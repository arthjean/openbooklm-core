//! Source processing service — RAG pipeline for document ingestion.
//!
//! Pipeline: extraction → chunking → contextualization → embedding → storage.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use sea_orm::{ActiveModelTrait, DatabaseConnection, Set, TransactionTrait};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::core::CoreState;
use crate::core::config::CoreConfig;
use crate::core::entitlements::{AuthorizationRequest, Operation, Permit, SharedEntitlementPolicy};
use crate::core::events::{DomainEvent, SharedEventSink};
use crate::core::principal::Principal;
use crate::entities::source::{self, SourceStatus};
use crate::error::AppError;
use crate::repositories::{ChunkRepository, OcrCacheRepository, SourceRepository};
use crate::services::source_events::SourceEventBroadcaster;
use crate::services::sources::{get_source, update_source_chunk_count, update_source_status};
use crate::types::SourceType;

/// PDF magic bytes: `%PDF-` (0x25 0x50 0x44 0x46 0x2D).
const PDF_MAGIC_BYTES: &[u8] = b"%PDF-";

/// Merge per-page native text with OCR results by page index.
///
/// For each OCR page, replaces the corresponding native text segment with the
/// OCR markdown. If an OCR page index exceeds the segment count, the segments
/// vector is extended with empty strings to accommodate it.
pub fn merge_ocr_text(page_segments: &[String], ocr_pages: &[crate::clients::OcrPage]) -> String {
    let mut segments: Vec<String> = page_segments.to_vec();

    for ocr_page in ocr_pages {
        let idx = ocr_page.index as usize;
        if idx >= segments.len() {
            segments.resize_with(idx + 1, String::new);
        }
        segments[idx].clone_from(&ocr_page.markdown);
    }

    segments.join("\n\n")
}

/// Validate that decoded bytes start with the PDF magic header.
pub(crate) fn validate_pdf_magic_bytes(bytes: &[u8]) -> Result<(), AppError> {
    if bytes.len() < PDF_MAGIC_BYTES.len() || &bytes[..PDF_MAGIC_BYTES.len()] != PDF_MAGIC_BYTES {
        return Err(AppError::Validation(
            "Invalid PDF file: does not start with %PDF- header".into(),
        ));
    }
    Ok(())
}

/// Dependencies for source processing, bundled to avoid too-many-arguments.
pub struct ProcessingDeps {
    pub db: DatabaseConnection,
    pub config: Arc<CoreConfig>,
    pub broadcaster: SourceEventBroadcaster,
    pub client_metrics: crate::clients::ClientMetrics,
    pub source_repo: Arc<dyn SourceRepository>,
    pub chunk_repo: Arc<dyn ChunkRepository>,
    pub embeddings: Option<crate::core::providers::SharedEmbeddingProvider>,
    pub firecrawl: Option<crate::clients::FirecrawlClient>,
    pub youtube: Option<crate::clients::YouTubeClient>,
    pub ocr: Option<crate::clients::MistralOcrClient>,
    pub ocr_cache: Arc<dyn OcrCacheRepository>,
    /// Authorizes and meters bounded work (OCR pages). No usage repository,
    /// plan string or SaaS profile reaches this pipeline.
    pub entitlements: SharedEntitlementPolicy,
    /// Receives ingestion outcome events.
    pub events: SharedEventSink,
    /// The account the source belongs to, carrying whatever opaque metadata the
    /// identity adapter attached.
    pub principal: Principal,
}

impl ProcessingDeps {
    /// Construct processing dependencies from the shared application state.
    ///
    /// All fields are extracted from `CoreState`, converting concrete repository
    /// types to trait objects for polymorphic use in the processing pipeline.
    pub fn from_state(state: &CoreState, principal: Principal) -> Self {
        Self {
            db: state.db.clone(),
            config: state.config.clone(),
            broadcaster: state.source_broadcaster.clone(),
            client_metrics: state.clients.metrics.clone(),
            source_repo: state.repos.sources.clone() as Arc<dyn SourceRepository>,
            chunk_repo: state.repos.chunks.clone() as Arc<dyn ChunkRepository>,
            embeddings: state.clients.embeddings.clone(),
            firecrawl: state.clients.firecrawl.clone(),
            youtube: state.clients.youtube.clone(),
            ocr: state.clients.ocr.clone(),
            ocr_cache: state.repos.ocr_cache.clone() as Arc<dyn OcrCacheRepository>,
            entitlements: state.entitlements.clone(),
            events: state.events.clone(),
            principal,
        }
    }

    /// The account id every event and permit is attributed to.
    pub fn account_id(&self) -> Uuid {
        self.principal.account_id
    }
}

/// Process a PDF source: decode, extract text, handle OCR fallback, merge results.
///
/// Returns `(extracted_content, effective_chunking_source_type, degraded_services)`.
async fn process_pdf_source(
    source: &source::Model,
    deps: &ProcessingDeps,
    source_id: Uuid,
    notebook_id: Uuid,
    source_repo: &dyn SourceRepository,
    broadcaster: &SourceEventBroadcaster,
    analytics: &AnalyticsCtx<'_>,
) -> Result<(String, SourceType, Vec<&'static str>), AppError> {
    let config = &deps.config;
    let mut degraded_services: Vec<&str> = Vec::new();
    let mut chunking_source_type = SourceType::Pdf;

    let pdf_bytes = Arc::new(
        BASE64
            .decode(&source.content)
            .map_err(|e| AppError::Internal(format!("Failed to decode PDF: {e}")))?,
    );

    validate_pdf_magic_bytes(&pdf_bytes)?;

    // Extract text and page count concurrently (both CPU-bound, wrapped
    // in spawn_blocking to avoid starving the tokio async executor).
    let bytes_for_pages = Arc::clone(&pdf_bytes);
    let bytes_for_count = Arc::clone(&pdf_bytes);

    let pages_handle = tokio::task::spawn_blocking(move || {
        crate::services::processor::extract_pdf_text_by_pages(&bytes_for_pages)
    });
    let count_handle = tokio::task::spawn_blocking(move || {
        crate::services::processor::get_pdf_page_count(&bytes_for_count)
    });

    let (pages_result, count_result) = tokio::join!(pages_handle, count_handle);

    let mut page_segments = pages_result
        .map_err(|e| AppError::Internal(format!("PDF text extraction panicked: {e}")))?
        .map_err(|e| {
            // Cannot use on_error! macro here — call set_error_status explicitly
            // in the caller if this returns Err
            AppError::Internal(format!("Failed to extract PDF text: {e}"))
        })?;

    let page_count = count_result
        .map_err(|e| AppError::Internal(format!("PDF page count task panicked: {e}")))?
        .unwrap_or(page_segments.len());

    // Full-document OCR fallback: when both extractors return no page
    // info, attempt OCR on the entire document (`pages: None`).
    let full_doc_ocr = page_segments.is_empty() && page_count == 0;

    // If lopdf found more pages than text extraction produced, pad with
    // empty strings so those pages are detected as needing OCR.
    if page_count > page_segments.len() {
        page_segments.resize(page_count, String::new());
    }

    // Detect pages with insufficient text for OCR fallback.
    let ocr_pages =
        crate::services::processor::detect_ocr_pages(&page_segments, config.ocr.min_text_threshold);

    let native_text = page_segments.join("\n\n");

    let content = if ocr_pages.is_empty() && !full_doc_ocr {
        // All pages have sufficient text — no OCR needed.
        native_text
    } else if let Some(ocr) = deps.ocr.as_ref() {
        // Pages need OCR and client is available.
        if full_doc_ocr {
            warn!(
                source_id = %source_id,
                scanned_pages = 0_usize,
                total_pages = 0_usize,
                full_doc_ocr = true,
                "ocr_fallback"
            );
        } else {
            warn!(
                source_id = %source_id,
                scanned_pages = ocr_pages.len(),
                total_pages = page_count,
                "ocr_fallback"
            );
        }

        // ── OCR plan/limit check ─────────────────────────────────
        // SAFETY: This billing check MUST remain BEFORE the cache
        // lookup below. See original comments for rationale.
        let mut ocr_pages = ocr_pages;
        if !full_doc_ocr && ocr_pages.len() > config.ocr.max_pages_per_request {
            warn!(
                source_id = %source_id,
                total_scanned = ocr_pages.len(),
                limit = config.ocr.max_pages_per_request,
                "OCR page list truncated to max_pages_per_request — \
                 remaining scanned pages will use native text"
            );
            ocr_pages.truncate(config.ocr.max_pages_per_request);
        }
        let pages_needed = if full_doc_ocr {
            i32::try_from(config.ocr.max_pages_per_request).unwrap_or(i32::MAX)
        } else {
            i32::try_from(ocr_pages.len()).unwrap_or(i32::MAX)
        };

        // Authorize the maximum work before the external call. The permit is
        // what records the pages actually processed afterwards, and it records
        // at most once. `source_id` is its operation id so a recording failure
        // is traceable to the source it belongs to. A reprocess calls the OCR
        // API again and is charged again, which is the intended behaviour.
        let ocr_permit: Result<Permit, AppError> = deps
            .entitlements
            .authorize(AuthorizationRequest::new(
                &deps.principal,
                Operation::ProcessOcrPages {
                    requested_pages: pages_needed,
                },
                source_id,
            ))
            .await;

        match ocr_permit {
            Err(e) => {
                if native_text.trim().is_empty() {
                    let user_msg = match &e {
                        AppError::Forbidden(_) => e.to_string(),
                        _ => {
                            "OCR processing is currently unavailable. Please try again.".to_string()
                        }
                    };
                    set_error_status(
                        source_repo,
                        broadcaster,
                        notebook_id,
                        source_id,
                        &user_msg,
                        analytics,
                    )
                    .await?;
                    return Err(e);
                }
                warn!(
                    source_id = %source_id,
                    error = %e,
                    "OCR limit exceeded — proceeding with partial text"
                );
                degraded_services.push("ocr");
                native_text
            }
            Ok(ocr_permit) => {
                // ── OCR cache check ─────────────────────────────────────
                let pdf_content_hash = crate::services::rag::utils::compute_bytes_hash(&pdf_bytes);
                let ocr_model = &config.ocr.model;

                let cache_result = deps
                    .ocr_cache
                    .find_by_hash(&pdf_content_hash, ocr_model)
                    .await;

                if let Err(ref e) = cache_result {
                    warn!(
                        source_id = %source_id,
                        error = %e,
                        "OCR cache lookup failed, proceeding as cache miss"
                    );
                }

                if let Some((cached_text, cached_pages)) = cache_result.unwrap_or(None) {
                    info!(
                        source_id = %source_id,
                        pages_processed = cached_pages,
                        duration_ms = 0_u64,
                        cache_hit = true,
                        "ocr_response"
                    );
                    broadcaster.broadcast_ocr_cache_hit(notebook_id, source_id);
                    chunking_source_type = SourceType::Markdown;
                    cached_text
                } else {
                    // Cache miss — call OCR API.
                    let ocr_page_selection = if full_doc_ocr { None } else { Some(ocr_pages) };
                    let ocr_total_pages = ocr_page_selection
                        .as_ref()
                        .map_or(0, |p| u32::try_from(p.len()).unwrap_or(u32::MAX));
                    let file_size_bytes = pdf_bytes.len();

                    info!(
                        source_id = %source_id,
                        pages_requested = ocr_total_pages,
                        full_doc_ocr,
                        file_size_bytes,
                        "ocr_request"
                    );

                    broadcaster.broadcast_ocr_started(notebook_id, source_id, ocr_total_pages);

                    let ocr_start = std::time::Instant::now();
                    let ocr_result = match ocr
                        .extract_text_from_pdf(&pdf_bytes, ocr_page_selection)
                        .await
                    {
                        Ok(v) => v,
                        Err(e) => {
                            tracing::error!(
                                source_id = %source_id,
                                error = %e,
                                "OCR processing failed"
                            );
                            let user_msg = "OCR processing failed. \
                            Please try again or upload a clearer scan.";
                            set_error_status(
                                source_repo,
                                broadcaster,
                                notebook_id,
                                source_id,
                                user_msg,
                                analytics,
                            )
                            .await?;
                            return Err(e);
                        }
                    };
                    let ocr_duration_ms =
                        u64::try_from(ocr_start.elapsed().as_millis()).unwrap_or(u64::MAX);

                    // Record the pages the provider actually processed, clamped to
                    // what the permit authorized.
                    let pages_processed = ocr_permit.clamp_units(
                        i32::try_from(ocr_result.pages_processed).unwrap_or(pages_needed),
                    );
                    if let Err(e) = deps.entitlements.record(&ocr_permit, pages_processed).await {
                        tracing::error!(
                            source_id = %source_id,
                            pages_processed,
                            error = %e,
                            "Failed to record OCR page usage — unbilled API cost"
                        );
                    }

                    let merged_text = if full_doc_ocr {
                        ocr_result
                            .pages
                            .iter()
                            .map(|p| p.markdown.as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n")
                    } else {
                        merge_ocr_text(&page_segments, &ocr_result.pages)
                    };

                    // Store in cache only if OCR produced meaningful output.
                    let cache_worthy = pages_processed > 0 && merged_text.trim().len() >= 10;
                    if cache_worthy {
                        if let Err(e) = deps
                            .ocr_cache
                            .store(&pdf_content_hash, ocr_model, &merged_text, pages_processed)
                            .await
                        {
                            warn!(
                                source_id = %source_id,
                                error = %e,
                                "Failed to store OCR result in cache"
                            );
                        }
                    } else {
                        warn!(
                            source_id = %source_id,
                            pages_processed,
                            merged_text_len = merged_text.trim().len(),
                            "Skipping OCR cache store — result too small or empty"
                        );
                    }

                    let clamped_pages = if full_doc_ocr {
                        ocr_result.pages_processed
                    } else {
                        ocr_result.pages_processed.min(ocr_total_pages)
                    };

                    info!(
                        source_id = %source_id,
                        pages_processed = clamped_pages,
                        duration_ms = ocr_duration_ms,
                        cache_hit = false,
                        "ocr_response"
                    );
                    broadcaster.broadcast_ocr_completed(notebook_id, source_id, clamped_pages);

                    chunking_source_type = SourceType::Markdown;
                    merged_text
                }
            }
        }
    } else {
        // OCR needed but client not available.
        if native_text.trim().is_empty() {
            let msg = "This PDF contains scanned pages that require OCR. \
                Upgrade to Pro to process scanned documents.";
            set_error_status(
                source_repo,
                broadcaster,
                notebook_id,
                source_id,
                msg,
                analytics,
            )
            .await?;
            return Err(AppError::Validation(msg.into()));
        }
        warn!(
            source_id = %source_id,
            scanned_pages = ocr_pages.len(),
            total_pages = page_count,
            "Proceeding with partial text — OCR not available for scanned pages"
        );
        degraded_services.push("ocr");
        native_text
    };

    Ok((content, chunking_source_type, degraded_services))
}

/// Process a source through the full RAG pipeline.
///
/// Stages: extraction → chunking → embedding → storage.
/// Note: Contextualization (Stage 3) is temporarily disabled to control costs.
/// See IMPORTANT_FUTUR.md at the repo root to re-enable it.
#[tracing::instrument(skip(deps), fields(%source_id, %notebook_id, source_type = source_type.as_str()))]
pub async fn process_source(
    deps: ProcessingDeps,
    source_id: Uuid,
    notebook_id: Uuid,
    source_type: SourceType,
) -> Result<(), AppError> {
    // NOTE: ContextualizationService import disabled — see IMPORTANT_FUTUR.md
    // use crate::services::rag::contextual::ContextualizationService;

    let pipeline_start = std::time::Instant::now();

    let db = &deps.db;
    let config = &deps.config;
    let broadcaster = &deps.broadcaster;
    let source_repo = deps.source_repo.as_ref();
    let chunk_repo = &deps.chunk_repo;
    let embeddings = &deps.embeddings;
    let firecrawl = &deps.firecrawl;
    let analytics = AnalyticsCtx {
        events: &deps.events,
        account_id: deps.account_id(),
        source_type,
        pipeline_start,
    };

    // Helper macro to handle errors consistently
    macro_rules! on_error {
        ($result:expr, $msg:literal) => {
            match $result {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!(concat!($msg, ": {}"), e);
                    set_error_status(
                        source_repo,
                        broadcaster,
                        notebook_id,
                        source_id,
                        &msg,
                        &analytics,
                    )
                    .await?;
                    return Err(e.into());
                }
            }
        };
    }

    // ── Stage 1: Extraction ──────────────────────────────────────────────
    update_source_status(source_repo, source_id, SourceStatus::Processing, None).await?;
    broadcaster.broadcast_status(notebook_id, source_id, "processing", None);

    let source = get_source(source_repo, source_id)
        .await?
        .ok_or_else(|| AppError::NotFound("Source not found".into()))?;

    let mut degraded_services: Vec<&str> = Vec::new();
    // Overridden to Markdown in the PDF+OCR path for section-aware chunking.
    let mut chunking_source_type = source_type;
    // Captured for YouTube timestamp enrichment after chunking.
    let mut youtube_video_id: Option<String> = None;

    let content = match source_type {
        SourceType::Web => {
            let url = source
                .metadata
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::Internal("Missing URL in metadata".into()))?;

            let firecrawl_client = firecrawl.as_ref().ok_or_else(|| {
                AppError::Internal("Firecrawl client not configured — cannot scrape URLs".into())
            })?;
            let scrape_result = on_error!(
                firecrawl_client.scrape_url(url).await,
                "Failed to fetch URL"
            );
            crate::services::content_cleaning::clean_scraped_content(&scrape_result.content)
        }
        SourceType::Pdf => {
            let (content, effective_type, pdf_degraded) = process_pdf_source(
                &source,
                &deps,
                source_id,
                notebook_id,
                source_repo,
                broadcaster,
                &analytics,
            )
            .await?;
            chunking_source_type = effective_type;
            degraded_services = pdf_degraded;
            content
        }
        SourceType::Docx => {
            let docx_bytes = BASE64
                .decode(&source.content)
                .map_err(|e| AppError::Internal(format!("Failed to decode DOCX: {e}")))?;

            // CPU-bound parsing wrapped in spawn_blocking (aligned with PDF handling)
            on_error!(
                tokio::task::spawn_blocking(move || {
                    crate::services::processor::extract_docx_text(&docx_bytes)
                })
                .await
                .map_err(|e| AppError::Internal(format!("DOCX task panicked: {e}")))?,
                "Failed to extract DOCX text"
            )
        }
        SourceType::Epub => {
            let epub_bytes = BASE64
                .decode(&source.content)
                .map_err(|e| AppError::Internal(format!("Failed to decode EPUB: {e}")))?;

            // CPU-bound parsing wrapped in spawn_blocking (aligned with PDF handling)
            let (text, metadata) = on_error!(
                tokio::task::spawn_blocking(move || {
                    crate::services::processor::extract_epub_text(&epub_bytes)
                })
                .await
                .map_err(|e| AppError::Internal(format!("EPUB task panicked: {e}")))?,
                "Failed to extract EPUB text"
            );

            // Store extracted metadata (title, author) on the source
            match serde_json::to_value(&metadata) {
                Ok(updated_metadata) => {
                    let mut source_model: source::ActiveModel = source.into();
                    source_model.metadata = Set(updated_metadata);
                    if let Err(e) = source_model.update(db).await {
                        warn!(
                            source_id = %source_id,
                            error = %e,
                            "Failed to update EPUB metadata on source"
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        source_id = %source_id,
                        error = %e,
                        "Failed to serialize EPUB metadata"
                    );
                }
            }

            text
        }
        SourceType::Text | SourceType::Markdown => source.content.clone(),
        SourceType::Youtube => {
            let url = source
                .metadata
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::Internal("Missing URL in YouTube source metadata".into()))?
                .to_string();

            let video_id = crate::clients::youtube::extract_youtube_video_id(&url)?;
            youtube_video_id = Some(video_id.clone());

            let youtube_client = deps
                .youtube
                .as_ref()
                .ok_or_else(|| AppError::Internal("YouTube client not configured".into()))?;

            let locale = deps.config.default_locale.as_deref().unwrap_or("en");

            // Fetch transcript and video details in parallel.
            let (transcript_result, details_result) = tokio::join!(
                youtube_client.fetch_transcript(&video_id, locale),
                youtube_client.fetch_video_details(&video_id),
            );

            let transcript = on_error!(transcript_result, "Failed to fetch YouTube transcript");
            let details = details_result.ok();

            // Update source title from video metadata if available.
            if let Some(ref d) = details {
                let mut source_model: source::ActiveModel = source.into();
                let mut meta = serde_json::json!({
                    "url": &url,
                    "video_id": &video_id,
                    "video_title": &d.title,
                    "channel_name": &d.author,
                    "duration_seconds": d.duration_seconds,
                    "language_code": &transcript.language_code,
                    "is_generated_captions": transcript.is_generated,
                });
                if let Some(ref thumb) = d.thumbnail_url {
                    meta["thumbnail_url"] = serde_json::Value::String(thumb.clone());
                }
                source_model.metadata = Set(meta);
                if !d.title.is_empty() {
                    source_model.title = Set(d.title.clone());
                }
                if let Err(e) = source_model.update(db).await {
                    warn!(
                        source_id = %source_id,
                        error = %e,
                        "Failed to update YouTube metadata on source"
                    );
                }
            }

            // Format transcript as timestamped Markdown.
            crate::clients::youtube::format_transcript_as_markdown(
                &transcript.snippets,
                details.as_ref().map(|d| d.title.as_str()).unwrap_or(""),
            )
        }
    };

    // ── Size + word count validation ───────────────────────────────────
    if let Err(e) = validate_content_limits(&content, config) {
        let msg = e.to_string();
        set_error_status(
            source_repo,
            broadcaster,
            notebook_id,
            source_id,
            &msg,
            &analytics,
        )
        .await?;
        return Err(e);
    }

    // ── Stage 2: Two-pass parent-child chunking ─────────────────────────
    // Pass 1: Split content into parent chunks (1024 tokens).
    // Pass 2: Split each parent into child chunks (256 tokens).
    // Children are the retrieval/embedding unit; parents provide LLM context.
    let parents = on_error!(
        crate::services::rag::chunking::chunk_content_with_parents(&content, chunking_source_type),
        "Failed to chunk content"
    );

    if parents.is_empty() {
        let msg = "No content to process";
        set_error_status(
            source_repo,
            broadcaster,
            notebook_id,
            source_id,
            msg,
            &analytics,
        )
        .await?;
        return Err(AppError::Validation(msg.into()));
    }

    // ── Stage 3: Contextualization — TEMPORARILY DISABLED ──────────────
    // Contextualization generates LLM-powered context prefixes for each chunk
    // via Anthropic API (Haiku), which improves retrieval quality but adds
    // significant cost (one API call per chunk). Disabled to control costs
    // until user base justifies it. See IMPORTANT_FUTUR.md to re-enable.
    //
    // Original pipeline: extraction → chunking → contextualization → embedding → storage
    // Current pipeline:  extraction → chunking → embedding → storage

    // Flatten parent/child hierarchy into a flat Vec<ChunkWithContext>.
    // Each child's `content` is the child text (for embedding + BM25),
    // and `parent_content` is the parent text (for LLM context).
    // Guards against excessive chunk fan-out (max 50k) before allocation.
    let mut chunks_with_context = match flatten_parent_child_chunks(&parents) {
        Ok(chunks) => chunks,
        Err(e) => {
            let msg = e.to_string();
            set_error_status(
                source_repo,
                broadcaster,
                notebook_id,
                source_id,
                &msg,
                &analytics,
            )
            .await?;
            return Err(e);
        }
    };

    // ── YouTube timestamp metadata enrichment ────────────────────────────
    // Parse [MM:SS] timestamps from chunk content and populate metadata
    // fields for deep-link citations.
    if let Some(ref vid) = youtube_video_id {
        for chunk in &mut chunks_with_context {
            let (ts_start, ts_end) =
                crate::clients::youtube::extract_timestamp_range(&chunk.content);
            chunk.metadata.video_id = Some(vid.clone());
            chunk.metadata.timestamp_start = ts_start;
            chunk.metadata.timestamp_end = ts_end;
            if let Some(start) = ts_start {
                chunk.metadata.citation_url = Some(format!(
                    "https://youtube.com/watch?v={vid}&t={}",
                    start as u64
                ));
            }
        }
    }

    let total_chunks = chunks_with_context.len();

    // ── Stage 3b: Content hash deduplication ─────────────────────────────
    // Reuse embeddings for unchanged content (same SHA-256 hash).
    // Note: hash is keyed on child text only (not parent_content). If contextual
    // embeddings are re-enabled (IMPORTANT_FUTUR.md), the dedup cache must
    // also incorporate parent_content since context prefixes would differ.
    let (needs_embedding_indices, reused_embeddings) = on_error!(
        compute_embedding_reuse(chunk_repo.as_ref(), source_id, &chunks_with_context).await,
        "Failed to query existing chunk hashes"
    );

    let reused_count = reused_embeddings.len();
    let new_count = needs_embedding_indices.len();
    info!(
        total_chunks = total_chunks,
        new_chunks = new_count,
        reused_chunks = reused_count,
        "Chunk deduplication results"
    );

    // ── Stage 4+5: Pipelined embedding + storage ──────────────────────────
    // Producer: embeds batches (reused or via Voyage AI), sends through channel.
    // Consumer: stores each batch incrementally within a single DB transaction.
    // The channel capacity (2) provides natural backpressure.
    update_source_status(source_repo, source_id, SourceStatus::Embedding, None).await?;
    broadcaster.broadcast_status(notebook_id, source_id, "embedding", None);

    let embedder = embeddings.as_ref().ok_or_else(|| {
        AppError::Internal("No embedding provider configured — cannot generate embeddings".into())
    })?;

    // Delete old chunks before the pipeline starts.
    chunk_repo.delete_for_source(source_id).await?;

    // Build work items: contiguous runs of chunks grouped by the Voyage batch size.
    // Each item knows which indices are reused vs need embedding.
    let batch_size = embedder.batch_size();
    let concurrency = embedder.concurrency();

    struct EmbeddedBatch {
        base_index: usize,
        chunks: Vec<crate::types::ChunkWithContext>,
        embeddings: Vec<Vec<f32>>,
    }

    let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbeddedBatch>(2);

    // --- Producer task ---
    let embedder_clone = Arc::clone(embedder);
    let chunks_owned = chunks_with_context;
    let broadcaster_clone = broadcaster.clone();
    let progress_total = u32::try_from(total_chunks).unwrap_or(u32::MAX);
    let progress_counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let progress_counter_clone = Arc::clone(&progress_counter);

    let producer_handle = tokio::spawn(async move {
        use futures::stream::{self, StreamExt};

        // Build batch work items: (batch_index, base_chunk_index, chunk_slice)
        let batch_work: Vec<(usize, usize, Vec<crate::types::ChunkWithContext>)> = chunks_owned
            .chunks(batch_size)
            .enumerate()
            .map(|(batch_idx, chunk_slice)| {
                (batch_idx, batch_idx * batch_size, chunk_slice.to_vec())
            })
            .collect();

        let results: Vec<Result<(), AppError>> = stream::iter(batch_work)
            .map(|(_batch_idx, base_index, batch_chunks)| {
                let tx = tx.clone();
                let embedder = &embedder_clone;
                let reused = &reused_embeddings;
                let progress = Arc::clone(&progress_counter_clone);
                let broadcaster = &broadcaster_clone;

                async move {
                    // Resolve embeddings for this batch: reuse cached or call Voyage AI
                    let mut batch_embeddings: Vec<Vec<f32>> =
                        Vec::with_capacity(batch_chunks.len());
                    let mut embed_offsets: Vec<usize> = Vec::new();
                    let mut embed_texts: Vec<String> = Vec::new();

                    for (offset, chunk) in batch_chunks.iter().enumerate() {
                        let global_idx = base_index + offset;
                        if let Some(cached) = reused.get(&global_idx) {
                            batch_embeddings.push(cached.clone());
                        } else {
                            let text = crate::services::embeddings::contextualized_text(
                                &chunk.content,
                                chunk.context_prefix.as_deref(),
                            );
                            embed_offsets.push(offset);
                            embed_texts.push(text);
                            batch_embeddings.push(Vec::new()); // placeholder
                        }
                    }

                    // Embed the texts that were not reused. Batching shape,
                    // rate limiting and usage reconciliation all belong to the
                    // provider now (US-020).
                    if !embed_texts.is_empty() {
                        let new_embeddings = embedder.embed_batch(&embed_texts).await?;
                        // Fail here rather than in the pgvector insert: a
                        // wrong-width vector must never reach a chunk row, and
                        // the error has to name the provider (US-020).
                        crate::core::providers::check_batch_widths(
                            embedder.as_ref(),
                            &new_embeddings,
                        )?;

                        // Place fresh embeddings into their correct offsets
                        for (offset, emb) in embed_offsets.into_iter().zip(new_embeddings) {
                            batch_embeddings[offset] = emb;
                        }
                    }

                    // Update progress
                    progress.fetch_add(
                        u32::try_from(batch_chunks.len()).unwrap_or(u32::MAX),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    broadcaster.broadcast_embedding_progress(
                        notebook_id,
                        source_id,
                        progress.load(std::sync::atomic::Ordering::Relaxed),
                        progress_total,
                    );

                    // Send to consumer
                    tx.send(EmbeddedBatch {
                        base_index,
                        chunks: batch_chunks,
                        embeddings: batch_embeddings,
                    })
                    .await
                    .map_err(|_| {
                        AppError::Internal("Pipeline consumer stopped unexpectedly".into())
                    })?;

                    Ok(())
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        // Drop the sender so the consumer's recv loop terminates
        drop(tx);

        // Return the first error (if any)
        for result in results {
            result?;
        }
        Ok::<(), AppError>(())
    });

    // --- Consumer loop (runs on current task) ---
    // Each batch commits independently to avoid holding a long-lived DB
    // transaction open while embedding batches complete (which can take
    // minutes for large documents and risk Neon connection timeouts).
    let embed_start = std::time::Instant::now();
    let mut consumer_error: Option<AppError> = None;
    let mut stored_count: usize = 0;

    while let Some(batch) = rx.recv().await {
        let batch_txn = match deps.db.begin().await {
            Ok(txn) => txn,
            Err(e) => {
                consumer_error = Some(AppError::Internal(format!(
                    "Failed to begin batch transaction: {e}"
                )));
                drop(rx);
                break;
            }
        };
        if let Err(e) = chunk_repo
            .store_chunk_batch(
                source_id,
                &batch.chunks,
                &batch.embeddings,
                batch.base_index,
                &batch_txn,
            )
            .await
        {
            if let Err(rollback_err) = batch_txn.rollback().await {
                tracing::error!(
                    %source_id,
                    error = %rollback_err,
                    "Failed to rollback batch transaction"
                );
            }
            consumer_error = Some(e);
            drop(rx);
            break;
        }
        if let Err(e) = batch_txn.commit().await {
            consumer_error = Some(AppError::Internal(format!("Failed to commit batch: {e}")));
            drop(rx);
            break;
        }
        stored_count += batch.chunks.len();
    }

    // Wait for producer to finish
    let producer_result = producer_handle
        .await
        .map_err(|e| AppError::Internal(format!("Pipeline producer task panicked: {e}")))?;

    // Handle errors
    if let Some(consumer_err) = consumer_error {
        set_error_status(
            source_repo,
            broadcaster,
            notebook_id,
            source_id,
            &format!("Failed to store chunks: {consumer_err}"),
            &analytics,
        )
        .await?;
        return Err(consumer_err);
    }
    if let Err(producer_err) = producer_result {
        set_error_status(
            source_repo,
            broadcaster,
            notebook_id,
            source_id,
            &format!("Failed to generate embeddings: {producer_err}"),
            &analytics,
        )
        .await?;
        return Err(producer_err);
    }

    let pipeline_ms = embed_start.elapsed().as_millis();
    info!(
        total_chunks = stored_count,
        pipeline_ms, "Pipeline embedding+storage completed"
    );

    // ── Finalize ─────────────────────────────────────────────────────────
    // Update chunk count first, then mark ready. If the second update fails,
    // the source stays in "embedding" status and can be reprocessed.
    // Chunk data integrity is already guaranteed by the store_chunks transaction.
    // Safe: total_chunks <= MAX_CHUNKS (50_000) which fits in i32.
    let chunk_count = i32::try_from(total_chunks).unwrap_or(i32::MAX);
    update_source_chunk_count(source_repo, source_id, chunk_count).await?;
    update_source_status(source_repo, source_id, SourceStatus::Ready, None).await?;

    if degraded_services.is_empty() {
        broadcaster.broadcast_ready(notebook_id, source_id, chunk_count);
    } else {
        let services: Vec<String> = degraded_services.iter().map(|s| (*s).to_string()).collect();
        info!(
            source_id = %source_id,
            degraded = ?services,
            "Source processed with degraded services"
        );
        broadcaster.broadcast_ready_degraded(notebook_id, source_id, chunk_count, services);
        analytics
            .events
            .emit(DomainEvent::SourceProcessingDegraded {
                account_id: analytics.account_id,
                notebook_id,
                source_id,
                degraded_services: degraded_services.clone(),
            });
    }

    analytics
        .events
        .emit(DomainEvent::SourceProcessingCompleted {
            account_id: analytics.account_id,
            notebook_id,
            source_id,
            source_type: analytics.source_type,
            duration_ms: analytics.elapsed_ms(),
            chunk_count,
        });

    info!(source_id = %source_id, chunk_count, "Source processed successfully");
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    #[test]
    fn validate_pdf_magic_bytes_accepts_valid_pdf() {
        let valid = b"%PDF-1.7 rest of file";
        assert!(validate_pdf_magic_bytes(valid).is_ok());
    }

    #[test]
    fn validate_pdf_magic_bytes_rejects_non_pdf() {
        let invalid = b"PK\x03\x04 not a pdf";
        assert!(validate_pdf_magic_bytes(invalid).is_err());
    }

    #[test]
    fn validate_pdf_magic_bytes_rejects_too_short() {
        let too_short = b"%PD";
        assert!(validate_pdf_magic_bytes(too_short).is_err());
    }

    /// Integration test: process a source when Anthropic is unavailable.
    ///
    /// Requires `TEST_DATABASE_URL` env var pointing to a PostgreSQL database
    /// with pgvector extension and Voyage AI API key configured. Run with:
    /// ```
    /// TEST_DATABASE_URL=postgres://... VOYAGE_API_KEY=... cargo test -- --ignored process_source_without_anthropic
    /// ```
    // TODO(US-017): Implement when test infrastructure is available.
    // This test validates that source processing completes with degraded
    // services when Anthropic is not configured (context_prefix = None,
    // SSE ready event includes degraded_services = ["contextualization"]).
    // Requires: test DB, source record, Voyage AI, Anthropic key NOT set.

    // ── Parent-child pipeline tests (US-009) ─────────────────────────────

    #[test]
    fn parent_child_flattening_produces_correct_chunk_with_context() {
        use crate::services::rag::chunking::chunk_content_with_parents;
        use crate::types::ChunkWithContext;

        // Generate a document large enough to produce multiple parents and children
        let content =
            "The quick brown fox jumps over the lazy dog near the river bank. ".repeat(500);

        let parents =
            chunk_content_with_parents(&content, SourceType::Text).expect("chunking should work");
        assert!(!parents.is_empty(), "Should produce at least one parent");

        // Flatten exactly as source_processing.rs does
        let total_children: usize = parents.iter().map(|(_, c, _)| c.len()).sum();
        let mut chunks: Vec<ChunkWithContext> = Vec::with_capacity(total_children);
        for (parent_text, children, parent_meta) in &parents {
            let shared_parent: Arc<str> = Arc::from(parent_text.as_str());
            for child_text in children {
                let content_hash = crate::services::rag::utils::compute_content_hash(child_text);
                let metadata = crate::types::ChunkMetadata {
                    section_header: parent_meta.section_header.clone(),
                    page_number: parent_meta.page_number,
                    position: chunks.len() as u32,
                    ..crate::types::ChunkMetadata::default()
                };
                chunks.push(ChunkWithContext {
                    content: child_text.clone(),
                    context_prefix: None,
                    parent_content: Some(Arc::clone(&shared_parent)),
                    metadata,
                    content_hash,
                });
            }
        }

        // Verify chunk count matches expected total
        assert_eq!(chunks.len(), total_children);
        assert!(
            chunks.len() >= 2 * parents.len(),
            "Should have at least 2× more children ({}) than parents ({})",
            chunks.len(),
            parents.len()
        );

        // Every chunk must have parent_content set
        for chunk in &chunks {
            assert!(
                chunk.parent_content.is_some(),
                "Every child chunk must have parent_content"
            );
        }

        // Content hash is from child text, not parent
        for chunk in &chunks {
            let expected_hash = crate::services::rag::utils::compute_content_hash(&chunk.content);
            assert_eq!(
                chunk.content_hash, expected_hash,
                "content_hash must be computed from child text"
            );
        }

        // Positions are sequential
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.metadata.position, i as u32);
        }

        // No chunk has context_prefix (contextualization disabled)
        for chunk in &chunks {
            assert!(chunk.context_prefix.is_none());
        }
    }

    #[test]
    fn parent_child_flattening_child_content_differs_from_parent() {
        use crate::services::rag::chunking::chunk_content_with_parents;

        // Use varied content so overlap doesn't produce identical children
        let content: String = (0..500)
            .map(|i| {
                format!(
                    "Sentence number {i} discusses topic {topic} in depth. ",
                    topic = i % 7
                )
            })
            .collect();

        let parents =
            chunk_content_with_parents(&content, SourceType::Pdf).expect("chunking should work");

        // Find a parent with multiple children
        let multi_child_parent = parents.iter().find(|(_, children, _)| children.len() > 1);
        assert!(
            multi_child_parent.is_some(),
            "Should have at least one parent with multiple children"
        );

        let (parent_text, children, _) = multi_child_parent.unwrap();

        // Each child should be shorter than its parent
        for child in children {
            assert!(
                child.len() < parent_text.len(),
                "Child ({} bytes) should be shorter than parent ({} bytes)",
                child.len(),
                parent_text.len()
            );
        }

        // Children should be different from each other
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                assert_ne!(
                    children[i], children[j],
                    "Children {i} and {j} should not be identical"
                );
            }
        }
    }
}

/// Validate that extracted content doesn't exceed size or word count limits.
///
/// User-facing messages are kept generic to avoid information disclosure;
/// exact values are logged at `warn!` for operator visibility.
fn validate_content_limits(content: &str, config: &CoreConfig) -> Result<(), AppError> {
    let max_size = config.security.max_source_size_bytes;
    if content.len() > max_size {
        warn!(
            content_bytes = content.len(),
            max_bytes = max_size,
            "Source content exceeds size limit"
        );
        return Err(AppError::Validation(
            "Source content is too large to process".into(),
        ));
    }

    let max_words = config.security.max_source_words;
    let word_count = content.split_whitespace().count();
    if word_count > max_words {
        warn!(word_count, max_words, "Source exceeds word count limit");
        return Err(AppError::Validation(
            "Source has too many words to process".into(),
        ));
    }

    Ok(())
}

/// Flatten a parent/child chunk hierarchy into a flat `Vec<ChunkWithContext>`.
///
/// Each child chunk gets a sequential global position and a reference to its parent
/// text (for LLM context). Returns an error if the total chunk count exceeds `MAX_CHUNKS`.
fn flatten_parent_child_chunks(
    parents: &[(String, Vec<String>, crate::types::ChunkMetadata)],
) -> Result<Vec<crate::types::ChunkWithContext>, AppError> {
    use crate::types::ChunkWithContext;
    const MAX_CHUNKS: usize = 50_000;

    let total_children: usize = parents.iter().map(|(_, children, _)| children.len()).sum();
    if total_children > MAX_CHUNKS {
        return Err(AppError::Validation(format!(
            "Document produced too many chunks ({total_children}), maximum is {MAX_CHUNKS}"
        )));
    }

    let mut chunks: Vec<ChunkWithContext> = Vec::with_capacity(total_children);
    for (parent_text, children, parent_meta) in parents {
        let shared_parent: Arc<str> = Arc::from(parent_text.as_str());
        let shared_header: Option<Arc<str>> = parent_meta.section_header.as_deref().map(Arc::from);
        for child_text in children {
            let content_hash = crate::services::rag::utils::compute_content_hash(child_text);
            let metadata = crate::types::ChunkMetadata {
                section_header: shared_header.as_ref().map(|h| String::from(&**h)),
                page_number: parent_meta.page_number,
                position: u32::try_from(chunks.len()).unwrap_or(u32::MAX),
                ..crate::types::ChunkMetadata::default()
            };
            chunks.push(ChunkWithContext {
                content: child_text.to_string(),
                context_prefix: None,
                parent_content: Some(Arc::clone(&shared_parent)),
                metadata,
                content_hash,
            });
        }
    }

    Ok(chunks)
}

/// Build a map of existing chunk content hashes to their embeddings for reuse.
///
/// Returns `(hash_map, needs_embedding_indices, reused_embeddings)`.
async fn compute_embedding_reuse(
    chunk_repo: &dyn ChunkRepository,
    source_id: Uuid,
    chunks: &[crate::types::ChunkWithContext],
) -> Result<(Vec<usize>, HashMap<usize, Vec<f32>>), AppError> {
    let existing = chunk_repo.get_chunks_with_hashes(source_id).await?;
    let existing_hash_map: HashMap<String, Vec<f32>> = existing
        .into_iter()
        .filter(|(_, hash, _)| !hash.is_empty())
        .map(|(_, hash, embedding)| (hash, embedding))
        .collect();

    let mut needs_embedding_indices: Vec<usize> = Vec::new();
    let mut reused_embeddings: HashMap<usize, Vec<f32>> = HashMap::new();

    for (i, chunk) in chunks.iter().enumerate() {
        if let Some(embedding) = existing_hash_map.get(&chunk.content_hash) {
            reused_embeddings.insert(i, embedding.clone());
        } else {
            needs_embedding_indices.push(i);
        }
    }

    Ok((needs_embedding_indices, reused_embeddings))
}

/// Event context for source processing outcomes.
struct AnalyticsCtx<'a> {
    events: &'a SharedEventSink,
    account_id: Uuid,
    source_type: SourceType,
    pipeline_start: std::time::Instant,
}

impl AnalyticsCtx<'_> {
    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.pipeline_start.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// Set error status on a source and broadcast the error event.
async fn set_error_status(
    source_repo: &dyn SourceRepository,
    broadcaster: &SourceEventBroadcaster,
    notebook_id: Uuid,
    source_id: Uuid,
    message: &str,
    analytics: &AnalyticsCtx<'_>,
) -> Result<(), AppError> {
    update_source_status(
        source_repo,
        source_id,
        SourceStatus::Error,
        Some(message.to_string()),
    )
    .await?;
    broadcaster.broadcast_error(notebook_id, source_id, message);
    analytics.events.emit(DomainEvent::SourceProcessingFailed {
        account_id: analytics.account_id,
        notebook_id,
        source_id,
        source_type: analytics.source_type,
        error_type: categorize_error(message),
        duration_ms: analytics.elapsed_ms(),
    });
    Ok(())
}

/// Categorize an error message into a short error type string.
fn categorize_error(message: &str) -> &'static str {
    let msg = message.to_lowercase();
    if msg.contains("ocr") {
        "ocr_error"
    } else if msg.contains("pdf") {
        "pdf_parse_error"
    } else if msg.contains("embedding") {
        "embedding_error"
    } else if msg.contains("timeout") || msg.contains("timed out") {
        "timeout"
    } else if msg.contains("chunk") {
        "chunking_error"
    } else if msg.contains("extract") || msg.contains("scrape") || msg.contains("firecrawl") {
        "extraction_error"
    } else if msg.contains("size") || msg.contains("too large") || msg.contains("empty") {
        "validation_error"
    } else {
        "unknown_error"
    }
}
