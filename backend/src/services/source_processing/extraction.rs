//! Stage 1: a stored source becomes text.
//!
//! Every source type ends here with the same three answers: the text to index,
//! the chunking contract that text should be split under, and which optional
//! services were degraded on the way. Nothing in this module writes a chunk, a
//! generation or a status — extraction that cannot decide anything about the
//! index is extraction that can be read on its own.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use sea_orm::{ActiveModelTrait, Set};
use std::sync::Arc;
use tracing::{info, warn};
use uuid::Uuid;

use crate::core::config::CoreConfig;
use crate::core::entitlements::{AuthorizationRequest, Operation, Permit};
use crate::entities::source;
use crate::error::AppError;
use crate::services::ingestion_tasks::IngestionTasks;
use crate::services::source_events::SourceEventBroadcaster;
use crate::types::SourceType;

use super::{PipelineFailure, ProcessingDeps, StageContext as _};

/// PDF magic bytes: `%PDF-` (0x25 0x50 0x44 0x46 0x2D).
const PDF_MAGIC_BYTES: &[u8] = b"%PDF-";

/// What extraction produces, whatever the source type was.
pub(super) struct Extracted {
    pub text: String,
    /// The type the content should be *chunked* as, which is not always the
    /// source's own: the PDF+OCR path produces Markdown and chunks accordingly.
    pub chunking_source_type: SourceType,
    /// Optional services that were unavailable but did not stop the run.
    pub degraded_services: Vec<&'static str>,
    /// Set for YouTube, consumed by timestamp enrichment after chunking.
    pub youtube_video_id: Option<String>,
}

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

/// Validate that extracted content doesn't exceed size or word count limits.
///
/// User-facing messages are kept generic to avoid information disclosure;
/// exact values are logged at `warn!` for operator visibility.
pub(super) fn validate_content_limits(content: &str, config: &CoreConfig) -> Result<(), AppError> {
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

/// Turn a stored source into indexable text.
///
/// The dispatch every source type goes through. Each arm owns one format and
/// returns the same shape, so adding a format does not change anything the
/// caller does with the result.
pub(super) async fn extract_content(
    deps: &ProcessingDeps,
    tasks: &IngestionTasks,
    source: source::Model,
    source_id: Uuid,
    notebook_id: Uuid,
    source_type: SourceType,
) -> Result<Extracted, PipelineFailure> {
    let broadcaster = &deps.broadcaster;

    match source_type {
        SourceType::Web => {
            let text = extract_web(deps, &source).await?;
            Ok(Extracted::plain(text, source_type))
        }
        SourceType::Pdf => {
            let (text, chunking_source_type, degraded_services) =
                extract_pdf(&source, deps, tasks, source_id, notebook_id, broadcaster).await?;
            Ok(Extracted {
                text,
                chunking_source_type,
                degraded_services,
                youtube_video_id: None,
            })
        }
        SourceType::Docx => {
            let text = extract_docx(tasks, &source).await?;
            Ok(Extracted::plain(text, source_type))
        }
        SourceType::Epub => {
            let text = extract_epub(deps, tasks, source, source_id).await?;
            Ok(Extracted::plain(text, source_type))
        }
        SourceType::Text | SourceType::Markdown => {
            Ok(Extracted::plain(source.content.clone(), source_type))
        }
        SourceType::Youtube => {
            let (text, video_id) = extract_youtube(deps, source, source_id).await?;
            Ok(Extracted {
                text,
                chunking_source_type: source_type,
                degraded_services: Vec::new(),
                youtube_video_id: Some(video_id),
            })
        }
    }
}

impl Extracted {
    /// Text chunked as its own source type, with nothing degraded.
    fn plain(text: String, source_type: SourceType) -> Self {
        Self {
            text,
            chunking_source_type: source_type,
            degraded_services: Vec::new(),
            youtube_video_id: None,
        }
    }
}

async fn extract_web(
    deps: &ProcessingDeps,
    source: &source::Model,
) -> Result<String, PipelineFailure> {
    let url = source
        .metadata
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            PipelineFailure::from_error(AppError::Internal("Missing URL in metadata".into()))
        })?;

    let firecrawl_client = deps.firecrawl.as_ref().ok_or_else(|| {
        PipelineFailure::from_error(AppError::Internal(
            "Firecrawl client not configured — cannot scrape URLs".into(),
        ))
    })?;
    let scrape_result = firecrawl_client
        .scrape_url(url)
        .await
        .stage("Failed to fetch URL")?;
    Ok(crate::services::content_cleaning::clean_scraped_content(
        &scrape_result.content,
    ))
}

async fn extract_docx(
    tasks: &IngestionTasks,
    source: &source::Model,
) -> Result<String, PipelineFailure> {
    let docx_bytes = BASE64
        .decode(&source.content)
        .map_err(|e| AppError::Internal(format!("Failed to decode DOCX: {e}")))
        .stage("Failed to decode DOCX")?;

    // CPU-bound parsing on the blocking pool. Counted by `tasks`, which cannot
    // abort it: the drain reports it as still running rather than claiming it
    // was cancelled (US-010).
    tasks
        .spawn_blocking(move || crate::services::processor::extract_docx_text(&docx_bytes))
        .await
        .map_err(|e| AppError::Internal(format!("DOCX task panicked: {e}")))
        .stage("DOCX extraction did not complete")?
        .stage("Failed to extract DOCX text")
}

async fn extract_epub(
    deps: &ProcessingDeps,
    tasks: &IngestionTasks,
    source: source::Model,
    source_id: Uuid,
) -> Result<String, PipelineFailure> {
    let epub_bytes = BASE64
        .decode(&source.content)
        .map_err(|e| AppError::Internal(format!("Failed to decode EPUB: {e}")))
        .stage("Failed to decode EPUB")?;

    let (text, metadata) = tasks
        .spawn_blocking(move || crate::services::processor::extract_epub_text(&epub_bytes))
        .await
        .map_err(|e| AppError::Internal(format!("EPUB task panicked: {e}")))
        .stage("EPUB extraction did not complete")?
        .stage("Failed to extract EPUB text")?;

    // Store extracted metadata (title, author) on the source. A failure here
    // costs the user a title, not their document, so it is logged rather than
    // ending the run.
    match serde_json::to_value(&metadata) {
        Ok(updated_metadata) => {
            let mut source_model: source::ActiveModel = source.into();
            source_model.metadata = Set(updated_metadata);
            if let Err(e) = source_model.update(&deps.db).await {
                warn!(%source_id, error = %e, "Failed to update EPUB metadata on source");
            }
        }
        Err(e) => warn!(%source_id, error = %e, "Failed to serialize EPUB metadata"),
    }

    Ok(text)
}

async fn extract_youtube(
    deps: &ProcessingDeps,
    source: source::Model,
    source_id: Uuid,
) -> Result<(String, String), PipelineFailure> {
    let url = source
        .metadata
        .get("url")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .ok_or_else(|| AppError::Internal("Missing URL in YouTube source metadata".into()))
        .stage("Failed to read YouTube metadata")?;

    let video_id = crate::clients::youtube::extract_youtube_video_id(&url)
        .stage("Failed to read YouTube video id")?;

    let youtube_client = deps.youtube.as_ref().ok_or_else(|| {
        PipelineFailure::from_error(AppError::Internal("YouTube client not configured".into()))
    })?;

    let locale = deps.config.default_locale.as_deref().unwrap_or("en");

    // Fetch transcript and video details in parallel.
    let (transcript_result, details_result) = tokio::join!(
        youtube_client.fetch_transcript(&video_id, locale),
        youtube_client.fetch_video_details(&video_id),
    );

    let transcript = transcript_result.stage("Failed to fetch YouTube transcript")?;
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
        if let Err(e) = source_model.update(&deps.db).await {
            warn!(%source_id, error = %e, "Failed to update YouTube metadata on source");
        }
    }

    let markdown = crate::clients::youtube::format_transcript_as_markdown(
        &transcript.snippets,
        details.as_ref().map(|d| d.title.as_str()).unwrap_or(""),
    );
    Ok((markdown, video_id))
}

/// Process a PDF source: decode, extract text, handle OCR fallback, merge results.
///
/// Returns `(extracted_content, effective_chunking_source_type, degraded_services)`.
async fn extract_pdf(
    source: &source::Model,
    deps: &ProcessingDeps,
    tasks: &IngestionTasks,
    source_id: Uuid,
    notebook_id: Uuid,
    broadcaster: &SourceEventBroadcaster,
) -> Result<(String, SourceType, Vec<&'static str>), PipelineFailure> {
    let config = &deps.config;
    let mut degraded_services: Vec<&str> = Vec::new();
    let mut chunking_source_type = SourceType::Pdf;

    let pdf_bytes = Arc::new(BASE64.decode(&source.content).map_err(|e| {
        PipelineFailure::from_error(AppError::Internal(format!("Failed to decode PDF: {e}")))
    })?);

    validate_pdf_magic_bytes(&pdf_bytes).map_err(PipelineFailure::from_error)?;

    // Extract text and page count concurrently (both CPU-bound, wrapped in
    // spawn_blocking to avoid starving the tokio async executor). Counted by
    // `tasks`: neither can be aborted, so a drain that outran its deadline has
    // to report them rather than claim they stopped (US-010).
    let bytes_for_pages = Arc::clone(&pdf_bytes);
    let bytes_for_count = Arc::clone(&pdf_bytes);

    let pages_handle = tasks.spawn_blocking(move || {
        crate::services::processor::extract_pdf_text_by_pages(&bytes_for_pages)
    });
    let count_handle = tasks
        .spawn_blocking(move || crate::services::processor::get_pdf_page_count(&bytes_for_count));

    let (pages_result, count_result) = tokio::join!(pages_handle, count_handle);

    let mut page_segments = pages_result
        .map_err(|e| {
            PipelineFailure::from_error(AppError::Internal(format!(
                "PDF text extraction panicked: {e}"
            )))
        })?
        .map_err(|e| {
            PipelineFailure::from_error(AppError::Internal(format!(
                "Failed to extract PDF text: {e}"
            )))
        })?;

    let page_count = count_result
        .map_err(|e| {
            PipelineFailure::from_error(AppError::Internal(format!(
                "PDF page count task panicked: {e}"
            )))
        })?
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
                    return Err(PipelineFailure::new(user_msg, e));
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
                            return Err(PipelineFailure::new(user_msg, e));
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
            return Err(PipelineFailure::new(msg, AppError::Validation(msg.into())));
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

#[cfg(test)]
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
}
