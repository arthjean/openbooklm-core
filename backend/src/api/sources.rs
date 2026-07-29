//! Source management API handlers.

use axum::{
    Json,
    extract::{Extension, Path, State},
    http::HeaderMap,
    response::{IntoResponse, Response, Sse, sse::Event},
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{convert::Infallible, time::Duration};
use tokio_stream::wrappers::BroadcastStream;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::api::common::{success_response, verify_notebook_access, verify_source_access};
use crate::api::common::{validate_string, validate_title, validate_url_for_ssrf};
use crate::core::CoreState;
use crate::core::entitlements::{AuthorizationRequest, Operation};
use crate::core::principal::Principal;
use crate::entities::source::SourceStatus;
use crate::error::AppError;
use crate::services::source_events::SourceEvent;
use crate::services::sources::{create_source, delete_source, list_sources, update_source_status};
use crate::types::SourceType;

// =============================================================================
// CONSTANTS
// =============================================================================

/// SSE keep-alive interval in seconds.
const SSE_KEEPALIVE_SECS: u64 = 15;

// =============================================================================
// VALIDATION HELPERS
// =============================================================================

use crate::services::source_processing::validate_pdf_magic_bytes;

/// ZIP magic bytes: `PK\x03\x04` (used by DOCX and EPUB).
const ZIP_MAGIC_BYTES: &[u8] = &[0x50, 0x4B, 0x03, 0x04];

/// Validate that decoded bytes start with the ZIP magic header (for DOCX/EPUB).
fn validate_zip_magic_bytes(bytes: &[u8]) -> Result<(), AppError> {
    if bytes.len() < ZIP_MAGIC_BYTES.len() || &bytes[..ZIP_MAGIC_BYTES.len()] != ZIP_MAGIC_BYTES {
        return Err(AppError::Validation(
            "Invalid file: does not start with ZIP (PK) header".into(),
        ));
    }
    Ok(())
}

// =============================================================================
// RESPONSE TYPES
// =============================================================================

/// Response for a source.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct SourceResponse {
    pub id: Uuid,
    pub notebook_id: Uuid,
    pub title: String,
    pub source_type: String,
    pub status: String,
    pub error_message: Option<String>,
    pub chunk_count: i32,
    pub metadata: serde_json::Value,
    pub created_at: String,
}

impl From<crate::entities::source::Model> for SourceResponse {
    fn from(s: crate::entities::source::Model) -> Self {
        Self {
            id: s.id,
            notebook_id: s.notebook_id,
            title: s.title,
            source_type: s.source_type,
            status: s.status,
            error_message: s.error_message,
            chunk_count: s.chunk_count,
            metadata: s.metadata,
            created_at: s.created_at.to_rfc3339(),
        }
    }
}

/// Response for listing sources.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct SourcesListResponse {
    pub sources: Vec<SourceResponse>,
}

/// Response for listing chunks.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ChunksListResponse {
    pub chunks: Vec<ChunkResponse>,
}

/// Response for a chunk.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ChunkResponse {
    pub id: Uuid,
    pub chunk_index: i32,
    pub content: String,
}

// =============================================================================
// REQUEST TYPES
// =============================================================================

/// Request for creating a source via JSON.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct CreateSourceRequest {
    pub title: String,
    pub source_type: String,
    pub content: Option<String>,
    pub url: Option<String>,
}

// =============================================================================
// HANDLERS
// =============================================================================

/// GET /api/notebooks/:notebook_id/sources - List all sources for a notebook.
#[utoipa::path(
    get,
    path = "/api/notebooks/{notebook_id}/sources",
    tag = "sources",
    params(("notebook_id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "The notebook's sources", body = SourcesListResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn list_sources_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
) -> Result<Json<SourcesListResponse>, AppError> {
    verify_notebook_access(state.repos.notebooks.as_ref(), &principal, notebook_id).await?;

    let sources = list_sources(state.repos.sources.as_ref(), notebook_id).await?;
    let sources = sources.into_iter().map(Into::into).collect();

    Ok(Json(SourcesListResponse { sources }))
}

/// POST /api/notebooks/:notebook_id/sources - Create a new source.
///
/// Supports JSON body for text/url sources and multipart for file uploads.
#[utoipa::path(
    post,
    path = "/api/notebooks/{notebook_id}/sources",
    tag = "sources",
    params(("notebook_id" = uuid::Uuid, Path, description = "Notebook ID")),
    request_body = CreateSourceRequest,
    responses(
        (status = 200, description = "The created source, queued for processing", body = SourceResponse),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 403, description = "Denied by the entitlement policy", body = crate::error::ProblemDetails),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn create_source_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    Json(payload): Json<CreateSourceRequest>,
) -> Result<Json<SourceResponse>, AppError> {
    verify_notebook_access(state.repos.notebooks.as_ref(), &principal, notebook_id).await?;

    let source_type = SourceType::try_from(payload.source_type.as_str())?;
    // Authorize before creating anything: a denied source creates no row and
    // spawns no processing task.
    state
        .entitlements
        .authorize(AuthorizationRequest::new(
            &principal,
            Operation::CreateSource {
                notebook_id,
                source_type,
            },
            Uuid::new_v4(),
        ))
        .await?;

    validate_title(&payload.title)?;
    let max_size = state.config.security.max_source_size_bytes;
    let (content, metadata) = extract_and_validate_content(source_type, &payload, max_size)?;

    let source = create_source(
        state.repos.sources.as_ref(),
        notebook_id,
        payload.title.trim().to_string(),
        source_type,
        content,
        metadata,
    )
    .await?;

    spawn_source_processing(&state, source.id, notebook_id, source_type, &principal);

    Ok(Json(source.into()))
}

// =============================================================================
// EXTRACTED HELPERS
// =============================================================================

/// Extract and validate content from the request payload based on source type.
///
/// Returns `(content, metadata)` where:
/// - Web sources: empty content + URL metadata (after SSRF validation)
/// - File sources (PDF/DOCX/EPUB): base64 content (after size + magic byte validation)
/// - Text/Markdown: raw content (after size validation)
fn extract_and_validate_content(
    source_type: SourceType,
    payload: &CreateSourceRequest,
    max_size: usize,
) -> Result<(String, Option<serde_json::Value>), AppError> {
    match source_type {
        SourceType::Web => extract_web_content(payload),
        SourceType::Youtube => extract_youtube_content(payload),
        SourceType::Pdf | SourceType::Docx | SourceType::Epub => {
            extract_file_content(source_type, payload, max_size)
        }
        SourceType::Text | SourceType::Markdown => extract_text_content(payload, max_size),
    }
}

/// Extract and validate web source: require URL, check SSRF.
fn extract_web_content(
    payload: &CreateSourceRequest,
) -> Result<(String, Option<serde_json::Value>), AppError> {
    let url = payload
        .url
        .as_deref()
        .ok_or_else(|| AppError::Validation("URL required for web sources".into()))?;
    validate_url_for_ssrf(url)?;
    Ok((String::new(), Some(json!({ "url": url }))))
}

/// Extract and validate YouTube source: require URL, validate YouTube domain, extract video ID.
fn extract_youtube_content(
    payload: &CreateSourceRequest,
) -> Result<(String, Option<serde_json::Value>), AppError> {
    let url = payload
        .url
        .as_deref()
        .ok_or_else(|| AppError::Validation("URL required for YouTube sources".into()))?;
    validate_url_for_ssrf(url)?;
    let video_id = crate::clients::youtube::extract_youtube_video_id(url)?;
    Ok((
        String::new(),
        Some(json!({ "url": url, "video_id": video_id })),
    ))
}

/// Extract and validate file source: require base64 content, check size and magic bytes.
fn extract_file_content(
    source_type: SourceType,
    payload: &CreateSourceRequest,
    max_size: usize,
) -> Result<(String, Option<serde_json::Value>), AppError> {
    let content = payload.content.as_ref().ok_or_else(|| {
        AppError::Validation("File content (base64) required for file sources".into())
    })?;

    // Validate base64 size (base64 is ~33% larger than raw)
    if content.len() > max_size * 4 / 3 + 4 {
        return Err(AppError::Validation("File too large".into()));
    }

    let decoded = BASE64
        .decode(content)
        .map_err(|_| AppError::Validation("Invalid base64 content".into()))?;

    match source_type {
        SourceType::Pdf => validate_pdf_magic_bytes(&decoded)?,
        SourceType::Docx | SourceType::Epub => validate_zip_magic_bytes(&decoded)?,
        _ => unreachable!(),
    }

    Ok((content.clone(), None))
}

/// Extract and validate text/markdown source: require content, check size.
fn extract_text_content(
    payload: &CreateSourceRequest,
    max_size: usize,
) -> Result<(String, Option<serde_json::Value>), AppError> {
    let content = payload
        .content
        .as_ref()
        .ok_or_else(|| AppError::Validation("Content required for text/markdown sources".into()))?;

    validate_string(content, max_size, "Content")?;
    Ok((content.clone(), None))
}

/// GET /api/sources/:id - Get a single source.
#[utoipa::path(
    get,
    path = "/api/sources/{id}",
    tag = "sources",
    params(("id" = uuid::Uuid, Path, description = "Source ID")),
    responses(
        (status = 200, description = "The source", body = SourceResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_source_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(source_id): Path<Uuid>,
) -> Result<Json<SourceResponse>, AppError> {
    let source = verify_source_access(state.repos.sources.as_ref(), &principal, source_id).await?;
    Ok(Json(source.into()))
}

/// GET /api/sources/:id/chunks - Get all chunks for a source.
#[utoipa::path(
    get,
    path = "/api/sources/{id}/chunks",
    tag = "sources",
    params(("id" = uuid::Uuid, Path, description = "Source ID")),
    responses(
        (status = 200, description = "The source's indexed chunks", body = ChunksListResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_source_chunks_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(source_id): Path<Uuid>,
) -> Result<Json<ChunksListResponse>, AppError> {
    verify_source_access(state.repos.sources.as_ref(), &principal, source_id).await?;

    let chunks = crate::repositories::ChunkRepository::get_for_source(
        state.repos.chunks.as_ref(),
        source_id,
    )
    .await?
    .into_iter()
    .map(|(id, chunk_index, content)| ChunkResponse {
        id,
        chunk_index,
        content,
    })
    .collect();

    Ok(Json(ChunksListResponse { chunks }))
}

/// DELETE /api/sources/:id - Delete a source.
#[utoipa::path(
    delete,
    path = "/api/sources/{id}",
    tag = "sources",
    params(("id" = uuid::Uuid, Path, description = "Source ID")),
    responses(
        (status = 200, description = "Deletion acknowledgement", body = serde_json::Value),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn delete_source_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(source_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    verify_source_access(state.repos.sources.as_ref(), &principal, source_id).await?;
    delete_source(
        state.repos.sources.as_ref(),
        source_id,
        principal.account_id,
    )
    .await?;

    Ok(success_response("Source deleted successfully"))
}

/// POST /api/sources/:id/reprocess - Reprocess a source.
#[utoipa::path(
    post,
    path = "/api/sources/{id}/reprocess",
    tag = "sources",
    params(("id" = uuid::Uuid, Path, description = "Source ID")),
    responses(
        (status = 200, description = "The source, re-queued for processing", body = SourceResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn reprocess_source_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(source_id): Path<Uuid>,
) -> Result<Json<SourceResponse>, AppError> {
    let source = verify_source_access(state.repos.sources.as_ref(), &principal, source_id).await?;

    let updated = update_source_status(
        state.repos.sources.as_ref(),
        source_id,
        SourceStatus::Pending,
        None,
    )
    .await?;
    state
        .source_broadcaster
        .broadcast_status(source.notebook_id, source_id, "pending", None);

    let source_type = SourceType::try_from(source.source_type.as_str())?;
    spawn_source_processing(
        &state,
        source_id,
        source.notebook_id,
        source_type,
        &principal,
    );

    Ok(Json(updated.into()))
}

// =============================================================================
// SOURCE PROCESSING
// =============================================================================

/// Maximum wall-clock time for a single source processing pipeline (10 minutes).
///
/// This prevents truly stuck processing from hanging forever. A 60-page PDF
/// typically completes in under 2 minutes; this limit allows ample headroom
/// for very large documents and rate-limited embedding batches.
const PROCESSING_TIMEOUT: Duration = Duration::from_secs(600);

/// Spawn async source processing task tracked for graceful shutdown.
///
/// Uses `task_tracker.spawn()` so the server waits for in-progress processing
/// before shutting down. If the task is cancelled by shutdown, the source is
/// marked as error so the user can reprocess it.
fn spawn_source_processing(
    state: &CoreState,
    source_id: Uuid,
    notebook_id: Uuid,
    source_type: SourceType,
    principal: &Principal,
) {
    use crate::services::source_processing::{ProcessingDeps, process_source};
    use crate::types::RequestContext;

    let deps = ProcessingDeps::from_state(state, principal.clone());
    let source_repo = state.repos.sources.clone();
    let broadcaster = state.source_broadcaster.clone();

    // Dependencies for post-processing suggestion generation
    let mistral = state.clients.mistral.clone();
    let chunk_repo = state.repos.chunks.clone();
    let notebook_repo = state.repos.notebooks.clone();

    // Capture the current request context so the spawned task inherits
    // the request_id for end-to-end tracing of downstream HTTP calls.
    let ctx = RequestContext::current();

    state.task_tracker.spawn("source-processing", async move {
        let result = tokio::time::timeout(
            PROCESSING_TIMEOUT,
            ctx.scope(process_source(deps, source_id, notebook_id, source_type)),
        )
        .await;
        match result {
            Ok(Ok(())) => {
                // Generate and cache suggested questions post-indexation.
                // Fire-and-forget: errors are logged but don't affect the source status.
                if let Err(e) = crate::services::suggestions::generate_and_store(
                    mistral.as_ref(),
                    chunk_repo.as_ref(),
                    notebook_repo.as_ref(),
                    notebook_id,
                )
                .await
                {
                    warn!(
                        error = %e,
                        %notebook_id,
                        "Failed to generate suggested questions post-indexation"
                    );
                }
            }
            Ok(Err(e)) => {
                error!(source_id = %source_id, error = %e, "Source processing failed");
            }
            Err(_elapsed) => {
                error!(
                    source_id = %source_id,
                    timeout_secs = PROCESSING_TIMEOUT.as_secs(),
                    "Source processing timed out"
                );
                let _ = update_source_status(
                    source_repo.as_ref(),
                    source_id,
                    SourceStatus::Error,
                    Some(
                        "Processing timed out — please try again or use a smaller document".into(),
                    ),
                )
                .await;
                broadcaster.broadcast_error(
                    notebook_id,
                    source_id,
                    "Processing timed out — please try again or use a smaller document",
                );
            }
        }
    });
}

// =============================================================================
// SSE ENDPOINT (R2.1)
// =============================================================================

/// GET /api/notebooks/:notebook_id/sources/events - Stream source status updates.
///
/// Events: `source:status`, `source:ready`, `source:error`, `source:resync`
///
/// Supports `Last-Event-ID` header for replay on reconnection. If provided,
/// all buffered events after that ID are replayed before the live stream.
#[utoipa::path(
    get,
    path = "/api/notebooks/{notebook_id}/sources/events",
    tag = "sources",
    params(("notebook_id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "Server-sent event stream. Framing, replay and `Last-Event-ID` semantics are specified in docs/contracts/sse-protocol-v1.md, which OpenAPI does not model.", content_type = "text/event-stream", body = crate::core::protocol::SourceEvent),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn source_events_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    headers: HeaderMap,
) -> Result<Response, AppError> {
    verify_notebook_access(state.repos.notebooks.as_ref(), &principal, notebook_id).await?;

    let last_event_id: Option<u64> = headers
        .get("Last-Event-ID")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse().ok());

    info!(
        notebook_id = %notebook_id,
        user_id = %principal.account_id,
        ?last_event_id,
        "SSE client connected"
    );

    // Build replay stream from buffered events (if reconnecting)
    let replay_events = last_event_id
        .map(|id| state.source_broadcaster.replay_since(notebook_id, id))
        .unwrap_or_default();

    let replay_stream = stream::iter(replay_events).filter_map(move |(event_id, event)| {
        let nb = notebook_id;
        async move { serialize_sse_event(&event, event_id, nb) }
    });

    // Live stream from broadcast channel
    let rx = state.source_broadcaster.subscribe(notebook_id);
    let live_stream = BroadcastStream::new(rx).filter_map(move |result| async move {
        match result {
            Ok((event_id, event)) => serialize_sse_event(&event, event_id, notebook_id),
            Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                warn!(notebook_id = %notebook_id, missed = n, "SSE client lagged");
                // Stream-level and per-subscriber: it carries no event ID, so a
                // reconnect cannot replay it.
                let event = SourceEvent::resync(n);
                let data = event.payload().ok()?.to_string();
                Some(Ok(Event::default().event(event.event_type()).data(data)))
            }
        }
    });

    let stream = replay_stream.chain(live_stream);

    let sse = Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(SSE_KEEPALIVE_SECS))
            .text("keep-alive"),
    );

    Ok(sse.into_response())
}

/// Serialize a source event to SSE format with an event ID.
///
/// The typed [`SourceEvent`] is the single authority for the wire form
/// (US-009). This function only frames it.
fn serialize_sse_event(
    event: &SourceEvent,
    event_id: u64,
    notebook_id: Uuid,
) -> Option<Result<Event, Infallible>> {
    match event
        .payload()
        .and_then(|data| serde_json::to_string(&data))
    {
        Ok(json) => Some(Ok(Event::default()
            .event(event.event_type())
            .data(json)
            .id(event_id.to_string()))),
        Err(e) => {
            error!(notebook_id = %notebook_id, error = %e, "Failed to serialize SSE event");
            None
        }
    }
}

// =============================================================================
// YouTube oEmbed title lookup
// =============================================================================

/// Query parameters for the YouTube title lookup endpoint.
#[derive(Debug, Deserialize, utoipa::ToSchema, utoipa::IntoParams)]
pub struct YouTubeTitleQuery {
    pub url: String,
}

/// Response for the YouTube title lookup.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct YouTubeTitleResponse {
    pub title: String,
    pub author: String,
}

/// GET /api/youtube/title?url=... — Fetch video title via YouTube oEmbed.
///
/// Lightweight proxy to avoid CORS issues. Returns `{ title, author }`.
/// Used by the frontend to auto-fill the source title on paste.
#[utoipa::path(
    get,
    path = "/api/youtube/title",
    tag = "sources",
    params(YouTubeTitleQuery),
    responses(
        (status = 200, description = "The video title and author", body = YouTubeTitleResponse),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn youtube_title_handler(
    State(_state): State<CoreState>,
    Extension(_principal): Extension<Principal>,
    axum::extract::Query(query): axum::extract::Query<YouTubeTitleQuery>,
) -> Result<Json<YouTubeTitleResponse>, AppError> {
    let video_id = crate::clients::youtube::extract_youtube_video_id(&query.url)?;
    let canonical_url = format!("https://www.youtube.com/watch?v={video_id}");
    let (title, author) = crate::clients::YouTubeClient::fetch_oembed_title(&canonical_url).await?;
    Ok(Json(YouTubeTitleResponse { title, author }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_pdf_bytes_pass_validation() {
        let pdf = b"%PDF-1.7 some content";
        assert!(validate_pdf_magic_bytes(pdf).is_ok());
    }

    #[test]
    fn minimal_pdf_header_passes() {
        assert!(validate_pdf_magic_bytes(b"%PDF-").is_ok());
    }

    #[test]
    fn png_header_rejected() {
        // PNG magic bytes: 0x89 0x50 0x4E 0x47
        let png = b"\x89PNG\r\n\x1a\n";
        let err = validate_pdf_magic_bytes(png).unwrap_err();
        assert!(
            err.to_string().contains("%PDF-"),
            "Error should mention expected header"
        );
    }

    #[test]
    fn plain_text_rejected() {
        let text = b"Hello, this is not a PDF";
        assert!(validate_pdf_magic_bytes(text).is_err());
    }

    #[test]
    fn empty_content_rejected() {
        assert!(validate_pdf_magic_bytes(b"").is_err());
    }

    #[test]
    fn too_short_content_rejected() {
        assert!(validate_pdf_magic_bytes(b"%PD").is_err());
    }

    #[test]
    fn valid_zip_bytes_pass_validation() {
        let zip = b"\x50\x4B\x03\x04 some content";
        assert!(validate_zip_magic_bytes(zip).is_ok());
    }

    #[test]
    fn minimal_zip_header_passes() {
        assert!(validate_zip_magic_bytes(&[0x50, 0x4B, 0x03, 0x04]).is_ok());
    }

    #[test]
    fn pdf_header_rejected_as_zip() {
        let pdf = b"%PDF-1.7";
        let err = validate_zip_magic_bytes(pdf).unwrap_err();
        assert!(
            err.to_string().contains("ZIP"),
            "Error should mention expected header"
        );
    }

    #[test]
    fn empty_content_rejected_as_zip() {
        assert!(validate_zip_magic_bytes(b"").is_err());
    }

    #[test]
    fn too_short_content_rejected_as_zip() {
        assert!(validate_zip_magic_bytes(b"PK").is_err());
    }

    #[test]
    fn zip_magic_bytes_constant_is_correct() {
        assert_eq!(ZIP_MAGIC_BYTES, &[0x50, 0x4B, 0x03, 0x04]);
    }

    // --- US-008: serialize_sse_event for OCR events ---

    #[test]
    fn serialize_ocr_started_event() {
        let source_id = Uuid::nil();
        let event = SourceEvent::ocr_started(source_id, 15);
        let result = serialize_sse_event(&event, 1, Uuid::nil());
        assert!(result.is_some(), "OCR started event should serialize");
    }

    #[test]
    fn serialize_ocr_progress_event() {
        let source_id = Uuid::nil();
        let event = SourceEvent::ocr_progress(source_id, 5, 15);
        let result = serialize_sse_event(&event, 2, Uuid::nil());
        assert!(result.is_some(), "OCR progress event should serialize");
    }

    #[test]
    fn serialize_ocr_completed_event() {
        let source_id = Uuid::nil();
        let event = SourceEvent::ocr_completed(source_id, 15);
        let result = serialize_sse_event(&event, 3, Uuid::nil());
        assert!(result.is_some(), "OCR completed event should serialize");
    }

    #[test]
    fn serialize_ocr_cache_hit_event() {
        let source_id = Uuid::nil();
        let event = SourceEvent::ocr_cache_hit(source_id);
        let result = serialize_sse_event(&event, 4, Uuid::nil());
        assert!(result.is_some(), "OCR cache hit event should serialize");
    }
}
