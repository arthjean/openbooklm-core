//! Core HTTP composition (US-013, EP-004).
//!
//! [`build_core_router`] returns every protected core route as an Axum 0.8
//! `Router<CoreState>` with **no** state applied and **no** identity middleware
//! attached. The composition root supplies both:
//!
//! - the reference server ([`crate::bin`] `openbooklm-server`) layers the
//!   static-token or loopback adapter from [`crate::core::identity`];
//! - the hosted product layers its Clerk middleware.
//!
//! Both must insert a [`Principal`](crate::core::Principal) into the request
//! extensions before the router runs: every handler here extracts one and a
//! missing extension is a 500, not a 401. Keeping the middleware out of this
//! function is what lets one router serve two identity models.
//!
//! [`build_core_health_router`] is separate because health must answer before
//! and without authentication.
//!
//! # One deliberate change from the pre-split composition
//!
//! The request timeout is applied here, around the non-streaming routes, so the
//! identity middleware the caller layers on top now runs *outside* it. Before
//! US-013 the hosted composition layered the timeout outermost, so a slow JWKS
//! refresh counted against the 120 s budget and surfaced as a 408.
//!
//! It cannot be both: the timeout must exclude the SSE routes, which means it
//! belongs to whoever knows which routes stream — this function — while the
//! identity adapter belongs to whoever knows what identity means. The practical
//! effect is that an identity adapter is now responsible for its own bounds. The
//! Clerk adapter's JWKS client has its own timeout; the reference adapter
//! performs no I/O at all.
//!
//! # Cancellation
//!
//! The streaming routes are deliberately excluded from the request timeout: an
//! SSE response is long-lived by construction and carries its own cancellation.
//! When a client disconnects, Axum drops the response body, the stream future
//! is dropped, and the provider request is cancelled with it. Work already
//! spawned on the [`TaskTracker`](crate::middleware::TaskTracker) (source
//! persistence, memory extraction) is not cancelled by a client disconnect.
//! Process shutdown signals and joins that work through the root task scope.
//! This is the behaviour US-002 captured and US-013 preserves.

use std::time::Duration;

use axum::error_handling::HandleErrorLayer;
use axum::{
    BoxError, Json, Router,
    extract::DefaultBodyLimit,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, patch, post},
};
use tower::ServiceBuilder;
use tower::timeout::TimeoutLayer;

use crate::api;
use crate::core::config::CoreConfig;
use crate::core::state::CoreState;
use crate::error::ProblemDetails;

/// Axum body limit for a source upload.
///
/// Files arrive base64-encoded inside a JSON body, which inflates them by about
/// a third, and the JSON wrapper adds a little more. Without the headroom a
/// file of exactly `max_source_size_bytes` would be rejected by the transport
/// before the handler's own size check could produce a domain error.
#[must_use]
pub fn upload_body_limit(config: &CoreConfig) -> usize {
    config.security.upload_limit_bytes * 4 / 3 + 4096
}

/// Every protected core route, without state and without identity middleware.
///
/// The caller applies `.with_state(core_state)` and layers an adapter that
/// inserts a `Principal`.
pub fn build_core_router(config: &CoreConfig) -> Router<CoreState> {
    let request_timeout = Duration::from_secs(config.async_config.request_timeout_secs);

    let non_streaming = core_api_routes(upload_body_limit(config)).layer(
        ServiceBuilder::new()
            .layer(HandleErrorLayer::new(handle_timeout_error))
            .layer(TimeoutLayer::new(request_timeout)),
    );

    Router::new()
        .merge(non_streaming)
        .merge(core_streaming_routes())
}

/// Unauthenticated health routes.
///
/// `/health/detailed` guards itself with `HEALTH_TOKEN` when one is configured;
/// it never depends on the identity adapter.
pub fn build_core_health_router() -> Router<CoreState> {
    Router::new()
        .route("/health", get(api::health::health_check))
        .route("/health/detailed", get(api::health::detailed_health_check))
}

/// Non-streaming protected core routes.
fn core_api_routes(upload_limit: usize) -> Router<CoreState> {
    Router::new()
        // Notebooks
        .route(
            "/api/notebooks",
            get(api::notebooks::list_notebooks_handler)
                .post(api::notebooks::create_notebook_handler),
        )
        .route(
            "/api/notebooks/{id}",
            get(api::notebooks::get_notebook_handler)
                .patch(api::notebooks::update_notebook_handler)
                .delete(api::notebooks::delete_notebook_handler),
        )
        // Memories
        .route(
            "/api/notebooks/{id}/memories",
            get(api::memory::list_memories_handler)
                .delete(api::memory::delete_all_memories_handler),
        )
        .route(
            "/api/memories/{id}",
            get(api::memory::get_memory_handler)
                .patch(api::memory::update_memory_handler)
                .delete(api::memory::delete_memory_handler),
        )
        // Sources (upload route has a higher body limit for file uploads)
        .route(
            "/api/notebooks/{notebook_id}/sources",
            get(api::sources::list_sources_handler)
                .post(api::sources::create_source_handler)
                .layer(DefaultBodyLimit::max(upload_limit)),
        )
        .route(
            "/api/sources/{id}",
            get(api::sources::get_source_handler).delete(api::sources::delete_source_handler),
        )
        .route(
            "/api/sources/{id}/chunks",
            get(api::sources::get_source_chunks_handler),
        )
        .route(
            "/api/sources/{id}/reprocess",
            post(api::sources::reprocess_source_handler),
        )
        // YouTube oEmbed title lookup (auth required — prevents unauthenticated abuse)
        .route(
            "/api/youtube/title",
            get(api::sources::youtube_title_handler),
        )
        // Suggestions
        .route(
            "/api/notebooks/{id}/suggestions",
            get(api::suggestions::get_suggestions_handler),
        )
        // Chat (non-streaming: GET history, DELETE clear)
        .route(
            "/api/notebooks/{id}/chat",
            get(api::chat::get_chat_history_handler).delete(api::chat::clear_chat_history_handler),
        )
        .route("/api/teaching-modes", get(api::chat::list_teaching_modes))
        // Notes
        .route(
            "/api/notebooks/{notebook_id}/notes",
            get(api::notes::list_notes_handler).post(api::notes::create_note_handler),
        )
        .route(
            "/api/notes/{id}",
            get(api::notes::get_note_handler)
                .patch(api::notes::update_note_handler)
                .delete(api::notes::delete_note_handler),
        )
        // RAG logs and metrics
        .route(
            "/api/rag-logs/{id}/feedback",
            patch(api::rag_logs::update_feedback_handler),
        )
        .route(
            "/api/notebooks/{id}/metrics",
            get(api::rag_logs::get_notebook_metrics_handler),
        )
        .route("/api/metrics", get(api::rag_logs::get_user_metrics_handler))
        // Settings
        .route(
            "/api/settings",
            get(api::settings::get_settings_handler).patch(api::settings::update_settings_handler),
        )
}

/// SSE routes — excluded from the global request timeout.
fn core_streaming_routes() -> Router<CoreState> {
    Router::new()
        .route(
            "/api/notebooks/{id}/chat",
            post(api::chat::send_message_handler),
        )
        .route(
            "/api/notebooks/{notebook_id}/sources/events",
            get(api::sources::source_events_handler),
        )
}

/// Converts `413 Payload Too Large` responses produced by [`DefaultBodyLimit`]
/// into RFC 7807 Problem Details JSON responses.
pub async fn map_payload_too_large(response: Response) -> Response {
    if response.status() == StatusCode::PAYLOAD_TOO_LARGE {
        let problem =
            ProblemDetails::payload_too_large("The request body exceeds the maximum allowed size");
        (StatusCode::PAYLOAD_TOO_LARGE, Json(problem)).into_response()
    } else {
        response
    }
}

/// Converts timeout errors from [`TimeoutLayer`] into RFC 7807 responses with
/// HTTP 408.
pub async fn handle_timeout_error(err: BoxError) -> Response {
    if err.is::<tower::timeout::error::Elapsed>() {
        let problem = ProblemDetails::request_timeout(
            "The server did not complete the request within the allowed time",
        );
        (StatusCode::REQUEST_TIMEOUT, Json(problem)).into_response()
    } else {
        tracing::error!(error = %err, "Unhandled middleware error");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ProblemDetails::internal_error("An internal error occurred")),
        )
            .into_response()
    }
}
