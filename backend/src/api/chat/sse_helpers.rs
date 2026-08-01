//! The chat SSE transport adapter.
//!
//! This module is the *only* place that knows chat events travel over SSE
//! (US-009). Everything upstream of it produces [`ChatEvent`] values and has no
//! opinion on framing, heartbeats or headers.

use axum::http::HeaderValue;
use axum::response::sse;

use crate::core::protocol::ChatEvent;

// ============================================================================
// SSE constants
// ============================================================================

/// Keep-alive interval for SSE streams (seconds).
pub const SSE_KEEPALIVE_SECS: u64 = 15;

// ============================================================================
// SSE response helpers
// ============================================================================

/// Apply standard SSE response headers to disable buffering and caching.
pub fn apply_sse_headers(response: &mut axum::response::Response) {
    response
        .headers_mut()
        .insert("X-Accel-Buffering", HeaderValue::from_static("no"));
    response.headers_mut().insert(
        "Cache-Control",
        HeaderValue::from_static("no-cache, no-store"),
    );
}

// ============================================================================
// Transport encoding
// ============================================================================

/// Frame a typed chat event as SSE.
///
/// A serialization failure is reported to the client as an `error` event rather
/// than silently dropping the frame; the payload types make it unreachable
/// today, but a stream that stalls with no explanation is the worse failure.
pub fn chat_event_to_sse(event: &ChatEvent) -> sse::Event {
    match event.payload_json() {
        Ok(json) => sse::Event::default().event(event.name()).data(json),
        Err(e) => {
            tracing::warn!(event = event.name(), error = %e, "Failed to serialize chat event");
            sse::Event::default()
                .event("error")
                .data(r#"{"message":"Internal serialization error"}"#)
        }
    }
}
