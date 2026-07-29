//! The chat SSE transport adapter.
//!
//! This module is the *only* place that knows chat events travel over SSE
//! (US-009). Everything upstream of it produces [`ChatEvent`] values and has no
//! opinion on framing, heartbeats or headers.

use std::collections::HashSet;

use axum::http::HeaderValue;
use axum::response::sse;

use crate::core::protocol::ChatEvent;
use crate::llm::citations::find_code_ranges;

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

// ============================================================================
// Citation detection
// ============================================================================

/// Detect new `[N]` citation patterns in text and return their indices.
///
/// Skips matches inside inline code and fenced code blocks.
#[allow(clippy::indexing_slicing)] // all accesses guarded by `i < bytes.len()` checks
pub fn detect_new_citations(text: &str, sent: &mut HashSet<usize>) -> Vec<usize> {
    let code_ranges = find_code_ranges(text);
    let mut new_citations = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'[' {
            let bracket_start = i;
            let start = i + 1;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i < bytes.len()
                && bytes[i] == b']'
                && i > start
                && !code_ranges
                    .iter()
                    .any(|&(cs, ce)| bracket_start >= cs && bracket_start < ce)
                && let Ok(n) = text[start..i].parse::<usize>()
                && sent.insert(n)
            {
                new_citations.push(n);
            }
        }
        i += 1;
    }

    new_citations
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_citations_basic() {
        let mut sent = HashSet::new();
        let result = detect_new_citations("See [1] and [2]", &mut sent);
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn detect_citations_skips_duplicates() {
        let mut sent = HashSet::new();
        let result = detect_new_citations("[1] and again [1]", &mut sent);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn detect_citations_skips_code() {
        let mut sent = HashSet::new();
        let result = detect_new_citations("Use `array[1]` for access [2].", &mut sent);
        assert_eq!(result, vec![2]);
    }
}
