//! Typed source processing events (US-009).
//!
//! Moved here from `services::source_events`, which keeps the broadcaster and
//! re-exports these types so existing import paths still resolve. The split is
//! deliberate: the broadcaster is runtime machinery, these types are the
//! contract.
//!
//! Before US-009 this enum had two serializations. The derive omitted
//! `error_message`, `progress` and an empty `degraded_services`; the SSE
//! handler hand-built a JSON object per variant and always wrote every key.
//! Clients parse the handler's form, so that is the one that survives: the
//! `skip_serializing_if` attributes are gone and the handler now uses the
//! derive. See `docs/contracts/known-drift.md` D-005.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Embedding progress for real-time UI updates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct EmbeddingProgress {
    pub chunks_done: u32,
    pub chunks_total: u32,
}

/// Data for source status change events (boxed to reduce enum size disparity).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SourceStatusData {
    pub source_id: Uuid,
    pub status: String,
    pub error_message: Option<String>,
    pub progress: Option<EmbeddingProgress>,
}

/// Events emitted during source processing.
///
/// Clients must ignore unknown variants without terminating the stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "event", content = "data")]
pub enum SourceEvent {
    /// Source status has changed.
    #[serde(rename = "source:status")]
    Status(Box<SourceStatusData>),

    /// Source processing is complete.
    #[serde(rename = "source:ready")]
    Ready {
        source_id: Uuid,
        chunk_count: i32,
        /// Services that were unavailable during processing (e.g., "contextualization").
        /// Empty when all services operated normally.
        degraded_services: Vec<String>,
    },

    /// Source processing failed.
    #[serde(rename = "source:error")]
    Error { source_id: Uuid, message: String },

    /// OCR extraction has started for a PDF source.
    #[serde(rename = "source:ocr_started")]
    OcrStarted { source_id: Uuid, total_pages: u32 },

    /// OCR page processing progress update.
    #[serde(rename = "source:ocr_progress")]
    OcrProgress {
        source_id: Uuid,
        current_page: u32,
        total_pages: u32,
    },

    /// OCR extraction completed successfully.
    #[serde(rename = "source:ocr_completed")]
    OcrCompleted {
        source_id: Uuid,
        pages_processed: u32,
    },

    /// OCR result served from cache — no API call needed.
    #[serde(rename = "source:ocr_cache_hit")]
    OcrCacheHit { source_id: Uuid },

    /// The replay buffer could not satisfy `Last-Event-ID`, or the subscriber
    /// lagged behind the broadcast channel. `missed` events were lost and the
    /// client must refetch the source list to resynchronize.
    ///
    /// Produced at the transport edge, never broadcast: it is per-subscriber.
    #[serde(rename = "source:resync")]
    Resync { missed: u64 },
}

impl SourceEvent {
    /// Create a status change event.
    #[must_use]
    pub fn status(
        source_id: Uuid,
        status: impl Into<String>,
        error_message: Option<String>,
    ) -> Self {
        Self::Status(Box::new(SourceStatusData {
            source_id,
            status: status.into(),
            error_message,
            progress: None,
        }))
    }

    /// Create a status change event with embedding progress.
    #[must_use]
    pub fn status_with_progress(
        source_id: Uuid,
        status: impl Into<String>,
        chunks_done: u32,
        chunks_total: u32,
    ) -> Self {
        Self::Status(Box::new(SourceStatusData {
            source_id,
            status: status.into(),
            error_message: None,
            progress: Some(EmbeddingProgress {
                chunks_done,
                chunks_total,
            }),
        }))
    }

    /// Create a ready event.
    #[must_use]
    pub const fn ready(source_id: Uuid, chunk_count: i32) -> Self {
        Self::Ready {
            source_id,
            chunk_count,
            degraded_services: Vec::new(),
        }
    }

    /// Create a ready event with degraded services information.
    #[must_use]
    pub const fn ready_degraded(
        source_id: Uuid,
        chunk_count: i32,
        degraded_services: Vec<String>,
    ) -> Self {
        Self::Ready {
            source_id,
            chunk_count,
            degraded_services,
        }
    }

    /// Create an error event.
    #[must_use]
    pub fn error(source_id: Uuid, message: impl Into<String>) -> Self {
        Self::Error {
            source_id,
            message: message.into(),
        }
    }

    /// Create an OCR started event.
    #[must_use]
    pub const fn ocr_started(source_id: Uuid, total_pages: u32) -> Self {
        Self::OcrStarted {
            source_id,
            total_pages,
        }
    }

    /// Create an OCR progress event.
    #[must_use]
    pub const fn ocr_progress(source_id: Uuid, current_page: u32, total_pages: u32) -> Self {
        Self::OcrProgress {
            source_id,
            current_page,
            total_pages,
        }
    }

    /// Create an OCR completed event.
    #[must_use]
    pub const fn ocr_completed(source_id: Uuid, pages_processed: u32) -> Self {
        Self::OcrCompleted {
            source_id,
            pages_processed,
        }
    }

    /// Create an OCR cache hit event.
    #[must_use]
    pub const fn ocr_cache_hit(source_id: Uuid) -> Self {
        Self::OcrCacheHit { source_id }
    }

    /// Create a resync event.
    #[must_use]
    pub const fn resync(missed: u64) -> Self {
        Self::Resync { missed }
    }

    /// The source this event concerns, or `None` for stream-level events.
    #[must_use]
    pub const fn source_id(&self) -> Option<Uuid> {
        match self {
            Self::Status(data) => Some(data.source_id),
            Self::Ready { source_id, .. }
            | Self::Error { source_id, .. }
            | Self::OcrStarted { source_id, .. }
            | Self::OcrProgress { source_id, .. }
            | Self::OcrCompleted { source_id, .. }
            | Self::OcrCacheHit { source_id, .. } => Some(*source_id),
            Self::Resync { .. } => None,
        }
    }

    /// SSE event name, matching the serde tag.
    #[must_use]
    pub const fn event_type(&self) -> &'static str {
        match self {
            Self::Status(_) => "source:status",
            Self::Ready { .. } => "source:ready",
            Self::Error { .. } => "source:error",
            Self::OcrStarted { .. } => "source:ocr_started",
            Self::OcrProgress { .. } => "source:ocr_progress",
            Self::OcrCompleted { .. } => "source:ocr_completed",
            Self::OcrCacheHit { .. } => "source:ocr_cache_hit",
            Self::Resync { .. } => "source:resync",
        }
    }

    /// The payload half of the event, as it appears on the SSE `data:` line.
    pub fn payload(&self) -> Result<serde_json::Value, serde_json::Error> {
        match self {
            Self::Status(data) => serde_json::to_value(data),
            Self::Ready {
                source_id,
                chunk_count,
                degraded_services,
            } => serde_json::to_value(serde_json::json!({
                "source_id": source_id,
                "chunk_count": chunk_count,
                "degraded_services": degraded_services,
            })),
            Self::Error { source_id, message } => serde_json::to_value(serde_json::json!({
                "source_id": source_id,
                "message": message,
            })),
            Self::OcrStarted {
                source_id,
                total_pages,
            } => serde_json::to_value(serde_json::json!({
                "source_id": source_id,
                "total_pages": total_pages,
            })),
            Self::OcrProgress {
                source_id,
                current_page,
                total_pages,
            } => serde_json::to_value(serde_json::json!({
                "source_id": source_id,
                "current_page": current_page,
                "total_pages": total_pages,
            })),
            Self::OcrCompleted {
                source_id,
                pages_processed,
            } => serde_json::to_value(serde_json::json!({
                "source_id": source_id,
                "pages_processed": pages_processed,
            })),
            Self::OcrCacheHit { source_id } => serde_json::to_value(serde_json::json!({
                "source_id": source_id,
            })),
            Self::Resync { missed } => serde_json::to_value(serde_json::json!({
                "missed": missed,
            })),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_type_matches_the_serde_tag() {
        let events = [
            SourceEvent::status(Uuid::nil(), "processing", None),
            SourceEvent::ready(Uuid::nil(), 3),
            SourceEvent::error(Uuid::nil(), "boom"),
            SourceEvent::ocr_started(Uuid::nil(), 4),
            SourceEvent::ocr_progress(Uuid::nil(), 1, 4),
            SourceEvent::ocr_completed(Uuid::nil(), 4),
            SourceEvent::ocr_cache_hit(Uuid::nil()),
            SourceEvent::resync(7),
        ];
        for event in events {
            let encoded = serde_json::to_value(&event).expect("serialize");
            assert_eq!(encoded["event"], event.event_type());
            assert_eq!(encoded["data"], event.payload().expect("payload"));
        }
    }

    #[test]
    fn optional_status_fields_stay_on_the_wire_as_null() {
        let payload = SourceEvent::status(Uuid::nil(), "processing", None)
            .payload()
            .expect("payload");
        assert!(
            payload
                .get("error_message")
                .is_some_and(serde_json::Value::is_null)
        );
        assert!(
            payload
                .get("progress")
                .is_some_and(serde_json::Value::is_null)
        );
    }

    #[test]
    fn an_empty_degraded_services_list_is_still_emitted() {
        let payload = SourceEvent::ready(Uuid::nil(), 3)
            .payload()
            .expect("payload");
        assert_eq!(payload["degraded_services"], serde_json::json!([]));
    }

    #[test]
    fn resync_is_a_stream_level_event() {
        assert_eq!(SourceEvent::resync(7).source_id(), None);
    }
}
