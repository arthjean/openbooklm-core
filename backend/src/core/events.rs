//! Domain events for non-critical side effects (US-008, EP-002).
//!
//! RAG, ingestion and chat used to call PostHog and Resend inline. They now
//! describe what happened and hand it to the injected [`EventSink`]. The public
//! core ships a no-op sink and a tracing sink; the private SaaS process
//! subscribes with its own consumer and translates what it cares about into
//! analytics and lifecycle email.
//!
//! # Delivery contract
//!
//! [`EventSink::emit`] is synchronous, non-blocking and infallible from the
//! caller's point of view. An implementation must enqueue and return: it may
//! not await I/O, take a contended lock or apply backpressure. Core request
//! latency therefore cannot be affected by a consumer's availability, which is
//! how the 50 ms ceiling in the PRD is met — the real cost is a bounded-channel
//! `try_send`. A sink whose queue is full drops the event and records the drop;
//! it never blocks the core and never fails the core result.
//!
//! Events carry internal account and resource identifiers plus operational
//! metrics. They must never carry an email address, an identity-provider
//! subject, a billing identifier or raw document content.

use std::sync::Arc;

use uuid::Uuid;

use crate::types::SourceType;

// ============================================================================
// Events
// ============================================================================

/// Something a core pipeline finished doing.
#[derive(Debug, Clone)]
pub enum DomainEvent {
    /// A source finished the ingestion pipeline successfully.
    SourceProcessingCompleted {
        account_id: Uuid,
        notebook_id: Uuid,
        source_id: Uuid,
        source_type: SourceType,
        duration_ms: u64,
        chunk_count: i32,
    },
    /// A source failed the ingestion pipeline.
    SourceProcessingFailed {
        account_id: Uuid,
        notebook_id: Uuid,
        source_id: Uuid,
        source_type: SourceType,
        /// Short category, never the raw error message.
        error_type: &'static str,
        duration_ms: u64,
    },
    /// A source completed with one or more capabilities unavailable
    /// (for example OCR skipped because the account's allowance ran out).
    SourceProcessingDegraded {
        account_id: Uuid,
        notebook_id: Uuid,
        source_id: Uuid,
        degraded_services: Vec<&'static str>,
    },
    /// A chat exchange produced a complete assistant response.
    ChatCompleted {
        account_id: Uuid,
        notebook_id: Uuid,
        provider: String,
        duration_ms: u64,
        context_chunks: usize,
    },
    /// A chat exchange terminated without a complete assistant response.
    ChatFailed {
        account_id: Uuid,
        notebook_id: Uuid,
        provider: String,
        /// Short category, never the raw error message.
        error_type: &'static str,
    },
}

impl DomainEvent {
    /// Short stable label used in logs and metrics.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::SourceProcessingCompleted { .. } => "source_processing_completed",
            Self::SourceProcessingFailed { .. } => "source_processing_failed",
            Self::SourceProcessingDegraded { .. } => "source_processing_degraded",
            Self::ChatCompleted { .. } => "chat_completed",
            Self::ChatFailed { .. } => "chat_failed",
        }
    }

    /// The account the event belongs to.
    #[must_use]
    pub fn account_id(&self) -> Uuid {
        match self {
            Self::SourceProcessingCompleted { account_id, .. }
            | Self::SourceProcessingFailed { account_id, .. }
            | Self::SourceProcessingDegraded { account_id, .. }
            | Self::ChatCompleted { account_id, .. }
            | Self::ChatFailed { account_id, .. } => *account_id,
        }
    }
}

// ============================================================================
// Sink
// ============================================================================

/// Receives domain events. Implementations must not block the caller.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: DomainEvent);
}

/// Shared handle to the configured sink.
pub type SharedEventSink = Arc<dyn EventSink>;

/// Discards every event. The default for a self-hosted deployment.
#[derive(Debug, Clone, Default)]
pub struct NoopEventSink;

impl EventSink for NoopEventSink {
    fn emit(&self, _event: DomainEvent) {}
}

/// Writes every event to the tracing subscriber. Useful for self-hosted
/// operators who want the same signal without an analytics vendor.
#[derive(Debug, Clone, Default)]
pub struct TracingEventSink;

impl EventSink for TracingEventSink {
    fn emit(&self, event: DomainEvent) {
        tracing::info!(event = event.label(), account_id = %event.account_id(), detail = ?event, "domain_event");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct RecordingSink {
        seen: Mutex<Vec<&'static str>>,
    }

    impl EventSink for RecordingSink {
        fn emit(&self, event: DomainEvent) {
            self.seen.lock().expect("sink mutex").push(event.label());
        }
    }

    fn sample_events(account_id: Uuid) -> Vec<DomainEvent> {
        vec![
            DomainEvent::SourceProcessingCompleted {
                account_id,
                notebook_id: Uuid::new_v4(),
                source_id: Uuid::new_v4(),
                source_type: SourceType::Pdf,
                duration_ms: 12,
                chunk_count: 4,
            },
            DomainEvent::SourceProcessingFailed {
                account_id,
                notebook_id: Uuid::new_v4(),
                source_id: Uuid::new_v4(),
                source_type: SourceType::Web,
                error_type: "extraction_error",
                duration_ms: 3,
            },
            DomainEvent::SourceProcessingDegraded {
                account_id,
                notebook_id: Uuid::new_v4(),
                source_id: Uuid::new_v4(),
                degraded_services: vec!["ocr"],
            },
            DomainEvent::ChatCompleted {
                account_id,
                notebook_id: Uuid::new_v4(),
                provider: "anthropic".into(),
                duration_ms: 900,
                context_chunks: 8,
            },
            DomainEvent::ChatFailed {
                account_id,
                notebook_id: Uuid::new_v4(),
                provider: "mistral".into(),
                error_type: "timeout",
            },
        ]
    }

    #[test]
    fn every_event_reports_its_account_and_label() {
        let account_id = Uuid::new_v4();
        for event in sample_events(account_id) {
            assert_eq!(event.account_id(), account_id);
            assert!(!event.label().is_empty());
        }
    }

    #[test]
    fn noop_sink_accepts_every_event() {
        let sink = NoopEventSink;
        for event in sample_events(Uuid::new_v4()) {
            sink.emit(event);
        }
    }

    #[test]
    fn sink_receives_every_emitted_event() {
        let sink = RecordingSink::default();
        let events = sample_events(Uuid::new_v4());
        let expected: Vec<&'static str> = events.iter().map(DomainEvent::label).collect();
        for event in events {
            sink.emit(event);
        }
        assert_eq!(*sink.seen.lock().expect("sink mutex"), expected);
    }
}
