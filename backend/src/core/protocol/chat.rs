//! Typed chat stream events (US-009).
//!
//! `ChatEvent` replaces the untyped `sse_event("<name>", json!({...}))` calls
//! that used to be spread across the handler, the streaming loop and the
//! orchestration service. Every payload that reaches a client is now a Rust
//! type, so a new field cannot be added on one side of the wire only.
//!
//! The enum carries `#[serde(tag = "event", content = "data")]` so a single
//! value round-trips as `{"event": "chunk", "data": {"text": "..."}}`. SSE
//! splits those two halves across the `event:` and `data:` lines, which is
//! what [`ChatEvent::name`] and [`ChatEvent::payload`] return.

use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::llm::Citation;

// ============================================================================
// Payloads
// ============================================================================

/// A partial text delta from the provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatChunk {
    pub text: String,
}

/// An inline `[N]` citation marker, emitted after final evidence validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatCitationRef {
    pub index: usize,
    pub source_id: Uuid,
}

/// The resolved citation set for the whole answer.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatCitations {
    pub citations: Vec<Citation>,
}

/// Retrieval quality metrics for this exchange.
///
/// `context_relevance` is `null` when the answer used no retrieved context.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatMetrics {
    pub context_relevance: Option<f32>,
}

/// Progress stage of the pipeline, for the client's thinking indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingStage {
    /// Hybrid retrieval is running.
    RetrievingContext,
    /// The query is being rewritten, proactively or after low-quality retrieval.
    ReformulatingQuery,
    /// The provider stream has been opened.
    Generating,
}

/// Thinking indicator payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatThinking {
    pub stage: ThinkingStage,
}

/// Non-fatal conditions the client localizes and renders.
///
/// A closed discriminator rather than free text: the message is the client's
/// to translate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum WarningKind {
    /// Retrieval scored below the corrective-RAG threshold.
    LowRetrievalQuality,
}

/// Warning payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatWarning {
    #[serde(rename = "type")]
    pub kind: WarningKind,
}

/// Server-side notices about how the prompt was assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatSystem {
    /// History was cut to fit the token budget; `kept` messages survived.
    HistoryTruncated { kept: usize },
    /// Dropped history is being summarized into memory.
    HistorySummarized { dropped_count: usize },
}

/// Suggested follow-up questions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatFollowUpSuggestions {
    pub suggestions: Vec<String>,
}

/// Terminal success payload.
///
/// `rag_log_id` is `null` when the answer used no retrieved context, so no RAG
/// log row exists to link feedback to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatDone {
    pub model: String,
    pub provider: String,
    pub rag_log_id: Option<Uuid>,
}

/// Terminal failure payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatError {
    pub message: String,
}

/// Terminal payload emitted when graceful shutdown interrupts a live stream.
///
/// Distinct from [`ChatError`] so the client can offer a retry instead of
/// reporting a provider failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatShutdown {
    pub message: String,
}

// ============================================================================
// The event
// ============================================================================

/// Apply a generic serialization function to whichever payload a `ChatEvent`
/// carries. Each arm monomorphizes `$f` separately, which is why one match can
/// serve both `to_value` and `to_string`.
macro_rules! dispatch_payload {
    ($event:expr, $f:path) => {
        match $event {
            Self::Chunk(p) => $f(p),
            Self::Thinking(p) => $f(p),
            Self::System(p) => $f(p),
            Self::Warning(p) => $f(p),
            Self::Citation(p) => $f(p),
            Self::Citations(p) => $f(p),
            Self::Metrics(p) => $f(p),
            Self::FollowUpSuggestions(p) => $f(p),
            Self::Done(p) => $f(p),
            Self::Error(p) => $f(p),
            Self::Shutdown(p) => $f(p),
        }
    };
}

/// Every event the chat stream can emit, in `v1` of the protocol.
///
/// Clients must ignore unknown variants without terminating the stream:
/// additive variants are not a breaking change.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "event", content = "data", rename_all = "snake_case")]
pub enum ChatEvent {
    /// Text delta. Emitted zero or more times.
    Chunk(ChatChunk),
    /// Pipeline stage indicator.
    Thinking(ChatThinking),
    /// Prompt assembly notice.
    System(ChatSystem),
    /// Non-fatal retrieval warning.
    Warning(ChatWarning),
    /// Inline citation marker.
    Citation(ChatCitationRef),
    /// Resolved citation set.
    Citations(ChatCitations),
    /// Retrieval quality metrics.
    Metrics(ChatMetrics),
    /// Follow-up questions. Always emitted before [`ChatEvent::Done`].
    FollowUpSuggestions(ChatFollowUpSuggestions),
    /// Terminal success.
    Done(ChatDone),
    /// Terminal failure.
    Error(ChatError),
    /// Terminal server shutdown.
    Shutdown(ChatShutdown),
}

impl ChatEvent {
    /// SSE event name, matching the serde tag.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Chunk(_) => "chunk",
            Self::Thinking(_) => "thinking",
            Self::System(_) => "system",
            Self::Warning(_) => "warning",
            Self::Citation(_) => "citation",
            Self::Citations(_) => "citations",
            Self::Metrics(_) => "metrics",
            Self::FollowUpSuggestions(_) => "follow_up_suggestions",
            Self::Done(_) => "done",
            Self::Error(_) => "error",
            Self::Shutdown(_) => "shutdown",
        }
    }

    /// The payload half of the event, encoded exactly as it appears on the SSE
    /// `data:` line.
    ///
    /// Prefer this over [`ChatEvent::payload`] on the wire: routing an `f32`
    /// through `serde_json::Value` widens it to `f64`, turning `0.72` into
    /// `0.7200000286102295`. Serializing straight to a string keeps the shorter
    /// `f32` form the API has always emitted.
    ///
    /// Serialization cannot fail for today's payloads — every field is a plain
    /// serde type with string map keys — but a future field could change that,
    /// so the error is surfaced instead of panicking.
    pub fn payload_json(&self) -> Result<String, serde_json::Error> {
        dispatch_payload!(self, serde_json::to_string)
    }

    /// The payload half of the event as a JSON value, for tests and contract
    /// generation. See [`ChatEvent::payload_json`] for the wire encoding.
    pub fn payload(&self) -> Result<serde_json::Value, serde_json::Error> {
        dispatch_payload!(self, serde_json::to_value)
    }

    /// Whether this event ends the stream.
    ///
    /// Exactly one terminal event reaches a client, enforced by
    /// [`ChatEventStream`].
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(self, Self::Done(_) | Self::Error(_) | Self::Shutdown(_))
    }

    // -- Constructors used by the orchestration and streaming layers --------

    #[must_use]
    pub fn chunk(text: impl Into<String>) -> Self {
        Self::Chunk(ChatChunk { text: text.into() })
    }

    #[must_use]
    pub const fn thinking(stage: ThinkingStage) -> Self {
        Self::Thinking(ChatThinking { stage })
    }

    #[must_use]
    pub const fn warning(kind: WarningKind) -> Self {
        Self::Warning(ChatWarning { kind })
    }

    #[must_use]
    pub const fn history_truncated(kept: usize) -> Self {
        Self::System(ChatSystem::HistoryTruncated { kept })
    }

    #[must_use]
    pub const fn history_summarized(dropped_count: usize) -> Self {
        Self::System(ChatSystem::HistorySummarized { dropped_count })
    }

    #[must_use]
    pub const fn citation(index: usize, source_id: Uuid) -> Self {
        Self::Citation(ChatCitationRef { index, source_id })
    }

    #[must_use]
    pub const fn citations(citations: Vec<Citation>) -> Self {
        Self::Citations(ChatCitations { citations })
    }

    #[must_use]
    pub const fn metrics(context_relevance: Option<f32>) -> Self {
        Self::Metrics(ChatMetrics { context_relevance })
    }

    #[must_use]
    pub const fn follow_up_suggestions(suggestions: Vec<String>) -> Self {
        Self::FollowUpSuggestions(ChatFollowUpSuggestions { suggestions })
    }

    #[must_use]
    pub fn done(
        model: impl Into<String>,
        provider: impl Into<String>,
        rag_log_id: Option<Uuid>,
    ) -> Self {
        Self::Done(ChatDone {
            model: model.into(),
            provider: provider.into(),
            rag_log_id,
        })
    }

    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(ChatError {
            message: message.into(),
        })
    }

    #[must_use]
    pub fn shutdown(message: impl Into<String>) -> Self {
        Self::Shutdown(ChatShutdown {
            message: message.into(),
        })
    }
}

// ============================================================================
// The stream handle
// ============================================================================

/// Buffer size for the chat event channel.
pub const CHAT_EVENT_BUFFER: usize = 100;

/// The producer half of a chat stream.
///
/// Wraps the channel so the one protocol rule that cannot be expressed in the
/// type system is enforced in one place: **at most one terminal event reaches
/// a client, and nothing follows it**. Before US-009 the truncation path
/// emitted `error` and then `done`, and a late failure could append `error`
/// after `done`; both are now impossible by construction rather than by
/// convention.
pub struct ChatEventStream {
    tx: mpsc::Sender<ChatEvent>,
    terminated: AtomicBool,
    generation_finished: AtomicBool,
}

impl ChatEventStream {
    #[must_use]
    pub const fn new(tx: mpsc::Sender<ChatEvent>) -> Self {
        Self {
            tx,
            terminated: AtomicBool::new(false),
            generation_finished: AtomicBool::new(false),
        }
    }

    /// Create a stream and its receiver with the default buffer.
    #[must_use]
    pub fn channel() -> (Self, mpsc::Receiver<ChatEvent>) {
        let (tx, rx) = mpsc::channel(CHAT_EVENT_BUFFER);
        (Self::new(tx), rx)
    }

    /// Emit an event. Returns `false` when the event was dropped because the
    /// client is gone or the stream already terminated.
    pub async fn emit(&self, event: ChatEvent) -> bool {
        if matches!(event, ChatEvent::Citation(_) | ChatEvent::Citations(_))
            && !self.generation_finished.load(Ordering::SeqCst)
        {
            tracing::warn!("Dropped citation data before generation finished");
            return false;
        }
        if matches!(event, ChatEvent::Chunk(_)) && self.generation_finished.load(Ordering::SeqCst) {
            tracing::warn!("Dropped text chunk after citation validation started");
            return false;
        }
        if event.is_terminal() {
            // `swap` makes the first terminal event win even if two failure
            // paths race: the loser is dropped rather than appended.
            if self.terminated.swap(true, Ordering::SeqCst) {
                tracing::debug!(
                    event = event.name(),
                    "Dropped terminal chat event: stream already terminated"
                );
                return false;
            }
        } else if self.terminated.load(Ordering::SeqCst) {
            tracing::debug!(
                event = event.name(),
                "Dropped chat event emitted after stream termination"
            );
            return false;
        }

        if self.tx.is_closed() {
            return false;
        }
        self.tx.send(event).await.is_ok()
    }

    /// Enqueue a non-terminal event without waiting for channel capacity.
    ///
    /// Citation validation uses this while a database lease is live: a stalled
    /// client may lose citation metadata, but cannot hold source publication
    /// locks indefinitely.
    pub fn try_emit_non_terminal(&self, event: ChatEvent) -> bool {
        if event.is_terminal() {
            tracing::warn!(
                event = event.name(),
                "Refused terminal event on non-terminal path"
            );
            return false;
        }
        if matches!(event, ChatEvent::Citation(_) | ChatEvent::Citations(_))
            && !self.generation_finished.load(Ordering::SeqCst)
        {
            tracing::warn!("Dropped citation data before generation finished");
            return false;
        }
        if matches!(event, ChatEvent::Chunk(_)) && self.generation_finished.load(Ordering::SeqCst) {
            tracing::warn!("Dropped text chunk after citation validation started");
            return false;
        }
        if self.terminated.load(Ordering::SeqCst) {
            tracing::debug!(
                event = event.name(),
                "Dropped chat event emitted after stream termination"
            );
            return false;
        }
        match self.tx.try_send(event) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(event)) => {
                tracing::warn!(
                    event = event.name(),
                    "Dropped chat event: client is backpressured"
                );
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => false,
        }
    }

    /// Close the text phase and allow validated per-citation events.
    pub fn finish_generation(&self) {
        self.generation_finished.store(true, Ordering::SeqCst);
    }

    /// Whether the client has disconnected.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    /// Whether a terminal event has already been emitted.
    #[must_use]
    pub fn is_terminated(&self) -> bool {
        self.terminated.load(Ordering::SeqCst)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(rx: &mut mpsc::Receiver<ChatEvent>) -> Vec<&'static str> {
        let mut names = Vec::new();
        while let Ok(event) = rx.try_recv() {
            names.push(event.name());
        }
        names
    }

    #[test]
    fn name_matches_the_serde_tag() {
        let event = ChatEvent::thinking(ThinkingStage::ReformulatingQuery);
        let encoded = serde_json::to_value(&event).expect("serialize");
        assert_eq!(encoded["event"], event.name());
        assert_eq!(encoded["data"], event.payload().expect("payload"));
    }

    #[test]
    fn warning_keeps_its_type_discriminator() {
        let event = ChatEvent::warning(WarningKind::LowRetrievalQuality);
        assert_eq!(
            event.payload().expect("payload"),
            serde_json::json!({ "type": "low_retrieval_quality" })
        );
    }

    #[test]
    fn system_variants_share_the_type_discriminator() {
        assert_eq!(
            ChatEvent::history_truncated(12).payload().expect("payload"),
            serde_json::json!({ "type": "history_truncated", "kept": 12 })
        );
        assert_eq!(
            ChatEvent::history_summarized(8).payload().expect("payload"),
            serde_json::json!({ "type": "history_summarized", "dropped_count": 8 })
        );
    }

    #[test]
    fn done_always_carries_rag_log_id() {
        let payload = ChatEvent::done("m", "p", None).payload().expect("payload");
        assert!(
            payload
                .get("rag_log_id")
                .is_some_and(serde_json::Value::is_null),
            "rag_log_id must be explicit null, not absent: {payload}"
        );
    }

    #[tokio::test]
    async fn only_the_first_terminal_event_is_delivered() {
        let (stream, mut rx) = ChatEventStream::channel();
        assert!(stream.emit(ChatEvent::chunk("hi")).await);
        assert!(stream.emit(ChatEvent::error("boom")).await);
        assert!(!stream.emit(ChatEvent::done("m", "p", None)).await);
        assert_eq!(drain(&mut rx), vec!["chunk", "error"]);
    }

    #[tokio::test]
    async fn nothing_is_emitted_after_termination() {
        let (stream, mut rx) = ChatEventStream::channel();
        assert!(stream.emit(ChatEvent::done("m", "p", None)).await);
        assert!(!stream.emit(ChatEvent::chunk("late")).await);
        assert!(stream.is_terminated());
        assert_eq!(drain(&mut rx), vec!["done"]);
    }

    #[tokio::test]
    async fn citation_events_are_allowed_only_after_the_text_phase() {
        let (stream, mut rx) = ChatEventStream::channel();
        let source_id = Uuid::new_v4();
        assert!(stream.emit(ChatEvent::chunk("answer [1]")).await);
        assert!(!stream.emit(ChatEvent::citation(1, source_id)).await);
        assert!(!stream.emit(ChatEvent::citations(Vec::new())).await);

        stream.finish_generation();
        assert!(stream.emit(ChatEvent::citation(1, source_id)).await);
        assert!(stream.emit(ChatEvent::citations(Vec::new())).await);
        assert!(!stream.emit(ChatEvent::chunk("late text")).await);
        assert_eq!(drain(&mut rx), vec!["chunk", "citation", "citations"]);
    }

    #[test]
    fn citation_enqueue_never_waits_for_a_backpressured_client() {
        let (stream, _rx) = ChatEventStream::channel();
        for _ in 0..CHAT_EVENT_BUFFER {
            assert!(stream.try_emit_non_terminal(ChatEvent::chunk("fill")));
        }
        stream.finish_generation();
        assert!(!stream.try_emit_non_terminal(ChatEvent::citations(Vec::new())));
    }

    #[tokio::test]
    async fn emitting_to_a_disconnected_client_is_not_an_error() {
        let (stream, rx) = ChatEventStream::channel();
        drop(rx);
        assert!(!stream.emit(ChatEvent::chunk("hi")).await);
        assert!(stream.is_closed());
    }
}
