//! The public event protocol, version `v1` (US-009, `tasks/prd-open-core.md`).
//!
//! One canonical definition of every event the core streams to a client. The
//! application layer emits these values; the HTTP adapter is solely responsible
//! for turning them into SSE framing. Nothing in this module may reference
//! Axum, `axum::response::sse` or any transport type: a queue consumer, a test
//! harness and the browser must all see the same payload.
//!
//! Ordering, optionality, reconnect and cancellation rules are specified in
//! `docs/contracts/sse-protocol-v1.md` and pinned by golden wire fixtures in
//! `contracts/baseline/sse/`, which the TypeScript parser tests read directly.

pub mod chat;
pub mod source;

pub use chat::{
    ChatChunk, ChatCitationRef, ChatCitations, ChatDone, ChatError, ChatEvent, ChatEventStream,
    ChatFollowUpSuggestions, ChatMetrics, ChatShutdown, ChatSystem, ChatThinking, ChatWarning,
    ThinkingStage, WarningKind,
};
pub use source::{EmbeddingProgress, SourceEvent, SourceStatusData};

/// Version of the event protocol described by this module.
///
/// Additive event variants and additive optional fields keep this version.
/// Removing a variant, renaming an event or changing a field's type is a
/// breaking change and requires `v2`.
pub const EVENT_PROTOCOL_VERSION: &str = "v1";
