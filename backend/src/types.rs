//! Shared domain types used across layer boundaries.
//!
//! This module is the **canonical** source for types that are needed by multiple
//! layers (API, services, repositories). Other modules import from `crate::types`
//! instead of reaching across layers directly.
//!
//! Dependency rule: this module may import from `crate::llm` and
//! `crate::repositories` (low-level), but **never** from `crate::services` or
//! `crate::api`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Re-export low-level types that cross layer boundaries.
pub use crate::llm::Citation;
pub use crate::repositories::ChunkSearchResult;

// ============================================================================
// Request Context (task-local propagation)
// ============================================================================

tokio::task_local! {
    static REQUEST_CONTEXT: RequestContext;
}

/// Request-scoped context propagated via task-local storage.
///
/// Carries the `X-Request-ID` from the incoming HTTP request through to
/// downstream service calls and outgoing HTTP client requests without
/// threading it through every function signature.
#[derive(Clone, Default)]
pub struct RequestContext {
    /// The X-Request-ID from the incoming HTTP request.
    pub request_id: Option<Arc<str>>,
}

impl RequestContext {
    /// Create a new context with a request ID.
    pub fn new(request_id: Arc<str>) -> Self {
        Self {
            request_id: Some(request_id),
        }
    }

    /// Get the current request context from task-local storage.
    ///
    /// Returns `Default` (no request ID) if called outside a scoped context
    /// (e.g., in background tasks or tests).
    pub fn current() -> Self {
        REQUEST_CONTEXT.try_with(Clone::clone).unwrap_or_default()
    }

    /// Execute a future within this request context.
    pub async fn scope<F: std::future::Future>(self, f: F) -> F::Output {
        REQUEST_CONTEXT.scope(self, f).await
    }
}

// ============================================================================
// SourceType
// ============================================================================

/// Supported source types for document ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    Pdf,
    Text,
    Markdown,
    Web,
    Docx,
    Epub,
    Youtube,
}

impl SourceType {
    /// Returns the string representation for database storage.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Text => "text",
            Self::Markdown => "markdown",
            Self::Web => "web",
            Self::Docx => "docx",
            Self::Epub => "epub",
            Self::Youtube => "youtube",
        }
    }
}

/// Valid source type strings for error messages.
const VALID_SOURCE_TYPES: &str = "pdf, text, txt, markdown, md, web, url, docx, epub, youtube, yt";

impl TryFrom<&str> for SourceType {
    type Error = crate::error::AppError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "pdf" => Ok(Self::Pdf),
            "text" | "txt" => Ok(Self::Text),
            "markdown" | "md" => Ok(Self::Markdown),
            "web" | "url" => Ok(Self::Web),
            "docx" => Ok(Self::Docx),
            "epub" => Ok(Self::Epub),
            "youtube" | "yt" => Ok(Self::Youtube),
            _ => Err(crate::error::AppError::Validation(format!(
                "Unknown source type: \"{s}\". Valid types: {VALID_SOURCE_TYPES}"
            ))),
        }
    }
}

impl From<SourceType> for String {
    fn from(t: SourceType) -> Self {
        t.as_str().to_owned()
    }
}

// ============================================================================
// SearchResult
// ============================================================================

/// Search result with source information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: Uuid,
    pub source_id: Uuid,
    pub source_title: String,
    pub chunk_index: i32,
    pub content: String,
    pub parent_content: Option<String>,
    pub relevance_score: f32,
    /// Chunk metadata (JSONB) — section_header, page_number, YouTube timestamps, etc.
    pub metadata: Option<serde_json::Value>,
}

impl From<ChunkSearchResult> for SearchResult {
    fn from(c: ChunkSearchResult) -> Self {
        Self {
            chunk_id: c.id,
            source_id: c.source_id,
            source_title: c.source_title,
            chunk_index: c.chunk_index,
            content: c.content,
            parent_content: c.parent_content,
            relevance_score: c.relevance_score,
            metadata: c.metadata,
        }
    }
}

impl From<&SearchResult> for crate::llm::types::CitableChunk {
    fn from(sr: &SearchResult) -> Self {
        Self {
            source_id: sr.source_id,
            chunk_index: sr.chunk_index,
            // Uses child content (not parent_content) intentionally: citations are
            // precise navigational anchors. The LLM receives parent_content via
            // format_context_for_llm in context.rs.
            content: sr.content.clone(),
            relevance_score: sr.relevance_score,
            metadata: sr.metadata.clone(),
        }
    }
}

// ============================================================================
// ChunkWithContext + ChunkMetadata
// ============================================================================

/// Chunk metadata for enhanced retrieval and citation display.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Closest parent header (for Markdown/structured docs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_header: Option<String>,
    /// Page number (for PDFs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<u32>,
    /// Position index within the source document.
    pub position: u32,
    /// Start timestamp in seconds (for YouTube sources).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_start: Option<f64>,
    /// End timestamp in seconds (for YouTube sources).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_end: Option<f64>,
    /// YouTube video ID (for YouTube sources).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_id: Option<String>,
    /// Deep-link citation URL (for YouTube sources).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_url: Option<String>,
}

/// A chunk with its optional context prefix, ready for storage.
#[derive(Debug, Clone)]
pub struct ChunkWithContext {
    /// The raw chunk text content.
    pub content: String,
    /// LLM-generated context prefix (None for legacy/failed chunks).
    pub context_prefix: Option<String>,
    /// Larger parent chunk text for small-to-big retrieval.
    /// None for legacy chunks that predate the parent-child architecture.
    /// Uses `Arc<str>` so all child chunks of the same parent share one allocation.
    pub parent_content: Option<Arc<str>>,
    /// Structured metadata.
    pub metadata: ChunkMetadata,
    /// SHA-256 hex digest of the content, used for deduplication during reprocessing.
    pub content_hash: String,
}

impl ChunkWithContext {
    /// Create a chunk without context prefix (legacy/non-contextualized).
    #[must_use]
    pub fn plain(content: String) -> Self {
        Self {
            content,
            context_prefix: None,
            parent_content: None,
            metadata: ChunkMetadata::default(),
            content_hash: String::new(),
        }
    }

    /// Create a chunk with a context prefix.
    #[must_use]
    pub fn with_prefix(content: String, prefix: String) -> Self {
        Self {
            content,
            context_prefix: Some(prefix),
            parent_content: None,
            metadata: ChunkMetadata::default(),
            content_hash: String::new(),
        }
    }
}

// ============================================================================
// Purge Task Status (for /health/detailed diagnostics)
// ============================================================================

/// Status of a single periodic purge task.
#[derive(Debug, Clone)]
pub struct PurgeTaskStatus {
    /// Total runs completed (success + failure).
    pub total_runs: u64,
    /// Total items deleted across all runs.
    pub total_deleted: u64,
    /// Total failures across all runs (cumulative, never resets).
    pub total_failures: u64,
    /// Consecutive failures (resets on success).
    pub consecutive_failures: u32,
    /// Timestamp of the last successful run.
    pub last_success_at: Option<Instant>,
    /// Timestamp of the last failure.
    pub last_failure_at: Option<Instant>,
    /// Duration of the last run (whether success or failure).
    pub last_run_duration_ms: Option<u64>,
    /// Configured normal interval.
    pub normal_interval_secs: u64,
}

impl PurgeTaskStatus {
    /// Create a new status for a task with the given normal interval.
    pub fn new(normal_interval: std::time::Duration) -> Self {
        Self {
            total_runs: 0,
            total_deleted: 0,
            total_failures: 0,
            consecutive_failures: 0,
            last_success_at: None,
            last_failure_at: None,
            last_run_duration_ms: None,
            normal_interval_secs: normal_interval.as_secs(),
        }
    }

    /// Record a successful run.
    pub fn record_success(&mut self, deleted: u64, duration_ms: u64) {
        self.total_runs += 1;
        self.total_deleted += deleted;
        self.consecutive_failures = 0;
        self.last_success_at = Some(Instant::now());
        self.last_run_duration_ms = Some(duration_ms);
    }

    /// Record a failed run.
    pub fn record_failure(&mut self, duration_ms: u64) {
        self.total_runs += 1;
        self.total_failures += 1;
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        self.last_failure_at = Some(Instant::now());
        self.last_run_duration_ms = Some(duration_ms);
    }

    /// Compute seconds since the last successful run.
    pub fn secs_since_last_success(&self) -> Option<u64> {
        self.last_success_at.map(|t| t.elapsed().as_secs())
    }

    /// Compute seconds since the last failure.
    pub fn secs_since_last_failure(&self) -> Option<u64> {
        self.last_failure_at.map(|t| t.elapsed().as_secs())
    }
}

/// Shared map of purge task statuses, keyed by task name.
#[derive(Clone, Default)]
pub struct PurgeTaskState {
    tasks: Arc<Mutex<HashMap<&'static str, PurgeTaskStatus>>>,
}

impl PurgeTaskState {
    /// Create a new, empty purge task state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a task with its normal interval.
    pub fn register(&self, name: &'static str, normal_interval: std::time::Duration) {
        self.tasks
            .lock()
            .insert(name, PurgeTaskStatus::new(normal_interval));
    }

    /// Record a successful run for a task.
    pub fn record_success(&self, name: &'static str, deleted: u64, duration_ms: u64) {
        if let Some(status) = self.tasks.lock().get_mut(name) {
            status.record_success(deleted, duration_ms);
        }
    }

    /// Record a failed run for a task.
    pub fn record_failure(&self, name: &'static str, duration_ms: u64) {
        if let Some(status) = self.tasks.lock().get_mut(name) {
            status.record_failure(duration_ms);
        }
    }

    /// Get a snapshot of all task statuses for reporting.
    pub fn snapshot(&self) -> HashMap<&'static str, PurgeTaskStatus> {
        self.tasks.lock().clone()
    }
}

// ============================================================================
// Memory Decay Tracker (US-004: at most once per 24h per notebook)
// ============================================================================

/// Cooldown duration between decay runs per notebook.
const DECAY_COOLDOWN: Duration = Duration::from_secs(24 * 60 * 60);

/// Tracks the last decay time per notebook to enforce the 24-hour cooldown.
///
/// Uses `DashMap` for lock-free concurrent access from multiple `tokio::spawn`
/// tasks. The `entry()` API provides atomic check-and-set semantics.
#[derive(Clone, Default)]
pub struct MemoryDecayTracker {
    last_decay: Arc<DashMap<Uuid, Instant>>,
}

impl MemoryDecayTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if decay should run for this notebook (not on cooldown).
    /// Atomically records the current time if returning `true`.
    ///
    /// **Fail-forward design**: The cooldown is consumed even if the subsequent
    /// decay operation fails. This prevents retry storms on persistent failures
    /// at the cost of skipping one decay cycle on transient errors.
    pub fn should_run(&self, notebook_id: Uuid) -> bool {
        match self.last_decay.entry(notebook_id) {
            dashmap::Entry::Occupied(mut entry) => {
                if entry.get().elapsed() >= DECAY_COOLDOWN {
                    *entry.get_mut() = Instant::now();
                    true
                } else {
                    false
                }
            }
            dashmap::Entry::Vacant(entry) => {
                entry.insert(Instant::now());
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_source_types_succeed() {
        let cases = [
            ("pdf", SourceType::Pdf),
            ("text", SourceType::Text),
            ("txt", SourceType::Text),
            ("markdown", SourceType::Markdown),
            ("md", SourceType::Markdown),
            ("web", SourceType::Web),
            ("url", SourceType::Web),
            ("docx", SourceType::Docx),
            ("epub", SourceType::Epub),
            ("youtube", SourceType::Youtube),
            ("yt", SourceType::Youtube),
        ];
        for (input, expected) in cases {
            let result = SourceType::try_from(input);
            assert_eq!(result.unwrap(), expected, "failed for input: {input}");
        }
    }

    #[test]
    fn valid_source_types_case_insensitive() {
        let cases = [
            ("PDF", SourceType::Pdf),
            ("Markdown", SourceType::Markdown),
            ("WEB", SourceType::Web),
        ];
        for (input, expected) in cases {
            let result = SourceType::try_from(input);
            assert_eq!(result.unwrap(), expected, "failed for input: {input}");
        }
    }

    #[test]
    fn unknown_source_types_fail() {
        let invalid = ["pdff", "html", "", "csv", "json", "   "];
        for input in invalid {
            let result = SourceType::try_from(input);
            assert!(result.is_err(), "expected error for input: \"{input}\"");
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("Unknown source type"),
                "error should mention unknown type, got: {msg}"
            );
            assert!(
                msg.contains("Valid types:"),
                "error should list valid types, got: {msg}"
            );
        }
    }

    #[test]
    fn source_type_as_str_roundtrips() {
        let types = [
            SourceType::Pdf,
            SourceType::Text,
            SourceType::Markdown,
            SourceType::Web,
            SourceType::Docx,
            SourceType::Epub,
            SourceType::Youtube,
        ];
        for t in types {
            let s = t.as_str();
            let roundtripped = SourceType::try_from(s).unwrap();
            assert_eq!(roundtripped, t, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn source_type_into_string_unchanged() {
        assert_eq!(String::from(SourceType::Pdf), "pdf");
        assert_eq!(String::from(SourceType::Web), "web");
    }

    // ── ChunkWithContext tests ────────────────────────────────────────────

    #[test]
    fn chunk_plain_has_no_parent_content() {
        let chunk = ChunkWithContext::plain("test".into());
        assert!(chunk.parent_content.is_none());
        assert!(chunk.context_prefix.is_none());
    }

    #[test]
    fn chunk_with_prefix_has_no_parent_content() {
        let chunk = ChunkWithContext::with_prefix("test".into(), "prefix".into());
        assert!(chunk.parent_content.is_none());
        assert!(chunk.context_prefix.is_some());
    }

    // ── PurgeTaskStatus tests ──────────────────────────────────────────────

    #[test]
    fn purge_status_new_has_zero_counters() {
        let status = PurgeTaskStatus::new(std::time::Duration::from_secs(86400));
        assert_eq!(status.total_runs, 0);
        assert_eq!(status.total_deleted, 0);
        assert_eq!(status.total_failures, 0);
        assert_eq!(status.consecutive_failures, 0);
        assert!(status.last_success_at.is_none());
        assert!(status.last_failure_at.is_none());
        assert!(status.last_run_duration_ms.is_none());
        assert_eq!(status.normal_interval_secs, 86400);
    }

    #[test]
    fn purge_status_record_success_increments_counters() {
        let mut status = PurgeTaskStatus::new(std::time::Duration::from_secs(3600));
        status.record_success(10, 50);
        assert_eq!(status.total_runs, 1);
        assert_eq!(status.total_deleted, 10);
        assert_eq!(status.total_failures, 0);
        assert_eq!(status.consecutive_failures, 0);
        assert!(status.last_success_at.is_some());
        assert_eq!(status.last_run_duration_ms, Some(50));
    }

    #[test]
    fn purge_status_record_failure_increments_total_and_consecutive() {
        let mut status = PurgeTaskStatus::new(std::time::Duration::from_secs(3600));
        status.record_failure(100);
        status.record_failure(200);
        assert_eq!(status.total_runs, 2);
        assert_eq!(status.total_failures, 2);
        assert_eq!(status.consecutive_failures, 2);
        assert!(status.last_failure_at.is_some());
        assert_eq!(status.last_run_duration_ms, Some(200));
    }

    #[test]
    fn purge_status_success_resets_consecutive_but_not_total_failures() {
        let mut status = PurgeTaskStatus::new(std::time::Duration::from_secs(3600));
        status.record_failure(10);
        status.record_failure(20);
        assert_eq!(status.total_failures, 2);
        assert_eq!(status.consecutive_failures, 2);

        status.record_success(5, 30);
        assert_eq!(status.total_failures, 2); // cumulative, never resets
        assert_eq!(status.consecutive_failures, 0); // resets on success
        assert_eq!(status.total_runs, 3);
        assert_eq!(status.total_deleted, 5);
    }

    #[test]
    fn purge_state_register_and_snapshot() {
        let state = PurgeTaskState::new();
        state.register("test-task", std::time::Duration::from_secs(600));
        state.record_success("test-task", 42, 10);

        let snapshot = state.snapshot();
        assert_eq!(snapshot.len(), 1);
        let status = &snapshot["test-task"];
        assert_eq!(status.total_runs, 1);
        assert_eq!(status.total_deleted, 42);
        assert_eq!(status.total_failures, 0);
    }

    #[test]
    fn purge_state_record_on_unregistered_task_is_noop() {
        let state = PurgeTaskState::new();
        state.record_success("unknown", 10, 5);
        state.record_failure("unknown", 5);
        assert!(state.snapshot().is_empty());
    }
}
