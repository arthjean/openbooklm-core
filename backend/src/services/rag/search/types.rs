//! Search types and request builders.

use serde::{Deserialize, Serialize};

use crate::error::AppError;
pub use crate::types::SearchResult;

// ============================================================================
// Constants
// ============================================================================

pub(super) const DEFAULT_LIMIT: i32 = 10;

/// Maximum results any single search can return.
/// Set to accommodate the max configurable `RETRIEVAL_POOL_SIZE` (500)
/// for the two-stage retrieval pipeline.
pub const MAX_LIMIT: i32 = 500;

// ============================================================================
// Types
// ============================================================================

/// Search mode for retrieval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Hybrid search combining dense + lexical with RRF fusion (default).
    #[default]
    Hybrid,
    /// Dense (vector) search only.
    Dense,
    /// Lexical (full-text) search only.
    Lexical,
}

/// Search request parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: i32,
    /// Minimum score, **on the domain the requested mode produces**: a dense
    /// similarity for [`SearchMode::Dense`], a lexical rank for
    /// [`SearchMode::Lexical`], an RRF value for [`SearchMode::Hybrid`]. The
    /// same number means something different in each, which is why it is
    /// documented against the mode rather than treated as a universal
    /// relevance floor (US-012).
    #[serde(default)]
    pub min_relevance: Option<f32>,
    #[serde(default)]
    pub mode: SearchMode,
}

const fn default_limit() -> i32 {
    DEFAULT_LIMIT
}

impl SearchRequest {
    /// Create a new search request.
    #[must_use]
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: DEFAULT_LIMIT,
            min_relevance: None,
            mode: SearchMode::default(),
        }
    }

    /// Set the result limit.
    #[must_use]
    pub const fn with_limit(mut self, limit: i32) -> Self {
        self.limit = limit;
        self
    }

    /// Set the search mode.
    #[must_use]
    pub const fn with_mode(mut self, mode: SearchMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set minimum relevance threshold.
    #[must_use]
    pub const fn with_min_relevance(mut self, min: f32) -> Self {
        self.min_relevance = Some(min);
        self
    }

    /// Get the clamped limit value.
    #[must_use]
    pub const fn clamped_limit(&self) -> i32 {
        clamp(self.limit, 1, MAX_LIMIT)
    }

    /// Validate and return trimmed query.
    pub(super) fn validated_query(&self) -> Result<&str, AppError> {
        let query = self.query.trim();
        if query.is_empty() {
            return Err(AppError::Validation("Search query cannot be empty".into()));
        }
        Ok(query)
    }
}

/// What one search produced, and what it cost.
///
/// Replaces the `(results, embed_ms, search_ms)` tuple: the retrieval trace
/// needs candidate counts per stage and the number of candidates rejected for
/// carrying a non-finite score (US-004, US-012), and a five-element tuple is
/// how those get silently dropped at a call site.
#[derive(Debug, Clone, Default)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    /// Time spent producing the query vector, or 0 on a cache hit.
    pub embed_ms: u128,
    /// Time spent in the repository search.
    pub search_ms: u128,
    /// Candidates dropped because their score was NaN or infinite.
    pub dropped_non_finite: usize,
    /// Candidates the dense search returned.
    pub dense_candidates: usize,
    /// Candidates the lexical search returned.
    pub lexical_candidates: usize,
    /// Whether the query embedding came from the cache.
    pub cache_hit: bool,
}

/// Result of corrective RAG retrieval.
#[derive(Debug, Clone)]
pub struct CorrectiveResult {
    /// The search results after possible reformulation.
    pub results: Vec<SearchResult>,
    /// Whether the final selection is the evidence that was requested.
    ///
    /// Replaces the average-score verdict: the number it averaged was an RRF
    /// rank artifact, and a threshold on it decided nothing meaningful
    /// (US-012).
    pub confidence: super::context::RetrievalConfidence,
    /// The query that was actually used for retrieval (original or reformulated).
    pub effective_query: String,
    /// Whether the query was reformulated after insufficient initial evidence.
    pub was_corrected: bool,
}

impl CorrectiveResult {
    /// Whether the caller should warn the user about retrieval quality.
    #[must_use]
    pub const fn low_quality_warning(&self) -> bool {
        !self.confidence.is_sufficient()
    }
}

/// Const clamp (stable since Rust 1.50 but const since 1.61).
pub(super) const fn clamp(val: i32, min: i32, max: i32) -> i32 {
    if val < min {
        min
    } else if val > max {
        max
    } else {
        val
    }
}
