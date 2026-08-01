//! The retrieval trace (US-004).
//!
//! One structured record per retrieval, carrying enough to diagnose a
//! regression and nothing that could leak a private corpus.
//!
//! # Redaction is structural, not editorial
//!
//! There is no field on [`RetrievalTrace`] that can hold source text or a raw
//! user query. Queries appear as [`query_hash`](RetrievalTrace::query_hash),
//! evidence appears as counts. That is deliberate: a trace whose safety depends
//! on every call site remembering to redact is a leak waiting for its first
//! hurried patch. Making the type incapable of carrying text moves the
//! guarantee from review to the compiler.
//!
//! # Why hashes rather than nothing
//!
//! An operator diagnosing "this question retrieves badly" needs to correlate
//! traces across the reformulation boundary and across retries. A truncated
//! SHA-256 does that without being reversible for anything but a guessed
//! candidate, which an operator who already has the question does not need.

use std::fmt;

use serde::Serialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

// ============================================================================
// Score domains
// ============================================================================

/// Which scale a candidate's score is expressed on.
///
/// Recorded so a report never compares an RRF rank score against a provider's
/// relevance score. US-012 turns this into typed score fields on the candidates
/// themselves; here it is the trace's record of which scale produced the
/// ordering that was measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreDomain {
    /// Reciprocal-rank-fusion score. A function of ranks, not of similarity.
    RrfRank,
    /// `1 - cosine_distance`, clamped at zero.
    DenseSimilarity,
    /// PostgreSQL `ts_rank_cd`, clamped at one.
    LexicalRank,
    /// A reranking provider's relevance score. Scale is provider-defined and
    /// not comparable across providers.
    RerankerRelevance,
    /// Context stuffing assigns every chunk the same score; the value carries
    /// no ranking information at all.
    StuffingUniform,
}

impl ScoreDomain {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RrfRank => "rrf_rank",
            Self::DenseSimilarity => "dense_similarity",
            Self::LexicalRank => "lexical_rank",
            Self::RerankerRelevance => "reranker_relevance",
            Self::StuffingUniform => "stuffing_uniform",
        }
    }
}

impl fmt::Display for ScoreDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Reason codes
// ============================================================================

/// A stable, enumerable explanation of what retrieval did.
///
/// Stable strings, because they end up in dashboards and in report diffs. A
/// renamed variant is a breaking change to whoever alerts on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    /// Retrieval ran and filled the requested number of contexts.
    Ok,
    /// The notebook has no chunk to search.
    EmptyCorpus,
    /// Retrieval ran and returned nothing.
    NoCandidates,
    /// Fewer unique contexts than requested survived selection.
    UnderfilledTopK,
    /// The whole notebook was loaded instead of being searched.
    StuffingApplied,
    /// No reranker was configured for this installation.
    RerankerAbsent,
    /// The reranker was called and failed; unreranked order was used.
    RerankerFailed,
    /// Deduplication collapsed candidates onto fewer unique parents than the
    /// requested limit, and no duplicate was reintroduced to pad the count.
    DedupShortfall,
    /// A candidate carried a non-finite score and was rejected.
    NonFiniteScore,
    /// The corpus has no relevance judgment for this query.
    MissingJudgment,
    /// A source outside the query's notebook appeared in the result set.
    IsolationBreach,
    /// The retrieval provider returned an error.
    ProviderError,
    /// Writing the interaction telemetry failed. Retrieval itself succeeded.
    TelemetryWriteFailed,
}

impl ReasonCode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::EmptyCorpus => "empty_corpus",
            Self::NoCandidates => "no_candidates",
            Self::UnderfilledTopK => "underfilled_top_k",
            Self::StuffingApplied => "stuffing_applied",
            Self::RerankerAbsent => "reranker_absent",
            Self::RerankerFailed => "reranker_failed",
            Self::DedupShortfall => "dedup_shortfall",
            Self::NonFiniteScore => "non_finite_score",
            Self::MissingJudgment => "missing_judgment",
            Self::IsolationBreach => "isolation_breach",
            Self::ProviderError => "provider_error",
            Self::TelemetryWriteFailed => "telemetry_write_failed",
        }
    }
}

impl fmt::Display for ReasonCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Trace payload
// ============================================================================

/// Candidate counts, one per pipeline stage.
///
/// A recall regression looks completely different depending on which of these
/// numbers moved; reporting only the last one makes the diagnosis guesswork.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct StageCounts {
    /// Candidates the dense search returned.
    pub dense: usize,
    /// Candidates the lexical search returned.
    pub lexical: usize,
    /// Candidates after fusion.
    pub fused: usize,
    /// Candidates the reranker was given.
    pub reranker_input: usize,
    /// Candidates the reranker returned.
    pub reranker_output: usize,
    /// Candidates after parent deduplication.
    pub deduplicated: usize,
    /// Contexts finally selected for the prompt.
    pub selected: usize,
}

/// Per-stage wall clock, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct StageDurations {
    pub embed_ms: u64,
    pub search_ms: u64,
    pub rerank_ms: u64,
    pub stuffing_load_ms: u64,
    pub total_ms: u64,
}

/// Token accounting for the evidence that reached, or did not reach, the model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct TokenCounts {
    /// Estimated tokens of context handed to the model.
    pub selected: usize,
    /// Estimated tokens of retrieved context dropped before generation.
    pub dropped: usize,
}

/// One retrieval, described without its content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RetrievalTrace {
    /// Index generations the result set was read from.
    ///
    /// Every retrieval query joins `sources.active_generation_id`, so this is a
    /// set of active generations by construction. Chat fills it from the
    /// retrieved chunks; the offline evaluator fills it from its corpus
    /// generation. More than one entry means the answer spanned several
    /// sources, never several versions of one (EP-002).
    pub generation_ids: Vec<Uuid>,
    pub notebook_id: Uuid,
    /// Truncated SHA-256 of the query as the user asked it.
    pub query_hash: String,
    /// Truncated SHA-256 of the reformulated query, when reformulation ran.
    pub reformulated_query_hash: Option<String>,
    /// Retrieval mode, as its stable wire name.
    pub mode: String,
    /// Scale of the score that produced the final ordering.
    pub score_domain: ScoreDomain,
    pub candidates: StageCounts,
    /// Distinct parent contexts in the final selection.
    pub unique_parents: usize,
    pub tokens: TokenCounts,
    /// Everything notable that happened, in a stable order.
    pub reasons: Vec<ReasonCode>,
    pub durations: StageDurations,
}

/// Hex characters kept from the SHA-256 digest.
///
/// 32 hex characters is 128 bits: collision-free for correlation, and short
/// enough to read in a log line.
const HASH_CHARS: usize = 32;

/// Truncated SHA-256 of a text, for correlation without disclosure.
#[must_use]
pub fn query_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    let full = format!("{digest:x}");
    full.chars().take(HASH_CHARS).collect()
}

impl RetrievalTrace {
    /// Start a trace for one retrieval.
    #[must_use]
    pub fn new(
        notebook_id: Uuid,
        query: &str,
        mode: impl Into<String>,
        domain: ScoreDomain,
    ) -> Self {
        Self {
            generation_ids: Vec::new(),
            notebook_id,
            query_hash: query_hash(query),
            reformulated_query_hash: None,
            mode: mode.into(),
            score_domain: domain,
            candidates: StageCounts::default(),
            unique_parents: 0,
            tokens: TokenCounts::default(),
            reasons: Vec::new(),
            durations: StageDurations::default(),
        }
    }

    /// Record the reformulated query, by hash.
    #[must_use]
    pub fn with_reformulation(mut self, reformulated: &str) -> Self {
        self.reformulated_query_hash = Some(query_hash(reformulated));
        self
    }

    /// Record a reason, keeping the list free of duplicates and in a stable
    /// order so two runs of the same retrieval produce the same trace.
    pub fn add_reason(&mut self, reason: ReasonCode) {
        if !self.reasons.contains(&reason) {
            self.reasons.push(reason);
            self.reasons.sort_unstable();
        }
    }

    /// `Ok` exactly when nothing else was recorded.
    pub fn finish(&mut self) {
        if self.reasons.is_empty() {
            self.reasons.push(ReasonCode::Ok);
        }
    }

    /// Emit the trace on the structured logging pipeline.
    ///
    /// One event, at `info`: an operator who turns this off loses retrieval
    /// diagnosis entirely, and one line per chat turn is affordable.
    pub fn emit(&self) {
        let reasons = self
            .reasons
            .iter()
            .map(|r| r.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let generations = self
            .generation_ids
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");

        tracing::info!(
            notebook_id = %self.notebook_id,
            generation_ids = %generations,
            query_hash = %self.query_hash,
            reformulated_query_hash = self.reformulated_query_hash.as_deref().unwrap_or(""),
            mode = %self.mode,
            score_domain = %self.score_domain,
            candidates_dense = self.candidates.dense,
            candidates_lexical = self.candidates.lexical,
            candidates_fused = self.candidates.fused,
            candidates_reranker_input = self.candidates.reranker_input,
            candidates_reranker_output = self.candidates.reranker_output,
            candidates_deduplicated = self.candidates.deduplicated,
            candidates_selected = self.candidates.selected,
            unique_parents = self.unique_parents,
            tokens_selected = self.tokens.selected,
            tokens_dropped = self.tokens.dropped,
            reasons = %reasons,
            embed_ms = self.durations.embed_ms,
            search_ms = self.durations.search_ms,
            rerank_ms = self.durations.rerank_ms,
            stuffing_load_ms = self.durations.stuffing_load_ms,
            total_ms = self.durations.total_ms,
            "rag_retrieval_trace"
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hash_is_stable_truncated_and_not_the_query() {
        let a = query_hash("what is the retry budget");
        assert_eq!(a, query_hash("what is the retry budget"));
        assert_ne!(a, query_hash("what is the retry budget?"));
        assert_eq!(a.len(), HASH_CHARS);
        assert!(!a.contains("retry"));
    }

    #[test]
    fn a_serialized_trace_carries_no_text_from_the_query_or_the_corpus() {
        let mut trace = RetrievalTrace::new(
            Uuid::nil(),
            "how many attempts does the retry budget allow",
            "hybrid",
            ScoreDomain::RrfRank,
        )
        .with_reformulation("retry budget attempts");
        trace.candidates.selected = 3;
        trace.finish();

        let rendered = serde_json::to_string(&trace).expect("trace serializes");
        for leaked in ["retry", "budget", "attempts", "how many"] {
            assert!(
                !rendered.contains(leaked),
                "trace leaked `{leaked}`: {rendered}"
            );
        }
    }

    #[test]
    fn reasons_are_deduplicated_and_ordered() {
        let mut trace =
            RetrievalTrace::new(Uuid::nil(), "q", "dense", ScoreDomain::DenseSimilarity);
        trace.add_reason(ReasonCode::UnderfilledTopK);
        trace.add_reason(ReasonCode::RerankerAbsent);
        trace.add_reason(ReasonCode::UnderfilledTopK);
        trace.finish();
        assert_eq!(
            trace.reasons,
            vec![ReasonCode::UnderfilledTopK, ReasonCode::RerankerAbsent]
        );

        // Two traces built in different orders compare equal, which is what
        // makes a report byte-stable.
        let mut other =
            RetrievalTrace::new(Uuid::nil(), "q", "dense", ScoreDomain::DenseSimilarity);
        other.add_reason(ReasonCode::RerankerAbsent);
        other.add_reason(ReasonCode::UnderfilledTopK);
        other.finish();
        assert_eq!(trace.reasons, other.reasons);
    }

    #[test]
    fn a_clean_retrieval_is_recorded_as_ok() {
        let mut trace = RetrievalTrace::new(Uuid::nil(), "q", "lexical", ScoreDomain::LexicalRank);
        trace.finish();
        assert_eq!(trace.reasons, vec![ReasonCode::Ok]);
    }

    #[test]
    fn wire_names_are_pinned() {
        assert_eq!(ScoreDomain::RrfRank.as_str(), "rrf_rank");
        assert_eq!(
            ScoreDomain::RerankerRelevance.as_str(),
            "reranker_relevance"
        );
        assert_eq!(ReasonCode::UnderfilledTopK.as_str(), "underfilled_top_k");
        assert_eq!(ReasonCode::IsolationBreach.as_str(), "isolation_breach");
        assert_eq!(
            ReasonCode::TelemetryWriteFailed.as_str(),
            "telemetry_write_failed"
        );
    }
}
