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

/// The scale a candidate's score is expressed on.
///
/// Lives in [`crate::types`] since US-012, because the score and its scale now
/// travel together on every candidate rather than being a property the trace
/// records after the fact. Re-exported here so the reporting vocabulary stays
/// one import.
pub use crate::types::ScoreDomain;

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
    /// Retrieval returned nothing, or fewer contexts than the evidence the
    /// caller required, so the query was reformulated and retried (US-012).
    CorrectiveRetrievalTriggered,
    /// Reformulation was not attempted: no history, nothing to disambiguate,
    /// no reformulator configured, or it had already run once for this turn
    /// (US-017).
    ReformulationSkipped,
    /// History was cut to fit the reformulation message and token budget
    /// before the provider call (US-017).
    ReformulationTruncated,
    /// Reformulation ran and retrieval used its output (US-017).
    ReformulationApplied,
    /// Reformulation was attempted and did not produce a usable query;
    /// retrieval continued once with the original (US-017).
    ReformulationFailed,
    /// Dense retrieval could not fill the requested top-k under the notebook
    /// filter, so the approximate index scanned out before finding enough
    /// matching rows (US-016).
    AnnUnderfilled,
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
            Self::CorrectiveRetrievalTriggered => "corrective_retrieval_triggered",
            Self::ReformulationSkipped => "reformulation_skipped",
            Self::ReformulationTruncated => "reformulation_truncated",
            Self::ReformulationApplied => "reformulation_applied",
            Self::ReformulationFailed => "reformulation_failed",
            Self::AnnUnderfilled => "ann_underfilled",
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

/// The set of reasons one retrieval accumulated.
///
/// Sorted and duplicate-free by construction. Both properties are load-bearing:
/// a reason recorded twice would double-count in a dashboard, and a set whose
/// order depended on which stage happened to observe it first would make two
/// runs of the same retrieval produce different report bytes (US-004).
///
/// It exists as a type because four places used to keep their own
/// `Vec<ReasonCode>` and their own "push if absent" loop, and merging two of
/// them meant re-filtering one into the other at a call site.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ReasonSet(Vec<ReasonCode>);

impl ReasonSet {
    /// Record a reason. Recording it again changes nothing.
    pub fn insert(&mut self, reason: ReasonCode) {
        if let Err(index) = self.0.binary_search(&reason) {
            self.0.insert(index, reason);
        }
    }

    /// Whether this reason was recorded.
    #[must_use]
    pub fn contains(&self, reason: ReasonCode) -> bool {
        self.0.binary_search(&reason).is_ok()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The reasons, in their stable order.
    #[must_use]
    pub fn as_slice(&self) -> &[ReasonCode] {
        &self.0
    }

    /// Comma-separated wire names, for one log field.
    #[must_use]
    pub fn joined(&self) -> String {
        self.0
            .iter()
            .map(|r| r.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }
}

impl Extend<ReasonCode> for ReasonSet {
    fn extend<T: IntoIterator<Item = ReasonCode>>(&mut self, iter: T) {
        for reason in iter {
            self.insert(reason);
        }
    }
}

impl FromIterator<ReasonCode> for ReasonSet {
    fn from_iter<T: IntoIterator<Item = ReasonCode>>(iter: T) -> Self {
        let mut set = Self::default();
        set.extend(iter);
        set
    }
}

impl<'a> IntoIterator for &'a ReasonSet {
    type Item = ReasonCode;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, ReasonCode>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter().copied()
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
    /// Scale of the score that produced the final ordering, or `None` when
    /// nothing ordered anything: an empty notebook and a retrieval that failed
    /// before its first query have no scale, and naming one anyway would put a
    /// number's provenance in an audit record that never had a number.
    pub score_domain: Option<ScoreDomain>,
    pub candidates: StageCounts,
    /// Distinct parent contexts in the final selection.
    pub unique_parents: usize,
    pub tokens: TokenCounts,
    /// Everything notable that happened, in a stable order.
    pub reasons: ReasonSet,
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
        domain: Option<ScoreDomain>,
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
            reasons: ReasonSet::default(),
            durations: StageDurations::default(),
        }
    }

    /// Record the reformulated query, by hash.
    #[must_use]
    pub fn with_reformulation(mut self, reformulated: &str) -> Self {
        self.reformulated_query_hash = Some(query_hash(reformulated));
        self
    }

    /// `Ok` exactly when nothing else was recorded.
    pub fn finish(&mut self) {
        if self.reasons.is_empty() {
            self.reasons.insert(ReasonCode::Ok);
        }
    }

    /// Emit the trace on the structured logging pipeline.
    ///
    /// One event, at `info`: an operator who turns this off loses retrieval
    /// diagnosis entirely, and one line per chat turn is affordable.
    pub fn emit(&self) {
        let reasons = self.reasons.joined();
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
            score_domain = self.score_domain.map_or("none", ScoreDomain::as_str),
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
            Some(ScoreDomain::RrfRank),
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
        let mut trace = RetrievalTrace::new(
            Uuid::nil(),
            "q",
            "dense",
            Some(ScoreDomain::DenseSimilarity),
        );
        trace.reasons.insert(ReasonCode::UnderfilledTopK);
        trace.reasons.insert(ReasonCode::RerankerAbsent);
        trace.reasons.insert(ReasonCode::UnderfilledTopK);
        trace.finish();
        assert_eq!(
            trace.reasons.as_slice(),
            [ReasonCode::UnderfilledTopK, ReasonCode::RerankerAbsent]
        );

        // Two traces built in different orders compare equal, which is what
        // makes a report byte-stable.
        let mut other = RetrievalTrace::new(
            Uuid::nil(),
            "q",
            "dense",
            Some(ScoreDomain::DenseSimilarity),
        );
        other.reasons.insert(ReasonCode::RerankerAbsent);
        other.reasons.insert(ReasonCode::UnderfilledTopK);
        other.finish();
        assert_eq!(trace.reasons, other.reasons);
    }

    #[test]
    fn a_reason_set_merges_without_a_call_site_filter() {
        let mut set: ReasonSet = [ReasonCode::RerankerAbsent, ReasonCode::Ok]
            .into_iter()
            .collect();
        set.extend([ReasonCode::Ok, ReasonCode::NoCandidates]);

        assert_eq!(
            set.as_slice(),
            [
                ReasonCode::Ok,
                ReasonCode::NoCandidates,
                ReasonCode::RerankerAbsent
            ],
            "a merged set stays sorted and duplicate-free"
        );
        assert!(set.contains(ReasonCode::NoCandidates));
        assert!(!set.contains(ReasonCode::EmptyCorpus));
        assert_eq!(set.joined(), "ok,no_candidates,reranker_absent");
    }

    #[test]
    fn a_clean_retrieval_is_recorded_as_ok() {
        let mut trace =
            RetrievalTrace::new(Uuid::nil(), "q", "lexical", Some(ScoreDomain::LexicalRank));
        trace.finish();
        assert_eq!(trace.reasons.as_slice(), [ReasonCode::Ok]);
    }

    #[test]
    fn a_retrieval_that_never_ranked_anything_reports_no_score_domain() {
        let mut trace = RetrievalTrace::new(Uuid::nil(), "q", "chat", None);
        trace.finish();
        let rendered = serde_json::to_string(&trace).expect("trace serializes");
        assert!(
            rendered.contains("\"score_domain\":null"),
            "an absent scale must serialize as null, not as a plausible default: {rendered}"
        );
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
