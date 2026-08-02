//! The retrieval pipeline, and its one output contract (EP-003).
//!
//! Candidates come from one of two places: a notebook small enough to fit
//! inside the requested limit is loaded whole, anything larger is searched.
//! Everything after that point is shared, which is the contract:
//!
//! - every branch returns between zero and the requested maximum number of
//!   unique parent contexts (US-013);
//! - the score that ordered them travels with them, and no decision compares
//!   two scales (US-012);
//! - diversification happens before reranking, reranking sees the whole pool,
//!   and the limit is applied last (US-014).
//!
//! The two sources differ in one property, and it is named rather than inferred
//! from a telemetry flag: a stuffed pool carries no ranking, so reranking,
//! preference ordering and sandwich presentation have nothing to act on and are
//! skipped. Diversification and selection are not skipped, because "one passage
//! is one context" holds however the passage was loaded.

use super::formatting::{entry_tokens, evidence_body, region_overhead_tokens};
use super::types::{CorrectiveResult, SearchResult};
use crate::core::config::CoreConfig;
use crate::core::providers::{EmbeddingProvider, Reranker};
use crate::error::AppError;
use crate::llm::prompts::EvidenceFormat;
use crate::repositories::{NotebookScope, SearchRepository};
use crate::services::rag::embedding_cache::EmbeddingCache;
use crate::services::rag::eval::trace::{ReasonCode, ReasonSet, StageCounts, query_hash};
use crate::services::rag::hyde::HydeService;
use crate::services::rag::provenance::QueryEmbeddingKind;
use crate::services::rag::query_reformulation::{ChatTurn, QueryReformulator};
use crate::types::ScoreDomain;

use super::preference::{PreferenceBoost, apply_preference_ordering};
use super::stuffing::max_context_stuffing_chunks;
use super::transforms::{
    collapse_parents, rerank_results, sandwich_order, select_final, unique_parent_count,
};
use super::{SearchMode, SearchRequest, search};

/// Maximum number of corrective reformulation attempts (1 retry).
pub const CORRECTIVE_RAG_MAX_RETRIES: u32 = 1;

// ============================================================================
// Retrieval confidence (US-012)
// ============================================================================

/// Why a retrieval did not produce the evidence that was asked for.
///
/// Deterministic facts about the result set, never a score comparison: without
/// a calibrated reranker there is no scale on which "0.42 is too low" means
/// anything, and the value the pipeline used to threshold was an RRF rank
/// artifact whose magnitude depends on `k` and the pool size.
///
/// There is deliberately no relevance-threshold variant. Adding one would
/// require a committed per-provider, per-model calibration artifact proving
/// where the useful cut sits on that provider's scale (US-012 AC-4); until such
/// an artifact exists, no code may compare a reranker score to a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsufficiencyReason {
    /// Retrieval returned nothing at all.
    NoCandidates,
    /// Fewer unique contexts than requested, while the notebook still held
    /// material that was not selected.
    UnderfilledEvidence { selected: usize, requested: usize },
}

/// Whether a retrieval produced the evidence the caller asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalConfidence {
    Sufficient,
    Insufficient(InsufficiencyReason),
}

impl RetrievalConfidence {
    #[must_use]
    pub const fn is_sufficient(self) -> bool {
        matches!(self, Self::Sufficient)
    }

    /// The reason code an insufficiency contributes to the trace.
    #[must_use]
    pub const fn reason_code(self) -> Option<ReasonCode> {
        match self {
            Self::Sufficient => None,
            Self::Insufficient(InsufficiencyReason::NoCandidates) => Some(ReasonCode::NoCandidates),
            Self::Insufficient(InsufficiencyReason::UnderfilledEvidence { .. }) => {
                Some(ReasonCode::UnderfilledTopK)
            }
        }
    }
}

// ============================================================================
// Pipeline outcome
// ============================================================================

/// Everything one retrieval produced besides the contexts themselves.
///
/// Timings, per-stage candidate counts, reason codes, the score domain that
/// ordered the output and whether the evidence was sufficient. The trace is
/// assembled from this rather than reconstructed from the surviving chunks:
/// counts for stages that dropped candidates cannot be recovered afterwards
/// (US-004, US-014).
#[derive(Debug, Clone, Default)]
pub struct PipelineOutcome {
    /// Time spent embedding the query (ms), 0 on a cache hit.
    pub embed_ms: u128,
    /// Time spent in search (dense + lexical) (ms).
    pub search_ms: u128,
    /// Time spent reranking (ms), or 0 if skipped.
    pub rerank_ms: u128,
    /// Time spent loading all chunks for context stuffing (ms).
    pub stuffing_load_ms: u128,
    /// Whether context stuffing was used (all chunks loaded directly).
    pub stuffed: bool,
    /// Whether the embedding cache was hit.
    pub cache_hit: bool,
    /// Candidate counts, one per pipeline stage.
    pub counts: StageCounts,
    /// Distinct parent contexts in the final selection.
    pub unique_parents: usize,
    /// Everything notable that happened.
    pub reasons: ReasonSet,
    /// The scale the returned ordering is expressed on, or `None` when nothing
    /// was ranked: an empty notebook and a retrieval that never ran have no
    /// scale, and defaulting to one would put a fabricated provenance in the
    /// trace.
    pub score_domain: Option<ScoreDomain>,
    /// Whether the requested evidence was produced.
    pub confidence: RetrievalConfidence,
}

impl Default for RetrievalConfidence {
    fn default() -> Self {
        // An outcome nobody filled describes a retrieval that produced nothing,
        // which is what a caller that skipped retrieval has.
        Self::Insufficient(InsufficiencyReason::NoCandidates)
    }
}

// ============================================================================
// Context retrieval
// ============================================================================

/// Parameters for context retrieval.
///
/// Replaces the 11-argument function signature with a named struct for clarity.
pub struct RetrievalParams<'a> {
    pub search_repo: &'a dyn SearchRepository,
    pub config: &'a CoreConfig,
    /// The account and notebook this retrieval may read (US-020 AC-2).
    pub scope: NotebookScope,
    pub query: &'a str,
    pub max_chunks: i32,
    pub embeddings: Option<&'a dyn EmbeddingProvider>,
    pub reranker: Option<&'a dyn Reranker>,
    pub hyde_service: Option<&'a HydeService>,
    pub embedding_cache: Option<&'a EmbeddingCache>,
    /// The role `query` plays. Selects the embedding cache namespace, so a
    /// reformulation and the question it came from cannot share a vector
    /// (US-011).
    pub embedding_kind: QueryEmbeddingKind,
    pub provider: &'a str,
    pub model: &'a str,
    pub preference_boost: Option<&'a PreferenceBoost>,
    /// Tokens the caller's prompt budget leaves for retrieved evidence.
    ///
    /// Only context stuffing consults it, and only to refuse: a notebook that
    /// fits the requested chunk limit but not the token budget is searched
    /// instead of loaded whole (US-018 AC-2). The authoritative pass still runs
    /// at prompt assembly; this is the same number, computed by
    /// [`evidence_allowance`](crate::services::chat::context_budget::evidence_allowance),
    /// so the two cannot disagree.
    pub evidence_token_budget: usize,
}

/// Whether a candidate pool carries an ordering worth acting on.
///
/// Named rather than derived from [`PipelineOutcome::stuffed`]: that field is
/// telemetry, and behaviour that reads telemetry drifts from it the first time
/// someone adds a third source of candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PoolOrder {
    /// Ordered by a score. Reranking, preferences and sandwich presentation
    /// apply.
    Ranked,
    /// Every candidate carries the same score, so the order is the order the
    /// notebook is written in. Nothing downstream may reorder it.
    Uniform,
}

/// Candidates and how they are ordered.
struct Pool {
    candidates: Vec<SearchResult>,
    order: PoolOrder,
}

/// Retrieve context for RAG chat.
///
/// Two sources, one contract. Both return at most `max_chunks` unique parent
/// contexts, and both report which score domain ordered them (US-013).
pub async fn retrieve_context(
    params: &RetrievalParams<'_>,
) -> Result<(Vec<SearchResult>, PipelineOutcome), AppError> {
    tracing::info!(
        notebook_id = %params.scope.notebook_id,
        "rag_pipeline: starting retrieval"
    );

    // The API boundary rejects a limit outside `[1, MAX_CONTEXT_CHUNKS]`, so
    // this conversion is total; the `max(0)` only keeps a mis-wired internal
    // caller from asking for a negative number of contexts.
    let final_limit = usize::try_from(params.max_chunks.max(0)).unwrap_or(0);

    let mut outcome = PipelineOutcome::default();

    // Hoist chunk count: used for the stuffing check, the rerank decision and
    // the underfill verdict. Single DB round-trip instead of three.
    let chunk_count = params
        .search_repo
        .count_chunks_for_notebook(params.scope)
        .await?;
    if chunk_count == 0 {
        let source_count = params
            .search_repo
            .count_sources_for_notebook(params.scope)
            .await?;
        outcome.reasons.insert(if source_count == 0 {
            ReasonCode::EmptyCorpus
        } else {
            ReasonCode::NoCandidates
        });
        return Ok((Vec::new(), outcome));
    }

    let pool = match stuff_notebook(params, chunk_count, final_limit, &mut outcome).await? {
        Some(pool) => Some(pool),
        None => search_notebook(params, chunk_count, final_limit, &mut outcome).await?,
    };
    // Neither source produced a pool: the notebook cannot be read at all, and
    // `outcome` already carries why.
    let Some(pool) = pool else {
        return Ok((Vec::new(), outcome));
    };

    let results = finalize(params, pool, chunk_count, final_limit, &mut outcome).await;
    Ok((results, outcome))
}

/// Load the whole notebook, when it fits.
///
/// Gated on the requested limit as well as on the model threshold. Loading 95
/// chunks for a request that asked for 15 was the largest violation of the
/// cardinality contract in the pipeline (US-013), and the guard is also what
/// US-018 will extend with the token budget.
///
/// What a whole notebook would cost as an evidence region, if it is too much.
///
/// `Some(cost)` when the candidates exceed `budget`, `None` when they fit. The
/// running total stops at the first chunk that blows the budget, so an
/// over-budget notebook is not measured to the end just to be refused, and the
/// pricing itself allocates nothing: the renderer writes into a token meter
/// rather than into a string that would be dropped on the next line.
fn stuffed_cost_over(candidates: &[SearchResult], budget: usize) -> Option<usize> {
    let mut total = region_overhead_tokens(EvidenceFormat::Inline);
    for (position, candidate) in candidates.iter().enumerate() {
        total += entry_tokens(
            EvidenceFormat::Inline,
            position + 1,
            candidate,
            evidence_body(candidate),
        );
        if total > budget {
            return Some(total);
        }
    }
    None
}

/// Returns `None` when stuffing does not apply, which is the signal to search.
async fn stuff_notebook(
    params: &RetrievalParams<'_>,
    chunk_count: i64,
    final_limit: usize,
    outcome: &mut PipelineOutcome,
) -> Result<Option<Pool>, AppError> {
    let threshold = max_context_stuffing_chunks(
        params.provider,
        params.model,
        params.config.context_stuffing_max_chunks,
    );
    if threshold == 0 {
        return Ok(None);
    }

    let ceiling = threshold.min(i64::try_from(final_limit).unwrap_or(i64::MAX));
    if chunk_count > ceiling {
        tracing::debug!(
            notebook_id = %params.scope.notebook_id,
            chunk_count,
            threshold,
            final_limit,
            "Context stuffing skipped: notebook exceeds the threshold or the requested limit"
        );
        return Ok(None);
    }

    // A turn with no evidence allowance cannot stuff anything, and finding that
    // out costs nothing here but a full notebook load below.
    let budget = params.evidence_token_budget;
    if budget == 0 {
        outcome.reasons.insert(ReasonCode::StuffingOverBudget);
        return Ok(None);
    }

    let load_start = std::time::Instant::now();
    let all_chunks = params
        .search_repo
        .get_all_chunks_for_notebook(params.scope)
        .await?;
    outcome.stuffing_load_ms = load_start.elapsed().as_millis();

    let candidates: Vec<SearchResult> = all_chunks
        .into_iter()
        .filter_map(|c| SearchResult::from_chunk(c, ScoreDomain::StuffingUniform))
        .collect();

    // Stuffing means "all of it, or none of it". A notebook whose chunks fit
    // the requested count but not the token budget would be loaded here and
    // trimmed at prompt assembly, which is a stuffed context that is not the
    // whole notebook — the one thing stuffing promises (US-018 AC-2). Searching
    // it instead at least ranks what survives.
    //
    // Priced with the inline renderer even for providers that take native
    // document blocks. The XML envelope is the larger of the two, so a native
    // provider is refused stuffing slightly earlier than it strictly needs to
    // be; the error is on the side that never overflows a window.
    if let Some(stuffed_tokens) = stuffed_cost_over(candidates.as_slice(), budget) {
        tracing::debug!(
            notebook_id = %params.scope.notebook_id,
            chunk_count,
            stuffed_tokens,
            evidence_token_budget = budget,
            "Context stuffing skipped: the notebook exceeds the token budget"
        );
        outcome.reasons.insert(ReasonCode::StuffingOverBudget);
        return Ok(None);
    }

    tracing::info!(
        notebook_id = %params.scope.notebook_id,
        chunk_count,
        threshold,
        final_limit,
        provider = params.provider,
        model = params.model,
        "Context stuffing: loaded all chunks directly"
    );

    outcome.stuffed = true;
    outcome.score_domain = Some(ScoreDomain::StuffingUniform);
    outcome.reasons.insert(ReasonCode::StuffingApplied);

    Ok(Some(Pool {
        candidates,
        order: PoolOrder::Uniform,
    }))
}

/// Embed the query and search, returning the fused candidate pool.
///
/// Returns `None` when no embedding provider is configured, which is reported
/// as a provider failure rather than as "no candidates": the distinction is
/// what stops an unconfigured installation from looking like an empty notebook
/// (US-012).
async fn search_notebook(
    params: &RetrievalParams<'_>,
    chunk_count: i64,
    final_limit: usize,
    outcome: &mut PipelineOutcome,
) -> Result<Option<Pool>, AppError> {
    let Some(embeddings) = params.embeddings else {
        tracing::warn!(
            notebook_id = %params.scope.notebook_id,
            "No embedding provider configured, skipping context retrieval"
        );
        outcome.reasons.insert(ReasonCode::ProviderError);
        return Ok(None);
    };

    let pool_size = params.config.retrieval_pool_size;
    let request = SearchRequest::new(params.query)
        .with_limit(pool_size)
        .with_mode(SearchMode::Hybrid);

    tracing::debug!(
        notebook_id = %params.scope.notebook_id,
        query_len = params.query.len(),
        retrieval_pool_size = pool_size,
        final_limit,
        "Starting context retrieval"
    );

    let query_embedder = super::QueryEmbedder {
        provider: embeddings,
        hyde: params.hyde_service,
        cache: params.embedding_cache,
        kind: params.embedding_kind,
    };
    let found = search(
        params.search_repo,
        &params.config.hybrid_search,
        params.scope,
        &request,
        &query_embedder,
    )
    .await?;

    outcome.embed_ms = found.embed_ms;
    outcome.search_ms = found.search_ms;
    outcome.cache_hit = found.cache_hit;
    outcome.counts.dense = found.dense_candidates;
    outcome.counts.lexical = found.lexical_candidates;
    outcome.counts.fused = found.results.len();
    outcome.score_domain = Some(ScoreDomain::RrfRank);
    if found.dropped_non_finite > 0 {
        outcome.reasons.insert(ReasonCode::NonFiniteScore);
    }

    // A dense search that came back short while the notebook still held rows
    // is the failure mode filtered ANN has: the index scan ran out before it
    // found enough rows passing the notebook filter. Recorded rather than
    // reported as a full result set (US-016).
    let dense_requested = usize::try_from(pool_size.max(0)).unwrap_or(0);
    let corpus = usize::try_from(chunk_count.max(0)).unwrap_or(usize::MAX);
    if found.dense_candidates < dense_requested.min(corpus) {
        outcome.reasons.insert(ReasonCode::AnnUnderfilled);
    }

    Ok(Some(Pool {
        candidates: found.results,
        order: PoolOrder::Ranked,
    }))
}

/// Diversify, rank, select: the steps every pool goes through.
///
/// Selection is the only step that changes membership. Sandwich ordering runs
/// after it and is presentation only (US-014).
async fn finalize(
    params: &RetrievalParams<'_>,
    pool: Pool,
    chunk_count: i64,
    final_limit: usize,
    outcome: &mut PipelineOutcome,
) -> Vec<SearchResult> {
    let Pool { candidates, order } = pool;

    // --- Diversify before ranking (US-014) ---
    let pooled = candidates.len();
    let diversified = collapse_parents(candidates);
    let collapsed = pooled - diversified.len();
    outcome.counts.deduplicated = diversified.len();

    let ranked = match order {
        PoolOrder::Ranked => {
            let mut ranked = rerank_pool(params, diversified, chunk_count, outcome).await;
            // --- Preferences: secondary ordering key, never a score edit ---
            if let Some(boost) = params.preference_boost {
                apply_preference_ordering(&mut ranked, boost);
            }
            ranked
        }
        // A uniform pool has no ranking to refine and no "best" to promote.
        // Reordering it would only replace the notebook's own reading order
        // with an arbitrary one.
        PoolOrder::Uniform => diversified,
    };

    // --- Selection: the only step that changes membership (US-013) ---
    let mut results = select_final(ranked, final_limit);
    outcome.counts.selected = results.len();
    outcome.unique_parents = unique_parent_count(&results);

    if results.len() < final_limit && collapsed > 0 {
        outcome.reasons.insert(ReasonCode::DedupShortfall);
    }

    // Underfill counts only while material was left behind. A stuffed pool is
    // the whole notebook, so there is nothing left by construction.
    let material_left = match order {
        PoolOrder::Ranked => i64::try_from(results.len()).unwrap_or(i64::MAX) < chunk_count,
        PoolOrder::Uniform => false,
    };
    outcome.confidence = verdict(results.len(), final_limit, material_left);
    if let Some(reason) = outcome.confidence.reason_code() {
        outcome.reasons.insert(reason);
    }

    // --- Sandwich ordering (lost-in-the-middle mitigation) ---
    // Presentation only, after selection: it reorders what was chosen and can
    // never change what was chosen (US-014).
    if order == PoolOrder::Ranked && results.len() >= 3 {
        results = sandwich_order(results);
    }

    results
}

/// Rerank the whole diversified pool, when a reranker applies (US-014).
async fn rerank_pool(
    params: &RetrievalParams<'_>,
    diversified: Vec<SearchResult>,
    chunk_count: i64,
    outcome: &mut PipelineOutcome,
) -> Vec<SearchResult> {
    if chunk_count < i64::from(params.config.rerank_min_chunks) {
        tracing::debug!(
            notebook_id = %params.scope.notebook_id,
            chunk_count,
            rerank_min_chunks = params.config.rerank_min_chunks,
            "Reranking skipped: chunk count below threshold"
        );
        return diversified;
    }
    if diversified.is_empty() {
        return diversified;
    }

    outcome.counts.reranker_input = diversified.len();
    let rerank_start = std::time::Instant::now();
    match rerank_results(params.reranker, params.query, diversified.as_slice()).await {
        Ok(Some(reranked)) => {
            outcome.rerank_ms = rerank_start.elapsed().as_millis();
            outcome.counts.reranker_output = reranked.len();
            outcome.score_domain = Some(ScoreDomain::RerankerRelevance);
            tracing::info!(
                rerank_ms = outcome.rerank_ms,
                notebook_id = %params.scope.notebook_id,
                "Reranking completed"
            );
            reranked
        }
        Ok(None) => {
            outcome.counts.reranker_input = 0;
            outcome.reasons.insert(ReasonCode::RerankerAbsent);
            diversified
        }
        Err(e) => {
            outcome.rerank_ms = rerank_start.elapsed().as_millis();
            outcome.reasons.insert(ReasonCode::RerankerFailed);
            tracing::warn!(
                notebook_id = %params.scope.notebook_id,
                error = %e,
                pool = diversified.len(),
                rerank_ms = outcome.rerank_ms,
                "Reranking failed, falling back to the fusion order"
            );
            diversified
        }
    }
}

/// Decide whether a selection is the evidence that was asked for.
///
/// `material_left` is what keeps a small notebook from paying for a
/// reformulation it cannot benefit from: a three-chunk notebook answering a
/// request for fifteen contexts has given everything it has, and calling that
/// "insufficient" would make every turn on it retry (US-012).
const fn verdict(selected: usize, requested: usize, material_left: bool) -> RetrievalConfidence {
    if selected == 0 {
        return RetrievalConfidence::Insufficient(InsufficiencyReason::NoCandidates);
    }
    if selected < requested && material_left {
        return RetrievalConfidence::Insufficient(InsufficiencyReason::UnderfilledEvidence {
            selected,
            requested,
        });
    }
    RetrievalConfidence::Sufficient
}

// ============================================================================
// Corrective retrieval
// ============================================================================

/// Parameters for corrective RAG context retrieval.
///
/// Composes [`RetrievalParams`] rather than repeating it: the base pass and the
/// corrected pass must retrieve identically, and a field list duplicated across
/// two structs is a field that eventually differs between them.
pub struct CorrectiveRetrievalParams<'a> {
    pub base: RetrievalParams<'a>,
    pub reformulator: Option<&'a QueryReformulator>,
    pub chat_history: &'a [ChatTurn],
    /// Whether this turn already spent its one reformulation before retrieval.
    ///
    /// The chat path reformulates a follow-up question proactively. When it
    /// did, the corrective pass must not reformulate the reformulation: that
    /// is the recursive loop US-017 forbids, and the second rewrite drifts
    /// further from what the user asked.
    pub already_reformulated: bool,
}

/// Retrieve context, reformulating once when the evidence is insufficient.
///
/// "Insufficient" is a deterministic fact: nothing came back, or fewer unique
/// contexts than requested came back while the notebook still held material.
/// It is never a score comparison: the value this used to threshold was an RRF
/// rank artifact, and thresholding a provider's reranker score would require a
/// calibration this build does not have (US-012).
///
/// Stuffing skips correction entirely: the whole notebook is already in
/// context, so there is nothing a rewritten query could find.
pub async fn retrieve_context_corrective(
    params: &CorrectiveRetrievalParams<'_>,
) -> Result<(CorrectiveResult, PipelineOutcome), AppError> {
    // Reasons that belong to the *turn* rather than to one retrieval pass.
    // Reformulation is attempted once for the turn and both passes inherit its
    // outcome, so it is accumulated here and merged into whichever pass is
    // kept, instead of being written into one outcome and fished back out of
    // it with a variant filter.
    let mut turn_reasons = ReasonSet::default();

    let pass = corrective_pass(params, &mut turn_reasons).await?;

    let Pass {
        results,
        mut outcome,
        query,
        was_corrected,
    } = pass;
    outcome.reasons.extend(&turn_reasons);

    Ok((
        CorrectiveResult {
            confidence: outcome.confidence,
            effective_query: query,
            was_corrected,
            results,
        },
        outcome,
    ))
}

/// The retrieval pass a corrective attempt settled on.
struct Pass {
    results: Vec<SearchResult>,
    outcome: PipelineOutcome,
    /// The query that produced `results`.
    query: String,
    /// Whether a reformulation ran for this turn.
    was_corrected: bool,
}

/// Retrieve, and retry once with a rewritten query when the evidence is short.
///
/// Every early return is "keep the first pass"; only the last decision can
/// choose the second. Splitting it out of the public function is what lets each
/// of those be one `return` instead of one call to a five-argument constructor.
async fn corrective_pass(
    params: &CorrectiveRetrievalParams<'_>,
    turn_reasons: &mut ReasonSet,
) -> Result<Pass, AppError> {
    let query = params.base.query;
    let (results, outcome) = retrieve_context(&params.base).await?;
    let uncorrected = |results, outcome| Pass {
        results,
        outcome,
        query: query.to_string(),
        was_corrected: false,
    };

    if outcome.stuffed || outcome.confidence.is_sufficient() {
        return Ok(uncorrected(results, outcome));
    }

    tracing::debug!(
        notebook_id = %params.base.scope.notebook_id,
        query_hash = %query_hash(query),
        confidence = ?outcome.confidence,
        result_count = results.len(),
        "Retrieval produced insufficient evidence"
    );

    // One corrective reformulation, if this turn has one left.
    let Some(reformulator) = params.reformulator else {
        turn_reasons.insert(ReasonCode::ReformulationSkipped);
        return Ok(uncorrected(results, outcome));
    };
    if params.already_reformulated {
        tracing::debug!(
            notebook_id = %params.base.scope.notebook_id,
            "Skipping corrective reformulation: this turn already reformulated once"
        );
        turn_reasons.insert(ReasonCode::ReformulationSkipped);
        return Ok(uncorrected(results, outcome));
    }

    let reformulation = reformulator.reformulate(query, params.chat_history).await;
    turn_reasons.extend(reformulation.outcome.reason_codes());

    if !reformulation.was_reformulated {
        return Ok(uncorrected(results, outcome));
    }
    turn_reasons.insert(ReasonCode::CorrectiveRetrievalTriggered);

    // Retry with the reformulated query. The corrective pass embeds a
    // *different* text under a different role; both facts belong in the cache
    // key, or the second pass would be served the first pass's vector and
    // corrective retrieval would be a no-op (US-011).
    let corrected_params = RetrievalParams {
        query: &reformulation.query,
        embedding_kind: QueryEmbeddingKind::Reformulated,
        ..params.base
    };
    let (corrected_results, corrected_outcome) = retrieve_context(&corrected_params).await?;

    tracing::debug!(
        notebook_id = %params.base.scope.notebook_id,
        query_hash = %query_hash(query),
        reformulated_query_hash = %query_hash(&reformulation.query),
        original_contexts = outcome.unique_parents,
        corrected_contexts = corrected_outcome.unique_parents,
        "Corrective retrieval completed"
    );

    // Which pass to keep is decided on counts, not on scores: the two passes
    // ran different queries, and comparing their score magnitudes, even
    // within one domain, compares two different questions (US-012).
    let corrected_is_better = corrected_outcome.confidence.is_sufficient()
        || corrected_outcome.unique_parents > outcome.unique_parents;

    Ok(if corrected_is_better {
        Pass {
            results: corrected_results,
            outcome: corrected_outcome,
            query: reformulation.query,
            was_corrected: true,
        }
    } else {
        Pass {
            was_corrected: true,
            ..uncorrected(results, outcome)
        }
    })
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use uuid::Uuid;

    use async_trait::async_trait;

    use super::*;
    use crate::core::config::tests::valid_core_config;
    use crate::core::providers::{DeterministicEmbedder, RerankedDocument};
    use crate::repositories::{ChunkSearchResult, RepoResult};
    use crate::types::RetrievalScore;

    // ====================================================================
    // Fixtures
    // ====================================================================

    /// A repository over a fixed corpus, ranking by lexical overlap.
    ///
    /// Deterministic and offline: the pipeline contract this file asserts is
    /// about counts and ordering, and a real provider would add nothing but
    /// flakiness.
    struct FakeRepo {
        chunks: Vec<ChunkSearchResult>,
        source_count: i64,
        /// Rows the dense search returns, when the index comes back short.
        dense_ceiling: Option<usize>,
    }

    impl FakeRepo {
        /// `count` chunks, `per_parent` of which share each parent passage.
        fn with_parents(count: usize, per_parent: usize) -> Self {
            let source_id = Uuid::new_v4();
            let chunks = (0..count)
                .map(|i| ChunkSearchResult {
                    id: Uuid::new_v4(),
                    generation_id: Uuid::nil(),
                    source_id,
                    chunk_index: i32::try_from(i).unwrap_or(0),
                    content: format!("retention policy chunk {i}"),
                    parent_content: Some(format!("parent {}", i / per_parent.max(1))),
                    source_title: "Handbook".to_string(),
                    // Descending, so the fake's order is the fake's ranking.
                    #[allow(clippy::cast_precision_loss)]
                    relevance_score: 1.0 - (i as f32) * 0.001,
                    metadata: None,
                })
                .collect();
            Self {
                chunks,
                source_count: 1,
                dense_ceiling: None,
            }
        }

        fn without_sources() -> Self {
            Self {
                chunks: Vec::new(),
                source_count: 0,
                dense_ceiling: None,
            }
        }

        /// A corpus whose dense search returns fewer rows than were asked for,
        /// which is what a filtered approximate scan does when it runs out of
        /// budget before finding enough matching rows (US-016).
        fn underfilled(count: usize, dense_ceiling: usize) -> Self {
            Self {
                dense_ceiling: Some(dense_ceiling),
                ..Self::with_parents(count, 1)
            }
        }
    }

    #[async_trait]
    impl SearchRepository for FakeRepo {
        async fn search_similar_chunks(
            &self,
            _scope: NotebookScope,
            _query_embedding: &[f32],
            limit: i32,
        ) -> RepoResult<Vec<ChunkSearchResult>> {
            let requested = usize::try_from(limit.max(0)).unwrap_or(0);
            let ceiling = self.dense_ceiling.unwrap_or(requested).min(requested);
            Ok(self.chunks.iter().take(ceiling).cloned().collect())
        }

        async fn search_lexical_chunks(
            &self,
            _scope: NotebookScope,
            _query: &str,
            limit: i32,
        ) -> RepoResult<Vec<ChunkSearchResult>> {
            Ok(self
                .chunks
                .iter()
                .take(usize::try_from(limit.max(0)).unwrap_or(0))
                .cloned()
                .collect())
        }

        async fn count_chunks_for_notebook(&self, _scope: NotebookScope) -> RepoResult<i64> {
            Ok(i64::try_from(self.chunks.len()).unwrap_or(i64::MAX))
        }

        async fn count_sources_for_notebook(&self, _scope: NotebookScope) -> RepoResult<i64> {
            Ok(self.source_count)
        }

        async fn get_all_chunks_for_notebook(
            &self,
            _scope: NotebookScope,
        ) -> RepoResult<Vec<ChunkSearchResult>> {
            Ok(self.chunks.clone())
        }
    }

    /// A reranker that reverses the pool, and counts what it was given.
    struct CountingReranker {
        seen: AtomicUsize,
        fails: bool,
    }

    impl CountingReranker {
        fn new() -> Self {
            Self {
                seen: AtomicUsize::new(0),
                fails: false,
            }
        }

        fn failing() -> Self {
            Self {
                seen: AtomicUsize::new(0),
                fails: true,
            }
        }
    }

    #[async_trait]
    impl Reranker for CountingReranker {
        fn name(&self) -> &str {
            "counting"
        }

        async fn rerank(
            &self,
            _query: &str,
            documents: &[String],
            _top_k: Option<usize>,
        ) -> Result<Vec<RerankedDocument>, AppError> {
            self.seen.store(documents.len(), Ordering::SeqCst);
            if self.fails {
                return Err(AppError::Internal("reranker unavailable".into()));
            }
            // Reverse the incoming order so a test can tell whether the
            // reranker's opinion survived to the final selection.
            Ok(documents
                .iter()
                .enumerate()
                .map(|(index, _)| RerankedDocument {
                    index,
                    #[allow(clippy::cast_precision_loss)]
                    relevance_score: index as f32,
                })
                .collect())
        }
    }

    fn params<'a>(
        repo: &'a dyn SearchRepository,
        config: &'a CoreConfig,
        embedder: &'a DeterministicEmbedder,
        reranker: Option<&'a dyn Reranker>,
        max_chunks: i32,
        provider: &'a str,
        model: &'a str,
    ) -> RetrievalParams<'a> {
        RetrievalParams {
            search_repo: repo,
            config,
            scope: NotebookScope::new(Uuid::new_v4(), Uuid::new_v4()),
            query: "retention policy",
            max_chunks,
            embeddings: Some(embedder),
            reranker,
            hyde_service: None,
            embedding_cache: None,
            embedding_kind: QueryEmbeddingKind::Direct,
            provider,
            model,
            preference_boost: None,
            // Large enough that these cases exercise the cardinality contract
            // rather than the token budget, which has its own tests.
            evidence_token_budget: usize::MAX,
        }
    }

    // ====================================================================
    // US-013: one final cardinality contract on every branch
    // ====================================================================

    #[tokio::test]
    async fn no_reranker_still_respects_the_requested_limit() {
        // The defect this replaces: with no reranker the pipeline returned the
        // whole pool: 50 candidates for a requested 15.
        let repo = FakeRepo::with_parents(60, 1);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 0;
        config.rerank_min_chunks = 1;
        let embedder = DeterministicEmbedder::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            15,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert_eq!(results.len(), 15, "the requested maximum is the maximum");
        assert!(outcome.reasons.contains(ReasonCode::RerankerAbsent));
        assert_eq!(outcome.score_domain, Some(ScoreDomain::RrfRank));
    }

    #[tokio::test]
    async fn stuffing_is_skipped_when_the_notebook_exceeds_the_requested_limit() {
        // 40 chunks, a model that would stuff up to 50, and a request for 15.
        // Stuffing all 40 would violate the contract, so the pipeline searches.
        let repo = FakeRepo::with_parents(40, 1);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 150;
        config.rerank_min_chunks = 10_000;
        let embedder = DeterministicEmbedder::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            15,
            "anthropic",
            "claude-haiku-4-5-20251001",
        ))
        .await
        .expect("retrieval succeeds");

        assert!(!outcome.stuffed, "stuffing must not exceed the request");
        assert_eq!(results.len(), 15);
    }

    #[tokio::test]
    async fn stuffing_applies_when_the_whole_notebook_fits_the_request() {
        let repo = FakeRepo::with_parents(8, 1);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 150;
        let embedder = DeterministicEmbedder::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            15,
            "anthropic",
            "claude-haiku-4-5-20251001",
        ))
        .await
        .expect("retrieval succeeds");

        assert!(outcome.stuffed);
        assert_eq!(results.len(), 8);
        assert!(results.len() <= 15, "still bounded by the request");
        assert_eq!(outcome.score_domain, Some(ScoreDomain::StuffingUniform));
        assert!(
            results.iter().all(|r| r.score == RetrievalScore::Stuffed),
            "a stuffed chunk carries no ranking information"
        );
        assert!(outcome.confidence.is_sufficient());
    }

    /// A notebook that fits the requested count but not the token budget is
    /// searched, not stuffed: a "stuffed" context the prompt then trims is not
    /// the whole notebook, which is the only thing stuffing promises
    /// (US-018 AC-2).
    #[tokio::test]
    async fn stuffing_is_skipped_when_the_notebook_exceeds_the_token_budget() {
        let repo = FakeRepo::with_parents(8, 1);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 150;
        let embedder = DeterministicEmbedder::new();

        let generous = params(
            &repo,
            &config,
            &embedder,
            None,
            15,
            "anthropic",
            "claude-haiku-4-5-20251001",
        );
        let (_, generous_outcome) = retrieve_context(&generous)
            .await
            .expect("retrieval succeeds");
        assert!(
            generous_outcome.stuffed,
            "the same notebook stuffs when the budget allows it"
        );

        let starved = RetrievalParams {
            evidence_token_budget: 10,
            ..params(
                &repo,
                &config,
                &embedder,
                None,
                15,
                "anthropic",
                "claude-haiku-4-5-20251001",
            )
        };
        let (results, outcome) = retrieve_context(&starved)
            .await
            .expect("retrieval succeeds");

        assert!(!outcome.stuffed, "the token budget refused the whole load");
        assert!(outcome.reasons.contains(ReasonCode::StuffingOverBudget));
        assert_eq!(
            outcome.score_domain,
            Some(ScoreDomain::RrfRank),
            "it fell back to search, which ranks"
        );
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn a_stuffed_notebook_collapses_its_overlapping_children_too() {
        // Eight chunks over four parent passages. Stuffing used to hand all
        // eight to the model, so the prompt carried each passage twice: the
        // defect US-014 removes everywhere else in the pipeline.
        let repo = FakeRepo::with_parents(8, 2);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 150;
        let embedder = DeterministicEmbedder::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            15,
            "anthropic",
            "claude-haiku-4-5-20251001",
        ))
        .await
        .expect("retrieval succeeds");

        assert!(outcome.stuffed);
        assert_eq!(results.len(), 4, "one passage is one context");
        assert_eq!(
            unique_parent_count(&results),
            results.len(),
            "a stuffed selection must not contain two children of one parent"
        );
        assert_eq!(outcome.counts.deduplicated, 4);
        assert!(
            outcome.confidence.is_sufficient(),
            "the whole notebook was loaded, so nothing was left behind"
        );
        // The reading order of the document survives: stuffing has no ranking
        // to impose one of its own.
        let indices: Vec<i32> = results.iter().map(|r| r.chunk_index).collect();
        assert_eq!(indices, vec![0, 2, 4, 6]);
    }

    #[tokio::test]
    async fn a_failed_reranker_still_respects_the_requested_limit() {
        let repo = FakeRepo::with_parents(60, 1);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 0;
        config.rerank_min_chunks = 1;
        let embedder = DeterministicEmbedder::new();
        let reranker = CountingReranker::failing();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            Some(&reranker),
            12,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert_eq!(results.len(), 12);
        assert!(outcome.reasons.contains(ReasonCode::RerankerFailed));
        assert_eq!(
            outcome.score_domain,
            Some(ScoreDomain::RrfRank),
            "a failed rerank leaves the fusion domain in place"
        );
    }

    #[tokio::test]
    async fn every_branch_returns_between_zero_and_the_requested_maximum() {
        let embedder = DeterministicEmbedder::new();
        let reranker = CountingReranker::new();

        /// One configuration branch of the cardinality contract.
        struct Branch<'a> {
            count: usize,
            per_parent: usize,
            limit: i32,
            stuffing: i32,
            rerank_min: i32,
            reranker: Option<&'a dyn Reranker>,
        }

        let branch = |count, per_parent, limit, stuffing, rerank_min, reranker| Branch {
            count,
            per_parent,
            limit,
            stuffing,
            rerank_min,
            reranker,
        };

        let matrix = vec![
            branch(0, 1, 10, 150, 1, None),
            branch(3, 1, 10, 150, 1, None),
            // Stuffing with overlapping children: the branch that used to
            // bypass diversification entirely.
            branch(8, 4, 15, 150, 1, None),
            branch(12, 3, 15, 150, 10_000, Some(&reranker as &dyn Reranker)),
            branch(60, 1, 15, 0, 1, Some(&reranker as &dyn Reranker)),
            branch(60, 1, 15, 0, 10_000, Some(&reranker as &dyn Reranker)),
            branch(60, 6, 15, 0, 1, Some(&reranker as &dyn Reranker)),
            branch(60, 6, 1, 0, 1, None),
            branch(200, 1, 20, 150, 1, Some(&reranker as &dyn Reranker)),
        ];

        for Branch {
            count,
            per_parent,
            limit,
            stuffing,
            rerank_min,
            reranker,
        } in matrix
        {
            let repo = FakeRepo::with_parents(count, per_parent);
            let mut config = valid_core_config();
            config.context_stuffing_max_chunks = stuffing;
            config.rerank_min_chunks = rerank_min;

            let (results, outcome) = retrieve_context(&params(
                &repo,
                &config,
                &embedder,
                reranker,
                limit,
                "anthropic",
                "claude-haiku-4-5-20251001",
            ))
            .await
            .expect("retrieval succeeds");

            let requested = usize::try_from(limit).expect("positive");
            assert!(
                results.len() <= requested,
                "{count} chunks / {per_parent} per parent / limit {limit}: returned {}",
                results.len()
            );
            assert_eq!(outcome.counts.selected, results.len());
            assert_eq!(
                unique_parent_count(&results),
                results.len(),
                "a selection must never contain two children of one parent \
                 ({count} chunks / {per_parent} per parent / limit {limit})"
            );
        }
    }

    #[tokio::test]
    async fn truncation_is_deterministic_across_runs() {
        let repo = FakeRepo::with_parents(60, 1);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 0;
        config.rerank_min_chunks = 10_000;
        let embedder = DeterministicEmbedder::new();

        let ids = |results: &[SearchResult]| results.iter().map(|r| r.chunk_id).collect::<Vec<_>>();

        let first = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            10,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds")
        .0;
        let second = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            10,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds")
        .0;

        assert_eq!(ids(&first), ids(&second));
    }

    #[tokio::test]
    async fn a_short_dense_result_set_is_recorded_rather_than_reported_as_success() {
        // The corpus holds 200 chunks and the pool asks for 50, but the dense
        // search returns 4. That is the filtered-ANN failure mode: without the
        // reason code it is indistinguishable from a notebook with four
        // passages (US-016 AC-5).
        let repo = FakeRepo::underfilled(200, 4);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 0;
        config.rerank_min_chunks = 10_000;
        config.retrieval_pool_size = 50;
        let embedder = DeterministicEmbedder::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            10,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert!(
            outcome.reasons.contains(ReasonCode::AnnUnderfilled),
            "an underfilled dense scan must be recorded: {:?}",
            outcome.reasons
        );
        assert!(results.len() <= 10);
        assert_eq!(
            outcome.counts.dense, 4,
            "the trace must carry what the dense leg actually returned"
        );
        // The final evidence can still be sufficient: the lexical leg fills the
        // pool the dense leg came up short on. That is exactly why the shortfall
        // needs its own reason code instead of being inferred from the result
        // count, which shows nothing.
        assert!(outcome.confidence.is_sufficient());
    }

    #[tokio::test]
    async fn a_full_dense_result_set_records_no_underfill() {
        let repo = FakeRepo::with_parents(200, 1);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 0;
        config.rerank_min_chunks = 10_000;
        config.retrieval_pool_size = 50;
        let embedder = DeterministicEmbedder::new();

        let (_, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            10,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert!(!outcome.reasons.contains(ReasonCode::AnnUnderfilled));
    }

    #[tokio::test]
    async fn an_empty_notebook_is_reported_as_an_empty_corpus() {
        let repo = FakeRepo::without_sources();
        let config = valid_core_config();
        let embedder = DeterministicEmbedder::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            10,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert!(results.is_empty());
        assert!(outcome.reasons.contains(ReasonCode::EmptyCorpus));
        assert_eq!(
            outcome.confidence,
            RetrievalConfidence::Insufficient(InsufficiencyReason::NoCandidates)
        );
        assert_eq!(
            outcome.score_domain, None,
            "nothing was ranked, so no scale ordered anything"
        );
    }

    #[tokio::test]
    async fn a_configured_source_without_active_chunks_is_no_evidence_not_no_sources() {
        let repo = FakeRepo::with_parents(0, 1);
        let config = valid_core_config();
        let embedder = DeterministicEmbedder::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            10,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert!(results.is_empty());
        assert!(!outcome.reasons.contains(ReasonCode::EmptyCorpus));
        assert!(outcome.reasons.contains(ReasonCode::NoCandidates));
    }

    // ====================================================================
    // US-014: diversify, rerank the pool, select last
    // ====================================================================

    #[tokio::test]
    async fn the_reranker_receives_the_diversified_pool_not_the_final_limit() {
        // 60 candidates, 3 children per parent → 20 unique contexts. The
        // reranker must see all 20, not the 15 the caller asked for.
        let repo = FakeRepo::with_parents(60, 3);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 0;
        config.rerank_min_chunks = 1;
        config.retrieval_pool_size = 60;
        let embedder = DeterministicEmbedder::new();
        let reranker = CountingReranker::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            Some(&reranker),
            15,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert_eq!(
            reranker.seen.load(Ordering::SeqCst),
            20,
            "the reranker must judge every distinct context in the pool"
        );
        assert_eq!(outcome.counts.reranker_input, 20);
        assert_eq!(results.len(), 15);
        assert_eq!(outcome.score_domain, Some(ScoreDomain::RerankerRelevance));
    }

    #[tokio::test]
    async fn a_pool_dominated_by_one_parent_returns_fewer_contexts_not_duplicates() {
        // 30 candidates, all children of one parent → exactly one context.
        let repo = FakeRepo::with_parents(30, 30);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 0;
        config.rerank_min_chunks = 10_000;
        let embedder = DeterministicEmbedder::new();

        let (results, outcome) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            10,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert_eq!(
            results.len(),
            1,
            "one passage is one context, however many children matched"
        );
        assert!(
            outcome.reasons.contains(ReasonCode::DedupShortfall),
            "the shortfall must be reported, not padded away"
        );
        assert_eq!(
            results[0].collapsed_children.len(),
            29,
            "the representative keeps the provenance of what it absorbed"
        );
    }

    #[tokio::test]
    async fn sandwich_ordering_reorders_the_selection_without_changing_it() {
        let repo = FakeRepo::with_parents(60, 1);
        let mut config = valid_core_config();
        config.context_stuffing_max_chunks = 0;
        config.rerank_min_chunks = 10_000;
        let embedder = DeterministicEmbedder::new();

        let (results, _) = retrieve_context(&params(
            &repo,
            &config,
            &embedder,
            None,
            5,
            "anthropic",
            "claude-sonnet-4-6-20260220",
        ))
        .await
        .expect("retrieval succeeds");

        assert_eq!(results.len(), 5);
        // The second-best moved to the end; membership is unchanged, which is
        // what "presentation only" means.
        let mut by_score = results.clone();
        by_score.sort_by(|a, b| a.score.cmp_desc(b.score));
        assert_eq!(by_score[0].chunk_id, results[0].chunk_id);
        assert_eq!(by_score[1].chunk_id, results[results.len() - 1].chunk_id);
    }

    // ====================================================================
    // US-012: confidence is a fact about the result set
    // ====================================================================

    #[test]
    fn an_exhausted_corpus_is_sufficient_even_below_the_requested_limit() {
        assert_eq!(verdict(3, 15, false), RetrievalConfidence::Sufficient);
        assert_eq!(
            verdict(0, 15, false),
            RetrievalConfidence::Insufficient(InsufficiencyReason::NoCandidates)
        );
        assert_eq!(
            verdict(4, 15, true),
            RetrievalConfidence::Insufficient(InsufficiencyReason::UnderfilledEvidence {
                selected: 4,
                requested: 15
            })
        );
        assert_eq!(verdict(15, 15, true), RetrievalConfidence::Sufficient);
    }

    #[test]
    fn an_insufficiency_maps_to_a_stable_reason_code() {
        assert_eq!(RetrievalConfidence::Sufficient.reason_code(), None);
        assert_eq!(
            RetrievalConfidence::Insufficient(InsufficiencyReason::NoCandidates).reason_code(),
            Some(ReasonCode::NoCandidates)
        );
        assert_eq!(
            RetrievalConfidence::Insufficient(InsufficiencyReason::UnderfilledEvidence {
                selected: 1,
                requested: 5
            })
            .reason_code(),
            Some(ReasonCode::UnderfilledTopK)
        );
    }
}
