//! Deterministic retrieval evaluation (US-002).
//!
//! One runner, four modes, one report. It drives the real
//! [`search`](crate::services::rag::search::search) orchestration over the
//! in-memory [`CorpusIndex`], so a change to fusion, limits or filtering shows
//! up here rather than in a user's notebook.
//!
//! # Two artifacts, on purpose
//!
//! [`RetrievalRun`] carries a deterministic [`RetrievalReport`] and a
//! machine-dependent [`LatencyReport`]. Keeping wall-clock out of the compared
//! payload is what lets "two runs at the same revision produce byte-identical
//! JSON" be a real assertion instead of a tolerance. It is the same split the
//! repository already makes for `contracts/baseline/latency/`.
//!
//! # What the numbers mean
//!
//! Every metric here is computed against the deterministic in-process embedder,
//! which is a hashing bag-of-words model. Lexical-overlap categories score well
//! and `semantic_paraphrase` scores near zero **by construction** — that is a
//! property of the fixture provider, not a defect of the pipeline. The PRD's
//! absolute targets (Recall@10 >= 0.90) are written for a real embedding
//! provider. What this report is for is *movement*: the same corpus, the same
//! provider, two revisions.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::config::HybridSearchConfig;
use crate::core::providers::DeterministicEmbedder;
use crate::error::AppError;
use crate::llm::budget::estimate_tokens;
use crate::repositories::SearchRepository;
use crate::services::rag::search::types::SearchResult;
use crate::services::rag::search::{SearchMode, SearchRequest, search};

use super::corpus::{EvalCorpus, EvalQuery, QueryCategory, Split};
use super::index::CorpusIndex;
use super::trace::{ReasonCode, RetrievalTrace, ScoreDomain};

/// Bumped whenever the report shape changes. A comparison across schema
/// versions is refused rather than silently misaligned.
pub const RETRIEVAL_REPORT_SCHEMA: &str = "rag-retrieval-eval/1";

/// Decimal places every reported ratio is rounded to.
///
/// Not cosmetic: it makes a report diff readable and keeps a last-bit
/// accumulation difference from reading as a regression.
const PRECISION: f64 = 1_000_000.0;

fn round(value: f64) -> f64 {
    if value.is_finite() {
        (value * PRECISION).round() / PRECISION
    } else {
        // Never emit NaN or infinity into a report (US-002 AC-5). A non-finite
        // aggregate means "no observation", and zero with an explicit reason
        // code alongside it is the honest encoding.
        0.0
    }
}

// ============================================================================
// Modes
// ============================================================================

/// Which retrieval path to measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    /// Vector search only, through the production orchestration.
    Dense,
    /// Full-text search only.
    Lexical,
    /// Dense and lexical, fused by reciprocal rank.
    Hybrid,
    /// Brute-force exact cosine, bypassing the orchestration.
    ///
    /// The reference the other modes are read against, and the shape US-015
    /// will use to measure a real approximate index.
    ExactReference,
}

impl RetrievalMode {
    pub const ALL: &'static [Self] = &[
        Self::Dense,
        Self::Lexical,
        Self::Hybrid,
        Self::ExactReference,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Lexical => "lexical",
            Self::Hybrid => "hybrid",
            Self::ExactReference => "exact_reference",
        }
    }

    /// Parse a mode from its wire name.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|m| m.as_str() == value)
    }

    /// The scale the final ordering is expressed on.
    #[must_use]
    pub const fn score_domain(self) -> ScoreDomain {
        match self {
            Self::Dense | Self::ExactReference => ScoreDomain::DenseSimilarity,
            Self::Lexical => ScoreDomain::LexicalRank,
            Self::Hybrid => ScoreDomain::RrfRank,
        }
    }
}

impl fmt::Display for RetrievalMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Run configuration
// ============================================================================

/// Everything that changes what a run measures.
///
/// Recorded verbatim in the report header: two reports whose configurations
/// differ are not comparable, and the comparison gate refuses them.
#[derive(Debug, Clone)]
pub struct RetrievalRunConfig {
    pub mode: RetrievalMode,
    pub split: Split,
    /// Results requested per query. The fill-rate denominator.
    pub limit: i32,
    pub fusion: HybridSearchConfig,
    /// Identifies the code under measurement.
    pub code_revision: String,
}

impl Default for RetrievalRunConfig {
    fn default() -> Self {
        Self {
            mode: RetrievalMode::Hybrid,
            // Tuning work gets the training split unless it says otherwise.
            split: Split::Train,
            limit: 10,
            fusion: HybridSearchConfig::default(),
            code_revision: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

/// Fusion parameters, as recorded in the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionReport {
    pub enabled: bool,
    pub rrf_k: f32,
    pub dense_weight: f32,
    pub sparse_weight: f32,
}

impl From<&HybridSearchConfig> for FusionReport {
    fn from(config: &HybridSearchConfig) -> Self {
        Self {
            enabled: config.enabled,
            rrf_k: config.rrf_k,
            dense_weight: config.dense_weight,
            sparse_weight: config.sparse_weight,
        }
    }
}

// ============================================================================
// Report
// ============================================================================

/// Ranking quality for one population of queries.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalMetrics {
    /// Queries in this population.
    pub queries: usize,
    /// Queries that carried relevance judgments, and so contributed to the
    /// ranking metrics below. The denominator, stated rather than implied.
    pub judged_queries: usize,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub recall_at_20: f64,
    pub mrr: f64,
    pub ndcg_at_10: f64,
    /// Fraction of labeled relevant sources present in the result set.
    pub source_recall: f64,
    /// Fraction of returned results whose parent context already appeared.
    pub duplicate_parent_rate: f64,
    /// Returned results over requested results, capped at one.
    pub top_k_fill_rate: f64,
    /// Result sets containing a chunk from a forbidden source. Any non-zero
    /// value blocks a release (US-004).
    pub isolation_failures: usize,
    /// Reason code counts. Present even when zero-valued keys are absent, so a
    /// new failure mode appears in the diff.
    pub reasons: BTreeMap<String, usize>,
}

/// What happened for one query.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QueryOutcome {
    pub query_id: String,
    pub category: String,
    pub split: String,
    pub answerable: bool,
    pub returned: usize,
    pub unique_parents: usize,
    /// 1-indexed positions at which a labeled relevant chunk was returned.
    pub hit_ranks: Vec<usize>,
    /// `null` when the query carries no judgment, never `NaN`.
    pub recall_at_10: Option<f64>,
    pub reciprocal_rank: Option<f64>,
    pub ndcg_at_10: Option<f64>,
    pub isolation_breach: bool,
    pub trace: RetrievalTrace,
}

/// The deterministic half of a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RetrievalReport {
    pub schema: String,
    /// The one field excluded from the byte-stability contract.
    pub generated_at: String,
    pub corpus_version: String,
    pub corpus_generation: String,
    pub chunking_version: String,
    pub code_revision: String,
    pub embedding_fingerprint: String,
    /// `"none"` here: the deterministic path has no cross-encoder, and a
    /// reranker score would be a different score domain anyway.
    pub reranker_fingerprint: String,
    pub mode: String,
    pub split: String,
    pub requested_limit: i32,
    pub fusion: FusionReport,
    /// What a reader must know before quoting a number from this file.
    pub notes: Vec<String>,
    pub overall: RetrievalMetrics,
    pub by_category: BTreeMap<String, RetrievalMetrics>,
    pub queries: Vec<QueryOutcome>,
}

/// The machine-dependent half of a run. Never compared, never asserted.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatencyReport {
    pub schema: String,
    pub mode: String,
    pub split: String,
    pub samples: usize,
    pub p50_us: u64,
    pub p95_us: u64,
    pub note: String,
}

/// One run's two artifacts.
#[derive(Debug, Clone)]
pub struct RetrievalRun {
    pub report: RetrievalReport,
    pub latency: LatencyReport,
}

// ============================================================================
// Runner
// ============================================================================

/// Score one query, and everything measured about it.
struct Scored {
    outcome: QueryOutcome,
    recall_at_5: Option<f64>,
    recall_at_20: Option<f64>,
    source_recall: Option<f64>,
    duplicate_parent_rate: f64,
    fill_rate: f64,
    latency_us: u64,
}

/// Run the retrieval evaluation.
///
/// `generated_at` is supplied by the caller rather than read from the clock:
/// the runner has no other source of nondeterminism, and taking the timestamp
/// as a parameter is what makes the determinism test possible without special
/// casing it.
///
/// # Errors
/// Returns the embedding provider's error. Nothing here touches the network.
pub async fn run_retrieval_eval(
    corpus: &EvalCorpus,
    index: &CorpusIndex,
    config: &RetrievalRunConfig,
    generated_at: &str,
) -> Result<RetrievalRun, AppError> {
    let embedder = DeterministicEmbedder::new();
    let queries = corpus.queries(config.split);

    let mut scored = Vec::with_capacity(queries.len());
    for query in &queries {
        scored.push(score_query(corpus, index, &embedder, config, query).await?);
    }

    let overall = aggregate(&scored, |_| true);
    let mut by_category = BTreeMap::new();
    for category in QueryCategory::ALL {
        let key = category.as_str();
        let metrics = aggregate(&scored, |s| s.outcome.category == key);
        // A category with no case in this split is reported as an empty
        // population rather than omitted, so the shape of the report does not
        // depend on the split.
        by_category.insert(key.to_owned(), metrics);
    }

    let mut latencies: Vec<u64> = scored.iter().map(|s| s.latency_us).collect();
    latencies.sort_unstable();

    let report = RetrievalReport {
        schema: RETRIEVAL_REPORT_SCHEMA.to_owned(),
        generated_at: generated_at.to_owned(),
        corpus_version: corpus.version().to_owned(),
        corpus_generation: corpus.generation().to_owned(),
        chunking_version: corpus.chunking_version().to_owned(),
        code_revision: config.code_revision.clone(),
        embedding_fingerprint: index.embedding_fingerprint(),
        reranker_fingerprint: "none".to_owned(),
        mode: config.mode.as_str().to_owned(),
        split: config.split.as_str().to_owned(),
        requested_limit: config.limit,
        fusion: FusionReport::from(&config.fusion),
        notes: notes_for(config.mode),
        overall,
        by_category,
        queries: scored.into_iter().map(|s| s.outcome).collect(),
    };

    let latency = LatencyReport {
        schema: RETRIEVAL_REPORT_SCHEMA.to_owned(),
        mode: config.mode.as_str().to_owned(),
        split: config.split.as_str().to_owned(),
        samples: latencies.len(),
        p50_us: percentile(&latencies, 50.0),
        p95_us: percentile(&latencies, 95.0),
        note: "Machine-dependent. Excluded from the byte-stability contract and \
               from the release comparison."
            .to_owned(),
    };

    Ok(RetrievalRun { report, latency })
}

fn notes_for(mode: RetrievalMode) -> Vec<String> {
    let mut notes = vec![
        "Embeddings come from the deterministic in-process hashing provider. \
         Semantic categories score low by construction; the PRD's absolute \
         targets assume a real embedding model."
            .to_owned(),
        "Lexical scores come from a BM25-shaped in-memory scorer, not from \
         PostgreSQL ts_rank_cd. They are comparable across revisions of this \
         repository and not against a database."
            .to_owned(),
    ];
    if mode == RetrievalMode::Hybrid {
        notes.push(
            "Fused results carry the production tie-break (score desc, chunk \
             id asc), applied inside reciprocal rank fusion since US-013. The \
             evaluator re-imposes the same order defensively, so a regression \
             in the pipeline's tie-break shows up as a report diff rather than \
             as a flapping report."
                .to_owned(),
        );
    }
    notes
}

/// Nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

/// Retrieve for one query and score the result set.
async fn score_query(
    corpus: &EvalCorpus,
    index: &CorpusIndex,
    embedder: &DeterministicEmbedder,
    config: &RetrievalRunConfig,
    query: &EvalQuery,
) -> Result<Scored, AppError> {
    let notebook_id = corpus
        .notebook(&query.notebook)
        .map_or_else(Uuid::nil, super::corpus::CorpusNotebook::uuid);

    let mut trace = RetrievalTrace::new(
        notebook_id,
        &query.query,
        config.mode.as_str(),
        Some(config.mode.score_domain()),
    );
    trace.generation_ids = vec![corpus.generation_id()];

    let started = Instant::now();
    let retrieved = prepare(index, embedder, config, notebook_id, &query.query).await?;
    #[allow(clippy::cast_possible_truncation)]
    let latency_us = started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64;

    if retrieved.dropped_non_finite > 0 {
        trace.reasons.insert(ReasonCode::NonFiniteScore);
    }
    let results = retrieved.results;
    let requested = usize::try_from(config.limit.max(0)).unwrap_or(0);

    // --- Counts and trace ------------------------------------------------
    let returned = results.len();
    let unique_parents = unique_parent_count(&results);
    trace.candidates.selected = returned;
    trace.unique_parents = unique_parents;
    match config.mode {
        RetrievalMode::Hybrid => {
            trace.candidates.fused = returned;
        }
        RetrievalMode::Lexical => trace.candidates.lexical = returned,
        RetrievalMode::Dense | RetrievalMode::ExactReference => trace.candidates.dense = returned,
    }
    trace.candidates.deduplicated = unique_parents;
    trace.tokens.selected = results
        .iter()
        .map(|r| estimate_tokens(r.parent_content.as_deref().unwrap_or(&r.content)))
        .sum();
    trace.durations.search_ms = 0;

    if returned == 0 {
        trace.reasons.insert(ReasonCode::NoCandidates);
    } else if returned < requested {
        trace.reasons.insert(ReasonCode::UnderfilledTopK);
    }
    if index
        .count_chunks_for_notebook(CorpusIndex::scope(notebook_id))
        .await
        .unwrap_or(0)
        == 0
    {
        trace.reasons.insert(ReasonCode::EmptyCorpus);
    }

    // --- Tenant isolation -------------------------------------------------
    let forbidden: BTreeSet<Uuid> = query
        .forbidden_sources
        .iter()
        .filter_map(|slug| corpus.source(slug).map(|(_, source)| source.uuid()))
        .collect();
    let isolation_breach = results.iter().any(|r| forbidden.contains(&r.source_id));
    if isolation_breach {
        trace.reasons.insert(ReasonCode::IsolationBreach);
    }

    // --- Relevance judgments ---------------------------------------------
    let relevant: BTreeSet<Uuid> = query
        .relevant_chunks
        .iter()
        .filter_map(|slug| corpus.chunk(slug).map(|(_, _, chunk)| chunk.uuid()))
        .collect();
    let relevant_sources: BTreeSet<Uuid> = query
        .relevant_sources
        .iter()
        .filter_map(|slug| corpus.source(slug).map(|(_, source)| source.uuid()))
        .collect();

    // An answerable query with no resolvable judgment is a corpus defect that
    // must be visible in the report, not an implicit zero.
    let judged = !relevant.is_empty();
    if query.answerable && !judged {
        trace.reasons.insert(ReasonCode::MissingJudgment);
    }

    let hit_ranks: Vec<usize> = results
        .iter()
        .enumerate()
        .filter(|(_, r)| relevant.contains(&r.chunk_id))
        .map(|(i, _)| i + 1)
        .collect();

    let recall = |k: usize| -> Option<f64> {
        if !judged {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let hits = hit_ranks.iter().filter(|rank| **rank <= k).count() as f64;
        #[allow(clippy::cast_precision_loss)]
        let total = relevant.len() as f64;
        Some(round(hits / total))
    };

    let reciprocal_rank = if judged {
        #[allow(clippy::cast_precision_loss)]
        Some(round(
            hit_ranks.first().map_or(0.0, |rank| 1.0 / *rank as f64),
        ))
    } else {
        None
    };

    let ndcg = if judged {
        Some(round(ndcg_at_10(&hit_ranks, relevant.len())))
    } else {
        None
    };

    let source_recall = if relevant_sources.is_empty() {
        None
    } else {
        let found: BTreeSet<Uuid> = results
            .iter()
            .map(|r| r.source_id)
            .filter(|id| relevant_sources.contains(id))
            .collect();
        #[allow(clippy::cast_precision_loss)]
        Some(round(found.len() as f64 / relevant_sources.len() as f64))
    };

    #[allow(clippy::cast_precision_loss)]
    let duplicate_parent_rate = if returned == 0 {
        0.0
    } else {
        round((returned - unique_parents) as f64 / returned as f64)
    };
    if unique_parents < returned {
        trace.reasons.insert(ReasonCode::DedupShortfall);
    }

    #[allow(clippy::cast_precision_loss)]
    let fill_rate = if requested == 0 {
        0.0
    } else {
        round((returned as f64 / requested as f64).min(1.0))
    };

    trace.finish();

    let recall_at_5 = recall(5);
    let recall_at_10 = recall(10);
    let recall_at_20 = recall(20);

    Ok(Scored {
        outcome: QueryOutcome {
            query_id: query.id.clone(),
            category: query.category.as_str().to_owned(),
            split: query.split.as_str().to_owned(),
            answerable: query.answerable,
            returned,
            unique_parents,
            hit_ranks,
            recall_at_10,
            reciprocal_rank,
            ndcg_at_10: ndcg,
            isolation_breach,
            trace,
        },
        recall_at_5,
        recall_at_20,
        source_recall,
        duplicate_parent_rate,
        fill_rate,
        latency_us,
    })
}

/// A prepared result set: filtered, totally ordered and truncated.
#[derive(Debug, Clone)]
pub struct RetrievedSet {
    pub results: Vec<SearchResult>,
    /// Candidates rejected for carrying a non-finite score.
    pub dropped_non_finite: usize,
}

/// Retrieve for one query, through the mode under measurement.
///
/// Public because the grounded-response producer (US-003) has to retrieve
/// through exactly this path: an evaluator that scored answers built over a
/// different retrieval than the one being measured would compare two things at
/// once.
///
/// # Errors
/// Returns the embedding provider's error.
pub async fn retrieve_for_query(
    index: &CorpusIndex,
    config: &RetrievalRunConfig,
    notebook_id: Uuid,
    query: &str,
) -> Result<RetrievedSet, AppError> {
    prepare(
        index,
        &DeterministicEmbedder::new(),
        config,
        notebook_id,
        query,
    )
    .await
}

/// Retrieve, then apply the three transformations every consumer needs.
async fn prepare(
    index: &CorpusIndex,
    embedder: &DeterministicEmbedder,
    config: &RetrievalRunConfig,
    notebook_id: Uuid,
    query: &str,
) -> Result<RetrievedSet, AppError> {
    // A non-finite score cannot be ranked and must not reach a metric. The
    // pipeline drops such a candidate at the conversion boundary and reports
    // the count, so the evaluator no longer has to re-filter: a
    // `RetrievalScore` that exists is finite by construction (US-012).
    let (mut results, dropped_non_finite) =
        retrieve_with(index, embedder, config, notebook_id, query).await?;

    // One ordering pass for every mode, before the cut: truncating an
    // unordered set is what would make the report depend on retrieval's
    // internal iteration order.
    stabilize(&mut results);
    results.truncate(usize::try_from(config.limit.max(0)).unwrap_or(0));

    Ok(RetrievedSet {
        results,
        dropped_non_finite,
    })
}

/// Dispatch to the mode under measurement.
async fn retrieve_with(
    index: &CorpusIndex,
    embedder: &DeterministicEmbedder,
    config: &RetrievalRunConfig,
    notebook_id: Uuid,
    query: &str,
) -> Result<(Vec<SearchResult>, usize), AppError> {
    // Every mode asks for the same pool: comparing modes at different pool
    // sizes would measure the pool, not the ranking.
    let pool = config.limit.max(1);

    match config.mode {
        RetrievalMode::ExactReference => {
            let chunks = index.exact_reference(notebook_id, query, pool).await?;
            let total = chunks.len();
            let results: Vec<SearchResult> = chunks
                .into_iter()
                .filter_map(|c| SearchResult::from_chunk(c, ScoreDomain::DenseSimilarity))
                .collect();
            let dropped = total - results.len();
            Ok((results, dropped))
        }
        mode => {
            let search_mode = match mode {
                RetrievalMode::Dense => SearchMode::Dense,
                RetrievalMode::Lexical => SearchMode::Lexical,
                _ => SearchMode::Hybrid,
            };
            let request = SearchRequest::new(query)
                .with_limit(pool)
                .with_mode(search_mode);
            // The evaluator drives corpus queries verbatim, with no HyDE and no
            // cache: an offline run must measure retrieval, not what a previous
            // run happened to leave behind.
            let query_embedder = crate::services::rag::search::QueryEmbedder::direct(embedder);
            let found = search(
                index,
                &config.fusion,
                CorpusIndex::scope(notebook_id),
                &request,
                &query_embedder,
            )
            .await?;
            Ok((found.results, found.dropped_non_finite))
        }
    }
}

/// Assert the total order the pipeline now guarantees.
///
/// Since US-013 the production path breaks score ties on the chunk identifier,
/// so a fused result set arrives here already totally ordered. This re-imposes
/// the same order for the modes that bypass fusion (the exact reference) and
/// keeps the evaluator honest if the pipeline's tie-break ever regresses: the
/// sort is idempotent when the invariant holds and repairs the report when it
/// does not, which a `debug_assert` alone could not do in a release run.
fn stabilize(results: &mut [SearchResult]) {
    results.sort_by(|a, b| {
        a.score
            .cmp_desc(b.score)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
}

/// Distinct parent contexts in a result set.
///
/// Two children of the same parent within the same source count once. Different
/// sources never collapse, matching the production deduplication rule that
/// preserves citation attribution.
fn unique_parent_count(results: &[SearchResult]) -> usize {
    let mut seen: BTreeSet<(Uuid, &str)> = BTreeSet::new();
    let mut singletons = 0;
    for result in results {
        match result.parent_content.as_deref() {
            Some(parent) => {
                seen.insert((result.source_id, parent));
            }
            // A chunk with no parent is its own context and cannot duplicate
            // another.
            None => singletons += 1,
        }
    }
    seen.len() + singletons
}

/// Binary-gain nDCG at 10.
fn ndcg_at_10(hit_ranks: &[usize], relevant_total: usize) -> f64 {
    if relevant_total == 0 {
        return 0.0;
    }
    let dcg: f64 = hit_ranks
        .iter()
        .filter(|rank| **rank <= 10)
        .map(|rank| {
            #[allow(clippy::cast_precision_loss)]
            let position = *rank as f64;
            1.0 / (position + 1.0).log2()
        })
        .sum();
    let ideal: f64 = (1..=relevant_total.min(10))
        .map(|rank| {
            #[allow(clippy::cast_precision_loss)]
            let position = rank as f64;
            1.0 / (position + 1.0).log2()
        })
        .sum();
    if ideal <= 0.0 { 0.0 } else { dcg / ideal }
}

/// Aggregate a population of scored queries.
fn aggregate(scored: &[Scored], select: impl Fn(&Scored) -> bool) -> RetrievalMetrics {
    let population: Vec<&Scored> = scored.iter().filter(|s| select(s)).collect();

    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for entry in &population {
        for reason in &entry.outcome.trace.reasons {
            *reasons.entry(reason.as_str().to_owned()).or_insert(0) += 1;
        }
    }

    // `mean` returns 0.0 for an empty population rather than NaN, and
    // `judged_queries` states how many observations produced the number.
    let mean = |values: Vec<f64>| -> f64 {
        if values.is_empty() {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let n = values.len() as f64;
            round(values.iter().sum::<f64>() / n)
        }
    };

    let judged: Vec<&&Scored> = population
        .iter()
        .filter(|s| s.outcome.recall_at_10.is_some())
        .collect();

    RetrievalMetrics {
        queries: population.len(),
        judged_queries: judged.len(),
        recall_at_5: mean(judged.iter().filter_map(|s| s.recall_at_5).collect()),
        recall_at_10: mean(
            judged
                .iter()
                .filter_map(|s| s.outcome.recall_at_10)
                .collect(),
        ),
        recall_at_20: mean(judged.iter().filter_map(|s| s.recall_at_20).collect()),
        mrr: mean(
            judged
                .iter()
                .filter_map(|s| s.outcome.reciprocal_rank)
                .collect(),
        ),
        ndcg_at_10: mean(judged.iter().filter_map(|s| s.outcome.ndcg_at_10).collect()),
        source_recall: mean(population.iter().filter_map(|s| s.source_recall).collect()),
        duplicate_parent_rate: mean(population.iter().map(|s| s.duplicate_parent_rate).collect()),
        top_k_fill_rate: mean(population.iter().map(|s| s.fill_rate).collect()),
        isolation_failures: population
            .iter()
            .filter(|s| s.outcome.isolation_breach)
            .count(),
        reasons,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const FIXED_TIME: &str = "2026-07-31T00:00:00Z";

    async fn corpus_and_index() -> (EvalCorpus, CorpusIndex) {
        let corpus = EvalCorpus::load_default().expect("corpus loads");
        let index = CorpusIndex::build(&corpus).await.expect("index builds");
        (corpus, index)
    }

    fn config(mode: RetrievalMode) -> RetrievalRunConfig {
        RetrievalRunConfig {
            mode,
            code_revision: "test".to_owned(),
            ..RetrievalRunConfig::default()
        }
    }

    #[tokio::test]
    async fn every_mode_produces_a_complete_report() {
        let (corpus, index) = corpus_and_index().await;
        for mode in RetrievalMode::ALL {
            let run = run_retrieval_eval(&corpus, &index, &config(*mode), FIXED_TIME)
                .await
                .expect("run");
            assert_eq!(run.report.mode, mode.as_str());
            assert_eq!(run.report.by_category.len(), QueryCategory::ALL.len());
            assert_eq!(run.report.queries.len(), run.report.overall.queries);
            assert!(run.report.overall.queries > 0, "{mode} scored no query");
        }
    }

    #[tokio::test]
    async fn two_runs_at_the_same_revision_are_byte_identical() {
        let (corpus, index) = corpus_and_index().await;
        // A second index, built independently: determinism must survive a
        // rebuild, not just a second call against the same vectors.
        let rebuilt = CorpusIndex::build(&corpus).await.expect("index");

        for mode in RetrievalMode::ALL {
            let first = run_retrieval_eval(&corpus, &index, &config(*mode), FIXED_TIME)
                .await
                .expect("run");
            let second = run_retrieval_eval(&corpus, &rebuilt, &config(*mode), FIXED_TIME)
                .await
                .expect("run");
            let a = serde_json::to_string_pretty(&first.report).expect("json");
            let b = serde_json::to_string_pretty(&second.report).expect("json");
            assert_eq!(a, b, "{mode} report is not byte-stable");
        }
    }

    #[tokio::test]
    async fn only_the_timestamp_varies_between_runs() {
        let (corpus, index) = corpus_and_index().await;
        let a = run_retrieval_eval(&corpus, &index, &config(RetrievalMode::Hybrid), "A")
            .await
            .expect("run");
        let mut b = run_retrieval_eval(&corpus, &index, &config(RetrievalMode::Hybrid), "B")
            .await
            .expect("run");
        assert_ne!(a.report.generated_at, b.report.generated_at);
        b.report.generated_at = a.report.generated_at.clone();
        assert_eq!(a.report, b.report);
    }

    #[tokio::test]
    async fn no_metric_is_ever_nan_or_infinite() {
        let (corpus, index) = corpus_and_index().await;
        for mode in RetrievalMode::ALL {
            let run = run_retrieval_eval(&corpus, &index, &config(*mode), FIXED_TIME)
                .await
                .expect("run");
            let rendered = serde_json::to_string(&run.report).expect("json");
            assert!(!rendered.contains("NaN"), "{mode}");
            assert!(!rendered.contains("null,\"recall"), "{mode}");
            for metrics in
                std::iter::once(&run.report.overall).chain(run.report.by_category.values())
            {
                for value in [
                    metrics.recall_at_5,
                    metrics.recall_at_10,
                    metrics.recall_at_20,
                    metrics.mrr,
                    metrics.ndcg_at_10,
                    metrics.source_recall,
                    metrics.duplicate_parent_rate,
                    metrics.top_k_fill_rate,
                ] {
                    assert!(value.is_finite(), "{mode}: {value}");
                    assert!((0.0..=1.0).contains(&value), "{mode}: {value}");
                }
            }
        }
    }

    #[tokio::test]
    async fn an_unanswerable_query_is_kept_with_null_ranking_metrics() {
        let (corpus, index) = corpus_and_index().await;
        let mut config = config(RetrievalMode::Hybrid);
        config.split = Split::Train;
        let run = run_retrieval_eval(&corpus, &index, &config, FIXED_TIME)
            .await
            .expect("run");

        let unanswerable: Vec<&QueryOutcome> = run
            .report
            .queries
            .iter()
            .filter(|q| !q.answerable)
            .collect();
        assert!(!unanswerable.is_empty(), "corpus has no unanswerable case");
        for outcome in unanswerable {
            assert!(outcome.recall_at_10.is_none(), "{}", outcome.query_id);
            assert!(outcome.ndcg_at_10.is_none(), "{}", outcome.query_id);
        }
        // Kept in the denominator: every query appears exactly once.
        assert_eq!(run.report.queries.len(), corpus.queries(Split::Train).len());
    }

    #[tokio::test]
    async fn an_empty_result_set_is_recorded_rather_than_dropped() {
        let (corpus, index) = corpus_and_index().await;
        // Lexical search over a corpus that contains none of the query terms is
        // the natural way to reach zero results.
        let run = run_retrieval_eval(&corpus, &index, &config(RetrievalMode::Lexical), FIXED_TIME)
            .await
            .expect("run");
        let empty: Vec<&QueryOutcome> = run
            .report
            .queries
            .iter()
            .filter(|q| q.returned == 0)
            .collect();
        for outcome in &empty {
            assert!(
                outcome.trace.reasons.contains(ReasonCode::NoCandidates),
                "{} returned nothing without a reason",
                outcome.query_id
            );
        }
        if !empty.is_empty() {
            assert!(
                run.report
                    .overall
                    .reasons
                    .contains_key(ReasonCode::NoCandidates.as_str())
            );
        }
    }

    #[tokio::test]
    async fn an_underfilled_result_set_carries_a_reason() {
        let (corpus, index) = corpus_and_index().await;
        let mut config = config(RetrievalMode::Dense);
        // Far more than any notebook holds, so every query underfills.
        config.limit = 10_000;
        let run = run_retrieval_eval(&corpus, &index, &config, FIXED_TIME)
            .await
            .expect("run");
        assert!(
            run.report.queries.iter().all(|q| {
                q.trace.reasons.contains(ReasonCode::UnderfilledTopK)
                    || q.trace.reasons.contains(ReasonCode::NoCandidates)
            }),
            "an underfilled run must say so"
        );
        assert!(run.report.overall.top_k_fill_rate < 1.0);
    }

    #[tokio::test]
    async fn tenant_isolation_holds_on_every_case() {
        let (corpus, index) = corpus_and_index().await;
        for mode in RetrievalMode::ALL {
            for split in [Split::Train, Split::Holdout] {
                let mut config = config(*mode);
                config.split = split;
                config.limit = 20;
                let run = run_retrieval_eval(&corpus, &index, &config, FIXED_TIME)
                    .await
                    .expect("run");
                assert_eq!(
                    run.report.overall.isolation_failures, 0,
                    "{mode}/{split} retrieved a forbidden source"
                );
            }
        }
    }

    #[tokio::test]
    async fn the_holdout_split_is_only_scored_when_named() {
        let (corpus, index) = corpus_and_index().await;
        let run = run_retrieval_eval(&corpus, &index, &config(RetrievalMode::Hybrid), FIXED_TIME)
            .await
            .expect("run");
        assert_eq!(run.report.split, "train");
        assert!(run.report.queries.iter().all(|q| q.split == "train"));
    }

    #[tokio::test]
    async fn a_trace_is_attached_to_every_query_and_carries_no_text() {
        let (corpus, index) = corpus_and_index().await;
        let run = run_retrieval_eval(&corpus, &index, &config(RetrievalMode::Hybrid), FIXED_TIME)
            .await
            .expect("run");
        let rendered = serde_json::to_string(&run.report.queries).expect("json");
        for query in corpus.queries(Split::Train) {
            let outcome = run
                .report
                .queries
                .iter()
                .find(|q| q.query_id == query.id)
                .expect("every query is reported");
            assert_eq!(outcome.trace.generation_ids, vec![corpus.generation_id()]);
            assert!(!outcome.trace.reasons.is_empty());
            assert!(
                !rendered.contains(&query.query),
                "trace leaked `{}`",
                query.id
            );
        }
    }

    // --- Metric arithmetic ------------------------------------------------

    #[test]
    fn ndcg_is_one_when_every_relevant_chunk_leads() {
        assert!((ndcg_at_10(&[1, 2], 2) - 1.0).abs() < 1e-12);
        assert!((ndcg_at_10(&[1], 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ndcg_is_zero_without_a_hit_or_without_a_judgment() {
        assert!(ndcg_at_10(&[], 3).abs() < f64::EPSILON);
        assert!(ndcg_at_10(&[1], 0).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_ignores_hits_past_the_cutoff() {
        assert!(ndcg_at_10(&[11, 12], 2).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_penalizes_a_lower_rank() {
        assert!(ndcg_at_10(&[1], 1) > ndcg_at_10(&[5], 1));
    }

    #[test]
    fn percentiles_use_nearest_rank_and_survive_an_empty_sample() {
        assert_eq!(percentile(&[], 50.0), 0);
        assert_eq!(percentile(&[10, 20, 30, 40], 50.0), 20);
        assert_eq!(percentile(&[10, 20, 30, 40], 95.0), 40);
    }

    #[test]
    fn round_never_lets_a_non_finite_value_reach_a_report() {
        assert!((round(f64::NAN)).abs() < f64::EPSILON);
        assert!((round(f64::INFINITY)).abs() < f64::EPSILON);
        assert!((round(0.123_456_789) - 0.123_457).abs() < 1e-12);
    }

    #[test]
    fn unique_parents_collapse_within_a_source_and_never_across_sources() {
        let source_a = Uuid::from_u128(1);
        let source_b = Uuid::from_u128(2);
        let make = |source_id: Uuid, parent: Option<&str>, n: u128| SearchResult {
            chunk_id: Uuid::from_u128(n),
            generation_id: Uuid::nil(),
            source_id,
            source_title: "t".to_owned(),
            chunk_index: 0,
            content: "c".to_owned(),
            parent_content: parent.map(str::to_owned),
            score: crate::types::RetrievalScore::Rrf(0.5),
            metadata: None,
            collapsed_children: Vec::new(),
        };

        // Two children of one parent in one source: one context.
        assert_eq!(
            unique_parent_count(&[make(source_a, Some("P"), 10), make(source_a, Some("P"), 11)]),
            1
        );
        // Same parent text, different sources: two contexts.
        assert_eq!(
            unique_parent_count(&[make(source_a, Some("P"), 10), make(source_b, Some("P"), 11)]),
            2
        );
        // Parentless chunks are each their own context.
        assert_eq!(
            unique_parent_count(&[make(source_a, None, 10), make(source_a, None, 11)]),
            2
        );
        assert_eq!(unique_parent_count(&[]), 0);
    }

    #[test]
    fn modes_round_trip_through_their_wire_names() {
        for mode in RetrievalMode::ALL {
            assert_eq!(RetrievalMode::parse(mode.as_str()), Some(*mode));
        }
        assert_eq!(RetrievalMode::parse("sparse"), None);
    }

    #[test]
    fn each_mode_declares_its_own_score_domain() {
        assert_eq!(
            RetrievalMode::Hybrid.score_domain(),
            ScoreDomain::RrfRank,
            "an RRF score is a rank score, not a similarity"
        );
        assert_eq!(
            RetrievalMode::Dense.score_domain(),
            ScoreDomain::DenseSimilarity
        );
        assert_eq!(
            RetrievalMode::Lexical.score_domain(),
            ScoreDomain::LexicalRank
        );
    }
}
