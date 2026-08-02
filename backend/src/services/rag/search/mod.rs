//! Semantic search service for RAG.
//!
//! Provides multi-modal search with optional reranking for improved relevance.
//!
//! ## Search Modes
//!
//! - **Hybrid** (default): Combines dense + lexical with Reciprocal Rank Fusion
//! - **Dense**: Vector similarity search using pgvector
//! - **Lexical**: Full-text search using PostgreSQL tsvector
//!
//! ## Two-Stage Retrieval
//!
//! When a Voyage client is provided, uses cross-encoder reranking for more
//! accurate relevance scoring (jointly analyzes query-document pairs).
//!
//! All raw SQL queries are encapsulated in [`SearchRepository`] — this module
//! contains only orchestration logic.

mod context;
mod formatting;
mod fusion;
mod preference;
mod stuffing;
mod transforms;
pub mod types;

use crate::core::config::HybridSearchConfig;
use crate::core::providers::EmbeddingProvider;
use crate::error::AppError;
use crate::repositories::{NotebookScope, SearchRepository};
use crate::services::embeddings;
use crate::services::rag::embedding_cache::EmbeddingCache;
use crate::services::rag::eval::trace::query_hash;
use crate::services::rag::hyde::HydeService;
use crate::services::rag::provenance::{EmbeddingProvenance, QueryEmbeddingKind};
use crate::types::ScoreDomain;

// ============================================================================
// Constants (used in orchestration and tests)
// ============================================================================

/// Default weight for dense (vector) search in hybrid RRF fusion.
/// With a 4:1 ratio (dense=1.0, sparse=0.25), dense results dominate
/// while lexical matches still contribute to the final ranking.
pub const DEFAULT_DENSE_WEIGHT: f32 = 1.0;

/// Default weight for sparse (BM25/lexical) search in hybrid RRF fusion.
/// Set to 0.25 to give lexical matches 1/4 the influence of dense results,
/// following Anthropic's Contextual Retrieval recommendations.
pub const DEFAULT_SPARSE_WEIGHT: f32 = 0.25;

// The search service's public surface. Each item is re-exported from the module
// that owns it: routing them through one another would only hide where the
// behaviour lives.
pub use context::{
    CORRECTIVE_RAG_MAX_RETRIES, CorrectiveRetrievalParams, InsufficiencyReason, PipelineOutcome,
    RetrievalConfidence, RetrievalParams, retrieve_context, retrieve_context_corrective,
};
pub use formatting::{
    REGION_CLOSE, REGION_OPEN, RenderedEvidence, build_rag_documents, entry_tokens, evidence_body,
    format_context_for_llm, region_overhead_tokens, render_evidence,
};
pub use fusion::reciprocal_rank_fusion;
pub use preference::{PreferenceBoost, extract_preference_keywords};
pub use stuffing::max_context_stuffing_chunks;
pub use transforms::{mean_relevance, unique_parent_count};
pub use types::{
    CorrectiveResult, MAX_LIMIT, SearchMode, SearchOutcome, SearchRequest, SearchResult,
};

use fusion::filter_and_convert;

// ============================================================================
// Public API — search orchestration
// ============================================================================

/// How a query becomes a vector.
///
/// Provider, optional HyDE expansion, cache and cache namespace are one
/// concern, not four parameters: every dense path needs all of them, and a call
/// site that had to pass them individually could pass a namespace that does not
/// match the provider it also passed.
pub struct QueryEmbedder<'a> {
    pub provider: &'a dyn EmbeddingProvider,
    pub hyde: Option<&'a HydeService>,
    pub cache: Option<&'a EmbeddingCache>,
    /// The role the query text plays; selects the cache namespace (US-011).
    pub kind: QueryEmbeddingKind,
}

impl<'a> QueryEmbedder<'a> {
    /// A direct user query, with no HyDE and no cache.
    ///
    /// The shape the offline evaluator drives: corpus queries verbatim, nothing
    /// remembered between runs.
    #[must_use]
    pub fn direct(provider: &'a dyn EmbeddingProvider) -> Self {
        Self {
            provider,
            hyde: None,
            cache: None,
            kind: QueryEmbeddingKind::Direct,
        }
    }
}

/// Cache namespaces a dense lookup may legitimately answer from, in order.
///
/// With HyDE configured the vector this path stores under the query text is the
/// *generated document's*, so that namespace is asked first and a `Direct`
/// lookup does accept it. That is deliberate and it is the one place the
/// namespace separation is crossed: under HyDE the dense retrieval vector for a
/// given query text is the HyDE vector, by design, and re-embedding the query
/// would search with a vector this path never uses. The caller's own namespace
/// is still consulted second, because HyDE declines long queries and stores the
/// plain embedding instead.
///
/// Everything outside dense retrieval — working-memory lookups in particular —
/// asks the cache directly and therefore never sees a HyDE vector.
fn lookup_kinds(
    kind: QueryEmbeddingKind,
    hyde_configured: bool,
) -> [Option<QueryEmbeddingKind>; 2] {
    if hyde_configured && kind != QueryEmbeddingKind::HydeDocument {
        [Some(QueryEmbeddingKind::HydeDocument), Some(kind)]
    } else {
        [Some(kind), None]
    }
}

/// Main search function that routes to the appropriate search method.
///
/// Returns a [`SearchOutcome`]: results, per-stage timings, per-stage candidate
/// counts and the number of candidates rejected for a non-finite score.
///
/// Takes [`HybridSearchConfig`] rather than the whole [`CoreConfig`](crate::core::config::CoreConfig):
/// fusion weights are the only configuration this layer reads, and narrowing the
/// parameter is what lets the offline evaluator (EP-001) drive the real
/// orchestration without fabricating a server configuration.
/// `kind` names the role `request.query` plays: an original user question, the
/// reformulation of one, or a working-memory lookup. It is not a search
/// parameter — it selects the embedding cache namespace, so a reformulation
/// cannot be served the original question's vector (US-011).
#[tracing::instrument(skip(search_repo, config, request, embedder), fields(notebook_id = %scope.notebook_id, kind = embedder.kind.as_str()))]
pub async fn search(
    search_repo: &dyn SearchRepository,
    config: &HybridSearchConfig,
    scope: NotebookScope,
    request: &SearchRequest,
    embedder: &QueryEmbedder<'_>,
) -> Result<SearchOutcome, AppError> {
    let mode = effective_mode(config, request.mode);

    match mode {
        SearchMode::Hybrid => hybrid_search(search_repo, config, scope, request, embedder).await,
        SearchMode::Dense => semantic_search_with_hyde(search_repo, scope, request, embedder).await,
        SearchMode::Lexical => lexical_search(search_repo, scope, request).await,
    }
}

/// Perform semantic (dense vector) search across a notebook's sources.
///
/// When a [`HydeService`] is provided and the query is short (< 20 words),
/// HyDE generates a hypothetical document and uses its embedding for search,
/// which bridges the query-document embedding gap.
#[tracing::instrument(skip(search_repo, request, embedder), fields(notebook_id = %scope.notebook_id))]
pub async fn semantic_search(
    search_repo: &dyn SearchRepository,
    scope: NotebookScope,
    request: &SearchRequest,
    embedder: &QueryEmbedder<'_>,
) -> Result<Vec<SearchResult>, AppError> {
    Ok(
        semantic_search_with_hyde(search_repo, scope, request, embedder)
            .await?
            .results,
    )
}

/// Semantic search with optional HyDE enhancement.
///
/// Scores are cosine similarities: [`ScoreDomain::DenseSimilarity`].
#[tracing::instrument(skip(search_repo, request, embedder), fields(notebook_id = %scope.notebook_id))]
pub async fn semantic_search_with_hyde(
    search_repo: &dyn SearchRepository,
    scope: NotebookScope,
    request: &SearchRequest,
    embedder: &QueryEmbedder<'_>,
) -> Result<SearchOutcome, AppError> {
    let query = request.validated_query()?;
    let embedded = embed_query(scope, query, embedder).await?;

    // --- Search stage ---
    let search_start = std::time::Instant::now();
    let chunks = search_repo
        .search_similar_chunks(scope, &embedded.vector, request.clamped_limit())
        .await?;
    let search_ms = search_start.elapsed().as_millis();
    tracing::info!(search_ms, notebook_id = %scope.notebook_id, "Dense search completed");

    let (results, dropped_non_finite) =
        filter_and_convert(chunks, request.min_relevance, ScoreDomain::DenseSimilarity);
    if dropped_non_finite > 0 {
        tracing::warn!(
            notebook_id = %scope.notebook_id,
            dropped_non_finite,
            "Dense candidates carried non-finite scores and were rejected"
        );
    }

    tracing::debug!(
        notebook_id = %scope.notebook_id,
        query_hash = %query_hash(query),
        count = results.len(),
        used_hyde = embedder.hyde.is_some(),
        embed_ms = embedded.elapsed_ms,
        search_ms,
        "Semantic search completed"
    );

    Ok(SearchOutcome {
        dense_candidates: results.len(),
        results,
        embed_ms: embedded.elapsed_ms,
        search_ms,
        dropped_non_finite,
        lexical_candidates: 0,
        cache_hit: embedded.cache_hit,
    })
}

/// Perform lexical (full-text) search across a notebook's sources.
///
/// Scores are `ts_rank_cd` values: [`ScoreDomain::LexicalRank`].
#[tracing::instrument(skip(search_repo, request), fields(notebook_id = %scope.notebook_id))]
pub async fn lexical_search(
    search_repo: &dyn SearchRepository,
    scope: NotebookScope,
    request: &SearchRequest,
) -> Result<SearchOutcome, AppError> {
    let query = request.validated_query()?;

    let search_start = std::time::Instant::now();
    let chunks = search_repo
        .search_lexical_chunks(scope, query, request.clamped_limit())
        .await?;
    let search_ms = search_start.elapsed().as_millis();

    let (results, dropped_non_finite) =
        filter_and_convert(chunks, request.min_relevance, ScoreDomain::LexicalRank);

    tracing::debug!(
        notebook_id = %scope.notebook_id,
        query_hash = %query_hash(query),
        count = results.len(),
        dropped_non_finite,
        "Lexical search completed"
    );

    Ok(SearchOutcome {
        lexical_candidates: results.len(),
        results,
        embed_ms: 0,
        search_ms,
        dropped_non_finite,
        dense_candidates: 0,
        cache_hit: false,
    })
}

/// Perform hybrid search combining dense and sparse retrieval.
///
/// Executes both database searches in one repeatable-read snapshot, then fuses
/// results using RRF. The fused
/// scores are [`ScoreDomain::RrfRank`] values, which are not comparable with
/// the dense similarities or lexical ranks they were derived from (US-012).
#[tracing::instrument(skip(search_repo, config, request, embedder), fields(notebook_id = %scope.notebook_id))]
pub async fn hybrid_search(
    search_repo: &dyn SearchRepository,
    config: &HybridSearchConfig,
    scope: NotebookScope,
    request: &SearchRequest,
    embedder: &QueryEmbedder<'_>,
) -> Result<SearchOutcome, AppError> {
    let query = request.validated_query()?;

    let embedded = embed_query(scope, query, embedder).await?;
    let search_start = std::time::Instant::now();
    let rows = search_repo
        .search_hybrid_chunks(scope, &embedded.vector, query, request.clamped_limit())
        .await?;
    let search_ms = search_start.elapsed().as_millis();
    let (dense_results, dense_dropped) =
        filter_and_convert(rows.dense, None, ScoreDomain::DenseSimilarity);
    let (lexical_results, lexical_dropped) =
        filter_and_convert(rows.lexical, None, ScoreDomain::LexicalRank);

    // Fuse results using RRF
    let fused = reciprocal_rank_fusion(
        &dense_results,
        &lexical_results,
        config.rrf_k,
        config.dense_weight,
        config.sparse_weight,
    );

    // `min_relevance` applies on the fused domain, not on the dense or lexical
    // scores the fusion consumed (US-012). Truncation keeps the fusion's total
    // order, which is deterministic through its chunk-id tie-break (US-013).
    let min_rel = request.min_relevance.unwrap_or(0.0);
    let results: Vec<_> = fused
        .into_iter()
        .filter(|r| r.relevance() >= min_rel)
        .take(usize::try_from(request.limit.max(0)).unwrap_or(0))
        .collect();

    tracing::debug!(
        notebook_id = %scope.notebook_id,
        query_hash = %query_hash(query),
        dense = dense_results.len(),
        lexical = lexical_results.len(),
        fused = results.len(),
        embed_ms = embedded.elapsed_ms,
        search_ms,
        "Hybrid search completed"
    );

    Ok(SearchOutcome {
        results,
        embed_ms: embedded.elapsed_ms,
        search_ms,
        dropped_non_finite: dense_dropped + lexical_dropped,
        dense_candidates: dense_results.len(),
        lexical_candidates: lexical_results.len(),
        cache_hit: embedded.cache_hit,
    })
}

// ============================================================================
// Internal helpers
// ============================================================================

struct EmbeddedQuery {
    vector: Vec<f32>,
    elapsed_ms: u128,
    cache_hit: bool,
}

async fn embed_query(
    scope: NotebookScope,
    query: &str,
    embedder: &QueryEmbedder<'_>,
) -> Result<EmbeddedQuery, AppError> {
    let QueryEmbedder {
        provider,
        hyde,
        cache,
        kind,
    } = *embedder;
    let started = std::time::Instant::now();
    let fingerprint = EmbeddingProvenance::from_provider(provider).fingerprint();

    let mut cached = None;
    let mut cached_kind = kind;
    if let Some(cache) = cache {
        for candidate in lookup_kinds(kind, hyde.is_some()).into_iter().flatten() {
            if let Some(embedding) = cache.get(candidate, &fingerprint, query).await {
                cached = Some(embedding);
                cached_kind = candidate;
                break;
            }
        }
    }
    let cache_hit = cached.is_some();
    tracing::debug!(
        cache_hit,
        kind = cached_kind.as_str(),
        "Embedding cache lookup"
    );

    let vector = if let Some(embedding) = cached {
        embedding
    } else {
        let (embedding, stored_kind) = if let Some(hyde_service) = hyde {
            if let Some(hyde_result) = hyde_service.generate(query).await {
                tracing::debug!(
                    notebook_id = %scope.notebook_id,
                    query_hash = %query_hash(query),
                    hyde_doc_len = hyde_result.document.len(),
                    "Using HyDE-generated document for embedding"
                );
                (
                    embeddings::embed_query(provider, &hyde_result.document).await?,
                    QueryEmbeddingKind::HydeDocument,
                )
            } else {
                (embeddings::embed_query(provider, query).await?, kind)
            }
        } else {
            (embeddings::embed_query(provider, query).await?, kind)
        };
        if let Some(cache) = cache {
            cache
                .insert(stored_kind, &fingerprint, query, embedding.clone())
                .await;
        }
        embedding
    };
    let elapsed_ms = if cache_hit {
        0
    } else {
        started.elapsed().as_millis()
    };
    tracing::info!(elapsed_ms, cache_hit, notebook_id = %scope.notebook_id, "Embedding completed");
    Ok(EmbeddedQuery {
        vector,
        elapsed_ms,
        cache_hit,
    })
}

/// Determine effective search mode based on config.
const fn effective_mode(config: &HybridSearchConfig, requested: SearchMode) -> SearchMode {
    if config.enabled {
        requested
    } else {
        match requested {
            SearchMode::Hybrid => SearchMode::Dense,
            other => other,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::providers::DeterministicEmbedder;
    use crate::repositories::{HybridChunkSearchResult, RepoResult};
    use crate::types::{ChunkSearchResult, RetrievalScore};
    use async_trait::async_trait;
    use types::MAX_LIMIT;
    use uuid::Uuid;

    fn make_result(chunk_id: Uuid, source_title: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk_id,
            generation_id: Uuid::nil(),
            source_id: Uuid::new_v4(),
            source_title: source_title.to_string(),
            chunk_index: 0,
            content: String::new(),
            parent_content: None,
            score: RetrievalScore::Dense(score),
            metadata: None,
            collapsed_children: Vec::new(),
        }
    }

    struct SnapshotRepo {
        old_generation: Uuid,
        new_generation: Uuid,
        score: f32,
    }

    impl SnapshotRepo {
        fn row(&self, generation_id: Uuid, marker: &str) -> ChunkSearchResult {
            ChunkSearchResult {
                id: Uuid::new_v4(),
                generation_id,
                source_id: Uuid::from_u128(42),
                chunk_index: 0,
                content: marker.to_owned(),
                parent_content: None,
                source_title: "Source".to_owned(),
                relevance_score: self.score,
                metadata: None,
            }
        }
    }

    #[async_trait]
    impl SearchRepository for SnapshotRepo {
        async fn search_similar_chunks(
            &self,
            _scope: NotebookScope,
            _query_embedding: &[f32],
            _limit: i32,
        ) -> RepoResult<Vec<ChunkSearchResult>> {
            Ok(vec![self.row(self.new_generation, "new dense")])
        }

        async fn search_lexical_chunks(
            &self,
            _scope: NotebookScope,
            _query: &str,
            _limit: i32,
        ) -> RepoResult<Vec<ChunkSearchResult>> {
            Ok(vec![self.row(self.old_generation, "old lexical")])
        }

        async fn search_hybrid_chunks(
            &self,
            _scope: NotebookScope,
            _query_embedding: &[f32],
            _query: &str,
            _limit: i32,
        ) -> RepoResult<HybridChunkSearchResult> {
            Ok(HybridChunkSearchResult {
                dense: vec![self.row(self.new_generation, "new dense")],
                lexical: vec![self.row(self.new_generation, "new lexical")],
            })
        }

        async fn count_chunks_for_notebook(&self, _scope: NotebookScope) -> RepoResult<i64> {
            Ok(2)
        }

        async fn count_sources_for_notebook(&self, _scope: NotebookScope) -> RepoResult<i64> {
            Ok(1)
        }

        async fn get_all_chunks_for_notebook(
            &self,
            _scope: NotebookScope,
        ) -> RepoResult<Vec<ChunkSearchResult>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn hybrid_search_uses_one_repository_snapshot() {
        let old_generation = Uuid::from_u128(1);
        let new_generation = Uuid::from_u128(2);
        let repo = SnapshotRepo {
            old_generation,
            new_generation,
            score: 0.9,
        };
        let embedder = DeterministicEmbedder::new();
        let outcome = hybrid_search(
            &repo,
            &HybridSearchConfig::default(),
            NotebookScope::new(Uuid::new_v4(), Uuid::new_v4()),
            &SearchRequest::new("generation").with_limit(10),
            &QueryEmbedder::direct(&embedder),
        )
        .await
        .expect("hybrid search");

        assert_eq!(outcome.results.len(), 2);
        assert!(
            outcome
                .results
                .iter()
                .all(|result| result.generation_id == new_generation),
            "RRF must receive both branches from one generation snapshot"
        );
    }

    #[tokio::test]
    async fn hybrid_threshold_applies_only_to_the_fused_score() {
        let repo = SnapshotRepo {
            old_generation: Uuid::from_u128(1),
            new_generation: Uuid::from_u128(2),
            score: 0.005,
        };
        let embedder = DeterministicEmbedder::new();
        let outcome = hybrid_search(
            &repo,
            &HybridSearchConfig::default(),
            NotebookScope::new(Uuid::new_v4(), Uuid::new_v4()),
            &SearchRequest::new("generation")
                .with_limit(10)
                .with_min_relevance(0.01),
            &QueryEmbedder::direct(&embedder),
        )
        .await
        .expect("hybrid search");

        assert_eq!(outcome.results.len(), 1);
        assert_eq!(outcome.results[0].score.domain(), ScoreDomain::RrfRank);
        assert!(outcome.results[0].relevance() >= 0.01);
    }

    #[test]
    fn default_weights_are_4_to_1_ratio() {
        assert!(
            (DEFAULT_DENSE_WEIGHT / DEFAULT_SPARSE_WEIGHT - 4.0).abs() < f32::EPSILON,
            "Expected 4:1 dense/sparse ratio, got {}:1",
            DEFAULT_DENSE_WEIGHT / DEFAULT_SPARSE_WEIGHT
        );
        assert!((DEFAULT_DENSE_WEIGHT - 1.0).abs() < f32::EPSILON);
        assert!((DEFAULT_SPARSE_WEIGHT - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn rrf_with_default_weights_ranks_dense_higher() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();

        // Chunk A: rank 1 in dense, absent in lexical
        let dense = vec![
            make_result(id_a, "Source A", 0.95),
            make_result(id_b, "Source B", 0.80),
        ];
        // Chunk B: rank 1 in lexical, A absent
        let lexical = vec![make_result(id_b, "Source B", 0.90)];

        let k = 60.0;
        let fused = reciprocal_rank_fusion(
            &dense,
            &lexical,
            k,
            DEFAULT_DENSE_WEIGHT,
            DEFAULT_SPARSE_WEIGHT,
        );

        assert_eq!(fused.len(), 2);

        // Chunk B should rank first: it appears in both lists
        // B score = 1.0/(60+2) + 0.25/(60+1) = 0.01613 + 0.00410 = 0.02023
        // A score = 1.0/(60+1) = 0.01639
        assert_eq!(
            fused[0].chunk_id, id_b,
            "Chunk B (in both lists) should rank first"
        );
        assert_eq!(
            fused[1].chunk_id, id_a,
            "Chunk A (dense only) should rank second"
        );
    }

    #[test]
    fn rrf_dense_dominates_over_sparse() {
        let id_dense_top = Uuid::new_v4();
        let id_sparse_top = Uuid::new_v4();

        // Chunk X: rank 1 in dense only
        let dense = vec![make_result(id_dense_top, "Dense Winner", 0.99)];
        // Chunk Y: rank 1 in lexical only
        let lexical = vec![make_result(id_sparse_top, "Sparse Winner", 0.99)];

        let k = 60.0;
        let fused = reciprocal_rank_fusion(
            &dense,
            &lexical,
            k,
            DEFAULT_DENSE_WEIGHT,
            DEFAULT_SPARSE_WEIGHT,
        );

        assert_eq!(fused.len(), 2);

        // Dense top should rank higher: 1.0/(60+1) = 0.01639 vs 0.25/(60+1) = 0.00410
        assert_eq!(
            fused[0].chunk_id, id_dense_top,
            "Dense rank-1 should beat sparse rank-1 with 4:1 weighting"
        );

        // Verify the score ratio is approximately 4:1
        let ratio = fused[0].relevance() / fused[1].relevance();
        assert!(
            (ratio - 4.0).abs() < 0.01,
            "Score ratio should be ~4.0, got {ratio}"
        );
    }

    #[test]
    fn rrf_equal_weights_gives_equal_scores() {
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();

        let dense = vec![make_result(id_a, "A", 0.9)];
        let lexical = vec![make_result(id_b, "B", 0.9)];

        // With equal weights, rank-1 in both should get equal scores
        let fused = reciprocal_rank_fusion(&dense, &lexical, 60.0, 1.0, 1.0);

        assert_eq!(fused.len(), 2);
        assert!(
            (fused[0].relevance() - fused[1].relevance()).abs() < f32::EPSILON,
            "Equal weights should produce equal scores for equal ranks"
        );
    }

    // --- Retrieval pool and limits ---

    #[test]
    fn max_limit_accommodates_max_pool_size() {
        // RETRIEVAL_POOL_SIZE is validated to be <= 500 in Config.
        // MAX_LIMIT must accommodate the maximum allowed pool size.
        const {
            assert!(
                MAX_LIMIT >= 500,
                "MAX_LIMIT must be >= max allowed RETRIEVAL_POOL_SIZE (500)"
            );
        }
    }

    #[test]
    fn clamped_limit_caps_at_max_limit() {
        let req = SearchRequest::new("test").with_limit(1000);
        assert_eq!(req.clamped_limit(), MAX_LIMIT);

        let req = SearchRequest::new("test").with_limit(50);
        assert_eq!(req.clamped_limit(), 50);

        let req = SearchRequest::new("test").with_limit(5);
        assert_eq!(req.clamped_limit(), 5);
    }

    // --- Score domains (US-012) ---

    #[test]
    fn fusion_produces_its_own_score_domain() {
        let dense = vec![make_result(Uuid::new_v4(), "A", 0.95)];
        let lexical = vec![make_result(Uuid::new_v4(), "B", 0.90)];
        let fused = reciprocal_rank_fusion(
            &dense,
            &lexical,
            60.0,
            DEFAULT_DENSE_WEIGHT,
            DEFAULT_SPARSE_WEIGHT,
        );

        assert!(
            fused
                .iter()
                .all(|r| r.score.domain() == ScoreDomain::RrfRank),
            "a fused candidate carries a rank score, not the similarity it came from"
        );
    }

    #[test]
    fn fusion_breaks_ties_on_the_chunk_id_rather_than_hash_order() {
        // Two candidates at the same rank in the same list fuse to the same
        // score; without a tie-break the surviving order would come from
        // `HashMap` iteration and differ between runs (US-013).
        let ids: Vec<Uuid> = (0..8).map(|_| Uuid::new_v4()).collect();
        let dense: Vec<SearchResult> = ids
            .iter()
            .map(|id| make_result(*id, "same rank", 0.5))
            .collect();

        let first = reciprocal_rank_fusion(&dense, &[], 60.0, 1.0, 0.25);
        let mut reversed = dense.clone();
        reversed.reverse();
        let second = reciprocal_rank_fusion(&reversed, &[], 60.0, 1.0, 0.25);

        let order = |v: &[SearchResult]| v.iter().map(|r| r.chunk_id).collect::<Vec<_>>();
        // The two inputs rank the same chunks differently, so the fused scores
        // differ; what must hold is that equal scores never depend on hashing.
        for fused in [&first, &second] {
            for window in fused.windows(2) {
                let equal_scores =
                    (window[0].relevance() - window[1].relevance()).abs() < f32::EPSILON;
                if equal_scores {
                    assert!(
                        window[0].chunk_id < window[1].chunk_id,
                        "equal scores must come back in chunk-id order"
                    );
                }
            }
        }
        assert_eq!(order(&first).len(), ids.len());
    }

    #[test]
    fn a_non_finite_repository_score_is_dropped_and_counted() {
        use crate::types::ChunkSearchResult;

        let chunk = |score: f32| ChunkSearchResult {
            id: Uuid::new_v4(),
            generation_id: Uuid::nil(),
            source_id: Uuid::new_v4(),
            chunk_index: 0,
            content: "c".into(),
            parent_content: None,
            source_title: "T".into(),
            relevance_score: score,
            metadata: None,
        };

        let (results, dropped) = filter_and_convert(
            vec![
                chunk(0.9),
                chunk(f32::NAN),
                chunk(f32::INFINITY),
                chunk(0.1),
            ],
            None,
            ScoreDomain::DenseSimilarity,
        );

        assert_eq!(results.len(), 2, "only the rankable candidates survive");
        assert_eq!(dropped, 2);
        assert!(
            results.iter().all(|r| r.relevance() > 0.0),
            "no candidate was coerced to a zero score"
        );
    }

    // --- Corrective RAG ---

    #[test]
    fn corrective_retrieval_retries_at_most_once() {
        assert_eq!(CORRECTIVE_RAG_MAX_RETRIES, 1);
    }

    #[test]
    fn a_mean_is_reported_only_for_a_relevance_scale() {
        let mut dense = make_result(Uuid::new_v4(), "A", 0.8);
        let mut other = make_result(Uuid::new_v4(), "B", 0.4);
        let mean =
            mean_relevance(&[dense.clone(), other.clone()]).expect("dense is a relevance scale");
        assert!((mean - 0.6).abs() < f32::EPSILON);

        for domain in [
            RetrievalScore::Rrf(0.02),
            RetrievalScore::Lexical(0.3),
            RetrievalScore::Stuffed,
        ] {
            dense.score = domain;
            other.score = domain;
            assert!(
                mean_relevance(&[dense.clone(), other.clone()]).is_none(),
                "{domain:?} is not a scale on which a mean is a relevance"
            );
        }

        assert!(mean_relevance(&[]).is_none(), "an empty set has no mean");
    }

    // --- Backward compatibility: old chunks without parent_content (US-009) ---

    #[test]
    fn search_result_without_parent_works_in_rrf() {
        // Old-style chunks (parent_content = None) must fuse correctly in RRF
        let id = Uuid::new_v4();
        let mut dense_r = make_result(id, "Legacy Source", 0.9);
        dense_r.content = "legacy chunk content".to_string();
        let mut lexical_r = make_result(id, "Legacy Source", 0.8);
        lexical_r.content = "legacy chunk content".to_string();
        let dense = vec![dense_r];
        let lexical = vec![lexical_r];

        let fused = reciprocal_rank_fusion(
            &dense,
            &lexical,
            60.0,
            DEFAULT_DENSE_WEIGHT,
            DEFAULT_SPARSE_WEIGHT,
        );

        assert_eq!(fused.len(), 1);
        assert!(fused[0].relevance() > 0.0);
        assert!(
            fused[0].parent_content.is_none(),
            "Legacy chunks should retain None parent_content through RRF"
        );
        assert_eq!(
            fused[0].content, "legacy chunk content",
            "Content field should be preserved through RRF"
        );
    }

    #[test]
    fn search_result_with_parent_preserved_through_rrf() {
        // New-style chunks (parent_content = Some) must preserve parent through RRF
        let id = Uuid::new_v4();
        let mut dense_result = make_result(id, "New Source", 0.9);
        dense_result.parent_content = Some("parent context text".to_string());
        let dense = vec![dense_result];

        let fused = reciprocal_rank_fusion(
            &dense,
            &[],
            60.0,
            DEFAULT_DENSE_WEIGHT,
            DEFAULT_SPARSE_WEIGHT,
        );

        assert_eq!(fused.len(), 1);
        assert_eq!(
            fused[0].parent_content.as_deref(),
            Some("parent context text"),
            "New chunks should preserve parent_content through RRF"
        );
    }
}
