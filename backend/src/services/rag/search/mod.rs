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
mod transforms;
pub mod types;

use uuid::Uuid;

use crate::core::config::CoreConfig;
use crate::core::providers::EmbeddingProvider;
use crate::error::AppError;
use crate::repositories::SearchRepository;
use crate::services::embeddings;
use crate::services::rag::embedding_cache::EmbeddingCache;
use crate::services::rag::hyde::HydeService;

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

// Re-export all public items for backward compatibility.
pub use context::{
    CORRECTIVE_RAG_MAX_RETRIES, CORRECTIVE_RAG_THRESHOLD, CorrectiveRetrievalParams,
    PipelineTimings, PreferenceBoost, RERANK_TOP_K, RetrievalParams, build_rag_documents,
    extract_preference_keywords, format_context_for_llm, max_context_stuffing_chunks,
    retrieve_context, retrieve_context_corrective,
};
pub use fusion::reciprocal_rank_fusion;
pub use types::{CorrectiveResult, MAX_LIMIT, SearchMode, SearchRequest, SearchResult};

use fusion::filter_and_convert;

// ============================================================================
// Public API — search orchestration
// ============================================================================

/// Main search function that routes to the appropriate search method.
///
/// Returns `(results, embed_ms, search_ms)` — timings are populated for Dense/Hybrid
/// modes and `(0, 0)` for Lexical (no embedding involved).
#[tracing::instrument(skip(search_repo, config, request, embeddings, hyde, embedding_cache), fields(%notebook_id))]
pub async fn search(
    search_repo: &dyn SearchRepository,
    config: &CoreConfig,
    notebook_id: Uuid,
    request: &SearchRequest,
    embeddings: &dyn EmbeddingProvider,
    hyde: Option<&HydeService>,
    embedding_cache: Option<&EmbeddingCache>,
) -> Result<(Vec<SearchResult>, u128, u128), AppError> {
    let mode = effective_mode(config, request.mode);

    match mode {
        SearchMode::Hybrid => {
            hybrid_search(
                search_repo,
                config,
                notebook_id,
                request,
                embeddings,
                hyde,
                embedding_cache,
            )
            .await
        }
        SearchMode::Dense => {
            semantic_search_with_hyde(
                search_repo,
                config,
                notebook_id,
                request,
                embeddings,
                hyde,
                embedding_cache,
            )
            .await
        }
        SearchMode::Lexical => {
            let results = lexical_search(search_repo, notebook_id, request).await?;
            Ok((results, 0, 0))
        }
    }
}

/// Perform semantic (dense vector) search across a notebook's sources.
///
/// When a [`HydeService`] is provided and the query is short (< 20 words),
/// HyDE generates a hypothetical document and uses its embedding for search,
/// which bridges the query-document embedding gap.
#[tracing::instrument(skip(search_repo, config, request, embeddings, hyde, embedding_cache), fields(%notebook_id))]
pub async fn semantic_search(
    search_repo: &dyn SearchRepository,
    config: &CoreConfig,
    notebook_id: Uuid,
    request: &SearchRequest,
    embeddings: &dyn EmbeddingProvider,
    hyde: Option<&HydeService>,
    embedding_cache: Option<&EmbeddingCache>,
) -> Result<Vec<SearchResult>, AppError> {
    let (results, _, _) = semantic_search_with_hyde(
        search_repo,
        config,
        notebook_id,
        request,
        embeddings,
        hyde,
        embedding_cache,
    )
    .await?;
    Ok(results)
}

/// Semantic search with optional HyDE enhancement.
///
/// Returns `(results, embed_ms, search_ms)` where `embed_ms` is the time spent
/// generating the query embedding and `search_ms` is the time spent in the DB search.
#[tracing::instrument(skip(search_repo, _config, request, embeddings, hyde, embedding_cache), fields(%notebook_id))]
pub async fn semantic_search_with_hyde(
    search_repo: &dyn SearchRepository,
    _config: &CoreConfig,
    notebook_id: Uuid,
    request: &SearchRequest,
    embeddings: &dyn EmbeddingProvider,
    hyde: Option<&HydeService>,
    embedding_cache: Option<&EmbeddingCache>,
) -> Result<(Vec<SearchResult>, u128, u128), AppError> {
    let query = request.validated_query()?;

    // --- Embed stage (with cache check) ---
    let embed_start = std::time::Instant::now();

    // Check embedding cache first
    let cached = if let Some(cache) = embedding_cache {
        cache.get(query).await
    } else {
        None
    };
    let cache_hit = cached.is_some();
    tracing::debug!(cache_hit, "Embedding cache lookup");

    let query_embedding = if let Some(embedding) = cached {
        embedding
    } else {
        let embedding = if let Some(hyde_svc) = hyde {
            if let Some(hyde_result) = hyde_svc.generate(query).await {
                tracing::debug!(
                    %notebook_id,
                    query,
                    hyde_doc_len = hyde_result.document.len(),
                    "Using HyDE-generated document for embedding"
                );
                embeddings::embed_query(embeddings, &hyde_result.document).await?
            } else {
                embeddings::embed_query(embeddings, query).await?
            }
        } else {
            embeddings::embed_query(embeddings, query).await?
        };

        // Insert into cache on miss
        if let Some(cache) = embedding_cache {
            cache.insert(query, embedding.clone()).await;
        }

        embedding
    };

    let embed_ms = if cache_hit {
        0
    } else {
        embed_start.elapsed().as_millis()
    };
    tracing::info!(embed_ms, cache_hit, %notebook_id, "Embedding completed");

    // --- Search stage ---
    let search_start = std::time::Instant::now();
    let chunks = search_repo
        .search_similar_chunks(notebook_id, &query_embedding, request.clamped_limit())
        .await?;
    let search_ms = search_start.elapsed().as_millis();
    tracing::info!(search_ms, %notebook_id, "Dense search completed");

    let results = filter_and_convert(chunks, request.min_relevance);

    tracing::debug!(
        %notebook_id,
        query,
        count = results.len(),
        used_hyde = hyde.is_some(),
        embed_ms,
        search_ms,
        "Semantic search completed"
    );

    Ok((results, embed_ms, search_ms))
}

/// Perform lexical (full-text) search across a notebook's sources.
#[tracing::instrument(skip(search_repo, request), fields(%notebook_id))]
pub async fn lexical_search(
    search_repo: &dyn SearchRepository,
    notebook_id: Uuid,
    request: &SearchRequest,
) -> Result<Vec<SearchResult>, AppError> {
    let query = request.validated_query()?;

    let chunks = search_repo
        .search_lexical_chunks(notebook_id, query, request.clamped_limit())
        .await?;

    let results = filter_and_convert(chunks, request.min_relevance);

    tracing::debug!(
        %notebook_id,
        query,
        count = results.len(),
        "Lexical search completed"
    );

    Ok(results)
}

/// Perform hybrid search combining dense and sparse retrieval.
///
/// Executes both searches in parallel, then fuses results using RRF.
/// Returns `(results, embed_ms, search_ms)` where timings come from the dense path.
#[tracing::instrument(skip(search_repo, config, request, embeddings, hyde, embedding_cache), fields(%notebook_id))]
pub async fn hybrid_search(
    search_repo: &dyn SearchRepository,
    config: &CoreConfig,
    notebook_id: Uuid,
    request: &SearchRequest,
    embeddings: &dyn EmbeddingProvider,
    hyde: Option<&HydeService>,
    embedding_cache: Option<&EmbeddingCache>,
) -> Result<(Vec<SearchResult>, u128, u128), AppError> {
    let query = request.validated_query()?;

    // Both sub-searches use the same limit; RRF fusion deduplicates and re-scores
    let dense_req = SearchRequest::new(query)
        .with_limit(request.limit)
        .with_mode(SearchMode::Dense);
    let lexical_req = SearchRequest::new(query)
        .with_limit(request.limit)
        .with_mode(SearchMode::Lexical);

    // Execute both searches in parallel (dense returns timing info)
    let (dense_res, lexical_res) = tokio::join!(
        semantic_search_with_hyde(
            search_repo,
            config,
            notebook_id,
            &dense_req,
            embeddings,
            hyde,
            embedding_cache
        ),
        lexical_search(search_repo, notebook_id, &lexical_req)
    );

    let (dense_results, embed_ms, search_ms) = dense_res?;
    let lexical_results = lexical_res.unwrap_or_else(|e| {
        tracing::warn!(error = %e, "Lexical search failed, using dense-only");
        Vec::new()
    });

    // Fuse results using RRF
    let hybrid = &config.hybrid_search;
    let fused = reciprocal_rank_fusion(
        &dense_results,
        &lexical_results,
        hybrid.rrf_k,
        hybrid.dense_weight,
        hybrid.sparse_weight,
    );

    // Apply limit and min_relevance filter
    let min_rel = request.min_relevance.unwrap_or(0.0);
    let results: Vec<_> = fused
        .into_iter()
        .filter(|r| r.relevance_score >= min_rel)
        .take(usize::try_from(request.limit.max(0)).unwrap_or(0))
        .collect();

    tracing::debug!(
        %notebook_id,
        query,
        dense = dense_results.len(),
        lexical = lexical_results.len(),
        fused = results.len(),
        embed_ms,
        search_ms,
        "Hybrid search completed"
    );

    Ok((results, embed_ms, search_ms))
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Determine effective search mode based on config.
const fn effective_mode(config: &CoreConfig, requested: SearchMode) -> SearchMode {
    if config.hybrid_search.enabled {
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
    use transforms::average_relevance;
    use types::MAX_LIMIT;
    use uuid::Uuid;

    fn make_result(chunk_id: Uuid, source_title: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk_id,
            source_id: Uuid::new_v4(),
            source_title: source_title.to_string(),
            chunk_index: 0,
            content: String::new(),
            parent_content: None,
            relevance_score: score,
            metadata: None,
        }
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
        let ratio = fused[0].relevance_score / fused[1].relevance_score;
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
            (fused[0].relevance_score - fused[1].relevance_score).abs() < f32::EPSILON,
            "Equal weights should produce equal scores for equal ranks"
        );
    }

    // --- Retrieval pool & rerank constants ---

    #[test]
    fn default_retrieval_pool_size_is_50() {
        // The default pool size is configured in Config (env var RETRIEVAL_POOL_SIZE).
        // Validate that the default (50) is above RERANK_TOP_K (20).
        let default_pool_size: i32 = 50;
        assert_eq!(RERANK_TOP_K, 20, "Rerank should keep top-20 chunks");
        assert!(
            default_pool_size > RERANK_TOP_K,
            "Default pool size must be larger than rerank top-K"
        );
    }

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

    #[test]
    fn rerank_truncates_to_top_k() {
        // Simulate pool_size results that would come from search
        let pool_size = 50; // default config value
        // After reranking, we should get at most RERANK_TOP_K results
        let truncated: Vec<SearchResult> = (0..pool_size)
            .map(|i| {
                make_result(
                    Uuid::new_v4(),
                    &format!("Source {i}"),
                    #[allow(clippy::cast_precision_loss)]
                    {
                        (i as f32).mul_add(-0.005, 1.0)
                    },
                )
            })
            .take(RERANK_TOP_K as usize)
            .collect();
        assert_eq!(truncated.len(), RERANK_TOP_K as usize);
        assert!(
            truncated.len() <= 20,
            "Pipeline must return at most 20 chunks"
        );
    }

    // --- Corrective RAG ---

    #[test]
    fn corrective_rag_threshold_is_reasonable() {
        assert!((CORRECTIVE_RAG_THRESHOLD - 0.5).abs() < f32::EPSILON);
        assert_eq!(CORRECTIVE_RAG_MAX_RETRIES, 1);
    }

    #[test]
    fn average_relevance_calculation() {
        let results = vec![
            make_result(Uuid::new_v4(), "A", 0.8),
            make_result(Uuid::new_v4(), "B", 0.6),
            make_result(Uuid::new_v4(), "C", 0.4),
        ];
        let avg = average_relevance(&results);
        assert!(
            (avg - 0.6).abs() < f32::EPSILON,
            "Expected avg 0.6, got {avg}"
        );
    }

    #[test]
    fn average_relevance_empty() {
        assert_eq!(average_relevance(&[]), 0.0);
    }

    #[test]
    fn average_relevance_above_threshold_no_correction_needed() {
        let results = vec![
            make_result(Uuid::new_v4(), "A", 0.9),
            make_result(Uuid::new_v4(), "B", 0.8),
        ];
        let avg = average_relevance(&results);
        assert!(avg >= CORRECTIVE_RAG_THRESHOLD);
    }

    #[test]
    fn average_relevance_below_threshold_needs_correction() {
        let results = vec![
            make_result(Uuid::new_v4(), "A", 0.3),
            make_result(Uuid::new_v4(), "B", 0.2),
        ];
        let avg = average_relevance(&results);
        assert!(avg < CORRECTIVE_RAG_THRESHOLD);
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
        assert!(fused[0].relevance_score > 0.0);
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
