//! Reciprocal Rank Fusion (RRF) for combining multi-modal search results.

use std::collections::HashMap;
use uuid::Uuid;

use crate::types::{ChunkSearchResult, SearchResult};

/// Reciprocal Rank Fusion to combine results from multiple retrieval methods.
///
/// Formula: `score(d) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)`
pub fn reciprocal_rank_fusion(
    dense: &[SearchResult],
    lexical: &[SearchResult],
    k: f32,
    dense_weight: f32,
    sparse_weight: f32,
) -> Vec<SearchResult> {
    let mut scores: HashMap<Uuid, (f32, SearchResult)> =
        HashMap::with_capacity(dense.len() + lexical.len());

    // Helper to add RRF contribution
    let mut add_scores = |results: &[SearchResult], weight: f32| {
        for (rank, result) in results.iter().enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let rrf = weight / (k + (rank + 1) as f32);
            scores
                .entry(result.chunk_id)
                .and_modify(|(score, _)| *score += rrf)
                .or_insert((rrf, result.clone()));
        }
    };

    add_scores(dense, dense_weight);
    add_scores(lexical, sparse_weight);

    // Collect, update scores, and sort descending
    let mut fused: Vec<_> = scores
        .into_values()
        .map(|(score, mut r)| {
            r.relevance_score = score;
            r
        })
        .collect();

    fused.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused
}

/// Filter chunks by min_relevance and convert to [`SearchResult`].
pub fn filter_and_convert(
    chunks: Vec<ChunkSearchResult>,
    min_relevance: Option<f32>,
) -> Vec<SearchResult> {
    let min = min_relevance.unwrap_or(0.0);
    chunks
        .into_iter()
        .filter(|c| c.relevance_score >= min)
        .map(SearchResult::from)
        .collect()
}
