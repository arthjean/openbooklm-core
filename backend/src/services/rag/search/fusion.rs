//! Reciprocal Rank Fusion (RRF) for combining multi-modal search results.

use std::collections::HashMap;
use uuid::Uuid;

use crate::types::{ChunkSearchResult, RetrievalScore, ScoreDomain, SearchResult};

/// Reciprocal Rank Fusion to combine results from multiple retrieval methods.
///
/// Formula: `score(d) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)`
///
/// The output is a new score domain: an RRF value is a function of *ranks* and
/// is not comparable with the dense similarity or lexical rank it was computed
/// from (US-012). Ties break on the chunk identifier so that two runs over the
/// same corpus produce the same order. A `HashMap` iteration order is not a
/// tie-breaker (US-013).
#[must_use]
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

    // Collect, re-domain the scores, and impose a total order.
    let mut fused: Vec<_> = scores
        .into_values()
        .filter_map(|(score, mut r)| {
            r.score = RetrievalScore::new(ScoreDomain::RrfRank, score)?;
            Some(r)
        })
        .collect();

    sort_by_score_then_id(&mut fused);
    fused
}

/// Impose a total order on a result set: score descending, chunk id ascending.
///
/// The tie-break is what makes truncation deterministic (US-013): with equal
/// scores (routine for lexical ranks, universal for stuffed chunks) the
/// surviving members would otherwise depend on hash iteration order, and the
/// same query would return different evidence on two identical runs.
pub(super) fn sort_by_score_then_id(results: &mut [SearchResult]) {
    results.sort_by(|a, b| {
        a.score
            .cmp_desc(b.score)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });
}

/// Filter chunks by `min_relevance` and convert to [`SearchResult`].
///
/// `domain` is the scale the producing query measured on; `min_relevance` is
/// interpreted on that same scale, and a caller that mixes the two gets a
/// meaningless filter, which is why the domain is a parameter rather than an
/// assumption (US-012).
///
/// Returns the converted results and the number of candidates dropped for
/// carrying a non-finite score, which the caller records as
/// [`ReasonCode::NonFiniteScore`](crate::services::rag::eval::trace::ReasonCode::NonFiniteScore).
pub fn filter_and_convert(
    chunks: Vec<ChunkSearchResult>,
    min_relevance: Option<f32>,
    domain: ScoreDomain,
) -> (Vec<SearchResult>, usize) {
    let min = min_relevance.unwrap_or(0.0);
    let mut dropped_non_finite = 0;
    let mut results = Vec::with_capacity(chunks.len());

    for chunk in chunks {
        // A rejected conversion means the score could not be ranked; a value
        // below the threshold is a legitimate filter. Only the first is a
        // defect, so they are counted apart.
        let Some(result) = SearchResult::from_chunk(chunk, domain) else {
            dropped_non_finite += 1;
            continue;
        };
        if result.relevance() >= min {
            results.push(result);
        }
    }

    (results, dropped_non_finite)
}
