//! Post-retrieval transforms: diversification, reranking, final selection and
//! sandwich presentation.
//!
//! The order these run in is the contract EP-003 fixes (US-014):
//!
//! ```text
//! fuse → collapse parents → rerank the whole diversified pool →
//!   preference as a secondary key → select the final limit → sandwich
//! ```
//!
//! Every step before selection changes *ranking*; selection is the only step
//! that changes *membership*; sandwich ordering runs after it and is
//! presentation only. Reversing any pair of those reintroduces one of the
//! defects the epic removes: reranking a pool already cut to the final limit
//! throws away the candidates the reranker exists to promote, and collapsing
//! after selection lets six children of one passage occupy six of fifteen
//! context slots.
//!
//! The preference key itself lives in [`super::preference`]; it is an ordering
//! input, not a transform of the evidence.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::core::providers::Reranker;
use crate::error::AppError;
use crate::types::{RetrievalScore, ScoreDomain};

use super::fusion::sort_by_score_then_id;
use super::types::SearchResult;

/// Collapse children that resolve to the same canonical parent context.
///
/// Input must be sorted by score descending, so the first occurrence of a
/// parent is its strongest child; that child represents the context and
/// absorbs the identifiers of the ones it replaced. A uniformly scored pool
/// satisfies that precondition trivially, and its first occurrence is the one
/// the notebook lists first. Chunks with no parent are their own context and
/// never collapse. The same parent text under two different sources stays two
/// contexts, because citation attribution differs.
///
/// Nothing is added back. The previous implementation re-admitted lower-scoring
/// duplicate children whenever deduplication dropped the count below a floor,
/// which is exactly "reintroduce duplicate parent contexts to reach a minimum
/// result count", forbidden by US-013. A pool dominated by one passage now
/// yields fewer unique contexts, and the caller records the shortfall (US-014).
pub(super) fn collapse_parents(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let original_count = results.len();
    if original_count == 0 {
        return results;
    }

    debug_assert!(
        results
            .windows(2)
            .all(|w| w[0].score.cmp_desc(w[1].score) != std::cmp::Ordering::Greater),
        "collapse_parents requires results sorted by score descending"
    );

    // Key → index of the representative already in `collapsed`.
    let mut representatives: HashMap<(Uuid, String), usize> =
        HashMap::with_capacity(original_count);
    let mut collapsed: Vec<SearchResult> = Vec::with_capacity(original_count);

    for result in results {
        let Some((source_id, parent)) = result.parent_key() else {
            collapsed.push(result);
            continue;
        };
        let key = (source_id, parent.to_owned());
        match representatives.get(&key) {
            Some(&index) => collapsed[index].collapsed_children.push(result.chunk_id),
            None => {
                representatives.insert(key, collapsed.len());
                collapsed.push(result);
            }
        }
    }

    let removed = original_count - collapsed.len();
    if removed > 0 {
        tracing::info!(
            original_count,
            unique_contexts = collapsed.len(),
            collapsed = removed,
            "Collapsed overlapping children onto their parent contexts"
        );
    }

    collapsed
}

/// Truncate to the final limit.
///
/// The ordering is the caller's: whichever stage ranked the pool last also
/// decided which contexts a cut keeps, and re-sorting here would silently
/// override a deliberate presentation order (the uniform pool's reading order,
/// in particular). The precondition is that the input is ordered, and the
/// `debug_assert` is what makes a caller that breaks it fail a test rather than
/// ship a differently-truncated result set (US-013).
pub(super) fn select_final(mut results: Vec<SearchResult>, limit: usize) -> Vec<SearchResult> {
    debug_assert!(
        results
            .windows(2)
            .all(|w| w[0].score.cmp_desc(w[1].score) != std::cmp::Ordering::Greater),
        "select_final requires results ordered by score descending"
    );
    results.truncate(limit);
    results
}

/// Reorder results in "sandwich" pattern to mitigate the lost-in-the-middle problem.
///
/// LLMs attend best to the start and end of context, with a significant attention
/// drop in the middle (Liu et al. 2023). Places highest-relevance first,
/// second-highest last, remaining in the middle.
///
/// Presentation only: it runs after selection and cannot change which contexts
/// were selected (US-014). Requires at least 3 results.
pub(super) fn sandwich_order(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    debug_assert!(results.len() >= 3, "sandwich_order requires >= 3 results");

    // results[0] = best (stays at position 0)
    // results[1] = second-best (moves to last position)
    // results[2..] = rest (fill the middle)
    let second_best = results.remove(1);
    results.push(second_best);
    results
}

/// Mean score of a result set, when the scale makes a mean meaningful.
///
/// `None` for RRF, lexical and stuffed sets: averaging a rank artifact produces
/// a number that looks like a confidence and is not one, and publishing it is
/// how "0.016 relevance" reaches an operator's dashboard (US-012).
pub fn mean_relevance(results: &[SearchResult]) -> Option<f32> {
    let first = results.first()?;
    if !first.score.domain().is_relevance_scale() {
        return None;
    }
    debug_assert!(
        results
            .iter()
            .all(|r| r.score.domain() == first.score.domain()),
        "a result set must be homogeneous in its score domain"
    );
    #[allow(clippy::cast_precision_loss)]
    let mean = results.iter().map(SearchResult::relevance).sum::<f32>() / results.len() as f32;
    Some(mean)
}

/// Rerank a candidate pool with the configured cross-encoder.
///
/// The whole diversified pool goes in, not the final limit: a reranker exists
/// to promote a candidate the first-stage ranking placed low, and it cannot do
/// that for a candidate that was already cut (US-014).
///
/// `None` is not an error: an installation without a reranker keeps the fusion
/// order, which is the same fallback a rerank failure takes. The returned
/// results carry [`ScoreDomain::RerankerRelevance`] scores: a different scale
/// from the fusion scores that went in, which is why the conversion is explicit
/// (US-012).
pub(super) async fn rerank_results(
    reranker: Option<&dyn Reranker>,
    query: &str,
    results: &[SearchResult],
) -> Result<Option<Vec<SearchResult>>, AppError> {
    if results.is_empty() {
        return Ok(None);
    }
    let Some(reranker) = reranker else {
        return Ok(None);
    };

    // Rerank using child content (the retrieval unit), not parent_content.
    let documents: Vec<_> = results.iter().map(|r| r.content.clone()).collect();
    let pool_size = results.len();
    let reranked = reranker.rerank(query, &documents, Some(pool_size)).await?;

    let mut reranked_results: Vec<_> = reranked
        .into_iter()
        .filter_map(|r| {
            let original = results.get(r.index)?;
            let score = RetrievalScore::new(ScoreDomain::RerankerRelevance, r.relevance_score)?;
            Some(SearchResult {
                score,
                ..original.clone()
            })
        })
        .collect();

    sort_by_score_then_id(&mut reranked_results);

    tracing::debug!(
        pool = pool_size,
        reranked = reranked_results.len(),
        top_score = ?reranked_results.first().map(SearchResult::relevance),
        "Reranked the diversified candidate pool"
    );

    Ok(Some(reranked_results))
}

/// Distinct parent contexts in a result set.
#[must_use]
pub fn unique_parent_count(results: &[SearchResult]) -> usize {
    let mut seen: HashSet<(Uuid, &str)> = HashSet::new();
    let mut singletons = 0;
    for result in results {
        match result.parent_key() {
            Some(key) => {
                seen.insert(key);
            }
            None => singletons += 1,
        }
    }
    seen.len() + singletons
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(score: f32, parent: Option<&str>, source_id: Uuid) -> SearchResult {
        SearchResult {
            chunk_id: Uuid::new_v4(),
            generation_id: Uuid::nil(),
            source_id,
            source_title: "Test".to_string(),
            chunk_index: 0,
            content: format!("child at score {score}"),
            parent_content: parent.map(String::from),
            score: RetrievalScore::Rrf(score),
            metadata: None,
            collapsed_children: Vec::new(),
        }
    }

    #[test]
    fn collapsing_keeps_the_strongest_child_and_its_provenance() {
        let src = Uuid::new_v4();
        let results = vec![
            scored(0.9, Some("parent A"), src),
            scored(0.8, Some("parent B"), src),
            scored(0.7, Some("parent A"), src),
            scored(0.6, Some("parent B"), src),
        ];
        let absorbed: Vec<Uuid> = vec![results[2].chunk_id, results[3].chunk_id];

        let collapsed = collapse_parents(results);

        assert_eq!(collapsed.len(), 2);
        assert!((collapsed[0].relevance() - 0.9).abs() < f32::EPSILON);
        assert!((collapsed[1].relevance() - 0.8).abs() < f32::EPSILON);
        assert_eq!(collapsed[0].collapsed_children, vec![absorbed[0]]);
        assert_eq!(collapsed[1].collapsed_children, vec![absorbed[1]]);
    }

    #[test]
    fn collapsing_never_reintroduces_a_duplicate_to_reach_a_count() {
        // Four children of one parent. The old implementation padded back up to
        // a floor of five; the contract now says one context is one context.
        let src = Uuid::new_v4();
        let results = (0..4)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                scored(0.9 - i as f32 * 0.1, Some("same parent"), src)
            })
            .collect();
        assert_eq!(collapse_parents(results).len(), 1);
    }

    #[test]
    fn chunks_without_a_parent_are_their_own_contexts() {
        let results = vec![
            scored(0.9, None, Uuid::new_v4()),
            scored(0.8, None, Uuid::new_v4()),
            scored(0.7, None, Uuid::new_v4()),
        ];
        assert_eq!(collapse_parents(results).len(), 3);
    }

    #[test]
    fn identical_text_in_two_sources_stays_two_contexts() {
        let results = vec![
            scored(0.9, Some("identical parent text"), Uuid::new_v4()),
            scored(0.7, Some("identical parent text"), Uuid::new_v4()),
        ];
        assert_eq!(
            collapse_parents(results).len(),
            2,
            "citation attribution differs, so the contexts differ"
        );
    }

    #[test]
    fn a_uniform_pool_collapses_in_its_own_reading_order() {
        // Every score equal, which is what stuffing produces. The first child
        // of each parent represents it, so the order the notebook is written in
        // survives diversification.
        let src = Uuid::new_v4();
        let mut results: Vec<SearchResult> = (0..6)
            .map(|i| {
                let mut r = scored(1.0, Some(&format!("parent {}", i / 2)), src);
                r.score = RetrievalScore::Stuffed;
                r.chunk_index = i;
                r
            })
            .collect();
        results.iter_mut().for_each(|r| r.content = String::new());

        let collapsed = collapse_parents(results);
        assert_eq!(
            collapsed.iter().map(|r| r.chunk_index).collect::<Vec<_>>(),
            vec![0, 2, 4]
        );
    }

    #[test]
    fn selection_truncates_without_reordering() {
        let src = Uuid::new_v4();
        let results = vec![
            scored(0.9, None, src),
            scored(0.8, None, src),
            scored(0.7, None, src),
        ];
        let ids: Vec<Uuid> = results.iter().take(2).map(|r| r.chunk_id).collect();
        let selected = select_final(results, 2);
        assert_eq!(
            selected.iter().map(|r| r.chunk_id).collect::<Vec<_>>(),
            ids,
            "selection keeps the order it was given"
        );
    }

    #[test]
    fn selection_below_the_limit_returns_everything() {
        let src = Uuid::new_v4();
        let results = vec![scored(0.9, None, src), scored(0.8, None, src)];
        assert_eq!(select_final(results, 10).len(), 2);
    }
}
