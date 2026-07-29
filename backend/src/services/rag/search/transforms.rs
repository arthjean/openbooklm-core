//! Post-retrieval transforms: preference boosting, deduplication, reranking,
//! and sandwich ordering.

use std::collections::HashSet;

use uuid::Uuid;

use crate::core::providers::Reranker;
use crate::error::AppError;

use super::context::{
    MIN_KEYWORD_LEN, PREFERENCE_BOOST_MULTIPLIER, PREFERENCE_TOPIC_BOOST, PreferenceBoost,
};
use super::types::SearchResult;

/// Apply preference boost to search results.
///
/// Two-layer boost:
/// 1. Source-level: chunks from sources with positive user feedback get a 1.15x multiplier
/// 2. Topic-level: chunks whose content overlaps with preference memory keywords get 1.05x
///
/// Re-sorts results by boosted score after application.
pub(super) fn apply_preference_boost(results: &mut [SearchResult], boost: &PreferenceBoost) {
    let mut any_boosted = false;

    for result in results.iter_mut() {
        let mut multiplier: f32 = 1.0;

        // Layer 1: source-level boost from positive rag_log feedback
        if boost.preferred_source_ids.contains(&result.source_id) {
            multiplier *= PREFERENCE_BOOST_MULTIPLIER;
        }

        // Layer 2: topic-level boost from preference memories
        if !boost.preference_keywords.is_empty()
            && has_topic_overlap(&result.content, &boost.preference_keywords)
        {
            multiplier *= PREFERENCE_TOPIC_BOOST;
        }

        if multiplier > 1.0 {
            result.relevance_score *= multiplier;
            any_boosted = true;
        }
    }

    // Re-sort after boosting to maintain descending score order
    if any_boosted {
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        tracing::debug!(
            preferred_sources = boost.preferred_source_ids.len(),
            preference_keywords = boost.preference_keywords.len(),
            "Applied preference boost to search results"
        );
    }
}

/// Check if chunk content contains any preference topic keywords.
///
/// Simple case-insensitive word overlap — a keyword must appear as a
/// substring in the lowercased chunk content.
pub(super) fn has_topic_overlap(content: &str, keywords: &[String]) -> bool {
    let lower = content.to_lowercase();
    keywords.iter().any(|kw| lower.contains(kw.as_str()))
}

/// Extract significant lowercased keywords from preference memory content.
///
/// Filters for words >= [`MIN_KEYWORD_LEN`] characters to exclude common
/// stop words while keeping meaningful terms like "technical", "machine",
/// "quantum", "detailed", etc.
pub fn extract_preference_keywords(preference_contents: &[String]) -> Vec<String> {
    let mut keywords: Vec<String> = preference_contents
        .iter()
        .flat_map(|content| {
            content.split_whitespace().filter_map(|word| {
                let clean: String = word
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase();
                if clean.len() >= MIN_KEYWORD_LEN {
                    Some(clean)
                } else {
                    None
                }
            })
        })
        .collect();

    keywords.sort();
    keywords.dedup();
    keywords
}

/// Deduplicate search results by parent content.
///
/// When multiple child chunks from the **same source** share identical
/// `parent_content`, only the highest-scoring child per parent is kept (input
/// must be sorted by score descending). Legacy chunks (`parent_content = None`)
/// are always retained.
///
/// If deduplication reduces the result count below `min_results`, lower-scoring
/// duplicate children are added back (sorted by score) until the minimum is met.
pub(super) fn deduplicate_parent_content(
    results: Vec<SearchResult>,
    min_results: usize,
) -> Vec<SearchResult> {
    let original_count = results.len();
    if original_count == 0 {
        return results;
    }

    // Key: (source_id, parent_content) — scoped per source to preserve
    // citation attribution when different sources contain identical text.
    let mut seen_parents: HashSet<(Uuid, String)> = HashSet::new();
    let mut deduped: Vec<SearchResult> = Vec::with_capacity(original_count);
    let mut duplicates: Vec<SearchResult> = Vec::new();

    // Input is sorted by score descending — first occurrence is the best per parent
    for result in results {
        match &result.parent_content {
            None => {
                // Legacy chunk (no parent-child) — always keep
                deduped.push(result);
            }
            Some(parent) => {
                let key = (result.source_id, parent.clone());
                if seen_parents.insert(key) {
                    deduped.push(result);
                } else {
                    duplicates.push(result);
                }
            }
        }
    }

    let duplicates_found = duplicates.len();

    // Relaxation: re-add lower-scoring children if dedup was too aggressive.
    if deduped.len() < min_results {
        let needed = min_results.saturating_sub(deduped.len());
        deduped.extend(duplicates.into_iter().take(needed));
    }

    let deduped_count = deduped.len();
    let chunks_removed = original_count - deduped_count;
    if duplicates_found > 0 {
        tracing::info!(
            original_count,
            deduped_count,
            chunks_removed,
            duplicates_found,
            "Parent content deduplication applied"
        );
    }

    deduped
}

/// Reorder results in "sandwich" pattern to mitigate the lost-in-the-middle problem.
///
/// LLMs attend best to the start and end of context, with a significant attention
/// drop in the middle (Liu et al. 2023). Places highest-relevance first,
/// second-highest last, remaining in the middle.
///
/// Input must be sorted by relevance_score descending. Requires at least 3 results.
pub(super) fn sandwich_order(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    debug_assert!(results.len() >= 3, "sandwich_order requires >= 3 results");

    // results[0] = best (stays at position 0)
    // results[1] = second-best (moves to last position)
    // results[2..] = rest (fill the middle)
    let second_best = results.remove(1);
    results.push(second_best);
    results
}

/// Calculate average relevance score for a set of results.
#[allow(clippy::cast_precision_loss)]
pub(super) fn average_relevance(results: &[SearchResult]) -> f32 {
    if results.is_empty() {
        return 0.0;
    }
    let sum: f32 = results.iter().map(|r| r.relevance_score).sum();
    sum / results.len() as f32
}

/// Rerank search results with the configured cross-encoder.
///
/// `None` is not an error: an installation without a reranker keeps the hybrid
/// search order, which is the same fallback a rerank failure already takes.
pub(super) async fn rerank_results(
    reranker: Option<&dyn Reranker>,
    query: &str,
    results: Vec<SearchResult>,
    top_k: usize,
) -> Result<Vec<SearchResult>, AppError> {
    if results.is_empty() {
        return Ok(results);
    }
    let Some(reranker) = reranker else {
        return Ok(results);
    };

    // Rerank using child content (the retrieval unit), not parent_content.
    let documents: Vec<_> = results.iter().map(|r| r.content.clone()).collect();
    let reranked = reranker.rerank(query, &documents, Some(top_k)).await?;

    let mut reranked_results: Vec<_> = reranked
        .into_iter()
        .filter_map(|r| {
            results.get(r.index).map(|orig| SearchResult {
                relevance_score: r.relevance_score,
                ..orig.clone()
            })
        })
        .collect();

    // Enforce descending score order — required by deduplicate_parent_content
    reranked_results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    tracing::debug!(
        original = results.len(),
        reranked = reranked_results.len(),
        top_score = ?reranked_results.first().map(|r| r.relevance_score),
        "Reranked results"
    );

    Ok(reranked_results)
}
