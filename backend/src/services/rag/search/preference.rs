//! User preferences as a secondary ordering key (US-014).
//!
//! A preference decides between candidates the ranker considers equivalent. It
//! never edits a score: multiplying a provider's relevance, which is what this
//! used to do, makes a preferred weak match indistinguishable from an
//! unpreferred strong one and destroys the very number the reranker was called
//! to produce.
//!
//! The type and its weights live together here rather than in the pipeline
//! module, so the transform that consumes them does not have to import its own
//! caller.

use std::collections::HashSet;

use uuid::Uuid;

use super::types::SearchResult;

/// Weight of a source the user gave positive feedback on, in the secondary
/// ordering key.
const SOURCE_TIER: u8 = 2;

/// Weight of an overlap with a preference memory topic, in the secondary
/// ordering key. Lower than the source tier: an explicit thumbs-up on a source
/// is a stronger signal than a keyword match on its text.
const TOPIC_TIER: u8 = 1;

/// Minimum word length for preference topic keyword extraction.
const MIN_KEYWORD_LEN: usize = 5;

/// Pre-computed preference boost parameters.
///
/// Built once per request in the chat handler and threaded into the retrieval
/// pipeline.
pub struct PreferenceBoost {
    /// Source IDs from RAG logs with positive user feedback.
    pub preferred_source_ids: HashSet<Uuid>,
    /// Lowercased keywords extracted from preference-type memories.
    pub preference_keywords: Vec<String>,
}

impl PreferenceBoost {
    /// Returns `true` if there are no boost signals at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.preferred_source_ids.is_empty() && self.preference_keywords.is_empty()
    }
}

/// How strongly a result matches the user's recorded preferences.
///
/// A small integer rather than a multiplier, for the reason in the module
/// documentation.
pub(super) fn preference_tier(result: &SearchResult, boost: &PreferenceBoost) -> u8 {
    let mut tier = 0;
    if boost.preferred_source_ids.contains(&result.source_id) {
        tier += SOURCE_TIER;
    }
    if !boost.preference_keywords.is_empty()
        && has_topic_overlap(&result.content, &boost.preference_keywords)
    {
        tier += TOPIC_TIER;
    }
    tier
}

/// Order results by score, then by preference, then by identity.
///
/// The score stays primary: a preference can decide between candidates the
/// ranker considers equivalent, and can never move a weaker candidate above a
/// stronger one. That is the whole of "explicit secondary ordering key"
/// (US-014).
pub(super) fn apply_preference_ordering(results: &mut [SearchResult], boost: &PreferenceBoost) {
    if boost.is_empty() {
        return;
    }

    results.sort_by(|a, b| {
        a.score
            .cmp_desc(b.score)
            .then_with(|| preference_tier(b, boost).cmp(&preference_tier(a, boost)))
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });

    tracing::debug!(
        preferred_sources = boost.preferred_source_ids.len(),
        preference_keywords = boost.preference_keywords.len(),
        promoted = results
            .iter()
            .filter(|r| preference_tier(r, boost) > 0)
            .count(),
        "Applied preference ordering as a secondary key"
    );
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
#[must_use]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RetrievalScore;

    fn scored(score: f32, source_id: Uuid) -> SearchResult {
        SearchResult {
            chunk_id: Uuid::new_v4(),
            generation_id: Uuid::nil(),
            source_id,
            source_title: "Test".to_string(),
            chunk_index: 0,
            content: format!("child at score {score}"),
            parent_content: None,
            score: RetrievalScore::Rrf(score),
            metadata: None,
            collapsed_children: Vec::new(),
        }
    }

    #[test]
    fn preferences_order_between_equals_and_never_overtake_a_stronger_result() {
        let preferred = Uuid::new_v4();
        let other = Uuid::new_v4();

        // A stronger unpreferred result and a weaker preferred one.
        let mut results = vec![scored(0.9, other), scored(0.5, preferred)];
        let boost = PreferenceBoost {
            preferred_source_ids: [preferred].into_iter().collect(),
            preference_keywords: vec![],
        };
        apply_preference_ordering(&mut results, &boost);

        assert_eq!(
            results[0].source_id, other,
            "a preference must not promote a weaker candidate"
        );
        assert!(
            (results[1].relevance() - 0.5).abs() < f32::EPSILON,
            "and it must not rewrite the score either"
        );

        // Equal scores: now the preference decides.
        let mut tied = vec![scored(0.7, other), scored(0.7, preferred)];
        apply_preference_ordering(&mut tied, &boost);
        assert_eq!(tied[0].source_id, preferred);
    }

    #[test]
    fn preference_tiers_rank_an_explicit_thumbs_up_above_a_keyword_match() {
        let preferred = Uuid::new_v4();
        let boost = PreferenceBoost {
            preferred_source_ids: [preferred].into_iter().collect(),
            preference_keywords: vec!["machine".to_string()],
        };

        let source_match = scored(0.5, preferred);
        let mut topic_match = scored(0.5, Uuid::new_v4());
        topic_match.content = "machine learning architectures".to_string();
        let neither = scored(0.5, Uuid::new_v4());

        assert!(preference_tier(&source_match, &boost) > preference_tier(&topic_match, &boost));
        assert!(preference_tier(&topic_match, &boost) > preference_tier(&neither, &boost));
    }

    #[test]
    fn an_empty_boost_leaves_the_order_untouched() {
        let boost = PreferenceBoost {
            preferred_source_ids: HashSet::new(),
            preference_keywords: vec![],
        };
        let mut results = vec![scored(0.2, Uuid::new_v4()), scored(0.9, Uuid::new_v4())];
        let before: Vec<Uuid> = results.iter().map(|r| r.chunk_id).collect();
        apply_preference_ordering(&mut results, &boost);
        let after: Vec<Uuid> = results.iter().map(|r| r.chunk_id).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn extract_keywords_filters_short_words() {
        let contents = vec!["The user prefers detailed technical explanations".to_string()];
        let keywords = extract_preference_keywords(&contents);
        assert!(keywords.contains(&"prefers".to_string()));
        assert!(keywords.contains(&"detailed".to_string()));
        assert!(keywords.contains(&"technical".to_string()));
        assert!(keywords.contains(&"explanations".to_string()));
        assert!(!keywords.contains(&"the".to_string()));
        assert!(!keywords.contains(&"user".to_string()));
    }

    #[test]
    fn extract_keywords_deduplicates() {
        let contents = vec![
            "The user prefers technical details".to_string(),
            "The user likes technical writing".to_string(),
        ];
        let keywords = extract_preference_keywords(&contents);
        assert_eq!(keywords.iter().filter(|k| *k == "technical").count(), 1);
    }

    #[test]
    fn extract_keywords_strips_punctuation() {
        let contents = vec!["prefers, (detailed) explanations.".to_string()];
        let keywords = extract_preference_keywords(&contents);
        assert!(keywords.contains(&"prefers".to_string()));
        assert!(keywords.contains(&"detailed".to_string()));
        assert!(keywords.contains(&"explanations".to_string()));
    }

    #[test]
    fn extract_keywords_empty_input() {
        assert!(extract_preference_keywords(&[]).is_empty());
    }

    #[test]
    fn has_topic_overlap_case_insensitive() {
        assert!(has_topic_overlap(
            "Technical Deep Learning",
            &["technical".to_string()]
        ));
        assert!(has_topic_overlap(
            "MACHINE LEARNING",
            &["machine".to_string()]
        ));
        assert!(!has_topic_overlap(
            "Cooking recipes",
            &["technical".to_string()]
        ));
    }
}
