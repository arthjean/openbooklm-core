//! Deduplication: resolve whether an extracted memory should be inserted, updated, or skipped.

use crate::repositories::MemorySearchResult;

use super::types::{ExtractedMemory, MemoryAction};

const DEDUP_THRESHOLD: f32 = 0.90;
const NEAR_EXACT_THRESHOLD: f32 = 0.97;

/// Resolve whether an extracted memory should be inserted, updated, or skipped.
pub(crate) fn resolve_upsert_action(
    extracted: ExtractedMemory,
    embedding: &[f32],
    similar: &[MemorySearchResult],
) -> MemoryAction {
    // Find the closest existing memory
    let closest = similar.iter().max_by(|a, b| {
        a.similarity
            .partial_cmp(&b.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let Some(closest) = closest else {
        return MemoryAction::Insert {
            memory: extracted,
            embedding: embedding.to_vec(),
        };
    };

    if closest.similarity > NEAR_EXACT_THRESHOLD {
        // Near-exact duplicate — skip entirely
        MemoryAction::Skip
    } else if closest.similarity >= DEDUP_THRESHOLD {
        // Semantically equivalent — update with new content, reinforce salience
        let reinforced_salience = (closest
            .memory
            .salience
            .mul_add(0.7, extracted.salience * 0.3)
            * 1.05)
            .min(1.0);
        MemoryAction::Update {
            existing_id: closest.memory.id,
            new_content: extracted.content,
            new_salience: reinforced_salience,
            embedding: embedding.to_vec(),
        }
    } else {
        MemoryAction::Insert {
            memory: extracted,
            embedding: embedding.to_vec(),
        }
    }
}
