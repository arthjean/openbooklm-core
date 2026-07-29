//! Temporal decay: reduce salience of stale memories and auto-delete below threshold.

use uuid::Uuid;

use crate::error::AppError;
use crate::repositories::MemoryRepository;

use super::types::DecayResult;

// ============================================================================
// Temporal decay (US-004)
// ============================================================================

/// Minimum days since last update before a memory is eligible for decay.
const DECAY_STALE_DAYS: i64 = 7;

/// Weekly decay factor applied per week of inactivity.
const DECAY_FACTOR_PER_WEEK: f32 = 0.95;

/// Salience threshold below which memories are auto-deleted.
const DECAY_DELETE_THRESHOLD: f32 = 0.1;

/// Apply temporal decay to stale memories in a notebook.
///
/// For each memory not updated in the last 7 days, multiplies salience by
/// `0.95^weeks_elapsed`. Memories with `memory_type = "conversation_summary"`
/// are exempt. After decay, memories with salience < 0.1 are auto-deleted.
pub async fn decay_memories(
    notebook_id: Uuid,
    repo: &dyn MemoryRepository,
) -> Result<DecayResult, AppError> {
    let all_memories = repo.list_for_notebook(notebook_id).await?;
    let now = chrono::Utc::now();
    let mut decayed_count = 0;

    for memory in &all_memories {
        // Conversation summaries are exempt from decay (managed by US-002)
        if memory.memory_type == "conversation_summary" {
            continue;
        }

        let days_since_update = (now - memory.updated_at.with_timezone(&chrono::Utc))
            .num_days()
            .max(0);

        if days_since_update < DECAY_STALE_DAYS {
            continue;
        }

        let weeks_elapsed = days_since_update / 7;
        let new_salience = (memory.salience
            * DECAY_FACTOR_PER_WEEK.powi(i32::try_from(weeks_elapsed).unwrap_or(i32::MAX)))
        .clamp(0.0, 1.0);

        repo.update_salience(memory.id, new_salience).await?;
        decayed_count += 1;
    }

    // Bulk-delete memories that fell below the threshold
    let deleted_count = repo
        .delete_below_salience(notebook_id, DECAY_DELETE_THRESHOLD)
        .await?;
    let deleted_count = usize::try_from(deleted_count).unwrap_or(usize::MAX);

    tracing::info!(
        %notebook_id,
        decayed_count,
        deleted_count,
        "Memory decay completed"
    );

    Ok(DecayResult {
        decayed_count,
        deleted_count,
    })
}
