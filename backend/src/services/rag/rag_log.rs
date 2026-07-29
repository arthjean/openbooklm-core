//! RAG evaluation logging service.
//!
//! Asynchronously logs RAG interaction details for quality analysis:
//! query, retrieved chunks, scores, reformulation, HyDE document, feedback.
//!
//! Logs are created non-blocking (via `tokio::spawn`) so they don't
//! delay the SSE response to the user.
//!
//! All database operations are delegated to the `RagLogRepository` trait.
//! This module owns the domain types and thin service-layer orchestration.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::AppError;
use crate::repositories::RagLogRepository;

// ============================================================================
// Types
// ============================================================================

/// Data for a single RAG interaction log entry.
#[derive(Debug, Clone, Serialize)]
pub struct RagLogEntry {
    pub notebook_id: Uuid,
    pub user_id: Uuid,
    pub query: String,
    pub reformulated_query: Option<String>,
    pub hyde_document: Option<String>,
    pub chunks_retrieved: Vec<ChunkLogEntry>,
    pub response_id: Option<Uuid>,
    pub retrieval_score_avg: Option<f32>,
    pub context_relevance: Option<f32>,
    pub model: Option<String>,
    pub provider: Option<String>,
}

/// Minimal chunk info for logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkLogEntry {
    pub chunk_id: Uuid,
    pub source_id: Uuid,
    pub chunk_index: i32,
    pub relevance_score: f32,
}

/// RAG quality metrics for a response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RagMetrics {
    /// Average relevance of retrieved chunks (from reranking scores).
    pub context_relevance: Option<f32>,
    /// Faithfulness of the answer to the source material (LLM-judged, async).
    pub answer_faithfulness: Option<f32>,
    /// Relevance of the answer to the original query (LLM-judged, async).
    pub answer_relevance: Option<f32>,
}

/// Scope for aggregated metrics queries — ensures the SQL column name
/// is always a compile-time constant, preventing injection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsScope {
    Notebook,
    User,
}

impl MetricsScope {
    pub const fn column_name(&self) -> &'static str {
        match self {
            Self::Notebook => "notebook_id",
            Self::User => "user_id",
        }
    }
}

/// User feedback on a RAG response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum UserFeedback {
    Positive,
    Negative,
}

impl UserFeedback {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
        }
    }
}

/// Aggregated metrics for display.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub struct AggregatedMetrics {
    pub total_interactions: i64,
    pub successful_retrievals: i64,
    pub avg_context_relevance: Option<f32>,
    pub avg_answer_faithfulness: Option<f32>,
    pub positive_feedback: i64,
    pub negative_feedback: i64,
}

impl AggregatedMetrics {
    /// Retrieval success rate as a percentage (0.0 to 100.0).
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn success_rate(&self) -> f32 {
        if self.total_interactions == 0 {
            return 0.0;
        }
        (self.successful_retrievals as f64 / self.total_interactions as f64 * 100.0) as f32
    }

    /// Positive feedback ratio as a percentage.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn positive_feedback_rate(&self) -> f32 {
        let total = self.positive_feedback + self.negative_feedback;
        if total == 0 {
            return 0.0;
        }
        (self.positive_feedback as f64 / total as f64 * 100.0) as f32
    }
}

// ============================================================================
// Constants
// ============================================================================

/// Default retention period for RAG logs in days.
pub const RAG_LOG_RETENTION_DAYS: i32 = 90;

// ============================================================================
// Public API — thin service layer
// ============================================================================

/// Create a RAG log entry synchronously, returning its ID.
///
/// Unlike [`log_rag_interaction`] (fire-and-forget), this awaits the insert
/// so the caller can include the log ID in the SSE response.
pub async fn create_rag_log(
    repo: &dyn RagLogRepository,
    entry: &RagLogEntry,
) -> Result<Uuid, AppError> {
    repo.create(entry).await
}

/// Update RAG metrics for an existing log entry.
pub async fn update_metrics(
    repo: &dyn RagLogRepository,
    log_id: Uuid,
    metrics: &RagMetrics,
) -> Result<(), AppError> {
    repo.update_metrics(log_id, metrics).await
}

/// Update user feedback for a RAG log entry.
///
/// Only the user who owns the log entry can update its feedback.
pub async fn update_feedback(
    repo: &dyn RagLogRepository,
    log_id: Uuid,
    user_id: Uuid,
    feedback: UserFeedback,
) -> Result<(), AppError> {
    repo.update_feedback(log_id, user_id, feedback.as_str())
        .await
}

/// Get aggregated RAG metrics for a notebook over the last N days.
pub async fn get_notebook_metrics(
    repo: &dyn RagLogRepository,
    notebook_id: Uuid,
    days: i32,
) -> Result<AggregatedMetrics, AppError> {
    repo.get_notebook_metrics(notebook_id, days).await
}

/// Get aggregated RAG metrics for a user across all notebooks.
pub async fn get_user_metrics(
    repo: &dyn RagLogRepository,
    user_id: Uuid,
    days: i32,
) -> Result<AggregatedMetrics, AppError> {
    repo.get_user_metrics(user_id, days).await
}

/// Batch lookup: find rag_log IDs for a set of chat message IDs (via response_id).
pub async fn get_rag_log_ids_for_messages(
    repo: &dyn RagLogRepository,
    message_ids: &[Uuid],
) -> Result<std::collections::HashMap<Uuid, (Uuid, Option<String>)>, AppError> {
    repo.get_rag_log_ids_for_messages(message_ids).await
}

/// Purge RAG logs older than the specified retention period.
///
/// Returns the number of deleted rows.
pub async fn purge_old_logs(
    repo: &dyn RagLogRepository,
    retention_days: i32,
) -> Result<u64, AppError> {
    repo.purge_old_logs(retention_days).await
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_feedback_as_str() {
        assert_eq!(UserFeedback::Positive.as_str(), "positive");
        assert_eq!(UserFeedback::Negative.as_str(), "negative");
    }

    #[test]
    fn aggregated_metrics_success_rate() {
        let metrics = AggregatedMetrics {
            total_interactions: 100,
            successful_retrievals: 95,
            avg_context_relevance: Some(0.75),
            avg_answer_faithfulness: Some(0.88),
            positive_feedback: 70,
            negative_feedback: 10,
        };

        assert!((metrics.success_rate() - 95.0).abs() < 0.1);
        assert!((metrics.positive_feedback_rate() - 87.5).abs() < 0.1);
    }

    #[test]
    fn aggregated_metrics_empty() {
        let metrics = AggregatedMetrics {
            total_interactions: 0,
            successful_retrievals: 0,
            avg_context_relevance: None,
            avg_answer_faithfulness: None,
            positive_feedback: 0,
            negative_feedback: 0,
        };

        assert_eq!(metrics.success_rate(), 0.0);
        assert_eq!(metrics.positive_feedback_rate(), 0.0);
    }

    #[test]
    fn retention_days_constant() {
        assert_eq!(RAG_LOG_RETENTION_DAYS, 90);
    }

    #[test]
    fn metrics_scope_column_names() {
        assert_eq!(MetricsScope::Notebook.column_name(), "notebook_id");
        assert_eq!(MetricsScope::User.column_name(), "user_id");
    }

    #[test]
    fn chunk_log_entry_serializes() {
        let entry = ChunkLogEntry {
            chunk_id: Uuid::new_v4(),
            source_id: Uuid::new_v4(),
            chunk_index: 3,
            relevance_score: 0.85,
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("chunk_id"));
        assert!(json.contains("0.85"));
    }
}
