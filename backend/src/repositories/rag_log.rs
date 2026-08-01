//! SeaORM implementation of RagLogRepository.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, ConnectionTrait, DatabaseConnection, DbBackend, EntityTrait,
    QueryFilter, QueryResult, Set, Statement,
};

use uuid::Uuid;

use crate::entities::rag_log;
use crate::error::AppError;
use crate::services::rag::rag_log::{AggregatedMetrics, MetricsScope, RagLogEntry, RagMetrics};

use super::traits::{RagLogRepository, RepoResult};

#[derive(Clone)]
pub struct SeaOrmRagLogRepository {
    db: DatabaseConnection,
}

impl SeaOrmRagLogRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    #[allow(clippy::unused_self)]
    fn stmt(&self, sql: &str, values: impl IntoIterator<Item = sea_orm::Value>) -> Statement {
        Statement::from_sql_and_values(DbBackend::Postgres, sql, values)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Shared aggregation query used by both notebook and user metrics.
///
/// The `scope` enum determines which column to filter on. The column name is derived
/// from `MetricsScope::column_name()` which returns a `&'static str`, making SQL
/// injection structurally impossible.
async fn query_aggregated_metrics(
    db: &DatabaseConnection,
    scope: MetricsScope,
    id: Uuid,
    days: i32,
) -> RepoResult<AggregatedMetrics> {
    let filter_column = scope.column_name();
    let sql = format!(
        r"
        SELECT
            COUNT(*) as total_interactions,
            COUNT(CASE WHEN retrieval_score_avg > 0.5 THEN 1 END) as successful_retrievals,
            AVG(context_relevance) as avg_context_relevance,
            AVG(answer_faithfulness) as avg_answer_faithfulness,
            COUNT(CASE WHEN user_feedback = 'positive' THEN 1 END) as positive_feedback,
            COUNT(CASE WHEN user_feedback = 'negative' THEN 1 END) as negative_feedback
        FROM rag_logs
        WHERE {filter_column} = $1
          AND created_at > NOW() - INTERVAL '1 day' * $2
    "
    );

    let rows = db
        .query_all(Statement::from_sql_and_values(
            DbBackend::Postgres,
            &sql,
            [id.into(), days.into()],
        ))
        .await?;

    let row = rows
        .first()
        .ok_or_else(|| AppError::Internal("No metrics result returned".into()))?;

    aggregate_metrics_from_row(row)
}

/// Extract [`AggregatedMetrics`] from a single SQL result row.
fn aggregate_metrics_from_row(row: &QueryResult) -> RepoResult<AggregatedMetrics> {
    Ok(AggregatedMetrics {
        total_interactions: row
            .try_get::<i64>("", "total_interactions")
            .inspect_err(|e| tracing::warn!(error = %e, "Failed to extract total_interactions"))
            .unwrap_or(0),
        successful_retrievals: row
            .try_get::<i64>("", "successful_retrievals")
            .inspect_err(|e| tracing::warn!(error = %e, "Failed to extract successful_retrievals"))
            .unwrap_or(0),
        avg_context_relevance: row
            .try_get::<Option<f32>>("", "avg_context_relevance")
            .ok()
            .flatten(),
        avg_answer_faithfulness: row
            .try_get::<Option<f32>>("", "avg_answer_faithfulness")
            .ok()
            .flatten(),
        positive_feedback: row
            .try_get::<i64>("", "positive_feedback")
            .inspect_err(|e| tracing::warn!(error = %e, "Failed to extract positive_feedback"))
            .unwrap_or(0),
        negative_feedback: row
            .try_get::<i64>("", "negative_feedback")
            .inspect_err(|e| tracing::warn!(error = %e, "Failed to extract negative_feedback"))
            .unwrap_or(0),
    })
}

// ============================================================================
// Trait implementation
// ============================================================================

#[async_trait]
impl RagLogRepository for SeaOrmRagLogRepository {
    async fn create(&self, entry: &RagLogEntry) -> RepoResult<Uuid> {
        let id = Uuid::new_v4();
        let chunks_json = serde_json::to_value(&entry.chunks_retrieved)
            .unwrap_or_else(|_| serde_json::Value::Array(vec![]));

        let active = rag_log::ActiveModel {
            id: Set(id),
            notebook_id: Set(entry.notebook_id),
            user_id: Set(entry.user_id),
            query: Set(String::new()),
            reformulated_query: Set(None),
            hyde_document: Set(None),
            query_hash: Set(entry.query_hash.clone()),
            reformulated_query_hash: Set(entry.reformulated_query_hash.clone()),
            chunks_retrieved: Set(chunks_json),
            response_id: Set(entry.response_id),
            retrieval_score_avg: Set(entry.retrieval_score_avg),
            context_relevance: Set(entry.context_relevance),
            answer_faithfulness: Set(None),
            answer_relevance: Set(None),
            model: Set(entry.model.clone()),
            provider: Set(entry.provider.clone()),
            user_feedback: Set(None),
            created_at: Set(Utc::now().into()),
        };

        active.insert(&self.db).await?;

        tracing::debug!(log_id = %id, notebook_id = %entry.notebook_id, "RAG log created");
        Ok(id)
    }

    async fn get_by_id(&self, id: Uuid) -> RepoResult<Option<rag_log::Model>> {
        Ok(rag_log::Entity::find_by_id(id).one(&self.db).await?)
    }

    async fn get_for_user(&self, id: Uuid, user_id: Uuid) -> RepoResult<rag_log::Model> {
        rag_log::Entity::find_by_id(id)
            .filter(rag_log::Column::UserId.eq(user_id))
            .one(&self.db)
            .await?
            .ok_or_else(|| AppError::NotFound("RAG log entry not found".into()))
    }

    async fn update_metrics(&self, log_id: Uuid, metrics: &RagMetrics) -> RepoResult<()> {
        let log = rag_log::Entity::find_by_id(log_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| AppError::NotFound("RAG log entry not found".into()))?;

        let mut active: rag_log::ActiveModel = log.into();
        active.context_relevance = Set(metrics.context_relevance);
        active.answer_faithfulness = Set(metrics.answer_faithfulness);
        active.answer_relevance = Set(metrics.answer_relevance);
        active.update(&self.db).await?;

        Ok(())
    }

    async fn update_feedback(&self, log_id: Uuid, user_id: Uuid, feedback: &str) -> RepoResult<()> {
        let log = rag_log::Entity::find_by_id(log_id)
            .filter(rag_log::Column::UserId.eq(user_id))
            .one(&self.db)
            .await?
            .ok_or_else(|| AppError::NotFound("RAG log entry not found".into()))?;

        let mut active: rag_log::ActiveModel = log.into();
        active.user_feedback = Set(Some(feedback.to_owned()));
        active.update(&self.db).await?;

        Ok(())
    }

    async fn get_notebook_metrics(
        &self,
        notebook_id: Uuid,
        days: i32,
    ) -> RepoResult<AggregatedMetrics> {
        query_aggregated_metrics(&self.db, MetricsScope::Notebook, notebook_id, days).await
    }

    async fn get_user_metrics(&self, user_id: Uuid, days: i32) -> RepoResult<AggregatedMetrics> {
        query_aggregated_metrics(&self.db, MetricsScope::User, user_id, days).await
    }

    async fn get_rag_log_ids_for_messages(
        &self,
        message_ids: &[Uuid],
    ) -> RepoResult<HashMap<Uuid, (Uuid, Option<String>)>> {
        if message_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let logs = rag_log::Entity::find()
            .filter(rag_log::Column::ResponseId.is_in(message_ids.to_vec()))
            .all(&self.db)
            .await?;

        let map: HashMap<Uuid, (Uuid, Option<String>)> = logs
            .into_iter()
            .filter_map(|log| {
                log.response_id
                    .map(|rid| (rid, (log.id, log.user_feedback)))
            })
            .collect();

        Ok(map)
    }

    async fn get_by_response_id(&self, response_id: Uuid) -> RepoResult<Option<rag_log::Model>> {
        Ok(rag_log::Entity::find()
            .filter(rag_log::Column::ResponseId.eq(response_id))
            .one(&self.db)
            .await?)
    }

    async fn get_preferred_source_ids(&self, notebook_id: Uuid) -> RepoResult<Vec<Uuid>> {
        // Extract distinct source_ids from chunks_retrieved JSONB for positive-feedback logs
        const SQL: &str = r"
            SELECT DISTINCT (elem->>'source_id')::uuid AS source_id
            FROM rag_logs,
                 jsonb_array_elements(chunks_retrieved) AS elem
            WHERE notebook_id = $1
              AND user_feedback = 'positive'
              AND elem->>'source_id' IS NOT NULL
        ";

        let rows = self
            .db
            .query_all(self.stmt(SQL, [notebook_id.into()]))
            .await?;

        let ids = rows
            .iter()
            .filter_map(|r| r.try_get::<Uuid>("", "source_id").ok())
            .collect();

        Ok(ids)
    }

    async fn purge_old_logs(&self, retention_days: i32) -> RepoResult<u64> {
        const SQL: &str = "DELETE FROM rag_logs WHERE created_at < NOW() - INTERVAL '1 day' * $1";

        let result = self
            .db
            .execute(self.stmt(SQL, [retention_days.into()]))
            .await?;

        let deleted = result.rows_affected();
        tracing::info!(deleted, retention_days, "RAG log purge completed");
        Ok(deleted)
    }
}
