//! SeaORM/raw SQL implementation of MemoryRepository for pgvector.
//!
//! Uses raw SQL for INSERT (VECTOR column) and similarity search (`<=>` operator),
//! and SeaORM for standard CRUD operations.

use async_trait::async_trait;
use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, ConnectionTrait, DatabaseConnection, DbBackend, EntityTrait,
    PaginatorTrait, QueryFilter, QueryOrder, QuerySelect, Set, Statement,
};
use uuid::Uuid;

use crate::entities::{NotebookMemory, notebook_memory};
use crate::error::MemoryError;
use crate::services::rag::utils::format_embedding;

use super::traits::{MemoryRepository, MemorySearchResult, RepoResult};

// ─── SQL Constants ───────────────────────────────────────────────────────────

const INSERT_MEMORY_SQL: &str = r"
    INSERT INTO notebook_memories (id, notebook_id, content, memory_type, metadata, salience, embedding, created_at, updated_at)
    VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::vector, $8, $9)
    RETURNING id, notebook_id, content, memory_type, metadata, salience, created_at, updated_at
";

const SEARCH_SIMILAR_SQL: &str = r"
    SELECT id, notebook_id, content, memory_type, metadata, salience, created_at, updated_at,
           (1.0 - (embedding <=> $1::vector)) as similarity
    FROM notebook_memories
    WHERE notebook_id = $2
    ORDER BY embedding <=> $1::vector
    LIMIT $3
";

const UPDATE_EMBEDDING_SQL: &str = r"
    UPDATE notebook_memories SET embedding = $1::vector, updated_at = $2 WHERE id = $3
";

const DELETE_ALL_FOR_NOTEBOOK_SQL: &str = "DELETE FROM notebook_memories WHERE notebook_id = $1";

const UPDATE_SALIENCE_SQL: &str =
    "UPDATE notebook_memories SET salience = $1, updated_at = $2 WHERE id = $3";

const TOUCH_MEMORY_SQL: &str = "UPDATE notebook_memories SET updated_at = $1 WHERE id = $2";

// ─── Repository ──────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct SeaOrmMemoryRepository {
    db: DatabaseConnection,
}

impl SeaOrmMemoryRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    #[allow(clippy::unused_self)]
    fn stmt(&self, sql: &str, values: impl IntoIterator<Item = sea_orm::Value>) -> Statement {
        Statement::from_sql_and_values(DbBackend::Postgres, sql, values)
    }
}

#[async_trait]
impl MemoryRepository for SeaOrmMemoryRepository {
    #[tracing::instrument(skip(self, content, metadata, embedding), fields(%notebook_id, %memory_type, %salience))]
    async fn create_with_embedding(
        &self,
        notebook_id: Uuid,
        content: &str,
        memory_type: &str,
        metadata: serde_json::Value,
        salience: f32,
        embedding: &[f32],
    ) -> RepoResult<notebook_memory::Model> {
        let id = Uuid::new_v4();
        let now = Utc::now().fixed_offset();
        let metadata_str = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());

        let row = self
            .db
            .query_one(self.stmt(
                INSERT_MEMORY_SQL,
                [
                    id.into(),
                    notebook_id.into(),
                    content.to_string().into(),
                    memory_type.to_string().into(),
                    metadata_str.into(),
                    salience.into(),
                    format_embedding(embedding).into(),
                    now.into(),
                    now.into(),
                ],
            ))
            .await?
            .ok_or_else(|| {
                MemoryError::Database(sea_orm::DbErr::Custom(
                    "INSERT RETURNING returned no rows".to_string(),
                ))
            })?;

        let model = notebook_memory::Model {
            id: row.try_get("", "id")?,
            notebook_id: row.try_get("", "notebook_id")?,
            content: row.try_get("", "content")?,
            memory_type: row.try_get("", "memory_type")?,
            metadata: row.try_get("", "metadata")?,
            salience: row.try_get("", "salience")?,
            created_at: row.try_get("", "created_at")?,
            updated_at: row.try_get("", "updated_at")?,
        };

        tracing::debug!(memory_id = %model.id, %notebook_id, %memory_type, "Memory created");
        Ok(model)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn list_for_notebook(
        &self,
        notebook_id: Uuid,
    ) -> RepoResult<Vec<notebook_memory::Model>> {
        Ok(NotebookMemory::find()
            .filter(notebook_memory::Column::NotebookId.eq(notebook_id))
            .order_by_desc(notebook_memory::Column::Salience)
            .order_by_desc(notebook_memory::Column::CreatedAt)
            .all(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self, query_embedding), fields(%notebook_id, %limit))]
    async fn search_similar(
        &self,
        notebook_id: Uuid,
        query_embedding: &[f32],
        limit: i32,
    ) -> RepoResult<Vec<MemorySearchResult>> {
        let embedding_str = format_embedding(query_embedding);

        let rows = self
            .db
            .query_all(self.stmt(
                SEARCH_SIMILAR_SQL,
                [embedding_str.into(), notebook_id.into(), limit.into()],
            ))
            .await?;

        let results: Vec<MemorySearchResult> = rows
            .into_iter()
            .map(|row| {
                let similarity: f64 = row.try_get("", "similarity")?;
                let memory = notebook_memory::Model {
                    id: row.try_get("", "id")?,
                    notebook_id: row.try_get("", "notebook_id")?,
                    content: row.try_get("", "content")?,
                    memory_type: row.try_get("", "memory_type")?,
                    metadata: row.try_get("", "metadata")?,
                    salience: row.try_get("", "salience")?,
                    created_at: row.try_get("", "created_at")?,
                    updated_at: row.try_get("", "updated_at")?,
                };
                Ok(MemorySearchResult {
                    memory,
                    #[allow(clippy::cast_possible_truncation)]
                    similarity: similarity.max(0.0) as f32,
                })
            })
            .collect::<Result<_, sea_orm::DbErr>>()?;

        tracing::debug!(%notebook_id, count = results.len(), %limit, "Memory similarity search");
        Ok(results)
    }

    async fn get_by_id(&self, memory_id: Uuid) -> RepoResult<Option<notebook_memory::Model>> {
        Ok(NotebookMemory::find_by_id(memory_id).one(&self.db).await?)
    }

    #[tracing::instrument(skip(self, embedding), fields(%memory_id))]
    async fn update(
        &self,
        memory_id: Uuid,
        content: Option<String>,
        salience: Option<f32>,
        metadata: Option<serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> RepoResult<notebook_memory::Model> {
        let existing = NotebookMemory::find_by_id(memory_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| MemoryError::NotFound {
                id: memory_id.to_string(),
            })?;

        let mut active: notebook_memory::ActiveModel = existing.into();

        if let Some(c) = content {
            active.content = Set(c);
        }
        if let Some(s) = salience {
            active.salience = Set(s);
        }
        if let Some(m) = metadata {
            active.metadata = Set(m);
        }
        active.updated_at = Set(Utc::now().fixed_offset());

        let model = active.update(&self.db).await?;

        // Update embedding via raw SQL if provided (SeaORM can't set VECTOR columns)
        if let Some(emb) = embedding {
            let now = Utc::now().fixed_offset();
            self.db
                .execute(self.stmt(
                    UPDATE_EMBEDDING_SQL,
                    [format_embedding(emb).into(), now.into(), memory_id.into()],
                ))
                .await?;
        }

        tracing::debug!(%memory_id, "Memory updated");
        Ok(model)
    }

    #[tracing::instrument(skip(self), fields(%memory_id))]
    async fn delete(&self, memory_id: Uuid) -> RepoResult<()> {
        let result = NotebookMemory::delete_by_id(memory_id)
            .exec(&self.db)
            .await?;

        if result.rows_affected == 0 {
            return Err(MemoryError::NotFound {
                id: memory_id.to_string(),
            }
            .into());
        }

        tracing::debug!(%memory_id, "Memory deleted");
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn delete_all_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64> {
        let result = self
            .db
            .execute(self.stmt(DELETE_ALL_FOR_NOTEBOOK_SQL, [notebook_id.into()]))
            .await?;

        let count = result.rows_affected();
        tracing::debug!(%notebook_id, %count, "All memories deleted for notebook");
        Ok(count)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn count_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64> {
        // Exclude conversation_summary — those are system-managed and do not
        // count against the per-plan memory limit (US-002, AC7).
        Ok(NotebookMemory::find()
            .filter(notebook_memory::Column::NotebookId.eq(notebook_id))
            .filter(notebook_memory::Column::MemoryType.ne("conversation_summary"))
            .count(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id, %memory_type))]
    async fn count_by_type(&self, notebook_id: Uuid, memory_type: &str) -> RepoResult<u64> {
        Ok(NotebookMemory::find()
            .filter(notebook_memory::Column::NotebookId.eq(notebook_id))
            .filter(notebook_memory::Column::MemoryType.eq(memory_type))
            .count(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id, %memory_type, %keep))]
    async fn delete_oldest_by_type(
        &self,
        notebook_id: Uuid,
        memory_type: &str,
        keep: u64,
    ) -> RepoResult<u64> {
        let count = self.count_by_type(notebook_id, memory_type).await?;
        if count <= keep {
            return Ok(0);
        }

        let overflow = count - keep;
        let oldest: Vec<notebook_memory::Model> = NotebookMemory::find()
            .filter(notebook_memory::Column::NotebookId.eq(notebook_id))
            .filter(notebook_memory::Column::MemoryType.eq(memory_type))
            .order_by_asc(notebook_memory::Column::CreatedAt)
            .limit(overflow)
            .all(&self.db)
            .await?;

        let ids: Vec<Uuid> = oldest.iter().map(|m| m.id).collect();
        if ids.is_empty() {
            return Ok(0);
        }

        let result = NotebookMemory::delete_many()
            .filter(notebook_memory::Column::Id.is_in(ids))
            .exec(&self.db)
            .await?;

        tracing::debug!(%notebook_id, %memory_type, deleted = result.rows_affected, "Evicted oldest memories by type");
        Ok(result.rows_affected)
    }

    #[tracing::instrument(skip(self), fields(%memory_id))]
    async fn touch_memory(&self, memory_id: Uuid) -> RepoResult<()> {
        let now = Utc::now().fixed_offset();
        let result = self
            .db
            .execute(self.stmt(TOUCH_MEMORY_SQL, [now.into(), memory_id.into()]))
            .await?;

        if result.rows_affected() == 0 {
            return Err(MemoryError::NotFound {
                id: memory_id.to_string(),
            }
            .into());
        }

        tracing::debug!(%memory_id, "Memory touched (updated_at refreshed)");
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(%memory_id, %new_salience))]
    async fn update_salience(&self, memory_id: Uuid, new_salience: f32) -> RepoResult<()> {
        let now = Utc::now().fixed_offset();
        let result = self
            .db
            .execute(self.stmt(
                UPDATE_SALIENCE_SQL,
                [new_salience.into(), now.into(), memory_id.into()],
            ))
            .await?;

        if result.rows_affected() == 0 {
            return Err(MemoryError::NotFound {
                id: memory_id.to_string(),
            }
            .into());
        }

        tracing::debug!(%memory_id, %new_salience, "Memory salience updated");
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(%notebook_id, %threshold))]
    async fn delete_below_salience(&self, notebook_id: Uuid, threshold: f32) -> RepoResult<u64> {
        let result = NotebookMemory::delete_many()
            .filter(notebook_memory::Column::NotebookId.eq(notebook_id))
            .filter(notebook_memory::Column::Salience.lt(threshold))
            .filter(notebook_memory::Column::MemoryType.ne("conversation_summary"))
            .exec(&self.db)
            .await?;

        tracing::debug!(%notebook_id, %threshold, deleted = result.rows_affected, "Deleted low-salience memories");
        Ok(result.rows_affected)
    }
}
