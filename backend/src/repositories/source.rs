//! SeaORM implementation of SourceRepository

use async_trait::async_trait;
use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, ConnectionTrait, DatabaseConnection, DbBackend, EntityTrait,
    PaginatorTrait, QueryFilter, QueryOrder, QuerySelect, Set, Statement, TransactionTrait,
};
use serde_json::json;
use uuid::Uuid;

use crate::entities::source::{self, SourceStatus};
use crate::entities::{Notebook, Source, notebook};
use crate::error::SourceError;
use crate::types::SourceType;

use super::traits::{RepoResult, SourceRepository};

#[derive(Clone)]
pub struct SeaOrmSourceRepository {
    db: DatabaseConnection,
}

impl SeaOrmSourceRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    /// Fetch a source by ID or return NotFound error.
    async fn fetch_or_not_found(&self, source_id: Uuid) -> RepoResult<source::Model> {
        Source::find_by_id(source_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                SourceError::NotFound {
                    id: source_id.to_string(),
                }
                .into()
            })
    }
}

#[async_trait]
impl SourceRepository for SeaOrmSourceRepository {
    #[tracing::instrument(skip(self, title, content, metadata), fields(%notebook_id))]
    async fn create(
        &self,
        notebook_id: Uuid,
        title: String,
        source_type: SourceType,
        content: String,
        metadata: Option<serde_json::Value>,
    ) -> RepoResult<source::Model> {
        let new_source = source::ActiveModel {
            id: Set(Uuid::new_v4()),
            notebook_id: Set(notebook_id),
            title: Set(title),
            source_type: Set(source_type.into()),
            content: Set(content),
            metadata: Set(metadata.unwrap_or_else(|| json!({}))),
            chunk_count: Set(0),
            status: Set(SourceStatus::Pending.into()),
            error_message: Set(None),
            created_at: Set(Utc::now().into()),
        };

        Ok(new_source.insert(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn get_by_id(&self, source_id: Uuid) -> RepoResult<Option<source::Model>> {
        Ok(Source::find_by_id(source_id).one(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%source_id, %user_id))]
    async fn get_for_user(&self, source_id: Uuid, user_id: Uuid) -> RepoResult<source::Model> {
        let source = self.fetch_or_not_found(source_id).await?;

        // Verify notebook ownership
        Notebook::find_by_id(source.notebook_id)
            .filter(notebook::Column::UserId.eq(user_id))
            .one(&self.db)
            .await?
            .ok_or_else(|| SourceError::NotFound {
                id: source_id.to_string(),
            })?;

        Ok(source)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn list_for_notebook(&self, notebook_id: Uuid) -> RepoResult<Vec<source::Model>> {
        Ok(Source::find()
            .filter(source::Column::NotebookId.eq(notebook_id))
            .order_by_desc(source::Column::CreatedAt)
            .all(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self), fields(%source_id, %user_id))]
    async fn delete(&self, source_id: Uuid, user_id: Uuid) -> RepoResult<()> {
        let source = self.fetch_or_not_found(source_id).await?;

        // Verify notebook ownership
        Notebook::find_by_id(source.notebook_id)
            .filter(notebook::Column::UserId.eq(user_id))
            .one(&self.db)
            .await?
            .ok_or_else(|| SourceError::NotFound {
                id: source_id.to_string(),
            })?;

        // Wrap chunk deletion + source deletion in a transaction so a failure
        // in either step rolls back both — no orphaned chunks left behind.
        self.db
            .transaction::<_, (), sea_orm::DbErr>(|txn| {
                Box::pin(async move {
                    // Delete chunks first (no-op if source has none)
                    txn.execute(Statement::from_sql_and_values(
                        DbBackend::Postgres,
                        "DELETE FROM chunks WHERE source_id = $1",
                        [source_id.into()],
                    ))
                    .await?;

                    // Then delete the source record
                    Source::delete_by_id(source_id).exec(txn).await?;

                    Ok(())
                })
            })
            .await
            .map_err(|e| match e {
                sea_orm::TransactionError::Connection(db_err)
                | sea_orm::TransactionError::Transaction(db_err) => db_err,
            })?;

        Ok(())
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn update_status(
        &self,
        source_id: Uuid,
        status: SourceStatus,
        error_message: Option<String>,
    ) -> RepoResult<source::Model> {
        let source = self.fetch_or_not_found(source_id).await?;
        let mut active: source::ActiveModel = source.into();

        active.status = Set(status.into());
        active.error_message = Set(error_message);

        Ok(active.update(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn update_chunk_count(
        &self,
        source_id: Uuid,
        chunk_count: i32,
    ) -> RepoResult<source::Model> {
        let source = self.fetch_or_not_found(source_id).await?;
        let mut active: source::ActiveModel = source.into();

        active.chunk_count = Set(chunk_count);

        Ok(active.update(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn count_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64> {
        Ok(Source::find()
            .filter(source::Column::NotebookId.eq(notebook_id))
            .count(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn count_web_sources_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64> {
        Ok(Source::find()
            .filter(source::Column::NotebookId.eq(notebook_id))
            .filter(source::Column::SourceType.is_in(["web", "youtube"]))
            .count(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self))]
    async fn get_by_status(&self, status: SourceStatus) -> RepoResult<Vec<source::Model>> {
        let status_str: String = status.into();
        Ok(Source::find()
            .filter(source::Column::Status.eq(status_str))
            .limit(1000)
            .all(&self.db)
            .await?)
    }
}
