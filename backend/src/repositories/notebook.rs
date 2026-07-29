//! SeaORM implementation of NotebookRepository

use async_trait::async_trait;
use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, ConnectionTrait, DatabaseConnection, EntityTrait,
    FromQueryResult, PaginatorTrait, QueryFilter, QueryOrder, Set, Statement,
};
use uuid::Uuid;

use crate::entities::{Notebook, Source, notebook, source};
use crate::error::NotebookError;

use super::traits::{NotebookRepository, NotebookWithSourceCount, RepoResult};

/// Raw SQL query result - maps directly to NotebookWithSourceCount.
#[derive(Debug, Clone, FromQueryResult)]
struct NotebookWithSourceCountRow {
    id: Uuid,
    user_id: Uuid,
    title: String,
    description: Option<String>,
    memory_enabled: bool,
    is_demo: bool,
    suggested_questions: serde_json::Value,
    created_at: chrono::DateTime<chrono::FixedOffset>,
    updated_at: chrono::DateTime<chrono::FixedOffset>,
    source_count: i64,
}

impl From<NotebookWithSourceCountRow> for NotebookWithSourceCount {
    fn from(row: NotebookWithSourceCountRow) -> Self {
        Self {
            id: row.id,
            user_id: row.user_id,
            title: row.title,
            description: row.description,
            memory_enabled: row.memory_enabled,
            is_demo: row.is_demo,
            suggested_questions: row.suggested_questions,
            created_at: row.created_at,
            updated_at: row.updated_at,
            source_count: row.source_count,
        }
    }
}

const LIST_WITH_COUNTS_SQL: &str = r"
    SELECT n.id, n.user_id, n.title, n.description, n.memory_enabled,
           n.is_demo, n.suggested_questions, n.created_at, n.updated_at,
           COUNT(s.id) as source_count
    FROM notebooks n
    LEFT JOIN sources s ON s.notebook_id = n.id
    WHERE n.user_id = $1
    GROUP BY n.id, n.user_id, n.title, n.description, n.memory_enabled,
             n.is_demo, n.suggested_questions, n.created_at, n.updated_at
    ORDER BY n.is_demo DESC, n.updated_at DESC
";

#[derive(Clone)]
pub struct SeaOrmNotebookRepository {
    db: DatabaseConnection,
}

impl SeaOrmNotebookRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }
}

#[async_trait]
impl NotebookRepository for SeaOrmNotebookRepository {
    #[tracing::instrument(skip(self, title, description), fields(%user_id))]
    async fn create(
        &self,
        user_id: Uuid,
        title: String,
        description: Option<String>,
    ) -> RepoResult<notebook::Model> {
        let now = Utc::now().into();

        let new_notebook = notebook::ActiveModel {
            id: Set(Uuid::new_v4()),
            user_id: Set(user_id),
            title: Set(title),
            description: Set(description),
            memory_enabled: Set(true),
            is_demo: Set(false),
            suggested_questions: Set(serde_json::json!([])),
            created_at: Set(now),
            updated_at: Set(now),
        };

        Ok(new_notebook.insert(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn get_by_id(&self, notebook_id: Uuid) -> RepoResult<Option<notebook::Model>> {
        Ok(Notebook::find_by_id(notebook_id).one(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id, %user_id))]
    async fn get_for_user(&self, notebook_id: Uuid, user_id: Uuid) -> RepoResult<notebook::Model> {
        Notebook::find_by_id(notebook_id)
            .filter(notebook::Column::UserId.eq(user_id))
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                NotebookError::NotFound {
                    id: notebook_id.to_string(),
                }
                .into()
            })
    }

    #[tracing::instrument(skip(self), fields(%user_id))]
    async fn list_for_user(&self, user_id: Uuid) -> RepoResult<Vec<notebook::Model>> {
        Ok(Notebook::find()
            .filter(notebook::Column::UserId.eq(user_id))
            .order_by_desc(notebook::Column::UpdatedAt)
            .all(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self), fields(%user_id))]
    async fn list_with_source_counts(
        &self,
        user_id: Uuid,
    ) -> RepoResult<Vec<NotebookWithSourceCount>> {
        let rows = NotebookWithSourceCountRow::find_by_statement(Statement::from_sql_and_values(
            self.db.get_database_backend(),
            LIST_WITH_COUNTS_SQL,
            [user_id.into()],
        ))
        .all(&self.db)
        .await?;

        let notebooks: Vec<_> = rows.into_iter().map(Into::into).collect();

        tracing::debug!(
            %user_id,
            count = notebooks.len(),
            "Fetched notebooks with source counts"
        );

        Ok(notebooks)
    }

    #[tracing::instrument(skip(self, title, description), fields(%notebook_id, %user_id))]
    async fn update(
        &self,
        notebook_id: Uuid,
        user_id: Uuid,
        title: Option<String>,
        description: Option<Option<String>>,
        memory_enabled: Option<bool>,
    ) -> RepoResult<notebook::Model> {
        let notebook = self.get_for_user(notebook_id, user_id).await?;
        let mut active: notebook::ActiveModel = notebook.into();

        if let Some(t) = title {
            active.title = Set(t);
        }
        if let Some(d) = description {
            active.description = Set(d);
        }
        if let Some(me) = memory_enabled {
            active.memory_enabled = Set(me);
        }
        active.updated_at = Set(Utc::now().into());

        Ok(active.update(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id, %user_id))]
    async fn delete(&self, notebook_id: Uuid, user_id: Uuid) -> RepoResult<()> {
        // Verify ownership first (will return NotFound if not owned)
        let notebook = self.get_for_user(notebook_id, user_id).await?;

        let result = Notebook::delete_by_id(notebook.id).exec(&self.db).await?;

        if result.rows_affected == 0 {
            return Err(NotebookError::NotFound {
                id: notebook_id.to_string(),
            }
            .into());
        }
        Ok(())
    }

    /// Count non-demo notebooks for a user (demo notebooks are excluded from quotas).
    #[tracing::instrument(skip(self), fields(%user_id))]
    async fn count_for_user(&self, user_id: Uuid) -> RepoResult<u64> {
        Ok(Notebook::find()
            .filter(notebook::Column::UserId.eq(user_id))
            .filter(notebook::Column::IsDemo.eq(false))
            .count(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id, %user_id))]
    async fn get_with_source_count(
        &self,
        notebook_id: Uuid,
        user_id: Uuid,
    ) -> RepoResult<(notebook::Model, u64)> {
        let notebook = self.get_for_user(notebook_id, user_id).await?;

        let source_count = Source::find()
            .filter(source::Column::NotebookId.eq(notebook_id))
            .count(&self.db)
            .await?;

        Ok((notebook, source_count))
    }

    #[tracing::instrument(skip(self), fields(%user_id))]
    async fn find_demo_for_user(&self, user_id: Uuid) -> RepoResult<Option<notebook::Model>> {
        Ok(Notebook::find()
            .filter(notebook::Column::UserId.eq(user_id))
            .filter(notebook::Column::IsDemo.eq(true))
            .one(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self, questions), fields(%notebook_id))]
    async fn update_suggested_questions(
        &self,
        notebook_id: Uuid,
        questions: Vec<String>,
    ) -> RepoResult<notebook::Model> {
        let notebook = Notebook::find_by_id(notebook_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| NotebookError::NotFound {
                id: notebook_id.to_string(),
            })?;

        let mut active: notebook::ActiveModel = notebook.into();
        active.suggested_questions = Set(serde_json::json!(questions));
        active.updated_at = Set(Utc::now().into());

        Ok(active.update(&self.db).await?)
    }
}
