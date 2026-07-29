//! SeaORM implementation of NoteRepository

use async_trait::async_trait;
use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, DatabaseConnection, EntityTrait, PaginatorTrait, QueryFilter,
    QueryOrder, Set,
};
use uuid::Uuid;

use crate::entities::{Note, Notebook, note, notebook};
use crate::error::{AppError, NoteError};

use super::traits::{NoteRepository, RepoResult};

#[derive(Clone)]
pub struct SeaOrmNoteRepository {
    db: DatabaseConnection,
}

impl SeaOrmNoteRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    async fn fetch_or_not_found(&self, note_id: Uuid) -> RepoResult<note::Model> {
        Note::find_by_id(note_id)
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                AppError::from(NoteError::NotFound {
                    id: note_id.to_string(),
                })
            })
    }
}

#[async_trait]
impl NoteRepository for SeaOrmNoteRepository {
    async fn create(
        &self,
        notebook_id: Uuid,
        title: String,
        content: String,
        original_message_id: Option<Uuid>,
    ) -> RepoResult<note::Model> {
        let now = Utc::now().into();
        let new_note = note::ActiveModel {
            id: Set(Uuid::new_v4()),
            notebook_id: Set(notebook_id),
            title: Set(title),
            content: Set(content),
            original_message_id: Set(original_message_id),
            created_at: Set(now),
            updated_at: Set(now),
        };

        let result = new_note.insert(&self.db).await?;
        tracing::debug!(note_id = %result.id, %notebook_id, "Note created");
        Ok(result)
    }

    async fn get_by_id(&self, note_id: Uuid) -> RepoResult<Option<note::Model>> {
        Ok(Note::find_by_id(note_id).one(&self.db).await?)
    }

    async fn get_for_user(&self, note_id: Uuid, user_id: Uuid) -> RepoResult<note::Model> {
        Note::find_by_id(note_id)
            .inner_join(notebook::Entity)
            .filter(notebook::Column::UserId.eq(user_id))
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                AppError::from(NoteError::NotFound {
                    id: note_id.to_string(),
                })
            })
    }

    async fn list_for_notebook(&self, notebook_id: Uuid) -> RepoResult<Vec<note::Model>> {
        Ok(Note::find()
            .filter(note::Column::NotebookId.eq(notebook_id))
            .order_by_desc(note::Column::CreatedAt)
            .all(&self.db)
            .await?)
    }

    async fn update(
        &self,
        note_id: Uuid,
        user_id: Uuid,
        title: Option<String>,
        content: Option<String>,
    ) -> RepoResult<note::Model> {
        let existing = self.fetch_or_not_found(note_id).await?;

        // Verify notebook ownership
        Notebook::find_by_id(existing.notebook_id)
            .filter(notebook::Column::UserId.eq(user_id))
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                AppError::from(NoteError::NotFound {
                    id: note_id.to_string(),
                })
            })?;

        let mut active: note::ActiveModel = existing.into();

        if let Some(t) = title {
            active.title = Set(t);
        }
        if let Some(c) = content {
            active.content = Set(c);
        }
        active.updated_at = Set(Utc::now().into());

        Ok(active.update(&self.db).await?)
    }

    async fn delete(&self, note_id: Uuid, user_id: Uuid) -> RepoResult<()> {
        let note = self.fetch_or_not_found(note_id).await?;

        // Verify notebook ownership
        Notebook::find_by_id(note.notebook_id)
            .filter(notebook::Column::UserId.eq(user_id))
            .one(&self.db)
            .await?
            .ok_or_else(|| {
                AppError::from(NoteError::NotFound {
                    id: note_id.to_string(),
                })
            })?;

        Note::delete_by_id(note_id).exec(&self.db).await?;

        tracing::debug!(%note_id, "Note deleted");
        Ok(())
    }

    async fn count_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64> {
        Ok(Note::find()
            .filter(note::Column::NotebookId.eq(notebook_id))
            .count(&self.db)
            .await?)
    }
}
