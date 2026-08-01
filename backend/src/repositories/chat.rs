//! SeaORM implementation of ChatRepository

use async_trait::async_trait;
use chrono::{DateTime, FixedOffset};
use sea_orm::{
    ActiveModelTrait, ColumnTrait, ConnectionTrait, DatabaseConnection, DatabaseTransaction,
    EntityTrait, PaginatorTrait, QueryFilter, QueryOrder, QuerySelect, Set,
};
use uuid::Uuid;

use crate::entities::{ChatMessage, chat_message};
use crate::error::ChatError;
use crate::llm::Citation;

use super::traits::{
    ChatRepository, DEFAULT_CHAT_HISTORY_LIMIT, MAX_CHAT_HISTORY_LIMIT, PaginatedChatHistory,
    RepoResult,
};

#[derive(Clone)]
pub struct SeaOrmChatRepository {
    db: DatabaseConnection,
}

impl SeaOrmChatRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    /// Base query filtered by notebook_id
    fn for_notebook(notebook_id: Uuid) -> sea_orm::Select<ChatMessage> {
        ChatMessage::find().filter(chat_message::Column::NotebookId.eq(notebook_id))
    }

    #[allow(clippy::too_many_arguments)]
    async fn insert_message<C: ConnectionTrait>(
        connection: &C,
        notebook_id: Uuid,
        role: &str,
        content: &str,
        citations: &[Citation],
        model: Option<&str>,
        session_id: Option<Uuid>,
    ) -> RepoResult<chat_message::Model> {
        let citations_json = serde_json::to_value(citations).map_err(|e| {
            ChatError::CitationsSerializationFailed {
                notebook_id: notebook_id.to_string(),
                reason: e.to_string(),
            }
        })?;

        let message = chat_message::ActiveModel {
            id: Set(Uuid::new_v4()),
            notebook_id: Set(notebook_id),
            role: Set(role.to_string()),
            content: Set(content.to_string()),
            citations: Set(citations_json),
            model: Set(model.map(String::from)),
            session_id: Set(session_id),
            ..Default::default()
        };

        let result = message
            .insert(connection)
            .await
            .map_err(|e| ChatError::SaveFailed {
                notebook_id: notebook_id.to_string(),
                reason: e.to_string(),
            })?;

        tracing::debug!(id = %result.id, %notebook_id, %role, "Message saved");
        Ok(result)
    }
}

#[async_trait]
impl ChatRepository for SeaOrmChatRepository {
    #[tracing::instrument(skip(self), fields(%message_id))]
    async fn get_by_id(&self, message_id: Uuid) -> RepoResult<Option<chat_message::Model>> {
        Ok(ChatMessage::find_by_id(message_id).one(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn get_conversation_up_to(
        &self,
        notebook_id: Uuid,
        up_to: DateTime<FixedOffset>,
    ) -> RepoResult<Vec<chat_message::Model>> {
        Ok(Self::for_notebook(notebook_id)
            .filter(chat_message::Column::CreatedAt.lte(up_to))
            .order_by_asc(chat_message::Column::CreatedAt)
            .limit(200)
            .all(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self, content, citations), fields(%notebook_id, %role))]
    async fn create_message(
        &self,
        notebook_id: Uuid,
        role: &str,
        content: &str,
        citations: &[Citation],
        model: Option<&str>,
        _agent_id: Option<Uuid>,
        session_id: Option<Uuid>,
    ) -> RepoResult<chat_message::Model> {
        Self::insert_message(
            &self.db,
            notebook_id,
            role,
            content,
            citations,
            model,
            session_id,
        )
        .await
    }

    #[tracing::instrument(skip(self, transaction, content, citations), fields(%notebook_id, %role))]
    async fn create_message_in_transaction(
        &self,
        transaction: &DatabaseTransaction,
        notebook_id: Uuid,
        role: &str,
        content: &str,
        citations: &[Citation],
        model: Option<&str>,
        _agent_id: Option<Uuid>,
        session_id: Option<Uuid>,
    ) -> RepoResult<chat_message::Model> {
        Self::insert_message(
            transaction,
            notebook_id,
            role,
            content,
            citations,
            model,
            session_id,
        )
        .await
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn get_latest_message(
        &self,
        notebook_id: Uuid,
    ) -> RepoResult<Option<chat_message::Model>> {
        Ok(Self::for_notebook(notebook_id)
            .order_by_desc(chat_message::Column::CreatedAt)
            .one(&self.db)
            .await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn get_history(
        &self,
        notebook_id: Uuid,
        limit: Option<u64>,
    ) -> RepoResult<Vec<chat_message::Model>> {
        let mut query =
            Self::for_notebook(notebook_id).order_by_asc(chat_message::Column::CreatedAt);

        if let Some(l) = limit {
            query = query.limit(l);
        }

        Ok(query.all(&self.db).await?)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn get_history_paginated(
        &self,
        notebook_id: Uuid,
        offset: Option<u64>,
        limit: Option<u64>,
    ) -> RepoResult<PaginatedChatHistory> {
        let offset = offset.unwrap_or(0);
        let limit = limit
            .unwrap_or(DEFAULT_CHAT_HISTORY_LIMIT)
            .min(MAX_CHAT_HISTORY_LIMIT);

        let total = Self::for_notebook(notebook_id).count(&self.db).await?;

        let mut messages = Self::for_notebook(notebook_id)
            .order_by_desc(chat_message::Column::CreatedAt)
            .offset(offset)
            .limit(limit)
            .all(&self.db)
            .await?;

        // Reverse to return messages in chronological order (oldest first) within each page.
        // DESC + offset gives us the correct "newest first" page selection, then reverse
        // ensures each page's messages are in ascending time order for display.
        messages.reverse();

        let has_more = offset + u64::try_from(messages.len()).unwrap_or(u64::MAX) < total;

        tracing::debug!(
            %notebook_id, %offset, %limit, %total,
            returned = messages.len(), %has_more,
            "Paginated chat history"
        );

        Ok(PaginatedChatHistory {
            messages,
            total,
            offset,
            limit,
            has_more,
        })
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn get_recent_history(
        &self,
        notebook_id: Uuid,
        max_messages: u64,
    ) -> RepoResult<Vec<chat_message::Model>> {
        let mut messages = Self::for_notebook(notebook_id)
            .order_by_desc(chat_message::Column::CreatedAt)
            .limit(max_messages)
            .all(&self.db)
            .await?;

        messages.reverse(); // Chronological order
        Ok(messages)
    }

    #[tracing::instrument(skip(self), fields(%notebook_id))]
    async fn clear_history(&self, notebook_id: Uuid) -> RepoResult<u64> {
        Ok(ChatMessage::delete_many()
            .filter(chat_message::Column::NotebookId.eq(notebook_id))
            .exec(&self.db)
            .await?
            .rows_affected)
    }
}
