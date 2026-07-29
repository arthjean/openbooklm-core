//! Chat message entity for conversation history.
//!
//! Stores user prompts and AI responses with RAG citations.
//! Messages are ordered by `created_at` within a notebook.

use sea_orm::entity::prelude::*;

/// Chat message in a notebook conversation.
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "chat_messages")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    pub notebook_id: Uuid,

    /// Message author: "user" | "assistant"
    pub role: String,

    /// Message text (markdown for assistant responses)
    pub content: String,

    /// RAG citations array (JSONB):
    /// `[{ "source_id": "uuid", "chunk_index": 0, "text": "...", "relevance_score": 0.95 }]`
    pub citations: Json,

    /// LLM model used (e.g., "claude-sonnet-4-6-20260220"), None for user messages
    pub model: Option<String>,

    /// Session grouping ID — shared by all messages in one conversation session.
    /// None for messages created before this feature was added.
    pub session_id: Option<Uuid>,

    pub created_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::notebook::Entity",
        from = "Column::NotebookId",
        to = "super::notebook::Column::Id"
    )]
    Notebook,
}

super::impl_related!(super::notebook::Entity, Relation::Notebook);

impl ActiveModelBehavior for ActiveModel {}
