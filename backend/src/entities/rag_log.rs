//! RAG evaluation log entity for quality tracking.
//!
//! Stores per-interaction query hashes, retrieved chunk identifiers, scores and
//! user feedback. Legacy raw-text columns remain only for write compatibility
//! and are scrubbed by a database trigger.
//!
//! Logs are created asynchronously (via `tokio::spawn`) so they don't
//! delay the SSE response to the user. Complex aggregation queries
//! remain as raw SQL in `services::rag_log`.

use sea_orm::entity::prelude::*;

/// RAG interaction log entry for quality analysis.
#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "rag_logs")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Notebook this interaction belongs to
    pub notebook_id: Uuid,

    /// User who initiated the query
    pub user_id: Uuid,

    /// Legacy compatibility column. Always empty after the redaction migration.
    pub query: String,

    /// Legacy compatibility column. Always NULL after the redaction migration.
    pub reformulated_query: Option<String>,

    /// Legacy compatibility column. Always NULL after the redaction migration.
    pub hyde_document: Option<String>,

    /// SHA-256 of the original user query.
    pub query_hash: String,

    /// SHA-256 of the reformulated query, when one was used.
    pub reformulated_query_hash: Option<String>,

    /// Retrieved chunks with metadata (JSONB):
    /// `[{ "chunk_id": "uuid", "source_id": "uuid", "chunk_index": 0, "relevance_score": 0.85 }]`
    pub chunks_retrieved: Json,

    /// Associated chat message response
    pub response_id: Option<Uuid>,

    /// Average relevance score across retrieved chunks
    pub retrieval_score_avg: Option<f32>,

    /// Contextual relevance score (from reranking)
    pub context_relevance: Option<f32>,

    /// Faithfulness of the answer to source material (LLM-judged)
    pub answer_faithfulness: Option<f32>,

    /// Relevance of the answer to the original query (LLM-judged)
    pub answer_relevance: Option<f32>,

    /// LLM model used (e.g., "claude-sonnet-4-6-20260220")
    pub model: Option<String>,

    /// LLM provider (e.g., "anthropic", "openai", "mistral")
    pub provider: Option<String>,

    /// User feedback: "positive" | "negative" | NULL
    pub user_feedback: Option<String>,

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
    #[sea_orm(
        belongs_to = "super::account::Entity",
        from = "Column::UserId",
        to = "super::account::Column::Id"
    )]
    Account,
    #[sea_orm(
        belongs_to = "super::chat_message::Entity",
        from = "Column::ResponseId",
        to = "super::chat_message::Column::Id"
    )]
    ChatMessage,
}

super::impl_related!(super::notebook::Entity, Relation::Notebook);
super::impl_related!(super::account::Entity, Relation::Account);
super::impl_related!(super::chat_message::Entity, Relation::ChatMessage);

impl ActiveModelBehavior for ActiveModel {}
