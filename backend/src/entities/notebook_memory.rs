//! Notebook memory entity for per-notebook user modeling.
//!
//! Stores extracted facts about the user's expertise, goals, preferences,
//! and key insights. Used to personalize AI responses across chat sessions.
//!
//! ## Memory Types
//!
//! | Type | Description |
//! |------|-------------|
//! | `fact` | Factual information mentioned by the user |
//! | `preference` | User preferences for topics, formats, depth |
//! | `summary` | Summarized insights from conversations |
//! | `expertise` | User's demonstrated knowledge areas |
//! | `goal` | Research goals and objectives |
//!
//! ## Database Schema
//!
//! ```sql
//! embedding VECTOR(1024)  -- Voyage AI voyage-context-3 dimension
//! ```
//!
//! The `embedding` column is handled via raw SQL in `repositories::memory`
//! because SeaORM doesn't have native pgvector support.

use sea_orm::entity::prelude::*;

/// A memory fact extracted from chat interactions within a notebook.
///
/// Note: `Eq` is not derived because the full DB row includes
/// a float vector column not represented in this struct.
#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "notebook_memories")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Parent notebook this memory belongs to
    pub notebook_id: Uuid,

    /// The memory content text
    pub content: String,

    /// Type: "fact", "preference", "summary", "expertise", "goal"
    pub memory_type: String,

    /// Structured metadata (extraction context, source message IDs, etc.)
    pub metadata: Json,

    /// Importance score (higher = more relevant to inject in prompts)
    pub salience: f32,

    // Note: `embedding VECTOR(1024)` exists in DB but is handled via raw SQL
    // See: repositories::memory (future US-004)
    pub created_at: DateTimeWithTimeZone,
    pub updated_at: DateTimeWithTimeZone,
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
