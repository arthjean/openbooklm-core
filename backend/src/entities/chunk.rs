//! Chunk entity for RAG vector search.
//!
//! Sources are split into chunks for embedding and retrieval.
//! Each chunk stores text content; embeddings are in a separate pgvector column.
//!
//! ## Database Schema
//!
//! ```sql
//! embedding VECTOR(1024)  -- Voyage AI voyage-context-3 dimension
//! ```
//!
//! The `embedding` column is handled via raw SQL in `repositories::chunk`
//! and `repositories::search` because SeaORM doesn't have native pgvector support.

use sea_orm::entity::prelude::*;

/// Text chunk with vector embedding for RAG retrieval.
///
/// Note: `Eq` is not derived because the full DB row includes
/// a float vector column not represented in this struct.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "chunks")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Parent source document
    pub source_id: Uuid,

    /// Position within source (0-indexed)
    pub chunk_index: i32,

    /// Text content for this chunk
    pub content: String,

    /// LLM-generated contextual prefix for Contextual Retrieval.
    /// NULL for legacy chunks that haven't been contextualized.
    pub context_prefix: Option<String>,

    /// Larger parent chunk text for small-to-big retrieval.
    /// NULL for legacy chunks that predate the parent-child architecture.
    pub parent_content: Option<String>,

    /// Structured metadata: section_header, page_number, position, etc.
    pub metadata: Json,

    // Note: `embedding VECTOR(1024)` exists in DB but is handled via raw SQL
    // See: repositories::chunk::SeaOrmChunkRepository, repositories::search::SeaOrmSearchRepository
    pub created_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::source::Entity",
        from = "Column::SourceId",
        to = "super::source::Column::Id"
    )]
    Source,
}

super::impl_related!(super::source::Entity, Relation::Source);

impl ActiveModelBehavior for ActiveModel {}
