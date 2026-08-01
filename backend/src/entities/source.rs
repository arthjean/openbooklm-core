//! Source entity for document storage in notebooks.
//!
//! Sources represent uploaded documents (PDF, text, markdown) or scraped web pages.
//! Each source is chunked and embedded for RAG retrieval.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Source Status
// ============================================================================

/// Processing status for async document ingestion pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceStatus {
    /// Queued for processing
    #[default]
    Pending,
    /// Currently being chunked/extracted
    Processing,
    /// Generating LLM context prefixes for chunks (Contextual Retrieval)
    Contextualizing,
    /// Generating vector embeddings
    Embedding,
    /// Successfully processed, ready for RAG
    Ready,
    /// Processing failed (check `error_message`)
    Error,
}

impl SourceStatus {
    /// Returns the string representation for database storage.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Processing => "processing",
            Self::Contextualizing => "contextualizing",
            Self::Embedding => "embedding",
            Self::Ready => "ready",
            Self::Error => "error",
        }
    }
}

impl From<&str> for SourceStatus {
    fn from(s: &str) -> Self {
        match s {
            "processing" => Self::Processing,
            "contextualizing" => Self::Contextualizing,
            "embedding" => Self::Embedding,
            "ready" => Self::Ready,
            "error" => Self::Error,
            _ => Self::Pending,
        }
    }
}

impl From<SourceStatus> for String {
    fn from(s: SourceStatus) -> Self {
        s.as_str().to_owned()
    }
}

// ============================================================================
// Entity
// ============================================================================

/// Source document stored in a notebook.
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "sources")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    pub notebook_id: Uuid,

    /// Display title (filename or page title)
    pub title: String,

    /// Type: "pdf" | "text" | "markdown" | "web"
    pub source_type: String,

    /// Extracted text content
    pub content: String,

    /// Original metadata (file info, URL, etc.)
    pub metadata: Json,

    /// Number of chunks created from this source
    pub chunk_count: i32,

    /// Published index generation used by every retrieval and citation read.
    pub active_generation_id: Option<Uuid>,

    /// Processing status: "pending" | "processing" | "contextualizing" | "embedding" | "ready" | "error"
    pub status: String,

    /// Error details if status == "error"
    pub error_message: Option<String>,

    pub created_at: DateTimeWithTimeZone,
}

impl Model {
    /// Parse status string into typed enum.
    #[inline]
    pub fn status_enum(&self) -> SourceStatus {
        SourceStatus::from(self.status.as_str())
    }
}

// ============================================================================
// Relations
// ============================================================================

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::notebook::Entity",
        from = "Column::NotebookId",
        to = "super::notebook::Column::Id"
    )]
    Notebook,
    #[sea_orm(has_many = "super::chunk::Entity")]
    Chunks,
}

super::impl_related!(super::notebook::Entity, Relation::Notebook);
super::impl_related!(super::chunk::Entity, Relation::Chunks);

impl ActiveModelBehavior for ActiveModel {}
