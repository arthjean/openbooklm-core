//! Index generation entity (US-006, EP-002).
//!
//! One row is one immutable attempt to index a source. It is created
//! `building`, gains chunks under its own id, and either becomes `published` in
//! a single transaction or `failed`. A generation is never edited into a
//! different shape: the replacement is a new row, which is what lets the
//! previous one stay searchable while it is built.
//!
//! ## Database Schema
//!
//! See `migration-core/src/core_track/m20260801_000001_index_generations.rs`.
//! The invariants live there, in the database, not in this struct.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Lifecycle
// ============================================================================

/// Where a generation is in its lifecycle.
///
/// Only `Published` is searchable, and only through `sources.active_generation_id`:
/// a published generation that has been replaced stays `Published` and becomes
/// unreachable by losing the pointer, which is exactly what makes rollback a
/// pointer move rather than a copy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GenerationState {
    /// Owned by one worker, accumulating chunks. At most one per source.
    #[default]
    Building,
    /// Validated and complete. Searchable while the source points at it.
    Published,
    /// Abandoned. Never becomes active; its chunks are reclaimable.
    Failed,
    /// Explicitly retired by an operator; kept only for audit.
    Superseded,
}

impl GenerationState {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Building => "building",
            Self::Published => "published",
            Self::Failed => "failed",
            Self::Superseded => "superseded",
        }
    }
}

impl From<&str> for GenerationState {
    fn from(s: &str) -> Self {
        match s {
            "published" => Self::Published,
            "failed" => Self::Failed,
            "superseded" => Self::Superseded,
            _ => Self::Building,
        }
    }
}

// ============================================================================
// Entity
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "source_index_generations")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// The source this generation indexes. Immutable.
    pub source_id: Uuid,

    /// `building` | `published` | `failed` | `superseded`.
    pub state: String,

    /// Chunks the build intends to store. Set once the content is chunked;
    /// NULL until then, and compared against `stored_chunk_count` at publication.
    pub expected_chunk_count: Option<i32>,

    /// Chunks actually written under this generation.
    pub stored_chunk_count: i32,

    /// Provider/model/width/normalization identity of every vector here.
    pub embedding_fingerprint: String,

    /// Schema version, size unit, sizer and geometry of every chunk here.
    pub chunking_fingerprint: String,

    pub embedding_provider: String,
    pub embedding_model: String,
    pub embedding_dimension: i32,
    pub embedding_normalization: String,
    pub chunking_schema_version: i32,
    pub chunk_size_unit: String,
    pub chunk_sizer_identity: String,

    /// Stable reason a `failed` generation was abandoned.
    pub failure_reason: Option<String>,

    pub created_at: DateTimeWithTimeZone,
    pub published_at: Option<DateTimeWithTimeZone>,
    pub failed_at: Option<DateTimeWithTimeZone>,
}

impl Model {
    /// Parse the stored state into the typed enum.
    #[inline]
    #[must_use]
    pub fn state_enum(&self) -> GenerationState {
        GenerationState::from(self.state.as_str())
    }
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
