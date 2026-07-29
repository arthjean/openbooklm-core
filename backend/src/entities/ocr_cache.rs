//! OCR cache entity — stores OCR results keyed by (content_hash, model)
//! to avoid redundant Mistral OCR API calls for identical PDF uploads.

use sea_orm::entity::prelude::*;

/// Cached OCR extraction result for a PDF document.
#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel)]
#[sea_orm(table_name = "ocr_cache")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// SHA-256 hex digest of the raw PDF bytes (always 64 chars).
    pub content_hash: String,

    /// OCR model used (e.g., "mistral-ocr-latest"). Cache is scoped by model
    /// so upgrading the model version naturally invalidates old entries.
    pub model: String,

    /// Merged OCR output text (Markdown).
    pub ocr_text: String,

    /// Number of pages processed by OCR.
    pub pages_processed: i32,

    pub created_at: DateTimeWithTimeZone,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
