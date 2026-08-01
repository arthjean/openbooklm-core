//! Sources service for document management.
//!
//! Delegates all database operations to the `SourceRepository` trait.

use uuid::Uuid;

use crate::entities::source::{self, SourceStatus};
use crate::error::AppError;
use crate::repositories::SourceRepository;
use crate::types::SourceType;

// ============================================================================
// Public API
// ============================================================================

/// Create a new source (initial creation, pending processing).
#[tracing::instrument(skip(repo, content, metadata), fields(%notebook_id))]
pub async fn create_source(
    repo: &dyn SourceRepository,
    notebook_id: Uuid,
    title: String,
    source_type: SourceType,
    content: String,
    metadata: Option<serde_json::Value>,
) -> Result<source::Model, AppError> {
    repo.create(notebook_id, title, source_type, content, metadata)
        .await
}

/// Get a source by ID.
#[tracing::instrument(skip(repo), fields(%source_id))]
pub async fn get_source(
    repo: &dyn SourceRepository,
    source_id: Uuid,
) -> Result<Option<source::Model>, AppError> {
    repo.get_by_id(source_id).await
}

/// Get a source by ID with notebook ownership verification.
#[tracing::instrument(skip(repo), fields(%source_id, %user_id))]
pub async fn get_source_for_user(
    repo: &dyn SourceRepository,
    source_id: Uuid,
    user_id: Uuid,
) -> Result<source::Model, AppError> {
    repo.get_for_user(source_id, user_id).await
}

/// List all sources for a notebook, ordered by creation date (newest first).
#[tracing::instrument(skip(repo), fields(%notebook_id))]
pub async fn list_sources(
    repo: &dyn SourceRepository,
    notebook_id: Uuid,
) -> Result<Vec<source::Model>, AppError> {
    repo.list_for_notebook(notebook_id).await
}

/// Delete a source (cascades to chunks).
#[tracing::instrument(skip(repo), fields(%source_id, %user_id))]
pub async fn delete_source(
    repo: &dyn SourceRepository,
    source_id: Uuid,
    user_id: Uuid,
) -> Result<(), AppError> {
    repo.delete(source_id, user_id).await
}

/// Update source status and optional error message.
#[tracing::instrument(skip(repo), fields(%source_id))]
pub async fn update_source_status(
    repo: &dyn SourceRepository,
    source_id: Uuid,
    status: SourceStatus,
    error_message: Option<String>,
) -> Result<source::Model, AppError> {
    repo.update_status(source_id, status, error_message).await
}

/// Set a source's chunk count directly.
///
/// Ingestion does not use this: since EP-002 the count is written by generation
/// publication, inside the transaction that moves the active pointer. See
/// [`SourceRepository::update_chunk_count`](crate::repositories::SourceRepository::update_chunk_count).
#[tracing::instrument(skip(repo), fields(%source_id))]
pub async fn update_source_chunk_count(
    repo: &dyn SourceRepository,
    source_id: Uuid,
    chunk_count: i32,
) -> Result<source::Model, AppError> {
    repo.update_chunk_count(source_id, chunk_count).await
}

/// Count sources in a notebook.
#[tracing::instrument(skip(repo), fields(%notebook_id))]
pub async fn count_sources(
    repo: &dyn SourceRepository,
    notebook_id: Uuid,
) -> Result<u64, AppError> {
    repo.count_for_notebook(notebook_id).await
}

/// Get all sources with a specific status.
#[tracing::instrument(skip(repo))]
pub async fn get_sources_by_status(
    repo: &dyn SourceRepository,
    status: SourceStatus,
) -> Result<Vec<source::Model>, AppError> {
    repo.get_by_status(status).await
}
