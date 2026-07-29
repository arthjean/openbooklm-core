//! Notebooks service for notebook management.
//!
//! Delegates all database operations to the `NotebookRepository` trait.

use uuid::Uuid;

use crate::entities::notebook;
use crate::error::AppError;
use crate::repositories::NotebookRepository;

// Re-export the repository type so existing import paths still work.
pub use crate::repositories::NotebookWithSourceCount;

// ============================================================================
// Public API
// ============================================================================

/// Create a new notebook.
#[tracing::instrument(skip(repo, title, description), fields(%user_id))]
pub async fn create_notebook(
    repo: &dyn NotebookRepository,
    user_id: Uuid,
    title: String,
    description: Option<String>,
) -> Result<notebook::Model, AppError> {
    repo.create(user_id, title, description).await
}

/// Get a notebook by ID.
#[tracing::instrument(skip(repo), fields(%notebook_id))]
pub async fn get_notebook(
    repo: &dyn NotebookRepository,
    notebook_id: Uuid,
) -> Result<Option<notebook::Model>, AppError> {
    repo.get_by_id(notebook_id).await
}

/// Get a notebook by ID, ensuring it belongs to the user.
#[tracing::instrument(skip(repo), fields(%notebook_id, %user_id))]
pub async fn get_notebook_for_user(
    repo: &dyn NotebookRepository,
    notebook_id: Uuid,
    user_id: Uuid,
) -> Result<notebook::Model, AppError> {
    repo.get_for_user(notebook_id, user_id).await
}

/// List all notebooks for a user, ordered by last update.
#[tracing::instrument(skip(repo), fields(%user_id))]
pub async fn list_notebooks(
    repo: &dyn NotebookRepository,
    user_id: Uuid,
) -> Result<Vec<notebook::Model>, AppError> {
    repo.list_for_user(user_id).await
}

/// List all notebooks for a user with source counts in a single query.
#[tracing::instrument(skip(repo), fields(%user_id))]
pub async fn list_notebooks_with_source_counts(
    repo: &dyn NotebookRepository,
    user_id: Uuid,
) -> Result<Vec<NotebookWithSourceCount>, AppError> {
    repo.list_with_source_counts(user_id).await
}

/// Update a notebook's title, description, and/or memory_enabled.
#[tracing::instrument(skip(repo, title, description), fields(%notebook_id, %user_id))]
pub async fn update_notebook(
    repo: &dyn NotebookRepository,
    notebook_id: Uuid,
    user_id: Uuid,
    title: Option<String>,
    description: Option<Option<String>>,
    memory_enabled: Option<bool>,
) -> Result<notebook::Model, AppError> {
    repo.update(notebook_id, user_id, title, description, memory_enabled)
        .await
}

/// Delete a notebook (cascades to sources, chunks, and chat messages).
#[tracing::instrument(skip(repo), fields(%notebook_id, %user_id))]
pub async fn delete_notebook(
    repo: &dyn NotebookRepository,
    notebook_id: Uuid,
    user_id: Uuid,
) -> Result<(), AppError> {
    repo.delete(notebook_id, user_id).await
}

/// Count notebooks for a user.
#[tracing::instrument(skip(repo), fields(%user_id))]
pub async fn count_notebooks(
    repo: &dyn NotebookRepository,
    user_id: Uuid,
) -> Result<u64, AppError> {
    repo.count_for_user(user_id).await
}

/// Get notebook with source count (two queries).
#[tracing::instrument(skip(repo), fields(%notebook_id, %user_id))]
pub async fn get_notebook_with_source_count(
    repo: &dyn NotebookRepository,
    notebook_id: Uuid,
    user_id: Uuid,
) -> Result<(notebook::Model, u64), AppError> {
    repo.get_with_source_count(notebook_id, user_id).await
}
