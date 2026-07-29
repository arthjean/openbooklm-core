//! Shared helpers for API handlers.
//!
//! Ownership checks take the public [`Principal`] (US-005): the account UUID is
//! already resolved by the identity adapter, so a core handler performs no
//! identity lookup of its own.
//!
//! Validation functions are centralized in [`crate::validation`] and re-exported here
//! for backward compatibility.

use axum::Json;
use uuid::Uuid;

use crate::core::principal::Principal;
use crate::error::AppError;
use crate::repositories::{NotebookRepository, SourceRepository};
use crate::services::notebooks::get_notebook_for_user;
use crate::services::sources::get_source_for_user;

// Re-export validation functions for backward compatibility with existing call sites.
pub use crate::validation::{
    validate_content, validate_description, validate_string, validate_system_prompt,
    validate_title, validate_url_for_ssrf,
};

// ============================================================================
// Response helpers
// ============================================================================

/// Build a standard JSON success response.
pub fn success_response(message: &str) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "success": true,
        "message": message
    }))
}

// ============================================================================
// Ownership helpers
// ============================================================================

/// Verify the principal owns the notebook.
///
/// Returns the repository's not-found error when the notebook belongs to
/// another account, preserving the existing information-hiding behaviour.
pub async fn verify_notebook_access(
    notebook_repo: &dyn NotebookRepository,
    principal: &Principal,
    notebook_id: Uuid,
) -> Result<(), AppError> {
    get_notebook_for_user(notebook_repo, notebook_id, principal.account_id).await?;
    Ok(())
}

/// Verify the principal owns the source and return it.
pub async fn verify_source_access(
    source_repo: &dyn SourceRepository,
    principal: &Principal,
    source_id: Uuid,
) -> Result<crate::entities::source::Model, AppError> {
    get_source_for_user(source_repo, source_id, principal.account_id).await
}
