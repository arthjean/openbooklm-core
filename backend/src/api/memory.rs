use axum::{
    Json,
    extract::{Extension, Path, State},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::CoreState;
use crate::core::entitlements::Quota;
use crate::core::principal::Principal;
use crate::error::{AppError, MemoryError};
use crate::repositories::{MemoryRepository, NotebookRepository};

/// Reported to clients when the entitlement policy imposes no memory limit.
const UNLIMITED_MEMORIES: i32 = i32::MAX;

// ============================================================================
// Response types
// ============================================================================

/// A single memory in the API response.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct MemoryResponse {
    pub id: Uuid,
    pub notebook_id: Uuid,
    pub content: String,
    pub memory_type: String,
    pub metadata: serde_json::Value,
    pub salience: f32,
    pub created_at: String,
    pub updated_at: String,
}

impl From<crate::entities::notebook_memory::Model> for MemoryResponse {
    fn from(m: crate::entities::notebook_memory::Model) -> Self {
        Self {
            id: m.id,
            notebook_id: m.notebook_id,
            content: m.content,
            memory_type: m.memory_type,
            metadata: m.metadata,
            salience: m.salience,
            created_at: m.created_at.to_rfc3339(),
            updated_at: m.updated_at.to_rfc3339(),
        }
    }
}

/// Response for listing notebook memories with plan usage info.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct MemoriesListResponse {
    pub memories: Vec<MemoryResponse>,
    pub limit: i32,
    pub count: i64,
}

/// Request for updating a memory.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct UpdateMemoryRequest {
    pub content: Option<String>,
    pub salience: Option<f32>,
    pub metadata: Option<serde_json::Value>,
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /api/notebooks/:id/memories — list all memories for a notebook.
///
/// Returns memories sorted by salience DESC with plan limit and count.
/// Validates that the authenticated user owns the notebook.
#[utoipa::path(
    get,
    path = "/api/notebooks/{id}/memories",
    tag = "memories",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "The notebook's memories", body = MemoriesListResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn list_memories_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
) -> Result<Json<MemoriesListResponse>, AppError> {
    // Verify notebook ownership (returns 404 if not found/not owned)
    state
        .repos
        .notebooks
        .get_for_user(notebook_id, principal.account_id)
        .await?;

    let limit = state
        .entitlements
        .quota(&principal, Quota::MemoriesPerNotebook { notebook_id })
        .await?
        .unwrap_or(UNLIMITED_MEMORIES);

    // Fetch memories and count in parallel
    let (memories, count) =
        tokio::try_join!(state.repos.memory.list_for_notebook(notebook_id), async {
            let c = state.repos.memory.count_for_notebook(notebook_id).await?;
            Ok::<i64, AppError>(i64::try_from(c).unwrap_or(i64::MAX))
        },)?;

    let memories = memories.into_iter().map(Into::into).collect();

    Ok(Json(MemoriesListResponse {
        memories,
        limit,
        count,
    }))
}

/// DELETE /api/notebooks/:id/memories — delete all memories for a notebook.
///
/// Validates that the authenticated user owns the notebook.
#[utoipa::path(
    delete,
    path = "/api/notebooks/{id}/memories",
    tag = "memories",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "Deletion acknowledgement", body = serde_json::Value),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn delete_all_memories_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Verify notebook ownership
    state
        .repos
        .notebooks
        .get_for_user(notebook_id, principal.account_id)
        .await?;

    let deleted = state
        .repos
        .memory
        .delete_all_for_notebook(notebook_id)
        .await?;

    Ok(Json(serde_json::json!({
        "message": "All memories deleted",
        "deleted": deleted,
    })))
}

// ============================================================================
// Individual memory handlers
// ============================================================================

/// Fetch a memory by ID and verify the authenticated user owns its notebook.
/// Returns 404 if memory not found, 403 if the user doesn't own the notebook.
async fn verify_memory_access(
    state: &CoreState,
    principal: &Principal,
    memory_id: Uuid,
) -> Result<crate::entities::notebook_memory::Model, AppError> {
    let memory = state
        .repos
        .memory
        .get_by_id(memory_id)
        .await?
        .ok_or(MemoryError::NotFound {
            id: memory_id.to_string(),
        })?;

    // Verify the user owns the notebook this memory belongs to.
    // get_for_user returns NotebookError::NotFound if not owned — map to 403.
    state
        .repos
        .notebooks
        .get_for_user(memory.notebook_id, principal.account_id)
        .await
        .map_err(|_| MemoryError::Forbidden {
            id: memory_id.to_string(),
        })?;

    Ok(memory)
}

/// GET /api/memories/:id — get a single memory by ID.
#[utoipa::path(
    get,
    path = "/api/memories/{id}",
    tag = "memories",
    params(("id" = uuid::Uuid, Path, description = "Memory ID")),
    responses(
        (status = 200, description = "The memory", body = MemoryResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_memory_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(memory_id): Path<Uuid>,
) -> Result<Json<MemoryResponse>, AppError> {
    let memory = verify_memory_access(&state, &principal, memory_id).await?;
    Ok(Json(MemoryResponse::from(memory)))
}

/// PATCH /api/memories/:id — update a memory's content, salience, or metadata.
#[utoipa::path(
    patch,
    path = "/api/memories/{id}",
    tag = "memories",
    params(("id" = uuid::Uuid, Path, description = "Memory ID")),
    request_body = UpdateMemoryRequest,
    responses(
        (status = 200, description = "The updated memory", body = MemoryResponse),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn update_memory_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(memory_id): Path<Uuid>,
    Json(payload): Json<UpdateMemoryRequest>,
) -> Result<Json<MemoryResponse>, AppError> {
    verify_memory_access(&state, &principal, memory_id).await?;

    let updated = state
        .repos
        .memory
        .update(
            memory_id,
            payload.content,
            payload.salience,
            payload.metadata,
            None,
        )
        .await?;

    Ok(Json(MemoryResponse::from(updated)))
}

/// DELETE /api/memories/:id — delete a single memory.
#[utoipa::path(
    delete,
    path = "/api/memories/{id}",
    tag = "memories",
    params(("id" = uuid::Uuid, Path, description = "Memory ID")),
    responses(
        (status = 200, description = "Deletion acknowledgement", body = serde_json::Value),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn delete_memory_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(memory_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    verify_memory_access(&state, &principal, memory_id).await?;

    state.repos.memory.delete(memory_id).await?;

    Ok(Json(serde_json::json!({
        "message": "Memory deleted",
    })))
}
