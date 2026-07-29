use axum::{
    Json,
    extract::{Extension, Path, State},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::api::common::success_response;
use crate::api::common::validate_title;
use crate::core::CoreState;
use crate::core::entitlements::{AuthorizationRequest, Operation};
use crate::core::principal::Principal;
use crate::entities::notebook;
use crate::error::AppError;
use crate::services::notebooks::{self, NotebookWithSourceCount};

/// Response for a notebook.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct NotebookResponse {
    pub id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub memory_enabled: bool,
    pub is_demo: bool,
    pub suggested_questions: Vec<String>,
    pub source_count: u64,
    pub created_at: String,
    pub updated_at: String,
}

impl From<NotebookWithSourceCount> for NotebookResponse {
    fn from(n: NotebookWithSourceCount) -> Self {
        Self {
            id: n.id,
            title: n.title,
            description: n.description,
            memory_enabled: n.memory_enabled,
            is_demo: n.is_demo,
            suggested_questions: serde_json::from_value(n.suggested_questions).unwrap_or_default(),
            source_count: u64::try_from(n.source_count.max(0)).unwrap_or(0),
            created_at: n.created_at.to_rfc3339(),
            updated_at: n.updated_at.to_rfc3339(),
        }
    }
}

impl From<(notebook::Model, u64)> for NotebookResponse {
    fn from((n, source_count): (notebook::Model, u64)) -> Self {
        Self {
            id: n.id,
            title: n.title,
            description: n.description,
            memory_enabled: n.memory_enabled,
            is_demo: n.is_demo,
            suggested_questions: serde_json::from_value(n.suggested_questions).unwrap_or_default(),
            source_count,
            created_at: n.created_at.to_rfc3339(),
            updated_at: n.updated_at.to_rfc3339(),
        }
    }
}

/// Response for listing notebooks.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct NotebooksListResponse {
    pub notebooks: Vec<NotebookResponse>,
}

/// GET /api/notebooks - List all notebooks for the current user.
///
/// Uses optimized single-query fetch with JOIN to avoid N+1 queries.
#[utoipa::path(
    get,
    path = "/api/notebooks",
    tag = "notebooks",
    responses(
        (status = 200, description = "The caller's notebooks", body = NotebooksListResponse),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn list_notebooks_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<NotebooksListResponse>, AppError> {
    let notebooks = notebooks::list_notebooks_with_source_counts(
        state.repos.notebooks.as_ref(),
        principal.account_id,
    )
    .await?
    .into_iter()
    .map(NotebookResponse::from)
    .collect();

    Ok(Json(NotebooksListResponse { notebooks }))
}

/// Request for creating a notebook.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct CreateNotebookRequest {
    pub title: String,
    pub description: Option<String>,
}

impl CreateNotebookRequest {
    fn validate(&self) -> Result<(), AppError> {
        validate_title(&self.title)
    }
}

/// POST /api/notebooks - Create a new notebook.
#[utoipa::path(
    post,
    path = "/api/notebooks",
    tag = "notebooks",
    request_body = CreateNotebookRequest,
    responses(
        (status = 200, description = "The created notebook", body = NotebookResponse),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 403, description = "Denied by the entitlement policy", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn create_notebook_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Json(payload): Json<CreateNotebookRequest>,
) -> Result<Json<NotebookResponse>, AppError> {
    payload.validate()?;

    // Denied operations create nothing: authorization precedes every write.
    state
        .entitlements
        .authorize(AuthorizationRequest::new(
            &principal,
            Operation::CreateNotebook,
            Uuid::new_v4(),
        ))
        .await?;

    let notebook = notebooks::create_notebook(
        state.repos.notebooks.as_ref(),
        principal.account_id,
        payload.title.trim().to_string(),
        payload.description,
    )
    .await?;

    Ok(Json((notebook, 0u64).into()))
}

/// GET /api/notebooks/:id - Get a single notebook.
#[utoipa::path(
    get,
    path = "/api/notebooks/{id}",
    tag = "notebooks",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "The notebook", body = NotebookResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_notebook_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
) -> Result<Json<NotebookResponse>, AppError> {
    let result = notebooks::get_notebook_with_source_count(
        state.repos.notebooks.as_ref(),
        notebook_id,
        principal.account_id,
    )
    .await?;

    Ok(Json(result.into()))
}

/// Request for updating a notebook.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct UpdateNotebookRequest {
    pub title: Option<String>,
    pub description: Option<Option<String>>,
    pub memory_enabled: Option<bool>,
}

impl UpdateNotebookRequest {
    fn validate(&self) -> Result<(), AppError> {
        if let Some(ref title) = self.title {
            validate_title(title)?;
        }
        Ok(())
    }
}

/// PATCH /api/notebooks/:id - Update a notebook.
#[utoipa::path(
    patch,
    path = "/api/notebooks/{id}",
    tag = "notebooks",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID")),
    request_body = UpdateNotebookRequest,
    responses(
        (status = 200, description = "The updated notebook", body = NotebookResponse),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn update_notebook_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    Json(payload): Json<UpdateNotebookRequest>,
) -> Result<Json<NotebookResponse>, AppError> {
    payload.validate()?;

    let notebook = notebooks::update_notebook(
        state.repos.notebooks.as_ref(),
        notebook_id,
        principal.account_id,
        payload.title.map(|t| t.trim().to_string()),
        payload.description,
        payload.memory_enabled,
    )
    .await?;

    let result = notebooks::get_notebook_with_source_count(
        state.repos.notebooks.as_ref(),
        notebook.id,
        principal.account_id,
    )
    .await?;

    Ok(Json(result.into()))
}

/// DELETE /api/notebooks/:id - Delete a notebook.
///
/// Demo notebooks (`is_demo = true`) cannot be deleted.
#[utoipa::path(
    delete,
    path = "/api/notebooks/{id}",
    tag = "notebooks",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "Deletion acknowledgement", body = serde_json::Value),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn delete_notebook_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    // Prevent deletion of the demo notebook
    let notebook = notebooks::get_notebook_for_user(
        state.repos.notebooks.as_ref(),
        notebook_id,
        principal.account_id,
    )
    .await?;
    if notebook.is_demo {
        return Err(AppError::Forbidden(
            "The demo notebook cannot be deleted.".into(),
        ));
    }

    notebooks::delete_notebook(
        state.repos.notebooks.as_ref(),
        notebook_id,
        principal.account_id,
    )
    .await?;

    Ok(success_response("Notebook deleted successfully"))
}
