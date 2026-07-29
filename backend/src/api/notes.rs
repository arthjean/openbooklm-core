//! Notes API endpoints for user notes within notebooks.

use axum::{
    Json,
    extract::{Extension, Path, State},
};
use serde::{Deserialize, Serialize};
use tracing::info;
use uuid::Uuid;

use crate::api::common::{
    success_response, validate_content, validate_title, verify_notebook_access,
};
use crate::core::CoreState;
use crate::core::principal::Principal;
use crate::error::AppError;
use crate::repositories::NoteRepository;

// =============================================================================
// TYPES
// =============================================================================

/// Request for creating a note.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct CreateNoteRequest {
    pub title: String,
    pub content: String,
    pub original_message_id: Option<Uuid>,
}

/// Request for updating a note.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct UpdateNoteRequest {
    pub title: Option<String>,
    pub content: Option<String>,
}

/// Response for a note.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct NoteResponse {
    pub id: Uuid,
    pub notebook_id: Uuid,
    pub title: String,
    pub content: String,
    pub original_message_id: Option<Uuid>,
    pub created_at: String,
    pub updated_at: String,
}

impl From<crate::entities::note::Model> for NoteResponse {
    fn from(n: crate::entities::note::Model) -> Self {
        Self {
            id: n.id,
            notebook_id: n.notebook_id,
            title: n.title,
            content: n.content,
            original_message_id: n.original_message_id,
            created_at: n.created_at.to_rfc3339(),
            updated_at: n.updated_at.to_rfc3339(),
        }
    }
}

/// Response for listing notes.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct NotesListResponse {
    pub notes: Vec<NoteResponse>,
}

// =============================================================================
// HELPERS
// =============================================================================

/// Verify note access and return the owning account ID.
async fn verify_note_access(
    state: &CoreState,
    principal: &Principal,
    note_id: Uuid,
) -> Result<Uuid, AppError> {
    let account_id = principal.account_id;
    state.repos.notes.get_for_user(note_id, account_id).await?;
    Ok(account_id)
}

// =============================================================================
// HANDLERS
// =============================================================================

/// GET /api/notebooks/:notebook_id/notes - List all notes for a notebook.
#[utoipa::path(
    get,
    path = "/api/notebooks/{notebook_id}/notes",
    tag = "notes",
    params(("notebook_id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "The notebook's notes", body = NotesListResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn list_notes_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
) -> Result<Json<NotesListResponse>, AppError> {
    verify_notebook_access(state.repos.notebooks.as_ref(), &principal, notebook_id).await?;

    let notes = state
        .repos
        .notes
        .list_for_notebook(notebook_id)
        .await?
        .into_iter()
        .map(Into::into)
        .collect();

    Ok(Json(NotesListResponse { notes }))
}

/// POST /api/notebooks/:notebook_id/notes - Create a new note.
#[utoipa::path(
    post,
    path = "/api/notebooks/{notebook_id}/notes",
    tag = "notes",
    params(("notebook_id" = uuid::Uuid, Path, description = "Notebook ID")),
    request_body = CreateNoteRequest,
    responses(
        (status = 200, description = "The created note", body = NoteResponse),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn create_note_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    Json(payload): Json<CreateNoteRequest>,
) -> Result<Json<NoteResponse>, AppError> {
    verify_notebook_access(state.repos.notebooks.as_ref(), &principal, notebook_id).await?;

    validate_title(&payload.title)?;
    validate_content(&payload.content)?;

    let note = state
        .repos
        .notes
        .create(
            notebook_id,
            payload.title.trim().to_string(),
            payload.content,
            payload.original_message_id,
        )
        .await?;

    info!(note_id = %note.id, notebook_id = %notebook_id, user_id = %principal.account_id, "Note created");

    Ok(Json(note.into()))
}

/// GET /api/notes/:id - Get a single note.
#[utoipa::path(
    get,
    path = "/api/notes/{id}",
    tag = "notes",
    params(("id" = uuid::Uuid, Path, description = "Note ID")),
    responses(
        (status = 200, description = "The note", body = NoteResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_note_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(note_id): Path<Uuid>,
) -> Result<Json<NoteResponse>, AppError> {
    let note = state
        .repos
        .notes
        .get_for_user(note_id, principal.account_id)
        .await?;

    Ok(Json(note.into()))
}

/// PATCH /api/notes/:id - Update a note.
#[utoipa::path(
    patch,
    path = "/api/notes/{id}",
    tag = "notes",
    params(("id" = uuid::Uuid, Path, description = "Note ID")),
    request_body = UpdateNoteRequest,
    responses(
        (status = 200, description = "The updated note", body = NoteResponse),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn update_note_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(note_id): Path<Uuid>,
    Json(payload): Json<UpdateNoteRequest>,
) -> Result<Json<NoteResponse>, AppError> {
    let user_id = verify_note_access(&state, &principal, note_id).await?;

    // Validate fields if provided
    if let Some(ref title) = payload.title {
        validate_title(title)?;
    }
    if let Some(ref content) = payload.content {
        validate_content(content)?;
    }

    let note = state
        .repos
        .notes
        .update(
            note_id,
            user_id,
            payload.title.map(|t| t.trim().to_string()),
            payload.content,
        )
        .await?;

    info!(note_id = %note_id, user_id = %user_id, "Note updated");

    Ok(Json(note.into()))
}

/// DELETE /api/notes/:id - Delete a note.
#[utoipa::path(
    delete,
    path = "/api/notes/{id}",
    tag = "notes",
    params(("id" = uuid::Uuid, Path, description = "Note ID")),
    responses(
        (status = 200, description = "Deletion acknowledgement", body = serde_json::Value),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn delete_note_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(note_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    let user_id = verify_note_access(&state, &principal, note_id).await?;

    state.repos.notes.delete(note_id, user_id).await?;

    info!(note_id = %note_id, user_id = %user_id, "Note deleted");

    Ok(success_response("Note deleted successfully"))
}
