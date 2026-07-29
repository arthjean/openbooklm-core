//! Suggested questions API handler.

use axum::{
    Json,
    extract::{Extension, Path, Query, State},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{core::CoreState, core::principal::Principal, error::AppError};

use super::common::verify_notebook_access;

#[derive(Deserialize, utoipa::ToSchema, utoipa::IntoParams)]
pub struct SuggestionsQuery {
    pub locale: Option<String>,
}

#[derive(Serialize, utoipa::ToSchema)]
pub struct SuggestionsResponse {
    pub suggestions: Vec<String>,
}

#[utoipa::path(
    get,
    path = "/api/notebooks/{id}/suggestions",
    tag = "suggestions",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID"), SuggestionsQuery),
    responses(
        (status = 200, description = "Suggested starter questions", body = SuggestionsResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_suggestions_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    Query(query): Query<SuggestionsQuery>,
) -> Result<Json<SuggestionsResponse>, AppError> {
    verify_notebook_access(state.repos.notebooks.as_ref(), &principal, notebook_id).await?;

    let locale = query.locale.as_deref().unwrap_or("en");

    let suggestions = crate::services::suggestions::generate_suggestions(
        state.clients.mistral.as_ref(),
        state.repos.chunks.as_ref(),
        notebook_id,
        locale,
    )
    .await?;

    Ok(Json(SuggestionsResponse { suggestions }))
}
