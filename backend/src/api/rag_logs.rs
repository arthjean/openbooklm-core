//! RAG log API endpoints for feedback and metrics.

use axum::{
    Json,
    extract::{Extension, Path, Query, State},
};
use serde::Deserialize;
use uuid::Uuid;

use crate::core::CoreState;
use crate::core::principal::Principal;
use crate::error::AppError;
use crate::services::notebooks::get_notebook_for_user;
use crate::services::rag::rag_log::{self, AggregatedMetrics, UserFeedback};

// ============================================================================
// Request types
// ============================================================================

/// Request body for updating feedback.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct UpdateFeedbackRequest {
    pub feedback: UserFeedback,
}

/// Query parameters for metrics.
#[derive(Debug, Deserialize, utoipa::ToSchema, utoipa::IntoParams)]
pub struct MetricsQuery {
    /// Number of days to aggregate (default: 30).
    pub days: Option<i32>,
}

// ============================================================================
// Response types
// ============================================================================

/// Extended metrics with computed rates.
#[derive(Debug, serde::Serialize, utoipa::ToSchema)]
pub struct MetricsResponse {
    #[serde(flatten)]
    pub metrics: AggregatedMetrics,
    pub success_rate: f32,
    pub positive_feedback_rate: f32,
}

impl From<AggregatedMetrics> for MetricsResponse {
    fn from(m: AggregatedMetrics) -> Self {
        let success_rate = m.success_rate();
        let positive_feedback_rate = m.positive_feedback_rate();
        Self {
            metrics: m,
            success_rate,
            positive_feedback_rate,
        }
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// PATCH /api/rag-logs/:id/feedback - Update feedback on a RAG log entry.
#[utoipa::path(
    patch,
    path = "/api/rag-logs/{id}/feedback",
    tag = "rag-logs",
    params(("id" = uuid::Uuid, Path, description = "RAG log ID")),
    request_body = UpdateFeedbackRequest,
    responses(
        (status = 200, description = "Feedback acknowledgement", body = serde_json::Value),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn update_feedback_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(log_id): Path<Uuid>,
    Json(payload): Json<UpdateFeedbackRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    rag_log::update_feedback(
        state.repos.rag_logs.as_ref(),
        log_id,
        principal.account_id,
        payload.feedback,
    )
    .await?;

    Ok(Json(serde_json::json!({
        "success": true,
        "feedback": payload.feedback.as_str()
    })))
}

/// GET /api/notebooks/:id/metrics - Get aggregated RAG metrics for a notebook.
#[utoipa::path(
    get,
    path = "/api/notebooks/{id}/metrics",
    tag = "rag-logs",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID"), MetricsQuery),
    responses(
        (status = 200, description = "Aggregated retrieval metrics for the notebook", body = MetricsResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_notebook_metrics_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    Query(query): Query<MetricsQuery>,
) -> Result<Json<MetricsResponse>, AppError> {
    get_notebook_for_user(
        state.repos.notebooks.as_ref(),
        notebook_id,
        principal.account_id,
    )
    .await?;

    let days = query.days.unwrap_or(30).clamp(1, 365);
    let metrics =
        rag_log::get_notebook_metrics(state.repos.rag_logs.as_ref(), notebook_id, days).await?;

    Ok(Json(metrics.into()))
}

/// GET /api/metrics - Get aggregated RAG metrics for the authenticated user.
#[utoipa::path(
    get,
    path = "/api/metrics",
    tag = "rag-logs",
    params(MetricsQuery),
    responses(
        (status = 200, description = "Aggregated retrieval metrics for the account", body = MetricsResponse),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_user_metrics_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Query(query): Query<MetricsQuery>,
) -> Result<Json<MetricsResponse>, AppError> {
    let days = query.days.unwrap_or(30).clamp(1, 365);
    let metrics =
        rag_log::get_user_metrics(state.repos.rag_logs.as_ref(), principal.account_id, days)
            .await?;

    Ok(Json(metrics.into()))
}
