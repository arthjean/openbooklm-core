//! Core account settings (US-011).
//!
//! Default provider and default model: the two preferences the core actually
//! implements. Onboarding progress used to share this response and this row;
//! it is a hosted-product concern and now lives in `saas::settings`, backed by
//! `saas_account_settings`.

use axum::{
    Json,
    extract::{Extension, State},
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::core::CoreState;
use crate::core::principal::Principal;
use crate::error::AppError;
use crate::repositories::AccountSettingsRepository;
use crate::validation::validate_provider;

// =============================================================================
// RESPONSE/REQUEST TYPES
// =============================================================================

/// Core account settings response.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct UserSettingsResponse {
    pub default_provider: String,
    pub default_model: String,
}

/// Request for updating default provider/model.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct UpdateDefaultsRequest {
    pub default_provider: Option<String>,
    pub default_model: Option<String>,
}

// =============================================================================
// HELPERS
// =============================================================================

/// Build the core settings response.
fn build_settings_response(
    settings: crate::entities::account_settings::Model,
) -> UserSettingsResponse {
    UserSettingsResponse {
        default_provider: settings.default_provider,
        default_model: settings.default_model,
    }
}

// =============================================================================
// HANDLERS
// =============================================================================

/// GET /api/settings - Get user settings.
#[utoipa::path(
    get,
    path = "/api/settings",
    tag = "settings",
    responses(
        (status = 200, description = "The account's core settings", body = UserSettingsResponse),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_settings_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
) -> Result<Json<UserSettingsResponse>, AppError> {
    let account_id = principal.account_id;
    let settings = state
        .repos
        .account_settings
        .get_or_create(account_id)
        .await?;

    Ok(Json(build_settings_response(settings)))
}

/// PATCH /api/settings - Update default provider and/or model.
#[utoipa::path(
    patch,
    path = "/api/settings",
    tag = "settings",
    request_body = UpdateDefaultsRequest,
    responses(
        (status = 200, description = "The updated settings", body = UserSettingsResponse),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn update_settings_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Json(payload): Json<UpdateDefaultsRequest>,
) -> Result<Json<UserSettingsResponse>, AppError> {
    let account_id = principal.account_id;

    // Validate provider if provided
    if let Some(ref provider) = payload.default_provider {
        validate_provider(provider)?;
    }

    let settings = state
        .repos
        .account_settings
        .update_defaults(account_id, payload.default_provider, payload.default_model)
        .await?;

    info!(
        account_id = %account_id,
        default_provider = %settings.default_provider,
        default_model = %settings.default_model,
        "Settings defaults updated"
    );

    Ok(Json(build_settings_response(settings)))
}
