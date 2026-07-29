//! Error handling module with domain-specific errors
//!
//! Uses thiserror for domain errors and RFC 7807 Problem Details for API responses.

use std::borrow::Cow;

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use thiserror::Error;

// ============================================================================
// RFC 7807 Problem Details
// ============================================================================

mod error_types {
    pub const VALIDATION: &str = "https://openbooklm.dev/errors/validation-error";
    pub const UNAUTHORIZED: &str = "https://openbooklm.dev/errors/unauthorized";
    pub const FORBIDDEN: &str = "https://openbooklm.dev/errors/forbidden";
    pub const NOT_FOUND: &str = "https://openbooklm.dev/errors/not-found";
    pub const RATE_LIMITED: &str = "https://openbooklm.dev/errors/rate-limited";
    pub const INTERNAL: &str = "https://openbooklm.dev/errors/internal-error";
    pub const PROVIDER: &str = "https://openbooklm.dev/errors/provider-error";
    pub const SERVICE_UNAVAILABLE: &str = "https://openbooklm.dev/errors/service-unavailable";
    pub const TIMEOUT: &str = "https://openbooklm.dev/errors/timeout";
    pub const PAYLOAD_TOO_LARGE: &str = "https://openbooklm.dev/errors/payload-too-large";
    pub const LIMIT_REACHED: &str = "https://openbooklm.dev/errors/limit-reached";
}

#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ProblemDetails {
    #[serde(rename = "type")]
    pub error_type: Cow<'static, str>,
    pub title: Cow<'static, str>,
    pub status: u16,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u32>,
}

/// Generates `ProblemDetails` factory methods that return `(StatusCode, ProblemDetails)`.
///
/// Supports two forms:
/// - **Simple**: `name => STATUS, TYPE_CONST, "Title";` generates `fn name(detail) -> (StatusCode, ProblemDetails)`
/// - **With retry**: `name(retry_after: u32) => STATUS, TYPE_CONST, "Title";` generates
///   `fn name(detail, retry_after: u32) -> (StatusCode, ProblemDetails)` with `retry_after` on the response
///
/// # Example
/// ```ignore
/// impl ProblemDetails {
///     problem_details! {
///         validation  => StatusCode::BAD_REQUEST, error_types::VALIDATION, "Validation Error";
///         rate_limited(retry_after: u32) => StatusCode::TOO_MANY_REQUESTS, error_types::RATE_LIMITED, "Rate Limited";
///     }
/// }
/// // Usage: ProblemDetails::validation("field is required")
/// // Usage: ProblemDetails::rate_limited("too many requests", 30)
/// ```
macro_rules! problem_details {
    ($($name:ident $(($extra:ident : $etype:ty))? => $status:expr, $type_const:expr, $title:literal);+ $(;)?) => {
        $(
            problem_details!(@single $name $(($extra : $etype))? => $status, $type_const, $title);
        )+
    };
    (@single $name:ident => $status:expr, $type_const:expr, $title:literal) => {
        /// Build the RFC 7807 response for this error class.
        ///
        /// Public because the error contract is part of the public core: adapters
        /// outside this crate must be able to produce a contract-compatible
        /// response without reimplementing the shape.
        pub fn $name(detail: impl Into<String>) -> (StatusCode, ProblemDetails) {
            ($status, ProblemDetails::new($status, $type_const, $title, detail))
        }
    };
    (@single $name:ident ($extra:ident : $etype:ty) => $status:expr, $type_const:expr, $title:literal) => {
        /// Build the RFC 7807 response for this error class, including `retry_after`.
        ///
        /// Public for the same reason as the simple form.
        pub fn $name(detail: impl Into<String>, $extra: $etype) -> (StatusCode, ProblemDetails) {
            ($status, ProblemDetails::new($status, $type_const, $title, detail).with_retry($extra))
        }
    };
}

impl ProblemDetails {
    fn new(
        status: StatusCode,
        error_type: &'static str,
        title: &'static str,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            error_type: Cow::Borrowed(error_type),
            title: Cow::Borrowed(title),
            status: status.as_u16(),
            detail: detail.into(),
            instance: None,
            retry_after: None,
        }
    }

    fn with_retry(mut self, seconds: u32) -> Self {
        self.retry_after = Some(seconds);
        self
    }

    problem_details! {
        validation        => StatusCode::BAD_REQUEST,            error_types::VALIDATION,         "Validation Error";
        unauthorized      => StatusCode::UNAUTHORIZED,           error_types::UNAUTHORIZED,       "Unauthorized";
        forbidden         => StatusCode::FORBIDDEN,              error_types::FORBIDDEN,          "Forbidden";
        not_found         => StatusCode::NOT_FOUND,              error_types::NOT_FOUND,          "Not Found";
        internal          => StatusCode::INTERNAL_SERVER_ERROR,  error_types::INTERNAL,           "Internal Server Error";
        provider_error    => StatusCode::BAD_GATEWAY,            error_types::PROVIDER,           "Provider Error";
        service_unavailable => StatusCode::SERVICE_UNAVAILABLE,  error_types::SERVICE_UNAVAILABLE, "Service Unavailable";
        timeout           => StatusCode::GATEWAY_TIMEOUT,        error_types::TIMEOUT,            "Request Timeout";
        limit_reached     => StatusCode::FORBIDDEN,              error_types::LIMIT_REACHED,      "Limit Reached";
        rate_limited(retry_after: u32) => StatusCode::TOO_MANY_REQUESTS, error_types::RATE_LIMITED, "Rate Limit Exceeded";
    }

    /// HTTP 413 Payload Too Large — used by the body size limit middleware.
    pub fn payload_too_large(detail: impl Into<String>) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            error_types::PAYLOAD_TOO_LARGE,
            "Payload Too Large",
            detail,
        )
    }

    /// HTTP 408 Request Timeout — used by the global request timeout middleware.
    pub fn request_timeout(detail: impl Into<String>) -> Self {
        Self::new(
            StatusCode::REQUEST_TIMEOUT,
            error_types::TIMEOUT,
            "Request Timeout",
            detail,
        )
    }

    /// HTTP 500 Internal Server Error — used for unhandled middleware errors.
    pub fn internal_error(detail: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            error_types::INTERNAL,
            "Internal Server Error",
            detail,
        )
    }
}

// ============================================================================
// Domain-Specific Errors
// ============================================================================

#[derive(Error, Debug)]
pub enum NotebookError {
    #[error("Notebook not found (id={id})")]
    NotFound { id: String },
    #[error("Notebook access denied (id={id})")]
    AccessDenied { id: String },
    #[error("Notebook limit reached: maximum {limit} notebooks allowed for your plan")]
    LimitReached { limit: i32 },
    #[error("Notebook creation failed: {reason}")]
    CreateFailed { reason: String },
    #[error("Notebook update failed: {reason} (id={id})")]
    UpdateFailed { id: String, reason: String },
    #[error("Notebook deletion failed: {reason} (id={id})")]
    DeleteFailed { id: String, reason: String },
    #[error("Notebook database error")]
    Database(#[from] sea_orm::DbErr),
}

#[derive(Error, Debug)]
pub enum SourceError {
    #[error("Source not found (id={id})")]
    NotFound { id: String },
    #[error("Source access denied (id={id})")]
    AccessDenied { id: String },
    #[error("Source limit reached: maximum {limit} sources per notebook")]
    LimitReached { limit: i32 },
    #[error("Source invalid type: {source_type}")]
    InvalidType { source_type: String },
    #[error("Source processing failed: {reason}")]
    ProcessingFailed { reason: String },
    #[error("PDF extraction failed: {reason}")]
    PdfExtractionFailed { reason: String },
    #[error("Web fetch failed for {url}: {reason}")]
    WebFetchFailed { url: String, reason: String },
    #[error("Source content is empty")]
    EmptyContent,
    #[error("Source content too large: {size} bytes (max: {max_size})")]
    ContentTooLarge { size: usize, max_size: usize },
    #[error("Source database error")]
    Database(#[from] sea_orm::DbErr),
}

#[derive(Error, Debug)]
pub enum RagError {
    #[error("RAG invalid chunk config: size={chunk_size}, overlap={overlap}")]
    InvalidChunkConfig { chunk_size: usize, overlap: usize },
    #[error("RAG chunking failed: {reason}")]
    ChunkingFailed { reason: String },
    #[error("RAG embedding failed: {reason}")]
    EmbeddingFailed { reason: String },
    #[error("RAG embedding timed out after {timeout_secs}s")]
    EmbeddingTimeout { timeout_secs: u64 },
    #[error("Embedding rate limited: {reason}")]
    EmbeddingRateLimited { reason: String, wait_secs: u32 },
    #[error("Vector store operation failed: {reason} (source_id={source_id})")]
    VectorStoreFailed { source_id: String, reason: String },
    #[error("No chunks found (source_id={source_id})")]
    NoChunksFound { source_id: String },
    #[error("RAG query is empty")]
    EmptyQuery,
    #[error("RAG query too long: {length} chars (max: {max_length})")]
    QueryTooLong { length: usize, max_length: usize },
    #[error("RAG database error")]
    Database(#[from] sea_orm::DbErr),
}

#[derive(Error, Debug)]
pub enum LlmError {
    #[error("LLM provider not configured: {provider}")]
    ProviderNotConfigured { provider: String },
    #[error("LLM API key missing (provider={provider})")]
    ApiKeyMissing { provider: String },
    #[error("LLM request failed for {provider}: {reason}")]
    RequestFailed { provider: String, reason: String },
    #[error("LLM response parse failed for {provider}: {reason}")]
    ResponseParseFailed { provider: String, reason: String },
    #[error("LLM rate limited for {provider}, retry after {retry_after}s")]
    RateLimited { provider: String, retry_after: u32 },
    #[error("LLM request timed out for {provider} after {timeout_secs}s")]
    Timeout { provider: String, timeout_secs: u64 },
    #[error("LLM stream error for {provider}: {reason}")]
    StreamError { provider: String, reason: String },
    #[error("LLM invalid model: {model}")]
    InvalidModel { model: String },
    #[error("LLM citation extraction failed: {reason}")]
    CitationExtractionFailed { reason: String },
}

#[derive(Error, Debug)]
pub enum BillingError {
    #[error("Subscription not found (user_id={user_id})")]
    SubscriptionNotFound { user_id: String },
    #[error("Billing invalid plan: {plan}")]
    InvalidPlan { plan: String },
    #[error("Stripe customer creation failed: {reason} (user_id={user_id})")]
    CustomerCreationFailed { user_id: String, reason: String },
    #[error("Stripe checkout failed: {reason} (user_id={user_id})")]
    CheckoutFailed { user_id: String, reason: String },
    #[error("Stripe portal creation failed: {reason} (user_id={user_id})")]
    PortalFailed { user_id: String, reason: String },
    #[error("Stripe webhook verification failed: {reason}")]
    WebhookVerificationFailed { reason: String },
    #[error("Stripe webhook processing failed: {reason} (event_type={event_type})")]
    WebhookProcessingFailed { event_type: String, reason: String },
    #[error("Billing invalid customer ID: {customer_id}")]
    InvalidCustomerId { customer_id: String },
    #[error("Billing limit exceeded: {resource} (limit={limit}, current={current})")]
    LimitExceeded {
        resource: String,
        limit: i32,
        current: i32,
    },
    #[error("Billing database error")]
    Database(#[from] sea_orm::DbErr),
}

#[derive(Error, Debug)]
pub enum ChatError {
    #[error("Chat message is empty")]
    EmptyMessage,
    #[error("Chat message too long: {length} chars (max={max_length})")]
    MessageTooLong { length: usize, max_length: usize },
    #[error("Chat daily limit exceeded: {limit} messages")]
    DailyLimitExceeded { limit: i32 },
    #[error("Chat message save failed: {reason} (notebook_id={notebook_id})")]
    SaveFailed { notebook_id: String, reason: String },
    #[error("Chat history retrieval failed: {reason} (notebook_id={notebook_id})")]
    HistoryRetrievalFailed { notebook_id: String, reason: String },
    #[error("Chat citations serialization failed: {reason} (notebook_id={notebook_id})")]
    CitationsSerializationFailed { notebook_id: String, reason: String },
    #[error("Chat database error")]
    Database(#[from] sea_orm::DbErr),
}

#[derive(Error, Debug)]
pub enum AuthError {
    #[error("Auth missing authorization header")]
    MissingAuthHeader,
    #[error("Auth invalid header format")]
    InvalidAuthHeader,
    #[error("Auth invalid JWT token: {reason}")]
    InvalidToken { reason: String },
    #[error("Auth token expired")]
    TokenExpired,
    #[error("Auth JWKS fetch failed: {reason}")]
    JwksFetchFailed { reason: String },
    #[error("Auth user not found (clerk_id={clerk_id})")]
    UserNotFound { clerk_id: String },
}

#[derive(Error, Debug)]
pub enum SettingsError {
    #[error("Settings not found (user_id={user_id})")]
    NotFound { user_id: String },
    #[error("Settings invalid provider: {provider}. Valid: mistral, anthropic, openai")]
    InvalidProvider { provider: String },
    #[error("Settings database error")]
    Database(#[from] sea_orm::DbErr),
}

#[derive(Error, Debug)]
pub enum NoteError {
    #[error("Note not found (id={id})")]
    NotFound { id: String },
    #[error("Note access denied (id={id})")]
    Forbidden { id: String },
    #[error("Note database error")]
    Database(#[from] sea_orm::DbErr),
}

#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory not found (id={id})")]
    NotFound { id: String },
    #[error("Memory access denied (id={id})")]
    Forbidden { id: String },
    #[error("Memory database error")]
    Database(#[from] sea_orm::DbErr),
}

// ============================================================================
// Main Application Error
// ============================================================================

#[derive(Error, Debug)]
pub enum AppError {
    // HTTP-level errors
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    #[error("Forbidden: {0}")]
    Forbidden(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Rate limited: {detail}")]
    RateLimited { detail: String, retry_after: u32 },
    #[error("Provider error: {0}")]
    ProviderError(String),
    #[error("Internal error: {0}")]
    Internal(String),
    // Database errors
    #[error("Database error: {0}")]
    Database(sea_orm::DbErr),
    // Domain errors
    #[error("{0}")]
    Notebook(NotebookError),
    #[error("{0}")]
    Source(SourceError),
    #[error("{0}")]
    Rag(RagError),
    #[error("{0}")]
    Llm(LlmError),
    #[error("{0}")]
    Billing(BillingError),
    #[error("{0}")]
    Chat(ChatError),
    #[error("{0}")]
    Auth(AuthError),
    #[error("{0}")]
    Settings(SettingsError),
    #[error("{0}")]
    Note(NoteError),
    #[error("{0}")]
    Memory(MemoryError),
}

// ============================================================================
// From implementations (macro-generated)
// ============================================================================

macro_rules! impl_from_domain_error {
    ($($variant:ident => $type:ty);+ $(;)?) => {
        $(
            impl From<$type> for AppError {
                fn from(err: $type) -> Self {
                    AppError::$variant(err)
                }
            }
        )+
    };
}

impl_from_domain_error! {
    Notebook => NotebookError;
    Source => SourceError;
    Rag => RagError;
    Llm => LlmError;
    Billing => BillingError;
    Chat => ChatError;
    Auth => AuthError;
    Settings => SettingsError;
    Note => NoteError;
    Memory => MemoryError;
}

impl From<sea_orm::DbErr> for AppError {
    fn from(err: sea_orm::DbErr) -> Self {
        Self::Database(err)
    }
}

// ============================================================================
// IntoResponse for Axum
// ============================================================================

#[allow(clippy::use_self)] // AppError:: is clearer than Self:: in match arms of IntoResponse
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        // Capture the original error detail for logging before the match consumes self.
        // This preserves unmasked error messages for 5xx responses (where the user-facing
        // detail is replaced with a generic message).
        let error_detail = self.to_string();

        let (status, problem) = match self {
            AppError::Validation(detail) => ProblemDetails::validation(detail),
            AppError::Unauthorized(detail) => ProblemDetails::unauthorized(detail),
            AppError::Forbidden(detail) => ProblemDetails::forbidden(detail),
            AppError::NotFound(detail) => ProblemDetails::not_found(detail),
            AppError::RateLimited {
                detail,
                retry_after,
            } => ProblemDetails::rate_limited(detail, retry_after),
            AppError::ProviderError(detail) => ProblemDetails::provider_error(detail),
            AppError::Internal(_) | AppError::Database(_) => {
                ProblemDetails::internal("An internal error occurred. Please try again later.")
            }
            AppError::Notebook(err) => err.into_response_parts(),
            AppError::Source(err) => err.into_response_parts(),
            AppError::Rag(err) => err.into_response_parts(),
            AppError::Llm(err) => err.into_response_parts(),
            AppError::Billing(err) => err.into_response_parts(),
            AppError::Chat(err) => err.into_response_parts(),
            AppError::Auth(err) => err.into_response_parts(),
            AppError::Settings(err) => err.into_response_parts(),
            AppError::Note(err) => err.into_response_parts(),
            AppError::Memory(err) => err.into_response_parts(),
        };

        // Log with severity based on HTTP status class:
        // - 4xx (client errors) → WARN: expected errors, not actionable alerts
        // - 5xx (server errors) → ERROR: genuine failures requiring investigation
        if status.is_server_error() {
            tracing::error!(
                status = status.as_u16(),
                error_type = %problem.error_type,
                "{error_detail}"
            );
        } else {
            tracing::warn!(
                status = status.as_u16(),
                error_type = %problem.error_type,
                "{error_detail}"
            );
        }

        (status, Json(problem)).into_response()
    }
}

// ============================================================================
// Domain error to HTTP response trait
// ============================================================================

trait IntoResponseParts {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails);
}

impl IntoResponseParts for NotebookError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::NotFound { .. } => ProblemDetails::not_found(self.to_string()),
            Self::AccessDenied { .. } => ProblemDetails::forbidden(self.to_string()),
            Self::LimitReached { .. } => ProblemDetails::limit_reached(self.to_string()),
            _ => ProblemDetails::internal(self.to_string()),
        }
    }
}

impl IntoResponseParts for SourceError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::NotFound { .. } => ProblemDetails::not_found(self.to_string()),
            Self::AccessDenied { .. } => ProblemDetails::forbidden(self.to_string()),
            Self::LimitReached { .. } => ProblemDetails::limit_reached(self.to_string()),
            Self::InvalidType { .. } | Self::EmptyContent | Self::ContentTooLarge { .. } => {
                ProblemDetails::validation(self.to_string())
            }
            Self::WebFetchFailed { .. } => ProblemDetails::provider_error(self.to_string()),
            _ => ProblemDetails::internal(self.to_string()),
        }
    }
}

impl IntoResponseParts for RagError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::EmptyQuery | Self::QueryTooLong { .. } | Self::InvalidChunkConfig { .. } => {
                ProblemDetails::validation(self.to_string())
            }
            Self::NoChunksFound { .. } => ProblemDetails::not_found(self.to_string()),
            Self::EmbeddingTimeout { .. } => ProblemDetails::timeout(self.to_string()),
            Self::EmbeddingRateLimited { wait_secs, .. } => {
                ProblemDetails::rate_limited(self.to_string(), *wait_secs)
            }
            _ => ProblemDetails::internal(self.to_string()),
        }
    }
}

impl IntoResponseParts for LlmError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::RateLimited { retry_after, .. } => {
                ProblemDetails::rate_limited(self.to_string(), *retry_after)
            }
            Self::InvalidModel { .. } => ProblemDetails::validation(self.to_string()),
            Self::ProviderNotConfigured { .. } | Self::ApiKeyMissing { .. } => {
                ProblemDetails::service_unavailable(self.to_string())
            }
            Self::Timeout { .. } => ProblemDetails::timeout(self.to_string()),
            _ => ProblemDetails::provider_error(self.to_string()),
        }
    }
}

impl IntoResponseParts for BillingError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::SubscriptionNotFound { .. } => ProblemDetails::not_found(self.to_string()),
            Self::InvalidPlan { .. } | Self::InvalidCustomerId { .. } => {
                ProblemDetails::validation(self.to_string())
            }
            Self::LimitExceeded { .. } => ProblemDetails::limit_reached(self.to_string()),
            Self::WebhookVerificationFailed { .. } => {
                ProblemDetails::unauthorized(self.to_string())
            }
            _ => ProblemDetails::provider_error(self.to_string()),
        }
    }
}

impl IntoResponseParts for ChatError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::EmptyMessage | Self::MessageTooLong { .. } => {
                ProblemDetails::validation(self.to_string())
            }
            Self::DailyLimitExceeded { .. } => {
                let now = chrono::Utc::now();
                let tomorrow_midnight = (now + chrono::Duration::days(1))
                    .date_naive()
                    .and_hms_opt(0, 0, 0)
                    .expect("valid midnight timestamp");
                let seconds_until_midnight =
                    (tomorrow_midnight - now.naive_utc()).num_seconds().max(1) as u32;
                ProblemDetails::rate_limited(self.to_string(), seconds_until_midnight)
            }
            _ => ProblemDetails::internal(self.to_string()),
        }
    }
}

impl IntoResponseParts for AuthError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::MissingAuthHeader
            | Self::InvalidAuthHeader
            | Self::InvalidToken { .. }
            | Self::TokenExpired => ProblemDetails::unauthorized(self.to_string()),
            Self::UserNotFound { .. } => ProblemDetails::not_found(self.to_string()),
            Self::JwksFetchFailed { .. } => ProblemDetails::service_unavailable(self.to_string()),
        }
    }
}

impl IntoResponseParts for SettingsError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::NotFound { .. } => ProblemDetails::not_found(self.to_string()),
            Self::InvalidProvider { .. } => ProblemDetails::validation(self.to_string()),
            Self::Database(_) => ProblemDetails::internal(self.to_string()),
        }
    }
}

impl IntoResponseParts for NoteError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::NotFound { .. } => ProblemDetails::not_found(self.to_string()),
            Self::Forbidden { .. } => ProblemDetails::forbidden(self.to_string()),
            Self::Database(_) => ProblemDetails::internal(self.to_string()),
        }
    }
}

impl IntoResponseParts for MemoryError {
    fn into_response_parts(self) -> (StatusCode, ProblemDetails) {
        match &self {
            Self::NotFound { .. } => ProblemDetails::not_found(self.to_string()),
            Self::Forbidden { .. } => ProblemDetails::forbidden(self.to_string()),
            Self::Database(_) => ProblemDetails::internal(self.to_string()),
        }
    }
}

// ============================================================================
// Type alias and extension traits
// ============================================================================

pub type AppResult<T> = Result<T, AppError>;

pub trait ResultExt<T> {
    fn log_err(self) -> Option<T>;
    fn warn_on_err(self) -> Option<T>;
}

impl<T, E: std::fmt::Debug> ResultExt<T> for Result<T, E> {
    #[track_caller]
    fn log_err(self) -> Option<T> {
        match self {
            Ok(v) => Some(v),
            Err(e) => {
                let loc = std::panic::Location::caller();
                tracing::error!(error = ?e, file = loc.file(), line = loc.line(), "Operation failed");
                None
            }
        }
    }

    #[track_caller]
    fn warn_on_err(self) -> Option<T> {
        match self {
            Ok(v) => Some(v),
            Err(e) => {
                let loc = std::panic::Location::caller();
                tracing::warn!(error = ?e, file = loc.file(), line = loc.line(), "Operation failed (non-critical)");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    #[test]
    fn payload_too_large_returns_413_with_problem_details() {
        let problem = ProblemDetails::payload_too_large("body too big");
        assert_eq!(problem.status, 413);
        assert_eq!(problem.title, "Payload Too Large");
        assert_eq!(problem.detail, "body too big");
        assert_eq!(
            problem.error_type.as_ref(),
            "https://openbooklm.dev/errors/payload-too-large"
        );
    }

    #[test]
    fn request_timeout_returns_408_with_problem_details() {
        let problem = ProblemDetails::request_timeout("timed out");
        assert_eq!(problem.status, 408);
        assert_eq!(problem.title, "Request Timeout");
        assert_eq!(problem.detail, "timed out");
    }

    #[test]
    fn internal_error_returns_500_with_problem_details() {
        let problem = ProblemDetails::internal_error("something broke");
        assert_eq!(problem.status, 500);
        assert_eq!(problem.title, "Internal Server Error");
        assert_eq!(problem.detail, "something broke");
    }

    // ========================================================================
    // US-004: Error severity mapping tests
    //
    // These tests verify that every AppError variant produces the correct HTTP
    // status class. Since logging severity is derived from the status class
    // (4xx → WARN, 5xx → ERROR), correct status codes guarantee correct severity.
    // ========================================================================

    /// Helper: convert an AppError to an HTTP response and return the status code.
    fn status_of(err: AppError) -> StatusCode {
        err.into_response().status()
    }

    #[test]
    fn direct_app_errors_4xx_produce_client_status() {
        assert!(status_of(AppError::Validation("t".into())).is_client_error());
        assert!(status_of(AppError::Unauthorized("t".into())).is_client_error());
        assert!(status_of(AppError::Forbidden("t".into())).is_client_error());
        assert!(status_of(AppError::NotFound("t".into())).is_client_error());
        assert!(
            status_of(AppError::RateLimited {
                detail: "t".into(),
                retry_after: 30,
            })
            .is_client_error()
        );
    }

    #[test]
    fn direct_app_errors_5xx_produce_server_status() {
        assert!(status_of(AppError::ProviderError("t".into())).is_server_error());
        assert!(status_of(AppError::Internal("t".into())).is_server_error());
        assert!(
            status_of(AppError::Database(sea_orm::DbErr::Custom("t".into()))).is_server_error()
        );
    }

    #[test]
    fn notebook_error_severity_mapping() {
        // 4xx variants (logged at WARN)
        assert!(status_of(NotebookError::NotFound { id: "1".into() }.into()).is_client_error());
        assert!(status_of(NotebookError::AccessDenied { id: "1".into() }.into()).is_client_error());
        assert!(status_of(NotebookError::LimitReached { limit: 3 }.into()).is_client_error());

        // 5xx variants (logged at ERROR)
        assert!(
            status_of(NotebookError::CreateFailed { reason: "t".into() }.into()).is_server_error()
        );
        assert!(
            status_of(
                NotebookError::UpdateFailed {
                    id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                NotebookError::DeleteFailed {
                    id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(NotebookError::Database(sea_orm::DbErr::Custom("t".into())).into())
                .is_server_error()
        );
    }

    #[test]
    fn source_error_severity_mapping() {
        // 4xx variants
        assert!(status_of(SourceError::NotFound { id: "1".into() }.into()).is_client_error());
        assert!(status_of(SourceError::AccessDenied { id: "1".into() }.into()).is_client_error());
        assert!(status_of(SourceError::LimitReached { limit: 10 }.into()).is_client_error());
        assert!(
            status_of(
                SourceError::InvalidType {
                    source_type: "pdff".into()
                }
                .into()
            )
            .is_client_error()
        );
        assert!(status_of(SourceError::EmptyContent.into()).is_client_error());
        assert!(
            status_of(
                SourceError::ContentTooLarge {
                    size: 100,
                    max_size: 50
                }
                .into()
            )
            .is_client_error()
        );

        // 5xx variants
        assert!(
            status_of(
                SourceError::WebFetchFailed {
                    url: "http://x".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(SourceError::ProcessingFailed { reason: "t".into() }.into())
                .is_server_error()
        );
        assert!(
            status_of(SourceError::PdfExtractionFailed { reason: "t".into() }.into())
                .is_server_error()
        );
        assert!(
            status_of(SourceError::Database(sea_orm::DbErr::Custom("t".into())).into())
                .is_server_error()
        );
    }

    #[test]
    fn rag_error_severity_mapping() {
        // 4xx variants
        assert!(
            status_of(
                RagError::InvalidChunkConfig {
                    chunk_size: 0,
                    overlap: 0
                }
                .into()
            )
            .is_client_error()
        );
        assert!(status_of(RagError::EmptyQuery.into()).is_client_error());
        assert!(
            status_of(
                RagError::QueryTooLong {
                    length: 100,
                    max_length: 50
                }
                .into()
            )
            .is_client_error()
        );
        assert!(
            status_of(
                RagError::NoChunksFound {
                    source_id: "1".into()
                }
                .into()
            )
            .is_client_error()
        );

        // 5xx variants
        assert!(
            status_of(RagError::ChunkingFailed { reason: "t".into() }.into()).is_server_error()
        );
        assert!(
            status_of(RagError::EmbeddingFailed { reason: "t".into() }.into()).is_server_error()
        );
        assert!(
            status_of(RagError::EmbeddingTimeout { timeout_secs: 30 }.into()).is_server_error()
        );
        assert!(
            status_of(
                RagError::EmbeddingRateLimited {
                    reason: "test".into(),
                    wait_secs: 60
                }
                .into()
            )
            .is_client_error()
        );
        assert!(
            status_of(
                RagError::VectorStoreFailed {
                    source_id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(RagError::Database(sea_orm::DbErr::Custom("t".into())).into())
                .is_server_error()
        );
    }

    #[test]
    fn llm_error_severity_mapping() {
        // 4xx variants
        assert!(status_of(LlmError::InvalidModel { model: "x".into() }.into()).is_client_error());
        assert!(
            status_of(
                LlmError::RateLimited {
                    provider: "x".into(),
                    retry_after: 10
                }
                .into()
            )
            .is_client_error()
        );

        // 5xx variants
        assert!(
            status_of(
                LlmError::ProviderNotConfigured {
                    provider: "x".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                LlmError::ApiKeyMissing {
                    provider: "x".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                LlmError::RequestFailed {
                    provider: "x".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                LlmError::ResponseParseFailed {
                    provider: "x".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                LlmError::Timeout {
                    provider: "x".into(),
                    timeout_secs: 30
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                LlmError::StreamError {
                    provider: "x".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(LlmError::CitationExtractionFailed { reason: "t".into() }.into())
                .is_server_error()
        );
    }

    #[test]
    fn billing_error_severity_mapping() {
        // 4xx variants
        assert!(
            status_of(
                BillingError::SubscriptionNotFound {
                    user_id: "1".into()
                }
                .into()
            )
            .is_client_error()
        );
        assert!(status_of(BillingError::InvalidPlan { plan: "x".into() }.into()).is_client_error());
        assert!(
            status_of(
                BillingError::InvalidCustomerId {
                    customer_id: "x".into()
                }
                .into()
            )
            .is_client_error()
        );
        assert!(
            status_of(
                BillingError::LimitExceeded {
                    resource: "x".into(),
                    limit: 10,
                    current: 11
                }
                .into()
            )
            .is_client_error()
        );
        assert!(
            status_of(BillingError::WebhookVerificationFailed { reason: "t".into() }.into())
                .is_client_error()
        );

        // 5xx variants
        assert!(
            status_of(
                BillingError::CustomerCreationFailed {
                    user_id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                BillingError::CheckoutFailed {
                    user_id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                BillingError::PortalFailed {
                    user_id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                BillingError::WebhookProcessingFailed {
                    event_type: "x".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(BillingError::Database(sea_orm::DbErr::Custom("t".into())).into())
                .is_server_error()
        );
    }

    #[test]
    fn chat_error_severity_mapping() {
        // 4xx variants
        assert!(status_of(ChatError::EmptyMessage.into()).is_client_error());
        assert!(
            status_of(
                ChatError::MessageTooLong {
                    length: 100,
                    max_length: 50
                }
                .into()
            )
            .is_client_error()
        );
        assert!(status_of(ChatError::DailyLimitExceeded { limit: 30 }.into()).is_client_error());

        // 5xx variants
        assert!(
            status_of(
                ChatError::SaveFailed {
                    notebook_id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                ChatError::HistoryRetrievalFailed {
                    notebook_id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(
                ChatError::CitationsSerializationFailed {
                    notebook_id: "1".into(),
                    reason: "t".into()
                }
                .into()
            )
            .is_server_error()
        );
        assert!(
            status_of(ChatError::Database(sea_orm::DbErr::Custom("t".into())).into())
                .is_server_error()
        );
    }

    #[test]
    fn auth_error_severity_mapping() {
        // 4xx variants
        assert!(status_of(AuthError::MissingAuthHeader.into()).is_client_error());
        assert!(status_of(AuthError::InvalidAuthHeader.into()).is_client_error());
        assert!(status_of(AuthError::InvalidToken { reason: "t".into() }.into()).is_client_error());
        assert!(status_of(AuthError::TokenExpired.into()).is_client_error());
        assert!(
            status_of(
                AuthError::UserNotFound {
                    clerk_id: "x".into()
                }
                .into()
            )
            .is_client_error()
        );

        // 5xx variants
        assert!(
            status_of(AuthError::JwksFetchFailed { reason: "t".into() }.into()).is_server_error()
        );
    }

    #[test]
    fn settings_error_severity_mapping() {
        // 4xx variants
        assert!(
            status_of(
                SettingsError::NotFound {
                    user_id: "1".into()
                }
                .into()
            )
            .is_client_error()
        );
        assert!(
            status_of(
                SettingsError::InvalidProvider {
                    provider: "x".into()
                }
                .into()
            )
            .is_client_error()
        );

        // 5xx variants
        assert!(
            status_of(SettingsError::Database(sea_orm::DbErr::Custom("t".into())).into())
                .is_server_error()
        );
    }

    #[test]
    fn note_error_severity_mapping() {
        // 4xx variants
        assert!(status_of(NoteError::NotFound { id: "1".into() }.into()).is_client_error());
        assert!(status_of(NoteError::Forbidden { id: "1".into() }.into()).is_client_error());

        // 5xx variants
        assert!(
            status_of(NoteError::Database(sea_orm::DbErr::Custom("t".into())).into())
                .is_server_error()
        );
    }

    #[test]
    fn daily_limit_exceeded_retry_after_is_positive() {
        let err = ChatError::DailyLimitExceeded { limit: 10 };
        let (status, details) = IntoResponseParts::into_response_parts(err);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert!(
            details.retry_after.unwrap_or(0) > 0,
            "retry_after must be > 0 for daily limits"
        );
    }
}
