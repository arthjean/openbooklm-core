//! Security middleware module
//!
//! This module provides security-related middleware for the API:
//! - Request ID generation and propagation
//! - Security headers (X-Content-Type-Options, X-Frame-Options, etc.)
//! - Rate limiting
//! - CORS configuration helpers
//! - Graceful shutdown and task tracking (B4.4, B4.5)

mod rate_limit;
mod request_id;
mod security_headers;
mod shutdown;

pub use rate_limit::{RateLimiter, create_rate_limit_middleware};
pub use request_id::request_id_middleware;
pub use security_headers::security_headers_middleware;
pub use shutdown::{SpawnRejected, TaskAdmission, TaskTracker, shutdown_signal};

use axum::http::{HeaderName, HeaderValue, Method, header};
use tower_http::cors::CorsLayer;

use crate::core::config::SecurityConfig;

use request_id::X_REQUEST_ID;

/// Custom header for CSRF protection (XMLHttpRequest marker)
const X_REQUESTED_WITH: HeaderName = HeaderName::from_static("x-requested-with");

/// Build a CORS layer with configured allowed origins
///
/// This creates a restrictive CORS configuration that only allows:
/// - Origins specified in the security config
/// - Standard HTTP methods (GET, POST, PUT, PATCH, DELETE, OPTIONS)
/// - Required headers (Authorization, Content-Type, Accept, X-Request-ID)
/// - Credentials (cookies, authorization headers)
pub fn build_cors_layer(security: &SecurityConfig) -> CorsLayer {
    let origins: Vec<HeaderValue> = security
        .allowed_origins
        .iter()
        .filter_map(|origin| origin.parse().ok())
        .collect();

    tracing::info!(
        origins = ?security.allowed_origins,
        "Configuring CORS with allowed origins"
    );

    if origins.is_empty() {
        tracing::warn!("No valid CORS origins configured, requests may be blocked");
    }

    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            header::ACCEPT,
            header::CACHE_CONTROL,
            HeaderName::from_static("last-event-id"),
            X_REQUEST_ID.clone(),
            X_REQUESTED_WITH,
        ])
        .expose_headers([header::CONTENT_TYPE, X_REQUEST_ID.clone()])
        .allow_credentials(true)
}
