//! Security headers middleware (OWASP recommendations)

use std::sync::LazyLock;

use axum::{
    extract::Request,
    http::{
        HeaderMap,
        header::{
            CONTENT_SECURITY_POLICY, HeaderName, HeaderValue, REFERRER_POLICY,
            X_CONTENT_TYPE_OPTIONS, X_FRAME_OPTIONS,
        },
    },
    middleware::Next,
    response::Response,
};

/// Pre-computed security headers (parsed once at startup).
static SECURITY_HEADERS: LazyLock<HeaderMap> = LazyLock::new(|| {
    let mut headers = HeaderMap::with_capacity(6);

    // Prevent MIME type sniffing
    headers.insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));

    // Prevent clickjacking
    headers.insert(X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));

    // Control referrer information
    headers.insert(
        REFERRER_POLICY,
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );

    // Basic CSP - adjust based on frontend needs
    headers.insert(
        CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("default-src 'self'; frame-ancestors 'none'"),
    );

    // Restrict powerful browser features
    headers.insert(
        HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("geolocation=(), microphone=(), camera=()"),
    );

    // Enforce HTTPS (aligned with frontend next.config.ts HSTS value)
    headers.insert(
        HeaderName::from_static("strict-transport-security"),
        HeaderValue::from_static("max-age=63072000; includeSubDomains; preload"),
    );

    headers
});

/// Adds security headers to all responses.
pub async fn security_headers_middleware(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    response.headers_mut().extend(SECURITY_HEADERS.clone());
    response
}
