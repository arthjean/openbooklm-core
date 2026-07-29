//! Reference-server identity contract (US-013, EP-004).
//!
//! Proves the two acceptance criteria that do not need a database: a protected
//! route returns RFC 7807 401 when static-token mode receives a missing or
//! invalid token, and loopback single-user mode refuses a non-loopback bind.
//!
//! The route under test stands in for a core handler: what matters is that the
//! middleware rejects before the handler runs, and that a `Principal` carrying
//! the configured account reaches the handler when it does not.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use axum::{
    Extension, Router,
    body::Body,
    http::{Request, StatusCode, header},
    middleware,
    routing::get,
};
use openbooklm::core::identity::{
    IdentityError, IdentityMode, ReferenceIdentity, reference_auth_middleware,
};
use openbooklm::core::principal::Principal;
use tower::ServiceExt;
use uuid::Uuid;

const TOKEN: &str = "b8f2c1d90a7e4653b8f2c1d90a7e4653";

fn account() -> Uuid {
    Uuid::parse_str("00000000-0000-4000-8000-000000000001").expect("fixed test UUID")
}

/// Echoes the account the middleware injected, so a 200 proves the identity
/// reached the handler rather than merely passing the guard.
async fn protected(Extension(principal): Extension<Principal>) -> String {
    principal.account_id.to_string()
}

fn app(mode: IdentityMode) -> Router {
    Router::new()
        .route("/api/notebooks", get(protected))
        .layer(middleware::from_fn_with_state(
            ReferenceIdentity::new(mode, account()),
            reference_auth_middleware,
        ))
}

async fn call(app: Router, auth: Option<&str>) -> (StatusCode, String, Option<String>) {
    let mut builder = Request::builder().uri("/api/notebooks");
    if let Some(value) = auth {
        builder = builder.header(header::AUTHORIZATION, value);
    }
    let response = app
        .oneshot(builder.body(Body::empty()).expect("request"))
        .await
        .expect("response");

    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("body");

    (
        status,
        String::from_utf8_lossy(&bytes).into_owned(),
        content_type,
    )
}

// ============================================================================
// Static-token mode
// ============================================================================

#[tokio::test]
async fn static_token_mode_rejects_a_missing_token() {
    let (status, body, content_type) =
        call(app(IdentityMode::StaticToken(TOKEN.into())), None).await;

    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // Same content type the hosted Clerk adapter emits: both reject through
    // `AppError::Unauthorized`, so the 401 shape is identical by construction.
    assert_eq!(content_type.as_deref(), Some("application/json"));
    assert!(body.contains("\"status\":401"), "RFC 7807 body: {body}");
    assert!(body.contains("\"type\":"), "RFC 7807 body: {body}");
    assert!(
        !body.contains(TOKEN),
        "the response must not echo the token"
    );
}

#[tokio::test]
async fn static_token_mode_rejects_an_invalid_token() {
    for presented in [
        "Bearer wrong",
        "Bearer ",
        TOKEN,                                            // no scheme
        "Basic dXNlcjpwYXNzd29yZA==",                     // wrong scheme
        &format!("Bearer {}", &TOKEN[..TOKEN.len() - 1]), // truncated
        &format!("Bearer {TOKEN}x"),                      // extended
    ] {
        let (status, body, _) = call(
            app(IdentityMode::StaticToken(TOKEN.into())),
            Some(presented),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "accepted `{presented}`");
        assert!(!body.contains(TOKEN), "leaked the token for `{presented}`");
    }
}

#[tokio::test]
async fn static_token_mode_accepts_the_configured_token() {
    let (status, body, _) = call(
        app(IdentityMode::StaticToken(TOKEN.into())),
        Some(&format!("Bearer {TOKEN}")),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, account().to_string());
}

// ============================================================================
// Loopback single-user mode
// ============================================================================

#[tokio::test]
async fn loopback_mode_needs_no_credential() {
    let (status, body, _) = call(app(IdentityMode::Loopback), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, account().to_string());
}

#[test]
fn loopback_mode_refuses_a_non_loopback_bind() {
    let public = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
    assert_eq!(
        IdentityMode::resolve(public, None),
        Err(IdentityError::PublicBindWithoutToken { bind: public })
    );
    assert!(IdentityMode::resolve(IpAddr::V6(Ipv6Addr::UNSPECIFIED), None).is_err());
    assert!(IdentityMode::resolve(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5)), None).is_err());

    // …and is allowed once a token makes the listener authenticated.
    assert!(IdentityMode::resolve(public, Some(TOKEN)).is_ok());
}
