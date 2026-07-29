//! Reference identity for self-hosted deployments (US-013, EP-004).
//!
//! The hosted product resolves a Clerk subject to an account. A self-hosted
//! operator has no identity provider and does not need one: OpenbookLM is a
//! single-operator tool by default. This module provides the two adapters the
//! PRD requires, both producing the same public
//! [`Principal`](crate::core::Principal) the Clerk adapter produces.
//!
//! | Mode | Requirement | Use |
//! |---|---|---|
//! | [`IdentityMode::Loopback`] | listener bound to a loopback address | local single-user |
//! | [`IdentityMode::StaticToken`] | `Authorization: Bearer <token>` | LAN, reverse proxy, container |
//!
//! # Why loopback mode refuses a public bind
//!
//! Loopback mode authorises every request. That is safe only while the kernel
//! guarantees the peer is on the same host. [`IdentityMode::from_env`] therefore
//! refuses to produce a `Loopback` mode for a non-loopback bind address, and
//! [`resolve`] is the only constructor: there is no way to reach an
//! unauthenticated public listener by mistake.
//!
//! # Token comparison
//!
//! Tokens are compared in constant time. A short-circuiting comparison leaks the
//! shared secret one byte at a time to anyone who can measure the response.

use std::net::IpAddr;

use axum::{
    extract::{Request, State},
    http::header::AUTHORIZATION,
    middleware::Next,
    response::{IntoResponse, Response},
};
use subtle::ConstantTimeEq;
use uuid::Uuid;

use crate::core::principal::Principal;
use crate::error::AppError;

/// Environment variable naming the shared bearer token.
pub const TOKEN_ENV: &str = "OPENBOOKLM_AUTH_TOKEN";
/// Environment variable pinning the single account UUID.
pub const ACCOUNT_ENV: &str = "OPENBOOKLM_ACCOUNT_ID";

/// Shortest token accepted. Below this a shared secret is guessable.
pub const MIN_TOKEN_LEN: usize = 32;

/// How the reference server authenticates a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdentityMode {
    /// Every request is the single operator. Only legal on a loopback bind.
    Loopback,
    /// `Authorization: Bearer <token>` must match the configured token.
    StaticToken(String),
}

/// Why a reference identity could not be configured.
#[derive(Debug, PartialEq, Eq)]
pub enum IdentityError {
    /// The listener is reachable from the network and no token was set.
    PublicBindWithoutToken { bind: IpAddr },
    /// A token was set but is too short to be a credential.
    TokenTooShort { len: usize },
    /// `OPENBOOKLM_ACCOUNT_ID` was set but is not a UUID.
    InvalidAccountId,
}

impl std::fmt::Display for IdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PublicBindWithoutToken { bind } => write!(
                f,
                "refusing to bind {bind}: single-user mode authorises every request and is only \
                 safe on a loopback address. Set {TOKEN_ENV} to serve a non-loopback address, or \
                 bind 127.0.0.1."
            ),
            Self::TokenTooShort { len } => write!(
                f,
                "{TOKEN_ENV} is {len} characters; at least {MIN_TOKEN_LEN} are required. Generate \
                 one with `openssl rand -hex 32`."
            ),
            Self::InvalidAccountId => {
                write!(f, "{ACCOUNT_ENV} is not a valid UUID")
            }
        }
    }
}

impl std::error::Error for IdentityError {}

/// The account every reference-server request maps to.
///
/// Fixed by `OPENBOOKLM_ACCOUNT_ID` when set, so the same database can be
/// reattached to a restarted container without stranding its notebooks.
/// Otherwise a stable well-known UUID is used, which keeps a fresh install
/// reproducible: the same value on every machine, no bootstrap step.
pub const DEFAULT_ACCOUNT_ID: Uuid = Uuid::from_u128(0x0000_0000_0000_4000_8000_0000_0000_0001);

/// Read the account UUID from the environment.
///
/// # Errors
/// Returns [`IdentityError::InvalidAccountId`] when the variable is set but
/// unparseable. An absent variable is not an error.
pub fn account_id_from_env() -> Result<Uuid, IdentityError> {
    match std::env::var(ACCOUNT_ENV) {
        Err(_) => Ok(DEFAULT_ACCOUNT_ID),
        Ok(raw) if raw.trim().is_empty() => Ok(DEFAULT_ACCOUNT_ID),
        Ok(raw) => raw
            .trim()
            .parse::<Uuid>()
            .map_err(|_| IdentityError::InvalidAccountId),
    }
}

impl IdentityMode {
    /// Choose a mode for `bind` from the environment.
    ///
    /// # Errors
    /// See [`IdentityError`]. Every variant fails startup before a socket is
    /// bound.
    pub fn from_env(bind: IpAddr) -> Result<Self, IdentityError> {
        let token = std::env::var(TOKEN_ENV)
            .ok()
            .filter(|t| !t.trim().is_empty());
        Self::resolve(bind, token.as_deref())
    }

    /// Choose a mode for `bind` given an optional configured token.
    ///
    /// # Errors
    /// See [`IdentityError`].
    pub fn resolve(bind: IpAddr, token: Option<&str>) -> Result<Self, IdentityError> {
        match token {
            Some(token) => {
                let token = token.trim();
                if token.len() < MIN_TOKEN_LEN {
                    return Err(IdentityError::TokenTooShort { len: token.len() });
                }
                Ok(Self::StaticToken(token.to_owned()))
            }
            None if is_loopback(bind) => Ok(Self::Loopback),
            None => Err(IdentityError::PublicBindWithoutToken { bind }),
        }
    }

    /// Short label for startup logs. Never renders the token.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Loopback => "loopback single-user",
            Self::StaticToken(_) => "static bearer token",
        }
    }

    /// Whether `header` authorises the request.
    #[must_use]
    pub fn accepts(&self, header: Option<&str>) -> bool {
        match self {
            Self::Loopback => true,
            Self::StaticToken(expected) => header
                .and_then(|h| h.strip_prefix("Bearer "))
                .map(str::trim)
                .is_some_and(|presented| presented.as_bytes().ct_eq(expected.as_bytes()).into()),
        }
    }
}

/// `0.0.0.0` and `::` bind every interface, so they are not loopback even
/// though a loopback client can reach them.
fn is_loopback(addr: IpAddr) -> bool {
    addr.is_loopback()
}

/// State for [`reference_auth_middleware`].
#[derive(Clone, Debug)]
pub struct ReferenceIdentity {
    mode: IdentityMode,
    account_id: Uuid,
}

impl ReferenceIdentity {
    #[must_use]
    pub fn new(mode: IdentityMode, account_id: Uuid) -> Self {
        Self { mode, account_id }
    }

    #[must_use]
    pub fn mode(&self) -> &IdentityMode {
        &self.mode
    }

    #[must_use]
    pub fn account_id(&self) -> Uuid {
        self.account_id
    }
}

/// Injects a [`Principal`] for the configured single account, or rejects.
///
/// Mirrors the hosted adapter's contract exactly: RFC 7807 with HTTP 401 and no
/// hint about what the expected credential is.
pub async fn reference_auth_middleware(
    State(identity): State<ReferenceIdentity>,
    mut request: Request,
    next: Next,
) -> Response {
    let header = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    if !identity.mode.accepts(header) {
        return unauthorized();
    }

    request
        .extensions_mut()
        .insert(Principal::new(identity.account_id));
    next.run(request).await
}

/// The hosted adapter rejects through `AppError::Unauthorized`; so does this
/// one, so both editions return byte-identical RFC 7807 401 bodies.
fn unauthorized() -> Response {
    AppError::Unauthorized("Invalid or missing authentication token".into()).into_response()
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    use super::*;

    const TOKEN: &str = "0123456789abcdef0123456789abcdef";

    fn v4(a: u8, b: u8, c: u8, d: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(a, b, c, d))
    }

    #[test]
    fn loopback_without_token_is_allowed() {
        assert_eq!(
            IdentityMode::resolve(v4(127, 0, 0, 1), None),
            Ok(IdentityMode::Loopback)
        );
        assert_eq!(
            IdentityMode::resolve(IpAddr::V6(Ipv6Addr::LOCALHOST), None),
            Ok(IdentityMode::Loopback)
        );
    }

    #[test]
    fn wildcard_bind_without_token_is_refused() {
        // 0.0.0.0 is reachable from the network: single-user mode must not run.
        let err = IdentityMode::resolve(v4(0, 0, 0, 0), None).unwrap_err();
        assert_eq!(
            err,
            IdentityError::PublicBindWithoutToken {
                bind: v4(0, 0, 0, 0)
            }
        );
        assert!(IdentityMode::resolve(IpAddr::V6(Ipv6Addr::UNSPECIFIED), None).is_err());
    }

    #[test]
    fn routable_bind_without_token_is_refused() {
        assert!(IdentityMode::resolve(v4(192, 168, 1, 10), None).is_err());
    }

    #[test]
    fn public_bind_with_token_is_allowed() {
        assert_eq!(
            IdentityMode::resolve(v4(0, 0, 0, 0), Some(TOKEN)),
            Ok(IdentityMode::StaticToken(TOKEN.to_string()))
        );
    }

    #[test]
    fn short_token_is_refused() {
        let err = IdentityMode::resolve(v4(0, 0, 0, 0), Some("tooshort")).unwrap_err();
        assert_eq!(err, IdentityError::TokenTooShort { len: 8 });
    }

    #[test]
    fn loopback_accepts_any_header() {
        let mode = IdentityMode::Loopback;
        assert!(mode.accepts(None));
        assert!(mode.accepts(Some("Bearer whatever")));
    }

    #[test]
    fn static_token_accepts_only_the_exact_bearer() {
        let mode = IdentityMode::StaticToken(TOKEN.to_string());
        assert!(mode.accepts(Some(&format!("Bearer {TOKEN}"))));
        assert!(!mode.accepts(None));
        assert!(!mode.accepts(Some(TOKEN)), "raw token without the scheme");
        assert!(!mode.accepts(Some("Bearer ")), "empty credential");
        assert!(
            !mode.accepts(Some(&format!("Bearer {TOKEN}x"))),
            "prefix match"
        );
        assert!(
            !mode.accepts(Some(&format!("Bearer {}", &TOKEN[..TOKEN.len() - 1]))),
            "truncated token"
        );
        assert!(!mode.accepts(Some("Basic dXNlcjpwYXNz")), "wrong scheme");
    }

    #[test]
    fn error_messages_never_contain_the_token() {
        let err = IdentityError::TokenTooShort { len: 8 };
        assert!(!err.to_string().contains("tooshort"));
    }

    #[test]
    fn label_never_contains_the_token() {
        assert!(
            !IdentityMode::StaticToken(TOKEN.to_string())
                .label()
                .contains(TOKEN)
        );
    }
}
