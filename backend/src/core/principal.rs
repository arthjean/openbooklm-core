//! The authenticated account the core authorizes against (US-005, EP-002).
//!
//! Core handlers never see a Clerk subject, a claim name or a plan role. They
//! receive a [`Principal`] carrying the stable internal account UUID that every
//! core ownership check already uses.
//!
//! An identity adapter (Clerk in the hosted product, a static token or loopback
//! single-user mode in the reference server) resolves its own identity to an
//! `account_id` and inserts the `Principal` into the request extensions before
//! any protected core handler runs.

use std::any::Any;
use std::sync::Arc;

use uuid::Uuid;

/// Opaque per-request identity metadata owned by the identity adapter.
///
/// The core stores and clones it but never inspects it. The adapter that
/// created it downcasts it back to its own type. This is what keeps
/// adapter-specific facts (a Clerk session id, an owner flag) out of the public
/// `Principal` definition while still letting the adapter's own entitlement
/// policy read them without a second extension lookup in core handlers.
#[derive(Clone)]
pub struct SessionContext(Arc<dyn Any + Send + Sync>);

impl SessionContext {
    pub fn new<T: Any + Send + Sync>(value: T) -> Self {
        Self(Arc::new(value))
    }

    /// Recover the adapter-owned metadata. Returns `None` for any other type.
    #[must_use]
    pub fn downcast_ref<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

impl std::fmt::Debug for SessionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never render the payload: it belongs to the private adapter.
        f.write_str("SessionContext(<opaque>)")
    }
}

/// The authenticated account for the current request.
#[derive(Clone, Debug)]
pub struct Principal {
    /// Stable internal account identifier. Every core ownership check uses this.
    pub account_id: Uuid,
    /// Optional adapter-owned session metadata, opaque to the core.
    pub session: Option<SessionContext>,
}

impl Principal {
    /// Build a principal with no adapter metadata.
    #[must_use]
    pub fn new(account_id: Uuid) -> Self {
        Self {
            account_id,
            session: None,
        }
    }

    /// Attach adapter-owned session metadata.
    #[must_use]
    pub fn with_session<T: Any + Send + Sync>(mut self, session: T) -> Self {
        self.session = Some(SessionContext::new(session));
        self
    }

    /// Read back adapter-owned session metadata, if it is of type `T`.
    #[must_use]
    pub fn session_as<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.session.as_ref().and_then(SessionContext::downcast_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct AdapterSession {
        is_owner: bool,
    }

    #[test]
    fn principal_without_session_reads_none() {
        let p = Principal::new(Uuid::new_v4());
        assert!(p.session_as::<AdapterSession>().is_none());
    }

    #[test]
    fn adapter_recovers_its_own_session_type() {
        let p = Principal::new(Uuid::new_v4()).with_session(AdapterSession { is_owner: true });
        assert_eq!(
            p.session_as::<AdapterSession>(),
            Some(&AdapterSession { is_owner: true })
        );
    }

    #[test]
    fn foreign_type_does_not_downcast() {
        let p = Principal::new(Uuid::new_v4()).with_session(AdapterSession { is_owner: true });
        assert!(p.session_as::<String>().is_none());
    }

    #[test]
    fn debug_never_renders_the_session_payload() {
        let p = Principal::new(Uuid::nil()).with_session("clerk-secret-subject".to_string());
        let rendered = format!("{p:?}");
        assert!(!rendered.contains("clerk-secret-subject"));
        assert!(rendered.contains("<opaque>"));
    }
}
