//! Entitlement policy for bounded core operations (US-006, US-007, EP-002).
//!
//! Core handlers never decide whether an account is allowed to do something.
//! They describe the operation, ask the injected [`EntitlementPolicy`], and act
//! on the answer. Two adapters implement it:
//!
//! - the public [`UnrestrictedPolicy`], which authorizes every structurally
//!   valid operation and requires no usage or subscription storage;
//! - a private SaaS adapter, which reproduces the hosted plan limits.
//!
//! Operations that consume metered resources (chat messages, OCR pages) return
//! an opaque [`Permit`]. The core performs the real work, then records what was
//! actually consumed through the same permit. Nothing is charged before the
//! provider has done paid work, and a permit records at most once.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use uuid::Uuid;

use crate::core::principal::Principal;
use crate::error::AppError;
use crate::types::SourceType;

// ============================================================================
// Operations
// ============================================================================

/// A bounded operation the core is about to perform.
///
/// Carries account and resource facts only: no plan name, no price identifier
/// and no private repository model.
#[derive(Debug, Clone)]
pub enum Operation {
    /// Create one notebook for the account.
    CreateNotebook,
    /// Add one source of the given type to a notebook.
    CreateSource {
        notebook_id: Uuid,
        source_type: SourceType,
    },
    /// Store one extracted memory in a notebook.
    CreateMemory { notebook_id: Uuid },
    /// Use a specific provider and model for generation.
    SelectModel { provider: String, model: String },
    /// Start one chat exchange in a notebook. Metered: consumption is recorded
    /// only once the provider has emitted real output.
    SendChatMessage { notebook_id: Uuid },
    /// Run OCR over at most `requested_pages` pages. Metered: consumption is
    /// recorded with the number of pages the provider actually processed.
    ProcessOcrPages { requested_pages: i32 },
}

/// A quota the core needs to report to the client (not an authorization).
#[derive(Debug, Clone)]
pub enum Quota {
    /// Maximum number of memories a notebook may store.
    MemoriesPerNotebook { notebook_id: Uuid },
}

/// Authorization input: who, what, and a stable identifier for the work.
pub struct AuthorizationRequest<'a> {
    pub principal: &'a Principal,
    pub operation: Operation,
    /// Identifier for this unit of work, carried on the permit and reported
    /// when recording fails so a private reconciliation can find it. Prefer a
    /// stable domain id (a source id) over a fresh UUID where one exists.
    pub operation_id: Uuid,
}

impl<'a> AuthorizationRequest<'a> {
    #[must_use]
    pub fn new(principal: &'a Principal, operation: Operation, operation_id: Uuid) -> Self {
        Self {
            principal,
            operation,
            operation_id,
        }
    }

    #[must_use]
    pub fn account_id(&self) -> Uuid {
        self.principal.account_id
    }
}

// ============================================================================
// Permit
// ============================================================================

/// Proof that an operation was authorized, plus the handle used to record the
/// work actually performed.
///
/// The payload is opaque: the policy that issued the permit is the only code
/// that reads it. Core code reads `max_units` and calls
/// [`EntitlementPolicy::record`].
#[derive(Clone)]
pub struct Permit {
    account_id: Uuid,
    operation_id: Uuid,
    max_units: Option<i32>,
    payload: Option<Arc<dyn Any + Send + Sync>>,
    recorded: Arc<AtomicBool>,
}

impl Permit {
    /// A permit that authorizes an unbounded amount of work and records nothing.
    #[must_use]
    pub fn unrestricted(account_id: Uuid, operation_id: Uuid) -> Self {
        Self {
            account_id,
            operation_id,
            max_units: None,
            payload: None,
            recorded: Arc::new(AtomicBool::new(false)),
        }
    }

    /// A permit bounded to `max_units` units of work.
    #[must_use]
    pub fn bounded(account_id: Uuid, operation_id: Uuid, max_units: i32) -> Self {
        Self {
            max_units: Some(max_units),
            ..Self::unrestricted(account_id, operation_id)
        }
    }

    /// Attach policy-owned data needed at recording time.
    #[must_use]
    pub fn with_payload<T: Any + Send + Sync>(mut self, payload: T) -> Self {
        self.payload = Some(Arc::new(payload));
        self
    }

    #[must_use]
    pub fn account_id(&self) -> Uuid {
        self.account_id
    }

    #[must_use]
    pub fn operation_id(&self) -> Uuid {
        self.operation_id
    }

    /// Maximum units of work authorized. `None` means unbounded.
    #[must_use]
    pub fn max_units(&self) -> Option<i32> {
        self.max_units
    }

    /// Clamp `units` to what this permit authorizes.
    #[must_use]
    pub fn clamp_units(&self, units: i32) -> i32 {
        match self.max_units {
            Some(max) => units.min(max),
            None => units,
        }
    }

    /// Recover the policy-owned payload.
    #[must_use]
    pub fn payload_as<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.payload.as_ref().and_then(|p| p.downcast_ref::<T>())
    }

    /// Claim the single recording slot for this permit.
    ///
    /// Returns `true` exactly once per permit — for the first caller, across
    /// every clone, whichever task gets there first. That makes recording
    /// idempotent for one authorized unit of work: SSE cancellation racing the
    /// completion path, a retry of the recording call itself, and any duplicated
    /// completion path sharing the permit all charge the account once.
    ///
    /// Work that is genuinely performed again — a source reprocessed, a new
    /// chat exchange — is authorized again and receives a new permit, so it is
    /// charged again. That is intended: the provider was paid twice.
    pub fn claim_recording(&self) -> bool {
        !self.recorded.swap(true, Ordering::SeqCst)
    }

    /// Release the recording slot after a failed record attempt so that a retry
    /// can charge the account.
    pub fn release_recording(&self) {
        self.recorded.store(false, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for Permit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Permit")
            .field("account_id", &self.account_id)
            .field("operation_id", &self.operation_id)
            .field("max_units", &self.max_units)
            .field("recorded", &self.recorded.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Policy
// ============================================================================

/// Authorizes bounded core operations and records metered consumption.
#[async_trait]
pub trait EntitlementPolicy: Send + Sync {
    /// Authorize one operation, or fail with the response the client should see.
    ///
    /// Implementations must fail closed: an unexpected internal failure denies
    /// the operation rather than allowing it.
    async fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<Permit, AppError>;

    /// Record `units` of work actually performed under `permit`.
    ///
    /// Called only after the provider has produced real output. Returning an
    /// error means the account has exhausted its allowance between
    /// authorization and recording; the core surfaces that to the client.
    async fn record(&self, _permit: &Permit, _units: i32) -> Result<(), AppError> {
        Ok(())
    }

    /// Report a quota the core has to include in a response.
    ///
    /// `None` means unlimited.
    async fn quota(&self, _principal: &Principal, _quota: Quota) -> Result<Option<i32>, AppError> {
        Ok(None)
    }
}

/// Shared handle to the configured policy.
pub type SharedEntitlementPolicy = Arc<dyn EntitlementPolicy>;

// ============================================================================
// Public adapter
// ============================================================================

/// Authorizes every structurally valid operation.
///
/// This is what a self-hosted deployment runs: no usage table, no subscription
/// table, no plan. It exists so that the public core and the hosted product
/// execute the same handler code.
#[derive(Debug, Clone, Default)]
pub struct UnrestrictedPolicy;

#[async_trait]
impl EntitlementPolicy for UnrestrictedPolicy {
    async fn authorize(&self, request: AuthorizationRequest<'_>) -> Result<Permit, AppError> {
        Ok(Permit::unrestricted(
            request.account_id(),
            request.operation_id,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Payload(i32);

    fn permit() -> Permit {
        Permit::bounded(Uuid::nil(), Uuid::nil(), 10)
    }

    #[test]
    fn recording_slot_is_claimed_once() {
        let p = permit();
        assert!(p.claim_recording());
        assert!(!p.claim_recording());
        assert!(!p.clone().claim_recording(), "clones share the same slot");
    }

    #[test]
    fn released_slot_can_be_reclaimed() {
        let p = permit();
        assert!(p.claim_recording());
        p.release_recording();
        assert!(p.claim_recording());
    }

    #[test]
    fn units_are_clamped_to_the_authorized_maximum() {
        assert_eq!(permit().clamp_units(25), 10);
        assert_eq!(permit().clamp_units(3), 3);
        assert_eq!(
            Permit::unrestricted(Uuid::nil(), Uuid::nil()).clamp_units(9_999),
            9_999
        );
    }

    #[test]
    fn payload_round_trips_only_for_its_own_type() {
        let p = permit().with_payload(Payload(7));
        assert_eq!(p.payload_as::<Payload>().map(|v| v.0), Some(7));
        assert!(p.payload_as::<String>().is_none());
    }

    #[tokio::test]
    async fn unrestricted_policy_authorizes_every_operation() {
        let policy = UnrestrictedPolicy;
        let principal = Principal::new(Uuid::new_v4());

        for operation in [
            Operation::CreateNotebook,
            Operation::CreateSource {
                notebook_id: Uuid::new_v4(),
                source_type: SourceType::Web,
            },
            Operation::CreateMemory {
                notebook_id: Uuid::new_v4(),
            },
            Operation::SelectModel {
                provider: "anthropic".into(),
                model: "claude-opus-4".into(),
            },
            Operation::SendChatMessage {
                notebook_id: Uuid::new_v4(),
            },
            Operation::ProcessOcrPages {
                requested_pages: 500,
            },
        ] {
            let permit = policy
                .authorize(AuthorizationRequest::new(
                    &principal,
                    operation,
                    Uuid::new_v4(),
                ))
                .await
                .expect("unrestricted policy denies nothing");
            assert_eq!(permit.max_units(), None);
            policy
                .record(&permit, 42)
                .await
                .expect("unrestricted recording never fails");
        }
    }

    #[tokio::test]
    async fn unrestricted_policy_reports_unlimited_quota() {
        let limit = UnrestrictedPolicy
            .quota(
                &Principal::new(Uuid::new_v4()),
                Quota::MemoriesPerNotebook {
                    notebook_id: Uuid::new_v4(),
                },
            )
            .await
            .expect("quota query succeeds");
        assert_eq!(limit, None);
    }
}
