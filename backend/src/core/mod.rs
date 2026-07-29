//! The open-source core: configuration, state and the seams every edition
//! shares (EP-002, `tasks/prd-open-core.md`).
//!
//! Nothing in this module may reference Clerk, Stripe, Resend or PostHog. The
//! private SaaS composition injects its adapters through the three public
//! seams defined here:
//!
//! | Seam | Public adapter | Private adapter |
//! |---|---|---|
//! | [`Principal`] | static token / loopback single user | Clerk JWT middleware |
//! | [`EntitlementPolicy`] | [`UnrestrictedPolicy`] | Stripe-backed plan limits |
//! | [`EventSink`] | [`NoopEventSink`] / [`TracingEventSink`] | PostHog + Resend consumer |
//! | [`EmbeddingProvider`] / [`Reranker`] | [`DeterministicEmbedder`] or a local model | Voyage AI |
//!
//! [`protocol`] is the fourth public surface, and the only one with no adapter:
//! it is the wire contract every edition emits identically (US-009).

pub mod catalog;
pub mod config;
pub mod entitlements;
pub mod events;
pub mod identity;
pub mod principal;
pub mod protocol;
pub mod providers;
pub mod router;
pub mod state;

pub use catalog::{
    CoreCatalog, ProviderCapabilities, TeachingModeEntry, ValidationLimits, catalog,
};
pub use config::{
    AsyncConfig, ConfigError, CoreConfig, DatabasePoolConfig, HybridSearchConfig,
    MISTRAL_OCR_MAX_FILE_SIZE, OcrConfig, SecurityConfig, VoyageRateLimitConfig,
};
pub use entitlements::{
    AuthorizationRequest, EntitlementPolicy, Operation, Permit, Quota, SharedEntitlementPolicy,
    UnrestrictedPolicy,
};
pub use events::{DomainEvent, EventSink, NoopEventSink, SharedEventSink, TracingEventSink};
pub use identity::{IdentityError, IdentityMode, ReferenceIdentity, reference_auth_middleware};
pub use principal::{Principal, SessionContext};
pub use protocol::{
    ChatEvent, ChatEventStream, EVENT_PROTOCOL_VERSION, SourceEvent, ThinkingStage, WarningKind,
};
pub use providers::{
    DETERMINISTIC_ENV, DeterministicEmbedder, DeterministicLlm, DimensionMismatch, EMBEDDING_DIM,
    EmbeddingProvider, RerankedDocument, Reranker, SharedEmbeddingProvider, SharedReranker,
    check_dimension, deterministic_requested,
};
pub use router::{build_core_health_router, build_core_router};
pub use state::{CoreState, CoreStateParts, ExternalClients, Repositories};
