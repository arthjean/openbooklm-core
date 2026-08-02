#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unimplemented
    )
)]

//! OpenbookLM backend.
//!
//! One crate, two editions. Everything declared unconditionally is the public
//! core: configuration, state, the RAG and ingestion pipelines, the LLM
//! providers, the core REST and SSE surface and the three seams a composition
//! injects into ([`core::Principal`], [`core::EntitlementPolicy`],
//! [`core::EventSink`]).
//!
//! The `saas` feature adds the hosted composition: Clerk identity, Stripe
//! entitlements, the PostHog/Resend event consumer and the hosted routes. It is
//! on by default in the private repository and does not exist in the public
//! one, whose manifest has neither the feature nor its dependencies (US-013).
//!
//! `cargo check --no-default-features` is therefore the local proof that the
//! core stands alone.

pub mod api;
pub mod clients;
pub mod core;
pub mod db;
pub mod entities;
pub mod error;
pub mod llm;
pub mod middleware;
pub mod repositories;
pub mod services;
pub mod types;
pub mod validation;
pub mod xml;

#[cfg(feature = "saas")]
pub mod app_state;
#[cfg(feature = "saas")]
pub mod auth;
#[cfg(feature = "saas")]
pub mod config;
#[cfg(feature = "saas")]
pub mod saas;

// Re-export commonly used types at crate root for convenience
pub use crate::core::{CoreState, ExternalClients, Repositories};
#[cfg(feature = "saas")]
pub use app_state::AppState;
