//! Core HTTP handlers.
//!
//! Every module here is part of the public product surface and is composed by
//! [`crate::core::router::build_core_router`]. Hosted-only handlers (billing,
//! webhooks, feedback, newsletter, stats, onboarding settings) are composed by
//! the hosted binary and live outside this module (US-013).

pub mod chat;
pub mod common;
pub mod health;
pub mod memory;
pub mod notebooks;
pub mod notes;
pub mod openapi;
pub mod rag_logs;
pub mod settings;
pub mod sources;
pub mod suggestions;
