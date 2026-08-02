//! Core business logic.
//!
//! RAG, ingestion, chat, notebooks, sources, memory and suggestions. Billing,
//! lifecycle email and onboarding are hosted concerns and live outside this
//! module (US-013).

// Sub-modules (grouped by domain)
pub mod rag;

// Standalone services
pub mod chat;
pub mod content_cleaning;
pub mod embeddings;
pub mod ingestion_tasks;
pub mod maintenance;
pub mod memory;
pub mod notebooks;
pub mod processor;
pub mod source_events;
pub mod source_processing;
pub mod sources;
pub mod suggestions;
