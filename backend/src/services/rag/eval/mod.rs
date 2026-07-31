//! Offline RAG evaluation (EP-001).
//!
//! The retrieval pipeline had deterministic unit tests and no way to answer
//! "did that change make retrieval better or worse". This module is the
//! evidence layer: a versioned corpus with explicit relevance judgments, a
//! retrieval runner, a grounded-response evaluator, and a baseline/comparison
//! gate that turns those numbers into a release decision.
//!
//! # Why it lives in the shipped crate
//!
//! Same reason as [`DeterministicEmbedder`](crate::core::providers::DeterministicEmbedder):
//! its consumers are the `rag-eval` binary, the default test suite and CI. An
//! evaluator compiled only under `#[cfg(test)]` would measure a different build
//! than the one being released.
//!
//! # Offline by construction
//!
//! Nothing here opens a socket. The corpus is a checked-in fixture, retrieval
//! runs against an in-memory index, and embeddings come from the deterministic
//! in-process provider. That is a hard requirement (FR-20), not a convenience:
//! a release gate that needs a commercial key is a gate nobody runs.
//!
//! # What each submodule owns
//!
//! | Module | Story | Owns |
//! |---|---|---|
//! | [`corpus`] | US-001 | Fixture schema, loading, validation |
//! | [`trace`] | US-004 | Redacted per-retrieval trace |

pub mod corpus;
pub mod trace;

pub use corpus::{
    CORPUS_RELATIVE_PATH, CorpusError, CorpusViolation, EvalCorpus, EvalQuery, QueryCategory,
    Split, synthetic_uuid,
};
pub use trace::{ReasonCode, RetrievalTrace, ScoreDomain};
