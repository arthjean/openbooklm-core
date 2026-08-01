//! RAG (Retrieval-Augmented Generation) pipeline sub-module.
//!
//! Groups all retrieval-related services: search, vector store, embeddings,
//! HyDE query expansion, query reformulation, contextual retrieval, and chunking.

pub mod contextual;
pub mod embedding_cache;
pub mod eval;
pub mod hyde;
pub mod provenance;
pub mod query_reformulation;
pub mod search;
pub mod utils;
pub mod vector_store;

pub mod chunking;
pub mod rag_log;
