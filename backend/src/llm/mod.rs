//! LLM abstraction layer
//!
//! This module provides a unified interface for working with different LLM providers.
//! It includes:
//! - `types` - Shared types for messages, citations, and events
//! - `prompts` - System prompt builders for RAG with citations and teaching modes
//! - `citations` - Citation extraction from LLM responses
//! - `budget` - Token budget management and history truncation
//! - `sse` - OpenAI-compatible SSE parsing
//! - `provider` - The `LlmProvider` trait that all providers implement

pub mod budget;
pub mod citations;
pub mod fallbacks;
pub mod prompts;
pub mod provider;
pub mod sse;
pub mod types;

// Re-export commonly used types at module level
pub use budget::{PromptBudget, context_window_for_model, estimate_tokens, fit_history};
pub use citations::{LocatedCitation, extract_citations};
pub use fallbacks::{FALLBACK_TEXTS, insufficient_evidence_text, no_sources_text};
pub use prompts::{build_messages, build_system_prompt};
pub use provider::{ByteStream, LlmProvider};
pub use sse::parse_openai_sse_data;
pub use types::{
    CitableChunk, Citation, LlmMessage, LlmStreamEvent, NativeCitation, ProviderInfo, RagDocument,
    Role, TeachingMode,
};
