//! LLM Provider trait for pluggable AI services.

use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::Stream;

use crate::error::AppError;

use super::types::{LlmMessage, LlmStreamEvent, ProviderInfo, RagDocument};

/// Boxed byte stream for SSE responses.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, AppError>> + Send>>;

/// Unified interface for LLM providers (Mistral, Anthropic, etc.).
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Provider identifier (e.g., "mistral", "anthropic").
    fn name(&self) -> &str;

    /// Default model for this provider.
    fn default_model(&self) -> &str;

    /// Whether the provider is available (API key configured, circuit not open).
    fn is_available(&self) -> bool;

    /// Whether this provider supports native document citations.
    ///
    /// When true, the chat handler should pass RAG documents via
    /// [`stream_chat`]'s `documents` parameter instead of embedding them
    /// as XML in the system prompt. The provider will emit
    /// [`LlmStreamEvent::NativeCitation`] events during streaming.
    fn supports_native_citations(&self) -> bool {
        false
    }

    /// Output tokens this provider asks the model to be able to write.
    ///
    /// Part of the one budgeting pass (US-018): the answer occupies the same
    /// context window as the prompt, so a request assembled without counting it
    /// is a request that fits until the model starts writing. Each client
    /// returns the `max_tokens` it actually sends, which is what keeps the
    /// budget and the wire from disagreeing.
    fn max_output_tokens(&self) -> usize {
        4096
    }

    /// List of supported models.
    fn supported_models(&self) -> Vec<String> {
        vec![self.default_model().to_owned()]
    }

    /// Provider metadata.
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: self.name().to_owned(),
            default_model: self.default_model().to_owned(),
            available: self.is_available(),
            models: self.supported_models(),
        }
    }

    /// Stream a chat completion.
    ///
    /// Returns raw SSE bytes; use `parse_sse_data` to interpret events.
    ///
    /// When `documents` is non-empty and the provider supports native citations,
    /// the documents are passed as structured content blocks and the provider
    /// emits [`LlmStreamEvent::NativeCitation`] events during streaming.
    /// Providers that don't support native citations ignore the documents
    /// parameter (the context should already be in the system prompt).
    ///
    /// `temperature` overrides the provider's default when set (e.g., from an
    /// agent config). `None` means use the provider's default.
    async fn stream_chat(
        &self,
        system_prompt: &str,
        messages: Vec<LlmMessage>,
        model: Option<&str>,
        documents: &[RagDocument],
        temperature: Option<f32>,
    ) -> Result<ByteStream, AppError>;

    /// Parse SSE data line into a stream event.
    fn parse_sse_data(&self, data: &str) -> Option<LlmStreamEvent>;
}
