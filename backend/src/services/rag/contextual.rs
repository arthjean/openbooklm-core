//! Contextual Retrieval service with Anthropic prompt caching.
//!
//! Generates LLM-powered context prefixes for document chunks to improve
//! retrieval quality, following Anthropic's Contextual Retrieval technique.
//!
//! Each chunk receives a short contextual prefix (50-100 tokens) that situates
//! it within the overall document, making embeddings and BM25 search more precise.
//!
//! ## Prompt Caching
//!
//! When processing multiple chunks from the same source, the document content
//! is marked with `cache_control` so that Anthropic caches it across requests.
//! Chunks are processed **sequentially** to maximize cache hits — the document
//! is tokenized once and reused for every chunk.
//!
//! ## Reference
//!
//! [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

use std::sync::Arc;

use serde::Serialize;

use crate::{
    clients::{AnthropicMessagesClient, ClientMetrics},
    error::{AppError, LlmError},
};

// ============================================================================
// Constants
// ============================================================================

/// Default model for chunk contextualization.
/// Haiku is fast and cheap, sufficient for generating 50-100 token context prefixes.
pub const DEFAULT_CONTEXT_MODEL: &str = "claude-haiku-4-5-20251001";

/// Maximum tokens for the context prefix response.
/// Context prefixes should be concise (50-100 tokens).
const MAX_CONTEXT_TOKENS: i32 = 128;

const SYSTEM_PROMPT: &str = "You are a document analysis expert. Your task is to provide a short, succinct context for a text chunk to improve search retrieval. Answer only with the succinct context and nothing else. Do not use phrases like \"This chunk\" or \"This section\". Write in the same language as the document.";

const CHUNK_INSTRUCTION: &str = "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.";

// ============================================================================
// Request types (Anthropic Messages API with prompt caching)
// ============================================================================

/// Top-level request with content block arrays for prompt caching support.
#[derive(Debug, Serialize)]
struct CachedApiRequest<'a> {
    model: &'a str,
    max_tokens: i32,
    system: Vec<TextBlock<'a>>,
    messages: Vec<CachedMessage<'a>>,
}

#[derive(Debug, Serialize)]
struct CachedMessage<'a> {
    role: &'a str,
    content: Vec<TextBlock<'a>>,
}

#[derive(Debug, Serialize)]
struct TextBlock<'a> {
    #[serde(rename = "type")]
    block_type: &'a str,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    cache_type: &'static str,
}

const EPHEMERAL_CACHE: CacheControl = CacheControl {
    cache_type: "ephemeral",
};

// ============================================================================
// Public types
// ============================================================================

/// Result of contextualizing a single chunk.
#[derive(Debug, Clone)]
pub struct ContextResult {
    /// The generated context prefix to prepend to the chunk.
    pub context_prefix: String,
    /// Token usage for cost tracking.
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens written to cache on this request (0 if cache was already warm).
    pub cache_creation_input_tokens: u32,
    /// Tokens read from cache on this request (0 on first request).
    pub cache_read_input_tokens: u32,
}

impl ContextResult {
    /// Whether this request hit the prompt cache (document was already cached).
    pub fn is_cache_hit(&self) -> bool {
        self.cache_read_input_tokens > 0
    }
}

/// Aggregate statistics for contextualizing all chunks of a source.
#[derive(Debug, Clone)]
pub struct SourceContextStats {
    pub total_chunks: usize,
    pub successful: usize,
    pub failed: usize,
    pub cache_hits: usize,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
    pub total_cache_creation_tokens: u32,
    pub total_cache_read_tokens: u32,
}

impl SourceContextStats {
    fn new() -> Self {
        Self {
            total_chunks: 0,
            successful: 0,
            failed: 0,
            cache_hits: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_creation_tokens: 0,
            total_cache_read_tokens: 0,
        }
    }

    fn record_success(&mut self, result: &ContextResult) {
        self.successful += 1;
        if result.is_cache_hit() {
            self.cache_hits += 1;
        }
        self.total_input_tokens += result.input_tokens;
        self.total_output_tokens += result.output_tokens;
        self.total_cache_creation_tokens += result.cache_creation_input_tokens;
        self.total_cache_read_tokens += result.cache_read_input_tokens;
    }

    fn record_failure(&mut self) {
        self.failed += 1;
    }

    /// Cache hit rate as a percentage (0.0 to 100.0).
    pub fn cache_hit_rate(&self) -> f32 {
        if self.successful == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            (self.cache_hits as f32 / self.successful as f32) * 100.0
        }
    }
}

// ============================================================================
// Service
// ============================================================================

/// Service that generates contextual prefixes for document chunks.
///
/// Uses a dedicated [`AnthropicMessagesClient`] with its own circuit breaker
/// and retry logic, separate from the main chat LLM client to avoid interference.
///
/// ## Prompt Caching
///
/// The system prompt and document content are marked with `cache_control: ephemeral`
/// so Anthropic caches them across sequential chunk requests. This reduces cost
/// by ~50% for sources with many chunks.
#[derive(Clone)]
pub struct ContextualizationService {
    client: AnthropicMessagesClient,
    model: String,
}

impl ContextualizationService {
    /// Create a new contextualization service from an API key.
    ///
    /// The `model` parameter allows overriding the default model.
    /// Pass `None` to use [`DEFAULT_CONTEXT_MODEL`] (Claude Haiku).
    pub fn new(
        api_key: impl Into<Arc<str>>,
        model: Option<String>,
        metrics: &ClientMetrics,
    ) -> Result<Self, AppError> {
        let client = AnthropicMessagesClient::new(
            api_key,
            "contextualization",
            30,
            metrics.provider("contextualization"),
        )?
        .with_beta_header("prompt-caching-2024-07-31");

        Ok(Self {
            client,
            model: model.unwrap_or_else(|| DEFAULT_CONTEXT_MODEL.to_string()),
        })
    }

    /// Check if the service is available (circuit breaker not open).
    pub fn is_available(&self) -> bool {
        self.client.is_available()
    }

    /// The model used for contextualization.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Generate a context prefix for a single chunk within a document.
    ///
    /// Uses prompt caching: the system prompt and document are marked with
    /// `cache_control` so subsequent calls with the same document reuse the cache.
    pub async fn contextualize_chunk(
        &self,
        document: &str,
        chunk: &str,
    ) -> Result<ContextResult, AppError> {
        let chunk_message = build_chunk_message(chunk);
        let request = build_cached_request(&self.model, document, &chunk_message);

        let response = self.client.send(&request).await?;

        let text = response
            .text()
            .ok_or_else(|| {
                AppError::from(LlmError::ResponseParseFailed {
                    provider: "anthropic".into(),
                    reason: "Empty content in Anthropic response".into(),
                })
            })?
            .to_string();

        Ok(ContextResult {
            context_prefix: text,
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
            cache_creation_input_tokens: response.usage.cache_creation_input_tokens,
            cache_read_input_tokens: response.usage.cache_read_input_tokens,
        })
    }

    /// Contextualize all chunks of a source sequentially to maximize cache hits.
    ///
    /// Chunks are processed in order so the document content stays in Anthropic's
    /// prompt cache. Returns a vec of `Option<ContextResult>` (None for failed chunks)
    /// and aggregate statistics.
    ///
    /// Failed chunks are logged but do not abort the batch — callers should store
    /// those chunks without a context prefix (graceful degradation).
    pub async fn contextualize_source_chunks(
        &self,
        document: &str,
        chunks: &[String],
    ) -> (Vec<Option<ContextResult>>, SourceContextStats) {
        let mut results = Vec::with_capacity(chunks.len());
        let mut stats = SourceContextStats::new();
        stats.total_chunks = chunks.len();

        for (i, chunk) in chunks.iter().enumerate() {
            match self.contextualize_chunk(document, chunk).await {
                Ok(result) => {
                    tracing::debug!(
                        chunk_index = i,
                        cache_hit = result.is_cache_hit(),
                        input_tokens = result.input_tokens,
                        output_tokens = result.output_tokens,
                        cache_created = result.cache_creation_input_tokens,
                        cache_read = result.cache_read_input_tokens,
                        "Chunk contextualized"
                    );
                    stats.record_success(&result);
                    results.push(Some(result));
                }
                Err(e) => {
                    tracing::warn!(
                        chunk_index = i,
                        error = %e,
                        "Failed to contextualize chunk, continuing without prefix"
                    );
                    stats.record_failure();
                    results.push(None);
                }
            }
        }

        tracing::info!(
            total_chunks = stats.total_chunks,
            successful = stats.successful,
            failed = stats.failed,
            total_input_tokens = stats.total_input_tokens,
            total_output_tokens = stats.total_output_tokens,
            cache_creation_tokens = stats.total_cache_creation_tokens,
            cache_read_tokens = stats.total_cache_read_tokens,
            cache_hit_rate = format!("{:.1}%", stats.cache_hit_rate()),
            "Source contextualization complete"
        );

        (results, stats)
    }
}

// ============================================================================
// Prompt construction
// ============================================================================

/// Build a cached API request for chunk contextualization.
///
/// The system prompt is marked with `cache_control: ephemeral`.
/// The document block is marked with `cache_control: ephemeral`.
/// The chunk + instructions block is NOT cached (changes every request).
fn build_cached_request<'a>(
    model: &'a str,
    document: &'a str,
    chunk_message: &'a str,
) -> CachedApiRequest<'a> {
    CachedApiRequest {
        model,
        max_tokens: MAX_CONTEXT_TOKENS,
        system: vec![TextBlock {
            block_type: "text",
            text: SYSTEM_PROMPT,
            cache_control: Some(EPHEMERAL_CACHE),
        }],
        messages: vec![CachedMessage {
            role: "user",
            content: vec![
                // Document block — cached across chunk requests
                TextBlock {
                    block_type: "text",
                    text: document,
                    cache_control: Some(EPHEMERAL_CACHE),
                },
                // Chunk + instructions — unique per request, NOT cached
                TextBlock {
                    block_type: "text",
                    text: chunk_message,
                    cache_control: None,
                },
            ],
        }],
    }
}

/// Build the chunk-specific part of the user message (not cached).
///
/// Uses the Anthropic-recommended format with `<chunk>` tags.
pub fn build_chunk_message(chunk: &str) -> String {
    format!(
        "Here is the chunk we want to situate within the whole document:\n\
         <chunk>\n{chunk}\n</chunk>\n\n\
         {CHUNK_INSTRUCTION}"
    )
}

/// Build the full user message for chunk contextualization (legacy, non-cached).
///
/// Uses the Anthropic-recommended format with `<document>` and `<chunk>` tags.
pub fn build_prompt(document: &str, chunk: &str) -> String {
    format!(
        "<document>\n{document}\n</document>\n\n\
         Here is the chunk we want to situate within the whole document:\n\
         <chunk>\n{chunk}\n</chunk>\n\n\
         {CHUNK_INSTRUCTION}"
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_contains_document_and_chunk_tags() {
        let doc = "Annual report 2024 for Tesla. Revenue grew 3% in Q3.";
        let chunk = "Revenue grew 3% in Q3.";

        let prompt = build_prompt(doc, chunk);

        assert!(prompt.contains("<document>"), "Must contain <document> tag");
        assert!(
            prompt.contains("</document>"),
            "Must contain </document> tag"
        );
        assert!(prompt.contains("<chunk>"), "Must contain <chunk> tag");
        assert!(prompt.contains("</chunk>"), "Must contain </chunk> tag");
        assert!(prompt.contains(doc), "Must contain the full document");
        assert!(prompt.contains(chunk), "Must contain the chunk");
    }

    #[test]
    fn build_prompt_has_correct_structure() {
        let prompt = build_prompt("doc content", "chunk content");

        let doc_pos = prompt
            .find("<document>")
            .expect("Prompt must contain <document> tag");
        let chunk_pos = prompt
            .find("<chunk>")
            .expect("Prompt must contain <chunk> tag");
        assert!(
            doc_pos < chunk_pos,
            "Document tag should appear before chunk tag"
        );

        let instruction_pos = prompt
            .find("Please give a short succinct context")
            .expect("Prompt must contain instruction text");
        assert!(
            chunk_pos < instruction_pos,
            "Instructions should appear after chunk"
        );
    }

    #[test]
    fn build_chunk_message_contains_chunk_tag_and_instruction() {
        let msg = build_chunk_message("some chunk text");

        assert!(msg.contains("<chunk>"), "Must contain <chunk> tag");
        assert!(
            msg.contains("some chunk text"),
            "Must contain chunk content"
        );
        assert!(
            msg.contains("Please give a short succinct context"),
            "Must contain instruction"
        );
        // Should NOT contain <document> — that's in a separate cached block
        assert!(
            !msg.contains("<document>"),
            "Must NOT contain <document> tag"
        );
    }

    #[test]
    fn cached_request_has_cache_control_on_system_and_document() {
        let chunk_msg = build_chunk_message("chunk");
        let request =
            build_cached_request("claude-haiku-4-5-20251001", "full document", &chunk_msg);

        // System prompt should have cache_control
        assert_eq!(request.system.len(), 1);
        assert!(
            request.system[0].cache_control.is_some(),
            "System prompt must have cache_control"
        );

        // User message should have 2 content blocks
        assert_eq!(request.messages.len(), 1);
        let content = &request.messages[0].content;
        assert_eq!(
            content.len(),
            2,
            "User message should have document + chunk blocks"
        );

        // Document block (index 0) should have cache_control
        assert!(
            content[0].cache_control.is_some(),
            "Document block must have cache_control"
        );
        assert_eq!(content[0].text, "full document");

        // Chunk block (index 1) should NOT have cache_control
        assert!(
            content[1].cache_control.is_none(),
            "Chunk block must NOT have cache_control"
        );
    }

    #[test]
    fn cached_request_serializes_correctly() {
        let chunk_msg = build_chunk_message("test chunk");
        let request = build_cached_request("claude-haiku-4-5-20251001", "test doc", &chunk_msg);

        let json = serde_json::to_value(&request).expect("CachedApiRequest must serialize to JSON");

        // Verify cache_control on system
        let system = json["system"]
            .as_array()
            .expect("'system' field must be a JSON array");
        assert!(system[0]["cache_control"].is_object());
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");

        // Verify cache_control on document block
        let content = json["messages"][0]["content"]
            .as_array()
            .expect("'messages[0].content' field must be a JSON array");
        assert!(content[0]["cache_control"].is_object());
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");

        // Verify NO cache_control on chunk block (should be absent due to skip_serializing_if)
        assert!(content[1].get("cache_control").is_none());
    }

    #[test]
    fn context_result_cache_hit_detection() {
        let hit = ContextResult {
            context_prefix: "test".into(),
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 5000,
        };
        assert!(hit.is_cache_hit());

        let miss = ContextResult {
            context_prefix: "test".into(),
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 5000,
            cache_read_input_tokens: 0,
        };
        assert!(!miss.is_cache_hit());
    }

    #[test]
    fn source_context_stats_cache_hit_rate() {
        let mut stats = SourceContextStats::new();
        stats.total_chunks = 5;

        // First chunk creates cache
        stats.record_success(&ContextResult {
            context_prefix: String::new(),
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 5000,
            cache_read_input_tokens: 0,
        });

        // Remaining 4 chunks hit cache
        for _ in 0..4 {
            stats.record_success(&ContextResult {
                context_prefix: String::new(),
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 5000,
            });
        }

        assert_eq!(stats.successful, 5);
        assert_eq!(stats.cache_hits, 4);
        assert_eq!(stats.total_cache_creation_tokens, 5000);
        assert_eq!(stats.total_cache_read_tokens, 20000);
        assert!(
            (stats.cache_hit_rate() - 80.0).abs() < 0.1,
            "4/5 = 80% hit rate"
        );
    }

    #[test]
    fn source_context_stats_with_failures() {
        let mut stats = SourceContextStats::new();
        stats.total_chunks = 3;

        stats.record_success(&ContextResult {
            context_prefix: String::new(),
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 5000,
            cache_read_input_tokens: 0,
        });
        stats.record_failure();
        stats.record_success(&ContextResult {
            context_prefix: String::new(),
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 5000,
        });

        assert_eq!(stats.successful, 2);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.total_input_tokens, 200);
    }

    #[test]
    fn default_context_model_is_haiku() {
        assert!(
            DEFAULT_CONTEXT_MODEL.contains("haiku"),
            "Default model should be Haiku for cost efficiency"
        );
    }

    #[test]
    fn service_accepts_custom_model() {
        let metrics = ClientMetrics::new();
        let svc = ContextualizationService::new(
            "test-key",
            Some("claude-sonnet-4-6-20260220".to_string()),
            &metrics,
        )
        .expect("HTTP client should build in test environment");
        assert_eq!(svc.model(), "claude-sonnet-4-6-20260220");
    }

    #[test]
    fn service_uses_default_model_when_none() {
        let metrics = ClientMetrics::new();
        let svc = ContextualizationService::new("test-key", None, &metrics)
            .expect("HTTP client should build in test environment");
        assert_eq!(svc.model(), DEFAULT_CONTEXT_MODEL);
    }

    #[test]
    fn max_context_tokens_is_reasonable() {
        const {
            assert!(
                MAX_CONTEXT_TOKENS <= 256,
                "Max tokens should be small for context prefixes"
            );
            assert!(
                MAX_CONTEXT_TOKENS >= 64,
                "Max tokens should allow at least 64 tokens"
            );
        }
    }

    #[test]
    fn build_prompt_does_not_contain_unexpected_tags() {
        let prompt = build_prompt("doc content", "chunk content");

        assert!(
            !prompt.contains("<summary>"),
            "Prompt must not contain unexpected <summary> tag"
        );
        assert!(
            !prompt.contains("<response>"),
            "Prompt must not contain unexpected <response> tag"
        );
        assert!(
            !prompt.contains("<system>"),
            "Prompt must not contain unexpected <system> tag"
        );
    }

    // --- US-012: Graceful degradation ---

    #[test]
    fn service_is_available_by_default() {
        let metrics = ClientMetrics::new();
        let svc = ContextualizationService::new("test-key", None, &metrics)
            .expect("HTTP client should build in test environment");
        // Fresh service with no failures should be available
        assert!(
            svc.is_available(),
            "Newly created service should be available (circuit closed)"
        );
    }

    #[test]
    fn source_context_stats_all_failed() {
        let mut stats = SourceContextStats::new();
        stats.total_chunks = 3;
        stats.record_failure();
        stats.record_failure();
        stats.record_failure();

        assert_eq!(stats.successful, 0);
        assert_eq!(stats.failed, 3);
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }
}
