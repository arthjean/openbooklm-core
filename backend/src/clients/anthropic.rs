//! Anthropic Claude client with retry logic and circuit breaker.
//!
//! Provides resilient access to Anthropic's Messages API with:
//! - Connection pooling, exponential backoff retry, circuit breaker
//! - Request metrics tracking and configurable timeouts
//!
//! Key differences from OpenAI-compatible APIs:
//! - Uses `x-api-key` header instead of `Authorization: Bearer`
//! - System prompt is a separate field, not a message

use std::{fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{
    circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState},
    metrics::{ClientMetrics, ProviderMetrics},
    openai_compat::llm_rate_limited_err,
    resilience::{
        RequestErrorKind, ResilientExecutor, build_http_client, check_response_status,
        handle_request_error, with_request_id, wrap_sse_stream,
    },
    retry::RetryConfig,
};
use crate::{
    core::config::CoreConfig,
    error::{AppError, LlmError},
    llm::{ByteStream, LlmMessage, LlmProvider, LlmStreamEvent},
};

/// Anthropic Messages API URL. Shared across clients and services.
pub const API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic API version header value. Shared across clients and services.
pub const API_VERSION: &str = "2023-06-01";

const DEFAULT_MODEL: &str = "claude-sonnet-4-6-20260220";
const MAX_TOKENS: i32 = 8192;

/// Build the standard Anthropic request headers (`x-api-key`, `anthropic-version`,
/// `content-type`).
///
/// Returns an error if the API key contains non-visible-ASCII characters
/// (e.g. BOM, zero-width spaces from copy-paste).
///
/// Service-specific headers (e.g. `anthropic-beta`) should be added to the
/// returned map or appended on the request builder.
pub fn anthropic_headers(api_key: &str) -> Result<reqwest::header::HeaderMap, AppError> {
    use reqwest::header::{HeaderMap, HeaderValue};

    let mut headers = HeaderMap::with_capacity(3);
    // x-api-key — Anthropic uses a custom header, not Authorization: Bearer
    headers.insert(
        "x-api-key",
        HeaderValue::from_str(api_key).map_err(|_| {
            AppError::Validation("API key contains invalid header characters".into())
        })?,
    );
    headers.insert("anthropic-version", HeaderValue::from_static(API_VERSION));
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    Ok(headers)
}

// ============================================================================
// Request/Response types
// ============================================================================

/// A message in the Anthropic Messages API.
///
/// The `content` field can be either a plain string (for simple text messages)
/// or a structured array of content blocks (for messages with documents).
#[derive(Debug, Serialize, Clone)]
struct Message {
    role: String,
    content: MessageContent,
}

/// Message content: either a plain string or structured content blocks.
///
/// Using `#[serde(untagged)]` so plain strings serialize as `"content": "text"`
/// and block arrays serialize as `"content": [...]`.
#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlockInput>),
}

/// A content block in a structured message.
#[derive(Debug, Serialize, Clone)]
#[serde(tag = "type")]
enum ContentBlockInput {
    /// A plain text block.
    #[serde(rename = "text")]
    Text { text: String },
    /// A document block with optional citation support.
    #[serde(rename = "document")]
    Document {
        source: DocumentSource,
        title: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
    },
}

/// Source content for a document block.
#[derive(Debug, Serialize, Clone)]
struct DocumentSource {
    #[serde(rename = "type")]
    source_type: &'static str,
    media_type: &'static str,
    data: String,
}

/// Citation configuration for a document block.
#[derive(Debug, Serialize, Clone)]
struct CitationsConfig {
    enabled: bool,
}

impl From<LlmMessage> for Message {
    fn from(msg: LlmMessage) -> Self {
        Self {
            role: msg.role.as_str().to_owned(),
            content: MessageContent::Text(msg.content),
        }
    }
}

/// System prompt content block with optional cache control.
///
/// Anthropic's prompt caching requires the system prompt to be structured as
/// an array of content blocks rather than a plain string. Adding
/// `cache_control: {type: "ephemeral"}` enables automatic caching with a
/// 5-minute TTL, giving a ~90% discount on cached input tokens.
#[derive(Debug, Serialize)]
struct SystemBlock<'a> {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: &'a str,
    cache_control: CacheControl,
}

#[derive(Debug, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    cache_type: &'static str,
}

impl CacheControl {
    const fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral",
        }
    }
}

#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'a str,
    max_tokens: i32,
    system: Vec<SystemBlock<'a>>,
    messages: Vec<Message>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    MessageStart {
        #[serde(rename = "message")]
        _message: MessageInfo,
    },
    ContentBlockStart {
        #[serde(rename = "index")]
        _index: i32,
        #[serde(rename = "content_block")]
        _content_block: ContentBlock,
    },
    ContentBlockDelta {
        #[serde(rename = "index")]
        _index: i32,
        delta: ContentDelta,
    },
    ContentBlockStop {
        #[serde(rename = "index")]
        _index: i32,
    },
    MessageDelta {
        #[serde(rename = "delta")]
        _delta: MessageDeltaInfo,
    },
    MessageStop,
    Ping,
    Error {
        error: ErrorInfo,
    },
}

#[derive(Debug, Deserialize)]
struct MessageInfo {
    #[serde(rename = "id")]
    _id: String,
    #[serde(rename = "type")]
    _msg_type: String,
    #[serde(rename = "role")]
    _role: String,
    #[serde(rename = "model")]
    _model: String,
    /// Token usage including prompt cache metrics.
    #[serde(default)]
    usage: Option<StreamUsage>,
}

/// Token usage from streaming `message_start` event, including cache metrics.
#[derive(Debug, Deserialize)]
#[allow(clippy::struct_field_names)] // matches Anthropic API response shape
struct StreamUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    _block_type: String,
    #[serde(rename = "text")]
    _text: Option<String>,
}

/// Delta within a content block: either text or a citation.
///
/// Uses `#[serde(tag = "type")]` to dispatch on the `"type"` field:
/// - `"text_delta"` → text content
/// - `"citations_delta"` → native citation from Citations API
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentDelta {
    TextDelta {
        text: String,
    },
    CitationsDelta {
        citation: RawCitation,
    },
    /// Catch-all for unknown delta types (signature_delta, etc.)
    #[serde(other)]
    Unknown,
}

/// Raw citation from the Anthropic Citations API.
///
/// Covers all citation location types (char, page, content_block).
/// We only need the document_index, cited_text, and document_title.
#[derive(Debug, Deserialize)]
struct RawCitation {
    /// 0-indexed position across all document blocks in the request.
    document_index: usize,
    /// Exact text from the source that was cited.
    cited_text: String,
    /// Title of the cited document.
    #[serde(default)]
    document_title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaInfo {
    #[serde(rename = "stop_reason")]
    _stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ErrorInfo {
    message: String,
}

// ============================================================================
// Client implementation
// ============================================================================

/// Anthropic Claude client with resilience patterns.
#[derive(Clone)]
pub struct AnthropicClient {
    http_client: reqwest::Client,
    api_key: Arc<str>,
    timeout: Duration,
    resilience: ResilientExecutor,
}

impl fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("timeout", &self.timeout)
            .field("resilience", &self.resilience)
            .finish_non_exhaustive()
    }
}

impl AnthropicClient {
    pub fn from_config(config: &CoreConfig, metrics: &ClientMetrics) -> Result<Self, LlmError> {
        let api_key = config.anthropic_api_key.as_deref().ok_or_else(|| {
            tracing::debug!("Anthropic API key not configured");
            LlmError::ApiKeyMissing {
                provider: "anthropic".into(),
            }
        })?;

        Self::new(
            api_key,
            Duration::from_secs(config.async_config.llm_timeout_secs),
            metrics.provider("anthropic"),
        )
    }

    pub fn new(
        api_key: impl Into<Arc<str>>,
        timeout: Duration,
        metrics: Arc<ProviderMetrics>,
    ) -> Result<Self, LlmError> {
        let http_client =
            build_http_client(None, 10).map_err(|reason| LlmError::RequestFailed {
                provider: "anthropic".into(),
                reason,
            })?;

        let retry_config = RetryConfig::new(3)
            .with_initial_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(30));

        let circuit_breaker = Arc::new(CircuitBreaker::new(
            "anthropic",
            CircuitBreakerConfig::new(5)
                .with_open_duration(Duration::from_secs(30))
                .with_success_threshold(2),
        ));

        Ok(Self {
            http_client,
            api_key: api_key.into(),
            timeout,
            resilience: ResilientExecutor::new("anthropic", retry_config, circuit_breaker, metrics)
                .with_timeout_secs(timeout.as_secs()),
        })
    }

    async fn execute_request(
        &self,
        request: &Request<'_>,
    ) -> Result<reqwest::Response, (AppError, Option<u16>, bool)> {
        let timeout_secs = self.timeout.as_secs();

        let headers = anthropic_headers(&self.api_key).map_err(|e| (e, None::<u16>, false))?;

        let response = with_request_id(self.http_client.post(API_URL).headers(headers))
            .json(request)
            .send()
            .await
            .map_err(|e| {
                handle_request_error(&e, |kind, description| match kind {
                    RequestErrorKind::Timeout => LlmError::Timeout {
                        provider: "anthropic".into(),
                        timeout_secs,
                    }
                    .into(),
                    _ => LlmError::RequestFailed {
                        provider: "anthropic".into(),
                        reason: description,
                    }
                    .into(),
                })
            })?;

        check_response_status(
            response,
            "anthropic",
            |code, body, retry_after| match code {
                429 | 529 => llm_rate_limited_err("anthropic", retry_after.unwrap_or(30).min(600)),
                _ => LlmError::RequestFailed {
                    provider: "anthropic".into(),
                    reason: format!("API returned {code}: {body}"),
                }
                .into(),
            },
        )
        .await
    }
}

super::impl_client_resilience_methods!(AnthropicClient, "anthropic");

#[async_trait]
#[allow(clippy::unnecessary_literal_bound)]
impl LlmProvider for AnthropicClient {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn default_model(&self) -> &str {
        DEFAULT_MODEL
    }

    fn is_available(&self) -> bool {
        self.resilience.circuit_breaker().state() != CircuitState::Open
    }

    fn supports_native_citations(&self) -> bool {
        true
    }

    /// The `max_tokens` every request below sends.
    fn max_output_tokens(&self) -> usize {
        usize::try_from(MAX_TOKENS).unwrap_or(4096)
    }

    fn supported_models(&self) -> Vec<String> {
        [
            "claude-opus-4-6-20260220",
            "claude-sonnet-4-6-20260220",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }

    async fn stream_chat(
        &self,
        system_prompt: &str,
        messages: Vec<LlmMessage>,
        model: Option<&str>,
        documents: &[crate::llm::RagDocument],
        temperature: Option<f32>,
    ) -> Result<ByteStream, AppError> {
        let model = model.unwrap_or(DEFAULT_MODEL);

        // Build messages: conversation history + current user message.
        // When documents are provided, the last user message is rebuilt as a
        // structured content array with document blocks + text block.
        let mut api_messages: Vec<Message> = messages
            .into_iter()
            .filter(|m| !m.is_system())
            .map(Message::from)
            .collect();

        if !documents.is_empty() {
            // Pop the last user message and rebuild it with document blocks
            if let Some(last_msg) = api_messages.pop() {
                let user_text = match last_msg.content {
                    MessageContent::Text(t) => t,
                    MessageContent::Blocks(blocks) => blocks
                        .into_iter()
                        .find_map(|b| match b {
                            ContentBlockInput::Text { text } => Some(text),
                            _ => None,
                        })
                        .unwrap_or_default(),
                };

                let mut blocks: Vec<ContentBlockInput> = documents
                    .iter()
                    .map(|doc| ContentBlockInput::Document {
                        source: DocumentSource {
                            source_type: "text",
                            media_type: "text/plain",
                            data: doc.content.clone(),
                        },
                        title: doc.title.clone(),
                        citations: Some(CitationsConfig { enabled: true }),
                    })
                    .collect();

                blocks.push(ContentBlockInput::Text { text: user_text });

                api_messages.push(Message {
                    role: "user".to_owned(),
                    content: MessageContent::Blocks(blocks),
                });

                tracing::debug!(
                    document_count = documents.len(),
                    "Built Anthropic request with native document citations"
                );
            }
        }

        let request = Request {
            model,
            max_tokens: MAX_TOKENS,
            system: vec![SystemBlock {
                block_type: "text",
                text: system_prompt,
                cache_control: CacheControl::ephemeral(),
            }],
            messages: api_messages,
            stream: true,
            temperature,
        };

        // Execute with resilience (circuit breaker + retry + metrics)
        let response = self
            .resilience
            .execute(
                |retry_secs| llm_rate_limited_err("anthropic", retry_secs),
                || self.execute_request(&request),
            )
            .await?;

        tracing::debug!(model, "Anthropic stream started");
        Ok(wrap_sse_stream(
            response,
            self.timeout,
            self.resilience.clone(),
        ))
    }

    fn parse_sse_data(&self, data: &str) -> Option<LlmStreamEvent> {
        match serde_json::from_str::<Event>(data).ok()? {
            Event::ContentBlockDelta { delta, .. } => match delta {
                ContentDelta::TextDelta { text } => Some(LlmStreamEvent::TextDelta { text }),
                ContentDelta::CitationsDelta { citation } => Some(LlmStreamEvent::NativeCitation {
                    citation: crate::llm::NativeCitation {
                        document_index: citation.document_index,
                        cited_text: citation.cited_text,
                        document_title: citation.document_title.unwrap_or_default(),
                    },
                }),
                ContentDelta::Unknown => None,
            },
            Event::MessageStop => Some(LlmStreamEvent::Done),
            Event::Error { error } => Some(LlmStreamEvent::Error {
                message: error.message,
            }),
            Event::MessageStart {
                _message: ref msg, ..
            } => {
                // Log prompt cache metrics when available
                if let Some(ref usage) = msg.usage
                    && (usage.cache_read_input_tokens > 0 || usage.cache_creation_input_tokens > 0)
                {
                    tracing::info!(
                        input_tokens = usage.input_tokens,
                        cache_read = usage.cache_read_input_tokens,
                        cache_creation = usage.cache_creation_input_tokens,
                        "Anthropic prompt cache metrics"
                    );
                }
                None
            }
            Event::ContentBlockStart { .. }
            | Event::ContentBlockStop { .. }
            | Event::MessageDelta { .. }
            | Event::Ping => None,
        }
    }
}

// ============================================================================
// Shared non-streaming Messages API types
// ============================================================================

/// Request type for the Anthropic Messages API (non-streaming).
///
/// Used by RAG services (HyDE, query reformulation) for simple requests.
/// For prompt caching requests (contextual retrieval), use a custom request
/// type with [`AnthropicMessagesClient::send`], which accepts any `Serialize` body.
#[derive(Debug, Serialize)]
pub struct MessagesRequest<'a> {
    pub model: &'a str,
    pub max_tokens: i32,
    pub system: &'a str,
    pub messages: Vec<MessagesRequestMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// A message in a Messages API request.
#[derive(Debug, Serialize)]
pub struct MessagesRequestMessage {
    pub role: String,
    pub content: String,
}

/// Response from the Anthropic Messages API (non-streaming).
#[derive(Debug, Deserialize)]
pub struct MessagesResponse {
    pub content: Vec<MessagesContentBlock>,
    pub usage: MessagesUsage,
}

impl MessagesResponse {
    /// Extract the first text content block from the response.
    pub fn text(&self) -> Option<&str> {
        self.content.iter().find_map(|b| b.text.as_deref())
    }
}

/// A content block in a Messages API response.
#[derive(Debug, Deserialize)]
pub struct MessagesContentBlock {
    pub text: Option<String>,
}

/// Token usage from a Messages API response.
///
/// Cache fields are populated only when using prompt caching
/// (requires the `anthropic-beta: prompt-caching-2024-07-31` header).
#[derive(Debug, Deserialize)]
pub struct MessagesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

// ============================================================================
// Non-streaming Messages client
// ============================================================================

/// Non-streaming Anthropic Messages API client with resilience.
///
/// Used by RAG services (contextual retrieval, HyDE, query reformulation)
/// for single-request/response interactions. Separate from [`AnthropicClient`]
/// which handles SSE streaming for chat.
///
/// Each service creates its own instance with its own circuit breaker and
/// retry configuration, named after the service (e.g. "contextualization").
#[derive(Clone)]
pub struct AnthropicMessagesClient {
    http_client: reqwest::Client,
    api_key: Arc<str>,
    timeout_secs: u64,
    resilience: ResilientExecutor,
    beta_header: Option<&'static str>,
}

impl fmt::Debug for AnthropicMessagesClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicMessagesClient")
            .field("timeout_secs", &self.timeout_secs)
            .field("resilience", &self.resilience)
            .field("beta_header", &self.beta_header)
            .finish_non_exhaustive()
    }
}

impl AnthropicMessagesClient {
    /// Create a new non-streaming Messages API client.
    ///
    /// Each service should create its own instance with a unique `service_name`
    /// so that circuit breakers and metrics are tracked independently.
    pub fn new(
        api_key: impl Into<Arc<str>>,
        service_name: &str,
        timeout_secs: u64,
        metrics: Arc<ProviderMetrics>,
    ) -> Result<Self, AppError> {
        let http_client = build_http_client(Some(Duration::from_secs(timeout_secs)), 5)
            .map_err(AppError::Internal)?;

        let retry_config = RetryConfig::new(3)
            .with_initial_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(15));

        let circuit_breaker = Arc::new(CircuitBreaker::new(
            service_name,
            CircuitBreakerConfig::new(5)
                .with_open_duration(Duration::from_secs(30))
                .with_success_threshold(2),
        ));

        Ok(Self {
            http_client,
            api_key: api_key.into(),
            timeout_secs,
            resilience: ResilientExecutor::new(
                service_name,
                retry_config,
                circuit_breaker,
                metrics,
            )
            .with_timeout_secs(timeout_secs),
            beta_header: None,
        })
    }

    /// Enable the `anthropic-beta` header (e.g. for prompt caching).
    #[must_use]
    pub fn with_beta_header(mut self, header: &'static str) -> Self {
        self.beta_header = Some(header);
        self
    }

    /// Whether the circuit breaker is allowing requests.
    pub fn is_available(&self) -> bool {
        self.resilience.circuit_breaker().state() != CircuitState::Open
    }

    /// Send a non-streaming request to the Messages API.
    ///
    /// Accepts any `Serialize` request body — use [`MessagesRequest`] for
    /// standard requests, or a custom cached request type for prompt caching.
    pub async fn send<R: Serialize + fmt::Debug>(
        &self,
        request: &R,
    ) -> Result<MessagesResponse, AppError> {
        self.resilience
            .execute(
                |retry_secs| llm_rate_limited_err("anthropic", retry_secs),
                || self.execute_send(request),
            )
            .await
    }

    async fn execute_send<R: Serialize + fmt::Debug>(
        &self,
        request: &R,
    ) -> Result<MessagesResponse, (AppError, Option<u16>, bool)> {
        let mut headers = anthropic_headers(&self.api_key).map_err(|e| (e, None::<u16>, false))?;
        if let Some(beta) = self.beta_header {
            headers.insert(
                "anthropic-beta",
                reqwest::header::HeaderValue::from_static(beta),
            );
        }

        let response = with_request_id(self.http_client.post(API_URL).headers(headers))
            .json(request)
            .send()
            .await
            .map_err(|e| {
                handle_request_error(&e, |kind, description| match kind {
                    RequestErrorKind::Timeout => LlmError::Timeout {
                        provider: "anthropic".into(),
                        timeout_secs: self.timeout_secs,
                    }
                    .into(),
                    _ => LlmError::RequestFailed {
                        provider: "anthropic".into(),
                        reason: description,
                    }
                    .into(),
                })
            })?;

        let response =
            check_response_status(
                response,
                "anthropic",
                |code, body, retry_after| match code {
                    429 | 529 => {
                        llm_rate_limited_err("anthropic", retry_after.unwrap_or(30).min(600))
                    }
                    _ => LlmError::RequestFailed {
                        provider: "anthropic".into(),
                        reason: format!("API returned {code}: {body}"),
                    }
                    .into(),
                },
            )
            .await?;

        // Parse JSON response. Invalid JSON is not retryable.
        response.json::<MessagesResponse>().await.map_err(|e| {
            let err = AppError::from(LlmError::ResponseParseFailed {
                provider: "anthropic".into(),
                reason: format!("Failed to parse Anthropic response: {e}"),
            });
            (err, None, false)
        })
    }
}

super::impl_client_resilience_methods!(AnthropicMessagesClient, "anthropic_messages");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::metrics::ClientMetrics;
    use crate::llm::Role;

    /// Verifies the resilience executor returns an error instead of panicking
    /// when all retry attempts are exhausted (connection refused to real API).
    #[tokio::test]
    async fn retry_returns_error_on_exhaustion() {
        let metrics = ClientMetrics::new();
        let client = AnthropicClient::new(
            "test-key",
            Duration::from_secs(1),
            metrics.provider("anthropic"),
        )
        .expect("HTTP client should build in test environment")
        .with_retry_config(
            RetryConfig::new(1)
                .with_initial_delay(Duration::from_millis(1))
                .with_max_delay(Duration::from_millis(10))
                .without_jitter(),
        );

        let messages = vec![LlmMessage {
            role: Role::User,
            content: "test".to_string(),
        }];

        // stream_chat should return an Err (connection refused), not panic
        let result = client.stream_chat("test", messages, None, &[], None).await;
        assert!(
            result.is_err(),
            "Retry loop must return an error, not panic"
        );
    }

    #[test]
    fn anthropic_headers_accepts_valid_key() {
        let result = anthropic_headers("sk-ant-api03-valid-key-1234");
        assert!(result.is_ok());
        let headers = result.unwrap();
        assert_eq!(
            headers.get("x-api-key").unwrap(),
            "sk-ant-api03-valid-key-1234"
        );
        assert_eq!(headers.get("anthropic-version").unwrap(), API_VERSION);
        assert_eq!(headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn anthropic_headers_rejects_non_ascii() {
        // Newline — control characters (< 0x20, except tab) are rejected by
        // http 1.0's HeaderValue::from_str. This can happen from malformed
        // copy-paste of BYOK keys.
        let result = anthropic_headers("sk-ant-key\nwith-newline");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, AppError::Validation(ref msg) if msg.contains("invalid header characters")),
            "Expected Validation error, got: {err:?}"
        );

        // Null byte
        let result = anthropic_headers("sk-ant-key\0with-null");
        assert!(result.is_err());
    }
}
