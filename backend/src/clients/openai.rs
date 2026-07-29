//! OpenAI client with retry logic and circuit breaker
//!
//! Provides resilient access to OpenAI's chat completion API.
//! Uses shared OpenAI-compatible types and helpers from `openai_compat`.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::core::config::CoreConfig;
use crate::error::{AppError, LlmError};
use crate::llm::{ByteStream, LlmMessage, LlmProvider, LlmStreamEvent};

use super::circuit_breaker::CircuitState;
use super::metrics::ClientMetrics;
use super::openai_compat::{
    ChatMessage, ChatRequest, build_llm_client, execute_chat_request, llm_rate_limited_err,
};
use super::resilience::{ResilientExecutor, wrap_sse_stream};

/// OpenAI API constants
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MODEL: &str = "gpt-5-mini";
const MAX_TOKENS: i32 = 4096;
const PROVIDER_NAME: &str = "openai";

/// OpenAI client with resilience patterns
#[derive(Clone)]
pub struct OpenAiClient {
    http_client: reqwest::Client,
    auth_header: reqwest::header::HeaderValue,
    timeout: Duration,
    resilience: ResilientExecutor,
}

impl std::fmt::Debug for OpenAiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiClient")
            .field("timeout", &self.timeout)
            .field("resilience", &self.resilience)
            .finish_non_exhaustive()
    }
}

impl OpenAiClient {
    /// Create a new OpenAI client from application config
    pub fn from_config(config: &CoreConfig, metrics: &ClientMetrics) -> Result<Self, LlmError> {
        let api_key = config.openai_api_key.as_deref().ok_or_else(|| {
            tracing::error!("OpenAI API key not configured");
            LlmError::ApiKeyMissing {
                provider: PROVIDER_NAME.to_string(),
            }
        })?;

        let timeout = Duration::from_secs(config.async_config.llm_timeout_secs);
        Self::new(api_key, timeout, metrics.provider(PROVIDER_NAME))
    }

    /// Create a new OpenAI client with custom settings
    pub fn new(
        api_key: impl Into<Arc<str>>,
        timeout: Duration,
        metrics: Arc<super::metrics::ProviderMetrics>,
    ) -> Result<Self, LlmError> {
        let api_key = api_key.into();
        let (http_client, auth_header, resilience) =
            build_llm_client(&api_key, PROVIDER_NAME, timeout, metrics, true, 30)?;

        Ok(Self {
            http_client,
            auth_header,
            timeout,
            resilience,
        })
    }
}

super::impl_client_resilience_methods!(OpenAiClient, PROVIDER_NAME);

#[async_trait]
impl LlmProvider for OpenAiClient {
    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn default_model(&self) -> &str {
        DEFAULT_MODEL
    }

    fn is_available(&self) -> bool {
        self.resilience.circuit_breaker().state() != CircuitState::Open
    }

    fn supported_models(&self) -> Vec<String> {
        ["gpt-5.2", "gpt-5-mini"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    async fn stream_chat(
        &self,
        system_prompt: &str,
        messages: Vec<LlmMessage>,
        model: Option<&str>,
        _documents: &[crate::llm::RagDocument],
        temperature: Option<f32>,
    ) -> Result<ByteStream, AppError> {
        let model = model.unwrap_or(DEFAULT_MODEL);

        let mut all_messages = vec![ChatMessage::system(system_prompt)];
        all_messages.extend(messages.into_iter().map(ChatMessage::from));

        let request = ChatRequest {
            model: model.to_string(),
            max_tokens: MAX_TOKENS,
            messages: all_messages,
            stream: true,
            response_format: None,
            temperature,
        };

        let response = self
            .resilience
            .execute(
                |retry_secs| llm_rate_limited_err(PROVIDER_NAME, retry_secs),
                || {
                    execute_chat_request(
                        &self.http_client,
                        &self.auth_header,
                        OPENAI_API_URL,
                        PROVIDER_NAME,
                        self.timeout.as_secs(),
                        &request,
                    )
                },
            )
            .await?;

        tracing::debug!(model, "LLM stream started successfully");
        Ok(wrap_sse_stream(
            response,
            self.timeout,
            self.resilience.clone(),
        ))
    }

    fn parse_sse_data(&self, data: &str) -> Option<LlmStreamEvent> {
        crate::llm::parse_openai_sse_data(data)
    }
}
