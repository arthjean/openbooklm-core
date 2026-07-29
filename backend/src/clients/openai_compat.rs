//! Shared types and helpers for OpenAI-compatible LLM APIs.
//!
//! Mistral and OpenAI use the same chat completion API format.
//! This module extracts the identical message/request types, constructor
//! logic, request execution, and error mapping to eliminate duplication
//! across those clients.

use std::sync::Arc;
use std::time::Duration;

use reqwest::header::HeaderValue;
use serde::Serialize;

use crate::error::{AppError, LlmError};
use crate::llm::LlmMessage;

use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use super::metrics::ProviderMetrics;
use super::resilience::{
    HttpResult, RequestErrorKind, ResilientExecutor, build_http_client, check_response_status,
    handle_request_error, with_request_id,
};
use super::retry::RetryConfig;

// ============================================================================
// Shared message and request types
// ============================================================================

/// Message format for OpenAI-compatible chat APIs.
///
/// Used by Mistral and OpenAI clients.
#[derive(Debug, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
}

impl From<LlmMessage> for ChatMessage {
    fn from(msg: LlmMessage) -> Self {
        Self {
            role: msg.role.as_str().to_owned(),
            content: msg.content,
        }
    }
}

/// Response format for JSON mode.
#[derive(Debug, Serialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

impl ResponseFormat {
    /// JSON object response format — instructs the model to output valid JSON.
    pub fn json_object() -> Self {
        Self {
            format_type: "json_object".to_string(),
        }
    }
}

/// Request body for OpenAI-compatible chat completion APIs.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub max_tokens: i32,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

// ============================================================================
// Shared constructor helpers
// ============================================================================

/// Build a `Bearer {api_key}` authorization header.
///
/// Returns an error if the API key contains non-visible-ASCII characters.
pub fn build_bearer_auth(api_key: &str, provider: &str) -> Result<HeaderValue, LlmError> {
    HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|_| LlmError::RequestFailed {
        provider: provider.into(),
        reason: "API key contains invalid header characters".into(),
    })
}

/// Build the common components for an OpenAI-compatible LLM client.
///
/// Returns `(http_client, auth_header, resilience_executor)`.
pub fn build_llm_client(
    api_key: &str,
    provider: &str,
    timeout: Duration,
    metrics: Arc<ProviderMetrics>,
    streaming: bool,
    rate_limit_window: u32,
) -> Result<(reqwest::Client, HeaderValue, ResilientExecutor), LlmError> {
    let auth_header = build_bearer_auth(api_key, provider)?;

    // Streaming clients must NOT set a global timeout (it kills SSE streams).
    let http_timeout = if streaming { None } else { Some(timeout) };
    let http_client =
        build_http_client(http_timeout, 10).map_err(|reason| LlmError::RequestFailed {
            provider: provider.into(),
            reason,
        })?;

    let retry_config = RetryConfig::new(3)
        .with_initial_delay(Duration::from_millis(500))
        .with_max_delay(Duration::from_secs(30));

    let circuit_breaker = Arc::new(CircuitBreaker::new(
        provider,
        CircuitBreakerConfig::new(5)
            .with_open_duration(Duration::from_secs(30))
            .with_success_threshold(2),
    ));

    let resilience = ResilientExecutor::new(provider, retry_config, circuit_breaker, metrics)
        .with_timeout_secs(timeout.as_secs())
        .with_rate_limit_window(rate_limit_window);

    Ok((http_client, auth_header, resilience))
}

// ============================================================================
// Shared request execution
// ============================================================================

/// Execute an HTTP POST to an OpenAI-compatible chat endpoint.
///
/// Handles request errors (timeout, connection) and response status codes
/// (429 rate limiting, server errors) with proper error mapping.
pub async fn execute_chat_request(
    http_client: &reqwest::Client,
    auth_header: &HeaderValue,
    api_url: &str,
    provider: &str,
    timeout_secs: u64,
    request: &impl Serialize,
) -> HttpResult<reqwest::Response> {
    let response = with_request_id(
        http_client
            .post(api_url)
            .header("Authorization", auth_header.clone())
            .header("Content-Type", "application/json"),
    )
    .json(request)
    .send()
    .await
    .map_err(|e| {
        handle_request_error(&e, |kind, description| match kind {
            RequestErrorKind::Timeout => LlmError::Timeout {
                provider: provider.into(),
                timeout_secs,
            }
            .into(),
            _ => LlmError::RequestFailed {
                provider: provider.into(),
                reason: description,
            }
            .into(),
        })
    })?;

    check_response_status(response, provider, |code, body, retry_after| match code {
        429 => llm_rate_limited_err(provider, retry_after.unwrap_or(30)),
        _ => LlmError::RequestFailed {
            provider: provider.into(),
            reason: format!("API returned {code}: {body}"),
        }
        .into(),
    })
    .await
}

// ============================================================================
// Shared error helpers
// ============================================================================

/// Build a rate-limited `AppError` for an LLM provider.
///
/// Used as the `circuit_open_err` closure across all LLM clients, and in
/// `check_response_status` for 429 responses. Caps retry_after at 600s
/// (10 min) to prevent unreasonable waits from malformed headers.
pub fn llm_rate_limited_err(provider: &str, retry_secs: u64) -> AppError {
    AppError::from(LlmError::RateLimited {
        provider: provider.to_string(),
        retry_after: u32::try_from(retry_secs.min(600)).unwrap_or(u32::MAX),
    })
}

/// Parse a non-streaming OpenAI-compatible response and extract the content string.
pub async fn parse_completion_response(
    response: reqwest::Response,
    provider: &str,
) -> Result<String, AppError> {
    let body: serde_json::Value = response.json().await.map_err(|e| {
        AppError::from(LlmError::ResponseParseFailed {
            provider: provider.into(),
            reason: format!("Failed to parse {provider} response: {e}"),
        })
    })?;

    body["choices"]
        .get(0)
        .and_then(|c| c["message"]["content"].as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            AppError::from(LlmError::ResponseParseFailed {
                provider: provider.into(),
                reason: format!("No content in {provider} response"),
            })
        })
}
