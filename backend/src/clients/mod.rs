//! External API clients module (Story B6)
//!
//! This module provides resilient client implementations for external services:
//! - Mistral AI (LLM + memory extraction)
//! - Anthropic Claude (LLM)
//! - OpenAI (LLM)
//! - Voyage AI (Embeddings)
//! - Firecrawl (Web scraping)
//!
//! All clients implement:
//! - Exponential backoff retry logic
//! - Circuit breaker pattern for fault tolerance
//! - Connection pooling via shared reqwest::Client
//! - Per-provider metrics (latency, errors, rate limits)
//! - Configurable timeouts
//!
//! LLM clients also implement the `LlmProvider` trait for unified access.

/// Generate the standard resilience builder/accessor methods for an HTTP client.
///
/// Every client struct that wraps a [`ResilientExecutor`] in a `resilience` field
/// needs the same four methods: two builder methods to override retry/circuit-breaker
/// config, and two accessors for current state and metrics.  This macro eliminates
/// the duplication (~20 lines per client, ~80 lines total across 4 clients).
///
/// # Arguments
///
/// * `$client` -- the client struct name (e.g. `AnthropicClient`)
/// * `$provider` -- a string literal used as the circuit-breaker service name
///   (e.g. `"anthropic"`)
///
/// # Generated methods
///
/// ```text
/// impl $client {
///     pub fn with_retry_config(mut self, config: RetryConfig) -> Self;
///     pub fn with_circuit_breaker(mut self, config: CircuitBreakerConfig) -> Self;
///     pub fn circuit_state(&self) -> CircuitState;
///     pub fn metrics(&self) -> MetricsSnapshot;
/// }
/// ```
///
/// # Example
///
/// ```text
/// use crate::clients::impl_client_resilience_methods;
///
/// pub struct MyClient {
///     resilience: ResilientExecutor,
/// }
///
/// impl_client_resilience_methods!(MyClient, "my_service");
/// ```
macro_rules! impl_client_resilience_methods {
    ($client:ty, $provider:expr) => {
        impl $client {
            /// Override the retry configuration (builder pattern).
            #[must_use]
            pub fn with_retry_config(mut self, config: $crate::clients::RetryConfig) -> Self {
                self.resilience.set_retry_config(config);
                self
            }

            /// Override the circuit breaker configuration (builder pattern).
            #[must_use]
            pub fn with_circuit_breaker(
                mut self,
                config: $crate::clients::CircuitBreakerConfig,
            ) -> Self {
                self.resilience.set_circuit_breaker(::std::sync::Arc::new(
                    $crate::clients::CircuitBreaker::new($provider, config),
                ));
                self
            }

            /// Current circuit breaker state (`Closed`, `Open`, or `HalfOpen`).
            pub fn circuit_state(&self) -> $crate::clients::CircuitState {
                self.resilience.circuit_breaker().state()
            }

            /// Snapshot of request metrics (latency, error rate, etc.).
            pub fn metrics(&self) -> $crate::clients::metrics::MetricsSnapshot {
                self.resilience.metrics().snapshot()
            }
        }
    };
}
pub(crate) use impl_client_resilience_methods;

mod anthropic;
mod circuit_breaker;
mod firecrawl;
mod llm_router;
mod metrics;
mod mistral;
mod mistral_ocr;
pub mod models;
mod openai;
mod openai_compat;
mod resilience;
mod retry;
mod voyage;
pub mod voyage_rate_limiter;
pub mod youtube;

// Public API - clients
pub use anthropic::{
    API_URL as ANTHROPIC_API_URL, API_VERSION as ANTHROPIC_API_VERSION, AnthropicClient,
    AnthropicMessagesClient, MessagesContentBlock, MessagesRequest, MessagesRequestMessage,
    MessagesResponse, MessagesUsage, anthropic_headers,
};
pub use firecrawl::FirecrawlClient;
pub use llm_router::{LlmRouter, ProviderSelection};
pub use mistral::MistralClient;
pub use mistral_ocr::{MistralOcrClient, OcrPage, OcrResult};
pub use openai::OpenAiClient;
pub use voyage::{RerankedDocument, VoyageClient};
pub use voyage_rate_limiter::VoyageRateLimiter;
pub use youtube::YouTubeClient;

// Resilience infrastructure — pub(crate) since these are implementation details
// used only within this crate. Types used by the impl_client_resilience_methods!
// macro or by modules outside clients/ (app_state, services).
// The remaining resilience types (HttpResult, ResilientExecutor, build_http_client,
// etc.) are only used within clients/* via `super::` imports.
pub(crate) use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use metrics::ProviderMetrics;
pub(crate) use metrics::{ClientMetrics, MetricsSnapshot};
pub(crate) use retry::RetryConfig;
