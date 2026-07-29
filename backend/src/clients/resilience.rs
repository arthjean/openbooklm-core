//! Shared HTTP resilience executor and helpers.
//!
//! Encapsulates the retry-with-backoff + circuit-breaker + metrics pattern
//! used across all HTTP clients. Replaces duplicated `execute_with_retry` /
//! `execute_with_resilience` logic (~400 lines) with a single implementation.
//!
//! Also provides shared HTTP error handling helpers (`handle_request_error`,
//! `check_response_status`) that eliminate ~150 lines of duplication across
//! the Anthropic, Mistral, Voyage, and Firecrawl clients (US-011).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::StreamExt;
use futures::stream::Stream;

use crate::error::{AppError, LlmError};
use crate::types::RequestContext;

use super::circuit_breaker::CircuitBreaker;
use super::metrics::{ProviderMetrics, RequestTimer};
use super::retry::{DefaultRetryPolicy, RetryConfig, RetryPolicy};

/// Result type for HTTP operations: `(error, status_code, is_retryable)`.
pub type HttpResult<T> = Result<T, (AppError, Option<u16>, bool)>;

/// Build a shared `reqwest::Client` with connection pooling and timeouts.
///
/// Pass `None` for `timeout` when building clients that consume SSE streams —
/// the global reqwest timeout covers the *entire* request lifecycle including
/// body reads, so it kills long-running streams. Streaming clients should
/// instead use [`wrap_sse_stream`] for per-chunk idle timeouts.
pub fn build_http_client(
    timeout: Option<Duration>,
    max_idle_per_host: usize,
) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .pool_idle_timeout(Duration::from_secs(90))
        .pool_max_idle_per_host(max_idle_per_host);

    if let Some(t) = timeout {
        builder = builder.timeout(t);
    }

    builder
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))
}

/// Add `X-Request-ID` header from the current task-local [`RequestContext`]
/// to an outgoing HTTP request builder.
///
/// If no request context is available (e.g., in background tasks), the builder
/// is returned unchanged.
pub fn with_request_id(builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match RequestContext::current().request_id {
        Some(id) => match reqwest::header::HeaderValue::from_str(&id) {
            Ok(value) => builder.header("x-request-id", value),
            Err(_) => builder,
        },
        None => builder,
    }
}

// ============================================================================
// Shared HTTP error handling (US-011)
// ============================================================================

/// Check whether an HTTP status code is retryable.
///
/// Retryable codes: 408 (Request Timeout), 429 (Too Many Requests),
/// 500-599 (Server Errors), and 529 (Anthropic overloaded).
pub fn is_retryable_status(code: u16) -> bool {
    matches!(code, 408 | 429 | 500..=599)
}

/// Describes the kind of `reqwest::Error` for domain-specific mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestErrorKind {
    /// The request timed out.
    Timeout,
    /// The connection could not be established.
    ConnectionFailed,
    /// Any other request error.
    Other,
}

/// Classify a `reqwest::Error` and produce an `HttpResult` error tuple.
///
/// The `map_err` closure receives the error kind and a human-readable
/// description so the caller can construct a domain-specific `AppError`
/// (e.g. `LlmError::Timeout`, `RagError::EmbeddingFailed`, etc.).
pub fn handle_request_error(
    e: &reqwest::Error,
    map_err: impl FnOnce(RequestErrorKind, String) -> AppError,
) -> (AppError, Option<u16>, bool) {
    let (kind, description) = if e.is_timeout() {
        (RequestErrorKind::Timeout, format!("Request timed out: {e}"))
    } else if e.is_connect() {
        (
            RequestErrorKind::ConnectionFailed,
            format!("Connection failed: {e}"),
        )
    } else {
        (RequestErrorKind::Other, format!("HTTP request failed: {e}"))
    };

    let is_retryable = matches!(
        kind,
        RequestErrorKind::Timeout | RequestErrorKind::ConnectionFailed
    );
    (map_err(kind, description), None, is_retryable)
}

/// Check an HTTP response status and return the response on success, or
/// consume it and produce an `HttpResult` error tuple on failure.
///
/// The `map_err` closure receives the status code, the error body text,
/// and an optional `Retry-After` value (seconds, parsed from the header
/// on 429 responses) so the caller can construct a domain-specific `AppError`.
pub async fn check_response_status(
    response: reqwest::Response,
    provider: &str,
    map_err: impl FnOnce(u16, String, Option<u64>) -> AppError,
) -> HttpResult<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let code = status.as_u16();

    // Extract Retry-After header before consuming the response body.
    let retry_after_secs = if code == 429 {
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
    } else {
        None
    };

    let body = response.text().await.unwrap_or_else(|e| {
        tracing::warn!(error = %e, "Failed to read error response body");
        "Unknown error".into()
    });

    // Sanitize body for the server-side log: strip control chars and cap length
    // to prevent log injection and excessive log volume from large error responses.
    let safe_log_body: String = body.chars().filter(|c| !c.is_control()).take(500).collect();
    tracing::error!(%status, error = %safe_log_body, provider, "API returned error");

    let err = map_err(code, body, retry_after_secs);
    Err((err, Some(code), is_retryable_status(code)))
}

// ============================================================================
// SSE stream wrapper with per-chunk idle timeout
// ============================================================================

/// Default wall-clock timeout for the entire LLM stream (5 minutes).
///
/// Override via `LLM_STREAM_WALL_CLOCK_TIMEOUT_SECS` env var.
fn stream_wall_clock_timeout() -> Duration {
    const MIN_WALL_CLOCK_SECS: u64 = 10;

    let secs = match std::env::var("LLM_STREAM_WALL_CLOCK_TIMEOUT_SECS") {
        Ok(v) => v.parse::<u64>().unwrap_or_else(|_| {
            tracing::warn!(value = %v, "Invalid LLM_STREAM_WALL_CLOCK_TIMEOUT_SECS, using default 300s");
            300
        }),
        Err(_) => 300,
    }
    .max(MIN_WALL_CLOCK_SECS);

    Duration::from_secs(secs)
}

/// Wrap a `reqwest::Response` byte stream with a per-chunk idle timeout
/// and a wall-clock timeout for the entire stream.
///
/// Each call to `.next()` on the inner stream must produce data within
/// `chunk_timeout`; if it doesn't, the stream yields a [`LlmError::StreamError`]
/// and terminates. Additionally, the entire stream is subject to a global
/// wall-clock timeout (default 300s, configurable via
/// `LLM_STREAM_WALL_CLOCK_TIMEOUT_SECS`).
pub fn wrap_sse_stream(
    response: reqwest::Response,
    chunk_timeout: Duration,
    resilience: ResilientExecutor,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, AppError>> + Send>> {
    use futures::stream::unfold;

    let inner = Box::pin(response.bytes_stream());
    let wall_clock = stream_wall_clock_timeout();
    let deadline = tokio::time::Instant::now() + wall_clock;

    let stream = unfold(
        Some((inner, chunk_timeout, resilience, deadline, wall_clock)),
        |state| async move {
            let (mut stream, timeout_dur, resilience, deadline, wall_clock) = state?;

            // Check wall-clock deadline before waiting for next chunk
            if tokio::time::Instant::now() >= deadline {
                let msg = format!(
                    "LLM stream exceeded wall-clock timeout of {}s",
                    wall_clock.as_secs()
                );
                resilience.record_stream_error(&msg);
                tracing::error!("{msg}");
                return Some((
                    Err(AppError::from(LlmError::StreamError {
                        provider: resilience.service_name.to_string(),
                        reason: msg,
                    })),
                    None,
                ));
            }

            match tokio::time::timeout(timeout_dur, stream.next()).await {
                Ok(Some(Ok(bytes))) => Some((
                    Ok(bytes),
                    Some((stream, timeout_dur, resilience, deadline, wall_clock)),
                )),
                Ok(Some(Err(e))) => {
                    resilience.record_stream_error(&e.to_string());
                    tracing::error!(error = %e, "LLM stream error");
                    Some((
                        Err(AppError::from(LlmError::StreamError {
                            provider: resilience.service_name.to_string(),
                            reason: e.to_string(),
                        })),
                        None,
                    ))
                }
                // Stream finished normally
                Ok(None) => None,
                // No data within the timeout window
                Err(_) => {
                    let msg = format!("No data received for {}s", timeout_dur.as_secs());
                    resilience.record_stream_error(&msg);
                    tracing::error!("{msg}");
                    Some((
                        Err(AppError::from(LlmError::StreamError {
                            provider: resilience.service_name.to_string(),
                            reason: msg,
                        })),
                        None,
                    ))
                }
            }
        },
    );

    Box::pin(stream)
}

/// Shared resilience executor with circuit breaker, retry, and metrics.
///
/// Use this to wrap any HTTP operation that needs retry + circuit breaker
/// + metrics without duplicating the loop boilerplate.
#[derive(Clone)]
pub struct ResilientExecutor {
    retry_config: RetryConfig,
    circuit_breaker: Arc<CircuitBreaker>,
    metrics: Arc<ProviderMetrics>,
    service_name: Arc<str>,
    timeout_secs: Option<u64>,
    rate_limit_window_secs: u32,
}

impl std::fmt::Debug for ResilientExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResilientExecutor")
            .field("service_name", &self.service_name)
            .field("circuit_breaker", &self.circuit_breaker)
            .finish_non_exhaustive()
    }
}

impl ResilientExecutor {
    /// Create a new executor with the given resilience configuration.
    pub fn new(
        service_name: impl Into<Arc<str>>,
        retry_config: RetryConfig,
        circuit_breaker: Arc<CircuitBreaker>,
        metrics: Arc<ProviderMetrics>,
    ) -> Self {
        Self {
            retry_config,
            circuit_breaker,
            metrics,
            service_name: service_name.into(),
            timeout_secs: None,
            rate_limit_window_secs: 60,
        }
    }

    /// Set the timeout (in seconds) for metrics recording.
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Set the rate limit window (in seconds) for metrics recording.
    pub fn with_rate_limit_window(mut self, secs: u32) -> Self {
        self.rate_limit_window_secs = secs;
        self
    }

    pub fn set_retry_config(&mut self, config: RetryConfig) {
        self.retry_config = config;
    }

    pub fn circuit_breaker(&self) -> &Arc<CircuitBreaker> {
        &self.circuit_breaker
    }

    pub fn set_circuit_breaker(&mut self, cb: Arc<CircuitBreaker>) {
        self.circuit_breaker = cb;
    }

    pub fn metrics(&self) -> &Arc<ProviderMetrics> {
        &self.metrics
    }

    /// Record a stream-level error in metrics and circuit breaker.
    pub fn record_stream_error(&self, error: &str) {
        self.metrics.record_failure(error, Duration::ZERO);
        self.circuit_breaker.record_failure();
    }

    /// Execute an operation with full resilience: circuit breaker check,
    /// retry with exponential backoff, and metrics recording.
    ///
    /// # Arguments
    ///
    /// * `circuit_open_err` – maps the circuit-breaker retry-after (seconds)
    ///   to a domain-specific `AppError` (e.g. `LlmError::RateLimited`).
    /// * `operation` – closure producing the HTTP request future.
    ///   Must return `HttpResult<T>` (Ok on success, `(AppError, Option<status>, retryable)` on error).
    pub async fn execute<T, F, Fut>(
        &self,
        circuit_open_err: impl FnOnce(u64) -> AppError,
        operation: F,
    ) -> Result<T, AppError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = HttpResult<T>>,
    {
        // Read request_id from task-local context for structured logging (US-007)
        let req_id = RequestContext::current().request_id;
        let request_id = req_id.as_deref().unwrap_or("-");

        // 1. Check circuit breaker
        self.circuit_breaker.allow_request().map_err(|e| {
            let retry_secs = e.retry_after.as_secs();
            tracing::warn!(
                service = %self.service_name,
                request_id,
                retry_after_secs = retry_secs,
                "Circuit breaker is open"
            );
            circuit_open_err(retry_secs)
        })?;

        // 2. Start metrics timer
        let timer = RequestTimer::start(self.metrics.clone());
        let policy = DefaultRetryPolicy;
        let mut attempt = 0u32;

        // 3. Retry loop
        loop {
            attempt += 1;

            match operation().await {
                Ok(result) => {
                    timer.success();
                    self.circuit_breaker.record_success();
                    return Ok(result);
                }
                Err((err, status, is_retryable)) => {
                    // Record timeout metrics
                    if let Some(secs) = self.timeout_secs
                        && status.is_none()
                        && is_retryable
                    {
                        let err_str = err.to_string();
                        if err_str.contains("timed out") || err_str.contains("timeout") {
                            self.metrics.record_timeout(secs);
                        }
                    }

                    // Rate limiting (429) — retry with Retry-After delay
                    // but do NOT record a circuit breaker failure, since
                    // 429s indicate the service is healthy but throttling.
                    if status == Some(429) {
                        self.metrics
                            .record_rate_limit(Some(self.rate_limit_window_secs));

                        // Exhausted retries → give up without tripping breaker
                        if attempt > self.retry_config.max_retries {
                            timer.failure(&err.to_string());
                            return Err(err);
                        }

                        // Use Retry-After from the error if available, else default
                        let delay = extract_retry_after(&err)
                            .unwrap_or_else(|| self.retry_config.delay_for_attempt(attempt - 1));

                        tracing::warn!(
                            attempt,
                            max_retries = self.retry_config.max_retries,
                            delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                            service = %self.service_name,
                            request_id,
                            "Rate limited (429), retrying after delay"
                        );

                        tokio::time::sleep(delay).await;
                        continue;
                    }

                    // Non-retryable or last attempt — bail
                    if !is_retryable || attempt > self.retry_config.max_retries {
                        timer.failure(&err.to_string());
                        self.circuit_breaker.record_failure();
                        return Err(err);
                    }

                    // Exponential backoff with jitter
                    let delay = policy
                        .get_retry_delay(status, None)
                        .unwrap_or_else(|| self.retry_config.delay_for_attempt(attempt - 1));

                    tracing::warn!(
                        attempt,
                        max_retries = self.retry_config.max_retries,
                        delay_ms = u64::try_from(delay.as_millis()).unwrap_or(u64::MAX),
                        ?status,
                        error = %err,
                        service = %self.service_name,
                        request_id,
                        "Retrying request"
                    );

                    tokio::time::sleep(delay).await;
                }
            }
        }
    }
}

/// Extract a retry-after duration from an `AppError`.
///
/// Looks for the `retry_after` field in `AppError::RateLimited` or
/// `LlmError::RateLimited`. Returns `None` if the error doesn't
/// carry a retry-after value, in which case the caller should fall
/// back to exponential backoff.
fn extract_retry_after(err: &AppError) -> Option<Duration> {
    match err {
        AppError::RateLimited { retry_after, .. } => {
            Some(Duration::from_secs(u64::from(*retry_after)))
        }
        AppError::Llm(crate::error::LlmError::RateLimited { retry_after, .. }) => {
            Some(Duration::from_secs(u64::from(*retry_after)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::circuit_breaker::{CircuitBreakerConfig, CircuitState};
    use std::sync::atomic::{AtomicU32, Ordering};

    /// 3 consecutive 429s must NOT open the circuit breaker.
    #[tokio::test]
    async fn consecutive_429s_do_not_open_circuit_breaker() {
        let cb = Arc::new(CircuitBreaker::new(
            "test",
            CircuitBreakerConfig::new(3) // opens after 3 failures
                .with_open_duration(Duration::from_secs(30))
                .with_success_threshold(1),
        ));
        let metrics = Arc::new(ProviderMetrics::new("test"));
        let executor = ResilientExecutor::new(
            "test",
            RetryConfig::new(3)
                .with_initial_delay(Duration::from_millis(1))
                .with_max_delay(Duration::from_millis(5))
                .without_jitter(),
            cb.clone(),
            metrics.clone(),
        );

        // Counter to track how many times the operation is called
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        // Operation that always returns 429
        let result: Result<(), AppError> = executor
            .execute(
                |_| AppError::Internal("circuit open".into()),
                || {
                    let cc = call_count_clone.clone();
                    async move {
                        cc.fetch_add(1, Ordering::SeqCst);
                        let err = AppError::from(crate::error::LlmError::RateLimited {
                            provider: "test".into(),
                            retry_after: 0, // zero-delay for test speed
                        });
                        Err::<(), (AppError, Option<u16>, bool)>((err, Some(429), true))
                    }
                },
            )
            .await;

        // Should have retried (initial + 3 retries = 4 calls total)
        assert_eq!(call_count.load(Ordering::SeqCst), 4);
        // Should have failed (exhausted retries)
        assert!(result.is_err());
        // Circuit breaker must still be CLOSED — 429s don't count as failures
        assert_eq!(cb.state(), CircuitState::Closed);
        // Rate limit metrics should be recorded
        assert_eq!(metrics.rate_limit_hits(), 4);
    }

    /// Non-429 errors still open the circuit breaker.
    /// Each failed execution records one circuit breaker failure (after
    /// exhausting retries). Two executions with threshold=2 trips the breaker.
    #[tokio::test]
    async fn server_errors_open_circuit_breaker() {
        let cb = Arc::new(CircuitBreaker::new(
            "test",
            CircuitBreakerConfig::new(2) // opens after 2 failures
                .with_open_duration(Duration::from_secs(30))
                .with_success_threshold(1),
        ));
        let metrics = Arc::new(ProviderMetrics::new("test"));
        let executor = ResilientExecutor::new(
            "test",
            RetryConfig::new(0) // no retries — fail immediately
                .with_initial_delay(Duration::from_millis(1))
                .with_max_delay(Duration::from_millis(5))
                .without_jitter(),
            cb.clone(),
            metrics,
        );

        // First failure — circuit still closed
        let result: Result<(), AppError> = executor
            .execute(
                |_| AppError::Internal("circuit open".into()),
                || async {
                    let err = AppError::Internal("server error".into());
                    Err::<(), (AppError, Option<u16>, bool)>((err, Some(500), true))
                },
            )
            .await;
        assert!(result.is_err());
        assert_eq!(cb.state(), CircuitState::Closed);

        // Second failure — circuit opens (threshold = 2)
        let result: Result<(), AppError> = executor
            .execute(
                |_| AppError::Internal("circuit open".into()),
                || async {
                    let err = AppError::Internal("server error".into());
                    Err::<(), (AppError, Option<u16>, bool)>((err, Some(500), true))
                },
            )
            .await;
        assert!(result.is_err());
        // 500s should have tripped the circuit breaker open
        assert_eq!(cb.state(), CircuitState::Open);
    }
}
