//! Retry logic with exponential backoff
//!
//! Implements retry patterns for transient failures (5xx, timeouts, connection errors).
//! Uses exponential backoff with jitter to avoid thundering herd.

use std::time::Duration;

use rand::Rng;

/// Default jitter factor (adds 0-25% random delay)
const JITTER_MAX_FACTOR: f64 = 0.25;

/// Default rate limit retry delay in seconds
const DEFAULT_RATE_LIMIT_DELAY_SECS: u64 = 30;

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not including the initial attempt)
    pub max_retries: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a new retry configuration with custom max retries
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Set the initial delay
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Set the maximum delay
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set the backoff multiplier
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Disable jitter
    pub fn without_jitter(mut self) -> Self {
        self.jitter = false;
        self
    }

    /// Calculate the delay for a given attempt (0-indexed)
    ///
    /// Uses exponential backoff: `initial_delay * multiplier^attempt`, capped at `max_delay`.
    /// Optionally adds up to 25% jitter to prevent thundering herd.
    #[allow(clippy::cast_precision_loss)] // ms values are small enough for f64
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_ms = self.initial_delay.as_millis() as f64;
        let max_ms = self.max_delay.as_millis() as f64;

        #[allow(clippy::cast_possible_wrap)]
        let delay_ms = (base_ms * self.backoff_multiplier.powi(attempt as i32)).min(max_ms);

        let final_ms = if self.jitter {
            delay_ms * (1.0 + rand::thread_rng().gen_range(0.0..JITTER_MAX_FACTOR))
        } else {
            delay_ms
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        Duration::from_millis(final_ms as u64)
    }
}

/// Retry policy for determining delay calculation from response metadata.
pub trait RetryPolicy {
    /// Get retry delay from response (e.g., from Retry-After header)
    fn get_retry_delay(&self, status: Option<u16>, headers: Option<&str>) -> Option<Duration>;
}

/// Default retry policy for HTTP errors
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultRetryPolicy;

impl RetryPolicy for DefaultRetryPolicy {
    fn get_retry_delay(&self, status: Option<u16>, headers: Option<&str>) -> Option<Duration> {
        // Only provide delay for rate limiting (429)
        if status != Some(429) {
            return None;
        }

        // Try to parse Retry-After header, fallback to default
        let secs = headers
            .and_then(|h| h.parse::<u64>().ok())
            .unwrap_or(DEFAULT_RATE_LIMIT_DELAY_SECS);

        Some(Duration::from_secs(secs))
    }
}
