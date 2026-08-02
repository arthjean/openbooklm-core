//! Voyage AI rate limiter for Tier 1 compliance.
//!
//! Implements a sliding-window rate limiter that enforces both RPM (requests
//! per minute) and TPM (tokens per minute) limits. Defaults are set for
//! Voyage AI Tier 1 (payment method added): 300 RPM and 2M TPM, with
//! headroom below the hard limits of 2,000 RPM and 8M TPM.
//!
//! The limiter blocks (`acquire`) until budget is available or times out.

use std::collections::VecDeque;
use std::time::Duration;

use tokio::sync::{Mutex, Notify};
use tokio::time::Instant;

use crate::core::config::VoyageRateLimitConfig;
use crate::error::{AppError, RagError};

/// Sliding-window rate limiter for Voyage AI requests.
pub struct VoyageRateLimiter {
    /// Timestamps of recent requests (sliding window for RPM).
    request_times: Mutex<VecDeque<Instant>>,
    /// (timestamp, token_count) pairs for recent requests (sliding window for TPM).
    token_usage: Mutex<VecDeque<(Instant, u32)>>,
    /// Configuration (RPM, TPM limits, max wait).
    config: VoyageRateLimitConfig,
    /// Notified when budget may have freed up (after window slides).
    notify: Notify,
}

/// Current usage statistics for monitoring.
#[derive(Debug, Clone)]
pub struct VoyageUsageStats {
    pub rpm_used: u32,
    pub rpm_remaining: u32,
    pub tpm_used: u32,
    pub tpm_remaining: u32,
}

impl VoyageRateLimiter {
    /// Create a new rate limiter from configuration.
    pub fn new(config: VoyageRateLimitConfig) -> Self {
        Self {
            request_times: Mutex::new(VecDeque::new()),
            token_usage: Mutex::new(VecDeque::new()),
            config,
            notify: Notify::new(),
        }
    }

    /// Wait until both RPM and TPM budgets allow a request with `estimated_tokens`.
    ///
    /// Returns the total time spent waiting. Fails with `EmbeddingRateLimited` if
    /// the wait would exceed `max_wait_secs`.
    #[allow(clippy::significant_drop_tightening)] // locks are held for atomic read-modify-write
    pub async fn acquire(&self, estimated_tokens: u32) -> Result<Duration, AppError> {
        let start = Instant::now();
        let max_wait = Duration::from_secs(self.config.max_wait_secs);
        let window = Duration::from_secs(60);

        loop {
            let now = Instant::now();
            let elapsed = now.duration_since(start);
            if elapsed >= max_wait {
                return Err(AppError::from(RagError::EmbeddingRateLimited {
                    reason: format!(
                        "Rate limit wait exceeded {}s (RPM={}, TPM={})",
                        self.config.max_wait_secs, self.config.max_rpm, self.config.max_tpm
                    ),
                    wait_secs: u32::try_from(self.config.max_wait_secs).unwrap_or(u32::MAX),
                }));
            }

            // Check RPM
            let cutoff = now.checked_sub(window).unwrap_or(now);
            let rpm_ok = {
                let mut times = self.request_times.lock().await;
                while times.front().is_some_and(|t| *t < cutoff) {
                    times.pop_front();
                }
                times.len() < self.config.max_rpm as usize
            };

            // Check TPM
            let tpm_ok = {
                let mut usage = self.token_usage.lock().await;
                while usage.front().is_some_and(|(t, _)| *t < cutoff) {
                    usage.pop_front();
                }
                let current_tpm: u32 = usage.iter().map(|(_, tokens)| tokens).sum();
                current_tpm + estimated_tokens <= self.config.max_tpm
            };

            if rpm_ok && tpm_ok {
                // Reserve the RPM slot now
                {
                    let mut times = self.request_times.lock().await;
                    times.push_back(now);
                }
                let waited = now.duration_since(start);
                if waited > Duration::from_millis(100) {
                    tracing::info!(
                        waited_ms = waited.as_millis(),
                        estimated_tokens,
                        "Rate limiter released"
                    );
                }
                return Ok(waited);
            }

            // Calculate how long until the oldest entry in the window expires
            let sleep_until = {
                let times = self.request_times.lock().await;
                let usage = self.token_usage.lock().await;

                let rpm_wait = if rpm_ok {
                    None
                } else {
                    times.front().map(|t| {
                        if *t > cutoff {
                            (*t + window).saturating_duration_since(now)
                        } else {
                            Duration::from_millis(100)
                        }
                    })
                };

                let tpm_wait = if tpm_ok {
                    None
                } else {
                    usage.front().map(|(t, _)| {
                        if *t > cutoff {
                            (*t + window).saturating_duration_since(now)
                        } else {
                            Duration::from_millis(100)
                        }
                    })
                };

                match (rpm_wait, tpm_wait) {
                    (Some(a), Some(b)) => a.min(b),
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => Duration::from_millis(100),
                }
            };

            let remaining_budget = max_wait.saturating_sub(now.duration_since(start));
            let actual_sleep = sleep_until.min(remaining_budget);

            tracing::debug!(
                sleep_ms = actual_sleep.as_millis(),
                rpm_ok,
                tpm_ok,
                "Rate limiter waiting for budget"
            );

            // Wait for either the sleep duration or a notification that budget freed
            tokio::select! {
                () = tokio::time::sleep(actual_sleep) => {},
                () = self.notify.notified() => {},
            }
        }
    }

    /// Record actual token usage after a successful request.
    ///
    /// The Voyage AI response includes `usage.total_tokens` which may differ from
    /// the estimate. Recording actual usage improves accuracy of TPM tracking.
    pub async fn record_usage(&self, actual_tokens: u32) {
        let now = Instant::now();
        {
            let mut usage = self.token_usage.lock().await;
            usage.push_back((now, actual_tokens));
        }

        // Notify waiters that the window state changed
        self.notify.notify_waiters();
    }

    /// Get current usage statistics.
    pub async fn stats(&self) -> VoyageUsageStats {
        let now = Instant::now();
        let window = Duration::from_secs(60);
        let cutoff = now.checked_sub(window).unwrap_or(now);

        let rpm_used = {
            let times = self.request_times.lock().await;
            u32::try_from(times.iter().filter(|t| **t >= cutoff).count()).unwrap_or(u32::MAX)
        };

        let tpm_used = {
            let usage = self.token_usage.lock().await;
            usage
                .iter()
                .filter(|(t, _)| *t >= cutoff)
                .map(|(_, tokens)| tokens)
                .sum()
        };

        VoyageUsageStats {
            rpm_used,
            rpm_remaining: self.config.max_rpm.saturating_sub(rpm_used),
            tpm_used,
            tpm_remaining: self.config.max_tpm.saturating_sub(tpm_used),
        }
    }
}

/// Estimate token count for a batch of texts.
///
/// Uses a simple heuristic: ~4 characters per token with a 10% safety margin.
/// This avoids pulling in a full tokenizer dependency for estimation.
pub fn estimate_tokens(texts: &[String]) -> u32 {
    let total_chars: usize = texts.iter().map(String::len).sum();
    // ~4 chars/token is a reasonable approximation for English text
    // Add 10% margin for safety
    #[allow(clippy::cast_precision_loss)]
    let estimated_f = (total_chars as f64 / 4.0 * 1.1).ceil();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let estimated = estimated_f as u32;
    // Minimum 1 token per text
    estimated.max(u32::try_from(texts.len()).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_basic() {
        let texts = vec!["Hello world".to_string()]; // 11 chars
        let tokens = estimate_tokens(&texts);
        // 11 / 4 * 1.1 = 3.025 → ceil = 4
        assert_eq!(tokens, 4);
    }

    #[test]
    fn estimate_tokens_empty() {
        let texts: Vec<String> = vec![];
        assert_eq!(estimate_tokens(&texts), 0);
    }

    #[test]
    fn estimate_tokens_multiple_texts() {
        let texts = vec![
            "Hello world".to_string(),       // 11 chars
            "How are you doing".to_string(), // 17 chars
        ];
        let tokens = estimate_tokens(&texts);
        // (11 + 17) / 4 * 1.1 = 7.7 → ceil = 8
        assert_eq!(tokens, 8);
    }

    #[test]
    fn estimate_tokens_minimum_one_per_text() {
        let texts = vec!["a".to_string()]; // 1 char
        let tokens = estimate_tokens(&texts);
        // 1 / 4 * 1.1 = 0.275 → ceil = 1, max(1, 1) = 1
        assert_eq!(tokens, 1);
    }

    #[tokio::test]
    async fn acquire_succeeds_under_limit() {
        let config = VoyageRateLimitConfig {
            max_rpm: 300,
            max_tpm: 2_000_000,
            batch_size: 128,
            concurrent_batches: 4,
            max_wait_secs: 5,
        };
        let limiter = VoyageRateLimiter::new(config);
        let result = limiter.acquire(100).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn stats_tracks_usage() {
        let config = VoyageRateLimitConfig {
            max_rpm: 300,
            max_tpm: 2_000_000,
            batch_size: 128,
            concurrent_batches: 4,
            max_wait_secs: 5,
        };
        let limiter = VoyageRateLimiter::new(config);

        // Acquire and record usage
        limiter.acquire(100).await.unwrap();
        limiter.record_usage(95).await;

        let stats = limiter.stats().await;
        assert_eq!(stats.rpm_used, 1);
        assert_eq!(stats.tpm_used, 95);
        assert_eq!(stats.rpm_remaining, 299);
    }

    #[tokio::test(start_paused = true)]
    async fn acquire_blocks_at_rpm_limit() {
        let config = VoyageRateLimitConfig {
            max_rpm: 2,
            max_tpm: 2_000_000,
            batch_size: 128,
            concurrent_batches: 4,
            max_wait_secs: 1,
        };
        let limiter = VoyageRateLimiter::new(config);

        // Exhaust RPM budget
        limiter.acquire(10).await.unwrap();
        limiter.acquire(10).await.unwrap();

        // Tokio's paused clock advances to the limiter deadline without making
        // the default offline suite sleep for a wall-clock second.
        let result = limiter.acquire(10).await;
        assert!(result.is_err());
    }
}
