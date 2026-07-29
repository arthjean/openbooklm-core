//! Client metrics collection
//!
//! Tracks latency, error rates, and rate limit hits per provider.
//! Uses lock-free atomics for counters with tracing integration for observability.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Sentinel value for uninitialized minimum latency
const MIN_LATENCY_UNSET: u64 = u64::MAX;

/// Memory ordering for metrics (relaxed is sufficient for statistics)
const ORD: Ordering = Ordering::Relaxed;

/// Metrics for a single provider
#[derive(Debug)]
pub struct ProviderMetrics {
    pub name: String,
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    rate_limit_hits: AtomicU64,
    timeout_errors: AtomicU64,
    total_duration_ms: AtomicU64,
    duration_request_count: AtomicU64,
    min_latency_ms: AtomicU64,
    max_latency_ms: AtomicU64,
    last_error: RwLock<Option<String>>,
}

impl ProviderMetrics {
    /// Create new metrics for a provider
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            rate_limit_hits: AtomicU64::new(0),
            timeout_errors: AtomicU64::new(0),
            total_duration_ms: AtomicU64::new(0),
            duration_request_count: AtomicU64::new(0),
            min_latency_ms: AtomicU64::new(MIN_LATENCY_UNSET),
            max_latency_ms: AtomicU64::new(0),
            last_error: RwLock::new(None),
        }
    }

    /// Record a request start
    pub fn record_request(&self) {
        self.total_requests.fetch_add(1, ORD);
    }

    /// Record a successful request with duration
    pub fn record_success(&self, duration: Duration) {
        self.successful_requests.fetch_add(1, ORD);
        self.record_duration(duration);

        tracing::debug!(
            provider = %self.name,
            duration_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
            "Request succeeded"
        );
    }

    /// Record a failed request
    pub fn record_failure(&self, error: &str, duration: Duration) {
        self.failed_requests.fetch_add(1, ORD);
        self.record_duration(duration);
        *self.last_error.write() = Some(error.to_string());

        tracing::warn!(
            provider = %self.name,
            duration_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX),
            error,
            "Request failed"
        );
    }

    /// Record a rate limit hit
    pub fn record_rate_limit(&self, retry_after: Option<u32>) {
        self.rate_limit_hits.fetch_add(1, ORD);
        self.failed_requests.fetch_add(1, ORD);

        tracing::warn!(provider = %self.name, ?retry_after, "Rate limit hit");
    }

    /// Record a timeout error
    pub fn record_timeout(&self, timeout_secs: u64) {
        self.timeout_errors.fetch_add(1, ORD);
        self.failed_requests.fetch_add(1, ORD);

        tracing::warn!(provider = %self.name, timeout_secs, "Request timed out");
    }

    /// Record request duration and update min/max latencies
    fn record_duration(&self, duration: Duration) {
        let ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
        self.total_duration_ms.fetch_add(ms, ORD);
        self.duration_request_count.fetch_add(1, ORD);

        // Update min/max using fetch_min/fetch_max (lock-free)
        self.min_latency_ms.fetch_min(ms, ORD);
        self.max_latency_ms.fetch_max(ms, ORD);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Getters
    // ─────────────────────────────────────────────────────────────────────────────

    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(ORD)
    }

    pub fn successful_requests(&self) -> u64 {
        self.successful_requests.load(ORD)
    }

    pub fn failed_requests(&self) -> u64 {
        self.failed_requests.load(ORD)
    }

    pub fn rate_limit_hits(&self) -> u64 {
        self.rate_limit_hits.load(ORD)
    }

    pub fn timeout_errors(&self) -> u64 {
        self.timeout_errors.load(ORD)
    }

    /// Get average latency in milliseconds
    pub fn avg_latency_ms(&self) -> Option<u64> {
        let count = self.duration_request_count.load(ORD);
        (count > 0).then(|| self.total_duration_ms.load(ORD) / count)
    }

    /// Get minimum latency in milliseconds
    pub fn min_latency_ms(&self) -> Option<u64> {
        let min = self.min_latency_ms.load(ORD);
        (min != MIN_LATENCY_UNSET).then_some(min)
    }

    /// Get maximum latency in milliseconds
    pub fn max_latency_ms(&self) -> Option<u64> {
        let max = self.max_latency_ms.load(ORD);
        // max is 0 only if no requests recorded (count check for edge case of 0ms latency)
        (max > 0 || self.duration_request_count.load(ORD) > 0).then_some(max)
    }

    /// Get error rate (0.0 to 1.0)
    pub fn error_rate(&self) -> f64 {
        let total = self.total_requests.load(ORD);
        if total == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.failed_requests.load(ORD) as f64 / total as f64
        }
    }

    /// Get last error message
    pub fn last_error(&self) -> Option<String> {
        self.last_error.read().clone()
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            provider: self.name.clone(),
            total_requests: self.total_requests(),
            successful_requests: self.successful_requests(),
            failed_requests: self.failed_requests(),
            rate_limit_hits: self.rate_limit_hits(),
            timeout_errors: self.timeout_errors(),
            avg_latency_ms: self.avg_latency_ms(),
            min_latency_ms: self.min_latency_ms(),
            max_latency_ms: self.max_latency_ms(),
            error_rate: self.error_rate(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.total_requests.store(0, ORD);
        self.successful_requests.store(0, ORD);
        self.failed_requests.store(0, ORD);
        self.rate_limit_hits.store(0, ORD);
        self.timeout_errors.store(0, ORD);
        self.total_duration_ms.store(0, ORD);
        self.duration_request_count.store(0, ORD);
        self.min_latency_ms.store(MIN_LATENCY_UNSET, ORD);
        self.max_latency_ms.store(0, ORD);
        *self.last_error.write() = None;
    }
}

/// Snapshot of metrics for reporting
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub provider: String,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub rate_limit_hits: u64,
    pub timeout_errors: u64,
    pub avg_latency_ms: Option<u64>,
    pub min_latency_ms: Option<u64>,
    pub max_latency_ms: Option<u64>,
    pub error_rate: f64,
}

/// Centralized metrics collection for all clients
#[derive(Clone, Default)]
pub struct ClientMetrics {
    providers: Arc<RwLock<HashMap<String, Arc<ProviderMetrics>>>>,
}

impl ClientMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create metrics for a provider
    pub fn provider(&self, name: &str) -> Arc<ProviderMetrics> {
        // Fast path: read lock
        if let Some(metrics) = self.providers.read().get(name) {
            return metrics.clone();
        }

        // Slow path: write lock with entry API
        self.providers
            .write()
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(ProviderMetrics::new(name)))
            .clone()
    }

    /// Get all provider snapshots
    pub fn all_snapshots(&self) -> Vec<MetricsSnapshot> {
        self.providers
            .read()
            .values()
            .map(|m| m.snapshot())
            .collect()
    }

    /// Log all metrics (for periodic reporting)
    pub fn log_all(&self) {
        for (name, metrics) in self.providers.read().iter() {
            let s = metrics.snapshot();
            tracing::info!(
                provider = %name,
                total_requests = s.total_requests,
                successful = s.successful_requests,
                failed = s.failed_requests,
                rate_limits = s.rate_limit_hits,
                timeouts = s.timeout_errors,
                avg_latency_ms = ?s.avg_latency_ms,
                error_rate = format!("{:.2}%", s.error_rate * 100.0),
                "Client metrics"
            );
        }
    }

    /// Reset all provider metrics
    pub fn reset_all(&self) {
        for metrics in self.providers.read().values() {
            metrics.reset();
        }
    }
}

/// Timer helper for measuring request duration
///
/// Automatically records failure if dropped without explicit success/failure call.
pub struct RequestTimer {
    start: Instant,
    metrics: Arc<ProviderMetrics>,
    recorded: bool,
}

impl RequestTimer {
    /// Start timing a request
    pub fn start(metrics: Arc<ProviderMetrics>) -> Self {
        metrics.record_request();
        Self {
            start: Instant::now(),
            metrics,
            recorded: false,
        }
    }

    /// Record success
    pub fn success(mut self) {
        self.metrics.record_success(self.start.elapsed());
        self.recorded = true;
    }

    /// Record failure
    pub fn failure(mut self, error: &str) {
        self.metrics.record_failure(error, self.start.elapsed());
        self.recorded = true;
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        if !self.recorded {
            self.metrics
                .record_failure("Request dropped without recording", self.start.elapsed());
        }
    }
}
