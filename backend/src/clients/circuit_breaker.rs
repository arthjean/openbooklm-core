//! Circuit breaker pattern implementation.
//!
//! Protects against cascading failures by temporarily stopping requests
//! to unhealthy services. Implements the standard three-state pattern:
//! - **Closed**: Normal operation, requests pass through
//! - **Open**: Service is unhealthy, requests fail fast
//! - **Half-Open**: Testing if service has recovered

use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use parking_lot::RwLock;

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Closed => "closed",
            Self::Open => "open",
            Self::HalfOpen => "half-open",
        })
    }
}

/// Configuration for circuit breaker behavior.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub open_duration: Duration,
    pub failure_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            open_duration: Duration::from_secs(30),
            failure_window: Duration::from_secs(60),
        }
    }
}

impl CircuitBreakerConfig {
    #[must_use]
    pub fn new(failure_threshold: u32) -> Self {
        Self {
            failure_threshold,
            ..Default::default()
        }
    }

    #[must_use]
    pub fn with_success_threshold(mut self, threshold: u32) -> Self {
        self.success_threshold = threshold;
        self
    }

    #[must_use]
    pub fn with_open_duration(mut self, duration: Duration) -> Self {
        self.open_duration = duration;
        self
    }

    #[must_use]
    pub fn with_failure_window(mut self, duration: Duration) -> Self {
        self.failure_window = duration;
        self
    }
}

struct InnerState {
    state: CircuitState,
    opened_at: Option<Instant>,
    failure_window_start: Option<Instant>,
}

/// Circuit breaker for a single service.
#[derive(Clone)]
pub struct CircuitBreaker {
    config: Arc<CircuitBreakerConfig>,
    service_name: Arc<str>,
    inner: Arc<RwLock<InnerState>>,
    failures: Arc<AtomicU32>,
    successes: Arc<AtomicU32>,
    total_requests: Arc<AtomicU64>,
    total_failures: Arc<AtomicU64>,
}

impl fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("CircuitBreaker")
            .field("service", &self.service_name)
            .field("state", &inner.state)
            .field("failures", &self.failures.load(Ordering::Relaxed))
            .field("successes", &self.successes.load(Ordering::Relaxed))
            .finish()
    }
}

/// Error when the circuit is open.
#[derive(Debug, Clone, thiserror::Error)]
#[error("Circuit breaker open for '{service}'. Retry after {}s.", .retry_after.as_secs())]
pub struct CircuitOpenError {
    pub service: Arc<str>,
    pub retry_after: Duration,
}

impl CircuitBreaker {
    pub fn new(service_name: impl Into<Arc<str>>, config: CircuitBreakerConfig) -> Self {
        Self {
            config: Arc::new(config),
            service_name: service_name.into(),
            inner: Arc::new(RwLock::new(InnerState {
                state: CircuitState::Closed,
                opened_at: None,
                failure_window_start: None,
            })),
            failures: Arc::new(AtomicU32::new(0)),
            successes: Arc::new(AtomicU32::new(0)),
            total_requests: Arc::new(AtomicU64::new(0)),
            total_failures: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn state(&self) -> CircuitState {
        self.maybe_transition_to_half_open();
        self.inner.read().state
    }

    /// Check if a request should be allowed.
    ///
    /// Returns `Ok(())` if request can proceed, `Err` if circuit is open.
    pub fn allow_request(&self) -> Result<(), CircuitOpenError> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.maybe_transition_to_half_open();

        let (state, opened_at) = {
            let inner = self.inner.read();
            (inner.state, inner.opened_at)
        };

        match state {
            CircuitState::Closed | CircuitState::HalfOpen => Ok(()),
            CircuitState::Open => {
                let retry_after = opened_at
                    .map(|t| self.config.open_duration.saturating_sub(t.elapsed()))
                    .unwrap_or(self.config.open_duration);

                Err(CircuitOpenError {
                    service: Arc::clone(&self.service_name),
                    retry_after,
                })
            }
        }
    }

    /// Record a successful request.
    pub fn record_success(&self) {
        let current = self.inner.read().state;

        match current {
            CircuitState::HalfOpen => {
                let count = self.successes.fetch_add(1, Ordering::Relaxed) + 1;
                if count >= self.config.success_threshold {
                    self.transition_to_closed();
                    tracing::info!(service = %self.service_name, "Circuit breaker closed after recovery");
                }
            }
            CircuitState::Closed => {
                self.failures.store(0, Ordering::Relaxed);
                self.inner.write().failure_window_start = None;
            }
            CircuitState::Open => {}
        }
    }

    fn transition_to_closed(&self) {
        {
            let mut inner = self.inner.write();
            inner.state = CircuitState::Closed;
            inner.opened_at = None;
            inner.failure_window_start = None;
        }
        self.failures.store(0, Ordering::Relaxed);
        self.successes.store(0, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_failure(&self) {
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        let current = self.inner.read().state;

        match current {
            CircuitState::Closed => self.handle_closed_failure(),
            CircuitState::HalfOpen => {
                self.transition_to_open("re-opened after failure in half-open");
            }
            CircuitState::Open => {}
        }
    }

    fn handle_closed_failure(&self) {
        let now = Instant::now();

        // Reset window if expired
        let window_expired = self
            .inner
            .read()
            .failure_window_start
            .map(|start| now.duration_since(start) > self.config.failure_window)
            .unwrap_or(true);

        if window_expired {
            self.failures.store(0, Ordering::Relaxed);
            self.inner.write().failure_window_start = Some(now);
        }

        let count = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        if count >= self.config.failure_threshold {
            {
                let mut inner = self.inner.write();
                inner.state = CircuitState::Open;
                inner.opened_at = Some(now);
            }

            tracing::warn!(
                service = %self.service_name,
                failure_count = count,
                open_duration_secs = self.config.open_duration.as_secs(),
                "Circuit breaker opened"
            );
        }
    }

    fn transition_to_open(&self, reason: &str) {
        {
            let mut inner = self.inner.write();
            inner.state = CircuitState::Open;
            inner.opened_at = Some(Instant::now());
        }
        self.successes.store(0, Ordering::Relaxed);

        tracing::warn!(service = %self.service_name, "Circuit breaker {reason}");
    }

    fn maybe_transition_to_half_open(&self) {
        let should_transition = {
            let inner = self.inner.read();
            inner.state == CircuitState::Open
                && inner
                    .opened_at
                    .is_some_and(|t| t.elapsed() >= self.config.open_duration)
        };

        if should_transition {
            let transitioned = {
                let mut inner = self.inner.write();
                // Double-check after acquiring write lock
                if inner.state == CircuitState::Open {
                    inner.state = CircuitState::HalfOpen;
                    true
                } else {
                    false
                }
            };
            if transitioned {
                self.successes.store(0, Ordering::Relaxed);
                self.failures.store(0, Ordering::Relaxed);
                tracing::info!(service = %self.service_name, "Circuit breaker half-open");
            }
        }
    }
}
