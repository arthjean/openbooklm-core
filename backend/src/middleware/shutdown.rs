//! Graceful shutdown module (B4.4, B4.5)
//!
//! Provides task tracking, active stream counting, and signal handling
//! for graceful shutdown with in-flight request draining.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio_util::{sync::CancellationToken, task::TaskTracker as TokioTaskTracker};

/// Coordinated shutdown manager combining task tracking with cancellation.
#[derive(Clone)]
pub struct ShutdownCoordinator {
    tracker: TokioTaskTracker,
    cancel_token: CancellationToken,
    active_streams: std::sync::Arc<AtomicUsize>,
}

impl ShutdownCoordinator {
    pub fn new() -> Self {
        Self {
            tracker: TokioTaskTracker::new(),
            cancel_token: CancellationToken::new(),
            active_streams: std::sync::Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Get the cancellation token for cooperative cancellation.
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Increment the active stream counter (called when an SSE stream starts).
    pub fn stream_started(&self) {
        self.active_streams.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the active stream counter (called when an SSE stream ends).
    pub fn stream_ended(&self) {
        self.active_streams.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get count of active SSE streams.
    pub fn active_stream_count(&self) -> usize {
        self.active_streams.load(Ordering::Relaxed)
    }

    /// Spawn a tracked task that respects cancellation.
    pub fn spawn<F>(&self, name: &'static str, future: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let cancel_token = self.cancel_token.clone();

        self.tracker.spawn(async move {
            tokio::select! {
                biased;
                () = cancel_token.cancelled() => {
                    tracing::debug!(task = name, "Task cancelled");
                }
                () = future => {
                    tracing::debug!(task = name, "Task completed");
                }
            }
        });
    }

    /// Get count of active tracked background tasks.
    pub fn task_count(&self) -> usize {
        self.tracker.len()
    }

    /// Initiate shutdown: cancel all tasks and wait with timeout.
    pub async fn shutdown(&self, timeout: Duration) {
        let active_tasks = self.task_count();
        let active_streams = self.active_stream_count();

        tracing::info!(
            active_tasks,
            active_streams,
            timeout_secs = timeout.as_secs(),
            "Initiating graceful shutdown, waiting for in-flight work to complete..."
        );

        self.cancel_token.cancel();
        self.tracker.close();

        let start = std::time::Instant::now();

        if tokio::time::timeout(timeout, self.tracker.wait())
            .await
            .is_err()
        {
            tracing::warn!(
                remaining_tasks = self.task_count(),
                remaining_streams = self.active_stream_count(),
                "Shutdown timeout reached, forcefully terminating remaining tasks"
            );
        }

        tracing::info!(
            elapsed_ms = start.elapsed().as_millis(),
            "Shutdown complete"
        );
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Backward-compatible alias.
pub type TaskTracker = ShutdownCoordinator;

/// Wait for shutdown signal (SIGTERM or SIGINT) (B4.5)
pub async fn shutdown_signal() {
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to install SIGTERM handler, only SIGINT will be available");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received SIGINT, initiating shutdown");
        }
        () = terminate => {
            tracing::info!("Received SIGTERM, initiating shutdown");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_counter_tracks_active_streams() {
        let coordinator = ShutdownCoordinator::new();
        assert_eq!(coordinator.active_stream_count(), 0);

        coordinator.stream_started();
        coordinator.stream_started();
        assert_eq!(coordinator.active_stream_count(), 2);

        coordinator.stream_ended();
        assert_eq!(coordinator.active_stream_count(), 1);

        coordinator.stream_ended();
        assert_eq!(coordinator.active_stream_count(), 0);
    }

    #[tokio::test]
    async fn shutdown_cancels_tracked_tasks() {
        let coordinator = ShutdownCoordinator::new();

        coordinator.spawn("test-task", async {
            // Long-running task that should be cancelled
            tokio::time::sleep(Duration::from_secs(3600)).await;
        });

        assert_eq!(coordinator.task_count(), 1);

        coordinator.shutdown(Duration::from_secs(5)).await;

        assert_eq!(coordinator.task_count(), 0);
    }
}
