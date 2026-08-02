//! Graceful shutdown and owned background tasks.
//!
//! The coordinator owns admission, cancellation and joins. It deliberately does
//! not select on cancellation around task futures: each task owns its terminal
//! path and receives a child token when it needs cooperative cancellation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use tokio::task::AbortHandle;
use tokio_util::task::task_tracker::TaskTrackerToken;
use tokio_util::{sync::CancellationToken, task::TaskTracker as TokioTaskTracker};

/// Returned when shutdown has closed background-task admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnRejected;

/// Reservation for one request that may create process-scoped work.
///
/// The reservation linearizes shutdown admission before the request performs
/// durable writes. It keeps the root scope non-empty until the request either
/// registers its task or exits without one.
pub struct TaskAdmission {
    coordinator: ShutdownCoordinator,
    reservation: Option<TaskTrackerToken>,
}

struct Lifecycle {
    accepting: bool,
    forced: bool,
    aborts: HashMap<u64, AbortHandle>,
}

/// Coordinated owner for process-scoped background tasks.
#[derive(Clone)]
pub struct ShutdownCoordinator {
    tracker: TokioTaskTracker,
    cancel_token: CancellationToken,
    lifecycle: Arc<Mutex<Lifecycle>>,
    next_task_id: Arc<AtomicU64>,
    active_streams: Arc<AtomicUsize>,
}

impl ShutdownCoordinator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracker: TokioTaskTracker::new(),
            cancel_token: CancellationToken::new(),
            lifecycle: Arc::new(Mutex::new(Lifecycle {
                accepting: true,
                forced: false,
                aborts: HashMap::new(),
            })),
            next_task_id: Arc::new(AtomicU64::new(1)),
            active_streams: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn lifecycle(&self) -> MutexGuard<'_, Lifecycle> {
        self.lifecycle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A child token that observes process shutdown without being able to cause it.
    #[must_use]
    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancel_token.child_token()
    }

    /// Reserve the right for a request to register one task later.
    ///
    /// The lifecycle lock makes this atomic with [`Self::begin_shutdown`]. A
    /// successful reservation remains valid after shutdown starts, preventing
    /// a request from committing durable state and then losing task admission.
    pub fn try_admit(&self) -> Result<TaskAdmission, SpawnRejected> {
        let lifecycle = self.lifecycle();
        if !lifecycle.accepting {
            return Err(SpawnRejected);
        }

        let reservation = self.tracker.token();
        drop(lifecycle);
        Ok(TaskAdmission {
            coordinator: self.clone(),
            reservation: Some(reservation),
        })
    }

    /// Register a process-scoped task if shutdown has not started.
    ///
    /// The future is tracked unchanged. Cancellation must be handled inside the
    /// future so its domain cleanup and terminal state cannot be bypassed.
    pub fn try_spawn<F>(&self, name: &'static str, future: F) -> Result<(), SpawnRejected>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let admission = self.try_admit()?;
        admission.spawn(name, future)
    }

    fn spawn_admitted<F>(&self, name: &'static str, future: F) -> Result<(), SpawnRejected>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let mut lifecycle = self.lifecycle();
        if lifecycle.forced {
            return Err(SpawnRejected);
        }

        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        let registration = TaskRegistration {
            task_id,
            lifecycle: Arc::clone(&self.lifecycle),
        };
        let handle = self.tracker.spawn(async move {
            let _registration = registration;
            future.await;
            tracing::debug!(task = name, "Task completed");
        });
        let abort = handle.abort_handle();

        // The lifecycle lock remains held until registration is complete. If a
        // very short task finishes first, its guard waits on this same lock and
        // removes the handle immediately after insertion.
        lifecycle.aborts.insert(task_id, abort);
        Ok(())
    }

    /// Atomically close admission, then notify every admitted task.
    ///
    /// Returns `true` only for the caller that initiated shutdown.
    pub fn begin_shutdown(&self) -> bool {
        let initiated = {
            let mut lifecycle = self.lifecycle();
            if !lifecycle.accepting {
                false
            } else {
                lifecycle.accepting = false;
                true
            }
        };

        if initiated {
            self.tracker.close();
            self.cancel_token.cancel();
        }
        initiated
    }

    /// Wait until shutdown has started and every admitted task has exited.
    pub async fn wait(&self) {
        self.cancel_token.cancelled().await;
        self.tracker.wait().await;
    }

    /// Abort every async task that remains after cooperative drain expires.
    #[must_use]
    pub fn abort_remaining(&self) -> usize {
        let aborts: Vec<AbortHandle> = {
            let mut lifecycle = self.lifecycle();
            lifecycle.forced = true;
            lifecycle.aborts.values().cloned().collect()
        };
        let remaining = aborts.len();
        for abort in aborts {
            abort.abort();
        }
        remaining
    }

    /// Convenience for bounded owners outside the server composition root.
    pub async fn shutdown(&self, timeout: Duration) {
        self.begin_shutdown();
        if tokio::time::timeout(timeout, self.wait()).await.is_err() {
            let aborted = self.abort_remaining();
            tracing::warn!(
                aborted,
                "Shutdown deadline reached; aborted remaining async tasks"
            );
            tokio::task::yield_now().await;
        }
    }

    /// Increment the active chat-stream counter.
    pub fn stream_started(&self) {
        self.active_streams.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the active chat-stream counter.
    pub fn stream_ended(&self) {
        self.active_streams.fetch_sub(1, Ordering::Relaxed);
    }

    /// Current number of active chat streams, for shutdown diagnostics.
    #[must_use]
    pub fn active_stream_count(&self) -> usize {
        self.active_streams.load(Ordering::Relaxed)
    }

    /// Current number of admitted background tasks.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tracker.len()
    }
}

impl TaskAdmission {
    /// Register the task reserved by this admission.
    pub fn spawn<F>(mut self, name: &'static str, future: F) -> Result<(), SpawnRejected>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        self.coordinator.spawn_admitted(name, future)?;

        // `spawn_admitted` creates the task's own tracker token before the
        // reservation is released, so a closed root scope cannot transiently
        // become empty between request admission and task ownership.
        drop(self.reservation.take());
        Ok(())
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

struct TaskRegistration {
    task_id: u64,
    lifecycle: Arc<Mutex<Lifecycle>>,
}

impl Drop for TaskRegistration {
    fn drop(&mut self) {
        let mut lifecycle = self
            .lifecycle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        lifecycle.aborts.remove(&self.task_id);
    }
}

/// Backward-compatible name used by application state.
pub type TaskTracker = ShutdownCoordinator;

/// Wait for SIGTERM or SIGINT.
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
    use std::sync::atomic::AtomicBool;

    use super::*;

    #[test]
    fn stream_counter_tracks_active_streams() {
        let coordinator = ShutdownCoordinator::new();
        assert_eq!(coordinator.active_stream_count(), 0);

        coordinator.stream_started();
        coordinator.stream_started();
        assert_eq!(coordinator.active_stream_count(), 2);

        coordinator.stream_ended();
        coordinator.stream_ended();
        assert_eq!(coordinator.active_stream_count(), 0);
    }

    #[tokio::test]
    async fn shutdown_runs_task_owned_cleanup_before_joining() {
        let coordinator = ShutdownCoordinator::new();
        let shutdown = coordinator.cancellation_token();
        let cleaned_up = Arc::new(AtomicBool::new(false));
        let cleaned_up_by_task = Arc::clone(&cleaned_up);

        coordinator
            .try_spawn("test-task", async move {
                shutdown.cancelled().await;
                cleaned_up_by_task.store(true, Ordering::SeqCst);
            })
            .expect("task admission must be open");

        coordinator.shutdown(Duration::from_secs(1)).await;

        assert!(cleaned_up.load(Ordering::SeqCst));
        assert_eq!(coordinator.task_count(), 0);
    }

    #[tokio::test]
    async fn shutdown_rejects_new_tasks() {
        let coordinator = ShutdownCoordinator::new();
        assert!(coordinator.begin_shutdown());
        assert!(!coordinator.begin_shutdown());
        assert!(coordinator.try_admit().is_err());
        assert!(
            coordinator
                .try_spawn("late-task", std::future::ready(()))
                .is_err()
        );
    }

    #[tokio::test]
    async fn shutdown_deadline_aborts_uncooperative_tasks() {
        let coordinator = ShutdownCoordinator::new();
        coordinator
            .try_spawn("uncooperative-task", std::future::pending())
            .expect("task admission must be open");

        coordinator.shutdown(Duration::from_millis(10)).await;

        tokio::time::timeout(Duration::from_millis(50), async {
            while coordinator.task_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("aborted task must leave the tracker");
    }

    #[tokio::test]
    async fn admitted_request_can_register_its_task_after_shutdown_starts() {
        let coordinator = ShutdownCoordinator::new();
        let admission = coordinator.try_admit().expect("request admission");
        let completed = Arc::new(AtomicBool::new(false));
        let completed_by_task = Arc::clone(&completed);

        coordinator.begin_shutdown();
        admission
            .spawn("admitted-task", async move {
                completed_by_task.store(true, Ordering::SeqCst);
            })
            .expect("gracefully admitted task");
        coordinator.wait().await;

        assert!(completed.load(Ordering::SeqCst));
        assert_eq!(coordinator.task_count(), 0);
    }

    #[tokio::test]
    async fn unused_request_admission_releases_the_root_scope() {
        let coordinator = ShutdownCoordinator::new();
        let admission = coordinator.try_admit().expect("request admission");

        coordinator.begin_shutdown();
        drop(admission);
        coordinator.wait().await;

        assert_eq!(coordinator.task_count(), 0);
    }

    #[tokio::test]
    async fn forced_shutdown_revokes_an_unmaterialized_admission() {
        let coordinator = ShutdownCoordinator::new();
        let admission = coordinator.try_admit().expect("request admission");
        let ran = Arc::new(AtomicBool::new(false));
        let ran_by_task = Arc::clone(&ran);

        coordinator.begin_shutdown();
        let _ = coordinator.abort_remaining();
        let result = admission.spawn("too-late-task", async move {
            ran_by_task.store(true, Ordering::SeqCst);
        });
        tokio::task::yield_now().await;

        assert!(result.is_err());
        assert!(!ran.load(Ordering::SeqCst));
    }
}
