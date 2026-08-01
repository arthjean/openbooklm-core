//! Owned, cancellable ingestion tasks (US-010, EP-002).
//!
//! Ingestion spawns async work — embedding batches, and whatever a future stage
//! adds. Before this module that work was detached: `tokio::spawn` returned a
//! handle the pipeline dropped on timeout, and the task carried on calling a
//! paid provider API for a source nobody was waiting for any more.
//!
//! The fix is ownership, not force. Three mechanisms, in this order:
//!
//! 1. **Admission closure.** A cancelled token makes every not-yet-started unit
//!    return immediately. Work that has not begun never begins.
//! 2. **Cooperative drain.** Started work observes the token at its own await
//!    points and unwinds. The drain waits for that, bounded by a deadline.
//! 3. **Abort.** Whatever is still running when the deadline passes is aborted.
//!
//! What the third step cannot reach is `spawn_blocking` work: a blocking closure
//! on the blocking pool runs to completion whatever anyone else wants.
//! [`DrainReport`] therefore separates *drained* from *abandoned*, and the
//! caller reports the difference rather than claiming everything stopped.
//!
//! This is a composition of `tokio_util`'s `TaskTracker` and `CancellationToken`
//! rather than a new abstraction over them: the same pair the server already
//! uses for graceful shutdown, scoped to one source's ingestion.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// How long cooperative cancellation is given before abort (US-010: 5 seconds).
pub const DRAIN_DEADLINE: Duration = Duration::from_secs(5);

/// What a drain actually achieved.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainReport {
    /// Tasks that observed cancellation and unwound on their own.
    pub drained: usize,
    /// Tasks still running at the deadline. Aborted if abortable; a blocking
    /// closure is not, and keeps running until it finishes.
    pub abandoned: usize,
    /// Blocking closures still executing on the blocking pool.
    ///
    /// Nothing can stop these. They are reported rather than counted as
    /// cancelled, because saying an OCR-sized PDF parse was cancelled when it
    /// is still burning a core is the lie US-010 exists to prevent.
    pub blocking_in_flight: usize,
    /// Whether the drain completed within [`DRAIN_DEADLINE`].
    pub within_deadline: bool,
}

impl DrainReport {
    /// True when every owned operation stopped on its own.
    #[must_use]
    pub const fn is_clean(&self) -> bool {
        self.abandoned == 0 && self.blocking_in_flight == 0 && self.within_deadline
    }
}

/// Owns every async task one ingestion run spawns.
///
/// Deliberately not `Clone`: exactly one owner exists, and it is the thing that
/// decides when the run is over. Clones are handed the [`CancellationToken`]
/// instead, which grants the right to observe cancellation and nothing else.
pub struct IngestionTasks {
    tracker: TaskTracker,
    token: CancellationToken,
    /// Kept so the deadline can escalate from cooperative to forced. Its length
    /// is also the spawn count, which is why no separate counter exists: two
    /// counters for one fact can disagree, one cannot.
    aborts: Mutex<Vec<AbortHandle>>,
    /// Blocking closures owned but not yet finished.
    ///
    /// Incremented when the closure is *handed to* the pool and decremented
    /// when it returns, so the window between spawn and first instruction is
    /// counted. Counting from inside the closure would leave a drain in that
    /// window reporting a clean stop while a closure was about to start —
    /// exactly the claim this module exists to avoid making.
    blocking_in_flight: Arc<AtomicUsize>,
}

impl IngestionTasks {
    /// Own a new set of tasks under `token`.
    ///
    /// Pass a child of the server's shutdown token so process shutdown cancels
    /// ingestion, while cancelling ingestion leaves the server alone.
    #[must_use]
    pub fn new(token: CancellationToken) -> Self {
        Self {
            tracker: TaskTracker::new(),
            token,
            aborts: Mutex::new(Vec::new()),
            blocking_in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// The token owned work observes. Cheap to clone.
    #[must_use]
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// Close admission and signal every owned task.
    ///
    /// Idempotent: the first terminal error, a timeout and a shutdown may all
    /// call it, and the second call is a no-op.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// Spawn a task this run owns.
    ///
    /// The returned handle is for the caller's own error propagation. Ownership
    /// stays here: dropping the handle does not detach the task, because the
    /// tracker still holds it and [`shutdown`](Self::shutdown) still waits for it.
    pub fn spawn<F>(&self, future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let handle = self.tracker.spawn(future);
        if let Ok(mut aborts) = self.aborts.lock() {
            aborts.push(handle.abort_handle());
        }
        handle
    }

    /// Run a CPU-bound closure on the blocking pool, counted.
    ///
    /// Nothing here can stop the closure once it starts — that is what
    /// "blocking" means. What this does provide is an honest count: the
    /// increment happens before the closure reaches the pool and the decrement
    /// when it returns, so neither a timeout that drops the returned future nor
    /// a drain that runs while the pool is still dispatching can observe a zero
    /// that is not true.
    ///
    /// Awaiting the returned handle yields a `JoinError` if the closure panics.
    pub fn spawn_blocking<F, R>(&self, f: F) -> JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let counter = Arc::clone(&self.blocking_in_flight);
        counter.fetch_add(1, Ordering::SeqCst);
        tokio::task::spawn_blocking(move || {
            // Decrement on the way out whatever happens, including a panic:
            // a closure that died still is not running.
            let _guard = InFlightGuard(counter);
            f()
        })
    }

    /// Cancel, drain within `deadline`, then abort what remains.
    ///
    /// Always call this before the run returns, on the success path too: a
    /// successful pipeline that leaves a task running has the same problem as a
    /// failed one, it is just harder to notice.
    pub async fn shutdown(&self, deadline: Duration) -> DrainReport {
        self.cancel();
        self.tracker.close();

        let spawned = self.aborts.lock().map_or(0, |a| a.len());
        let within_deadline = tokio::time::timeout(deadline, self.tracker.wait())
            .await
            .is_ok();

        let blocking_in_flight = self.blocking_in_flight.load(Ordering::SeqCst);

        if within_deadline {
            return DrainReport {
                drained: spawned,
                abandoned: 0,
                blocking_in_flight,
                within_deadline: true,
            };
        }

        // Past the deadline. Abort what can be aborted and count what is left:
        // `TaskTracker::len` is the tasks that have not finished, which after
        // the aborts are the ones abort cannot reach.
        if let Ok(aborts) = self.aborts.lock() {
            for handle in aborts.iter() {
                handle.abort();
            }
        }
        let abandoned = self.tracker.len();
        DrainReport {
            drained: spawned.saturating_sub(abandoned),
            abandoned,
            blocking_in_flight,
            within_deadline: false,
        }
    }
}

/// Decrements the in-flight count when a blocking closure leaves the pool.
struct InFlightGuard(Arc<AtomicUsize>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[tokio::test]
    async fn cooperative_work_drains_within_the_deadline() {
        let tasks = IngestionTasks::new(CancellationToken::new());
        let token = tasks.token();
        tasks.spawn(async move {
            token.cancelled().await;
        });

        let report = tasks.shutdown(DRAIN_DEADLINE).await;
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.drained, 1);
    }

    #[tokio::test(start_paused = true)]
    async fn work_ignoring_cancellation_is_reported_as_abandoned() {
        let tasks = IngestionTasks::new(CancellationToken::new());
        tasks.spawn(std::future::pending::<()>());

        let report = tasks.shutdown(Duration::from_millis(50)).await;
        assert!(!report.within_deadline, "{report:?}");
        assert_eq!(report.abandoned, 1, "{report:?}");
        assert!(!report.is_clean(), "abandoned work is not a clean drain");
    }

    #[tokio::test]
    async fn cancellation_closes_admission_before_work_starts() {
        let tasks = IngestionTasks::new(CancellationToken::new());
        let started = Arc::new(AtomicUsize::new(0));

        tasks.cancel();

        // Admission check is the first thing the unit does, exactly as the
        // embedding batches do it.
        let token = tasks.token();
        let counter = Arc::clone(&started);
        let handle = tasks.spawn(async move {
            if token.is_cancelled() {
                return;
            }
            counter.fetch_add(1, Ordering::SeqCst);
        });
        handle.await.expect("task must not panic");

        assert_eq!(
            started.load(Ordering::SeqCst),
            0,
            "no unit may start after cancellation"
        );
    }

    #[tokio::test]
    async fn blocking_work_still_running_is_reported_not_claimed_cancelled() {
        let tasks = IngestionTasks::new(CancellationToken::new());
        let (release, gate) = std::sync::mpsc::channel();

        // A blocking closure nothing can abort. The handle is deliberately not
        // awaited before the drain: this is the shape a timeout produces when it
        // drops the future waiting on `spawn_blocking`.
        let handle = tasks.spawn_blocking(move || gate.recv());
        // No wait for the closure to start: the count is taken when the closure
        // is handed to the pool, so the drain cannot observe a false zero even
        // if the pool has not dispatched it yet.

        let report = tasks.shutdown(Duration::from_millis(50)).await;
        assert_eq!(
            report.blocking_in_flight, 1,
            "a running blocking closure must be reported, not counted as cancelled: {report:?}"
        );
        assert!(!report.is_clean(), "{report:?}");

        release.send(()).expect("release blocking closure");
        handle
            .await
            .expect("blocking closure")
            .expect("release signal");
    }

    #[tokio::test]
    async fn completed_blocking_work_leaves_a_clean_report() {
        let tasks = IngestionTasks::new(CancellationToken::new());
        let value = tasks
            .spawn_blocking(|| 21_u32 * 2)
            .await
            .expect("blocking closure must not panic");
        assert_eq!(value, 42);

        let report = tasks.shutdown(DRAIN_DEADLINE).await;
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(report.blocking_in_flight, 0);
    }

    /// The window this counter exists to cover: a drain that runs before the
    /// blocking pool has dispatched the closure must still report it.
    #[tokio::test]
    async fn a_blocking_closure_counts_from_the_moment_it_is_handed_over() {
        let tasks = IngestionTasks::new(CancellationToken::new());
        let (release, gate) = std::sync::mpsc::channel();

        let handle = tasks.spawn_blocking(move || gate.recv());
        assert_eq!(
            tasks.blocking_in_flight.load(Ordering::SeqCst),
            1,
            "the count must be taken at hand-over, not at first instruction"
        );

        release.send(()).expect("release blocking closure");
        handle
            .await
            .expect("blocking closure")
            .expect("release signal");
        let report = tasks.shutdown(DRAIN_DEADLINE).await;
        assert!(report.is_clean(), "{report:?}");
    }

    #[tokio::test]
    async fn shutdown_is_idempotent() {
        let tasks = IngestionTasks::new(CancellationToken::new());
        let first = tasks.shutdown(DRAIN_DEADLINE).await;
        let second = tasks.shutdown(DRAIN_DEADLINE).await;
        assert!(first.is_clean() && second.is_clean());
    }
}
