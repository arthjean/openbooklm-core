//! Owned maintenance for persistent data with bounded retention.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::middleware::TaskTracker;
use crate::repositories::{GenerationRepository, OcrCacheRepository, RagLogRepository, RepoResult};
use crate::services::rag::rag_log::RAG_LOG_RETENTION_DAYS;
use crate::services::source_processing::{
    GENERATION_RETENTION_HOURS, PROCESSING_TIMEOUT, STALE_BUILD_MULTIPLIER, STALE_BUILD_REASON,
};
use crate::types::PurgeTaskState;

pub const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const RAG_LOG_TASK: &str = "rag-log-retention";
const OCR_CACHE_TASK: &str = "ocr-cache-retention";
const GENERATION_TASK: &str = "index-generation-retention";

/// Start one process-owned loop. Every pass runs each retention domain
/// independently, so a RAG-log failure cannot suppress generation recovery.
pub fn start_maintenance_task(
    task_tracker: &TaskTracker,
    status: &PurgeTaskState,
    rag_logs: Arc<dyn RagLogRepository>,
    ocr_cache: Arc<dyn OcrCacheRepository>,
    generations: Arc<dyn GenerationRepository>,
) {
    status.register(RAG_LOG_TASK, MAINTENANCE_INTERVAL);
    status.register(OCR_CACHE_TASK, MAINTENANCE_INTERVAL);
    status.register(GENERATION_TASK, MAINTENANCE_INTERVAL);

    let status = status.clone();
    let shutdown = task_tracker.cancellation_token();
    if task_tracker
        .try_spawn("data-retention", async move {
            run_periodically(shutdown, MAINTENANCE_INTERVAL, move || {
                let status = status.clone();
                let rag_logs = rag_logs.clone();
                let ocr_cache = ocr_cache.clone();
                let generations = generations.clone();
                async move {
                    run_maintenance_pass(
                        &status,
                        async move { rag_logs.purge_old_logs(RAG_LOG_RETENTION_DAYS).await },
                        async move { ocr_cache.purge_unowned().await },
                        async move { run_generation_maintenance(generations.as_ref()).await },
                    )
                    .await;
                }
            })
            .await;
        })
        .is_err()
    {
        tracing::warn!("Data-retention maintenance not started because shutdown is active");
    }
}

async fn run_periodically<F, Fut>(
    shutdown: tokio_util::sync::CancellationToken,
    interval: Duration,
    mut operation: F,
) where
    F: FnMut() -> Fut,
    Fut: Future<Output = ()>,
{
    loop {
        if shutdown.is_cancelled() {
            break;
        }

        operation().await;

        tokio::select! {
            biased;
            () = shutdown.cancelled() => break,
            () = tokio::time::sleep(interval) => {}
        }
    }
}

async fn run_maintenance_pass<R, O, G>(
    status: &PurgeTaskState,
    rag_logs: R,
    ocr_cache: O,
    generations: G,
) where
    R: Future<Output = RepoResult<u64>>,
    O: Future<Output = RepoResult<u64>>,
    G: Future<Output = RepoResult<u64>>,
{
    record_run(status, RAG_LOG_TASK, rag_logs).await;
    record_run(status, OCR_CACHE_TASK, ocr_cache).await;
    record_run(status, GENERATION_TASK, generations).await;
}

async fn run_generation_maintenance(repo: &dyn GenerationRepository) -> RepoResult<u64> {
    let stale_after_secs = i64::try_from(
        PROCESSING_TIMEOUT
            .saturating_mul(STALE_BUILD_MULTIPLIER)
            .as_secs(),
    )
    .unwrap_or(i64::MAX);

    // Reclaim still runs when stale recovery fails. The two operations touch
    // disjoint terminal states and each is safe to retry on the next pass.
    let recovered = repo
        .fail_stale_builds(stale_after_secs, STALE_BUILD_REASON)
        .await;
    let reclaimed = repo.reclaim_all(GENERATION_RETENTION_HOURS).await;

    match (recovered, reclaimed) {
        (Ok(recovered), Ok(reclaimed)) => {
            tracing::info!(
                recovered,
                reclaimed,
                retention_hours = GENERATION_RETENTION_HOURS,
                "Index-generation maintenance completed"
            );
            Ok(reclaimed)
        }
        (Err(recovery_error), Ok(_)) => Err(recovery_error),
        (Ok(_), Err(reclaim_error)) => Err(reclaim_error),
        (Err(recovery_error), Err(reclaim_error)) => {
            tracing::warn!(error = %reclaim_error, "Index-generation reclaim also failed");
            Err(recovery_error)
        }
    }
}

async fn record_run<F>(status: &PurgeTaskState, name: &'static str, future: F)
where
    F: Future<Output = RepoResult<u64>>,
{
    let started = Instant::now();
    match future.await {
        Ok(deleted) => {
            status.record_success(
                name,
                deleted,
                u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            );
        }
        Err(error) => {
            status.record_failure(
                name,
                u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            );
            tracing::warn!(task = name, error = %error, "Data-retention maintenance failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AppError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test(start_paused = true)]
    async fn periodic_loop_runs_immediately_then_stops_on_shutdown() {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let runs = Arc::new(AtomicUsize::new(0));
        let loop_runs = runs.clone();
        let handle = tokio::spawn(run_periodically(
            shutdown.clone(),
            MAINTENANCE_INTERVAL,
            move || {
                let runs = loop_runs.clone();
                async move {
                    runs.fetch_add(1, Ordering::SeqCst);
                }
            },
        ));

        tokio::task::yield_now().await;
        assert_eq!(runs.load(Ordering::SeqCst), 1);

        tokio::time::advance(MAINTENANCE_INTERVAL).await;
        tokio::task::yield_now().await;
        assert_eq!(runs.load(Ordering::SeqCst), 2);

        shutdown.cancel();
        handle.await.expect("periodic loop exits cleanly");
    }

    #[tokio::test]
    async fn a_failed_retention_domain_does_not_suppress_the_others() {
        let status = PurgeTaskState::new();
        status.register(RAG_LOG_TASK, MAINTENANCE_INTERVAL);
        status.register(OCR_CACHE_TASK, MAINTENANCE_INTERVAL);
        status.register(GENERATION_TASK, MAINTENANCE_INTERVAL);

        run_maintenance_pass(
            &status,
            async { Err(AppError::Internal("RAG purge failed".to_owned())) },
            async { Ok(2) },
            async { Ok(3) },
        )
        .await;

        let snapshot = status.snapshot();
        assert_eq!(snapshot[RAG_LOG_TASK].total_failures, 1);
        assert_eq!(snapshot[OCR_CACHE_TASK].total_deleted, 2);
        assert_eq!(snapshot[GENERATION_TASK].total_deleted, 3);
    }

    #[tokio::test]
    async fn tracked_run_records_success() {
        let status = PurgeTaskState::new();
        status.register(RAG_LOG_TASK, MAINTENANCE_INTERVAL);

        record_run(&status, RAG_LOG_TASK, async { Ok(7) }).await;

        let snapshot = status.snapshot();
        let task = snapshot.get(RAG_LOG_TASK).expect("registered task");
        assert_eq!(task.total_runs, 1);
        assert_eq!(task.total_deleted, 7);
        assert_eq!(task.total_failures, 0);
    }

    #[tokio::test]
    async fn tracked_run_records_failure() {
        let status = PurgeTaskState::new();
        status.register(GENERATION_TASK, MAINTENANCE_INTERVAL);

        record_run(&status, GENERATION_TASK, async {
            Err(AppError::Internal("maintenance failed".to_owned()))
        })
        .await;

        let snapshot = status.snapshot();
        let task = snapshot.get(GENERATION_TASK).expect("registered task");
        assert_eq!(task.total_runs, 1);
        assert_eq!(task.total_failures, 1);
        assert_eq!(task.consecutive_failures, 1);
    }
}
