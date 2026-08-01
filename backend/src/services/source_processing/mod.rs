//! Source processing service — RAG pipeline for document ingestion.
//!
//! Pipeline: extraction → chunking → embedding → storage → publication. The
//! first stages live in [`extraction`] and [`indexing`]; this module owns what
//! surrounds them: who is allowed to build, when the build stops, and which of
//! the two terminal outcomes it reaches.
//!
//! Contextualization (the former stage 3) is disabled to control cost. See
//! IMPORTANT_FUTUR.md at the repo root to re-enable it.
//!
//! ## Generations (EP-002)
//!
//! Reprocessing never touches the index a search is reading. A run claims one
//! *building* generation, writes every batch under it, validates it, and only
//! then moves the source's active pointer in a single transaction. Extraction,
//! embedding, storage, validation, timeout and shutdown failures all end the
//! same way: the building generation becomes `failed` and the previously active
//! generation is still there, still complete, still searchable.
//!
//! Claiming is also what makes a duplicate reprocess request harmless. The
//! partial unique index on `(source_id) WHERE state = 'building'` allows one
//! owner; every other caller gets `None` and returns the source's current state
//! without spawning anything.

mod extraction;
mod indexing;

pub use extraction::merge_ocr_text;
pub(crate) use extraction::validate_pdf_magic_bytes;

use sea_orm::DatabaseConnection;
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};
use uuid::Uuid;

use crate::core::CoreState;
use crate::core::config::CoreConfig;
use crate::core::entitlements::SharedEntitlementPolicy;
use crate::core::events::{DomainEvent, SharedEventSink};
use crate::core::principal::Principal;
use crate::core::providers::{EmbeddingProvider, SharedEmbeddingProvider};
use crate::entities::source::SourceStatus;
use crate::error::AppError;
use crate::repositories::{
    ChunkRepository, GenerationRepository, OcrCacheRepository, SourceRepository,
};
use crate::services::ingestion_tasks::{DRAIN_DEADLINE, IngestionTasks};
use crate::services::rag::provenance::{
    ChunkingProvenance, EmbeddingProvenance, GenerationProvenance,
};
use crate::services::source_events::SourceEventBroadcaster;
use crate::services::sources::{get_source, update_source_status};
use crate::types::SourceType;

/// How long one source may spend being processed before the run is abandoned.
///
/// Lives here rather than at the API boundary because it also defines a stale
/// build: recovery calls a generation abandoned after [`STALE_BUILD_MULTIPLIER`]
/// times this value, and the two numbers only mean anything together.
pub const PROCESSING_TIMEOUT: Duration = Duration::from_secs(600);

/// How long a replaced generation stays rollback-eligible.
///
/// At least one prior complete generation and at least 24 hours, per
/// `docs/architecture/index-generations.md`. Reclaim keeps the newest replaced
/// generation whatever its age, so this bounds how far back an operator can go,
/// not whether they can go back at all.
pub const GENERATION_RETENTION_HOURS: i32 = 24;

/// Attach a stage description to a failing step.
///
/// The messages this produces are what the user sees and what the generation
/// stores as its failure reason, so every fallible step in the pipeline names
/// the stage it belongs to.
pub(crate) trait StageContext<T> {
    /// # Errors
    /// Returns the original error wrapped in a [`PipelineFailure`] whose
    /// message is prefixed with `stage`.
    fn stage(self, stage: &str) -> Result<T, PipelineFailure>;
}

impl<T, E: Into<AppError>> StageContext<T> for Result<T, E> {
    fn stage(self, stage: &str) -> Result<T, PipelineFailure> {
        self.map_err(|e| {
            let e = e.into();
            PipelineFailure::new(format!("{stage}: {e}"), e)
        })
    }
}

/// Dependencies for source processing, bundled to avoid too-many-arguments.
pub struct ProcessingDeps {
    pub db: DatabaseConnection,
    pub config: Arc<CoreConfig>,
    pub broadcaster: SourceEventBroadcaster,
    pub source_repo: Arc<dyn SourceRepository>,
    pub chunk_repo: Arc<dyn ChunkRepository>,
    /// Owns the index-generation lifecycle: claim, publish, fail, reclaim.
    pub generation_repo: Arc<dyn GenerationRepository>,
    pub embeddings: Option<SharedEmbeddingProvider>,
    pub firecrawl: Option<crate::clients::FirecrawlClient>,
    pub youtube: Option<crate::clients::YouTubeClient>,
    pub ocr: Option<crate::clients::MistralOcrClient>,
    pub ocr_cache: Arc<dyn OcrCacheRepository>,
    /// Authorizes and meters bounded work (OCR pages). No usage repository,
    /// plan string or SaaS profile reaches this pipeline.
    pub entitlements: SharedEntitlementPolicy,
    /// Receives ingestion outcome events.
    pub events: SharedEventSink,
    /// The account the source belongs to, carrying whatever opaque metadata the
    /// identity adapter attached.
    pub principal: Principal,
    /// The server's shutdown signal. Ingestion derives a child token from it, so
    /// shutting the process down cancels ingestion while a cancelled ingestion
    /// leaves the server running (US-010).
    pub shutdown: CancellationToken,
}

impl ProcessingDeps {
    /// Construct processing dependencies from the shared application state.
    ///
    /// All fields are extracted from `CoreState`, converting concrete repository
    /// types to trait objects for polymorphic use in the processing pipeline.
    pub fn from_state(state: &CoreState, principal: Principal) -> Self {
        Self {
            db: state.db.clone(),
            config: state.config.clone(),
            broadcaster: state.source_broadcaster.clone(),
            source_repo: state.repos.sources.clone() as Arc<dyn SourceRepository>,
            chunk_repo: state.repos.chunks.clone() as Arc<dyn ChunkRepository>,
            generation_repo: state.repos.generations.clone() as Arc<dyn GenerationRepository>,
            embeddings: state.clients.embeddings.clone(),
            firecrawl: state.clients.firecrawl.clone(),
            youtube: state.clients.youtube.clone(),
            ocr: state.clients.ocr.clone(),
            ocr_cache: state.repos.ocr_cache.clone() as Arc<dyn OcrCacheRepository>,
            entitlements: state.entitlements.clone(),
            events: state.events.clone(),
            principal,
            shutdown: state.task_tracker.cancellation_token(),
        }
    }

    /// The account id every event and permit is attributed to.
    pub fn account_id(&self) -> Uuid {
        self.principal.account_id
    }
}

// ============================================================================
// Ownership (US-009)
// ============================================================================

/// A claimed, exclusive right to build one source's next index.
///
/// Held by exactly one worker. Every status update in the run carries its
/// `generation_id`, which is what stops a worker whose ownership has moved on
/// from reporting an outcome for someone else's build.
#[must_use]
#[derive(Debug, Clone)]
pub struct IndexOwnership {
    pub generation_id: Uuid,
    pub provenance: GenerationProvenance,
}

/// How long a building generation may exist before recovery calls it abandoned.
///
/// Twice the processing timeout: a build still inside its own deadline is not
/// stale, and a build that outlived twice its deadline has no live owner —
/// `process_source` cannot exceed one deadline plus one drain.
pub const STALE_BUILD_MULTIPLIER: u32 = 2;

/// Reason recorded on a generation reclaimed from a process that is gone.
pub const STALE_BUILD_REASON: &str =
    "Indexing was interrupted (process restart or crash) and did not finish";

/// The provenance a generation built now would carry.
///
/// Derived from the live embedding provider and the compiled chunking contract,
/// never from configuration text: a fingerprint that does not come from the
/// thing it describes is not evidence of anything.
///
/// # Errors
/// Returns [`AppError::Internal`] when the provider reports provenance the
/// schema will not accept.
pub fn current_provenance(
    embedder: &dyn EmbeddingProvider,
    source_type: SourceType,
) -> Result<GenerationProvenance, AppError> {
    let provenance = GenerationProvenance {
        embedding: EmbeddingProvenance::from_provider(embedder),
        chunking: ChunkingProvenance::current(crate::services::rag::chunking::parent_chunk_size(
            source_type,
        )),
    };
    provenance.validate()?;
    Ok(provenance)
}

/// Claim the right to rebuild a source's index, or report who has it.
///
/// Returns `None` when another worker already owns a building generation for
/// this source. The caller then reports the source's current state rather than
/// starting a second worker (US-009).
///
/// Before claiming, generations older than [`STALE_BUILD_MULTIPLIER`] × the
/// processing timeout are failed: their owner cannot still exist, and leaving
/// them would block the source forever behind a worker that died.
///
/// Takes the two things a claim actually needs rather than a whole pipeline's
/// dependencies: ownership is decided at the API boundary, and that boundary
/// should not have to assemble an ingestion run to ask the question.
///
/// # Errors
/// Propagates repository errors, and rejects a deployment with no embedding
/// provider or with provenance the schema will not accept — which is cheaper to
/// find out before claiming ownership than after.
pub async fn claim_index_ownership(
    generations: &dyn GenerationRepository,
    embedder: Option<&SharedEmbeddingProvider>,
    source_id: Uuid,
    source_type: SourceType,
    processing_timeout: Duration,
) -> Result<Option<IndexOwnership>, AppError> {
    let embedder = embedder.ok_or_else(|| {
        AppError::Internal("No embedding provider configured — cannot generate embeddings".into())
    })?;
    let provenance = current_provenance(embedder.as_ref(), source_type)?;

    let stale_after = i64::try_from(
        processing_timeout
            .saturating_mul(STALE_BUILD_MULTIPLIER)
            .as_secs(),
    )
    .unwrap_or(i64::MAX);
    let recovered = generations
        .fail_stale_builds(stale_after, STALE_BUILD_REASON)
        .await?;
    if recovered > 0 {
        warn!(
            recovered,
            stale_after_secs = stale_after,
            "Failed abandoned building generations before claiming ownership"
        );
    }

    let claimed = generations.claim(source_id, &provenance).await?;
    Ok(claimed.map(|generation_id| IndexOwnership {
        generation_id,
        provenance,
    }))
}

// ============================================================================
// Failure reporting
// ============================================================================

/// A stage failure, carrying the message the user should see.
///
/// Stages return this instead of writing the source's status themselves. One
/// place — [`process_source`] — turns a failure into a failed generation, one
/// broadcast and one domain event, which is what "emit exactly one terminal
/// source event" (US-010) means mechanically rather than by discipline.
#[derive(Debug)]
pub struct PipelineFailure {
    /// Shown to the user and stored as the generation's failure reason.
    pub message: String,
    /// The underlying error, propagated to the caller.
    pub error: AppError,
}

impl PipelineFailure {
    fn new(message: impl Into<String>, error: AppError) -> Self {
        Self {
            message: message.into(),
            error,
        }
    }

    /// A failure whose user-facing message is the error itself.
    fn from_error(error: AppError) -> Self {
        Self {
            message: error.to_string(),
            error,
        }
    }
}

/// User-facing message for a build that outlived its deadline.
const TIMEOUT_MESSAGE: &str = "Processing timed out — please try again or use a smaller document";

/// What a completed build produced, before it is published.
struct BuildOutcome {
    degraded_services: Vec<&'static str>,
    total_chunks: usize,
}

/// Process a source through the full RAG pipeline.
///
/// The caller must already hold [`IndexOwnership`] from
/// [`claim_index_ownership`]. Passing ownership in rather than claiming it here
/// is what lets the API boundary answer a duplicate request without starting a
/// second worker (US-009).
///
/// Every exit path goes through one of two places: publication on success, or
/// [`fail_generation`] otherwise. Both are terminal, and exactly one of them
/// runs, which is what keeps "one terminal source event" (US-010) a property of
/// the control flow rather than a rule someone has to remember.
///
/// # Errors
/// Propagates the stage error that ended the run. The previously active
/// generation is untouched in every failure case.
#[tracing::instrument(
    skip(deps, ownership),
    fields(%source_id, %notebook_id, source_type = source_type.as_str(), generation_id = %ownership.generation_id)
)]
pub async fn process_source(
    deps: ProcessingDeps,
    ownership: IndexOwnership,
    source_id: Uuid,
    notebook_id: Uuid,
    source_type: SourceType,
    processing_timeout: Duration,
) -> Result<(), AppError> {
    let analytics = AnalyticsCtx {
        events: &deps.events,
        account_id: deps.account_id(),
        source_type,
        pipeline_start: std::time::Instant::now(),
    };

    // The run owns its tasks under a child of the server's shutdown token:
    // shutting the process down cancels ingestion, and a cancelled ingestion
    // leaves the server alone.
    let tasks = IngestionTasks::new(deps.shutdown.child_token());

    // The deadline lives here rather than around the whole spawned task. The
    // difference matters: `timeout` cancels by dropping the future it wraps, so
    // anything owned *inside* that future would be dropped without a chance to
    // drain. `tasks` is owned outside it and survives to be shut down properly.
    let build = tokio::time::timeout(
        processing_timeout,
        run_build(
            &deps,
            &tasks,
            &ownership,
            source_id,
            notebook_id,
            source_type,
        ),
    )
    .await;

    // Ownership of every async task ends here, on every path — success, stage
    // failure and timeout alike. A successful run that leaves a task embedding
    // in the background has the same problem as a failed one.
    let drain = tasks.shutdown(DRAIN_DEADLINE).await;
    if !drain.is_clean() {
        warn!(
            source_id = %source_id,
            generation_id = %ownership.generation_id,
            drained = drain.drained,
            abandoned = drain.abandoned,
            blocking_in_flight = drain.blocking_in_flight,
            drain_deadline_secs = DRAIN_DEADLINE.as_secs(),
            "Ingestion drain did not complete cleanly — the operations above were \
             still running at the deadline and were not cancelled"
        );
    }

    let failure = match build {
        Ok(Ok(outcome)) => {
            match publish_generation(
                &deps,
                &ownership,
                source_id,
                notebook_id,
                &analytics,
                &outcome,
            )
            .await
            {
                Ok(()) => return Ok(()),
                Err(failure) => failure,
            }
        }
        Ok(Err(failure)) => failure,
        Err(_elapsed) => {
            warn!(
                source_id = %source_id,
                timeout_secs = processing_timeout.as_secs(),
                "Source processing timed out"
            );
            PipelineFailure::new(
                TIMEOUT_MESSAGE,
                AppError::Internal(format!(
                    "source processing exceeded its {}s deadline",
                    processing_timeout.as_secs()
                )),
            )
        }
    };

    fail_generation(
        &deps,
        &ownership,
        source_id,
        notebook_id,
        &analytics,
        failure,
    )
    .await
}

/// Extraction, chunking, embedding and storage under one building generation.
///
/// Writes nothing outside the generation it was given: the active index is not
/// read for anything but embedding reuse, and is never deleted or mutated
/// (FR-03).
async fn run_build(
    deps: &ProcessingDeps,
    tasks: &IngestionTasks,
    ownership: &IndexOwnership,
    source_id: Uuid,
    notebook_id: Uuid,
    source_type: SourceType,
) -> Result<BuildOutcome, PipelineFailure> {
    update_source_status(
        deps.source_repo.as_ref(),
        source_id,
        SourceStatus::Processing,
        None,
    )
    .await
    .stage("Failed to record processing status")?;
    deps.broadcaster
        .broadcast_status(notebook_id, source_id, "processing", None);

    let source = get_source(deps.source_repo.as_ref(), source_id)
        .await
        .map_err(PipelineFailure::from_error)?
        .ok_or_else(|| {
            PipelineFailure::from_error(AppError::NotFound("Source not found".into()))
        })?;

    let extracted =
        extraction::extract_content(deps, tasks, source, source_id, notebook_id, source_type)
            .await?;

    extraction::validate_content_limits(&extracted.text, &deps.config)
        .map_err(PipelineFailure::from_error)?;

    let total_chunks =
        indexing::build_chunks(deps, tasks, ownership, &extracted, source_id, notebook_id).await?;

    Ok(BuildOutcome {
        degraded_services: extracted.degraded_services,
        total_chunks,
    })
}

/// The atomic step: validate the generation, publish it, report readiness.
///
/// Publication is the *only* place a source's active pointer moves forward.
/// Until it commits, every reader still sees the previous generation; after it
/// commits, every reader sees the new one. There is no third state.
async fn publish_generation(
    deps: &ProcessingDeps,
    ownership: &IndexOwnership,
    source_id: Uuid,
    notebook_id: Uuid,
    analytics: &AnalyticsCtx<'_>,
    outcome: &BuildOutcome,
) -> Result<(), PipelineFailure> {
    let published = deps
        .generation_repo
        .publish(
            ownership.generation_id,
            source_id,
            ownership.provenance.embedding.dimension,
        )
        .await
        .map_err(|e| {
            PipelineFailure::new(format!("Failed to publish the rebuilt index: {e}"), e)
        })?;

    let chunk_count = published.chunk_count;
    let broadcaster = &deps.broadcaster;

    if outcome.degraded_services.is_empty() {
        broadcaster.broadcast_ready(notebook_id, source_id, chunk_count);
    } else {
        let services: Vec<String> = outcome
            .degraded_services
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        info!(
            source_id = %source_id,
            degraded = ?services,
            "Source processed with degraded services"
        );
        broadcaster.broadcast_ready_degraded(notebook_id, source_id, chunk_count, services);
        analytics
            .events
            .emit(DomainEvent::SourceProcessingDegraded {
                account_id: analytics.account_id,
                notebook_id,
                source_id,
                degraded_services: outcome.degraded_services.clone(),
            });
    }

    analytics
        .events
        .emit(DomainEvent::SourceProcessingCompleted {
            account_id: analytics.account_id,
            notebook_id,
            source_id,
            source_type: analytics.source_type,
            duration_ms: analytics.elapsed_ms(),
            chunk_count,
        });

    info!(
        source_id = %source_id,
        generation_id = %ownership.generation_id,
        chunk_count,
        declared_chunks = outcome.total_chunks,
        "Source processed successfully"
    );

    reclaim_obsolete_generations(deps, source_id).await;
    Ok(())
}

/// Reclaim this source's superseded generations, once publication has committed.
///
/// A publication is the only event that makes a generation obsolete, so it is
/// also the only moment worth looking — which is what keeps retention a real
/// policy rather than a documented one nothing ever runs.
///
/// Deliberately outside the publication transaction and deliberately not
/// fallible: the new index is already live, and disk left behind is an
/// operational cost, not a reason to report a successful rebuild as a failure.
/// Reclaim itself never touches the active generation or the rollback target.
async fn reclaim_obsolete_generations(deps: &ProcessingDeps, source_id: Uuid) {
    match deps
        .generation_repo
        .reclaim(source_id, GENERATION_RETENTION_HOURS)
        .await
    {
        Ok(0) => {}
        Ok(reclaimed) => info!(
            %source_id,
            reclaimed,
            retention_hours = GENERATION_RETENTION_HOURS,
            "Reclaimed obsolete index generations"
        ),
        Err(e) => warn!(
            %source_id,
            error = %e,
            "Failed to reclaim obsolete index generations — the active and \
             rollback-eligible generations are unaffected"
        ),
    }
}

/// The single terminal failure path: fail the generation, tell the user once.
///
/// Marking the generation `failed` is what preserves the previous index: the
/// active pointer is not touched, so a source that had a working index keeps it
/// and stays `ready`, and only a source that never had one reports `error`.
async fn fail_generation(
    deps: &ProcessingDeps,
    ownership: &IndexOwnership,
    source_id: Uuid,
    notebook_id: Uuid,
    analytics: &AnalyticsCtx<'_>,
    failure: PipelineFailure,
) -> Result<(), AppError> {
    if let Err(e) = deps
        .generation_repo
        .mark_failed(ownership.generation_id, source_id, &failure.message)
        .await
    {
        // The generation stays `building` and recovery will reclaim it after
        // the stale deadline. Reporting the original failure matters more than
        // reporting this one, so it is logged rather than returned.
        tracing::error!(
            source_id = %source_id,
            generation_id = %ownership.generation_id,
            error = %e,
            "Failed to mark the building generation as failed"
        );
    }

    deps.broadcaster
        .broadcast_error(notebook_id, source_id, &failure.message);
    analytics.events.emit(DomainEvent::SourceProcessingFailed {
        account_id: analytics.account_id,
        notebook_id,
        source_id,
        source_type: analytics.source_type,
        error_type: categorize_error(&failure.message),
        duration_ms: analytics.elapsed_ms(),
    });

    Err(failure.error)
}

/// Event context for source processing outcomes.
struct AnalyticsCtx<'a> {
    events: &'a SharedEventSink,
    account_id: Uuid,
    source_type: SourceType,
    pipeline_start: std::time::Instant,
}

impl AnalyticsCtx<'_> {
    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.pipeline_start.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// Categorize an error message into a short error type string.
fn categorize_error(message: &str) -> &'static str {
    let msg = message.to_lowercase();
    if msg.contains("ocr") {
        "ocr_error"
    } else if msg.contains("pdf") {
        "pdf_parse_error"
    } else if msg.contains("embedding") {
        "embedding_error"
    } else if msg.contains("timeout") || msg.contains("timed out") {
        "timeout"
    } else if msg.contains("chunk") {
        "chunking_error"
    } else if msg.contains("extract") || msg.contains("scrape") || msg.contains("firecrawl") {
        "extraction_error"
    } else if msg.contains("size") || msg.contains("too large") || msg.contains("empty") {
        "validation_error"
    } else {
        "unknown_error"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_context_prefixes_the_stage_and_keeps_the_error() {
        let failure: PipelineFailure = Err::<(), _>(AppError::Validation("empty".into()))
            .stage("Failed to chunk content")
            .expect_err("an error must stay an error");

        assert!(
            failure.message.starts_with("Failed to chunk content: "),
            "{}",
            failure.message
        );
        assert!(matches!(failure.error, AppError::Validation(_)));
    }

    #[test]
    fn recovery_waits_longer_than_a_run_is_allowed_to_take() {
        // A build inside one deadline can still have a live owner, so recovery
        // must not be able to reclaim it.
        assert!(
            PROCESSING_TIMEOUT.saturating_mul(STALE_BUILD_MULTIPLIER) > PROCESSING_TIMEOUT,
            "a stale build must be older than one deadline"
        );
    }

    #[test]
    fn categorize_error_names_the_stage_that_failed() {
        assert_eq!(categorize_error("OCR processing failed"), "ocr_error");
        assert_eq!(categorize_error("Processing timed out"), "timeout");
        assert_eq!(categorize_error("something else"), "unknown_error");
    }
}
