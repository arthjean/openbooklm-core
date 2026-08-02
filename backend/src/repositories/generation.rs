//! Index generation lifecycle (US-007, US-008, US-009, EP-002).
//!
//! Every statement here is raw SQL for the same reason the chunk repository is:
//! the invariants are database constraints, and the code that depends on them
//! has to be able to see which constraint it is relying on.
//!
//! The one rule worth stating up front: **the active pointer moves in exactly
//! one transaction, and nothing else ever writes it**. Readers therefore see
//! either the whole previous generation or the whole replacement. There is no
//! window in which a source has half of each, because there is no statement
//! that could produce one.

use async_trait::async_trait;
use sea_orm::{
    ConnectionTrait, DatabaseConnection, DatabaseTransaction, DbBackend, Statement,
    TransactionTrait,
};
use uuid::Uuid;

use crate::error::{AppError, RagError};
use crate::services::rag::provenance::GenerationProvenance;

use super::traits::{GenerationRepository, PublicationOutcome, RepoResult};

// ============================================================================
// SQL
// ============================================================================

/// Claim ownership of a source's index by inserting the only `building` row it
/// is allowed to have.
///
/// `source_index_generations_one_building` is what makes this a compare-and-set:
/// a second concurrent worker gets a unique violation rather than a second
/// index. `ON CONFLICT DO NOTHING` turns that into an empty result, which the
/// caller reads as "someone else owns this".
const CLAIM_OWNERSHIP_SQL: &str = r"
    INSERT INTO source_index_generations (
        source_id, state, embedding_fingerprint, chunking_fingerprint,
        embedding_provider, embedding_model, embedding_dimension,
        embedding_normalization, chunking_schema_version,
        chunk_size_unit, chunk_sizer_identity
    )
    VALUES ($1, 'building', $2, $3, $4, $5, $6, $7, $8, $9, $10)
    ON CONFLICT (source_id) WHERE (state = 'building') DO NOTHING
    RETURNING id
";

const FIND_BUILDING_SQL: &str = r"
    SELECT id
    FROM source_index_generations
    WHERE source_id = $1 AND state = 'building'
";

/// Freeze the generation before inspecting any child row. Chunk writers hold
/// `FOR SHARE` on the same row, so this lock waits for every admitted batch and
/// prevents every later batch from entering once publication changes state.
const LOCK_FOR_PUBLICATION_SQL: &str = r"
    SELECT id
    FROM source_index_generations
    WHERE id = $1 AND source_id = $2 AND state = 'building'
    FOR UPDATE
";

const RECORD_BUILD_PLAN_SQL: &str = r"
    UPDATE source_index_generations
       SET expected_chunk_count = $2,
           chunking_fingerprint = $3,
           chunking_schema_version = $4,
           chunk_size_unit = $5,
           chunk_sizer_identity = $6
     WHERE id = $1 AND state = 'building'
";

/// Recount from `chunks` rather than trusting an incremented counter: the
/// counter is a cache, the rows are the truth, and publication validates the
/// truth.
const SYNC_STORED_COUNT_SQL: &str = r"
    UPDATE source_index_generations g
       SET stored_chunk_count = (SELECT count(*) FROM chunks c WHERE c.generation_id = g.id)
     WHERE g.id = $1
 RETURNING g.stored_chunk_count
";

const MARK_FAILED_SQL: &str = r"
    UPDATE source_index_generations
       SET state = 'failed', failure_reason = $2, failed_at = now()
     WHERE id = $1 AND state = 'building'
";

/// Validate and publish in one statement pair inside one transaction.
///
/// The `WHERE` clause carries the whole contract: the generation must still be
/// building, must belong to the source it claims, must know how many chunks it
/// expected, and must have stored exactly that many. A generation failing any
/// of them updates zero rows and never becomes active.
const PUBLISH_GENERATION_SQL: &str = r"
    UPDATE source_index_generations
       SET state = 'published', published_at = now()
     WHERE id = $1
       AND source_id = $2
       AND state = 'building'
       AND expected_chunk_count IS NOT NULL
       AND expected_chunk_count = stored_chunk_count
       AND stored_chunk_count > 0
 RETURNING stored_chunk_count
";

const POINT_SOURCE_SQL: &str = r"
    UPDATE sources
       SET active_generation_id = $1, chunk_count = $3, status = 'ready', error_message = NULL
     WHERE id = $2
";

/// Serialize every active-pointer move for one source. Publication's final
/// `UPDATE sources` takes the same row lock, so a rollback either observes the
/// pointer before that publication or after it, never a stale value in between.
const LOCK_SOURCE_SQL: &str = r"
    SELECT id
    FROM sources
    WHERE id = $1
    FOR UPDATE
";

/// Every embedding under a generation must have the configured width and only
/// finite components.
///
/// pgvector rejects a NaN or infinite component at insert time, so this is a
/// defence in depth against a future write path that bypasses the type, and the
/// width check is the one that actually catches a provider change mid-build.
const VALIDATE_VECTORS_SQL: &str = r"
    SELECT count(*) FILTER (WHERE embedding IS NULL)                     AS null_vectors,
           count(*) FILTER (WHERE vector_dims(embedding) <> $2)          AS wrong_width,
           count(*) FILTER (WHERE embedding::text ~ '(NaN|Infinity)')    AS non_finite,
           count(*)                                                      AS total
    FROM chunks WHERE generation_id = $1
";

const PREVIOUS_PUBLISHED_SQL: &str = r"
    SELECT g.id, g.stored_chunk_count
    FROM source_index_generations g
    JOIN sources s ON s.id = g.source_id
    JOIN source_index_generations active ON active.id = s.active_generation_id
    WHERE g.source_id = $1
      AND g.state = 'published'
      AND g.id <> s.active_generation_id
      AND g.embedding_fingerprint = active.embedding_fingerprint
    ORDER BY g.published_at DESC, g.id DESC
    LIMIT 1
";

const LIST_FOR_SOURCE_SQL: &str = r"
    SELECT id
    FROM source_index_generations
    WHERE source_id = $1
    ORDER BY created_at DESC
";

/// Generations a reclaim pass may delete.
///
/// Three exclusions, each protecting a different guarantee: the active pointer
/// (never delete what search reads), the newest compatible published
/// generation (never delete the rollback target), and anything younger than
/// the retention window (never delete what an operator is still deciding
/// about).
const RECLAIMABLE_SQL: &str = r"
    SELECT g.id
    FROM source_index_generations g
    JOIN sources s ON s.id = g.source_id
    WHERE g.source_id = $1
      AND (s.active_generation_id IS NULL OR g.id <> s.active_generation_id)
      AND g.id <> COALESCE((
            SELECT p.id FROM source_index_generations p
            WHERE p.source_id = g.source_id AND p.state = 'published'
              AND (s.active_generation_id IS NULL OR p.id <> s.active_generation_id)
              AND p.embedding_fingerprint = COALESCE((
                    SELECT active.embedding_fingerprint
                    FROM source_index_generations active
                    WHERE active.id = s.active_generation_id
                  ), '')
            ORDER BY p.published_at DESC, p.id DESC LIMIT 1
          ), '00000000-0000-0000-0000-000000000000'::uuid)
      AND g.created_at < now() - make_interval(hours => $2)
      AND g.state <> 'building'
";

const DELETE_GENERATION_SQL: &str = "DELETE FROM source_index_generations WHERE id = $1";

/// Building generations older than the deadline, with the source they block.
const STALE_BUILDING_SQL: &str = r"
    SELECT g.id, g.source_id
    FROM source_index_generations g
    WHERE g.state = 'building'
      AND g.created_at < now() - make_interval(secs => $1)
";

const FAIL_STALE_SQL: &str = r"
    UPDATE source_index_generations
       SET state = 'failed', failure_reason = $2, failed_at = now()
     WHERE id = $1 AND state = 'building'
";

/// Restore the source's own status after a build failed, without touching the
/// active pointer.
///
/// A source that still has an active generation stays `ready`: its previous
/// index is intact and searchable, and reporting `error` would tell the user
/// their document disappeared when it did not.
///
/// The `NOT EXISTS` guard is what stops a superseded worker from reporting an
/// outcome for someone else's build (US-009). A worker whose generation was
/// reclaimed can still be running; when it finally fails, another owner may
/// already be building, and overwriting that owner's `processing` status with
/// `ready` would tell the user a rebuild finished that has not started.
const RESTORE_SOURCE_STATUS_SQL: &str = r"
    UPDATE sources s
       SET status = CASE WHEN s.active_generation_id IS NULL THEN 'error' ELSE 'ready' END,
           error_message = CASE WHEN s.active_generation_id IS NULL THEN $2 ELSE NULL END
     WHERE s.id = $1
       AND NOT EXISTS (
           SELECT 1 FROM source_index_generations g
           WHERE g.source_id = s.id AND g.state = 'building'
       )
";

// ============================================================================
// Implementation
// ============================================================================

#[derive(Clone)]
pub struct SeaOrmGenerationRepository {
    db: DatabaseConnection,
}

impl SeaOrmGenerationRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    fn stmt(sql: &str, values: impl IntoIterator<Item = sea_orm::Value>) -> Statement {
        Statement::from_sql_and_values(DbBackend::Postgres, sql, values)
    }
}

/// Read one `bigint` aggregate out of a row, defaulting to 0.
fn count_of(row: &sea_orm::QueryResult, column: &str) -> i64 {
    row.try_get::<i64>("", column).unwrap_or(0)
}

#[async_trait]
impl GenerationRepository for SeaOrmGenerationRepository {
    #[tracing::instrument(skip(self, provenance), fields(%source_id))]
    async fn claim(
        &self,
        source_id: Uuid,
        provenance: &GenerationProvenance,
    ) -> RepoResult<Option<Uuid>> {
        let embedding = &provenance.embedding;
        let chunking = &provenance.chunking;
        let rows = self
            .db
            .query_all(Self::stmt(
                CLAIM_OWNERSHIP_SQL,
                [
                    source_id.into(),
                    embedding.fingerprint().into(),
                    chunking.fingerprint().into(),
                    embedding.provider.clone().into(),
                    embedding.model.clone().into(),
                    i32::try_from(embedding.dimension)
                        .unwrap_or(i32::MAX)
                        .into(),
                    embedding.normalization.as_str().into(),
                    chunking.schema_version.into(),
                    chunking.size_unit.into(),
                    chunking.sizer_identity.into(),
                ],
            ))
            .await?;

        rows.first()
            .map(|row| row.try_get::<Uuid>("", "id").map_err(AppError::from))
            .transpose()
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn find_building(&self, source_id: Uuid) -> RepoResult<Option<Uuid>> {
        let rows = self
            .db
            .query_all(Self::stmt(FIND_BUILDING_SQL, [source_id.into()]))
            .await?;

        rows.first()
            .map(|row| row.try_get::<Uuid>("", "id").map_err(AppError::from))
            .transpose()
    }

    #[tracing::instrument(skip(self, chunking), fields(%generation_id, %expected))]
    async fn record_build_plan(
        &self,
        generation_id: Uuid,
        expected: i32,
        chunking: &crate::services::rag::provenance::ChunkingProvenance,
    ) -> RepoResult<()> {
        self.db
            .execute(Self::stmt(
                RECORD_BUILD_PLAN_SQL,
                [
                    generation_id.into(),
                    expected.into(),
                    chunking.fingerprint().into(),
                    chunking.schema_version.into(),
                    chunking.size_unit.into(),
                    chunking.sizer_identity.into(),
                ],
            ))
            .await?;
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(%generation_id, %source_id))]
    async fn publish(
        &self,
        generation_id: Uuid,
        source_id: Uuid,
        expected_dimension: usize,
    ) -> RepoResult<PublicationOutcome> {
        let txn = self.db.begin().await?;

        // Validation and publication share one transaction on purpose: a vector
        // written between the check and the pointer move would otherwise be
        // published unchecked.
        let outcome = publish_in_txn(&txn, generation_id, source_id, expected_dimension).await;

        match outcome {
            Ok(result) => {
                txn.commit().await?;
                Ok(result)
            }
            Err(e) => {
                if let Err(rollback) = txn.rollback().await {
                    tracing::error!(%generation_id, error = %rollback, "publication rollback failed");
                }
                Err(e)
            }
        }
    }

    #[tracing::instrument(skip(self, reason), fields(%generation_id, %source_id))]
    async fn mark_failed(
        &self,
        generation_id: Uuid,
        source_id: Uuid,
        reason: &str,
    ) -> RepoResult<()> {
        let txn = self.db.begin().await?;
        txn.execute(Self::stmt(
            MARK_FAILED_SQL,
            [generation_id.into(), reason.into()],
        ))
        .await?;
        txn.execute(Self::stmt(
            RESTORE_SOURCE_STATUS_SQL,
            [source_id.into(), reason.into()],
        ))
        .await?;
        txn.commit().await?;
        Ok(())
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn rollback_to_previous(&self, source_id: Uuid) -> RepoResult<Option<Uuid>> {
        let txn = self.db.begin().await?;

        let source = txn
            .query_one(Self::stmt(LOCK_SOURCE_SQL, [source_id.into()]))
            .await?;
        if source.is_none() {
            txn.rollback().await?;
            return Ok(None);
        }

        let rows = txn
            .query_all(Self::stmt(PREVIOUS_PUBLISHED_SQL, [source_id.into()]))
            .await?;
        let Some(row) = rows.first() else {
            txn.rollback().await?;
            return Ok(None);
        };
        let previous: Uuid = row.try_get("", "id")?;
        let chunk_count: i32 = row.try_get("", "stored_chunk_count")?;

        // Same statement as publication uses. Rollback is a pointer move, not a
        // copy: the chunks of the previous generation were never deleted.
        txn.execute(Self::stmt(
            POINT_SOURCE_SQL,
            [previous.into(), source_id.into(), chunk_count.into()],
        ))
        .await?;
        txn.commit().await?;
        Ok(Some(previous))
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn list_for_source(&self, source_id: Uuid) -> RepoResult<Vec<Uuid>> {
        let rows = self
            .db
            .query_all(Self::stmt(LIST_FOR_SOURCE_SQL, [source_id.into()]))
            .await?;

        rows.into_iter()
            .map(|row| row.try_get("", "id"))
            .collect::<Result<_, sea_orm::DbErr>>()
            .map_err(Into::into)
    }

    #[tracing::instrument(skip(self), fields(%source_id, %retention_hours))]
    async fn reclaim(&self, source_id: Uuid, retention_hours: i32) -> RepoResult<u64> {
        let txn = self.db.begin().await?;
        // Publication and rollback move the active pointer while holding this
        // same row lock. Keeping it through selection and deletion prevents a
        // fingerprint change from making a selected generation the new
        // rollback target between those two operations.
        txn.query_one(Self::stmt(LOCK_SOURCE_SQL, [source_id.into()]))
            .await?;
        let rows = txn
            .query_all(Self::stmt(
                RECLAIMABLE_SQL,
                [source_id.into(), retention_hours.into()],
            ))
            .await?;

        let mut reclaimed = 0;
        for row in rows {
            let id: Uuid = row.try_get("", "id")?;
            reclaimed += txn
                .execute(Self::stmt(DELETE_GENERATION_SQL, [id.into()]))
                .await?
                .rows_affected();
        }
        txn.commit().await?;
        Ok(reclaimed)
    }

    #[tracing::instrument(skip(self), fields(%older_than_secs))]
    async fn fail_stale_builds(&self, older_than_secs: i64, reason: &str) -> RepoResult<u64> {
        let rows = self
            .db
            .query_all(Self::stmt(STALE_BUILDING_SQL, [older_than_secs.into()]))
            .await?;

        let mut failed = 0;
        for row in rows {
            let generation_id: Uuid = row.try_get("", "id")?;
            let source_id: Uuid = row.try_get("", "source_id")?;
            let txn = self.db.begin().await?;
            txn.execute(Self::stmt(
                FAIL_STALE_SQL,
                [generation_id.into(), reason.into()],
            ))
            .await?;
            txn.execute(Self::stmt(
                RESTORE_SOURCE_STATUS_SQL,
                [source_id.into(), reason.into()],
            ))
            .await?;
            txn.commit().await?;
            failed += 1;
        }
        Ok(failed)
    }
}

/// The publication transaction: validate, mark published, move the pointer.
///
/// Split out so the whole sequence is visible in one place. Every early return
/// leaves the source pointing where it already pointed.
async fn publish_in_txn(
    txn: &DatabaseTransaction,
    generation_id: Uuid,
    source_id: Uuid,
    expected_dimension: usize,
) -> Result<PublicationOutcome, AppError> {
    let dimension = i32::try_from(expected_dimension).unwrap_or(i32::MAX);

    let locked = txn
        .query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            LOCK_FOR_PUBLICATION_SQL,
            [generation_id.into(), source_id.into()],
        ))
        .await?;
    if locked.is_none() {
        return Err(vector_error(
            source_id,
            "generation is not owned by this source or is no longer building",
        ));
    }

    // ── Vector validation ────────────────────────────────────────────────
    let rows = txn
        .query_all(Statement::from_sql_and_values(
            DbBackend::Postgres,
            VALIDATE_VECTORS_SQL,
            [generation_id.into(), dimension.into()],
        ))
        .await?;
    let Some(row) = rows.first() else {
        return Err(vector_error(
            source_id,
            "vector validation returned no rows",
        ));
    };

    let total = count_of(row, "total");
    if total == 0 {
        return Err(vector_error(
            source_id,
            "generation holds zero chunks — content produced no indexable text",
        ));
    }
    for (column, message) in [
        ("null_vectors", "chunk(s) have no embedding"),
        ("wrong_width", "chunk(s) have the wrong embedding width"),
        (
            "non_finite",
            "chunk(s) hold a non-finite embedding component",
        ),
    ] {
        let count = count_of(row, column);
        if count > 0 {
            return Err(vector_error(source_id, &format!("{count} {message}")));
        }
    }

    // ── Count reconciliation ─────────────────────────────────────────────
    let stored_rows = txn
        .query_all(Statement::from_sql_and_values(
            DbBackend::Postgres,
            SYNC_STORED_COUNT_SQL,
            [generation_id.into()],
        ))
        .await?;
    if stored_rows.is_empty() {
        return Err(vector_error(source_id, "generation no longer exists"));
    }

    // ── Publication ──────────────────────────────────────────────────────
    let published = txn
        .query_all(Statement::from_sql_and_values(
            DbBackend::Postgres,
            PUBLISH_GENERATION_SQL,
            [generation_id.into(), source_id.into()],
        ))
        .await?;
    let Some(published_row) = published.first() else {
        return Err(vector_error(
            source_id,
            "generation is no longer a building generation of this source, or its stored chunk \
             count disagrees with the count it declared",
        ));
    };
    let chunk_count: i32 = published_row.try_get("", "stored_chunk_count")?;

    // ── The single atomic pointer move ───────────────────────────────────
    txn.execute(Statement::from_sql_and_values(
        DbBackend::Postgres,
        POINT_SOURCE_SQL,
        [generation_id.into(), source_id.into(), chunk_count.into()],
    ))
    .await?;

    Ok(PublicationOutcome {
        generation_id,
        chunk_count,
    })
}

fn vector_error(source_id: Uuid, reason: &str) -> AppError {
    RagError::VectorStoreFailed {
        source_id: source_id.to_string(),
        reason: format!("generation publication refused: {reason}"),
    }
    .into()
}
