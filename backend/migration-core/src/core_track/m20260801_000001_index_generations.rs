//! Immutable index generations (US-006, EP-002).
//!
//! A source index becomes an immutable *generation* with exactly one atomic
//! publication point. The invariants this migration installs are the ones
//! `docs/architecture/index-generations.md` specifies, and they are enforced by
//! the database rather than by convention:
//!
//! | Invariant | Enforced by |
//! |---|---|
//! | at most one building generation per source | partial unique index `source_index_generations_one_building` |
//! | no duplicate chunk position inside a generation | unique index `chunks_generation_chunk_index_unique` |
//! | a chunk belongs to a generation of *its own* source | composite FK `chunks_generation_fk` |
//! | a source can only publish a generation it owns | composite FK `sources_active_generation_fk` |
//! | a referenced generation cannot be reclaimed | the same FK, `NO ACTION` |
//!
//! Both composite foreign keys reference `(id, source_id)`, which is why that
//! pair carries its own unique constraint. `NO ACTION` rather than
//! `ON DELETE SET NULL` is deliberate: the column-list form of `SET NULL` needs
//! PostgreSQL 15 and this build supports 14, and `NO ACTION` is checked at the
//! end of the statement, so deleting a source still cascades cleanly while
//! deleting a *published* generation on its own is refused.
//!
//! The migration is additive and idempotent. Existing chunks are backfilled
//! into one published generation per source, carrying `legacy` provenance: their
//! real provider, model and chunking configuration were never recorded, and
//! inventing a fingerprint would let a later reprocess reuse embeddings it
//! cannot prove compatible. A `legacy` fingerprint never matches a live one, so
//! the first reprocess rebuilds instead.

use sea_orm_migration::prelude::*;
use sea_orm_migration::sea_orm::{ConnectionTrait, DatabaseBackend, FromQueryResult, Statement};

#[derive(DeriveMigrationName)]
pub struct Migration;

/// Provenance recorded for chunks that predate generations.
///
/// Consumed by `openbooklm::services::rag::provenance`, which treats any
/// fingerprint starting with `legacy:` as incompatible with every live one.
const LEGACY_EMBEDDING_FINGERPRINT: &str = "legacy:unknown";
const LEGACY_CHUNKING_FINGERPRINT: &str = "legacy:unknown";

/// The one embedding width the core has ever stored, pinned by `vector(1024)`.
const LEGACY_EMBEDDING_DIMENSION: i32 = 1024;

/// A defect and the source it belongs to. `source_id` is selected as text so
/// this crate needs no UUID feature to report it.
#[derive(Debug, FromQueryResult)]
struct Diagnostic {
    source_id: String,
    detail: String,
}

/// Refuse to publish an ambiguous generation.
///
/// Legacy rows can carry three defects that a generation cannot represent:
/// a duplicate `(source_id, chunk_index)` pair has no single position inside a
/// generation, a NULL embedding is not searchable, and a chunk whose source no
/// longer exists belongs to no generation at all. Each aborts with the source id
/// that caused it, because "the backfill failed" is not something an operator
/// can act on.
///
/// Every check is scoped to the rows the backfill will actually claim
/// (`generation_id IS NULL`), which is why it runs *after* that column is
/// added. That scope is what keeps a replay idempotent rather than merely
/// idempotent-on-a-fresh-database: once generations exist, two of them
/// legitimately hold the same `(source_id, chunk_index)` pair — uniqueness is
/// `(generation_id, chunk_index)`, and keeping the previous generation for
/// rollback is the normal state, not a defect. Validating the whole table would
/// abort on healthy data.
async fn assert_backfillable<C: ConnectionTrait>(db: &C) -> Result<(), DbErr> {
    let checks: [(&str, &str); 3] = [
        (
            "duplicate chunk positions",
            r"SELECT source_id::text AS source_id,
                     'chunk_index ' || chunk_index::text || ' appears ' ||
                     count(*)::text || ' times' AS detail
              FROM chunks WHERE generation_id IS NULL
              GROUP BY source_id, chunk_index HAVING count(*) > 1
              ORDER BY source_id LIMIT 5",
        ),
        (
            "missing embeddings",
            r"SELECT source_id::text AS source_id,
                     count(*)::text || ' chunk(s) have a NULL embedding' AS detail
              FROM chunks WHERE embedding IS NULL AND generation_id IS NULL
              GROUP BY source_id ORDER BY source_id LIMIT 5",
        ),
        (
            "orphaned chunks",
            r"SELECT c.source_id::text AS source_id,
                     count(*)::text || ' chunk(s) reference a missing source' AS detail
              FROM chunks c LEFT JOIN sources s ON s.id = c.source_id
              WHERE s.id IS NULL AND c.generation_id IS NULL
              GROUP BY c.source_id ORDER BY c.source_id LIMIT 5",
        ),
    ];

    for (kind, sql) in checks {
        let rows = Diagnostic::find_by_statement(Statement::from_string(
            DatabaseBackend::Postgres,
            sql.to_owned(),
        ))
        .all(db)
        .await?;

        if let Some(first) = rows.first() {
            let extra = rows
                .iter()
                .skip(1)
                .map(|r| format!("\n  source {}: {}", r.source_id, r.detail))
                .collect::<String>();
            return Err(DbErr::Custom(format!(
                "index-generation backfill aborted — {kind} in `chunks`:\n  \
                 source {}: {}{extra}\n\n\
                 Fix the listed sources (deduplicate, re-embed or delete their chunks) and run \
                 the migration again. The backfill refuses to publish a generation whose \
                 contents it cannot prove complete.",
                first.source_id, first.detail
            )));
        }
    }

    Ok(())
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // ── Generation identity ──────────────────────────────────────────
        db.execute_unprepared(
            r"
            CREATE TABLE IF NOT EXISTS source_index_generations (
                id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_id               UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                state                   VARCHAR(20) NOT NULL,
                expected_chunk_count    INTEGER,
                stored_chunk_count      INTEGER NOT NULL DEFAULT 0,
                embedding_fingerprint   TEXT NOT NULL,
                chunking_fingerprint    TEXT NOT NULL,
                embedding_provider      TEXT NOT NULL,
                embedding_model         TEXT NOT NULL,
                embedding_dimension     INTEGER NOT NULL,
                embedding_normalization TEXT NOT NULL,
                chunking_schema_version INTEGER NOT NULL,
                chunk_size_unit         TEXT NOT NULL,
                chunk_sizer_identity    TEXT NOT NULL,
                failure_reason          TEXT,
                created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
                published_at            TIMESTAMPTZ,
                failed_at               TIMESTAMPTZ,
                CONSTRAINT source_index_generations_state_check
                    CHECK (state IN ('building', 'published', 'failed', 'superseded')),
                CONSTRAINT source_index_generations_id_source_unique UNIQUE (id, source_id)
            );

            CREATE INDEX IF NOT EXISTS idx_source_index_generations_source
                ON source_index_generations(source_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_source_index_generations_state
                ON source_index_generations(state, created_at);
            CREATE UNIQUE INDEX IF NOT EXISTS source_index_generations_one_building
                ON source_index_generations(source_id) WHERE (state = 'building');
            ",
        )
        .await?;

        // ── Chunk and source membership columns ──────────────────────────
        db.execute_unprepared(
            r"
            ALTER TABLE chunks ADD COLUMN IF NOT EXISTS generation_id UUID;
            ALTER TABLE sources ADD COLUMN IF NOT EXISTS active_generation_id UUID;
            ",
        )
        .await?;

        // Validation runs here, not before, so it can scope itself to the rows
        // the backfill will claim. Both statements above are additive, and the
        // whole `up` shares one transaction, so a refusal below leaves nothing
        // behind.
        assert_backfillable(db).await?;

        // ── Backfill ─────────────────────────────────────────────────────
        // One published generation per source that already has chunks, in a
        // single statement per step: the whole migration runs inside SeaORM's
        // transaction, so a partial backfill is not a state this can leave
        // behind. Sources with no chunks keep a NULL pointer, which is what an
        // unindexed source means.
        db.execute_unprepared(&format!(
            r"
            INSERT INTO source_index_generations (
                source_id, state, expected_chunk_count, stored_chunk_count,
                embedding_fingerprint, chunking_fingerprint,
                embedding_provider, embedding_model, embedding_dimension,
                embedding_normalization, chunking_schema_version,
                chunk_size_unit, chunk_sizer_identity, published_at
            )
            SELECT c.source_id, 'published', count(*)::int, count(*)::int,
                   '{LEGACY_EMBEDDING_FINGERPRINT}', '{LEGACY_CHUNKING_FINGERPRINT}',
                   'legacy', 'legacy', {LEGACY_EMBEDDING_DIMENSION},
                   'unknown', 0, 'unknown', 'legacy', now()
            FROM chunks c
            JOIN sources s ON s.id = c.source_id
            WHERE c.generation_id IS NULL AND s.active_generation_id IS NULL
            GROUP BY c.source_id;

            UPDATE chunks c
               SET generation_id = g.id
              FROM source_index_generations g
             WHERE g.source_id = c.source_id
               AND g.embedding_fingerprint = '{LEGACY_EMBEDDING_FINGERPRINT}'
               AND c.generation_id IS NULL;

            UPDATE sources s
               SET active_generation_id = g.id
              FROM source_index_generations g
             WHERE g.source_id = s.id
               AND g.embedding_fingerprint = '{LEGACY_EMBEDDING_FINGERPRINT}'
               AND s.active_generation_id IS NULL;
            "
        ))
        .await?;

        // ── Constraints ──────────────────────────────────────────────────
        // `NOT NULL` last: every remaining chunk now has a generation, and the
        // constraint is what stops a future write from skipping one.
        db.execute_unprepared(
            r"
            ALTER TABLE chunks ALTER COLUMN generation_id SET NOT NULL;

            CREATE UNIQUE INDEX IF NOT EXISTS chunks_generation_chunk_index_unique
                ON chunks(generation_id, chunk_index);
            CREATE INDEX IF NOT EXISTS idx_chunks_generation_id ON chunks(generation_id);
            CREATE INDEX IF NOT EXISTS idx_sources_active_generation
                ON sources(active_generation_id);
            ",
        )
        .await?;

        // `ADD CONSTRAINT` has no `IF NOT EXISTS`, so re-running the migration
        // on a database that already carries these would fail on the duplicate
        // name rather than on anything meaningful. The guard makes the whole
        // `up` a no-op the second time.
        db.execute_unprepared(
            r"
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'chunks_generation_fk'
                ) THEN
                    ALTER TABLE chunks ADD CONSTRAINT chunks_generation_fk
                        FOREIGN KEY (generation_id, source_id)
                        REFERENCES source_index_generations(id, source_id)
                        ON DELETE CASCADE;
                END IF;

                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'sources_active_generation_fk'
                ) THEN
                    ALTER TABLE sources ADD CONSTRAINT sources_active_generation_fk
                        FOREIGN KEY (active_generation_id, id)
                        REFERENCES source_index_generations(id, source_id);
                END IF;
            END $$;
            ",
        )
        .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Reversible for a fresh install only, like the baseline. Dropping
        // `generation_id` discards which generation each chunk belonged to;
        // the chunks themselves survive, so a downgrade leaves a searchable
        // pre-generation corpus rather than an empty one.
        manager
            .get_connection()
            .execute_unprepared(
                r"
                ALTER TABLE sources DROP CONSTRAINT IF EXISTS sources_active_generation_fk;
                ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_generation_fk;
                DROP INDEX IF EXISTS chunks_generation_chunk_index_unique;
                DROP INDEX IF EXISTS idx_chunks_generation_id;
                DROP INDEX IF EXISTS idx_sources_active_generation;
                ALTER TABLE sources DROP COLUMN IF EXISTS active_generation_id;
                ALTER TABLE chunks DROP COLUMN IF EXISTS generation_id;
                DROP TABLE IF EXISTS source_index_generations;
                ",
            )
            .await?;
        Ok(())
    }
}
