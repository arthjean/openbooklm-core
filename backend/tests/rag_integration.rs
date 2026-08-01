//! Index-generation integration suite (EP-002).
//!
//! Every test here needs a real PostgreSQL with pgvector, because every claim
//! EP-002 makes is a claim about database semantics: what a concurrent reader
//! observes across a commit, what a unique index refuses, what a foreign key
//! protects. Asserting those against a mock would assert what the mock does.
//!
//! ```bash
//! TEST_DATABASE_URL=postgres://openbooklm:openbooklm@localhost:5432/openbooklm \
//!   cargo test --no-default-features --test rag_integration -- --ignored
//! ```
//!
//! Each test provisions its own schema-isolated fixture and cleans up after
//! itself, so the suite is re-runnable against a persistent database.

#![cfg(test)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement};
use uuid::Uuid;

use openbooklm::core::config::DatabasePoolConfig;
use openbooklm::core::providers::{EMBEDDING_DIM, EmbeddingProvider};
use openbooklm::error::AppError;
use openbooklm::repositories::{
    ChunkRepository, GenerationRepository, SeaOrmChunkRepository, SeaOrmGenerationRepository,
    SeaOrmSearchRepository, SearchRepository,
};
use openbooklm::services::rag::provenance::{
    ChunkingProvenance, EmbeddingProvenance, GenerationProvenance, Normalization,
};
use openbooklm::types::{ChunkMetadata, ChunkWithContext};
use openbooklm_migration_core::MigratorTrait;
use openbooklm_migration_core::core_track::CoreMigrator;

// ============================================================================
// Fixture
// ============================================================================

/// The number of deterministic schedules the publication stress test runs.
///
/// The PRD's Definition of Done for EP-002 names 1,000; this is that number,
/// not a sample of it.
const PUBLICATION_SCHEDULES: usize = 1_000;

/// Concurrent reprocess requests the ownership test issues (US-009).
const OWNERSHIP_REQUESTS: usize = 100;

struct Fixture {
    db: DatabaseConnection,
    account_id: Uuid,
    notebook_id: Uuid,
    chunks: SeaOrmChunkRepository,
    generations: SeaOrmGenerationRepository,
    search: SeaOrmSearchRepository,
}

impl Fixture {
    /// Connect, apply the core track, and create one synthetic account/notebook.
    ///
    /// Returns `None` when `TEST_DATABASE_URL` is unset, so an accidental run
    /// without a database skips rather than fails misleadingly.
    ///
    /// The migration is applied here rather than assumed, because assuming it
    /// is what makes an unmigrated database fail as `relation "accounts" does
    /// not exist` in twenty-one tests at once, which reads like a regression
    /// and is not one. `CoreMigrator::up` is idempotent, so on a database that
    /// is already migrated this costs one query against the history table.
    async fn setup() -> Option<Self> {
        let db_url = std::env::var("TEST_DATABASE_URL").ok()?;
        let db = openbooklm::db::connect(&db_url, &DatabasePoolConfig::default())
            .await
            .expect("connect to the test database");
        CoreMigrator::up(&db, None)
            .await
            .expect("apply the core track to the test database");

        let account_id = Uuid::new_v4();
        let notebook_id = Uuid::new_v4();
        exec(
            &db,
            "INSERT INTO accounts (id) VALUES ($1)",
            [account_id.into()],
        )
        .await;
        exec(
            &db,
            "INSERT INTO notebooks (id, user_id, title) VALUES ($1, $2, 'EP-002 fixture')",
            [notebook_id.into(), account_id.into()],
        )
        .await;

        Some(Self {
            chunks: SeaOrmChunkRepository::new(&db),
            generations: SeaOrmGenerationRepository::new(&db),
            search: SeaOrmSearchRepository::new(&db),
            db,
            account_id,
            notebook_id,
        })
    }

    async fn create_source(&self, title: &str) -> Uuid {
        let source_id = Uuid::new_v4();
        exec(
            &self.db,
            "INSERT INTO sources (id, notebook_id, title, source_type, content, status)
             VALUES ($1, $2, $3, 'text', 'fixture content', 'pending')",
            [source_id.into(), self.notebook_id.into(), title.into()],
        )
        .await;
        source_id
    }

    /// Build and publish one complete generation, returning its id.
    async fn publish_generation(
        &self,
        source_id: Uuid,
        marker: &str,
        count: usize,
        provenance: &GenerationProvenance,
    ) -> Uuid {
        let generation_id = self
            .generations
            .claim(source_id, provenance)
            .await
            .expect("claim")
            .expect("no competing build");
        let (chunks, embeddings) = synthetic_chunks(marker, count);
        self.chunks
            .store_chunks(generation_id, source_id, &chunks, &embeddings)
            .await
            .expect("store chunks");
        self.generations
            .record_build_plan(
                generation_id,
                i32::try_from(count).expect("fixture size fits i32"),
                &provenance.chunking,
            )
            .await
            .expect("record build plan");
        let _published = self
            .generations
            .publish(generation_id, source_id, EMBEDDING_DIM)
            .await
            .expect("publish");
        generation_id
    }

    /// Every chunk the notebook's active generations expose, via lexical search.
    async fn active_contents(&self, query: &str) -> Vec<String> {
        self.search
            .search_lexical_chunks(self.notebook_id, query, 500)
            .await
            .expect("lexical search")
            .into_iter()
            .map(|r| r.content)
            .collect()
    }

    async fn cleanup(&self) {
        exec(
            &self.db,
            "DELETE FROM accounts WHERE id = $1",
            [self.account_id.into()],
        )
        .await;
    }
}

async fn exec(
    db: &DatabaseConnection,
    sql: &str,
    values: impl IntoIterator<Item = sea_orm::Value>,
) {
    db.execute(Statement::from_sql_and_values(
        DbBackend::Postgres,
        sql,
        values,
    ))
    .await
    .unwrap_or_else(|e| panic!("statement failed: {sql}\n{e}"));
}

async fn scalar_i64(
    db: &DatabaseConnection,
    sql: &str,
    values: impl IntoIterator<Item = sea_orm::Value>,
) -> i64 {
    db.query_one(Statement::from_sql_and_values(
        DbBackend::Postgres,
        sql,
        values,
    ))
    .await
    .expect("query")
    .expect("one row")
    .try_get::<i64>("", "value")
    .expect("bigint column named `value`")
}

fn provenance(model: &str) -> GenerationProvenance {
    GenerationProvenance {
        embedding: EmbeddingProvenance {
            provider: "test".into(),
            model: model.into(),
            dimension: EMBEDDING_DIM,
            normalization: Normalization::Unit,
        },
        chunking: ChunkingProvenance::current(1024),
    }
}

/// `count` chunks whose text carries `marker`, with distinct finite vectors.
///
/// The marker is what makes a mixed read visible: a result set holding two
/// markers is a result set spanning two generations.
fn synthetic_chunks(marker: &str, count: usize) -> (Vec<ChunkWithContext>, Vec<Vec<f32>>) {
    let chunks: Vec<ChunkWithContext> = (0..count)
        .map(|i| ChunkWithContext {
            content: format!("generation {marker} passage {i} discusses retrieval invariants"),
            context_prefix: None,
            parent_content: None,
            metadata: ChunkMetadata {
                position: u32::try_from(i).unwrap_or(0),
                ..Default::default()
            },
            content_hash: format!("{marker}-{i}"),
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let embeddings: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            let mut v = vec![0.001_f32; EMBEDDING_DIM];
            v[i % EMBEDDING_DIM] = 1.0;
            v
        })
        .collect();
    (chunks, embeddings)
}

macro_rules! fixture_or_skip {
    () => {
        match Fixture::setup().await {
            Some(f) => f,
            None => {
                eprintln!("skipped: TEST_DATABASE_URL is not set");
                return;
            }
        }
    };
}

// ============================================================================
// US-005: the publication model
// ============================================================================

/// Readers observe the old generation before the publication commit, and the
/// complete new one after it. Nothing observes a mixture.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_reader_sees_the_old_generation_until_publication_commits() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("publication visibility").await;
    let prov = provenance("model-a");

    f.publish_generation(source_id, "old", 3, &prov).await;
    assert_eq!(f.active_contents("generation").await.len(), 3);

    // Build the replacement beside the active one. Its rows exist in `chunks`
    // from this point on, and are invisible to every reader.
    let replacement = f
        .generations
        .claim(source_id, &prov)
        .await
        .expect("claim")
        .expect("no competing build");
    let (chunks, embeddings) = synthetic_chunks("new", 5);
    f.chunks
        .store_chunks(replacement, source_id, &chunks, &embeddings)
        .await
        .expect("store replacement chunks");
    f.generations
        .record_build_plan(replacement, 5, &prov.chunking)
        .await
        .expect("record build plan");

    let during = f.active_contents("generation").await;
    assert_eq!(
        during.len(),
        3,
        "a building generation must be invisible; saw {during:?}"
    );
    assert!(
        during.iter().all(|c| c.contains("old")),
        "reader saw replacement content before publication: {during:?}"
    );

    let _published = f
        .generations
        .publish(replacement, source_id, EMBEDDING_DIM)
        .await
        .expect("publish");

    let after = f.active_contents("generation").await;
    assert_eq!(
        after.len(),
        5,
        "publication must expose the whole replacement"
    );
    assert!(
        after.iter().all(|c| c.contains("new")),
        "publication left old-generation rows visible: {after:?}"
    );

    f.cleanup().await;
}

/// A failure at the publication boundary leaves the previous generation active
/// and the source's readiness untouched.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_failure_at_publication_preserves_the_active_generation() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("publication failure").await;
    let prov = provenance("model-a");
    let original = f.publish_generation(source_id, "old", 4, &prov).await;

    // A replacement that stores fewer chunks than it declared: the exact shape
    // an interrupted build leaves behind.
    let replacement = f
        .generations
        .claim(source_id, &prov)
        .await
        .expect("claim")
        .expect("no competing build");
    let (chunks, embeddings) = synthetic_chunks("new", 2);
    f.chunks
        .store_chunks(replacement, source_id, &chunks, &embeddings)
        .await
        .expect("store partial chunks");
    f.generations
        .record_build_plan(replacement, 6, &prov.chunking)
        .await
        .expect("record build plan");

    let err = f
        .generations
        .publish(replacement, source_id, EMBEDDING_DIM)
        .await
        .expect_err("an incomplete generation must not publish");
    assert!(
        err.to_string().contains("disagrees"),
        "the refusal must name the count mismatch: {err}"
    );

    let active = scalar_i64(
        &f.db,
        "SELECT count(*) AS value FROM sources WHERE id = $1 AND active_generation_id = $2",
        [source_id.into(), original.into()],
    )
    .await;
    assert_eq!(active, 1, "the previous generation must still be active");

    let visible = f.active_contents("generation").await;
    assert_eq!(visible.len(), 4);
    assert!(visible.iter().all(|c| c.contains("old")));

    f.cleanup().await;
}

// ============================================================================
// US-006: schema invariants
// ============================================================================

/// The database refuses the four states EP-002 declares impossible.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn the_schema_refuses_every_forbidden_state() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_a = f.create_source("invariants a").await;
    let source_b = f.create_source("invariants b").await;
    let generation_a = f.publish_generation(source_a, "a", 2, &prov).await;

    // 1. One building generation per source.
    let first = f
        .generations
        .claim(source_a, &prov)
        .await
        .expect("claim")
        .expect("first claim wins");
    assert!(
        f.generations
            .claim(source_a, &prov)
            .await
            .expect("claim must not error")
            .is_none(),
        "a second building generation must be refused, not queued"
    );

    // 2. No duplicate chunk position inside a generation.
    let (chunks, embeddings) = synthetic_chunks("dup", 1);
    f.chunks
        .store_chunks(first, source_a, &chunks, &embeddings)
        .await
        .expect("first write");
    f.chunks
        .store_chunks(first, source_a, &chunks, &embeddings)
        .await
        .expect("a retried write must be idempotent, not a duplicate");
    let rows = scalar_i64(
        &f.db,
        "SELECT count(*) AS value FROM chunks WHERE generation_id = $1",
        [first.into()],
    )
    .await;
    assert_eq!(rows, 1, "the retry created a duplicate chunk position");

    // 3. A source cannot publish another source's generation.
    let cross =
        f.db.execute(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "UPDATE sources SET active_generation_id = $1 WHERE id = $2",
            [generation_a.into(), source_b.into()],
        ))
        .await;
    assert!(
        cross.is_err(),
        "cross-source publication must violate a foreign key"
    );

    // 4. A referenced generation cannot be deleted.
    let referenced =
        f.db.execute(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "DELETE FROM source_index_generations WHERE id = $1",
            [generation_a.into()],
        ))
        .await;
    assert!(
        referenced.is_err(),
        "deleting the active generation must violate a foreign key"
    );

    f.cleanup().await;
}

/// Deleting a source still cascades cleanly through generations and chunks.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn deleting_a_source_reclaims_its_generations_and_chunks() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("cascade").await;
    let generation = f
        .publish_generation(source_id, "cascade", 3, &provenance("model-a"))
        .await;

    exec(
        &f.db,
        "DELETE FROM sources WHERE id = $1",
        [source_id.into()],
    )
    .await;

    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM source_index_generations WHERE id = $1",
            [generation.into()],
        )
        .await,
        0
    );
    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM chunks WHERE generation_id = $1",
            [generation.into()],
        )
        .await,
        0
    );

    f.cleanup().await;
}

// ============================================================================
// US-006: the migration itself
// ============================================================================

/// A scratch database created and dropped by one migration test.
///
/// Each test owns a fresh database rather than a schema, because the migration
/// operates on `public` table names and a shared database would let two tests
/// see each other's `chunks`.
struct ScratchDb {
    admin_url: String,
    name: String,
    url: String,
}

impl ScratchDb {
    /// `None` when `TEST_DATABASE_URL` is unset.
    async fn create(suffix: &str) -> Option<Self> {
        let base = std::env::var("TEST_DATABASE_URL").ok()?;
        let admin_url = base
            .rsplit_once('/')
            .map(|(head, _)| format!("{head}/postgres"))?;
        let name = format!("obl_migration_{suffix}");

        let admin = openbooklm::db::connect(&admin_url, &DatabasePoolConfig::default())
            .await
            .expect("connect to the maintenance database");
        exec(&admin, &format!(r#"DROP DATABASE IF EXISTS "{name}""#), []).await;
        exec(&admin, &format!(r#"CREATE DATABASE "{name}""#), []).await;

        let url = base
            .rsplit_once('/')
            .map(|(head, _)| format!("{head}/{name}"))?;
        Some(Self {
            admin_url,
            name,
            url,
        })
    }

    async fn connect(&self) -> DatabaseConnection {
        openbooklm::db::connect(&self.url, &DatabasePoolConfig::default())
            .await
            .expect("connect to the scratch database")
    }

    /// Apply the whole core track.
    async fn migrate(&self) -> Result<(), sea_orm::DbErr> {
        let db = self.connect().await;
        CoreMigrator::up(&db, None).await
    }

    /// Undo only the generation migration, leaving a pre-EP-002 database.
    async fn rewind_generations(&self) {
        let db = self.connect().await;
        for sql in [
            "ALTER TABLE sources DROP CONSTRAINT IF EXISTS sources_active_generation_fk",
            "ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_generation_fk",
            "DROP INDEX IF EXISTS chunks_generation_chunk_index_unique",
            "ALTER TABLE sources DROP COLUMN IF EXISTS active_generation_id",
            "ALTER TABLE chunks DROP COLUMN IF EXISTS generation_id",
            "DROP TABLE IF EXISTS source_index_generations",
            "DELETE FROM seaql_migrations_core WHERE version = 'm20260801_000001_index_generations'",
        ] {
            exec(&db, sql, []).await;
        }
    }

    /// One account, one notebook, one source: the minimum a chunk needs.
    async fn seed_source(&self, db: &DatabaseConnection) -> Uuid {
        let account_id = Uuid::new_v4();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();
        exec(
            db,
            "INSERT INTO accounts (id) VALUES ($1)",
            [account_id.into()],
        )
        .await;
        exec(
            db,
            "INSERT INTO notebooks (id, user_id, title) VALUES ($1, $2, 'legacy')",
            [notebook_id.into(), account_id.into()],
        )
        .await;
        exec(
            db,
            "INSERT INTO sources (id, notebook_id, title, source_type, content, status)
             VALUES ($1, $2, 'legacy source', 'text', 'legacy', 'ready')",
            [source_id.into(), notebook_id.into()],
        )
        .await;
        source_id
    }

    async fn drop(self) {
        let admin = openbooklm::db::connect(&self.admin_url, &DatabasePoolConfig::default())
            .await
            .expect("connect to the maintenance database");
        exec(
            &admin,
            &format!(r#"DROP DATABASE IF EXISTS "{}" WITH (FORCE)"#, self.name),
            [],
        )
        .await;
    }
}

macro_rules! scratch_or_skip {
    ($suffix:expr) => {
        match ScratchDb::create($suffix).await {
            Some(db) => db,
            None => {
                eprintln!("skipped: TEST_DATABASE_URL is not set");
                return;
            }
        }
    };
}

/// A fresh database reaches the generation schema, and applying the migration
/// twice changes nothing.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn the_migration_is_idempotent_on_a_fresh_database() {
    let scratch = scratch_or_skip!("fresh");
    scratch.migrate().await.expect("first apply");
    let db = scratch.connect().await;

    // The invariants the migration is responsible for installing.
    for (label, sql) in [
        (
            "generation table",
            "SELECT count(*)::bigint AS value FROM information_schema.tables
             WHERE table_name = 'source_index_generations'",
        ),
        (
            "chunk membership column",
            "SELECT count(*)::bigint AS value FROM information_schema.columns
             WHERE table_name = 'chunks' AND column_name = 'generation_id'
               AND is_nullable = 'NO'",
        ),
        (
            "source pointer column",
            "SELECT count(*)::bigint AS value FROM information_schema.columns
             WHERE table_name = 'sources' AND column_name = 'active_generation_id'",
        ),
        (
            "one-building index",
            "SELECT count(*)::bigint AS value FROM pg_indexes
             WHERE indexname = 'source_index_generations_one_building'",
        ),
        (
            "chunk position uniqueness",
            "SELECT count(*)::bigint AS value FROM pg_indexes
             WHERE indexname = 'chunks_generation_chunk_index_unique'",
        ),
        (
            "composite foreign keys",
            "SELECT count(*)::bigint AS value FROM pg_constraint
             WHERE conname IN ('chunks_generation_fk', 'sources_active_generation_fk')",
        ),
    ] {
        let found = scalar_i64(&db, sql, []).await;
        assert!(found >= 1, "{label} was not installed by the migration");
    }

    // Re-applying the whole track is a no-op: the history table already records
    // this version, and the DDL guards make a forced replay harmless too.
    scratch
        .migrate()
        .await
        .expect("second apply must be a no-op");
    scratch.rewind_generations().await;
    scratch
        .migrate()
        .await
        .expect("a forced replay must not fail on existing objects");

    scratch.drop().await;
}

/// An upgraded database keeps every chunk it had, in exactly one published
/// generation per source, with the source pointing at it.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn the_backfill_preserves_every_existing_chunk() {
    let scratch = scratch_or_skip!("backfill");
    scratch.migrate().await.expect("apply");
    scratch.rewind_generations().await;

    let db = scratch.connect().await;
    let indexed = scratch.seed_source(&db).await;
    // A second source with no chunks: an unindexed source must stay unindexed
    // rather than acquire an empty generation.
    let unindexed = {
        let row = db
            .query_one(Statement::from_sql_and_values(
                DbBackend::Postgres,
                "SELECT notebook_id FROM sources WHERE id = $1",
                [indexed.into()],
            ))
            .await
            .expect("query")
            .expect("one row");
        let notebook_id: Uuid = row.try_get("", "notebook_id").expect("notebook");
        let id = Uuid::new_v4();
        exec(
            &db,
            "INSERT INTO sources (id, notebook_id, title, source_type, content, status)
             VALUES ($1, $2, 'never indexed', 'text', 'x', 'pending')",
            [id.into(), notebook_id.into()],
        )
        .await;
        id
    };

    exec(
        &db,
        "INSERT INTO chunks (source_id, chunk_index, content, embedding, content_hash)
         SELECT $1, i, 'legacy chunk ' || i,
                array_fill(0.01::real, ARRAY[1024])::vector, 'hash' || i
         FROM generate_series(0, 6) AS i",
        [indexed.into()],
    )
    .await;

    scratch.migrate().await.expect("upgrade");

    assert_eq!(
        scalar_i64(
            &db,
            "SELECT count(*)::bigint AS value FROM chunks WHERE source_id = $1",
            [indexed.into()],
        )
        .await,
        7,
        "the backfill lost chunks"
    );
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT count(*)::bigint AS value FROM source_index_generations WHERE source_id = $1",
            [indexed.into()],
        )
        .await,
        1,
        "exactly one generation per source with chunks"
    );
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT count(*)::bigint AS value FROM chunks c
             JOIN sources s ON s.id = c.source_id AND s.active_generation_id = c.generation_id
             WHERE c.source_id = $1",
            [indexed.into()],
        )
        .await,
        7,
        "every backfilled chunk must be reachable through the active pointer"
    );
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT count(*)::bigint AS value FROM sources
             WHERE id = $1 AND active_generation_id IS NULL",
            [unindexed.into()],
        )
        .await,
        1,
        "a source with no chunks must not acquire a generation"
    );

    // Backfilled provenance is `legacy`, which never matches a live fingerprint:
    // the first reprocess rebuilds rather than reusing vectors of unknown origin.
    let fingerprint: String = db
        .query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT embedding_fingerprint AS value FROM source_index_generations
             WHERE source_id = $1",
            [indexed.into()],
        ))
        .await
        .expect("query")
        .expect("one row")
        .try_get("", "value")
        .expect("fingerprint");
    assert!(fingerprint.starts_with("legacy:"), "{fingerprint}");

    let reusable = SeaOrmChunkRepository::new(&db)
        .get_reusable_embeddings(indexed, &provenance("model-a").embedding.fingerprint())
        .await
        .expect("reuse lookup");
    assert!(
        reusable.is_empty(),
        "legacy vectors must not be reused under a live fingerprint"
    );

    scratch.drop().await;
}

/// A replay on a database that has already been reprocessed is still a no-op.
///
/// The regression this pins: after EP-002 two generations of one source
/// legitimately hold the same `(source_id, chunk_index)` pair — uniqueness is
/// `(generation_id, chunk_index)`, and keeping the previous generation for
/// rollback is the normal state. A backfill check that looked at the whole
/// `chunks` table read that as "duplicate chunk positions" and aborted on
/// healthy data. Scoping the checks to `generation_id IS NULL` is what makes the
/// migration idempotent on a *used* database, not merely a fresh one.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_replay_after_a_reprocess_is_still_a_no_op() {
    let scratch = scratch_or_skip!("replay_after_reprocess");
    scratch.migrate().await.expect("apply");

    let db = scratch.connect().await;
    let source_id = scratch.seed_source(&db).await;

    // Two published generations of one source, both holding positions 0..=3 —
    // the shape a single reprocess leaves behind.
    let prov = provenance("model-a");
    let generations = SeaOrmGenerationRepository::new(&db);
    let chunk_repo = SeaOrmChunkRepository::new(&db);
    for marker in ["first", "second"] {
        let generation = generations
            .claim(source_id, &prov)
            .await
            .expect("claim")
            .expect("claim wins");
        let (chunks, embeddings) = synthetic_chunks(marker, 4);
        chunk_repo
            .store_chunks(generation, source_id, &chunks, &embeddings)
            .await
            .expect("store");
        generations
            .record_build_plan(generation, 4, &prov.chunking)
            .await
            .expect("record build plan");
        let _published = generations
            .publish(generation, source_id, EMBEDDING_DIM)
            .await
            .expect("publish");
    }
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT count(*)::bigint AS value FROM chunks WHERE source_id = $1",
            [source_id.into()],
        )
        .await,
        8,
        "two generations of four chunks must coexist"
    );

    // Forget the migration ever ran, keeping the data exactly as it is. This is
    // the state a replay actually meets: schema and generations present, history
    // row gone.
    exec(
        &db,
        "DELETE FROM seaql_migrations_core WHERE version = 'm20260801_000001_index_generations'",
        [],
    )
    .await;

    scratch
        .migrate()
        .await
        .expect("a replay must not read two generations of one source as a defect");

    // Nothing moved: no generation was added, none was lost, and the active
    // pointer still names the generation that was published last.
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT count(*)::bigint AS value FROM source_index_generations WHERE source_id = $1",
            [source_id.into()],
        )
        .await,
        2,
        "the replay invented or removed a generation"
    );
    assert_eq!(
        scalar_i64(
            &db,
            "SELECT count(*)::bigint AS value FROM chunks c
             JOIN sources s ON s.id = c.source_id AND s.active_generation_id = c.generation_id
             WHERE c.source_id = $1 AND c.content LIKE '%second%'",
            [source_id.into()],
        )
        .await,
        4,
        "the active pointer must still reach the newest generation, and only it"
    );

    scratch.drop().await;
}

/// Legacy defects abort the migration with the source that caused them, rather
/// than publishing a generation whose contents cannot be proven.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn the_backfill_aborts_on_ambiguous_legacy_chunks() {
    for (suffix, defect_sql, expected) in [
        (
            "duplicates",
            "INSERT INTO chunks (source_id, chunk_index, content, embedding)
             VALUES ($1, 0, 'a', array_fill(0.01::real, ARRAY[1024])::vector),
                    ($1, 0, 'b', array_fill(0.01::real, ARRAY[1024])::vector)",
            "duplicate chunk positions",
        ),
        (
            "nullvectors",
            "INSERT INTO chunks (source_id, chunk_index, content, embedding)
             VALUES ($1, 0, 'a', NULL)",
            "missing embeddings",
        ),
    ] {
        let scratch = scratch_or_skip!(suffix);
        scratch.migrate().await.expect("apply");
        scratch.rewind_generations().await;

        let db = scratch.connect().await;
        let source_id = scratch.seed_source(&db).await;
        exec(&db, defect_sql, [source_id.into()]).await;

        let err = scratch
            .migrate()
            .await
            .expect_err("an ambiguous corpus must abort the migration");
        let message = err.to_string();
        assert!(message.contains(expected), "{message}");
        assert!(
            message.contains(&source_id.to_string()),
            "the diagnostic must name the source that caused it: {message}"
        );

        // Nothing was published: the abort left the database as it was.
        assert_eq!(
            scalar_i64(
                &db,
                "SELECT count(*)::bigint AS value FROM information_schema.tables
                 WHERE table_name = 'source_index_generations'",
                [],
            )
            .await,
            0,
            "the aborted migration must not leave a partial generation schema"
        );

        scratch.drop().await;
    }
}

// ============================================================================
// US-007: validation before publication
// ============================================================================

/// Publication validates counts, widths and finiteness, and refuses an empty
/// generation with a stable reason.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn publication_refuses_a_generation_it_cannot_prove_complete() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");

    // Zero chunks.
    let empty_source = f.create_source("empty").await;
    let empty = f
        .generations
        .claim(empty_source, &prov)
        .await
        .expect("claim")
        .expect("claim wins");
    f.generations
        .record_build_plan(empty, 0, &prov.chunking)
        .await
        .expect("record build plan");
    let err = f
        .generations
        .publish(empty, empty_source, EMBEDDING_DIM)
        .await
        .expect_err("an empty generation must never become active");
    assert!(
        err.to_string().contains("zero chunks"),
        "the reason must be stable and specific: {err}"
    );
    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM sources
             WHERE id = $1 AND active_generation_id IS NOT NULL",
            [empty_source.into()],
        )
        .await,
        0
    );

    // Wrong declared width: the same rows, validated against another dimension.
    let width_source = f.create_source("width").await;
    let width = f
        .generations
        .claim(width_source, &prov)
        .await
        .expect("claim")
        .expect("claim wins");
    let (chunks, embeddings) = synthetic_chunks("width", 2);
    f.chunks
        .store_chunks(width, width_source, &chunks, &embeddings)
        .await
        .expect("store chunks");
    f.generations
        .record_build_plan(width, 2, &prov.chunking)
        .await
        .expect("record build plan");
    let err = f
        .generations
        .publish(width, width_source, 768)
        .await
        .expect_err("a width disagreement must block publication");
    assert!(
        err.to_string().contains("wrong embedding width"),
        "the reason must name the width: {err}"
    );

    f.cleanup().await;
}

/// A failed build leaves the previous generation active and the source ready.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_failed_build_leaves_a_previously_indexed_source_searchable() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("failed build").await;
    let original = f.publish_generation(source_id, "old", 3, &prov).await;

    let replacement = f
        .generations
        .claim(source_id, &prov)
        .await
        .expect("claim")
        .expect("claim wins");
    f.generations
        .mark_failed(
            replacement,
            source_id,
            "embedding provider rejected the batch",
        )
        .await
        .expect("mark failed");

    let row =
        f.db.query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT status, active_generation_id FROM sources WHERE id = $1",
            [source_id.into()],
        ))
        .await
        .expect("query")
        .expect("one row");
    assert_eq!(
        row.try_get::<String>("", "status").expect("status"),
        "ready",
        "a source whose previous index is intact must not be reported as failed"
    );
    assert_eq!(
        row.try_get::<Uuid>("", "active_generation_id")
            .expect("pointer"),
        original
    );
    assert_eq!(f.active_contents("generation").await.len(), 3);

    // And the source can be claimed again.
    assert!(
        f.generations
            .claim(source_id, &prov)
            .await
            .expect("claim")
            .is_some(),
        "a failed generation must not block the next build"
    );

    f.cleanup().await;
}

/// A first build that fails reports the source as failed: there is no previous
/// index to fall back to, and claiming otherwise would be a lie.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_failed_first_build_reports_the_source_as_failed() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("first build failure").await;
    let generation = f
        .generations
        .claim(source_id, &provenance("model-a"))
        .await
        .expect("claim")
        .expect("claim wins");

    f.generations
        .mark_failed(generation, source_id, "extraction produced no text")
        .await
        .expect("mark failed");

    let row =
        f.db.query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT status, error_message FROM sources WHERE id = $1",
            [source_id.into()],
        ))
        .await
        .expect("query")
        .expect("one row");
    assert_eq!(
        row.try_get::<String>("", "status").expect("status"),
        "error"
    );
    assert_eq!(
        row.try_get::<String>("", "error_message").expect("message"),
        "extraction produced no text"
    );

    f.cleanup().await;
}

// ============================================================================
// US-008: publication, reads, rollback, reclaim
// ============================================================================

/// The Definition of Done: 1,000 schedules, zero mixed reads.
///
/// Each schedule publishes a replacement against two readers: one racing the
/// publication at a schedule-dependent offset, and one strictly after it
/// commits. The racing reader may legitimately observe either generation — both
/// answers are correct — so the assertion is never "the new one won". It is that
/// no result set ever spans both, and that the post-commit reader always sees
/// the whole replacement.
///
/// The offset is derived from the schedule index and from one calibration
/// measurement of how long a publication takes on this machine. A fixed offset
/// would be reproducible but useless: it lands on the same side of the commit
/// every time, and the racing reader would never actually straddle the window
/// it exists to test. Calibrating spreads the 1,000 schedules across it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_thousand_publication_schedules_produce_no_mixed_read() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("stress").await;
    f.publish_generation(source_id, "gen0", 4, &prov).await;

    // Calibration: how long one publication takes here, uncontended.
    let calibration = {
        let generation = f
            .generations
            .claim(source_id, &prov)
            .await
            .expect("claim")
            .expect("claim wins");
        let (chunks, embeddings) = synthetic_chunks("calibration", 4);
        f.chunks
            .store_chunks(generation, source_id, &chunks, &embeddings)
            .await
            .expect("store");
        f.generations
            .record_build_plan(generation, 4, &prov.chunking)
            .await
            .expect("record build plan");
        let start = std::time::Instant::now();
        let _published = f
            .generations
            .publish(generation, source_id, EMBEDDING_DIM)
            .await
            .expect("publish");
        start.elapsed()
    };
    // Sixteen offsets spanning twice the publication duration, so schedules
    // land before, inside and after the commit window.
    let offset_step = (calibration.as_micros() / 8).max(1) as u64;

    let mut mixed = 0usize;
    let mut observed_old = 0usize;
    let mut observed_new = 0usize;

    for schedule in 0..PUBLICATION_SCHEDULES {
        // Schedule 0 follows the calibration publication, so the marker it
        // replaces is the calibration one rather than `gen0`.
        let previous_marker = if schedule == 0 {
            "calibration".to_owned()
        } else {
            format!("gen{schedule}")
        };
        let next_marker = format!("gen{}", schedule + 1);

        let replacement = f
            .generations
            .claim(source_id, &prov)
            .await
            .expect("claim")
            .expect("claim wins");
        let (chunks, embeddings) = synthetic_chunks(&next_marker, 4);
        f.chunks
            .store_chunks(replacement, source_id, &chunks, &embeddings)
            .await
            .expect("store replacement");
        f.generations
            .record_build_plan(replacement, 4, &prov.chunking)
            .await
            .expect("record build plan");

        let generations = f.generations.clone();
        let publisher = tokio::spawn(async move {
            generations
                .publish(replacement, source_id, EMBEDDING_DIM)
                .await
        });

        // Walk the read across the publication window: schedules land before,
        // during and after the commit rather than all at the same point.
        let offset = Duration::from_micros((schedule % 16) as u64 * offset_step);
        let search = f.search.clone();
        let notebook_id = f.notebook_id;
        let reader = tokio::spawn(async move {
            if !offset.is_zero() {
                tokio::time::sleep(offset).await;
            }
            search
                .search_lexical_chunks(notebook_id, "generation", 100)
                .await
        });

        let _published = publisher.await.expect("publisher join").expect("publish");
        let racing = reader.await.expect("reader join").expect("read");

        assert_eq!(
            racing.len(),
            4,
            "schedule {schedule}: a read must return exactly one generation's chunks"
        );
        let has_old = racing.iter().any(|r| r.content.contains(&previous_marker));
        let has_new = racing.iter().any(|r| r.content.contains(&next_marker));
        match (has_old, has_new) {
            (true, true) => mixed += 1,
            (true, false) => observed_old += 1,
            (false, true) => observed_new += 1,
            (false, false) => panic!(
                "schedule {schedule}: read {} rows carrying neither generation's marker",
                racing.len()
            ),
        }

        // After the commit there is only one correct answer.
        let settled = f.active_contents("generation").await;
        assert_eq!(settled.len(), 4, "schedule {schedule}: post-commit read");
        assert!(
            settled.iter().all(|c| c.contains(&next_marker)),
            "schedule {schedule}: a committed publication left old rows visible"
        );
    }

    assert_eq!(
        mixed, 0,
        "{mixed} of {PUBLICATION_SCHEDULES} schedules observed a mixed generation"
    );
    println!(
        "calibrated publication: {}µs; {PUBLICATION_SCHEDULES} racing reads: \
         {observed_old} saw the previous generation, {observed_new} saw the replacement, \
         {mixed} mixed; {PUBLICATION_SCHEDULES} post-commit reads all saw the replacement",
        calibration.as_micros()
    );
    assert!(
        observed_old > 0 && observed_new > 0,
        "the racing reader never straddled the publication window \
         ({observed_old} before, {observed_new} after) — the schedule spread is not \
         exercising the race this test exists to check"
    );

    f.cleanup().await;
}

/// Rollback repoints a source at its previous complete generation without
/// copying anything.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn rollback_returns_to_the_previous_complete_generation() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("rollback").await;

    let first = f.publish_generation(source_id, "first", 3, &prov).await;
    let second = f.publish_generation(source_id, "second", 5, &prov).await;
    assert_eq!(f.active_contents("generation").await.len(), 5);

    let chunk_rows_before = scalar_i64(
        &f.db,
        "SELECT count(*) AS value FROM chunks WHERE source_id = $1",
        [source_id.into()],
    )
    .await;

    let target = f
        .generations
        .rollback_to_previous(source_id)
        .await
        .expect("rollback")
        .expect("a previous generation exists");
    assert_eq!(target, first);
    assert_ne!(target, second);

    let after = f.active_contents("generation").await;
    assert_eq!(after.len(), 3);
    assert!(after.iter().all(|c| c.contains("first")));
    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM chunks WHERE source_id = $1",
            [source_id.into()],
        )
        .await,
        chunk_rows_before,
        "rollback must move a pointer, not copy or delete chunks"
    );
    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT chunk_count::bigint AS value FROM sources WHERE id = $1",
            [source_id.into()],
        )
        .await,
        3,
        "the public chunk count must follow the pointer"
    );

    f.cleanup().await;
}

/// A source with only one generation has nothing to roll back to, and says so
/// rather than unpublishing it.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn rollback_without_a_predecessor_changes_nothing() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("no rollback target").await;
    let only = f
        .publish_generation(source_id, "only", 2, &provenance("model-a"))
        .await;

    assert!(
        f.generations
            .rollback_to_previous(source_id)
            .await
            .expect("rollback")
            .is_none()
    );
    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM sources WHERE id = $1 AND active_generation_id = $2",
            [source_id.into(), only.into()],
        )
        .await,
        1
    );
    assert_eq!(f.active_contents("generation").await.len(), 2);

    f.cleanup().await;
}

/// Reclaim removes obsolete generations and refuses to touch the active one or
/// the rollback target.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn reclaim_never_removes_a_referenced_or_rollback_eligible_generation() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("reclaim").await;

    let oldest = f.publish_generation(source_id, "oldest", 2, &prov).await;
    let previous = f.publish_generation(source_id, "previous", 2, &prov).await;
    let active = f.publish_generation(source_id, "active", 2, &prov).await;

    // Age every generation past the retention window so only the exclusions
    // decide what survives.
    exec(
        &f.db,
        "UPDATE source_index_generations SET created_at = now() - interval '48 hours'
         WHERE source_id = $1",
        [source_id.into()],
    )
    .await;

    let reclaimed = f.generations.reclaim(source_id, 24).await.expect("reclaim");
    assert_eq!(reclaimed, 1, "only the oldest generation was reclaimable");

    let surviving = f
        .generations
        .list_for_source(source_id)
        .await
        .expect("list");
    assert!(
        surviving.contains(&active),
        "the active generation was removed"
    );
    assert!(
        surviving.contains(&previous),
        "the rollback target was removed"
    );
    assert!(!surviving.contains(&oldest));

    // The rollback target still works after a reclaim pass.
    assert_eq!(
        f.generations
            .rollback_to_previous(source_id)
            .await
            .expect("rollback")
            .expect("target survives"),
        previous
    );

    f.cleanup().await;
}

/// A generation inside the retention window is never reclaimed, however
/// obsolete it is.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn reclaim_respects_the_retention_window() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("retention").await;
    f.publish_generation(source_id, "one", 2, &prov).await;
    f.publish_generation(source_id, "two", 2, &prov).await;
    f.publish_generation(source_id, "three", 2, &prov).await;

    assert_eq!(
        f.generations.reclaim(source_id, 24).await.expect("reclaim"),
        0,
        "generations younger than the window must be kept"
    );

    f.cleanup().await;
}

/// Every RAG read path is scoped to the active generation, not just search.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn every_read_path_is_scoped_to_the_active_generation() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("read scope").await;
    f.publish_generation(source_id, "active", 3, &prov).await;

    // A building generation with more chunks than the active one: if any read
    // path forgot the pointer, its count would be wrong rather than absent.
    let building = f
        .generations
        .claim(source_id, &prov)
        .await
        .expect("claim")
        .expect("claim wins");
    let (chunks, embeddings) = synthetic_chunks("building", 9);
    f.chunks
        .store_chunks(building, source_id, &chunks, &embeddings)
        .await
        .expect("store building chunks");

    assert_eq!(
        f.search
            .count_chunks_for_notebook(f.notebook_id)
            .await
            .expect("count"),
        3,
        "count must not see the building generation"
    );
    assert_eq!(
        f.search
            .get_all_chunks_for_notebook(f.notebook_id)
            .await
            .expect("stuffing load")
            .len(),
        3,
        "context stuffing must not see the building generation"
    );
    assert_eq!(
        f.chunks
            .get_for_source(source_id)
            .await
            .expect("chunk listing")
            .len(),
        3,
        "chunk listing must not see the building generation"
    );
    let sample = f
        .chunks
        .sample_chunks_for_notebook(f.notebook_id, 20)
        .await
        .expect("sample");
    assert_eq!(sample.len(), 3);
    assert!(sample.iter().all(|c| c.contains("active")));

    let dense = f
        .search
        .search_similar_chunks(f.notebook_id, &vec![0.001_f32; EMBEDDING_DIM], 50)
        .await
        .expect("dense search");
    assert_eq!(
        dense.len(),
        3,
        "dense search must not see the building generation"
    );

    f.cleanup().await;
}

// ============================================================================
// US-009: single-owner reprocessing
// ============================================================================

/// A hundred concurrent claims produce exactly one owner.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_hundred_concurrent_requests_produce_one_owner() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("ownership").await;

    let mut handles = Vec::with_capacity(OWNERSHIP_REQUESTS);
    for _ in 0..OWNERSHIP_REQUESTS {
        let generations = f.generations.clone();
        let prov = prov.clone();
        handles.push(tokio::spawn(async move {
            generations.claim(source_id, &prov).await
        }));
    }

    let mut owners = Vec::new();
    for handle in handles {
        let claimed = handle.await.expect("join").expect("claim must not error");
        if let Some(id) = claimed {
            owners.push(id);
        }
    }

    assert_eq!(
        owners.len(),
        1,
        "{} requests claimed ownership; exactly one may",
        owners.len()
    );
    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM source_index_generations
             WHERE source_id = $1 AND state = 'building'",
            [source_id.into()],
        )
        .await,
        1
    );

    // The single owner publishes; the result is one active pointer and unique
    // chunk positions.
    let owner = owners[0];
    let (chunks, embeddings) = synthetic_chunks("owned", 6);
    f.chunks
        .store_chunks(owner, source_id, &chunks, &embeddings)
        .await
        .expect("store");
    f.generations
        .record_build_plan(owner, 6, &prov.chunking)
        .await
        .expect("record build plan");
    let _published = f
        .generations
        .publish(owner, source_id, EMBEDDING_DIM)
        .await
        .expect("publish");

    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM (
                 SELECT generation_id, chunk_index FROM chunks WHERE source_id = $1
                 GROUP BY generation_id, chunk_index HAVING count(*) > 1
             ) duplicates",
            [source_id.into()],
        )
        .await,
        0,
        "duplicate chunk positions survived the concurrency test"
    );
    assert_eq!(f.active_contents("generation").await.len(), 6);

    f.cleanup().await;
}

/// A worker whose ownership has moved on cannot publish over the new owner.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_superseded_worker_cannot_publish() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("superseded worker").await;
    let active = f.publish_generation(source_id, "active", 3, &prov).await;

    // The first worker claims, then loses ownership to recovery.
    let stale = f
        .generations
        .claim(source_id, &prov)
        .await
        .expect("claim")
        .expect("claim wins");
    let (chunks, embeddings) = synthetic_chunks("stale", 2);
    f.chunks
        .store_chunks(stale, source_id, &chunks, &embeddings)
        .await
        .expect("store");
    f.generations
        .record_build_plan(stale, 2, &prov.chunking)
        .await
        .expect("record build plan");

    exec(
        &f.db,
        "UPDATE source_index_generations SET created_at = now() - interval '1 hour' WHERE id = $1",
        [stale.into()],
    )
    .await;
    let recovered = f
        .generations
        .fail_stale_builds(60, "interrupted")
        .await
        .expect("recovery");
    assert_eq!(recovered, 1);

    // The stale worker finishes and tries to publish.
    let err = f
        .generations
        .publish(stale, source_id, EMBEDDING_DIM)
        .await
        .expect_err("a failed generation must not publish");
    assert!(
        err.to_string().contains("no longer a building generation"),
        "{err}"
    );

    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM sources WHERE id = $1 AND active_generation_id = $2",
            [source_id.into(), active.into()],
        )
        .await,
        1
    );
    // And a new owner is free to claim.
    assert!(
        f.generations
            .claim(source_id, &prov)
            .await
            .expect("claim")
            .is_some()
    );

    f.cleanup().await;
}

/// A superseded worker cannot report an outcome over the owner that replaced it.
///
/// The failure path restores the source's status, and a naive restore would
/// overwrite the new owner's `processing` with `ready` — telling the user a
/// rebuild finished that had not started (US-009).
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_superseded_worker_cannot_rewrite_the_new_owners_status() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("status ownership").await;
    f.publish_generation(source_id, "active", 3, &prov).await;

    // Worker A claims, then loses ownership to recovery.
    let stale = f
        .generations
        .claim(source_id, &prov)
        .await
        .expect("claim")
        .expect("claim wins");
    exec(
        &f.db,
        "UPDATE source_index_generations SET created_at = now() - interval '1 hour' WHERE id = $1",
        [stale.into()],
    )
    .await;
    assert_eq!(
        f.generations
            .fail_stale_builds(60, "interrupted")
            .await
            .expect("recovery"),
        1
    );

    // Worker B takes over and reports that it is working.
    let fresh = f
        .generations
        .claim(source_id, &prov)
        .await
        .expect("claim")
        .expect("the reclaimed slot is free");
    assert_ne!(fresh, stale);
    exec(
        &f.db,
        "UPDATE sources SET status = 'embedding' WHERE id = $1",
        [source_id.into()],
    )
    .await;

    // Worker A finally notices its own failure and reports it.
    f.generations
        .mark_failed(stale, source_id, "provider timed out")
        .await
        .expect("mark failed");

    assert_eq!(
        f.db.query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT status FROM sources WHERE id = $1",
            [source_id.into()],
        ))
        .await
        .expect("query")
        .expect("one row")
        .try_get::<String>("", "status")
        .expect("status"),
        "embedding",
        "the superseded worker overwrote the current owner's status"
    );

    f.cleanup().await;
}

/// Recovery only reclaims builds past the deadline, never a live one.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn recovery_leaves_a_build_inside_its_deadline_alone() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("live build").await;
    let live = f
        .generations
        .claim(source_id, &provenance("model-a"))
        .await
        .expect("claim")
        .expect("claim wins");

    assert_eq!(
        f.generations
            .fail_stale_builds(1_200, "interrupted")
            .await
            .expect("recovery"),
        0,
        "a build inside its deadline must not be reclaimed"
    );
    assert!(
        f.generations
            .find_building(source_id)
            .await
            .expect("find")
            .is_some_and(|id| id == live)
    );

    f.cleanup().await;
}

// ============================================================================
// US-011: provenance-keyed reuse
// ============================================================================

/// Reuse is keyed on the embedding fingerprint, not the content hash alone.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_changed_embedding_fingerprint_reuses_nothing() {
    let f = fixture_or_skip!();
    let model_a = provenance("model-a");
    let model_b = provenance("model-b");
    let source_id = f.create_source("provenance").await;
    f.publish_generation(source_id, "a", 4, &model_a).await;

    let same_model = f
        .chunks
        .get_reusable_embeddings(source_id, &model_a.embedding.fingerprint())
        .await
        .expect("reuse lookup");
    assert_eq!(
        same_model.len(),
        4,
        "the same model must be able to reuse its own vectors"
    );

    let other_model = f
        .chunks
        .get_reusable_embeddings(source_id, &model_b.embedding.fingerprint())
        .await
        .expect("reuse lookup");
    assert!(
        other_model.is_empty(),
        "a different model reused {} vectors from another vector space",
        other_model.len()
    );

    f.cleanup().await;
}

/// The provenance a generation was claimed under is what it stores.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_generation_stores_the_provenance_it_was_built_under() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("stored provenance").await;
    let generation = f.publish_generation(source_id, "p", 2, &prov).await;

    let row =
        f.db.query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT embedding_fingerprint, chunking_fingerprint, embedding_provider,
                    embedding_model, embedding_dimension, embedding_normalization,
                    chunking_schema_version, chunk_size_unit, chunk_sizer_identity
             FROM source_index_generations WHERE id = $1",
            [generation.into()],
        ))
        .await
        .expect("query")
        .expect("one row");

    assert_eq!(
        row.try_get::<String>("", "embedding_fingerprint")
            .expect("fingerprint"),
        prov.embedding.fingerprint()
    );
    assert_eq!(
        row.try_get::<String>("", "chunking_fingerprint")
            .expect("fingerprint"),
        prov.chunking.fingerprint()
    );
    assert_eq!(
        row.try_get::<String>("", "embedding_provider")
            .expect("provider"),
        "test"
    );
    assert_eq!(
        row.try_get::<String>("", "embedding_model").expect("model"),
        "model-a"
    );
    assert_eq!(
        row.try_get::<i32>("", "embedding_dimension")
            .expect("dimension"),
        i32::try_from(EMBEDDING_DIM).expect("fits"),
    );
    assert_eq!(
        row.try_get::<String>("", "embedding_normalization")
            .expect("normalization"),
        "unit"
    );
    assert_eq!(
        row.try_get::<String>("", "chunk_size_unit").expect("unit"),
        openbooklm::services::rag::provenance::CHUNK_SIZE_UNIT
    );
    assert_eq!(
        row.try_get::<String>("", "chunk_sizer_identity")
            .expect("sizer"),
        openbooklm::services::rag::provenance::CHUNK_SIZER_IDENTITY
    );
    assert_eq!(
        row.try_get::<i32>("", "chunking_schema_version")
            .expect("schema version"),
        openbooklm::services::rag::provenance::CHUNKING_SCHEMA_VERSION
    );

    f.cleanup().await;
}

// ============================================================================
// US-010: cancellation through the real pipeline
// ============================================================================

/// Records every domain event a run emits, so "exactly one terminal event"
/// (US-010) is counted rather than asserted about code shape.
#[derive(Default)]
struct RecordingSink {
    events: std::sync::Mutex<Vec<&'static str>>,
}

impl RecordingSink {
    /// The terminal events: a run may emit at most one.
    fn terminal(&self) -> Vec<&'static str> {
        self.events
            .lock()
            .expect("sink mutex")
            .iter()
            .copied()
            .filter(|label| {
                matches!(
                    *label,
                    "source_processing_completed" | "source_processing_failed"
                )
            })
            .collect()
    }
}

impl openbooklm::core::events::EventSink for RecordingSink {
    fn emit(&self, event: openbooklm::core::events::DomainEvent) {
        self.events.lock().expect("sink mutex").push(event.label());
    }
}

/// An embedding provider that counts calls and can block until released.
///
/// The call counter is what makes "zero calls started after cancellation" a
/// measurement rather than an assertion about code shape.
struct CountingEmbedder {
    started: Arc<AtomicUsize>,
    block: Arc<AtomicBool>,
}

#[async_trait]
impl EmbeddingProvider for CountingEmbedder {
    fn name(&self) -> &str {
        "counting"
    }

    fn model(&self) -> &str {
        "counting-v1"
    }

    fn dimension(&self) -> usize {
        EMBEDDING_DIM
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        self.started.fetch_add(1, Ordering::SeqCst);
        while self.block.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok(texts
            .iter()
            .map(|_| vec![0.001_f32; EMBEDDING_DIM])
            .collect())
    }

    async fn embed_query(&self, _text: &str) -> Result<Vec<f32>, AppError> {
        Ok(vec![0.001_f32; EMBEDDING_DIM])
    }

    fn batch_size(&self) -> usize {
        1
    }

    fn concurrency(&self) -> usize {
        1
    }
}

/// A cancelled ingestion starts no further provider call and preserves the
/// active generation.
///
/// The deadline is the pipeline's own: `process_source` is given a short one,
/// so the run times out mid-embedding with the provider deliberately slow.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_timed_out_run_stops_calling_the_provider_and_preserves_the_index() {
    use openbooklm::core::entitlements::UnrestrictedPolicy;
    use openbooklm::core::principal::Principal;
    use openbooklm::services::source_processing::{
        ProcessingDeps, claim_index_ownership, process_source,
    };
    use openbooklm::types::SourceType;
    use tokio_util::sync::CancellationToken;

    let f = fixture_or_skip!();
    let prov = provenance("counting:counting-v1");
    let source_id = f.create_source("cancellation").await;
    let original = f.publish_generation(source_id, "old", 3, &prov).await;

    // Content large enough to produce many batches, so cancellation lands
    // between them rather than after the last one.
    let content = "Retrieval invariants and generation publication. ".repeat(4_000);
    exec(
        &f.db,
        "UPDATE sources SET content = $2 WHERE id = $1",
        [source_id.into(), content.into()],
    )
    .await;

    let started = Arc::new(AtomicUsize::new(0));
    let block = Arc::new(AtomicBool::new(true));
    let embedder = Arc::new(CountingEmbedder {
        started: Arc::clone(&started),
        block: Arc::clone(&block),
    }) as Arc<dyn EmbeddingProvider>;
    let sink = Arc::new(RecordingSink::default());

    let deps = ProcessingDeps {
        db: f.db.clone(),
        config: Arc::new(openbooklm::core::config::CoreConfig::from_env()),
        broadcaster: openbooklm::services::source_events::SourceEventBroadcaster::new(),
        source_repo: Arc::new(openbooklm::repositories::SeaOrmSourceRepository::new(&f.db)),
        chunk_repo: Arc::new(SeaOrmChunkRepository::new(&f.db)),
        generation_repo: Arc::new(SeaOrmGenerationRepository::new(&f.db)),
        embeddings: Some(embedder),
        firecrawl: None,
        youtube: None,
        ocr: None,
        ocr_cache: Arc::new(openbooklm::repositories::SeaOrmOcrCacheRepository::new(
            &f.db,
        )),
        entitlements: Arc::new(UnrestrictedPolicy),
        events: Arc::clone(&sink) as Arc<dyn openbooklm::core::events::EventSink>,
        principal: Principal::new(f.account_id),
        shutdown: CancellationToken::new(),
    };

    let ownership = claim_index_ownership(
        deps.generation_repo.as_ref(),
        deps.embeddings.as_ref(),
        source_id,
        SourceType::Text,
        Duration::from_secs(2),
    )
    .await
    .expect("claim must not error")
    .expect("a failed or absent build must not block the claim");

    let run = process_source(
        deps,
        ownership.clone(),
        source_id,
        f.notebook_id,
        SourceType::Text,
        // Short enough that the blocked provider guarantees a timeout.
        Duration::from_millis(400),
    );

    let run_started = std::time::Instant::now();
    let outcome = run.await;
    let run_elapsed = run_started.elapsed();
    assert!(
        outcome.is_err(),
        "a blocked provider must not report success"
    );

    // The deadline plus the drain window is the whole budget. Overrunning it
    // means the drain waited on work it promised to bound (US-010).
    assert!(
        run_elapsed
            < Duration::from_millis(400)
                + openbooklm::services::ingestion_tasks::DRAIN_DEADLINE
                + Duration::from_secs(2),
        "the run took {run_elapsed:?}, beyond its deadline plus the {:?} drain window",
        openbooklm::services::ingestion_tasks::DRAIN_DEADLINE
    );

    let terminal = sink.terminal();
    assert_eq!(
        terminal,
        vec!["source_processing_failed"],
        "a timed-out run must emit exactly one terminal event"
    );

    let at_cancellation = started.load(Ordering::SeqCst);
    assert!(
        at_cancellation >= 1,
        "the provider was never called, so \"no calls after cancellation\" would be vacuous"
    );
    // One second of grace: no *new* call may start after cancellation. The one
    // already in flight is allowed to be there — it is what the drain waits for.
    tokio::time::sleep(Duration::from_secs(1)).await;
    let after = started.load(Ordering::SeqCst);
    block.store(false, Ordering::SeqCst);

    assert_eq!(
        after,
        at_cancellation,
        "{} provider call(s) started more than a second after cancellation",
        after - at_cancellation
    );

    // The previous index is intact and the source still reports ready.
    let row =
        f.db.query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT status, active_generation_id FROM sources WHERE id = $1",
            [source_id.into()],
        ))
        .await
        .expect("query")
        .expect("one row");
    assert_eq!(
        row.try_get::<String>("", "status").expect("status"),
        "ready"
    );
    assert_eq!(
        row.try_get::<Uuid>("", "active_generation_id")
            .expect("pointer"),
        original
    );
    assert_eq!(f.active_contents("generation").await.len(), 3);

    // And the timed-out generation is failed, not left blocking the source.
    let state: String =
        f.db.query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT state FROM source_index_generations WHERE id = $1",
            [ownership.generation_id.into()],
        ))
        .await
        .expect("query")
        .expect("one row")
        .try_get("", "state")
        .expect("state");
    assert_eq!(state, "failed");

    f.cleanup().await;
}

/// Process shutdown reaches ingestion through the token, with the same outcome
/// as a timeout: the building generation fails, the active one survives, and
/// exactly one terminal event is emitted (US-010).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_shutdown_signal_stops_ingestion_and_preserves_the_index() {
    use openbooklm::core::entitlements::UnrestrictedPolicy;
    use openbooklm::core::principal::Principal;
    use openbooklm::services::source_processing::{
        ProcessingDeps, claim_index_ownership, process_source,
    };
    use openbooklm::types::SourceType;
    use tokio_util::sync::CancellationToken;

    let f = fixture_or_skip!();
    let prov = provenance("counting:counting-v1");
    let source_id = f.create_source("shutdown").await;
    let original = f.publish_generation(source_id, "old", 3, &prov).await;

    let content = "Retrieval invariants and generation publication. ".repeat(4_000);
    exec(
        &f.db,
        "UPDATE sources SET content = $2 WHERE id = $1",
        [source_id.into(), content.into()],
    )
    .await;

    let started = Arc::new(AtomicUsize::new(0));
    let block = Arc::new(AtomicBool::new(true));
    let embedder = Arc::new(CountingEmbedder {
        started: Arc::clone(&started),
        block: Arc::clone(&block),
    }) as Arc<dyn EmbeddingProvider>;
    let sink = Arc::new(RecordingSink::default());
    let shutdown = CancellationToken::new();

    let deps = ProcessingDeps {
        db: f.db.clone(),
        config: Arc::new(openbooklm::core::config::CoreConfig::from_env()),
        broadcaster: openbooklm::services::source_events::SourceEventBroadcaster::new(),
        source_repo: Arc::new(openbooklm::repositories::SeaOrmSourceRepository::new(&f.db)),
        chunk_repo: Arc::new(SeaOrmChunkRepository::new(&f.db)),
        generation_repo: Arc::new(SeaOrmGenerationRepository::new(&f.db)),
        embeddings: Some(embedder),
        firecrawl: None,
        youtube: None,
        ocr: None,
        ocr_cache: Arc::new(openbooklm::repositories::SeaOrmOcrCacheRepository::new(
            &f.db,
        )),
        entitlements: Arc::new(UnrestrictedPolicy),
        events: Arc::clone(&sink) as Arc<dyn openbooklm::core::events::EventSink>,
        principal: Principal::new(f.account_id),
        shutdown: shutdown.clone(),
    };

    let ownership = claim_index_ownership(
        deps.generation_repo.as_ref(),
        deps.embeddings.as_ref(),
        source_id,
        SourceType::Text,
        Duration::from_secs(60),
    )
    .await
    .expect("claim must not error")
    .expect("claim wins");

    // A generous deadline: what ends this run is the shutdown signal, not time.
    let run = tokio::spawn(process_source(
        deps,
        ownership.clone(),
        source_id,
        f.notebook_id,
        SourceType::Text,
        Duration::from_secs(60),
    ));

    // Let the first batch reach the provider, then shut down.
    while started.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let at_shutdown = started.load(Ordering::SeqCst);
    let signalled = std::time::Instant::now();
    shutdown.cancel();

    let outcome = run.await.expect("join");
    let drained_in = signalled.elapsed();
    block.store(false, Ordering::SeqCst);

    assert!(outcome.is_err(), "a shut-down run must not report success");
    assert!(
        drained_in < openbooklm::services::ingestion_tasks::DRAIN_DEADLINE + Duration::from_secs(2),
        "the run took {drained_in:?} to drain after the shutdown signal"
    );
    assert_eq!(
        started.load(Ordering::SeqCst),
        at_shutdown,
        "a provider call started after the shutdown signal"
    );
    assert_eq!(
        sink.terminal(),
        vec!["source_processing_failed"],
        "a shut-down run must emit exactly one terminal event"
    );

    let row =
        f.db.query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT s.status, s.active_generation_id, g.state
             FROM sources s JOIN source_index_generations g ON g.id = $2
             WHERE s.id = $1",
            [source_id.into(), ownership.generation_id.into()],
        ))
        .await
        .expect("query")
        .expect("one row");
    assert_eq!(
        row.try_get::<String>("", "status").expect("status"),
        "ready"
    );
    assert_eq!(
        row.try_get::<Uuid>("", "active_generation_id")
            .expect("pointer"),
        original
    );
    assert_eq!(row.try_get::<String>("", "state").expect("state"), "failed");
    assert_eq!(f.active_contents("generation").await.len(), 3);

    f.cleanup().await;
}
