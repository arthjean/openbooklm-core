#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented
)]

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

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement, TransactionTrait};
use uuid::Uuid;

use openbooklm::core::config::DatabasePoolConfig;
use openbooklm::core::providers::{EMBEDDING_DIM, EmbeddingProvider};
use openbooklm::error::AppError;
use openbooklm::repositories::{
    APPROVED_STRATEGY, ChunkRepository, GenerationRepository, NotebookScope, SeaOrmChunkRepository,
    SeaOrmGenerationRepository, SeaOrmSearchRepository, SeaOrmSourceRepository, SearchRepository,
    SourceRepository, VectorCapabilities,
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

    /// The owner scope every search in this fixture runs under (US-020).
    fn scope(&self) -> NotebookScope {
        NotebookScope::new(self.account_id, self.notebook_id)
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

    /// Publish one generation of `count` chunks carrying clustered vectors.
    ///
    /// Used by the reduced recall test: `synthetic_chunks` produces one-hot
    /// vectors, which are equidistant from everything and make a recall
    /// comparison vacuous. These cluster, the way real embeddings do.
    async fn seed_dense_source(&self, source_id: Uuid, count: usize, cluster: usize) -> Uuid {
        let provenance = provenance("dense-recall");
        let generation_id = self
            .generations
            .claim(source_id, &provenance)
            .await
            .expect("claim")
            .expect("no competing build");

        let chunks: Vec<ChunkWithContext> = (0..count)
            .map(|i| ChunkWithContext {
                content: format!("dense passage {i} in cluster {cluster}"),
                context_prefix: None,
                parent_content: None,
                metadata: ChunkMetadata {
                    position: u32::try_from(i).unwrap_or(0),
                    ..Default::default()
                },
                content_hash: format!("dense-{cluster}-{i}"),
            })
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..count).map(|i| dense_vector(i, cluster)).collect();

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
            .search_lexical_chunks(self.scope(), query, 500)
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

async fn optional_uuid(
    db: &DatabaseConnection,
    sql: &str,
    values: impl IntoIterator<Item = sea_orm::Value>,
) -> Option<Uuid> {
    db.query_one(Statement::from_sql_and_values(
        DbBackend::Postgres,
        sql,
        values,
    ))
    .await
    .expect("query")
    .map(|row| {
        row.try_get::<Uuid>("", "value")
            .expect("UUID column named `value`")
    })
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

fn embedding_fingerprint(model: &str) -> String {
    provenance(model).embedding.fingerprint()
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

/// A citation lease pins the source pointer through persistence and enqueue.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn an_active_generation_lease_blocks_pointer_publication() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("citation lease").await;
    let provenance = provenance("model-a");
    let old_generation = f.publish_generation(source_id, "old", 2, &provenance).await;
    let replacement = f
        .generations
        .claim(source_id, &provenance)
        .await
        .expect("claim")
        .expect("claim wins");
    let (chunks, embeddings) = synthetic_chunks("new", 2);
    f.chunks
        .store_chunks(replacement, source_id, &chunks, &embeddings)
        .await
        .expect("store replacement");
    f.generations
        .record_build_plan(replacement, 2, &provenance.chunking)
        .await
        .expect("record build plan");

    let sources = SeaOrmSourceRepository::new(&f.db);
    let lease = sources
        .lock_active_generation(source_id, old_generation)
        .await
        .expect("lock active generation")
        .expect("old generation is active");
    let generations = f.generations.clone();
    let mut publisher = tokio::spawn(async move {
        generations
            .publish(replacement, source_id, EMBEDDING_DIM)
            .await
    });
    tokio::task::yield_now().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut publisher)
            .await
            .is_err(),
        "source pointer moved while a citation lease was live"
    );

    lease.release().await.expect("release citation lease");
    let _published = publisher
        .await
        .expect("publisher join")
        .expect("publish after citation enqueue");
    assert!(
        sources
            .lock_active_generation(source_id, old_generation)
            .await
            .expect("check old generation")
            .is_none(),
        "superseded generation must no longer be leasable"
    );
    f.cleanup().await;
}

/// Publication is an immutability boundary, not merely a pointer update.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_published_generation_rejects_every_late_chunk_write() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("published immutability").await;
    let generation_id = f
        .publish_generation(source_id, "original", 2, &provenance("model-a"))
        .await;
    let before = f.active_contents("generation").await;
    let (chunks, embeddings) = synthetic_chunks("mutated", 2);

    let error = f
        .chunks
        .store_chunks(generation_id, source_id, &chunks, &embeddings)
        .await
        .expect_err("a published generation must reject a retry");

    assert!(
        error.to_string().contains("no longer building"),
        "the rejection must name the immutable state: {error}"
    );
    assert_eq!(f.active_contents("generation").await, before);
    f.cleanup().await;
}

/// Publication waits for a batch that was admitted while the generation was
/// building, then closes the gate before moving the active pointer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn publication_waits_for_an_admitted_chunk_batch() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("writer publication overlap").await;
    let provenance = provenance("model-a");
    let generation_id = f
        .generations
        .claim(source_id, &provenance)
        .await
        .expect("claim")
        .expect("claim wins");
    let (first_chunks, first_embeddings) = synthetic_chunks("overlap", 2);
    f.chunks
        .store_chunks(generation_id, source_id, &first_chunks, &first_embeddings)
        .await
        .expect("store initial chunks");
    f.generations
        .record_build_plan(generation_id, 3, &provenance.chunking)
        .await
        .expect("record build plan");

    let writer = f.db.begin().await.expect("writer transaction");
    let (last_chunk, last_embedding) = synthetic_chunks("overlap", 1);
    f.chunks
        .store_chunk_batch(
            generation_id,
            source_id,
            &last_chunk,
            &last_embedding,
            2,
            &writer,
        )
        .await
        .expect("admit final batch");

    let generations = f.generations.clone();
    let mut publisher = tokio::spawn(async move {
        generations
            .publish(generation_id, source_id, EMBEDDING_DIM)
            .await
    });
    tokio::task::yield_now().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut publisher)
            .await
            .is_err(),
        "publisher crossed validation while an admitted writer held the generation lock"
    );

    writer.commit().await.expect("commit final batch");
    let published = publisher
        .await
        .expect("publisher join")
        .expect("publish after writer");
    assert_eq!(published.chunk_count, 3);
    assert_eq!(f.active_contents("generation").await.len(), 3);
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
        "INSERT INTO ocr_cache (
             id, source_id, content_hash, model, ocr_text, pages_processed
         ) VALUES ($1, $2, $3, 'test-model', 'derived text', 1)",
        [
            Uuid::new_v4().into(),
            source_id.into(),
            format!("{:064x}", 1).into(),
        ],
    )
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
    assert_eq!(
        scalar_i64(
            &f.db,
            "SELECT count(*) AS value FROM ocr_cache WHERE source_id = $1",
            [source_id.into()],
        )
        .await,
        0
    );

    f.cleanup().await;
}

/// The compatibility trigger makes raw telemetry structurally impossible even
/// when an older binary still writes the legacy columns.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn rag_log_legacy_writes_are_redacted_before_storage() {
    let f = fixture_or_skip!();
    let log_id = Uuid::new_v4();
    exec(
        &f.db,
        "INSERT INTO rag_logs (
             id, notebook_id, user_id, query, reformulated_query, hyde_document
         ) VALUES ($1, $2, $3, 'raw question', 'raw reformulation', 'raw hyde')",
        [log_id.into(), f.notebook_id.into(), f.account_id.into()],
    )
    .await;

    let row =
        f.db.query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT query, reformulated_query, hyde_document, query_hash
             FROM rag_logs WHERE id = $1",
            [log_id.into()],
        ))
        .await
        .expect("read redacted log")
        .expect("log exists");
    assert_eq!(row.try_get::<String>("", "query").expect("query"), "");
    assert_eq!(
        row.try_get::<Option<String>>("", "reformulated_query")
            .expect("reformulated query"),
        None
    );
    assert_eq!(
        row.try_get::<Option<String>>("", "hyde_document")
            .expect("HyDE document"),
        None
    );
    assert_eq!(
        row.try_get::<String>("", "query_hash").expect("query hash"),
        ""
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
/// Each schedule publishes a replacement against two readers: one hybrid read
/// racing publication at a schedule-dependent offset, and one lexical read
/// strictly after it commits. The racing reader may legitimately observe
/// either generation. Dense and lexical branches must still share one snapshot,
/// and the post-commit reader must see the whole replacement.
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
    let mut hybrid_query = vec![0.0; EMBEDDING_DIM];
    hybrid_query[0] = 1.0;

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
        let scope = f.scope();
        let query_embedding = hybrid_query.clone();
        let fingerprint = prov.embedding.fingerprint();
        let reader = tokio::spawn(async move {
            if !offset.is_zero() {
                tokio::time::sleep(offset).await;
            }
            search
                .search_hybrid_chunks(scope, &query_embedding, &fingerprint, "generation", 100)
                .await
        });

        let _published = publisher.await.expect("publisher join").expect("publish");
        let racing = reader.await.expect("reader join").expect("read");

        assert_eq!(racing.dense.len(), 4, "schedule {schedule}: dense fill");
        assert_eq!(racing.lexical.len(), 4, "schedule {schedule}: lexical fill");
        let rows: Vec<_> = racing.dense.iter().chain(&racing.lexical).collect();
        let generations: HashSet<_> = rows.iter().map(|row| row.generation_id).collect();
        if generations.len() > 1 {
            mixed += 1;
        }
        let has_old = rows.iter().any(|r| r.content.contains(&previous_marker));
        let has_new = rows.iter().any(|r| r.content.contains(&next_marker));
        match (has_old, has_new) {
            (true, true) => {}
            (true, false) => observed_old += 1,
            (false, true) => observed_new += 1,
            (false, false) => panic!(
                "schedule {schedule}: read {} rows carrying neither generation's marker",
                rows.len()
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

#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn rollback_never_activates_an_incompatible_embedding_generation() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("rollback fingerprint").await;
    let old = provenance("model-a");
    let current = provenance("model-b");
    let old_generation = f.publish_generation(source_id, "old", 2, &old).await;
    let current_generation = f
        .publish_generation(source_id, "current", 2, &current)
        .await;

    let target = f
        .generations
        .rollback_to_previous(source_id)
        .await
        .expect("rollback query");

    assert_eq!(target, None);
    assert_eq!(
        optional_uuid(
            &f.db,
            "SELECT active_generation_id AS value FROM sources WHERE id = $1",
            [source_id.into()],
        )
        .await,
        Some(current_generation)
    );
    assert_ne!(old_generation, current_generation);

    f.cleanup().await;
}

/// Rollback must wait behind any other active-pointer move for the same source.
/// Publication takes this row lock through its final `UPDATE sources`, so this
/// is the serialization point that prevents a stale rollback from overwriting
/// a concurrently published generation.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn rollback_serializes_on_the_source_pointer() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("rollback-lock").await;
    let first = f.publish_generation(source_id, "first", 2, &prov).await;
    f.publish_generation(source_id, "second", 2, &prov).await;

    let blocker = f.db.begin().await.expect("begin blocker");
    blocker
        .query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT id FROM sources WHERE id = $1 FOR UPDATE",
            [source_id.into()],
        ))
        .await
        .expect("lock source");

    let generations = f.generations.clone();
    let mut rollback =
        tokio::spawn(async move { generations.rollback_to_previous(source_id).await });
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut rollback)
            .await
            .is_err(),
        "rollback moved the pointer while another pointer transaction held the source row"
    );

    blocker.commit().await.expect("release source lock");
    let target = rollback
        .await
        .expect("rollback task")
        .expect("rollback")
        .expect("previous generation");
    assert_eq!(target, first);

    f.cleanup().await;
}

/// Publication and rollback are both real pointer moves here. Across a
/// thousand injected schedules the final pointer and rollback target must
/// correspond to one of the only two legal serial orders.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn a_thousand_publication_rollback_schedules_are_linearizable() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("rollback-publication-stress").await;
    f.publish_generation(source_id, "oldest", 1, &prov).await;
    f.publish_generation(source_id, "active", 1, &prov).await;

    let mut rollback_first = 0usize;
    let mut publication_first = 0usize;
    for schedule in 0..PUBLICATION_SCHEDULES {
        let active_before = optional_uuid(
            &f.db,
            "SELECT active_generation_id AS value FROM sources WHERE id = $1",
            [source_id.into()],
        )
        .await
        .expect("source has an active generation");
        let newest_before = optional_uuid(
            &f.db,
            "SELECT id AS value FROM source_index_generations
             WHERE source_id = $1 AND state = 'published'
             ORDER BY published_at DESC, id DESC LIMIT 1",
            [source_id.into()],
        )
        .await
        .expect("source has a published generation");
        let previous_before = optional_uuid(
            &f.db,
            "SELECT id AS value FROM source_index_generations
             WHERE source_id = $1 AND state = 'published' AND id <> $2
             ORDER BY published_at DESC, id DESC LIMIT 1",
            [source_id.into(), active_before.into()],
        )
        .await
        .expect("source has a rollback target");

        let replacement = f
            .generations
            .claim(source_id, &prov)
            .await
            .expect("claim")
            .expect("claim wins");
        let marker = format!("rollback-race-{schedule}");
        let (chunks, embeddings) = synthetic_chunks(&marker, 1);
        f.chunks
            .store_chunks(replacement, source_id, &chunks, &embeddings)
            .await
            .expect("store replacement");
        f.generations
            .record_build_plan(replacement, 1, &prov.chunking)
            .await
            .expect("record build plan");

        let rollback_delay = Duration::from_millis(usize::from(schedule % 2 == 1) as u64);
        let publish_delay = Duration::from_millis(usize::from(schedule % 2 == 0) as u64);
        let rollback_repo = f.generations.clone();
        let rollback = tokio::spawn(async move {
            if !rollback_delay.is_zero() {
                tokio::time::sleep(rollback_delay).await;
            }
            rollback_repo.rollback_to_previous(source_id).await
        });
        let publish_repo = f.generations.clone();
        let publisher = tokio::spawn(async move {
            if !publish_delay.is_zero() {
                tokio::time::sleep(publish_delay).await;
            }
            publish_repo
                .publish(replacement, source_id, EMBEDDING_DIM)
                .await
        });

        let rollback_target = rollback
            .await
            .expect("rollback join")
            .expect("rollback")
            .expect("rollback target");
        let _published = publisher.await.expect("publisher join").expect("publish");
        let final_active = optional_uuid(
            &f.db,
            "SELECT active_generation_id AS value FROM sources WHERE id = $1",
            [source_id.into()],
        )
        .await
        .expect("source remains active");

        if final_active == replacement {
            rollback_first += 1;
            assert_eq!(
                rollback_target, previous_before,
                "schedule {schedule}: rollback-before-publication chose a stale target"
            );
        } else {
            publication_first += 1;
            assert_eq!(
                rollback_target, newest_before,
                "schedule {schedule}: rollback-after-publication did not choose the prior latest generation"
            );
            assert_eq!(
                final_active, rollback_target,
                "schedule {schedule}: final pointer disagrees with the serialized rollback"
            );
        }
    }

    assert!(rollback_first > 0, "no schedule serialized rollback first");
    assert!(
        publication_first > 0,
        "no schedule serialized publication first"
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

    // Age every generation past the retention window and force a publication
    // timestamp tie. Reclaim must use the same UUID tie-break as rollback or it
    // can preserve a different predecessor from the one rollback will select.
    exec(
        &f.db,
        "UPDATE source_index_generations
         SET created_at = now() - interval '48 hours',
             published_at = now() - interval '48 hours'
         WHERE source_id = $1",
        [source_id.into()],
    )
    .await;
    let rollback_target = std::cmp::max(oldest, previous);
    let obsolete = if rollback_target == oldest {
        previous
    } else {
        oldest
    };

    let reclaimed = f.generations.reclaim(source_id, 24).await.expect("reclaim");
    assert_eq!(
        reclaimed, 1,
        "only the non-target generation was reclaimable"
    );

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
        surviving.contains(&rollback_target),
        "the rollback target was removed"
    );
    assert!(!surviving.contains(&obsolete));

    // The rollback target still works after a reclaim pass.
    assert_eq!(
        f.generations
            .rollback_to_previous(source_id)
            .await
            .expect("rollback")
            .expect("target survives"),
        rollback_target
    );

    f.cleanup().await;
}

/// Reclaim holds the same source-row lock as publication and rollback from
/// candidate selection through deletion.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn reclaim_serializes_on_the_source_pointer() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("reclaim-lock").await;
    f.publish_generation(source_id, "oldest", 2, &prov).await;
    f.publish_generation(source_id, "previous", 2, &prov).await;
    f.publish_generation(source_id, "active", 2, &prov).await;
    exec(
        &f.db,
        "UPDATE source_index_generations SET created_at = now() - interval '48 hours'
         WHERE source_id = $1",
        [source_id.into()],
    )
    .await;

    let blocker = f.db.begin().await.expect("begin blocker");
    blocker
        .query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            "SELECT id FROM sources WHERE id = $1 FOR UPDATE",
            [source_id.into()],
        ))
        .await
        .expect("lock source");

    let generations = f.generations.clone();
    let mut reclaim = tokio::spawn(async move { generations.reclaim(source_id, 24).await });
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut reclaim)
            .await
            .is_err(),
        "reclaim selected candidates while an active-pointer transaction held the source row"
    );

    blocker.commit().await.expect("release source lock");
    assert_eq!(reclaim.await.expect("reclaim task").expect("reclaim"), 1);

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
            .count_chunks_for_notebook(f.scope())
            .await
            .expect("count"),
        3,
        "count must not see the building generation"
    );
    assert_eq!(
        f.search
            .get_all_chunks_for_notebook(f.scope())
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
        .search_similar_chunks(
            f.scope(),
            &vec![0.001_f32; EMBEDDING_DIM],
            &embedding_fingerprint("model-a"),
            50,
        )
        .await
        .expect("dense search");
    assert_eq!(
        dense.len(),
        3,
        "dense search must not see the building generation"
    );

    f.cleanup().await;
}

/// Every read path is scoped to the owner as well as to the notebook (US-020).
///
/// The handler checks access before retrieval, but an embedding call, a
/// reformulation call and a reranker call sit between that check and this SQL.
/// A notebook whose ownership changed in that window has to stop returning
/// content, and the only layer that can be true of is the query itself
/// (PRD edge case 8).
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn no_read_path_returns_a_notebook_to_another_account() {
    let f = fixture_or_skip!();
    let prov = provenance("model-a");
    let source_id = f.create_source("owner scope").await;
    f.publish_generation(source_id, "owned", 4, &prov).await;

    let stranger = NotebookScope::new(Uuid::new_v4(), f.notebook_id);
    let embedding = vec![0.001_f32; EMBEDDING_DIM];

    // The owner sees the source, so the assertions below are not vacuous.
    assert_eq!(
        f.search
            .count_chunks_for_notebook(f.scope())
            .await
            .expect("count"),
        4
    );

    assert_eq!(
        f.search
            .count_chunks_for_notebook(stranger)
            .await
            .expect("count"),
        0,
        "counting another account's notebook must return nothing"
    );
    assert!(
        f.search
            .get_all_chunks_for_notebook(stranger)
            .await
            .expect("stuffing load")
            .is_empty(),
        "context stuffing must not load another account's notebook"
    );
    assert!(
        f.search
            .search_similar_chunks(stranger, &embedding, &embedding_fingerprint("model-a"), 50,)
            .await
            .expect("dense search")
            .is_empty(),
        "dense search must not read another account's notebook"
    );
    assert!(
        f.search
            .search_lexical_chunks(stranger, "owned", 50)
            .await
            .expect("lexical search")
            .is_empty(),
        "lexical search must not read another account's notebook"
    );

    f.cleanup().await;
}

/// Dense and hybrid retrieval refuse an active generation produced in another
/// vector space, even when both providers expose the schema's 1024 dimensions.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn vector_backed_search_rejects_an_incompatible_active_generation() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("fingerprint scope").await;
    let model_a = provenance("model-a");
    f.publish_generation(source_id, "fingerprint", 3, &model_a)
        .await;
    let query = vec![0.001_f32; EMBEDDING_DIM];

    let compatible = f
        .search
        .search_similar_chunks(f.scope(), &query, &model_a.embedding.fingerprint(), 10)
        .await
        .expect("compatible dense search");
    assert_eq!(compatible.len(), 3);

    let incompatible_fingerprint = embedding_fingerprint("model-b");
    let dense = f
        .search
        .search_similar_chunks(f.scope(), &query, &incompatible_fingerprint, 10)
        .await
        .expect_err("incompatible dense search must fail");
    assert!(
        dense.to_string().contains("fingerprint mismatch"),
        "{dense}"
    );

    let hybrid = f
        .search
        .search_hybrid_chunks(
            f.scope(),
            &query,
            &incompatible_fingerprint,
            "fingerprint",
            10,
        )
        .await
        .expect_err("incompatible hybrid search must fail");
    assert!(
        hybrid.to_string().contains("fingerprint mismatch"),
        "{hybrid}"
    );

    // Lexical-only mode has no vector-space contract and remains available.
    assert_eq!(
        f.search
            .search_lexical_chunks(f.scope(), "fingerprint", 10)
            .await
            .expect("lexical search")
            .len(),
        3
    );

    f.cleanup().await;
}

/// The fingerprint check and dense query must observe one active-generation
/// snapshot. A concurrent publication may yield old compatible rows or the
/// structured mismatch, never successful missing evidence.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn dense_search_never_turns_a_fingerprint_race_into_missing_evidence() {
    let f = fixture_or_skip!();
    let source_id = f.create_source("dense fingerprint race").await;
    let model_a = provenance("model-a");
    let model_b = provenance("model-b");
    let generation_a = f
        .publish_generation(source_id, "model-a", 3, &model_a)
        .await;
    let generation_b = f
        .publish_generation(source_id, "model-b", 3, &model_b)
        .await;
    exec(
        &f.db,
        "UPDATE sources SET active_generation_id = $2, chunk_count = 3 WHERE id = $1",
        [source_id.into(), generation_a.into()],
    )
    .await;

    // The compatibility statement reads sources/generations, then the dense
    // statement reaches chunks. Blocking only chunks lets the first statement
    // establish its snapshot before the active pointer changes.
    let blocker = f.db.begin().await.expect("begin chunks blocker");
    blocker
        .execute_unprepared("LOCK TABLE chunks IN ACCESS EXCLUSIVE MODE")
        .await
        .expect("lock chunks");

    let query = vec![0.001_f32; EMBEDDING_DIM];
    let search = f.search.clone();
    let scope = f.scope();
    let expected_fingerprint = model_a.embedding.fingerprint();
    let reader = tokio::spawn(async move {
        search
            .search_similar_chunks(scope, &query, &expected_fingerprint, 10)
            .await
    });

    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let waiting = scalar_i64(
                &f.db,
                "SELECT count(*)::bigint AS value
                 FROM pg_locks
                 WHERE relation = to_regclass($1)
                   AND mode = 'AccessShareLock'
                   AND NOT granted",
                ["chunks".into()],
            )
            .await;
            if waiting > 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dense query never reached the chunks lock");

    exec(
        &f.db,
        "UPDATE sources SET active_generation_id = $2 WHERE id = $1",
        [source_id.into(), generation_b.into()],
    )
    .await;
    blocker.commit().await.expect("release chunks lock");

    let rows = reader
        .await
        .expect("reader task")
        .expect("repeatable-read search keeps its compatible snapshot");
    assert_eq!(
        rows.len(),
        3,
        "fingerprint race became successful missing evidence"
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
    fail_on_call: Option<usize>,
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
        let call = self.started.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_on_call == Some(call) {
            return Err(AppError::ProviderError(format!(
                "synthetic embedding failure on call {call}"
            )));
        }
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
        fail_on_call: None,
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

/// A provider failure stops the lazy batch stream at the first failed call.
///
/// With concurrency one, any third call necessarily started after the second
/// returned its terminal error. This makes the admission guarantee exact.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn an_embedding_error_stops_admitting_new_provider_calls() {
    use openbooklm::core::entitlements::UnrestrictedPolicy;
    use openbooklm::core::principal::Principal;
    use openbooklm::services::source_processing::{
        ProcessingDeps, claim_index_ownership, process_source,
    };
    use openbooklm::types::SourceType;
    use tokio_util::sync::CancellationToken;

    let f = fixture_or_skip!();
    let prov = provenance("counting-v1");
    let source_id = f.create_source("provider failure").await;
    let original = f.publish_generation(source_id, "old", 3, &prov).await;
    let content = "Retrieval invariants and generation publication. ".repeat(4_000);
    exec(
        &f.db,
        "UPDATE sources SET content = $2 WHERE id = $1",
        [source_id.into(), content.into()],
    )
    .await;

    let started = Arc::new(AtomicUsize::new(0));
    let embedder = Arc::new(CountingEmbedder {
        started: Arc::clone(&started),
        block: Arc::new(AtomicBool::new(false)),
        fail_on_call: Some(2),
    }) as Arc<dyn EmbeddingProvider>;
    let sink = Arc::new(RecordingSink::default());
    let deps = ProcessingDeps {
        db: f.db.clone(),
        config: Arc::new(openbooklm::core::config::CoreConfig::from_env()),
        broadcaster: openbooklm::services::source_events::SourceEventBroadcaster::new(),
        source_repo: Arc::new(SeaOrmSourceRepository::new(&f.db)),
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
        Duration::from_secs(10),
    )
    .await
    .expect("claim must not error")
    .expect("claim must succeed");

    let outcome = process_source(
        deps,
        ownership,
        source_id,
        f.notebook_id,
        SourceType::Text,
        Duration::from_secs(10),
    )
    .await;

    assert!(outcome.is_err());
    assert_eq!(started.load(Ordering::SeqCst), 2);
    assert_eq!(sink.terminal(), vec!["source_processing_failed"]);
    assert_eq!(
        optional_uuid(
            &f.db,
            "SELECT active_generation_id AS value FROM sources WHERE id = $1",
            [source_id.into()],
        )
        .await,
        Some(original)
    );

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
    use openbooklm::middleware::TaskTracker;
    use openbooklm::services::source_processing::{
        ProcessingDeps, claim_index_ownership, process_source,
    };
    use openbooklm::types::SourceType;

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
        fail_on_call: None,
    }) as Arc<dyn EmbeddingProvider>;
    let sink = Arc::new(RecordingSink::default());
    let task_tracker = TaskTracker::new();

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
        shutdown: task_tracker.cancellation_token(),
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
    let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel();
    let run_ownership = ownership.clone();
    let notebook_id = f.notebook_id;
    task_tracker
        .try_spawn("source-processing", async move {
            let outcome = process_source(
                deps,
                run_ownership,
                source_id,
                notebook_id,
                SourceType::Text,
                Duration::from_secs(60),
            )
            .await;
            let _ = outcome_tx.send(outcome);
        })
        .expect("source processing admission");

    // Let the first batch reach the provider, then shut down.
    while started.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let at_shutdown = started.load(Ordering::SeqCst);
    let signalled = std::time::Instant::now();
    task_tracker.begin_shutdown();

    task_tracker.wait().await;
    let outcome = outcome_rx.await.expect("owned task outcome");
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

// ============================================================================
// US-016 — the approved filtered-ANN strategy
// ============================================================================

/// Per-query scan settings must not survive the retrieval transaction.
///
/// `SET LOCAL` is transaction-scoped by definition, but the setting is applied
/// on a *pooled* connection, and a leak here would be invisible: the next
/// borrower of that connection would silently run someone else's scan mode, and
/// the only symptom would be a recall number that moves for no reason.
///
/// The probes run concurrently and each reports its backend PID. Sequential
/// probes would prove nothing: the pool hands the same connection back every
/// time, so a loop of twenty would interrogate one backend twenty times. The
/// test asserts it reached more than one, then asserts every one of them is
/// clean.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn per_query_scan_settings_do_not_leak_into_pooled_connections() {
    let Some(f) = Fixture::setup().await else {
        return;
    };

    let source_id = f.create_source("scan settings").await;
    let provenance = provenance("scan-settings");
    f.publish_generation(source_id, "leak", 3, &provenance)
        .await;

    // Retrievals that apply every setting the approved strategy needs, run
    // concurrently so several pooled connections carry one.
    let query = vec![0.1_f32; EMBEDDING_DIM];
    let fingerprint = provenance.embedding.fingerprint();
    let searches = (0..8).map(|_| {
        f.search
            .search_similar_chunks(f.scope(), &query, &fingerprint, 10)
    });
    for result in futures::future::join_all(searches).await {
        result.expect("dense search");
    }

    // `DatabasePoolConfig::default()` allows ten connections. Concurrent
    // probes force the pool to hand out more than one of them.
    let probes = (0..8).map(|_| {
        f.db.query_one(Statement::from_string(
            DbBackend::Postgres,
            // `missing_ok` matters: pgvector registers `hnsw.*` when its
            // library loads into a backend, so a pooled connection that has
            // never run a vector expression does not know the setting at all.
            // That is not a leak either, which is why NULL is accepted and any
            // other value is not.
            "SELECT pg_backend_pid() AS pid,
                    current_setting('hnsw.iterative_scan', true) AS iterative_scan,
                    current_setting('hnsw.ef_search', true) AS ef_search,
                    current_setting('hnsw.max_scan_tuples', true) AS max_scan_tuples",
        ))
    });

    let mut backends: HashSet<i32> = HashSet::new();
    for probe in futures::future::join_all(probes).await {
        let row = probe.expect("probe").expect("one row");
        backends.insert(row.try_get::<i32>("", "pid").expect("backend pid"));
        for (column, default) in [
            ("iterative_scan", "off"),
            ("ef_search", "40"),
            ("max_scan_tuples", "20000"),
        ] {
            let value: Option<String> = row.try_get("", column).expect("setting value");
            if let Some(value) = value {
                assert_eq!(
                    value, default,
                    "hnsw.{column} leaked out of the retrieval transaction"
                );
            }
        }
    }

    assert!(
        backends.len() > 1,
        "the probes all landed on one backend ({backends:?}), so nothing about \
         the pool was tested"
    );

    f.cleanup().await;
}

/// The reduced recall test CI runs (US-016 AC-4).
///
/// The full 100,000-row comparison is `tests/ann_benchmark.rs`, an explicit
/// performance test. This one seeds 3,000 chunks across two notebooks so that
/// the notebook under test holds 10% of them, and asserts that the production
/// dense query returns the same top-10 an exact scan does. It is small enough
/// for a CI job and still fails if the approved scan strategy is dropped: with
/// a plain HNSW scan and post-filtering the fill rate collapses long before the
/// ordering does.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn filtered_dense_search_matches_exact_search_on_a_reduced_corpus() {
    let Some(f) = Fixture::setup().await else {
        return;
    };

    // 300 rows in the notebook under test, 2,700 in a second notebook that
    // shares the index. Filter selectivity is 10% of the seeded rows.
    let target = f.create_source("target").await;
    let target_generation = f.seed_dense_source(target, 300, 0).await;
    let other_notebook = Uuid::new_v4();
    exec(
        &f.db,
        "INSERT INTO notebooks (id, user_id, title) VALUES ($1, $2, 'ann noise')",
        [other_notebook.into(), f.account_id.into()],
    )
    .await;
    let noise_source = Uuid::new_v4();
    exec(
        &f.db,
        "INSERT INTO sources (id, notebook_id, title, source_type, content, status)
         VALUES ($1, $2, 'noise', 'text', 'noise', 'pending')",
        [noise_source.into(), other_notebook.into()],
    )
    .await;
    f.seed_dense_source(noise_source, 2_700, 1).await;

    let queries: Vec<Vec<f32>> = (0..10).map(|i| dense_vector(i * 7, 0)).collect();

    let mut matched = 0usize;
    let mut returned = 0usize;
    for query in &queries {
        let approximate = f
            .search
            .search_similar_chunks(f.scope(), query, &embedding_fingerprint("dense-recall"), 10)
            .await
            .expect("dense search");
        let exact = exact_top_ids(&f.db, f.notebook_id, query, 10).await;

        returned += approximate.len();
        matched += approximate.iter().filter(|r| exact.contains(&r.id)).count();

        assert!(
            approximate
                .iter()
                .all(|r| r.generation_id == target_generation),
            "search must stay inside the active generation"
        );

        // strict_order is the reason this strategy was chosen over
        // relaxed_order: the API and the fusion layer both consume dense
        // results in distance order (US-016 AC-3).
        for pair in approximate.windows(2) {
            assert!(
                pair[0].relevance_score >= pair[1].relevance_score,
                "dense results came back out of distance order: {} then {}",
                pair[0].relevance_score,
                pair[1].relevance_score
            );
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let recall = matched as f64 / (queries.len() * 10) as f64;
    #[allow(clippy::cast_precision_loss)]
    let fill = returned as f64 / (queries.len() * 10) as f64;
    assert!(
        recall >= 0.95,
        "Recall@10 against exact search was {recall:.3}, below the approved 0.95"
    );
    assert!(
        fill >= 0.99,
        "top-k fill was {fill:.3}: the filtered scan came back short"
    );

    exec(
        &f.db,
        "DELETE FROM notebooks WHERE id = $1",
        [other_notebook.into()],
    )
    .await;
    f.cleanup().await;
}

/// The pgvector build must offer the strategy the repository applies.
#[tokio::test]
#[ignore = "Requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn the_deployed_pgvector_supports_the_approved_strategy() {
    let Some(f) = Fixture::setup().await else {
        return;
    };

    let capabilities = VectorCapabilities::probe(&f.db).await.expect("probe");
    assert!(
        capabilities.iterative_scan,
        "pgvector {} cannot run the approved filtered scan",
        capabilities.extension_version
    );
    capabilities
        .ensure_supports(APPROVED_STRATEGY)
        .expect("the reference image satisfies the documented minimum");

    f.cleanup().await;
}

/// A deterministic vector that clusters with `cluster` and varies with `index`.
fn dense_vector(index: usize, cluster: usize) -> Vec<f32> {
    (0..EMBEDDING_DIM)
        .map(|d| {
            #[allow(clippy::cast_precision_loss)]
            let base = ((cluster * 13 + d) as f32 * 0.017).sin();
            #[allow(clippy::cast_precision_loss)]
            let jitter = ((index * 31 + d) as f32 * 0.0007).cos() * 0.05;
            base + jitter
        })
        .collect()
}

/// Exact top-`limit` chunk ids for a notebook, bypassing the index.
///
/// `+ 0` on the distance is what keeps the planner off the HNSW index without
/// touching a planner GUC: a session setting applied on a pooled connection is
/// exactly the leak the test above forbids.
async fn exact_top_ids(
    db: &DatabaseConnection,
    notebook_id: Uuid,
    query: &[f32],
    limit: usize,
) -> Vec<Uuid> {
    let vector = format!(
        "[{}]",
        query
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let rows = db
        .query_all(Statement::from_sql_and_values(
            DbBackend::Postgres,
            format!(
                "SELECT c.id
                 FROM chunks c
                 JOIN sources s ON c.source_id = s.id AND s.active_generation_id = c.generation_id
                 WHERE s.notebook_id = $1
                 ORDER BY (c.embedding <=> $2::vector) + 0
                 LIMIT {limit}"
            ),
            [notebook_id.into(), vector.into()],
        ))
        .await
        .expect("exact search");
    rows.into_iter()
        .map(|row| row.try_get::<Uuid>("", "id").expect("id"))
        .collect()
}
