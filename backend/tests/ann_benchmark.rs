#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented
)]

//! Filtered HNSW versus exact search (US-015).
//!
//! The measurement US-016 is not allowed to skip. It seeds 100,000 synthetic
//! chunks, then compares exact vector search against every approximate strategy
//! the deployed pgvector build offers, at four filter selectivities, and writes
//! `contracts/baseline/ann/filtered-hnsw.json`.
//!
//! ```bash
//! TEST_DATABASE_URL=postgres://openbooklm:openbooklm@localhost:5432/openbooklm \
//!   cargo test --no-default-features --test ann_benchmark -- --ignored --nocapture
//! ```
//!
//! # Why a fixture table and not `chunks`
//!
//! The benchmark bulk-loads 100,000 vectors and then builds the index, which is
//! an order of magnitude faster than maintaining an HNSW index across 100,000
//! inserts. Doing that to the product table would mean dropping and recreating
//! a production index inside a test. `ann_bench_sources` / `ann_bench_chunks`
//! reproduce the exact shape the production query joins over, with the same
//! HNSW parameters as the baseline migration, and are dropped at the end.
//!
//! # Why the 100% case has no predicate
//!
//! Selectivity is the fraction of *indexed* rows that survive the filter. Four
//! notebooks inside one corpus cannot each hold 100%, so the 100% case runs the
//! same query with a predicate that excludes nothing. Every other case names a
//! notebook holding exactly its fraction, so the four measured selectivities
//! are 100%, 10%, 1% and 0.1% rather than approximations of them.

#![cfg(test)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement, TransactionTrait};
use serde::Serialize;
use uuid::Uuid;

use openbooklm::core::config::DatabasePoolConfig;
use openbooklm::repositories::ann::{IterativeMode, ScanStrategy, VectorCapabilities};

// ============================================================================
// Benchmark parameters
// ============================================================================

/// Rows seeded. The PRD's floor, not a sample of it.
const CORPUS_SIZE: usize = 100_000;

/// Embedding width, pinned by the core schema.
const DIMENSION: usize = 1_024;

/// Distinct clusters the synthetic vectors are drawn from.
///
/// Uniform noise in 1,024 dimensions makes every point nearly equidistant from
/// every other, which would make any recall number meaningless. Clusters give
/// the index something to approximate.
const CLUSTERS: usize = 256;

/// Queries per (selectivity, strategy) cell.
const QUERIES: usize = 30;

/// Untimed queries run before each timed cell.
///
/// A full pass, not a token three. The exact reference reads the whole table
/// and evicts the HNSW index from the page cache, so the next strategy measured
/// pays for reloading it: at 100% selectivity that showed up as a plain HNSW
/// scan reporting 84 ms where the identical plan, touching the identical 42
/// tuples, ran in 2.4 ms once warm. A benchmark whose first strategy per cell
/// measures the previous strategy's eviction is measuring cache state, not
/// strategies.
const WARMUP_QUERIES: usize = QUERIES;

/// Results each query asks for.
const TOP_K: usize = 10;

/// Thresholds a strategy must meet to be recommended (US-015 AC-4).
const MIN_RECALL: f64 = 0.95;
const MIN_FILL: f64 = 0.99;
const MAX_P95_MS: f64 = 300.0;

/// How far out of distance order two adjacent rows may be before it counts.
///
/// pgvector computes distances in single precision inside the index and the
/// `SELECT` list recomputes them, so two rows that the index considers tied can
/// come back with recomputed distances differing in the last bit. Measured
/// across every strategy here, those disagreements sit around 1e-8, while a
/// mode that genuinely reorders rows produces gaps orders of magnitude larger.
/// The tolerance separates the two instead of reporting float noise as an
/// ordering failure; `max_order_violation` publishes the largest gap seen, so
/// the choice of tolerance stays auditable.
const ORDER_TOLERANCE: f64 = 1e-6;

/// The reference environment this recommendation is valid for.
const REFERENCE_ENVIRONMENT: &str = "pgvector/pgvector:pg16 container, HNSW m=16 ef_construction=128, \
     1024-dimensional vectors, 100000 rows, local loopback connection";

// ============================================================================
// Report
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct Measurement {
    strategy: String,
    /// Mean Recall@10 against exact search on the same query set.
    recall_at_10: f64,
    /// Mean returned/requested, capped at one.
    fill_rate: f64,
    p50_ms: f64,
    p95_ms: f64,
    /// Rows the plan actually touched, from `EXPLAIN ANALYZE` on one query.
    tuples_scanned: i64,
    /// Queries whose rows came back out of ascending distance order by more
    /// than [`ORDER_TOLERANCE`].
    ///
    /// Measured rather than assumed from the strategy's declared mode: the
    /// fusion layer consumes dense *rank*, so a strategy that reorders rows
    /// changes what fusion computes, and the report has to be able to show it
    /// (US-016 AC-3).
    out_of_order_queries: usize,
    /// The largest backward step in distance seen, tolerance included. What
    /// makes the tolerance auditable rather than a magic constant.
    max_order_violation: f64,
    /// The plan, reduced to what a reader compares between strategies.
    plan: PlanSummary,
    /// The `SET LOCAL` statements this strategy ran with.
    settings: Vec<String>,
    /// Whether this strategy meets every threshold at this selectivity.
    meets_thresholds: bool,
    /// The thresholds it failed, if any.
    blocking: Vec<String>,
}

/// One query plan, reduced to what a decision needs.
///
/// The full `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` output is ~15 KB per
/// measurement, and its per-node timings differ on every run, so a committed
/// report full of them cannot be diffed against the next one: the whole file
/// changes whether or not anything did. What a reader compares is which access
/// path ran, how many rows it produced and how many the filter threw away.
#[derive(Debug, Clone, Serialize)]
struct PlanSummary {
    /// `Limit → Index Scan using ann_bench_chunks_embedding`, and so on.
    shape: String,
    /// Whether the vector index was used at all.
    uses_vector_index: bool,
    /// Rows the top node produced.
    actual_rows: i64,
    /// Rows discarded by the notebook filter across the plan.
    rows_removed_by_filter: i64,
    /// Shared buffers hit and read, summed over the plan.
    shared_blocks_hit: i64,
    shared_blocks_read: i64,
}

#[derive(Debug, Clone, Serialize)]
struct SelectivityCase {
    label: String,
    /// Rows passing the filter over rows in the table.
    measured_selectivity: f64,
    matching_rows: i64,
    measurements: Vec<Measurement>,
}

#[derive(Debug, Clone, Serialize)]
struct Recommendation {
    /// The strategy to implement, or `null` when none qualifies.
    strategy: Option<String>,
    settings: Vec<String>,
    /// Why this one and not another.
    rationale: String,
    /// Dimensions that blocked every rejected strategy.
    blocking: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkReport {
    schema: &'static str,
    postgres_version: String,
    pgvector_version: String,
    iterative_scan_supported: bool,
    reference_environment: &'static str,
    corpus_size: usize,
    dimension: usize,
    clusters: usize,
    queries_per_cell: usize,
    /// Untimed queries run before each cell. Recorded because a report whose
    /// latencies were taken cold is a different measurement.
    warmup_queries_per_cell: usize,
    top_k: usize,
    thresholds: Thresholds,
    cases: Vec<SelectivityCase>,
    recommendation: Recommendation,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct Thresholds {
    min_recall_at_10: f64,
    min_fill_rate: f64,
    max_p95_ms: f64,
    /// Backward step in distance below which an inversion is float noise
    /// rather than a reordering. See [`ORDER_TOLERANCE`].
    order_tolerance: f64,
}

// ============================================================================
// Fixture
// ============================================================================

struct Bench {
    db: DatabaseConnection,
    /// Notebook per case label, `None` for the unfiltered 100% case.
    notebooks: Vec<(&'static str, Option<Uuid>, i64)>,
}

async fn exec(db: &DatabaseConnection, sql: &str) {
    db.execute_unprepared(sql)
        .await
        .unwrap_or_else(|e| panic!("statement failed: {sql}\n{e}"));
}

async fn scalar_i64(db: &DatabaseConnection, sql: &str) -> i64 {
    db.query_one(Statement::from_string(DbBackend::Postgres, sql))
        .await
        .expect("query")
        .expect("one row")
        .try_get::<i64>("", "value")
        .expect("i64 column named value")
}

impl Bench {
    /// Connect, create the fixture tables, seed them and build the index.
    ///
    /// Returns `None` when `TEST_DATABASE_URL` is unset, so an accidental run
    /// without a database skips rather than fails misleadingly. Re-running
    /// against an already-seeded database reuses the corpus: the seed is
    /// deterministic, and rebuilding it costs minutes.
    async fn setup() -> Option<Self> {
        let url = std::env::var("TEST_DATABASE_URL").ok()?;
        let db = openbooklm::db::connect(&url, &DatabasePoolConfig::default())
            .await
            .expect("connect to the test database");

        exec(&db, "CREATE EXTENSION IF NOT EXISTS vector").await;

        let seeded = scalar_i64(
            &db,
            "SELECT COALESCE((SELECT count(*) FROM information_schema.tables \
             WHERE table_name = 'ann_bench_chunks'), 0)::bigint AS value",
        )
        .await;
        let existing = if seeded > 0 {
            scalar_i64(
                &db,
                "SELECT count(*)::bigint AS value FROM ann_bench_chunks",
            )
            .await
        } else {
            0
        };

        if existing != i64::try_from(CORPUS_SIZE).expect("corpus size fits i64") {
            Self::seed(&db).await;
        } else {
            eprintln!("ann_benchmark: reusing the seeded corpus ({existing} rows)");
        }

        let notebooks = Self::case_notebooks(&db).await;
        Some(Self { db, notebooks })
    }

    /// Build and fill the fixture tables.
    ///
    /// Everything runs inside **one transaction**, and that is not a tidiness
    /// preference. `setseed` and `maintenance_work_mem` are per-session state,
    /// and the pool hands out up to ten connections: issued outside a
    /// transaction, the seed would apply to whichever backend happened to serve
    /// that statement while the inserts ran on others, so the corpus would not
    /// be reproducible and the memory setting would not reach the index build.
    /// It is the same reason [`VectorCapabilities::probe`] pins a connection,
    /// and the same reason the production query scopes its settings with `SET
    /// LOCAL` rather than `SET`.
    async fn seed(db: &DatabaseConnection) {
        eprintln!("ann_benchmark: seeding {CORPUS_SIZE} vectors, this takes minutes");
        let txn = db.begin().await.expect("begin the seeding transaction");

        txn.execute_unprepared(
            "DROP TABLE IF EXISTS ann_bench_chunks;
             DROP TABLE IF EXISTS ann_bench_sources;",
        )
        .await
        .expect("drop previous fixture tables");

        // The production shape: chunks reach their notebook through a source,
        // and the join carries the active-generation equality (EP-002).
        txn.execute_unprepared(&format!(
            "CREATE TABLE ann_bench_sources (
                 id UUID PRIMARY KEY,
                 notebook_id UUID NOT NULL,
                 active_generation_id UUID NOT NULL,
                 label TEXT NOT NULL
             );
             CREATE TABLE ann_bench_chunks (
                 id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                 source_id UUID NOT NULL REFERENCES ann_bench_sources(id),
                 generation_id UUID NOT NULL,
                 chunk_index INTEGER NOT NULL,
                 embedding vector({DIMENSION})
             );"
        ))
        .await
        .expect("create fixture tables");

        // 0.1%, 1%, 10% and the remainder. The fractions are of CORPUS_SIZE, so
        // each case notebook holds exactly its selectivity.
        let sizes: [(&str, usize); 4] = [
            ("nb_0_1", CORPUS_SIZE / 1000),
            ("nb_1", CORPUS_SIZE / 100),
            ("nb_10", CORPUS_SIZE / 10),
            (
                "nb_rest",
                CORPUS_SIZE - CORPUS_SIZE / 1000 - CORPUS_SIZE / 100 - CORPUS_SIZE / 10,
            ),
        ];

        // `max_parallel_maintenance_workers = 0` is not a performance choice.
        // A parallel HNSW build partitions the work across workers, so the
        // graph it produces depends on how many of them the server started,
        // and a fixture whose index differs between runs cannot support a
        // recall comparison across revisions. It also keeps the build off
        // dynamic shared memory, which the reference container sizes at 64 MB:
        // with `maintenance_work_mem` actually reaching the backend, a parallel
        // build asks for half a gigabyte of it and fails.
        txn.execute_unprepared(
            "SELECT setseed(0.42);
             SET LOCAL maintenance_work_mem = '512MB';
             SET LOCAL max_parallel_maintenance_workers = 0;",
        )
        .await
        .expect("pin the random seed and the build parameters for this transaction");

        let mut offset = 0usize;
        for (label, size) in sizes {
            let source_id = Uuid::new_v4();
            let notebook_id = Uuid::new_v4();
            let generation_id = Uuid::new_v4();
            txn.execute(Statement::from_sql_and_values(
                DbBackend::Postgres,
                "INSERT INTO ann_bench_sources (id, notebook_id, active_generation_id, label)
                 VALUES ($1, $2, $3, $4)",
                [
                    source_id.into(),
                    notebook_id.into(),
                    generation_id.into(),
                    label.into(),
                ],
            ))
            .await
            .expect("insert source");

            // Clustered vectors: a deterministic centroid per cluster plus
            // noise, so the index has structure to approximate.
            txn.execute_unprepared(&format!(
                "INSERT INTO ann_bench_chunks (source_id, generation_id, chunk_index, embedding)
                 SELECT '{source_id}', '{generation_id}', i,
                        (SELECT array_agg(
                                    sin((((i + {offset}) % {CLUSTERS}) * 13 + d)::double precision * 0.017)
                                    + 0.25 * (random() - 0.5)
                                ORDER BY d)::real[]::vector
                         FROM generate_series(1, {DIMENSION}) d)
                 FROM generate_series(1, {size}) i"
            ))
            .await
            .expect("insert chunks");
            offset += size;
            eprintln!("ann_benchmark: seeded {label} with {size} rows");
        }

        eprintln!("ann_benchmark: building the HNSW index");
        txn.execute_unprepared(
            "CREATE INDEX ann_bench_chunks_embedding ON ann_bench_chunks
                 USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
             CREATE INDEX ann_bench_chunks_source ON ann_bench_chunks(source_id);",
        )
        .await
        .expect("build the indexes");

        txn.commit().await.expect("commit the seeded corpus");

        // ANALYZE cannot run inside the transaction that created the rows, so
        // it follows the commit.
        exec(db, "ANALYZE ann_bench_chunks; ANALYZE ann_bench_sources;").await;
    }

    /// The notebooks each selectivity case filters on, with their row counts.
    async fn case_notebooks(db: &DatabaseConnection) -> Vec<(&'static str, Option<Uuid>, i64)> {
        let rows = db
            .query_all(Statement::from_string(
                DbBackend::Postgres,
                "SELECT s.notebook_id, count(*)::bigint AS value
                 FROM ann_bench_chunks c
                 JOIN ann_bench_sources s ON c.source_id = s.id
                 GROUP BY s.notebook_id
                 ORDER BY count(*) ASC",
            ))
            .await
            .expect("group query");

        let mut sized: Vec<(Uuid, i64)> = rows
            .into_iter()
            .map(|row| {
                (
                    row.try_get::<Uuid>("", "notebook_id").expect("uuid"),
                    row.try_get::<i64>("", "value").expect("count"),
                )
            })
            .collect();
        sized.sort_by_key(|(_, count)| *count);

        let total = scalar_i64(db, "SELECT count(*)::bigint AS value FROM ann_bench_chunks").await;
        let mut cases: Vec<(&'static str, Option<Uuid>, i64)> = vec![("100%", None, total)];
        for (label, index) in [("0.1%", 0usize), ("1%", 1), ("10%", 2)] {
            let (id, count) = sized[index];
            cases.push((label, Some(id), count));
        }
        cases
    }

    /// Query vectors: existing embeddings, so every query sits inside a
    /// cluster the way a real question sits near its passage.
    async fn query_vectors(&self) -> Vec<String> {
        let rows = self
            .db
            .query_all(Statement::from_string(
                DbBackend::Postgres,
                format!(
                    "SELECT embedding::text AS value FROM ann_bench_chunks
                     ORDER BY id LIMIT {QUERIES}"
                ),
            ))
            .await
            .expect("query vectors");
        rows.into_iter()
            .map(|row| row.try_get::<String>("", "value").expect("vector text"))
            .collect()
    }

    /// Apply a strategy's settings to an open transaction.
    ///
    /// One statement, exactly as retrieval sends them.
    async fn apply(txn: &impl ConnectionTrait, strategy: ScanStrategy) {
        let preamble = strategy.session_preamble();
        if preamble.is_empty() {
            return;
        }
        txn.execute_unprepared(&preamble)
            .await
            .unwrap_or_else(|e| panic!("settings failed: {preamble}\n{e}"));
    }

    /// Run one search and return the ordered `(id, distance)` rows plus the
    /// elapsed time.
    async fn search(
        &self,
        strategy: ScanStrategy,
        notebook: Option<Uuid>,
        vector: &str,
    ) -> (Vec<(Uuid, f64)>, f64) {
        let txn = self.db.begin().await.expect("begin");
        Self::apply(&txn, strategy).await;

        let sql = search_sql(notebook, strategy);
        let started = Instant::now();
        let rows = txn
            .query_all(Statement::from_string(
                DbBackend::Postgres,
                sql.replace("$VECTOR", vector)
                    .replace("$NOTEBOOK", &notebook.unwrap_or_default().to_string()),
            ))
            .await
            .expect("search");
        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
        txn.commit().await.expect("commit");

        (
            rows.into_iter()
                .map(|row| {
                    (
                        row.try_get::<Uuid>("", "id").expect("id"),
                        row.try_get::<f64>("", "distance").expect("distance"),
                    )
                })
                .collect(),
            elapsed,
        )
    }

    /// `EXPLAIN (ANALYZE, FORMAT JSON)` for one query, reduced to a summary.
    async fn explain(
        &self,
        strategy: ScanStrategy,
        notebook: Option<Uuid>,
        vector: &str,
    ) -> (PlanSummary, i64) {
        let txn = self.db.begin().await.expect("begin");
        Self::apply(&txn, strategy).await;
        let sql = format!(
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {}",
            search_sql(notebook, strategy)
                .replace("$VECTOR", vector)
                .replace("$NOTEBOOK", &notebook.unwrap_or_default().to_string())
        );
        let row = txn
            .query_one(Statement::from_string(DbBackend::Postgres, &sql))
            .await
            .expect("explain")
            .expect("one row");
        let plan: serde_json::Value = row.try_get("", "QUERY PLAN").expect("plan json");
        txn.commit().await.expect("commit");

        let scanned = tuples_scanned(&plan);
        (summarize_plan(&plan), scanned)
    }
}

/// The production dense query, over the fixture tables.
///
/// The 100% case drops the notebook predicate: excluding nothing is what 100%
/// selectivity means, and keeping a predicate that matches every row would
/// measure the predicate instead.
///
/// The exact reference orders by `distance + 0`. No vector index can serve that
/// expression, so the ordering is computed from every matching row, and the
/// harness never has to disable a planner feature on a pooled connection to get
/// ground truth.
fn search_sql(notebook: Option<Uuid>, strategy: ScanStrategy) -> String {
    let filter = if notebook.is_some() {
        "WHERE s.notebook_id = '$NOTEBOOK'"
    } else {
        ""
    };
    let order = if strategy.is_exact_reference() {
        "(c.embedding <=> '$VECTOR'::vector) + 0"
    } else {
        "c.embedding <=> '$VECTOR'::vector"
    };
    format!(
        "SELECT c.id, (c.embedding <=> '$VECTOR'::vector) AS distance
         FROM ann_bench_chunks c
         JOIN ann_bench_sources s
           ON c.source_id = s.id AND s.active_generation_id = c.generation_id
         {filter}
         ORDER BY {order}
         LIMIT {TOP_K}"
    )
}

/// Reduce an `EXPLAIN` tree to the few facts a strategy comparison rests on.
///
/// See [`PlanSummary`] for why the full tree is not what gets committed.
fn summarize_plan(plan: &serde_json::Value) -> PlanSummary {
    fn walk(node: &serde_json::Value, summary: &mut PlanSummary, depth: usize) {
        let node_type = node
            .get("Node Type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let index = node
            .get("Index Name")
            .and_then(serde_json::Value::as_str)
            .map(|name| format!(" using {name}"))
            .unwrap_or_default();
        if depth > 0 {
            summary.shape.push_str(" → ");
        }
        summary.shape.push_str(node_type);
        summary.shape.push_str(&index);
        if index.contains("embedding") {
            summary.uses_vector_index = true;
        }

        let i64_field = |name: &str| {
            node.get(name)
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0)
        };
        summary.rows_removed_by_filter += i64_field("Rows Removed by Filter");
        summary.shared_blocks_hit += i64_field("Shared Hit Blocks");
        summary.shared_blocks_read += i64_field("Shared Read Blocks");

        if let Some(children) = node.get("Plans").and_then(serde_json::Value::as_array) {
            for child in children {
                walk(child, summary, depth + 1);
            }
        }
    }

    let mut summary = PlanSummary {
        shape: String::new(),
        uses_vector_index: false,
        actual_rows: -1,
        rows_removed_by_filter: 0,
        shared_blocks_hit: 0,
        shared_blocks_read: 0,
    };
    if let Some(root) = plan.as_array().and_then(|a| a.first())
        && let Some(node) = root.get("Plan")
    {
        summary.actual_rows = node
            .get("Actual Rows")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(-1);
        walk(node, &mut summary, 0);
    }
    summary
}

/// Sum of rows produced and rows discarded across every node of a plan.
///
/// "Tuples scanned" is not one number PostgreSQL reports: an HNSW scan that
/// returns 4 rows after discarding 19,996 did far more work than one that
/// returned 10 immediately, and only the sum shows it.
fn tuples_scanned(plan: &serde_json::Value) -> i64 {
    fn walk(node: &serde_json::Value, total: &mut i64) {
        if let Some(rows) = node.get("Actual Rows").and_then(serde_json::Value::as_i64) {
            let loops = node
                .get("Actual Loops")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(1);
            *total += rows * loops;
        }
        if let Some(removed) = node
            .get("Rows Removed by Filter")
            .and_then(serde_json::Value::as_i64)
        {
            *total += removed;
        }
        if let Some(children) = node.get("Plans").and_then(serde_json::Value::as_array) {
            for child in children {
                walk(child, total);
            }
        }
    }

    let mut total = 0;
    if let Some(root) = plan.as_array().and_then(|a| a.first())
        && let Some(node) = root.get("Plan")
    {
        walk(node, &mut total);
    }
    total
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn report_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("backend/ always has a parent")
        .join("contracts/baseline/ann/filtered-hnsw.json")
}

// ============================================================================
// The benchmark
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "Seeds 100k vectors; requires PostgreSQL with pgvector (TEST_DATABASE_URL)"]
async fn filtered_hnsw_is_measured_against_exact_search() {
    let Some(bench) = Bench::setup().await else {
        eprintln!("ann_benchmark: TEST_DATABASE_URL unset, skipping");
        return;
    };

    let capabilities = VectorCapabilities::probe(&bench.db)
        .await
        .expect("probe pgvector");
    let postgres_version = bench
        .db
        .query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT version() AS value",
        ))
        .await
        .expect("version")
        .expect("row")
        .try_get::<String>("", "value")
        .expect("version string");

    // Every strategy this build can actually run. An unsupported mode is
    // recorded as a blocking dimension rather than silently skipped (US-015).
    let mut strategies = vec![ScanStrategy::Hnsw { ef_search: 100 }];
    let mut unsupported: Vec<String> = Vec::new();
    for mode in [IterativeMode::StrictOrder, IterativeMode::RelaxedOrder] {
        for max_scan_tuples in [20_000, 100_000] {
            let strategy = ScanStrategy::HnswIterative {
                mode,
                ef_search: 100,
                max_scan_tuples,
            };
            if capabilities.ensure_supports(strategy).is_ok() {
                strategies.push(strategy);
            } else {
                unsupported.push(strategy.label());
            }
        }
    }
    // Cheapest first, by the strategy's own cost, so "the cheapest that
    // qualifies" is a property of the strategies and not of the loop above.
    strategies.sort_by_key(|s| s.cost_rank());

    let vectors = bench.query_vectors().await;
    assert_eq!(vectors.len(), QUERIES, "the corpus must supply every query");

    let total_rows = scalar_i64(
        &bench.db,
        "SELECT count(*)::bigint AS value FROM ann_bench_chunks",
    )
    .await;

    let mut cases = Vec::new();
    for (label, notebook, matching_rows) in bench.notebooks.clone() {
        // Ground truth for this selectivity: exact search, same queries.
        let mut truth: Vec<BTreeSet<Uuid>> = Vec::with_capacity(vectors.len());
        for vector in &vectors {
            let (ids, _) = bench.search(ScanStrategy::Exact, notebook, vector).await;
            truth.push(ids.into_iter().map(|(id, _)| id).collect());
        }

        let mut measurements = Vec::new();
        for strategy in std::iter::once(ScanStrategy::Exact).chain(strategies.iter().copied()) {
            // Warm up before timing. The first query of a cell pays for pages
            // the previous cell evicted, and a cold P95 next to a warm one
            // compares cache states rather than strategies.
            for vector in vectors.iter().take(WARMUP_QUERIES) {
                let _ = bench.search(strategy, notebook, vector).await;
            }

            let mut recalls = Vec::with_capacity(vectors.len());
            let mut fills = Vec::with_capacity(vectors.len());
            let mut latencies = Vec::with_capacity(vectors.len());
            let mut out_of_order = 0usize;
            let mut max_violation = 0.0_f64;

            let mut first_returned = 0usize;
            for (index, vector) in vectors.iter().enumerate() {
                let (rows, elapsed) = bench.search(strategy, notebook, vector).await;
                if index == 0 {
                    first_returned = rows.len();
                }
                // Distance order is a requirement, not a preference: fusion
                // consumes dense rank (US-016 AC-3). Measured per query rather
                // than trusted from the strategy's declared mode.
                let violation = rows
                    .windows(2)
                    .map(|w| w[0].1 - w[1].1)
                    .fold(0.0_f64, f64::max);
                max_violation = max_violation.max(violation);
                if violation > ORDER_TOLERANCE {
                    out_of_order += 1;
                }

                let expected = &truth[index];
                #[allow(clippy::cast_precision_loss)]
                let hits = rows.iter().filter(|(id, _)| expected.contains(id)).count() as f64;
                #[allow(clippy::cast_precision_loss)]
                let denominator = expected.len().max(1) as f64;
                recalls.push(hits / denominator);
                #[allow(clippy::cast_precision_loss)]
                fills.push((rows.len() as f64 / TOP_K as f64).min(1.0));
                latencies.push(elapsed);
            }

            latencies.sort_by(|a, b| a.partial_cmp(b).expect("finite latencies"));
            #[allow(clippy::cast_precision_loss)]
            let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
            let recall = mean(&recalls);
            let fill = mean(&fills);
            let p95 = percentile(&latencies, 95.0);

            let (plan, scanned) = bench.explain(strategy, notebook, &vectors[0]).await;

            // The plan and the timed queries must describe the same execution.
            // When they disagree, something outside this loop changed how the
            // query ran, and every number in this cell is suspect.
            assert_eq!(
                plan.actual_rows,
                i64::try_from(first_returned).expect("row count fits i64"),
                "{} @ {label}: the plan returned {} rows and the timed query \
                 returned {first_returned}",
                strategy.label(),
                plan.actual_rows
            );

            let mut blocking = Vec::new();
            if recall < MIN_RECALL {
                blocking.push(format!("recall_at_10 {recall:.3} < {MIN_RECALL}"));
            }
            if fill < MIN_FILL {
                blocking.push(format!("fill_rate {fill:.3} < {MIN_FILL}"));
            }
            if p95 > MAX_P95_MS {
                blocking.push(format!("p95_ms {p95:.1} > {MAX_P95_MS}"));
            }
            if out_of_order > 0 {
                blocking.push(format!(
                    "{out_of_order}/{} queries returned rows out of distance order \
                     (largest backward step {max_violation:.2e} > {ORDER_TOLERANCE:.0e})",
                    vectors.len()
                ));
            }
            if !strategy.preserves_distance_order() {
                blocking.push(
                    "the mode does not guarantee distance order, which fusion consumes".to_owned(),
                );
            }

            measurements.push(Measurement {
                strategy: strategy.label(),
                recall_at_10: round(recall),
                fill_rate: round(fill),
                p50_ms: round(percentile(&latencies, 50.0)),
                p95_ms: round(p95),
                tuples_scanned: scanned,
                out_of_order_queries: out_of_order,
                max_order_violation: max_violation,
                plan,
                settings: strategy.session_settings(),
                meets_thresholds: blocking.is_empty(),
                blocking,
            });
        }

        #[allow(clippy::cast_precision_loss)]
        let measured = matching_rows as f64 / total_rows.max(1) as f64;
        cases.push(SelectivityCase {
            label: label.to_owned(),
            measured_selectivity: round(measured),
            matching_rows,
            measurements,
        });
    }

    // The recommendation: the cheapest approximate strategy that clears every
    // threshold at every selectivity. `strategies` is sorted by cost, so "the
    // first that qualifies" is the cheapest that qualifies. Exact search is
    // excluded on purpose: it reads the whole table, so recommending it would
    // be recommending that the index be abandoned.
    let candidates: Vec<String> = strategies
        .iter()
        .map(|s| s.label())
        .filter(|label| {
            cases.iter().all(|case| {
                case.measurements
                    .iter()
                    .any(|m| &m.strategy == label && m.meets_thresholds)
            })
        })
        .collect();

    let blocking: Vec<String> = cases
        .iter()
        .flat_map(|case| {
            case.measurements
                .iter()
                .filter(|m| m.strategy != "exact" && !m.meets_thresholds)
                .map(|m| format!("{} @ {}: {}", m.strategy, case.label, m.blocking.join("; ")))
        })
        .chain(
            unsupported
                .iter()
                .map(|s| format!("{s}: unsupported by this pgvector build")),
        )
        .collect();

    let chosen = candidates.first().cloned();
    let settings = strategies
        .iter()
        .find(|s| Some(s.label()) == chosen)
        .map(|s| s.session_settings())
        .unwrap_or_default();

    let recommendation = Recommendation {
        rationale: chosen.as_ref().map_or_else(
            || {
                "No approximate strategy met Recall@10 >= 0.95, fill >= 0.99, P95 <= 300 ms \
                 and exact distance order at every selectivity. US-016 is blocked."
                    .to_owned()
            },
            |label| {
                format!(
                    "{label} is the cheapest strategy, ordered by ScanStrategy::cost_rank, \
                     that met Recall@10 >= {MIN_RECALL}, fill >= {MIN_FILL}, \
                     P95 <= {MAX_P95_MS} ms and exact distance order at every measured \
                     selectivity."
                )
            },
        ),
        strategy: chosen.clone(),
        settings,
        blocking: blocking.clone(),
    };

    let report = BenchmarkReport {
        schema: "rag-ann-benchmark/1",
        postgres_version,
        pgvector_version: capabilities.extension_version.clone(),
        iterative_scan_supported: capabilities.iterative_scan,
        reference_environment: REFERENCE_ENVIRONMENT,
        corpus_size: CORPUS_SIZE,
        dimension: DIMENSION,
        clusters: CLUSTERS,
        queries_per_cell: QUERIES,
        warmup_queries_per_cell: WARMUP_QUERIES,
        top_k: TOP_K,
        thresholds: Thresholds {
            min_recall_at_10: MIN_RECALL,
            min_fill_rate: MIN_FILL,
            max_p95_ms: MAX_P95_MS,
            order_tolerance: ORDER_TOLERANCE,
        },
        cases,
        recommendation,
    };

    let path = report_path();
    std::fs::create_dir_all(path.parent().expect("a parent directory")).expect("create report dir");
    let mut rendered = serde_json::to_string_pretty(&report).expect("serialize report");
    rendered.push('\n');
    std::fs::write(&path, rendered).expect("write report");
    eprintln!("ann_benchmark: wrote {}", path.display());

    // Non-zero exit when nothing qualifies, with the blocking dimension in the
    // failure message and in the committed report (US-015 AC-5).
    assert!(
        chosen.is_some(),
        "no filtered-ANN strategy met the thresholds; US-016 is blocked by: {}",
        blocking.join(" | ")
    );
}

fn round(value: f64) -> f64 {
    if value.is_finite() {
        (value * 1_000.0).round() / 1_000.0
    } else {
        0.0
    }
}
