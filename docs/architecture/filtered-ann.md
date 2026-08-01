# Filtered approximate search

The measurement behind EP-003's dense-retrieval strategy (US-015), and the
strategy it approved (US-016).

Measured on PostgreSQL 16.14 with pgvector 0.8.5, the version the reference
image (`pgvector/pgvector:pg16`) ships. The report this document summarises is
`contracts/baseline/ann/filtered-hnsw.json`, produced by
`backend/tests/ann_benchmark.rs`.

## The problem

A dense retrieval query filters by notebook and orders by vector distance:

```sql
SELECT c.id FROM chunks c
JOIN sources s ON c.source_id = s.id AND s.active_generation_id = c.generation_id
WHERE s.notebook_id = $1
ORDER BY c.embedding <=> $2::vector
LIMIT 10
```

PostgreSQL runs the HNSW index scan first and applies the notebook filter to
what comes out. With `hnsw.ef_search = 100` the scan yields about 100
candidates for the *whole corpus*, and a notebook holding 0.1% of it keeps
roughly none of them. The query returns one row, or zero, and reports success.
Nothing in the response distinguishes that from a notebook with one relevant
passage.

## What was measured

100,000 synthetic 1,024-dimensional vectors drawn from 256 clusters, in one
table with the same HNSW parameters as the baseline migration
(`m = 16, ef_construction = 128`). Thirty query vectors, `LIMIT 10`, at four
filter selectivities. Recall is against exact search over the same filtered
rows; the exact reference orders by `(embedding <=> $v) + 0`, an expression no
vector index can serve, so ground truth needs no planner setting.

| Selectivity | Strategy | Recall@10 | Fill | P50 | P95 | Tuples scanned |
|---|---|---|---|---|---|---|
| 100% | `hnsw.ef_search=100` | 0.997 | 1.000 | 2.2 ms | 2.4 ms | 41 |
| 100% | iterative `strict_order` | 0.997 | 1.000 | 2.2 ms | 2.6 ms | 41 |
| 10% | `hnsw.ef_search=100` | 0.877 | 0.880 | 2.4 ms | 2.8 ms | 194 |
| 10% | iterative `strict_order` | 0.983 | 1.000 | 2.5 ms | 3.5 ms | 194 |
| 1% | `hnsw.ef_search=100` | 0.120 | 0.120 | 2.5 ms | 2.7 ms | 208 |
| 1% | iterative `strict_order` | 1.000 | 1.000 | 5.8 ms | 6.5 ms | 1,624 |
| 0.1% | `hnsw.ef_search=100` | **0.010** | **0.010** | 2.3 ms | 2.7 ms | 204 |
| 0.1% | iterative `strict_order` | 1.000 | 1.000 | 33.0 ms | 43.3 ms | 17,432 |

The full matrix, including `relaxed_order`, `max_scan_tuples = 100000`, the
plan summaries and the exact-search reference row, is in the report.

The shape of the failure is worth stating plainly: at 0.1% selectivity the
strategy this repository shipped before EP-003 returned **1% of the evidence
that existed**, in under 3 ms, with no error.

## How the harness earns those numbers

Three properties of the measurement, each of which was wrong at some point and
would have produced a plausible report anyway:

- **The corpus is reproducible.** Seeding runs inside one transaction, so
  `setseed` and the index-build parameters apply to the backend that does the
  work. Issued outside one, they land on whichever pooled connection serves
  that statement and the corpus stops being a fixture. The index is built with
  `max_parallel_maintenance_workers = 0`, because a parallel HNSW build
  partitions the work and produces a different graph depending on how many
  workers started.
- **Latencies are warm.** Each cell runs a full untimed pass before the timed
  one. Exact search reads the whole table and evicts the index, so a cell
  measured cold after it reports the eviction, not the strategy: that showed up
  as a single-scan HNSW query timed at 84 ms where the same plan over the same
  41 tuples runs in 2.2 ms warm.
- **Ordering is measured, not assumed.** Every query is checked for rows
  returned out of ascending distance order, and the report publishes the
  largest backward step it saw. Recomputed distances disagree with the index's
  own order by about 1e-7, so the threshold is 1e-6; `relaxed_order` at 10%
  selectivity produces steps of 5.4e-5, two orders of magnitude above it. The
  requirement is real: fusion consumes dense *rank*.

## What was approved

```
hnsw.ef_search      = 100
hnsw.iterative_scan = strict_order
hnsw.max_scan_tuples = 20000
```

Recorded as `APPROVED_STRATEGY` in `backend/src/repositories/ann.rs`, beside the
query it parameterises, and applied by
`SeaOrmSearchRepository::search_similar_chunks` as a single `SET LOCAL`
statement.

- **Iterative over single-pass**, because the single-pass numbers above are not
  a tuning shortfall: no `ef_search` that keeps 0.1% selectivity usable is
  affordable at 100%.
- **`strict_order` over `relaxed_order`**, because relaxed order measurably
  returns rows out of distance order (2 of 30 queries at 10% selectivity) while
  buying nothing: its latencies sit within noise of strict order at every
  selectivity. The benchmark rejects it on the measurement, not on a comment.
- **`max_scan_tuples = 20000`**, pgvector's default, because raising it to
  100,000 changed no recall number and no latency percentile: the scan already
  stops on finding enough filtered rows.

The recommendation is the cheapest qualifying strategy, ordered by
`ScanStrategy::cost_rank` rather than by the order strategies happen to be
listed in, so the report's "cheapest" is a property a reader can check.

Every setting is applied with `SET LOCAL` inside the retrieval transaction, so
it reverts on commit and cannot reach the next borrower of a pooled connection.
`per_query_scan_settings_do_not_leak_into_pooled_connections` in
`backend/tests/rag_integration.rs` asserts that, with concurrent probes across
distinct backend PIDs: sequential probes would interrogate one connection
repeatedly and prove nothing about the pool.

## Required version

**pgvector 0.8.0 or newer.** `hnsw.iterative_scan` appeared in 0.8.0; on an
older build the approved strategy cannot run.

The server probes the extension at startup and refuses to start with a message
naming the requirement and what it found, rather than degrading silently. The
degradation is the reason: an older build does not error on retrieval, it
returns less evidence, which reaches a user as a thinner answer and an operator
as nothing at all.

## Reproducing

```bash
podman run -d --name openbooklm-pg \
  -e POSTGRES_USER=openbooklm -e POSTGRES_PASSWORD=openbooklm -e POSTGRES_DB=openbooklm \
  -p 5432:5432 pgvector/pgvector:pg16

cd backend
TEST_DATABASE_URL=postgres://openbooklm:openbooklm@localhost:5432/openbooklm \
  cargo test --no-default-features --release --test ann_benchmark -- --ignored --nocapture
```

The first run seeds the corpus, which takes a few minutes; later runs reuse it.
The test writes the report and fails when no strategy meets Recall@10 >= 0.95,
fill >= 0.99, P95 <= 300 ms and distance ordering at every selectivity, naming
the blocking dimension. A CI-sized version of the same comparison, over 3,000
rows at 10% selectivity, runs in `rag_integration` as
`filtered_dense_search_matches_exact_search_on_a_reduced_corpus`.

Latency figures are machine-dependent and are not a release gate. What the gate
compares is recall, fill and ordering, which are not.
