# Latency baseline

**Story:** US-002 (EP-001, `tasks/prd-open-core.md`)
**Consumer:** US-019 — hosted cutover allows at most a 10% P95 regression against
these numbers during the first 24 hours.
**Harness:** `backend/tests/latency_baseline.rs`
**Raw report:** `contracts/baseline/latency/*.json`

## Status

All four operations are recorded. One caveat, stated up front because it changes
how the numbers may be used: the database measurements were taken against a
**local pgvector container over loopback**, not against the hosted Neon instance.
Round-trip network latency to a managed Postgres dominates both figures in
production. Read them as a **code**-regression baseline, not as a prediction of
hosted latency. US-019 re-measures against the hosted instance class.

## Method

`measure` and `measure_async` in the harness discard 20 warm-up iterations, then
time 200 samples per operation. Percentiles use the nearest-rank definition: P95
is the smallest sample at or above the 95th percentile of the sorted set.

The harness asserts **no** wall-clock threshold. Absolute timings are
machine-dependent, and a test that fails on a slow CI runner teaches people to
ignore it. The harness produces numbers; this document records them; US-019
compares against them at deployment time.

## What "time to first token" measures here

The offline measurement drives a synthetic provider stream and stops at the first
`TextDelta`. It measures the pipeline's own overhead — stream construction, SSE
frame parsing, event dispatch — with provider network latency removed.

That is deliberate. Provider latency is not something the open-core split can
regress; pipeline overhead is. An end-to-end time-to-first-token figure against a
real provider is a separate measurement and belongs to the US-019 rehearsal, not
to a unit-test harness.

## Recorded measurements

**Captured:** 2026-07-28
**Environment:** AMD Ryzen 7 7800X3D, 16 threads, Linux 7.1.5, rustc 1.97.1,
**release** profile, 200 samples after 20 warm-up iterations.
**Database:** `pgvector/pgvector:pg16` (PostgreSQL 16.14) in a local container over
loopback, all migrations applied.

| Operation | P50 | P95 |
|---|---|---|
| `source_creation_validation` | 0.18 µs | 0.20 µs |
| `chat_time_to_first_token` | 0.16 µs | 0.20 µs |
| `notebook_listing` | 100 µs | 134 µs |
| `lexical_search` | 358 µs | 463 µs |

Raw values are stored in nanoseconds in `contracts/baseline/latency/*.json`. A
microsecond field would record the two offline operations as `0`.

**Fixture for the database measurements.** One synthetic account, 5 notebooks,
and 4 text sources of 40 chunks each in the measured notebook, so 160 indexed
chunks. The lexical query is `retrieval`, a term seeded into every chunk, and the
test asserts it returns rows before measuring: a lexical baseline over an empty
result set times query planning against zero tuples, not retrieval, and would
make the US-019 gate compare noise.

The two offline figures measure the pipeline's own overhead only. They will catch
an order-of-magnitude regression — a validation path that starts allocating per
call, a parser that starts doing work per frame — and nothing subtler.

## Reproducing

```bash
cd backend

# offline, part of the default suite
cargo test --test latency_baseline -- --nocapture

# database-backed, requires pgvector PostgreSQL with migrations applied
TEST_DATABASE_URL=postgres://user:pass@host/db \
  cargo test --test latency_baseline -- --ignored --nocapture

# persist contracts/baseline/latency/*.json
UPDATE_BASELINE=1 cargo test --test latency_baseline -- --nocapture
```

A local pgvector container is enough to reproduce:

```bash
podman run -d --name obl-pg -p 5433:5432 \
  -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=openbooklm_test \
  docker.io/pgvector/pgvector:pg16
cd backend && cargo run -p migration --bin migration -- up \
  -u postgres://test:test@localhost:5433/openbooklm_test
```

Always compare like for like. The recorded numbers are `--release`; a debug run is
roughly an order of magnitude slower on the offline operations and is not
comparable.

## Before the cutover

Two gaps remain, both owned by US-019 rather than by this harness:

1. **Re-measure against the hosted database.** The recorded figures are loopback
   to a local container. Managed Postgres adds round-trip latency that dominates
   both queries, so the current numbers would make the 10% gate trivially pass on
   a hosted run and tell you nothing.
2. **Record an end-to-end chat time to first token** against the real provider
   mix. The harness deliberately excludes provider network latency, which the
   open-core split cannot regress, but the cutover gate is about what users
   experience.

Until both are done, treat this document as a code-regression baseline only.
