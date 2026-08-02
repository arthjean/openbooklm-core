# Database migrations

**Story:** US-012 (EP-003, `tasks/prd-open-core.md`)
**Code:** `backend/migration/`
**Verified against:** PostgreSQL 17 with pgvector

Three tracks, three history tables, one rule: **a migration is never edited
after it has been applied.**

| Track | Migrator | History table | Owns |
|---|---|---|---|
| legacy | `Migrator` | `seaql_migrations` | the applied history of the hosted database |
| core | `core_track::CoreMigrator` | `seaql_migrations_core` | the public core schema and every future change to it |
| SaaS | `saas_track::SaasMigrator` | `seaql_migrations_saas` | identity, billing, usage, webhooks, feedback, newsletter |

Separate history tables are what let the public and private schemas version
independently. With one shared table, a public release could not add a core
migration without the private repository also knowing about it.

## The three database paths

### 1. Fresh public install (self-hosted)

```bash
migration core-up -u "$DATABASE_URL"
```

Creates the core schema, the pgvector extension, the HNSW indexes and the
`content_tsv` trigger. No `users`, `subscriptions`, `usage` or `identities`
table exists: a self-hosted install has no identity provider and no billing.

### 2. Fresh hosted install

```bash
migration core-up -u "$DATABASE_URL"
migration saas-up -u "$DATABASE_URL"
```

Core first, always: `identities`, `saas_account_settings` and `micro_feedback`
reference core tables.

### 3. Existing hosted database

```bash
migration validate -u "$DATABASE_URL"   # must report `Legacy`
migration bridge   -u "$DATABASE_URL"
```

The bridge records both baselines as satisfied **without running their SQL**.
The legacy history already created every table the baselines describe;
re-running them would fail on existing objects and, if forced, destroy data.

The bridge is idempotent, and refuses to act on a history it cannot account
for. After it runs, `core-up` and `saas-up` are no-ops until a new migration is
added to either track.

`backend/entrypoint.sh` runs validate → up → bridge → core-up → saas-up on every
container start. Each step is a no-op once satisfied, so the same script serves
all three paths.

## One documented schema difference

A **fresh** install has `notebooks.user_id` and `rag_logs.user_id` referencing
`accounts(id)`. A **bridged** legacy database still references `users(id)`.

Both columns hold the same UUIDs — US-011 backfilled `accounts` from `users`
without changing a single identifier — so ownership and every query behave
identically. Repointing the constraint is a *contraction*, which this PRD
forbids; it belongs to the later cleanup PRD, after the rollback window.

Two legacy tables, `agents` and `notebook_agents`, exist only on bridged
databases. They have no entity, no repository and no reader; they are on the
same contraction list.

## Migration state validation

`migration validate` classifies the database and applies nothing:

| State | Meaning | Next step |
|---|---|---|
| `Empty` | no history | `core-up` (+ `saas-up` if hosted) |
| `Legacy` | complete legacy history, not yet bridged | `bridge` |
| `Bridged` | legacy history plus both baselines recorded | `core-up`, `saas-up` |
| `Tracked` | tracks only, no legacy history | `core-up`, `saas-up` |
| `Divergent` | anything else | **stop** — see the remediation it prints |

`Divergent` covers a partially applied legacy history, a version this build does
not recognise, and a bridge interrupted between its two writes. In every case
the running code and the database disagree about what the schema is, and
applying more SQL on top of that disagreement is how a migration corrupts data
instead of failing.

`core-up` and `saas-up` run `validate` first and refuse to proceed on a
divergent database, so the check cannot be skipped by calling them directly.

## Concurrency

Every verb holds a PostgreSQL advisory lock (`pg_advisory_lock`, key
`MIGRATION_LOCK_KEY`). If two instances of the same new release start together,
the lock makes the second wait and then find nothing to apply. This serializes
migrators, not application writes from an older release. A migration that
changes a writer protocol requires all older instances to stop first.

The lock is session-scoped, so a killed process releases it when its connection
closes.

## Rollback

**Never** use `migration fresh`, `migration refresh`, or `migration down` on a
hosted database. All three drop tables. The core baseline's `down` in particular
would drop tables it never created on a bridged database.

Rollback is a *deployment* operation, not a down migration. Additive schema does
not by itself guarantee binary compatibility: constraints and writer protocols
also matter.

For migrations explicitly marked rollback-compatible, redeploy the previous
binary and leave the expanded schema in place. Across
`m20260801_000001_index_generations`, restore the backup and then start the
previous binary. Its validator refuses the unknown migration and its writer
cannot populate `chunks.generation_id`.

Restore the pre-upgrade backup for an incompatible migration:

```bash
pg_restore --clean --if-exists -d "$DATABASE_URL" backup.dump
```

That loses everything written since the backup, which is why the coordinated
upgrade starts with a verified dump and a maintenance window.

The rollback window is one stable hosted release. After it closes, a later PRD
removes the legacy columns and tables, and rollback past that point requires a
restore.

## Applied core migrations

| Version | Adds |
|---|---|
| `m20260729_000001_core_baseline` | the complete core schema for a fresh install |
| `m20260801_000001_index_generations` | immutable index generations, the active-generation pointer, and the backfill of existing chunks (EP-002) |
| `m20260801_000002_rag_log_redaction` | query hashes plus structural scrubbing of legacy raw RAG log fields (US-004) |
| `m20260802_000001_data_retention` | source ownership and cascade deletion for derived OCR cache entries |

The RAG-log redaction is intentionally irreversible: existing raw query,
reformulation and HyDE values are cleared during `up`, and the trigger rejects
future writes to those legacy fields. A `down` migration cannot reconstruct
that private text.

The data-retention migration adds source ownership and cascade deletion for new
OCR cache writes. `source_id` stays nullable for one rolling-compatibility
window because the previous writer does not populate it; the new reader ignores
nullable legacy rows, and the migration discards the pre-existing unowned
cache. The reference server's daily retention pass removes any nullable rows
written while an older process is still draining. No source document is
removed: OCR cache rows are recomputable derived data.

`m20260801_000001_index_generations` is additive and idempotent, and refuses to
run on a corpus it cannot represent: among the chunks it would backfill,
duplicate `(source_id, chunk_index)` pairs, NULL embeddings, or chunks whose
source no longer exists abort the migration naming the source that caused it.
Fix the listed sources and run it again. Chunks that already belong to a
generation are not re-validated, so a replay on a database that has been
reprocessed since stays a no-op.

It also changes what the previous binary can do. The old code writes chunks
without a `generation_id`, which the new `NOT NULL` constraint refuses, so the
migration and the new binary deploy in one stop-first maintenance window. The
advisory lock serializes migrators only; it does not make a rolling deploy safe. See
[architecture/index-generations.md](architecture/index-generations.md) for the
model, its invariants and the tests that verify them.

## Adding a migration

- **Core schema change** → new file in `backend/migration/src/core_track/`,
  appended to `CoreMigrator::migrations()`. It ships in a public release.
- **SaaS schema change** → new file in `backend/migration/src/saas_track/`,
  appended to `SaasMigrator::migrations()`. Private.
- **Never** add to the legacy list. It is closed.

Expand first, contract later: add a column or table, backfill, deploy readers,
and only remove the old shape after a full release window.
