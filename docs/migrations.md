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
`MIGRATION_LOCK_KEY`). A rolling deploy starts the new instance before stopping
the old one, so two processes can reach the migrator simultaneously; the lock
makes the second wait and then find nothing to apply. Verified with two
concurrent `core-up` runs against one empty database: both exit 0, one row is
written.

The lock is session-scoped, so a killed process releases it when its connection
closes.

## Rollback

**Never** use `migration fresh`, `migration refresh`, or `migration down` on a
hosted database. All three drop tables. The core baseline's `down` in particular
would drop tables it never created on a bridged database.

Rollback is a *deployment* operation, not a schema operation. Every migration in
this PRD is additive, so the previous application version runs unchanged against
the expanded schema — that is the whole point of the expand/contract discipline.

1. **Redeploy the previous application version.** Backend and frontend. The
   expanded schema is a superset of what it expects.
2. **Verify.** Sign-in, notebook list, source ingestion, one chat exchange with
   a citation, Stripe portal in test mode.
3. **Leave the schema alone.** The new tables are unread by the old code. They
   cost storage and nothing else.

Restore from backup only if the data itself is wrong, not merely the schema:

```bash
pg_restore --clean --if-exists -d "$DATABASE_URL" backup.dump
```

That *is* destructive and loses everything written since the backup. It is the
last resort, not the rollback procedure.

The rollback window is one stable hosted release. After it closes, a later PRD
removes the legacy columns and tables, and rollback past that point requires a
restore.

## Applied core migrations

| Version | Adds |
|---|---|
| `m20260729_000001_core_baseline` | the complete core schema for a fresh install |
| `m20260801_000001_index_generations` | immutable index generations, the active-generation pointer, and the backfill of existing chunks (EP-002) |

`m20260801_000001_index_generations` is additive and idempotent, and refuses to
run on a corpus it cannot represent: among the chunks it would backfill,
duplicate `(source_id, chunk_index)` pairs, NULL embeddings, or chunks whose
source no longer exists abort the migration naming the source that caused it.
Fix the listed sources and run it again. Chunks that already belong to a
generation are not re-validated, so a replay on a database that has been
reprocessed since stays a no-op.

It also changes what the previous binary can do. The old code writes chunks
without a `generation_id`, which the new `NOT NULL` constraint refuses, so the
migration and the new binary deploy together — which is what the server already
does under its advisory lock. See
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
