# Upgrading OpenbookLM

The core is pre-1.0. This document says what an upgrade guarantees, what it
does not, and what to do when one goes wrong.

## What a version means

The Rust core and the TypeScript SDK share one version. `0.2.0` of the server
goes with `0.2.0` of `@openbooklm/sdk`; a mismatched pair is unsupported and the
SDK's compatibility check will say so.

Until `1.0.0`:

| Surface | Stability |
|---|---|
| REST paths, request and response bodies in `contracts/openapi.json` | breaking changes only in a minor version, listed in the release notes |
| SSE event names, ordering and terminal semantics (`docs/contracts/event-protocol-v1.md`) | same |
| `openbooklm::core` items: `CoreState`, `CoreConfig`, `Principal`, `EntitlementPolicy`, `EventSink`, `ChatEvent`, `SourceEvent`, `build_core_router` | the intended embedding surface; changes are called out |
| Every other Rust item | not stable, may move in any release |
| Core schema | additive only; no column or table is removed in a minor version |
| Environment variables | a removed variable is accepted and ignored for one minor version before it becomes an error |

**Additive SSE events are not a breaking change.** A client must ignore an
event name it does not know and keep the stream open. If your integration
treats an unknown event as an error, fix that before upgrading anything.

## Supported jumps

- **Patch** (`0.2.0` → `0.2.1`): drop-in. No migration, no configuration change.
- **Minor** (`0.2.x` → `0.3.0`): supported, one step at a time. Read the release
  notes; there may be a migration and an SDK bump.
- **Skipping minors** (`0.2.0` → `0.5.0`): not supported. Migrations are only
  tested against the version before them. Upgrade through each minor.
- **Downgrade**: supported only within a minor, and only back to a version that
  knows every applied migration. The server refuses to start otherwise rather
  than run against a schema it cannot account for.

## Procedure

### 1. Back up

Non-negotiable. There is no down migration to fall back on.

```bash
pg_dump "$DATABASE_URL" --format=custom --file=openbooklm-$(date +%F).dump
```

Verify the dump restores into a scratch database before continuing. An
unverified backup is not a backup.

### 2. Check the state before touching anything

```bash
openbooklm-migrate validate -u "$DATABASE_URL"
```

`Empty` or `Tracked` means proceed. `Divergent` means **stop**: the running
code and the database disagree about what the schema is, and applying more SQL
on top of that disagreement is how a migration corrupts data instead of failing.
The command prints what it found and what to do.

### 3. Stop the server, upgrade, start it

```bash
docker compose -f docker/docker-compose.yml pull
docker compose -f docker/docker-compose.yml up -d
```

The server validates the migration state and applies pending core migrations on
start, under a Postgres advisory lock. Two instances started by a rolling
deploy serialise; they do not race.

To apply migrations separately — the right choice when you want the schema
change and the code change to be distinct steps:

```bash
openbooklm-migrate up -u "$DATABASE_URL"
```

### 4. Verify

```bash
curl -fsS localhost:3001/health
curl -fsS -H "x-health-token: $HEALTH_TOKEN" localhost:3001/health/detailed | jq .
```

`/health/detailed` reports per-dependency status, pool metrics and
circuit-breaker state. Then ingest one source and ask one question: a healthy
process with an unreachable embedding provider looks fine until you try to use
it.

## Rollback

**Never run `down`, `fresh` or `refresh`.** They are not exposed by
`openbooklm-migrate` for that reason. Rollback is: previous binary, previous
database.

1. Stop the server.
2. Restore the dump from step 1 into a fresh database.
3. Point `DATABASE_URL` at it.
4. Start the previous version.
5. Confirm with `openbooklm-migrate validate`.

Because migrations are additive, the previous binary usually runs against the
*upgraded* schema too — new columns are simply unread. That is the fast path
when the problem is in the code rather than the data, and it avoids losing
anything written since the upgrade. Try it before restoring, and check the
release notes: a release that says otherwise means the fast path is closed.

## Configuration changes

Removed variables are ignored for one minor version, with a warning naming the
replacement. Renamed variables accept both spellings for that window. A new
*required* variable only ever appears in a minor version and is listed at the
top of its release notes.

Startup validation reports every configuration problem at once, before binding a
socket. If the server starts, its configuration is complete.

## Contract changes

Every release publishes `contracts/openapi.json` and the matching SDK. To see
what changed between two releases:

```bash
git diff v0.2.0..v0.3.0 -- contracts/openapi.json
```

CI rejects a removal, a new required field, or an incompatible type change
unless the release is declared breaking, so the diff is small by construction.
