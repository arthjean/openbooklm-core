# Index generations

The design proof for EP-002 (US-005). It defines the invariants US-006 through
US-011 implement, and records what was verified against PostgreSQL rather than
assumed.

Verified on PostgreSQL 16.14 with pgvector 0.8.5. The schema uses no feature
newer than PostgreSQL 14, which is the minimum the README declares.

## The problem

Before EP-002, reprocessing a source deleted its chunks and then wrote the
replacement in independently committed batches. Three consequences followed from
that one ordering:

- A provider, database, timeout or process failure between the delete and the
  last batch left the source with a partial index, and search read it.
- Two workers could interleave their deletes and inserts, because nothing
  expressed "this source is being rebuilt by someone".
- A retried batch could duplicate chunk positions, because no constraint said a
  position belonged to exactly one row.

## The model

A source index is an immutable **generation**. A rebuild creates a new one
beside the active one, fills it, proves it complete, and then moves a single
pointer. Nothing deletes the active generation; it stops being read because the
pointer moved, not because its rows went away.

### Identity and lifecycle

`source_index_generations` holds one row per attempt:

| Field | Meaning |
|---|---|
| `id` | immutable generation identity |
| `source_id` | the source it indexes; never changes |
| `state` | `building` → `published` \| `failed`, or `superseded` by an operator |
| `expected_chunk_count` | what the build declared it would store; NULL until chunked |
| `stored_chunk_count` | what it actually stored, recounted at publication |
| `embedding_fingerprint` | provider, model, width, normalization |
| `chunking_fingerprint` | schema version, size unit, sizer, geometry |
| `failure_reason` | stable reason a `failed` generation was abandoned |

`sources.active_generation_id` is the nullable pointer. NULL means the source has
no searchable index — never indexed, or its first build failed.

A published generation that has been replaced stays `published`. It becomes
unreachable by losing the pointer, which is what makes rollback a pointer move
rather than a copy.

### Invariants, and what enforces each

Every one of these is a database constraint. Code that depends on an invariant
can name the constraint that guarantees it.

| Invariant | Enforced by |
|---|---|
| at most one building generation per source | partial unique index `source_index_generations_one_building` |
| no duplicate chunk position inside a generation | unique index `chunks_generation_chunk_index_unique` |
| a chunk belongs to a generation of *its own* source | composite FK `chunks_generation_fk` on `(generation_id, source_id)` |
| a source can only publish a generation it owns | composite FK `sources_active_generation_fk` on `(active_generation_id, id)` |
| a referenced generation cannot be reclaimed | the same FK, `NO ACTION` |
| at most one active generation per source | `active_generation_id` is a single column |

Both composite foreign keys reference `(id, source_id)`, which is why that pair
carries its own unique constraint.

`NO ACTION` rather than `ON DELETE SET NULL` is deliberate. The column-list form
of `SET NULL` requires PostgreSQL 15, and `NO ACTION` is checked at the end of
the statement: deleting a *source* still cascades cleanly, because by the time
the check runs the referencing row is gone, while deleting a *published
generation* on its own is refused. Cleanup safety is therefore a constraint, not
a code path that has to remember.

### Publication

One transaction, in this order:

1. Validate every vector under the generation: none NULL, all of the configured
   width, none holding a non-finite component.
2. Recount `stored_chunk_count` from `chunks`. The counter is a cache; the rows
   are the truth.
3. `UPDATE ... SET state = 'published'` guarded by `source_id`, `state =
   'building'`, `expected_chunk_count IS NOT NULL`, `expected_chunk_count =
   stored_chunk_count` and `stored_chunk_count > 0`. A generation failing any of
   them updates zero rows and the transaction aborts.
4. `UPDATE sources SET active_generation_id, chunk_count, status = 'ready'`.

Steps 1 and 2 share the transaction with steps 3 and 4 on purpose: a vector
written between the check and the pointer move would otherwise be published
unchecked.

**What a concurrent reader observes.** Under `READ COMMITTED`, a reader's
statement takes its snapshot at statement start. The pointer move is a single
row update inside one transaction, so a reader either sees the pre-commit
pointer — and therefore the whole previous generation — or the post-commit
pointer, and therefore the whole replacement. There is no snapshot in which the
pointer holds a value that never existed.

Verified: `a_reader_sees_the_old_generation_until_publication_commits`, and
`a_thousand_publication_schedules_produce_no_mixed_read`, which races a reader
against 1,000 publications at calibrated offsets spanning the commit window and
asserts zero mixed result sets across 2,000 reads.

### Failure

Any failure — extraction, embedding, storage, channel, validation, timeout,
shutdown — marks the building generation `failed` and touches nothing else. The
source's own status is then derived, not asserted:

```sql
status = CASE WHEN active_generation_id IS NULL THEN 'error' ELSE 'ready' END
```

A source whose previous index is intact stays `ready`, because it is. Reporting
`error` would tell the user their document disappeared when it did not.

Verified: `a_failure_at_publication_preserves_the_active_generation`,
`a_failed_build_leaves_a_previously_indexed_source_searchable`,
`a_failed_first_build_reports_the_source_as_failed`.

### Reads

Every RAG read path joins `sources` on **both** `id` and
`active_generation_id = chunks.generation_id`: dense search, lexical search, the
chunk count, the context-stuffing load, the chunk listing, and the notebook
sample used for suggested questions. Citations resolve against the retrieved
context, so they inherit the same scope.

It is a join predicate rather than a `WHERE` clause so that there is no way to
add a filter to one of these queries and forget the generation — the generation
is part of how `sources` is reached.

Verified: `every_read_path_is_scoped_to_the_active_generation`.

### Ownership

Claiming ownership is an insert of the one `building` row a source is allowed to
have, with `ON CONFLICT (source_id) WHERE (state = 'building') DO NOTHING`. The
winner gets a generation id; every other caller gets `None` and returns the
source's current state without spawning a worker.

Both a compare-and-set field and a uniqueness constraint were considered. The
constraint alone is enough here because the `building` row *is* the ownership
record: it carries the state a separate field would have carried, and it cannot
disagree with the generation it describes.

A build older than twice the processing timeout has no live owner —
`process_source` cannot exceed one deadline plus one drain — so recovery marks
it `failed` before the next claim. A worker whose generation was reclaimed
cannot publish: the publication guard requires `state = 'building'`.

Verified: `a_hundred_concurrent_requests_produce_one_owner`,
`a_superseded_worker_cannot_publish`,
`recovery_leaves_a_build_inside_its_deadline_alone`.

### Rollback and reclaim

Rollback repoints a source at its newest `published` generation that is not the
active one, using the same `UPDATE sources` statement publication uses. It
copies nothing and deletes nothing, and public response shapes do not change.
Before selecting that predecessor it locks the `sources` row `FOR UPDATE`.
Publication's pointer update takes the same row lock, so a concurrent rollback
observes either the pointer before publication or the committed pointer after
publication and cannot overwrite a newer generation from a stale snapshot.

Reclaim deletes unreferenced generations older than the retention window, with
three exclusions: the active generation, the newest other published generation
(the rollback target), and anything inside the window. Deletions run one
statement per generation outside any shared transaction, so a generation that
became referenced since the scan fails its foreign key and is skipped while the
others still go.

Retention is at least one prior complete generation and at least 24 hours.
Cleanup never runs inside the publication transaction.

**When it runs.** Immediately after a publication commits, for the source that
was just published, in `source_processing::reclaim_obsolete_generations`. A
publication is the only event that makes a generation obsolete, so it is the
only moment worth looking, and it needs no scheduler to be a real policy rather
than a documented one. The call is deliberately not fallible: the new index is
already live, and disk left behind is an operational cost, not a reason to
report a successful rebuild as a failure.

Verified: `rollback_returns_to_the_previous_complete_generation`,
`rollback_serializes_on_the_source_pointer`,
`a_thousand_publication_rollback_schedules_are_linearizable`,
`rollback_without_a_predecessor_changes_nothing`,
`reclaim_never_removes_a_referenced_or_rollback_eligible_generation`,
`reclaim_respects_the_retention_window`.

## Migration and rollout

`m20260801_000001_index_generations` is additive and idempotent. It creates the
generation table, adds `chunks.generation_id` and `sources.active_generation_id`,
backfills, then installs the constraints — `NOT NULL` last, once every row has a
generation.

**Backfill.** One published generation per source that already has chunks, in
one statement per step inside the migration's transaction. A partial backfill is
not a state this can leave behind. Sources with no chunks keep a NULL pointer,
which is what an unindexed source means.

Backfilled generations carry `legacy:unknown` provenance. Their real provider,
model and chunking configuration were never recorded, and inventing a
fingerprint would let a later reprocess reuse embeddings it cannot prove
compatible. A `legacy:` fingerprint never matches a live one, so the first
reprocess rebuilds instead.

**Refusal.** Three legacy defects abort the migration with the source that
caused them, rather than publishing a generation whose contents cannot be
proven: duplicate `(source_id, chunk_index)` pairs, NULL embeddings, and chunks
whose source no longer exists.

Every one of those checks is scoped to `generation_id IS NULL`, which is why
validation runs *after* that column is added rather than before. The scope is
what makes a replay idempotent on a database that has been used rather than only
on a fresh one: once generations exist, two of them legitimately hold the same
`(source_id, chunk_index)` pair, because uniqueness is
`(generation_id, chunk_index)` and keeping the previous generation for rollback
is the normal state. Validating the whole table would abort on healthy data.

**Forward deployment.** The migration changes the chunk writer protocol. The
old code writes without a `generation_id`, which the `NOT NULL` constraint
refuses. Stop every old instance before the new binary applies the migration;
the advisory lock only serializes migrators and does not protect application
writes. Rolling back means redeploying the previous binary *and* restoring the
backup taken before the upgrade, per `docs/upgrading.md`.

**Batch size.** The backfill is a single grouped `INSERT ... SELECT` plus two
`UPDATE`s. On the synthetic upgraded fixture this is not a measurable cost; a
deployment with a corpus large enough to care should measure it against its own
snapshot before upgrading, because the decision belongs to whoever owns that
database.

Verified: `the_migration_is_idempotent_on_a_fresh_database`,
`a_replay_after_a_reprocess_is_still_a_no_op`,
`the_backfill_preserves_every_existing_chunk`,
`the_backfill_aborts_on_ambiguous_legacy_chunks`.

## Provenance

A generation records what makes its vectors and positions comparable:
`emb:v1:<provider>:<model>:<dim>:<normalization>` and
`chunk:v<schema>:<unit>:<sizer>:<parent>/<child>+<overlap>`.

Reuse is keyed on `(content_hash, embedding_fingerprint)`. Identical text
embedded by a different model is a different vector, and reusing it would mix
two vector spaces inside one index. The query embedding cache is keyed the same
way, plus a namespace separating direct queries, reformulations, HyDE documents
and working-memory lookups.

Verified: `a_changed_embedding_fingerprint_reuses_nothing`,
`a_generation_stores_the_provenance_it_was_built_under`, and the
`embedding_cache` unit tests.

## Cancellation

Ingestion owns its async tasks under a child of the server's shutdown token.
Cancellation proceeds in three steps: admission closure (a cancelled token makes
every not-yet-started unit return immediately), a cooperative drain bounded by a
5-second deadline, then abort.

What abort cannot reach is `spawn_blocking` work. The drain report separates
*drained* from *abandoned* from *blocking in flight*, and the caller logs the
difference rather than claiming everything stopped.

The processing deadline lives inside `process_source` rather than around the
spawned task. `tokio::time::timeout` cancels by dropping the future it wraps, so
a deadline applied from outside would drop the task owner without giving it a
chance to drain — which is precisely how the previous implementation left
embedding calls running after a timeout.

Verified: `a_timed_out_run_stops_calling_the_provider_and_preserves_the_index`,
`a_shutdown_signal_stops_ingestion_and_preserves_the_index`, and the
`ingestion_tasks` unit tests.

## Running the proofs

```bash
cd backend
TEST_DATABASE_URL=postgres://openbooklm:openbooklm@localhost:5432/openbooklm \
  cargo test --no-default-features --test rag_integration -- --ignored
```

The suite provisions its own fixtures and scratch databases and cleans up after
itself, so it is re-runnable against a persistent database.
