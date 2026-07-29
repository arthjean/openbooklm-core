# Golden contract baseline

Generated fixtures freezing the current public REST and SSE behavior (US-002,
`tasks/prd-open-core.md`). They are the compatibility target for the open-core
split: EP-002 and EP-003 may not change them without an explicit contract
decision.

**Do not hand-edit these files.** They are produced from the Rust types.

## Layout

| Path | Contents |
|---|---|
| `rest/*.json` | Response DTOs, success and failure, per domain |
| `rest/problem-details.json` | RFC 7807 shapes for every error class |
| `sse/chat.json` | One payload per chat SSE event name |
| `sse/source.json` | One payload per `SourceEvent` variant, plus `source:resync` |
| `rag/citation-extraction.json` | Citation extraction over synthetic chunks |
| `latency/*.json` | P50/P95 measurements; machine-dependent, see below |

Each file maps a case name to the recorded value.

## Producers and consumers

- **Producer:** `backend/tests/contract_baseline.rs`. Serializes live Rust types
  and compares against these files.
- **Consumer:** `frontend/src/lib/__tests__/contract-drift.test.ts`. Feeds the
  recorded payloads through the real TypeScript parsers to pin the Rust/TS
  differences catalogued in `docs/contracts/known-drift.md`.
- **Future consumer:** `packages/sdk-ts` (US-010).

## Regenerating

```bash
cd backend
UPDATE_BASELINE=1 cargo test --test contract_baseline
git diff ../contracts/baseline    # review every line: this is a contract change
```

Regeneration is deterministic: two consecutive runs with no source change produce
byte-identical files.

A regenerated fixture always contains the **complete** serialization of the live
value. A field added to a Rust type therefore appears in the diff instead of
being silently dropped, which is what `assert_no_dropped_fields` enforces on
every non-regenerating run.

## Latency

`latency/*.json` is machine-dependent and is written only under
`UPDATE_BASELINE=1`. No test asserts a wall-clock threshold. The numbers recorded
on the reference environment live in `docs/contracts/latency-baseline.md`, and
comparison against them is a deployment gate (US-019), not a unit test.

## Synthetic data only

Every identifier, title and body in these fixtures is synthetic. Production-derived
data, real account identifiers and real email addresses are forbidden here by PRD
hard constraint, and the frontend test asserts their absence.
