# Known Rust/TypeScript contract drift

**Story:** US-002 (EP-001, `tasks/prd-open-core.md`)
**Baseline commit:** `4c3b06b`
**Success metric:** these cases reach zero once US-009 (typed `ChatEvent`),
US-010 (generated SDK) and US-018 (frontend migration) land.

**Status.** D-001 to D-007 are **resolved by US-009**: the protocol they
disagreed about is specified in `docs/contracts/sse-protocol-v1.md` and
generated from `backend/src/core/protocol/`. Their regression tests were
rewritten in the same commit and now assert the fixed behavior.

D-008 to D-011 are **resolved at the source by US-010**: `contracts/openapi.json`
and `packages/sdk-ts` now carry one generated definition of every REST type, and
the SDK's declarations are correct for all four cases. They stay listed as open
against the *frontend*, which still uses its handwritten copies; US-018 deletes
those, following `docs/contracts/sdk-replacement-map.md`.

Rust and TypeScript currently define the REST and SSE contracts independently.
Every case below is a real difference between what the server emits and what the
client declares. Each entry states the **canonical** behavior, the owning story,
and the regression test that pins it.

Regression tests live in `frontend/src/lib/__tests__/contract-drift.test.ts` and
`backend/tests/contract_baseline.rs`. They assert the behavior that exists
**today**. When the owning story fixes a case, its test must be changed in the
same commit: that is the point, the fix becomes visible instead of silent.

| ID | Surface | Canonical | Owner | Status |
|---|---|---|---|---|
| D-001 | `thinking.stage` values | Rust | US-009 | resolved |
| D-002 | `system` event missing from the client union | Rust | US-009 | resolved |
| D-003 | `warning` payload shape | Rust | US-009 | resolved |
| D-004 | `shutdown` event missing from the client union | Rust | US-009 | resolved |
| D-005 | `SourceEvent` derived vs handwritten SSE encoding | Handler | US-009 | resolved |
| D-006 | `source:resync` has no Rust variant | TypeScript | US-009 | resolved |
| D-007 | Chat stream termination order | PRD | US-009 | resolved |
| D-008 | Nullable on the wire, optional in TypeScript | Wire | US-010 | SDK correct, frontend pending US-018 |
| D-009 | `ChatMessage.context_relevance` never sent | Rust | US-010 | SDK correct, frontend pending US-018 |
| D-010 | Client view state declared on the REST DTO | Rust | US-010, US-018 | SDK correct, frontend pending US-018 |
| D-011 | `source_type` / `status` closed only in TypeScript | Generated enum | US-010 | partial, see below |

---

## D-001 — `thinking.stage` enumerates fewer values in TypeScript

**Rust** emits three stages:

- `retrieving_context` — `backend/src/api/chat/mod.rs`
- `generating` — `backend/src/api/chat/mod.rs`
- `reformulating_query` — `backend/src/services/chat/orchestration.rs`, twice
  (proactive reformulation for follow-ups, and corrective-RAG re-query)

**TypeScript** declares only two:

```ts
export interface ChatThinkingEvent {
  stage: "retrieving_context" | "generating";
}
```

**Impact.** `reformulating_query` reaches the UI as a `thinking` event whose
`stage` is outside the declared union. TypeScript does not validate at runtime,
so nothing throws: any `switch` on `stage` silently falls through and the user
sees no progress indicator during query reformulation.

**Canonical: Rust.** Three stages. The client union gains `reformulating_query`.

**Resolved (US-009).** `ThinkingStage` is a Rust enum with three variants;
`ChatThinkingStage` in `frontend/src/lib/api/chat.ts` and the store's
`ThinkingStage` declare the same three. The thinking indicator falls back to its
generic label for `reformulating_query`; a dedicated label is US-018's.

## D-002 — `system` is emitted but absent from the client union

**Rust** emits `system` with two payload shapes:

- `{"type": "history_truncated", "kept": <n>}`
- `{"type": "history_summarized", "dropped_count": <n>}`

**TypeScript** has no `system` member in `ChatStreamEvent` and no `"system"` in
`KNOWN_EVENT_TYPES`.

**Impact.** The parser falls through to its shape-based compatibility branch,
which tests `"text" in data`, `"citations" in data`, `"context_relevance" in
data`, `"model" in data`, `"message" in data`. A `system` payload matches none of
them, so `parseChatSSEData` returns `null` and the event is dropped. The user is
never told their conversation history was truncated.

**Canonical: Rust.** `system` is part of the protocol and must be typed. The
`{type, ...}` discriminator becomes a proper variant in US-009.

**Resolved (US-009).** `ChatSystem` is a `#[serde(tag = "type")]` enum in Rust
and a discriminated union in TypeScript. `system` is in `KNOWN_EVENT_TYPES`, so
both payloads now reach the client.

## D-003 — `warning` payload shape disagrees

**Rust** emits `{"type": "low_retrieval_quality"}`.

**TypeScript** declares `ChatWarningEvent` as `{ message: string }`.

**Impact.** `warning` is in `KNOWN_EVENT_TYPES`, so the event is delivered, but
typed with a field that never arrives. Any consumer rendering `data.message`
renders `undefined`.

**Canonical: Rust.** A `type` discriminator, not free text, because the client
must localize the message. US-009 types it as a closed variant set.

**Resolved (US-009).** `WarningKind` is a closed Rust enum;
`ChatWarningEvent` declares `{ type: "low_retrieval_quality" }`.

## D-004 — `shutdown` is emitted but absent from the client union

**Rust** emits `{"message": "Server shutting down"}` under the `shutdown` event
name when graceful shutdown interrupts an in-flight stream.

**TypeScript** has no `shutdown` member.

**Impact.** The shape-based fallback matches `"message" in data` and reclassifies
the event as `{type: "error"}`. This is benign today, and arguably the desired
UX, but it is accidental: the client cannot distinguish an operator-initiated
restart from a provider failure, so it cannot offer a retry.

**Canonical: Rust.** `shutdown` stays a distinct name. US-009 types it and lets
the client decide the presentation.

**Resolved (US-009).** `shutdown` is a `ChatEvent` variant and a member of the
client union, so a restart is no longer indistinguishable from a provider
failure.

## D-005 — `SourceEvent` has two different serializations

`SourceEvent` derives `Serialize` with `#[serde(tag = "event", content = "data")]`
and `skip_serializing_if` on `error_message`, `progress` and an empty
`degraded_services`.

`serialize_sse_event` in `backend/src/api/sources.rs` ignores that derive and
hand-builds a `serde_json::json!` object per variant, always writing every key.

**Impact.** The two encodings disagree on optional fields: the derived form omits
them, the wire form emits explicit `null`. The TypeScript `SourceStatusEvent`
declares `error_message: string | null` and `progress?`, which matches the wire
form, not the derive. Any future consumer of the derived form (a queue, a log
sink, a test helper) would see a different shape than the browser.

**Canonical: the handler wire form**, because it is what clients already parse.
`contracts/baseline/sse/source.json` records the derived form so the divergence
is visible; US-009 makes one serialization authoritative and deletes the other.

**Resolved (US-009).** The three `skip_serializing_if` attributes are gone and
`serialize_sse_event` now frames `SourceEvent::payload()`. One serialization,
matching what clients parse. The fixture gained `error_message: null`,
`progress: null` and `degraded_services: []` accordingly.

## D-006 — `source:resync` exists on the wire and in TypeScript but not in Rust

`SourceEvent` has no `Resync` variant. The handler constructs the event inline
when the replay buffer cannot satisfy `Last-Event-ID` or the broadcast receiver
lags. TypeScript declares `SourceResyncEvent` with `{ missed: number }`.

**Impact.** The Rust enum is not the source of truth for the source stream, so
generating the SSE contract from it would produce an incomplete union and drop
`source:resync` from the SDK.

**Canonical: TypeScript.** The event is real and needed. US-009 adds the Rust
variant so a generated contract is complete.

**Resolved (US-009).** `SourceEvent::Resync { missed }` exists. It is produced
only at the transport edge and carries no SSE `id:`, so a reconnect cannot
replay it — `SourceEvent::source_id()` returns `None` for it, which is why that
method now returns an `Option`.

## D-007 — chat stream termination order contradicts the intended protocol

Two ordering facts today:

1. `follow_up_suggestions` is emitted **after** `done`
   (`backend/src/api/chat/streaming.rs`).
2. The response-truncation path emits `error` and then `done`, in that order,
   and returns `Ok`.

The PRD requires suggestions before `done`, `done` as the terminal successful
event, and exactly one terminal event with no `done` after an `error`.

**Impact.** A client that closes the stream on `done` — the documented contract —
never receives follow-up suggestions. A client that treats `error` as terminal
sees a `done` it should not. Both behaviors are currently load-bearing in the UI,
so reordering is a behavior change, not a bug fix.

**Canonical: the PRD.** US-009 moves suggestions before `done` and makes `error`
unconditionally terminal. Pinned here so the change is reviewed rather than
absorbed.

**Resolved (US-009).** Both changed:

1. `follow_up_suggestions` is emitted before `done`. This delays `done` by up to
   the 5 s suggestion timeout when Mistral is configured and the mode is not
   Quiz. That cost buys the documented contract: a client may now close on
   `done` without losing suggestions.
2. The truncation path emits `error` and stops. `ChatEventStream` enforces the
   rule for every path: the first terminal event wins and nothing follows it, so
   `error` then `done` is unrepresentable rather than merely avoided.

---

## D-008 — nullable on the wire, optional in TypeScript

Rust `Option<T>` fields **without** `skip_serializing_if` serialize to explicit
`null`. `frontend/src/types/core.ts` declares the same fields with `?`, which
under `strictNullChecks` means "absent or `T`" and does **not** admit `null`.

Verified against `contracts/baseline/rest/`:

| Wire field | Emitted | TypeScript declares |
|---|---|---|
| `Notebook.description` | `null` | `description?: string` |
| `Source.error_message` | `null` | `error_message?: string` |
| `Note.original_message_id` | `null` | `original_message_id?: string` |
| `ChatMessage.model` | `null` | `model?: string` |

**Impact.** The declared type is unsound: a `null` flows into code that the type
system says received `string | undefined`. Every `if (x)` guard happens to hide
it, so nothing fails today, but `x ?? fallback` and `x === undefined` behave
differently for `null` than the types promise.

Fields that *do* carry `skip_serializing_if` — `rag_log_id`, `feedback`,
`session_id`, and every optional `Citation` field — are genuinely absent from the
payload and match their `?` declarations. The inconsistency is per-field, which
is exactly what a generated contract removes.

**Canonical: the wire.** US-010 generates these as `T | null` where the server
emits `null` and as optional where it omits the key.

## D-009 — `ChatMessage.context_relevance` is declared but never sent

`frontend/src/types/core.ts` declares `context_relevance?: number | null` on
`ChatMessage`. `ChatMessageResponse` in `backend/src/services/chat/mod.rs` has no
such field and the chat history endpoint never emits it. The value exists only as
the payload of the `metrics` SSE event, per response rather than per message.

**Impact.** A phantom field. Any consumer reading `message.context_relevance`
from history gets `undefined` forever.

**Canonical: Rust.** The field is removed from the client type in US-010; per
-response relevance stays on the `metrics` event.

## D-010 — client-side view state declared on the REST DTO

`frontend/src/types/core.ts` adds `embedding_progress`, `ocr_progress` and
`ocr_cache_hit` to `Source`. None are returned by `GET /api/notebooks/:id/sources`.
The client merges them from `source:status`, `source:ocr_progress` and
`source:ocr_cache_hit` events into the cached object.

**Impact.** The DTO and the view model are the same type, so the SDK cannot
generate `Source` without either dropping fields the UI needs or exporting UI
state as part of the public contract.

**Canonical: Rust for the DTO.** US-010 generates `Source` from the Rust
response; US-018 keeps the merged view state as a separate private type in the
frontend.

## D-011 — `source_type` and `status` are closed unions in TypeScript only

`SourceResponse.source_type` and `SourceResponse.status` are `String` in Rust.
TypeScript narrows them to closed unions (7 source types, 7 statuses).

**Impact.** A new source type or status added server-side is a silent type
violation on the client rather than a compile error, because the value arrives at
runtime through `JSON.parse` and is never validated.

**Canonical: a closed set, generated.** US-010 promotes both to Rust enums so the
union has one source of truth and adding a variant is a visible contract change.

---

## Coverage note

The REST surface is duplicated by hand on both sides. Cases D-008 through D-011
are the differences visible from the response structs and
`frontend/src/types/core.ts` at this commit; they are not a proof of exhaustive
equivalence, which only generation gives. US-010 removes the duplication by
generating `contracts/openapi.json` and `packages/sdk-ts` from the Rust
definitions. Until then, `contracts/baseline/rest/` is the shared reference.

---

## Resolutions from US-010

**D-008.** `contracts/openapi.json` is generated from the Rust response types, so
a field without `skip_serializing_if` is emitted as `T | null` and one with it is
optional. The SDK's `Notebook.description`, `Source.error_message`,
`Note.original_message_id` and `ChatMessage.model` are nullable; `rag_log_id`,
`feedback`, `session_id` and the optional `Citation` fields are optional. Both
are correct per field, which is what generation buys over a hand-kept copy.

**D-009.** `ChatMessage` is generated from `ChatMessageResponse`, which has no
`context_relevance`. The phantom field cannot be declared in the SDK because
nothing generates it. Per-response relevance stays on the `metrics` event.

**D-010.** The SDK's `Source` is the REST DTO and nothing more.
`docs/contracts/sdk-replacement-map.md` shows the private `SourceViewModel` that
US-018 composes from it for the three fields the UI merges from SSE.

**D-011 — partially resolved.** The closed sets now have one source of truth:
`SOURCE_TYPES` and `SOURCE_STATUSES` are generated from the Rust `SourceType`
and `SourceStatus` enums via `contracts/core-constants.json`, and the SDK derives
its unions from them. `SOURCE_STATUSES` therefore no longer contains `"ocr"`,
which the server never sends.

What is **not** done: `SourceResponse.source_type` and `SourceResponse.status`
are still `String` in Rust, so the OpenAPI schema types them as `string` rather
than as the enum. Typing the response fields means converting a database string
into an enum on every read, and an unrecognized value would then fail the whole
list endpoint instead of one row. That trade is a wire-level change with a
migration question attached, not a contract-generation question, so it is
deliberately left out of US-010. Until a story takes it, a consumer that wants
narrowing does it explicitly:

```ts
import { SOURCE_TYPES, type SourceType } from "@openbooklm/sdk";

const isKnownType = (value: string): value is SourceType =>
  (SOURCE_TYPES as readonly string[]).includes(value);
```
