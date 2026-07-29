# SSE baseline — current wire behavior

> **Superseded for the chat and source streams by
> `docs/contracts/sse-protocol-v1.md` (US-009).** This document remains the
> record of the pre-US-009 behavior captured by US-002. Where the two disagree,
> the differences are the ones catalogued as D-001 to D-007 in
> `docs/contracts/known-drift.md`, and `v1` is authoritative.

**Story:** US-002 (EP-001, `tasks/prd-open-core.md`)
**Status:** descriptive baseline of the code at `4c3b06b`, not a specification.

This document records what the server emits **today**. US-009 replaces the
untyped chat payloads with a typed `ChatEvent` and publishes a versioned `v1`
protocol; until then, this file plus `contracts/baseline/sse/` is the
compatibility target that seam extraction must not break.

Executable counterpart: `backend/tests/contract_baseline.rs`. Every event name
listed here has at least one recorded payload, and the test fails if the code
emits a name with no fixture.

---

## 1. Chat stream

`POST /api/notebooks/:id/chat` (and the regenerate endpoint) respond with
`text/event-stream`.

**Transport**

| Property | Value | Source |
|---|---|---|
| Channel buffer | 100 events | `SSE_CHANNEL_BUFFER` |
| Keep-alive | 15 s | `SSE_KEEPALIVE_SECS` |
| `X-Accel-Buffering` | `no` | `apply_sse_headers` |
| `Cache-Control` | `no-cache, no-store` | `apply_sse_headers` |
| Event id | **not set** on chat events | — |
| `Last-Event-ID` replay | **not supported** on chat | — |

Chat events carry no `id:` field, so a dropped chat connection cannot be
resumed. Reconnecting restarts generation. Source events behave differently
(§2).

**Events**

| `event:` | Payload | When |
|---|---|---|
| `thinking` | `{"stage": "retrieving_context" \| "reformulating_query" \| "generating"}` | Progress before the first token |
| `system` | `{"type": "history_truncated", "kept": <n>}` | History dropped to fit the token budget |
| `system` | `{"type": "history_summarized", "dropped_count": <n>}` | Dropped turns were summarized into memory |
| `warning` | `{"type": "low_retrieval_quality"}` | Corrective RAG judged retrieval weak |
| `chunk` | `{"text": "<partial>"}` | Each provider text delta |
| `citation` | `{"index": <1-based>, "source_id": "<uuid>"}` | Incremental, as a `[N]` marker appears, or per native citation |
| `citations` | `{"citations": [<Citation>]}` | Once, after generation, full resolved set |
| `metrics` | `{"context_relevance": <f32> \| null}` | Once, after `citations` |
| `done` | `{"model": "<id>", "provider": "<name>", "rag_log_id": "<uuid>"?}` | Terminal success |
| `follow_up_suggestions` | `{"suggestions": ["…"]}` | After `done`, best-effort |
| `error` | `{"message": "<text>"}` | Terminal failure |
| `shutdown` | `{"message": "Server shutting down"}` | Graceful shutdown interrupts the stream |

**Ordering and termination**

1. `thinking(retrieving_context)` → optional `thinking(reformulating_query)` →
   optional `warning` → `thinking(generating)`.
2. `system` events may appear between prompt assembly and generation.
3. Zero or more `chunk`, interleaved with `citation`.
4. `citations`, then `metrics`.
5. `done` is the terminal successful event.
6. `follow_up_suggestions` is emitted **after** `done`, best-effort: it is
   skipped for Quiz mode, when Mistral is unavailable, on a 5 s timeout, or when
   the response is not valid JSON.
7. `error` is terminal on failure. Chunks already delivered are kept and are not
   retracted.

**Known deviations from the intended v1 protocol**

- `follow_up_suggestions` after `done` contradicts "`done` is the terminal
  successful event". US-009 moves it before `done`. Recorded here so the move is
  a reviewed contract change, not an accidental reordering.
- One failure path emits `error` and then still emits `done`
  (`backend/src/api/chat/streaming.rs`, provider-error branch). US-009 makes
  `error` unconditionally terminal.

**Cancellation**

The handler holds a `CancellationToken`. Client disconnect drops the mpsc
receiver, `send_sse` observes a closed channel and stops sending. Provider work
already in flight is cancelled where the provider supports it. Persistence of a
partial assistant message follows the current behavior in `streaming.rs` and is
not changed by this baseline.

---

## 2. Source processing stream

`GET /api/notebooks/:id/sources/events` responds with `text/event-stream`.

**Transport**

| Property | Value | Source |
|---|---|---|
| Broadcast capacity | 100 events per notebook | `CHANNEL_CAPACITY` |
| Replay buffer | 200 events per notebook | `REPLAY_BUFFER_CAPACITY` |
| Event id | monotonic `u64` per notebook, set on every event | `serialize_sse_event` |
| `Last-Event-ID` replay | supported | — |
| Channel reclamation | idle 300 s, hard age 3600 s, swept every 60 s | `SseCleanupConfig` |

**Events**

| `event:` | Payload |
|---|---|
| `source:status` | `{"source_id", "status", "error_message": string\|null, "progress": {"chunks_done","chunks_total"}\|null}` |
| `source:ready` | `{"source_id", "chunk_count", "degraded_services": [string]}` |
| `source:error` | `{"source_id", "message"}` |
| `source:ocr_started` | `{"source_id", "total_pages"}` |
| `source:ocr_progress` | `{"source_id", "current_page", "total_pages"}` |
| `source:ocr_completed` | `{"source_id", "pages_processed"}` |
| `source:ocr_cache_hit` | `{"source_id"}` |
| `source:resync` | `{"missed": <n>}` |

**Replay and resync**

- On connect with `Last-Event-ID: <n>`, the server replays buffered events with
  id `> n` before switching to live delivery.
- When the requested id is older than the replay buffer, or the broadcast
  receiver lags, the server emits `source:resync` with the number of missed
  events. The client must refetch source state; the gap is not recoverable.
- `source:resync` is the one event with **no** `SourceEvent` enum variant: the
  handler builds it inline. It has a baseline fixture regardless, because the
  wire protocol is the contract, not the enum.

**Serialization asymmetry**

`SourceEvent` derives `Serialize` with `#[serde(tag = "event", content = "data")]`,
but the HTTP handler does not use that representation. `serialize_sse_event`
re-encodes each variant by hand into the SSE `event:`/`data:` framing, and the
two encodings differ: the derived form omits `error_message`, `progress` and an
empty `degraded_services` via `skip_serializing_if`, while the handler always
writes those keys, emitting explicit `null`. `contracts/baseline/sse/source.json`
records the **derived** form. The handler form is documented in
`docs/contracts/known-drift.md` as D-005 and is unified in US-009.

---

## 3. Rules that must survive the split

These hold today and must still hold after EP-002 and EP-003:

- Event names are stable strings. Renaming one is a breaking change.
- A client that receives an unknown `event:` name must ignore it and stay
  connected. The current TypeScript chat parser does this; see
  `docs/contracts/known-drift.md` D-004 for where it does not.
- `done` is the terminal successful event for chat.
- Source event ids are monotonic per notebook and are the only replay key.
- No event payload carries an email address, a Clerk subject, a Stripe
  identifier or raw document content beyond the cited excerpt.
