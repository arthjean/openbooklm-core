# Frontend → SDK replacement map

**Story:** US-010 (EP-003, `tasks/prd-open-core.md`) defines it; **US-018**
executes it.
**Status:** every row below is **transitional**. The frontend still uses its
handwritten declaration; the SDK export exists and is contract-tested.

The private frontend duplicates the core contract by hand. `packages/sdk-ts`
now generates the same contract from the Rust definitions, so each handwritten
declaration has exactly one replacement. This document is that mapping, written
before the migration so US-018 is a mechanical substitution rather than a
redesign, and so a reviewer can see at a glance what stays private.

Import the SDK as `@openbooklm/sdk`.

## `frontend/src/types/core.ts`

| Handwritten | SDK replacement | Note |
|---|---|---|
| `Notebook` | `Notebook` | `description` becomes `string \| null` (drift D-008) |
| `Source` | `Source` | see **Source view state** below |
| `SourceType` | `SourceType` | derived from `SOURCE_TYPES`, generated from the Rust enum |
| `SourceStatus` | `SourceStatus` | loses `"ocr"`, which the server never sends (drift D-011) |
| `ChatMessage` | `ChatMessage` | loses `context_relevance`, which is never sent (drift D-009) |
| `Citation` | `Citation` | identical |
| `Chunk` | `Chunk` | identical |
| `Note` | `Note` | `original_message_id` becomes `string \| null` (drift D-008) |
| `NotebookMemory` | `Memory` | identical shape, contract name |
| `MemoryType`, `MemoryMetadata` | `Memory["memory_type"]`, `Memory["metadata"]` | narrowed by the generated schema |

### Source view state

`Source.embedding_progress`, `Source.ocr_progress` and `Source.ocr_cache_hit`
are **not** on the REST DTO and never were (drift D-010). The client merges them
from `source:status`, `source:ocr_progress` and `source:ocr_cache_hit` events.

US-018 keeps them, in a private type that composes the public one:

```ts
import type { Source, SourceEventOf } from "@openbooklm/sdk";

/** Private: the SDK `Source` plus the fields the UI merges from SSE. */
export interface SourceViewModel extends Source {
  embedding_progress?: SourceEventOf<"source:status">["data"]["progress"];
  ocr_progress?: { current_page: number; total_pages: number };
  ocr_cache_hit?: boolean;
}
```

## `frontend/src/lib/api/*.ts`

Each module's exported functions become methods on one `OpenbookLMClient`.

| Module | SDK replacement | Fate |
|---|---|---|
| `notebooks.ts` | `client.listNotebooks/createNotebook/getNotebook/updateNotebook/deleteNotebook` | replaced |
| `sources.ts` | `client.listSources/createSource/getSource/getSourceChunks/deleteSource/reprocessSource/youtubeTitle` | replaced |
| `notes.ts` | `client.listNotes/createNote/getNote/updateNote/deleteNote` | replaced |
| `memories.ts` | `client.listMemories/getMemory/updateMemory/deleteMemory/deleteAllMemories` | replaced |
| `chat.ts` | `client.getChatHistory/clearChatHistory/listTeachingModes/sendMessage` | replaced |
| `rag-logs.ts` | `client.submitFeedback/getNotebookMetrics/getAccountMetrics` | replaced |
| `suggestions.ts` | `client.getSuggestions` | replaced |
| `settings.ts` | `client.getSettings/updateSettings` | **partially**: onboarding stays private until US-011 splits it |
| `health.ts` | `client.health/detailedHealth` | replaced |
| `billing.ts` | none | **stays private** |
| `feedback.ts` | none | **stays private** |

## SSE

| Handwritten | SDK replacement |
|---|---|
| `ChatStreamEvent` union in `lib/api/chat.ts` | `ChatEvent` |
| `ChatChunkEvent`, `ChatCitationsEvent`, … | `ChatChunk`, `ChatCitations`, … |
| `parseChatStream`, `parseChatSSEData` | `parseChatEvent` + `readEventStream`, or `client.sendMessage` |
| `SourceStatusEvent`, `SourceReadyEvent`, … in `lib/sse.ts` | `SourceEventOf<"source:status">`, … |
| `SourceResyncEvent` | `SourceEventOf<"source:resync">` |
| `ThinkingStage` in `lib/stores/streaming-store.ts` | `ChatThinking["stage"]` |

The SDK's `parseChatEvent` yields `{event, data}` envelopes, while the
frontend's parser yields `{type, data}`. US-018 either adapts the call sites or
adds a one-line mapper; the payloads are byte-identical either way, which the
shared golden fixtures prove.

## Constants

| Handwritten | SDK replacement |
|---|---|
| `MAX_MESSAGE_LENGTH` in `lib/api/chat.ts` | `VALIDATION_LIMITS.max_message_length` |
| any hardcoded chunk or history bound | `VALIDATION_LIMITS.*` |
| teaching-mode lists | `TEACHING_MODES`, `DEFAULT_TEACHING_MODE` |
| model and provider lists | `PROVIDERS` |

## What is **not** replaced

These stay in the private frontend, by design. The SDK must never export them:

- Billing plans, plan ranks, prices, upgrade presentation, Stripe identifiers.
- Clerk claims, session shapes, sign-in components.
- PostHog event names and properties.
- Feedback, micro-feedback, newsletter and onboarding DTOs.
- Every UI-only type: view models, form state, store shapes, i18n keys.

`packages/sdk-ts/test/contract.test.ts` asserts the negative: the published
surface contains none of `clerk`, `stripe`, `posthog`, `resend`, `subscription`
or `price_`, and documents no `/api/billing`, `/api/webhooks`, `/api/feedback`,
`/api/micro-feedback` or `/api/public/` route.
