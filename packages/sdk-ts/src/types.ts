/**
 * The public REST contract, renamed for readability.
 *
 * Every type here is an alias into `generated/openapi.ts`, which
 * `openapi-typescript` emits from `contracts/openapi.json`. Nothing is
 * redeclared: an alias cannot drift from the schema it points at, whereas a
 * handwritten copy is exactly the drift this package exists to remove
 * (`docs/contracts/known-drift.md`).
 */

import type { components, operations, paths } from "./generated/openapi.js";

export type { components, operations, paths };

type Schemas = components["schemas"];

// ── Resources ────────────────────────────────────────────────────────

export type Notebook = Schemas["NotebookResponse"];
export type NotebookList = Schemas["NotebooksListResponse"];
export type CreateNotebookRequest = Schemas["CreateNotebookRequest"];
export type UpdateNotebookRequest = Schemas["UpdateNotebookRequest"];

export type Source = Schemas["SourceResponse"];
export type SourceList = Schemas["SourcesListResponse"];
export type CreateSourceRequest = Schemas["CreateSourceRequest"];
export type Chunk = Schemas["ChunkResponse"];
export type ChunkList = Schemas["ChunksListResponse"];

export type Note = Schemas["NoteResponse"];
export type NoteList = Schemas["NotesListResponse"];
export type CreateNoteRequest = Schemas["CreateNoteRequest"];
export type UpdateNoteRequest = Schemas["UpdateNoteRequest"];

export type Memory = Schemas["MemoryResponse"];
export type MemoryList = Schemas["MemoriesListResponse"];
export type UpdateMemoryRequest = Schemas["UpdateMemoryRequest"];

export type ChatMessage = Schemas["ChatMessageResponse"];
export type ChatHistory = Schemas["ChatHistoryResponse"];
export type SendMessageRequest = Schemas["SendMessageRequest"];
export type TeachingModeInfo = Schemas["TeachingModeInfo"];
export type TeachingModes = Schemas["TeachingModesResponse"];

export type Citation = Schemas["Citation"];

export type Metrics = Schemas["MetricsResponse"];
export type UpdateFeedbackRequest = Schemas["UpdateFeedbackRequest"];

export type Settings = Schemas["UserSettingsResponse"];
export type UpdateSettingsRequest = Schemas["UpdateDefaultsRequest"];

export type Suggestions = Schemas["SuggestionsResponse"];
export type YouTubeTitle = Schemas["YouTubeTitleResponse"];

export type Health = Schemas["HealthResponse"];
export type DetailedHealth = Schemas["DetailedHealthResponse"];

// ── Errors ───────────────────────────────────────────────────────────

/** RFC 7807 problem details. Every non-2xx core response has this shape. */
export type ProblemDetails = Schemas["ProblemDetails"];

/**
 * Thrown by every client method on a non-2xx response.
 *
 * `problem` is `null` only when the server returned a body that is not RFC 7807
 * — a proxy error page, for instance. Callers that branch on `problem.type`
 * must handle that case.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly problem: ProblemDetails | null;

  constructor(status: number, problem: ProblemDetails | null, fallback: string) {
    super(problem?.detail ?? problem?.title ?? fallback);
    this.name = "ApiError";
    this.status = status;
    this.problem = problem;
  }

  /** Whether the server asked the caller to retry, and after how long. */
  get retryAfterSeconds(): number | null {
    return this.problem?.retry_after ?? null;
  }
}
