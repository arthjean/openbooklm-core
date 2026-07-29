/**
 * Typed client for the OpenbookLM core API.
 *
 * Every method's request and response types come from `types.ts`, which aliases
 * the generated schemas. A route that changes shape in Rust changes shape here
 * on the next `bun run generate`, and the compiler reports every affected call
 * site.
 *
 * The client is deliberately thin: `fetch`, a bearer token, and RFC 7807 error
 * mapping. Caching, retries and request deduplication belong to the consumer.
 */

import {
  ApiError,
  type ChatHistory,
  type Chunk,
  type ChunkList,
  type CreateNoteRequest,
  type CreateNotebookRequest,
  type CreateSourceRequest,
  type DetailedHealth,
  type Health,
  type Memory,
  type MemoryList,
  type Metrics,
  type Note,
  type NoteList,
  type Notebook,
  type NotebookList,
  type ProblemDetails,
  type SendMessageRequest,
  type Settings,
  type Source,
  type SourceList,
  type Suggestions,
  type TeachingModes,
  type UpdateFeedbackRequest,
  type UpdateMemoryRequest,
  type UpdateNoteRequest,
  type UpdateNotebookRequest,
  type UpdateSettingsRequest,
  type YouTubeTitle,
} from "./types.js";
import { type ChatEvent, type SourceEvent, parseChatEvent, parseSourceEvent, readEventStream } from "./events.js";

export interface ClientOptions {
  /** Base URL of the core server, e.g. `http://localhost:3001`. */
  baseUrl: string;
  /**
   * Bearer token, or a function returning one. A function is re-invoked per
   * request, so a rotating token does not require rebuilding the client.
   */
  token?: string | (() => string | Promise<string>);
  /** Injected in tests, or to add tracing to every request. */
  fetch?: typeof globalThis.fetch;
}

interface RequestOptions {
  query?: Record<string, string | number | boolean | undefined>;
  body?: unknown;
  signal?: AbortSignal;
}

export class OpenbookLMClient {
  readonly #baseUrl: string;
  readonly #token: ClientOptions["token"];
  readonly #fetch: typeof globalThis.fetch;

  constructor(options: ClientOptions) {
    this.#baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.#token = options.token;
    this.#fetch = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  // ── Notebooks ──────────────────────────────────────────────────────

  listNotebooks(o?: RequestOptions): Promise<NotebookList> {
    return this.#json("GET", "/api/notebooks", o);
  }
  createNotebook(body: CreateNotebookRequest, o?: RequestOptions): Promise<Notebook> {
    return this.#json("POST", "/api/notebooks", { ...o, body });
  }
  getNotebook(id: string, o?: RequestOptions): Promise<Notebook> {
    return this.#json("GET", `/api/notebooks/${enc(id)}`, o);
  }
  updateNotebook(id: string, body: UpdateNotebookRequest, o?: RequestOptions): Promise<Notebook> {
    return this.#json("PATCH", `/api/notebooks/${enc(id)}`, { ...o, body });
  }
  deleteNotebook(id: string, o?: RequestOptions): Promise<unknown> {
    return this.#json("DELETE", `/api/notebooks/${enc(id)}`, o);
  }

  // ── Sources ────────────────────────────────────────────────────────

  listSources(notebookId: string, o?: RequestOptions): Promise<SourceList> {
    return this.#json("GET", `/api/notebooks/${enc(notebookId)}/sources`, o);
  }
  createSource(notebookId: string, body: CreateSourceRequest, o?: RequestOptions): Promise<Source> {
    return this.#json("POST", `/api/notebooks/${enc(notebookId)}/sources`, { ...o, body });
  }
  getSource(id: string, o?: RequestOptions): Promise<Source> {
    return this.#json("GET", `/api/sources/${enc(id)}`, o);
  }
  getSourceChunks(id: string, o?: RequestOptions): Promise<ChunkList> {
    return this.#json("GET", `/api/sources/${enc(id)}/chunks`, o);
  }
  deleteSource(id: string, o?: RequestOptions): Promise<unknown> {
    return this.#json("DELETE", `/api/sources/${enc(id)}`, o);
  }
  reprocessSource(id: string, o?: RequestOptions): Promise<Source> {
    return this.#json("POST", `/api/sources/${enc(id)}/reprocess`, o);
  }
  youtubeTitle(url: string, o?: RequestOptions): Promise<YouTubeTitle> {
    return this.#json("GET", "/api/youtube/title", { ...o, query: { url } });
  }

  // ── Notes ──────────────────────────────────────────────────────────

  listNotes(notebookId: string, o?: RequestOptions): Promise<NoteList> {
    return this.#json("GET", `/api/notebooks/${enc(notebookId)}/notes`, o);
  }
  createNote(notebookId: string, body: CreateNoteRequest, o?: RequestOptions): Promise<Note> {
    return this.#json("POST", `/api/notebooks/${enc(notebookId)}/notes`, { ...o, body });
  }
  getNote(id: string, o?: RequestOptions): Promise<Note> {
    return this.#json("GET", `/api/notes/${enc(id)}`, o);
  }
  updateNote(id: string, body: UpdateNoteRequest, o?: RequestOptions): Promise<Note> {
    return this.#json("PATCH", `/api/notes/${enc(id)}`, { ...o, body });
  }
  deleteNote(id: string, o?: RequestOptions): Promise<unknown> {
    return this.#json("DELETE", `/api/notes/${enc(id)}`, o);
  }

  // ── Memories ───────────────────────────────────────────────────────

  listMemories(notebookId: string, o?: RequestOptions): Promise<MemoryList> {
    return this.#json("GET", `/api/notebooks/${enc(notebookId)}/memories`, o);
  }
  deleteAllMemories(notebookId: string, o?: RequestOptions): Promise<unknown> {
    return this.#json("DELETE", `/api/notebooks/${enc(notebookId)}/memories`, o);
  }
  getMemory(id: string, o?: RequestOptions): Promise<Memory> {
    return this.#json("GET", `/api/memories/${enc(id)}`, o);
  }
  updateMemory(id: string, body: UpdateMemoryRequest, o?: RequestOptions): Promise<Memory> {
    return this.#json("PATCH", `/api/memories/${enc(id)}`, { ...o, body });
  }
  deleteMemory(id: string, o?: RequestOptions): Promise<unknown> {
    return this.#json("DELETE", `/api/memories/${enc(id)}`, o);
  }

  // ── Chat ───────────────────────────────────────────────────────────

  getChatHistory(
    notebookId: string,
    page?: { offset?: number; limit?: number },
    o?: RequestOptions,
  ): Promise<ChatHistory> {
    return this.#json("GET", `/api/notebooks/${enc(notebookId)}/chat`, { ...o, query: page });
  }
  clearChatHistory(notebookId: string, o?: RequestOptions): Promise<unknown> {
    return this.#json("DELETE", `/api/notebooks/${enc(notebookId)}/chat`, o);
  }
  listTeachingModes(o?: RequestOptions): Promise<TeachingModes> {
    return this.#json("GET", "/api/teaching-modes", o);
  }

  /**
   * Send a message and yield the typed event stream.
   *
   * Yields until a terminal event or the connection ends. Unknown event names
   * are skipped without ending iteration, which is what the `v1` protocol
   * requires of a client older than the core it is talking to.
   */
  async *sendMessage(
    notebookId: string,
    body: SendMessageRequest,
    o?: RequestOptions,
  ): AsyncGenerator<ChatEvent> {
    const response = await this.#request("POST", `/api/notebooks/${enc(notebookId)}/chat`, {
      ...o,
      body,
    });
    if (!response.body) return;
    for await (const raw of readEventStream(response.body)) {
      const event = parseChatEvent(raw);
      if (event) yield event;
    }
  }

  /**
   * Subscribe to a notebook's source processing events.
   *
   * `lastEventId` resumes from a previous connection. If the server cannot
   * satisfy it, the first event is `source:resync` and the caller must refetch
   * the source list.
   */
  async *sourceEvents(
    notebookId: string,
    lastEventId?: string,
    o?: RequestOptions,
  ): AsyncGenerator<SourceEvent & { id: string | null }> {
    const response = await this.#request(
      "GET",
      `/api/notebooks/${enc(notebookId)}/sources/events`,
      o,
      lastEventId ? { "Last-Event-ID": lastEventId } : undefined,
    );
    if (!response.body) return;
    for await (const raw of readEventStream(response.body)) {
      const event = parseSourceEvent(raw);
      if (event) yield { ...event, id: raw.id };
    }
  }

  // ── Retrieval quality ──────────────────────────────────────────────

  submitFeedback(logId: string, body: UpdateFeedbackRequest, o?: RequestOptions): Promise<unknown> {
    return this.#json("PATCH", `/api/rag-logs/${enc(logId)}/feedback`, { ...o, body });
  }
  getNotebookMetrics(notebookId: string, days?: number, o?: RequestOptions): Promise<Metrics> {
    return this.#json("GET", `/api/notebooks/${enc(notebookId)}/metrics`, { ...o, query: { days } });
  }
  getAccountMetrics(days?: number, o?: RequestOptions): Promise<Metrics> {
    return this.#json("GET", "/api/metrics", { ...o, query: { days } });
  }
  getSuggestions(notebookId: string, o?: RequestOptions): Promise<Suggestions> {
    return this.#json("GET", `/api/notebooks/${enc(notebookId)}/suggestions`, o);
  }

  // ── Settings and health ────────────────────────────────────────────

  getSettings(o?: RequestOptions): Promise<Settings> {
    return this.#json("GET", "/api/settings", o);
  }
  updateSettings(body: UpdateSettingsRequest, o?: RequestOptions): Promise<Settings> {
    return this.#json("PATCH", "/api/settings", { ...o, body });
  }
  health(o?: RequestOptions): Promise<Health> {
    return this.#json("GET", "/health", o);
  }
  detailedHealth(o?: RequestOptions): Promise<DetailedHealth> {
    return this.#json("GET", "/health/detailed", o);
  }

  // ── Transport ──────────────────────────────────────────────────────

  async #request(
    method: string,
    path: string,
    options?: RequestOptions,
    extraHeaders?: Record<string, string>,
  ): Promise<Response> {
    const url = new URL(this.#baseUrl + path);
    for (const [key, value] of Object.entries(options?.query ?? {})) {
      if (value !== undefined) url.searchParams.set(key, String(value));
    }

    const headers: Record<string, string> = { Accept: "application/json", ...extraHeaders };
    const token = typeof this.#token === "function" ? await this.#token() : this.#token;
    if (token) headers.Authorization = `Bearer ${token}`;
    if (options?.body !== undefined) headers["Content-Type"] = "application/json";

    const response = await this.#fetch(url.toString(), {
      method,
      headers,
      body: options?.body === undefined ? undefined : JSON.stringify(options.body),
      signal: options?.signal,
    });

    if (!response.ok) throw await toApiError(response);
    return response;
  }

  async #json<T>(method: string, path: string, options?: RequestOptions): Promise<T> {
    const response = await this.#request(method, path, options);
    return (await response.json()) as T;
  }
}

function enc(segment: string): string {
  return encodeURIComponent(segment);
}

async function toApiError(response: Response): Promise<ApiError> {
  let problem: ProblemDetails | null = null;
  try {
    const body: unknown = await response.json();
    if (body !== null && typeof body === "object" && "status" in body && "title" in body) {
      problem = body as ProblemDetails;
    }
  } catch {
    // A non-JSON body (proxy error page, empty 502) leaves `problem` null.
  }
  return new ApiError(response.status, problem, `${response.status} ${response.statusText}`);
}
