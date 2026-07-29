/**
 * The SSE event protocol, version `v1`.
 *
 * The unions are aliases into the generated schemas: `ChatEvent` and
 * `SourceEvent` are `#[serde(tag = "event", content = "data")]` enums in Rust,
 * so `openapi-typescript` already produces a discriminated union keyed on
 * `event`. This module adds only what a schema cannot express — which events
 * terminate a stream, and how to parse a frame — as specified in
 * `docs/contracts/sse-protocol-v1.md`.
 */

import type { components } from "./generated/openapi.js";

type Schemas = components["schemas"];

// ── Chat stream ──────────────────────────────────────────────────────

/** Every event the chat stream can emit. Discriminated on `event`. */
export type ChatEvent = Schemas["ChatEvent"];
export type ChatEventName = ChatEvent["event"];

export type ChatChunk = Schemas["ChatChunk"];
export type ChatCitationRef = Schemas["ChatCitationRef"];
export type ChatCitations = Schemas["ChatCitations"];
export type ChatMetrics = Schemas["ChatMetrics"];
export type ChatThinking = Schemas["ChatThinking"];
export type ChatSystem = Schemas["ChatSystem"];
export type ChatWarning = Schemas["ChatWarning"];
export type ChatFollowUpSuggestions = Schemas["ChatFollowUpSuggestions"];
export type ChatDone = Schemas["ChatDone"];
export type ChatError = Schemas["ChatError"];
export type ChatShutdown = Schemas["ChatShutdown"];

/** Narrow a `ChatEvent` to one variant. */
export type ChatEventOf<N extends ChatEventName> = Extract<ChatEvent, { event: N }>;

/**
 * Events that end the stream. Exactly one reaches a client, and nothing
 * follows it.
 */
export const CHAT_TERMINAL_EVENTS = ["done", "error", "shutdown"] as const;

const CHAT_EVENT_NAMES = new Set<string>([
  "chunk",
  "thinking",
  "system",
  "warning",
  "citation",
  "citations",
  "metrics",
  "follow_up_suggestions",
  ...CHAT_TERMINAL_EVENTS,
]);

export function isChatTerminal(event: ChatEvent): boolean {
  return (CHAT_TERMINAL_EVENTS as readonly string[]).includes(event.event);
}

// ── Source stream ────────────────────────────────────────────────────

/** Every event the source processing stream can emit. */
export type SourceEvent = Schemas["SourceEvent"];
export type SourceEventName = SourceEvent["event"];
export type SourceEventOf<N extends SourceEventName> = Extract<SourceEvent, { event: N }>;

export type SourceStatusData = Schemas["SourceStatusData"];
export type EmbeddingProgress = Schemas["EmbeddingProgress"];

const SOURCE_EVENT_NAMES = new Set<string>([
  "source:status",
  "source:ready",
  "source:error",
  "source:ocr_started",
  "source:ocr_progress",
  "source:ocr_completed",
  "source:ocr_cache_hit",
  "source:resync",
]);

// ── Parsing ──────────────────────────────────────────────────────────

/** One raw SSE frame: the `event:` name and the `data:` payload, unparsed. */
export interface RawEvent {
  event: string | null;
  data: string;
}

/**
 * Parse a raw chat frame.
 *
 * Returns `null` for an unknown event name or malformed JSON. The protocol
 * requires a client to *ignore* those and keep reading: additive event variants
 * are not a breaking change, so an older SDK must survive a newer core.
 */
export function parseChatEvent(raw: RawEvent): ChatEvent | null {
  return parseEvent(raw, CHAT_EVENT_NAMES) as ChatEvent | null;
}

/** Parse a raw source frame. Unknown names are ignored, as for chat. */
export function parseSourceEvent(raw: RawEvent): SourceEvent | null {
  return parseEvent(raw, SOURCE_EVENT_NAMES) as SourceEvent | null;
}

function parseEvent(raw: RawEvent, known: ReadonlySet<string>): unknown {
  if (raw.event === null || !known.has(raw.event)) return null;
  let data: unknown;
  try {
    data = JSON.parse(raw.data);
  } catch {
    return null;
  }
  return { event: raw.event, data };
}

/**
 * Split a `text/event-stream` body into raw frames.
 *
 * Handles multi-line `data:` fields and carries `id:` through for the source
 * stream's `Last-Event-ID` replay.
 */
export async function* readEventStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<RawEvent & { id: string | null }> {
  const decoder = new TextDecoder();
  const reader = body.getReader();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const parsed = parseFrame(frame);
        if (parsed) yield parsed;
        boundary = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): (RawEvent & { id: string | null }) | null {
  let event: string | null = null;
  let id: string | null = null;
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith(":")) continue; // comment, used for keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const rest = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = rest;
    else if (field === "data") dataLines.push(rest);
    else if (field === "id") id = rest;
  }

  if (dataLines.length === 0) return null;
  return { event, id, data: dataLines.join("\n") };
}
