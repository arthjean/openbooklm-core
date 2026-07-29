/**
 * The SDK is checked against the same golden fixtures the Rust tests generate.
 *
 * `contracts/baseline/sse/*.json` is written by `backend/tests/contract_baseline.rs`
 * from the typed `ChatEvent` and `SourceEvent`. Reading it here — rather than
 * writing payloads by hand — is what makes "Rust fixtures and TypeScript parser
 * fixtures use identical payloads" a fact rather than an intention.
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  CHAT_TERMINAL_EVENTS,
  CORE_CATALOG,
  EVENT_PROTOCOL_VERSION,
  PROVIDERS,
  SOURCE_STATUSES,
  SOURCE_TYPES,
  TEACHING_MODES,
  VALIDATION_LIMITS,
  isChatTerminal,
  parseChatEvent,
  parseSourceEvent,
  readEventStream,
  type ChatEvent,
  type SourceEvent,
} from "../src/index.js";

const ROOT = path.resolve(import.meta.dirname, "../../..");

/** The subset of an OpenAPI schema node this file inspects. */
interface SchemaNode {
  oneOf?: Array<{ properties?: { event?: { enum?: string[] } } }>;
}

/** The `event` discriminator value of every variant of a tagged-union schema. */
function variantNames(node: SchemaNode | undefined): string[] {
  return (node?.oneOf ?? []).flatMap((variant) => variant.properties?.event?.enum ?? []);
}

function json(relative: string): Record<string, unknown> {
  return JSON.parse(readFileSync(path.join(ROOT, relative), "utf8"));
}

const chatSse = json("contracts/baseline/sse/chat.json");
const sourceSse = json("contracts/baseline/sse/source.json");
const openapi = json("contracts/openapi.json");

/** Recorded chat fixtures, keyed by the SSE event name they belong to. */
const CHAT_CASES: Array<[string, string]> = [
  ["chunk", "chunk"],
  ["thinking", "thinking_retrieving_context"],
  ["thinking", "thinking_reformulating_query"],
  ["thinking", "thinking_generating"],
  ["system", "system_history_truncated"],
  ["system", "system_history_summarized"],
  ["warning", "warning"],
  ["citation", "citation"],
  ["citations", "citations"],
  ["metrics", "metrics"],
  ["metrics", "metrics_no_context"],
  ["follow_up_suggestions", "follow_up_suggestions"],
  ["done", "done"],
  ["done", "done_without_rag_log"],
  ["error", "error"],
  ["shutdown", "shutdown"],
];

describe("chat events", () => {
  it.each(CHAT_CASES)("parses the recorded %s payload (%s)", (name, fixture) => {
    const payload = chatSse[fixture];
    expect(payload, `missing fixture ${fixture}`).toBeDefined();
    const parsed = parseChatEvent({ event: name, data: JSON.stringify(payload) });
    expect(parsed).toEqual({ event: name, data: payload });
  });

  it("covers every event name in the generated union", () => {
    const declared = new Set(CHAT_CASES.map(([name]) => name));
    const schemas = (openapi.components as { schemas: Record<string, SchemaNode> }).schemas;
    const generated = new Set(variantNames(schemas.ChatEvent));
    expect([...generated].sort()).toEqual([...declared].sort());
  });

  it("ignores an unknown event without failing", () => {
    expect(parseChatEvent({ event: "some_future_event", data: "{}" })).toBeNull();
    expect(parseChatEvent({ event: null, data: "{}" })).toBeNull();
  });

  it("ignores malformed JSON without throwing", () => {
    expect(parseChatEvent({ event: "chunk", data: "{not json" })).toBeNull();
  });

  it("classifies exactly the terminal events as terminal", () => {
    for (const [name, fixture] of CHAT_CASES) {
      const event = parseChatEvent({ event: name, data: JSON.stringify(chatSse[fixture]) });
      expect(isChatTerminal(event as ChatEvent)).toBe(
        (CHAT_TERMINAL_EVENTS as readonly string[]).includes(name),
      );
    }
  });

  it("types done.rag_log_id as nullable, matching the wire", () => {
    const parsed = parseChatEvent({
      event: "done",
      data: JSON.stringify(chatSse.done_without_rag_log),
    });
    expect(parsed).not.toBeNull();
    const done = parsed as Extract<ChatEvent, { event: "done" }>;
    expect(done.data.rag_log_id).toBeNull();
  });
});

describe("source events", () => {
  it.each(Object.keys(sourceSse))("parses the recorded %s envelope", (fixture) => {
    const recorded = sourceSse[fixture] as { event: string; data: unknown };
    const parsed = parseSourceEvent({ event: recorded.event, data: JSON.stringify(recorded.data) });
    expect(parsed).toEqual(recorded);
  });

  it("carries resync as a first-class variant", () => {
    const parsed = parseSourceEvent({ event: "source:resync", data: '{"missed":7}' });
    expect(parsed).toEqual({ event: "source:resync", data: { missed: 7 } });
  });

  it("keeps the optional status fields present on the wire", () => {
    const status = (sourceSse.status_processing as { data: Record<string, unknown> }).data;
    expect("error_message" in status).toBe(true);
    const ready = (sourceSse.ready as { data: Record<string, unknown> }).data;
    expect(ready.degraded_services).toEqual([]);
  });
});

describe("event stream framing", () => {
  async function frames(body: string) {
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(body));
        controller.close();
      },
    });
    const out: Array<{ event: string | null; data: string; id: string | null }> = [];
    for await (const frame of readEventStream(stream)) out.push(frame);
    return out;
  }

  it("splits frames, keeps ids and skips keep-alive comments", async () => {
    const parsed = await frames(
      ':keep-alive\n\nevent: source:ready\nid: 42\ndata: {"source_id":"x","chunk_count":1,"degraded_services":[]}\n\n',
    );
    expect(parsed).toHaveLength(1);
    expect(parsed[0]?.event).toBe("source:ready");
    expect(parsed[0]?.id).toBe("42");
  });

  it("joins multi-line data fields", async () => {
    const parsed = await frames('event: chunk\ndata: {"text":\ndata: "hi"}\n\n');
    expect(parsed[0]?.data).toBe('{"text":\n"hi"}');
    expect(parseChatEvent(parsed[0] as { event: string; data: string })).toEqual({
      event: "chunk",
      data: { text: "hi" },
    });
  });
});

describe("catalog", () => {
  it("matches the protocol version the events belong to", () => {
    expect(EVENT_PROTOCOL_VERSION).toBe("v1");
  });

  it("exports the server-enforced limits", () => {
    expect(VALIDATION_LIMITS.max_message_length).toBeGreaterThan(0);
    expect(VALIDATION_LIMITS.max_context_chunks).toBeGreaterThanOrEqual(
      VALIDATION_LIMITS.default_max_context_chunks,
    );
    expect(VALIDATION_LIMITS.max_chat_history_limit).toBeGreaterThanOrEqual(
      VALIDATION_LIMITS.default_chat_history_limit,
    );
  });

  it("exports source types, statuses and teaching modes", () => {
    expect(SOURCE_TYPES).toContain("pdf");
    expect(SOURCE_STATUSES).toContain("ready");
    // `ocr` is an event, never a status: the client union used to disagree.
    expect(SOURCE_STATUSES as readonly string[]).not.toContain("ocr");
    expect(TEACHING_MODES.map((m) => m.id)).toContain("deep");
    expect(TEACHING_MODES.filter((m) => m.is_default)).toHaveLength(1);
  });

  it("exports provider capabilities without naming a commercial vendor", () => {
    expect(PROVIDERS.map((p) => p.provider)).toEqual(["mistral", "anthropic", "openai"]);
    expect(PROVIDERS.find((p) => p.provider === "anthropic")?.native_citations).toBe(true);
  });
});

describe("public boundary", () => {
  const surface = JSON.stringify([openapi, CORE_CATALOG]).toLowerCase();

  it.each(["clerk", "stripe", "posthog", "resend", "subscription", "price_"])(
    "does not expose %s",
    (needle) => {
      expect(surface).not.toContain(needle);
    },
  );

  it.each(["/api/billing", "/api/webhooks", "/api/feedback", "/api/micro-feedback", "/api/public/"])(
    "does not document the SaaS route %s",
    (prefix) => {
      const documented = Object.keys(openapi.paths as Record<string, unknown>);
      expect(documented.filter((p) => p.startsWith(prefix))).toEqual([]);
    },
  );
});
