/**
 * Generated from contracts/core-constants.json by scripts/generate-catalog.mjs.
 * Do not edit: run `bun run generate` in packages/sdk-ts instead.
 *
 * Source of truth: backend/src/core/catalog.rs
 */

/** Everything the core exposes that is not a request or response shape. */
export const CORE_CATALOG = {
  "event_protocol_version": "v1",
  "validation": {
    "max_title_length": 255,
    "max_description_length": 1000,
    "max_system_prompt_length": 10000,
    "max_message_length": 10000,
    "default_max_context_chunks": 15,
    "max_context_chunks": 20,
    "default_chat_history_limit": 50,
    "max_chat_history_limit": 200,
  },
  "source_types": [
    "pdf",
    "text",
    "markdown",
    "web",
    "docx",
    "epub",
    "youtube",
  ],
  "source_statuses": [
    "pending",
    "processing",
    "contextualizing",
    "embedding",
    "ready",
    "error",
  ],
  "teaching_modes": [
    {
      "id": "flash",
      "name": "Flash",
      "icon": "⚡",
      "description": "Quick essential summary",
      "is_default": false,
    },
    {
      "id": "deep",
      "name": "Deep",
      "icon": "🧠",
      "description": "Complete detailed exploration",
      "is_default": true,
    },
    {
      "id": "quiz",
      "name": "Quiz",
      "icon": "❓",
      "description": "Interactive multiple-choice quiz",
      "is_default": false,
    },
    {
      "id": "glossary",
      "name": "Glossary",
      "icon": "📖",
      "description": "Key terms extraction",
      "is_default": false,
    },
    {
      "id": "summary",
      "name": "Summary",
      "icon": "📄",
      "description": "Structured summary with sections",
      "is_default": false,
    },
    {
      "id": "timeline",
      "name": "Timeline",
      "icon": "🕐",
      "description": "Chronological events",
      "is_default": false,
    },
  ],
  "default_teaching_mode": "deep",
  "providers": [
    {
      "provider": "mistral",
      "native_citations": false,
      "models": [],
    },
    {
      "provider": "anthropic",
      "native_citations": true,
      "models": [
        {
          "id": "claude-opus-4-6-20260220",
          "name": "Claude Opus 4.6",
          "description": "Most capable model for complex tasks",
          "context_window": 200000,
        },
        {
          "id": "claude-sonnet-4-6-20260220",
          "name": "Claude Sonnet 4.6",
          "description": "Best for complex agents and coding",
          "context_window": 200000,
        },
        {
          "id": "claude-haiku-4-5-20251001",
          "name": "Claude Haiku 4.5",
          "description": "Fastest model with near-frontier intelligence",
          "context_window": 200000,
        },
      ],
    },
    {
      "provider": "openai",
      "native_citations": false,
      "models": [
        {
          "id": "gpt-5.2",
          "name": "GPT-5.2",
          "description": "Advanced reasoning",
          "context_window": 400000,
        },
        {
          "id": "gpt-5-mini",
          "name": "GPT-5 mini",
          "description": "Fast and affordable",
          "context_window": 400000,
        },
      ],
    },
  ],
} as const;

/** Version of the SSE event protocol these constants belong to. */
export const EVENT_PROTOCOL_VERSION = CORE_CATALOG.event_protocol_version;

/** Server-enforced input bounds. Respecting these avoids every avoidable 400. */
export const VALIDATION_LIMITS = CORE_CATALOG.validation;

/** Source types the core can ingest. */
export const SOURCE_TYPES = CORE_CATALOG.source_types;
export type SourceType = (typeof SOURCE_TYPES)[number];

/** Statuses a source can hold. `ocr` is an event, never a status. */
export const SOURCE_STATUSES = CORE_CATALOG.source_statuses;
export type SourceStatus = (typeof SOURCE_STATUSES)[number];

/** Teaching modes, with the presentation metadata the API returns. */
export const TEACHING_MODES = CORE_CATALOG.teaching_modes;
export type TeachingMode = (typeof TEACHING_MODES)[number]["id"];
export const DEFAULT_TEACHING_MODE = CORE_CATALOG.default_teaching_mode;

/** Providers the core can route to, and the models each pins. */
export const PROVIDERS = CORE_CATALOG.providers;
export type ProviderName = (typeof PROVIDERS)[number]["provider"];
