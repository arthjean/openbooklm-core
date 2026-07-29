// Emit `src/generated/catalog.ts` from `contracts/core-constants.json` (US-010).
//
// The JSON is produced by `cargo run --bin contracts` from the Rust
// definitions. This script only re-types it: every value here exists because a
// Rust constant, enum or catalogue function produced it, so a limit is never
// written twice.
//
// Run through `bun run generate`, which regenerates the OpenAPI types first.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(here, "../../../contracts/core-constants.json");
const target = resolve(here, "../src/generated/catalog.ts");

/** @type {import("node:fs").PathLike} */
const raw = readFileSync(source, "utf8");
const catalog = JSON.parse(raw);

/** Render a JSON value as a TypeScript literal with `as const`-friendly formatting. */
function literal(value, indent = 0) {
  const pad = "  ".repeat(indent);
  const padInner = "  ".repeat(indent + 1);
  if (Array.isArray(value)) {
    if (value.length === 0) return "[]";
    const items = value.map((v) => `${padInner}${literal(v, indent + 1)}`);
    return `[\n${items.join(",\n")},\n${pad}]`;
  }
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value).map(
      ([k, v]) => `${padInner}${JSON.stringify(k)}: ${literal(v, indent + 1)}`,
    );
    return `{\n${entries.join(",\n")},\n${pad}}`;
  }
  return JSON.stringify(value);
}

const header = `/**
 * Generated from contracts/core-constants.json by scripts/generate-catalog.mjs.
 * Do not edit: run \`bun run generate\` in packages/sdk-ts instead.
 *
 * Source of truth: backend/src/core/catalog.rs
 */
`;

const body = `
/** Everything the core exposes that is not a request or response shape. */
export const CORE_CATALOG = ${literal(catalog)} as const;

/** Version of the SSE event protocol these constants belong to. */
export const EVENT_PROTOCOL_VERSION = CORE_CATALOG.event_protocol_version;

/** Server-enforced input bounds. Respecting these avoids every avoidable 400. */
export const VALIDATION_LIMITS = CORE_CATALOG.validation;

/** Source types the core can ingest. */
export const SOURCE_TYPES = CORE_CATALOG.source_types;
export type SourceType = (typeof SOURCE_TYPES)[number];

/** Statuses a source can hold. \`ocr\` is an event, never a status. */
export const SOURCE_STATUSES = CORE_CATALOG.source_statuses;
export type SourceStatus = (typeof SOURCE_STATUSES)[number];

/** Teaching modes, with the presentation metadata the API returns. */
export const TEACHING_MODES = CORE_CATALOG.teaching_modes;
export type TeachingMode = (typeof TEACHING_MODES)[number]["id"];
export const DEFAULT_TEACHING_MODE = CORE_CATALOG.default_teaching_mode;

/** Providers the core can route to, and the models each pins. */
export const PROVIDERS = CORE_CATALOG.providers;
export type ProviderName = (typeof PROVIDERS)[number]["provider"];
`;

mkdirSync(dirname(target), { recursive: true });
writeFileSync(target, header + body);
console.log(`wrote ${target}`);
