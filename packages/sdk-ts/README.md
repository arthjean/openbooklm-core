# `@openbooklm/sdk`

Typed client for the OpenbookLM core API.

Everything this package exports is generated from, or aliased to, the Rust
definitions in the public core. No core type is written twice.

```bash
bun add @openbooklm/sdk
```

```ts
import { OpenbookLMClient, VALIDATION_LIMITS } from "@openbooklm/sdk";

const client = new OpenbookLMClient({
  baseUrl: "http://localhost:3001",
  token: process.env.OPENBOOKLM_TOKEN,
});

const { notebooks } = await client.listNotebooks();

for await (const event of client.sendMessage(notebooks[0].id, {
  message: "What does chapter two argue?",
  max_context_chunks: VALIDATION_LIMITS.default_max_context_chunks,
  teaching_mode: "deep",
})) {
  if (event.event === "chunk") process.stdout.write(event.data.text);
  if (event.event === "citations") console.log(event.data.citations);
}
```

## What is in here

| Export | Source |
|---|---|
| REST request and response types | `contracts/openapi.json`, from the `#[utoipa::path]` annotations |
| `OpenbookLMClient` | one method per core route, typed by the above |
| `ChatEvent`, `SourceEvent` and their parsers | the `ChatEvent`/`SourceEvent` Rust enums |
| `VALIDATION_LIMITS`, `SOURCE_TYPES`, `SOURCE_STATUSES`, `TEACHING_MODES`, `PROVIDERS` | `contracts/core-constants.json`, from `backend/src/core/catalog.rs` |
| `ProblemDetails`, `ApiError` | the RFC 7807 responses the core returns |

## What is deliberately not in here

Billing plans, prices, identity-provider claims, analytics events and
proprietary UI types belong to the hosted product, not the core. The contract
test asserts their absence rather than trusting the author.

## Streams

`ChatEvent` and `SourceEvent` are discriminated unions keyed on `event`. What a
schema cannot express — ordering, terminal events, replay, cancellation — is
specified in [`docs/contracts/sse-protocol-v1.md`](../../docs/contracts/sse-protocol-v1.md).

Two rules matter to every consumer:

- **Ignore unknown events.** `parseChatEvent` and `parseSourceEvent` return
  `null` for an event name they do not know. Keep reading; additive variants are
  not a breaking change.
- **One terminal event.** `done`, `error` or `shutdown` ends a chat stream, and
  nothing follows it. `follow_up_suggestions` always arrives before `done`.

## Regenerating

```bash
cd backend && cargo run --bin contracts   # contracts/*.json from Rust
cd packages/sdk-ts && bun run generate    # src/generated/* from contracts/
```

`./scripts/check-contracts.sh` at the repository root verifies both steps are
byte-reproducible and that the SDK still agrees with the golden wire fixtures.

## Versioning

The SDK version matches the core release it was generated from. Mixing versions
is unsupported: `EVENT_PROTOCOL_VERSION` and the OpenAPI `info.version` are the
two values to compare when diagnosing a mismatch.

`./scripts/check-openapi-compat.py <baseline> [candidate]` rejects a contract
change that would break an existing client unless the release is declared
breaking.

## License

Apache-2.0.
