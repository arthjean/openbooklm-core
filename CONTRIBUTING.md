# Contributing to OpenbookLM

OpenbookLM's core is published under the Apache License 2.0. This document defines
how contributions are accepted, what the compatibility promises are, and which
mechanical gates a change must pass.

The hosted SaaS built on top of this core is a separate, private product. Nothing
in this repository requires a Clerk, Stripe, Resend or PostHog account.

## Developer Certificate of Origin

OpenbookLM uses the [Developer Certificate of Origin 1.1](https://developercertificate.org/),
not a Contributor License Agreement. Every commit must carry a `Signed-off-by`
trailer matching the author identity:

```
Signed-off-by: Jane Doe <jane@example.com>
```

`git commit -s` adds it. To sign off a branch you already wrote:

```bash
git rebase --signoff main
```

Signing off means you certify the DCO terms: you wrote the contribution, or you
have the right to submit it under Apache-2.0, and you understand the contribution
and its record are public and permanent.

Pull requests without a sign-off on every commit are not merged.

## Licensing of contributions

By signing off you license your contribution under Apache-2.0, including the
patent grant in section 3. Do not paste code whose license you cannot identify.
New third-party assets (fonts, icons, fixtures, datasets) must state their license
in the pull request description; unclear provenance blocks the publication gate.

## Scope: what belongs here

**In scope for the public core**

- RAG: chunking, embedding, hybrid retrieval, reranking, citation extraction
- Document processing: PDF, web, text, Markdown, DOCX, EPUB, transcript ingestion
- LLM provider integrations and resilience (retry, circuit breaker, routing)
- Core persistence: notebooks, sources, chunks, notes, chat messages, memories
- Core REST and SSE handlers, request validation, error contracts
- Core migrations, generated contracts, the TypeScript SDK, self-host assets
- The reference server (`backend/src/bin/openbooklm-server.rs`), its identity
  modes and the Docker assets that run it

**Out of scope for the public core**

- Identity vendors, subscription plans, prices, metering and quota policy
- Analytics and lifecycle email providers
- The proprietary Next.js application and its UI behavior
- Hosted deployment topology and commercial operations

Commercial behavior is injected by adapters. If a change needs a plan, a price,
a Clerk claim or an analytics event, it belongs behind an interface rather than
inside a core module. `docs/open-core-boundary.md` is authoritative and
`scripts/check-open-core-boundary.sh` enforces it.

## High-value contribution paths

| Path | Where | What makes it land |
|---|---|---|
| New LLM provider | `backend/src/clients/`, `backend/src/llm/` | Implements `LlmProvider`, uses `ResilientExecutor`, adds model metadata, ships fake-provider tests |
| New source format | `backend/src/services/processor.rs`, `backend/src/services/rag/chunking.rs` | Deterministic extraction, a chunking profile, golden fixtures |
| Retrieval quality | `backend/src/services/rag/search/` | A measurable claim plus a regression test; scoring changes need a baseline diff |
| Reliability | `backend/src/clients/`, `backend/src/middleware/` | A reproducing test for the failure being fixed |
| Ingestion robustness | `backend/src/services/source_processing.rs` | Degradation is observable through the event sink, never silent |
| Self-hosting DX | `docker/`, `docs/self-hosting/`, `backend/src/bin/openbooklm-server.rs` | The Compose smoke path still reaches a cited answer from a clean checkout |
| SDK ergonomics | `packages/sdk-ts/` | Generated types stay generated; only hand-written wrappers change |

## Contract requirements

Core REST and SSE surfaces are contracts, not implementation details.

- REST DTOs and RFC 7807 problem shapes are generated into `contracts/`. Changing
  a public type means regenerating the artifacts in the same commit.
- SSE event names, payload shapes, ordering and terminal semantics are versioned
  separately. `done` is the terminal successful chat event. Additive events must
  be ignorable by older clients without terminating the stream.
- Golden fixtures in `backend/tests/fixtures/` are the executable baseline.
  A fixture change is a contract change and must be described in the pull request.
- Removing a field, adding a required field or changing a type incompatibly is a
  breaking change and needs an explicit major/pre-1.0 release decision.

## Migration requirements

- Applied migration files are immutable. Never edit, rename or reorder one.
- Schema evolution is additive: expand, backfill, then contract in a later release.
- Column and table removals are a separate, later change after at least one
  compatible release window.
- A migration must be safe to run twice. It must also be safe while an older
  binary serves traffic unless its release notes declare a coordinated
  stop-first maintenance window and backup-only rollback. Never describe such a
  migration as rolling-compatible.

## Quality gates

A pull request must pass, from `backend/`:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
cargo deny check licenses bans advisories
```

and, from the repository root:

```bash
./scripts/check-open-core-boundary.sh
./scripts/check-contracts.sh
python3 scripts/check-public-manifest.py
python3 scripts/scan-git-history-secrets.py --working-tree
```

If you work in this repository rather than the public one, also run the core
edition on its own:

```bash
cd backend && cargo check --no-default-features --all-targets
```

The `saas` feature is what separates the two editions inside one crate. A core
change that only compiles with the feature on has acquired a hosted dependency,
and this command is where that shows up — not in the public repository's CI.

Tests must stay fast. The default `cargo test` suite runs offline in under two
seconds: no real network, no commercial key, no sleep-based waiting. Anything
that needs PostgreSQL is `#[ignore]` and runs with `TEST_DATABASE_URL` set.

All fixtures are synthetic. Never commit production-derived data, real account
identifiers or real email addresses.

## Security

Do not open a public issue for a vulnerability. Follow `SECURITY.md`.

## Conduct

Participation is governed by `CODE_OF_CONDUCT.md`.

## Trademarks

The code is Apache-2.0; the name and marks are not. See `TRADEMARK.md`.
