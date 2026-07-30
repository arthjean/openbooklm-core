# OpenbookLM 0.0.1

The first public release of the OpenbookLM core: the RAG, ingestion, retrieval,
citation and multi-LLM chat implementation that runs the hosted product, with a
reference server you can operate yourself.

## What you get

- **Ingestion** for PDF (with OCR fallback for scanned pages), DOCX, EPUB,
  Markdown, plain text, web pages and YouTube transcripts.
- **Retrieval**: two-pass chunking, pgvector dense search and PostgreSQL
  full-text search fused with Reciprocal Rank Fusion, optional reranking,
  optional HyDE.
- **Generation** through Anthropic, OpenAI and Mistral behind one router with
  circuit breaking, retry and failover.
- **Citations** on every answer, resolved to chunk-level source references.
- **A reference server** (`openbooklm-server`) with health endpoints, request
  IDs, security headers, rate limiting, graceful shutdown and two identity
  modes: loopback single-user and static bearer token.
- **A generated contract**: `contracts/openapi.json` and `@openbooklm/sdk`, both
  produced from the Rust definitions and checked against them in CI.
- **The core schema** as a single baseline migration, with startup validation
  and a Postgres advisory lock around execution.

## What is compatibility-stable

The public API is **pre-1.0**. Read this before you build on it.

**Stable, and changed only in a minor version with release notes:**

- Every REST path, request body and response body in `contracts/openapi.json`.
- The RFC 7807 error shape: `type`, `title`, `status`, `detail`.
- SSE event names, their payloads, their ordering rules and their terminal
  semantics, as documented in `docs/contracts/event-protocol-v1.md`. `done` is
  the terminal successful chat event; a typed `error` event is never followed by
  `done`.
- These Rust items, which are the intended embedding surface: `CoreState`,
  `CoreConfig`, `Principal`, `EntitlementPolicy`, `EventSink`, `ChatEvent`,
  `SourceEvent`, `build_core_router`, `build_core_health_router`.
- The core schema, additively. No column or table is removed in a minor version.
- Environment variable names, with one minor version of deprecation before a
  removal becomes an error.

**Not stable. Expect it to move without notice:**

- Every Rust item outside `openbooklm::core` — repositories, services, clients,
  entities, the chunker, the retrieval internals. They are `pub` because the
  reference server and the tests need them, not as a promise.
- Chunk boundaries, retrieval scores, rank ordering and prompt text. Retrieval
  quality work will change what comes back for a given query. Do not assert on
  exact chunk text or score values.
- Log lines, metric names and the `/health/detailed` body shape.
- The internal layout of `packages/sdk-ts` beyond its documented exports.

**Explicitly not part of this release:**

- Multi-tenancy. The reference server maps every request to one account. If you
  need real accounts, implement the `Principal` seam.
- A web interface. The hosted product's UI is proprietary and is not published.
- Offline local inference. Embeddings and generation call hosted providers.
- Any database other than PostgreSQL with pgvector.

## Additive changes and older clients

A newer core may emit an SSE event name an older client does not know. That is
an **additive** change, not a breaking one: a client must ignore the event and
keep the stream open. The shipped SDK does this. If you wrote your own parser,
make sure it does too — otherwise the next minor release will look like an
outage.

## Requirements

- Rust 1.88+ (Edition 2024) to build from source.
- PostgreSQL 14+ with the `pgvector` extension.
- A Voyage AI key for embeddings, and at least one of Anthropic, OpenAI or
  Mistral for generation.
- Optionally a Firecrawl key for web sources. Without it, web sources are
  unavailable and every other source type works.

No Clerk, Stripe, PostHog or Resend value is read, required or accepted.

## Getting started

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
curl localhost:3001/health
```

The README walks through creating a notebook, ingesting a source and getting a
cited answer.

## Known limitations

- Retrieval quality is tuned for English and French. Other languages work and
  are less well measured.
- OCR requires a Mistral key and is off by default; a scanned PDF ingested
  without it produces an empty document rather than an error at upload time.
- The rate limiter is in-memory unless you configure Upstash Redis, so it is
  per-process rather than per-deployment.
- Contextual retrieval (LLM-generated chunk prefixes) is implemented but
  disabled: it multiplies ingestion cost and has not been measured against the
  current chunker.

## Security

Report vulnerabilities through `SECURITY.md`, not a public issue. The container
carries an SPDX SBOM, and the CycloneDX source SBOM is attached to the release.
Every published artifact has a GitHub artifact attestation. An attestation proves
what built the artifact — it is not a security audit and does not claim to be.
