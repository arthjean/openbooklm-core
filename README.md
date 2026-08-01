# OpenbookLM

Upload documents, ask questions, get answers with citations back to the source.
A complete RAG backend in Rust: ingestion, hybrid retrieval, reranking,
grounded generation and a streaming chat API, with a TypeScript SDK for the
public contract.

This repository is the OpenbookLM **core**. It is the same code the hosted
product runs: the retrieval pipeline, the chunker, the provider clients, the
REST and SSE surface and the schema are not a reduced edition. What is *not*
here is the hosted composition around it — accounts, billing, analytics and the
proprietary web UI.

Licensed under [Apache-2.0](LICENSE).

---

## What it does

| | |
|---|---|
| **Sources** | PDF (with OCR fallback for scans), DOCX, EPUB, Markdown, plain text, web pages, YouTube transcripts |
| **Retrieval** | two-pass chunking, pgvector dense search + PostgreSQL full-text search fused with Reciprocal Rank Fusion, optional reranking, optional HyDE |
| **Generation** | Anthropic, OpenAI and Mistral behind one router with circuit breaking, retry and automatic failover |
| **Citations** | every answer carries chunk-level references back to the source document |
| **Transport** | REST for everything, Server-Sent Events for chat and ingestion progress |
| **Contract** | generated OpenAPI document plus a typed TypeScript SDK, both checked against the Rust definitions in CI |

Notebooks, sources, notes, per-notebook memory and RAG logging are all core.

## Architecture

```
                      ┌──────────────────────────────────────────┐
   HTTP / SSE ───────►│  build_core_router()                     │
                      │    REST handlers, SSE transport          │
                      └───────────────┬──────────────────────────┘
                                      │  Principal (injected)
                      ┌───────────────▼──────────────────────────┐
                      │  services/                               │
                      │    rag · chat · sources · memory         │
                      └───────────────┬──────────────────────────┘
                        ┌─────────────┼─────────────┐
                        ▼             ▼             ▼
                   repositories/   clients/     seams:
                   PostgreSQL      LLM /        Principal
                   + pgvector      embeddings   EntitlementPolicy
                                   / rerank     EventSink
```

Four **seams** let one implementation serve very different deployments:

| Seam | Reference server | Hosted product |
|---|---|---|
| `Principal` | loopback single user, or a static bearer token | an identity provider mapped to an account |
| `EntitlementPolicy` | `UnrestrictedPolicy` — every valid operation is allowed | plan limits and metering |
| `EventSink` | no-op, or the tracing sink | analytics and lifecycle email consumers |
| `EmbeddingProvider` / `Reranker` | Voyage AI, or the deterministic fixtures | Voyage AI |

`EmbeddingProvider` and `Reranker` are deliberately separate: an installation
can embed without a cross-encoder, and retrieval already degrades to unreranked
results. `EmbeddingProvider::dimension` is checked against the schema's vector
width at startup, so a model of the wrong size fails to boot instead of
silently indexing chunks no query can match.

A composition root injects its adapters into `CoreState` and calls
`build_core_router`. `src/bin/openbooklm-server.rs` is a complete, supported
example — read it if you are embedding the core in your own binary.

## Quick start

Prerequisites: Rust 1.88+, PostgreSQL 14+ with `pgvector` **0.8.0 or newer**, a
[Voyage AI](https://www.voyageai.com/) key for embeddings, and at least one of
Anthropic, OpenAI or Mistral for generation.

### With Docker Compose

```bash
cp .env.example .env      # fill in VOYAGE_API_KEY and one LLM key
docker compose -f docker/docker-compose.yml up --build
curl localhost:3001/health
```

Compose brings up PostgreSQL with pgvector and the reference server, applies the
core migrations on start and exposes the API on `:3001`.

### From source

```bash
cp .env.example .env      # fill in DATABASE_URL, VOYAGE_API_KEY and one LLM key
cd backend
cargo run --bin openbooklm-server
```

### Without a provider account

`OPENBOOKLM_DETERMINISTIC_PROVIDERS=true` swaps embeddings, reranking and
generation for in-process fixtures. The whole path works, including citations,
with no key at all. Retrieval becomes token overlap and answers are quotes of
the retrieved chunk, so it is a way to see the shape of the product and to run
integration tests, not a way to serve anyone.

### First cited answer

```bash
# Loopback mode: no token needed for local requests.
NB=$(curl -sX POST localhost:3001/api/notebooks \
       -H 'content-type: application/json' \
       -d '{"title":"First notebook"}' | jq -r .id)

curl -sX POST "localhost:3001/api/notebooks/$NB/sources" \
     -H 'content-type: application/json' \
     -d '{"source_type":"text","title":"Note","content":"OpenbookLM fuses dense and lexical retrieval with RRF."}'

# Watch ingestion, then ask. `citations` events carry the source references.
curl -N -X POST "localhost:3001/api/notebooks/$NB/chat" \
     -H 'content-type: application/json' \
     -d '{"message":"How does retrieval work?"}'
```

## Authentication

The reference server has two modes and picks one at startup:

- **Loopback single user.** No token configured and the listener is bound to a
  loopback address: every local request is the operator. This is the default.
- **Static bearer token.** `OPENBOOKLM_AUTH_TOKEN` set (32 characters minimum):
  every request must carry `Authorization: Bearer <token>`.

Binding a non-loopback address without a token is **refused at startup**, not
warned about. Single-user mode authorises every request, and that is only safe
while the kernel guarantees the peer is on the same machine.

Neither mode is multi-tenant. If you need accounts, implement the `Principal`
seam against your own identity provider.

## Configuration

Every variable is documented in [`.env.example`](.env.example). These matter:

| Variable | |
|---|---|
| `DATABASE_URL` | PostgreSQL with pgvector 0.8.0+. Required. |
| `VOYAGE_API_KEY` | embeddings. Required — retrieval cannot index without it. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `MISTRAL_API_KEY` | at least one required. |
| `FIRECRAWL_API_KEY` | optional; web sources need it. |
| `OPENBOOKLM_DETERMINISTIC_PROVIDERS` | optional; replaces all of the above with fixtures. |

Startup fails, before binding a socket, when a required value is missing or
malformed. Every error is reported at once rather than one restart at a time.

**Web sources and SSRF.** The core never fetches a user-supplied URL itself.
Scraping is delegated to Firecrawl, which performs post-resolution IP
validation server-side. Without `FIRECRAWL_API_KEY`, web sources are simply
unavailable; every other source type works. Do not add direct URL fetching
without implementing that validation yourself.

## The contract

`contracts/openapi.json` is generated from the Rust request and response types,
and `packages/sdk-ts` is generated from it. Both are committed, and CI fails if
they drift from their sources.

```bash
cd backend && cargo run --bin contracts     # regenerate
./scripts/check-contracts.sh                # verify no drift
```

SSE is documented separately in `docs/contracts/event-protocol-v1.md`, because
OpenAPI does not model event ordering, replay or terminal events. The protocol
is versioned: unknown additive events must be ignored by a client, never treated
as a stream error.

## Database

PostgreSQL with pgvector 0.8.0 or newer. HNSW index on chunk embeddings, GIN
index on the full-text vector, 1024-dimension embeddings.

The version floor is filtered dense retrieval. Notebook-scoped search needs
`hnsw.iterative_scan`, which pgvector added in 0.8.0; without it a notebook
holding a small share of the corpus silently receives a fraction of its own
evidence. The server probes the extension at startup and refuses to run on an
older build. The measurement and the chosen parameters are in
[docs/architecture/filtered-ann.md](docs/architecture/filtered-ann.md).

```bash
cd backend
cargo run --bin openbooklm-migrate -- validate    # classify, apply nothing
cargo run --bin openbooklm-migrate -- up          # apply pending core migrations
```

There is no `down`, no `fresh` and no `refresh`: all three destroy data. Going
back means restoring the backup described in [docs/upgrading.md](docs/upgrading.md).
The server applies migrations itself on start, under a Postgres advisory lock,
so a rolling deploy serialises rather than races.

## Development

```bash
cd backend
cargo test                                  # full suite, under two seconds
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

`CONTRIBUTING.md` covers where changes belong, the contract and migration
requirements, and the DCO sign-off every commit needs.

## Versioning and compatibility

The public API is **pre-1.0**. Until `1.0.0`:

- The REST surface described by `contracts/openapi.json` and the SSE protocol at
  `v1` are the compatibility contract. Breaking changes to either land in a
  minor version and are listed in the release notes.
- Rust items outside `openbooklm::core` are **not** stable. `CoreState`,
  `CoreConfig`, `Principal`, `EntitlementPolicy`, `EventSink`, `ChatEvent`,
  `SourceEvent` and `build_core_router` are the intended embedding surface;
  everything else may move without notice.
- The core schema only ever gains migrations. Columns and tables are not
  removed inside a minor version.
- The Rust core and the TypeScript SDK share one version. Use the matching
  pair.

See [docs/upgrading.md](docs/upgrading.md) for supported version jumps, backup
and rollback.

## Support boundary

What is supported: the reference server on the documented stack — PostgreSQL
14+ with pgvector 0.8.0+, the named providers, Linux x86-64, Docker Compose or
a single binary.

What is not: multi-tenant hosting, other databases, other vector stores,
offline local inference, and deployment topologies that are not one process and
one database. Issues are welcome for all of it; a fix landing is not promised.

Security reports go to the process in [SECURITY.md](SECURITY.md) — please do not
open a public issue for a vulnerability.

## Relationship to the hosted product

OpenbookLM is also offered as a hosted service. That product is built by
composing this core with private adapters for identity, billing and analytics,
plus a proprietary web interface. The core is not a teaser for it: the hosted
edition adds operations, not capability.

Apache-2.0 permits commercial use, hosting included.
