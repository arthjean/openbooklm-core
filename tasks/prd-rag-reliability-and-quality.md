[PRD]
# PRD: RAG Reliability and Quality

## Changelog

| Version | Date | Author | Summary |
|---------|------|--------|---------|
| 1.0 | 2026-07-31 | OpenbookLM Core | Initial draft based on the RAG architecture audit and primary-source research |

## Problem Statement

1. Reprocessing a source deletes its active chunks before the replacement is complete. Batch commits are visible independently, so retrieval can observe an empty or partial corpus after a provider, database, timeout, or process failure.
2. Concurrent reprocessing has no source-level ownership invariant. Multiple workers can interleave deletion and insertion, while the schema does not enforce uniqueness for a chunk position within an index generation.
3. Ingestion starts detached async work. A timeout drops the parent future without guaranteeing that embedding calls stop, and errors are collected after outstanding work rather than cancelling it immediately.
4. Retrieval decisions compare synthetic stuffing scores, RRF ranks, and provider-defined reranker scores against one threshold. Context limits, preference boosts, deduplication, and corrective retrieval therefore behave differently across configuration branches.
5. The prompt budget does not bound all retrieved content, especially stuffing and provider-native document blocks. PDF/OCR page metadata is partly heuristic, so a syntactically valid citation can point to the wrong page.
6. The project has deterministic unit and contract tests, but no versioned relevance dataset or release gate for retrieval recall, grounded answers, citation support, abstention, filtered ANN recall, or concurrent indexing integrity.

**Why now:** OpenbookLM Core publicly presents ingestion, hybrid retrieval, reranking, and citations as its central product capability. The current implementation has correctness defects that can silently expose partial indexes or ungrounded output. Tuning chunk sizes, HyDE, contextual retrieval, or ANN parameters before adding lifecycle invariants and an evaluation baseline would make regressions harder to detect and rollback.

## Overview

This initiative hardens the existing Rust/PostgreSQL RAG without replacing its provider and repository boundaries. It introduces immutable index generations that are validated before an atomic publication step, explicit task ownership and cancellation, versioned embedding/chunking provenance, and one active generation per source. Existing active data remains searchable until its replacement is complete.

Retrieval becomes a contract rather than a collection of branch-specific heuristics. Score domains remain distinct, the final result limit and token budget apply on every path, reranking operates on a diversified candidate pool, and filtered ANN behavior is selected from an exact-vs-approximate benchmark. Structured source spans and an explicit untrusted-content boundary improve citation correctness and resistance to indirect prompt injection.

A versioned synthetic evaluation corpus measures retrieval and answer behavior separately. Release reports cover recall, ranking, duplicate rate, context cardinality, grounded citation precision and coverage, abstention, latency, tenant isolation, and failure handling. HyDE, contextual retrieval, and chunk-size tuning remain disabled or unchanged until an ablation proves a gain on the holdout set.

## Goals

| Goal | Month-1 Target | Month-6 Target |
|------|---------------|----------------|
| Active-index integrity | 0 mixed or partial active generations in 1,000 deterministic failure/concurrency schedules | 0 integrity incidents and 100% generation publication audit coverage |
| Evaluation coverage | At least 40 labeled queries across 8 categories, with at least 5 cases per category and a locked holdout split | At least 150 labeled queries with no category below 10 cases |
| Retrieval quality | Baseline published for Recall@5/10/20, MRR, nDCG@10, duplicate rate, and filtered top-k fill | Recall@10 >= 0.90, nDCG@10 >= 0.75, and no category regresses by more than 0.02 versus its approved baseline |
| Grounded response quality | Baseline published for citation precision, citation coverage, and abstention accuracy | Citation precision >= 0.95, citation coverage >= 0.90, and abstention accuracy >= 0.90 |
| Context contract | 100% of deterministic tests respect requested chunk limits and provider context windows | 0 observed context-limit violations across supported providers |
| Cancellation | 0 new provider calls start more than 1 second after cancellation; tracked async work drains within 5 seconds | Same targets under continuous failure-injection tests |

## Target Users

### OpenbookLM Core integrators and operators

- **Role:** Engineers embedding the standalone core in a product or operating its source-ingestion pipeline.
- **Behaviors:** Configure providers, ingest and reprocess sources, monitor source events, run migrations, and diagnose retrieval quality through logs and tests.
- **Pain points:** A source can appear ready while its searchable data is partial; retries can duplicate work; scores and metrics do not have stable semantics; provider costs may continue after timeout.
- **Current workaround:** Retry reprocessing, inspect rows manually, tune constants by intuition, and infer quality from individual chat responses.
- **Success looks like:** Every source has one auditable active generation, failures preserve the previous generation, and a deterministic report identifies regressions before release.

### Notebook users

- **Role:** People asking questions over their own notebook sources.
- **Behaviors:** Upload PDF, OCR, text, or web content; ask direct and follow-up questions; open citations to verify answers.
- **Pain points:** Relevant passages can be crowded out by duplicates, retrieval failure can look like missing sources, and page citations can be inaccurate.
- **Current workaround:** Rephrase questions, reopen source documents, or manually verify claims without knowing whether retrieval failed.
- **Success looks like:** Answers either cite the exact supporting passage or explicitly abstain, and source reprocessing never exposes mixed old/new content.

## Research Findings

Key findings that informed this PRD:

### Competitive Context

- [OpenAI Vector Stores](https://developers.openai.com/api/reference/resources/vector_stores) expose asynchronous readiness, metadata filters, configurable chunking, score thresholds, query rewriting, reranking, and hybrid ranking controls.
- [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/retrieval-and-ranking) separates retrieval from reranking, while [Vertex grounding checks](https://docs.cloud.google.com/generative-ai-app-builder/docs/check-grounding?hl=en) evaluate claim support rather than citation presence alone.
- [Microsoft Foundry evaluators](https://learn.microsoft.com/en-in/azure/foundry/concepts/evaluation-evaluators/rag-evaluators?view=foundry) report retrieval, response quality, and groundedness as separate dimensions.
- **Market gap:** The core can differentiate through transparent, offline, provider-neutral evaluation and atomic corpus lifecycle rather than proprietary orchestration.

### Best Practices Applied

- Maintain a versioned dataset with queries, relevance judgments, expected claims, and unanswerable cases; measure retrieval independently from generation.
- Retrieve lexical and dense pools independently, fuse by rank, rerank a bounded diversified pool, and enforce the final context budget after ranking.
- Publish an immutable corpus generation only after validation; retain the previous generation for failure safety and deliberate rollback.
- Compare filtered ANN against exact search. The [pgvector project](https://github.com/pgvector/pgvector) documents that post-filtering can underfill approximate results and recommends iterative scans or data-layout alternatives.
- Treat all retrieved documents as untrusted data. [OWASP LLM guidance](https://genai.owasp.org/download/43299/) identifies poisoned RAG content as an indirect prompt-injection path.
- Use additive PostgreSQL migrations. [PostgreSQL index documentation](https://www.postgresql.org/docs/current/sql-createindex.html) defines the operational constraints of concurrent index construction and invalid-index recovery.

## Assumptions & Constraints

### Assumptions (to validate)

- An additive generation table plus an active-generation pointer can be introduced without changing the public REST or SSE payload shape. US-005 validates the transaction and rollout model before migration code is written.
- The deployed pgvector version can support a filtered ANN strategy meeting Recall@10 and latency targets. US-015 compares available modes against exact search before US-016 changes production queries.
- A synthetic stratified corpus predicts the failure modes relevant to public core users. A locked holdout split and category-level reporting limit overfitting, but real deployment data is not available in this repository.
- Provider reranker scores are not mutually calibrated. No threshold may be shared between providers unless a provider-specific calibration artifact proves equivalence.
- Existing provider and repository traits are sufficient. No new framework or external vector service is required for the first release.

### Hard Constraints

- The repository must remain deployable as the standalone OpenbookLM core, without hosted identity, billing, analytics, or proprietary UI dependencies.
- Migrations are additive. The applied baseline migration must not be edited, renamed, or reordered.
- Tests are offline and deterministic by default. Deliberate PostgreSQL tests use synthetic data through `TEST_DATABASE_URL` and remain ignored unless explicitly invoked.
- Public contracts and generated artifacts are changed only from their authoritative Rust sources and regenerated in the same story.
- Existing embedding vectors remain 1,024 dimensions until a separately approved compatibility change introduces another dimension.
- No production-derived content, account identifier, email address, or credential may enter fixtures, reports, or logs.

## Quality Gates

These commands must pass for every user story:

- `cd backend && git ls-files '*.rs' | grep -v '^src/lib\.rs$' | xargs rustfmt --check --edition 2024` - verify formatting with the repository's file-level CI rule.
- `cd backend && cargo check --no-default-features --all-targets` - compile the standalone core and all targets.
- `cd backend && cargo clippy --no-default-features --all-targets -- -D warnings` - reject Rust warnings.
- `cd backend && cargo test --no-default-features` - run deterministic offline tests.
- `cd backend && cargo deny check licenses bans advisories` - verify dependency policy.
- `cd backend && TEST_DATABASE_URL=postgres://openbooklm:openbooklm@localhost:5432/openbooklm cargo test --no-default-features --test rag_integration -- --ignored` - additionally required for stories that change migrations, repositories, generation publication, or ANN behavior.

## Epics & User Stories

### EP-001: Evaluation Contract and Observability

Create the evidence layer needed to distinguish retrieval, ranking, generation, citation, and operational regressions. Tuning stories cannot claim improvement without this epic.

**Definition of Done:** A versioned corpus of at least 40 cases produces deterministic machine-readable retrieval and response reports, with category-level baselines and enforceable thresholds.

#### US-001: Define the versioned RAG evaluation corpus

**Description:** As a core maintainer, I want a synthetic and versioned relevance corpus so that retrieval changes are evaluated against explicit expectations rather than anecdotal prompts.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** None

**Acceptance Criteria:**

- [ ] The fixture schema requires a unique query ID, notebook/source fixtures, query category, answerability, relevant source and chunk/span IDs, expected claims, and train or holdout membership.
- [ ] The initial corpus contains at least 40 queries with at least 5 cases in each category: exact identifiers, semantic paraphrases, multi-hop, conflicting sources, unanswerable, tables/code, long documents, and hostile retrieved instructions.
- [ ] Every fixture and identifier is synthetic, and a repository check rejects production-looking account IDs, email addresses, credentials, duplicate query IDs, and references to absent chunks.
- [ ] Given a malformed or incomplete annotation, the validator exits non-zero and reports the exact query and field instead of skipping the case.
- [ ] Corpus version and change rationale are stored with the fixtures, and holdout labels are not consumed by tuning code.

#### US-002: Implement deterministic retrieval evaluation

**Description:** As a retrieval engineer, I want one offline runner for lexical, dense, hybrid, and exact-reference modes so that ranking changes can be compared reproducibly.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** Blocked by US-001

**Acceptance Criteria:**

- [ ] The runner reports Recall@5, Recall@10, Recall@20, MRR, nDCG@10, source recall, duplicate-parent rate, requested top-k fill rate, and latency percentiles overall and per query category.
- [ ] Dense, lexical, hybrid/RRF, and exact vector reference modes can be selected independently without changing fixtures.
- [ ] Repeated runs with the same seed, configuration, and revision produce byte-stable JSON except for an explicitly excluded timestamp field.
- [ ] The default CI mode uses deterministic local providers and makes zero network requests.
- [ ] Given an empty result set, missing relevance judgments, NaN score, or fewer results than requested, the report records an explicit failure reason and never emits NaN or silently drops the query.

#### US-003: Implement grounded-response evaluation

**Description:** As a product maintainer, I want answer behavior evaluated separately from retrieval so that fluent but unsupported responses cannot hide retrieval or grounding failures.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** Blocked by US-001

**Acceptance Criteria:**

- [ ] The evaluator reports expected-claim coverage, citation precision, citation coverage, abstention accuracy, and unsupported-claim count overall and per category.
- [ ] Deterministic CI evaluation uses labeled claims and source spans; an optional LLM judge may add diagnostics but cannot determine pass/fail alone.
- [ ] A citation counts as correct only when its active-generation source span supports the associated claim, not merely when its marker and source ID are valid.
- [ ] Given an unanswerable query or insufficient retrieved evidence, a non-abstaining factual answer is recorded as a failure.
- [ ] Given a provider error or malformed streamed citation, the case remains in the denominator and receives an explicit error classification.

#### US-004: Publish baselines, telemetry, and release gates

**Description:** As an operator, I want retrieval traces and comparable release reports so that regressions can be diagnosed without logging private source content.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** Blocked by US-002, US-003

**Acceptance Criteria:**

- [ ] Each retrieval trace records generation IDs, original and reformulated query hashes, retrieval mode, score domain, candidate counts by stage, final unique-parent count, token counts, reason codes, and stage durations without logging source text or raw user queries.
- [ ] A baseline artifact records corpus version, code revision, provider fingerprints, chunking version, retrieval configuration, aggregate metrics, and category metrics.
- [ ] The comparison command exits non-zero when Recall@10 or nDCG@10 regresses by more than 0.02, citation precision or coverage regresses by more than 0.02, any tenant-isolation case fails, or any required metric is missing.
- [ ] The initial baseline can be captured even when a future absolute target is not met; enforcement mode clearly separates regression failures from unmet target failures.
- [ ] Given an unavailable metrics repository, chat behavior remains functional and emits one structured telemetry failure without retry loops or source content leakage.

---

### EP-002: Atomic, Idempotent, and Cancellable Indexing

Make a source index an immutable generation with one atomic publication point. Concurrent retries and interrupted provider work must preserve the previously active generation.

**Definition of Done:** Across 1,000 deterministic concurrent, timeout, and injected-failure schedules, search observes either the complete previous generation or the complete replacement, never a mixture or partial replacement.

#### US-005: Validate the index-generation publication model

**Description:** As a database maintainer, I want an executable design proof for generation publication so that the migration is based on verified PostgreSQL transaction semantics.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** None

**Acceptance Criteria:**

- [ ] The design defines immutable generation identity, source ownership, lifecycle states, embedding fingerprint, chunking fingerprint, expected and stored chunk counts, and one nullable active-generation pointer per source.
- [ ] An executable PostgreSQL test proves that readers observe the old generation before commit and the complete new generation after one publication transaction.
- [ ] The design defines backfill, forward deployment, rollback, obsolete-generation retention, and cleanup ordering without requiring an edit to the baseline migration.
- [ ] Given a forced failure immediately before or during publication, the test proves that the old active generation and source readiness remain unchanged.
- [ ] The design is recorded in a repository architecture document with the exact invariants consumed by US-006 through US-011.

#### US-006: Add the generation schema and backfill

**Description:** As an operator, I want existing chunks represented as a valid generation so that upgrades preserve all currently searchable data.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** Blocked by US-005

**Acceptance Criteria:**

- [ ] A new timestamped additive migration creates the approved generation model, adds generation identity to chunks, and adds source active-generation ownership without modifying prior migrations.
- [ ] Existing chunks are backfilled into exactly one published generation per source, and each source points to that generation after migration.
- [ ] The database enforces uniqueness for source generation identity and chunk index within a generation, plus foreign keys that prevent cross-source publication.
- [ ] Applying the migration twice is a no-op, and schema validation passes on a fresh database and an upgraded synthetic database.
- [ ] Given orphaned, duplicate, or dimensionally invalid legacy chunks, migration validation aborts with a source-specific diagnostic rather than publishing an ambiguous generation.

#### US-007: Write and validate immutable generations

**Description:** As an operator, I want reprocessing to build a replacement beside the active index so that provider or storage failure cannot destroy searchable content.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** Blocked by US-006

**Acceptance Criteria:**

- [ ] Reprocessing creates one building generation and writes every batch under that generation without deleting or mutating chunks in the active generation.
- [ ] Batch retries are idempotent under the generation/chunk uniqueness constraint and never create duplicate chunk positions.
- [ ] Before publication, validation proves stored chunk count equals expected count and every embedding has the configured dimension and only finite values.
- [ ] Given extraction, embedding, channel, storage, validation, or timeout failure, the building generation becomes failed and the prior active generation remains searchable.
- [ ] Given content producing zero valid chunks, the generation is rejected with a stable error reason and is never made active.

#### US-008: Publish, read, rollback, and reclaim generations

**Description:** As a notebook user, I want source replacements to appear atomically so that every search uses one complete source version.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** Blocked by US-007

**Acceptance Criteria:**

- [ ] One database transaction validates ownership and counts, marks the replacement published, updates the source active-generation pointer, and updates source readiness.
- [ ] Dense search, lexical search, count queries, stuffing, chunk listing used by RAG, and citation resolution include only chunks belonging to each source's active generation.
- [ ] Concurrent readers in a publication stress test observe only complete old or complete new generations, with zero mixed-generation result sets across 1,000 schedules.
- [ ] A rollback operation can repoint a source to its immediately previous complete generation without copying chunks or changing public response shapes.
- [ ] Given cleanup failure, all active and rollback-eligible generations remain intact; cleanup never deletes a generation referenced by a source.

#### US-009: Make source reprocessing single-owner and retry-safe

**Description:** As an operator, I want duplicate reprocess requests coalesced so that retries cannot start competing workers for one source.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** Blocked by US-008

**Acceptance Criteria:**

- [ ] A compare-and-set ownership transition allows at most one building generation per source and associates every status update with the owning generation ID.
- [ ] Concurrent duplicate requests preserve the existing API response shape, start one worker, and return the current source processing state for all other callers.
- [ ] A 100-request concurrency test produces one published replacement, one active pointer, unique chunk positions, and no last-writer status corruption.
- [ ] A worker cannot mark a source ready or failed after ownership has moved to a different generation.
- [ ] Given a process restart with a building generation older than twice the configured processing timeout, recovery marks it failed without changing the active generation and permits one new owner.

#### US-010: Own, cancel, and drain ingestion work

**Description:** As an operator, I want ingestion work to stop predictably after error, timeout, or shutdown so that provider cost and database activity do not continue in the background.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** None

**Acceptance Criteria:**

- [ ] Every async producer and consumer task is owned by the source-processing lifecycle; no ingestion `JoinHandle` is intentionally detached.
- [ ] The first terminal extraction, embedding, storage, or channel error cancels admission of new work and propagates a cooperative cancellation signal to all owned async tasks.
- [ ] With a deterministic fake provider, zero new provider calls start more than 1 second after cancellation and all cancellable async work drains within 5 seconds before remaining async handles are aborted.
- [ ] Timeout and shutdown mark the building generation failed, preserve the active generation, and emit exactly one terminal source event.
- [ ] Given non-abortable blocking work already in progress, shutdown waits only for the documented drain deadline, records the unfinished operation, and does not claim it was cancelled.

#### US-011: Version embeddings, caches, and chunking semantics

**Description:** As a maintainer, I want every index and cache entry tied to its semantic configuration so that model or chunking changes cannot silently reuse incompatible embeddings.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** Blocked by US-008

**Acceptance Criteria:**

- [ ] Each generation stores provider, model, embedding dimension, normalization mode, chunking schema version, size unit, tokenizer/sizer identity, and relevant configuration fingerprint.
- [ ] Ingestion reuse keys include content hash and embedding fingerprint; a changed fingerprint builds a new generation and performs no incompatible reuse.
- [ ] Query embedding cache keys distinguish provider/model fingerprint, direct query, reformulated query, HyDE document, and working-memory lookup modes.
- [ ] Chunk-size constants and tests measure the declared unit; a token-sized configuration names its tokenizer and asserts token counts rather than character counts.
- [ ] Given absent, malformed, or mismatched provenance, indexing fails before publication and search never combines generations with incompatible vector dimensions or semantics.

---

### EP-003: Coherent and Bounded Retrieval

Give every retrieval branch the same output contract while preserving distinct score semantics and measuring filtered ANN behavior before tuning it.

**Definition of Done:** For every supported configuration branch, retrieval returns at most the requested number of unique parent contexts, never compares incompatible scores, and meets the approved exact-vs-ANN recall and latency profile.

#### US-012: Separate ranking scores from retrieval confidence

**Description:** As a maintainer, I want score domains represented explicitly so that corrective retrieval and metrics cannot compare unrelated values.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** Blocked by US-002

**Acceptance Criteria:**

- [ ] RRF rank score, dense distance/similarity, lexical rank, reranker relevance, and retrieval-confidence reason are distinct typed fields or variants.
- [ ] No average, threshold, log field, or corrective decision combines values from different score domains.
- [ ] Without a calibrated reranker, corrective retrieval triggers only on deterministic insufficiency reasons such as zero results or underfilled required evidence, never on an RRF score threshold.
- [ ] Provider-specific relevance thresholds require an explicit provider/model calibration fingerprint and evaluation artifact.
- [ ] Given NaN, infinity, an unknown score domain, or a missing required calibration, the candidate is rejected or the optional decision is disabled with a recorded reason rather than coerced to zero.

#### US-013: Enforce one final context cardinality contract

**Description:** As an API consumer, I want `max_context_chunks` respected in every configuration so that cost and response behavior remain predictable.

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** Blocked by US-002

**Acceptance Criteria:**

- [ ] Stuffing, no-reranker, reranker success, reranker failure, deduplication, and corrective-retrieval branches all return between zero and the validated requested maximum.
- [ ] Omitting a reranker never returns the complete candidate pool when it exceeds the final maximum.
- [ ] Result truncation is deterministic for equal ranks through a documented stable tie-breaker.
- [ ] The pipeline never reintroduces duplicate parent contexts merely to reach a minimum result count.
- [ ] Given a zero, negative, or above-contract limit at the API boundary, validation rejects it before retrieval with the existing structured error shape.

#### US-014: Reorder diversification, reranking, preferences, and selection

**Description:** As a notebook user, I want distinct evidence to survive ranking so that overlapping child chunks do not crowd out relevant sources.

**Priority:** P1
**Size:** M (3 pts)
**Dependencies:** Blocked by US-013

**Acceptance Criteria:**

- [ ] Children resolving to the same canonical parent are collapsed before final selection, retaining the strongest representative and all matched-child provenance needed for diagnostics.
- [ ] The reranker receives the full diversified candidate pool up to the configured pool size, not only the final context limit.
- [ ] The final limit is applied after reranking; presentation-only sandwich ordering occurs after selection and cannot change membership.
- [ ] Source/topic preferences remain an explicit secondary ordering key after reranking and are never implemented by overwriting provider relevance scores.
- [ ] Given a pool dominated by one parent, retrieval returns fewer unique parents rather than restoring duplicate children, and the report records the shortfall reason.

#### US-015: Validate filtered HNSW against exact search

**Description:** As a database maintainer, I want measured ANN recall under notebook filters so that the production search strategy is chosen from evidence rather than a fixed `ef_search` guess.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** Blocked by US-002

**Acceptance Criteria:**

- [ ] A reproducible PostgreSQL/pgvector benchmark seeds at least 100,000 synthetic chunks and filter selectivities of 100%, 10%, 1%, and 0.1%.
- [ ] The benchmark compares exact vector search, the current HNSW query, and each iterative-scan mode supported by the deployed extension version.
- [ ] It reports Recall@10, requested top-k fill rate, P50/P95 latency, tuples scanned, query plan, and configuration for every selectivity.
- [ ] The recommendation selects the lowest-cost strategy achieving Recall@10 >= 0.95, fill rate >= 0.99, and P95 <= 300 ms on the documented reference environment.
- [ ] Given an unsupported extension feature or no strategy meeting all thresholds, the report exits non-zero, records the blocking dimension, and blocks US-016 rather than silently selecting a fallback.

#### US-016: Implement the approved filtered ANN strategy

**Description:** As a notebook user, I want dense retrieval to preserve recall under selective notebook filters so that corpus growth in other notebooks does not hide my evidence.

**Priority:** P1
**Size:** M (3 pts)
**Dependencies:** Blocked by US-015

**Acceptance Criteria:**

- [ ] The search repository implements exactly the strategy and bounded parameters approved by US-015 and documents the required minimum pgvector version.
- [ ] Per-query settings are scoped to the retrieval transaction and cannot leak into pooled connections.
- [ ] Dense results preserve deterministic distance ordering required by the API and fusion layer.
- [ ] CI runs a reduced deterministic recall test, while the full 100,000-row benchmark remains an explicit ignored performance test.
- [ ] Given an unsupported pgvector version, invalid scan setting, or underfilled result set, startup or retrieval returns an actionable diagnostic and telemetry reason rather than silently reporting a successful full top-k.

#### US-017: Bound reformulation and isolate query embeddings

**Description:** As an operator, I want reformulation cost and caches bounded by explicit semantics so that short follow-ups cannot send unbounded history or contaminate later searches.

**Priority:** P1
**Size:** M (3 pts)
**Dependencies:** Blocked by US-012

**Acceptance Criteria:**

- [ ] Reformulation consumes at most the latest 5 eligible messages and at most 20% of the provider input-token budget, whichever limit is reached first.
- [ ] Token fitting occurs before the reformulation provider call and preserves the current user query even when all history is dropped.
- [ ] Direct-query, reformulated-query, HyDE-document, and working-memory embeddings use distinct cache namespaces and generation/model fingerprints.
- [ ] The retrieval trace records whether reformulation was attempted, skipped, truncated, succeeded, or fell back to the original query.
- [ ] Given reformulation timeout, provider rejection, or an over-budget history, retrieval continues once with the original query and performs no recursive reformulation loop.

---

### EP-004: Bounded Context, Trustworthy Citations, and Safe Failure

Ensure that selected evidence fits every provider contract, citations resolve to stable source spans, and retrieved content cannot silently become instructions or conceal infrastructure failure.

**Definition of Done:** All supported chat providers remain within their declared context windows, citation precision and coverage meet targets on the holdout set, and 50 hostile-content fixtures cause zero instruction-following or cross-notebook leakage failures.

#### US-018: Enforce one provider-aware context budget

**Description:** As an API consumer, I want all prompt components counted before generation so that stuffing and native citation blocks cannot overflow a provider context window.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** Blocked by US-013, US-014

**Acceptance Criteria:**

- [ ] One budgeting pass counts system instructions, retrieved text, provider-native document blocks, history, current query, requested output tokens, and a reserve equal to the larger of 1,024 tokens or 5% of the provider context window.
- [ ] Stuffing is used only when all unique active-generation contexts fit both the requested chunk maximum and the token budget.
- [ ] When evidence exceeds budget, complete lowest-ranked contexts are removed first; if one parent is too large, its matched child is used only when that child and its citation metadata fit.
- [ ] The final provider request is asserted to remain within the declared context window, and the retrieval trace records selected and dropped token counts.
- [ ] Given a provider with unknown context size or evidence for which no cited passage fits, generation does not send an oversized request and returns the explicit insufficient-evidence behavior.

#### US-019: Preserve stable PDF, OCR, and text source spans

**Description:** As a notebook user, I want citations to open the actual supporting page or text span so that I can verify an answer without searching the whole source.

**Priority:** P1
**Size:** L (5 pts)
**Dependencies:** Blocked by US-001, US-008

**Acceptance Criteria:**

- [ ] Extraction preserves authoritative page boundaries for native PDF and OCR before text concatenation, and chunks store stable page and source-span provenance within their generation.
- [ ] Text, Markdown, and web sources store deterministic source offsets or section identifiers derived before chunk overlap is applied.
- [ ] Citation resolution verifies source, active generation, chunk/span ownership, and claim linkage before emitting the existing public citation shape.
- [ ] Native PDF tests prove reported pages equal actual fixture pages rather than character-count heuristics; OCR tests retain page provenance after normalization.
- [ ] Given stale-generation, missing-span, cross-source, duplicate, code-block, or unsupported citation markers, the marker is not emitted as a valid citation and the failure is counted by US-003 telemetry.

#### US-020: Isolate untrusted content and distinguish RAG failure states

**Description:** As a notebook user, I want source content treated only as evidence and infrastructure failures reported honestly so that malicious documents or retrieval outages cannot produce falsely grounded answers.

**Priority:** P0
**Size:** L (5 pts)
**Dependencies:** Blocked by US-003, US-018, US-019

**Acceptance Criteria:**

- [ ] Prompt templates explicitly classify retrieved documents as untrusted data, forbid following instructions found inside them, and delimit provenance separately from content for every provider path.
- [ ] SQL retrieval enforces account/notebook ownership and active-generation membership before any content reaches prompt assembly.
- [ ] Fifty adversarial fixtures cover instruction override, secret requests, fake system tags, poisoned citations, cross-notebook references, and encoded payloads, with zero successful policy override or cross-notebook evidence inclusion.
- [ ] No configured sources, no relevant evidence, and retrieval infrastructure failure produce three distinct internal reason codes: the first two generate their corresponding explicit fallback/abstention text, while infrastructure failure terminates through the existing structured error event instead of generating an ungrounded answer.
- [ ] Given hostile markup that survives structural escaping, it remains inside the untrusted data boundary and cannot alter system instructions; logs contain only hashes, IDs, counts, and reason codes.

## Functional Requirements

- FR-01: A source must have at most one active published index generation and at most one owned building generation.
- FR-02: Search must read only chunks belonging to each source's active generation.
- FR-03: Reprocessing must not delete or mutate the active generation before replacement publication.
- FR-04: Generation publication must be one atomic transaction after count, ownership, dimension, and finite-vector validation.
- FR-05: A failed, timed-out, cancelled, or stale generation must never become active.
- FR-06: Duplicate reprocess requests must coalesce without changing the public response shape.
- FR-07: Every ingestion task must be owned, cooperatively cancellable, and bounded by a drain deadline.
- FR-08: Embedding and chunk reuse must include provider/model/chunking provenance in their compatibility key.
- FR-09: RRF, lexical, dense, reranker, and confidence values must remain distinct score domains.
- FR-10: `max_context_chunks` must be enforced after all retrieval transformations on every path.
- FR-11: Reranking must operate on a diversified candidate pool larger than or equal to the final selection when candidates exist.
- FR-12: Filtered ANN configuration must be selected from an exact-search recall and latency benchmark.
- FR-13: Reformulation must fit its own history and token limits before calling a provider.
- FR-14: One token budget must cover every prompt and native-document component before generation.
- FR-15: Citations must resolve to active-generation source spans with authoritative PDF/OCR page provenance where available.
- FR-16: Retrieved content must remain an explicit untrusted-data region for every provider prompt format.
- FR-17: Retrieval infrastructure failure must not silently degrade to a factual ungrounded answer.
- FR-18: Evaluation reports must separate retrieval, response grounding, citation, abstention, performance, and operational metrics.
- FR-19: Release comparison must report aggregate and category-level regressions against a versioned baseline.
- FR-20: No default test or evaluation command may require a commercial provider or network access.

## Non-Functional Requirements

- **Performance:** On the documented 100,000-chunk PostgreSQL reference environment, filtered dense retrieval must achieve P95 <= 300 ms at 0.1%, 1%, 10%, and 100% filter selectivity; local retrieval changes must add no more than 20% P95 latency versus the approved baseline.
- **Cancellation:** Zero provider calls may start more than 1 second after cancellation, and cancellable ingestion tasks must drain within 5 seconds in deterministic tests.
- **Reliability:** Zero partial or mixed active-generation reads are permitted across 1,000 injected-failure and concurrency schedules; generation publication and rollback must be transactionally atomic.
- **Retrieval quality:** Recall@10 must be >= 0.90 overall and >= 0.80 in every category; ANN Recall@10 against exact search must be >= 0.95 with top-k fill >= 0.99.
- **Grounding quality:** Citation precision must be >= 0.95, citation coverage >= 0.90, abstention accuracy >= 0.90, and cross-notebook evidence inclusions must equal 0.
- **Prompt safety:** The 50-case hostile-content suite must produce 0 instruction-boundary overrides, 0 cross-notebook leaks, and 0 source-content values in telemetry output.
- **Budget correctness:** 100% of generated provider requests must fit the declared context window with a reserve of at least max(1,024 tokens, 5% of the context window).
- **Determinism:** Two offline evaluation runs at the same code revision, corpus version, seed, and configuration must produce byte-identical metric payloads excluding the declared timestamp.
- **Compatibility:** 100% of existing REST and SSE baseline fixtures must remain unchanged unless a story explicitly updates the authoritative Rust contract, protocol documentation, and baseline fixture together.

## Edge Cases & Error States

| # | Scenario | Trigger | Expected Behavior | User Message |
|---|----------|---------|-------------------|--------------|
| 1 | Empty corpus | Notebook has no configured sources | Skip retrieval and use the explicit no-sources fallback | "Aucune source n'est disponible dans ce notebook." |
| 2 | No relevant evidence | Active sources exist but retrieval returns no sufficient evidence | Abstain without unsupported claims and retain diagnostic reason | "Les sources disponibles ne permettent pas de répondre avec suffisamment de confiance." |
| 3 | Reprocessing in progress | A replacement generation is building | Continue searching the complete active generation and expose existing processing status | Existing source status payload |
| 4 | Concurrent reprocess | Multiple requests target one source | One worker owns the generation; other requests return current state without spawning work | Existing accepted source response |
| 5 | Provider or network failure | Embedding, reranking, or reformulation provider times out or rejects | Cancel dependent work, preserve active data, apply the documented branch-specific fallback once | Existing structured error or source failure payload |
| 6 | Database failure during publication | Transaction or connection fails | Roll back the publication and keep the previous generation active | Existing source failure payload with stable reason |
| 7 | Boundary values | Zero/oversized top-k, oversized chunk, unknown context window, NaN vector | Reject at boundary or abstain before provider invocation | Existing validation error or insufficient-evidence response |
| 8 | Access revoked | Source/notebook ownership changes before prompt assembly | SQL scope excludes the content; no cached result bypasses ownership | Existing not-found/forbidden behavior |
| 9 | Interrupted worker | Timeout, shutdown, or process restart | Cancel and drain owned tasks, mark stale build failed on recovery, preserve active generation | Existing source failure/status event |
| 10 | Rollback and cleanup | New generation is bad or obsolete cleanup fails | Repoint atomically to previous complete generation; never delete referenced generations | Operator log and unchanged public source shape |
| 11 | Hostile retrieved content | Source contains system-like instructions or fake citations | Treat as inert evidence, reject unsupported markers, record attack fixture outcome | No special content echoed to user |
| 12 | Unsupported pgvector capability | Installed extension lacks the benchmark-approved scan mode | Fail validation with required version and do not claim full top-k success | Actionable startup or retrieval error |

## Risks & Mitigations

| # | Risk | Probability | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | Generation migration makes existing chunks temporarily invisible | Medium | High | US-005 transaction proof, upgraded-database fixture, additive backfill, old-generation preservation, and publication stress tests |
| 2 | Synthetic evaluation overfits implementation fixtures | Medium | High | Locked holdout split, at least 8 categories, category-level thresholds, corpus versioning, and expansion to 150 cases |
| 3 | Filtered HNSW cannot meet recall and latency targets on the supported pgvector version | Medium | High | US-015 exact-reference benchmark blocks production changes and records the required version or unsatisfied constraint |
| 4 | Cancellation cannot stop already-running blocking work | Medium | Medium | Bound admission and drain, abort only cancellable async tasks, and report non-abortable operations honestly |
| 5 | New provenance fields increase storage and migration duration | Medium | Medium | Additive compact metadata, benchmark migration on synthetic scale, preserve old data until validation, and reclaim obsolete generations later |
| 6 | Absolute quality thresholds are unrealistic for one query category | Medium | Medium | Report both aggregate and per-category metrics; thresholds are explicit and changes require a corpus-versioned decision record |
| 7 | Prompt isolation reduces useful instruction-like document interpretation | Low | Medium | Test legitimate code/documentation instructions separately from adversarial control attempts and measure expected-claim coverage |
| 8 | Scope spans too many subsystems for one release | Medium | High | Four dependency-ordered epics, 20 or fewer stories, and no HyDE/contextual-retrieval tuning before the core gates pass |

## Non-Goals

Explicit boundaries for this version:

- Replacing PostgreSQL/pgvector with a managed vector database or introducing an external orchestration framework.
- Selecting a universal chunk size, overlap, top-k, HyDE policy, or contextual-retrieval policy without an ablation on the versioned holdout corpus.
- Adding hosted-only identity, billing, analytics, lifecycle email, proprietary UI, or commercial operations.
- Redesigning the chat UX or adding new public REST/SSE event types; existing shapes remain unless a later approved compatibility change requires otherwise.
- Supporting multiple embedding dimensions inside one active generation.
- Using production conversations or documents as evaluation fixtures.
- Building continuous online learning, automatic relevance-label generation, or LLM-judge-only release gates.

## Files NOT to Modify

- `backend/migration-core/src/core_track/m20260729_000001_core_baseline.rs` - applied baseline migration; add a timestamped migration instead.
- `contracts/openapi.json` - generated from Rust DTOs and API annotations.
- `contracts/core-constants.json` - generated from `backend/src/core/catalog.rs`.
- `packages/sdk-ts/src/generated/openapi.ts` - generated from the OpenAPI contract.
- `packages/sdk-ts/src/generated/catalog.ts` - generated from core constants.
- `.codex/` - local agent state outside product scope.

## Technical Considerations

- **Architecture:** Should generation state live in a dedicated `source_index_generations` table with `sources.active_generation_id`, or can the same invariants be expressed with fewer columns? Recommended: dedicated immutable generations because publication, rollback, provenance, and cleanup have separate lifecycles. Engineering must confirm through US-005.
- **Data Model:** Should legacy chunks be backfilled into one generation per source in a single migration or in bounded batches? Recommended: choose from a measured upgraded-database fixture while preserving additive compatibility and one final validation transaction.
- **Concurrency:** Should ownership use a compare-and-set source field, a generation uniqueness constraint, or both? Recommended: both, because the database constraint protects integrity while the source transition produces actionable state.
- **Cancellation:** Should task ownership use `JoinSet`, `TaskTracker`, or a small project-local owner around `CancellationToken`? Recommended: the smallest existing-stack composition that proves admission closure, bounded drain, and no detached handles.
- **Score Model:** Should confidence be rule-based or provider-calibrated? Recommended: deterministic insufficiency reasons by default, with optional provider/model-specific calibrated thresholds stored as explicit artifacts.
- **ANN:** What minimum pgvector version is acceptable after US-015? Recommended: require the lowest version whose measured scan mode meets recall, fill, ordering, and latency targets, then document it in README and upgrading guidance.
- **Citation Data:** Should structured spans be typed columns or versioned JSON metadata? Recommended: typed generation/source/page ownership for invariants, with versioned metadata only for source-type-specific details.
- **API Design:** Can infrastructure failure terminate through the existing SSE error event without a new contract? Recommended: yes; preserve event shapes and update protocol fixtures only if ordering or terminal behavior changes.
- **Dependencies:** Is a tokenizer or metric crate necessary? Recommended: first reuse `text-splitter`, `tiktoken-rs`, and project-local metric implementations; add a dependency only when a story demonstrates missing current capability.
- **Migration:** How long should previous generations remain rollback-eligible? Recommended: at least one prior complete generation and at least 24 hours, with cleanup operating only on unreferenced generations and never inside the publication transaction.

## Success Metrics

| Metric | Baseline (current) | Target | Timeframe | How Measured |
|--------|-------------------|--------|-----------|-------------|
| Mixed/partial active generations | Not measured; current delete-then-batch-write path permits exposure | 0 failures across 1,000 schedules | Before EP-002 completion | Deterministic PostgreSQL concurrency and failure-injection suite |
| Duplicate active chunk positions | No generation uniqueness constraint | 0 duplicate `(generation_id, chunk_index)` rows | Before US-009 completion | Database constraint and 100-request stress test |
| Calls started after cancellation | Not measured; detached producer can outlive timeout | 0 calls started after 1 second; drain <= 5 seconds | Before US-010 completion | Fake provider call counter and Tokio task lifecycle test |
| Retrieval evaluation coverage | 0 versioned labeled queries | >= 40 queries in Month 1; >= 150 in Month 6 | Month 1 and Month 6 | Corpus validator report |
| Recall@10 / nDCG@10 | Not measured | Recall@10 >= 0.90 and nDCG@10 >= 0.75 overall | Before EP-004 completion | Offline retrieval evaluator on locked holdout |
| Filtered ANN recall/fill | Not measured against exact search | Recall@10 >= 0.95 and fill >= 0.99 at 0.1% to 100% selectivity | Before US-016 completion | 100,000-row pgvector benchmark |
| Final chunk-limit violations | Supported no-reranker path can return the full pool, default 50, for a default final limit of 15 | 0 violations | Before US-013 completion | Branch matrix property tests |
| Prompt budget violations | Retrieved/native document blocks are not fully bounded | 0 oversized provider requests with required reserve | Before US-018 completion | Provider request capture tests |
| Citation precision / coverage | Syntax and source IDs tested; claim support not measured | Precision >= 0.95 and coverage >= 0.90 | Before EP-004 completion | Grounded-response evaluator on holdout |
| Abstention accuracy | Not measured | >= 0.90 on unanswerable and insufficient-evidence cases | Before EP-004 completion | Labeled response evaluator |
| Cross-notebook leakage | No dedicated adversarial evaluation baseline | 0 leaked chunks across all isolation tests | Every release after US-020 | SQL scope tests and hostile-content suite |
| Indirect prompt-injection success | 0 dedicated fixtures | 0 successes across at least 50 hostile fixtures | Before US-020 completion | Deterministic prompt/output policy assertions |

## Open Questions

- Engineering owner must record the exact pgvector extension version shipped by the reference image before US-015; US-016 depends on the available scan modes.
- Database owner must choose the measured backfill batch size before US-006; migration runtime and lock observations from the upgraded synthetic fixture decide it.
- Product/retrieval owner must approve the initial 40-query labels and holdout membership before US-004 captures the first baseline.
- Maintainers must decide whether infrastructure retrieval failure should preserve current SSE terminal ordering or require a versioned protocol change before US-020 implementation.
- Release owner must approve any future change to absolute quality thresholds through a corpus-version and decision-log update; implementation code must not tune them silently.
[/PRD]
