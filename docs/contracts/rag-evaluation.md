# RAG evaluation contract

How retrieval and answer quality are measured, what each number means, and what
blocks a release. Implements EP-001 of
`tasks/prd-rag-reliability-and-quality.md`.

The fixtures and commands live in [contracts/eval/](../../contracts/eval/README.md).

## Why three artifacts and not one

Retrieval and generation fail differently, and one number hides both. A fluent
answer built over the wrong passages and a correct answer whose evidence was
never retrieved look identical from the outside. So the evaluation splits into:

| Artifact | Question it answers |
|---|---|
| Retrieval report | Did the right passages come back, and in what order? |
| Grounded-response report | Did the answer assert what the sources support, cite it correctly, and abstain when it should? |
| Latency report | How long did it take on this machine? |

The first two are deterministic and byte-comparable. The third is
machine-dependent and is never asserted or compared — the same separation the
repository already makes for `contracts/baseline/latency/`.

## The corpus

40 labeled queries over 3 synthetic notebooks, 14 sources and 41 chunks,
stratified over eight failure modes with at least 5 cases each:

`exact_identifier`, `semantic_paraphrase`, `multi_hop`, `conflicting_sources`,
`unanswerable`, `tables_and_code`, `long_document`, `hostile_instructions`.

The split is locked: 24 training cases and 16 holdout cases, with both splits
populated in every category. `EvalCorpus::tuning_queries()` returns the training
split, and holdout labels are reachable only by naming `Split::Holdout`. That is
the whole mechanism — there is no accessor that hands out holdout labels by
accident.

Some queries carry `forbidden_sources`: sources in **another** notebook that must
never appear in a result set. Retrieving one is a tenant-isolation failure, which
blocks a release unconditionally.

## Retrieval metrics

Computed per query, then averaged overall and per category. Queries with no
relevance judgment — every `unanswerable` case — stay in the population and
report `null` for the ranking metrics rather than a fabricated zero.
`judged_queries` states the denominator explicitly.

| Metric | Definition |
|---|---|
| `recall_at_5` / `_10` / `_20` | Labeled relevant chunks in the top k, over labeled relevant chunks |
| `mrr` | Reciprocal of the first relevant rank; 0 when none appears |
| `ndcg_at_10` | Binary-gain nDCG, ideal computed over `min(relevant, 10)` |
| `source_recall` | Labeled relevant sources present, over labeled relevant sources |
| `duplicate_parent_rate` | Returned results whose parent context already appeared, over returned |
| `top_k_fill_rate` | Returned over requested, capped at 1 |
| `isolation_failures` | Result sets containing a forbidden source |
| `reasons` | Reason-code counts across the population |

### Modes

`dense`, `lexical` and `hybrid` run the production
`services::rag::search::search` orchestration. `exact_reference` is brute-force
exact cosine, bypassing the orchestration: it is the ground truth an approximate
index is measured against, and the shape US-015 will reuse against real pgvector.

### What the numbers do and do not mean

Embeddings come from the deterministic in-process hashing provider, so
`semantic_paraphrase` scores low **by construction** — synonyms do not match a
bag-of-words model. Lexical scores come from a BM25-shaped in-memory scorer, not
from PostgreSQL `ts_rank_cd`. Both facts are recorded in every report's `notes`.

These numbers are comparable across revisions of this repository at one corpus
version. They are not comparable against a database, against another provider, or
against a published benchmark.

### Determinism

Two runs at the same revision, corpus version and configuration produce
byte-identical JSON except for `generated_at`. The evaluator takes the timestamp
as a parameter rather than reading the clock, so there is no second source of
variance.

That used to need an intervention here: reciprocal rank fusion collects into a
hash map and sorts stably, so equal scores came out in hash-iteration order and
the evaluator re-sorted by `(score desc, chunk id asc)` before scoring. Since
US-013 the tie-break lives in fusion itself, where it also makes production
truncation deterministic. The evaluator still re-imposes the same order, now as
a check rather than a repair: if the pipeline's tie-break regresses, the report
diffs instead of flapping.

## Grounded-response metrics

| Metric | Definition |
|---|---|
| `expected_claim_coverage` | Expected claims the answers asserted, over expected claims labeled |
| `citation_precision` | Correct citations over citations emitted |
| `citation_coverage` | Asserted claims with at least one correct citation, over asserted claims |
| `abstention_accuracy` | Cases whose abstention decision matched the evidence |
| `unsupported_claims` | Asserted claims with no correct supporting citation |

Claim assertion is decided by the corpus's literal `answer_markers`, never by a
model. A `ClaimJudge` may be configured; its output lands in `diagnostics` and in
no metric. A gate that an LLM judge could move would be exactly as reproducible
as the judge.

### When a citation counts

Four conditions, all required:

1. Its `(source_id, chunk_index)` resolves to a chunk that exists.
2. That chunk belongs to the active generation.
3. It lives in the notebook the question was asked in.
4. Its quoted span is a passage of that chunk which supports a claim the answer
   made.

A citation that satisfies only the first is the failure this metric exists to
catch. It is counted as wrong and classified: `unknown_chunk`,
`stale_generation`, `cross_notebook`, `span_mismatch`, `unrelated_to_claim`.

The chat path refuses the same things before a citation is ever emitted (US-019):
a marker that resolves to nothing retrieved, a missing or stale active
generation, a claim without lexical support in the cited passage, a
provider-native quote the document does not contain, and a marker written inside
a code fence. Refusals are counted and reach the trace as `citation_rejected`,
so a regression appears as refusals rather than as silently lower coverage.
The active source row stays locked through event enqueue. Partial responses
persist no citations because they never reach this validation boundary.

### Abstention

Abstention is correct exactly when the question cannot be answered from what was
retrieved: either the corpus never had the answer, or retrieval did not surface
it. Answering anyway is `missing_abstention`; abstaining over sufficient evidence
is `spurious_abstention`. A provider error keeps the case in every denominator
with a `provider_error` classification — dropping it would quietly shrink the
population.

## The retrieval trace

Every chat turn emits one structured `rag_retrieval_trace` event carrying
generation ids, query hashes, mode, score domain, per-stage candidate counts,
unique parent count, token counts, reason codes and per-stage durations.

It carries no query text and no source content, and cannot: `RetrievalTrace` has
no field that could hold either. Queries appear as a truncated SHA-256, which is
enough to correlate a reformulation with its original and to follow a retry, and
is not reversible. The redaction is structural rather than editorial because a
guarantee that depends on every call site remembering it is a leak waiting for
its first hurried patch.

`ScoreDomain` records which scale produced the ordering: `rrf_rank`,
`dense_similarity`, `lexical_rank`, `reranker_relevance`, `stuffing_uniform`. An
RRF score is a function of ranks and a reranker score is provider-defined; they
are not on the same scale, and recording which applies is what keeps a later
consumer from averaging them. US-012 turns this into typed fields on the
candidates themselves.

## The hostile-content suite

`contracts/eval/adversarial/cases.json` holds fifty synthetic hostile documents
over six families — instruction override, secret request, fake system tag,
poisoned citation, cross-notebook reference, encoded payload — and
`rag-eval adversarial` assembles each into a real prompt.

```bash
cargo run --bin rag-eval -- adversarial            # exits non-zero on any violation
cargo run --bin rag-eval -- adversarial --out /tmp/isolation.json
```

It is a release gate, not only a unit test: it fails when the suite drops below
fifty cases, when an attack family loses its last fixture, when a payload breaks
a boundary property, or when a citation resolves outside the retrieved set.

It asserts structure, not behaviour. A model's refusal is not reproducible in
this offline gate; what is reproducible is whether the payload could close its
own element, forge a second data policy, or change a byte of the instructions
that follow the evidence. These checks are necessary but do not prove model
behavior. EP-004 stays open until a provider-specific behavioral run covers all
fifty cases with zero successful instruction following.

Cross-notebook reach is answered one layer down, where it is enforced: every
search query joins `notebooks.user_id`, so a scope naming another account
returns nothing. See
[docs/architecture/prompt-assembly.md](../architecture/prompt-assembly.md).

## The release gate

`rag-eval compare --baseline A --candidate B` exits non-zero when:

- `recall_at_10` or `ndcg_at_10` fell by more than **0.02**, overall or in any
  category that has judged queries on either side;
- `citation_precision` or `citation_coverage` fell by more than **0.02**;
- any tenant-isolation case failed, or any citation resolved cross-notebook;
- a required metric is absent from either document;
- the two artifacts are not comparable (different corpus version, chunking
  version, provider fingerprint, mode, split, requested limit or fusion
  parameters).

Absolute targets are separate. `--enforce regression_only` (the default) reports
them and does not block; `--enforce regression_and_targets` blocks. The
distinction exists because the month-6 targets — Recall@10 ≥ 0.90, nDCG@10 ≥
0.75, citation precision ≥ 0.95, citation coverage ≥ 0.90, abstention accuracy ≥
0.90 — are written for a real embedding provider, and a gate that refused to
record a below-target first baseline would leave the project with no baseline at
all.

A tenant-isolation failure and a missing metric block in both modes. Neither is a
quality trade-off: one is a leak, the other is a report that cannot be read.

Changing a target is a release-owner decision recorded with a corpus version
bump, not an implementation detail.

## Offline by construction

No command here reads a credential or opens a socket (FR-20). The corpus is a
checked-in fixture, retrieval runs against an in-memory index, and generation
uses the in-process deterministic model. That is a requirement rather than a
convenience: a release gate that needs a commercial key is a gate nobody runs.
