# RAG evaluation corpus and baselines

The evidence layer for retrieval and answer quality (EP-001,
`tasks/prd-rag-reliability-and-quality.md`). Tuning stories may not claim an
improvement without a measurement taken here.

Read [docs/contracts/rag-evaluation.md](../../docs/contracts/rag-evaluation.md)
for what each metric means and how the release gate decides.

## Layout

| Path | Contents |
|---|---|
| `corpus/manifest.json` | Corpus version, rationale, changelog, coverage contract |
| `corpus/notebooks.json` | Synthetic notebooks, sources, chunks, spans |
| `corpus/queries.json` | Labeled queries: category, split, judgments, expected claims |
| `baseline/hybrid-train.json` | Approved measurement on the training split |
| `baseline/hybrid-holdout.json` | Approved measurement on the holdout split |
| `adversarial/cases.json` | Fifty hostile documents for the `rag-eval adversarial` gate (US-020) |

## Commands

```bash
cd backend

# The corpus is valid, synthetic and covers every category
cargo run --bin rag-eval -- validate

# Measure retrieval; --mode is dense | lexical | hybrid | exact_reference
cargo run --bin rag-eval -- retrieval --mode hybrid --split train

# Measure answer behavior over the same retrieval
cargo run --bin rag-eval -- grounding --split train

# Every hostile fixture stays inside the untrusted-data boundary
cargo run --bin rag-eval -- adversarial

# Produce a candidate artifact and compare it against the approved one
cargo run --bin rag-eval -- baseline --out /tmp/candidate.json \
    --revision "$(git rev-parse --short HEAD)"
cargo run --bin rag-eval -- compare \
    --baseline ../contracts/eval/baseline/hybrid-train.json \
    --candidate /tmp/candidate.json
```

`compare` exits non-zero on a regression beyond 0.02, a tenant-isolation
failure, a missing required metric, or an incomparable pair. Add
`--enforce regression_and_targets` to also block on an unmet absolute target.

## Regenerating the baselines

```bash
cd backend
UPDATE_BASELINE=1 cargo test --no-default-features --test rag_eval
git diff ../contracts/eval/baseline    # review every line: this is a quality change
```

`backend/tests/rag_eval.rs` fails when a committed baseline no longer describes
the code, so a quality change cannot land unnoticed. Regeneration is
deterministic: two consecutive runs at one revision produce identical bytes.

## Editing the corpus

The corpus is hand-authored and reviewed; nothing generates it. When adding a
case:

1. Give it a unique `id`, a category, and a `split`. Keep both splits populated
   in every category.
2. An answerable query needs at least one `relevant_chunk`, the sources that own
   them in `relevant_sources`, and at least one expected claim with
   `answer_markers` — the literal substrings a correct answer must contain. An
   unanswerable query carries none of these.
3. `forbidden_sources` records **cross-notebook** isolation only. A source in the
   query's own notebook is legitimately searchable; listing one there would turn
   correct retrieval into a blocking failure.
4. Chunk `span`s must tile the source: with chunks joined by a blank line in
   index order, each span is the byte range its chunk occupies. `rag-eval
   validate` checks this and names the first chunk that does not fit.
5. Bump `corpus_version` and add a changelog entry. Reports and baselines record
   the version, and a comparison across versions is refused rather than
   silently misaligned.

Then regenerate the baselines: a corpus change moves the numbers by definition.

## Synthetic data only

Every notebook, source, chunk, query and identifier is written for this
repository. Production-derived content, real account identifiers, real email
addresses and credentials are forbidden by PRD hard constraint. Two checks
enforce it: `EvalCorpus::validate` walks the parsed structure, and
`backend/tests/rag_eval.rs` scans the raw bytes of every fixture file.

Identifiers are readable slugs (`src-alpha-runbook`), projected to UUIDs by a
stable SHA-256 derivation at load time. No UUID literal may appear in a fixture —
the validator rejects one on sight, because a hand-written UUID is exactly what a
production-derived identifier would look like.
