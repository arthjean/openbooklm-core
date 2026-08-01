# Prompt assembly, context budget and the untrusted boundary

The contract EP-004 implements: what reaches a provider, how it is bounded, what
a citation has to prove, and what happens when there is nothing to answer from.
Implements US-018, US-019 and US-020 of
`tasks/prd-rag-reliability-and-quality.md`.

## One budgeting pass

Every component that occupies the context window is counted once, before the
provider call, by `services::chat::context_budget::fit_prompt`. The arithmetic
lives in `llm::budget`.

`api::chat::turn::prepare` is where a turn is assembled: validate, budget,
retrieve, fit, assert. It returns one `TurnOutcome` per way the exchange can
end — stream from a provider, answer with a constant, terminate with `error` —
so the HTTP handler does nothing but frame typed events as SSE. The two endings
that involve no provider live in `api::chat::fallback`.

The window is spent in a fixed priority order. Each step takes what it needs
from what the previous ones left:

| Order | Component | If it does not fit |
|---|---|---|
| 1 | Reserve: `max(1 024, 5% of the window)` | never spent |
| 2 | Requested output tokens (`LlmProvider::max_output_tokens`) | never spent |
| 3 | System instructions and the current question | the turn is refused |
| 4 | Memory block, capped at 10% of the window | dropped whole |
| 5 | Retrieved evidence, capped at 60% of the window | lowest-ranked contexts dropped |
| 6 | Conversation history | oldest turns dropped |

Two properties are load-bearing:

**The evidence allowance is computed before retrieval.** Context stuffing loads
a whole notebook instead of searching it, and "the whole notebook fits" has to
mean the same thing in the retrieval pipeline and in the prompt. Both call
`evidence_allowance`, so a notebook that fits the requested chunk count but not
the token budget is searched rather than loaded and then trimmed. The reason
code is `stuffing_over_budget`.

**Evidence is priced by the renderer that sends it.** `write_entry` renders one
`<source>` element into a `fmt::Write` sink. `format_context_for_llm` gives it a
`String`; `entry_tokens` gives it a `TokenMeter`, which counts bytes and CJK
characters without building anything. So the budget measures the exact bytes the
renderer would emit, rather than estimating the passage and adding a constant
for the markup, and it does so without allocating the string it is about to
discard. Provider-native document blocks go through the same `entry_tokens` with
the other `EvidenceFormat`: they are not prompt text, but they occupy the same
window.

### A parent that does not fit degrades to its child

Evidence is retrieved as child chunks standing for a broader parent passage, and
the parent is normally what the prompt carries. When one parent exceeds the
remaining allowance, the matched child is sent in its place, provided the child
*and its provenance attributes* fit — a citation whose source id and page were
dropped to save tokens is a citation the reader cannot open. Reason code:
`parent_downgraded_to_child`.

Admission stops at the first context that fits in neither form, so what survives
is always a prefix of the ranking. Reason code: `evidence_dropped_for_budget`.

### An undeclared window is a refusal

`context_window_for_model` returns `Option`. A provider this build cannot size
gets no budget and no request: the previous 128 000-token fallback was a guess,
and a model with a smaller window rejects the request only after the prompt has
been assembled and paid for. Reason code: `context_window_unknown`.

The assembled request is checked once more before it is sent
(`PromptBudget::admits`). That check is not a `debug_assert`: a release build
must refuse an oversized request, not send it.

## The untrusted data boundary

Retrieved passages are data. The prompt says so, and the structure enforces it.

```
<data_policy> … five absolute rules … </data_policy>

<untrusted_source_data>
<source index="1" source_id="…" title="…" page="3">
<content>
… the document's own bytes, XML-escaped …
</content>
</source>
</untrusted_source_data>

<memory> … </memory>          ← outside the region: the user's own conversation
<role> … <instructions> …     ← the mode template
<language> …
```

Provenance is what the *system* knows and travels as attributes. Content is what
the *document* says and travels inside `<content>`, escaped. A passage that
writes `</content><data_policy>` cannot close the element it is in, because the
five structural characters never survive escaping.

Providers with native citation support receive the same policy, worded for
attached document blocks, and no inline region.

`llm::prompts::EvidenceFormat` has no "no evidence" variant. A prompt built here
always has evidence attached. The templates that told the model to answer from
its own knowledge are gone: a retrieval outage and an empty notebook produced a
fluent, confident, ungrounded answer that read exactly like a grounded one
(FR-17).

That enum is the turn's single switch, decided once from the provider. It
selects the data policy in `llm::prompts`, the renderer in
`services::rag::search::render_evidence`, and the per-entry price in
`entry_tokens`. Nothing downstream re-derives "does this provider take native
documents" from a boolean.

### The hostile-content suite

`contracts/eval/adversarial/cases.json` holds fifty synthetic hostile documents
across six families: instruction override, secret request, fake system tag,
poisoned citation, cross-notebook reference, encoded payload.
`rag-eval adversarial` assembles each one into a real prompt and checks
properties that do not depend on a model's mood:

- exactly the data policy blocks the builder wrote;
- exactly one evidence region;
- one closing `</content>` per rendered source;
- the instructions after the region are byte-identical to the evidence-free
  assembly;
- structural characters survive as entities, not as markup;
- citation markers the payload wrote resolve only to retrieved evidence.

Model behaviour is not asserted, because it is not reproducible in the offline
gate. These checks prove the structural boundary only. EP-004 remains open
until a provider-specific behavioral evaluation demonstrates zero successful
instruction following on the same fixtures.

Cross-notebook reach is answered where it is enforced rather than in the prompt:
`NotebookScope` carries the account and the notebook, and all four search
queries join `notebooks.user_id`. The handler's access check and the SQL are
separated by an embedding call, possibly a reformulation call and a reranker
call; ownership can be revoked in that window, and only the query itself can
notice.

## Pages and spans

A paginated source reaches the chunker as pages
(`chunking::SourceText::paginated`), never as a joined string. Cleaning happens
per page, so joining the cleaned pages yields both the text to split and the
byte offset where each page begins.

Both chunking passes use the splitter's `chunk_indices`, so every parent and
every child carries the exact byte range it occupies. Parents do not overlap and
partition the source; children overlap by design and each still carries its own
range.

| Metadata | Meaning |
|---|---|
| `page_number` | first authoritative page the chunk covers |
| `page_end` | last page it covers; equal to `page_number` in the ordinary case |
| `span_start` / `span_end` | byte range in the cleaned source text |

This replaces a heuristic that divided a character offset by 3 000 and called
the quotient a page. On short pages it drifted from page two onwards, and a
citation that opens the wrong page is worse than one that opens none: the reader
checks the wrong paragraph and believes it.

OCR keeps page identity through the content-hash cache. Pages are stored joined
by a form feed (`U+000C`), which chunking strips from page text, so the round
trip is exact. The cache key carries the payload schema (`model#pages-v1`), so
entries written before US-019 — one joined string with no page boundaries — are
unreachable rather than decoded as a single page. Those documents are OCR'd once
more; the alternative was a population of sources whose every citation said
page 1, hidden behind a cache hit.

### A marker is not a citation

A citation is emitted only when seven things hold:

1. the marker resolves to a chunk retrieved this turn;
2. that chunk carries an index generation (a nil generation was never published);
3. that generation remains active while a source-row lease is held through
   event enqueue;
4. its recorded span and pages describe a passage that can exist - `span_start <
   span_end`, `page_number <= page_end`, no last page without a first;
5. the immediately preceding claim has a conservative lexical support signal in
   that passage, including matching values and polarity;
6. a provider-native quoted passage is a passage of that chunk;
7. the marker is not inside a code span or fence.

Rule 4 is the span half of US-019 AC-3. The chunker writes both ends in one
pass, so a violation means the row did not come from it: a hand-edited index, a
truncated write, a generation from another schema. A chunk with no span at all
is not a violation — notebooks indexed before US-019 carry none, and they stay
citable.

Everything else is refused and counted. No public citation event is sent while
the answer is still streaming. Provider-native citations get the same
treatment, through the same checks: a `document_index` the request never sent, a
`cited_text` the document does not contain, a stale generation, an unrelated
claim or an incoherent span. Both paths read
provenance through `llm::types::ChunkProvenance`, which parses the stored
metadata once, typed, instead of probing the JSON key by key in each of them.
The count reaches the retrieval trace as `citation_rejected`, so a regression
shows up as refusals rather than as silently lower citation coverage.

Interrupted, truncated and shutdown responses never reach final validation, so
their persisted partial messages carry no citations even when marker text was
already streamed.

## Three ways to have no answer

An empty result set is three different incidents, and only one of them is an
error. `services::chat::orchestration::RetrievalFailure` classifies them at the
point where the outcome is still whole.

| Incident | Reason code | Behaviour |
|---|---|---|
| The notebook has no source | `empty_corpus` | Answers `no_sources_text(locale)` |
| Sources exist, nothing relevant came back | `no_candidates` | Answers `insufficient_evidence_text(locale)` |
| Retrieval did not run | `provider_error` | Terminates through the existing SSE `error` event |
| The request cannot be measured | `context_window_unknown`, `prompt_over_budget` | Terminates through the existing SSE `error` event |

The first two are constants, not generated text: "there is nothing to search" is
a fact about the notebook, and asking a model to phrase it is what produced an
ungrounded answer instead. Each locale has its own pair, and all of them live in
`llm::fallbacks::FALLBACK_TEXTS`. That table is what `reads_as_abstention` matches,
so the grounded-response report scores such a turn as an abstention rather than
as an unsupported answer. The table exists because the coupling used to run the
other way: the evaluator matched a French substring, so every locale's answer had
to embed the French sentence, and the metric was dictating the product copy.

Both fallbacks emit the same terminal event sequence a generated answer does —
content, citations, metrics, `done` — because the transport contract does not
change just because no model was involved. No permit is recorded: nothing was
generated, so nothing is charged.

The last two terminate with `error` and no `done`, which is the SSE protocol's
existing terminal failure shape. This was the PRD's open question about whether
an infrastructure failure needs a versioned protocol change: it does not. Event
names, payloads and ordering are unchanged; only the set of conditions that
reach the terminal error grew.

A budget refusal is deliberately *not* answered with an abstention. The sources
are fine; the deployment is not. Returning "the available sources do not support
a confident answer" for an undeclared context window would be a false statement
about the user's notebook, it would score as a correct abstention in the
grounded-response report, and it would let a misconfigured model emit
`ChatCompleted` for every turn it silently refused.

## Related documents

- [docs/contracts/sse-protocol-v1.md](../contracts/sse-protocol-v1.md) — the wire contract
- [docs/contracts/rag-evaluation.md](../contracts/rag-evaluation.md) — what the reports measure
- [docs/architecture/index-generations.md](index-generations.md) — what "active generation" means
