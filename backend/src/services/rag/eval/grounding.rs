//! Grounded-response evaluation (US-003).
//!
//! Retrieval quality and answer quality fail differently, and a single number
//! hides both. This module measures the second one: did the answer assert what
//! the sources support, did every citation resolve to a span that carries the
//! claim, and did it abstain when it should have.
//!
//! # A pure evaluator with a separate producer
//!
//! [`run_grounding_eval`] takes answers as input. It never calls a provider.
//! That split is what lets every branch — a malformed citation, a stale
//! generation, a fluent answer over an unanswerable question — be exercised
//! from a fixture instead of coaxed out of a model.
//!
//! [`answer_with_deterministic_pipeline`] is the producer CI uses: real
//! retrieval, the real prompt assembly, the in-process deterministic model, real
//! [`extract_citations`]. No network, no key.
//!
//! # A citation is not correct because it parses
//!
//! Four things have to hold before a citation counts (US-003 AC-3): its
//! `(source_id, chunk_index)` must resolve to a chunk that exists, that chunk
//! must belong to the active generation, it must live in the notebook the
//! question was asked in, and its quoted span must actually be a passage of that
//! chunk which supports a claim the answer made. A citation that only satisfies
//! the first is exactly the failure mode this metric exists to catch, so it is
//! counted as wrong and classified.
//!
//! # The judge advises, it does not decide
//!
//! [`ClaimJudge`] output lands in `diagnostics` and in no metric. An LLM judge
//! that could move a release gate would make the gate as reproducible as the
//! judge, which is not at all.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::core::providers::DeterministicLlm;
use crate::error::AppError;
use crate::llm::citations::extract_citations;
use crate::llm::types::{CitableChunk, Citation};
use crate::services::rag::search::format_context_for_llm;

use super::corpus::{EvalCorpus, EvalQuery, QueryCategory, Split};
use super::index::CorpusIndex;
use super::retrieval::{RetrievalRunConfig, retrieve_for_query};

/// Bumped whenever the report shape changes.
pub const GROUNDING_REPORT_SCHEMA: &str = "rag-grounding-eval/1";

/// Sentences that mean "I am not answering from these sources".
///
/// The first two are the fallbacks the PRD's edge-case table specifies for an
/// empty corpus and for insufficient evidence; US-020 is what makes the chat
/// path emit them. The third is what the in-process deterministic model says
/// when retrieval handed it nothing. Matching is on a normalized prefix so that
/// a trailing clause does not defeat it.
pub const ABSTENTION_MARKERS: &[&str] = &[
    "aucune source n'est disponible",
    "les sources disponibles ne permettent pas de répondre",
    "no source was retrieved",
];

fn round(value: f64) -> f64 {
    if value.is_finite() {
        (value * 1_000_000.0).round() / 1_000_000.0
    } else {
        0.0
    }
}

// ============================================================================
// Inputs
// ============================================================================

/// What a model did with one query.
#[derive(Debug, Clone)]
pub enum AnswerOutcome {
    /// The model answered, with whatever citations were extracted from it.
    Answered {
        text: String,
        citations: Vec<Citation>,
    },
    /// The model declined to answer from the sources.
    Abstained { text: String },
    /// The provider failed. The case stays in every denominator.
    ProviderError { reason: String },
}

/// One case handed to the evaluator.
#[derive(Debug, Clone)]
pub struct AnswerCase {
    /// Must match a query id in the corpus.
    pub query_id: String,
    /// Generations the answer's evidence was read from.
    pub generation_ids: Vec<Uuid>,
    /// Whether retrieval actually surfaced a labeled relevant chunk.
    ///
    /// Supplied by the caller from the retrieval run: "insufficient retrieved
    /// evidence" is a property of what happened, not of the corpus label, and
    /// abstention is the correct behavior in both cases (US-003 AC-4).
    pub evidence_sufficient: bool,
    pub outcome: AnswerOutcome,
}

/// An optional second opinion. Advisory only.
pub trait ClaimJudge: Send + Sync {
    fn name(&self) -> &str;
    /// Free-form observations about one answer. Never consulted by a metric.
    fn diagnose(&self, query: &EvalQuery, answer: &str) -> Vec<String>;
}

// ============================================================================
// Verdicts
// ============================================================================

/// Why one citation did or did not count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CitationVerdict {
    /// Resolves, current, in-notebook, and its span supports an asserted claim.
    Correct,
    /// `(source_id, chunk_index)` matches nothing in the corpus.
    UnknownChunk,
    /// The chunk exists but the answer read it from a superseded generation.
    StaleGeneration,
    /// The chunk belongs to a different notebook than the question.
    CrossNotebook,
    /// The quoted text is not a passage of the cited chunk.
    SpanMismatch,
    /// The chunk is real and current, but supports no claim the answer made.
    UnrelatedToClaim,
}

impl CitationVerdict {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Correct => "correct",
            Self::UnknownChunk => "unknown_chunk",
            Self::StaleGeneration => "stale_generation",
            Self::CrossNotebook => "cross_notebook",
            Self::SpanMismatch => "span_mismatch",
            Self::UnrelatedToClaim => "unrelated_to_claim",
        }
    }
}

impl fmt::Display for CitationVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How one case ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseClassification {
    /// Answered when it should, abstained when it should.
    Ok,
    /// The provider errored. Counted, never silently dropped.
    ProviderError,
    /// Evidence was absent or insufficient and the model answered anyway.
    MissingAbstention,
    /// Evidence was sufficient and the model abstained.
    SpuriousAbstention,
    /// Answered without emitting a single citation.
    Uncited,
    /// The corpus has no query with this id.
    UnknownQuery,
}

impl CaseClassification {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::ProviderError => "provider_error",
            Self::MissingAbstention => "missing_abstention",
            Self::SpuriousAbstention => "spurious_abstention",
            Self::Uncited => "uncited",
            Self::UnknownQuery => "unknown_query",
        }
    }
}

// ============================================================================
// Report
// ============================================================================

/// Answer quality for one population of cases.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundingMetrics {
    pub cases: usize,
    /// Expected claims the answers asserted, over expected claims labeled.
    pub expected_claim_coverage: f64,
    /// Correct citations over citations emitted.
    pub citation_precision: f64,
    /// Asserted claims carrying at least one correct citation, over asserted
    /// claims.
    pub citation_coverage: f64,
    /// Cases whose abstention decision matched the evidence.
    pub abstention_accuracy: f64,
    /// Asserted claims with no correct supporting citation.
    pub unsupported_claims: usize,
    pub citations_emitted: usize,
    pub citations_correct: usize,
    pub citation_verdicts: BTreeMap<String, usize>,
    pub classifications: BTreeMap<String, usize>,
}

/// What happened for one case.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CaseOutcome {
    pub query_id: String,
    pub category: String,
    pub split: String,
    pub classification: CaseClassification,
    pub expected_claims: usize,
    pub asserted_claims: usize,
    pub supported_claims: usize,
    pub unsupported_claims: usize,
    pub citations_emitted: usize,
    pub citations_correct: usize,
    pub citation_verdicts: Vec<CitationVerdict>,
    pub abstained: bool,
    pub abstention_expected: bool,
    pub abstention_correct: bool,
    /// Advisory judge output. Never read by a metric.
    pub diagnostics: Vec<String>,
}

/// The grounded-response report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroundingReport {
    pub schema: String,
    /// The one field excluded from the byte-stability contract.
    pub generated_at: String,
    pub corpus_version: String,
    pub corpus_generation: String,
    pub code_revision: String,
    pub split: String,
    /// Name of the advisory judge, or `"none"`.
    pub judge: String,
    pub notes: Vec<String>,
    pub overall: GroundingMetrics,
    pub by_category: BTreeMap<String, GroundingMetrics>,
    pub cases: Vec<CaseOutcome>,
}

// ============================================================================
// Evaluator
// ============================================================================

/// Configuration for one grounded-response run.
#[derive(Debug, Clone)]
pub struct GroundingRunConfig {
    pub split: Split,
    pub code_revision: String,
}

impl Default for GroundingRunConfig {
    fn default() -> Self {
        Self {
            split: Split::Train,
            code_revision: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

/// Score a set of answers against the corpus.
///
/// `generated_at` is a parameter for the same reason as in the retrieval
/// runner: the evaluator otherwise has no source of nondeterminism, and taking
/// the clock as input keeps it that way.
#[must_use]
pub fn run_grounding_eval(
    corpus: &EvalCorpus,
    cases: &[AnswerCase],
    config: &GroundingRunConfig,
    judge: Option<&dyn ClaimJudge>,
    generated_at: &str,
) -> GroundingReport {
    let outcomes: Vec<CaseOutcome> = cases
        .iter()
        .map(|case| score_case(corpus, case, judge))
        .collect();

    let mut by_category = BTreeMap::new();
    for category in QueryCategory::ALL {
        let key = category.as_str();
        by_category.insert(key.to_owned(), aggregate(&outcomes, |o| o.category == key));
    }

    GroundingReport {
        schema: GROUNDING_REPORT_SCHEMA.to_owned(),
        generated_at: generated_at.to_owned(),
        corpus_version: corpus.version().to_owned(),
        corpus_generation: corpus.generation().to_owned(),
        code_revision: config.code_revision.clone(),
        split: config.split.as_str().to_owned(),
        judge: judge.map_or_else(|| "none".to_owned(), |j| j.name().to_owned()),
        notes: vec![
            "Claim assertion is decided by the corpus's literal answer markers, \
             never by a model. A judge, when configured, only adds diagnostics."
                .to_owned(),
            "A citation counts only when its chunk resolves, belongs to the \
             active generation and the query's notebook, and its quoted span \
             supports an asserted claim."
                .to_owned(),
        ],
        overall: aggregate(&outcomes, |_| true),
        by_category,
        cases: outcomes,
    }
}

/// Whether a text reads as an abstention.
#[must_use]
pub fn reads_as_abstention(text: &str) -> bool {
    let normalized = text.trim().to_lowercase();
    ABSTENTION_MARKERS
        .iter()
        .any(|marker| normalized.contains(marker))
}

fn score_case(
    corpus: &EvalCorpus,
    case: &AnswerCase,
    judge: Option<&dyn ClaimJudge>,
) -> CaseOutcome {
    let Some(query) = corpus.all_queries().iter().find(|q| q.id == case.query_id) else {
        // An answer for a query the corpus does not have cannot be scored, but
        // dropping it would quietly shrink the denominator.
        return CaseOutcome {
            query_id: case.query_id.clone(),
            category: "unknown".to_owned(),
            split: "unknown".to_owned(),
            classification: CaseClassification::UnknownQuery,
            expected_claims: 0,
            asserted_claims: 0,
            supported_claims: 0,
            unsupported_claims: 0,
            citations_emitted: 0,
            citations_correct: 0,
            citation_verdicts: Vec::new(),
            abstained: false,
            abstention_expected: false,
            abstention_correct: false,
            diagnostics: Vec::new(),
        };
    };

    // Abstention is correct exactly when the question cannot be answered from
    // what was retrieved: either the corpus never had the answer, or retrieval
    // did not surface it.
    let abstention_expected = !query.answerable || !case.evidence_sufficient;

    let (answer_text, citations, abstained, mut classification) = match &case.outcome {
        AnswerOutcome::ProviderError { .. } => (
            String::new(),
            Vec::new(),
            false,
            CaseClassification::ProviderError,
        ),
        AnswerOutcome::Abstained { text } => {
            (text.clone(), Vec::new(), true, CaseClassification::Ok)
        }
        AnswerOutcome::Answered { text, citations } => {
            // A model that emitted the fallback sentence abstained, whatever
            // the producer labeled it.
            let abstained = reads_as_abstention(text);
            (
                text.clone(),
                if abstained {
                    Vec::new()
                } else {
                    citations.clone()
                },
                abstained,
                CaseClassification::Ok,
            )
        }
    };

    let provider_failed = classification == CaseClassification::ProviderError;

    // --- Which expected claims did the answer actually assert? -----------
    let asserted: Vec<&super::corpus::ExpectedClaim> = if provider_failed || abstained {
        Vec::new()
    } else {
        query
            .expected_claims
            .iter()
            .filter(|claim| asserts_claim(&answer_text, claim))
            .collect()
    };

    // --- Citation verdicts ------------------------------------------------
    let verdicts: Vec<CitationVerdict> = citations
        .iter()
        .map(|citation| judge_citation(corpus, query, case, &asserted, citation))
        .collect();
    let citations_correct = verdicts
        .iter()
        .filter(|v| **v == CitationVerdict::Correct)
        .count();

    // A claim is supported when a correct citation points at a chunk the corpus
    // says carries it.
    let correctly_cited: BTreeSet<String> = citations
        .iter()
        .zip(&verdicts)
        .filter(|(_, verdict)| **verdict == CitationVerdict::Correct)
        .filter_map(|(citation, _)| resolve_chunk_slug(corpus, citation))
        .collect();

    let supported_claims = asserted
        .iter()
        .filter(|claim| {
            claim
                .supported_by
                .iter()
                .any(|slug| correctly_cited.contains(slug))
        })
        .count();

    if !provider_failed {
        if abstained && !abstention_expected {
            classification = CaseClassification::SpuriousAbstention;
        } else if !abstained && abstention_expected {
            classification = CaseClassification::MissingAbstention;
        } else if !abstained && citations.is_empty() {
            classification = CaseClassification::Uncited;
        }
    }

    let diagnostics = judge.map_or_else(Vec::new, |j| j.diagnose(query, &answer_text));

    CaseOutcome {
        query_id: query.id.clone(),
        category: query.category.as_str().to_owned(),
        split: query.split.as_str().to_owned(),
        classification,
        expected_claims: query.expected_claims.len(),
        asserted_claims: asserted.len(),
        supported_claims,
        unsupported_claims: asserted.len() - supported_claims,
        citations_emitted: citations.len(),
        citations_correct,
        citation_verdicts: verdicts,
        abstained,
        abstention_expected,
        abstention_correct: abstained == abstention_expected,
        diagnostics,
    }
}

/// Does the answer literally assert the claim?
///
/// Every marker must appear. Deterministic and dumb on purpose: the alternative
/// is a model deciding, and then the release gate inherits its variance.
fn asserts_claim(answer: &str, claim: &super::corpus::ExpectedClaim) -> bool {
    let normalized = answer.to_lowercase();
    !claim.answer_markers.is_empty()
        && claim
            .answer_markers
            .iter()
            .all(|marker| normalized.contains(&marker.to_lowercase()))
}

/// The corpus chunk a citation points at, by slug.
fn resolve_chunk_slug(corpus: &EvalCorpus, citation: &Citation) -> Option<String> {
    corpus.notebooks().iter().find_map(|notebook| {
        notebook.sources.iter().find_map(|source| {
            if source.uuid() != citation.source_id {
                return None;
            }
            source
                .chunks
                .iter()
                .find(|chunk| chunk.index == citation.chunk_index)
                .map(|chunk| chunk.id.clone())
        })
    })
}

/// Apply the four correctness conditions, in the order that produces the most
/// specific verdict.
fn judge_citation(
    corpus: &EvalCorpus,
    query: &EvalQuery,
    case: &AnswerCase,
    asserted: &[&super::corpus::ExpectedClaim],
    citation: &Citation,
) -> CitationVerdict {
    let Some(slug) = resolve_chunk_slug(corpus, citation) else {
        return CitationVerdict::UnknownChunk;
    };
    let Some((notebook, _, chunk)) = corpus.chunk(&slug) else {
        return CitationVerdict::UnknownChunk;
    };

    if !case.generation_ids.contains(&corpus.generation_id()) {
        return CitationVerdict::StaleGeneration;
    }
    if notebook.id != query.notebook {
        return CitationVerdict::CrossNotebook;
    }
    // The public `Citation` carries the quoted passage, truncated with an
    // ellipsis. Strip it before checking that the quote is really a prefix of
    // the chunk: a quote that is not is a span that does not exist.
    let quoted = citation.text.trim_end_matches('.').trim();
    if !quoted.is_empty() && !chunk.content.contains(quoted) {
        return CitationVerdict::SpanMismatch;
    }
    if !asserted
        .iter()
        .any(|claim| claim.supported_by.contains(&slug))
    {
        return CitationVerdict::UnrelatedToClaim;
    }
    CitationVerdict::Correct
}

fn aggregate(outcomes: &[CaseOutcome], select: impl Fn(&CaseOutcome) -> bool) -> GroundingMetrics {
    let population: Vec<&CaseOutcome> = outcomes.iter().filter(|o| select(o)).collect();

    let mut citation_verdicts: BTreeMap<String, usize> = BTreeMap::new();
    let mut classifications: BTreeMap<String, usize> = BTreeMap::new();
    for outcome in &population {
        *classifications
            .entry(outcome.classification.as_str().to_owned())
            .or_insert(0) += 1;
        for verdict in &outcome.citation_verdicts {
            *citation_verdicts
                .entry(verdict.as_str().to_owned())
                .or_insert(0) += 1;
        }
    }

    let sum = |f: fn(&CaseOutcome) -> usize| -> usize { population.iter().map(|o| f(o)).sum() };

    let expected_claims = sum(|o| o.expected_claims);
    let asserted_claims = sum(|o| o.asserted_claims);
    let supported_claims = sum(|o| o.supported_claims);
    let citations_emitted = sum(|o| o.citations_emitted);
    let citations_correct = sum(|o| o.citations_correct);
    let abstention_correct = population.iter().filter(|o| o.abstention_correct).count();

    // Every ratio guards its own denominator: a population with nothing to
    // measure reports zero next to the count that explains it, never NaN.
    #[allow(clippy::cast_precision_loss)]
    let ratio = |numerator: usize, denominator: usize| -> f64 {
        if denominator == 0 {
            0.0
        } else {
            round(numerator as f64 / denominator as f64)
        }
    };

    GroundingMetrics {
        cases: population.len(),
        expected_claim_coverage: ratio(asserted_claims, expected_claims),
        citation_precision: ratio(citations_correct, citations_emitted),
        citation_coverage: ratio(supported_claims, asserted_claims),
        abstention_accuracy: ratio(abstention_correct, population.len()),
        unsupported_claims: sum(|o| o.unsupported_claims),
        citations_emitted,
        citations_correct,
        citation_verdicts,
        classifications,
    }
}

// ============================================================================
// Deterministic producer
// ============================================================================

/// Produce answers offline: real retrieval, real prompt, in-process model, real
/// citation extraction.
///
/// This is what the default CI mode evaluates. It makes no network request and
/// needs no credential (FR-20).
///
/// # Errors
/// Returns the embedding provider's error.
pub async fn answer_with_deterministic_pipeline(
    corpus: &EvalCorpus,
    index: &CorpusIndex,
    config: &RetrievalRunConfig,
) -> Result<Vec<AnswerCase>, AppError> {
    let mut cases = Vec::new();

    for query in corpus.queries(config.split) {
        let notebook_id = corpus
            .notebook(&query.notebook)
            .map_or_else(Uuid::nil, super::corpus::CorpusNotebook::uuid);
        let results = retrieve_for_query(index, config, notebook_id, &query.query)
            .await?
            .results;

        // "Sufficient evidence" means a labeled relevant chunk was actually
        // retrieved, not that the corpus says one exists.
        let relevant: BTreeSet<Uuid> = query
            .relevant_chunks
            .iter()
            .filter_map(|slug| corpus.chunk(slug).map(|(_, _, chunk)| chunk.uuid()))
            .collect();
        let evidence_sufficient = results.iter().any(|r| relevant.contains(&r.chunk_id));

        let system_prompt = format!(
            "You are a helpful assistant.\n\n{}",
            format_context_for_llm(&results)
        );
        let text = DeterministicLlm::answer_for(&system_prompt);

        let citable: Vec<CitableChunk> = results.iter().map(CitableChunk::from).collect();
        let citations = extract_citations(&text, &citable);

        let outcome = if reads_as_abstention(&text) {
            AnswerOutcome::Abstained { text }
        } else {
            AnswerOutcome::Answered { text, citations }
        };

        cases.push(AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient,
            outcome,
        });
    }

    Ok(cases)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::rag::eval::retrieval::RetrievalMode;

    const FIXED_TIME: &str = "2026-07-31T00:00:00Z";

    fn corpus() -> EvalCorpus {
        EvalCorpus::load_default().expect("corpus loads")
    }

    fn config() -> GroundingRunConfig {
        GroundingRunConfig {
            split: Split::Train,
            code_revision: "test".to_owned(),
        }
    }

    /// The first answerable training query, and the chunk that answers it.
    fn answerable_case(corpus: &EvalCorpus) -> (&EvalQuery, String, i32, Uuid) {
        let query = corpus
            .queries(Split::Train)
            .into_iter()
            .find(|q| q.answerable && !q.expected_claims.is_empty())
            .expect("corpus has an answerable training query");
        let slug = query.expected_claims[0].supported_by[0].clone();
        let (_, source, chunk) = corpus.chunk(&slug).expect("labeled chunk exists");
        (query, chunk.content.clone(), chunk.index, source.uuid())
    }

    fn citation(source_id: Uuid, chunk_index: i32, text: &str) -> Citation {
        Citation {
            source_id,
            chunk_index,
            text: text.to_owned(),
            relevance_score: 0.9,
            section_header: None,
            page_number: None,
            timestamp_start: None,
            timestamp_end: None,
            video_id: None,
            citation_url: None,
        }
    }

    /// An answer that asserts every expected claim of `query`.
    fn perfect_answer(query: &EvalQuery) -> String {
        let mut text = String::from("Answer: ");
        for claim in &query.expected_claims {
            for marker in &claim.answer_markers {
                text.push_str(marker);
                text.push(' ');
            }
        }
        text
    }

    #[test]
    fn a_correct_citation_over_an_asserted_claim_scores_perfectly() {
        let corpus = corpus();
        let (query, content, chunk_index, source_id) = answerable_case(&corpus);

        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: true,
            outcome: AnswerOutcome::Answered {
                text: perfect_answer(query),
                citations: vec![citation(source_id, chunk_index, &content)],
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(report.overall.citations_correct, 1);
        assert!((report.overall.citation_precision - 1.0).abs() < f64::EPSILON);
        assert!((report.overall.citation_coverage - 1.0).abs() < f64::EPSILON);
        assert_eq!(report.overall.unsupported_claims, 0);
        assert_eq!(report.cases[0].classification, CaseClassification::Ok);
    }

    #[test]
    fn a_valid_marker_over_a_span_that_does_not_carry_the_text_is_not_correct() {
        let corpus = corpus();
        let (query, _, chunk_index, source_id) = answerable_case(&corpus);

        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: true,
            outcome: AnswerOutcome::Answered {
                text: perfect_answer(query),
                // Right source, right chunk index, invented passage.
                citations: vec![citation(source_id, chunk_index, "a passage nobody wrote")],
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(
            report.cases[0].citation_verdicts,
            vec![CitationVerdict::SpanMismatch]
        );
        assert_eq!(report.overall.citations_correct, 0);
        assert!(report.overall.unsupported_claims > 0);
    }

    #[test]
    fn a_citation_from_a_superseded_generation_is_not_correct() {
        let corpus = corpus();
        let (query, content, chunk_index, source_id) = answerable_case(&corpus);

        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![Uuid::from_u128(999)],
            evidence_sufficient: true,
            outcome: AnswerOutcome::Answered {
                text: perfect_answer(query),
                citations: vec![citation(source_id, chunk_index, &content)],
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(
            report.cases[0].citation_verdicts,
            vec![CitationVerdict::StaleGeneration]
        );
    }

    #[test]
    fn a_citation_into_another_notebook_is_classified_as_cross_notebook() {
        let corpus = corpus();
        let (query, _, _, _) = answerable_case(&corpus);

        // A chunk of a source in some other notebook.
        let (other_source, other_chunk) = corpus
            .notebooks()
            .iter()
            .filter(|nb| nb.id != query.notebook)
            .find_map(|nb| {
                nb.sources
                    .first()
                    .and_then(|s| s.chunks.first().map(|c| (s, c)))
            })
            .expect("corpus has a second notebook");

        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: true,
            outcome: AnswerOutcome::Answered {
                text: perfect_answer(query),
                citations: vec![citation(
                    other_source.uuid(),
                    other_chunk.index,
                    &other_chunk.content,
                )],
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(
            report.cases[0].citation_verdicts,
            vec![CitationVerdict::CrossNotebook]
        );
    }

    #[test]
    fn an_unresolvable_marker_is_classified_rather_than_ignored() {
        let corpus = corpus();
        let (query, _, _, _) = answerable_case(&corpus);

        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: true,
            outcome: AnswerOutcome::Answered {
                text: perfect_answer(query),
                citations: vec![citation(Uuid::from_u128(4242), 7, "anything")],
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(
            report.cases[0].citation_verdicts,
            vec![CitationVerdict::UnknownChunk]
        );
        assert_eq!(report.overall.citations_emitted, 1);
        assert_eq!(
            report.overall.citation_verdicts.get("unknown_chunk"),
            Some(&1)
        );
    }

    #[test]
    fn answering_an_unanswerable_question_is_a_failure() {
        let corpus = corpus();
        let query = corpus
            .queries(Split::Train)
            .into_iter()
            .find(|q| !q.answerable)
            .expect("corpus has an unanswerable training query");

        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: false,
            outcome: AnswerOutcome::Answered {
                text: "The deadline is thirty days.".to_owned(),
                citations: Vec::new(),
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(
            report.cases[0].classification,
            CaseClassification::MissingAbstention
        );
        assert!(!report.cases[0].abstention_correct);
        assert!(report.overall.abstention_accuracy.abs() < f64::EPSILON);
    }

    #[test]
    fn abstaining_with_sufficient_evidence_is_also_a_failure() {
        let corpus = corpus();
        let (query, _, _, _) = answerable_case(&corpus);

        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: true,
            outcome: AnswerOutcome::Abstained {
                text: "Les sources disponibles ne permettent pas de répondre.".to_owned(),
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(
            report.cases[0].classification,
            CaseClassification::SpuriousAbstention
        );
    }

    #[test]
    fn insufficient_retrieved_evidence_makes_abstention_the_correct_answer() {
        let corpus = corpus();
        let (query, _, _, _) = answerable_case(&corpus);

        // Answerable by the corpus, but retrieval surfaced nothing relevant.
        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: false,
            outcome: AnswerOutcome::Abstained {
                text: "Aucune source n'est disponible dans ce notebook.".to_owned(),
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(report.cases[0].classification, CaseClassification::Ok);
        assert!(report.cases[0].abstention_correct);
    }

    #[test]
    fn a_provider_error_stays_in_the_denominator_with_a_classification() {
        let corpus = corpus();
        let (query, _, _, _) = answerable_case(&corpus);

        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: true,
            outcome: AnswerOutcome::ProviderError {
                reason: "upstream timeout".to_owned(),
            },
        };

        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(report.overall.cases, 1, "the case must not be dropped");
        assert_eq!(
            report.overall.classifications.get("provider_error"),
            Some(&1)
        );
        assert_eq!(report.overall.expected_claim_coverage, 0.0);
    }

    #[test]
    fn an_unknown_query_id_is_reported_rather_than_dropped() {
        let corpus = corpus();
        let case = AnswerCase {
            query_id: "q-not-in-the-corpus".to_owned(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: true,
            outcome: AnswerOutcome::Abstained {
                text: String::new(),
            },
        };
        let report = run_grounding_eval(&corpus, &[case], &config(), None, FIXED_TIME);
        assert_eq!(report.overall.cases, 1);
        assert_eq!(
            report.cases[0].classification,
            CaseClassification::UnknownQuery
        );
    }

    #[test]
    fn no_metric_is_ever_nan() {
        let corpus = corpus();
        let report = run_grounding_eval(&corpus, &[], &config(), None, FIXED_TIME);
        for metrics in std::iter::once(&report.overall).chain(report.by_category.values()) {
            for value in [
                metrics.expected_claim_coverage,
                metrics.citation_precision,
                metrics.citation_coverage,
                metrics.abstention_accuracy,
            ] {
                assert!(value.is_finite());
            }
        }
        assert!(
            !serde_json::to_string(&report)
                .expect("json")
                .contains("NaN")
        );
    }

    /// A judge that condemns everything. If it could move a metric, it would.
    struct HostileJudge;

    impl ClaimJudge for HostileJudge {
        fn name(&self) -> &str {
            "hostile"
        }

        fn diagnose(&self, _query: &EvalQuery, _answer: &str) -> Vec<String> {
            vec!["every claim is unsupported and every citation is a lie".to_owned()]
        }
    }

    #[test]
    fn a_judge_adds_diagnostics_and_moves_no_metric() {
        let corpus = corpus();
        let (query, content, chunk_index, source_id) = answerable_case(&corpus);
        let case = AnswerCase {
            query_id: query.id.clone(),
            generation_ids: vec![corpus.generation_id()],
            evidence_sufficient: true,
            outcome: AnswerOutcome::Answered {
                text: perfect_answer(query),
                citations: vec![citation(source_id, chunk_index, &content)],
            },
        };

        let without = run_grounding_eval(
            &corpus,
            std::slice::from_ref(&case),
            &config(),
            None,
            FIXED_TIME,
        );
        let judge = HostileJudge;
        let with = run_grounding_eval(&corpus, &[case], &config(), Some(&judge), FIXED_TIME);

        assert_eq!(without.overall, with.overall, "a judge decided a metric");
        assert!(without.cases[0].diagnostics.is_empty());
        assert!(!with.cases[0].diagnostics.is_empty());
        assert_eq!(with.judge, "hostile");
    }

    #[tokio::test]
    async fn the_deterministic_pipeline_produces_one_case_per_query_offline() {
        let corpus = corpus();
        let index = CorpusIndex::build(&corpus).await.expect("index");
        let config = RetrievalRunConfig {
            mode: RetrievalMode::Hybrid,
            code_revision: "test".to_owned(),
            ..RetrievalRunConfig::default()
        };

        let cases = answer_with_deterministic_pipeline(&corpus, &index, &config)
            .await
            .expect("producer");
        assert_eq!(cases.len(), corpus.queries(Split::Train).len());

        let report = run_grounding_eval(&corpus, &cases, &self::config(), None, FIXED_TIME);
        assert_eq!(report.overall.cases, cases.len());
        // Whatever the numbers, every case is classified.
        assert_eq!(
            report.overall.classifications.values().sum::<usize>(),
            cases.len()
        );
    }

    #[tokio::test]
    async fn the_deterministic_grounding_report_is_byte_stable() {
        let corpus = corpus();
        let index = CorpusIndex::build(&corpus).await.expect("index");
        let retrieval = RetrievalRunConfig {
            code_revision: "test".to_owned(),
            ..RetrievalRunConfig::default()
        };

        let first = answer_with_deterministic_pipeline(&corpus, &index, &retrieval)
            .await
            .expect("producer");
        let second = answer_with_deterministic_pipeline(&corpus, &index, &retrieval)
            .await
            .expect("producer");

        let a = run_grounding_eval(&corpus, &first, &config(), None, FIXED_TIME);
        let b = run_grounding_eval(&corpus, &second, &config(), None, FIXED_TIME);
        assert_eq!(
            serde_json::to_string_pretty(&a).expect("json"),
            serde_json::to_string_pretty(&b).expect("json")
        );
    }

    #[test]
    fn abstention_markers_cover_the_documented_fallbacks() {
        assert!(reads_as_abstention(
            "Aucune source n'est disponible dans ce notebook."
        ));
        assert!(reads_as_abstention(
            "Les sources disponibles ne permettent pas de répondre avec suffisamment de confiance."
        ));
        assert!(reads_as_abstention(
            "No source was retrieved for this question."
        ));
        assert!(!reads_as_abstention("The retry budget is four attempts."));
    }
}
