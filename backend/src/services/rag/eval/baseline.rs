//! Baselines and the release gate (US-004).
//!
//! A metric with nothing to compare against is a number. This module turns two
//! runs into a decision: capture one as a [`Baseline`], compare the next one,
//! and exit non-zero when something got worse.
//!
//! # Regression and target are different failures
//!
//! The PRD's absolute targets (Recall@10 >= 0.90, citation precision >= 0.95)
//! are month-6 goals against a real embedding provider. The first baseline
//! captured on the deterministic provider will not meet them, and a gate that
//! refused to record it would leave the project with no baseline at all.
//!
//! So [`Enforcement`] separates the two. [`Enforcement::RegressionOnly`] blocks
//! on "worse than last time" and reports unmet targets as information.
//! [`Enforcement::RegressionAndTargets`] blocks on both. A release owner moves
//! between them deliberately; nothing infers it.
//!
//! # What always blocks
//!
//! A tenant-isolation failure and a missing required metric block in both modes.
//! Neither is a quality trade-off: one is a leak, the other is a report that
//! cannot be read.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::grounding::{GroundingMetrics, GroundingReport};
use super::retrieval::{FusionReport, RetrievalMetrics, RetrievalReport};

/// Bumped whenever the artifact shape changes. Comparison across schema
/// versions is refused rather than silently misaligned.
pub const BASELINE_SCHEMA: &str = "rag-eval-baseline/1";

/// How far a metric may fall before it counts as a regression.
///
/// 0.02 absolute, from the PRD. Applied to Recall@10, nDCG@10, citation
/// precision and citation coverage, overall and per category.
pub const REGRESSION_TOLERANCE: f64 = 0.02;

// ============================================================================
// Targets
// ============================================================================

/// Absolute quality goals.
///
/// The PRD's month-6 numbers. A release owner changes these through a
/// corpus-versioned decision record; implementation code must not tune them
/// silently, which is why they are constants with one named override point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Targets {
    pub recall_at_10: f64,
    pub ndcg_at_10: f64,
    pub citation_precision: f64,
    pub citation_coverage: f64,
    pub abstention_accuracy: f64,
}

impl Default for Targets {
    fn default() -> Self {
        Self {
            recall_at_10: 0.90,
            ndcg_at_10: 0.75,
            citation_precision: 0.95,
            citation_coverage: 0.90,
            abstention_accuracy: 0.90,
        }
    }
}

/// Which failures block a release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Enforcement {
    /// Block on regressions, isolation failures and missing metrics. Report
    /// unmet targets without blocking.
    RegressionOnly,
    /// Also block on unmet absolute targets.
    RegressionAndTargets,
}

impl Enforcement {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RegressionOnly => "regression_only",
            Self::RegressionAndTargets => "regression_and_targets",
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "regression_only" => Some(Self::RegressionOnly),
            "regression_and_targets" => Some(Self::RegressionAndTargets),
            _ => None,
        }
    }
}

impl fmt::Display for Enforcement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Baseline artifact
// ============================================================================

/// The retrieval half of a baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalSection {
    pub mode: String,
    pub split: String,
    pub requested_limit: i32,
    pub fusion: FusionReport,
    pub overall: RetrievalMetrics,
    pub by_category: BTreeMap<String, RetrievalMetrics>,
}

/// The grounded-response half of a baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundingSection {
    pub split: String,
    pub overall: GroundingMetrics,
    pub by_category: BTreeMap<String, GroundingMetrics>,
}

/// One approved measurement of the system.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Baseline {
    pub schema: String,
    /// The one field excluded from the byte-stability contract.
    pub generated_at: String,
    pub corpus_version: String,
    pub corpus_generation: String,
    pub chunking_version: String,
    pub code_revision: String,
    pub embedding_fingerprint: String,
    pub reranker_fingerprint: String,
    pub retrieval: RetrievalSection,
    pub grounding: GroundingSection,
    pub notes: Vec<String>,
}

impl Baseline {
    /// Build a baseline from one retrieval run and one grounded-response run.
    ///
    /// The two must describe the same corpus: comparing a retrieval report
    /// against a grounding report from a different corpus version would produce
    /// an artifact that looks coherent and is not.
    ///
    /// # Errors
    /// Returns a description of the mismatch when the two reports disagree
    /// about the corpus, the generation or the code revision.
    pub fn capture(
        retrieval: &RetrievalReport,
        grounding: &GroundingReport,
    ) -> Result<Self, String> {
        if retrieval.corpus_version != grounding.corpus_version {
            return Err(format!(
                "retrieval measured corpus {} and grounding measured {}",
                retrieval.corpus_version, grounding.corpus_version
            ));
        }
        if retrieval.corpus_generation != grounding.corpus_generation {
            return Err(format!(
                "retrieval read generation {} and grounding read {}",
                retrieval.corpus_generation, grounding.corpus_generation
            ));
        }
        if retrieval.code_revision != grounding.code_revision {
            return Err(format!(
                "retrieval measured revision {} and grounding measured {}",
                retrieval.code_revision, grounding.code_revision
            ));
        }

        let mut notes = retrieval.notes.clone();
        notes.extend(grounding.notes.iter().cloned());

        Ok(Self {
            schema: BASELINE_SCHEMA.to_owned(),
            generated_at: retrieval.generated_at.clone(),
            corpus_version: retrieval.corpus_version.clone(),
            corpus_generation: retrieval.corpus_generation.clone(),
            chunking_version: retrieval.chunking_version.clone(),
            code_revision: retrieval.code_revision.clone(),
            embedding_fingerprint: retrieval.embedding_fingerprint.clone(),
            reranker_fingerprint: retrieval.reranker_fingerprint.clone(),
            retrieval: RetrievalSection {
                mode: retrieval.mode.clone(),
                split: retrieval.split.clone(),
                requested_limit: retrieval.requested_limit,
                fusion: retrieval.fusion.clone(),
                overall: retrieval.overall.clone(),
                by_category: retrieval.by_category.clone(),
            },
            grounding: GroundingSection {
                split: grounding.split.clone(),
                overall: grounding.overall.clone(),
                by_category: grounding.by_category.clone(),
            },
            notes,
        })
    }
}

// ============================================================================
// Required metrics
// ============================================================================

/// JSON pointers a baseline must carry to be comparable at all.
///
/// Checked against the raw document rather than the parsed struct: a baseline
/// written by an older revision can deserialize with defaults and look complete,
/// and "any required metric is missing" has to catch exactly that.
const REQUIRED_POINTERS: &[&str] = &[
    "/schema",
    "/corpus_version",
    "/code_revision",
    "/embedding_fingerprint",
    "/chunking_version",
    "/retrieval/mode",
    "/retrieval/split",
    "/retrieval/requested_limit",
    "/retrieval/fusion",
    "/retrieval/overall/recall_at_10",
    "/retrieval/overall/ndcg_at_10",
    "/retrieval/overall/isolation_failures",
    "/grounding/overall/citation_precision",
    "/grounding/overall/citation_coverage",
    "/grounding/overall/abstention_accuracy",
];

/// Required pointers absent from a baseline document.
#[must_use]
pub fn missing_metrics(document: &serde_json::Value) -> Vec<String> {
    REQUIRED_POINTERS
        .iter()
        .filter(|pointer| document.pointer(pointer).is_none())
        .map(|pointer| (*pointer).to_owned())
        .collect()
}

// ============================================================================
// Comparison
// ============================================================================

/// One metric that moved.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Delta {
    /// `retrieval.overall.recall_at_10`, `grounding.multi_hop.citation_precision`, …
    pub metric: String,
    pub previous: f64,
    pub current: f64,
    pub delta: f64,
}

/// One absolute target that is not met.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TargetMiss {
    pub metric: String,
    pub value: f64,
    pub target: f64,
}

/// The result of comparing a candidate against an approved baseline.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ComparisonReport {
    pub schema: String,
    pub enforcement: Enforcement,
    pub tolerance: f64,
    pub previous_revision: String,
    pub current_revision: String,
    /// Reasons the two artifacts cannot be compared at all.
    pub incomparable: Vec<String>,
    /// Required pointers absent from one of the two documents.
    pub missing_metrics: Vec<String>,
    /// Metrics that fell by more than the tolerance.
    pub regressions: Vec<Delta>,
    /// Metrics that rose by more than the tolerance. Informational.
    pub improvements: Vec<Delta>,
    /// Populations whose result sets contained a forbidden source.
    pub isolation_failures: Vec<String>,
    /// Absolute goals not yet reached.
    pub unmet_targets: Vec<TargetMiss>,
}

impl ComparisonReport {
    /// Whether this comparison blocks a release.
    ///
    /// Isolation failures, missing metrics and incomparability block in every
    /// mode. Unmet targets block only when the release owner asked for it.
    #[must_use]
    pub fn blocking(&self) -> bool {
        !self.incomparable.is_empty()
            || !self.missing_metrics.is_empty()
            || !self.regressions.is_empty()
            || !self.isolation_failures.is_empty()
            || (self.enforcement == Enforcement::RegressionAndTargets
                && !self.unmet_targets.is_empty())
    }

    /// A short human summary, for a CI log.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.blocking() {
            format!(
                "BLOCKED: {} incomparable, {} missing, {} regressions, {} isolation failures, \
                 {} unmet targets ({})",
                self.incomparable.len(),
                self.missing_metrics.len(),
                self.regressions.len(),
                self.isolation_failures.len(),
                self.unmet_targets.len(),
                self.enforcement
            )
        } else {
            format!(
                "OK: no regression beyond {:.2}, {} improvements, {} unmet targets (not enforced \
                 in {})",
                self.tolerance,
                self.improvements.len(),
                self.unmet_targets.len(),
                self.enforcement
            )
        }
    }
}

/// Metrics compared for regression, with their extractors.
///
/// Only the four the PRD names. Adding one here changes what blocks a release,
/// which is a decision and not a refactor.
fn retrieval_pairs(metrics: &RetrievalMetrics) -> [(&'static str, f64); 2] {
    [
        ("recall_at_10", metrics.recall_at_10),
        ("ndcg_at_10", metrics.ndcg_at_10),
    ]
}

fn grounding_pairs(metrics: &GroundingMetrics) -> [(&'static str, f64); 2] {
    [
        ("citation_precision", metrics.citation_precision),
        ("citation_coverage", metrics.citation_coverage),
    ]
}

/// Compare a candidate baseline against an approved one.
#[must_use]
pub fn compare(
    previous: &Baseline,
    current: &Baseline,
    enforcement: Enforcement,
    targets: &Targets,
) -> ComparisonReport {
    let mut incomparable = Vec::new();

    // Two measurements of different things are not a comparison. Say so rather
    // than reporting a delta nobody can act on.
    let mut require_same = |field: &str, a: &str, b: &str| {
        if a != b {
            incomparable.push(format!("{field}: baseline `{a}`, candidate `{b}`"));
        }
    };
    require_same("schema", &previous.schema, &current.schema);
    require_same(
        "corpus_version",
        &previous.corpus_version,
        &current.corpus_version,
    );
    require_same(
        "chunking_version",
        &previous.chunking_version,
        &current.chunking_version,
    );
    require_same(
        "embedding_fingerprint",
        &previous.embedding_fingerprint,
        &current.embedding_fingerprint,
    );
    require_same(
        "reranker_fingerprint",
        &previous.reranker_fingerprint,
        &current.reranker_fingerprint,
    );
    require_same(
        "retrieval.mode",
        &previous.retrieval.mode,
        &current.retrieval.mode,
    );
    require_same(
        "retrieval.split",
        &previous.retrieval.split,
        &current.retrieval.split,
    );
    require_same(
        "grounding.split",
        &previous.grounding.split,
        &current.grounding.split,
    );
    if previous.retrieval.requested_limit != current.retrieval.requested_limit {
        incomparable.push(format!(
            "retrieval.requested_limit: baseline {}, candidate {}",
            previous.retrieval.requested_limit, current.retrieval.requested_limit
        ));
    }
    if previous.retrieval.fusion != current.retrieval.fusion {
        incomparable.push("retrieval.fusion: fusion parameters differ".to_owned());
    }

    let mut regressions = Vec::new();
    let mut improvements = Vec::new();
    let mut missing_metrics = Vec::new();

    let mut record = |metric: String, before: f64, after: f64| {
        let delta = round6(after - before);
        if delta < -REGRESSION_TOLERANCE {
            regressions.push(Delta {
                metric,
                previous: before,
                current: after,
                delta,
            });
        } else if delta > REGRESSION_TOLERANCE {
            improvements.push(Delta {
                metric,
                previous: before,
                current: after,
                delta,
            });
        }
    };

    for (name, before) in retrieval_pairs(&previous.retrieval.overall) {
        let after = retrieval_pairs(&current.retrieval.overall)
            .into_iter()
            .find(|(n, _)| *n == name)
            .map_or(0.0, |(_, v)| v);
        record(format!("retrieval.overall.{name}"), before, after);
    }
    for (name, before) in grounding_pairs(&previous.grounding.overall) {
        let after = grounding_pairs(&current.grounding.overall)
            .into_iter()
            .find(|(n, _)| *n == name)
            .map_or(0.0, |(_, v)| v);
        record(format!("grounding.overall.{name}"), before, after);
    }

    // Per category. A category that disappears from the candidate is a missing
    // metric, not a silent pass.
    for (category, before_metrics) in &previous.retrieval.by_category {
        let Some(after_metrics) = current.retrieval.by_category.get(category) else {
            missing_metrics.push(format!("retrieval.by_category.{category}"));
            continue;
        };
        // A category with no judged query in either run carries no signal;
        // comparing its zeros would manufacture noise.
        if before_metrics.judged_queries == 0 && after_metrics.judged_queries == 0 {
            continue;
        }
        for ((name, before), (_, after)) in retrieval_pairs(before_metrics)
            .into_iter()
            .zip(retrieval_pairs(after_metrics))
        {
            record(format!("retrieval.{category}.{name}"), before, after);
        }
    }
    for (category, before_metrics) in &previous.grounding.by_category {
        let Some(after_metrics) = current.grounding.by_category.get(category) else {
            missing_metrics.push(format!("grounding.by_category.{category}"));
            continue;
        };
        if before_metrics.cases == 0 && after_metrics.cases == 0 {
            continue;
        }
        for ((name, before), (_, after)) in grounding_pairs(before_metrics)
            .into_iter()
            .zip(grounding_pairs(after_metrics))
        {
            record(format!("grounding.{category}.{name}"), before, after);
        }
    }

    // Isolation is not a quality dimension; any failure blocks.
    let mut isolation_failures = Vec::new();
    if current.retrieval.overall.isolation_failures > 0 {
        isolation_failures.push(format!(
            "retrieval.overall: {} case(s) retrieved a forbidden source",
            current.retrieval.overall.isolation_failures
        ));
    }
    for (category, metrics) in &current.retrieval.by_category {
        if metrics.isolation_failures > 0 {
            isolation_failures.push(format!(
                "retrieval.{category}: {} case(s) retrieved a forbidden source",
                metrics.isolation_failures
            ));
        }
    }
    for (category, metrics) in &current.grounding.by_category {
        let leaked = metrics
            .citation_verdicts
            .get("cross_notebook")
            .copied()
            .unwrap_or(0);
        if leaked > 0 {
            isolation_failures.push(format!(
                "grounding.{category}: {leaked} citation(s) into another notebook"
            ));
        }
    }

    let unmet_targets = unmet(current, targets);

    // Sorted so a comparison report is itself byte-stable.
    regressions.sort_by(|a, b| a.metric.cmp(&b.metric));
    improvements.sort_by(|a, b| a.metric.cmp(&b.metric));
    missing_metrics.sort();
    isolation_failures.sort();
    incomparable.sort();

    ComparisonReport {
        schema: BASELINE_SCHEMA.to_owned(),
        enforcement,
        tolerance: REGRESSION_TOLERANCE,
        previous_revision: previous.code_revision.clone(),
        current_revision: current.code_revision.clone(),
        incomparable,
        missing_metrics,
        regressions,
        improvements,
        isolation_failures,
        unmet_targets,
    }
}

/// Absolute targets a baseline does not meet.
#[must_use]
pub fn unmet(baseline: &Baseline, targets: &Targets) -> Vec<TargetMiss> {
    let mut misses = Vec::new();
    let mut check = |metric: &str, value: f64, target: f64| {
        if value + f64::EPSILON < target {
            misses.push(TargetMiss {
                metric: metric.to_owned(),
                value,
                target,
            });
        }
    };
    check(
        "retrieval.overall.recall_at_10",
        baseline.retrieval.overall.recall_at_10,
        targets.recall_at_10,
    );
    check(
        "retrieval.overall.ndcg_at_10",
        baseline.retrieval.overall.ndcg_at_10,
        targets.ndcg_at_10,
    );
    check(
        "grounding.overall.citation_precision",
        baseline.grounding.overall.citation_precision,
        targets.citation_precision,
    );
    check(
        "grounding.overall.citation_coverage",
        baseline.grounding.overall.citation_coverage,
        targets.citation_coverage,
    );
    check(
        "grounding.overall.abstention_accuracy",
        baseline.grounding.overall.abstention_accuracy,
        targets.abstention_accuracy,
    );
    misses
}

fn round6(value: f64) -> f64 {
    if value.is_finite() {
        (value * 1_000_000.0).round() / 1_000_000.0
    } else {
        0.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn retrieval_metrics(recall: f64, ndcg: f64) -> RetrievalMetrics {
        RetrievalMetrics {
            queries: 10,
            judged_queries: 8,
            recall_at_10: recall,
            ndcg_at_10: ndcg,
            ..RetrievalMetrics::default()
        }
    }

    fn grounding_metrics(precision: f64, coverage: f64) -> GroundingMetrics {
        GroundingMetrics {
            cases: 10,
            citation_precision: precision,
            citation_coverage: coverage,
            abstention_accuracy: 1.0,
            ..GroundingMetrics::default()
        }
    }

    fn baseline(recall: f64, ndcg: f64, precision: f64, coverage: f64) -> Baseline {
        Baseline {
            schema: BASELINE_SCHEMA.to_owned(),
            generated_at: "2026-07-31T00:00:00Z".to_owned(),
            corpus_version: "1.0.0".to_owned(),
            corpus_generation: "gen-1".to_owned(),
            chunking_version: "chunking-v1".to_owned(),
            code_revision: "rev-a".to_owned(),
            embedding_fingerprint: "deterministic/hash/1024".to_owned(),
            reranker_fingerprint: "none".to_owned(),
            retrieval: RetrievalSection {
                mode: "hybrid".to_owned(),
                split: "train".to_owned(),
                requested_limit: 10,
                fusion: FusionReport {
                    enabled: true,
                    rrf_k: 60.0,
                    dense_weight: 1.0,
                    sparse_weight: 0.25,
                },
                overall: retrieval_metrics(recall, ndcg),
                by_category: BTreeMap::from([(
                    "multi_hop".to_owned(),
                    retrieval_metrics(recall, ndcg),
                )]),
            },
            grounding: GroundingSection {
                split: "train".to_owned(),
                overall: grounding_metrics(precision, coverage),
                by_category: BTreeMap::from([(
                    "multi_hop".to_owned(),
                    grounding_metrics(precision, coverage),
                )]),
            },
            notes: Vec::new(),
        }
    }

    #[test]
    fn an_unchanged_candidate_does_not_block() {
        let previous = baseline(0.8, 0.7, 0.9, 0.85);
        let current = baseline(0.8, 0.7, 0.9, 0.85);
        let report = compare(
            &previous,
            &current,
            Enforcement::RegressionOnly,
            &Targets::default(),
        );
        assert!(!report.blocking(), "{report:?}");
        assert!(report.regressions.is_empty());
    }

    #[test]
    fn a_drop_inside_the_tolerance_does_not_block() {
        let previous = baseline(0.80, 0.70, 0.90, 0.85);
        let current = baseline(0.785, 0.69, 0.89, 0.84);
        let report = compare(
            &previous,
            &current,
            Enforcement::RegressionOnly,
            &Targets::default(),
        );
        assert!(report.regressions.is_empty(), "{:?}", report.regressions);
        assert!(!report.blocking());
    }

    #[test]
    fn recall_falling_past_the_tolerance_blocks() {
        let previous = baseline(0.80, 0.70, 0.90, 0.85);
        let current = baseline(0.75, 0.70, 0.90, 0.85);
        let report = compare(
            &previous,
            &current,
            Enforcement::RegressionOnly,
            &Targets::default(),
        );
        assert!(report.blocking());
        assert!(
            report
                .regressions
                .iter()
                .any(|d| d.metric == "retrieval.overall.recall_at_10"),
            "{:?}",
            report.regressions
        );
        // The category carrying the same drop is reported too.
        assert!(
            report
                .regressions
                .iter()
                .any(|d| d.metric == "retrieval.multi_hop.recall_at_10")
        );
    }

    #[test]
    fn ndcg_citation_precision_and_coverage_each_block_on_their_own() {
        for (recall, ndcg, precision, coverage, expected) in [
            (0.80, 0.60, 0.90, 0.85, "retrieval.overall.ndcg_at_10"),
            (
                0.80,
                0.70,
                0.80,
                0.85,
                "grounding.overall.citation_precision",
            ),
            (
                0.80,
                0.70,
                0.90,
                0.70,
                "grounding.overall.citation_coverage",
            ),
        ] {
            let report = compare(
                &baseline(0.80, 0.70, 0.90, 0.85),
                &baseline(recall, ndcg, precision, coverage),
                Enforcement::RegressionOnly,
                &Targets::default(),
            );
            assert!(
                report.regressions.iter().any(|d| d.metric == expected),
                "{expected} did not block: {:?}",
                report.regressions
            );
            assert!(report.blocking());
        }
    }

    #[test]
    fn a_tenant_isolation_failure_blocks_in_every_mode() {
        let previous = baseline(0.80, 0.70, 0.90, 0.85);
        let mut current = baseline(0.80, 0.70, 0.90, 0.85);
        current.retrieval.overall.isolation_failures = 1;

        for enforcement in [
            Enforcement::RegressionOnly,
            Enforcement::RegressionAndTargets,
        ] {
            let report = compare(&previous, &current, enforcement, &Targets::default());
            assert!(!report.isolation_failures.is_empty());
            assert!(report.blocking(), "{enforcement} let a leak through");
        }
    }

    #[test]
    fn a_cross_notebook_citation_blocks_too() {
        let previous = baseline(0.80, 0.70, 0.90, 0.85);
        let mut current = baseline(0.80, 0.70, 0.90, 0.85);
        current
            .grounding
            .by_category
            .get_mut("multi_hop")
            .expect("category")
            .citation_verdicts
            .insert("cross_notebook".to_owned(), 2);

        let report = compare(
            &previous,
            &current,
            Enforcement::RegressionOnly,
            &Targets::default(),
        );
        assert!(report.blocking());
        assert!(
            report
                .isolation_failures
                .iter()
                .any(|f| f.contains("cross_notebook") || f.contains("another notebook"))
        );
    }

    #[test]
    fn an_unmet_target_is_reported_but_only_blocks_when_enforced() {
        // Far below every target, which is where the first deterministic
        // baseline actually sits.
        let low = baseline(0.30, 0.25, 0.40, 0.30);
        let report = compare(&low, &low, Enforcement::RegressionOnly, &Targets::default());
        assert_eq!(report.unmet_targets.len(), 4);
        assert!(
            !report.blocking(),
            "capturing a first baseline below target must be possible"
        );

        let enforced = compare(
            &low,
            &low,
            Enforcement::RegressionAndTargets,
            &Targets::default(),
        );
        assert!(enforced.blocking());
        assert!(
            enforced.regressions.is_empty(),
            "targets are not regressions"
        );
    }

    #[test]
    fn a_category_missing_from_the_candidate_is_a_missing_metric() {
        let previous = baseline(0.80, 0.70, 0.90, 0.85);
        let mut current = baseline(0.80, 0.70, 0.90, 0.85);
        current.retrieval.by_category.clear();
        current.grounding.by_category.clear();

        let report = compare(
            &previous,
            &current,
            Enforcement::RegressionOnly,
            &Targets::default(),
        );
        assert_eq!(
            report.missing_metrics,
            vec![
                "grounding.by_category.multi_hop",
                "retrieval.by_category.multi_hop"
            ]
        );
        assert!(report.blocking());
    }

    #[test]
    fn a_required_pointer_absent_from_the_document_is_detected() {
        let complete = serde_json::to_value(baseline(0.8, 0.7, 0.9, 0.85)).expect("json");
        assert!(missing_metrics(&complete).is_empty());

        let mut stripped = complete;
        stripped
            .pointer_mut("/grounding/overall")
            .and_then(serde_json::Value::as_object_mut)
            .expect("object")
            .remove("citation_precision");
        assert_eq!(
            missing_metrics(&stripped),
            vec!["/grounding/overall/citation_precision"]
        );
    }

    #[test]
    fn a_different_corpus_or_configuration_is_incomparable() {
        let previous = baseline(0.80, 0.70, 0.90, 0.85);

        let mut other_corpus = baseline(0.80, 0.70, 0.90, 0.85);
        other_corpus.corpus_version = "2.0.0".to_owned();

        let mut other_limit = baseline(0.80, 0.70, 0.90, 0.85);
        other_limit.retrieval.requested_limit = 20;

        let mut other_fusion = baseline(0.80, 0.70, 0.90, 0.85);
        other_fusion.retrieval.fusion.rrf_k = 30.0;

        for candidate in [other_corpus, other_limit, other_fusion] {
            let report = compare(
                &previous,
                &candidate,
                Enforcement::RegressionOnly,
                &Targets::default(),
            );
            assert!(!report.incomparable.is_empty());
            assert!(report.blocking());
        }
    }

    #[test]
    fn an_empty_category_on_both_sides_produces_no_noise() {
        let mut previous = baseline(0.80, 0.70, 0.90, 0.85);
        let mut current = baseline(0.80, 0.70, 0.90, 0.85);
        for base in [&mut previous, &mut current] {
            let metrics = base
                .retrieval
                .by_category
                .get_mut("multi_hop")
                .expect("category");
            metrics.judged_queries = 0;
            metrics.recall_at_10 = 0.0;
            metrics.ndcg_at_10 = 0.0;
        }
        // The overall numbers still differ from zero, so this isolates the
        // category rule.
        let report = compare(
            &previous,
            &current,
            Enforcement::RegressionOnly,
            &Targets::default(),
        );
        assert!(
            report
                .regressions
                .iter()
                .all(|d| !d.metric.starts_with("retrieval.multi_hop")),
            "{:?}",
            report.regressions
        );
    }

    #[test]
    fn a_comparison_report_is_itself_deterministic() {
        let previous = baseline(0.90, 0.80, 0.95, 0.90);
        let current = baseline(0.50, 0.40, 0.50, 0.40);
        let a = compare(
            &previous,
            &current,
            Enforcement::RegressionAndTargets,
            &Targets::default(),
        );
        let b = compare(
            &previous,
            &current,
            Enforcement::RegressionAndTargets,
            &Targets::default(),
        );
        assert_eq!(
            serde_json::to_string(&a).expect("json"),
            serde_json::to_string(&b).expect("json")
        );
        assert!(a.summary().starts_with("BLOCKED"));
    }

    #[test]
    fn improvements_are_reported_without_blocking() {
        let previous = baseline(0.50, 0.40, 0.50, 0.40);
        let current = baseline(0.90, 0.80, 0.95, 0.90);
        let report = compare(
            &previous,
            &current,
            Enforcement::RegressionOnly,
            &Targets::default(),
        );
        assert!(!report.blocking());
        assert!(!report.improvements.is_empty());
        assert!(report.summary().starts_with("OK"));
    }

    #[test]
    fn enforcement_modes_round_trip_through_their_wire_names() {
        for mode in [
            Enforcement::RegressionOnly,
            Enforcement::RegressionAndTargets,
        ] {
            assert_eq!(Enforcement::parse(mode.as_str()), Some(mode));
        }
        assert_eq!(Enforcement::parse("strict"), None);
    }

    #[test]
    fn targets_are_the_prd_numbers() {
        let targets = Targets::default();
        assert!((targets.recall_at_10 - 0.90).abs() < f64::EPSILON);
        assert!((targets.ndcg_at_10 - 0.75).abs() < f64::EPSILON);
        assert!((targets.citation_precision - 0.95).abs() < f64::EPSILON);
        assert!((targets.citation_coverage - 0.90).abs() < f64::EPSILON);
        assert!((targets.abstention_accuracy - 0.90).abs() < f64::EPSILON);
        assert!((REGRESSION_TOLERANCE - 0.02).abs() < f64::EPSILON);
    }

    #[test]
    fn a_baseline_round_trips_through_json() {
        let original = baseline(0.8, 0.7, 0.9, 0.85);
        let rendered = serde_json::to_string(&original).expect("json");
        let parsed: Baseline = serde_json::from_str(&rendered).expect("parse");
        assert_eq!(original, parsed);
    }
}
