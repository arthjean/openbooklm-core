//! The RAG evaluation contract, end to end (EP-001).
//!
//! Unit tests inside `services::rag::eval` cover each metric and each failure
//! branch. This file covers the properties that only exist across the whole
//! thing: the checked-in corpus is valid and synthetic, two runs at one revision
//! produce identical bytes, the committed baselines still describe this code,
//! and the release gate blocks what it says it blocks.
//!
//! ## Regenerating the baselines
//!
//! ```bash
//! cd backend
//! UPDATE_BASELINE=1 cargo test --no-default-features --test rag_eval
//! git diff ../contracts/eval/baseline   # review every line: this is a quality change
//! ```
//!
//! ## Offline by construction
//!
//! Nothing here opens a socket. The corpus is a checked-in fixture, retrieval
//! runs in memory, and generation uses the in-process deterministic model
//! (FR-20).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use openbooklm::services::rag::eval::baseline::{
    Baseline, Enforcement, Targets, compare, missing_metrics, unmet,
};
use openbooklm::services::rag::eval::corpus::{EvalCorpus, QueryCategory, Split};
use openbooklm::services::rag::eval::grounding::{
    GroundingRunConfig, answer_with_deterministic_pipeline, run_grounding_eval,
};
use openbooklm::services::rag::eval::index::CorpusIndex;
use openbooklm::services::rag::eval::retrieval::{
    RetrievalMode, RetrievalRunConfig, run_retrieval_eval,
};

/// Fixed so that the timestamp — the one field excluded from the byte-stability
/// contract — never moves in a committed artifact.
const FIXED_TIME: &str = "unspecified";

/// Fixed so that a baseline diff reflects a behavior change and not a version
/// bump. The real revision belongs to whoever runs the gate in CI.
const FIXED_REVISION: &str = "corpus-v1-deterministic";

fn baseline_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("backend/ always has a parent")
        .join("contracts/eval/baseline")
}

fn updating() -> bool {
    std::env::var("UPDATE_BASELINE").is_ok_and(|v| v != "0" && !v.is_empty())
}

fn corpus() -> EvalCorpus {
    EvalCorpus::load_default().expect("the checked-in corpus loads")
}

fn config(split: Split) -> RetrievalRunConfig {
    RetrievalRunConfig {
        mode: RetrievalMode::Hybrid,
        split,
        limit: 10,
        code_revision: FIXED_REVISION.to_owned(),
        ..RetrievalRunConfig::default()
    }
}

/// Produce the baseline for one split, through the same path the binary uses.
async fn capture(split: Split) -> Baseline {
    let corpus = corpus();
    let index = CorpusIndex::build(&corpus).await.expect("index builds");
    let retrieval = config(split);

    let run = run_retrieval_eval(&corpus, &index, &retrieval, FIXED_TIME)
        .await
        .expect("retrieval evaluation");
    let cases = answer_with_deterministic_pipeline(&corpus, &index, &retrieval)
        .await
        .expect("answer production");
    let grounding = run_grounding_eval(
        &corpus,
        &cases,
        &GroundingRunConfig {
            split,
            code_revision: FIXED_REVISION.to_owned(),
        },
        None,
        FIXED_TIME,
    );

    Baseline::capture(&run.report, &grounding).expect("the two reports agree")
}

fn render(value: &impl serde::Serialize) -> String {
    let mut rendered = serde_json::to_string_pretty(value).expect("json");
    rendered.push('\n');
    rendered
}

// ============================================================================
// US-001 — the corpus
// ============================================================================

#[test]
fn the_checked_in_corpus_has_no_violation() {
    let corpus = corpus();
    let violations = corpus.validate();
    assert!(
        violations.is_empty(),
        "the corpus that gates releases must itself be valid:\n{}",
        violations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn the_corpus_meets_the_prd_coverage_bar() {
    let corpus = corpus();
    assert!(
        corpus.all_queries().len() >= 40,
        "PRD requires at least 40 labeled queries, found {}",
        corpus.all_queries().len()
    );
    for (category, count) in corpus.cases_per_category() {
        assert!(
            count >= 5,
            "PRD requires at least 5 cases per category; `{category}` has {count}"
        );
    }
    assert_eq!(
        corpus.cases_per_category().len(),
        QueryCategory::ALL.len(),
        "every category must be represented"
    );
}

#[test]
fn the_holdout_split_is_locked_and_invisible_to_tuning() {
    let corpus = corpus();
    let holdout = corpus.queries(Split::Holdout);
    let train = corpus.queries(Split::Train);

    assert!(!holdout.is_empty(), "an empty holdout is not a split");
    assert!(!train.is_empty());
    assert_eq!(holdout.len() + train.len(), corpus.all_queries().len());

    // The only accessor a tuning story would reach for returns training cases.
    let tuning: Vec<&str> = corpus
        .tuning_queries()
        .iter()
        .map(|q| q.id.as_str())
        .collect();
    for query in &holdout {
        assert!(
            !tuning.contains(&query.id.as_str()),
            "{} is holdout and reachable from tuning_queries()",
            query.id
        );
    }

    // Every category keeps cases on both sides, so a category-level holdout
    // number is never computed from zero observations.
    for category in QueryCategory::ALL {
        assert!(
            holdout.iter().any(|q| q.category == *category),
            "category `{category}` has no holdout case"
        );
        assert!(
            train.iter().any(|q| q.category == *category),
            "category `{category}` has no training case"
        );
    }
}

#[test]
fn the_fixture_files_contain_nothing_that_looks_like_production_data() {
    // `EvalCorpus::validate` walks the parsed structure. This walks the raw
    // bytes, which also covers a field the schema does not model yet.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("parent")
        .join("contracts/eval/corpus");

    let patterns: &[(&str, &str)] = &[
        (
            "email address",
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        ),
        ("secret key", r"sk-[A-Za-z0-9]{16,}"),
        ("AWS access key id", r"AKIA[0-9A-Z]{16}"),
        (
            "raw UUID literal",
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        ),
    ];

    for name in ["manifest.json", "notebooks.json", "queries.json"] {
        let body = std::fs::read_to_string(dir.join(name)).expect("fixture is readable");
        for (label, pattern) in patterns {
            let regex = regex::Regex::new(pattern).expect("pattern compiles");
            assert!(
                !regex.is_match(&body),
                "{name} contains something shaped like a {label}"
            );
        }
    }
}

#[test]
fn a_malformed_case_fails_loudly_and_names_the_query() {
    // A corpus copied to a scratch directory with one field removed: the loader
    // must refuse it rather than evaluate 39 of 40 cases.
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("parent")
        .join("contracts/eval/corpus");
    let scratch =
        std::env::temp_dir().join(format!("openbooklm-eval-broken-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch");

    for name in ["manifest.json", "notebooks.json"] {
        std::fs::copy(source.join(name), scratch.join(name)).expect("copy");
    }
    let queries = std::fs::read_to_string(source.join("queries.json")).expect("read");
    let broken = queries.replacen("      \"category\": ", "      \"categry\": ", 1);
    assert_ne!(
        broken, queries,
        "the fixture shape changed; update this test"
    );
    std::fs::write(scratch.join("queries.json"), broken).expect("write");

    let error = EvalCorpus::load(&scratch).expect_err("a malformed case must not be skipped");
    let rendered = error.to_string();
    assert!(rendered.contains("q-"), "must name the query: {rendered}");
    assert!(
        rendered.contains("category"),
        "must name the field: {rendered}"
    );

    let _ = std::fs::remove_dir_all(&scratch);
}

// ============================================================================
// US-002 / US-003 — determinism and offline operation
// ============================================================================

#[tokio::test]
async fn every_retrieval_mode_produces_a_byte_identical_report_twice() {
    let corpus = corpus();
    for mode in RetrievalMode::ALL {
        let mut config = config(Split::Train);
        config.mode = *mode;

        // Two independently built indexes: determinism has to survive a
        // rebuild, not only a second call over the same vectors.
        let first_index = CorpusIndex::build(&corpus).await.expect("index");
        let second_index = CorpusIndex::build(&corpus).await.expect("index");

        let first = run_retrieval_eval(&corpus, &first_index, &config, FIXED_TIME)
            .await
            .expect("run");
        let second = run_retrieval_eval(&corpus, &second_index, &config, FIXED_TIME)
            .await
            .expect("run");

        assert_eq!(
            render(&first.report),
            render(&second.report),
            "{mode} retrieval report is not byte-stable"
        );
    }
}

#[tokio::test]
async fn the_default_run_needs_no_provider_credential() {
    // Clearing the keys proves the runner never reads one. It does not, on its
    // own, prove no socket is opened — that guarantee is structural: the only
    // provider the runner constructs is the in-process deterministic embedder,
    // which has no HTTP client, and the only model is `DeterministicLlm`.
    for key in [
        "VOYAGE_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "FIRECRAWL_API_KEY",
    ] {
        // SAFETY: the integration test binary runs this case on one thread and
        // no other case reads these variables.
        unsafe { std::env::remove_var(key) };
    }

    let corpus = corpus();
    let index = CorpusIndex::build(&corpus).await.expect("index");
    let run = run_retrieval_eval(&corpus, &index, &config(Split::Train), FIXED_TIME)
        .await
        .expect("retrieval runs with no credential");

    assert!(
        run.report
            .embedding_fingerprint
            .starts_with("deterministic/"),
        "the CI path must use the deterministic provider, got {}",
        run.report.embedding_fingerprint
    );
    assert_eq!(run.report.reranker_fingerprint, "none");

    let cases = answer_with_deterministic_pipeline(&corpus, &index, &config(Split::Train))
        .await
        .expect("generation runs with no credential");
    assert_eq!(cases.len(), corpus.queries(Split::Train).len());
}

#[tokio::test]
async fn no_query_is_dropped_and_no_metric_is_nan() {
    let corpus = corpus();
    let index = CorpusIndex::build(&corpus).await.expect("index");

    for split in [Split::Train, Split::Holdout] {
        for mode in RetrievalMode::ALL {
            let mut config = config(split);
            config.mode = *mode;
            let run = run_retrieval_eval(&corpus, &index, &config, FIXED_TIME)
                .await
                .expect("run");

            assert_eq!(
                run.report.queries.len(),
                corpus.queries(split).len(),
                "{mode}/{split} dropped a query"
            );
            for outcome in &run.report.queries {
                assert!(
                    !outcome.trace.reasons.is_empty(),
                    "{} has no reason code",
                    outcome.query_id
                );
            }
            let rendered = render(&run.report);
            assert!(!rendered.contains("NaN"), "{mode}/{split}");
            assert!(!rendered.contains("Infinity"), "{mode}/{split}");
        }
    }
}

#[tokio::test]
async fn retrieval_never_crosses_a_notebook_boundary() {
    let corpus = corpus();
    let index = CorpusIndex::build(&corpus).await.expect("index");

    for split in [Split::Train, Split::Holdout] {
        for mode in RetrievalMode::ALL {
            let mut config = config(split);
            config.mode = *mode;
            config.limit = 20;
            let run = run_retrieval_eval(&corpus, &index, &config, FIXED_TIME)
                .await
                .expect("run");
            assert_eq!(
                run.report.overall.isolation_failures, 0,
                "{mode}/{split} retrieved a forbidden source"
            );
        }
    }
}

// ============================================================================
// US-004 — baselines and the release gate
// ============================================================================

#[tokio::test]
async fn the_committed_baselines_still_describe_this_code() {
    for (split, file) in [
        (Split::Train, "hybrid-train.json"),
        (Split::Holdout, "hybrid-holdout.json"),
    ] {
        let captured = capture(split).await;
        let path = baseline_dir().join(file);

        if updating() {
            std::fs::create_dir_all(baseline_dir()).expect("baseline directory");
            std::fs::write(&path, render(&captured)).expect("write baseline");
            continue;
        }

        let recorded = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "missing baseline {}: {e}\nrun `UPDATE_BASELINE=1 cargo test \
                 --no-default-features --test rag_eval` to create it",
                path.display()
            )
        });

        assert_eq!(
            recorded,
            render(&captured),
            "{file} no longer describes this code.\n\
             If the change is intended, regenerate with `UPDATE_BASELINE=1 cargo test \
             --no-default-features --test rag_eval` and review the diff as a quality change."
        );
    }
}

#[tokio::test]
async fn a_baseline_compared_against_itself_does_not_block() {
    let baseline = capture(Split::Train).await;
    let report = compare(
        &baseline,
        &baseline,
        Enforcement::RegressionOnly,
        &Targets::default(),
    );
    assert!(!report.blocking(), "{report:?}");
    assert!(report.regressions.is_empty());
    assert!(report.missing_metrics.is_empty());
    assert!(report.isolation_failures.is_empty());
}

#[tokio::test]
async fn the_committed_baseline_carries_every_required_metric() {
    for file in ["hybrid-train.json", "hybrid-holdout.json"] {
        let path = baseline_dir().join(file);
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue; // covered by the regeneration test above
        };
        let document: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
        assert!(
            missing_metrics(&document).is_empty(),
            "{file} is missing {:?}",
            missing_metrics(&document)
        );
    }
}

#[tokio::test]
async fn the_first_baseline_may_sit_below_the_absolute_targets() {
    // The whole point of the two enforcement modes: the deterministic provider
    // does not reach the month-6 targets, and that must not stop a baseline
    // from being recorded (US-004 AC-4).
    let baseline = capture(Split::Train).await;
    let misses = unmet(&baseline, &Targets::default());

    let permissive = compare(
        &baseline,
        &baseline,
        Enforcement::RegressionOnly,
        &Targets::default(),
    );
    assert_eq!(
        permissive.unmet_targets, misses,
        "targets are still reported"
    );
    assert!(
        !permissive.blocking(),
        "recording a below-target baseline must be possible"
    );

    if !misses.is_empty() {
        let strict = compare(
            &baseline,
            &baseline,
            Enforcement::RegressionAndTargets,
            &Targets::default(),
        );
        assert!(
            strict.blocking(),
            "enforcement mode must be able to block on an unmet target"
        );
        assert!(
            strict.regressions.is_empty(),
            "an unmet target is not a regression"
        );
    }
}

#[tokio::test]
async fn a_manufactured_regression_blocks_the_gate() {
    let previous = capture(Split::Train).await;
    let mut current = previous.clone();
    current.code_revision = "candidate".to_owned();
    current.retrieval.overall.recall_at_10 =
        (previous.retrieval.overall.recall_at_10 - 0.05).max(0.0);

    let report = compare(
        &previous,
        &current,
        Enforcement::RegressionOnly,
        &Targets::default(),
    );
    assert!(report.blocking(), "a 0.05 drop must block");
    assert!(
        report
            .regressions
            .iter()
            .any(|d| d.metric == "retrieval.overall.recall_at_10")
    );
}

#[tokio::test]
async fn a_manufactured_isolation_failure_blocks_the_gate() {
    let previous = capture(Split::Train).await;
    let mut current = previous.clone();
    current.retrieval.overall.isolation_failures = 1;

    for enforcement in [
        Enforcement::RegressionOnly,
        Enforcement::RegressionAndTargets,
    ] {
        let report = compare(&previous, &current, enforcement, &Targets::default());
        assert!(
            report.blocking(),
            "{enforcement} let a tenant-isolation failure through"
        );
    }
}

#[tokio::test]
async fn a_baseline_missing_a_required_metric_blocks_before_any_delta() {
    let baseline = capture(Split::Train).await;
    let mut document = serde_json::to_value(&baseline).expect("json");
    document
        .pointer_mut("/retrieval/overall")
        .and_then(serde_json::Value::as_object_mut)
        .expect("object")
        .remove("recall_at_10");

    let missing = missing_metrics(&document);
    assert_eq!(missing, vec!["/retrieval/overall/recall_at_10"]);
}

#[tokio::test]
async fn a_baseline_from_another_corpus_is_refused_rather_than_compared() {
    let previous = capture(Split::Train).await;
    let mut current = previous.clone();
    current.corpus_version = "2.0.0".to_owned();

    let report = compare(
        &previous,
        &current,
        Enforcement::RegressionOnly,
        &Targets::default(),
    );
    assert!(!report.incomparable.is_empty());
    assert!(report.blocking());
}
