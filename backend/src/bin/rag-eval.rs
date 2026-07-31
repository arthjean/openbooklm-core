//! Offline RAG evaluation command (EP-001).
//!
//! ```bash
//! cargo run --bin rag-eval -- validate
//! cargo run --bin rag-eval -- retrieval --mode hybrid --split train
//! cargo run --bin rag-eval -- grounding --split train
//! cargo run --bin rag-eval -- baseline --out contracts/eval/baseline/hybrid-train.json
//! cargo run --bin rag-eval -- compare \
//!     --baseline contracts/eval/baseline/hybrid-train.json \
//!     --candidate /tmp/candidate.json
//! ```
//!
//! Every subcommand is offline: the corpus is a checked-in fixture, retrieval
//! runs against an in-memory index, and generation uses the in-process
//! deterministic model. No credential is read and no socket is opened (FR-20).
//!
//! # Exit codes
//!
//! `0` when the requested check passed, `1` when it failed, `2` when the
//! invocation itself was wrong. `validate` fails on any corpus violation;
//! `compare` fails on a regression, a tenant-isolation failure, a missing
//! required metric, an incomparable pair, and — only under
//! `--enforce regression_and_targets` — an unmet absolute target.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use openbooklm::services::rag::eval::baseline::{
    Baseline, Enforcement, Targets, compare, missing_metrics,
};
use openbooklm::services::rag::eval::corpus::{EvalCorpus, Split, default_corpus_dir};
use openbooklm::services::rag::eval::grounding::{
    GroundingRunConfig, answer_with_deterministic_pipeline, run_grounding_eval,
};
use openbooklm::services::rag::eval::index::CorpusIndex;
use openbooklm::services::rag::eval::retrieval::{
    RetrievalMode, RetrievalRunConfig, run_retrieval_eval,
};

const USAGE: &str = "\
rag-eval — offline RAG evaluation (EP-001)

USAGE:
    rag-eval validate  [--corpus DIR]
    rag-eval retrieval [--mode MODE] [--split SPLIT] [--limit N] [--out FILE]
                       [--latency-out FILE] [--revision REV] [--now TIMESTAMP]
    rag-eval grounding [--mode MODE] [--split SPLIT] [--limit N] [--out FILE]
                       [--revision REV] [--now TIMESTAMP]
    rag-eval baseline  --out FILE [--mode MODE] [--split SPLIT] [--limit N]
                       [--revision REV] [--now TIMESTAMP]
    rag-eval compare   --baseline FILE --candidate FILE
                       [--enforce regression_only|regression_and_targets] [--out FILE]

MODE     dense | lexical | hybrid | exact_reference   (default: hybrid)
SPLIT    train | holdout                              (default: train)

`--now` fixes the report timestamp, which is the one field excluded from the
byte-stability contract. Pass it to produce a reproducible artifact.
";

/// The invocation was malformed. Distinct from a failed check.
const EXIT_USAGE: u8 = 2;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().cloned() else {
        eprint!("{USAGE}");
        return ExitCode::from(EXIT_USAGE);
    };

    let flags = match parse_flags(&args[1..]) {
        Ok(flags) => flags,
        Err(message) => {
            eprintln!("{message}\n");
            eprint!("{USAGE}");
            return ExitCode::from(EXIT_USAGE);
        }
    };

    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("could not start a runtime: {e}");
            return ExitCode::FAILURE;
        }
    };

    match command.as_str() {
        "validate" => runtime.block_on(validate(&flags)),
        "retrieval" => runtime.block_on(retrieval(&flags)),
        "grounding" => runtime.block_on(grounding(&flags)),
        "baseline" => runtime.block_on(baseline(&flags)),
        "compare" => compare_command(&flags),
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command `{other}`\n");
            eprint!("{USAGE}");
            ExitCode::from(EXIT_USAGE)
        }
    }
}

// ============================================================================
// Flags
// ============================================================================

type Flags = HashMap<String, String>;

/// Parse `--key value` pairs. Deliberately tiny: the crate has no argument
/// parser, and adding one for five options would be a dependency nobody asked
/// for.
fn parse_flags(args: &[String]) -> Result<Flags, String> {
    let mut flags = Flags::new();
    let mut index = 0;
    while index < args.len() {
        let Some(key) = args[index].strip_prefix("--") else {
            return Err(format!("unexpected argument `{}`", args[index]));
        };
        let Some(value) = args.get(index + 1) else {
            return Err(format!("`--{key}` needs a value"));
        };
        if value.starts_with("--") {
            return Err(format!("`--{key}` needs a value, got `{value}`"));
        }
        flags.insert(key.to_owned(), value.clone());
        index += 2;
    }
    Ok(flags)
}

fn corpus_dir(flags: &Flags) -> PathBuf {
    flags
        .get("corpus")
        .map_or_else(default_corpus_dir, PathBuf::from)
}

fn revision(flags: &Flags) -> String {
    flags
        .get("revision")
        .cloned()
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_owned())
}

/// The report timestamp.
///
/// Defaults to the literal `unspecified` rather than to the clock: a caller who
/// wants a timestamp passes one, and one who does not gets a reproducible
/// artifact instead of a surprise diff.
fn now(flags: &Flags) -> String {
    flags
        .get("now")
        .cloned()
        .unwrap_or_else(|| "unspecified".to_owned())
}

fn split(flags: &Flags) -> Result<Split, String> {
    match flags.get("split").map(String::as_str) {
        None | Some("train") => Ok(Split::Train),
        Some("holdout") => Ok(Split::Holdout),
        Some(other) => Err(format!("unknown split `{other}`")),
    }
}

fn retrieval_config(flags: &Flags) -> Result<RetrievalRunConfig, String> {
    let mode = match flags.get("mode") {
        None => RetrievalMode::Hybrid,
        Some(value) => {
            RetrievalMode::parse(value).ok_or_else(|| format!("unknown mode `{value}`"))?
        }
    };
    let limit = match flags.get("limit") {
        None => 10,
        Some(value) => value
            .parse::<i32>()
            .map_err(|_| format!("`--limit` must be a whole number, got `{value}`"))?,
    };
    if limit <= 0 {
        return Err("`--limit` must be greater than zero".to_owned());
    }

    Ok(RetrievalRunConfig {
        mode,
        split: split(flags)?,
        limit,
        code_revision: revision(flags),
        ..RetrievalRunConfig::default()
    })
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    let mut rendered = serde_json::to_string_pretty(value)
        .map_err(|e| format!("could not render {}: {e}", path.display()))?;
    rendered.push('\n');
    std::fs::write(path, rendered).map_err(|e| format!("could not write {}: {e}", path.display()))
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("{} is not valid JSON: {e}", path.display()))
}

/// Load the corpus and refuse to measure an invalid one.
///
/// Every runner goes through here. Publishing numbers from a corpus that fails
/// its own validator would make the report look authoritative and be worthless.
async fn load_validated(flags: &Flags) -> Result<(EvalCorpus, CorpusIndex), String> {
    let dir = corpus_dir(flags);
    let corpus = EvalCorpus::load(&dir).map_err(|e| format!("corpus: {e}"))?;

    let violations = corpus.validate();
    if !violations.is_empty() {
        for violation in &violations {
            eprintln!("  {violation}");
        }
        return Err(format!(
            "corpus at {} has {} violation(s); fix them before measuring anything",
            dir.display(),
            violations.len()
        ));
    }

    let index = CorpusIndex::build(&corpus)
        .await
        .map_err(|e| format!("could not index the corpus: {e}"))?;
    Ok((corpus, index))
}

// ============================================================================
// Commands
// ============================================================================

async fn validate(flags: &Flags) -> ExitCode {
    let dir = corpus_dir(flags);
    let corpus = match EvalCorpus::load(&dir) {
        Ok(corpus) => corpus,
        Err(e) => {
            eprintln!("FAIL {e}");
            return ExitCode::FAILURE;
        }
    };

    let violations = corpus.validate();
    if violations.is_empty() {
        println!(
            "ok  corpus {} — {} notebooks, {} chunks, {} queries ({} train, {} holdout)",
            corpus.version(),
            corpus.notebooks().len(),
            corpus.chunk_count(),
            corpus.all_queries().len(),
            corpus.queries(Split::Train).len(),
            corpus.queries(Split::Holdout).len(),
        );
        for (category, count) in corpus.cases_per_category() {
            println!("    {:<22} {count}", category.as_str());
        }
        return ExitCode::SUCCESS;
    }

    eprintln!("FAIL {} corpus violation(s):", violations.len());
    for violation in &violations {
        eprintln!("  {violation}");
    }
    ExitCode::FAILURE
}

async fn retrieval(flags: &Flags) -> ExitCode {
    let config = match retrieval_config(flags) {
        Ok(config) => config,
        Err(message) => return fail_usage(&message),
    };
    let (corpus, index) = match load_validated(flags).await {
        Ok(pair) => pair,
        Err(message) => return fail(&message),
    };

    let run = match run_retrieval_eval(&corpus, &index, &config, &now(flags)).await {
        Ok(run) => run,
        Err(e) => return fail(&format!("retrieval evaluation failed: {e}")),
    };

    if let Some(path) = flags.get("out")
        && let Err(message) = write_json(Path::new(path), &run.report)
    {
        return fail(&message);
    }
    if let Some(path) = flags.get("latency-out")
        && let Err(message) = write_json(Path::new(path), &run.latency)
    {
        return fail(&message);
    }
    if flags.get("out").is_none() {
        match serde_json::to_string_pretty(&run.report) {
            Ok(rendered) => println!("{rendered}"),
            Err(e) => return fail(&format!("could not render the report: {e}")),
        }
    }

    eprintln!(
        "ok  {} {} — recall@10 {:.3}, nDCG@10 {:.3}, fill {:.3}, isolation failures {} \
         (p50 {} µs, p95 {} µs)",
        run.report.mode,
        run.report.split,
        run.report.overall.recall_at_10,
        run.report.overall.ndcg_at_10,
        run.report.overall.top_k_fill_rate,
        run.report.overall.isolation_failures,
        run.latency.p50_us,
        run.latency.p95_us,
    );
    ExitCode::SUCCESS
}

async fn grounding(flags: &Flags) -> ExitCode {
    let retrieval_config = match retrieval_config(flags) {
        Ok(config) => config,
        Err(message) => return fail_usage(&message),
    };
    let (corpus, index) = match load_validated(flags).await {
        Ok(pair) => pair,
        Err(message) => return fail(&message),
    };

    let cases = match answer_with_deterministic_pipeline(&corpus, &index, &retrieval_config).await {
        Ok(cases) => cases,
        Err(e) => return fail(&format!("answer production failed: {e}")),
    };

    let config = GroundingRunConfig {
        split: retrieval_config.split,
        code_revision: retrieval_config.code_revision.clone(),
    };
    let report = run_grounding_eval(&corpus, &cases, &config, None, &now(flags));

    if let Some(path) = flags.get("out") {
        if let Err(message) = write_json(Path::new(path), &report) {
            return fail(&message);
        }
    } else {
        match serde_json::to_string_pretty(&report) {
            Ok(rendered) => println!("{rendered}"),
            Err(e) => return fail(&format!("could not render the report: {e}")),
        }
    }

    eprintln!(
        "ok  grounding {} — claim coverage {:.3}, citation precision {:.3}, coverage {:.3}, \
         abstention {:.3}, unsupported claims {}",
        report.split,
        report.overall.expected_claim_coverage,
        report.overall.citation_precision,
        report.overall.citation_coverage,
        report.overall.abstention_accuracy,
        report.overall.unsupported_claims,
    );
    ExitCode::SUCCESS
}

async fn baseline(flags: &Flags) -> ExitCode {
    let Some(out) = flags.get("out") else {
        return fail_usage("`baseline` needs `--out FILE`");
    };
    let retrieval_config = match retrieval_config(flags) {
        Ok(config) => config,
        Err(message) => return fail_usage(&message),
    };
    let (corpus, index) = match load_validated(flags).await {
        Ok(pair) => pair,
        Err(message) => return fail(&message),
    };

    let run = match run_retrieval_eval(&corpus, &index, &retrieval_config, &now(flags)).await {
        Ok(run) => run,
        Err(e) => return fail(&format!("retrieval evaluation failed: {e}")),
    };
    let cases = match answer_with_deterministic_pipeline(&corpus, &index, &retrieval_config).await {
        Ok(cases) => cases,
        Err(e) => return fail(&format!("answer production failed: {e}")),
    };
    let grounding_report = run_grounding_eval(
        &corpus,
        &cases,
        &GroundingRunConfig {
            split: retrieval_config.split,
            code_revision: retrieval_config.code_revision.clone(),
        },
        None,
        &now(flags),
    );

    let artifact = match Baseline::capture(&run.report, &grounding_report) {
        Ok(artifact) => artifact,
        Err(message) => return fail(&format!("cannot capture a baseline: {message}")),
    };

    if let Err(message) = write_json(Path::new(out), &artifact) {
        return fail(&message);
    }
    if let Some(path) = flags.get("latency-out")
        && let Err(message) = write_json(Path::new(path), &run.latency)
    {
        return fail(&message);
    }

    // Unmet targets are printed, never fatal: the first baseline of a project
    // is captured below target by definition (US-004 AC-4).
    let misses = openbooklm::services::rag::eval::baseline::unmet(&artifact, &Targets::default());
    eprintln!("ok  baseline written to {out}");
    for miss in &misses {
        eprintln!(
            "    below target: {} = {:.3} (target {:.3})",
            miss.metric, miss.value, miss.target
        );
    }
    ExitCode::SUCCESS
}

fn compare_command(flags: &Flags) -> ExitCode {
    let (Some(previous_path), Some(current_path)) = (flags.get("baseline"), flags.get("candidate"))
    else {
        return fail_usage("`compare` needs `--baseline FILE` and `--candidate FILE`");
    };
    let enforcement = match flags.get("enforce").map(String::as_str) {
        None => Enforcement::RegressionOnly,
        Some(value) => match Enforcement::parse(value) {
            Some(mode) => mode,
            None => return fail_usage(&format!("unknown enforcement mode `{value}`")),
        },
    };

    let mut documents = Vec::new();
    for (label, path) in [("baseline", previous_path), ("candidate", current_path)] {
        match read_json(Path::new(path)) {
            Ok(document) => documents.push((label, path, document)),
            Err(message) => return fail(&message),
        }
    }

    // A required metric absent from either document blocks before anything is
    // compared: a delta computed against a default is a fabricated number.
    let mut missing = Vec::new();
    for (label, path, document) in &documents {
        for pointer in missing_metrics(document) {
            missing.push(format!("{label} ({path}): missing {pointer}"));
        }
    }
    if !missing.is_empty() {
        eprintln!("FAIL {} required metric(s) missing:", missing.len());
        for entry in &missing {
            eprintln!("  {entry}");
        }
        return ExitCode::FAILURE;
    }

    let mut parsed = Vec::new();
    for (label, path, document) in documents {
        match serde_json::from_value::<Baseline>(document) {
            Ok(baseline) => parsed.push(baseline),
            Err(e) => return fail(&format!("{label} ({path}) is not a baseline: {e}")),
        }
    }
    let (Some(previous), Some(current)) = (parsed.first(), parsed.get(1)) else {
        return fail("could not read both baselines");
    };

    let report = compare(previous, current, enforcement, &Targets::default());

    if let Some(path) = flags.get("out")
        && let Err(message) = write_json(Path::new(path), &report)
    {
        return fail(&message);
    }

    for entry in &report.incomparable {
        eprintln!("  incomparable: {entry}");
    }
    for entry in &report.missing_metrics {
        eprintln!("  missing: {entry}");
    }
    for delta in &report.regressions {
        eprintln!(
            "  regression: {} {:.3} -> {:.3} ({:+.3})",
            delta.metric, delta.previous, delta.current, delta.delta
        );
    }
    for entry in &report.isolation_failures {
        eprintln!("  isolation: {entry}");
    }
    for miss in &report.unmet_targets {
        eprintln!(
            "  below target: {} = {:.3} (target {:.3})",
            miss.metric, miss.value, miss.target
        );
    }
    for delta in &report.improvements {
        eprintln!(
            "  improvement: {} {:.3} -> {:.3} ({:+.3})",
            delta.metric, delta.previous, delta.current, delta.delta
        );
    }

    eprintln!("{}", report.summary());
    if report.blocking() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn fail(message: &str) -> ExitCode {
    eprintln!("FAIL {message}");
    ExitCode::FAILURE
}

fn fail_usage(message: &str) -> ExitCode {
    eprintln!("{message}\n");
    eprint!("{USAGE}");
    ExitCode::from(EXIT_USAGE)
}
