//! Generate the public contract artifacts (US-010).
//!
//! ```bash
//! cargo run --bin contracts            # write contracts/*.json
//! cargo run --bin contracts -- --check # fail if a committed artifact is stale
//! ```
//!
//! Two artifacts, one generator:
//!
//! - `contracts/openapi.json` — request and response shapes, from the
//!   `#[utoipa::path]` annotations on the core handlers.
//! - `contracts/core-constants.json` — the values a client needs before it can
//!   build a valid request: validation limits, teaching modes, source types,
//!   provider capabilities. OpenAPI has nowhere to put these.
//!
//! `--check` is what `scripts/check-contracts.sh` and public CI run. Every
//! artifact is rendered before any is written, so a failure never leaves one
//! file regenerated and another stale.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Repository-root-relative paths of the generated contracts.
const OPENAPI: &str = "contracts/openapi.json";
const CATALOG: &str = "contracts/core-constants.json";

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<repo>/backend`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

/// Serialize a document with a trailing newline, matching what an editor and
/// `git diff` expect from a checked-in JSON file.
fn render<T: serde::Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let mut json = serde_json::to_string_pretty(value)?;
    json.push('\n');
    Ok(json)
}

/// Render every artifact before writing any of them.
fn render_all() -> Result<Vec<(&'static str, String)>, serde_json::Error> {
    Ok(vec![
        (OPENAPI, render(&openbooklm::api::openapi::document())?),
        (CATALOG, render(&openbooklm::core::catalog())?),
    ])
}

fn main() -> ExitCode {
    let check = std::env::args().any(|arg| arg == "--check");
    let root = repo_root();

    let artifacts = match render_all() {
        Ok(artifacts) => artifacts,
        Err(e) => {
            eprintln!("contract generation failed, no artifact was written: {e}");
            return ExitCode::FAILURE;
        }
    };

    if check {
        let mut stale = Vec::new();
        for (relative, rendered) in &artifacts {
            match std::fs::read_to_string(root.join(relative)) {
                Ok(committed) if committed == *rendered => {
                    println!("{relative} is up to date");
                }
                Ok(_) => stale.push(format!("{relative} is stale")),
                Err(e) => stale.push(format!("{relative} is missing or unreadable ({e})")),
            }
        }
        if stale.is_empty() {
            return ExitCode::SUCCESS;
        }
        eprintln!(
            "{}\nRun `cd backend && cargo run --bin contracts`, review the diff, and commit it.",
            stale.join("\n")
        );
        return ExitCode::FAILURE;
    }

    for (relative, rendered) in &artifacts {
        let path = root.join(relative);
        if let Some(parent) = path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            eprintln!("could not create {}: {e}", parent.display());
            return ExitCode::FAILURE;
        }
        if let Err(e) = std::fs::write(&path, rendered) {
            eprintln!("could not write {relative}: {e}");
            return ExitCode::FAILURE;
        }
        println!("wrote {relative} ({} bytes)", rendered.len());
    }
    ExitCode::SUCCESS
}
