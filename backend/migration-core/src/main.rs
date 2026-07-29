//! Public core migration CLI.
//!
//! ```bash
//! openbooklm-migrate validate -u "$DATABASE_URL"   # classify, apply nothing
//! openbooklm-migrate up       -u "$DATABASE_URL"   # apply the core track
//! openbooklm-migrate status   -u "$DATABASE_URL"   # what is applied
//! ```
//!
//! There is no `down`, no `fresh` and no `refresh`. Every one of them destroys
//! data, and an operator who needs to go back restores the backup the upgrade
//! guide told them to take. `up` validates first and refuses to apply SQL on
//! top of a history it cannot account for.
//!
//! Every verb takes the migration advisory lock, so two instances started by a
//! rolling deploy serialise instead of racing.

use openbooklm_migration_core::core_track::{CoreMigrator, with_migration_lock};
use openbooklm_migration_core::validate_core_state;
use sea_orm_migration::prelude::*;
use sea_orm_migration::sea_orm::{Database, DatabaseConnection};

const USAGE: &str = "\
openbooklm-migrate — the OpenbookLM core schema

USAGE:
    openbooklm-migrate <validate|up|status> [-u <database-url>]

The database URL comes from -u/--database-url, or from DATABASE_URL.

VERBS:
    validate   classify the database against this build's expected history
    up         apply pending core migrations (validates first)
    status     print applied and pending core migrations
";

/// Read `-u <url>` or `--database-url <url>`, falling back to `DATABASE_URL`.
fn database_url(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == "-u" || arg == "--database-url" {
            return iter.next().cloned();
        }
    }
    std::env::var("DATABASE_URL").ok()
}

async fn connect(args: &[String]) -> DatabaseConnection {
    let Some(url) = database_url(args) else {
        eprintln!("no database URL: pass -u <url> or set DATABASE_URL");
        std::process::exit(2);
    };
    match Database::connect(&url).await {
        Ok(db) => db,
        Err(e) => {
            // `e` renders the connection error, not the URL, so no password
            // reaches the log.
            eprintln!("could not connect to the database: {e}");
            std::process::exit(2);
        }
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.first().map(String::as_str).unwrap_or("") {
        "validate" => {
            let db = connect(&args).await;
            let state = validate_core_state(&db).await.unwrap_or_else(|e| {
                eprintln!("validation failed: {e}");
                std::process::exit(1);
            });
            println!(
                "state: {:?}\ncore:  {} applied",
                state.kind, state.core_applied
            );
            if let Some(remediation) = state.remediation() {
                eprintln!("\n{remediation}");
                std::process::exit(1);
            }
        }
        "up" => {
            let db = connect(&args).await;
            let state = validate_core_state(&db).await.unwrap_or_else(|e| {
                eprintln!("validation failed: {e}");
                std::process::exit(1);
            });
            if let Some(remediation) = state.remediation() {
                eprintln!("{remediation}");
                std::process::exit(1);
            }
            if let Err(e) =
                with_migration_lock(&db, async || CoreMigrator::up(&db, None).await).await
            {
                eprintln!("migration failed: {e}");
                std::process::exit(1);
            }
            println!("core schema up to date");
        }
        "status" => {
            let db = connect(&args).await;
            if let Err(e) = CoreMigrator::status(&db).await {
                eprintln!("status failed: {e}");
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
}
