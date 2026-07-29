//! Startup migration validation (US-012, EP-003).
//!
//! Called before any migrator runs. It answers one question — *which of the
//! four states is this database in?* — and refuses to guess:
//!
//! | State | Meaning | What the caller does |
//! |---|---|---|
//! | `Empty` | no history at all | fresh install: run the migrators |
//! | `Legacy` | complete legacy history, no tracks | run the bridge, then the migrators |
//! | `Bridged` / `Tracked` | tracks present | run the migrators |
//! | `Divergent` | anything else | **stop**, print remediation |
//!
//! Divergent is the case that matters. A partially applied legacy history, an
//! unknown version, or a track table holding a version this build does not know
//! about all mean the running code and the database disagree about what the
//! schema is. Applying more SQL on top of that disagreement is how a migration
//! corrupts data instead of failing.

use sea_orm_migration::prelude::*;
use sea_orm_migration::sea_orm::{ConnectionTrait, DatabaseBackend, FromQueryResult, Statement};

/// The legacy history table, unchanged since the first migration.
pub const LEGACY_MIGRATION_TABLE: &str = "seaql_migrations";

/// What this build believes exists, supplied by the caller.
///
/// Validation takes the expected history rather than importing the migrators.
/// That keeps this module free of any track dependency — which matters, because
/// the public core ships it while the SaaS track stays private — and it lets a
/// test drive every state without constructing a migrator.
#[derive(Debug, Clone)]
pub struct ExpectedVersions {
    pub legacy: Vec<String>,
    pub core: Vec<String>,
    pub saas: Vec<String>,
    pub core_table: &'static str,
    pub saas_table: &'static str,
    pub core_baseline: &'static str,
    pub saas_baseline: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateKind {
    Empty,
    Legacy,
    Bridged,
    Tracked,
    Divergent(String),
}

#[derive(Debug, Clone)]
pub struct MigrationState {
    pub kind: StateKind,
    pub legacy_applied: usize,
    pub core_applied: usize,
    pub saas_applied: usize,
}

impl MigrationState {
    /// The remediation an operator should run, or `None` when the state is fine.
    #[must_use]
    pub fn remediation(&self) -> Option<String> {
        let StateKind::Divergent(reason) = &self.kind else {
            return None;
        };
        Some(format!(
            "Database migration state is incompatible: {reason}\n\
             \n\
             Do NOT run `migration fresh`, `migration refresh` or a down migration: all three\n\
             destroy data. Instead:\n\
             \n\
               1. Take a backup:      pg_dump \"$DATABASE_URL\" > backup.sql\n\
               2. Inspect the history: psql \"$DATABASE_URL\" -c \\\n\
                    'SELECT version FROM {LEGACY_MIGRATION_TABLE} ORDER BY version'\n\
               3. Compare it with the list this build expects (see docs/migrations.md)\n\
               4. Deploy the build whose expected list matches, or restore the backup\n"
        ))
    }
}

#[derive(Debug, FromQueryResult)]
struct VersionRow {
    version: String,
}

async fn table_exists<C: ConnectionTrait>(db: &C, table: &str) -> Result<bool, DbErr> {
    #[derive(Debug, FromQueryResult)]
    struct Exists {
        found: bool,
    }
    Ok(Exists::find_by_statement(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables
         WHERE table_schema = current_schema() AND table_name = $1) AS found",
        [table.into()],
    ))
    .one(db)
    .await?
    .is_some_and(|e| e.found))
}

async fn applied_versions<C: ConnectionTrait>(db: &C, table: &str) -> Result<Vec<String>, DbErr> {
    if !table_exists(db, table).await? {
        return Ok(Vec::new());
    }
    Ok(VersionRow::find_by_statement(Statement::from_string(
        DatabaseBackend::Postgres,
        format!(r#"SELECT version FROM "{table}" ORDER BY version"#),
    ))
    .all(db)
    .await?
    .into_iter()
    .map(|r| r.version)
    .collect())
}

/// Classify the database's migration state without applying any SQL.
pub async fn validate_with<C: ConnectionTrait>(
    db: &C,
    expected: &ExpectedVersions,
) -> Result<MigrationState, DbErr> {
    let legacy = applied_versions(db, LEGACY_MIGRATION_TABLE).await?;
    let core = applied_versions(db, expected.core_table).await?;
    let saas = applied_versions(db, expected.saas_table).await?;

    let expected_legacy = &expected.legacy;
    let known_core = &expected.core;
    let known_saas = &expected.saas;

    let state = |kind: StateKind| MigrationState {
        kind,
        legacy_applied: legacy.len(),
        core_applied: core.len(),
        saas_applied: saas.len(),
    };

    // A build that expects no legacy history is a public build. Finding one
    // means the database belongs to the hosted edition, which owns tables and
    // columns this binary knows nothing about. Say so plainly instead of
    // letting the checks below report a confusing "interrupted bridge".
    if expected_legacy.is_empty() && !legacy.is_empty() {
        return Ok(state(StateKind::Divergent(format!(
            "this database has a `{LEGACY_MIGRATION_TABLE}` history with {} applied migration(s), \
             but this build owns the core schema only. It belongs to a different edition; start \
             it with the binary that owns its history.",
            legacy.len()
        ))));
    }

    // A track table holding a version this build does not know about means the
    // database was migrated by a newer build. Rolling back the code without
    // rolling back the schema is a supported operation only inside the
    // compatibility window, and only for versions this build recognises.
    for (label, applied, known) in [("core", &core, known_core), ("saas", &saas, known_saas)] {
        if let Some(unknown) = applied.iter().find(|v| !known.contains(v)) {
            return Ok(state(StateKind::Divergent(format!(
                "the {label} track has applied `{unknown}`, which this build does not know. \
                 The database was migrated by a newer release."
            ))));
        }
    }

    if !core.is_empty() || !saas.is_empty() {
        // On a legacy database the bridge records both baselines in one
        // transaction, so exactly one track means it was interrupted. On a new
        // database `core-up` legitimately runs before `saas-up`, and a public
        // install never initialises the SaaS track at all — neither is a
        // divergence.
        if !legacy.is_empty() && (core.is_empty() || saas.is_empty()) {
            return Ok(state(StateKind::Divergent(
                "this database has a legacy history but only one of the core and SaaS tracks is \
                 initialised; the bridge was interrupted"
                    .to_owned(),
            )));
        }
        let bridged = !legacy.is_empty()
            && core.iter().any(|v| v == expected.core_baseline)
            && saas.iter().any(|v| v == expected.saas_baseline);
        return Ok(state(if bridged {
            StateKind::Bridged
        } else {
            StateKind::Tracked
        }));
    }

    if legacy.is_empty() {
        return Ok(state(StateKind::Empty));
    }

    if let Some(unknown) = legacy.iter().find(|v| !expected_legacy.contains(v)) {
        return Ok(state(StateKind::Divergent(format!(
            "the legacy history contains `{unknown}`, which this build does not know"
        ))));
    }
    let missing: Vec<&String> = expected_legacy
        .iter()
        .filter(|v| !legacy.contains(v))
        .collect();
    if !missing.is_empty() {
        return Ok(state(StateKind::Divergent(format!(
            "the legacy history is incomplete: {} migration(s) never applied, starting with `{}`",
            missing.len(),
            missing.first().map_or("?", |v| v.as_str())
        ))));
    }

    Ok(state(StateKind::Legacy))
}
