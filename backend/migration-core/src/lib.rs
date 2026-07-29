//! The public core schema (US-012, US-013).
//!
//! One crate, two responsibilities:
//!
//! - [`core_track::CoreMigrator`] owns the core baseline and every future core
//!   schema change, in its own history table `seaql_migrations_core`.
//! - [`validate`] classifies a database against an expected history and refuses
//!   to guess what an unknown state means.
//!
//! It is deliberately separate from the private `migration` crate. That crate
//! carries the legacy applied history and the SaaS track, neither of which a
//! self-hosted installation has or should have. The dependency points one way:
//! `migration` depends on this crate and re-exports it, never the reverse.
//!
//! A fresh public database reaches the full core schema with
//! [`core_track::CoreMigrator::up`] and nothing else.

pub mod core_track;
pub mod validate;

pub use sea_orm_migration::prelude::*;

/// Every core-track migration this build knows, in order.
///
/// Derived from the migrator rather than hand-listed: a list that can drift
/// from the migrator is worse than no list.
#[must_use]
pub fn core_versions() -> Vec<String> {
    core_track::CoreMigrator::migrations()
        .iter()
        .map(|m| m.name().to_owned())
        .collect()
}

/// The expected history for a **public** installation.
///
/// The legacy and SaaS lists are empty because a self-hosted database has
/// neither. A public database that reports either is divergent, which is the
/// correct answer: it is a hosted database being started by the wrong binary.
#[must_use]
pub fn expected_core_only() -> validate::ExpectedVersions {
    validate::ExpectedVersions {
        legacy: Vec::new(),
        core: core_versions(),
        saas: Vec::new(),
        core_table: core_track::CORE_MIGRATION_TABLE,
        // A public build never looks for a SaaS history table: an empty name
        // matches nothing, so the SaaS track is invisible rather than special-cased.
        saas_table: "",
        core_baseline: core_track::CORE_BASELINE,
        saas_baseline: "",
    }
}

/// Classify a public database against the core-only expected history.
///
/// # Errors
/// Propagates any database error from the queries it runs.
pub async fn validate_core_state<C: sea_orm_migration::sea_orm::ConnectionTrait>(
    db: &C,
) -> Result<validate::MigrationState, DbErr> {
    validate::validate_with(db, &expected_core_only()).await
}
