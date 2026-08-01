//! The public core migration track (US-012, EP-003).
//!
//! `CoreMigrator` owns the core schema and every future change to it. It has
//! its own history table, `seaql_migrations_core`, so the public and private
//! tracks can advance independently and neither can accidentally record the
//! other's progress.
//!
//! Three databases, three paths:
//!
//! | Database | How it reaches the core schema |
//! |---|---|
//! | fresh public | `CoreMigrator::up` alone |
//! | fresh hosted | `CoreMigrator::up`, then `SaasMigrator::up` |
//! | existing hosted | the bridge records the baseline as satisfied; no core SQL runs |

mod m20260729_000001_core_baseline;
mod m20260801_000001_index_generations;
mod m20260801_000002_rag_log_redaction;

use sea_orm_migration::prelude::*;
use sea_orm_migration::sea_orm::{ConnectionTrait, DatabaseBackend, Statement};

/// History table for the core track. Distinct from the legacy
/// `seaql_migrations` and from `seaql_migrations_saas`.
pub const CORE_MIGRATION_TABLE: &str = "seaql_migrations_core";

/// Version string of the core baseline. The bridge writes exactly this value.
pub const CORE_BASELINE: &str = "m20260729_000001_core_baseline";

pub struct CoreMigrator;

#[async_trait::async_trait]
impl MigratorTrait for CoreMigrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20260729_000001_core_baseline::Migration),
            Box::new(m20260801_000001_index_generations::Migration),
            Box::new(m20260801_000002_rag_log_redaction::Migration),
        ]
    }

    fn migration_table_name() -> sea_orm::DynIden {
        Alias::new(CORE_MIGRATION_TABLE).into_iden()
    }
}

// ============================================================================
// Serialization
// ============================================================================

/// Advisory lock key for migration execution.
///
/// A rolling deploy starts the new instance before stopping the old one, so two
/// processes can reach the migrator at the same moment. Postgres advisory locks
/// are session-scoped and cost nothing when uncontended; the alternative — each
/// instance racing to apply the same `CREATE TABLE` — surfaces as a confusing
/// duplicate-object error on whichever instance loses.
///
/// The constant is arbitrary but must never change: it is the lock's identity.
pub const MIGRATION_LOCK_KEY: i64 = 0x0BB1_0CA1_0000_0001_u64 as i64;

/// Run `f` while holding the migration advisory lock.
///
/// The lock is released even if `f` fails, because the guard's release runs on
/// both paths. It is *not* released if the process is killed mid-migration —
/// Postgres releases it when the session ends, which is the behaviour we want.
pub async fn with_migration_lock<C, F, T>(db: &C, f: F) -> Result<T, DbErr>
where
    C: ConnectionTrait,
    F: AsyncFnOnce() -> Result<T, DbErr>,
{
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "SELECT pg_advisory_lock($1)",
        [MIGRATION_LOCK_KEY.into()],
    ))
    .await?;

    let result = f().await;

    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "SELECT pg_advisory_unlock($1)",
        [MIGRATION_LOCK_KEY.into()],
    ))
    .await?;

    result
}
