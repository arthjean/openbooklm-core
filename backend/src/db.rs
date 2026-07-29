//! Database connection with configurable connection pool.

use sea_orm::{ConnectOptions, Database, DatabaseConnection};
use std::time::Duration;

use crate::core::config::DatabasePoolConfig;

/// Connect to the database with a configured connection pool.
///
/// Optimized for Neon PostgreSQL serverless (compute suspends after idle).
///
/// - `min_connections(0)`: Allow Neon compute to fully scale to zero
/// - `test_before_acquire(true)`: Catch stale connections after Neon suspend
/// - `connect_lazy(true)`: Server starts even if Neon compute is waking up
/// - `idle_timeout(270s)`: Evict connections before Neon's 5-min suspend window
pub async fn connect(
    url: &str,
    pool: &DatabasePoolConfig,
) -> Result<DatabaseConnection, sea_orm::DbErr> {
    tracing::info!(
        min = pool.min_connections,
        max = pool.max_connections,
        connect_timeout = pool.connect_timeout_secs,
        acquire_timeout = pool.acquire_timeout_secs,
        idle_timeout = pool.idle_timeout_secs,
        max_lifetime = pool.max_lifetime_secs,
        "Connecting to database"
    );

    let opts = ConnectOptions::new(url)
        .min_connections(pool.min_connections)
        .max_connections(pool.max_connections)
        .connect_timeout(Duration::from_secs(pool.connect_timeout_secs))
        .acquire_timeout(Duration::from_secs(pool.acquire_timeout_secs))
        .idle_timeout(Duration::from_secs(pool.idle_timeout_secs))
        .max_lifetime(Duration::from_secs(pool.max_lifetime_secs))
        .test_before_acquire(true)
        .connect_lazy(true)
        .sqlx_logging(true)
        .sqlx_logging_level(tracing::log::LevelFilter::Debug)
        .to_owned();

    let db = Database::connect(opts).await?;
    tracing::info!("Database connection pool established");

    Ok(db)
}

/// Connect with default pool configuration (min=0, max=10, Neon-optimized).
pub async fn connect_with_defaults(url: &str) -> Result<DatabaseConnection, sea_orm::DbErr> {
    connect(url, &DatabasePoolConfig::default()).await
}
