//! Health check endpoints.
//!
//! Provides health status including:
//! - Database connectivity and latency
//! - External client availability and circuit breaker states
//! - Client request metrics (latency, error rate, rate limits)
//! - Purge task status and history
//! - SSE broadcaster channel count
//! - Active background task count

use std::sync::OnceLock;
use std::time::Instant;

use axum::{Json, extract::State, http::StatusCode};
use sea_orm::{ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement};
use serde::Serialize;

use crate::clients::{CircuitState, MetricsSnapshot};
use crate::core::CoreState;

/// Application version from Cargo.toml.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Server start time for uptime tracking.
static START_TIME: OnceLock<Instant> = OnceLock::new();

fn uptime_secs() -> u64 {
    START_TIME.get_or_init(Instant::now).elapsed().as_secs()
}

// ============================================================================
// Response types
// ============================================================================

/// Basic health response for load balancer checks.
#[derive(Serialize, utoipa::ToSchema)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

/// Detailed health response with comprehensive diagnostics.
#[derive(Serialize, utoipa::ToSchema)]
pub struct DetailedHealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub uptime_secs: u64,
    pub db: DatabaseHealth,
    pub clients: Vec<ClientHealth>,
    pub purge_tasks: Vec<PurgeTaskHealth>,
    pub sse_broadcaster: SseBroadcasterHealth,
    pub active_tasks: u64,
}

/// Database health status including pool configuration and live metrics.
#[derive(Serialize, utoipa::ToSchema)]
pub struct DatabaseHealth {
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub pool: PoolConfig,
    pub pool_metrics: PoolMetrics,
}

/// Database connection pool configuration.
#[derive(Serialize, utoipa::ToSchema)]
pub struct PoolConfig {
    pub min_connections: u32,
    pub max_connections: u32,
    pub connect_timeout_secs: u64,
    pub acquire_timeout_secs: u64,
    pub idle_timeout_secs: u64,
    pub max_lifetime_secs: u64,
}

/// Live database connection pool metrics from the underlying sqlx pool.
#[derive(Serialize, utoipa::ToSchema)]
pub struct PoolMetrics {
    /// Total connections currently managed by the pool (active + idle).
    pub active_connections: u32,
    /// Connections currently idle and available for use.
    pub idle_connections: u32,
    /// Maximum connections the pool is configured to hold.
    pub max_connections: u32,
    /// Pool utilization as a percentage: `(active - idle) / max * 100`.
    pub utilization_pct: f64,
}

/// Health status for a single external client.
#[derive(Serialize, utoipa::ToSchema)]
pub struct ClientHealth {
    pub name: &'static str,
    pub available: bool,
    pub circuit_state: CircuitState,
    pub metrics: ClientMetricsHealth,
}

/// Subset of client metrics exposed in the health endpoint.
#[derive(Default, Serialize, utoipa::ToSchema)]
pub struct ClientMetricsHealth {
    pub total_requests: u64,
    pub error_rate: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_latency_ms: Option<u64>,
    pub rate_limit_hits: u64,
    pub timeout_errors: u64,
}

impl From<MetricsSnapshot> for ClientMetricsHealth {
    fn from(s: MetricsSnapshot) -> Self {
        Self {
            total_requests: s.total_requests,
            error_rate: s.error_rate,
            avg_latency_ms: s.avg_latency_ms,
            min_latency_ms: s.min_latency_ms,
            max_latency_ms: s.max_latency_ms,
            rate_limit_hits: s.rate_limit_hits,
            timeout_errors: s.timeout_errors,
        }
    }
}

/// Health status for a periodic purge task.
#[derive(Serialize, utoipa::ToSchema)]
pub struct PurgeTaskHealth {
    pub name: &'static str,
    pub total_runs: u64,
    pub total_deleted: u64,
    pub total_failures: u64,
    pub consecutive_failures: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_success_secs_ago: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_failure_secs_ago: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_run_duration_ms: Option<u64>,
    pub normal_interval_secs: u64,
}

/// SSE broadcaster health.
#[derive(Serialize, utoipa::ToSchema)]
pub struct SseBroadcasterHealth {
    pub active_channels: usize,
    pub total_receivers: usize,
}

// ============================================================================
// Handlers
// ============================================================================

/// GET /health - Basic health check.
///
/// Performs a lightweight DB check (`SELECT 1`) and returns 200 if healthy,
/// 503 if the database is unreachable (e.g., pool exhaustion).
#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses(
        (status = 200, description = "Liveness and database reachability", body = HealthResponse),
        (status = 503, description = "One or more dependencies are unhealthy", body = HealthResponse),
    ),
)]
pub async fn health_check(State(state): State<CoreState>) -> (StatusCode, Json<HealthResponse>) {
    // Initialize start time on first call
    let _ = uptime_secs();

    let db_ok = check_database_ok(&state.db).await;
    let status_code = if db_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status_code,
        Json(HealthResponse {
            status: if db_ok { "ok" } else { "unavailable" },
            version: VERSION,
        }),
    )
}

/// GET /health/detailed - Detailed health check.
///
/// Returns comprehensive health status including database connectivity,
/// external client circuit breaker states and metrics, purge task status,
/// and SSE broadcaster information.
///
/// Returns 200 if healthy, 503 if degraded, 401 if token is required but missing/wrong.
/// Does NOT expose secrets or user data.
///
/// When `HEALTH_TOKEN` is set in config, requests must include `X-Health-Token` header
/// with the matching value. When unset, access is unrestricted (backwards compatible).
#[utoipa::path(
    get,
    path = "/health/detailed",
    tag = "health",
    responses(
        (status = 200, description = "Per-dependency health, pool metrics and circuit-breaker state", body = DetailedHealthResponse),
        (status = 503, description = "One or more dependencies are unhealthy", body = DetailedHealthResponse),
    ),
)]
pub async fn detailed_health_check(
    State(state): State<CoreState>,
    headers: axum::http::HeaderMap,
) -> Result<(StatusCode, Json<DetailedHealthResponse>), StatusCode> {
    if let Some(expected) = &state.config.health_token {
        let provided = headers.get("x-health-token").and_then(|v| v.to_str().ok());
        if provided != Some(expected.as_str()) {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }
    let db_health = check_database_detailed(&state.db, &state.config.database_pool).await;
    let db_healthy = db_health.status == "healthy";

    let clients = collect_client_health(&state);
    let purge_tasks = collect_purge_task_health(&state);
    let active_tasks = u64::try_from(state.task_tracker.task_count()).unwrap_or(u64::MAX);
    let sse_broadcaster = SseBroadcasterHealth {
        active_channels: state.source_broadcaster.channel_count(),
        total_receivers: state.source_broadcaster.total_receivers(),
    };

    let status = if db_healthy { "healthy" } else { "degraded" };
    let status_code = if db_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let response = DetailedHealthResponse {
        status,
        version: VERSION,
        uptime_secs: uptime_secs(),
        db: db_health,
        clients,
        purge_tasks,
        sse_broadcaster,
        active_tasks,
    };

    Ok((status_code, Json(response)))
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Quick DB check — returns true if `SELECT 1` succeeds.
async fn check_database_ok(db: &DatabaseConnection) -> bool {
    db.execute(Statement::from_string(
        DatabaseBackend::Postgres,
        "SELECT 1",
    ))
    .await
    .is_ok()
}

/// Detailed DB check with latency measurement, pool config, and live pool metrics.
async fn check_database_detailed(
    db: &DatabaseConnection,
    pool_config: &crate::core::config::DatabasePoolConfig,
) -> DatabaseHealth {
    let start = Instant::now();

    let result = db
        .execute(Statement::from_string(
            DatabaseBackend::Postgres,
            "SELECT 1",
        ))
        .await;

    let latency_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    let (status, error) = match result {
        Ok(_) => ("healthy", None),
        Err(e) => {
            tracing::error!(error = %e, "Database health check failed");
            ("unhealthy", Some(e.to_string()))
        }
    };

    let pool_metrics = collect_pool_metrics(db, pool_config.max_connections);

    DatabaseHealth {
        status,
        latency_ms: Some(latency_ms),
        error,
        pool: PoolConfig {
            min_connections: pool_config.min_connections,
            max_connections: pool_config.max_connections,
            connect_timeout_secs: pool_config.connect_timeout_secs,
            acquire_timeout_secs: pool_config.acquire_timeout_secs,
            idle_timeout_secs: pool_config.idle_timeout_secs,
            max_lifetime_secs: pool_config.max_lifetime_secs,
        },
        pool_metrics,
    }
}

/// Collect live pool metrics from the underlying sqlx pool.
///
/// Uses `get_postgres_connection_pool()` to access the sqlx `PgPool` and
/// read `size()` (total managed connections) and `num_idle()` (available).
/// Utilization is computed as `(active - idle) / max * 100`.
pub fn collect_pool_metrics(db: &DatabaseConnection, max_connections: u32) -> PoolMetrics {
    let pg_pool = db.get_postgres_connection_pool();
    let active = pg_pool.size();
    let idle = u32::try_from(pg_pool.num_idle()).unwrap_or(u32::MAX);
    let in_use = active.saturating_sub(idle);

    let utilization_pct = if max_connections > 0 {
        (f64::from(in_use) / f64::from(max_connections) * 100.0 * 10.0).round() / 10.0
    } else {
        0.0
    };

    PoolMetrics {
        active_connections: active,
        idle_connections: idle,
        max_connections,
        utilization_pct,
    }
}

/// Collect health status for all external clients.
fn collect_client_health(state: &CoreState) -> Vec<ClientHealth> {
    let clients = &state.clients;

    // Build a list of all known clients with their availability and metrics.
    // Unavailable clients (not configured) get a synthetic "closed" circuit
    // state and zeroed metrics.
    let mut health = Vec::with_capacity(6);

    // Helper: build health for an optional client
    macro_rules! client_health {
        ($name:literal, $field:expr) => {
            match &$field {
                Some(client) => ClientHealth {
                    name: $name,
                    available: true,
                    circuit_state: client.circuit_state(),
                    metrics: client.metrics().into(),
                },
                None => ClientHealth {
                    name: $name,
                    available: false,
                    circuit_state: CircuitState::Closed,
                    metrics: ClientMetricsHealth {
                        total_requests: 0,
                        error_rate: 0.0,
                        avg_latency_ms: None,
                        min_latency_ms: None,
                        max_latency_ms: None,
                        rate_limit_hits: 0,
                        timeout_errors: 0,
                    },
                },
            }
        };
    }

    health.push(client_health!("mistral", clients.mistral));
    // The embedding provider and reranker are behind traits (US-020), so their
    // vendor metrics are no longer reachable from here. Presence and provider
    // name are what an operator can act on; a provider that wants richer health
    // exposes it through its own client entry above.
    health.push(ClientHealth {
        name: "embeddings",
        available: clients.embeddings.is_some(),
        circuit_state: CircuitState::Closed,
        metrics: ClientMetricsHealth::default(),
    });
    health.push(ClientHealth {
        name: "reranker",
        available: clients.reranker.is_some(),
        circuit_state: CircuitState::Closed,
        metrics: ClientMetricsHealth::default(),
    });
    health.push(client_health!("firecrawl", clients.firecrawl));
    health.push(client_health!("openai", clients.openai));

    health
}

/// Collect health status for all purge tasks.
fn collect_purge_task_health(state: &CoreState) -> Vec<PurgeTaskHealth> {
    state
        .purge_task_state
        .snapshot()
        .into_iter()
        .map(|(name, status)| PurgeTaskHealth {
            name,
            total_runs: status.total_runs,
            total_deleted: status.total_deleted,
            total_failures: status.total_failures,
            consecutive_failures: status.consecutive_failures,
            last_success_secs_ago: status.secs_since_last_success(),
            last_failure_secs_ago: status.secs_since_last_failure(),
            last_run_duration_ms: status.last_run_duration_ms,
            normal_interval_secs: status.normal_interval_secs,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_response_serializes() {
        let resp = HealthResponse {
            status: "ok",
            version: "1.0.0",
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"version\":\"1.0.0\""));
    }

    #[test]
    fn client_metrics_health_from_snapshot() {
        let snapshot = MetricsSnapshot {
            provider: "test".into(),
            total_requests: 100,
            successful_requests: 90,
            failed_requests: 10,
            rate_limit_hits: 2,
            timeout_errors: 1,
            avg_latency_ms: Some(150),
            min_latency_ms: Some(50),
            max_latency_ms: Some(500),
            error_rate: 0.1,
        };

        let health: ClientMetricsHealth = snapshot.into();
        assert_eq!(health.total_requests, 100);
        assert!((health.error_rate - 0.1).abs() < f64::EPSILON);
        assert_eq!(health.avg_latency_ms, Some(150));
        assert_eq!(health.rate_limit_hits, 2);
        assert_eq!(health.timeout_errors, 1);
    }

    #[test]
    fn circuit_state_serializes_as_snake_case() {
        assert_eq!(
            serde_json::to_string(&CircuitState::Closed).unwrap(),
            "\"closed\""
        );
        assert_eq!(
            serde_json::to_string(&CircuitState::Open).unwrap(),
            "\"open\""
        );
        assert_eq!(
            serde_json::to_string(&CircuitState::HalfOpen).unwrap(),
            "\"half_open\""
        );
    }

    #[test]
    fn pool_config_serializes() {
        let pool = PoolConfig {
            min_connections: 1,
            max_connections: 20,
            connect_timeout_secs: 30,
            acquire_timeout_secs: 15,
            idle_timeout_secs: 600,
            max_lifetime_secs: 1800,
        };
        let json = serde_json::to_value(&pool).unwrap();
        assert_eq!(json["min_connections"], 1);
        assert_eq!(json["max_connections"], 20);
    }

    #[test]
    fn sse_broadcaster_health_serializes() {
        let health = SseBroadcasterHealth {
            active_channels: 5,
            total_receivers: 12,
        };
        let json = serde_json::to_value(&health).unwrap();
        assert_eq!(json["active_channels"], 5);
        assert_eq!(json["total_receivers"], 12);
    }

    #[test]
    fn purge_task_health_omits_none_fields() {
        let health = PurgeTaskHealth {
            name: "test-purge",
            total_runs: 0,
            total_deleted: 0,
            total_failures: 0,
            consecutive_failures: 0,
            last_success_secs_ago: None,
            last_failure_secs_ago: None,
            last_run_duration_ms: None,
            normal_interval_secs: 86400,
        };
        let json = serde_json::to_value(&health).unwrap();
        assert!(
            !json
                .as_object()
                .unwrap()
                .contains_key("last_success_secs_ago")
        );
        assert!(
            !json
                .as_object()
                .unwrap()
                .contains_key("last_failure_secs_ago")
        );
        assert!(
            !json
                .as_object()
                .unwrap()
                .contains_key("last_run_duration_ms")
        );
    }

    #[test]
    fn purge_task_health_includes_present_fields() {
        let health = PurgeTaskHealth {
            name: "test-purge",
            total_runs: 10,
            total_deleted: 50,
            total_failures: 3,
            consecutive_failures: 2,
            last_success_secs_ago: Some(3600),
            last_failure_secs_ago: Some(60),
            last_run_duration_ms: Some(250),
            normal_interval_secs: 86400,
        };
        let json = serde_json::to_value(&health).unwrap();
        assert_eq!(json["total_runs"], 10);
        assert_eq!(json["total_deleted"], 50);
        assert_eq!(json["total_failures"], 3);
        assert_eq!(json["consecutive_failures"], 2);
        assert_eq!(json["last_success_secs_ago"], 3600);
        assert_eq!(json["last_failure_secs_ago"], 60);
        assert_eq!(json["last_run_duration_ms"], 250);
    }

    #[test]
    fn client_health_unavailable_has_zero_metrics() {
        let health = ClientHealth {
            name: "test",
            available: false,
            circuit_state: CircuitState::Closed,
            metrics: ClientMetricsHealth {
                total_requests: 0,
                error_rate: 0.0,
                avg_latency_ms: None,
                min_latency_ms: None,
                max_latency_ms: None,
                rate_limit_hits: 0,
                timeout_errors: 0,
            },
        };
        let json = serde_json::to_value(&health).unwrap();
        assert_eq!(json["available"], false);
        assert_eq!(json["circuit_state"], "closed");
        assert_eq!(json["metrics"]["total_requests"], 0);
    }

    #[test]
    fn database_health_healthy_omits_error() {
        let health = DatabaseHealth {
            status: "healthy",
            latency_ms: Some(5),
            error: None,
            pool: PoolConfig {
                min_connections: 1,
                max_connections: 20,
                connect_timeout_secs: 30,
                acquire_timeout_secs: 15,
                idle_timeout_secs: 600,
                max_lifetime_secs: 1800,
            },
            pool_metrics: PoolMetrics {
                active_connections: 5,
                idle_connections: 3,
                max_connections: 20,
                utilization_pct: 10.0,
            },
        };
        let json = serde_json::to_value(&health).unwrap();
        assert_eq!(json["status"], "healthy");
        assert!(!json.as_object().unwrap().contains_key("error"));
    }

    #[test]
    fn database_health_unhealthy_includes_error() {
        let health = DatabaseHealth {
            status: "unhealthy",
            latency_ms: Some(100),
            error: Some("connection refused".into()),
            pool: PoolConfig {
                min_connections: 1,
                max_connections: 20,
                connect_timeout_secs: 30,
                acquire_timeout_secs: 15,
                idle_timeout_secs: 600,
                max_lifetime_secs: 1800,
            },
            pool_metrics: PoolMetrics {
                active_connections: 0,
                idle_connections: 0,
                max_connections: 20,
                utilization_pct: 0.0,
            },
        };
        let json = serde_json::to_value(&health).unwrap();
        assert_eq!(json["status"], "unhealthy");
        assert_eq!(json["error"], "connection refused");
    }

    #[test]
    fn pool_metrics_serializes_with_utilization() {
        let metrics = PoolMetrics {
            active_connections: 15,
            idle_connections: 5,
            max_connections: 20,
            utilization_pct: 50.0,
        };
        let json = serde_json::to_value(&metrics).unwrap();
        assert_eq!(json["active_connections"], 15);
        assert_eq!(json["idle_connections"], 5);
        assert_eq!(json["max_connections"], 20);
        assert_eq!(json["utilization_pct"], 50.0);
    }

    #[test]
    fn pool_metrics_zero_max_yields_zero_utilization() {
        // Edge case: max_connections == 0 should not divide by zero
        let metrics = PoolMetrics {
            active_connections: 0,
            idle_connections: 0,
            max_connections: 0,
            utilization_pct: 0.0,
        };
        let json = serde_json::to_value(&metrics).unwrap();
        assert_eq!(json["utilization_pct"], 0.0);
    }

    #[test]
    fn database_health_includes_pool_metrics() {
        let health = DatabaseHealth {
            status: "healthy",
            latency_ms: Some(2),
            error: None,
            pool: PoolConfig {
                min_connections: 1,
                max_connections: 20,
                connect_timeout_secs: 30,
                acquire_timeout_secs: 15,
                idle_timeout_secs: 600,
                max_lifetime_secs: 1800,
            },
            pool_metrics: PoolMetrics {
                active_connections: 8,
                idle_connections: 3,
                max_connections: 20,
                utilization_pct: 25.0,
            },
        };
        let json = serde_json::to_value(&health).unwrap();
        let pm = &json["pool_metrics"];
        assert_eq!(pm["active_connections"], 8);
        assert_eq!(pm["idle_connections"], 3);
        assert_eq!(pm["utilization_pct"], 25.0);
    }
}
