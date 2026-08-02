#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unimplemented
    )
)]

//! The OpenbookLM reference server (US-013, EP-004).
//!
//! A complete, supported self-hosted binary: the public core router, the public
//! core migration track, health endpoints, request IDs, security headers, rate
//! limiting and graceful shutdown. It needs PostgreSQL with pgvector and at
//! least one embedding provider and one LLM provider. It needs no Clerk,
//! Stripe, Resend or PostHog value and reads none.
//!
//! ```bash
//! DATABASE_URL=postgres://openbooklm@localhost/openbooklm \
//! VOYAGE_API_KEY=… ANTHROPIC_API_KEY=… \
//!   openbooklm-server
//! ```
//!
//! # Startup order
//!
//! Configuration, then identity, then database, then migrations, then the
//! socket. Every failure mode the PRD names is reachable *before* a port is
//! bound, so a misconfigured server never accepts a request it cannot serve.
//!
//! # What it deliberately does not do
//!
//! No user management, no plans, no analytics. Identity is one account
//! ([`core::identity`](openbooklm::core::identity)) and entitlements are
//! [`UnrestrictedPolicy`]: every structurally valid operation is authorised.
//! Multi-tenant hosting is the commercial edition's job, not a missing feature
//! here.

use std::future::Future;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use axum::{Router, extract::DefaultBodyLimit, middleware as axum_mw};
use openbooklm::{
    core::{
        CoreConfig, ReferenceIdentity, UnrestrictedPolicy,
        identity::{IdentityMode, account_id_from_env, reference_auth_middleware},
        router::{build_core_health_router, build_core_router, map_payload_too_large},
        state::{CoreState, CoreStateParts},
    },
    db,
    middleware::{
        RateLimiter, TaskTracker, build_cors_layer, create_rate_limit_middleware,
        request_id_middleware, security_headers_middleware, shutdown_signal,
    },
    repositories::{APPROVED_STRATEGY, VectorCapabilities},
    services::source_events::{SourceEventBroadcaster, SseCleanupConfig},
    types::PurgeTaskState,
};
use openbooklm_migration_core::{
    MigratorTrait,
    core_track::{CoreMigrator, with_migration_lock},
    sea_orm::ConnectionTrait,
    validate_core_state,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Address the listener binds. Defaults to loopback: a fresh install is not
/// reachable from the network until the operator says so, and saying so
/// requires a token (see [`IdentityMode`]).
const BIND_ENV: &str = "OPENBOOKLM_BIND";
const DEFAULT_BIND: &str = "127.0.0.1";

/// Set to `1`/`true` to emit domain events to the tracing subscriber instead of
/// discarding them.
const EVENT_LOG_ENV: &str = "OPENBOOKLM_EVENT_LOG";

fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    init_tracing();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(run());

    // Application-owned async work has already drained or been aborted. Tokio
    // cannot stop a spawn_blocking closure that exceeded that budget, so waiting
    // again here would make the configured shutdown deadline untrue.
    runtime.shutdown_timeout(Duration::ZERO);
    result
}

async fn run() -> anyhow::Result<()> {
    // 1. Configuration. Every missing or malformed core value is reported at
    //    once; commercial variables are neither read nor required.
    let config = CoreConfig::load().map_err(|e| {
        anyhow::anyhow!(
            "configuration is invalid:\n{e}\n\nSee .env.example for the core variables."
        )
    })?;

    // 2. Identity. Resolved before the socket exists, so single-user mode
    //    cannot end up serving a network address.
    let bind: IpAddr = std::env::var(BIND_ENV)
        .unwrap_or_else(|_| DEFAULT_BIND.to_string())
        .parse()
        .map_err(|_| anyhow::anyhow!("{BIND_ENV} is not a valid IP address"))?;
    let mode = IdentityMode::from_env(bind)?;
    let account_id = account_id_from_env()?;
    tracing::info!(mode = mode.label(), %account_id, "Identity configured");

    // 3. Database, then migrations, then the account row the single operator owns.
    let db = db::connect(&config.database_url, &config.database_pool).await?;
    apply_core_migrations(&db).await?;
    require_vector_capabilities(&db).await?;
    ensure_account(&db, account_id).await?;

    let (state, task_tracker) = build_core_state(&config, db);
    let shutdown_timeout = Duration::from_secs(config.async_config.shutdown_timeout_secs);
    let server_owner = task_tracker.clone();
    let server = async move {
        require_provider_capabilities(&state)?;
        openbooklm::services::maintenance::start_maintenance_task(
            &state.task_tracker,
            &state.purge_task_state,
            state.repos.rag_logs.clone(),
            state.repos.ocr_cache.clone(),
            state.repos.generations.clone(),
        );

        let identity = ReferenceIdentity::new(mode, account_id);
        let app = build_app(state, &config, identity);

        let addr = SocketAddr::new(bind, config.server_port);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!(%addr, "OpenbookLM core is ready.");

        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown_signal().await;
                server_owner.begin_shutdown();
            })
            .await?;
        Ok::<(), anyhow::Error>(())
    };

    run_server_lifecycle(server, task_tracker, shutdown_timeout).await
}

async fn run_server_lifecycle<F>(
    server: F,
    task_tracker: TaskTracker,
    shutdown_timeout: Duration,
) -> anyhow::Result<()>
where
    F: Future<Output = anyhow::Result<()>>,
{
    let shutdown_observer = task_tracker.cancellation_token();
    tokio::pin!(server);
    let mut forced = false;
    let server_result = tokio::select! {
        biased;
        () = shutdown_observer.cancelled() => {
            let drain = async {
                let (server_result, ()) = tokio::join!(server.as_mut(), task_tracker.wait());
                server_result
            };
            match tokio::time::timeout(shutdown_timeout, drain).await {
                Ok(server_result) => server_result,
                Err(_) => {
                    forced = true;
                    let aborted = task_tracker.abort_remaining();
                    tracing::warn!(
                        aborted,
                        remaining_tasks = task_tracker.task_count(),
                        remaining_chat_streams = task_tracker.active_stream_count(),
                        timeout_secs = shutdown_timeout.as_secs(),
                        "Shutdown deadline reached; aborting remaining async work"
                    );
                    tokio::task::yield_now().await;
                    Ok(())
                }
            }
        }
        server_result = server.as_mut() => {
            // Startup failures and unexpected server exits use the same root
            // cancellation, bounded drain and abort path as a process signal.
            task_tracker.begin_shutdown();
            if tokio::time::timeout(shutdown_timeout, task_tracker.wait()).await.is_err() {
                forced = true;
                let aborted = task_tracker.abort_remaining();
                tracing::warn!(
                    aborted,
                    remaining_tasks = task_tracker.task_count(),
                    timeout_secs = shutdown_timeout.as_secs(),
                    "Shutdown deadline reached after server exit; aborting remaining async work"
                );
                tokio::task::yield_now().await;
            }
            server_result
        }
    };

    if forced {
        tracing::warn!("Server stopped after the graceful shutdown deadline");
    } else {
        tracing::info!("Server shut down successfully");
    }
    server_result
}

fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::new(
        std::env::var("RUST_LOG").unwrap_or_else(|_| "info,tower_http=debug".into()),
    );
    let registry = tracing_subscriber::registry().with(env_filter);

    if std::env::var("LOG_FORMAT")
        .unwrap_or_default()
        .eq_ignore_ascii_case("json")
    {
        registry
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        registry.with(tracing_subscriber::fmt::layer()).init();
    }
}

/// Validate the migration state, then apply the core track under the advisory
/// lock.
///
/// Validation runs first and unconditionally: a database this build cannot
/// account for stops startup with a remediation command rather than receiving
/// more SQL. The lock makes two instances started together serialise.
async fn apply_core_migrations(db: &sea_orm::DatabaseConnection) -> anyhow::Result<()> {
    let state = validate_core_state(db).await?;
    if let Some(remediation) = state.remediation() {
        anyhow::bail!("{remediation}");
    }
    tracing::info!(
        state = ?state.kind,
        applied = state.core_applied,
        "Core migration state validated"
    );

    with_migration_lock(db, async || CoreMigrator::up(db, None).await).await?;
    tracing::info!("Core schema up to date");
    Ok(())
}

/// Refuse to start on a pgvector build that cannot run the approved filtered
/// scan (US-016).
///
/// Failing here rather than at query time is the whole point: an unsupported
/// build does not error on retrieval, it silently returns fewer rows than the
/// filter had available, which reaches the user as a notebook with less
/// evidence and reaches the operator as nothing at all.
async fn require_vector_capabilities(db: &sea_orm::DatabaseConnection) -> anyhow::Result<()> {
    let capabilities = VectorCapabilities::probe(db).await?;
    capabilities.ensure_supports(APPROVED_STRATEGY)?;
    tracing::info!(
        pgvector = %capabilities.extension_version,
        strategy = %APPROVED_STRATEGY.label(),
        "Filtered dense retrieval strategy available"
    );
    Ok(())
}

/// Create the single operator's account row if it does not exist.
///
/// Ownership foreign keys point at `accounts(id)`, so the row must exist before
/// the first notebook is created. `ON CONFLICT DO NOTHING` makes a restart and
/// a reattached volume both no-ops.
async fn ensure_account(
    db: &sea_orm::DatabaseConnection,
    account_id: uuid::Uuid,
) -> anyhow::Result<()> {
    use openbooklm_migration_core::sea_orm::{DatabaseBackend, Statement};

    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO accounts (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
        [account_id.into()],
    ))
    .await?;
    Ok(())
}

fn build_core_state(
    config: &CoreConfig,
    db: sea_orm::DatabaseConnection,
) -> (CoreState, TaskTracker) {
    let task_tracker = TaskTracker::new();
    let repos = openbooklm::Repositories::new(&db);
    let clients = openbooklm::ExternalClients::from_config(config);
    clients.log_status();

    let source_broadcaster =
        SourceEventBroadcaster::with_cleanup_config(SseCleanupConfig::from_env());
    source_broadcaster.start_cleanup_task(&task_tracker);

    let purge_task_state = PurgeTaskState::new();

    let events: openbooklm::core::SharedEventSink = if env_flag(EVENT_LOG_ENV) {
        Arc::new(openbooklm::core::TracingEventSink)
    } else {
        Arc::new(openbooklm::core::NoopEventSink)
    };

    let state = CoreState::new(CoreStateParts {
        db,
        config: Arc::new(config.clone()),
        task_tracker: task_tracker.clone(),
        repos,
        clients,
        source_broadcaster,
        purge_task_state,
        // Self-hosting has no plans, no quotas and no metering.
        entitlements: Arc::new(UnrestrictedPolicy),
        events,
    });

    (state, task_tracker)
}

/// Refuse to start without the capabilities the product is made of, or with an
/// embedding provider the schema cannot store.
///
/// Retrieval needs an embedding provider and chat needs an LLM provider. A
/// server missing either would accept an upload and then fail every request,
/// which is worse than not starting. A provider whose vectors are the wrong
/// width is worse still: the ingestion would appear to succeed and no query
/// would ever match. The messages name the capability and the variable that
/// supplies it, and print no credential.
fn require_provider_capabilities(state: &CoreState) -> anyhow::Result<()> {
    let mut missing = Vec::new();

    match state.clients.embeddings.as_deref() {
        None => missing.push(
            "embeddings: set VOYAGE_API_KEY. Retrieval cannot index or search a source without it."
                .to_owned(),
        ),
        Some(embedder) => {
            openbooklm::core::providers::check_dimension(embedder)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            tracing::info!(
                provider = embedder.name(),
                model = embedder.model(),
                dimension = embedder.dimension(),
                "Embedding provider ready"
            );
        }
    }

    if state.clients.llm_router.provider_count() == 0 {
        missing.push(
            "chat: set at least one of ANTHROPIC_API_KEY, OPENAI_API_KEY or MISTRAL_API_KEY."
                .to_owned(),
        );
    }

    if missing.is_empty() {
        Ok(())
    } else {
        anyhow::bail!(
            "No compatible provider is configured.\n  - {}",
            missing.join("\n  - ")
        )
    }
}

fn build_app(state: CoreState, config: &CoreConfig, identity: ReferenceIdentity) -> Router {
    let rate_limiter = RateLimiter::new(
        config.security.rate_limit_rpm,
        state.task_tracker.clone(),
        config.upstash_redis_url.as_deref(),
        config.upstash_redis_token.as_deref(),
    );

    let protected = build_core_router(config)
        .layer(axum_mw::from_fn_with_state(
            identity,
            reference_auth_middleware,
        ))
        .with_state(state.clone());

    let health = build_core_health_router().with_state(state);

    Router::new()
        .merge(protected)
        .merge(health)
        .layer(axum_mw::map_response(map_payload_too_large))
        .layer(DefaultBodyLimit::max(config.security.body_limit_bytes))
        .layer(axum_mw::from_fn(security_headers_middleware))
        .layer(axum_mw::from_fn(request_id_middleware))
        .layer(axum_mw::from_fn(create_rate_limit_middleware(
            rate_limiter,
            config.security.trusted_proxy_mode,
        )))
        .layer(build_cors_layer(&config.security))
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[tokio::test]
    async fn startup_failure_cancels_and_drains_root_tasks() {
        let task_tracker = TaskTracker::new();
        let shutdown = task_tracker.cancellation_token();
        let cleaned = Arc::new(AtomicBool::new(false));
        let cleaned_by_task = Arc::clone(&cleaned);
        task_tracker
            .try_spawn("startup-cleanup", async move {
                shutdown.cancelled().await;
                cleaned_by_task.store(true, Ordering::SeqCst);
            })
            .expect("task admission");

        let result = run_server_lifecycle(
            async { anyhow::bail!("startup failed") },
            task_tracker,
            Duration::from_secs(1),
        )
        .await;

        assert!(result.is_err());
        assert!(cleaned.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn unexpected_server_exit_aborts_work_past_the_deadline() {
        let task_tracker = TaskTracker::new();
        task_tracker
            .try_spawn("stuck-cleanup", std::future::pending())
            .expect("task admission");

        run_server_lifecycle(
            std::future::ready(Ok(())),
            task_tracker.clone(),
            Duration::from_millis(10),
        )
        .await
        .expect("server result");

        tokio::time::timeout(Duration::from_millis(50), async {
            while task_tracker.task_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("aborted work must leave the root scope");
    }
}
