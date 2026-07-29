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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    init_tracing();

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
    ensure_account(&db, account_id).await?;

    let (state, task_tracker) = build_core_state(&config, db);
    require_provider_capabilities(&state)?;

    let identity = ReferenceIdentity::new(mode, account_id);
    let app = build_app(state, &config, identity);

    let addr = SocketAddr::new(bind, config.server_port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "OpenbookLM core is ready.");

    let shutdown_timeout = Duration::from_secs(config.async_config.shutdown_timeout_secs);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            task_tracker.shutdown(shutdown_timeout).await;
        })
        .await?;

    tracing::info!("Server shut down successfully");
    Ok(())
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
    source_broadcaster.start_cleanup_task();

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
        purge_task_state: PurgeTaskState::new(),
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
        state.task_tracker.cancellation_token(),
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
