//! Core application state (US-004, EP-002).
//!
//! [`CoreState`] holds everything the open-source core needs to serve a
//! request: the database handle, core repositories, provider clients, the SSE
//! broadcaster, caches, task coordination and the two injected seams
//! ([`EntitlementPolicy`](crate::core::EntitlementPolicy) and
//! [`EventSink`](crate::core::EventSink)).
//!
//! It contains no Clerk, Stripe, Resend or PostHog type. The private SaaS
//! composition wraps it in `AppState` and derefs to it, so both editions share
//! one constructed instance of every repository and client.

use std::sync::Arc;

use sea_orm::DatabaseConnection;

use crate::clients::{
    ClientMetrics, FirecrawlClient, LlmRouter, MistralClient, MistralOcrClient, OpenAiClient,
    VoyageClient, VoyageRateLimiter,
};
use crate::core::config::CoreConfig;
use crate::core::entitlements::SharedEntitlementPolicy;
use crate::core::events::SharedEventSink;
use crate::core::providers::{
    DeterministicEmbedder, DeterministicLlm, SharedEmbeddingProvider, SharedReranker,
    deterministic_requested,
};
use crate::middleware::TaskTracker;
use crate::repositories::{
    SeaOrmAccountRepository, SeaOrmAccountSettingsRepository, SeaOrmChatRepository,
    SeaOrmChunkRepository, SeaOrmMemoryRepository, SeaOrmNoteRepository, SeaOrmNotebookRepository,
    SeaOrmOcrCacheRepository, SeaOrmRagLogRepository, SeaOrmSearchRepository,
    SeaOrmSourceRepository,
};
use crate::services::rag::embedding_cache::EmbeddingCache;
use crate::services::rag::hyde::HydeService;
use crate::services::rag::query_reformulation::QueryReformulator;
use crate::services::source_events::SourceEventBroadcaster;
use crate::types::{MemoryDecayTracker, PurgeTaskState};

// ============================================================================
// Repositories
// ============================================================================

/// Holds concrete core repository implementations behind `Arc` for shared access.
///
/// Uses concrete types instead of trait objects for simplicity. All public
/// methods on the concrete types come from trait impls (the only inherent
/// method is `new()`). Callers use trait method syntax exclusively, so
/// swapping implementations later requires changing only this struct
/// and `Repositories::new()`.
///
/// SaaS-only repositories (subscriptions, usage, identity mappings) live in the
/// private `SaasRepositories` struct instead.
///
/// **Test strategy**: Integration tests with a test database, not mock
/// repositories.
#[derive(Clone)]
pub struct Repositories {
    /// The core's ownership root (US-011).
    pub accounts: Arc<SeaOrmAccountRepository>,
    /// Core preferences: default provider and model (US-011).
    pub account_settings: Arc<SeaOrmAccountSettingsRepository>,
    pub notebooks: Arc<SeaOrmNotebookRepository>,
    pub sources: Arc<SeaOrmSourceRepository>,
    pub chunks: Arc<SeaOrmChunkRepository>,
    pub search: Arc<SeaOrmSearchRepository>,
    pub chat: Arc<SeaOrmChatRepository>,
    pub notes: Arc<SeaOrmNoteRepository>,
    pub rag_logs: Arc<SeaOrmRagLogRepository>,
    pub memory: Arc<SeaOrmMemoryRepository>,
    pub ocr_cache: Arc<SeaOrmOcrCacheRepository>,
}

impl Repositories {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self {
            accounts: Arc::new(SeaOrmAccountRepository::new(db)),
            account_settings: Arc::new(SeaOrmAccountSettingsRepository::new(db)),
            notebooks: Arc::new(SeaOrmNotebookRepository::new(db)),
            sources: Arc::new(SeaOrmSourceRepository::new(db)),
            chunks: Arc::new(SeaOrmChunkRepository::new(db)),
            search: Arc::new(SeaOrmSearchRepository::new(db)),
            chat: Arc::new(SeaOrmChatRepository::new(db)),
            notes: Arc::new(SeaOrmNoteRepository::new(db)),
            rag_logs: Arc::new(SeaOrmRagLogRepository::new(db)),
            memory: Arc::new(SeaOrmMemoryRepository::new(db)),
            ocr_cache: Arc::new(SeaOrmOcrCacheRepository::new(db)),
        }
    }
}

// ============================================================================
// External Clients
// ============================================================================

#[derive(Clone)]
pub struct ExternalClients {
    pub metrics: ClientMetrics,
    pub http_client: reqwest::Client,
    pub mistral: Option<MistralClient>,
    /// Text to vectors. A trait object, so the core never names a vendor (US-020).
    pub embeddings: Option<SharedEmbeddingProvider>,
    /// Optional cross-encoder. Retrieval degrades to unreranked results without it.
    pub reranker: Option<SharedReranker>,
    pub firecrawl: Option<FirecrawlClient>,
    pub openai: Option<OpenAiClient>,
    pub llm_router: LlmRouter,
    pub query_reformulator: Option<QueryReformulator>,
    pub hyde_service: Option<HydeService>,
    pub ocr: Option<MistralOcrClient>,
    pub youtube: Option<crate::clients::YouTubeClient>,
}

impl ExternalClients {
    pub fn from_config(config: &CoreConfig) -> Self {
        let metrics = ClientMetrics::new();

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("Failed to build shared HTTP client");

        let mistral = MistralClient::from_config(config, &metrics)
            .inspect_err(|e| tracing::warn!(error = %e, "Mistral client not available"))
            .ok();

        let voyage_rate_limiter =
            Arc::new(VoyageRateLimiter::new(config.voyage_rate_limit.clone()));

        // Deterministic providers are opt-in and total: when requested they
        // replace embeddings, reranking and generation together, so a run can
        // never be half fixture and half vendor.
        let deterministic = deterministic_requested();
        if deterministic {
            tracing::warn!(
                "Deterministic providers installed. Retrieval and generation are fixtures, \
                 not models — never serve real users with this."
            );
        }

        let voyage = VoyageClient::from_config(config, &metrics)
            .inspect_err(|e| tracing::warn!(error = %e, "Voyage client not available"))
            .ok()
            .map(|v| {
                v.with_rate_limiter(voyage_rate_limiter)
                    .with_batch_size(config.voyage_rate_limit.batch_size)
                    .with_concurrent_batches(config.voyage_rate_limit.concurrent_batches)
                    .with_model(&config.voyage_embedding_model)
            });

        let (embeddings, reranker): (Option<SharedEmbeddingProvider>, Option<SharedReranker>) =
            if deterministic {
                (Some(Arc::new(DeterministicEmbedder::new())), None)
            } else {
                let shared = voyage.map(Arc::new);
                (
                    shared.clone().map(|v| v as SharedEmbeddingProvider),
                    shared.map(|v| v as SharedReranker),
                )
            };

        let firecrawl = FirecrawlClient::from_config(config, &metrics)
            .inspect_err(|e| tracing::warn!(error = %e, "Firecrawl client not available"))
            .ok();

        let openai = crate::clients::OpenAiClient::from_config(config, &metrics)
            .inspect_err(|e| tracing::warn!(error = %e, "OpenAI client not available"))
            .ok();

        let llm_router = if deterministic {
            LlmRouter::single(Arc::new(DeterministicLlm))
        } else {
            LlmRouter::from_config(config, &metrics)
        };

        let query_reformulator = config.anthropic_api_key.as_deref().and_then(|key| {
            QueryReformulator::new(key, &metrics)
                .inspect_err(|e| tracing::warn!(error = %e, "Query reformulator not available"))
                .ok()
        });

        let hyde_service = if config.hyde_enabled {
            config.anthropic_api_key.as_deref().and_then(|key| {
                HydeService::new(key, &metrics)
                    .inspect_err(|e| tracing::warn!(error = %e, "HyDE service not available"))
                    .ok()
            })
        } else {
            None
        };

        let ocr = if config.ocr.enabled {
            MistralOcrClient::from_config(config, &metrics)
                .inspect_err(|e| tracing::warn!(error = %e, "Mistral OCR client not available"))
                .ok()
        } else {
            None
        };

        let youtube = crate::clients::YouTubeClient::from_config(config, &metrics);

        Self {
            metrics,
            http_client,
            mistral,
            embeddings,
            reranker,
            firecrawl,
            openai,
            llm_router,
            query_reformulator,
            hyde_service,
            ocr,
            youtube,
        }
    }

    pub fn log_status(&self) {
        tracing::info!(
            mistral = self.mistral.is_some(),
            embeddings = self.embeddings.as_ref().map(|e| e.name()),
            reranker = self.reranker.as_ref().map(|r| r.name()),
            firecrawl = self.firecrawl.is_some(),
            openai = self.openai.is_some(),
            youtube = self.youtube.is_some(),
            query_reformulator = self.query_reformulator.is_some(),
            hyde = self.hyde_service.is_some(),
            ocr = self.ocr.is_some(),
            llm_providers = self.llm_router.provider_count(),
            llm_default = self.llm_router.default_provider_name(),
            "External clients initialized"
        );
    }

    pub fn log_metrics(&self) {
        self.metrics.log_all();
    }
}

// ============================================================================
// Core state
// ============================================================================

/// Everything the open-source core needs to serve a request.
#[derive(Clone)]
pub struct CoreState {
    pub db: DatabaseConnection,
    pub config: Arc<CoreConfig>,
    pub task_tracker: TaskTracker,
    pub repos: Repositories,
    pub clients: ExternalClients,
    pub source_broadcaster: SourceEventBroadcaster,
    pub embedding_cache: EmbeddingCache,
    pub memory_decay_tracker: MemoryDecayTracker,
    pub purge_task_state: PurgeTaskState,
    /// Decides whether a bounded operation is allowed and records metered work.
    pub entitlements: SharedEntitlementPolicy,
    /// Receives non-critical side-effect events.
    pub events: SharedEventSink,
}

/// Inputs to [`CoreState::new`] that the composition root chooses.
///
/// `repos` and `clients` are passed in rather than constructed here so that a
/// composition root needing its own handle to a core repository (the hosted
/// entitlement adapter does) shares the one instance instead of building a
/// second.
pub struct CoreStateParts {
    pub db: DatabaseConnection,
    pub config: Arc<CoreConfig>,
    pub task_tracker: TaskTracker,
    pub repos: Repositories,
    pub clients: ExternalClients,
    pub source_broadcaster: SourceEventBroadcaster,
    pub purge_task_state: PurgeTaskState,
    pub entitlements: SharedEntitlementPolicy,
    pub events: SharedEventSink,
}

impl CoreState {
    pub fn new(parts: CoreStateParts) -> Self {
        Self {
            db: parts.db,
            config: parts.config,
            task_tracker: parts.task_tracker,
            repos: parts.repos,
            clients: parts.clients,
            source_broadcaster: parts.source_broadcaster,
            embedding_cache: EmbeddingCache::new(),
            memory_decay_tracker: MemoryDecayTracker::new(),
            purge_task_state: parts.purge_task_state,
            entitlements: parts.entitlements,
            events: parts.events,
        }
    }
}
