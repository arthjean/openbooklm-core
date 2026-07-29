//! Core configuration loaded from environment variables (US-004, EP-002).
//!
//! [`CoreConfig`] holds every setting the open-source core needs to run:
//! database, HTTP, RAG, providers, OCR and rate limiting. It deliberately
//! contains no Clerk, Stripe, Resend or PostHog field — those live in the
//! private [`crate::config::Config`] wrapper, which derefs to this struct.
//!
//! Validation collects every error (not fail-on-first) so an operator sees all
//! misconfiguration in a single pass.

use std::{env, str::FromStr};

// ============================================================================
// Configuration error type
// ============================================================================

/// Error returned when configuration validation fails.
///
/// Contains all validation errors at once so operators can fix everything
/// in a single pass rather than playing whack-a-mole.
#[derive(Debug)]
pub struct ConfigError {
    pub errors: Vec<String>,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Configuration validation failed ({} error(s)):",
            self.errors.len()
        )?;
        for (i, error) in self.errors.iter().enumerate() {
            write!(f, "\n  {}. {}", i + 1, error)?;
        }
        Ok(())
    }
}

impl std::error::Error for ConfigError {}

// ============================================================================
// Helper functions for env var parsing
// ============================================================================

pub(crate) fn env_var(key: &str) -> Result<String, env::VarError> {
    env::var(key)
}

pub(crate) fn env_or<T: FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn env_or_else<F: FnOnce() -> String>(key: &str, default: F) -> String {
    env::var(key).unwrap_or_else(|_| default())
}

pub(crate) fn env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .map(|v| !matches!(v.to_lowercase().as_str(), "false" | "0" | "no"))
        .unwrap_or(default)
}

fn env_list(key: &str, default: Vec<String>) -> Vec<String> {
    env::var(key)
        .map(|s| s.split(',').map(|o| o.trim().to_string()).collect())
        .unwrap_or(default)
}

// ============================================================================
// Configuration structs
// ============================================================================

#[derive(Clone, Debug)]
pub struct DatabasePoolConfig {
    pub min_connections: u32,
    pub max_connections: u32,
    pub connect_timeout_secs: u64,
    pub acquire_timeout_secs: u64,
    pub idle_timeout_secs: u64,
    pub max_lifetime_secs: u64,
}

impl Default for DatabasePoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 0,       // Allow Neon compute to fully suspend (scale to zero)
            max_connections: 10,      // Neon 0.25 CU supports ~97 connections; keep pool modest
            connect_timeout_secs: 15, // Budget for Neon cold start (~500ms) + TLS + network
            acquire_timeout_secs: 15, // Same budget for pool slot acquisition
            idle_timeout_secs: 270,   // 4.5 min — evict before Neon's 5 min suspend window
            max_lifetime_secs: 1800,  // 30 minutes — periodic refresh catches stale connections
        }
    }
}

impl DatabasePoolConfig {
    pub fn from_env() -> Self {
        Self {
            min_connections: env_or("DB_POOL_MIN", 0),
            max_connections: env_or("DB_POOL_MAX", 10),
            connect_timeout_secs: env_or("DB_CONNECT_TIMEOUT", 15),
            acquire_timeout_secs: env_or("DB_ACQUIRE_TIMEOUT", 15),
            idle_timeout_secs: env_or("DB_IDLE_TIMEOUT", 270),
            max_lifetime_secs: env_or("DB_MAX_LIFETIME", 1800),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SecurityConfig {
    pub allowed_origins: Vec<String>,
    pub max_search_query_length: usize,
    pub max_source_size_bytes: usize,
    pub max_source_words: usize,
    pub rate_limit_rpm: u32,
    pub body_limit_bytes: usize,
    pub upload_limit_bytes: usize,
    /// When true (default in production), trust `X-Real-IP` / `X-Forwarded-For`
    /// headers set by the reverse proxy (Caddy) for rate limiting.
    /// When false, ignore proxy headers and use the TCP socket IP instead.
    pub trusted_proxy_mode: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            allowed_origins: vec!["http://localhost:3000".to_string()],
            max_search_query_length: 500,
            max_source_size_bytes: 200 * 1024 * 1024, // 200MB
            max_source_words: 500_000,
            rate_limit_rpm: 300,
            body_limit_bytes: 1024 * 1024,         // 1MB
            upload_limit_bytes: 200 * 1024 * 1024, // 200MB
            trusted_proxy_mode: true,
        }
    }
}

impl SecurityConfig {
    fn from_env(frontend_url: &str) -> Self {
        Self {
            allowed_origins: env_list("CORS_ALLOWED_ORIGINS", vec![frontend_url.to_string()]),
            max_search_query_length: env_or("MAX_SEARCH_QUERY_LENGTH", 500),
            max_source_size_bytes: env_or("MAX_SOURCE_SIZE_BYTES", 200 * 1024 * 1024),
            max_source_words: env_or("MAX_SOURCE_WORDS", 500_000),
            rate_limit_rpm: env_or("RATE_LIMIT_RPM", 300),
            body_limit_bytes: env_or("BODY_LIMIT_BYTES", 1024 * 1024),
            upload_limit_bytes: env_or("UPLOAD_LIMIT_BYTES", 200 * 1024 * 1024),
            trusted_proxy_mode: env_bool("TRUSTED_PROXY_MODE", true),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AsyncConfig {
    pub llm_timeout_secs: u64,
    pub embedding_timeout_secs: u64,
    pub shutdown_timeout_secs: u64,
    pub request_timeout_secs: u64,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            llm_timeout_secs: 60,
            embedding_timeout_secs: 120,
            shutdown_timeout_secs: 30,
            request_timeout_secs: 30,
        }
    }
}

impl AsyncConfig {
    fn from_env() -> Self {
        Self {
            llm_timeout_secs: env_or("LLM_TIMEOUT_SECS", 60),
            embedding_timeout_secs: env_or("EMBEDDING_TIMEOUT_SECS", 120),
            shutdown_timeout_secs: env_or("SHUTDOWN_TIMEOUT_SECS", 30),
            request_timeout_secs: env_or("REQUEST_TIMEOUT_SECS", 30),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HybridSearchConfig {
    pub enabled: bool,
    pub rrf_k: f32,
    pub dense_weight: f32,
    pub sparse_weight: f32,
}

impl Default for HybridSearchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rrf_k: 60.0,
            // 4:1 dense/BM25 ratio per Anthropic Contextual Retrieval recommendations
            // See also: services::search::DEFAULT_DENSE_WEIGHT / DEFAULT_SPARSE_WEIGHT
            dense_weight: 1.0,
            sparse_weight: 0.25,
        }
    }
}

impl HybridSearchConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool("HYBRID_SEARCH_ENABLED", true),
            rrf_k: env_or("HYBRID_SEARCH_RRF_K", 60.0),
            dense_weight: env_or("HYBRID_SEARCH_DENSE_WEIGHT", 1.0),
            sparse_weight: env_or("HYBRID_SEARCH_SPARSE_WEIGHT", 0.25),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoyageRateLimitConfig {
    /// Maximum requests per minute (valid: 1–10,000; Tier 1 allows 2,000, default 300 for headroom)
    pub max_rpm: u32,
    /// Maximum tokens per minute (valid: 1,000–50,000,000; Tier 1 allows 8M, default 2M for headroom)
    pub max_tpm: u32,
    /// Batch size for embedding requests (valid: 1–1,000; Voyage AI documented max is 128 for standard requests)
    pub batch_size: usize,
    /// Number of embedding batches to send concurrently (valid: 1–16; default 4)
    pub concurrent_batches: usize,
    /// Maximum time to wait for rate limit budget (seconds)
    pub max_wait_secs: u64,
}

impl Default for VoyageRateLimitConfig {
    fn default() -> Self {
        Self {
            max_rpm: 300,
            max_tpm: 2_000_000,
            batch_size: 128,
            concurrent_batches: 4,
            max_wait_secs: 60,
        }
    }
}

impl VoyageRateLimitConfig {
    fn from_env() -> Self {
        Self {
            max_rpm: env_or("VOYAGE_MAX_RPM", 300),
            max_tpm: env_or("VOYAGE_MAX_TPM", 2_000_000),
            batch_size: env_or("VOYAGE_BATCH_SIZE", 128),
            concurrent_batches: env_or("VOYAGE_CONCURRENT_BATCHES", 4),
            max_wait_secs: env_or("VOYAGE_MAX_WAIT_SECS", 60),
        }
    }
}

#[derive(Clone, Debug)]
pub struct OcrConfig {
    /// Whether OCR fallback is enabled for scanned PDFs.
    pub enabled: bool,
    /// Mistral OCR model identifier.
    pub model: String,
    /// Timeout for OCR API requests in seconds.
    pub timeout_secs: u64,
    /// Minimum characters per page below which OCR is triggered.
    pub min_text_threshold: usize,
    /// Maximum pages per OCR API request (Mistral limit: 1,000).
    pub max_pages_per_request: usize,
    /// Maximum PDF file size in bytes for OCR (Mistral limit: 50 MB).
    pub max_file_size_bytes: usize,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: "mistral-ocr-latest".to_string(),
            timeout_secs: 120,
            min_text_threshold: 50,
            max_pages_per_request: 1000,
            max_file_size_bytes: MISTRAL_OCR_MAX_FILE_SIZE,
        }
    }
}

impl OcrConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool("OCR_ENABLED", false),
            model: env_or_else("OCR_MODEL", || "mistral-ocr-latest".to_string())
                .trim()
                .to_string(),
            timeout_secs: env_or("OCR_TIMEOUT_SECS", 120),
            min_text_threshold: env_or("OCR_MIN_TEXT_THRESHOLD", 50),
            max_pages_per_request: env_or("OCR_MAX_PAGES_PER_REQUEST", 1000),
            max_file_size_bytes: env_or("OCR_MAX_FILE_SIZE_BYTES", MISTRAL_OCR_MAX_FILE_SIZE),
        }
    }
}

// ============================================================================
// Core config
// ============================================================================

/// Configuration required by the open-source core.
///
/// Contains no commercial-service field. A deployment that sets no Clerk,
/// Stripe, Resend or PostHog variable still produces a valid `CoreConfig`.
#[derive(Clone, Debug)]
pub struct CoreConfig {
    // Database
    pub database_url: String,
    pub database_pool: DatabasePoolConfig,
    // Server
    pub server_port: u16,
    pub frontend_url: String,
    // Sub-configs
    pub security: SecurityConfig,
    pub async_config: AsyncConfig,
    pub hybrid_search: HybridSearchConfig,
    pub voyage_rate_limit: VoyageRateLimitConfig,
    pub ocr: OcrConfig,
    // Voyage AI embedding model
    pub voyage_embedding_model: String,
    // RAG pipeline tuning
    pub retrieval_pool_size: i32,
    pub hyde_enabled: bool,
    pub rerank_min_chunks: i32,
    pub context_stuffing_max_chunks: i32,
    // Processing services (optional)
    pub firecrawl_api_key: Option<String>,
    pub voyage_api_key: Option<String>,
    // LLM providers (optional)
    pub anthropic_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub mistral_api_key: Option<String>,
    // Upstash Redis (optional, enables distributed rate limiting)
    pub upstash_redis_url: Option<String>,
    pub upstash_redis_token: Option<String>,
    // Health check protection (optional, guards /health/detailed)
    pub health_token: Option<String>,
    // YouTube (optional, proxy for transcript extraction)
    pub youtube_proxy_url: Option<String>,
    // YouTube InnerTube API key (optional, falls back to embedded public key)
    pub youtube_innertube_api_key: Option<String>,
    // Default locale for transcript language selection
    pub default_locale: Option<String>,
}

/// Timeout bounds for operational timeouts (connect, acquire, request, etc.).
const MIN_TIMEOUT_SECS: u64 = 1;
const MAX_TIMEOUT_SECS: u64 = 600;

/// Maximum pool size — prevents accidental resource exhaustion.
const MAX_POOL_CONNECTIONS: u32 = 100;

/// Mistral OCR maximum file size per request (50 MB).
pub const MISTRAL_OCR_MAX_FILE_SIZE: usize = 50 * 1024 * 1024;

/// Push an error if `value` is outside `[MIN_TIMEOUT_SECS, MAX_TIMEOUT_SECS]`.
fn validate_timeout(errors: &mut Vec<String>, name: &str, value: u64) {
    if !(MIN_TIMEOUT_SECS..=MAX_TIMEOUT_SECS).contains(&value) {
        errors.push(format!(
            "{name} must be between {MIN_TIMEOUT_SECS} and {MAX_TIMEOUT_SECS} seconds, got {value}"
        ));
    }
}

/// Push an error if `value` is empty or whitespace-only.
pub(crate) fn check_required(errors: &mut Vec<String>, name: &str, value: &str) {
    if value.trim().is_empty() {
        errors.push(format!("{name} is required but not set or empty"));
    }
}

impl CoreConfig {
    /// Load the core configuration from environment variables.
    ///
    /// Does not validate — the caller composes core and (optionally) private
    /// validation so that every error is reported in one response.
    pub fn from_env() -> Self {
        let frontend_url = env_or_else("FRONTEND_URL", || "http://localhost:3000".to_string());

        Self {
            // Required (use empty string if missing — validate() will catch it)
            database_url: env_or_else("DATABASE_URL", String::new),
            server_port: env_or("PORT", 3001),
            frontend_url: frontend_url.clone(),
            // Sub-configs
            database_pool: DatabasePoolConfig::from_env(),
            security: SecurityConfig::from_env(&frontend_url),
            async_config: AsyncConfig::from_env(),
            hybrid_search: HybridSearchConfig::from_env(),
            voyage_rate_limit: VoyageRateLimitConfig::from_env(),
            ocr: OcrConfig::from_env(),
            // Voyage AI embedding model
            voyage_embedding_model: env_or_else("VOYAGE_EMBEDDING_MODEL", || {
                "voyage-4".to_string()
            }),
            // RAG pipeline tuning
            retrieval_pool_size: env_or("RETRIEVAL_POOL_SIZE", 50),
            hyde_enabled: env_bool("HYDE_ENABLED", false),
            rerank_min_chunks: env_or("RERANK_MIN_CHUNKS", 200),
            context_stuffing_max_chunks: env_or("CONTEXT_STUFFING_MAX_CHUNKS", 150),
            // Optional API keys
            firecrawl_api_key: env_var("FIRECRAWL_API_KEY").ok(),
            voyage_api_key: env_var("VOYAGE_API_KEY").ok(),
            anthropic_api_key: env_var("ANTHROPIC_API_KEY").ok(),
            openai_api_key: env_var("OPENAI_API_KEY").ok(),
            mistral_api_key: env_var("MISTRAL_API_KEY").ok(),
            upstash_redis_url: env_var("UPSTASH_REDIS_URL").ok(),
            upstash_redis_token: env_var("UPSTASH_REDIS_TOKEN").ok(),
            health_token: env_var("HEALTH_TOKEN").ok(),
            youtube_proxy_url: env_var("YOUTUBE_PROXY_URL").ok(),
            youtube_innertube_api_key: env_var("YOUTUBE_INNERTUBE_API_KEY").ok(),
            default_locale: env_var("DEFAULT_LOCALE").ok(),
        }
    }

    /// Load and validate the core configuration on its own.
    ///
    /// This is the entry point for the public reference server: it never looks
    /// at a Clerk, Stripe, Resend or PostHog variable.
    pub fn load() -> Result<Self, ConfigError> {
        let config = Self::from_env();
        config.validate()?;
        Ok(config)
    }

    /// Validate every core value, returning all errors at once.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();
        self.collect_errors(&mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError { errors })
        }
    }

    /// Append every core validation error to `errors`.
    ///
    /// Exposed to the private wrapper so that core and SaaS misconfiguration
    /// are reported in a single response rather than two startup attempts.
    pub(crate) fn collect_errors(&self, errors: &mut Vec<String>) {
        check_required(errors, "DATABASE_URL", &self.database_url);
        self.validate_database(errors);
        self.validate_timeouts(errors);
        self.validate_urls(errors);
        self.validate_rag_pipeline(errors);
        self.validate_voyage(errors);
        self.validate_ocr(errors);
    }

    fn validate_database(&self, errors: &mut Vec<String>) {
        // Database URL format (skip if empty — already reported by check_required)
        if !self.database_url.is_empty()
            && !self.database_url.starts_with("postgres://")
            && !self.database_url.starts_with("postgresql://")
        {
            errors.push(format!(
                "DATABASE_URL must start with postgres:// or postgresql://, got: {}",
                &self.database_url[..self.database_url.len().min(30)]
            ));
        }

        // Pool size bounds
        if self.database_pool.min_connections > self.database_pool.max_connections {
            errors.push(format!(
                "DB_POOL_MIN ({}) must be <= DB_POOL_MAX ({})",
                self.database_pool.min_connections, self.database_pool.max_connections
            ));
        }
        if self.database_pool.max_connections > MAX_POOL_CONNECTIONS {
            errors.push(format!(
                "DB_POOL_MAX ({}) must be <= {MAX_POOL_CONNECTIONS}",
                self.database_pool.max_connections
            ));
        }
    }

    fn validate_timeouts(&self, errors: &mut Vec<String>) {
        validate_timeout(
            errors,
            "DB_CONNECT_TIMEOUT",
            self.database_pool.connect_timeout_secs,
        );
        validate_timeout(
            errors,
            "DB_ACQUIRE_TIMEOUT",
            self.database_pool.acquire_timeout_secs,
        );
        validate_timeout(
            errors,
            "LLM_TIMEOUT_SECS",
            self.async_config.llm_timeout_secs,
        );
        validate_timeout(
            errors,
            "EMBEDDING_TIMEOUT_SECS",
            self.async_config.embedding_timeout_secs,
        );
        validate_timeout(
            errors,
            "REQUEST_TIMEOUT_SECS",
            self.async_config.request_timeout_secs,
        );
        validate_timeout(
            errors,
            "SHUTDOWN_TIMEOUT_SECS",
            self.async_config.shutdown_timeout_secs,
        );
    }

    fn validate_urls(&self, errors: &mut Vec<String>) {
        if !self.frontend_url.starts_with("http://") && !self.frontend_url.starts_with("https://") {
            errors.push(format!(
                "FRONTEND_URL must start with http:// or https://, got: {}",
                self.frontend_url
            ));
        }
    }

    fn validate_rag_pipeline(&self, errors: &mut Vec<String>) {
        if self.retrieval_pool_size < 20 {
            errors.push(format!(
                "RETRIEVAL_POOL_SIZE must be >= 20 (RERANK_TOP_K), got {}",
                self.retrieval_pool_size
            ));
        }
        if self.retrieval_pool_size > 500 {
            errors.push(format!(
                "RETRIEVAL_POOL_SIZE must be <= 500, got {}",
                self.retrieval_pool_size
            ));
        }
        if self.rerank_min_chunks < 0 {
            errors.push(format!(
                "RERANK_MIN_CHUNKS must be >= 0, got {}",
                self.rerank_min_chunks
            ));
        }
        if self.rerank_min_chunks > 10000 {
            errors.push(format!(
                "RERANK_MIN_CHUNKS must be <= 10000, got {}",
                self.rerank_min_chunks
            ));
        }
        if self.context_stuffing_max_chunks < 0 {
            errors.push(format!(
                "CONTEXT_STUFFING_MAX_CHUNKS must be >= 0, got {}",
                self.context_stuffing_max_chunks
            ));
        }
        if self.context_stuffing_max_chunks > 500 {
            errors.push(format!(
                "CONTEXT_STUFFING_MAX_CHUNKS must be <= 500, got {}",
                self.context_stuffing_max_chunks
            ));
        }
    }

    fn validate_voyage(&self, errors: &mut Vec<String>) {
        // Warn on unknown model, don't reject — allows new models
        const KNOWN_MODELS: &[&str] = &["voyage-4", "voyage-4-lite", "voyage-4-large"];

        if !(1..=10_000).contains(&self.voyage_rate_limit.max_rpm) {
            errors.push(format!(
                "VOYAGE_MAX_RPM must be between 1 and 10,000, got {}",
                self.voyage_rate_limit.max_rpm
            ));
        }
        if !(1_000..=50_000_000).contains(&self.voyage_rate_limit.max_tpm) {
            errors.push(format!(
                "VOYAGE_MAX_TPM must be between 1,000 and 50,000,000, got {}",
                self.voyage_rate_limit.max_tpm
            ));
        }
        if !(1..=1_000).contains(&self.voyage_rate_limit.batch_size) {
            errors.push(format!(
                "VOYAGE_BATCH_SIZE must be between 1 and 1,000, got {}",
                self.voyage_rate_limit.batch_size
            ));
        }
        if !(1..=16).contains(&self.voyage_rate_limit.concurrent_batches) {
            errors.push(format!(
                "VOYAGE_CONCURRENT_BATCHES must be between 1 and 16, got {}",
                self.voyage_rate_limit.concurrent_batches
            ));
        }

        if self.voyage_embedding_model.trim().is_empty() {
            errors.push("VOYAGE_EMBEDDING_MODEL must not be empty".to_string());
        } else if !KNOWN_MODELS.contains(&self.voyage_embedding_model.as_str()) {
            tracing::warn!(
                model = %self.voyage_embedding_model,
                known = ?KNOWN_MODELS,
                "Unknown Voyage AI embedding model — proceeding anyway"
            );
        }
    }

    fn validate_ocr(&self, errors: &mut Vec<String>) {
        // Unconditional: model and file-size are logged/stored even when disabled
        if self.ocr.model.len() > 64 || self.ocr.model.chars().any(char::is_control) {
            errors
                .push("OCR_MODEL contains control characters or exceeds 64 characters".to_string());
        }
        if self.ocr.max_file_size_bytes > MISTRAL_OCR_MAX_FILE_SIZE {
            errors.push(format!(
                "OCR_MAX_FILE_SIZE_BYTES must be <= {} (50 MB Mistral limit), got {}",
                MISTRAL_OCR_MAX_FILE_SIZE, self.ocr.max_file_size_bytes
            ));
        }
        if !self.ocr.enabled {
            return;
        }
        if self.mistral_api_key.is_none() {
            errors.push(
                "OCR_ENABLED is true but MISTRAL_API_KEY is not set. \
                 OCR requires a Mistral API key."
                    .to_string(),
            );
        }
        validate_timeout(errors, "OCR_TIMEOUT_SECS", self.ocr.timeout_secs);
        if self.ocr.model.trim().is_empty() {
            errors.push("OCR_MODEL must not be empty when OCR is enabled".to_string());
        } else {
            const KNOWN_OCR_MODELS: &[&str] = &["mistral-ocr-latest", "mistral-ocr-2505"];
            if !KNOWN_OCR_MODELS.contains(&self.ocr.model.as_str()) {
                tracing::warn!(
                    model = %self.ocr.model,
                    known = ?KNOWN_OCR_MODELS,
                    "Unknown OCR model — proceeding anyway"
                );
            }
        }
        if self.ocr.min_text_threshold == 0 {
            errors.push(
                "OCR_MIN_TEXT_THRESHOLD must be >= 1 \
                 (0 would send all pages to OCR unconditionally)"
                    .to_string(),
            );
        }
        if !(1..=1_000).contains(&self.ocr.max_pages_per_request) {
            errors.push(format!(
                "OCR_MAX_PAGES_PER_REQUEST must be between 1 and 1,000, got {}",
                self.ocr.max_pages_per_request
            ));
        }
        if self.ocr.max_file_size_bytes == 0 {
            errors.push("OCR_MAX_FILE_SIZE_BYTES must be >= 1 when OCR is enabled".to_string());
        }
    }

    /// Log a summary of the validated core configuration (non-secret fields only).
    pub fn log_summary(&self) {
        tracing::info!(
            server_port = self.server_port,
            frontend_url = %self.frontend_url,
            db_pool_min = self.database_pool.min_connections,
            db_pool_max = self.database_pool.max_connections,
            llm_timeout_secs = self.async_config.llm_timeout_secs,
            embedding_timeout_secs = self.async_config.embedding_timeout_secs,
            request_timeout_secs = self.async_config.request_timeout_secs,
            shutdown_timeout_secs = self.async_config.shutdown_timeout_secs,
            retrieval_pool_size = self.retrieval_pool_size,
            hyde_enabled = self.hyde_enabled,
            rerank_min_chunks = self.rerank_min_chunks,
            context_stuffing_max_chunks = self.context_stuffing_max_chunks,
            hybrid_search = self.hybrid_search.enabled,
            rate_limit_rpm = self.security.rate_limit_rpm,
            body_limit_bytes = self.security.body_limit_bytes,
            voyage_rate_limit_rpm = self.voyage_rate_limit.max_rpm,
            voyage_rate_limit_tpm = self.voyage_rate_limit.max_tpm,
            voyage_batch_size = self.voyage_rate_limit.batch_size,
            voyage_concurrent_batches = self.voyage_rate_limit.concurrent_batches,
            voyage_embedding_model = %self.voyage_embedding_model,
            ocr_enabled = self.ocr.enabled,
            ocr_model = %self.ocr.model,
            has_anthropic = self.anthropic_api_key.is_some(),
            has_openai = self.openai_api_key.is_some(),
            has_mistral = self.mistral_api_key.is_some(),
            has_voyage = self.voyage_api_key.is_some(),
            has_firecrawl = self.firecrawl_api_key.is_some(),
            has_redis = self.upstash_redis_url.is_some(),
            "Core configuration validated"
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a valid `CoreConfig` for testing. All required fields are populated
    /// with valid placeholder values so that `validate()` passes.
    pub(crate) fn valid_core_config() -> CoreConfig {
        CoreConfig {
            database_url: "postgres://localhost/test".into(),
            database_pool: DatabasePoolConfig::default(),
            server_port: 3001,
            frontend_url: "http://localhost:3000".into(),
            security: SecurityConfig::default(),
            async_config: AsyncConfig::default(),
            hybrid_search: HybridSearchConfig::default(),
            voyage_rate_limit: VoyageRateLimitConfig::default(),
            ocr: OcrConfig::default(),
            voyage_embedding_model: "voyage-4".into(),
            retrieval_pool_size: 50,
            hyde_enabled: false,
            rerank_min_chunks: 200,
            context_stuffing_max_chunks: 150,
            firecrawl_api_key: None,
            voyage_api_key: None,
            anthropic_api_key: None,
            openai_api_key: None,
            mistral_api_key: None,
            upstash_redis_url: None,
            upstash_redis_token: None,
            health_token: None,
            youtube_proxy_url: None,
            youtube_innertube_api_key: None,
            default_locale: None,
        }
    }

    // ========================================================================
    // Sub-config defaults
    // ========================================================================

    #[test]
    fn database_pool_default_acquire_timeout_is_15_seconds() {
        let pool = DatabasePoolConfig::default();
        assert_eq!(pool.acquire_timeout_secs, 15);
    }

    #[test]
    fn database_pool_from_env_reads_acquire_timeout() {
        // SAFETY: test runs in a single thread; no other code reads this env var concurrently
        unsafe { env::set_var("DB_ACQUIRE_TIMEOUT", "10") };
        let pool = DatabasePoolConfig::from_env();
        assert_eq!(pool.acquire_timeout_secs, 10);
        unsafe { env::remove_var("DB_ACQUIRE_TIMEOUT") };
    }

    #[test]
    fn database_pool_from_env_uses_default_when_unset() {
        // SAFETY: test runs in a single thread; no other code reads this env var concurrently
        unsafe { env::remove_var("DB_ACQUIRE_TIMEOUT") };
        let pool = DatabasePoolConfig::from_env();
        assert_eq!(pool.acquire_timeout_secs, 15);
    }

    #[test]
    fn security_config_default_body_limit_is_1mb() {
        let security = SecurityConfig::default();
        assert_eq!(security.body_limit_bytes, 1024 * 1024);
    }

    #[test]
    fn security_config_default_upload_limit_is_200mb() {
        let security = SecurityConfig::default();
        assert_eq!(security.upload_limit_bytes, 200 * 1024 * 1024);
    }

    // ========================================================================
    // US-004: the core config is valid without any commercial variable
    // ========================================================================

    #[test]
    fn valid_core_config_passes_validation() {
        assert!(valid_core_config().validate().is_ok());
    }

    #[test]
    fn core_config_is_valid_without_clerk_or_stripe_variables() {
        // US-004: "Given Clerk and Stripe variables are absent, when CoreConfig
        // is valid, then core state construction succeeds." CoreConfig has no
        // field able to hold those values, so the only thing to assert is that
        // a config carrying nothing but core settings validates.
        let cfg = valid_core_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn core_validation_reports_every_missing_variable_at_once() {
        let mut cfg = valid_core_config();
        cfg.database_url = String::new();
        cfg.frontend_url = "not-a-url".into();
        cfg.retrieval_pool_size = 19;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("DATABASE_URL")));
        assert!(errors.iter().any(|e| e.contains("FRONTEND_URL")));
        assert!(errors.iter().any(|e| e.contains("RETRIEVAL_POOL_SIZE")));
    }

    // -- Database URL --

    #[test]
    fn empty_database_url_is_error() {
        let mut cfg = valid_core_config();
        cfg.database_url = String::new();
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("DATABASE_URL")));
    }

    #[test]
    fn invalid_database_url_scheme_is_error() {
        let mut cfg = valid_core_config();
        cfg.database_url = "mysql://localhost/test".into();
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("DATABASE_URL") && e.contains("postgres://"))
        );
    }

    #[test]
    fn valid_config_with_postgresql_url() {
        let mut cfg = valid_core_config();
        cfg.database_url = "postgresql://localhost/test".into();
        assert!(cfg.validate().is_ok());
    }

    // -- Pool sizes --

    #[test]
    fn min_connections_greater_than_max_is_error() {
        let mut cfg = valid_core_config();
        cfg.database_pool.min_connections = 25;
        cfg.database_pool.max_connections = 10;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("DB_POOL_MIN") && e.contains("DB_POOL_MAX"))
        );
    }

    #[test]
    fn max_connections_over_100_is_error() {
        let mut cfg = valid_core_config();
        cfg.database_pool.max_connections = 101;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("DB_POOL_MAX") && e.contains("100"))
        );
    }

    #[test]
    fn max_connections_at_100_is_ok() {
        let mut cfg = valid_core_config();
        cfg.database_pool.max_connections = 100;
        assert!(cfg.validate().is_ok());
    }

    // -- Timeout bounds --

    #[test]
    fn zero_timeout_is_error() {
        let mut cfg = valid_core_config();
        cfg.async_config.llm_timeout_secs = 0;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("LLM_TIMEOUT_SECS")));
    }

    #[test]
    fn timeout_over_600_is_error() {
        let mut cfg = valid_core_config();
        cfg.async_config.request_timeout_secs = 601;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("REQUEST_TIMEOUT_SECS")));
    }

    #[test]
    fn timeout_at_boundary_values_ok() {
        let mut cfg = valid_core_config();
        cfg.async_config.llm_timeout_secs = 1;
        cfg.async_config.embedding_timeout_secs = 600;
        cfg.database_pool.connect_timeout_secs = 1;
        cfg.database_pool.acquire_timeout_secs = 600;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn all_timeout_fields_validated() {
        // Note: OCR_TIMEOUT_SECS is excluded — it is only validated when ocr.enabled is true.
        // See ocr_enabled_with_invalid_timeout_is_error for OCR timeout validation.
        let mut cfg = valid_core_config();
        cfg.database_pool.connect_timeout_secs = 0;
        cfg.database_pool.acquire_timeout_secs = 601;
        cfg.async_config.llm_timeout_secs = 0;
        cfg.async_config.embedding_timeout_secs = 0;
        cfg.async_config.request_timeout_secs = 999;
        cfg.async_config.shutdown_timeout_secs = 0;
        let errors = cfg.validate().unwrap_err().errors;
        assert_eq!(
            errors.len(),
            6,
            "Expected 6 timeout errors, got: {errors:?}"
        );
    }

    // -- Frontend URL --

    #[test]
    fn valid_config_with_https_frontend() {
        let mut cfg = valid_core_config();
        cfg.frontend_url = "https://app.example.com".into();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn invalid_frontend_url_scheme_is_error() {
        let mut cfg = valid_core_config();
        cfg.frontend_url = "ftp://example.com".into();
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("FRONTEND_URL") && e.contains("http://"))
        );
    }

    // -- Retrieval pool size --

    #[test]
    fn default_retrieval_pool_size_is_valid() {
        let cfg = valid_core_config();
        assert_eq!(cfg.retrieval_pool_size, 50);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn retrieval_pool_size_below_rerank_top_k_is_error() {
        let mut cfg = valid_core_config();
        cfg.retrieval_pool_size = 19;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("RETRIEVAL_POOL_SIZE") && e.contains(">= 20"))
        );
    }

    #[test]
    fn retrieval_pool_size_above_500_is_error() {
        let mut cfg = valid_core_config();
        cfg.retrieval_pool_size = 501;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("RETRIEVAL_POOL_SIZE") && e.contains("<= 500"))
        );
    }

    #[test]
    fn retrieval_pool_size_at_boundaries_ok() {
        let mut cfg = valid_core_config();
        cfg.retrieval_pool_size = 20;
        assert!(cfg.validate().is_ok());

        cfg.retrieval_pool_size = 500;
        assert!(cfg.validate().is_ok());
    }

    // -- Rerank min chunks --

    #[test]
    fn default_rerank_min_chunks_is_valid() {
        let cfg = valid_core_config();
        assert_eq!(cfg.rerank_min_chunks, 200);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rerank_min_chunks_below_zero_is_error() {
        let mut cfg = valid_core_config();
        cfg.rerank_min_chunks = -1;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("RERANK_MIN_CHUNKS") && e.contains(">= 0"))
        );
    }

    #[test]
    fn rerank_min_chunks_above_10000_is_error() {
        let mut cfg = valid_core_config();
        cfg.rerank_min_chunks = 10001;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("RERANK_MIN_CHUNKS") && e.contains("<= 10000"))
        );
    }

    #[test]
    fn rerank_min_chunks_at_boundaries_ok() {
        let mut cfg = valid_core_config();
        cfg.rerank_min_chunks = 0;
        assert!(cfg.validate().is_ok());

        cfg.rerank_min_chunks = 10000;
        assert!(cfg.validate().is_ok());
    }

    // -- Context stuffing max chunks --

    #[test]
    fn default_context_stuffing_max_chunks_is_valid() {
        let cfg = valid_core_config();
        assert_eq!(cfg.context_stuffing_max_chunks, 150);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn context_stuffing_max_chunks_below_zero_is_error() {
        let mut cfg = valid_core_config();
        cfg.context_stuffing_max_chunks = -1;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("CONTEXT_STUFFING_MAX_CHUNKS") && e.contains(">= 0"))
        );
    }

    #[test]
    fn context_stuffing_max_chunks_above_500_is_error() {
        let mut cfg = valid_core_config();
        cfg.context_stuffing_max_chunks = 501;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("CONTEXT_STUFFING_MAX_CHUNKS") && e.contains("<= 500"))
        );
    }

    #[test]
    fn context_stuffing_max_chunks_at_boundaries_ok() {
        let mut cfg = valid_core_config();
        cfg.context_stuffing_max_chunks = 0;
        assert!(cfg.validate().is_ok());

        cfg.context_stuffing_max_chunks = 500;
        assert!(cfg.validate().is_ok());
    }

    // -- OCR config --

    #[test]
    fn ocr_disabled_by_default() {
        let cfg = valid_core_config();
        assert!(!cfg.ocr.enabled);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn ocr_enabled_without_mistral_key_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = None;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("OCR_ENABLED") && e.contains("MISTRAL_API_KEY"))
        );
    }

    #[test]
    fn ocr_enabled_with_mistral_key_ok() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn ocr_disabled_without_mistral_key_ok() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = false;
        cfg.mistral_api_key = None;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn ocr_default_values_are_correct() {
        let ocr = OcrConfig::default();
        assert!(!ocr.enabled);
        assert_eq!(ocr.model, "mistral-ocr-latest");
        assert_eq!(ocr.timeout_secs, 120);
        assert_eq!(ocr.min_text_threshold, 50);
        assert_eq!(ocr.max_pages_per_request, 1000);
        assert_eq!(ocr.max_file_size_bytes, MISTRAL_OCR_MAX_FILE_SIZE);
    }

    #[test]
    fn ocr_enabled_with_empty_model_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.model = "   ".into();
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("OCR_MODEL")));
    }

    #[test]
    fn ocr_enabled_with_invalid_timeout_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.timeout_secs = 0;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("OCR_TIMEOUT_SECS")));
    }

    #[test]
    fn ocr_enabled_with_timeout_over_600_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.timeout_secs = 601;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("OCR_TIMEOUT_SECS")));
    }

    #[test]
    fn ocr_enabled_with_invalid_max_pages_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.max_pages_per_request = 1001;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("OCR_MAX_PAGES_PER_REQUEST"))
        );
    }

    #[test]
    fn ocr_enabled_with_zero_max_pages_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.max_pages_per_request = 0;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("OCR_MAX_PAGES_PER_REQUEST"))
        );
    }

    #[test]
    fn ocr_enabled_with_zero_file_size_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.max_file_size_bytes = 0;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("OCR_MAX_FILE_SIZE_BYTES")));
    }

    #[test]
    fn ocr_enabled_with_file_size_over_50mb_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.max_file_size_bytes = 50 * 1024 * 1024 + 1;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("OCR_MAX_FILE_SIZE_BYTES") && e.contains("50 MB"))
        );
    }

    #[test]
    fn ocr_enabled_with_zero_min_text_threshold_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.min_text_threshold = 0;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(errors.iter().any(|e| e.contains("OCR_MIN_TEXT_THRESHOLD")));
    }

    #[test]
    fn ocr_enabled_with_control_chars_in_model_is_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = true;
        cfg.mistral_api_key = Some("test-key".into());
        cfg.ocr.model = "mistral\nocr".into();
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("OCR_MODEL") && e.contains("control characters"))
        );
    }

    #[test]
    fn ocr_disabled_with_control_chars_in_model_is_still_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = false;
        cfg.ocr.model = "mistral\x00ocr".into();
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("OCR_MODEL") && e.contains("control characters"))
        );
    }

    #[test]
    fn ocr_disabled_with_file_size_over_50mb_is_still_error() {
        let mut cfg = valid_core_config();
        cfg.ocr.enabled = false;
        cfg.ocr.max_file_size_bytes = MISTRAL_OCR_MAX_FILE_SIZE + 1;
        let errors = cfg.validate().unwrap_err().errors;
        assert!(
            errors
                .iter()
                .any(|e| e.contains("OCR_MAX_FILE_SIZE_BYTES") && e.contains("50 MB"))
        );
    }

    // -- ConfigError display --

    #[test]
    fn config_error_display_lists_all_errors() {
        let err = ConfigError {
            errors: vec!["error one".into(), "error two".into()],
        };
        let display = err.to_string();
        assert!(display.contains("2 error(s)"));
        assert!(display.contains("1. error one"));
        assert!(display.contains("2. error two"));
    }
}
