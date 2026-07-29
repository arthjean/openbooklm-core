//! Voyage AI embeddings client with batch handling and rate limiting
//!
//! Provides resilient access to Voyage AI's embeddings API with:
//! - Automatic batching (128 texts per batch, Voyage AI limit)
//! - Connection pooling via shared reqwest::Client
//! - Exponential backoff retry for transient failures
//! - Circuit breaker for fault tolerance
//! - Request metrics tracking
//! - Configurable timeouts
//!
//! ## Models
//!
//! - `voyage-4`: General-purpose embeddings for RAG (1024-dim, 32K context)
//! - `voyage-4-lite`: Low-latency, low-cost embeddings (1024-dim)
//! - `voyage-4-large`: Highest quality embeddings (1024-dim)
//!
//! ## Reranking
//!
//! Use `VoyageReranker` for cross-encoder reranking after initial retrieval.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

use crate::core::config::CoreConfig;
use crate::error::{AppError, RagError};

use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use super::metrics::{ClientMetrics, ProviderMetrics};
use super::resilience::{
    HttpResult, RequestErrorKind, ResilientExecutor, build_http_client, check_response_status,
    handle_request_error, with_request_id,
};
use super::retry::RetryConfig;
use super::voyage_rate_limiter::{VoyageRateLimiter, estimate_tokens};

/// Voyage AI API constants
const VOYAGE_EMBEDDINGS_URL: &str = "https://api.voyageai.com/v1/embeddings";
#[allow(dead_code)] // Reserved for future contextualized embeddings feature
const VOYAGE_CONTEXTUALIZED_URL: &str = "https://api.voyageai.com/v1/contextualizedembeddings";
const VOYAGE_RERANK_URL: &str = "https://api.voyageai.com/v1/rerank";

/// Voyage AI models
/// All voyage-4 variants (voyage-4, voyage-4-lite, voyage-4-large) default to 1024-dim.
/// If a model change requires a different dimension, a DB migration is needed to resize
/// the pgvector column — that's a separate concern from model configurability.
const DEFAULT_EMBEDDING_MODEL: &str = "voyage-4";
const VOYAGE_RERANK_MODEL: &str = "rerank-2.5";

const BATCH_SIZE: usize = 128; // Voyage AI maximum batch size
const PROVIDER_NAME: &str = "voyage";

/// Result of a single embedding batch: (batch_index, embeddings).
type BatchResult = Result<(usize, Vec<Vec<f32>>), AppError>;

/// Voyage AI standard embedding request
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
    input_type: String,
}

/// Voyage AI contextualized embedding request
/// Each inner Vec contains chunks from the same document, ordered by position
#[allow(dead_code)] // Reserved for future contextualized embeddings feature
#[derive(Debug, Serialize)]
struct ContextualizedEmbeddingRequest {
    documents: Vec<Vec<String>>,
    model: String,
    input_type: String,
}

/// Voyage AI rerank request
#[derive(Debug, Serialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
}

/// Voyage AI embedding response
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    usage: Option<UsageData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

/// Voyage AI rerank response
#[derive(Debug, Deserialize)]
struct RerankResponse {
    data: Vec<RerankData>,
    usage: Option<UsageData>,
}

#[derive(Debug, Deserialize)]
struct RerankData {
    index: usize,
    relevance_score: f32,
}

#[derive(Debug, Deserialize)]
struct UsageData {
    total_tokens: Option<i32>,
}

// `RerankedDocument` now lives in `core::providers`: it is part of the
// public `Reranker` contract, not a Voyage detail (US-020).
pub use crate::core::providers::RerankedDocument;

/// Voyage AI embeddings client with resilience patterns
#[derive(Clone)]
pub struct VoyageClient {
    http_client: reqwest::Client,
    api_key: Arc<str>,
    model: String,
    timeout: Duration,
    resilience: ResilientExecutor,
    rate_limiter: Option<Arc<VoyageRateLimiter>>,
    batch_size: usize,
    concurrent_batches: usize,
}

impl std::fmt::Debug for VoyageClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoyageClient")
            .field("timeout", &self.timeout)
            .field("resilience", &self.resilience)
            .finish_non_exhaustive()
    }
}

impl VoyageClient {
    /// Create a new Voyage client from application config
    pub fn from_config(config: &CoreConfig, metrics: &ClientMetrics) -> Result<Self, RagError> {
        let api_key = config.voyage_api_key.as_deref().ok_or_else(|| {
            tracing::error!("Voyage AI API key not configured");
            RagError::EmbeddingFailed {
                reason: "Voyage AI API key not configured".to_string(),
            }
        })?;

        let timeout = Duration::from_secs(config.async_config.embedding_timeout_secs);

        Self::new(api_key, timeout, metrics.provider("voyage"))
    }

    /// Create a new Voyage client with custom settings
    pub fn new(
        api_key: impl Into<Arc<str>>,
        timeout: Duration,
        metrics: Arc<ProviderMetrics>,
    ) -> Result<Self, RagError> {
        let http_client = build_http_client(Some(timeout), 10)
            .map_err(|reason| RagError::EmbeddingFailed { reason })?;

        let retry_config = RetryConfig::new(3)
            .with_initial_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(30));

        let circuit_breaker = Arc::new(CircuitBreaker::new(
            "voyage",
            CircuitBreakerConfig::new(5)
                .with_open_duration(Duration::from_secs(30))
                .with_success_threshold(2),
        ));

        Ok(Self {
            http_client,
            api_key: api_key.into(),
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            timeout,
            resilience: ResilientExecutor::new("voyage", retry_config, circuit_breaker, metrics)
                .with_timeout_secs(timeout.as_secs()),
            rate_limiter: None,
            batch_size: BATCH_SIZE,
            concurrent_batches: 4,
        })
    }

    /// Attach a rate limiter to this client (builder pattern).
    #[must_use]
    pub fn with_rate_limiter(mut self, rl: Arc<VoyageRateLimiter>) -> Self {
        self.rate_limiter = Some(rl);
        self
    }

    /// Set the batch size for embedding requests (builder pattern).
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the number of concurrent embedding batches (builder pattern).
    #[must_use]
    pub fn with_concurrent_batches(mut self, n: usize) -> Self {
        self.concurrent_batches = n;
        self
    }

    /// Set the embedding model (builder pattern).
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Get the effective batch size for external callers.
    pub fn effective_batch_size(&self) -> usize {
        self.batch_size
    }

    /// Access the rate limiter (if attached) for per-batch rate limiting in the pipeline.
    pub(crate) fn rate_limiter(&self) -> Option<&Arc<VoyageRateLimiter>> {
        self.rate_limiter.as_ref()
    }

    /// Get the configured concurrency level.
    pub(crate) fn concurrency(&self) -> usize {
        self.concurrent_batches
    }

    /// Generate embeddings for documents (for storage)
    ///
    /// Automatically batches requests according to the configured batch size,
    /// sends up to `concurrent_batches` in parallel via `buffer_unordered`,
    /// and applies rate limiting if a rate limiter is attached.
    pub async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let text_count = texts.len();
        let batch_size = self.batch_size;
        let concurrency = self.concurrent_batches;
        let total_batches = text_count.div_ceil(batch_size);

        tracing::info!(
            total_batches,
            concurrency,
            total_texts = text_count,
            "Starting parallel embedding"
        );

        // Build (batch_index, batch_slice) pairs and process concurrently.
        // Each future returns (batch_index, embeddings) so we can reorder.
        let batches: Vec<(usize, Vec<String>)> = texts
            .chunks(batch_size)
            .enumerate()
            .map(|(i, chunk)| (i, chunk.to_vec()))
            .collect();

        let results: Vec<BatchResult> = stream::iter(batches)
            .map(|(batch_index, batch)| async move {
                // Rate limiting: acquire budget before sending the request
                if let Some(ref rl) = self.rate_limiter {
                    let estimated = estimate_tokens(&batch);
                    let waited = rl.acquire(estimated).await?;
                    if waited > Duration::from_millis(100) {
                        tracing::info!(
                            batch = batch_index,
                            waited_ms = waited.as_millis(),
                            estimated_tokens = estimated,
                            "Rate limiter wait before batch"
                        );
                    }
                }

                let (batch_embeddings, tokens_used) =
                    self.embed_batch_with_usage(&batch, "document").await?;

                // Record actual token usage for accurate TPM tracking
                if let Some(ref rl) = self.rate_limiter
                    && let Some(tokens) = tokens_used
                {
                    rl.record_usage(u32::try_from(tokens).unwrap_or(0)).await;
                }

                tracing::debug!(
                    batch = batch_index,
                    batch_size = batch.len(),
                    ?tokens_used,
                    "Processed embedding batch"
                );

                Ok((batch_index, batch_embeddings))
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        // Reorder by batch index and check for errors (first error wins)
        let mut indexed: Vec<(usize, Vec<Vec<f32>>)> = Vec::with_capacity(total_batches);
        for result in results {
            indexed.push(result?);
        }
        indexed.sort_by_key(|(idx, _)| *idx);

        let all_embeddings: Vec<Vec<f32>> =
            indexed.into_iter().flat_map(|(_, embs)| embs).collect();

        tracing::debug!(
            total_embeddings = all_embeddings.len(),
            "Document embeddings generated"
        );
        Ok(all_embeddings)
    }

    /// Generate embeddings with a progress callback `(chunks_done, chunks_total)`.
    ///
    /// Batches are processed concurrently via `buffer_unordered`. Progress is
    /// tracked with an `AtomicU32` so multiple completing batches can safely
    /// increment the counter; the callback is invoked after each batch lands.
    pub async fn embed_documents_with_progress<F>(
        &self,
        texts: &[String],
        mut on_progress: F,
    ) -> Result<Vec<Vec<f32>>, AppError>
    where
        F: FnMut(u32, u32),
    {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let text_count = texts.len();
        let batch_size = self.batch_size;
        let concurrency = self.concurrent_batches;
        let total = u32::try_from(text_count).unwrap_or(u32::MAX);

        // Atomic counter so concurrent batches can safely report progress
        let progress = Arc::new(AtomicU32::new(0));

        let batches: Vec<(usize, Vec<String>)> = texts
            .chunks(batch_size)
            .enumerate()
            .map(|(i, chunk)| (i, chunk.to_vec()))
            .collect();

        let results: Vec<BatchResult> = stream::iter(batches)
            .map(|(batch_index, batch)| {
                let progress = Arc::clone(&progress);
                async move {
                    // Rate limiting
                    if let Some(ref rl) = self.rate_limiter {
                        let estimated = estimate_tokens(&batch);
                        let waited = rl.acquire(estimated).await?;
                        if waited > Duration::from_millis(100) {
                            tracing::info!(
                                batch = batch_index,
                                waited_ms = waited.as_millis(),
                                "Rate limiter wait before batch"
                            );
                        }
                    }

                    let (batch_embeddings, tokens_used) =
                        self.embed_batch_with_usage(&batch, "document").await?;

                    if let Some(ref rl) = self.rate_limiter
                        && let Some(tokens) = tokens_used
                    {
                        rl.record_usage(u32::try_from(tokens).unwrap_or(0)).await;
                    }

                    // Atomically bump progress by the number of texts in this batch
                    progress.fetch_add(
                        u32::try_from(batch.len()).unwrap_or(u32::MAX),
                        Ordering::Relaxed,
                    );

                    Ok((batch_index, batch_embeddings))
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        // Reorder by batch index and check for errors
        let mut indexed: Vec<(usize, Vec<Vec<f32>>)> = Vec::with_capacity(results.len());
        for result in results {
            let (idx, embs) = result?;
            indexed.push((idx, embs));
            // Report progress after each result is consumed
            on_progress(progress.load(Ordering::Relaxed), total);
        }
        indexed.sort_by_key(|(idx, _)| *idx);

        Ok(indexed.into_iter().flat_map(|(_, embs)| embs).collect())
    }

    /// Generate embedding for a query (for search)
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>, AppError> {
        tracing::debug!(query_length = text.len(), "Generating query embedding");

        // Rate limit single query embeddings too
        if let Some(ref rl) = self.rate_limiter {
            let estimated = estimate_tokens(&[text.to_string()]);
            rl.acquire(estimated).await?;
        }

        let (mut embeddings, tokens_used) = self
            .embed_batch_with_usage(&[text.to_string()], "query")
            .await?;

        if let Some(ref rl) = self.rate_limiter
            && let Some(tokens) = tokens_used
        {
            rl.record_usage(u32::try_from(tokens).unwrap_or(0)).await;
        }

        embeddings.pop().ok_or_else(|| {
            tracing::error!("Voyage AI returned empty embedding response");
            AppError::from(RagError::EmbeddingFailed {
                reason: "No embedding returned from Voyage AI".to_string(),
            })
        })
    }

    /// Embed a batch of texts with retry and circuit breaker.
    /// Returns (embeddings, tokens_used).
    ///
    /// `pub(crate)` so the pipeline in `source_processing.rs` can call individual
    /// batches and stream results through an `mpsc` channel.
    pub(crate) async fn embed_batch_with_usage(
        &self,
        texts: &[String],
        input_type: &str,
    ) -> Result<(Vec<Vec<f32>>, Option<i32>), AppError> {
        tracing::debug!(model = %self.model, batch_size = texts.len(), "Sending embedding request");

        let request = EmbeddingRequest {
            input: texts.to_vec(),
            model: self.model.clone(),
            input_type: input_type.to_string(),
        };

        self.resilience
            .execute(
                |retry_secs| {
                    AppError::from(RagError::EmbeddingFailed {
                        reason: format!(
                            "Voyage AI service unavailable. Retry after {retry_secs} seconds."
                        ),
                    })
                },
                || self.execute_embedding_request(&request, input_type),
            )
            .await
    }

    /// Execute a single embedding request, returning embeddings + token usage.
    async fn execute_embedding_request(
        &self,
        request: &EmbeddingRequest,
        input_type: &str,
    ) -> HttpResult<(Vec<Vec<f32>>, Option<i32>)> {
        let response = self
            .send_request(VOYAGE_EMBEDDINGS_URL, request)
            .await
            .map_err(|e| {
                handle_request_error(&e, |kind, description| {
                    self.map_voyage_error(kind, description, "embedding", Some(input_type))
                })
            })?;

        let response =
            check_response_status(response, PROVIDER_NAME, |code, body, _retry_after| {
                AppError::from(RagError::EmbeddingFailed {
                    reason: format!("Voyage AI embedding returned {code}: {body}"),
                })
            })
            .await?;

        let embedding_response: EmbeddingResponse =
            self.parse_json_response(response, "embedding").await?;

        self.log_usage(&embedding_response.usage, input_type);

        let tokens_used = embedding_response
            .usage
            .as_ref()
            .and_then(|u| u.total_tokens);

        // Sort by index to ensure correct order
        let mut data = embedding_response.data;
        data.sort_by_key(|d| d.index);

        Ok((data.into_iter().map(|d| d.embedding).collect(), tokens_used))
    }

    /// Rerank documents by relevance to a query
    ///
    /// Uses the rerank-2.5 cross-encoder model for more accurate relevance scoring
    /// than embedding similarity. Should be used after initial vector retrieval.
    pub async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
    ) -> Result<Vec<RerankedDocument>, AppError> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        tracing::debug!(
            query_len = query.len(),
            document_count = documents.len(),
            ?top_k,
            "Reranking documents"
        );

        // Rate limit rerank requests
        if let Some(ref rl) = self.rate_limiter {
            let mut all_texts: Vec<String> = documents.to_vec();
            all_texts.push(query.to_string());
            let estimated = estimate_tokens(&all_texts);
            rl.acquire(estimated).await?;
        }

        let request = RerankRequest {
            query: query.to_string(),
            documents: documents.to_vec(),
            model: VOYAGE_RERANK_MODEL.to_string(),
            top_k,
        };

        let reranked = self
            .resilience
            .execute(
                |retry_secs| {
                    AppError::from(RagError::EmbeddingFailed {
                        reason: format!(
                            "Voyage AI service unavailable. Retry after {retry_secs} seconds."
                        ),
                    })
                },
                || self.execute_rerank_request(&request),
            )
            .await?;

        // Record actual token usage for rerank
        if let Some(ref rl) = self.rate_limiter {
            // Rerank usage is not always reported, estimate from inputs
            let mut all_texts: Vec<String> = request.documents.clone();
            all_texts.push(request.query.clone());
            let estimated = estimate_tokens(&all_texts);
            rl.record_usage(estimated).await;
        }

        tracing::debug!(
            result_count = reranked.len(),
            top_score = reranked.first().map(|r| r.relevance_score),
            "Reranking completed"
        );

        Ok(reranked)
    }

    /// Execute a single rerank request
    async fn execute_rerank_request(
        &self,
        request: &RerankRequest,
    ) -> HttpResult<Vec<RerankedDocument>> {
        let response = self
            .send_request(VOYAGE_RERANK_URL, request)
            .await
            .map_err(|e| {
                handle_request_error(&e, |kind, description| {
                    self.map_voyage_error(kind, description, "rerank", None)
                })
            })?;

        let response =
            check_response_status(response, PROVIDER_NAME, |code, body, _retry_after| {
                AppError::from(RagError::EmbeddingFailed {
                    reason: format!("Voyage AI rerank returned {code}: {body}"),
                })
            })
            .await?;

        let rerank_response: RerankResponse = self.parse_json_response(response, "rerank").await?;

        self.log_usage(&rerank_response.usage, "rerank");

        // Results are already sorted by relevance score (descending)
        Ok(rerank_response
            .data
            .into_iter()
            .map(|d| RerankedDocument {
                index: d.index,
                relevance_score: d.relevance_score,
            })
            .collect())
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Private HTTP helpers
    // ─────────────────────────────────────────────────────────────────────────────

    /// Send an authenticated POST request
    async fn send_request<T: Serialize>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<reqwest::Response, reqwest::Error> {
        with_request_id(
            self.http_client
                .post(url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json"),
        )
        .json(body)
        .send()
        .await
    }

    /// Map a request error to a Voyage-specific `AppError`.
    fn map_voyage_error(
        &self,
        kind: RequestErrorKind,
        description: String,
        operation: &str,
        input_type: Option<&str>,
    ) -> AppError {
        match kind {
            RequestErrorKind::Timeout => {
                let timeout_secs = self.timeout.as_secs();
                tracing::warn!(
                    timeout_secs,
                    ?input_type,
                    "Voyage AI {operation} request timed out"
                );
                AppError::from(RagError::EmbeddingTimeout { timeout_secs })
            }
            _ => AppError::from(RagError::EmbeddingFailed {
                reason: description,
            }),
        }
    }

    /// Parse JSON response with proper error handling
    async fn parse_json_response<T: serde::de::DeserializeOwned>(
        &self,
        response: reqwest::Response,
        operation: &str,
    ) -> HttpResult<T> {
        response.json().await.map_err(|e| {
            tracing::error!(error = %e, "Failed to parse Voyage AI {operation} response");
            (
                AppError::from(RagError::EmbeddingFailed {
                    reason: format!("Failed to parse {operation} response: {e}"),
                }),
                None,
                false,
            )
        })
    }

    /// Log token usage if available
    #[allow(clippy::unused_self)]
    fn log_usage(&self, usage: &Option<UsageData>, operation: &str) {
        if let Some(UsageData {
            total_tokens: Some(tokens),
        }) = usage
        {
            tracing::debug!(tokens, "Voyage AI {operation} tokens used");
        }
    }
}

super::impl_client_resilience_methods!(VoyageClient, "voyage");

// ============================================================================
// Public seams (US-020)
// ============================================================================
//
// The inherent methods above stay: they carry the batching, rate limiting and
// progress reporting that only this client has. The trait impls are the narrow
// surface the core depends on, so retrieval never names Voyage.

impl VoyageClient {
    /// Embed one batch, acquiring and reconciling the rate-limit budget.
    ///
    /// The ingestion pipeline used to do this itself, which put Voyage's token
    /// accounting inside a core service. It belongs here: the estimate, the
    /// wait and the reconciliation against reported usage are all facts about
    /// this vendor's API (US-020).
    pub async fn embed_batch_rate_limited(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, AppError> {
        if let Some(rl) = self.rate_limiter() {
            let estimated = crate::clients::voyage_rate_limiter::estimate_tokens(texts);
            let waited = rl.acquire(estimated).await?;
            if waited > std::time::Duration::from_millis(100) {
                tracing::info!(
                    waited_ms = waited.as_millis(),
                    batch_len = texts.len(),
                    "Rate limiter wait before embedding batch"
                );
            }
        }

        let (embeddings, tokens_used) = self.embed_batch_with_usage(texts, "document").await?;

        // Reconcile the estimate against what the API actually charged.
        if let Some(rl) = self.rate_limiter()
            && let Some(tokens) = tokens_used
        {
            rl.record_usage(u32::try_from(tokens).unwrap_or(0)).await;
        }

        Ok(embeddings)
    }
}

#[async_trait::async_trait]
impl crate::core::providers::EmbeddingProvider for VoyageClient {
    fn name(&self) -> &str {
        "voyage"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        // Every voyage-4 variant returns 1024. A future model of another width
        // would need this to become a per-model lookup; the dimension guard at
        // startup is what would catch the omission.
        crate::core::providers::EMBEDDING_DIM
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        VoyageClient::embed_documents(self, texts).await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, AppError> {
        VoyageClient::embed_query(self, text).await
    }

    fn batch_size(&self) -> usize {
        self.effective_batch_size()
    }

    fn concurrency(&self) -> usize {
        VoyageClient::concurrency(self)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        self.embed_batch_rate_limited(texts).await
    }
}

#[async_trait::async_trait]
impl crate::core::providers::Reranker for VoyageClient {
    fn name(&self) -> &str {
        "voyage"
    }

    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
    ) -> Result<Vec<RerankedDocument>, AppError> {
        VoyageClient::rerank(self, query, documents, top_k).await
    }
}
