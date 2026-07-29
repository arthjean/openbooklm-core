//! Firecrawl web scraping client with retry and circuit breaker
//!
//! Provides resilient access to Firecrawl's web scraping API with:
//! - Connection pooling via shared reqwest::Client
//! - Exponential backoff retry for transient failures
//! - Circuit breaker for fault tolerance
//! - Request metrics tracking
//! - Configurable timeouts

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::core::config::CoreConfig;
use crate::error::{AppError, SourceError};

use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use super::metrics::{ClientMetrics, ProviderMetrics};
use super::resilience::{
    HttpResult, ResilientExecutor, build_http_client, check_response_status, handle_request_error,
    with_request_id,
};
use super::retry::RetryConfig;

/// Firecrawl API constants
const FIRECRAWL_API_URL: &str = "https://api.firecrawl.dev/v1/scrape";
const DEFAULT_TIMEOUT_SECS: u64 = 60; // Web scraping can be slow
const PROVIDER_NAME: &str = "firecrawl";

/// Firecrawl scrape request
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ScrapeRequest {
    url: String,
    formats: Vec<String>,
    only_main_content: bool,
    remove_base64_images: bool,
    exclude_tags: Vec<String>,
}

/// Firecrawl scrape response
#[derive(Debug, Deserialize)]
struct FirecrawlResponse {
    success: bool,
    data: Option<FirecrawlData>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FirecrawlData {
    markdown: Option<String>,
    content: Option<String>,
    metadata: Option<FirecrawlMetadata>,
}

#[derive(Debug, Deserialize)]
struct FirecrawlMetadata {
    title: Option<String>,
    description: Option<String>,
    #[serde(rename = "sourceURL")]
    source_url: Option<String>,
}

/// Result of scraping a URL
#[derive(Debug, Clone)]
pub struct ScrapeResult {
    pub title: String,
    pub content: String,
    pub description: Option<String>,
    pub source_url: String,
}

/// Firecrawl web scraping client with resilience patterns
#[derive(Clone)]
pub struct FirecrawlClient {
    http_client: reqwest::Client,
    api_key: Arc<str>,
    timeout: Duration,
    resilience: ResilientExecutor,
}

impl std::fmt::Debug for FirecrawlClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FirecrawlClient")
            .field("timeout", &self.timeout)
            .field("resilience", &self.resilience)
            .finish_non_exhaustive()
    }
}

impl FirecrawlClient {
    /// Create a new Firecrawl client from application config
    pub fn from_config(config: &CoreConfig, metrics: &ClientMetrics) -> Result<Self, SourceError> {
        let api_key = config.firecrawl_api_key.as_deref().ok_or_else(|| {
            tracing::error!("Firecrawl API key not configured");
            SourceError::WebFetchFailed {
                url: "N/A".to_string(),
                reason: "Firecrawl API key not configured".to_string(),
            }
        })?;

        Self::new(
            api_key,
            Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            metrics.provider(PROVIDER_NAME),
        )
        .map_err(|e| SourceError::WebFetchFailed {
            url: "N/A".to_string(),
            reason: format!("Failed to initialize Firecrawl client: {e}"),
        })
    }

    /// Create a new Firecrawl client with custom settings
    pub fn new(
        api_key: impl Into<Arc<str>>,
        timeout: Duration,
        metrics: Arc<ProviderMetrics>,
    ) -> Result<Self, SourceError> {
        let http_client = Self::make_http_client(timeout)?;

        let retry_config = RetryConfig::new(3)
            .with_initial_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(30));

        let circuit_breaker = Arc::new(CircuitBreaker::new(
            PROVIDER_NAME,
            CircuitBreakerConfig::new(5)
                .with_open_duration(Duration::from_secs(60))
                .with_success_threshold(2),
        ));

        Ok(Self {
            http_client,
            api_key: api_key.into(),
            timeout,
            resilience: ResilientExecutor::new(
                PROVIDER_NAME,
                retry_config,
                circuit_breaker,
                metrics,
            )
            .with_timeout_secs(timeout.as_secs()),
        })
    }

    /// Build HTTP client with given timeout
    fn make_http_client(timeout: Duration) -> Result<reqwest::Client, SourceError> {
        build_http_client(Some(timeout), 5)
            .map_err(|reason| SourceError::ProcessingFailed { reason })
    }

    /// Create a client with custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, SourceError> {
        self.timeout = timeout;
        self.http_client = Self::make_http_client(timeout)?;
        Ok(self)
    }

    /// Scrape a URL and extract content
    pub async fn scrape_url(&self, url: &str) -> Result<ScrapeResult, AppError> {
        // Validate URL
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(AppError::from(SourceError::WebFetchFailed {
                url: url.to_string(),
                reason: "Invalid URL: must start with http:// or https://".to_string(),
            }));
        }

        tracing::debug!(url, "Scraping URL with Firecrawl");

        let request = ScrapeRequest {
            url: url.to_string(),
            formats: vec!["markdown".to_string()],
            only_main_content: true,
            remove_base64_images: true,
            exclude_tags: vec![
                "nav", "footer", "aside", "form", "iframe", "noscript", "script", "style",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        };

        let result = self
            .resilience
            .execute(
                |retry_secs| {
                    AppError::from(SourceError::WebFetchFailed {
                        url: url.to_string(),
                        reason: format!(
                            "Firecrawl service unavailable. Retry after {retry_secs} seconds."
                        ),
                    })
                },
                || self.execute_request(&request, url),
            )
            .await?;

        tracing::info!(
            url,
            title = %result.title,
            content_length = result.content.len(),
            "Successfully scraped URL"
        );

        Ok(result)
    }

    /// Execute a single scrape request
    async fn execute_request(
        &self,
        request: &ScrapeRequest,
        url: &str,
    ) -> HttpResult<ScrapeResult> {
        let response = self.send_request_with_timeout(request, url).await?;

        // Check HTTP status using shared helper, then parse Firecrawl-specific response
        let url_owned = url.to_string();
        let response =
            check_response_status(response, PROVIDER_NAME, |code, body, _retry_after| {
                AppError::from(SourceError::WebFetchFailed {
                    url: url_owned,
                    reason: format!("Firecrawl returned {code}: {body}"),
                })
            })
            .await?;

        self.parse_firecrawl_response(response, url).await
    }

    /// Send HTTP request with timeout wrapper
    async fn send_request_with_timeout(
        &self,
        request: &ScrapeRequest,
        url: &str,
    ) -> HttpResult<reqwest::Response> {
        let response_result = tokio::time::timeout(
            self.timeout,
            with_request_id(
                self.http_client
                    .post(FIRECRAWL_API_URL)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json"),
            )
            .json(request)
            .send(),
        )
        .await;

        match response_result {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => {
                let url = url.to_string();
                Err(handle_request_error(&e, |_kind, description| {
                    AppError::from(SourceError::WebFetchFailed {
                        url,
                        reason: description,
                    })
                }))
            }
            Err(_) => {
                let timeout_secs = self.timeout.as_secs();
                tracing::warn!(timeout_secs, url, "Firecrawl request timed out");
                Err(self.web_fetch_error(
                    url,
                    format!("Request timed out after {timeout_secs} seconds"),
                    true,
                ))
            }
        }
    }

    /// Parse and validate Firecrawl response
    async fn parse_firecrawl_response(
        &self,
        response: reqwest::Response,
        url: &str,
    ) -> HttpResult<ScrapeResult> {
        let firecrawl_response: FirecrawlResponse = response.json().await.map_err(|e| {
            tracing::error!(error = %e, url, "Failed to parse Firecrawl response");
            self.web_fetch_error(url, format!("Failed to parse response: {e}"), false)
        })?;

        if !firecrawl_response.success {
            let error_msg = firecrawl_response
                .error
                .unwrap_or_else(|| "Unknown error".to_string());
            tracing::error!(url, error = %error_msg, "Firecrawl scrape failed");
            return Err(self.web_fetch_error(url, format!("Firecrawl failed: {error_msg}"), false));
        }

        let data = firecrawl_response.data.ok_or_else(|| {
            tracing::error!(url, "Firecrawl returned no data");
            self.web_fetch_error(url, "Firecrawl returned no data".to_string(), false)
        })?;

        let content = data.markdown.or(data.content).ok_or_else(|| {
            tracing::error!(url, "Firecrawl returned no content");
            self.web_fetch_error(url, "Firecrawl returned no content".to_string(), false)
        })?;

        if content.trim().is_empty() {
            return Err((AppError::from(SourceError::EmptyContent), None, false));
        }

        let metadata = data.metadata.as_ref();
        Ok(ScrapeResult {
            title: metadata
                .and_then(|m| m.title.clone())
                .unwrap_or_else(|| extract_title_from_url(url)),
            content,
            description: metadata.and_then(|m| m.description.clone()),
            source_url: metadata
                .and_then(|m| m.source_url.clone())
                .unwrap_or_else(|| url.to_string()),
        })
    }

    /// Helper to create web fetch error tuple
    #[allow(clippy::unused_self)]
    fn web_fetch_error(
        &self,
        url: &str,
        reason: String,
        is_retryable: bool,
    ) -> (AppError, Option<u16>, bool) {
        (
            AppError::from(SourceError::WebFetchFailed {
                url: url.to_string(),
                reason,
            }),
            None,
            is_retryable,
        )
    }
}

super::impl_client_resilience_methods!(FirecrawlClient, PROVIDER_NAME);

/// Extract a simple title from URL as fallback
fn extract_title_from_url(url: &str) -> String {
    url.split('/')
        .rfind(|s| !s.is_empty() && !s.contains(':'))
        .map(|s| s.replace(['-', '_'], " "))
        .unwrap_or_else(|| "Web Page".to_string())
}
