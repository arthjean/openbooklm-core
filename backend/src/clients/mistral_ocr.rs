//! Mistral OCR client with retry and circuit breaker
//!
//! Provides resilient access to Mistral's OCR API with:
//! - Base64 PDF encoding for direct upload
//! - Per-page selection for lazy OCR (only scanned pages)
//! - Exponential backoff retry for transient failures
//! - Circuit breaker for fault tolerance
//! - Request metrics tracking
//! - Configurable timeouts

use std::sync::Arc;
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

use crate::core::config::CoreConfig;
use crate::error::{AppError, SourceError};

use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use super::metrics::{ClientMetrics, ProviderMetrics};
use super::resilience::{
    HttpResult, ResilientExecutor, build_http_client, check_response_status, with_request_id,
};
use super::retry::RetryConfig;

/// Mistral OCR API constants
const MISTRAL_API_BASE: &str = "https://api.mistral.ai";
const MISTRAL_OCR_PATH: &str = "/v1/ocr";
const PROVIDER_NAME: &str = "mistral_ocr";

// ── Request types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OcrDocument {
    #[serde(rename = "type")]
    doc_type: &'static str,
    document_url: String,
}

#[derive(Serialize)]
struct OcrRequest {
    model: String,
    document: OcrDocument,
    #[serde(skip_serializing_if = "Option::is_none")]
    pages: Option<Vec<u32>>,
    include_image_base64: bool,
}

// ── Response types ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OcrApiResponse {
    pages: Vec<OcrPage>,
    usage_info: Option<OcrUsageInfo>,
}

#[derive(Debug, Deserialize)]
struct OcrUsageInfo {
    pages_processed: u32,
}

/// A single OCR-processed page with its extracted Markdown text.
#[derive(Debug, Clone, Deserialize)]
pub struct OcrPage {
    /// 0-indexed page number (page 1 of the PDF is index 0).
    pub index: u32,
    pub markdown: String,
}

/// Result of an OCR extraction containing all processed pages.
#[derive(Debug, Clone)]
pub struct OcrResult {
    pub pages: Vec<OcrPage>,
    pub pages_processed: u32,
}

// ── Client ───────────────────────────────────────────────────────────────────

/// Mistral OCR client with resilience patterns
#[derive(Clone)]
pub struct MistralOcrClient {
    http_client: reqwest::Client,
    api_key: Arc<str>,
    model: String,
    max_file_size_bytes: usize,
    timeout: Duration,
    resilience: ResilientExecutor,
    base_url: String,
}

impl std::fmt::Debug for MistralOcrClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralOcrClient")
            .field("model", &self.model)
            .field("timeout", &self.timeout)
            .field("resilience", &self.resilience)
            .finish_non_exhaustive()
    }
}

impl MistralOcrClient {
    /// Create a new Mistral OCR client from application config.
    pub fn from_config(config: &CoreConfig, metrics: &ClientMetrics) -> Result<Self, SourceError> {
        let api_key = config.mistral_api_key.as_deref().ok_or_else(|| {
            tracing::error!("Mistral API key not configured for OCR");
            SourceError::PdfExtractionFailed {
                reason: "Mistral API key not configured".to_string(),
            }
        })?;

        let timeout = Duration::from_secs(config.ocr.timeout_secs);
        let model = config.ocr.model.clone();
        let max_file_size_bytes = config.ocr.max_file_size_bytes;

        Self::new(
            api_key,
            model,
            max_file_size_bytes,
            timeout,
            metrics.provider(PROVIDER_NAME),
        )
        .map_err(|e| SourceError::PdfExtractionFailed {
            reason: format!("Failed to initialize Mistral OCR client: {e}"),
        })
    }

    /// Create a new Mistral OCR client with custom settings.
    pub fn new(
        api_key: impl Into<Arc<str>>,
        model: String,
        max_file_size_bytes: usize,
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
            model,
            max_file_size_bytes,
            timeout,
            resilience: ResilientExecutor::new(
                PROVIDER_NAME,
                retry_config,
                circuit_breaker,
                metrics,
            )
            .with_timeout_secs(timeout.as_secs()),
            base_url: MISTRAL_API_BASE.to_string(),
        })
    }

    /// Override the base URL (for testing with mock servers).
    ///
    /// Not part of the public API — used only by integration tests to redirect
    /// requests to a wiremock server. Must NEVER be wired to user input or
    /// environment variables without URL validation (scheme allowlist, no
    /// RFC 1918 addresses) to prevent SSRF.
    ///
    /// The `url` parameter should be a root URL without path (e.g., `http://localhost:1234`).
    #[doc(hidden)]
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build HTTP client with given timeout.
    fn make_http_client(timeout: Duration) -> Result<reqwest::Client, SourceError> {
        build_http_client(Some(timeout), 5)
            .map_err(|reason| SourceError::ProcessingFailed { reason })
    }

    /// Extract text from a PDF via Mistral OCR.
    ///
    /// Encodes the PDF bytes as base64, sends them to the Mistral OCR API,
    /// and returns extracted Markdown text per page.
    ///
    /// If `pages` is `Some`, only those 0-indexed pages are processed (lazy OCR).
    /// If `pages` is `None`, all pages in the PDF are processed.
    pub async fn extract_text_from_pdf(
        &self,
        pdf_bytes: &[u8],
        pages: Option<Vec<u32>>,
    ) -> Result<OcrResult, AppError> {
        if pdf_bytes.is_empty() {
            return Err(AppError::from(SourceError::PdfExtractionFailed {
                reason: "PDF content is empty".to_string(),
            }));
        }

        if pdf_bytes.len() > self.max_file_size_bytes {
            return Err(AppError::from(SourceError::PdfExtractionFailed {
                reason: format!(
                    "PDF size ({} bytes) exceeds OCR limit ({} bytes)",
                    pdf_bytes.len(),
                    self.max_file_size_bytes
                ),
            }));
        }

        let file_size_bytes = pdf_bytes.len();
        let pages_requested = pages
            .as_ref()
            .map(|p| u32::try_from(p.len()).unwrap_or(u32::MAX));

        tracing::debug!(
            file_size_bytes,
            pages_requested = ?pages_requested,
            model = %self.model,
            "Extracting text via Mistral OCR"
        );

        // Build the data URL in a single allocation to avoid a ~67MB intermediate
        // copy when encoding large PDFs (50MB raw → ~67MB base64).
        let prefix = "data:application/pdf;base64,";
        let encoded_len = pdf_bytes.len().div_ceil(3) * 4; // exact base64 output size
        let mut document_url = String::with_capacity(prefix.len() + encoded_len);
        document_url.push_str(prefix);
        STANDARD.encode_string(pdf_bytes, &mut document_url);

        let request = OcrRequest {
            model: self.model.clone(),
            document: OcrDocument {
                doc_type: "document_url",
                document_url,
            },
            pages,
            include_image_base64: false,
        };

        let ocr_start = std::time::Instant::now();

        let result = self
            .resilience
            .execute(
                |retry_secs| {
                    AppError::from(SourceError::PdfExtractionFailed {
                        reason: format!(
                            "Mistral OCR service unavailable. Retry after {retry_secs} seconds."
                        ),
                    })
                },
                || self.execute_ocr_request(&request),
            )
            .await?;

        let duration_ms = u64::try_from(ocr_start.elapsed().as_millis()).unwrap_or(u64::MAX);

        tracing::debug!(
            pages_processed = result.pages_processed,
            pages_requested = pages_requested.unwrap_or(0),
            file_size_bytes,
            duration_ms,
            model = %self.model,
            "Mistral OCR completed successfully"
        );

        Ok(result)
    }

    /// Execute a single OCR request with status checking.
    async fn execute_ocr_request(&self, request: &OcrRequest) -> HttpResult<OcrResult> {
        let response = self.send_request_with_timeout(request).await?;

        let response =
            check_response_status(response, PROVIDER_NAME, |code, body, _retry_after| {
                let safe_body: String =
                    body.chars().filter(|c| !c.is_control()).take(200).collect();
                AppError::from(SourceError::PdfExtractionFailed {
                    reason: format!("Mistral OCR returned {code}: {safe_body}"),
                })
            })
            .await?;

        self.parse_ocr_response(response).await
    }

    /// Send HTTP request with timeout wrapper.
    async fn send_request_with_timeout(
        &self,
        request: &OcrRequest,
    ) -> HttpResult<reqwest::Response> {
        let url = format!("{}{MISTRAL_OCR_PATH}", self.base_url.trim_end_matches('/'));
        let response_result = tokio::time::timeout(
            self.timeout,
            with_request_id(
                self.http_client
                    .post(&url)
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
                // Log full error server-side, but strip URL from user-facing message
                // to avoid leaking internal endpoint details (e.g. api.mistral.ai/v1/ocr).
                tracing::warn!(error = %e, "Mistral OCR network error");
                // Classify retryability before consuming `e` with `without_url()`.
                let is_retryable = e.is_timeout() || e.is_connect();
                let safe_reason = format!("OCR service error: {}", e.without_url());
                Err((
                    AppError::from(SourceError::PdfExtractionFailed {
                        reason: safe_reason,
                    }),
                    None,
                    is_retryable,
                ))
            }
            Err(_) => {
                let timeout_secs = self.timeout.as_secs();
                tracing::warn!(timeout_secs, "Mistral OCR request timed out");
                // OCR timeouts are not retryable: a 120s timeout on a large PDF
                // upload indicates the file is too large to process, not a transient fault.
                // Retrying would amplify the stall to ~8 minutes (4 attempts × 120s).
                Err(self.ocr_error(
                    format!("Request timed out after {timeout_secs} seconds"),
                    false,
                ))
            }
        }
    }

    /// Parse and validate OCR response.
    async fn parse_ocr_response(&self, response: reqwest::Response) -> HttpResult<OcrResult> {
        let ocr_response: OcrApiResponse = response.json().await.map_err(|e| {
            // Log full error server-side, but strip URL from user-facing message
            // to avoid leaking internal endpoint details.
            tracing::error!(error = %e, "Failed to parse Mistral OCR response");
            self.ocr_error(
                format!("Failed to parse OCR response: {}", e.without_url()),
                false,
            )
        })?;

        // Fallback to pages.len() when usage_info is absent (observed in some Mistral
        // API responses). This may under-count if Mistral processed more pages than it
        // returned (e.g., blank pages omitted). Downstream billing uses this value, so
        // the approximation errs on the side of under-charging rather than over-charging.
        let pages_processed = ocr_response
            .usage_info
            .map(|u| u.pages_processed)
            .unwrap_or(u32::try_from(ocr_response.pages.len()).unwrap_or(u32::MAX));

        if pages_processed == 0 {
            tracing::warn!(
                provider = PROVIDER_NAME,
                model = %self.model,
                "Mistral OCR returned zero pages"
            );
        }

        Ok(OcrResult {
            pages: ocr_response.pages,
            pages_processed,
        })
    }

    /// Helper to create OCR error tuple.
    #[allow(clippy::unused_self)]
    fn ocr_error(&self, reason: String, is_retryable: bool) -> (AppError, Option<u16>, bool) {
        (
            AppError::from(SourceError::PdfExtractionFailed { reason }),
            None,
            is_retryable,
        )
    }
}

super::impl_client_resilience_methods!(MistralOcrClient, PROVIDER_NAME);

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal client for unit tests (no real HTTP).
    fn test_client() -> MistralOcrClient {
        let metrics = Arc::new(ProviderMetrics::new("mistral_ocr_test"));
        MistralOcrClient::new(
            "test-key",
            "mistral-ocr-latest".to_string(),
            50 * 1024 * 1024,
            Duration::from_secs(10),
            metrics,
        )
        .expect("test client should build")
    }

    #[tokio::test]
    async fn extract_empty_pdf_bytes_returns_error() {
        let client = test_client();
        let result = client.extract_text_from_pdf(&[], None).await;
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("PDF content is empty"),
            "expected 'PDF content is empty', got: {err}"
        );
    }

    #[tokio::test]
    async fn extract_oversized_pdf_returns_error() {
        // Use a tiny limit (1 KB) to avoid allocating 50 MB in unit tests.
        let metrics = Arc::new(ProviderMetrics::new("mistral_ocr_test"));
        let client = MistralOcrClient::new(
            "test-key",
            "mistral-ocr-latest".to_string(),
            1024, // 1 KB limit
            Duration::from_secs(10),
            metrics,
        )
        .expect("test client should build");

        let oversized = vec![0u8; 1025];
        let result = client.extract_text_from_pdf(&oversized, None).await;
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("exceeds OCR limit"),
            "expected 'exceeds OCR limit', got: {err}"
        );
    }

    #[test]
    fn parse_ocr_response_with_usage_info() {
        let json = serde_json::json!({
            "pages": [
                { "index": 0, "markdown": "# Page 1\nHello world" },
                { "index": 1, "markdown": "## Page 2\nFoo bar" }
            ],
            "usage_info": { "pages_processed": 5 }
        });

        let response: OcrApiResponse = serde_json::from_value(json).unwrap();
        assert_eq!(response.pages.len(), 2);
        assert_eq!(response.pages[0].index, 0);
        assert_eq!(response.pages[0].markdown, "# Page 1\nHello world");
        assert_eq!(response.pages[1].index, 1);
        assert_eq!(response.usage_info.unwrap().pages_processed, 5);
    }

    #[test]
    fn parse_ocr_response_without_usage_info_falls_back_to_pages_len() {
        let json = serde_json::json!({
            "pages": [
                { "index": 0, "markdown": "content" },
                { "index": 2, "markdown": "more content" }
            ]
        });

        let response: OcrApiResponse = serde_json::from_value(json).unwrap();
        assert!(response.usage_info.is_none());
        // Fallback: pages_processed = pages.len()
        let pages_processed = response
            .usage_info
            .map(|u| u.pages_processed)
            .unwrap_or(response.pages.len() as u32);
        assert_eq!(pages_processed, 2);
    }

    #[test]
    fn parse_ocr_response_with_zero_pages_processed() {
        let json = serde_json::json!({
            "pages": [],
            "usage_info": { "pages_processed": 0 }
        });

        let response: OcrApiResponse = serde_json::from_value(json).unwrap();
        let pages_processed = response
            .usage_info
            .map(|u| u.pages_processed)
            .unwrap_or(response.pages.len() as u32);
        assert_eq!(pages_processed, 0);
        assert!(response.pages.is_empty());
    }

    #[test]
    fn parse_ocr_response_with_empty_pages_no_usage_info() {
        let json = serde_json::json!({
            "pages": []
        });

        let response: OcrApiResponse = serde_json::from_value(json).unwrap();
        assert!(response.usage_info.is_none());
        let pages_processed = response
            .usage_info
            .map(|u| u.pages_processed)
            .unwrap_or(response.pages.len() as u32);
        assert_eq!(pages_processed, 0);
    }

    #[test]
    fn ocr_request_serializes_without_pages_when_none() {
        let req = OcrRequest {
            model: "mistral-ocr-latest".to_string(),
            document: OcrDocument {
                doc_type: "document_url",
                document_url: "data:application/pdf;base64,AAAA".to_string(),
            },
            pages: None,
            include_image_base64: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert!(!json.as_object().unwrap().contains_key("pages"));
        assert_eq!(json["include_image_base64"], false);
    }

    #[test]
    fn ocr_request_serializes_with_pages_when_some() {
        let req = OcrRequest {
            model: "mistral-ocr-latest".to_string(),
            document: OcrDocument {
                doc_type: "document_url",
                document_url: "data:application/pdf;base64,AAAA".to_string(),
            },
            pages: Some(vec![0, 2, 5]),
            include_image_base64: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["pages"], serde_json::json!([0, 2, 5]));
    }

    #[test]
    fn debug_impl_does_not_leak_api_key() {
        let client = test_client();
        let debug = format!("{client:?}");
        assert!(debug.contains("MistralOcrClient"));
        assert!(debug.contains("mistral-ocr-latest"));
        // API key must never appear in Debug output
        assert!(
            !debug.contains("test-key"),
            "Debug output must not contain the API key"
        );
    }
}
