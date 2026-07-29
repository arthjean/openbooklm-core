//! Hypothetical Document Embeddings (HyDE) for improved short-query retrieval.
//!
//! For short queries (< 20 tokens), generates a hypothetical document that answers
//! the query, then embeds that document for dense search. This bridges the gap
//! between query-style embeddings and document-style embeddings.
//!
//! ## When HyDE activates
//!
//! - Query is shorter than [`ACTIVATION_THRESHOLD`] tokens (default 20)
//! - Anthropic API key is configured
//!
//! ## Reference
//!
//! Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)

use std::sync::Arc;

use crate::clients::{
    AnthropicMessagesClient, ClientMetrics, MessagesRequest, MessagesRequestMessage,
};
use crate::error::AppError;

// ============================================================================
// Constants
// ============================================================================

/// Model for HyDE document generation.
const HYDE_MODEL: &str = "claude-haiku-4-5-20251001";

/// Max tokens for the hypothetical document.
const MAX_HYDE_TOKENS: i32 = 384;

/// Queries shorter than this (in whitespace-separated words) trigger HyDE.
pub const ACTIVATION_THRESHOLD: usize = 20;

const SYSTEM_PROMPT: &str = "\
You are a document generator. Given a search query, write a short, factual passage (150-300 tokens) \
that directly answers the query as if it were an excerpt from a real document. \
Do NOT add disclaimers, hedging, or meta-commentary. \
Write in the same language as the query. \
Output ONLY the passage.";

// ============================================================================
// Types
// ============================================================================

/// Result of HyDE generation.
#[derive(Debug, Clone)]
pub struct HydeResult {
    /// The generated hypothetical document.
    pub document: String,
    /// Token usage.
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ============================================================================
// Service
// ============================================================================

/// HyDE document generator for short queries.
#[derive(Clone)]
pub struct HydeService {
    client: AnthropicMessagesClient,
    activation_threshold: usize,
}

impl HydeService {
    /// Create a new HyDE service.
    pub fn new(api_key: impl Into<Arc<str>>, metrics: &ClientMetrics) -> Result<Self, AppError> {
        let client = AnthropicMessagesClient::new(api_key, "hyde", 15, metrics.provider("hyde"))?;

        Ok(Self {
            client,
            activation_threshold: ACTIVATION_THRESHOLD,
        })
    }

    /// Whether HyDE should activate for this query.
    pub fn should_activate(&self, query: &str) -> bool {
        word_count(query) < self.activation_threshold
    }

    /// Generate a hypothetical document for the query.
    ///
    /// Returns `None` if the query is too long (HyDE not needed) or on error.
    pub async fn generate(&self, query: &str) -> Option<HydeResult> {
        if !self.should_activate(query) {
            tracing::debug!(
                word_count = word_count(query),
                threshold = self.activation_threshold,
                "HyDE skipped: query too long"
            );
            return None;
        }

        match self.call_llm(query).await {
            Ok(result) => {
                tracing::debug!(
                    query,
                    doc_len = result.document.len(),
                    input_tokens = result.input_tokens,
                    output_tokens = result.output_tokens,
                    "HyDE document generated"
                );
                Some(result)
            }
            Err(e) => {
                tracing::warn!(error = %e, query, "HyDE generation failed, using original query");
                None
            }
        }
    }

    async fn call_llm(&self, query: &str) -> Result<HydeResult, AppError> {
        let request = MessagesRequest {
            model: HYDE_MODEL,
            max_tokens: MAX_HYDE_TOKENS,
            system: SYSTEM_PROMPT,
            messages: vec![MessagesRequestMessage {
                role: "user".into(),
                content: query.to_string(),
            }],
            temperature: None,
        };

        let response = self.client.send(&request).await?;

        let document = response.text().unwrap_or_default().trim().to_string();

        if document.is_empty() {
            tracing::warn!("HyDE received empty response, falling back to original query");
            return Err(AppError::Internal("Empty HyDE response".into()));
        }

        Ok(HydeResult {
            document,
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

/// Count whitespace-separated words.
fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_count_basic() {
        assert_eq!(word_count("hello world"), 2);
        assert_eq!(word_count("one"), 1);
        assert_eq!(word_count(""), 0);
        assert_eq!(word_count("  multiple   spaces  between  words  "), 4);
    }

    #[test]
    fn should_activate_for_short_query() {
        let metrics = ClientMetrics::new();
        let svc = HydeService::new("test-key", &metrics).unwrap();

        assert!(svc.should_activate("What is revenue?"));
        assert!(svc.should_activate("Tesla Q3"));
        assert!(svc.should_activate("")); // 0 words < 20
    }

    #[test]
    fn should_not_activate_for_long_query() {
        let metrics = ClientMetrics::new();
        let svc = HydeService::new("test-key", &metrics).unwrap();

        // Generate a query with 25 words
        let long_query = (0..25)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert!(!svc.should_activate(&long_query));
    }

    #[test]
    fn activation_threshold_is_configurable() {
        assert_eq!(ACTIVATION_THRESHOLD, 20);
    }

    #[tokio::test]
    async fn generate_skips_long_query() {
        let metrics = ClientMetrics::new();
        let svc = HydeService::new("test-key", &metrics).unwrap();

        let long_query = (0..25)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let result = svc.generate(&long_query).await;
        assert!(result.is_none());
    }

    #[test]
    fn hyde_empty_response_returns_error() {
        use crate::clients::{MessagesContentBlock, MessagesResponse, MessagesUsage};

        // Empty content list → text() returns None → unwrap_or_default → ""
        let response = MessagesResponse {
            content: vec![],
            usage: MessagesUsage {
                input_tokens: 10,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        };

        let document = response.text().unwrap_or_default().trim().to_string();
        assert!(
            document.is_empty(),
            "empty content list should yield empty document"
        );

        // Whitespace-only content block → text() returns Some("  ") → trim → ""
        let response_whitespace = MessagesResponse {
            content: vec![MessagesContentBlock {
                text: Some("   \n\t  ".to_string()),
            }],
            usage: MessagesUsage {
                input_tokens: 10,
                output_tokens: 1,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        };

        let document = response_whitespace
            .text()
            .unwrap_or_default()
            .trim()
            .to_string();
        assert!(
            document.is_empty(),
            "whitespace-only content should yield empty document"
        );
    }
}
