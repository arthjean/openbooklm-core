//! Conversation-aware query reformulation for RAG retrieval.
//!
//! Reformulates ambiguous follow-up questions (e.g., "and for last year?")
//! into standalone queries using chat history context, so retrieval finds
//! the right chunks even when the user's question is implicit.
//!
//! Uses Claude Haiku for fast, cheap reformulation (< 500ms target).

use std::sync::Arc;

use crate::clients::{
    AnthropicMessagesClient, ClientMetrics, MessagesRequest, MessagesRequestMessage,
};
use crate::error::AppError;
use crate::services::rag::eval::trace::query_hash;

// ============================================================================
// Constants
// ============================================================================

/// Default model for query reformulation.
const REFORMULATION_MODEL: &str = "claude-haiku-4-5-20251001";

/// Maximum tokens for the reformulated query.
const MAX_REFORMULATION_TOKENS: i32 = 256;

/// Default number of recent chat messages to include as context.
pub const DEFAULT_HISTORY_CONTEXT: usize = 5;

const SYSTEM_PROMPT: &str = "\
You are a search query reformulator. Given a conversation history and the user's latest question, \
rewrite the question as a standalone, self-contained search query that captures the full intent. \
Include any relevant entities, dates, or context from the conversation that would help find the answer. \
Output ONLY the reformulated query, nothing else. \
If the question is already clear and self-contained, return it unchanged. \
Write in the same language as the user's question.";

// ============================================================================
// Types
// ============================================================================

/// A chat message for reformulation context.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// Result of query reformulation.
#[derive(Debug, Clone)]
pub struct ReformulationResult {
    /// The reformulated query.
    pub query: String,
    /// Whether the query was actually changed.
    pub was_reformulated: bool,
    /// Token usage for cost tracking.
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ============================================================================
// Service
// ============================================================================

/// Reformulates queries using conversation context.
#[derive(Clone)]
pub struct QueryReformulator {
    client: AnthropicMessagesClient,
}

impl QueryReformulator {
    /// Create a new reformulator from an Anthropic API key.
    pub fn new(api_key: impl Into<Arc<str>>, metrics: &ClientMetrics) -> Result<Self, AppError> {
        let client = AnthropicMessagesClient::new(
            api_key,
            "query_reformulation",
            15,
            metrics.provider("query_reformulation"),
        )?;

        Ok(Self { client })
    }

    /// Reformulate a query given conversation history.
    ///
    /// If the history is empty or the query appears self-contained, returns the
    /// original query. On error, falls back to the original query (graceful degradation).
    pub async fn reformulate(&self, query: &str, history: &[ChatTurn]) -> ReformulationResult {
        // No history → nothing to reformulate
        if history.is_empty() {
            return ReformulationResult {
                query: query.to_string(),
                was_reformulated: false,
                input_tokens: 0,
                output_tokens: 0,
            };
        }

        match self.call_llm(query, history).await {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    query_hash = %query_hash(query),
                    "Query reformulation failed, using original query"
                );
                ReformulationResult {
                    query: query.to_string(),
                    was_reformulated: false,
                    input_tokens: 0,
                    output_tokens: 0,
                }
            }
        }
    }

    async fn call_llm(
        &self,
        query: &str,
        history: &[ChatTurn],
    ) -> Result<ReformulationResult, AppError> {
        // Build messages: history + current query
        let mut messages: Vec<MessagesRequestMessage> = history
            .iter()
            .map(|turn| MessagesRequestMessage {
                role: turn.role.clone(),
                content: turn.content.clone(),
            })
            .collect();

        messages.push(MessagesRequestMessage {
            role: "user".into(),
            content: format!(
                "Reformulate this search query given the conversation above:\n\n{query}"
            ),
        });

        let request = MessagesRequest {
            model: REFORMULATION_MODEL,
            max_tokens: MAX_REFORMULATION_TOKENS,
            system: SYSTEM_PROMPT,
            messages,
            temperature: None,
        };

        let response = self.client.send(&request).await?;

        let reformulated = response.text().unwrap_or(query).trim().to_string();

        let was_reformulated = reformulated != query;

        if was_reformulated {
            tracing::debug!(
                query_hash = %query_hash(query),
                reformulated_query_hash = %query_hash(&reformulated),
                "Query reformulated"
            );
        }

        Ok(ReformulationResult {
            query: reformulated,
            was_reformulated,
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn empty_history_returns_original() {
        let metrics = ClientMetrics::new();
        let reformulator = QueryReformulator::new("test-key", &metrics)
            .expect("HTTP client should build in test environment");

        let result = reformulator.reformulate("What is revenue?", &[]).await;
        assert_eq!(result.query, "What is revenue?");
        assert!(!result.was_reformulated);
    }

    #[test]
    fn default_history_context_is_reasonable() {
        const {
            assert!(DEFAULT_HISTORY_CONTEXT >= 3);
            assert!(DEFAULT_HISTORY_CONTEXT <= 10);
        }
    }

    #[test]
    fn chat_turn_construction() {
        let turn = ChatTurn {
            role: "user".into(),
            content: "Tell me about Tesla".into(),
        };
        assert_eq!(turn.role, "user");
        assert_eq!(turn.content, "Tell me about Tesla");
    }
}
