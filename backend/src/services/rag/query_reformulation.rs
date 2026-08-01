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
use crate::llm::budget::{context_window_for_model, estimate_tokens};
use crate::services::rag::eval::trace::{ReasonCode, query_hash};

// ============================================================================
// Constants
// ============================================================================

/// Provider of the reformulation model, for context-window lookup.
const REFORMULATION_PROVIDER: &str = "anthropic";

/// Default model for query reformulation.
const REFORMULATION_MODEL: &str = "claude-haiku-4-5-20251001";

/// Maximum tokens for the reformulated query.
const MAX_REFORMULATION_TOKENS: i32 = 256;

/// Default number of recent chat messages to include as context.
pub const DEFAULT_HISTORY_CONTEXT: usize = 5;

/// Share of the reformulation model's input window this call may spend.
///
/// Reformulation is a cheap disambiguation step in front of retrieval, not a
/// second conversation. Twenty percent keeps a long thread from turning every
/// follow-up into a full-context provider call, and the message cap below
/// usually binds first, which is the point of enforcing both (US-017).
const INPUT_BUDGET_SHARE: f64 = 0.20;

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

/// What reformulation did, for the retrieval trace (US-017).
///
/// Reported rather than inferred from whether the query changed: "skipped
/// because there was no history" and "attempted and the provider refused" are
/// the same string comparison downstream and completely different operational
/// facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReformulationOutcome {
    /// Not attempted: no eligible history, or nothing to disambiguate.
    Skipped,
    /// Attempted with the full eligible history.
    Applied,
    /// Attempted after the history was cut to fit the message or token budget.
    Truncated,
    /// Attempted and failed; the caller continues with the original query.
    Failed,
}

impl ReformulationOutcome {
    /// The reason codes this outcome contributes to a retrieval trace.
    #[must_use]
    pub fn reason_codes(self) -> Vec<ReasonCode> {
        match self {
            Self::Skipped => vec![ReasonCode::ReformulationSkipped],
            Self::Applied => vec![ReasonCode::ReformulationApplied],
            // Truncation is a property of the attempt, not an alternative to
            // it: an operator needs to see both that it ran and that it ran on
            // a cut history.
            Self::Truncated => vec![
                ReasonCode::ReformulationApplied,
                ReasonCode::ReformulationTruncated,
            ],
            Self::Failed => vec![ReasonCode::ReformulationFailed],
        }
    }
}

/// Result of query reformulation.
#[derive(Debug, Clone)]
pub struct ReformulationResult {
    /// The reformulated query.
    pub query: String,
    /// Whether the query was actually changed.
    pub was_reformulated: bool,
    /// What the attempt did, for the trace.
    pub outcome: ReformulationOutcome,
    /// Token usage for cost tracking.
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl ReformulationResult {
    /// The original query, unchanged, with the reason it was not rewritten.
    fn unchanged(query: &str, outcome: ReformulationOutcome) -> Self {
        Self {
            query: query.to_string(),
            was_reformulated: false,
            outcome,
            input_tokens: 0,
            output_tokens: 0,
        }
    }
}

/// The history slice a reformulation call may send, and whether it was cut.
///
/// Both limits are applied here, before the provider call: fitting after the
/// call is how an over-budget history becomes a provider rejection instead of
/// a truncation (US-017).
fn fit_history<'a>(history: &'a [ChatTurn], query: &str) -> (Vec<&'a ChatTurn>, bool) {
    let eligible = history.len().min(DEFAULT_HISTORY_CONTEXT);
    let mut truncated = history.len() > eligible;

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let token_budget = (context_window_for_model(REFORMULATION_PROVIDER, REFORMULATION_MODEL)
        as f64
        * INPUT_BUDGET_SHARE) as usize;

    // The current query is never dropped. A reformulation call without it has
    // nothing to reformulate, so it is charged against the budget first and
    // the history fills whatever remains.
    let mut spent = estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(query);
    let mut kept: Vec<&ChatTurn> = Vec::with_capacity(eligible);

    // Newest first: a follow-up is disambiguated by what was just said.
    for turn in history[history.len() - eligible..].iter().rev() {
        let cost = estimate_tokens(&turn.content);
        if spent + cost > token_budget {
            truncated = true;
            break;
        }
        spent += cost;
        kept.push(turn);
    }

    kept.reverse();
    (kept, truncated)
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
    /// At most the five most recent turns and at most 20% of the reformulation
    /// model's input window reach the provider, whichever binds first; the
    /// current query always survives (US-017). On timeout or provider error
    /// the original query comes back once, and the caller does not retry.
    pub async fn reformulate(&self, query: &str, history: &[ChatTurn]) -> ReformulationResult {
        // No history → nothing to disambiguate against
        if history.is_empty() {
            return ReformulationResult::unchanged(query, ReformulationOutcome::Skipped);
        }

        let (fitted, truncated) = fit_history(history, query);
        if fitted.is_empty() {
            // Everything was over budget. Retrieval continues with the query as
            // typed rather than sending a history-less rewrite request that
            // could only echo it back.
            tracing::debug!(
                query_hash = %query_hash(query),
                history_turns = history.len(),
                "Reformulation skipped: no history turn fits the token budget"
            );
            return ReformulationResult::unchanged(query, ReformulationOutcome::Skipped);
        }

        match self.call_llm(query, &fitted).await {
            Ok(mut result) => {
                if truncated {
                    result.outcome = ReformulationOutcome::Truncated;
                }
                result
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    query_hash = %query_hash(query),
                    "Query reformulation failed, using original query"
                );
                ReformulationResult::unchanged(query, ReformulationOutcome::Failed)
            }
        }
    }

    async fn call_llm(
        &self,
        query: &str,
        history: &[&ChatTurn],
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
            // The call ran. A provider that judged the query already
            // self-contained answered it: that is neither a skip nor a
            // failure, and `was_reformulated` already carries the difference.
            outcome: ReformulationOutcome::Applied,
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

    // ── History and token budget (US-017) ─────────────────────────────────

    fn turns(count: usize, words: usize) -> Vec<ChatTurn> {
        (0..count)
            .map(|i| ChatTurn {
                role: if i % 2 == 0 { "user" } else { "assistant" }.into(),
                content: format!("turn{i} ").repeat(words),
            })
            .collect()
    }

    #[test]
    fn at_most_the_five_latest_turns_are_sent() {
        let history = turns(12, 3);
        let (fitted, truncated) = fit_history(&history, "and for last year?");

        assert_eq!(fitted.len(), DEFAULT_HISTORY_CONTEXT);
        assert!(truncated, "dropping seven turns is a truncation");
        // The five kept must be the five most recent, in conversation order.
        let kept: Vec<&str> = fitted.iter().map(|t| t.content.as_str()).collect();
        let expected: Vec<&str> = history[7..].iter().map(|t| t.content.as_str()).collect();
        assert_eq!(kept, expected);
    }

    #[test]
    fn a_short_history_is_sent_whole_and_is_not_marked_truncated() {
        let history = turns(3, 2);
        let (fitted, truncated) = fit_history(&history, "what about revenue?");
        assert_eq!(fitted.len(), 3);
        assert!(!truncated);
    }

    #[test]
    fn the_token_budget_binds_before_the_message_cap_on_huge_turns() {
        // Each turn is far larger than 20% of the Haiku input window, so not
        // one of them fits alongside the query.
        let window = context_window_for_model(REFORMULATION_PROVIDER, REFORMULATION_MODEL);
        let oversized = "word ".repeat(window);
        let history = vec![
            ChatTurn {
                role: "user".into(),
                content: oversized.clone(),
            },
            ChatTurn {
                role: "assistant".into(),
                content: oversized,
            },
        ];

        let (fitted, truncated) = fit_history(&history, "and after that?");
        assert!(
            fitted.is_empty(),
            "an over-budget history must be cut, not sent"
        );
        assert!(truncated);
    }

    #[test]
    fn the_budget_is_a_fifth_of_the_reformulation_window() {
        let window = context_window_for_model(REFORMULATION_PROVIDER, REFORMULATION_MODEL);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let budget = (window as f64 * INPUT_BUDGET_SHARE) as usize;

        // One turn just under the budget fits; the same turn plus a second one
        // does not, which is what makes the limit a budget and not a count.
        // `estimate_tokens` counts bytes/4, so a turn of 6 * budget * 0.4 bytes
        // is about 60% of the budget: one fits, two cannot.
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let words = ((budget as f64) * 0.4) as usize;
        let filler = "token ".repeat(words);
        let history = vec![
            ChatTurn {
                role: "user".into(),
                content: filler.clone(),
            },
            ChatTurn {
                role: "assistant".into(),
                content: filler,
            },
        ];
        let (fitted, truncated) = fit_history(&history, "follow up");
        assert_eq!(fitted.len(), 1, "only the newest turn fits the budget");
        assert!(truncated);
    }

    #[tokio::test]
    async fn an_empty_history_is_a_skip_not_a_failure() {
        let metrics = ClientMetrics::new();
        let reformulator = QueryReformulator::new("test-key", &metrics).expect("client builds");
        let result = reformulator.reformulate("What is revenue?", &[]).await;
        assert_eq!(result.outcome, ReformulationOutcome::Skipped);
        assert_eq!(
            result.outcome.reason_codes(),
            vec![ReasonCode::ReformulationSkipped]
        );
    }

    #[test]
    fn a_truncated_attempt_reports_both_that_it_ran_and_that_it_was_cut() {
        assert_eq!(
            ReformulationOutcome::Truncated.reason_codes(),
            vec![
                ReasonCode::ReformulationApplied,
                ReasonCode::ReformulationTruncated
            ]
        );
        assert_eq!(
            ReformulationOutcome::Failed.reason_codes(),
            vec![ReasonCode::ReformulationFailed]
        );
    }
}
