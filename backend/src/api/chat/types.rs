//! Request/response types for the chat API.

use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::llm::TeachingMode;
use crate::repositories::{DEFAULT_CHAT_HISTORY_LIMIT, MAX_CHAT_HISTORY_LIMIT};
use crate::services::chat::ChatMessageResponse;
use crate::validation;

// ============================================================================
// Constants
// ============================================================================

pub const DEFAULT_MAX_CONTEXT_CHUNKS: i32 = 15;

/// Maximum allowed context chunks per query (server-enforced cap for DoS prevention).
pub const MAX_CONTEXT_CHUNKS: i32 = 20;

/// Maximum messages to fetch from DB before token-based truncation.
/// Generous upper bound — actual truncation is token-aware.
pub(super) const MAX_HISTORY_FETCH: u64 = 50;

// ============================================================================
// Request types
// ============================================================================

/// Request for sending a chat message.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct SendMessageRequest {
    pub message: String,
    #[serde(default = "default_max_context_chunks")]
    pub max_context_chunks: i32,
    /// Optional LLM provider (e.g., "mistral", "anthropic")
    pub provider: Option<String>,
    /// Optional model override
    pub model: Option<String>,
    /// Teaching mode for pedagogical adaptation
    #[serde(default)]
    pub teaching_mode: TeachingMode,
    /// UI locale for generating suggestions in the correct language (e.g. "fr", "en")
    pub locale: Option<String>,
}

fn default_max_context_chunks() -> i32 {
    DEFAULT_MAX_CONTEXT_CHUNKS
}

impl SendMessageRequest {
    pub(crate) fn validate(&self) -> Result<&str, AppError> {
        validation::validate_message(&self.message)
    }

    /// Returns `max_context_chunks`, or a validation error.
    ///
    /// Rejected rather than clamped since US-013. Clamping made the contract
    /// unobservable: a client asking for 100 contexts got 20 and a client
    /// asking for 0 got 1, both silently, so neither could tell that the
    /// server had substituted a different request. The retrieval pipeline
    /// enforces this exact number on every branch, which is only meaningful if
    /// the number is the one the caller asked for.
    pub(crate) fn validated_max_context_chunks(&self) -> Result<i32, AppError> {
        if (1..=MAX_CONTEXT_CHUNKS).contains(&self.max_context_chunks) {
            Ok(self.max_context_chunks)
        } else {
            Err(AppError::Validation(format!(
                "max_context_chunks must be between 1 and {MAX_CONTEXT_CHUNKS}, got {}",
                self.max_context_chunks
            )))
        }
    }
}

/// Query parameters for paginated chat history.
#[derive(Debug, Deserialize, utoipa::ToSchema, utoipa::IntoParams)]
pub struct ChatHistoryQuery {
    pub offset: Option<u64>,
    pub limit: Option<u64>,
}

impl ChatHistoryQuery {
    /// Returns `(offset, limit)` with server-enforced bounds.
    ///
    /// `limit` is clamped to [`MAX_CHAT_HISTORY_LIMIT`] (200).
    /// `offset` defaults to 0 (always non-negative as `u64`).
    pub(super) fn clamped(&self) -> (u64, u64) {
        validation::validate_pagination(
            self.offset,
            self.limit,
            DEFAULT_CHAT_HISTORY_LIMIT,
            MAX_CHAT_HISTORY_LIMIT,
        )
    }
}

// ============================================================================
// Response types
// ============================================================================

/// Response for paginated chat history.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ChatHistoryResponse {
    pub messages: Vec<ChatMessageResponse>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
    pub has_more: bool,
}

/// Teaching mode info for frontend.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct TeachingModeInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub icon: &'static str,
    pub description: &'static str,
}

impl From<TeachingMode> for TeachingModeInfo {
    fn from(mode: TeachingMode) -> Self {
        Self {
            id: match mode {
                TeachingMode::Flash => "flash",
                TeachingMode::Deep => "deep",
                TeachingMode::Quiz => "quiz",
                TeachingMode::Glossary => "glossary",
                TeachingMode::Summary => "summary",
                TeachingMode::Timeline => "timeline",
            },
            name: mode.display_name(),
            icon: mode.icon(),
            description: mode.description(),
        }
    }
}

/// Response for teaching modes list.
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct TeachingModesResponse {
    pub modes: Vec<TeachingModeInfo>,
    pub default: &'static str,
}

// ============================================================================
// Token budget tests (imports from llm::helpers)
// ============================================================================

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LlmMessage;
    use crate::llm::budget::{allocate_token_budget, fit_prompt_to_budget};

    #[test]
    fn fit_prompt_from_types_module() {
        let messages = vec![LlmMessage::user("Hello"), LlmMessage::assistant("Hi there")];
        let budget = allocate_token_budget("anthropic", "claude-sonnet-4-6-20260220");
        let result = fit_prompt_to_budget("system prompt", "rag context", None, &messages, &budget);
        assert!(!result.was_truncated);
        assert_eq!(result.messages.len(), 2);
    }

    // ====================================================================
    // US-013: max_context_chunks is validated, not clamped
    // ====================================================================

    fn make_request(max_context_chunks: i32) -> SendMessageRequest {
        SendMessageRequest {
            message: "test".into(),
            max_context_chunks,
            provider: None,
            model: None,
            teaching_mode: TeachingMode::default(),
            locale: None,
        }
    }

    #[test]
    fn a_limit_inside_the_contract_is_returned_unchanged() {
        for value in [1, 10, MAX_CONTEXT_CHUNKS] {
            assert_eq!(
                make_request(value)
                    .validated_max_context_chunks()
                    .expect("inside the contract"),
                value
            );
        }
    }

    #[test]
    fn a_limit_above_the_contract_is_rejected_rather_than_clamped() {
        for value in [MAX_CONTEXT_CHUNKS + 1, 100, i32::MAX] {
            let error = make_request(value)
                .validated_max_context_chunks()
                .expect_err("above the contract");
            assert!(
                matches!(error, AppError::Validation(_)),
                "an out-of-contract limit is a validation error, got {error:?}"
            );
            assert!(
                error.to_string().contains(&MAX_CONTEXT_CHUNKS.to_string()),
                "the error must name the bound: {error}"
            );
        }
    }

    #[test]
    fn a_zero_or_negative_limit_is_rejected() {
        for value in [0, -5, i32::MIN] {
            assert!(
                make_request(value).validated_max_context_chunks().is_err(),
                "{value} contexts is not a request the pipeline can honour"
            );
        }
    }

    // ====================================================================
    // US-003: Pagination clamping
    // ====================================================================

    #[test]
    fn pagination_defaults() {
        let query = ChatHistoryQuery {
            offset: None,
            limit: None,
        };
        let (offset, limit) = query.clamped();
        assert_eq!(offset, 0);
        assert_eq!(limit, DEFAULT_CHAT_HISTORY_LIMIT);
    }

    #[test]
    fn pagination_limit_clamped_to_max() {
        let query = ChatHistoryQuery {
            offset: None,
            limit: Some(500),
        };
        let (_, limit) = query.clamped();
        assert_eq!(limit, MAX_CHAT_HISTORY_LIMIT);
    }

    #[test]
    fn pagination_limit_at_max_unchanged() {
        let query = ChatHistoryQuery {
            offset: None,
            limit: Some(MAX_CHAT_HISTORY_LIMIT),
        };
        let (_, limit) = query.clamped();
        assert_eq!(limit, MAX_CHAT_HISTORY_LIMIT);
    }

    #[test]
    fn pagination_explicit_values_preserved() {
        let query = ChatHistoryQuery {
            offset: Some(42),
            limit: Some(25),
        };
        let (offset, limit) = query.clamped();
        assert_eq!(offset, 42);
        assert_eq!(limit, 25);
    }

    #[test]
    fn pagination_extreme_limit_clamped() {
        let query = ChatHistoryQuery {
            offset: Some(0),
            limit: Some(u64::MAX),
        };
        let (_, limit) = query.clamped();
        assert_eq!(limit, MAX_CHAT_HISTORY_LIMIT);
    }
}
