//! Chat service for message management and persistence.
//!
//! Delegates all data access to repository traits.
//! Orchestration logic (validation, RAG, memory) lives in [`orchestration`].

pub mod orchestration;

use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use crate::entities::chat_message;
use crate::error::AppError;
use crate::llm::{Citation, LlmMessage, Role};
use crate::repositories::ChatRepository;

// Re-export types from the repository layer so existing import paths still work.
pub use crate::repositories::{
    DEFAULT_CHAT_HISTORY_LIMIT, MAX_CHAT_HISTORY_LIMIT, PaginatedChatHistory,
};

/// Chat message response DTO.
///
/// Lives in the service layer (not API) because it's shared by both
/// the API handlers and the export service.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ChatMessageResponse {
    pub id: Uuid,
    pub notebook_id: Uuid,
    pub role: String,
    pub content: String,
    pub citations: Vec<Citation>,
    pub model: Option<String>,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rag_log_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Uuid>,
}

impl From<chat_message::Model> for ChatMessageResponse {
    fn from(model: chat_message::Model) -> Self {
        let message_id = model.id;

        let citations: Vec<Citation> = serde_json::from_value(model.citations)
            .inspect_err(
                |e| warn!(message_id = %message_id, error = %e, "Failed to parse citations"),
            )
            .unwrap_or_default();

        Self {
            id: message_id,
            notebook_id: model.notebook_id,
            role: model.role,
            content: model.content,
            citations,
            model: model.model,
            created_at: model.created_at.to_rfc3339(),
            rag_log_id: None,
            feedback: None,
            session_id: model.session_id,
        }
    }
}

// ============================================================================
// Parameters
// ============================================================================

/// Parameters for creating a new chat message.
pub struct CreateMessageParams<'a> {
    pub repo: &'a dyn ChatRepository,
    pub notebook_id: Uuid,
    pub role: &'a str,
    pub content: &'a str,
    pub citations: &'a [Citation],
    pub model: Option<&'a str>,
    pub agent_id: Option<Uuid>,
    pub session_id: Option<Uuid>,
}

// ============================================================================
// Public API — delegates to ChatRepository
// ============================================================================

/// Create a new chat message.
#[tracing::instrument(skip_all, fields(notebook_id = %params.notebook_id))]
pub async fn create_message(
    params: CreateMessageParams<'_>,
) -> Result<chat_message::Model, AppError> {
    params
        .repo
        .create_message(
            params.notebook_id,
            params.role,
            params.content,
            params.citations,
            params.model,
            params.agent_id,
            params.session_id,
        )
        .await
}

/// Get chat history for a notebook (legacy, no pagination).
#[tracing::instrument(skip(repo), fields(%notebook_id))]
pub async fn get_chat_history(
    repo: &dyn ChatRepository,
    notebook_id: Uuid,
    limit: Option<u64>,
) -> Result<Vec<chat_message::Model>, AppError> {
    repo.get_history(notebook_id, limit).await
}

/// Get paginated chat history for a notebook.
#[tracing::instrument(skip(repo), fields(%notebook_id))]
pub async fn get_chat_history_paginated(
    repo: &dyn ChatRepository,
    notebook_id: Uuid,
    offset: Option<u64>,
    limit: Option<u64>,
) -> Result<PaginatedChatHistory, AppError> {
    repo.get_history_paginated(notebook_id, offset, limit).await
}

/// Get recent chat history for context (last N messages, chronological order).
#[tracing::instrument(skip(repo), fields(%notebook_id))]
pub async fn get_recent_history(
    repo: &dyn ChatRepository,
    notebook_id: Uuid,
    max_messages: u64,
) -> Result<Vec<chat_message::Model>, AppError> {
    repo.get_recent_history(notebook_id, max_messages).await
}

/// Clear chat history for a notebook.
#[tracing::instrument(skip(repo), fields(%notebook_id))]
pub async fn clear_chat_history(
    repo: &dyn ChatRepository,
    notebook_id: Uuid,
) -> Result<u64, AppError> {
    repo.clear_history(notebook_id).await
}

/// Convert chat history to LLM message format.
///
/// Borrows the input so the caller can retain the raw models without cloning
/// the entire Vec. Only `content` is cloned per message.
#[tracing::instrument(skip_all, fields(message_count = history.len()))]
pub fn history_to_llm_messages(history: &[chat_message::Model]) -> Vec<LlmMessage> {
    history
        .iter()
        .map(|msg| LlmMessage {
            role: Role::from(msg.role.as_str()),
            content: msg.content.clone(),
        })
        .collect()
}
