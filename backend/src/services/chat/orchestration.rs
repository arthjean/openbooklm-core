//! Chat orchestration — business logic extracted from `api/chat/mod.rs`.
//!
//! Contains the pure helpers, data types, and async orchestration functions
//! that were previously private helpers in the API handler module.

use std::sync::Arc;

use uuid::Uuid;

use crate::api::chat::types::SendMessageRequest;
use crate::api::common::verify_notebook_access;
use crate::core::CoreState;
use crate::core::config::CoreConfig;
use crate::core::entitlements::{AuthorizationRequest, Operation, Permit};
use crate::core::principal::Principal;
use crate::core::protocol::{ChatEvent, ChatEventStream, ThinkingStage, WarningKind};
use crate::entities::chat_message;
use crate::error::AppError;
use crate::llm::{LlmMessage, LlmProvider, TeachingMode, build_system_prompt};
use crate::repositories::{MemoryRepository, RagLogRepository, SearchRepository};
use crate::services::memory::{format_memory_for_prompt, select_core_memories};
use crate::services::rag::eval::trace::query_hash;
use crate::services::rag::query_reformulation::ChatTurn;
use crate::services::rag::search::{
    CorrectiveRetrievalParams, PipelineTimings, PreferenceBoost, RetrievalParams,
    extract_preference_keywords, retrieve_context, retrieve_context_corrective,
};
use crate::types::SearchResult;

// ============================================================================
// Data types
// ============================================================================

/// Validated and authorized chat request data.
pub(crate) struct ValidatedRequest {
    pub message: String,
    /// Permit for this chat exchange. Consumption is recorded only after the
    /// provider emits its first non-empty token.
    pub permit: Permit,
    pub provider: Arc<dyn LlmProvider>,
    pub model_for_stream: Option<String>,
    /// Resolved model ID (e.g. "claude-sonnet-4-6-20260220").
    pub model_id: String,
    /// Whether memory extraction is enabled for this notebook.
    pub memory_enabled: bool,
    /// Session ID for grouping messages within a conversation session.
    pub session_id: Uuid,
}

/// Loaded chat history (not yet truncated — truncation happens in `fit_prompt_to_budget`).
pub(crate) struct ChatHistory {
    /// Raw DB models (used for building ChatTurn history for corrective RAG).
    pub raw: Vec<chat_message::Model>,
    /// LLM-formatted messages (all loaded, not yet truncated).
    pub all_messages: Vec<LlmMessage>,
}

/// What retrieval produced for one chat turn.
///
/// Replaces the `(chunks, timings)` pair so that the reformulated query can
/// reach the retrieval trace. It travels as text and is hashed at the trace
/// boundary: the caller needs the value to decide whether reformulation
/// happened, and the log needs never to see it.
pub(crate) struct RetrievedContext {
    pub chunks: Vec<SearchResult>,
    pub timings: PipelineTimings,
    /// The standalone query retrieval actually used, when it differs from what
    /// the user typed.
    pub reformulated_query: Option<String>,
}

impl RetrievedContext {
    /// Retrieval failed or was skipped. Chat continues without evidence.
    pub(crate) fn empty() -> Self {
        Self {
            chunks: Vec::new(),
            timings: PipelineTimings::default(),
            reformulated_query: None,
        }
    }
}

/// Parameters for RAG context retrieval.
pub(crate) struct RagContextParams<'a> {
    pub search_repo: &'a dyn SearchRepository,
    pub config: &'a CoreConfig,
    pub notebook_id: Uuid,
    pub query: &'a str,
    pub max_chunks: i32,
    pub embeddings: &'a Option<crate::core::providers::SharedEmbeddingProvider>,
    pub reranker: &'a Option<crate::core::providers::SharedReranker>,
    pub hyde_service: Option<&'a crate::services::rag::hyde::HydeService>,
    pub reformulator: Option<&'a crate::services::rag::query_reformulation::QueryReformulator>,
    pub history: &'a [chat_message::Model],
    pub events: &'a ChatEventStream,
    pub embedding_cache: Option<&'a crate::services::rag::embedding_cache::EmbeddingCache>,
    pub provider: &'a str,
    pub model: &'a str,
    /// Pre-computed preference boost signals.
    pub preference_boost: Option<PreferenceBoost>,
}

// ============================================================================
// Pure helpers
// ============================================================================

/// Session timeout: if the most recent message is older than this, start a new session.
pub(crate) const SESSION_TIMEOUT_SECS: i64 = 30 * 60; // 30 minutes

/// Determine whether to reuse an existing session ID based on elapsed time.
///
/// Returns `Some(session_id)` if the last message is within the session timeout
/// and has a session_id. Returns `None` otherwise (caller generates a new UUID).
pub(crate) fn should_reuse_session(
    elapsed: chrono::Duration,
    latest_session_id: Option<Uuid>,
) -> Option<Uuid> {
    if elapsed < chrono::Duration::seconds(SESSION_TIMEOUT_SECS) {
        latest_session_id
    } else {
        None
    }
}

/// Heuristic to detect queries that likely need reformulation before retrieval.
///
/// A query is flagged as needing proactive reformulation if ANY of:
/// - Short query (< 60 bytes) — likely implicit/referential
/// - Contains common referential pronouns or deictic words
///
/// This is intentionally broad — false positives are cheap (one Haiku call,
/// ~200ms, which returns the original query unchanged if it's already standalone).
/// False negatives are expensive (failed retrieval → bad answer).
pub(crate) fn needs_proactive_reformulation(query: &str) -> bool {
    // Check for referential/deictic words that suggest the query depends on context.
    // Covers both French and English.
    const REFERENTIAL_MARKERS: &[&str] = &[
        // English
        " it ",
        " its ",
        " this ",
        " that ",
        " these ",
        " those ",
        " them ",
        " they ",
        " their ",
        " he ",
        " she ",
        " him ",
        " her ",
        "the same",
        "the previous",
        "the last",
        "the above",
        "compared to",
        "compare with",
        "what about",
        "how about",
        // French
        " il ",
        " elle ",
        " ils ",
        " elles ",
        " ce ",
        " cet ",
        " cette ",
        " ces ",
        " celui ",
        " celle ",
        " ceux ",
        " celles ",
        " le même",
        " la même",
        " les mêmes",
        " précédent",
        " dernier",
        " ci-dessus",
        "par rapport",
        "et pour",
        "qu'en est",
    ];

    // Short queries are often follow-ups ("And for 2023?", "What about X?")
    if query.len() < 60 {
        return true;
    }

    let lower = query.to_lowercase();
    REFERENTIAL_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Build the final system prompt with memory when present.
///
/// Memory is injected between `</sources>` and `<role>` by appending to the
/// context string. This avoids modifying all prompt templates individually.
pub(crate) fn build_system_prompt_for_chat(
    context: &str,
    teaching_mode: TeachingMode,
    locale: &str,
    memory: Option<&str>,
) -> String {
    // Append <memory> block to context so it appears between </sources> and <role>
    let context_with_memory: std::borrow::Cow<'_, str> = match memory {
        Some(mem) => format!("{context}\n\n{mem}").into(),
        None => context.into(),
    };

    build_system_prompt(&context_with_memory, teaching_mode, locale)
}

// ============================================================================
// Async helpers
// ============================================================================

/// Validate the chat request, check access/limits, save user message, and resolve LLM provider.
///
/// Takes `&CoreState` as a pragmatic single dependency — this function touches
/// 5+ repos/clients (notebooks, chat, llm_router, entitlement policy, event
/// sink) making explicit deps unwieldy.
pub(crate) async fn validate_and_authorize(
    state: &CoreState,
    principal: &Principal,
    notebook_id: Uuid,
    payload: &SendMessageRequest,
) -> Result<ValidatedRequest, AppError> {
    let message = payload.validate()?.to_owned();
    verify_notebook_access(state.repos.notebooks.as_ref(), principal, notebook_id).await?;

    // Fetch notebook to read memory_enabled flag (notebook was already validated above)
    let notebook = crate::services::notebooks::get_notebook_for_user(
        state.repos.notebooks.as_ref(),
        notebook_id,
        principal.account_id,
    )
    .await?;

    // Fast-fail authorization (read-only). The permit is what charges the
    // account, and only after the provider emits its first token.
    let permit = state
        .entitlements
        .authorize(AuthorizationRequest::new(
            principal,
            Operation::SendChatMessage { notebook_id },
            Uuid::new_v4(),
        ))
        .await?;

    // Resolve LLM provider with automatic fallback + model mapping
    // Done before persisting the user message so that model access failures
    // don't leave orphan user messages with no assistant response.
    let selection = state
        .clients
        .llm_router
        .get_provider(payload.provider.as_deref(), payload.model.as_deref())?;
    let model_id = selection
        .model_override
        .as_deref()
        .or(payload.model.as_deref())
        .unwrap_or_else(|| selection.provider.default_model())
        .to_owned();
    state
        .entitlements
        .authorize(AuthorizationRequest::new(
            principal,
            Operation::SelectModel {
                provider: selection.provider.name().to_owned(),
                model: model_id.clone(),
            },
            Uuid::new_v4(),
        ))
        .await?;

    // Resolve session ID: reuse from last message if < 30 min ago, else new
    let session_id = resolve_session_id(state.repos.chat.as_ref(), notebook_id).await;

    // Persist the user's message (after all validation passes)
    super::create_message(super::CreateMessageParams {
        repo: state.repos.chat.as_ref(),
        notebook_id,
        role: "user",
        content: &message,
        citations: &[],
        model: None,
        agent_id: None,
        session_id: Some(session_id),
    })
    .await?;

    Ok(ValidatedRequest {
        message,
        permit,
        provider: selection.provider,
        model_for_stream: selection.model_override.or_else(|| payload.model.clone()),
        model_id,
        memory_enabled: notebook.memory_enabled,
        session_id,
    })
}

/// Resolve the session ID for a new message.
///
/// Reuses the session_id from the most recent message if it was created within
/// 30 minutes. Otherwise generates a new UUID.
///
/// **Known limitation (TOCTOU):** This read-then-write pattern is not atomic.
/// Two concurrent requests to the same notebook can both read the same latest
/// message and independently resolve session IDs, potentially producing
/// inconsistent groupings. This is acceptable because session IDs are advisory
/// (used for analytics/summarization, not access control) and concurrent
/// requests to the same single-user notebook are extremely rare in practice.
pub(crate) async fn resolve_session_id(
    chat_repo: &dyn crate::repositories::ChatRepository,
    notebook_id: Uuid,
) -> Uuid {
    if let Ok(Some(latest)) = chat_repo.get_latest_message(notebook_id).await {
        let now = chrono::Utc::now();
        let message_time = latest.created_at.to_utc();
        let elapsed = now.signed_duration_since(message_time);
        if let Some(sid) = should_reuse_session(elapsed, latest.session_id) {
            return sid;
        }
    }
    Uuid::new_v4()
}

/// Load recent chat history from DB.
///
/// Returns raw + LLM-formatted messages. Token-aware truncation is deferred
/// to [`fit_prompt_to_budget`] which accounts for all prompt segments.
pub(crate) async fn load_chat_history(
    chat_repo: &dyn crate::repositories::ChatRepository,
    notebook_id: Uuid,
    max_history: u64,
) -> Result<ChatHistory, AppError> {
    let raw = super::get_recent_history(chat_repo, notebook_id, max_history).await?;
    let all_messages = super::history_to_llm_messages(&raw);
    Ok(ChatHistory { raw, all_messages })
}

/// Notify the client if history was truncated to fit the token budget.
pub(crate) async fn notify_history_truncation(
    events: &ChatEventStream,
    was_truncated: bool,
    total_before: usize,
    kept: usize,
    notebook_id: Uuid,
    provider_name: &str,
) {
    if was_truncated {
        tracing::info!(
            %notebook_id,
            original = total_before,
            kept,
            provider = provider_name,
            "Chat history truncated to fit token budget"
        );
        events.emit(ChatEvent::history_truncated(kept)).await;
    }
}

/// Build preference boost signals for RAG retrieval.
///
/// Fetches source IDs with positive user feedback from rag_logs and extracts
/// topic keywords from preference-type memories. Returns `None` if no signals
/// exist (zero-cost default path).
pub(crate) async fn build_preference_boost(
    rag_log_repo: &dyn RagLogRepository,
    memory_repo: &dyn MemoryRepository,
    notebook_id: Uuid,
    memory_enabled: bool,
) -> Option<PreferenceBoost> {
    // Fetch source IDs from rag_logs with positive feedback
    let preferred_source_ids = match rag_log_repo.get_preferred_source_ids(notebook_id).await {
        Ok(ids) => ids.into_iter().collect::<std::collections::HashSet<_>>(),
        Err(e) => {
            tracing::warn!(%notebook_id, error = %e, "Failed to load preferred source IDs");
            std::collections::HashSet::new()
        }
    };

    // Extract preference keywords from memory (only if memory is enabled)
    let preference_keywords = if memory_enabled {
        match memory_repo.list_for_notebook(notebook_id).await {
            Ok(memories) => {
                let preference_contents: Vec<String> = memories
                    .iter()
                    .filter(|m| m.memory_type == "preference")
                    .map(|m| m.content.clone())
                    .collect();
                extract_preference_keywords(&preference_contents)
            }
            Err(e) => {
                tracing::warn!(%notebook_id, error = %e, "Failed to load preference memories for boost");
                vec![]
            }
        }
    } else {
        vec![]
    };

    let boost = PreferenceBoost {
        preferred_source_ids,
        preference_keywords,
    };

    if boost.is_empty() {
        None
    } else {
        tracing::debug!(
            %notebook_id,
            preferred_sources = boost.preferred_source_ids.len(),
            preference_keywords = boost.preference_keywords.len(),
            "Built preference boost for RAG retrieval"
        );
        Some(boost)
    }
}

/// Retrieve RAG context chunks, with proactive query reformulation for follow-ups
/// and corrective RAG when a reformulator is available.
///
/// Two-stage reformulation:
/// 1. **Proactive** (before retrieval): If the query looks like a follow-up
///    (short, contains pronouns, not the first message), reformulate to a
///    standalone query before retrieval. This prevents embedding search failures
///    on implicit references like "What about that one?" or "Compare with the
///    previous chapter."
/// 2. **Corrective** (after retrieval): If initial retrieval quality is low
///    (`avg_relevance < 0.5`), reformulate and retry (existing behavior).
///
/// Emits typed thinking and warning events as retrieval progresses.
/// Falls back to an empty context on any error.
pub(crate) async fn retrieve_rag_context(params: &RagContextParams<'_>) -> RetrievedContext {
    let RagContextParams {
        search_repo,
        config,
        notebook_id,
        query,
        max_chunks,
        embeddings,
        reranker,
        hyde_service,
        reformulator,
        history,
        events,
        embedding_cache,
        provider,
        model,
        preference_boost,
    } = params;
    let search_repo = *search_repo;
    let notebook_id = *notebook_id;
    let max_chunks = *max_chunks;
    if let Some(reformulator) = reformulator {
        let chat_turns: Vec<ChatTurn> = history
            .iter()
            .map(|msg| ChatTurn {
                role: msg.role.clone(),
                content: msg.content.clone(),
            })
            .collect();

        // --- Proactive query reformulation for follow-up questions ---
        // Reformulate before retrieval when the query appears non-standalone.
        // This is separate from corrective RAG (which reformulates *after*
        // low-quality retrieval) — it prevents the retrieval failure entirely.
        let effective_query: std::borrow::Cow<'_, str> =
            if !chat_turns.is_empty() && needs_proactive_reformulation(query) {
                let result = reformulator.reformulate(query, &chat_turns).await;
                if result.was_reformulated {
                    // Hashes, not text: a retrieval log must never carry the
                    // question a user typed (US-004).
                    tracing::info!(
                        %notebook_id,
                        query_hash = %query_hash(query),
                        reformulated_query_hash = %query_hash(&result.query),
                        "Proactive query reformulation for follow-up"
                    );
                    events
                        .emit(ChatEvent::thinking(ThinkingStage::ReformulatingQuery))
                        .await;
                    result.query.into()
                } else {
                    (*query).into()
                }
            } else {
                (*query).into()
            };

        match retrieve_context_corrective(&CorrectiveRetrievalParams {
            search_repo,
            config,
            notebook_id,
            query: &effective_query,
            max_chunks,
            embeddings: embeddings.as_deref(),
            reranker: reranker.as_deref(),
            hyde_service: *hyde_service,
            reformulator: Some(reformulator),
            chat_history: &chat_turns,
            embedding_cache: *embedding_cache,
            provider,
            model,
            preference_boost: preference_boost.as_ref(),
        })
        .await
        {
            Ok((corrective, timings)) => {
                tracing::info!(
                    %notebook_id,
                    chunks_retrieved = corrective.results.len(),
                    avg_relevance = corrective.avg_relevance,
                    was_corrected = corrective.was_corrected,
                    low_quality = corrective.low_quality_warning,
                    effective_query_hash = %query_hash(&corrective.effective_query),
                    "Corrective RAG retrieval"
                );

                if corrective.was_corrected {
                    events
                        .emit(ChatEvent::thinking(ThinkingStage::ReformulatingQuery))
                        .await;
                }

                if corrective.low_quality_warning {
                    events
                        .emit(ChatEvent::warning(WarningKind::LowRetrievalQuality))
                        .await;
                }

                let reformulated_query = (corrective.effective_query != *query)
                    .then(|| corrective.effective_query.clone());

                RetrievedContext {
                    chunks: corrective.results,
                    timings,
                    reformulated_query,
                }
            }
            Err(e) => {
                tracing::warn!(%notebook_id, error = %e, "Corrective RAG failed, proceeding without context");
                RetrievedContext::empty()
            }
        }
    } else {
        // No reformulator — use standard retrieval
        let retrieval_params = RetrievalParams {
            search_repo,
            config,
            notebook_id,
            query,
            max_chunks,
            embeddings: embeddings.as_deref(),
            reranker: reranker.as_deref(),
            hyde_service: *hyde_service,
            embedding_cache: *embedding_cache,
            provider,
            model,
            preference_boost: preference_boost.as_ref(),
        };
        match retrieve_context(&retrieval_params).await {
            Ok((chunks, timings)) => {
                tracing::info!(
                    %notebook_id,
                    chunks_retrieved = chunks.len(),
                    top_score = chunks.first().map(|c| c.relevance_score),
                    "Retrieved context chunks for RAG"
                );
                RetrievedContext {
                    chunks,
                    timings,
                    reformulated_query: None,
                }
            }
            Err(e) => {
                tracing::warn!(%notebook_id, error = %e, "Failed to retrieve context, proceeding without");
                RetrievedContext::empty()
            }
        }
    }
}

/// Load notebook memory for prompt injection.
///
/// Returns a formatted `<memory>` XML block with Core Memory (always-on user
/// profile) and Working Memory (query-relevant facts). Returns `None` if
/// memory is disabled, no memories exist, or embedding fails.
///
/// Uses the embedding cache to avoid redundant provider calls — the query
/// embedding is typically already cached by the RAG pipeline.
pub(crate) async fn load_memory_for_prompt(
    memory_enabled: bool,
    notebook_id: Uuid,
    query: &str,
    memory_repo: &dyn MemoryRepository,
    embeddings: Option<&dyn crate::core::providers::EmbeddingProvider>,
    embedding_cache: &crate::services::rag::embedding_cache::EmbeddingCache,
) -> (Option<String>, Vec<crate::entities::notebook_memory::Model>) {
    if !memory_enabled {
        return (None, vec![]);
    }

    // Load all memories for this notebook
    let all_memories = match memory_repo.list_for_notebook(notebook_id).await {
        Ok(memories) => memories,
        Err(e) => {
            tracing::warn!(%notebook_id, error = %e, "Failed to load memories for prompt");
            return (None, vec![]);
        }
    };

    if all_memories.is_empty() {
        return (None, vec![]);
    }

    // Select core memories (expertise/goal/preference with salience > 0.5)
    let core = select_core_memories(&all_memories);

    // Get query embedding for working memory search (try cache first, then Voyage)
    let working = match get_query_embedding(query, embeddings, embedding_cache).await {
        Some(embedding) => memory_repo
            .search_similar(notebook_id, &embedding, 5)
            .await
            .unwrap_or_default(),
        None => vec![],
    };

    // Touch referenced working memories to prevent temporal decay
    for r in &working {
        if r.similarity >= 0.6
            && r.memory.memory_type != "conversation_summary"
            && let Err(e) = memory_repo.touch_memory(r.memory.id).await
        {
            tracing::warn!(memory_id = %r.memory.id, error = %e, "Failed to touch working memory");
        }
    }

    let result = format_memory_for_prompt(&core, &working);

    if result.is_some() {
        tracing::debug!(
            %notebook_id,
            core_count = core.len(),
            working_count = working.len(),
            "Injecting memory into system prompt"
        );
    }

    (result, all_memories)
}

/// Get the query embedding, trying the cache first to avoid redundant API calls.
pub(crate) async fn get_query_embedding(
    query: &str,
    embeddings: Option<&dyn crate::core::providers::EmbeddingProvider>,
    cache: &crate::services::rag::embedding_cache::EmbeddingCache,
) -> Option<Vec<f32>> {
    // Try cache first — the RAG pipeline likely already embedded this query
    if let Some(cached) = cache.get(query).await {
        return Some(cached);
    }

    // Fall back to the configured provider
    let embeddings = embeddings?;
    match embeddings.embed_query(query).await {
        Ok(embedding) => Some(embedding),
        Err(e) => {
            tracing::warn!(error = %e, "Failed to embed query for working memory search");
            None
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::notebook_memory;
    use crate::repositories::{MemorySearchResult, RepoResult};
    use async_trait::async_trait;
    use chrono::{DateTime, FixedOffset, Utc};
    use std::collections::HashMap;

    fn empty_metrics() -> crate::services::rag::rag_log::AggregatedMetrics {
        crate::services::rag::rag_log::AggregatedMetrics {
            total_interactions: 0,
            successful_retrievals: 0,
            avg_context_relevance: None,
            avg_answer_faithfulness: None,
            positive_feedback: 0,
            negative_feedback: 0,
        }
    }

    // ── Mock RagLogRepository ──────────────────────────────────────────

    struct MockRagLogRepo {
        preferred_source_ids: Vec<Uuid>,
    }

    impl MockRagLogRepo {
        fn empty() -> Self {
            Self {
                preferred_source_ids: vec![],
            }
        }

        fn with_preferred(ids: Vec<Uuid>) -> Self {
            Self {
                preferred_source_ids: ids,
            }
        }
    }

    #[async_trait]
    impl RagLogRepository for MockRagLogRepo {
        async fn create(
            &self,
            _entry: &crate::services::rag::rag_log::RagLogEntry,
        ) -> RepoResult<Uuid> {
            Ok(Uuid::new_v4())
        }
        async fn get_by_id(
            &self,
            _id: Uuid,
        ) -> RepoResult<Option<crate::entities::rag_log::Model>> {
            Ok(None)
        }
        async fn get_for_user(
            &self,
            _id: Uuid,
            _user_id: Uuid,
        ) -> RepoResult<crate::entities::rag_log::Model> {
            Err(AppError::NotFound("not found".into()))
        }
        async fn update_metrics(
            &self,
            _log_id: Uuid,
            _metrics: &crate::services::rag::rag_log::RagMetrics,
        ) -> RepoResult<()> {
            Ok(())
        }
        async fn update_feedback(
            &self,
            _log_id: Uuid,
            _user_id: Uuid,
            _feedback: &str,
        ) -> RepoResult<()> {
            Ok(())
        }
        async fn get_notebook_metrics(
            &self,
            _notebook_id: Uuid,
            _days: i32,
        ) -> RepoResult<crate::services::rag::rag_log::AggregatedMetrics> {
            Ok(empty_metrics())
        }
        async fn get_user_metrics(
            &self,
            _user_id: Uuid,
            _days: i32,
        ) -> RepoResult<crate::services::rag::rag_log::AggregatedMetrics> {
            Ok(empty_metrics())
        }
        async fn get_rag_log_ids_for_messages(
            &self,
            _message_ids: &[Uuid],
        ) -> RepoResult<HashMap<Uuid, (Uuid, Option<String>)>> {
            Ok(HashMap::new())
        }
        async fn get_by_response_id(
            &self,
            _response_id: Uuid,
        ) -> RepoResult<Option<crate::entities::rag_log::Model>> {
            Ok(None)
        }
        async fn purge_old_logs(&self, _retention_days: i32) -> RepoResult<u64> {
            Ok(0)
        }
        async fn get_preferred_source_ids(&self, _notebook_id: Uuid) -> RepoResult<Vec<Uuid>> {
            Ok(self.preferred_source_ids.clone())
        }
    }

    // ── Mock MemoryRepository ──────────────────────────────────────────

    struct MockMemoryRepo {
        memories: Vec<notebook_memory::Model>,
    }

    impl MockMemoryRepo {
        fn empty() -> Self {
            Self { memories: vec![] }
        }

        fn with_memories(memories: Vec<notebook_memory::Model>) -> Self {
            Self { memories }
        }
    }

    fn make_memory(memory_type: &str, content: &str, salience: f32) -> notebook_memory::Model {
        let now: DateTime<FixedOffset> = Utc::now().fixed_offset();
        notebook_memory::Model {
            id: Uuid::new_v4(),
            notebook_id: Uuid::new_v4(),
            content: content.to_owned(),
            memory_type: memory_type.to_owned(),
            metadata: serde_json::json!({}),
            salience,
            created_at: now,
            updated_at: now,
        }
    }

    #[async_trait]
    impl MemoryRepository for MockMemoryRepo {
        async fn create_with_embedding(
            &self,
            _notebook_id: Uuid,
            _content: &str,
            _memory_type: &str,
            _metadata: serde_json::Value,
            _salience: f32,
            _embedding: &[f32],
        ) -> RepoResult<notebook_memory::Model> {
            Err(AppError::NotFound("not implemented".into()))
        }
        async fn list_for_notebook(
            &self,
            _notebook_id: Uuid,
        ) -> RepoResult<Vec<notebook_memory::Model>> {
            Ok(self.memories.clone())
        }
        async fn search_similar(
            &self,
            _notebook_id: Uuid,
            _query_embedding: &[f32],
            _limit: i32,
        ) -> RepoResult<Vec<MemorySearchResult>> {
            Ok(vec![])
        }
        async fn get_by_id(&self, _memory_id: Uuid) -> RepoResult<Option<notebook_memory::Model>> {
            Ok(None)
        }
        async fn update(
            &self,
            _memory_id: Uuid,
            _content: Option<String>,
            _salience: Option<f32>,
            _metadata: Option<serde_json::Value>,
            _embedding: Option<&[f32]>,
        ) -> RepoResult<notebook_memory::Model> {
            Err(AppError::NotFound("not implemented".into()))
        }
        async fn delete(&self, _memory_id: Uuid) -> RepoResult<()> {
            Ok(())
        }
        async fn delete_all_for_notebook(&self, _notebook_id: Uuid) -> RepoResult<u64> {
            Ok(0)
        }
        async fn count_for_notebook(&self, _notebook_id: Uuid) -> RepoResult<u64> {
            Ok(0)
        }
        async fn count_by_type(&self, _notebook_id: Uuid, _memory_type: &str) -> RepoResult<u64> {
            Ok(0)
        }
        async fn delete_oldest_by_type(
            &self,
            _notebook_id: Uuid,
            _memory_type: &str,
            _keep: u64,
        ) -> RepoResult<u64> {
            Ok(0)
        }
        async fn touch_memory(&self, _memory_id: Uuid) -> RepoResult<()> {
            Ok(())
        }
        async fn update_salience(&self, _memory_id: Uuid, _new_salience: f32) -> RepoResult<()> {
            Ok(())
        }
        async fn delete_below_salience(
            &self,
            _notebook_id: Uuid,
            _threshold: f32,
        ) -> RepoResult<u64> {
            Ok(0)
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // should_reuse_session
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn should_reuse_session_within_window() {
        let sid = Uuid::new_v4();
        let elapsed = chrono::Duration::seconds(60);
        assert_eq!(should_reuse_session(elapsed, Some(sid)), Some(sid));
    }

    #[test]
    fn should_reuse_session_outside_window() {
        let sid = Uuid::new_v4();
        let elapsed = chrono::Duration::seconds(SESSION_TIMEOUT_SECS + 1);
        assert_eq!(should_reuse_session(elapsed, Some(sid)), None);
    }

    #[test]
    fn should_reuse_session_no_previous_session() {
        let elapsed = chrono::Duration::seconds(60);
        assert_eq!(should_reuse_session(elapsed, None), None);
    }

    #[test]
    fn should_reuse_session_zero_duration() {
        let sid = Uuid::new_v4();
        let elapsed = chrono::Duration::seconds(0);
        assert_eq!(should_reuse_session(elapsed, Some(sid)), Some(sid));
    }

    #[test]
    fn should_reuse_session_exactly_at_boundary() {
        let sid = Uuid::new_v4();
        let elapsed = chrono::Duration::seconds(SESSION_TIMEOUT_SECS);
        assert_eq!(should_reuse_session(elapsed, Some(sid)), None);
    }

    #[test]
    fn should_reuse_session_near_boundary() {
        let sid = Uuid::new_v4();
        let elapsed = chrono::Duration::seconds(29 * 60); // 29 minutes — just inside
        assert_eq!(should_reuse_session(elapsed, Some(sid)), Some(sid));
    }

    #[test]
    fn should_reuse_session_long_gap() {
        let sid = Uuid::new_v4();
        let elapsed = chrono::Duration::seconds(2 * 60 * 60); // 2 hours
        assert_eq!(should_reuse_session(elapsed, Some(sid)), None);
    }

    #[test]
    fn should_reuse_session_zero_elapsed_no_session() {
        // No session_id even with zero elapsed → None (caller generates new)
        assert_eq!(
            should_reuse_session(chrono::Duration::seconds(0), None),
            None
        );
    }

    #[test]
    fn should_reuse_session_negative_elapsed_clock_skew() {
        // Negative elapsed from DB timestamp ahead of app clock → reuse
        let sid = Uuid::new_v4();
        let elapsed = chrono::Duration::seconds(-5);
        assert_eq!(should_reuse_session(elapsed, Some(sid)), Some(sid));
    }

    // ════════════════════════════════════════════════════════════════════
    // needs_proactive_reformulation
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn needs_reformulation_short_query() {
        assert!(needs_proactive_reformulation("What about that?"));
    }

    #[test]
    fn needs_reformulation_referential_english() {
        let query = "Can you explain this concept in more detail and provide additional examples for better understanding?";
        assert!(needs_proactive_reformulation(query));
    }

    #[test]
    fn needs_reformulation_referential_french() {
        let query = "Pouvez-vous expliquer cette notion en détail et fournir des exemples supplémentaires pour mieux comprendre?";
        assert!(needs_proactive_reformulation(query));
    }

    #[test]
    fn needs_reformulation_long_standalone_query() {
        let query = "Explain the differences between supervised and unsupervised machine learning algorithms in the context of natural language processing applications";
        assert!(!needs_proactive_reformulation(query));
    }

    #[test]
    fn needs_reformulation_empty_query() {
        assert!(needs_proactive_reformulation(""));
    }

    // ════════════════════════════════════════════════════════════════════
    // build_preference_boost
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn preference_boost_empty_returns_none() {
        let rag_repo = MockRagLogRepo::empty();
        let mem_repo = MockMemoryRepo::empty();
        let result = build_preference_boost(&rag_repo, &mem_repo, Uuid::new_v4(), true).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn preference_boost_source_ids_only() {
        let source_id = Uuid::new_v4();
        let rag_repo = MockRagLogRepo::with_preferred(vec![source_id]);
        let mem_repo = MockMemoryRepo::empty();
        let result = build_preference_boost(&rag_repo, &mem_repo, Uuid::new_v4(), true).await;
        let boost = result.expect("should have boost");
        assert!(boost.preferred_source_ids.contains(&source_id));
        assert!(boost.preference_keywords.is_empty());
    }

    #[tokio::test]
    async fn preference_boost_keywords_only() {
        let rag_repo = MockRagLogRepo::empty();
        let mem = make_memory(
            "preference",
            "I prefer technical deep-dive explanations",
            0.8,
        );
        let mem_repo = MockMemoryRepo::with_memories(vec![mem]);
        let result = build_preference_boost(&rag_repo, &mem_repo, Uuid::new_v4(), true).await;
        let boost = result.expect("should have boost");
        assert!(boost.preferred_source_ids.is_empty());
        assert!(!boost.preference_keywords.is_empty());
    }

    #[tokio::test]
    async fn preference_boost_combined() {
        let source_id = Uuid::new_v4();
        let rag_repo = MockRagLogRepo::with_preferred(vec![source_id]);
        let mem = make_memory(
            "preference",
            "I prefer detailed technical explanations",
            0.8,
        );
        let mem_repo = MockMemoryRepo::with_memories(vec![mem]);
        let result = build_preference_boost(&rag_repo, &mem_repo, Uuid::new_v4(), true).await;
        let boost = result.expect("should have boost");
        assert!(boost.preferred_source_ids.contains(&source_id));
        assert!(!boost.preference_keywords.is_empty());
    }

    #[tokio::test]
    async fn preference_boost_memory_disabled() {
        let source_id = Uuid::new_v4();
        let rag_repo = MockRagLogRepo::with_preferred(vec![source_id]);
        let mem = make_memory("preference", "I prefer technical explanations", 0.8);
        let mem_repo = MockMemoryRepo::with_memories(vec![mem]);
        let result = build_preference_boost(&rag_repo, &mem_repo, Uuid::new_v4(), false).await;
        let boost = result.expect("should have boost from source IDs");
        assert!(boost.preferred_source_ids.contains(&source_id));
        assert!(boost.preference_keywords.is_empty());
    }

    // ════════════════════════════════════════════════════════════════════
    // load_memory_for_prompt
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn load_memory_disabled_returns_none() {
        let mem_repo = MockMemoryRepo::empty();
        let cache = crate::services::rag::embedding_cache::EmbeddingCache::new();
        let (block, mems) =
            load_memory_for_prompt(false, Uuid::new_v4(), "test", &mem_repo, None, &cache).await;
        assert!(block.is_none());
        assert!(mems.is_empty());
    }

    #[tokio::test]
    async fn load_memory_no_memories_returns_none() {
        let mem_repo = MockMemoryRepo::empty();
        let cache = crate::services::rag::embedding_cache::EmbeddingCache::new();
        let (block, mems) =
            load_memory_for_prompt(true, Uuid::new_v4(), "test", &mem_repo, None, &cache).await;
        assert!(block.is_none());
        assert!(mems.is_empty());
    }

    #[tokio::test]
    async fn load_memory_with_core_memories() {
        let mem = make_memory("expertise", "User is an expert in Rust", 0.9);
        let mem_repo = MockMemoryRepo::with_memories(vec![mem]);
        let cache = crate::services::rag::embedding_cache::EmbeddingCache::new();
        let (block, mems) =
            load_memory_for_prompt(true, Uuid::new_v4(), "test", &mem_repo, None, &cache).await;
        assert!(block.is_some());
        assert!(block.unwrap().contains("<memory>"));
        assert_eq!(mems.len(), 1);
    }

    #[tokio::test]
    async fn load_memory_voyage_none_skips_working_memory() {
        let mem = make_memory("expertise", "User knows machine learning", 0.9);
        let mem_repo = MockMemoryRepo::with_memories(vec![mem]);
        let cache = crate::services::rag::embedding_cache::EmbeddingCache::new();
        let (block, _) =
            load_memory_for_prompt(true, Uuid::new_v4(), "test", &mem_repo, None, &cache).await;
        assert!(block.is_some());
    }

    // ════════════════════════════════════════════════════════════════════
    // build_system_prompt_for_chat
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn system_prompt_without_memory() {
        let prompt =
            build_system_prompt_for_chat("<sources>doc</sources>", TeachingMode::Deep, "en", None);
        assert!(!prompt.contains("<memory>"));
        assert!(!prompt.is_empty());
    }

    #[test]
    fn system_prompt_with_memory() {
        let memory = "<memory>\n<core>User is a Rust developer</core>\n</memory>";
        let prompt = build_system_prompt_for_chat(
            "<sources>doc</sources>",
            TeachingMode::Deep,
            "en",
            Some(memory),
        );
        assert!(prompt.contains("<memory>"));
        assert!(prompt.contains("User is a Rust developer"));
    }

    #[test]
    fn system_prompt_quiz_mode() {
        let prompt =
            build_system_prompt_for_chat("<sources>doc</sources>", TeachingMode::Quiz, "en", None);
        let deep_prompt =
            build_system_prompt_for_chat("<sources>doc</sources>", TeachingMode::Deep, "en", None);
        assert_ne!(prompt, deep_prompt);
    }

    // ════════════════════════════════════════════════════════════════════
    // get_query_embedding
    // ════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn embedding_cache_hit() {
        let cache = crate::services::rag::embedding_cache::EmbeddingCache::new();
        let embedding = vec![0.1_f32, 0.2, 0.3];
        cache.insert("test query", embedding.clone()).await;
        let result = get_query_embedding("test query", None, &cache).await;
        assert_eq!(result, Some(embedding));
    }

    #[tokio::test]
    async fn embedding_cache_miss_no_voyage() {
        let cache = crate::services::rag::embedding_cache::EmbeddingCache::new();
        let result = get_query_embedding("test query", None, &cache).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn embedding_cache_miss_different_query() {
        let cache = crate::services::rag::embedding_cache::EmbeddingCache::new();
        // Cache a different query — lookup for "other query" should miss
        cache.insert("cached query", vec![1.0, 2.0]).await;
        let result = get_query_embedding("other query", None, &cache).await;
        assert!(result.is_none());
    }
}
