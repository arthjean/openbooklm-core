//! Chat API endpoints with SSE streaming.
//!
//! Features:
//! - POST streaming with CancellationToken for proper client disconnect handling
//! - Rate limiting per user plan
//! - Paginated chat history

pub(crate) mod sse_helpers;
pub(crate) mod streaming;
pub mod types;

use std::convert::Infallible;
use std::time::Duration;

use axum::{
    Json,
    extract::{Extension, Path, Query, State},
    response::{IntoResponse, Response, Sse, sse},
};
use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::api::common::verify_notebook_access;
use crate::core::CoreState;
use crate::core::events::DomainEvent;
use crate::core::principal::Principal;
use crate::core::protocol::{ChatEvent, ChatEventStream, ThinkingStage};
use crate::error::AppError;
use crate::llm::{TeachingMode, allocate_token_budget, build_messages, fit_prompt_to_budget};
use crate::services::chat::{
    ChatMessageResponse, clear_chat_history, get_chat_history_paginated,
    orchestration::{
        RagContextParams, build_preference_boost, build_system_prompt_for_chat, load_chat_history,
        load_memory_for_prompt, notify_history_truncation, retrieve_rag_context,
        validate_and_authorize,
    },
};
use crate::services::memory::{MIN_DROPPED_FOR_SUMMARY, load_conversation_summaries};
use crate::services::rag::rag_log::get_rag_log_ids_for_messages;
use crate::services::rag::search::{build_rag_documents, format_context_for_llm};

use sse_helpers::{SSE_KEEPALIVE_SECS, apply_sse_headers, chat_event_to_sse};
use streaming::{CancellableStream, StreamContext, stream_llm_response};
use types::{
    ChatHistoryQuery, ChatHistoryResponse, MAX_HISTORY_FETCH, SendMessageRequest,
    TeachingModesResponse,
};

// ============================================================================
// Handlers
// ============================================================================

/// POST /api/notebooks/{id}/chat - Send a message and get streaming response.
#[utoipa::path(
    post,
    path = "/api/notebooks/{id}/chat",
    tag = "chat",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID")),
    request_body = SendMessageRequest,
    responses(
        (status = 200, description = "Server-sent event stream. Ordering, terminal events and cancellation are specified in docs/contracts/sse-protocol-v1.md, which OpenAPI does not model.", content_type = "text/event-stream", body = crate::core::protocol::ChatEvent),
        (status = 400, description = "Validation failed", body = crate::error::ProblemDetails),
        (status = 403, description = "Denied by the entitlement policy", body = crate::error::ProblemDetails),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn send_message_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    Json(payload): Json<SendMessageRequest>,
) -> Result<Response, AppError> {
    // 1. Validate input, check access + rate limits, save user message, resolve provider
    let req = validate_and_authorize(&state, &principal, notebook_id, &payload).await?;

    // 2. Setup the typed event channel + cancellation. The handler is the only
    //    layer below this point that knows the transport is SSE.
    let (out, rx) = ChatEventStream::channel();
    let cancel_token = CancellationToken::new();

    // 3. Load chat history (truncation deferred to fit_prompt_to_budget)
    let hist = load_chat_history(state.repos.chat.as_ref(), notebook_id, MAX_HISTORY_FETCH).await?;

    // 4. Build preference boost signals + Retrieve RAG context
    let preference_boost = build_preference_boost(
        state.repos.rag_logs.as_ref(),
        state.repos.memory.as_ref(),
        notebook_id,
        req.memory_enabled,
    )
    .await;
    out.emit(ChatEvent::thinking(ThinkingStage::RetrievingContext))
        .await;
    let retrieved = retrieve_rag_context(&RagContextParams {
        search_repo: state.repos.search.as_ref(),
        config: &state.config,
        notebook_id,
        query: &req.message,
        max_chunks: payload.clamped_max_context_chunks(),
        embeddings: &state.clients.embeddings,
        reranker: &state.clients.reranker,
        hyde_service: state.clients.hyde_service.as_ref(),
        reformulator: state.clients.query_reformulator.as_ref(),
        history: &hist.raw,
        events: &out,
        embedding_cache: Some(&state.embedding_cache),
        provider: req.provider.name(),
        model: &req.model_id,
        preference_boost,
    })
    .await;
    let context_chunks = retrieved.chunks;
    let rag_timings = retrieved.timings;
    let reformulated_query = retrieved.reformulated_query;

    // 5. Build system prompt
    out.emit(ChatEvent::thinking(ThinkingStage::Generating))
        .await;
    let locale = payload.locale.as_deref().unwrap_or("en");

    // 5a. Native citations (Anthropic Citations API).
    let use_native_citations = req.provider.supports_native_citations();
    let rag_documents = if use_native_citations {
        build_rag_documents(&context_chunks)
    } else {
        Vec::new()
    };

    let context = if use_native_citations {
        String::new()
    } else {
        format_context_for_llm(&context_chunks)
    };

    // 5b. Load notebook memory (core + working) for prompt injection
    let (memory_block, all_memories) = load_memory_for_prompt(
        req.memory_enabled,
        notebook_id,
        &req.message,
        state.repos.memory.as_ref(),
        state.clients.embeddings.as_deref(),
        &state.embedding_cache,
    )
    .await;

    let system_prompt = build_system_prompt_for_chat(
        &context,
        payload.teaching_mode,
        locale,
        memory_block.as_deref(),
    );

    // 6. Fit all segments within the unified token budget
    let budget = allocate_token_budget(req.provider.name(), &req.model_id);
    let fitted = fit_prompt_to_budget(
        &system_prompt,
        &context,
        memory_block.as_deref(),
        &hist.all_messages,
        &budget,
    );
    let was_truncated = fitted.was_truncated;
    let kept_count = fitted.messages.len();
    let dropped_messages = fitted.dropped_messages;
    let dropped_count = dropped_messages.len();

    // 6b. Notify about history truncation + summarization
    notify_history_truncation(
        &out,
        was_truncated,
        hist.all_messages.len(),
        kept_count,
        notebook_id,
        req.provider.name(),
    )
    .await;
    if dropped_count > MIN_DROPPED_FOR_SUMMARY
        && state.clients.mistral.is_some()
        && req.memory_enabled
    {
        out.emit(ChatEvent::history_summarized(dropped_count)).await;
    }

    // 6c. Inject conversation summaries at the top of history
    let mut final_messages = Vec::with_capacity(fitted.messages.len() + 4);
    if req.memory_enabled {
        let summaries = load_conversation_summaries(&all_memories);
        if !summaries.is_empty() {
            tracing::debug!(
                %notebook_id,
                summary_count = summaries.len(),
                "Injecting conversation summaries into history"
            );
            final_messages.extend(summaries);
        }
    }
    final_messages.extend(fitted.messages);

    let messages = build_messages(&final_messages, &req.message);

    // 7. Spawn streaming task and return SSE response
    let stream_ctx = StreamContext {
        provider: req.provider,
        rag_log_repo: state.repos.rag_logs.clone(),
        chat_repo: state.repos.chat.clone(),
        entitlements: state.entitlements.clone(),
        permit: req.permit,
        events: state.events.clone(),
        notebook_id,
        system_prompt,
        messages,
        context_chunks,
        llm_timeout: Duration::from_secs(state.config.async_config.llm_timeout_secs),
        model: req.model_for_stream,
        shutdown_token: state.task_tracker.cancellation_token(),
        principal: principal.clone(),
        teaching_mode: payload.teaching_mode,
        mistral: state.clients.mistral.clone(),
        user_question: req.message,
        locale: payload.locale.as_deref().unwrap_or("en").to_owned(),
        rag_timings,
        reformulated_query,
        memory_enabled: req.memory_enabled,
        embeddings: state.clients.embeddings.clone(),
        memory_repo: state.repos.memory.clone(),
        dropped_messages,
        memory_decay_tracker: state.memory_decay_tracker.clone(),
        session_id: req.session_id,
        source_repo: state.repos.sources.clone(),
        conversation_turn: hist.all_messages.len() + 1,
        rag_documents,
    };
    Ok(spawn_sse_stream(stream_ctx, out, rx, cancel_token, &state))
}

/// GET /api/notebooks/{id}/chat - Get paginated chat history.
#[utoipa::path(
    get,
    path = "/api/notebooks/{id}/chat",
    tag = "chat",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID"), ChatHistoryQuery),
    responses(
        (status = 200, description = "A page of chat history", body = ChatHistoryResponse),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn get_chat_history_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    Query(query): Query<ChatHistoryQuery>,
) -> Result<Json<ChatHistoryResponse>, AppError> {
    verify_notebook_access(state.repos.notebooks.as_ref(), &principal, notebook_id).await?;

    let (offset, limit) = query.clamped();
    let paginated = get_chat_history_paginated(
        state.repos.chat.as_ref(),
        notebook_id,
        Some(offset),
        Some(limit),
    )
    .await?;

    let message_ids: Vec<Uuid> = paginated.messages.iter().map(|m| m.id).collect();
    let rag_log_map =
        get_rag_log_ids_for_messages(state.repos.rag_logs.as_ref(), &message_ids).await?;

    let messages: Vec<ChatMessageResponse> = paginated
        .messages
        .into_iter()
        .map(|m| {
            let (rag_log_id, feedback) = rag_log_map
                .get(&m.id)
                .map(|(id, fb)| (Some(*id), fb.clone()))
                .unwrap_or((None, None));
            let mut resp = ChatMessageResponse::from(m);
            resp.rag_log_id = rag_log_id;
            resp.feedback = feedback;
            resp
        })
        .collect();

    Ok(Json(ChatHistoryResponse {
        messages,
        total: paginated.total,
        offset: paginated.offset,
        limit: paginated.limit,
        has_more: paginated.has_more,
    }))
}

/// DELETE /api/notebooks/{id}/chat - Clear chat history.
#[utoipa::path(
    delete,
    path = "/api/notebooks/{id}/chat",
    tag = "chat",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "Deletion acknowledgement", body = serde_json::Value),
        (status = 404, description = "Not found, or owned by another account", body = crate::error::ProblemDetails),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn clear_chat_history_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, AppError> {
    verify_notebook_access(state.repos.notebooks.as_ref(), &principal, notebook_id).await?;

    let deleted = clear_chat_history(state.repos.chat.as_ref(), notebook_id).await?;

    Ok(Json(serde_json::json!({
        "success": true,
        "deleted_count": deleted
    })))
}

// ============================================================================
// Teaching modes endpoint
// ============================================================================

/// GET /api/teaching-modes - List available teaching modes.
#[utoipa::path(
    get,
    path = "/api/teaching-modes",
    tag = "chat",
    responses(
        (status = 200, description = "Available teaching modes and the default", body = TeachingModesResponse),
        (status = 401, description = "Missing or invalid credentials", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn list_teaching_modes() -> Json<TeachingModesResponse> {
    Json(TeachingModesResponse {
        modes: vec![
            TeachingMode::Flash.into(),
            TeachingMode::Deep.into(),
            TeachingMode::Quiz.into(),
            TeachingMode::Glossary.into(),
            TeachingMode::Summary.into(),
            TeachingMode::Timeline.into(),
        ],
        default: "deep",
    })
}

// ============================================================================
// SSE stream spawning (handler-only concern — depends on Axum response types)
// ============================================================================

/// Spawn the LLM streaming task and frame its typed events as an SSE response.
///
/// This is the transport boundary: `stream_llm_response` produces
/// [`ChatEvent`] values, and everything below converts them to SSE framing,
/// heartbeats and headers.
fn spawn_sse_stream(
    stream_ctx: StreamContext,
    out: ChatEventStream,
    rx: mpsc::Receiver<ChatEvent>,
    cancel_token: CancellationToken,
    state: &CoreState,
) -> Response {
    let handle = {
        let cancel_token = cancel_token.clone();
        // Captured before the context moves into the task: every failure path
        // out of `stream_llm_response` reports one typed ChatFailed event.
        let events = state.events.clone();
        let account_id = stream_ctx.principal.account_id;
        let failed_notebook_id = stream_ctx.notebook_id;
        let provider_name = stream_ctx.provider.name().to_owned();

        tokio::spawn(async move {
            if let Err(e) = stream_llm_response(stream_ctx, &out, cancel_token).await {
                events.emit(DomainEvent::ChatFailed {
                    account_id,
                    notebook_id: failed_notebook_id,
                    provider: provider_name,
                    error_type: streaming::chat_error_type(&e),
                });
                // The one terminal event for every failure path. `ChatEventStream`
                // drops it if the stream already terminated.
                out.emit(ChatEvent::error(e.to_string())).await;
            }
        })
    };

    let frames =
        ReceiverStream::new(rx).map(|event| Ok::<_, Infallible>(chat_event_to_sse(&event)));
    let stream = CancellableStream::new(frames, cancel_token, handle, state.task_tracker.clone());
    let sse = Sse::new(stream).keep_alive(
        sse::KeepAlive::new()
            .interval(Duration::from_secs(SSE_KEEPALIVE_SECS))
            .text("keep-alive"),
    );

    let mut response = sse.into_response();
    apply_sse_headers(&mut response);
    response
}
