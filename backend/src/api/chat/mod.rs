//! Chat API endpoints with SSE streaming.
//!
//! Features:
//! - POST streaming with CancellationToken for proper client disconnect handling
//! - Rate limiting per user plan
//! - Paginated chat history

mod citation_resolution;
pub(crate) mod fallback;
pub(crate) mod sse_helpers;
pub(crate) mod streaming;
pub(crate) mod turn;
pub mod types;

use std::convert::Infallible;
use std::time::Duration;

use axum::{
    Json,
    extract::{Extension, Path, Query, State},
    response::{IntoResponse, Response, Sse, sse},
};
use futures::StreamExt;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::{CancellationToken, DropGuard};
use uuid::Uuid;

use crate::api::common::verify_notebook_access;
use crate::core::CoreState;
use crate::core::events::DomainEvent;
use crate::core::principal::Principal;
use crate::core::protocol::{ChatEvent, ChatEventStream};
use crate::error::AppError;
use crate::llm::TeachingMode;
use crate::services::chat::{ChatMessageResponse, clear_chat_history, get_chat_history_paginated};
use crate::services::rag::rag_log::get_rag_log_ids_for_messages;
use crate::types::RequestContext;

use sse_helpers::{SSE_KEEPALIVE_SECS, apply_sse_headers, chat_event_to_sse};
use streaming::{CancellableStream, StreamContext, stream_llm_response};
use turn::TurnOutcome;
use types::{ChatHistoryQuery, ChatHistoryResponse, SendMessageRequest, TeachingModesResponse};

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
        (status = 503, description = "Server shutdown has closed task admission", body = crate::error::ProblemDetails),
    ),
    security(("bearer_auth" = [])),
)]
pub async fn send_message_handler(
    State(state): State<CoreState>,
    Extension(principal): Extension<Principal>,
    Path(notebook_id): Path<Uuid>,
    Json(payload): Json<SendMessageRequest>,
) -> Result<Response, AppError> {
    // Materialize root ownership before `prepare` can persist the user message.
    // Forced shutdown can then abort either the complete turn or nothing.
    let task_admission = state.task_tracker.try_admit().map_err(|_| {
        AppError::ServiceUnavailable("Server is shutting down; retry this request".to_owned())
    })?;

    // The typed event channel and its cancellation token. This handler is the
    // only layer below this point that knows the transport is SSE; the owned
    // turn speaks `ChatEvent`.
    let (out, rx) = ChatEventStream::channel();
    let cancel_token = CancellationToken::new();
    let cancel_guard = cancel_token.clone().drop_guard();
    let (ready_tx, ready_rx) = oneshot::channel();
    let task_state = state.clone();
    let task_cancel = cancel_token.clone();
    let request_context = RequestContext::current();
    task_admission
        .spawn(
            "chat-turn",
            request_context.scope(async move {
                let outcome =
                    turn::prepare(&task_state, &principal, notebook_id, &payload, &out).await;
                match outcome {
                    Ok(outcome) => {
                        if ready_tx.send(Ok(())).is_err() {
                            task_cancel.cancel();
                        }
                        run_prepared_turn(outcome, out, task_cancel, &task_state).await;
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            }),
        )
        .map_err(|_| shutdown_unavailable())?;
    ready_rx.await.map_err(|_| shutdown_unavailable())??;

    Ok(sse_response(rx, cancel_guard, &state))
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

/// DELETE /api/notebooks/{id}/chat - Clear chat history and its RAG traces.
#[utoipa::path(
    delete,
    path = "/api/notebooks/{id}/chat",
    tag = "chat",
    params(("id" = uuid::Uuid, Path, description = "Notebook ID")),
    responses(
        (status = 200, description = "Messages and associated RAG logs deleted", body = serde_json::Value),
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

enum ProviderFreeOutcome {
    Answer(fallback::FallbackContext),
    Failed(fallback::FailureContext),
}

async fn run_prepared_turn(
    outcome: TurnOutcome,
    out: ChatEventStream,
    client_cancel: CancellationToken,
    state: &CoreState,
) {
    match outcome {
        TurnOutcome::Stream(context) => {
            run_llm_stream(*context, out, client_cancel, state).await;
        }
        TurnOutcome::Answer(context) => {
            run_provider_free(
                ProviderFreeOutcome::Answer(context),
                out,
                client_cancel,
                state.task_tracker.cancellation_token(),
            )
            .await;
        }
        TurnOutcome::Failed(context) => {
            run_provider_free(
                ProviderFreeOutcome::Failed(context),
                out,
                client_cancel,
                state.task_tracker.cancellation_token(),
            )
            .await;
        }
    }
}

async fn run_provider_free(
    outcome: ProviderFreeOutcome,
    out: ChatEventStream,
    client_cancel: CancellationToken,
    shutdown: CancellationToken,
) {
    tokio::select! {
        biased;
        () = shutdown.cancelled() => {
            out.emit(ChatEvent::shutdown("Server shutting down")).await;
        }
        () = client_cancel.cancelled() => {}
        () = async {
            match outcome {
                ProviderFreeOutcome::Answer(context) => {
                    fallback::stream_grounded_fallback(context, &out).await;
                }
                ProviderFreeOutcome::Failed(context) => {
                    fallback::stream_turn_failure(context, &out).await;
                }
            }
        } => {}
    }
}

/// Run the LLM branch inside the already-owned chat task.
///
/// Every failure path out of `stream_llm_response` reports one typed
/// `ChatFailed` event and one terminal `error`.
async fn run_llm_stream(
    stream_ctx: StreamContext,
    out: ChatEventStream,
    cancel_token: CancellationToken,
    state: &CoreState,
) {
    // Captured before the context moves into the task.
    let events = state.events.clone();
    let account_id = stream_ctx.principal.account_id;
    let failed_notebook_id = stream_ctx.notebook_id;
    let provider_name = stream_ctx.provider.name().to_owned();

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
}

fn shutdown_unavailable() -> AppError {
    AppError::ServiceUnavailable("Server is shutting down; retry this request".to_owned())
}

/// Frame the owned task's typed events as the SSE response.
///
/// The transport boundary: everything above produces [`ChatEvent`] values, and
/// everything below converts them to SSE framing, heartbeats and headers.
fn sse_response(
    rx: mpsc::Receiver<ChatEvent>,
    cancel_guard: DropGuard,
    state: &CoreState,
) -> Response {
    let frames =
        ReceiverStream::new(rx).map(|event| Ok::<_, Infallible>(chat_event_to_sse(&event)));
    let stream = CancellableStream::new(frames, cancel_guard, state.task_tracker.clone());
    let sse = Sse::new(stream).keep_alive(
        sse::KeepAlive::new()
            .interval(Duration::from_secs(SSE_KEEPALIVE_SECS))
            .text("keep-alive"),
    );

    let mut response = sse.into_response();
    apply_sse_headers(&mut response);
    response
}
