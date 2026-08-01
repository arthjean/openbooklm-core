//! Turns that end without a provider call (US-018, US-020).
//!
//! Three of them exist, and the difference between them is the whole point:
//!
//! - The notebook has no source, or retrieval returned nothing relevant. Those
//!   are *answers*. The assistant returns the documented sentence, the turn ends
//!   with `done`, and the client needs no special case.
//! - Retrieval never ran, or the request could not be measured against the
//!   model's context window. Those are *failures*. Answering anyway would
//!   produce a fluent, confident, ungrounded reply indistinguishable from a
//!   grounded one (FR-17), so the turn terminates through the existing
//!   structured `error` event and nothing is generated.
//!
//! A configuration defect is not an abstention. Telling the user "the available
//! sources do not support a confident answer" when the real cause is an
//! undeclared context window is a false statement about their notebook, and it
//! would let a broken deployment report `ChatCompleted` for every turn.

use std::sync::Arc;

use uuid::Uuid;

use crate::core::events::{DomainEvent, SharedEventSink};
use crate::core::protocol::{ChatEvent, ChatEventStream};
use crate::repositories::ChatRepository;
use crate::services::chat::{CreateMessageParams, create_message};
use crate::services::rag::eval::trace::{ReasonCode, RetrievalTrace};

use super::streaming::chat_error_type;

/// A turn whose answer is a constant.
pub(super) struct FallbackContext {
    pub chat_repo: Arc<dyn ChatRepository>,
    pub events: SharedEventSink,
    pub account_id: Uuid,
    pub notebook_id: Uuid,
    pub session_id: Uuid,
    pub provider: String,
    pub model: String,
    /// The sentence to return. Never model output.
    pub answer: &'static str,
    /// The user's question, hashed into the trace and never logged.
    pub query: String,
    pub reason: ReasonCode,
}

/// A turn that produced no answer at all.
pub(super) struct FailureContext {
    pub events: SharedEventSink,
    pub account_id: Uuid,
    pub notebook_id: Uuid,
    pub provider: String,
    /// The user's question, hashed into the trace and never logged.
    pub query: String,
    pub reason: ReasonCode,
    /// Domain-event classification, for operators reading failure counters.
    pub error_type: &'static str,
    /// What the client is told. Never carries source text or configuration.
    pub message: &'static str,
}

/// Answer with the documented fallback text, without calling a provider.
///
/// Emits the same terminal event sequence a generated answer does — content,
/// citations, metrics, `done` — because the transport contract does not change
/// just because no model was involved (US-020 AC-4). The permit is deliberately
/// not recorded: nothing was generated, so nothing is charged.
pub(super) async fn stream_grounded_fallback(ctx: FallbackContext, out: &ChatEventStream) {
    emit_retrieval_trace(ctx.notebook_id, &ctx.query, ctx.reason);

    out.emit(ChatEvent::chunk(ctx.answer)).await;
    out.emit(ChatEvent::citations(Vec::new())).await;
    out.emit(ChatEvent::metrics(None)).await;

    if let Err(e) = create_message(CreateMessageParams {
        repo: ctx.chat_repo.as_ref(),
        notebook_id: ctx.notebook_id,
        role: "assistant",
        content: ctx.answer,
        citations: &[],
        model: Some(&ctx.model),
        agent_id: None,
        session_id: Some(ctx.session_id),
    })
    .await
    {
        tracing::warn!(
            notebook_id = %ctx.notebook_id,
            error_kind = chat_error_type(&e),
            "Could not persist the fallback answer; the client still received it"
        );
    }

    out.emit(ChatEvent::done(&ctx.model, &ctx.provider, None))
        .await;
    ctx.events.emit(DomainEvent::ChatCompleted {
        account_id: ctx.account_id,
        notebook_id: ctx.notebook_id,
        provider: ctx.provider,
        duration_ms: 0,
        context_chunks: 0,
    });
}

/// Terminate the turn through the existing structured error event.
///
/// The SSE contract is unchanged: `error` is terminal and no `done` follows it.
/// No assistant message is persisted, because there is no answer to persist.
pub(super) async fn stream_turn_failure(ctx: FailureContext, out: &ChatEventStream) {
    tracing::error!(
        notebook_id = %ctx.notebook_id,
        provider = %ctx.provider,
        reason = ctx.reason.as_str(),
        error_type = ctx.error_type,
        "Turn terminated without generating an answer"
    );

    emit_retrieval_trace(ctx.notebook_id, &ctx.query, ctx.reason);
    ctx.events.emit(DomainEvent::ChatFailed {
        account_id: ctx.account_id,
        notebook_id: ctx.notebook_id,
        provider: ctx.provider,
        error_type: ctx.error_type,
    });
    out.emit(ChatEvent::error(ctx.message)).await;
}

/// Emit a trace for a turn that produced no evidence.
///
/// The reason code is the whole point: "the notebook is empty", "nothing
/// relevant came back", "retrieval never ran" and "the request could not be
/// measured" produce the same absence of chunks and are four different
/// incidents (US-018 AC-5, US-020 AC-4).
fn emit_retrieval_trace(notebook_id: Uuid, query: &str, reason: ReasonCode) {
    let mut trace = RetrievalTrace::new(notebook_id, query, "chat", None);
    trace.reasons.insert(reason);
    trace.finish();
    trace.emit();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::events::NoopEventSink;
    use crate::repositories::{PaginatedChatHistory, RepoResult};

    /// A chat repository that records what was persisted.
    #[derive(Default)]
    struct RecordingChatRepo {
        saved: std::sync::Mutex<Vec<(String, String)>>,
    }

    #[async_trait::async_trait]
    impl ChatRepository for RecordingChatRepo {
        async fn create_message(
            &self,
            notebook_id: Uuid,
            role: &str,
            content: &str,
            citations: &[crate::llm::Citation],
            model: Option<&str>,
            _agent_id: Option<Uuid>,
            session_id: Option<Uuid>,
        ) -> RepoResult<crate::entities::chat_message::Model> {
            self.saved
                .lock()
                .expect("lock")
                .push((role.to_owned(), content.to_owned()));
            Ok(crate::entities::chat_message::Model {
                id: Uuid::new_v4(),
                notebook_id,
                role: role.to_owned(),
                content: content.to_owned(),
                citations: serde_json::to_value(citations).unwrap_or(serde_json::Value::Null),
                model: model.map(str::to_owned),
                session_id,
                created_at: chrono::Utc::now().fixed_offset(),
            })
        }

        async fn get_by_id(
            &self,
            _message_id: Uuid,
        ) -> RepoResult<Option<crate::entities::chat_message::Model>> {
            Ok(None)
        }

        async fn get_latest_message(
            &self,
            _notebook_id: Uuid,
        ) -> RepoResult<Option<crate::entities::chat_message::Model>> {
            Ok(None)
        }

        async fn get_conversation_up_to(
            &self,
            _notebook_id: Uuid,
            _up_to: chrono::DateTime<chrono::FixedOffset>,
        ) -> RepoResult<Vec<crate::entities::chat_message::Model>> {
            Ok(Vec::new())
        }

        async fn get_history(
            &self,
            _notebook_id: Uuid,
            _limit: Option<u64>,
        ) -> RepoResult<Vec<crate::entities::chat_message::Model>> {
            Ok(Vec::new())
        }

        async fn get_history_paginated(
            &self,
            _notebook_id: Uuid,
            _offset: Option<u64>,
            _limit: Option<u64>,
        ) -> RepoResult<PaginatedChatHistory> {
            Ok(PaginatedChatHistory {
                messages: Vec::new(),
                total: 0,
                offset: 0,
                limit: 0,
                has_more: false,
            })
        }

        async fn get_recent_history(
            &self,
            _notebook_id: Uuid,
            _max_messages: u64,
        ) -> RepoResult<Vec<crate::entities::chat_message::Model>> {
            Ok(Vec::new())
        }

        async fn clear_history(&self, _notebook_id: Uuid) -> RepoResult<u64> {
            Ok(0)
        }
    }

    fn drain(rx: &mut tokio::sync::mpsc::Receiver<ChatEvent>) -> Vec<String> {
        let mut rendered = Vec::new();
        while let Ok(event) = rx.try_recv() {
            rendered.push(serde_json::to_string(&event).expect("event serializes"));
        }
        rendered
    }

    /// The empty-notebook turn: the documented sentence, the same terminal
    /// event sequence, and no provider call (US-020 AC-4).
    #[tokio::test]
    async fn a_notebook_with_no_source_answers_with_the_documented_sentence() {
        let repo = Arc::new(RecordingChatRepo::default());
        let (out, mut rx) = ChatEventStream::channel();
        let answer = crate::llm::fallbacks::no_sources_text("fr");

        stream_grounded_fallback(
            FallbackContext {
                chat_repo: Arc::clone(&repo) as Arc<dyn ChatRepository>,
                events: Arc::new(NoopEventSink),
                account_id: Uuid::new_v4(),
                notebook_id: Uuid::new_v4(),
                session_id: Uuid::new_v4(),
                provider: "anthropic".to_owned(),
                model: "claude-sonnet-4-6-20260220".to_owned(),
                answer,
                query: "what is the retention window".to_owned(),
                reason: ReasonCode::EmptyCorpus,
            },
            &out,
        )
        .await;
        drop(out);

        let rendered = drain(&mut rx);
        assert!(rendered[0].contains(answer), "the answer is streamed first");
        assert!(
            rendered.last().expect("a terminal event").contains("done"),
            "the turn terminates with done, as a generated one does"
        );
        assert!(
            rendered.iter().any(|e| e.contains("citations")),
            "the citations event is still emitted, empty"
        );

        let saved = repo.saved.lock().expect("lock");
        assert_eq!(saved.len(), 1, "one assistant message was persisted");
        assert_eq!(saved[0], ("assistant".to_owned(), answer.to_owned()));
    }

    /// A turn that could not be measured is a failure, not an abstention: it
    /// ends with the terminal `error`, never with `done`, and persists nothing.
    #[tokio::test]
    async fn an_unmeasurable_turn_terminates_with_error_and_no_answer() {
        let (out, mut rx) = ChatEventStream::channel();

        stream_turn_failure(
            FailureContext {
                events: Arc::new(NoopEventSink),
                account_id: Uuid::new_v4(),
                notebook_id: Uuid::new_v4(),
                provider: "anthropic".to_owned(),
                query: "what is the retention window".to_owned(),
                reason: ReasonCode::ContextWindowUnknown,
                error_type: "context_window_unknown",
                message: super::super::turn::UNMEASURABLE_REQUEST_MESSAGE,
            },
            &out,
        )
        .await;
        drop(out);

        let rendered = drain(&mut rx);
        assert_eq!(rendered.len(), 1, "one terminal event and nothing else");
        assert!(rendered[0].contains("error"), "{}", rendered[0]);
        assert!(
            !rendered[0].contains("done"),
            "error is terminal; no done follows it"
        );
    }

    /// The failure message must not read like an abstention: a client and a
    /// grounded-response run both have to be able to tell them apart.
    #[tokio::test]
    async fn a_failure_message_is_not_one_of_the_documented_abstentions() {
        for message in [
            super::super::turn::UNMEASURABLE_REQUEST_MESSAGE,
            super::super::turn::RETRIEVAL_UNAVAILABLE_MESSAGE,
        ] {
            assert!(
                !crate::llm::fallbacks::FALLBACK_TEXTS.contains(&message),
                "{message} would be scored as an abstention"
            );
        }
    }
}
