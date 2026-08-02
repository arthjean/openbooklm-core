//! Preparing one chat turn: everything decided before the first token.
//!
//! The handler used to hold this inline, and it grew a numbered comment per
//! step and four near-identical early returns, one per way a turn can end
//! without a model. Here the pipeline runs top to bottom and every exit is a
//! [`TurnOutcome`], so the handler's only remaining job is the one thing it is
//! uniquely able to do: frame typed events as an SSE response.
//!
//! # Order, and why it is this order
//!
//! The evidence allowance gates context stuffing (US-018 AC-2), so the budget
//! must exist *before* retrieval runs. Everything it depends on — the
//! instructions, the memory block, the question — is knowable then. Retrieval
//! follows, then the one budgeting pass over what will actually be sent, then
//! the assertion that the assembled request fits its window (US-018 AC-4).

use std::time::Duration;

use uuid::Uuid;

use crate::core::CoreState;
use crate::core::principal::Principal;
use crate::core::protocol::{ChatEvent, ChatEventStream, ThinkingStage};
use crate::error::AppError;
use crate::llm::fallbacks::{insufficient_evidence_text, no_sources_text};
use crate::llm::prompts::{EvidenceFormat, build_system_prompt, system_prompt_shell};
use crate::llm::{LlmMessage, PromptBudget, build_messages};
use crate::services::chat::context_budget::{
    BudgetRefusal, PromptInputs, evidence_allowance, fit_prompt,
};
use crate::services::chat::orchestration::{
    RagContextParams, RetrievalFailure, ValidatedRequest, build_preference_boost,
    load_chat_history, load_memory_for_prompt, notify_history_truncation, retrieve_rag_context,
    validate_and_authorize,
};
use crate::services::memory::{MIN_DROPPED_FOR_SUMMARY, load_conversation_summaries};
use crate::services::rag::eval::trace::{ReasonCode, RetrievalTrace, TokenCounts};
use crate::services::rag::search::{PipelineOutcome, render_evidence};

use super::fallback::{FailureContext, FallbackContext, fallback_trace};
use super::streaming::{
    StreamContext, build_retrieval_trace, build_retrieval_trace_from_generation_ids,
};
use super::types::{MAX_HISTORY_FETCH, SendMessageRequest};

/// What the client is told when the request cannot be measured.
///
/// Deliberately not one of the documented abstentions: the sources are fine,
/// the deployment is not, and saying otherwise would be a false statement about
/// the user's notebook.
pub(super) const UNMEASURABLE_REQUEST_MESSAGE: &str =
    "This model cannot be used for this notebook right now. No answer was generated.";

/// What the client is told when retrieval never ran.
pub(super) const RETRIEVAL_UNAVAILABLE_MESSAGE: &str =
    "Retrieval is unavailable for this notebook. No answer was generated.";

/// How a prepared turn ends.
///
/// One variant per way the exchange can terminate, each already carrying
/// everything its task needs. The handler matches once and spawns.
pub(super) enum TurnOutcome {
    /// Generate an answer from a provider.
    Stream(Box<StreamContext>),
    /// Answer with a documented constant, without calling a provider.
    Answer(FallbackContext),
    /// Terminate through the structured error event without answering.
    Failed(FailureContext),
}

/// Validate, retrieve, budget and assemble one turn.
///
/// # Errors
/// Returns [`AppError`] only for the pre-stream failures the HTTP layer reports
/// directly: validation, authorization, rate limiting, history loading. Once the
/// exchange is under way, every other ending is a [`TurnOutcome`].
pub(super) async fn prepare(
    state: &CoreState,
    principal: &Principal,
    notebook_id: Uuid,
    payload: &SendMessageRequest,
    out: &ChatEventStream,
) -> Result<TurnOutcome, AppError> {
    // Validate input, check access and rate limits, save the user message,
    // resolve the provider.
    let req = validate_and_authorize(state, principal, notebook_id, payload).await?;
    let locale = payload.locale.as_deref().unwrap_or("en").to_owned();
    let teaching_mode = payload.teaching_mode;
    let format = EvidenceFormat::for_provider(req.provider.supports_native_citations());

    // Truncation is decided by the budgeting pass, not here.
    let hist = load_chat_history(state.repos.chat.as_ref(), notebook_id, MAX_HISTORY_FETCH).await?;

    let (memory_block, all_memories) = load_memory_for_prompt(
        req.memory_enabled,
        notebook_id,
        &req.message,
        state.repos.memory.as_ref(),
        state.clients.embeddings.as_deref(),
        &state.embedding_cache,
    )
    .await;

    // Without the memory block: the budgeting pass prices memory separately so
    // that it can drop it whole, and counting it here as well would charge the
    // turn for it twice.
    let instructions = system_prompt_shell(format, None, teaching_mode, &locale);

    let Some(budget) = PromptBudget::for_model(
        req.provider.name(),
        &req.model_id,
        req.provider.max_output_tokens(),
    ) else {
        // No declared window means no request can be measured, and an
        // unmeasured request is one the provider may reject after the prompt
        // has been assembled and paid for (US-018 AC-5).
        tracing::error!(
            %notebook_id,
            provider = req.provider.name(),
            model = %req.model_id,
            "No declared context window for this model"
        );
        return Ok(failed(
            state,
            principal,
            &req,
            notebook_id,
            ReasonCode::ContextWindowUnknown,
            "context_window_unknown",
            UNMEASURABLE_REQUEST_MESSAGE,
        ));
    };

    // --- Retrieval ------------------------------------------------------
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
        scope: req.scope,
        query: &req.message,
        max_chunks: req.max_context_chunks,
        embeddings: &state.clients.embeddings,
        reranker: &state.clients.reranker,
        hyde_service: state.clients.hyde_service.as_ref(),
        reformulator: state.clients.query_reformulator.as_ref(),
        history: &hist.raw,
        events: out,
        embedding_cache: Some(&state.embedding_cache),
        provider: req.provider.name(),
        model: &req.model_id,
        preference_boost,
        evidence_token_budget: evidence_allowance(
            &budget,
            &instructions,
            memory_block.as_deref(),
            &req.message,
        ),
    })
    .await;
    let mut rag_outcome = retrieved.outcome;
    let reformulated_query = retrieved.reformulated_query;

    // No evidence is three different incidents, and only one of them is an
    // error (US-020 AC-4).
    if let Some(failure) = retrieved.failure {
        let reason = failure.reason_code();
        rag_outcome.reasons.insert(reason);
        let trace = build_retrieval_trace(
            notebook_id,
            &req.message,
            reformulated_query.as_deref(),
            &retrieved.chunks,
            &rag_outcome,
            TokenCounts::default(),
            0,
            0,
        );
        return Ok(match failure {
            RetrievalFailure::Infrastructure => failed_with_trace(
                state,
                principal,
                &req,
                notebook_id,
                ReasonCode::ProviderError,
                "retrieval_unavailable",
                RETRIEVAL_UNAVAILABLE_MESSAGE,
                trace,
            ),
            RetrievalFailure::NoSources => answer_with_trace(
                state,
                principal,
                &req,
                notebook_id,
                no_sources_text(&locale),
                trace,
            ),
            RetrievalFailure::NoEvidence => answer_with_trace(
                state,
                principal,
                &req,
                notebook_id,
                insufficient_evidence_text(&locale),
                trace,
            ),
        });
    }

    // --- One budgeting pass over everything that will be sent -----------
    out.emit(ChatEvent::thinking(ThinkingStage::Generating))
        .await;

    // Conversation summaries are the oldest end of the history, so they are
    // budgeted with it rather than smuggled in after the fitting pass.
    let summaries = if req.memory_enabled {
        load_conversation_summaries(&all_memories)
    } else {
        Vec::new()
    };
    let summary_count = summaries.len();
    let mut history: Vec<LlmMessage> = summaries;
    history.extend(hist.all_messages.iter().cloned());

    let inputs = PromptInputs {
        budget,
        format,
        instructions: &instructions,
        memory: memory_block.as_deref(),
        query: &req.message,
        history: &history,
    };
    let fitted = match fit_prompt(&inputs, retrieved.chunks) {
        Ok(fitted) => fitted,
        Err(refusal) => {
            let trace = build_budget_refusal_trace(
                notebook_id,
                &req.message,
                reformulated_query.as_deref(),
                &rag_outcome,
                &refusal,
            );
            return Ok(match refusal {
                BudgetRefusal::MandatoryComponents {
                    needed, allowance, ..
                } => {
                    tracing::error!(
                        %notebook_id,
                        provider = req.provider.name(),
                        model = %req.model_id,
                        window = budget.window(),
                        needed,
                        allowance,
                        "Instructions and question alone exceed the window"
                    );
                    failed_with_trace(
                        state,
                        principal,
                        &req,
                        notebook_id,
                        ReasonCode::PromptOverBudget,
                        "prompt_over_budget",
                        UNMEASURABLE_REQUEST_MESSAGE,
                        trace,
                    )
                }
                BudgetRefusal::NoEvidenceFits { .. } => {
                    tracing::info!(
                        %notebook_id,
                        provider = req.provider.name(),
                        model = %req.model_id,
                        "No retrieved passage fits the provider context window"
                    );
                    answer_with_trace(
                        state,
                        principal,
                        &req,
                        notebook_id,
                        insufficient_evidence_text(&locale),
                        trace,
                    )
                }
            });
        }
    };
    rag_outcome.reasons.extend(&fitted.reasons);

    let context_chunks = fitted.evidence;
    let evidence = render_evidence(format, &context_chunks);
    let system_prompt = build_system_prompt(
        format,
        &evidence.region,
        fitted
            .memory_kept
            .then_some(memory_block.as_deref())
            .flatten(),
        teaching_mode,
        &locale,
    );

    // --- History truncation and summarization notices -------------------
    //
    // The dropped messages are the oldest prefix of the combined history, so the
    // first `summary_count` of them are injected summaries. Feeding those back
    // into summarization would summarize summaries, turn after turn.
    let dropped_messages: Vec<LlmMessage> = fitted
        .history
        .dropped
        .into_iter()
        .skip(summary_count)
        .collect();
    let dropped_count = dropped_messages.len();
    notify_history_truncation(
        out,
        fitted.history.was_truncated,
        history.len(),
        fitted.history.messages.len(),
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

    let messages = build_messages(&fitted.history.messages, &req.message);

    // Native documents are outside the system prompt, but still inside the
    // provider request. The fitting pass measured their complete rendered
    // blocks, so the final assertion must add that amount back explicitly.
    let native_document_tokens = match format {
        EvidenceFormat::Inline => 0,
        EvidenceFormat::NativeDocuments => fitted.tokens.selected,
    };
    // The assertion US-018 AC-4 asks for, on the request as it will be sent.
    if !budget.admits_with_additional_prompt_tokens(
        &system_prompt,
        &messages,
        native_document_tokens,
    ) {
        tracing::error!(
            %notebook_id,
            provider = req.provider.name(),
            model = %req.model_id,
            window = budget.window(),
            "Assembled request exceeds the declared context window"
        );
        rag_outcome.reasons.insert(ReasonCode::PromptOverBudget);
        let trace = build_retrieval_trace(
            notebook_id,
            &req.message,
            reformulated_query.as_deref(),
            &context_chunks,
            &rag_outcome,
            fitted.tokens,
            0,
            0,
        );
        return Ok(failed_with_trace(
            state,
            principal,
            &req,
            notebook_id,
            ReasonCode::PromptOverBudget,
            "prompt_over_budget",
            UNMEASURABLE_REQUEST_MESSAGE,
            trace,
        ));
    }

    Ok(TurnOutcome::Stream(Box::new(StreamContext {
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
        task_tracker: state.task_tracker.clone(),
        principal: principal.clone(),
        teaching_mode,
        mistral: state.clients.mistral.clone(),
        user_question: req.message,
        locale,
        rag_outcome,
        evidence_tokens: fitted.tokens,
        reformulated_query,
        memory_enabled: req.memory_enabled,
        embeddings: state.clients.embeddings.clone(),
        memory_repo: state.repos.memory.clone(),
        dropped_messages,
        memory_decay_tracker: state.memory_decay_tracker.clone(),
        session_id: req.session_id,
        source_repo: state.repos.sources.clone(),
        conversation_turn: hist.all_messages.len() + 1,
        rag_documents: evidence.documents,
    })))
}

/// Preserve every retrieval field when budgeting terminates a turn.
fn build_budget_refusal_trace(
    notebook_id: Uuid,
    query: &str,
    reformulated_query: Option<&str>,
    outcome: &PipelineOutcome,
    refusal: &BudgetRefusal,
) -> RetrievalTrace {
    let (generation_ids, tokens, record_no_candidates, reason) = match refusal {
        BudgetRefusal::MandatoryComponents {
            tokens,
            generation_ids,
            ..
        } => (
            generation_ids.clone(),
            *tokens,
            generation_ids.is_empty(),
            ReasonCode::PromptOverBudget,
        ),
        BudgetRefusal::NoEvidenceFits {
            tokens,
            generation_ids,
        } => (
            generation_ids.clone(),
            *tokens,
            true,
            ReasonCode::EvidenceDroppedForBudget,
        ),
    };
    let mut outcome = outcome.clone();
    outcome.reasons.insert(reason);
    build_retrieval_trace_from_generation_ids(
        notebook_id,
        query,
        reformulated_query,
        generation_ids,
        record_no_candidates,
        &outcome,
        tokens,
        0,
        0,
    )
}

/// A constant answer that retains the complete retrieval record.
fn answer_with_trace(
    state: &CoreState,
    principal: &Principal,
    req: &ValidatedRequest,
    notebook_id: Uuid,
    text: &'static str,
    trace: RetrievalTrace,
) -> TurnOutcome {
    TurnOutcome::Answer(FallbackContext {
        chat_repo: state.repos.chat.clone(),
        events: state.events.clone(),
        account_id: principal.account_id,
        notebook_id,
        session_id: req.session_id,
        provider: req.provider.name().to_owned(),
        model: req.model_id.clone(),
        answer: text,
        trace,
    })
}

/// A turn that ends without an answer.
fn failed(
    state: &CoreState,
    principal: &Principal,
    req: &ValidatedRequest,
    notebook_id: Uuid,
    reason: ReasonCode,
    error_type: &'static str,
    message: &'static str,
) -> TurnOutcome {
    let trace = fallback_trace(notebook_id, &req.message, reason);
    failed_with_trace(
        state,
        principal,
        req,
        notebook_id,
        reason,
        error_type,
        message,
        trace,
    )
}

/// A turn that ends without an answer after retrieval produced a full trace.
#[allow(clippy::too_many_arguments)] // one terminal turn context
fn failed_with_trace(
    state: &CoreState,
    principal: &Principal,
    req: &ValidatedRequest,
    notebook_id: Uuid,
    reason: ReasonCode,
    error_type: &'static str,
    message: &'static str,
    trace: RetrievalTrace,
) -> TurnOutcome {
    TurnOutcome::Failed(FailureContext {
        events: state.events.clone(),
        account_id: principal.account_id,
        notebook_id,
        provider: req.provider.name().to_owned(),
        reason,
        trace,
        error_type,
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::rag::eval::trace::StageCounts;
    use crate::services::rag::search::RetrievalConfidence;
    use crate::types::{RetrievalScore, ScoreDomain, SearchResult};

    #[test]
    fn budget_refusal_mapping_preserves_the_complete_retrieval_trace() {
        let notebook_id = Uuid::new_v4();
        let generation_id = Uuid::new_v4();
        let query = "oversized evidence";
        let inputs = PromptInputs {
            budget: PromptBudget::new(16_000, 1_000),
            format: EvidenceFormat::Inline,
            instructions: "policy",
            memory: None,
            query,
            history: &[],
        };
        let refusal = fit_prompt(
            &inputs,
            vec![SearchResult {
                chunk_id: Uuid::new_v4(),
                generation_id,
                source_id: Uuid::new_v4(),
                source_title: "Synthetic runbook".to_owned(),
                chunk_index: 0,
                content: "child ".repeat(40_000),
                parent_content: Some("parent ".repeat(40_000)),
                score: RetrievalScore::Rrf(0.5),
                metadata: None,
                collapsed_children: Vec::new(),
            }],
        )
        .expect_err("neither the parent nor child should fit");
        let pipeline = PipelineOutcome {
            embed_ms: 11,
            search_ms: 22,
            rerank_ms: 33,
            counts: StageCounts {
                dense: 20,
                lexical: 18,
                fused: 30,
                reranker_input: 15,
                reranker_output: 10,
                deduplicated: 8,
                selected: 1,
            },
            unique_parents: 1,
            reasons: [ReasonCode::DedupShortfall].into_iter().collect(),
            score_domain: Some(ScoreDomain::RrfRank),
            confidence: RetrievalConfidence::Sufficient,
            ..PipelineOutcome::default()
        };

        let trace = build_budget_refusal_trace(
            notebook_id,
            query,
            Some("oversized source evidence"),
            &pipeline,
            &refusal,
        );

        assert_eq!(trace.generation_ids, vec![generation_id]);
        assert!(trace.reformulated_query_hash.is_some());
        assert_eq!(trace.score_domain, pipeline.score_domain);
        assert_eq!(trace.candidates, pipeline.counts);
        assert_eq!(trace.unique_parents, pipeline.unique_parents);
        assert_eq!(trace.durations.embed_ms, 11);
        assert_eq!(trace.durations.search_ms, 22);
        assert_eq!(trace.durations.rerank_ms, 33);
        assert_eq!(trace.tokens.selected, 0);
        assert!(trace.tokens.dropped > 0);
        assert!(trace.reasons.contains(ReasonCode::DedupShortfall));
        assert!(trace.reasons.contains(ReasonCode::NoCandidates));
        assert!(trace.reasons.contains(ReasonCode::EvidenceDroppedForBudget));
    }
}
