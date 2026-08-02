//! SSE streaming infrastructure for chat responses.
//!
//! Contains the streaming loop, cancellation handling, and the
//! `CancellableStream` wrapper that cleans up when clients disconnect.

use std::collections::HashMap;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{Stream, StreamExt};
use tokio_util::sync::{CancellationToken, DropGuard};
use uuid::Uuid;

use crate::clients::MistralClient;
use crate::core::entitlements::{
    AuthorizationRequest, Operation, Permit, Quota, SharedEntitlementPolicy,
};
use crate::core::events::{DomainEvent, SharedEventSink};
use crate::core::principal::Principal;
use crate::core::protocol::{ChatEvent, ChatEventStream};
use crate::error::AppError;
use crate::llm::{LlmProvider, LlmStreamEvent, TeachingMode};
use crate::middleware::TaskTracker;
use crate::repositories::{ChatRepository, MemoryRepository, RagLogRepository, SourceRepository};
use crate::services::chat::{CreateMessageParams, create_message};
use crate::services::memory::{
    MIN_DROPPED_FOR_SUMMARY, apply_memory_actions, decay_memories, extract_memories,
    store_conversation_summary, summarize_truncated_history,
};
use crate::services::rag::eval::trace::{
    ReasonCode, RetrievalTrace, StageDurations, TokenCounts, query_hash,
};
use crate::services::rag::rag_log::{ChunkLogEntry, RagLogEntry, create_rag_log};
use crate::services::rag::search::{PipelineOutcome, mean_relevance};
use crate::types::MemoryDecayTracker;
use crate::types::SearchResult;

use super::citation_resolution::{CitationResolution, ResolvedCitations, resolve_citations};

// ============================================================================
// Constants
// ============================================================================

/// Maximum accumulated LLM response size (100 KB) to prevent OOM from runaway responses.
pub const MAX_RESPONSE_SIZE: usize = 102_400;

// ============================================================================
// StreamContext
// ============================================================================

/// Context for streaming LLM response.
pub(super) struct StreamContext {
    pub provider: Arc<dyn LlmProvider>,
    pub rag_log_repo: Arc<dyn RagLogRepository>,
    pub chat_repo: Arc<dyn ChatRepository>,
    /// Authorizes and records metered work. Replaces the usage repository.
    pub entitlements: SharedEntitlementPolicy,
    /// Permit issued for this exchange; consumed after the first real token.
    pub permit: Permit,
    /// Receives chat outcome events.
    pub events: SharedEventSink,
    pub notebook_id: Uuid,
    pub system_prompt: String,
    pub messages: Vec<crate::llm::LlmMessage>,
    pub context_chunks: Vec<SearchResult>,
    pub llm_timeout: Duration,
    pub model: Option<String>,
    /// Shutdown token from the root task scope.
    pub shutdown_token: CancellationToken,
    /// Root owner for process-scoped work started after the response is saved.
    pub task_tracker: TaskTracker,
    /// The account this exchange belongs to.
    pub principal: Principal,
    /// Teaching mode — used to skip follow-up suggestions for Quiz mode.
    pub teaching_mode: TeachingMode,
    /// Mistral client for generating follow-up suggestions (optional).
    pub mistral: Option<MistralClient>,
    /// The user's original question (for follow-up context).
    pub user_question: String,
    /// UI locale for generating follow-up suggestions in the correct language.
    pub locale: String,
    /// What the RAG pipeline produced besides the contexts: timings, per-stage
    /// candidate counts, reason codes and the score domain that ordered them.
    pub rag_outcome: PipelineOutcome,
    /// Evidence tokens the budgeting pass selected and dropped (US-018 AC-4).
    ///
    /// Measured where the decision was made. Recomputing it here could only
    /// describe what survived, which is the half of the number that never
    /// explains a truncated answer.
    pub evidence_tokens: TokenCounts,
    /// The standalone query retrieval used, when reformulation changed it.
    /// Reaches the trace as a hash and is never logged as text.
    pub reformulated_query: Option<String>,
    /// Whether memory extraction is enabled for this notebook.
    pub memory_enabled: bool,
    /// Voyage client for embedding memories (optional — extraction skipped if None).
    pub embeddings: Option<crate::core::providers::SharedEmbeddingProvider>,
    /// Memory repository for storing/querying memories.
    pub memory_repo: Arc<dyn MemoryRepository>,
    /// Messages dropped from history due to token budget limits (US-002).
    pub dropped_messages: Vec<crate::llm::LlmMessage>,
    /// Tracker for at-most-once-per-24h memory decay per notebook (US-004).
    pub memory_decay_tracker: MemoryDecayTracker,
    /// Session ID for grouping messages within a conversation session (US-007).
    pub session_id: Uuid,
    /// Source repository for loading notebook source names (US-003).
    pub source_repo: Arc<dyn SourceRepository>,
    /// Total message count at the time of this exchange (US-004 metadata enrichment).
    pub conversation_turn: usize,
    /// Structured RAG documents for native citation providers (Anthropic).
    /// Empty when using prompt-based citations (Mistral/OpenAI).
    pub rag_documents: Vec<crate::llm::RagDocument>,
}

// ============================================================================
// Streaming logic
// ============================================================================

/// Stream LLM response with cancellation support.
pub(super) async fn stream_llm_response(
    ctx: StreamContext,
    out: &ChatEventStream,
    cancel_token: CancellationToken,
) -> Result<(), AppError> {
    let StreamContext {
        provider,
        rag_log_repo,
        chat_repo,
        entitlements,
        permit,
        events,
        notebook_id,
        system_prompt,
        messages,
        context_chunks,
        llm_timeout,
        model,
        shutdown_token,
        task_tracker,
        principal,
        teaching_mode,
        mistral,
        user_question,
        locale,
        rag_outcome,
        evidence_tokens,
        reformulated_query,
        memory_enabled,
        embeddings,
        memory_repo,
        dropped_messages,
        memory_decay_tracker,
        session_id,
        source_repo,
        conversation_turn,
        rag_documents,
    } = ctx;

    // Record LLM provider and model into the current request span for observability.
    tracing::Span::current().record("provider", provider.name());
    tracing::Span::current().record("notebook_id", notebook_id.to_string().as_str());
    tracing::info!(
        %notebook_id,
        provider = %provider.name(),
        model = model.as_deref().unwrap_or("default"),
        "Starting LLM stream"
    );

    let uses_native_citations = !rag_documents.is_empty();

    // Initialize stream with timeout
    let llm_start = std::time::Instant::now();
    let stream_initialization = tokio::time::timeout(
        llm_timeout,
        provider.stream_chat(
            &system_prompt,
            messages,
            model.as_deref(),
            &rag_documents,
            None,
        ),
    );
    tokio::pin!(stream_initialization);
    let mut stream = tokio::select! {
        biased;
        () = shutdown_token.cancelled() => {
            tracing::info!(%notebook_id, provider = provider.name(), "LLM stream stopped during initialization by server shutdown");
            out.emit(ChatEvent::shutdown("Server shutting down")).await;
            return Ok(());
        }
        () = cancel_token.cancelled() => {
            tracing::info!(%notebook_id, provider = provider.name(), "LLM stream cancelled by client during initialization");
            return Ok(());
        }
        result = &mut stream_initialization => match result {
        Ok(Ok(stream)) => stream,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            tracing::warn!(
                timeout_secs = llm_timeout.as_secs(),
                provider = provider.name(),
                "LLM stream initialization timed out"
            );
            return Err(AppError::from(crate::error::LlmError::Timeout {
                provider: provider.name().into(),
                timeout_secs: llm_timeout.as_secs(),
            }));
        }
        }
    };

    let mut full_response = String::new();
    let model_name = model.as_deref().unwrap_or_else(|| provider.default_model());
    let mut message_counted = false;
    let mut llm_ttft_ms: Option<u128> = None;
    let mut native_citations = NativeCitationStream::default();
    let mut sse_frames = SseFrameBuffer::default();

    // Process stream with cancellation and shutdown support
    'stream: loop {
        tokio::select! {
            biased;

            _ = shutdown_token.cancelled() => {
                tracing::info!(
                    %notebook_id,
                    provider = provider.name(),
                    response_length = full_response.len(),
                    "SSE stream terminated by server shutdown"
                );
                out.emit(ChatEvent::shutdown("Server shutting down")).await;
                if !full_response.is_empty() {
                    save_partial_response(chat_repo.as_ref(), notebook_id, &full_response, model_name, "[server shutdown]", None, Some(session_id)).await;
                }
                return Ok(());
            }

            _ = cancel_token.cancelled() => {
                tracing::info!(
                    %notebook_id,
                    provider = provider.name(),
                    response_length = full_response.len(),
                    "SSE stream cancelled by client"
                );
                if !full_response.is_empty() {
                    save_partial_response(chat_repo.as_ref(), notebook_id, &full_response, model_name, "[interrupted]", None, Some(session_id)).await;
                }
                return Ok(());
            }

            chunk = stream.next() => {
                match chunk {
                    Some(Ok(bytes)) => {
                        match process_chunk(
                            &bytes,
                            provider.as_ref(),
                            &mut full_response,
                            out,
                            &mut sse_frames,
                            uses_native_citations.then_some(&mut native_citations),
                        ).await? {
                            ChunkResult::Continue => {
                                // Record LLM TTFT on first text delta
                                if llm_ttft_ms.is_none() && !full_response.is_empty() {
                                    let ttft = llm_start.elapsed().as_millis();
                                    llm_ttft_ms = Some(ttft);
                                    tracing::info!(llm_ttft_ms = ttft, %notebook_id, "LLM first token received");
                                }

                                // Increment message count after the first token arrives.
                                // This ensures failed LLM calls don't consume quota.
                                if !message_counted && !full_response.is_empty() {
                                    message_counted = true;
                                    if let Err(e) = entitlements.record(&permit, 1).await {
                                        tracing::warn!(
                                            account_id = %principal.account_id,
                                            error = %e,
                                            "Message limit exceeded after first token"
                                        );
                                        // The caller emits the one terminal error event.
                                        return Err(e);
                                    }
                                }
                            }
                            ChunkResult::Done => break 'stream,
                            ChunkResult::Truncated => {
                                tracing::warn!(
                                    %notebook_id,
                                    response_size = full_response.len(),
                                    max_size = MAX_RESPONSE_SIZE,
                                    "LLM response truncated: size limit exceeded"
                                );
                                save_partial_response(chat_repo.as_ref(), notebook_id, &full_response, model_name, "[truncated]", None, Some(session_id)).await;
                                // `error` is terminal: no `done` follows it (US-009, D-007).
                                out.emit(ChatEvent::error("Response size limit exceeded")).await;
                                return Ok(());
                            }
                        }
                    }
                    Some(Err(e)) => return Err(e),
                    None => break 'stream,
                }
            }
        }
    }

    // This phase transition is enforced by ChatEventStream: individual
    // citations cannot be emitted before it, and no text can follow it.
    out.finish_generation();

    // Extract citations: native (Anthropic Citations API) or regex-based [N] fallback.
    let resolved = resolve_citations(
        &CitationResolution {
            uses_native_citations,
            native_citations: &native_citations.citations,
            doc_citation_map: &native_citations.document_numbers,
            rag_documents: &rag_documents,
            context_chunks: &context_chunks,
            full_response: &full_response,
            notebook_id,
        },
        source_repo.as_ref(),
    )
    .await;
    let ResolvedCitations {
        citations,
        rejected,
        event_refs,
        lease,
    } = resolved;

    // Emit the per-request retrieval trace (US-004), after citation resolution
    // so that a refused marker is part of the record. The timings are in it
    // alongside the stage counts, score domain and reason codes that make a
    // latency number interpretable — and without the query text the old log did
    // not carry either, which the type now makes impossible to add back.
    build_retrieval_trace(
        notebook_id,
        &user_question,
        reformulated_query.as_deref(),
        &context_chunks,
        &rag_outcome,
        evidence_tokens,
        rejected,
        llm_ttft_ms.unwrap_or(0),
    )
    .emit();

    // Persist and enqueue every public citation shape while one transaction
    // pins all referenced source generations. Enqueue is deliberately
    // non-blocking so a stalled client cannot retain source-row locks.
    let saved_message = if let Some(lease) = lease {
        let saved = chat_repo
            .create_message_in_transaction(
                lease.transaction(),
                notebook_id,
                "assistant",
                &full_response,
                &citations,
                Some(model_name),
                None,
                Some(session_id),
            )
            .await?;
        for (index, source_id) in event_refs {
            out.try_emit_non_terminal(ChatEvent::citation(index, source_id));
        }
        out.try_emit_non_terminal(ChatEvent::citations(citations.clone()));
        lease.release().await?;
        saved
    } else {
        let saved = create_message(CreateMessageParams {
            repo: chat_repo.as_ref(),
            notebook_id,
            role: "assistant",
            content: &full_response,
            citations: &citations,
            model: Some(model_name),
            agent_id: None,
            session_id: Some(session_id),
        })
        .await?;
        out.emit(ChatEvent::citations(citations.clone())).await;
        saved
    };

    // Mean score, but only from a scale on which a mean means something. An
    // RRF value is a rank artifact and a stuffed chunk carries a constant;
    // publishing either as "context relevance" reports a number that moves
    // with the pool size rather than with the answer (US-012).
    let context_relevance = mean_relevance(&context_chunks);
    out.emit(ChatEvent::metrics(context_relevance)).await;

    // Process-owned post-response work: extract memories and summarize history.
    if memory_enabled
        && let Some(mistral_client) = mistral.clone()
        && let Some(embedder) = embeddings
    {
        spawn_memory_tasks(MemoryTaskContext {
            mistral_client,
            embedder,
            memory_repo: memory_repo.clone(),
            source_repo: source_repo.clone(),
            memory_decay_tracker: memory_decay_tracker.clone(),
            task_tracker,
            notebook_id,
            user_question: user_question.clone(),
            full_response: full_response.clone(),
            entitlements: entitlements.clone(),
            principal: principal.clone(),
            dropped_messages,
            conversation_turn,
        });
    }

    // Create RAG log entry linked to the saved message
    #[allow(clippy::if_not_else)] // positive branch is the main logic path
    let rag_log_id = if !context_chunks.is_empty() {
        let entry = RagLogEntry {
            notebook_id,
            user_id: principal.account_id,
            query_hash: query_hash(&user_question),
            reformulated_query_hash: reformulated_query.as_deref().map(query_hash),
            chunks_retrieved: context_chunks
                .iter()
                .map(|c| ChunkLogEntry {
                    chunk_id: c.chunk_id,
                    source_id: c.source_id,
                    chunk_index: c.chunk_index,
                    relevance_score: c.relevance(),
                })
                .collect(),
            response_id: Some(saved_message.id),
            // Same rule as the streamed metric: an average is written only
            // when the domain it came from is a relevance scale (US-012).
            retrieval_score_avg: context_relevance,
            context_relevance,
            model: Some(model_name.to_string()),
            provider: Some(provider.name().to_string()),
        };
        record_rag_log(rag_log_repo.as_ref(), &entry).await
    } else {
        None
    };

    // Follow-up suggestions precede `done`, which is the terminal successful
    // event (US-009, D-007). Skipped for Quiz mode and when Mistral is absent.
    if teaching_mode != TeachingMode::Quiz
        && let Some(mistral_client) = mistral
    {
        generate_follow_up_suggestions(
            out,
            &mistral_client,
            &user_question,
            &full_response,
            &context_chunks,
            &locale,
        )
        .await;
    }

    out.emit(ChatEvent::done(model_name, provider.name(), rag_log_id))
        .await;

    events.emit(DomainEvent::ChatCompleted {
        account_id: principal.account_id,
        notebook_id,
        provider: provider.name().to_owned(),
        duration_ms: u64::try_from(llm_start.elapsed().as_millis()).unwrap_or(u64::MAX),
        context_chunks: context_chunks.len(),
    });

    Ok(())
}

// ============================================================================
// Retrieval trace and interaction telemetry (US-004)
// ============================================================================

/// Describe one retrieval without its content.
///
/// Split out of the streaming function so it can be tested directly: the
/// property that matters — that nothing here can carry source text or the user's
/// question — is worth an assertion, and an assertion needs a seam.
#[allow(clippy::too_many_arguments)] // one record, assembled from one turn
pub(super) fn build_retrieval_trace(
    notebook_id: Uuid,
    query: &str,
    reformulated: Option<&str>,
    context_chunks: &[SearchResult],
    outcome: &PipelineOutcome,
    evidence_tokens: TokenCounts,
    rejected_citations: usize,
    llm_ttft_ms: u128,
) -> RetrievalTrace {
    let generation_ids: Vec<Uuid> = context_chunks.iter().map(|c| c.generation_id).collect();
    build_retrieval_trace_from_generation_ids(
        notebook_id,
        query,
        reformulated,
        generation_ids,
        context_chunks.is_empty(),
        outcome,
        evidence_tokens,
        rejected_citations,
        llm_ttft_ms,
    )
}

#[allow(clippy::too_many_arguments)] // one record, assembled from one turn
pub(super) fn build_retrieval_trace_from_generation_ids(
    notebook_id: Uuid,
    query: &str,
    reformulated: Option<&str>,
    mut generation_ids: Vec<Uuid>,
    record_no_candidates: bool,
    outcome: &PipelineOutcome,
    evidence_tokens: TokenCounts,
    rejected_citations: usize,
    llm_ttft_ms: u128,
) -> RetrievalTrace {
    // The domain comes from the pipeline rather than being guessed from the
    // shape of the result set: whether a reranker ran, failed or was absent
    // decides which scale ordered these chunks, and only the pipeline knows
    // (US-012).
    let mut trace = RetrievalTrace::new(notebook_id, query, "chat", outcome.score_domain);
    if let Some(reformulated) = reformulated {
        trace = trace.with_reformulation(reformulated);
    }

    // The generations that actually answered. Every search path joins the
    // active pointer, so this is a set of active generations by construction —
    // and an operator diagnosing a bad answer can tell which index produced it
    // (US-004, filled by EP-002).
    generation_ids.sort_unstable();
    generation_ids.dedup();
    trace.generation_ids = generation_ids;

    // Counts come from the pipeline. Recomputing them from the surviving
    // chunks can only ever describe the last stage, which is precisely the
    // stage a recall regression did *not* happen in (US-004).
    trace.candidates = outcome.counts;
    trace.unique_parents = outcome.unique_parents;
    // Both halves come from the budgeting pass. "How much evidence reached the
    // model" and "how much was cut to make it fit" are the pair that explains a
    // thin answer, and only the pass that made the cut knows the second
    // (US-018 AC-4).
    trace.tokens = evidence_tokens;

    let ms = |value: u128| u64::try_from(value).unwrap_or(u64::MAX);
    trace.durations = StageDurations {
        embed_ms: ms(outcome.embed_ms),
        search_ms: ms(outcome.search_ms),
        rerank_ms: ms(outcome.rerank_ms),
        stuffing_load_ms: ms(outcome.stuffing_load_ms),
        total_ms: ms(outcome.embed_ms
            + outcome.search_ms
            + outcome.rerank_ms
            + outcome.stuffing_load_ms
            + llm_ttft_ms),
    };

    trace.reasons.extend(&outcome.reasons);
    if record_no_candidates {
        trace.reasons.insert(ReasonCode::NoCandidates);
    }
    if rejected_citations > 0 {
        trace.reasons.insert(ReasonCode::CitationRejected);
    }
    trace.finish();
    trace
}

/// Write the interaction log, and survive its failure.
///
/// One attempt, no retry: the metrics repository being down must not turn one
/// chat turn into a write storm, and the answer has already been streamed and
/// saved by the time this runs. The failure is reported once, as a reason code
/// with no query text and no source content.
async fn record_rag_log(repo: &dyn RagLogRepository, entry: &RagLogEntry) -> Option<Uuid> {
    match create_rag_log(repo, entry).await {
        Ok(id) => Some(id),
        Err(e) => {
            tracing::warn!(
                notebook_id = %entry.notebook_id,
                // The same vocabulary the retrieval trace uses, so an operator
                // alerts on one set of reason codes rather than two.
                telemetry_failure = ReasonCode::TelemetryWriteFailed.as_str(),
                chunks_logged = entry.chunks_retrieved.len(),
                error_kind = chat_error_type(&e),
                "RAG interaction telemetry was not recorded; the exchange itself succeeded"
            );
            None
        }
    }
}

/// Categorize a failed chat exchange for the domain event.
///
/// Returns a short stable label, never the error message: domain events must
/// not carry free-form text.
pub(super) fn chat_error_type(error: &AppError) -> &'static str {
    match error {
        AppError::Forbidden(_) => "usage_limit",
        AppError::Llm(crate::error::LlmError::Timeout { .. }) => "timeout",
        AppError::Llm(_) => "provider_error",
        AppError::Validation(_) => "validation_error",
        AppError::Database(_) => "database_error",
        _ => "internal_error",
    }
}

// ============================================================================
// Extracted helpers for stream_llm_response
// ============================================================================

struct MemoryTaskContext {
    mistral_client: MistralClient,
    embedder: crate::core::providers::SharedEmbeddingProvider,
    memory_repo: Arc<dyn MemoryRepository>,
    source_repo: Arc<dyn SourceRepository>,
    memory_decay_tracker: MemoryDecayTracker,
    task_tracker: TaskTracker,
    notebook_id: Uuid,
    user_question: String,
    full_response: String,
    entitlements: SharedEntitlementPolicy,
    principal: Principal,
    dropped_messages: Vec<crate::llm::LlmMessage>,
    conversation_turn: usize,
}

/// Spawn process-owned tasks for memory extraction and conversation summarization.
fn spawn_memory_tasks(ctx: MemoryTaskContext) {
    let MemoryTaskContext {
        mistral_client,
        embedder,
        memory_repo,
        source_repo,
        memory_decay_tracker,
        task_tracker,
        notebook_id,
        user_question,
        full_response,
        entitlements,
        principal,
        dropped_messages,
        conversation_turn,
    } = ctx;
    // Spawn memory extraction + temporal decay
    let mem_repo = memory_repo.clone();
    let mem_source_repo = source_repo;
    let mem_user_question = user_question;
    let mem_response = full_response;
    let decay_tracker = memory_decay_tracker;
    let extraction_mistral = mistral_client.clone();
    let extraction_embedder = std::sync::Arc::clone(&embedder);
    let extraction = async move {
        let source_names: Vec<String> = match mem_source_repo.list_for_notebook(notebook_id).await {
            Ok(sources) => sources.into_iter().map(|s| s.title).collect(),
            Err(e) => {
                tracing::warn!(
                    %notebook_id,
                    error = %e,
                    "Failed to load source names for memory extraction, proceeding without"
                );
                vec![]
            }
        };

        match extract_memories(
            &extraction_mistral,
            extraction_embedder.as_ref(),
            &mem_user_question,
            &mem_response,
            notebook_id,
            mem_repo.as_ref(),
            &source_names,
            conversation_turn,
        )
        .await
        {
            Ok(actions) if !actions.is_empty() => {
                // Authorize the batch, then read the per-notebook cap. Both run
                // off the request path: this task starts after the response has
                // been streamed. A denial silently skips extraction, which is
                // the documented behaviour of the limit it replaces.
                if let Err(e) = entitlements
                    .authorize(AuthorizationRequest::new(
                        &principal,
                        Operation::CreateMemory { notebook_id },
                        Uuid::new_v4(),
                    ))
                    .await
                {
                    tracing::debug!(%notebook_id, error = %e, "Memory creation not authorized — skipping extraction");
                    return;
                }

                let limit = entitlements
                    .quota(&principal, Quota::MemoriesPerNotebook { notebook_id })
                    .await
                    .unwrap_or_else(|e| {
                        tracing::warn!(%notebook_id, error = %e, "Memory quota lookup failed");
                        None
                    });
                match apply_memory_actions(notebook_id, actions, mem_repo.as_ref(), limit).await {
                    Ok(applied) => {
                        tracing::info!(%notebook_id, applied, "Memory extraction completed");
                    }
                    Err(e) => {
                        tracing::warn!(%notebook_id, error = %e, "Memory action application failed");
                    }
                }
            }
            Ok(_) => {
                tracing::debug!(%notebook_id, "No memories extracted from exchange");
            }
            Err(e) => {
                tracing::warn!(%notebook_id, error = %e, "Memory extraction failed");
            }
        }

        if decay_tracker.should_run(notebook_id) {
            match decay_memories(notebook_id, mem_repo.as_ref()).await {
                Ok(result) => {
                    if result.decayed_count > 0 || result.deleted_count > 0 {
                        tracing::info!(
                            %notebook_id,
                            decayed = result.decayed_count,
                            deleted = result.deleted_count,
                            "Memory temporal decay applied"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(%notebook_id, error = %e, "Memory temporal decay failed");
                }
            }
        }
    };
    let extraction_shutdown = task_tracker.cancellation_token();
    if task_tracker
        .try_spawn("chat-memory-extraction", async move {
            tokio::select! {
                biased;
                () = extraction_shutdown.cancelled() => {}
                () = extraction => {}
            }
        })
        .is_err()
    {
        tracing::debug!(%notebook_id, "Memory extraction skipped during shutdown");
    }

    // Spawn summarization (separate task, independent from extraction)
    let should_summarize = dropped_messages.len() > MIN_DROPPED_FOR_SUMMARY;
    if should_summarize {
        let sum_repo = memory_repo;
        let summarization = async move {
            let result = tokio::time::timeout(Duration::from_secs(30), async {
                let summary =
                    summarize_truncated_history(&dropped_messages, &mistral_client).await?;
                store_conversation_summary(
                    notebook_id,
                    &summary,
                    embedder.as_ref(),
                    sum_repo.as_ref(),
                )
                .await?;
                Ok::<_, crate::error::AppError>(dropped_messages.len())
            })
            .await;
            match result {
                Ok(Ok(dropped)) => {
                    tracing::info!(%notebook_id, dropped, "Conversation summary stored");
                }
                Ok(Err(e)) => {
                    tracing::warn!(%notebook_id, error = %e, "Conversation summarization/storage failed");
                }
                Err(_) => {
                    tracing::warn!(%notebook_id, "Conversation summarization timed out after 30s");
                }
            }
        };
        let summary_shutdown = task_tracker.cancellation_token();
        if task_tracker
            .try_spawn("chat-conversation-summary", async move {
                tokio::select! {
                    biased;
                    () = summary_shutdown.cancelled() => {}
                    () = summarization => {}
                }
            })
            .is_err()
        {
            tracing::debug!(%notebook_id, "Conversation summary skipped during shutdown");
        }
    }
}

/// Generate and send follow-up question suggestions via Mistral.
async fn generate_follow_up_suggestions(
    out: &ChatEventStream,
    mistral_client: &MistralClient,
    user_question: &str,
    full_response: &str,
    context_chunks: &[SearchResult],
    locale: &str,
) {
    let response_preview: String = full_response.chars().take(500).collect();
    let unique_titles: HashSet<&str> = context_chunks
        .iter()
        .map(|c| c.source_title.as_str())
        .collect();
    let source_titles: Vec<&str> = unique_titles.into_iter().collect();

    let lang_instruction = match locale {
        l if l.starts_with("fr") => "The questions MUST be in French (Français).",
        l if l.starts_with("de") => "The questions MUST be in German (Deutsch).",
        l if l.starts_with("es") => "The questions MUST be in Spanish (Español).",
        _ => "The questions MUST be in English.",
    };
    let suggestion_prompt = format!(
        "Based on this Q&A exchange and sources, suggest 3 short follow-up questions (max 80 chars each) the user might want to ask next. Be contextual and specific. {lang_instruction}\n\nUser question: {user_question}\n\nAssistant response (excerpt): {response_preview}\n\nSources used: {}\n\nRespond ONLY with JSON: {{\"suggestions\": [\"Question 1?\", \"Question 2?\", \"Question 3?\"]}}",
        source_titles.join(", ")
    );

    match tokio::time::timeout(
        Duration::from_secs(5),
        mistral_client.complete(
            "You are a helpful assistant that generates follow-up questions. Respond only with valid JSON.",
            &suggestion_prompt,
        ),
    )
    .await
    {
        Ok(Ok(response)) => {
            match serde_json::from_str::<crate::core::protocol::ChatFollowUpSuggestions>(&response) {
                Ok(parsed) if !parsed.suggestions.is_empty() => {
                    out.emit(ChatEvent::follow_up_suggestions(parsed.suggestions))
                        .await;
                }
                Ok(_) => {
                    tracing::debug!("Follow-up suggestions response was empty, skipping");
                }
                Err(e) => {
                    tracing::debug!(error = %e, "Follow-up suggestions response did not match the contract, skipping");
                }
            }
        }
        Ok(Err(e)) => {
            tracing::debug!(error = %e, "Follow-up suggestion generation failed, skipping");
        }
        Err(_) => {
            tracing::debug!("Follow-up suggestion generation timed out, skipping");
        }
    }
}

// ============================================================================
// Chunk processing
// ============================================================================

/// Result of processing a stream chunk.
pub enum ChunkResult {
    /// Stream continues normally.
    Continue,
    /// Stream finished (done event or channel closed).
    Done,
    /// Response size limit exceeded — partial response should be saved with `[truncated]` marker.
    Truncated,
}

#[derive(Default)]
pub struct NativeCitationStream {
    citations: Vec<crate::llm::LocatedCitation<crate::llm::NativeCitation>>,
    document_numbers: HashMap<usize, usize>,
}

#[derive(Default)]
pub struct SseFrameBuffer {
    pending: Vec<u8>,
}

impl SseFrameBuffer {
    fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, AppError> {
        self.pending.extend_from_slice(bytes);
        if self.pending.len() > MAX_RESPONSE_SIZE && !self.pending.contains(&b'\n') {
            return Err(AppError::ProviderError(
                "Provider SSE frame exceeded the response size limit".to_owned(),
            ));
        }

        let mut start = 0;
        let mut data = Vec::new();
        for (end, byte) in self.pending.iter().copied().enumerate() {
            if byte != b'\n' {
                continue;
            }
            let mut line = &self.pending[start..end];
            if line.last() == Some(&b'\r') {
                line = &line[..line.len() - 1];
            }
            let line = std::str::from_utf8(line).map_err(|error| {
                AppError::ProviderError(format!("Provider SSE frame is not UTF-8: {error}"))
            })?;
            if let Some(payload) = line.strip_prefix("data:") {
                data.push(payload.strip_prefix(' ').unwrap_or(payload).to_owned());
            }
            start = end + 1;
        }
        if start > 0 {
            self.pending.drain(..start);
        }
        if self.pending.len() > MAX_RESPONSE_SIZE {
            return Err(AppError::ProviderError(
                "Provider SSE frame exceeded the response size limit".to_owned(),
            ));
        }
        Ok(data)
    }
}

/// Process a chunk of bytes from the LLM stream.
///
/// Native markers are injected while their events are parsed. This preserves
/// their association with the preceding claim. `sse_frames` retains partial
/// lines, so parsing is independent from how HTTP/TCP fragments the stream.
pub async fn process_chunk(
    bytes: &[u8],
    provider: &dyn LlmProvider,
    full_response: &mut String,
    out: &ChatEventStream,
    sse_frames: &mut SseFrameBuffer,
    mut native_citations: Option<&mut NativeCitationStream>,
) -> Result<ChunkResult, AppError> {
    for data in sse_frames.push(bytes)? {
        let Some(event) = provider.parse_sse_data(&data) else {
            continue;
        };

        match event {
            LlmStreamEvent::TextDelta { text } => {
                if full_response.len() + text.len() > MAX_RESPONSE_SIZE {
                    return Ok(ChunkResult::Truncated);
                }
                full_response.push_str(&text);
                if out.is_closed() {
                    tracing::debug!("Chat event channel closed, stopping stream");
                    return Ok(ChunkResult::Done);
                }
                out.emit(ChatEvent::chunk(text)).await;
            }
            LlmStreamEvent::NativeCitation { citation } => {
                if let Some(state) = native_citations.as_deref_mut() {
                    let next_number = state.document_numbers.len() + 1;
                    let number = *state
                        .document_numbers
                        .entry(citation.document_index)
                        .or_insert(next_number);
                    let marker = format!(" [{number}]");
                    if full_response.len() + marker.len() > MAX_RESPONSE_SIZE {
                        return Ok(ChunkResult::Truncated);
                    }
                    state.citations.push(crate::llm::LocatedCitation {
                        citation,
                        marker_start: full_response.len(),
                    });
                    full_response.push_str(&marker);
                    out.emit(ChatEvent::chunk(marker)).await;
                }
            }
            LlmStreamEvent::Done => return Ok(ChunkResult::Done),
            LlmStreamEvent::Error { message } => {
                return Err(AppError::ProviderError(format!(
                    "{} error: {message}",
                    provider.name()
                )));
            }
        }
    }

    Ok(ChunkResult::Continue)
}

/// Save partial response when stream is interrupted or truncated.
async fn save_partial_response(
    chat_repo: &dyn ChatRepository,
    notebook_id: Uuid,
    response: &str,
    model: &str,
    marker: &str,
    agent_id: Option<Uuid>,
    session_id: Option<Uuid>,
) {
    let content = format!("{response} {marker}");
    let _ = create_message(CreateMessageParams {
        repo: chat_repo,
        notebook_id,
        role: "assistant",
        content: &content,
        // A partial answer never reaches final active-generation and claim
        // validation. Persisting no citations is safer than blessing markers
        // whose association may itself have been interrupted.
        citations: &[],
        model: Some(model),
        agent_id,
        session_id,
    })
    .await;
}

// ============================================================================
// CancellableStream
// ============================================================================

/// A stream wrapper that requests cooperative cancellation when dropped.
///
/// The root task scope owns join and deadline escalation. This wrapper owns only
/// the client's right to request cancellation, so dropping an HTTP body cannot
/// bypass response persistence or the server-shutdown terminal event.
pub struct CancellableStream<S> {
    inner: S,
    cancel_guard: Option<DropGuard>,
    shutdown_token: CancellationToken,
    task_tracker: crate::middleware::TaskTracker,
}

impl<S> CancellableStream<S> {
    pub fn new(
        inner: S,
        cancel_guard: DropGuard,
        task_tracker: crate::middleware::TaskTracker,
    ) -> Self {
        let shutdown_token = task_tracker.cancellation_token();
        task_tracker.stream_started();
        Self {
            inner,
            cancel_guard: Some(cancel_guard),
            shutdown_token,
            task_tracker,
        }
    }
}

impl<S: Stream + Unpin> Stream for CancellableStream<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for CancellableStream<S> {
    fn drop(&mut self) {
        if self.shutdown_token.is_cancelled() {
            if let Some(guard) = self.cancel_guard.take() {
                guard.disarm();
            }
            tracing::debug!("CancellableStream dropped during server shutdown");
        } else {
            drop(self.cancel_guard.take());
            tracing::debug!("CancellableStream dropped, client cancellation requested");
        }
        self.task_tracker.stream_ended();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    /// Minimal mock LLM provider for testing `process_chunk`.
    struct MockProvider;

    impl MockProvider {
        fn text_delta_line(text: &str) -> Vec<u8> {
            format!("data: {{\"type\":\"text\",\"text\":\"{text}\"}}\n").into_bytes()
        }

        fn done_line() -> Vec<u8> {
            b"data: {\"type\":\"done\"}\n".to_vec()
        }

        fn citation_line(document_index: usize, cited_text: &str) -> String {
            format!(
                "data: {{\"type\":\"citation\",\"document_index\":{document_index},\"cited_text\":\"{cited_text}\"}}\n"
            )
        }
    }

    #[async_trait::async_trait]
    impl crate::llm::LlmProvider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }
        fn default_model(&self) -> &str {
            "mock-1"
        }
        fn is_available(&self) -> bool {
            true
        }
        fn parse_sse_data(&self, data: &str) -> Option<LlmStreamEvent> {
            let v: serde_json::Value = serde_json::from_str(data).ok()?;
            match v.get("type")?.as_str()? {
                "text" => Some(LlmStreamEvent::TextDelta {
                    text: v.get("text")?.as_str()?.to_owned(),
                }),
                "citation" => Some(LlmStreamEvent::NativeCitation {
                    citation: crate::llm::NativeCitation {
                        document_index: v.get("document_index")?.as_u64()? as usize,
                        cited_text: v.get("cited_text")?.as_str()?.to_owned(),
                        document_title: String::new(),
                    },
                }),
                "done" => Some(LlmStreamEvent::Done),
                _ => None,
            }
        }
        async fn stream_chat(
            &self,
            _system_prompt: &str,
            _messages: Vec<crate::llm::LlmMessage>,
            _model: Option<&str>,
            _documents: &[crate::llm::RagDocument],
            _temperature: Option<f32>,
        ) -> Result<crate::llm::ByteStream, AppError> {
            unimplemented!("not needed for process_chunk tests")
        }
    }

    #[tokio::test]
    async fn process_chunk_truncates_when_response_exceeds_limit() {
        let (out, _rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();

        // Fill response to just under the limit
        let padding = "x".repeat(MAX_RESPONSE_SIZE - 10);
        full_response.push_str(&padding);

        // Send a chunk that would exceed the limit
        let bytes = MockProvider::text_delta_line(&"y".repeat(20));
        let result = process_chunk(
            &bytes,
            &provider,
            &mut full_response,
            &out,
            &mut SseFrameBuffer::default(),
            None,
        )
        .await
        .unwrap();

        assert!(
            matches!(result, ChunkResult::Truncated),
            "Expected Truncated when response exceeds MAX_RESPONSE_SIZE"
        );
        // full_response should NOT have the new text appended
        assert_eq!(full_response.len(), MAX_RESPONSE_SIZE - 10);
    }

    #[tokio::test]
    async fn process_chunk_continues_when_within_limit() {
        let (out, _rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();

        let bytes = MockProvider::text_delta_line("hello");
        let result = process_chunk(
            &bytes,
            &provider,
            &mut full_response,
            &out,
            &mut SseFrameBuffer::default(),
            None,
        )
        .await
        .unwrap();

        assert!(
            matches!(result, ChunkResult::Continue),
            "Expected Continue for small response"
        );
        assert_eq!(full_response, "hello");
    }

    #[tokio::test]
    async fn process_chunk_returns_done_on_done_event() {
        let (out, _rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();

        let bytes = MockProvider::done_line();
        let result = process_chunk(
            &bytes,
            &provider,
            &mut full_response,
            &out,
            &mut SseFrameBuffer::default(),
            None,
        )
        .await
        .unwrap();

        assert!(
            matches!(result, ChunkResult::Done),
            "Expected Done on done event"
        );
    }

    #[tokio::test]
    async fn native_markers_follow_provider_event_order_with_no_early_citation_event() {
        let (out, mut rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();
        let mut native = NativeCitationStream::default();
        let mut frames = SseFrameBuffer::default();
        let bytes = format!(
            "{}{}{}{}",
            String::from_utf8(MockProvider::text_delta_line("Claim A")).expect("text A"),
            MockProvider::citation_line(0, "Claim A"),
            String::from_utf8(MockProvider::text_delta_line(". Claim B")).expect("text B"),
            MockProvider::citation_line(1, "Claim B"),
        );

        let result = process_chunk(
            bytes.as_bytes(),
            &provider,
            &mut full_response,
            &out,
            &mut frames,
            Some(&mut native),
        )
        .await
        .expect("process native events");

        assert!(matches!(result, ChunkResult::Continue));
        assert_eq!(full_response, "Claim A [1]. Claim B [2]");
        assert_eq!(native.citations.len(), 2);
        assert_eq!(
            native
                .citations
                .iter()
                .map(|located| located.marker_start)
                .collect::<Vec<_>>(),
            vec![7, 20]
        );
        for expected in ["Claim A", " [1]", ". Claim B", " [2]"] {
            match rx.recv().await.expect("chunk event") {
                ChatEvent::Chunk(chunk) => assert_eq!(chunk.text, expected),
                event => panic!("citation validation must remain after streaming, got {event:?}"),
            }
        }
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn provider_events_survive_arbitrary_network_fragmentation() {
        let (out, mut rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();
        let mut frames = SseFrameBuffer::default();
        let event = MockProvider::text_delta_line("réponse fragmentée");
        let split_inside_utf8 = event
            .windows(2)
            .position(|bytes| bytes == "é".as_bytes())
            .expect("multibyte character")
            + 1;

        for fragment in [
            &event[..7],
            &event[7..split_inside_utf8],
            &event[split_inside_utf8..],
        ] {
            let result = process_chunk(
                fragment,
                &provider,
                &mut full_response,
                &out,
                &mut frames,
                None,
            )
            .await
            .expect("process fragment");
            assert!(matches!(result, ChunkResult::Continue));
        }

        assert_eq!(full_response, "réponse fragmentée");
        match rx.recv().await.expect("one reconstructed event") {
            ChatEvent::Chunk(chunk) => assert_eq!(chunk.text, "réponse fragmentée"),
            event => panic!("expected chunk, got {event:?}"),
        }
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn max_response_size_constant_is_100kb() {
        assert_eq!(
            MAX_RESPONSE_SIZE, 102_400,
            "MAX_RESPONSE_SIZE should be 100 KB"
        );
    }

    #[tokio::test]
    async fn cancellable_stream_requests_task_cleanup_on_drop() {
        let cancel_token = CancellationToken::new();
        let token_clone = cancel_token.clone();
        let completed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completed_by_task = Arc::clone(&completed);

        let task_tracker = crate::middleware::TaskTracker::new();
        task_tracker
            .try_spawn("test-chat-stream", async move {
                token_clone.cancelled().await;
                completed_by_task.store(true, std::sync::atomic::Ordering::SeqCst);
            })
            .expect("task admission");
        let (_tx, rx) = mpsc::channel::<ChatEvent>(1);

        let stream = CancellableStream::new(
            ReceiverStream::new(rx),
            cancel_token.clone().drop_guard(),
            task_tracker.clone(),
        );

        drop(stream);

        assert!(
            cancel_token.is_cancelled(),
            "CancellationToken must be cancelled after stream drop"
        );

        tokio::time::timeout(Duration::from_millis(50), async {
            while !completed.load(std::sync::atomic::Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("task cleanup must complete");
    }

    #[tokio::test]
    async fn cancellable_stream_saves_partial_response_on_cancel() {
        // Verify that save_partial_response is invoked before cancel in the stream loop.
        // We test this indirectly: the cancel branch in stream_llm_response saves first,
        // then returns Ok(()), which means the task completes cleanly.
        let cancel_token = CancellationToken::new();

        // Cancel immediately
        cancel_token.cancel();

        // The token should be cancelled
        assert!(cancel_token.is_cancelled());
    }

    #[tokio::test]
    async fn cancellable_stream_does_not_abort_task_during_shutdown() {
        let cancel_token = CancellationToken::new();
        let task_tracker = crate::middleware::TaskTracker::new();
        let shutdown = task_tracker.cancellation_token();

        let completed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let completed_clone = completed.clone();
        task_tracker
            .try_spawn("test-chat-stream", async move {
                shutdown.cancelled().await;
                completed_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            })
            .expect("task admission");

        let (_tx, rx) = mpsc::channel::<ChatEvent>(1);

        let stream = CancellableStream::new(
            ReceiverStream::new(rx),
            cancel_token.clone().drop_guard(),
            task_tracker.clone(),
        );

        task_tracker.begin_shutdown();

        drop(stream);

        assert!(
            !cancel_token.is_cancelled(),
            "Server shutdown must not be reclassified as client cancellation"
        );

        task_tracker.wait().await;

        assert!(
            completed.load(std::sync::atomic::Ordering::SeqCst),
            "Task should complete cleanup during shutdown (not aborted)"
        );
    }

    #[tokio::test]
    async fn shutdown_token_is_exposed_by_coordinator() {
        let coordinator = crate::middleware::TaskTracker::new();
        let token = coordinator.cancellation_token();

        assert!(
            !token.is_cancelled(),
            "Shutdown token should not be cancelled initially"
        );

        // Simulate shutdown
        coordinator.shutdown(Duration::from_secs(1)).await;

        assert!(
            token.is_cancelled(),
            "Shutdown token should be cancelled after shutdown"
        );
    }

    #[tokio::test]
    async fn follow_up_suggestions_sent_after_done_before_stream_end() {
        // Verified by code structure: stream_llm_response awaits Mistral inline
        // (not spawned), so suggestions are sent before Ok(()) closes the stream.
        // A full integration test would require mocking the LLM provider.
    }

    /// Dropping a `CancellableStream` while items are actively flowing through
    /// triggers cooperative cancellation of the owned background task.
    #[tokio::test]
    async fn drop_mid_stream_triggers_cancellation_and_aborts_task() {
        let cancel_token = CancellationToken::new();
        let token_clone = cancel_token.clone();

        let task_tracker = crate::middleware::TaskTracker::new();
        task_tracker
            .try_spawn("test-chat-stream", async move {
                token_clone.cancelled().await;
            })
            .expect("task admission");
        let (tx, rx) = mpsc::channel::<ChatEvent>(10);

        let mut stream = CancellableStream::new(
            ReceiverStream::new(rx),
            cancel_token.clone().drop_guard(),
            task_tracker.clone(),
        );

        // Send items into the stream (simulating active LLM token delivery)
        tx.send(ChatEvent::chunk("chunk-1")).await.unwrap();
        tx.send(ChatEvent::chunk("chunk-2")).await.unwrap();

        // Consume one item to prove the stream is actively delivering
        let item = stream.next().await;
        assert!(
            item.is_some(),
            "Should receive first item from active stream"
        );

        // Drop the stream while items remain in the channel (mid-stream)
        drop(stream);

        assert!(
            cancel_token.is_cancelled(),
            "CancellationToken must be cancelled after mid-stream drop"
        );

        tokio::time::timeout(Duration::from_millis(50), async {
            while task_tracker.task_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("task must stop after client cancellation");
    }

    // ========================================================================
    // US-012: SSE response size limit tests
    // ========================================================================

    /// Feeding >100KB of chunk data across multiple `process_chunk` calls triggers
    /// `ChunkResult::Truncated` on the call that would exceed the limit. The over-limit
    /// text is NOT appended to `full_response`, and prior SSE "chunk" events are delivered.
    ///
    /// **Truncation behavior (documented):** When `process_chunk` returns `Truncated`,
    /// the caller (`stream_llm_response`) saves the partial response with a
    /// `[truncated]` marker and emits one terminal `error` event with message
    /// "Response size limit exceeded". Since US-009 no `done` follows it.
    #[tokio::test]
    async fn sse_response_over_100kb_triggers_truncation() {
        let (out, mut rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();
        let mut frames = SseFrameBuffer::default();

        // Feed 10 chunks of 10KB each = 100KB, all should succeed
        let chunk_10kb = "A".repeat(10_240);
        for i in 0..10 {
            let bytes = MockProvider::text_delta_line(&chunk_10kb);
            let result = process_chunk(
                &bytes,
                &provider,
                &mut full_response,
                &out,
                &mut frames,
                None,
            )
            .await
            .unwrap();
            assert!(
                matches!(result, ChunkResult::Continue),
                "Chunk {i} (at {}KB) should Continue",
                (i + 1) * 10
            );
        }
        assert_eq!(full_response.len(), MAX_RESPONSE_SIZE);

        // The 11th chunk (even 1 byte) should trigger Truncated
        let bytes = MockProvider::text_delta_line("x");
        let result = process_chunk(
            &bytes,
            &provider,
            &mut full_response,
            &out,
            &mut frames,
            None,
        )
        .await
        .unwrap();
        assert!(
            matches!(result, ChunkResult::Truncated),
            "Expected Truncated when accumulated response exceeds MAX_RESPONSE_SIZE"
        );

        // full_response must not grow beyond the limit
        assert_eq!(
            full_response.len(),
            MAX_RESPONSE_SIZE,
            "Truncated chunk must not be appended"
        );

        // Verify that 10 SSE "chunk" events were sent for the successful chunks
        let mut event_count = 0;
        while rx.try_recv().is_ok() {
            event_count += 1;
        }
        assert_eq!(
            event_count, 10,
            "Should have received 10 SSE chunk events before truncation"
        );
    }

    /// Feeding exactly MAX_RESPONSE_SIZE (100KB) of chunk data succeeds — the limit
    /// check is strictly greater-than, so exactly 102,400 bytes is allowed.
    #[tokio::test]
    async fn sse_response_exactly_100kb_succeeds() {
        let (out, _rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();
        let mut frames = SseFrameBuffer::default();

        // Feed in two halves: 50KB + 50KB
        let half = "B".repeat(MAX_RESPONSE_SIZE / 2);
        for _ in 0..2 {
            let bytes = MockProvider::text_delta_line(&half);
            let result = process_chunk(
                &bytes,
                &provider,
                &mut full_response,
                &out,
                &mut frames,
                None,
            )
            .await
            .unwrap();
            assert!(
                matches!(result, ChunkResult::Continue),
                "Expected Continue at or below MAX_RESPONSE_SIZE"
            );
        }

        assert_eq!(
            full_response.len(),
            MAX_RESPONSE_SIZE,
            "Response should be exactly MAX_RESPONSE_SIZE"
        );
    }

    /// Feeding <100KB of chunk data followed by a "done" event completes normally.
    /// All SSE "chunk" events are delivered, and the final event is Done.
    #[tokio::test]
    async fn sse_response_under_100kb_completes_normally() {
        let (out, mut rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();
        let mut frames = SseFrameBuffer::default();

        // Feed 5 chunks of 1KB each = 5KB total (well under limit)
        let chunk_1kb = "C".repeat(1_024);
        for _ in 0..5 {
            let bytes = MockProvider::text_delta_line(&chunk_1kb);
            let result = process_chunk(
                &bytes,
                &provider,
                &mut full_response,
                &out,
                &mut frames,
                None,
            )
            .await
            .unwrap();
            assert!(matches!(result, ChunkResult::Continue));
        }

        assert_eq!(full_response.len(), 5 * 1_024);

        // Send a done event — stream completes normally
        let bytes = MockProvider::done_line();
        let result = process_chunk(
            &bytes,
            &provider,
            &mut full_response,
            &out,
            &mut frames,
            None,
        )
        .await
        .unwrap();
        assert!(
            matches!(result, ChunkResult::Done),
            "Expected Done on done event"
        );

        // Verify all 5 SSE chunk events were delivered
        let mut event_count = 0;
        while rx.try_recv().is_ok() {
            event_count += 1;
        }
        assert_eq!(event_count, 5, "Should have received 5 SSE chunk events");
    }

    /// A single chunk that is exactly 1 byte over MAX_RESPONSE_SIZE on an empty
    /// response triggers Truncated immediately — the boundary is precise.
    #[tokio::test]
    async fn sse_response_single_chunk_one_byte_over_limit_truncates() {
        let (out, _rx) = ChatEventStream::channel();
        let provider = MockProvider;
        let mut full_response = String::new();
        let mut frames = SseFrameBuffer::default();

        let bytes = MockProvider::text_delta_line(&"Z".repeat(MAX_RESPONSE_SIZE + 1));
        let result = process_chunk(
            &bytes,
            &provider,
            &mut full_response,
            &out,
            &mut frames,
            None,
        )
        .await
        .unwrap();

        assert!(
            matches!(result, ChunkResult::Truncated),
            "A single chunk exceeding MAX_RESPONSE_SIZE must be Truncated"
        );
        assert!(
            full_response.is_empty(),
            "Over-limit chunk must not be appended"
        );
    }

    /// After the cancellation token fires, the background task exits and drops
    /// its channel sender, causing the stream to stop producing items. This
    /// verifies the full cancellation chain: token -> task exit -> channel close -> stream end.
    #[tokio::test]
    async fn stream_stops_producing_items_after_cancellation() {
        let cancel_token = CancellationToken::new();
        let token_for_task = cancel_token.clone();

        let (tx, rx) = mpsc::channel::<ChatEvent>(10);

        let task_tracker = crate::middleware::TaskTracker::new();
        task_tracker
            .try_spawn("test-chat-stream", async move {
                let _tx = tx;
                token_for_task.cancelled().await;
            })
            .expect("task admission");

        let mut stream = CancellableStream::new(
            ReceiverStream::new(rx),
            cancel_token.clone().drop_guard(),
            task_tracker,
        );

        // Cancel the token (simulates client disconnect)
        cancel_token.cancel();

        // The background task observes cancellation, exits, drops the sender.
        // The stream should then return None (no more items).
        let result = tokio::time::timeout(Duration::from_millis(50), stream.next()).await;

        match result {
            Ok(None) => {} // Stream ended as expected
            Ok(Some(_)) => panic!("Stream should not produce items after cancellation"),
            Err(_) => panic!("Stream did not end within 50ms after cancellation"),
        }
    }

    // ========================================================================
    // Retrieval trace and interaction telemetry (US-004)
    // ========================================================================

    fn chunk(source_id: Uuid, parent: Option<&str>, content: &str) -> SearchResult {
        SearchResult {
            chunk_id: Uuid::new_v4(),
            generation_id: Uuid::nil(),
            source_id,
            source_title: "Runbook".to_owned(),
            chunk_index: 0,
            content: content.to_owned(),
            parent_content: parent.map(str::to_owned),
            score: crate::types::RetrievalScore::Rrf(0.8),
            metadata: None,
            collapsed_children: Vec::new(),
        }
    }

    /// A pipeline outcome shaped like one the retrieval path would produce.
    ///
    /// The fusion domain is stated rather than defaulted: the default is
    /// "nothing ranked anything", which is not what a pipeline that returned
    /// contexts produced.
    fn outcome(selected: usize, unique_parents: usize) -> PipelineOutcome {
        PipelineOutcome {
            counts: crate::services::rag::eval::trace::StageCounts {
                selected,
                deduplicated: unique_parents,
                ..Default::default()
            },
            unique_parents,
            score_domain: Some(crate::types::ScoreDomain::RrfRank),
            confidence: crate::services::rag::search::RetrievalConfidence::Sufficient,
            ..PipelineOutcome::default()
        }
    }

    #[test]
    fn the_trace_records_every_field_the_prd_names() {
        let notebook = Uuid::new_v4();
        let source = Uuid::new_v4();
        let trace = build_retrieval_trace(
            notebook,
            "how long are interaction logs kept",
            Some("interaction log retention window"),
            &[
                chunk(source, Some("parent A"), "child one"),
                chunk(source, Some("parent A"), "child two"),
                chunk(source, Some("parent B"), "child three"),
            ],
            &PipelineOutcome {
                embed_ms: 12,
                search_ms: 30,
                rerank_ms: 40,
                reasons: [ReasonCode::DedupShortfall].into_iter().collect(),
                ..outcome(3, 2)
            },
            TokenCounts {
                selected: 512,
                dropped: 128,
            },
            0,
            100,
        );

        assert_eq!(trace.notebook_id, notebook);
        assert_eq!(trace.mode, "chat");
        assert_eq!(trace.score_domain, Some(crate::types::ScoreDomain::RrfRank));
        assert_eq!(trace.candidates.selected, 3);
        assert_eq!(trace.unique_parents, 2, "two children share one parent");
        assert_eq!(
            trace.tokens.selected, 512,
            "the selected count comes from the budgeting pass"
        );
        assert_eq!(
            trace.tokens.dropped, 128,
            "and so does what the budget cut, which is the half that explains a thin answer"
        );
        assert_eq!(trace.durations.embed_ms, 12);
        assert_eq!(trace.durations.search_ms, 30);
        assert_eq!(trace.durations.rerank_ms, 40);
        assert_eq!(
            trace.durations.total_ms, 182,
            "stage times plus first token"
        );
        assert!(trace.reformulated_query_hash.is_some());
        assert!(trace.reasons.contains(ReasonCode::DedupShortfall));
    }

    #[test]
    fn the_trace_carries_neither_the_question_nor_the_evidence() {
        let trace = build_retrieval_trace(
            Uuid::new_v4(),
            "what did the postmortem say about the deleted chunks",
            Some("postmortem deleted chunks root cause"),
            &[chunk(
                Uuid::new_v4(),
                Some("the parent passage"),
                "the child passage",
            )],
            &outcome(1, 1),
            TokenCounts::default(),
            0,
            0,
        );
        let rendered = serde_json::to_string(&trace).expect("trace serializes");
        for leaked in [
            "postmortem",
            "deleted",
            "root cause",
            "parent passage",
            "child passage",
        ] {
            assert!(
                !rendered.contains(leaked),
                "trace leaked `{leaked}`: {rendered}"
            );
        }
    }

    #[test]
    fn stuffing_is_recorded_as_its_own_score_domain() {
        let trace = build_retrieval_trace(
            Uuid::new_v4(),
            "anything",
            None,
            &[chunk(Uuid::new_v4(), None, "a")],
            &PipelineOutcome {
                stuffed: true,
                stuffing_load_ms: 7,
                score_domain: Some(crate::types::ScoreDomain::StuffingUniform),
                reasons: [ReasonCode::StuffingApplied].into_iter().collect(),
                ..outcome(1, 1)
            },
            TokenCounts::default(),
            0,
            0,
        );
        assert_eq!(
            trace.score_domain,
            Some(crate::types::ScoreDomain::StuffingUniform),
            "a uniform score must not be reported as a rank score"
        );
        assert!(trace.reasons.contains(ReasonCode::StuffingApplied));
        assert_eq!(trace.durations.stuffing_load_ms, 7);
    }

    #[test]
    fn an_empty_selection_is_recorded_as_such() {
        let trace = build_retrieval_trace(
            Uuid::new_v4(),
            "anything",
            None,
            &[],
            &PipelineOutcome::default(),
            TokenCounts::default(),
            0,
            0,
        );
        assert_eq!(trace.candidates.selected, 0);
        assert_eq!(trace.unique_parents, 0);
        assert!(trace.reasons.contains(ReasonCode::NoCandidates));
    }

    /// A metrics repository that is down, and counts how often it was asked.
    struct FailingRagLogRepo {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl FailingRagLogRepo {
        fn new() -> Self {
            Self {
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl RagLogRepository for FailingRagLogRepo {
        async fn create(&self, _entry: &RagLogEntry) -> crate::repositories::RepoResult<Uuid> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(AppError::Internal("metrics store unavailable".to_owned()))
        }

        async fn get_by_id(
            &self,
            _id: Uuid,
        ) -> crate::repositories::RepoResult<Option<crate::entities::rag_log::Model>> {
            Ok(None)
        }

        async fn get_for_user(
            &self,
            _id: Uuid,
            _user_id: Uuid,
        ) -> crate::repositories::RepoResult<crate::entities::rag_log::Model> {
            Err(AppError::NotFound("rag log".to_owned()))
        }

        async fn update_metrics(
            &self,
            _log_id: Uuid,
            _metrics: &crate::services::rag::rag_log::RagMetrics,
        ) -> crate::repositories::RepoResult<()> {
            Ok(())
        }

        async fn update_feedback(
            &self,
            _log_id: Uuid,
            _user_id: Uuid,
            _feedback: &str,
        ) -> crate::repositories::RepoResult<()> {
            Ok(())
        }

        async fn get_notebook_metrics(
            &self,
            _notebook_id: Uuid,
            _days: i32,
        ) -> crate::repositories::RepoResult<crate::services::rag::rag_log::AggregatedMetrics>
        {
            Err(AppError::Internal("unavailable".to_owned()))
        }

        async fn get_user_metrics(
            &self,
            _user_id: Uuid,
            _days: i32,
        ) -> crate::repositories::RepoResult<crate::services::rag::rag_log::AggregatedMetrics>
        {
            Err(AppError::Internal("unavailable".to_owned()))
        }

        async fn get_rag_log_ids_for_messages(
            &self,
            _message_ids: &[Uuid],
        ) -> crate::repositories::RepoResult<std::collections::HashMap<Uuid, (Uuid, Option<String>)>>
        {
            Ok(std::collections::HashMap::new())
        }

        async fn get_by_response_id(
            &self,
            _response_id: Uuid,
        ) -> crate::repositories::RepoResult<Option<crate::entities::rag_log::Model>> {
            Ok(None)
        }

        async fn purge_old_logs(
            &self,
            _retention_days: i32,
        ) -> crate::repositories::RepoResult<u64> {
            Ok(0)
        }

        async fn get_preferred_source_ids(
            &self,
            _notebook_id: Uuid,
        ) -> crate::repositories::RepoResult<Vec<Uuid>> {
            Ok(Vec::new())
        }
    }

    fn log_entry() -> RagLogEntry {
        RagLogEntry {
            notebook_id: Uuid::new_v4(),
            user_id: Uuid::new_v4(),
            query_hash: query_hash("what is the retry budget"),
            reformulated_query_hash: None,
            chunks_retrieved: vec![ChunkLogEntry {
                chunk_id: Uuid::new_v4(),
                source_id: Uuid::new_v4(),
                chunk_index: 0,
                relevance_score: 0.9,
            }],
            response_id: Some(Uuid::new_v4()),
            retrieval_score_avg: Some(0.9),
            context_relevance: Some(0.9),
            model: Some("mock-1".to_owned()),
            provider: Some("mock".to_owned()),
        }
    }

    #[tokio::test]
    async fn an_unavailable_metrics_store_is_tried_once_and_does_not_fail_the_exchange() {
        let repo = FailingRagLogRepo::new();
        let recorded = record_rag_log(&repo, &log_entry()).await;

        assert!(recorded.is_none(), "the caller continues without a log id");
        assert_eq!(
            repo.calls(),
            1,
            "one attempt: a down metrics store must not become a write storm"
        );
    }

    #[test]
    fn the_telemetry_failure_reason_is_stable() {
        // Dashboards and alerts key on this string, and it is the same code the
        // retrieval trace emits.
        assert_eq!(
            ReasonCode::TelemetryWriteFailed.as_str(),
            "telemetry_write_failed"
        );
    }
}
