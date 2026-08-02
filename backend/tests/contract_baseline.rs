#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented
)]

//! Golden contract baseline for the open-core split (US-002, EP-001).
//!
//! These tests freeze the **current** public REST and SSE behavior so that the
//! seam extraction in EP-002 and EP-003 cannot silently change it. They are a
//! compatibility target, not a specification: when a fixture and the code
//! disagree, the code changed and the change must be deliberate.
//!
//! Fixtures live in `contracts/baseline/` at the repository root, outside
//! `backend/`, because the TypeScript SDK and its parser tests consume the same
//! files (US-010).
//!
//! ## Regenerating
//!
//! ```bash
//! UPDATE_BASELINE=1 cargo test --test contract_baseline
//! git diff contracts/baseline   # review every line: this is a contract change
//! ```
//!
//! Regeneration always writes the **complete** serialization of the live value.
//! A fixture can therefore never be a subset of what the code emits, which is
//! what `assert_no_dropped_fields` enforces on every run.
//!
//! ## Offline by construction
//!
//! Nothing here touches PostgreSQL, a provider API or a commercial key. The one
//! provider used is `FakeLlmProvider`, defined below.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

// ============================================================================
// Deterministic synthetic data
// ============================================================================

/// All identifiers are synthetic and stable. No production-derived value is
/// permitted in a public fixture (PRD hard constraint).
const NOTEBOOK_ID: &str = "11111111-1111-4111-8111-111111111111";
const SOURCE_ID: &str = "22222222-2222-4222-8222-222222222222";
const NOTE_ID: &str = "33333333-3333-4333-8333-333333333333";
const MESSAGE_ID: &str = "44444444-4444-4444-8444-444444444444";
const MEMORY_ID: &str = "55555555-5555-4555-8555-555555555555";
const RAG_LOG_ID: &str = "66666666-6666-4666-8666-666666666666";
const SESSION_ID: &str = "77777777-7777-4777-8777-777777777777";
const CHUNK_ID: &str = "88888888-8888-4888-8888-888888888888";

const CREATED_AT: &str = "2026-01-01T00:00:00+00:00";
const UPDATED_AT: &str = "2026-01-02T00:00:00+00:00";

fn uuid(s: &str) -> Uuid {
    Uuid::parse_str(s).unwrap()
}

// ============================================================================
// Fixture harness
// ============================================================================

fn baseline_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("backend/ always has a parent")
        .join("contracts/baseline")
}

fn updating() -> bool {
    std::env::var("UPDATE_BASELINE").is_ok_and(|v| v != "0" && !v.is_empty())
}

fn read_fixture(relative: &str) -> Value {
    let path = baseline_root().join(relative);
    match std::fs::read_to_string(&path) {
        Ok(raw) => serde_json::from_str(&raw)
            .unwrap_or_else(|e| panic!("{} is not valid JSON: {e}", path.display())),
        Err(_) if updating() => json!({}),
        Err(e) => panic!(
            "missing baseline fixture {}: {e}\n\
             run `UPDATE_BASELINE=1 cargo test --test contract_baseline` to create it",
            path.display()
        ),
    }
}

fn write_fixture(relative: &str, value: &Value) {
    let path = baseline_root().join(relative);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut rendered = serde_json::to_string_pretty(value).unwrap();
    rendered.push('\n');
    std::fs::write(&path, rendered).unwrap();
}

/// Recursively assert that every field present in `actual` also exists in
/// `recorded`.
///
/// This is the guard behind the US-002 acceptance criterion "a field that cannot
/// be represented by the baseline schema fails generation rather than being
/// dropped": a struct field added upstream shows up in `actual` and has nowhere
/// to go in the fixture, so the run fails and names the exact JSON pointer.
fn assert_no_dropped_fields(pointer: &str, actual: &Value, recorded: &Value) {
    match (actual, recorded) {
        (Value::Object(a), Value::Object(r)) => {
            for (key, av) in a {
                let child = format!("{pointer}/{key}");
                let Some(rv) = r.get(key) else {
                    panic!(
                        "BASELINE DROPPED FIELD at `{child}`: the code emits this field but the \
                         fixture has no place for it.\n\
                         Regenerate with `UPDATE_BASELINE=1 cargo test --test contract_baseline` \
                         and review the diff as a contract change."
                    );
                };
                assert_no_dropped_fields(&child, av, rv);
            }
        }
        (Value::Array(a), Value::Array(r)) => {
            for (i, av) in a.iter().enumerate() {
                if let Some(rv) = r.get(i) {
                    assert_no_dropped_fields(&format!("{pointer}/{i}"), av, rv);
                }
            }
        }
        _ => {}
    }
}

/// Compare one named case against its recorded baseline.
struct Baseline {
    relative: &'static str,
    cases: BTreeMap<String, Value>,
    recorded: Value,
}

impl Baseline {
    fn open(relative: &'static str) -> Self {
        Self {
            relative,
            cases: BTreeMap::new(),
            recorded: read_fixture(relative),
        }
    }

    fn case(&mut self, name: &str, value: &impl Serialize) {
        // Round-trip through the JSON text form so the comparison sees exactly
        // what a fixture file can hold. Without this, float formatting makes an
        // in-memory value and its serialized form compare unequal.
        let actual: Value = serde_json::from_str(&serde_json::to_string(value).unwrap()).unwrap();

        if updating() {
            self.cases.insert(name.to_owned(), actual);
            return;
        }

        if let Some(recorded) = self.recorded.get(name) {
            assert_no_dropped_fields(name, &actual, recorded);
            assert_eq!(
                &actual, recorded,
                "\nBASELINE DRIFT in {}[{name}]\n\
                 The public contract changed. If the change is intended, regenerate with \
                 `UPDATE_BASELINE=1 cargo test --test contract_baseline` and describe the \
                 contract change in the pull request.\n",
                self.relative
            );
        } else {
            panic!(
                "missing baseline case `{name}` in {}\n\
                 run `UPDATE_BASELINE=1 cargo test --test contract_baseline` to record it",
                self.relative
            );
        }
        self.cases.insert(name.to_owned(), actual);
    }

    /// Persist when regenerating, and always assert the fixture records no case
    /// the code no longer produces.
    fn finish(self) {
        if updating() {
            let value = Value::Object(self.cases.into_iter().collect());
            write_fixture(self.relative, &value);
            return;
        }
        if let Value::Object(recorded) = &self.recorded {
            let stale: Vec<&String> = recorded
                .keys()
                .filter(|k| !self.cases.contains_key(*k))
                .collect();
            assert!(
                stale.is_empty(),
                "{} records cases the code no longer produces: {stale:?}",
                self.relative
            );
        }
    }
}

// ============================================================================
// REST: success and failure shapes
// ============================================================================

#[test]
fn baseline_notebook_responses() {
    use openbooklm::api::notebooks::{NotebookResponse, NotebooksListResponse};

    let full = NotebookResponse {
        id: uuid(NOTEBOOK_ID),
        title: "Baseline notebook".into(),
        description: Some("A synthetic notebook used by the contract baseline.".into()),
        memory_enabled: true,
        is_demo: false,
        suggested_questions: vec!["What is in this notebook?".into()],
        source_count: 3,
        created_at: CREATED_AT.into(),
        updated_at: UPDATED_AT.into(),
    };
    let minimal = NotebookResponse {
        id: uuid(NOTEBOOK_ID),
        title: "Minimal notebook".into(),
        description: None,
        memory_enabled: false,
        is_demo: true,
        suggested_questions: vec![],
        source_count: 0,
        created_at: CREATED_AT.into(),
        updated_at: CREATED_AT.into(),
    };

    let mut b = Baseline::open("rest/notebooks.json");
    b.case("notebook_full", &full);
    b.case("notebook_minimal", &minimal);
    b.case(
        "notebook_list",
        &NotebooksListResponse {
            notebooks: vec![minimal],
        },
    );
    b.case(
        "notebook_list_empty",
        &NotebooksListResponse { notebooks: vec![] },
    );
    b.finish();
}

#[test]
fn baseline_source_responses() {
    use openbooklm::api::sources::{
        ChunkResponse, ChunksListResponse, SourceResponse, SourcesListResponse,
    };

    let ready = SourceResponse {
        id: uuid(SOURCE_ID),
        notebook_id: uuid(NOTEBOOK_ID),
        title: "Baseline source".into(),
        source_type: "text".into(),
        status: "ready".into(),
        error_message: None,
        chunk_count: 12,
        metadata: json!({ "content_hash": "0000000000000000" }),
        created_at: CREATED_AT.into(),
    };
    let failed = SourceResponse {
        id: uuid(SOURCE_ID),
        notebook_id: uuid(NOTEBOOK_ID),
        title: "Broken source".into(),
        source_type: "web".into(),
        status: "error".into(),
        error_message: Some("Extraction failed".into()),
        chunk_count: 0,
        metadata: json!({}),
        created_at: CREATED_AT.into(),
    };

    let mut b = Baseline::open("rest/sources.json");
    b.case("source_ready", &ready);
    b.case("source_failed", &failed);
    b.case(
        "source_list",
        &SourcesListResponse {
            sources: vec![ready],
        },
    );
    b.case(
        "chunk_list",
        &ChunksListResponse {
            chunks: vec![ChunkResponse {
                id: uuid(CHUNK_ID),
                chunk_index: 0,
                content: "Synthetic chunk content.".into(),
            }],
        },
    );
    b.finish();
}

#[test]
fn baseline_note_responses() {
    use openbooklm::api::notes::{NoteResponse, NotesListResponse};

    let derived = NoteResponse {
        id: uuid(NOTE_ID),
        notebook_id: uuid(NOTEBOOK_ID),
        title: "Saved answer".into(),
        content: "Answer body with a citation [1].".into(),
        original_message_id: Some(uuid(MESSAGE_ID)),
        created_at: CREATED_AT.into(),
        updated_at: UPDATED_AT.into(),
    };
    let standalone = NoteResponse {
        id: uuid(NOTE_ID),
        notebook_id: uuid(NOTEBOOK_ID),
        title: "Manual note".into(),
        content: String::new(),
        original_message_id: None,
        created_at: CREATED_AT.into(),
        updated_at: CREATED_AT.into(),
    };

    let mut b = Baseline::open("rest/notes.json");
    b.case("note_from_message", &derived);
    b.case("note_standalone", &standalone);
    b.case(
        "note_list",
        &NotesListResponse {
            notes: vec![standalone],
        },
    );
    b.finish();
}

#[test]
fn baseline_memory_responses() {
    use openbooklm::api::memory::{MemoriesListResponse, MemoryResponse};

    let memory = MemoryResponse {
        id: uuid(MEMORY_ID),
        notebook_id: uuid(NOTEBOOK_ID),
        content: "The reader prefers concise answers.".into(),
        memory_type: "preference".into(),
        metadata: json!({ "source": "chat" }),
        salience: 0.75,
        created_at: CREATED_AT.into(),
        updated_at: UPDATED_AT.into(),
    };

    let mut b = Baseline::open("rest/memory.json");
    b.case("memory", &memory);
    b.case(
        "memory_list",
        &MemoriesListResponse {
            memories: vec![memory],
            limit: 50,
            count: 1,
        },
    );
    b.case(
        "memory_list_empty",
        &MemoriesListResponse {
            memories: vec![],
            limit: 0,
            count: 0,
        },
    );
    b.finish();
}

#[test]
fn baseline_settings_responses() {
    use openbooklm::api::settings::UserSettingsResponse;

    // US-011 removed `onboarding_state` from this response. It is a
    // hosted-funnel concern and now has its own private endpoint,
    // `GET/PATCH /api/settings/onboarding`, backed by `saas_account_settings`.
    let mut b = Baseline::open("rest/settings.json");
    b.case(
        "settings_default",
        &UserSettingsResponse {
            default_provider: "mistral".into(),
            default_model: "mistral-small-latest".into(),
        },
    );
    b.case(
        "settings_non_default_provider",
        &UserSettingsResponse {
            default_provider: "anthropic".into(),
            default_model: "claude-sonnet-4-6-20260220".into(),
        },
    );
    b.finish();
}

#[test]
fn baseline_chat_responses() {
    use openbooklm::api::chat::types::{
        ChatHistoryResponse, TeachingModeInfo, TeachingModesResponse,
    };
    use openbooklm::llm::{Citation, TeachingMode};
    use openbooklm::services::chat::ChatMessageResponse;

    let citation = Citation {
        source_id: uuid(SOURCE_ID),
        chunk_index: 0,
        text: "Cited passage.".into(),
        relevance_score: 0.91,
        section_header: Some("Introduction".into()),
        page_number: Some(2),
        timestamp_start: None,
        timestamp_end: None,
        video_id: None,
        citation_url: None,
    };
    let media_citation = Citation {
        source_id: uuid(SOURCE_ID),
        chunk_index: 4,
        text: "Spoken passage.".into(),
        relevance_score: 0.66,
        section_header: None,
        page_number: None,
        timestamp_start: Some(12.5),
        timestamp_end: Some(31.0),
        video_id: Some("dQw4w9WgXcQ".into()),
        citation_url: Some("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=12s".into()),
    };

    let user_message = ChatMessageResponse {
        id: uuid(MESSAGE_ID),
        notebook_id: uuid(NOTEBOOK_ID),
        role: "user".into(),
        content: "What does the source say?".into(),
        citations: vec![],
        model: None,
        created_at: CREATED_AT.into(),
        rag_log_id: None,
        feedback: None,
        session_id: None,
    };
    let assistant_message = ChatMessageResponse {
        id: uuid(MESSAGE_ID),
        notebook_id: uuid(NOTEBOOK_ID),
        role: "assistant".into(),
        content: "It says this [1].".into(),
        citations: vec![citation, media_citation],
        model: Some("claude-sonnet-4-6-20260220".into()),
        created_at: UPDATED_AT.into(),
        rag_log_id: Some(uuid(RAG_LOG_ID)),
        feedback: Some("positive".into()),
        session_id: Some(uuid(SESSION_ID)),
    };

    let mut b = Baseline::open("rest/chat.json");
    b.case("message_user", &user_message);
    b.case("message_assistant", &assistant_message);
    b.case(
        "history",
        &ChatHistoryResponse {
            messages: vec![user_message, assistant_message],
            total: 2,
            offset: 0,
            limit: 50,
            has_more: false,
        },
    );
    // `TeachingMode::ALL` drives the fixture, so adding a mode changes the
    // baseline instead of quietly shipping an undocumented mode.
    b.case(
        "teaching_modes",
        &TeachingModesResponse {
            modes: TeachingMode::ALL
                .iter()
                .copied()
                .map(TeachingModeInfo::from)
                .collect(),
            default: "deep",
        },
    );
    assert_eq!(
        TeachingMode::default(),
        TeachingMode::Deep,
        "the teaching-modes endpoint hardcodes `default: \"deep\"`; \
         changing the Rust default without changing the endpoint would drift"
    );
    b.finish();
}

#[test]
fn baseline_rag_log_responses() {
    use openbooklm::api::rag_logs::MetricsResponse;
    use openbooklm::services::rag::rag_log::AggregatedMetrics;

    let populated = AggregatedMetrics {
        total_interactions: 40,
        successful_retrievals: 36,
        avg_context_relevance: Some(0.72),
        avg_answer_faithfulness: Some(0.88),
        positive_feedback: 9,
        negative_feedback: 1,
    };
    let empty = AggregatedMetrics {
        total_interactions: 0,
        successful_retrievals: 0,
        avg_context_relevance: None,
        avg_answer_faithfulness: None,
        positive_feedback: 0,
        negative_feedback: 0,
    };

    let mut b = Baseline::open("rest/rag-logs.json");
    b.case("metrics_populated", &MetricsResponse::from(populated));
    b.case("metrics_empty", &MetricsResponse::from(empty));
    b.finish();
}

/// RFC 7807 failure shapes. These are the contract for every unhappy path, so
/// they are frozen independently of the handler that produces them.
#[test]
fn baseline_problem_details() {
    use openbooklm::error::ProblemDetails;

    let mut b = Baseline::open("rest/problem-details.json");
    for (name, (_status, problem)) in [
        (
            "validation",
            ProblemDetails::validation("title is required"),
        ),
        (
            "unauthorized",
            ProblemDetails::unauthorized("Missing or invalid token"),
        ),
        (
            "forbidden",
            ProblemDetails::forbidden("Model not available on your plan"),
        ),
        ("not_found", ProblemDetails::not_found("Notebook not found")),
        (
            "internal",
            ProblemDetails::internal("Internal server error"),
        ),
        (
            "limit_reached",
            ProblemDetails::limit_reached("Daily message limit reached"),
        ),
        (
            "rate_limited",
            ProblemDetails::rate_limited("Too many requests", 30),
        ),
    ] {
        b.case(name, &problem);
    }
    b.finish();
}

// ============================================================================
// SSE: source processing stream
// ============================================================================

#[test]
fn baseline_source_sse_events() {
    use openbooklm::core::protocol::{EmbeddingProgress, SourceEvent, SourceStatusData};

    let source_id = uuid(SOURCE_ID);

    let variants = vec![
        (
            "status_processing",
            SourceEvent::Status(Box::new(SourceStatusData {
                source_id,
                status: "processing".into(),
                error_message: None,
                progress: Some(EmbeddingProgress {
                    chunks_done: 4,
                    chunks_total: 12,
                }),
            })),
        ),
        (
            "status_error",
            SourceEvent::Status(Box::new(SourceStatusData {
                source_id,
                status: "error".into(),
                error_message: Some("Extraction failed".into()),
                progress: None,
            })),
        ),
        (
            "ready",
            SourceEvent::Ready {
                source_id,
                chunk_count: 12,
                degraded_services: vec![],
            },
        ),
        (
            "ready_degraded",
            SourceEvent::Ready {
                source_id,
                chunk_count: 12,
                degraded_services: vec!["contextualization".into()],
            },
        ),
        (
            "error",
            SourceEvent::Error {
                source_id,
                message: "Unsupported source type".into(),
            },
        ),
        (
            "ocr_started",
            SourceEvent::OcrStarted {
                source_id,
                total_pages: 8,
            },
        ),
        (
            "ocr_progress",
            SourceEvent::OcrProgress {
                source_id,
                current_page: 3,
                total_pages: 8,
            },
        ),
        (
            "ocr_completed",
            SourceEvent::OcrCompleted {
                source_id,
                pages_processed: 8,
            },
        ),
        ("ocr_cache_hit", SourceEvent::OcrCacheHit { source_id }),
        // Stream-level, not source-level: the transport emits it when the
        // replay buffer cannot satisfy `Last-Event-ID` or a subscriber lags.
        ("resync", SourceEvent::Resync { missed: 7 }),
    ];

    // Exhaustiveness: adding a `SourceEvent` variant without a fixture is a
    // compile error here, not a silently missing baseline case.
    for (_, variant) in &variants {
        match variant {
            SourceEvent::Status(_)
            | SourceEvent::Ready { .. }
            | SourceEvent::Error { .. }
            | SourceEvent::OcrStarted { .. }
            | SourceEvent::OcrProgress { .. }
            | SourceEvent::OcrCompleted { .. }
            | SourceEvent::OcrCacheHit { .. }
            | SourceEvent::Resync { .. } => {}
        }
    }

    let mut b = Baseline::open("sse/source.json");
    for (name, variant) in &variants {
        b.case(name, variant);
    }
    b.finish();
}

/// US-009 collapsed the two `SourceEvent` serializations into one (D-005). The
/// enum derive is now what the SSE handler frames, so the payload half of the
/// envelope and the handler's wire form are the same object by construction.
#[test]
fn source_event_has_one_serialization() {
    use openbooklm::core::protocol::SourceEvent;

    for event in [
        SourceEvent::status(uuid(SOURCE_ID), "processing", None),
        SourceEvent::ready(uuid(SOURCE_ID), 12),
        SourceEvent::resync(7),
    ] {
        let envelope = serde_json::to_value(&event).expect("serialize envelope");
        assert_eq!(envelope["event"], event.event_type());
        assert_eq!(envelope["data"], event.payload().expect("payload"));
    }

    // Optional fields stay on the wire as explicit null, which is what the
    // TypeScript `SourceStatusEvent` has always declared.
    let status = SourceEvent::status(uuid(SOURCE_ID), "processing", None)
        .payload()
        .expect("payload");
    assert!(status["error_message"].is_null());
    assert!(status["progress"].is_null());
    assert_eq!(
        SourceEvent::ready(uuid(SOURCE_ID), 12)
            .payload()
            .expect("payload")["degraded_services"],
        json!([])
    );
}

// ============================================================================
// SSE: chat stream
// ============================================================================

#[test]
fn baseline_chat_sse_events() {
    use openbooklm::core::protocol::{ChatEvent, ThinkingStage, WarningKind};
    use openbooklm::llm::Citation;

    let source_id = uuid(SOURCE_ID);

    // Canonical wire payloads, one per event name, built from the typed
    // `ChatEvent` rather than from `json!` literals at each emission site
    // (US-009). The enum is now the only place an event can be defined.
    let variants: Vec<(&str, ChatEvent)> = vec![
        ("chunk", ChatEvent::chunk("It says this ")),
        (
            "thinking_retrieving_context",
            ChatEvent::thinking(ThinkingStage::RetrievingContext),
        ),
        (
            "thinking_reformulating_query",
            ChatEvent::thinking(ThinkingStage::ReformulatingQuery),
        ),
        (
            "thinking_generating",
            ChatEvent::thinking(ThinkingStage::Generating),
        ),
        ("system_history_truncated", ChatEvent::history_truncated(12)),
        (
            "system_history_summarized",
            ChatEvent::history_summarized(8),
        ),
        (
            "warning",
            ChatEvent::warning(WarningKind::LowRetrievalQuality),
        ),
        (
            "citation_after_validation",
            ChatEvent::citation(1, source_id),
        ),
        (
            "citations",
            ChatEvent::citations(vec![Citation {
                source_id,
                chunk_index: 0,
                text: "Cited passage.".into(),
                relevance_score: 0.91,
                section_header: Some("Introduction".into()),
                page_number: Some(2),
                timestamp_start: None,
                timestamp_end: None,
                video_id: None,
                citation_url: None,
            }]),
        ),
        ("metrics", ChatEvent::metrics(Some(0.72))),
        ("metrics_no_context", ChatEvent::metrics(None)),
        (
            "follow_up_suggestions",
            ChatEvent::follow_up_suggestions(vec!["What about the second chapter?".into()]),
        ),
        (
            "done",
            ChatEvent::done(
                "claude-sonnet-4-6-20260220",
                "anthropic",
                Some(uuid(RAG_LOG_ID)),
            ),
        ),
        (
            "done_without_rag_log",
            ChatEvent::done("mistral-small-latest", "mistral", None),
        ),
        ("error", ChatEvent::error("Provider unavailable")),
        ("shutdown", ChatEvent::shutdown("Server shutting down")),
    ];

    // Exhaustiveness: adding a `ChatEvent` variant without a fixture is a
    // compile error here. This replaces the source-scanning guard US-002 needed
    // while chat events were untyped JSON.
    for (_, variant) in &variants {
        match variant {
            ChatEvent::Chunk(_)
            | ChatEvent::Thinking(_)
            | ChatEvent::System(_)
            | ChatEvent::Warning(_)
            | ChatEvent::Citation(_)
            | ChatEvent::Citations(_)
            | ChatEvent::Metrics(_)
            | ChatEvent::FollowUpSuggestions(_)
            | ChatEvent::Done(_)
            | ChatEvent::Error(_)
            | ChatEvent::Shutdown(_) => {}
        }
    }

    // Record the bytes that actually reach the client, not a `Value` round-trip:
    // routing an f32 through `Value` widens 0.72 into 0.7200000286102295.
    let wire = |event: &ChatEvent| -> Value {
        serde_json::from_str(&event.payload_json().expect("encode payload")).expect("valid JSON")
    };

    let mut b = Baseline::open("sse/chat.json");
    for (name, variant) in &variants {
        b.case(name, &wire(variant));
    }
    b.finish();

    // The SSE event name and the serde tag cannot drift apart.
    for (_, variant) in &variants {
        let envelope = serde_json::to_value(variant).expect("serialize envelope");
        assert_eq!(envelope["event"], variant.name());
        assert_eq!(envelope["data"], variant.payload().expect("payload"));
    }
}

/// The `v1` protocol rules that no single fixture can express: exactly one
/// terminal event, `done` last on success, and nothing after termination.
#[test]
fn chat_protocol_v1_termination_rules() {
    use openbooklm::core::protocol::{ChatEvent, EVENT_PROTOCOL_VERSION};

    assert_eq!(EVENT_PROTOCOL_VERSION, "v1");

    for terminal in [
        ChatEvent::done("m", "p", None),
        ChatEvent::error("boom"),
        ChatEvent::shutdown("bye"),
    ] {
        assert!(
            terminal.is_terminal(),
            "{} must be terminal",
            terminal.name()
        );
    }
    for non_terminal in [
        ChatEvent::chunk("x"),
        ChatEvent::metrics(None),
        ChatEvent::follow_up_suggestions(vec!["q?".into()]),
    ] {
        assert!(
            !non_terminal.is_terminal(),
            "{} must not be terminal",
            non_terminal.name()
        );
    }

    let doc = include_str!("../../docs/contracts/sse-protocol-v1.md");
    for rule in [
        "`done` is the terminal successful event",
        "follow_up_suggestions",
        "Last-Event-ID",
        "cancellation",
    ] {
        assert!(
            doc.contains(rule),
            "docs/contracts/sse-protocol-v1.md must document: {rule}"
        );
    }
}

/// A malformed provider stream produces one typed `error` and no `done`.
#[tokio::test]
async fn malformed_provider_stream_emits_one_terminal_error() {
    use openbooklm::core::protocol::{ChatEvent, ChatEventStream};

    let (out, mut rx) = ChatEventStream::channel();
    assert!(out.emit(ChatEvent::chunk("partial")).await);
    assert!(out.emit(ChatEvent::error("Provider unavailable")).await);
    // Whatever the caller does next, the stream is closed to further events.
    assert!(!out.emit(ChatEvent::done("m", "p", None)).await);

    let mut names = Vec::new();
    while let Ok(event) = rx.try_recv() {
        names.push(event.name());
    }
    assert_eq!(names, vec!["chunk", "error"]);
}

// ============================================================================
// Fake provider: core pipeline tests take no external dependency
// ============================================================================

/// Deterministic in-memory `LlmProvider`.
///
/// Proves the provider seam is implementable without a network call or a
/// commercial key, which is what lets EP-004's public CI run the RAG and chat
/// pipelines offline.
struct FakeLlmProvider {
    deltas: Vec<&'static str>,
}

#[async_trait::async_trait]
impl openbooklm::llm::LlmProvider for FakeLlmProvider {
    fn name(&self) -> &str {
        "fake"
    }

    fn default_model(&self) -> &str {
        "fake-model-1"
    }

    fn is_available(&self) -> bool {
        true
    }

    async fn stream_chat(
        &self,
        _system_prompt: &str,
        _messages: Vec<openbooklm::llm::LlmMessage>,
        _model: Option<&str>,
        _documents: &[openbooklm::llm::RagDocument],
        _temperature: Option<f32>,
    ) -> Result<openbooklm::llm::ByteStream, openbooklm::error::AppError> {
        let frames: Vec<Result<bytes::Bytes, openbooklm::error::AppError>> = self
            .deltas
            .iter()
            .map(|text| {
                let payload = json!({ "choices": [{ "delta": { "content": text }, "index": 0 }] });
                Ok(bytes::Bytes::from(format!("data: {payload}\n\n")))
            })
            .chain(std::iter::once(Ok(bytes::Bytes::from_static(
                b"data: [DONE]\n\n",
            ))))
            .collect();
        Ok(Box::pin(futures::stream::iter(frames)))
    }

    fn parse_sse_data(&self, data: &str) -> Option<openbooklm::llm::LlmStreamEvent> {
        openbooklm::llm::parse_openai_sse_data(data)
    }
}

#[tokio::test]
async fn fake_provider_streams_without_external_services() {
    use futures::StreamExt;
    use openbooklm::llm::{LlmProvider, LlmStreamEvent};

    let provider = FakeLlmProvider {
        deltas: vec!["It says ", "this [1]."],
    };
    let mut stream = provider
        .stream_chat("system", vec![], None, &[], None)
        .await
        .unwrap();

    let mut text = String::new();
    let mut saw_done = false;
    while let Some(frame) = stream.next().await {
        let bytes = frame.unwrap();
        for line in String::from_utf8_lossy(&bytes).lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            match provider.parse_sse_data(data) {
                Some(LlmStreamEvent::TextDelta { text: delta }) => text.push_str(&delta),
                Some(LlmStreamEvent::Done) => saw_done = true,
                _ => {}
            }
        }
    }

    assert_eq!(text, "It says this [1].");
    assert!(saw_done, "fake provider must terminate with Done");
}

/// Citation extraction is a core RAG behavior with no external dependency.
/// Freezing it here keeps the seam refactors from changing which `[N]` markers
/// resolve to which chunk.
#[test]
fn citation_extraction_baseline() {
    use openbooklm::llm::{CitableChunk, extract_citations};

    let chunks = vec![
        CitableChunk {
            source_id: uuid(SOURCE_ID),
            generation_id: uuid(SOURCE_ID),
            chunk_index: 0,
            content: "The first synthetic passage.".into(),
            relevance_score: 0.91,
            metadata: Some(json!({
                "section_header": "Introduction",
                "page_number": 2,
                "position": 0,
                "span_start": 0,
                "span_end": "The first synthetic passage.".len(),
            })),
        },
        CitableChunk {
            source_id: uuid(SOURCE_ID),
            generation_id: uuid(SOURCE_ID),
            chunk_index: 1,
            content: "The second synthetic passage.".into(),
            relevance_score: 0.55,
            metadata: Some(json!({
                "position": 1,
                "span_start": 30,
                "span_end": 30 + "The second synthetic passage.".len(),
            })),
        },
    ];

    let citations = extract_citations("First point [1]. Second point [2]. Repeat [1].", &chunks);

    let mut b = Baseline::open("rag/citation-extraction.json");
    b.case("two_markers_with_repeat", &citations);
    b.case(
        "no_markers",
        &extract_citations("No citation markers here.", &chunks),
    );
    b.case(
        "marker_in_code_span",
        &extract_citations("Use `array[1]` for access.", &chunks),
    );
    b.finish();
}
