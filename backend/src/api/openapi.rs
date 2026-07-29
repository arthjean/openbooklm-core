//! The public REST contract (US-010).
//!
//! `contracts/openapi.json` is generated from this document by
//! `cargo run --bin contracts`, and `packages/sdk-ts` is generated from that
//! file. The Rust handler annotations are the only place any of it is written
//! by hand.
//!
//! Only **core** routes appear here. Billing, webhooks, feedback,
//! micro-feedback, newsletter and stats are SaaS surfaces: they stay out of the
//! public contract and out of the SDK, which is what
//! `scripts/check-open-core-boundary.sh` classifies them as.
//!
//! SSE endpoints are listed with their payload schema and `text/event-stream`
//! content type. OpenAPI cannot express event ordering, replay or terminal
//! events, so those rules live in `docs/contracts/sse-protocol-v1.md` and the
//! response descriptions point at it.

use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::{Modify, OpenApi};

/// Registers the bearer scheme the `security(("bearer_auth" = []))` annotations
/// reference.
///
/// The core contract states only that a bearer token is required. Which
/// identity provider issues it is an adapter concern: Clerk for the hosted
/// product, a static token or loopback single-user mode for the reference
/// server. Naming a provider here would leak a SaaS decision into the public
/// contract.
struct BearerAuth;

impl Modify for BearerAuth {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "bearer_auth",
                SecurityScheme::Http(
                    HttpBuilder::new()
                        .scheme(HttpAuthScheme::Bearer)
                        .bearer_format("JWT")
                        .description(Some(
                            "Bearer token verified by the composed identity adapter.",
                        ))
                        .build(),
                ),
            );
        }
    }
}

#[derive(OpenApi)]
#[openapi(
    info(
        title = "OpenbookLM Core API",
        description = "The public core of OpenbookLM: notebooks, sources, chunks, notes, \
                       memories, chat and retrieval metrics. Event-stream semantics are \
                       specified separately in docs/contracts/sse-protocol-v1.md.",
        license(name = "Apache-2.0", url = "https://www.apache.org/licenses/LICENSE-2.0"),
        // Taken from the crate rather than written here: a hardcoded version
        // drifts the moment the crate is bumped, and the release gate only
        // checks the tag against the manifest and the SDK, so a contract
        // claiming a version nobody published would pass unnoticed.
        version = env!("CARGO_PKG_VERSION")
    ),
    modifiers(&BearerAuth),
    tags(
        (name = "notebooks", description = "Notebook lifecycle"),
        (name = "sources", description = "Source ingestion, chunks and the processing event stream"),
        (name = "notes", description = "Notes saved from chat or written directly"),
        (name = "memories", description = "Per-notebook long-term memory"),
        (name = "chat", description = "Cited chat over a notebook's sources"),
        (name = "rag-logs", description = "Retrieval feedback and aggregated quality metrics"),
        (name = "settings", description = "Core account preferences"),
        (name = "suggestions", description = "Suggested starter questions"),
        (name = "health", description = "Liveness and dependency health"),
    ),
    paths(
        crate::api::notebooks::list_notebooks_handler,
        crate::api::notebooks::create_notebook_handler,
        crate::api::notebooks::get_notebook_handler,
        crate::api::notebooks::update_notebook_handler,
        crate::api::notebooks::delete_notebook_handler,
        crate::api::sources::list_sources_handler,
        crate::api::sources::create_source_handler,
        crate::api::sources::get_source_handler,
        crate::api::sources::get_source_chunks_handler,
        crate::api::sources::delete_source_handler,
        crate::api::sources::reprocess_source_handler,
        crate::api::sources::youtube_title_handler,
        crate::api::sources::source_events_handler,
        crate::api::notes::list_notes_handler,
        crate::api::notes::create_note_handler,
        crate::api::notes::get_note_handler,
        crate::api::notes::update_note_handler,
        crate::api::notes::delete_note_handler,
        crate::api::memory::list_memories_handler,
        crate::api::memory::delete_all_memories_handler,
        crate::api::memory::get_memory_handler,
        crate::api::memory::update_memory_handler,
        crate::api::memory::delete_memory_handler,
        crate::api::chat::send_message_handler,
        crate::api::chat::get_chat_history_handler,
        crate::api::chat::clear_chat_history_handler,
        crate::api::chat::list_teaching_modes,
        crate::api::rag_logs::update_feedback_handler,
        crate::api::rag_logs::get_notebook_metrics_handler,
        crate::api::rag_logs::get_user_metrics_handler,
        crate::api::settings::get_settings_handler,
        crate::api::settings::update_settings_handler,
        crate::api::suggestions::get_suggestions_handler,
        crate::api::health::health_check,
        crate::api::health::detailed_health_check,
    ),
    components(schemas(
        crate::error::ProblemDetails,
        crate::core::protocol::ChatEvent,
        crate::core::protocol::SourceEvent,
    ))
)]
pub struct CoreApi;

/// The generated document, with the OpenAPI version and component ordering the
/// generator writes to `contracts/openapi.json`.
#[must_use]
pub fn document() -> utoipa::openapi::OpenApi {
    CoreApi::openapi()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The routes the private SaaS adds on top of the core. None of them may
    /// appear in the public contract, because the SDK generated from it is
    /// published.
    ///
    /// The complementary assertion — that no commercial vendor is *named*
    /// anywhere in the document — lives in
    /// `packages/sdk-ts/test/contract.test.ts`. It cannot live here:
    /// `check-open-core-boundary.sh` greps public Rust for vendor names, and
    /// would read the list of names being forbidden as a dependency on them.
    const SAAS_PATH_FRAGMENTS: &[&str] = &[
        "/api/billing",
        "/api/webhooks",
        "/api/feedback",
        "/api/micro-feedback",
        "/api/public/newsletter",
        "/api/public/stats",
        "/api/settings/onboarding",
    ];

    #[test]
    fn contract_excludes_every_saas_route() {
        let doc = document();
        for path in doc.paths.paths.keys() {
            for fragment in SAAS_PATH_FRAGMENTS {
                assert!(
                    !path.starts_with(fragment),
                    "SaaS route {path} leaked into the public contract"
                );
            }
        }
    }

    #[test]
    fn every_core_route_is_documented() {
        let doc = document();
        // One entry per distinct path template registered in `main.rs` for the
        // core surface. A route added to the router without an annotation fails
        // here rather than silently missing from the SDK.
        let expected = [
            "/api/notebooks",
            "/api/notebooks/{id}",
            "/api/notebooks/{id}/chat",
            "/api/notebooks/{id}/memories",
            "/api/notebooks/{id}/metrics",
            "/api/notebooks/{id}/suggestions",
            "/api/notebooks/{notebook_id}/notes",
            "/api/notebooks/{notebook_id}/sources",
            "/api/notebooks/{notebook_id}/sources/events",
            "/api/memories/{id}",
            "/api/sources/{id}",
            "/api/sources/{id}/chunks",
            "/api/sources/{id}/reprocess",
            "/api/notes/{id}",
            "/api/rag-logs/{id}/feedback",
            "/api/metrics",
            "/api/settings",
            "/api/teaching-modes",
            "/api/youtube/title",
            "/health",
            "/health/detailed",
        ];
        for path in expected {
            assert!(
                doc.paths.paths.contains_key(path),
                "core route {path} is missing from the OpenAPI document"
            );
        }
        assert_eq!(
            doc.paths.paths.len(),
            expected.len(),
            "the document has paths not listed in this test: {:?}",
            doc.paths
                .paths
                .keys()
                .filter(|p| !expected.contains(&p.as_str()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn sse_routes_declare_the_event_stream_content_type() {
        let doc = document();
        for path in [
            "/api/notebooks/{id}/chat",
            "/api/notebooks/{notebook_id}/sources/events",
        ] {
            let item = doc.paths.paths.get(path).expect("path present");
            let encoded = serde_json::to_string(item).expect("serialize path item");
            assert!(
                encoded.contains("text/event-stream"),
                "{path} must declare text/event-stream"
            );
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let first = serde_json::to_string(&document()).expect("serialize");
        let second = serde_json::to_string(&document()).expect("serialize");
        assert_eq!(
            first, second,
            "two consecutive generations must be byte-equal"
        );
    }
}
