//! The non-REST half of the public contract (US-010).
//!
//! OpenAPI describes request and response *shapes*. It does not carry the
//! values a client needs before it can build a valid request: how long a title
//! may be, which teaching modes exist, which source types are supported, which
//! models a provider offers.
//!
//! Those are generated into `contracts/core-constants.json` by
//! `cargo run --bin contracts`, alongside `contracts/openapi.json`, and become
//! typed constants in `packages/sdk-ts`. One generator, two artifacts, zero
//! handwritten duplicates: a limit exists once, in Rust, and every consumer
//! reads it from here.

use serde::{Deserialize, Serialize};

use crate::api::chat::types::{DEFAULT_MAX_CONTEXT_CHUNKS, MAX_CONTEXT_CHUNKS};
use crate::clients::models::{ModelInfo, anthropic_models, mistral_models, openai_models};
use crate::entities::source::SourceStatus;
use crate::llm::TeachingMode;
use crate::repositories::{DEFAULT_CHAT_HISTORY_LIMIT, MAX_CHAT_HISTORY_LIMIT};
use crate::types::SourceType;
use crate::validation::{
    MAX_DESCRIPTION_LENGTH, MAX_MESSAGE_LENGTH, MAX_SYSTEM_PROMPT_LENGTH, MAX_TITLE_LENGTH,
    VALID_PROVIDERS,
};

/// Server-enforced input bounds. A client that respects these never sees a 400.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationLimits {
    /// Notebooks, sources and notes.
    pub max_title_length: usize,
    pub max_description_length: usize,
    pub max_system_prompt_length: usize,
    pub max_message_length: usize,
    /// Retrieved chunks per chat request. Values outside the range are clamped
    /// server-side, not rejected.
    pub default_max_context_chunks: i32,
    pub max_context_chunks: i32,
    /// Chat history pagination. `limit` is clamped, not rejected.
    pub default_chat_history_limit: u64,
    pub max_chat_history_limit: u64,
}

impl Default for ValidationLimits {
    fn default() -> Self {
        Self::current()
    }
}

impl ValidationLimits {
    #[must_use]
    pub const fn current() -> Self {
        Self {
            max_title_length: MAX_TITLE_LENGTH,
            max_description_length: MAX_DESCRIPTION_LENGTH,
            max_system_prompt_length: MAX_SYSTEM_PROMPT_LENGTH,
            max_message_length: MAX_MESSAGE_LENGTH,
            default_max_context_chunks: DEFAULT_MAX_CONTEXT_CHUNKS,
            max_context_chunks: MAX_CONTEXT_CHUNKS,
            default_chat_history_limit: DEFAULT_CHAT_HISTORY_LIMIT,
            max_chat_history_limit: MAX_CHAT_HISTORY_LIMIT,
        }
    }
}

/// One teaching mode, with the presentation metadata the API already returns
/// from `GET /api/teaching-modes`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeachingModeEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub icon: &'static str,
    pub description: &'static str,
    pub is_default: bool,
}

/// The models a provider exposes, and whether the core can cite from them
/// natively.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub provider: &'static str,
    /// Whether the provider returns structured citations, as opposed to the
    /// core extracting `[N]` markers from the answer text.
    pub native_citations: bool,
    /// Stable models supported by the public client contract. A provider's
    /// runtime discovery endpoint may return additional models.
    pub models: Vec<ModelInfo>,
}

/// Everything a client needs that is not a request or response shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreCatalog {
    /// Version of the SSE event protocol these constants belong to.
    pub event_protocol_version: &'static str,
    pub validation: ValidationLimits,
    pub source_types: Vec<&'static str>,
    pub source_statuses: Vec<&'static str>,
    pub teaching_modes: Vec<TeachingModeEntry>,
    pub default_teaching_mode: &'static str,
    pub providers: Vec<ProviderCapabilities>,
}

/// Build the catalog from the live Rust definitions.
///
/// Every list is derived from an exhaustive `match` or a `const` in the module
/// that owns it, so adding a source type or teaching mode without regenerating
/// the contract is a `check-contracts.sh` failure rather than a silent omission.
#[must_use]
pub fn catalog() -> CoreCatalog {
    let default_mode = TeachingMode::default();
    let teaching_modes = [
        TeachingMode::Flash,
        TeachingMode::Deep,
        TeachingMode::Quiz,
        TeachingMode::Glossary,
        TeachingMode::Summary,
        TeachingMode::Timeline,
    ]
    .into_iter()
    .map(|mode| TeachingModeEntry {
        id: teaching_mode_id(mode),
        name: mode.display_name(),
        icon: mode.icon(),
        description: mode.description(),
        is_default: mode == default_mode,
    })
    .collect();

    CoreCatalog {
        event_protocol_version: super::protocol::EVENT_PROTOCOL_VERSION,
        validation: ValidationLimits::current(),
        source_types: [
            SourceType::Pdf,
            SourceType::Text,
            SourceType::Markdown,
            SourceType::Web,
            SourceType::Docx,
            SourceType::Epub,
            SourceType::Youtube,
        ]
        .iter()
        .map(SourceType::as_str)
        .collect(),
        source_statuses: [
            SourceStatus::Pending,
            SourceStatus::Processing,
            SourceStatus::Contextualizing,
            SourceStatus::Embedding,
            SourceStatus::Ready,
            SourceStatus::Error,
        ]
        .iter()
        .map(SourceStatus::as_str)
        .collect(),
        teaching_modes,
        default_teaching_mode: teaching_mode_id(default_mode),
        providers: provider_capabilities(),
    }
}

/// The wire identifier of a teaching mode, matching its serde representation.
const fn teaching_mode_id(mode: TeachingMode) -> &'static str {
    match mode {
        TeachingMode::Flash => "flash",
        TeachingMode::Deep => "deep",
        TeachingMode::Quiz => "quiz",
        TeachingMode::Glossary => "glossary",
        TeachingMode::Summary => "summary",
        TeachingMode::Timeline => "timeline",
    }
}

/// Provider catalogue, ordered to match [`VALID_PROVIDERS`].
///
fn provider_capabilities() -> Vec<ProviderCapabilities> {
    VALID_PROVIDERS
        .iter()
        .map(|provider| match *provider {
            "mistral" => ProviderCapabilities {
                provider: "mistral",
                native_citations: false,
                models: mistral_models(),
            },
            "anthropic" => ProviderCapabilities {
                provider: "anthropic",
                native_citations: true,
                models: anthropic_models(),
            },
            "openai" => ProviderCapabilities {
                provider: "openai",
                native_citations: false,
                models: openai_models(),
            },
            other => ProviderCapabilities {
                provider: other,
                native_citations: false,
                models: Vec::new(),
            },
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_generation_is_deterministic() {
        assert_eq!(catalog(), catalog());
    }

    #[test]
    fn every_provider_named_by_validation_is_described() {
        let described: Vec<&str> = catalog().providers.iter().map(|p| p.provider).collect();
        assert_eq!(described, VALID_PROVIDERS.to_vec());
    }

    #[test]
    fn mistral_defaults_are_part_of_the_public_contract() {
        let c = catalog();
        let mistral = c
            .providers
            .iter()
            .find(|provider| provider.provider == "mistral")
            .expect("Mistral provider capabilities");
        let model_ids: Vec<&str> = mistral
            .models
            .iter()
            .map(|model| model.id.as_str())
            .collect();

        assert_eq!(
            model_ids,
            vec!["mistral-small-latest", "mistral-large-latest"]
        );
        assert_eq!(mistral.models[0].context_window, Some(32_768));
        assert_eq!(mistral.models[1].context_window, Some(131_072));
    }

    #[test]
    fn teaching_mode_ids_match_the_endpoint() {
        let catalog_ids: Vec<&str> = catalog().teaching_modes.iter().map(|m| m.id).collect();
        let endpoint_ids: Vec<&str> = [
            TeachingMode::Flash,
            TeachingMode::Deep,
            TeachingMode::Quiz,
            TeachingMode::Glossary,
            TeachingMode::Summary,
            TeachingMode::Timeline,
        ]
        .into_iter()
        .map(|mode| crate::api::chat::types::TeachingModeInfo::from(mode).id)
        .collect();
        assert_eq!(catalog_ids, endpoint_ids);
    }

    #[test]
    fn exactly_one_teaching_mode_is_default() {
        let c = catalog();
        let defaults: Vec<&str> = c
            .teaching_modes
            .iter()
            .filter(|m| m.is_default)
            .map(|m| m.id)
            .collect();
        assert_eq!(defaults, vec![c.default_teaching_mode]);
    }

    #[test]
    fn source_type_and_status_sets_are_complete() {
        let c = catalog();
        assert_eq!(
            c.source_types,
            vec!["pdf", "text", "markdown", "web", "docx", "epub", "youtube"]
        );
        // `ocr` is deliberately absent: it is an OCR *event*, never a stored
        // status. The TypeScript union used to declare it (drift D-011).
        assert_eq!(
            c.source_statuses,
            vec![
                "pending",
                "processing",
                "contextualizing",
                "embedding",
                "ready",
                "error"
            ]
        );
    }

    #[test]
    fn limits_match_the_validators_that_enforce_them() {
        let limits = ValidationLimits::current();
        let too_long = "a".repeat(limits.max_message_length + 1);
        assert!(crate::validation::validate_message(&too_long).is_err());
        assert!(
            crate::validation::validate_message(&"a".repeat(limits.max_message_length)).is_ok()
        );

        let long_title = "a".repeat(limits.max_title_length + 1);
        assert!(crate::validation::validate_title(&long_title).is_err());
        assert!(crate::validation::validate_title(&"a".repeat(limits.max_title_length)).is_ok());
    }
}
