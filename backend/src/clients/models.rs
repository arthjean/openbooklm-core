//! Provider model catalogs: static lists and API-fetched lists.
//!
//! Used by settings API for listing available models per provider.

use serde::{Deserialize, Serialize};

use crate::error::AppError;

pub const ANTHROPIC_CONTEXT_WINDOW: u32 = 200_000;
pub const OPENAI_GPT5_CONTEXT_WINDOW: u32 = 400_000;
pub const MISTRAL_SMALL_CONTEXT_WINDOW: u32 = 32_768;
pub const MISTRAL_LARGE_CONTEXT_WINDOW: u32 = 131_072;

/// Model info returned from provider APIs.
#[derive(Debug, Clone, Serialize, PartialEq, Eq, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
}

/// Static Anthropic models (no public list API).
pub fn anthropic_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "claude-opus-4-6-20260220".into(),
            name: "Claude Opus 4.6".into(),
            description: Some("Most capable model for complex tasks".into()),
            context_window: Some(ANTHROPIC_CONTEXT_WINDOW),
        },
        ModelInfo {
            id: "claude-sonnet-4-6-20260220".into(),
            name: "Claude Sonnet 4.6".into(),
            description: Some("Best for complex agents and coding".into()),
            context_window: Some(ANTHROPIC_CONTEXT_WINDOW),
        },
        ModelInfo {
            id: "claude-haiku-4-5-20251001".into(),
            name: "Claude Haiku 4.5".into(),
            description: Some("Fastest model with near-frontier intelligence".into()),
            context_window: Some(ANTHROPIC_CONTEXT_WINDOW),
        },
    ]
}

/// Static OpenAI models.
pub fn openai_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "gpt-5.2".into(),
            name: "GPT-5.2".into(),
            description: Some("Advanced reasoning".into()),
            context_window: Some(OPENAI_GPT5_CONTEXT_WINDOW),
        },
        ModelInfo {
            id: "gpt-5-mini".into(),
            name: "GPT-5 mini".into(),
            description: Some("Fast and affordable".into()),
            context_window: Some(OPENAI_GPT5_CONTEXT_WINDOW),
        },
    ]
}

/// Stable Mistral models exposed through the public client contract.
///
/// The settings API can still discover additional models from Mistral at
/// runtime. This subset gives clients deterministic defaults without an API
/// key or network request.
pub fn mistral_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "mistral-small-latest".into(),
            name: "Mistral Small".into(),
            description: Some("Fast and efficient".into()),
            context_window: Some(MISTRAL_SMALL_CONTEXT_WINDOW),
        },
        ModelInfo {
            id: "mistral-large-latest".into(),
            name: "Mistral Large".into(),
            description: Some("Advanced reasoning".into()),
            context_window: Some(MISTRAL_LARGE_CONTEXT_WINDOW),
        },
    ]
}

/// Resolve the context window declared by the static public model catalog.
///
/// Returning `None` for dynamically discovered or unknown models keeps callers
/// from inventing a window that the provider may reject.
#[must_use]
pub fn context_window_for_model(provider: &str, model: &str) -> Option<u32> {
    models_for_provider(provider)?
        .into_iter()
        .find(|candidate| candidate.id == model)
        .and_then(|candidate| candidate.context_window)
}

/// Model identifiers supported by a provider in this build.
///
/// Provider metadata and request budgeting both derive from the same catalog,
/// so a model cannot be advertised without a declared window.
#[must_use]
pub fn model_ids_for_provider(provider: &str) -> Vec<String> {
    models_for_provider(provider)
        .unwrap_or_default()
        .into_iter()
        .map(|model| model.id)
        .collect()
}

fn models_for_provider(provider: &str) -> Option<Vec<ModelInfo>> {
    match provider {
        "anthropic" => Some(anthropic_models()),
        "openai" => Some(openai_models()),
        "mistral" => Some(mistral_models()),
        _ => None,
    }
}

/// Format a model ID into a human-readable name.
pub fn format_model_name(id: &str) -> String {
    let name = id
        .replace("-latest", "")
        .replace("-preview", " Preview")
        .replace("mistral-", "Mistral ")
        .replace("open-", "Open ")
        .replace("codestral-", "Codestral ")
        .replace("pixtral-", "Pixtral ")
        .replace("gpt-4-", "GPT-4 ")
        .replace("gpt-3.5-", "GPT-3.5 ")
        .replace("o1-", "o1 ")
        .replace("o3-", "o3 ");

    name.split('-')
        .map(capitalize_first)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Capitalize first letter of a word.
fn capitalize_first(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().chain(chars).collect(),
    }
}

// =============================================================================
// API-FETCHED MODELS
// =============================================================================

/// Fetch available models from a provider's API.
pub async fn fetch_provider_models(
    client: &reqwest::Client,
    provider: &str,
    api_key: &str,
) -> Result<Vec<ModelInfo>, AppError> {
    match provider {
        "mistral" => fetch_mistral_models(client, api_key).await,
        "anthropic" => Ok(anthropic_models()),
        "openai" => Ok(openai_models()),
        _ => Err(AppError::Validation(format!(
            "Unknown provider: {provider}"
        ))),
    }
}

/// Make authenticated GET request to a provider API.
async fn provider_api_get<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    provider: &str,
) -> Result<T, AppError> {
    let response = client
        .get(url)
        .header("Authorization", format!("Bearer {api_key}"))
        .send()
        .await
        .map_err(|e| AppError::ProviderError(format!("Failed to fetch {provider} models: {e}")))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(AppError::ProviderError(format!(
            "{provider} API error ({status}): {body}"
        )));
    }

    response
        .json()
        .await
        .map_err(|e| AppError::ProviderError(format!("Failed to parse {provider} response: {e}")))
}

/// Fetch models from Mistral API.
async fn fetch_mistral_models(
    client: &reqwest::Client,
    api_key: &str,
) -> Result<Vec<ModelInfo>, AppError> {
    #[derive(Deserialize)]
    struct Response {
        data: Vec<Model>,
    }
    #[derive(Deserialize)]
    struct Model {
        id: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        max_context_length: Option<u32>,
    }

    let data: Response = provider_api_get(
        client,
        "https://api.mistral.ai/v1/models",
        api_key,
        "Mistral",
    )
    .await?;

    Ok(data
        .data
        .into_iter()
        .filter(|m| !m.id.contains("embed") && !m.id.contains("moderation"))
        .map(|m| ModelInfo {
            name: format_model_name(&m.id),
            id: m.id,
            description: m.description,
            context_window: m.max_context_length,
        })
        .collect())
}
