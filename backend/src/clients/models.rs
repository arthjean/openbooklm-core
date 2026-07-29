//! Provider model catalogs: static lists and API-fetched lists.
//!
//! Used by settings API for listing available models per provider.

use serde::{Deserialize, Serialize};

use crate::error::AppError;

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
            context_window: Some(200_000),
        },
        ModelInfo {
            id: "claude-sonnet-4-6-20260220".into(),
            name: "Claude Sonnet 4.6".into(),
            description: Some("Best for complex agents and coding".into()),
            context_window: Some(200_000),
        },
        ModelInfo {
            id: "claude-haiku-4-5-20251001".into(),
            name: "Claude Haiku 4.5".into(),
            description: Some("Fastest model with near-frontier intelligence".into()),
            context_window: Some(200_000),
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
            context_window: Some(400_000),
        },
        ModelInfo {
            id: "gpt-5-mini".into(),
            name: "GPT-5 mini".into(),
            description: Some("Fast and affordable".into()),
            context_window: Some(400_000),
        },
    ]
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
