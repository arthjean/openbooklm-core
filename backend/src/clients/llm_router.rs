//! LLM Router for provider selection and fallback
//!
//! The router manages multiple LLM providers and handles:
//! - Provider selection by name
//! - Default provider configuration
//! - Fallback to available providers when preferred is unavailable
//! - Provider availability checking via circuit breaker state

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::config::CoreConfig;
use crate::error::{AppError, LlmError};
use crate::llm::{LlmProvider, ProviderInfo};

use super::anthropic::AnthropicClient;
use super::metrics::ClientMetrics;
use super::mistral::MistralClient;
use super::openai::OpenAiClient;

/// Provider priority order for default selection and fallback
const PROVIDER_PRIORITY: &[&str] = &["mistral", "anthropic", "openai"];

/// Result of provider selection, possibly with a model override for fallback.
pub struct ProviderSelection {
    /// The selected LLM provider.
    pub provider: Arc<dyn LlmProvider>,
    /// When falling back, the model is remapped to an equivalent on the target
    /// provider. `None` means "use whatever the caller requested or the
    /// provider's default".
    pub model_override: Option<String>,
    /// True when a different provider was selected due to the requested one
    /// being unavailable.
    pub is_fallback: bool,
}

impl std::fmt::Debug for ProviderSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderSelection")
            .field("provider", &self.provider.name())
            .field("model_override", &self.model_override)
            .field("is_fallback", &self.is_fallback)
            .finish()
    }
}

/// Router for selecting and managing LLM providers
#[derive(Clone)]
pub struct LlmRouter {
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    default_provider: String,
}

impl std::fmt::Debug for LlmRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmRouter")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .field("default_provider", &self.default_provider)
            .finish()
    }
}

impl LlmRouter {
    /// Route every request to one provider.
    ///
    /// Used by the deterministic composition (US-020), where routing, failover
    /// and priority have nothing to choose between.
    pub fn single(provider: Arc<dyn LlmProvider>) -> Self {
        let name = provider.name().to_owned();
        let mut providers: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        providers.insert(name.clone(), provider);
        Self {
            providers,
            default_provider: name,
        }
    }

    /// Create a new LlmRouter from application config
    ///
    /// Initializes all configured providers and sets the default based on priority.
    pub fn from_config(config: &CoreConfig, metrics: &ClientMetrics) -> Self {
        let mut providers: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();

        // Initialize providers - each logs its own success
        if let Ok(client) = MistralClient::from_config(config, metrics) {
            tracing::info!("Mistral LLM provider initialized");
            providers.insert("mistral".to_string(), Arc::new(client));
        }

        if let Ok(client) = AnthropicClient::from_config(config, metrics) {
            tracing::info!("Anthropic LLM provider initialized");
            providers.insert("anthropic".to_string(), Arc::new(client));
        }

        if let Ok(client) = OpenAiClient::from_config(config, metrics) {
            tracing::info!("OpenAI LLM provider initialized");
            providers.insert("openai".to_string(), Arc::new(client));
        }

        // Select default provider by priority
        let default_provider = PROVIDER_PRIORITY
            .iter()
            .find(|&&name| providers.contains_key(name))
            .map(|&s| s.to_string())
            .unwrap_or_default();

        if default_provider.is_empty() {
            tracing::warn!("No LLM providers configured!");
        } else {
            tracing::info!(
                default = %default_provider,
                available = ?providers.keys().collect::<Vec<_>>(),
                "LLM router initialized"
            );
        }

        Self {
            providers,
            default_provider,
        }
    }

    /// Create a new LlmRouter with custom providers
    pub fn new(
        providers: HashMap<String, Arc<dyn LlmProvider>>,
        default_provider: impl Into<String>,
    ) -> Self {
        Self {
            providers,
            default_provider: default_provider.into(),
        }
    }

    /// Get a provider by name (or default), with automatic fallback if unavailable.
    ///
    /// When the requested provider is unavailable and a fallback is used, the
    /// returned `ProviderSelection` includes a `model_override` with an
    /// equivalent model for the fallback provider (e.g. Anthropic models map
    /// to `"mistral-large-latest"` when falling back to Mistral).
    pub fn get_provider(
        &self,
        name: Option<&str>,
        requested_model: Option<&str>,
    ) -> Result<ProviderSelection, AppError> {
        let provider_name = name.unwrap_or(&self.default_provider);

        if provider_name.is_empty() {
            return Err(LlmError::ProviderNotConfigured {
                provider: "any".to_string(),
            }
            .into());
        }

        let provider =
            self.providers
                .get(provider_name)
                .ok_or_else(|| LlmError::ProviderNotConfigured {
                    provider: provider_name.to_string(),
                })?;

        if provider.is_available() {
            return Ok(ProviderSelection {
                provider: provider.clone(),
                model_override: None,
                is_fallback: false,
            });
        }

        // Provider unavailable — try fallback with model mapping
        if let Some(fallback) = self.find_available_fallback(provider_name) {
            let fallback_model =
                map_model_for_fallback(provider_name, requested_model, fallback.name());

            tracing::info!(
                requested_provider = provider_name,
                fallback_provider = fallback.name(),
                requested_model = requested_model.unwrap_or("default"),
                fallback_model = %fallback_model,
                "Falling back to alternate provider with mapped model"
            );

            return Ok(ProviderSelection {
                provider: fallback,
                model_override: Some(fallback_model),
                is_fallback: true,
            });
        }

        Err(LlmError::RateLimited {
            provider: provider_name.to_string(),
            retry_after: 30,
        }
        .into())
    }

    /// Get a specific provider without fallback
    pub fn get_provider_exact(&self, name: &str) -> Result<Arc<dyn LlmProvider>, AppError> {
        self.providers.get(name).cloned().ok_or_else(|| {
            LlmError::ProviderNotConfigured {
                provider: name.to_string(),
            }
            .into()
        })
    }

    /// Find an available fallback provider (excluding the specified one)
    fn find_available_fallback(&self, exclude: &str) -> Option<Arc<dyn LlmProvider>> {
        // First try priority-ordered providers
        let from_priority = PROVIDER_PRIORITY.iter().filter_map(|&name| {
            (name != exclude)
                .then(|| self.providers.get(name))
                .flatten()
                .filter(|p| p.is_available())
                .cloned()
        });

        // Then try any other available provider
        let from_any = self
            .providers
            .values()
            .filter(|p| p.name() != exclude && p.is_available())
            .cloned();

        from_priority.chain(from_any).next()
    }

    /// Get the default provider name
    pub fn default_provider_name(&self) -> &str {
        &self.default_provider
    }

    /// Check if any provider is available
    pub fn has_available_provider(&self) -> bool {
        self.providers.values().any(|p| p.is_available())
    }

    /// List all configured provider names
    pub fn provider_names(&self) -> Vec<&str> {
        self.providers.keys().map(String::as_str).collect()
    }

    /// Get information about all providers
    pub fn provider_info(&self) -> Vec<ProviderInfo> {
        self.providers.values().map(|p| p.info()).collect()
    }

    /// Get the number of configured providers
    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

/// Map a model name to an equivalent on the fallback provider.
///
/// When the original provider is down and we fall back, we can't send
/// e.g. `"claude-sonnet-4-6-20260220"` to Mistral. This function picks
/// a reasonable equivalent:
///
/// - **Anthropic → Mistral**: any Claude model → `"mistral-large-latest"`
/// - **Mistral → Anthropic**: any Mistral model → `"claude-haiku-4-5-20251001"` (cheapest)
/// - **Unknown**: use the fallback provider's default model
fn map_model_for_fallback(
    _original_provider: &str,
    _requested_model: Option<&str>,
    fallback_provider: &str,
) -> String {
    match fallback_provider {
        "mistral" => "mistral-large-latest".to_string(),
        "anthropic" => "claude-haiku-4-5-20251001".to_string(),
        "openai" => "gpt-5-mini".to_string(),
        _ => fallback_provider.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ByteStream, LlmMessage, LlmStreamEvent};

    /// Stub provider for testing router logic without real HTTP clients.
    struct StubProvider {
        name: &'static str,
        available: bool,
        default_model: &'static str,
    }

    #[async_trait::async_trait]
    impl LlmProvider for StubProvider {
        fn name(&self) -> &str {
            self.name
        }
        fn default_model(&self) -> &str {
            self.default_model
        }
        fn is_available(&self) -> bool {
            self.available
        }
        async fn stream_chat(
            &self,
            _system_prompt: &str,
            _messages: Vec<LlmMessage>,
            _model: Option<&str>,
            _documents: &[crate::llm::RagDocument],
            _temperature: Option<f32>,
        ) -> Result<ByteStream, AppError> {
            unimplemented!("stub")
        }
        fn parse_sse_data(&self, _data: &str) -> Option<LlmStreamEvent> {
            None
        }
    }

    #[test]
    fn fallback_maps_anthropic_model_to_mistral() {
        let mut providers: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        providers.insert(
            "anthropic".into(),
            Arc::new(StubProvider {
                name: "anthropic",
                available: false,
                default_model: "claude-sonnet-4-6-20260220",
            }),
        );
        providers.insert(
            "mistral".into(),
            Arc::new(StubProvider {
                name: "mistral",
                available: true,
                default_model: "mistral-small-latest",
            }),
        );

        let router = LlmRouter::new(providers, "anthropic");
        let selection = router
            .get_provider(Some("anthropic"), Some("claude-sonnet-4-6-20260220"))
            .expect("should fallback");

        assert!(selection.is_fallback);
        assert_eq!(selection.provider.name(), "mistral");
        assert_eq!(
            selection.model_override.as_deref(),
            Some("mistral-large-latest")
        );
    }

    #[test]
    fn no_fallback_when_provider_available() {
        let mut providers: HashMap<String, Arc<dyn LlmProvider>> = HashMap::new();
        providers.insert(
            "anthropic".into(),
            Arc::new(StubProvider {
                name: "anthropic",
                available: true,
                default_model: "claude-sonnet-4-6-20260220",
            }),
        );

        let router = LlmRouter::new(providers, "anthropic");
        let selection = router
            .get_provider(Some("anthropic"), Some("claude-sonnet-4-6-20260220"))
            .expect("should succeed");

        assert!(!selection.is_fallback);
        assert!(selection.model_override.is_none());
        assert_eq!(selection.provider.name(), "anthropic");
    }
}
