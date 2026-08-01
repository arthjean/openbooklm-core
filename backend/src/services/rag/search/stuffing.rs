//! Context stuffing thresholds.
//!
//! A notebook small enough to fit inside the requested context limit is loaded
//! whole instead of being searched. How small "small enough" is depends on what
//! the generation model costs per input token, which is the table below.
//!
//! The threshold is a ceiling on candidates, not a licence to exceed the
//! caller's request: the pipeline takes the minimum of this value and the
//! requested context limit, and the loaded chunks go through the same
//! diversification and selection as searched ones (US-013).

/// Compute the effective context stuffing threshold for a given model.
///
/// Returns the maximum number of chunks that can be loaded directly into the
/// LLM context (bypassing embed → search → rerank). The threshold is the
/// **minimum** of the per-model default and the global override.
///
/// When `global_override` is 0, stuffing is disabled for all models.
/// Unknown models default to 0 (no stuffing).
#[must_use]
pub fn max_context_stuffing_chunks(provider: &str, model: &str, global_override: i32) -> i64 {
    if global_override == 0 {
        return 0;
    }

    let per_model: i32 = match provider {
        // Tier 1 — stuff aggressively (input < $0.25/M)
        "mistral" if model.starts_with("mistral-small-") => 95,
        "openai" if model.starts_with("gpt-5-mini") => 150,
        // Tier 2 — stuff moderately (input $0.50-$1/M)
        "mistral" if model.starts_with("mistral-large-") => 80,
        "anthropic" if model.starts_with("claude-haiku-4-5-") => 50,
        // Tier 3 — stuff minimally (input $1.75+/M)
        "openai" if model.starts_with("gpt-5.2") => 30,
        "anthropic" if model.starts_with("claude-sonnet-4-6-") => 30,
        "anthropic" if model.starts_with("claude-opus-4-6-") => 0, // never stuff, too expensive
        // Unknown models: no stuffing
        _ => 0,
    };

    i64::from(per_model.min(global_override))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stuffing_disabled_when_global_override_zero() {
        assert_eq!(
            max_context_stuffing_chunks("mistral", "mistral-small-latest", 0),
            0
        );
    }

    #[test]
    fn stuffing_tier1_aggressive() {
        assert_eq!(
            max_context_stuffing_chunks("mistral", "mistral-small-latest", 150),
            95
        );
        assert_eq!(
            max_context_stuffing_chunks("openai", "gpt-5-mini", 150),
            150
        );
    }

    #[test]
    fn stuffing_tier2_moderate() {
        assert_eq!(
            max_context_stuffing_chunks("mistral", "mistral-large-latest", 150),
            80
        );
        assert_eq!(
            max_context_stuffing_chunks("anthropic", "claude-haiku-4-5-20251001", 150),
            50
        );
    }

    #[test]
    fn stuffing_tier3_minimal() {
        assert_eq!(
            max_context_stuffing_chunks("openai", "gpt-5.2-turbo", 150),
            30
        );
        assert_eq!(
            max_context_stuffing_chunks("anthropic", "claude-sonnet-4-6-20260220", 150),
            30
        );
    }

    #[test]
    fn stuffing_opus_always_zero() {
        assert_eq!(
            max_context_stuffing_chunks("anthropic", "claude-opus-4-6-20260220", 150),
            0
        );
        assert_eq!(
            max_context_stuffing_chunks("anthropic", "claude-opus-4-6-20260220", 500),
            0
        );
    }

    #[test]
    fn stuffing_unknown_model_returns_zero() {
        assert_eq!(max_context_stuffing_chunks("unknown", "some-model", 150), 0);
        assert_eq!(
            max_context_stuffing_chunks("anthropic", "claude-999", 150),
            0
        );
    }

    #[test]
    fn stuffing_global_override_acts_as_ceiling() {
        assert_eq!(
            max_context_stuffing_chunks("mistral", "mistral-small-latest", 50),
            50
        );
        assert_eq!(
            max_context_stuffing_chunks("mistral", "mistral-large-latest", 200),
            80
        );
    }
}
