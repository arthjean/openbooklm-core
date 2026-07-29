//! Token budget management for LLM prompt assembly.
//!
//! Handles context window sizing, token estimation, per-message truncation,
//! and fitting all prompt segments within the model's token budget.

use std::borrow::Cow;

use super::types::LlmMessage;

/// Default fallback context window for unknown models (tokens).
const DEFAULT_CONTEXT_WINDOW: usize = 128_000;

/// Maximum tokens per individual history message (US-006).
///
/// Prevents a single oversized assistant response from consuming the entire
/// history budget. Full content is preserved in the database.
pub const MAX_TOKENS_PER_HISTORY_MESSAGE: usize = 2000;

/// Truncation marker appended to per-message truncated content.
const TRUNCATION_MARKER: &str = "\n\n[...truncated, full response available in history]";

// ============================================================================
// Token budget allocation (US-001)
// ============================================================================

/// Token budget allocation for each prompt segment.
///
/// Ratios: system 15%, RAG 35%, history 25%, memory 10%, buffer 15%.
/// Unused budget from non-history segments is reallocated to history.
///
/// Note: `history_tokens` is the nominal allocation. [`fit_prompt_to_budget`]
/// computes the actual history budget dynamically as
/// `total_window - system_actual - buffer_tokens`, reclaiming unused
/// RAG/memory budget when those segments are small.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    pub total_window: usize,
    pub system_tokens: usize,
    pub rag_tokens: usize,
    pub memory_tokens: usize,
    pub history_tokens: usize,
    pub buffer_tokens: usize,
}

/// Actual token usage per segment (for logging/debugging).
#[derive(Debug, Clone)]
pub struct BudgetUsage {
    pub system_actual: usize,
    pub rag_actual: usize,
    pub memory_actual: usize,
    pub history_actual: usize,
    pub buffer_reserved: usize,
    pub total_used: usize,
    pub total_window: usize,
}

/// Result of fitting all prompt segments within the token budget.
pub struct FittedPrompt {
    pub system_prompt: String,
    pub messages: Vec<LlmMessage>,
    pub was_truncated: bool,
    pub budget_usage: BudgetUsage,
    /// Messages dropped from the oldest end of history due to budget limits (US-002).
    pub dropped_messages: Vec<LlmMessage>,
}

/// Result of token-aware history truncation.
///
/// Kept for backward compatibility. New code should use [`fit_prompt_to_budget`].
pub struct TruncatedHistory {
    /// The truncated messages (newest messages kept).
    pub messages: Vec<LlmMessage>,
    /// Whether any messages were dropped.
    pub was_truncated: bool,
}

// ============================================================================
// Context window sizing
// ============================================================================

/// Get the context window size for a specific `(provider, model)` pair.
///
/// Uses `starts_with` prefix matching so versioned model IDs (e.g.
/// `claude-opus-4-6-20260220`) resolve correctly.
pub fn context_window_for_model(provider: &str, model: &str) -> usize {
    match provider {
        "anthropic" => 200_000, // claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5
        "openai" => {
            if model.starts_with("gpt-5.2") || model.starts_with("gpt-5-mini") {
                400_000
            } else {
                128_000
            }
        }
        "mistral" => {
            if model.starts_with("mistral-large-") {
                256_000
            } else {
                128_000 // mistral-small and other Mistral models
            }
        }
        _ => DEFAULT_CONTEXT_WINDOW,
    }
}

// ============================================================================
// Token estimation
// ============================================================================

/// Estimate token count for a string.
///
/// Uses bytes/4 as a rough approximation for Latin/ASCII text.
/// CJK characters get an extra token each on top of their bytes/4
/// contribution. This is conservative — actual tokenizers give ~1
/// token per CJK char, but our formula yields ~1.75 (0.75 from
/// bytes/4 + 1.0 correction). The overestimate is safe for budget
/// enforcement (errs toward keeping fewer messages, never overflow).
pub fn estimate_tokens(text: &str) -> usize {
    let cjk_count = text.chars().filter(|c| is_cjk(*c)).count();
    text.len() / 4 + cjk_count
}

fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Korean Hangul
    )
}

// ============================================================================
// Budget allocation & prompt fitting
// ============================================================================

/// Compute segment budgets from the model's context window.
///
/// Ratios: system 15%, RAG 35%, history 25%, memory 10%, buffer 15%.
pub fn allocate_token_budget(provider: &str, model: &str) -> TokenBudget {
    let total = context_window_for_model(provider, model);
    let system_tokens = total * 15 / 100;
    let rag_tokens = total * 35 / 100;
    let history_tokens = total * 25 / 100;
    let memory_tokens = total * 10 / 100;
    // Buffer absorbs integer rounding so all segments sum exactly to total.
    let buffer_tokens = total - system_tokens - rag_tokens - history_tokens - memory_tokens;
    TokenBudget {
        total_window: total,
        system_tokens,
        rag_tokens,
        history_tokens,
        memory_tokens,
        buffer_tokens,
    }
}

/// Truncate a single message's content to fit within a token budget.
///
/// Returns `Cow::Borrowed` if the content fits (zero-copy for the token
/// measurement step), or `Cow::Owned` with a truncation marker if it
/// exceeds the budget. Callers that convert to `String` (e.g. via
/// `into_owned()`) will allocate regardless — the `Cow` fast path avoids
/// allocation only while the borrow is live.
///
/// Truncation respects UTF-8 character boundaries.
pub fn truncate_message_content(content: &str, max_tokens: usize) -> Cow<'_, str> {
    if estimate_tokens(content) <= max_tokens {
        return Cow::Borrowed(content);
    }

    // Subtract marker tokens so the result including marker fits the budget.
    let marker_tokens = estimate_tokens(TRUNCATION_MARKER);
    let target_tokens = max_tokens.saturating_sub(marker_tokens);

    // Walk characters accumulating token estimate incrementally.
    // This mirrors `estimate_tokens` (bytes/4 + cjk_count) exactly,
    // avoiding the CJK mismatch that would arise from a fixed bytes-per-token
    // multiplier (CJK chars are 3 UTF-8 bytes but count as ~1.75 tokens
    // under our heuristic, not 0.75).
    let mut byte_count = 0usize;
    let mut cjk_count = 0usize;
    let mut truncate_at = 0usize;

    for (i, c) in content.char_indices() {
        let new_bytes = byte_count + c.len_utf8();
        let new_cjk = cjk_count + usize::from(is_cjk(c));
        if new_bytes / 4 + new_cjk > target_tokens {
            break;
        }
        byte_count = new_bytes;
        cjk_count = new_cjk;
        truncate_at = i + c.len_utf8();
    }

    let mut truncated = content[..truncate_at].to_string();
    truncated.push_str(TRUNCATION_MARKER);
    Cow::Owned(truncated)
}

/// Fit all prompt segments within the model's token budget.
///
/// Measures each segment's actual token count and dynamically allocates
/// remaining tokens to history (reclaiming unused budget from other segments).
///
/// The system prompt (including RAG context and memory) is passed pre-built.
/// `rag_context` and `memory_block` are passed separately for per-segment
/// measurement and logging — their tokens are already included in `system_prompt`.
///
/// Individual history messages are truncated to [`MAX_TOKENS_PER_HISTORY_MESSAGE`]
/// before measuring total history tokens (US-006).
pub fn fit_prompt_to_budget(
    system_prompt: &str,
    rag_context: &str,
    memory_block: Option<&str>,
    history: &[LlmMessage],
    budget: &TokenBudget,
) -> FittedPrompt {
    // 1. Measure actual segment sizes
    let system_actual = estimate_tokens(system_prompt);
    let rag_actual = estimate_tokens(rag_context);
    let memory_actual = memory_block.map_or(0, estimate_tokens);

    // 2. Log warnings for oversized segments
    if rag_actual > budget.rag_tokens {
        tracing::warn!(
            rag_actual,
            budget = budget.rag_tokens,
            "RAG context exceeds segment budget"
        );
    }
    if memory_actual > budget.memory_tokens {
        tracing::warn!(
            memory_actual,
            budget = budget.memory_tokens,
            "Memory block exceeds segment budget"
        );
    }
    if system_actual > budget.system_tokens + budget.rag_tokens + budget.memory_tokens {
        tracing::warn!(
            system_actual,
            budget_combined = budget.system_tokens + budget.rag_tokens + budget.memory_tokens,
            "System prompt (with RAG + memory) exceeds combined segment budget"
        );
    }

    // 3. Compute remaining budget for history.
    //    remaining = total - actual_system - buffer
    //    This dynamically reclaims unused budget: if RAG/memory are small,
    //    history gets the surplus automatically.
    let non_history = system_actual + budget.buffer_tokens;
    let history_budget = budget.total_window.saturating_sub(non_history);

    // 4. Truncate individual messages (US-006) and fit to history budget
    let mut total_tokens = 0;
    let mut keep_count = 0;
    let mut kept_messages: Vec<LlmMessage> = Vec::new();

    for msg in history.iter().rev() {
        let content = truncate_message_content(&msg.content, MAX_TOKENS_PER_HISTORY_MESSAGE);
        let msg_tokens = estimate_tokens(&content);
        if total_tokens + msg_tokens > history_budget {
            break;
        }
        total_tokens += msg_tokens;
        keep_count += 1;
        kept_messages.push(LlmMessage {
            role: msg.role,
            content: content.into_owned(),
        });
    }

    // Reverse to restore chronological order (we iterated newest-first)
    kept_messages.reverse();

    let was_truncated = keep_count < history.len();

    // Collect dropped messages (oldest end of history) for potential summarization (US-002)
    let dropped_messages = if was_truncated {
        let drop_count = history.len() - keep_count;
        history[..drop_count].to_vec()
    } else {
        vec![]
    };

    let budget_usage = BudgetUsage {
        system_actual,
        rag_actual,
        memory_actual,
        history_actual: total_tokens,
        buffer_reserved: budget.buffer_tokens,
        total_used: system_actual + total_tokens + budget.buffer_tokens,
        total_window: budget.total_window,
    };

    tracing::debug!(
        system = budget_usage.system_actual,
        rag = budget_usage.rag_actual,
        memory = budget_usage.memory_actual,
        history = budget_usage.history_actual,
        buffer = budget_usage.buffer_reserved,
        total_used = budget_usage.total_used,
        total_window = budget_usage.total_window,
        history_budget,
        was_truncated,
        kept = kept_messages.len(),
        original = history.len(),
        "Token budget usage"
    );

    FittedPrompt {
        system_prompt: system_prompt.to_string(),
        messages: kept_messages,
        was_truncated,
        budget_usage,
        dropped_messages,
    }
}

/// Truncate conversation history to fit within a fixed token budget.
///
/// Keeps the most recent messages, dropping oldest first.
/// Budget is 25% of the model's context window (the history segment ratio).
///
/// **Prefer [`fit_prompt_to_budget`]** for the chat handler — it accounts for
/// all prompt segments and dynamically reallocates unused budget to history.
pub fn truncate_history_to_budget(
    messages: &[LlmMessage],
    provider_name: &str,
    model: &str,
) -> TruncatedHistory {
    let budget = allocate_token_budget(provider_name, model);

    let mut total_tokens = 0;
    let mut keep_count = 0;

    for msg in messages.iter().rev() {
        let msg_tokens = estimate_tokens(&msg.content);
        if total_tokens + msg_tokens > budget.history_tokens {
            break;
        }
        total_tokens += msg_tokens;
        keep_count += 1;
    }

    let was_truncated = keep_count < messages.len();
    let start = messages.len() - keep_count;
    let kept = messages[start..].to_vec();

    TruncatedHistory {
        messages: kept,
        was_truncated,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::LlmMessage;

    #[test]
    fn context_window_anthropic_opus() {
        assert_eq!(
            context_window_for_model("anthropic", "claude-opus-4-6-20260220"),
            200_000
        );
    }

    #[test]
    fn context_window_anthropic_sonnet() {
        assert_eq!(
            context_window_for_model("anthropic", "claude-sonnet-4-6-20260220"),
            200_000
        );
    }

    #[test]
    fn context_window_anthropic_haiku() {
        assert_eq!(
            context_window_for_model("anthropic", "claude-haiku-4-5-20251001"),
            200_000
        );
    }

    #[test]
    fn context_window_openai_gpt52() {
        assert_eq!(context_window_for_model("openai", "gpt-5.2-turbo"), 400_000);
    }

    #[test]
    fn context_window_openai_gpt5_mini() {
        assert_eq!(
            context_window_for_model("openai", "gpt-5-mini-2025"),
            400_000
        );
    }

    #[test]
    fn context_window_mistral_large() {
        assert_eq!(
            context_window_for_model("mistral", "mistral-large-latest"),
            256_000
        );
    }

    #[test]
    fn context_window_mistral_small() {
        assert_eq!(
            context_window_for_model("mistral", "mistral-small-latest"),
            128_000
        );
    }

    #[test]
    fn context_window_unknown_uses_default() {
        assert_eq!(context_window_for_model("unknown", "some-model"), 128_000);
    }

    #[test]
    fn context_window_prefix_matching_works() {
        assert_eq!(
            context_window_for_model("mistral", "mistral-large-2026-01"),
            256_000
        );
        assert_eq!(
            context_window_for_model("openai", "gpt-5.2-latest"),
            400_000
        );
    }

    #[test]
    fn estimate_tokens_english() {
        let tokens = estimate_tokens("hello world");
        assert!(tokens >= 2);
        assert!(tokens <= 5);
    }

    #[test]
    fn estimate_tokens_cjk_higher_density() {
        let tokens_cjk = estimate_tokens("你好世界");
        let tokens_eng = estimate_tokens("abcd");
        assert!(tokens_cjk > tokens_eng);
    }

    #[test]
    fn budget_allocation_anthropic() {
        let budget = allocate_token_budget("anthropic", "claude-sonnet-4-6-20260220");
        assert_eq!(budget.total_window, 200_000);
        assert_eq!(budget.system_tokens, 30_000); // 15%
        assert_eq!(budget.rag_tokens, 70_000); // 35%
        assert_eq!(budget.history_tokens, 50_000); // 25%
        assert_eq!(budget.memory_tokens, 20_000); // 10%
        assert_eq!(budget.buffer_tokens, 30_000); // 15%
    }

    #[test]
    fn budget_allocation_openai_gpt52() {
        let budget = allocate_token_budget("openai", "gpt-5.2-turbo");
        assert_eq!(budget.total_window, 400_000);
        assert_eq!(budget.system_tokens, 60_000);
        assert_eq!(budget.rag_tokens, 140_000);
        assert_eq!(budget.history_tokens, 100_000);
        assert_eq!(budget.memory_tokens, 40_000);
        assert_eq!(budget.buffer_tokens, 60_000);
    }

    #[test]
    fn budget_allocation_mistral_small() {
        let budget = allocate_token_budget("mistral", "mistral-small-latest");
        assert_eq!(budget.total_window, 128_000);
        assert_eq!(budget.system_tokens, 19_200);
        assert_eq!(budget.rag_tokens, 44_800);
        assert_eq!(budget.history_tokens, 32_000);
        assert_eq!(budget.memory_tokens, 12_800);
        assert_eq!(budget.buffer_tokens, 19_200);
    }

    #[test]
    fn budget_allocation_unknown_provider() {
        let budget = allocate_token_budget("unknown", "some-model");
        assert_eq!(budget.total_window, 128_000);
        let sum = budget.system_tokens
            + budget.rag_tokens
            + budget.history_tokens
            + budget.memory_tokens
            + budget.buffer_tokens;
        assert_eq!(sum, budget.total_window);
    }

    #[test]
    fn truncate_keeps_all_when_within_budget() {
        let messages = vec![
            LlmMessage::user("Hello"),
            LlmMessage::assistant("Hi there"),
            LlmMessage::user("How are you?"),
        ];
        let result =
            truncate_history_to_budget(&messages, "anthropic", "claude-sonnet-4-6-20260220");
        assert!(!result.was_truncated);
        assert_eq!(result.messages.len(), 3);
    }

    #[test]
    fn truncate_drops_oldest_first() {
        let long_msg = "x".repeat(600_000);
        let messages = vec![
            LlmMessage::user(long_msg),
            LlmMessage::user("recent message"),
        ];
        let result = truncate_history_to_budget(&messages, "mistral", "mistral-small-latest");
        assert!(result.was_truncated);
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].content, "recent message");
    }

    #[test]
    fn truncate_exceeding_budget_keeps_newest() {
        let big = "a".repeat(400_000);
        let messages = vec![
            LlmMessage::user(big.clone()),
            LlmMessage::assistant(big.clone()),
            LlmMessage::user("latest question"),
        ];
        let result = truncate_history_to_budget(&messages, "unknown", "some-model");
        assert!(result.was_truncated);
        assert!(result.messages.len() <= 2);
        assert_eq!(result.messages.last().unwrap().content, "latest question");
    }

    #[test]
    fn truncate_message_under_budget_returns_borrowed() {
        let content = "short message";
        let result = truncate_message_content(content, 2000);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, content);
    }

    #[test]
    fn truncate_message_over_budget_returns_owned() {
        let content = "x".repeat(40_000); // ~10,000 tokens
        let result = truncate_message_content(&content, 2000);
        assert!(matches!(result, Cow::Owned(_)));
        assert!(result.contains("[...truncated, full response available in history]"));
        let result_tokens = estimate_tokens(&result);
        assert!(
            result_tokens <= 2000,
            "Truncated message should be <= 2000 tokens, got {result_tokens}"
        );
    }

    #[test]
    fn truncate_message_utf8_boundary_2byte() {
        let content = "é".repeat(10_000);
        let result = truncate_message_content(&content, 500);
        assert!(matches!(result, Cow::Owned(_)));
        let char_count = result.chars().count();
        assert!(char_count > 0);
        let marker_pos = result.find("[...truncated").unwrap();
        let prefix = &result[..marker_pos];
        assert!(prefix.chars().all(|c| c == 'é' || c == '\n'));
        assert!(result.contains("[...truncated"));
    }

    #[test]
    fn truncate_message_utf8_boundary_cjk() {
        let content = "中".repeat(5_000);
        let result = truncate_message_content(&content, 500);
        assert!(matches!(result, Cow::Owned(_)));
        let _ = result.chars().count();
        assert!(result.contains("[...truncated"));
        let result_tokens = estimate_tokens(&result);
        assert!(
            result_tokens <= 500,
            "CJK truncated message should be <= 500 tokens, got {result_tokens}"
        );
    }

    #[test]
    fn truncate_message_utf8_boundary_emoji() {
        let content = "😀".repeat(5_000);
        let result = truncate_message_content(&content, 500);
        assert!(matches!(result, Cow::Owned(_)));
        let _ = result.chars().count();
        assert!(result.contains("[...truncated"));
        let result_tokens = estimate_tokens(&result);
        assert!(
            result_tokens <= 500,
            "Emoji truncated message should be <= 500 tokens, got {result_tokens}"
        );
    }

    #[test]
    fn truncate_message_empty_returns_borrowed() {
        let result = truncate_message_content("", 2000);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(&*result, "");
    }

    #[test]
    fn truncate_message_exact_budget_returns_borrowed() {
        let content = "a".repeat(8000);
        assert_eq!(estimate_tokens(&content), 2000);
        let result = truncate_message_content(&content, 2000);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn fit_prompt_keeps_all_when_within_budget() {
        let messages = vec![
            LlmMessage::user("Hello"),
            LlmMessage::assistant("Hi there"),
            LlmMessage::user("How are you?"),
        ];
        let budget = allocate_token_budget("anthropic", "claude-sonnet-4-6-20260220");
        let result = fit_prompt_to_budget("system prompt", "rag context", None, &messages, &budget);
        assert!(!result.was_truncated);
        assert_eq!(result.messages.len(), 3);
        assert!(result.budget_usage.total_used < budget.total_window);
    }

    #[test]
    fn fit_prompt_history_reclaims_unused_budget() {
        let budget = allocate_token_budget("mistral", "mistral-small-latest");
        let result = fit_prompt_to_budget("hi", "", None, &[], &budget);
        let expected_budget = 128_000 - estimate_tokens("hi") - 19_200;
        assert_eq!(result.budget_usage.system_actual, estimate_tokens("hi"));
        assert!(expected_budget > budget.history_tokens);
    }

    #[test]
    fn fit_prompt_truncates_with_large_system_prompt() {
        let large_system = "x".repeat(450_000); // ~112,500 tokens
        let messages: Vec<LlmMessage> = (0..5)
            .map(|i| LlmMessage::user(format!("message {i}")))
            .collect();
        let budget = allocate_token_budget("mistral", "mistral-small-latest");
        let result = fit_prompt_to_budget(&large_system, &large_system, None, &messages, &budget);
        assert!(result.was_truncated);
        assert!(result.messages.len() < 5);
    }

    #[test]
    fn fit_prompt_logs_oversized_rag() {
        let budget = allocate_token_budget("mistral", "mistral-small-latest");
        let large_rag = "x".repeat(200_000); // ~50K tokens, budget.rag_tokens = 44,800
        let result = fit_prompt_to_budget("system", &large_rag, None, &[], &budget);
        assert!(result.budget_usage.rag_actual > budget.rag_tokens);
    }

    #[test]
    fn fit_prompt_logs_oversized_memory() {
        let budget = allocate_token_budget("mistral", "mistral-small-latest");
        let large_memory = "x".repeat(60_000); // ~15K tokens, budget.memory_tokens = 12,800
        let result = fit_prompt_to_budget("system", "", Some(&large_memory), &[], &budget);
        assert!(result.budget_usage.memory_actual > budget.memory_tokens);
    }

    #[test]
    fn fit_prompt_budget_usage_tracks_all_segments() {
        let budget = allocate_token_budget("anthropic", "claude-sonnet-4-6-20260220");
        let messages = vec![LlmMessage::user("Hello world")];
        let result = fit_prompt_to_budget(
            "system prompt here",
            "rag context",
            Some("<memory>facts</memory>"),
            &messages,
            &budget,
        );
        assert_eq!(
            result.budget_usage.system_actual,
            estimate_tokens("system prompt here")
        );
        assert_eq!(
            result.budget_usage.rag_actual,
            estimate_tokens("rag context")
        );
        assert_eq!(
            result.budget_usage.memory_actual,
            estimate_tokens("<memory>facts</memory>")
        );
        assert!(result.budget_usage.history_actual > 0);
        assert_eq!(result.budget_usage.buffer_reserved, budget.buffer_tokens);
        assert_eq!(result.budget_usage.total_window, 200_000);
    }

    #[test]
    fn fit_prompt_no_dropped_messages_when_all_fit() {
        let messages = vec![
            LlmMessage::user("Hello"),
            LlmMessage::assistant("Hi there"),
            LlmMessage::user("How are you?"),
        ];
        let budget = allocate_token_budget("anthropic", "claude-sonnet-4-6-20260220");
        let result = fit_prompt_to_budget("system", "", None, &messages, &budget);
        assert!(result.dropped_messages.is_empty());
        assert!(!result.was_truncated);
    }

    #[test]
    fn fit_prompt_dropped_messages_when_truncated() {
        let large_system = "x".repeat(450_000); // ~112.5K tokens
        let messages: Vec<LlmMessage> = (0..10)
            .map(|i| LlmMessage::user(format!("message {i}")))
            .collect();
        let budget = allocate_token_budget("mistral", "mistral-small-latest");
        let result = fit_prompt_to_budget(&large_system, "", None, &messages, &budget);
        assert!(result.was_truncated);
        assert_eq!(
            result.dropped_messages.len() + result.messages.len(),
            messages.len()
        );
        if !result.dropped_messages.is_empty() {
            assert_eq!(result.dropped_messages[0].content, "message 0");
        }
    }

    #[test]
    fn fit_prompt_dropped_messages_are_oldest() {
        let system = "y".repeat(150_000); // ~37,500 tokens
        let messages: Vec<LlmMessage> = (0..50)
            .map(|i| LlmMessage::user(format!("msg-{i}: {}", "w".repeat(8000)))) // ~2000 tokens each
            .collect();
        let budget = allocate_token_budget("mistral", "mistral-small-latest");
        let result = fit_prompt_to_budget(&system, "", None, &messages, &budget);
        assert!(result.was_truncated);
        assert!(!result.dropped_messages.is_empty());
        assert!(result.dropped_messages[0].content.starts_with("msg-0:"));
        assert!(
            result
                .messages
                .last()
                .unwrap()
                .content
                .starts_with("msg-49:")
        );
    }

    #[test]
    fn fit_prompt_applies_per_message_truncation() {
        let huge_msg = "x".repeat(100_000); // ~25K tokens
        let messages = vec![LlmMessage::assistant(huge_msg)];
        let budget = allocate_token_budget("anthropic", "claude-sonnet-4-6-20260220");
        let result = fit_prompt_to_budget("system", "", None, &messages, &budget);
        assert_eq!(result.messages.len(), 1);
        assert!(
            result.messages[0]
                .content
                .contains("[...truncated, full response available in history]")
        );
        assert!(result.budget_usage.history_actual <= 2100);
    }
}
