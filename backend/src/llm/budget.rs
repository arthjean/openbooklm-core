//! One token budget for the whole provider request (US-018).
//!
//! Every component that reaches a provider is counted here: the system
//! instructions, the retrieved evidence (as prompt text or as provider-native
//! document blocks), the memory block, the conversation history, the current
//! question, the tokens the answer is allowed to use, and a reserve.
//!
//! # Why a declared window and not a default
//!
//! [`context_window_for_model`] returns `Option`. A provider whose window this
//! build does not know has no budget, and a request that cannot be measured is
//! not sent: the previous 128,000-token fallback was a guess that happened to be
//! wrong in the only direction that matters, since a model with a smaller window
//! rejects the request after the prompt has been assembled and paid for
//! (US-018 AC-5).
//!
//! # Allocation order
//!
//! The window is spent in a fixed priority order, and each step takes what it
//! needs from what the previous ones left:
//!
//! 1. **Reserve** — `max(1,024, 5% of the window)`, never spent. Tokenizer
//!    disagreement is real: this module estimates, the provider counts.
//! 2. **Output** — what the client asks the model to be able to write.
//! 3. **Instructions and the current question** — mandatory. When they do not
//!    fit, nothing is sent.
//! 4. **Memory**, capped at [`MEMORY_PERCENT`] of the window; dropped whole
//!    rather than truncated mid-fact.
//! 5. **Evidence**, capped at [`EVIDENCE_PERCENT`] of the window.
//! 6. **History**, which gets the remainder, oldest turns dropped first.
//!
//! Steps 4 to 6 are applied by [`crate::services::chat::context_budget`], which
//! is where the evidence is visible. This module owns the arithmetic, the two
//! pure fitting primitives, and [`TokenMeter`] — the sink the evidence renderer
//! writes into when it is being priced rather than sent.

use std::borrow::Cow;
use std::fmt::{self, Write as _};

use crate::clients::models::context_window_for_model as catalog_context_window_for_model;

use super::types::LlmMessage;

/// Maximum tokens per individual history message.
///
/// Prevents a single oversized assistant response from consuming the entire
/// history budget. Full content is preserved in the database.
pub const MAX_TOKENS_PER_HISTORY_MESSAGE: usize = 2000;

/// Truncation marker appended to per-message truncated content.
const TRUNCATION_MARKER: &str = "\n\n[...truncated, full response available in history]";

/// Floor of the reserve held back from every request.
///
/// The PRD fixes it: a reserve of `max(1,024 tokens, 5% of the context window)`.
pub const MIN_RESERVE_TOKENS: usize = 1024;

/// Share of the window that forms the reserve, when it exceeds the floor.
const RESERVE_PERCENT: usize = 5;

/// Ceiling on retrieved evidence, as a share of the window.
///
/// Evidence is bounded even when the window is nearly empty, so that a long
/// conversation does not lose all of its history to one oversized passage. With
/// a 20-context limit and 1,024-token parents the cap only binds on the smallest
/// windows, which is the case it exists for.
const EVIDENCE_PERCENT: usize = 60;

/// Ceiling on the memory block, as a share of the window.
const MEMORY_PERCENT: usize = 10;

/// Conservative allowance for provider-side request framing outside content.
const REQUEST_OVERHEAD_TOKENS: usize = 16;

/// Conservative allowance for the role and separators of one message.
const MESSAGE_OVERHEAD_TOKENS: usize = 8;

// ============================================================================
// Context window sizing
// ============================================================================

/// Declared context window for a `(provider, model)` pair, in tokens.
///
/// `None` means this build cannot state the window, which is a refusal to
/// generate rather than a licence to guess (US-018 AC-5).
///
/// Only model identifiers present in the static public catalog resolve. An
/// unlisted identifier has no safe window until its metadata is propagated to
/// this boundary.
#[must_use]
pub fn context_window_for_model(provider: &str, model: &str) -> Option<usize> {
    // The in-process model behind the offline evaluator and the CI smoke path
    // is not part of the public provider catalog.
    let window = if (provider, model) == ("deterministic", "deterministic-echo-v1") {
        128_000
    } else {
        catalog_context_window_for_model(provider, model)?
    };
    usize::try_from(window).ok()
}

// ============================================================================
// Budget
// ============================================================================

/// The token arithmetic of one provider request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptBudget {
    window: usize,
    reserve: usize,
    output: usize,
}

impl PromptBudget {
    /// Build a budget for a declared window and a requested output size.
    #[must_use]
    pub fn new(window: usize, output: usize) -> Self {
        let reserve = MIN_RESERVE_TOKENS.max(window * RESERVE_PERCENT / 100);
        Self {
            window,
            reserve,
            output,
        }
    }

    /// Build a budget for a model, or `None` when its window is not declared.
    #[must_use]
    pub fn for_model(provider: &str, model: &str, output: usize) -> Option<Self> {
        context_window_for_model(provider, model).map(|window| Self::new(window, output))
    }

    #[must_use]
    pub const fn window(&self) -> usize {
        self.window
    }

    #[must_use]
    pub const fn reserve(&self) -> usize {
        self.reserve
    }

    #[must_use]
    pub const fn output(&self) -> usize {
        self.output
    }

    /// Tokens the prompt itself may occupy.
    #[must_use]
    pub const fn prompt_allowance(&self) -> usize {
        self.window
            .saturating_sub(self.reserve)
            .saturating_sub(self.output)
            .saturating_sub(REQUEST_OVERHEAD_TOKENS)
    }

    /// Ceiling on retrieved evidence.
    #[must_use]
    pub const fn evidence_cap(&self) -> usize {
        self.window * EVIDENCE_PERCENT / 100
    }

    /// Ceiling on the memory block.
    #[must_use]
    pub const fn memory_cap(&self) -> usize {
        self.window * MEMORY_PERCENT / 100
    }

    /// Whether an assembled request fits, with its reserve intact.
    ///
    /// The assertion US-018 AC-4 asks for. It is a check and not a
    /// `debug_assert`: a release build must refuse an oversized request rather
    /// than send it.
    #[must_use]
    pub fn admits(&self, system_prompt: &str, messages: &[LlmMessage]) -> bool {
        self.admits_with_additional_prompt_tokens(system_prompt, messages, 0)
    }

    /// Whether the final request fits when the provider receives prompt tokens
    /// outside `system_prompt` and `messages`, such as Anthropic document
    /// blocks.
    #[must_use]
    pub fn admits_with_additional_prompt_tokens(
        &self,
        system_prompt: &str,
        messages: &[LlmMessage],
        additional_prompt_tokens: usize,
    ) -> bool {
        request_tokens(system_prompt, messages)
            .saturating_add(additional_prompt_tokens)
            .saturating_add(self.output)
            .saturating_add(self.reserve)
            <= self.window
    }
}

/// Estimated tokens of an assembled request, excluding output and reserve.
#[must_use]
pub fn request_tokens(system_prompt: &str, messages: &[LlmMessage]) -> usize {
    REQUEST_OVERHEAD_TOKENS
        + estimate_tokens(system_prompt)
        + messages
            .iter()
            .map(|m| message_tokens(&m.content))
            .sum::<usize>()
}

// ============================================================================
// Token estimation
// ============================================================================

/// Counts the tokens of everything written into it, without building a string.
///
/// The single definition of the bound: one token per UTF-8 byte. Supported
/// provider tokenizers cannot emit more content tokens than input bytes, since
/// every content token consumes at least one byte. This intentionally trades
/// some context utilization for a provider-independent upper bound; request and
/// message framing are counted separately.
///
/// It is a [`fmt::Write`] sink, which is what makes it interchangeable with the
/// `String` the evidence renderer writes into. Pricing an entry means running
/// the renderer against a meter instead of against a buffer: same code, same
/// bytes, no allocation. An estimate computed by any other route is an estimate
/// that eventually disagrees with what is sent.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokenMeter {
    bytes: usize,
}

impl TokenMeter {
    /// Tokens written into this meter so far.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.bytes
    }
}

impl fmt::Write for TokenMeter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.bytes += s.len();
        Ok(())
    }
}

/// Estimate token count for a string.
///
/// One string fed through [`TokenMeter`], so the standalone estimate and the
/// renderer's price can never drift apart.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    let mut meter = TokenMeter::default();
    let _ = meter.write_str(text);
    meter.tokens()
}

pub(crate) fn message_tokens(content: &str) -> usize {
    estimate_tokens(content) + MESSAGE_OVERHEAD_TOKENS
}

// ============================================================================
// History fitting
// ============================================================================

/// Conversation history cut to a token budget.
#[derive(Debug, Clone, Default)]
pub struct FittedHistory {
    /// Kept messages, in chronological order.
    pub messages: Vec<LlmMessage>,
    /// Messages dropped from the oldest end, for summarization.
    pub dropped: Vec<LlmMessage>,
    pub was_truncated: bool,
    /// Estimated tokens of `messages`.
    pub tokens: usize,
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
#[must_use]
pub fn truncate_message_content(content: &str, max_tokens: usize) -> Cow<'_, str> {
    if estimate_tokens(content) <= max_tokens {
        return Cow::Borrowed(content);
    }

    // Subtract marker tokens so the result including marker fits the budget.
    let marker_tokens = estimate_tokens(TRUNCATION_MARKER);
    let target_tokens = max_tokens.saturating_sub(marker_tokens);

    // Walk characters while preserving UTF-8 boundaries. The estimate is the
    // byte count, so no second tokenizer approximation is involved here.
    let mut byte_count = 0usize;
    let mut truncate_at = 0usize;

    for (i, c) in content.char_indices() {
        let new_bytes = byte_count + c.len_utf8();
        if new_bytes > target_tokens {
            break;
        }
        byte_count = new_bytes;
        truncate_at = i + c.len_utf8();
    }

    let mut truncated = content[..truncate_at].to_string();
    truncated.push_str(TRUNCATION_MARKER);
    Cow::Owned(truncated)
}

/// Keep the newest history that fits `budget_tokens`.
///
/// Individual messages are first truncated to
/// [`MAX_TOKENS_PER_HISTORY_MESSAGE`], then the oldest are dropped until the
/// remainder fits.
#[must_use]
pub fn fit_history(history: &[LlmMessage], budget_tokens: usize) -> FittedHistory {
    let mut total_tokens = 0;
    let mut kept: Vec<LlmMessage> = Vec::new();

    for msg in history.iter().rev() {
        let content = truncate_message_content(&msg.content, MAX_TOKENS_PER_HISTORY_MESSAGE);
        let msg_tokens = message_tokens(&content);
        if total_tokens + msg_tokens > budget_tokens {
            break;
        }
        total_tokens += msg_tokens;
        kept.push(LlmMessage {
            role: msg.role,
            content: content.into_owned(),
        });
    }

    // Reverse to restore chronological order (we iterated newest-first).
    kept.reverse();

    let was_truncated = kept.len() < history.len();
    let dropped = if was_truncated {
        history[..history.len() - kept.len()].to_vec()
    } else {
        Vec::new()
    };

    FittedHistory {
        messages: kept,
        dropped,
        was_truncated,
        tokens: total_tokens,
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
    fn context_window_anthropic_models_share_one_window() {
        for model in [
            "claude-opus-4-6-20260220",
            "claude-sonnet-4-6-20260220",
            "claude-haiku-4-5-20251001",
        ] {
            assert_eq!(context_window_for_model("anthropic", model), Some(200_000));
        }
    }

    #[test]
    fn context_window_openai_variants() {
        assert_eq!(context_window_for_model("openai", "gpt-5.2"), Some(400_000));
        assert_eq!(
            context_window_for_model("openai", "gpt-5-mini"),
            Some(400_000)
        );
        assert_eq!(context_window_for_model("openai", "gpt-4o-mini"), None);
    }

    #[test]
    fn context_window_mistral_variants() {
        assert_eq!(
            context_window_for_model("mistral", "mistral-large-latest"),
            Some(131_072)
        );
        assert_eq!(
            context_window_for_model("mistral", "mistral-small-latest"),
            Some(32_768)
        );
        assert_eq!(
            context_window_for_model("mistral", "mistral-small-unknown"),
            None,
            "an unlisted model must not inherit an invented window"
        );
    }

    /// The defect US-018 AC-5 names: an unknown provider used to be budgeted
    /// against an invented 128,000-token window.
    #[test]
    fn an_undeclared_provider_has_no_window_rather_than_a_default() {
        assert_eq!(context_window_for_model("unknown", "some-model"), None);
        assert!(PromptBudget::for_model("unknown", "some-model", 4096).is_none());
    }

    #[test]
    fn the_offline_provider_declares_its_window() {
        assert_eq!(
            context_window_for_model("deterministic", "deterministic-echo-v1"),
            Some(128_000)
        );
    }

    #[test]
    fn estimate_tokens_english() {
        let tokens = estimate_tokens("hello world");
        assert_eq!(tokens, "hello world".len());
    }

    #[test]
    fn estimate_is_a_conservative_utf8_byte_bound() {
        for text in [
            "550e8400-e29b-41d4-a716-446655440000",
            "fn main() { println!(\"hello\"); }",
            "مرحبا بالعالم",
            "你好世界",
            "😀🧪",
        ] {
            assert_eq!(estimate_tokens(text), text.len());
        }
    }

    /// The property the evidence renderer relies on: writing the pieces of a
    /// string into a meter prices the same string, minus only the truncation
    /// the integer division loses at each boundary.
    #[test]
    fn a_meter_fed_in_pieces_prices_the_whole() {
        let pieces = [
            "<source index=\"1\" ",
            "title=\"Notes\">",
            "中文とか",
            "</source>",
        ];
        let joined: String = pieces.concat();

        let mut meter = TokenMeter::default();
        for piece in pieces {
            let _ = meter.write_str(piece);
        }

        let whole = estimate_tokens(&joined);
        assert_eq!(meter.tokens(), whole);
    }

    #[test]
    fn the_reserve_is_the_larger_of_the_floor_and_five_percent() {
        // 5% of 200,000 is 10,000, which exceeds the 1,024 floor.
        let large = PromptBudget::new(200_000, 8192);
        assert_eq!(large.reserve(), 10_000);

        // 5% of 8,000 is 400, so the floor applies.
        let small = PromptBudget::new(8_000, 1_000);
        assert_eq!(small.reserve(), MIN_RESERVE_TOKENS);
    }

    #[test]
    fn the_prompt_allowance_excludes_the_reserve_and_the_answer() {
        let budget = PromptBudget::new(100_000, 4_000);
        assert_eq!(budget.reserve(), 5_000);
        assert_eq!(
            budget.prompt_allowance(),
            100_000 - 5_000 - 4_000 - REQUEST_OVERHEAD_TOKENS
        );
    }

    #[test]
    fn an_output_request_larger_than_the_window_leaves_no_allowance() {
        let budget = PromptBudget::new(8_000, 32_000);
        assert_eq!(budget.prompt_allowance(), 0);
    }

    #[test]
    fn the_evidence_and_memory_caps_are_shares_of_the_window() {
        let budget = PromptBudget::new(200_000, 8192);
        assert_eq!(budget.evidence_cap(), 120_000);
        assert_eq!(budget.memory_cap(), 20_000);
    }

    #[test]
    fn admits_rejects_a_request_that_would_eat_the_reserve() {
        let budget = PromptBudget::new(10_000, 1_000);
        // Reserve, output and request framing leave 7,960 content tokens.
        let fits = "x".repeat(7_000);
        assert!(budget.admits(&fits, &[]));

        let overflows = "x".repeat(8_000);
        assert!(!budget.admits(&overflows, &[]));
    }

    #[test]
    fn admits_counts_the_messages_too() {
        let budget = PromptBudget::new(10_000, 1_000);
        let system = "x".repeat(7_000);
        let message = LlmMessage::user("y".repeat(2_000));
        assert!(budget.admits(&system, &[]));
        assert!(!budget.admits(&system, std::slice::from_ref(&message)));
    }

    #[test]
    fn admits_counts_provider_native_document_blocks() {
        let budget = PromptBudget::new(2_500, 100);
        assert!(budget.admits_with_additional_prompt_tokens("system", &[], 1_300));
        assert!(!budget.admits_with_additional_prompt_tokens("system", &[], 1_400));
    }

    #[test]
    fn truncate_message_under_budget_returns_borrowed() {
        let result = truncate_message_content("short message", 2000);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn truncate_message_over_budget_returns_owned() {
        let content = "x".repeat(40_000); // ~10,000 tokens
        let result = truncate_message_content(&content, 2000);
        assert!(matches!(result, Cow::Owned(_)));
        assert!(result.contains("[...truncated, full response available in history]"));
        assert!(estimate_tokens(&result) <= 2000);
    }

    #[test]
    fn truncate_message_utf8_boundary_2byte() {
        let content = "é".repeat(10_000);
        let result = truncate_message_content(&content, 500);
        let marker_pos = result.find("[...truncated").expect("marker");
        assert!(result[..marker_pos].chars().all(|c| c == 'é' || c == '\n'));
    }

    #[test]
    fn truncate_message_utf8_boundary_cjk_and_emoji() {
        for content in ["中".repeat(5_000), "😀".repeat(5_000)] {
            let result = truncate_message_content(&content, 500);
            assert!(result.contains("[...truncated"));
            assert!(estimate_tokens(&result) <= 500);
        }
    }

    #[test]
    fn truncate_message_empty_and_exact_budget_return_borrowed() {
        assert!(matches!(
            truncate_message_content("", 2000),
            Cow::Borrowed(_)
        ));
        let exact = "a".repeat(2000);
        assert_eq!(estimate_tokens(&exact), 2000);
        assert!(matches!(
            truncate_message_content(&exact, 2000),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn history_that_fits_is_kept_whole_and_in_order() {
        let history = vec![
            LlmMessage::user("Hello"),
            LlmMessage::assistant("Hi there"),
            LlmMessage::user("How are you?"),
        ];
        let fitted = fit_history(&history, 10_000);
        assert!(!fitted.was_truncated);
        assert_eq!(fitted.messages.len(), 3);
        assert_eq!(fitted.messages[0].content, "Hello");
        assert!(fitted.dropped.is_empty());
        assert!(fitted.tokens > 0);
    }

    #[test]
    fn history_drops_the_oldest_first() {
        let history: Vec<LlmMessage> = (0..10)
            .map(|i| LlmMessage::user(format!("msg-{i}: {}", "w".repeat(1_000))))
            .collect();
        // Each message costs just over 1,000 tokens including framing, so two
        // fit and a third does not.
        let fitted = fit_history(&history, 3_000);
        assert!(fitted.was_truncated);
        assert_eq!(fitted.messages.len(), 2);
        assert!(fitted.messages[0].content.starts_with("msg-8:"));
        assert!(fitted.messages[1].content.starts_with("msg-9:"));
        assert_eq!(fitted.dropped.len(), 8);
        assert!(fitted.dropped[0].content.starts_with("msg-0:"));
    }

    #[test]
    fn one_oversized_message_is_truncated_before_it_is_weighed() {
        let history = vec![LlmMessage::assistant("x".repeat(100_000))]; // ~25,000 tokens
        let fitted = fit_history(&history, 3_000);
        assert_eq!(fitted.messages.len(), 1);
        assert!(
            fitted.messages[0]
                .content
                .contains("[...truncated, full response available in history]")
        );
        assert!(fitted.tokens <= MAX_TOKENS_PER_HISTORY_MESSAGE + MESSAGE_OVERHEAD_TOKENS);
    }

    #[test]
    fn a_zero_budget_keeps_no_history() {
        let history = vec![LlmMessage::user("Hello")];
        let fitted = fit_history(&history, 0);
        assert!(fitted.messages.is_empty());
        assert_eq!(fitted.dropped.len(), 1);
        assert!(fitted.was_truncated);
    }
}
