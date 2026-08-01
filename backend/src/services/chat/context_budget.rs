//! The one budgeting pass over an assembled prompt (US-018).
//!
//! [`crate::llm::budget`] owns the token arithmetic and
//! [`crate::services::rag::search`] owns the price of a rendered context; this
//! module owns the decisions that need to see the evidence. It runs once per
//! turn, before the provider call, and it is the only place that decides what is
//! dropped.
//!
//! # Priority order
//!
//! Reserve and output tokens are removed from the window first and never
//! reallocated. What remains pays, in order, for the instructions and the
//! current question (mandatory), the memory block (dropped whole if it does not
//! fit), the retrieved evidence (lowest-ranked contexts removed first), and
//! finally the history (oldest turns removed first).
//!
//! `allocate` is that order as arithmetic, and both entry points go through it:
//! [`evidence_allowance`], which the retrieval pipeline gates context stuffing
//! on, and [`fit_prompt`], which spends it. Two implementations of one priority
//! order is two numbers that eventually disagree about whether a notebook
//! fits.
//!
//! # A parent that does not fit is not simply dropped
//!
//! Evidence is retrieved as child chunks that stand for a broader parent
//! passage, and the parent is what the prompt normally carries. When one parent
//! is too large for the remaining allowance, the matched child is sent in its
//! place provided the child *and its provenance* fit: a citation whose page and
//! source id were dropped to save tokens is a citation the reader cannot open
//! (US-018 AC-3).
//!
//! # Dropping stops at the first context that does not fit
//!
//! The contract is "complete lowest-ranked contexts are removed first", so the
//! pass admits from the top and stops. Skipping over a large context to admit a
//! smaller, lower-ranked one would silently reorder the evidence by size.

use crate::llm::LlmMessage;
use crate::llm::budget::{FittedHistory, PromptBudget, estimate_tokens, fit_history};
use crate::llm::prompts::EvidenceFormat;
use crate::services::rag::eval::trace::{ReasonCode, ReasonSet, TokenCounts};
use crate::services::rag::search::{entry_tokens, evidence_body, region_overhead_tokens};
use crate::types::SearchResult;

/// Everything the pass needs besides the evidence itself.
pub struct PromptInputs<'a> {
    pub budget: PromptBudget,
    pub format: EvidenceFormat,
    /// The system prompt without its evidence region **and without memory**.
    ///
    /// Memory is priced separately because it can be dropped whole; including
    /// it here as well would charge the turn for it twice and shrink the
    /// evidence allowance by its size.
    pub instructions: &'a str,
    pub memory: Option<&'a str>,
    /// The question this turn is asking.
    pub query: &'a str,
    /// Conversation history, oldest first, before truncation.
    pub history: &'a [LlmMessage],
}

/// What the pass decided.
#[derive(Debug)]
pub struct BudgetedPrompt {
    /// Contexts that fit, in their retrieval order. A context whose parent was
    /// too large carries its child instead, with `parent_content` cleared.
    pub evidence: Vec<SearchResult>,
    /// Whether the memory block survived the budget.
    pub memory_kept: bool,
    pub history: FittedHistory,
    /// Evidence tokens selected and dropped, for the retrieval trace.
    pub tokens: TokenCounts,
    pub reasons: ReasonSet,
}

/// Why no request could be assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetRefusal {
    /// Tokens the mandatory components need.
    pub needed: usize,
    /// Tokens the window allows the prompt.
    pub allowance: usize,
}

/// How the window is divided, before any evidence is looked at.
///
/// The one place the priority order is arithmetic. Both entry points go through
/// it, so the number the retrieval pipeline gates stuffing on and the number the
/// fitting pass spends are the same number by construction rather than by two
/// implementations agreeing.
struct Allocation {
    memory_kept: bool,
    memory_tokens: usize,
    /// Tokens left for evidence and history together.
    remaining: usize,
    /// Tokens evidence may occupy: `remaining`, capped by the evidence share.
    evidence_budget: usize,
}

/// Divide the window, or refuse the turn.
///
/// # Errors
/// Returns [`BudgetRefusal`] when the instructions and the current question
/// alone exceed the prompt allowance.
fn allocate(
    budget: &PromptBudget,
    instructions: &str,
    memory: Option<&str>,
    query: &str,
) -> Result<Allocation, BudgetRefusal> {
    let mandatory = estimate_tokens(instructions) + estimate_tokens(query);
    let allowance = budget.prompt_allowance();
    if mandatory > allowance {
        return Err(BudgetRefusal {
            needed: mandatory,
            allowance,
        });
    }
    let mut remaining = allowance - mandatory;

    // Memory is kept whole or not at all: truncating it cuts a fact in half.
    let memory_tokens = memory.map_or(0, estimate_tokens);
    let memory_kept =
        memory_tokens > 0 && memory_tokens <= budget.memory_cap() && memory_tokens <= remaining;
    if memory_kept {
        remaining -= memory_tokens;
    }

    Ok(Allocation {
        memory_kept,
        memory_tokens,
        remaining,
        evidence_budget: remaining.min(budget.evidence_cap()),
    })
}

/// Tokens available to evidence, given everything that outranks it.
///
/// Public because the retrieval pipeline gates context stuffing on the same
/// number (US-018 AC-2): "the whole notebook fits" has to mean the same thing
/// in both places, or stuffing loads a corpus the budget then trims.
///
/// A turn whose instructions alone overflow the window has no allowance at all,
/// which is the same answer [`fit_prompt`] gives it a moment later.
#[must_use]
pub fn evidence_allowance(
    budget: &PromptBudget,
    instructions: &str,
    memory: Option<&str>,
    query: &str,
) -> usize {
    allocate(budget, instructions, memory, query).map_or(0, |allocation| allocation.evidence_budget)
}

/// Fit evidence, memory and history into the window.
///
/// # Errors
/// Returns [`BudgetRefusal`] when the instructions and the current question
/// alone exceed the prompt allowance. The caller must not send a request in
/// that case (US-018 AC-5).
pub fn fit_prompt(
    inputs: &PromptInputs<'_>,
    evidence: Vec<SearchResult>,
) -> Result<BudgetedPrompt, BudgetRefusal> {
    let mut reasons = ReasonSet::default();

    let Allocation {
        memory_kept,
        memory_tokens,
        mut remaining,
        evidence_budget,
    } = allocate(
        &inputs.budget,
        inputs.instructions,
        inputs.memory,
        inputs.query,
    )?;
    if !memory_kept && memory_tokens > 0 {
        reasons.insert(ReasonCode::MemoryDroppedForBudget);
    }

    // --- Evidence: admitted from the top, in rank order -------------------
    let mut spent = if evidence.is_empty() {
        0
    } else {
        region_overhead_tokens(inputs.format)
    };
    let mut selected: Vec<SearchResult> = Vec::with_capacity(evidence.len());
    let mut dropped_tokens = 0usize;
    let mut admitting = true;

    for (position, mut context) in evidence.into_iter().enumerate() {
        let index = position + 1;
        let parent_cost = entry_tokens(inputs.format, index, &context, evidence_body(&context));

        if !admitting {
            dropped_tokens += parent_cost;
            continue;
        }

        if spent + parent_cost <= evidence_budget {
            spent += parent_cost;
            selected.push(context);
            continue;
        }

        // The parent does not fit. Its matched child may, and the child keeps
        // the same provenance attributes because they are rendered from the
        // result, not from the passage.
        let child_cost = context
            .parent_content
            .is_some()
            .then(|| entry_tokens(inputs.format, index, &context, &context.content));

        match child_cost {
            Some(cost) if spent + cost <= evidence_budget => {
                context.parent_content = None;
                spent += cost;
                selected.push(context);
                reasons.insert(ReasonCode::ParentDowngradedToChild);
            }
            _ => {
                dropped_tokens += parent_cost;
                admitting = false;
            }
        }
    }

    if dropped_tokens > 0 {
        reasons.insert(ReasonCode::EvidenceDroppedForBudget);
    }
    if selected.is_empty() {
        spent = 0;
    }
    remaining = remaining.saturating_sub(spent);

    // --- History: the remainder, oldest turns dropped first ---------------
    let history = fit_history(inputs.history, remaining);

    Ok(BudgetedPrompt {
        evidence: selected,
        memory_kept,
        history,
        tokens: TokenCounts {
            selected: spent,
            dropped: dropped_tokens,
        },
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use super::*;
    use crate::types::RetrievalScore;

    /// A context whose parent passage is `parent_chars` long and whose child is
    /// `child_chars` long.
    fn context(rank: usize, parent_chars: usize, child_chars: usize) -> SearchResult {
        SearchResult {
            chunk_id: Uuid::new_v4(),
            generation_id: Uuid::nil(),
            source_id: Uuid::new_v4(),
            source_title: format!("Doc {rank}"),
            chunk_index: i32::try_from(rank).unwrap_or(0),
            content: format!("c{rank}").repeat(child_chars / 2),
            parent_content: Some(format!("p{rank}").repeat(parent_chars / 2)),
            score: RetrievalScore::Rrf(1.0 - rank as f32 * 0.01),
            metadata: None,
            collapsed_children: Vec::new(),
        }
    }

    /// A context with no parent passage: nothing to downgrade to, so it is
    /// admitted whole or dropped whole.
    fn flat_context(rank: usize, chars: usize) -> SearchResult {
        SearchResult {
            parent_content: None,
            content: format!("f{rank}").repeat(chars / 2),
            ..context(rank, 2, 2)
        }
    }

    fn inputs(budget: PromptBudget) -> PromptInputs<'static> {
        PromptInputs {
            budget,
            format: EvidenceFormat::Inline,
            instructions: "system instructions",
            memory: None,
            query: "what is the retry budget",
            history: &[],
        }
    }

    #[test]
    fn everything_fits_when_the_window_is_large() {
        let evidence: Vec<SearchResult> = (0..5).map(|i| context(i, 400, 100)).collect();
        let fitted = fit_prompt(&inputs(PromptBudget::new(200_000, 8192)), evidence).expect("fits");
        assert_eq!(fitted.evidence.len(), 5);
        assert_eq!(fitted.tokens.dropped, 0);
        assert!(fitted.tokens.selected > 0);
        assert!(fitted.reasons.is_empty());
    }

    #[test]
    fn the_lowest_ranked_contexts_are_the_ones_removed() {
        // 8,000-char passages cost ~2,000 tokens each; a 16,000-token window
        // has a 9,600-token evidence cap, so only the first few survive. They
        // carry no parent, so there is no smaller form to fall back to and the
        // pass has to choose which contexts to keep.
        let evidence: Vec<SearchResult> = (0..10).map(|i| flat_context(i, 8_000)).collect();
        let titles: Vec<String> = evidence.iter().map(|c| c.source_title.clone()).collect();

        let fitted = fit_prompt(&inputs(PromptBudget::new(16_000, 1_000)), evidence).expect("fits");

        assert!(!fitted.evidence.is_empty() && fitted.evidence.len() < 10);
        for (position, kept) in fitted.evidence.iter().enumerate() {
            assert_eq!(
                kept.source_title, titles[position],
                "selection must keep the ranking prefix"
            );
        }
        assert!(fitted.tokens.dropped > 0);
        assert!(
            fitted
                .reasons
                .contains(ReasonCode::EvidenceDroppedForBudget)
        );
    }

    #[test]
    fn an_oversized_parent_is_replaced_by_its_matched_child() {
        // One context whose parent cannot fit but whose child can.
        let evidence = vec![context(0, 40_000, 200)];
        let fitted = fit_prompt(&inputs(PromptBudget::new(16_000, 1_000)), evidence).expect("fits");

        assert_eq!(fitted.evidence.len(), 1);
        assert!(
            fitted.evidence[0].parent_content.is_none(),
            "the child is what was sent"
        );
        assert!(fitted.reasons.contains(ReasonCode::ParentDowngradedToChild));
        // Provenance survives the downgrade: it is rendered from the result.
        assert_eq!(fitted.evidence[0].source_title, "Doc 0");
    }

    #[test]
    fn a_context_whose_child_does_not_fit_either_is_dropped_with_the_rest() {
        let evidence = vec![context(0, 200_000, 200_000), context(1, 40, 40)];
        let fitted = fit_prompt(&inputs(PromptBudget::new(16_000, 1_000)), evidence).expect("fits");
        assert!(fitted.evidence.is_empty());
        assert_eq!(fitted.tokens.selected, 0);
        assert!(fitted.tokens.dropped > 0);
    }

    #[test]
    fn memory_is_dropped_whole_rather_than_truncated() {
        let big_memory = "m".repeat(4 * 20_000); // ~20,000 tokens
        let budget = PromptBudget::new(100_000, 4_000);
        let params = PromptInputs {
            memory: Some(&big_memory),
            ..inputs(budget)
        };
        // 10% of 100,000 is 10,000, so a 20,000-token block cannot be kept.
        let fitted = fit_prompt(&params, vec![context(0, 400, 100)]).expect("fits");
        assert!(!fitted.memory_kept);
        assert!(fitted.reasons.contains(ReasonCode::MemoryDroppedForBudget));

        let small_memory = "m".repeat(400);
        let params = PromptInputs {
            memory: Some(&small_memory),
            ..inputs(budget)
        };
        let fitted = fit_prompt(&params, vec![context(0, 400, 100)]).expect("fits");
        assert!(fitted.memory_kept);
    }

    #[test]
    fn history_gets_what_evidence_left() {
        let history: Vec<LlmMessage> = (0..20)
            .map(|i| LlmMessage::user(format!("turn-{i} {}", "h".repeat(4_000)))) // ~1,000 tokens
            .collect();
        let budget = PromptBudget::new(16_000, 1_000);
        let params = PromptInputs {
            history: &history,
            ..inputs(budget)
        };

        let fitted = fit_prompt(&params, vec![flat_context(0, 8_000)]).expect("fits");
        assert!(fitted.history.was_truncated);
        assert!(!fitted.history.messages.is_empty());
        assert!(fitted.history.messages.len() < history.len());
        // The kept history is the newest suffix.
        assert!(
            fitted
                .history
                .messages
                .last()
                .expect("kept at least one")
                .content
                .starts_with("turn-19")
        );
    }

    #[test]
    fn instructions_that_cannot_fit_refuse_the_request() {
        let instructions = "i".repeat(4 * 40_000); // ~40,000 tokens
        let budget = PromptBudget::new(16_000, 1_000);
        let params = PromptInputs {
            instructions: &instructions,
            ..inputs(budget)
        };
        let refusal = fit_prompt(&params, Vec::new()).expect_err("must refuse");
        assert!(refusal.needed > refusal.allowance);
    }

    /// A pool whose parents are all too large is not dropped: each context
    /// falls back to its matched child, in rank order, and the evidence
    /// survives at lower resolution rather than disappearing (US-018 AC-3).
    #[test]
    fn a_pool_of_oversized_parents_degrades_to_children_rather_than_emptying() {
        let evidence: Vec<SearchResult> = (0..8).map(|i| context(i, 20_000, 200)).collect();
        let fitted = fit_prompt(&inputs(PromptBudget::new(16_000, 1_000)), evidence).expect("fits");

        assert_eq!(
            fitted.evidence.len(),
            8,
            "every context kept a form that fit"
        );
        assert!(
            fitted.evidence[0].parent_content.is_some(),
            "the top-ranked context still fit at full resolution"
        );
        assert!(
            fitted.evidence[1..]
                .iter()
                .all(|c| c.parent_content.is_none()),
            "the rest were sent as their children"
        );
        assert!(fitted.reasons.contains(ReasonCode::ParentDowngradedToChild));
        assert_eq!(fitted.tokens.dropped, 0);
    }

    #[test]
    fn native_documents_are_priced_and_bounded_too() {
        let evidence: Vec<SearchResult> = (0..10).map(|i| flat_context(i, 8_000)).collect();
        let budget = PromptBudget::new(16_000, 1_000);
        let params = PromptInputs {
            format: EvidenceFormat::NativeDocuments,
            ..inputs(budget)
        };
        let fitted = fit_prompt(&params, evidence).expect("fits");
        assert!(fitted.evidence.len() < 10, "native blocks are counted");
        assert!(fitted.tokens.selected <= budget.evidence_cap());
    }

    /// The advertised allowance and the budget the pass actually spends against
    /// are one computation, including when memory is in play. They used to be
    /// two, and they disagreed whenever memory fit its cap but not the window.
    #[test]
    fn the_allowance_helper_is_the_budget_the_pass_uses() {
        let budget = PromptBudget::new(16_000, 1_000);
        let memory = "m".repeat(4 * 1_200); // fits the 1,600-token cap
        for memory in [None, Some(memory.as_str())] {
            let params = PromptInputs {
                memory,
                ..inputs(budget)
            };
            let allowance = evidence_allowance(&budget, params.instructions, memory, params.query);

            let evidence: Vec<SearchResult> = (0..10).map(|i| flat_context(i, 8_000)).collect();
            let fitted = fit_prompt(&params, evidence).expect("fits");
            assert!(
                fitted.tokens.selected <= allowance,
                "the pass spent {} of an advertised {allowance}",
                fitted.tokens.selected
            );

            // And the advertised number is not merely an upper bound: one more
            // context than fits would have to exceed it.
            let one_more: Vec<SearchResult> = (0..11).map(|i| flat_context(i, 8_000)).collect();
            let saturated = fit_prompt(&params, one_more).expect("fits");
            assert_eq!(
                saturated.tokens.selected, fitted.tokens.selected,
                "the same allowance admits the same prefix"
            );
        }
    }

    /// A turn whose instructions alone overflow the window has no evidence
    /// allowance, so retrieval never loads a corpus for a request that cannot
    /// be sent.
    #[test]
    fn instructions_that_cannot_fit_advertise_no_allowance() {
        let instructions = "i".repeat(4 * 40_000);
        let budget = PromptBudget::new(16_000, 1_000);
        assert_eq!(
            evidence_allowance(&budget, &instructions, None, "question"),
            0
        );
    }

    #[test]
    fn no_evidence_costs_no_region_overhead() {
        let fitted =
            fit_prompt(&inputs(PromptBudget::new(200_000, 8192)), Vec::new()).expect("fits");
        assert_eq!(fitted.tokens.selected, 0);
        assert_eq!(fitted.tokens.dropped, 0);
    }
}
