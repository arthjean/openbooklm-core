//! Extraction prompt template and prompt formatting for memory system prompt injection.

use crate::entities::notebook_memory;
use crate::repositories::MemorySearchResult;

// ============================================================================
// Extraction prompt
// ============================================================================

pub(crate) const EXTRACTION_SYSTEM_PROMPT: &str = r#"You are a memory extraction assistant for a document research tool. Extract memorable facts about the user from their chat messages.

The user's notebook may contain sources listed in a <notebook_sources> block in the user prompt. When the conversation references these sources, include the relevant source name in the memory content or context field.

## Output schema
Return ONLY a valid JSON object — no prose, no markdown code fences:
{
  "memories": [
    {
      "content": "specific, self-contained statement about the user (25-40 words)",
      "memory_type": "fact|preference|expertise|goal|summary",
      "salience": 0.0-1.0,
      "context": "what topic/source/question triggered this memory"
    }
  ]
}

## Memory types
- "fact"       — concrete information stated by the user (occupation, project, institution, dataset, specific detail)
- "preference" — how the user wants responses or engages with material (depth, format, statistical rigor, language)
- "expertise"  — demonstrated knowledge level and domain (inferred from vocabulary, questions asked, corrections given)
- "goal"       — stated research objective or question in this notebook
- "summary"    — insight, cross-source conclusion, or synthesis the user reached during the conversation. For summary-type memories, capture the specific insight or conclusion reached, referencing which sources or arguments led to it.

## What makes a good memory
Write memories as specific, self-contained third-person statements that include the topic, depth of knowledge, and source context when available. Avoid generic statements.
- INCLUDE: domain, depth signal, source reference when relevant, specific topics not general categories
- AVOID: generic category labels ("interested in X"), single-word topics, statements true of most users

## Salience guide
- 0.9-1.0: Explicit, unambiguous self-disclosure with specifics. Examples: "I am a cardiologist", "My goal is to find evidence for X in these sources", "I work at Y hospital on Z project." These facts directly shape every future response.
- 0.6-0.8: Strong signal with domain specificity but inferred rather than stated. Examples: requesting hazard ratios and CIs (reveals statistical sophistication), asking about clinical application of a trial (reveals practitioner context), expressing surprise at a finding (reveals prior expectation and existing expertise).
- 0.3-0.5: Preference or context signal with limited specificity. Examples: format preferences, topic interest without depth evidence. Useful but not unique to this user.
- Below 0.3: Do not extract. Skip pleasantries, restatements of assistant output, acknowledgments ("ok", "got it", "interesting").

## Context field
The "context" field situates this memory in the conversation for retrieval purposes. Write it as: what topic was being discussed + which sources were referenced (if any) + what the user's intent was.

## Summary-type memories
When the user reaches a conclusion, discovers a connection between sources, or synthesizes information, extract it as a summary. Include which sources or arguments led to the insight.

Summary memories preserve the user's analytical work — the connections they drew, the patterns they noticed, or the conclusions they reached by comparing material. The "context" field MUST reference the specific sources compared or synthesized.

Example:
BAD:  {"content": "The user learned about heart failure treatment", "memory_type": "summary", "salience": 0.5, "context": "heart failure"}
GOOD: {"content": "The user concluded that SGLT2 inhibitors reduce heart failure hospitalization by ~25%, synthesizing data from the DAPA-HF and EMPEROR-Reduced trial sources.", "memory_type": "summary", "salience": 0.8, "context": "Synthesizing DAPA-HF and EMPEROR-Reduced trial data on SGLT2 hospitalization outcomes"}

## Extraction rules
1. Extract ONLY from the USER message. The assistant response is context only — do not extract from it.
2. Write each memory as a self-contained third-person statement that someone reading it later can understand without seeing the original conversation. Include the specific topic, not the general category.
3. One fact per memory entry — split compound statements into separate entries.
4. If <notebook_sources> are listed, include the relevant source name in the content or context when the user references it.
5. Omit pleasantries, filler words, and facts already obvious from context.
6. If no memorable facts exist, return {"memories": []}.

## Examples

### Example 1 — professional identity with source reference
User: "Can you summarize the hospitalization rate section in DAPA-HF Trial Results? I cite this regularly with my heart failure patients when discussing SGLT2 inhibitors."

BAD:  {"content": "The user is interested in cardiology", "memory_type": "expertise", "salience": 0.4, "context": "cardiology question"}
GOOD: {"content": "The user is a clinician who regularly cites SGLT2 inhibitor trial data with heart failure patients, specifically using the DAPA-HF Trial Results source for hospitalization outcomes.", "memory_type": "expertise", "salience": 0.95, "context": "Discussing DAPA-HF hospitalization data for clinical use"}

### Example 2 — cross-source synthesis (summary type)
User: "Both the EMPEROR-Reduced and DAPA-HF sources show roughly 25% hospitalization reduction — that consistency is stronger than I expected."

BAD:  {"content": "The user learned about heart failure treatment", "memory_type": "summary", "salience": 0.5, "context": "heart failure topic"}
GOOD: {"content": "The user synthesized EMPEROR-Reduced and DAPA-HF sources, noting consistent ~25% hospitalization reduction across both RCTs — finding the cross-trial consistency stronger than expected.", "memory_type": "summary", "salience": 0.85, "context": "Cross-trial synthesis of SGLT2 hospitalization outcomes"}

### Example 3 — implicit expertise signal (no direct self-disclosure)
User: "That summary is too simplified — I need the actual hazard ratios and 95% confidence intervals, not just percentages."

BAD:  {"content": "The user prefers detailed explanations", "memory_type": "preference", "salience": 0.4, "context": "asked for more detail"}
GOOD: {"content": "The user requires hazard ratios and 95% confidence intervals rather than simplified percentage summaries, indicating advanced statistical literacy or clinical research methodology background.", "memory_type": "preference", "salience": 0.78, "context": "Rejected simplified summary, requested formal statistical measures"}

## Language
Write memories in the same language as the user's message.

Output ONLY the JSON object. If there are no memorable facts, output {"memories": []}."#;

// ============================================================================
// Prompt formatting
// ============================================================================

/// Memory types that qualify for the always-on Core Memory block.
const CORE_MEMORY_TYPES: &[&str] = &["expertise", "goal", "preference"];

/// Salience threshold for core memory inclusion.
const CORE_SALIENCE_THRESHOLD: f32 = 0.5;

/// Maximum number of core memories. With context annotations (~50 tokens each),
/// 15 core memories ≈ 750 tokens — within the 10% token budget.
const MAX_CORE_MEMORIES: usize = 15;

/// Similarity threshold for working memory inclusion.
const WORKING_SIMILARITY_THRESHOLD: f32 = 0.6;

/// Maximum character length for context annotations in the memory prompt block.
const MAX_CONTEXT_ANNOTATION_LEN: usize = 60;

/// Select core memories from a full memory list.
///
/// Filters by type (expertise/goal/preference) and salience > 0.5,
/// sorted by salience descending, capped at ~500 tokens.
#[must_use]
pub fn select_core_memories(
    all_memories: &[notebook_memory::Model],
) -> Vec<&notebook_memory::Model> {
    let mut core: Vec<&notebook_memory::Model> = all_memories
        .iter()
        .filter(|m| {
            CORE_MEMORY_TYPES.contains(&m.memory_type.as_str())
                && m.salience > CORE_SALIENCE_THRESHOLD
        })
        .collect();

    core.sort_by(|a, b| {
        b.salience
            .partial_cmp(&a.salience)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    core.truncate(MAX_CORE_MEMORIES);
    core
}

/// Escape XML special characters to prevent injection into XML-structured prompts.
pub(crate) fn escape_xml_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

/// Truncate a string at a word boundary, appending "…" if it exceeds `max_chars`.
fn truncate_at_word_boundary(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let byte_pos: usize = s.char_indices().nth(max_chars).map_or(s.len(), |(i, _)| i);
    let boundary = s[..byte_pos]
        .rfind([' ', ','])
        .filter(|&b| b > 0)
        .unwrap_or(byte_pos);
    format!("{}…", &s[..boundary])
}

/// Format a single memory line with optional context annotation from metadata.
///
/// If `metadata.extracted_from_topic` exists and is non-empty, appends it as a
/// parenthetical: `[TYPE] content (context: annotation)`.
/// Both content and annotation are XML-escaped to prevent injection into the prompt structure.
/// Internal newlines are normalized to spaces so each memory remains a single prompt line.
fn format_memory_line(memory_type: &str, content: &str, metadata: &serde_json::Value) -> String {
    // Normalize newlines to spaces to prevent line-injection within the <core>/<working> block
    let sanitized_content = content.replace(['\n', '\r'], " ");
    let base = format!(
        "[{}] {}",
        escape_xml_chars(&memory_type.to_uppercase()),
        escape_xml_chars(&sanitized_content)
    );

    if let Some(topic) = metadata
        .get("extracted_from_topic")
        .and_then(|v| v.as_str())
    {
        let topic = topic.trim().replace(['\n', '\r'], " ");
        if !topic.is_empty() {
            let truncated = truncate_at_word_boundary(&topic, MAX_CONTEXT_ANNOTATION_LEN);
            let annotation = escape_xml_chars(&truncated);
            return format!("{base} (context: {annotation})");
        }
    }

    base
}

/// Format memories into a `<memory>` XML block for system prompt injection.
///
/// Two sections:
/// - `<core>`: always-on persona facts (expertise, goals, preferences)
/// - `<working>`: query-relevant facts from semantic search
///
/// Returns `None` if both sections are empty.
#[must_use]
pub fn format_memory_for_prompt(
    core_memories: &[&notebook_memory::Model],
    relevant_memories: &[MemorySearchResult],
) -> Option<String> {
    let mut parts = Vec::new();

    // Core memory section
    if !core_memories.is_empty() {
        let core_lines: Vec<String> = core_memories
            .iter()
            .map(|m| format_memory_line(&m.memory_type, &m.content, &m.metadata))
            .collect();
        parts.push(format!("<core>\n{}\n</core>", core_lines.join("\n")));
    }

    // Working memory section (filtered by similarity threshold)
    // Exclude conversation_summary type — those are injected into history, not the memory block
    let working_lines: Vec<String> = relevant_memories
        .iter()
        .filter(|r| {
            r.similarity >= WORKING_SIMILARITY_THRESHOLD
                && r.memory.memory_type != "conversation_summary"
        })
        .map(|r| format_memory_line(&r.memory.memory_type, &r.memory.content, &r.memory.metadata))
        .collect();

    if !working_lines.is_empty() {
        parts.push(format!(
            "<working>\n{}\n</working>",
            working_lines.join("\n")
        ));
    }

    if parts.is_empty() {
        return None;
    }

    Some(format!("<memory>\n{}\n</memory>", parts.join("\n")))
}
