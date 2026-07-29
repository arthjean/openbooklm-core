//! Memory extraction: signal filtering, LLM extraction, response parsing, metadata enrichment.

use uuid::Uuid;

use crate::clients::MistralClient;
use crate::core::providers::EmbeddingProvider;
use crate::error::AppError;
use crate::repositories::MemoryRepository;

use super::deduplication::resolve_upsert_action;
use super::prompts::{EXTRACTION_SYSTEM_PROMPT, escape_xml_chars};
use super::types::{ExtractedMemory, MemoryAction};

// ============================================================================
// Signal filtering
// ============================================================================

/// Trivial patterns that should skip memory extraction entirely.
/// Case-insensitive exact match after stripping punctuation.
const TRIVIAL_PATTERNS: &[&str] = &[
    "ok",
    "okay",
    "yes",
    "no",
    "yep",
    "nope",
    "sure",
    "thanks",
    "thank you",
    "merci",
    "oui",
    "non",
    "d'accord",
    "ok merci",
    "parfait",
    "super",
    "great",
    "got it",
    "understood",
    "noted",
    "cool",
    "awesome",
    "nice",
    "good",
    "fine",
    "alright",
    "right",
];

/// Returns `true` if the message is too trivial to extract memories from.
///
/// Two-layer check:
/// 1. Length gate: messages under 20 chars are trivial
/// 2. Pattern match: common acknowledgments in English + French
#[must_use]
pub fn is_trivial_message(msg: &str) -> bool {
    let trimmed = msg.trim();

    // Layer 1: length gate
    if trimmed.len() < 20 {
        return true;
    }

    // Layer 2: trivial pattern matching
    let stripped: String = trimmed
        .chars()
        .filter(|c| c.is_alphabetic() || c.is_whitespace())
        .collect();
    let stripped = stripped.trim().to_lowercase();

    TRIVIAL_PATTERNS.iter().any(|p| stripped == *p)
}

// ============================================================================
// Extraction
// ============================================================================

/// Maximum number of source names included in the extraction prompt.
const MAX_SOURCE_NAMES: usize = 20;

/// Maximum length (in characters) of each source name in the extraction prompt.
const MAX_SOURCE_NAME_LEN: usize = 50;

/// Maximum characters of user message passed to the extraction LLM prompt.
/// ~1k tokens — the extraction LLM only needs enough context to understand the user's intent.
const MAX_USER_MSG_FOR_EXTRACTION: usize = 4_000;

/// Maximum characters of assistant response passed to the extraction LLM prompt.
/// Keeps extraction cost bounded while providing sufficient conversational context.
const MAX_ASSISTANT_FOR_EXTRACTION: usize = 2_000;

/// Truncate a string at the nearest char boundary at or below `max_bytes`.
fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Build the user prompt for memory extraction, optionally including notebook source names.
///
/// When `source_names` is non-empty, a `<notebook_sources>` XML block is prepended
/// before the user/assistant messages. Names are truncated to [`MAX_SOURCE_NAME_LEN`]
/// characters and the list is capped at [`MAX_SOURCE_NAMES`] entries.
/// Source names are sanitized to prevent XML structural injection.
///
/// User message and assistant response are length-capped and XML-escaped to prevent
/// prompt injection and unbounded cost.
pub(crate) fn build_extraction_user_prompt(
    user_message: &str,
    assistant_response: &str,
    source_names: &[String],
) -> String {
    let mut prompt = String::new();

    if !source_names.is_empty() {
        prompt.push_str("<notebook_sources>\n");
        for name in source_names.iter().take(MAX_SOURCE_NAMES) {
            // Truncate to first 50 chars, escape XML special chars, strip newlines
            let truncated: String = name.chars().take(MAX_SOURCE_NAME_LEN).collect();
            let sanitized: String = escape_xml_chars(&truncated)
                .chars()
                .filter(|c| !matches!(c, '\n' | '\r'))
                .collect();
            if sanitized.is_empty() {
                continue;
            }
            prompt.push_str("- ");
            prompt.push_str(&sanitized);
            prompt.push('\n');
        }
        prompt.push_str("</notebook_sources>\n\n");
    }

    // Cap content length to bound extraction LLM cost and limit injection surface
    let user_capped = truncate_at_char_boundary(user_message, MAX_USER_MSG_FOR_EXTRACTION);
    let assistant_capped =
        truncate_at_char_boundary(assistant_response, MAX_ASSISTANT_FOR_EXTRACTION);

    prompt.push_str("<user_message>\n");
    prompt.push_str(&escape_xml_chars(user_capped));
    prompt.push_str("\n</user_message>\n\n<assistant_response>\n");
    prompt.push_str(&escape_xml_chars(assistant_capped));
    prompt.push_str("\n</assistant_response>\n\nExtract memorable facts about the user.");

    prompt
}

/// Maximum length for the user message preview stored in metadata.
const MAX_USER_MESSAGE_PREVIEW: usize = 150;

/// Enrich extracted memories with runtime context metadata.
///
/// Adds `active_sources`, `user_message_preview`, and `conversation_turn`
/// to each memory's metadata. Called after extraction, before dedup/storage.
///
/// Note: `extracted_from_topic` and `source` are set earlier in
/// [`parse_extraction_response()`] from the LLM output. This function adds
/// the runtime-only fields that aren't part of the LLM response.
pub(crate) fn enrich_metadata(
    memories: &mut [ExtractedMemory],
    source_names: &[String],
    user_message: &str,
    conversation_turn: usize,
) {
    // Build user message preview: first 150 characters, truncated at char boundary
    let preview = if user_message.len() <= MAX_USER_MESSAGE_PREVIEW {
        user_message.to_string()
    } else {
        let truncate_at = user_message
            .char_indices()
            .take_while(|(i, _)| *i < MAX_USER_MESSAGE_PREVIEW)
            .last()
            .map_or_else(
                || MAX_USER_MESSAGE_PREVIEW.min(user_message.len()),
                |(i, c)| i + c.len_utf8(),
            );
        user_message[..truncate_at].to_string()
    };

    // Build enrichment values once, share across all memories
    let sources_value = serde_json::Value::Array(
        source_names
            .iter()
            .take(MAX_SOURCE_NAMES)
            .map(|s| serde_json::Value::String(s.clone()))
            .collect(),
    );
    let preview_value = serde_json::Value::String(preview);
    let turn_value = serde_json::json!(conversation_turn);

    for mem in memories.iter_mut() {
        // Graceful fallback: if metadata is somehow not an object, replace it.
        if !mem.metadata.is_object() {
            tracing::warn!("Memory metadata is not a JSON object, replacing with empty object");
            mem.metadata = serde_json::json!({});
        }
        if let Some(obj) = mem.metadata.as_object_mut() {
            obj.insert("active_sources".to_string(), sources_value.clone());
            obj.insert("user_message_preview".to_string(), preview_value.clone());
            obj.insert("conversation_turn".to_string(), turn_value.clone());
        }
    }
}

/// Extract structured memories from a chat exchange using Mistral Small.
///
/// Returns a list of [`MemoryAction`]s after signal filtering, extraction,
/// embedding, and deduplication against existing memories.
#[allow(clippy::too_many_arguments)]
pub async fn extract_memories(
    mistral: &MistralClient,
    embeddings: &dyn EmbeddingProvider,
    user_message: &str,
    assistant_response: &str,
    notebook_id: Uuid,
    memory_repo: &dyn MemoryRepository,
    source_names: &[String],
    conversation_turn: usize,
) -> Result<Vec<MemoryAction>, AppError> {
    // Step 1: signal filter
    if is_trivial_message(user_message) {
        tracing::debug!("Skipping memory extraction: trivial message");
        return Ok(vec![]);
    }

    // Step 2: call Mistral to extract facts
    let user_prompt = build_extraction_user_prompt(user_message, assistant_response, source_names);

    let raw_json = mistral
        .complete_json(EXTRACTION_SYSTEM_PROMPT, &user_prompt, 1024)
        .await?;

    // Step 3: parse extraction response
    let mut extracted = parse_extraction_response(&raw_json)?;
    if extracted.is_empty() {
        return Ok(vec![]);
    }

    // Step 3b: enrich metadata with runtime context
    enrich_metadata(
        &mut extracted,
        source_names,
        user_message,
        conversation_turn,
    );

    // Step 4: batch-embed all extracted memories at once (single API call)
    let texts: Vec<String> = extracted.iter().map(|m| m.content.clone()).collect();
    let vectors = embeddings.embed_documents(&texts).await?;

    // Step 5: batch-deduplicate against existing memories (single query instead of N)
    let batch_results = memory_repo
        .batch_search_similar(notebook_id, &vectors, 5)
        .await?;

    let actions: Vec<MemoryAction> = extracted
        .into_iter()
        .zip(vectors)
        .zip(batch_results)
        .map(|((mem, embedding), similar)| resolve_upsert_action(mem, &embedding, &similar))
        .collect();

    Ok(actions)
}

/// Execute a list of [`MemoryAction`]s against the repository.
///
/// When `memory_limit` is provided, inserts are skipped once the notebook
/// reaches the plan's memory cap. Updates and skips are unaffected.
pub async fn apply_memory_actions(
    notebook_id: Uuid,
    actions: Vec<MemoryAction>,
    memory_repo: &dyn MemoryRepository,
    memory_limit: Option<i32>,
) -> Result<usize, AppError> {
    // Pre-check: if a limit is set, get the current count once
    let mut current_count = match memory_limit {
        Some(_) => {
            i32::try_from(memory_repo.count_for_notebook(notebook_id).await?).unwrap_or(i32::MAX)
        }
        None => 0,
    };

    let mut applied = 0;
    for action in actions {
        match action {
            MemoryAction::Insert { memory, embedding } => {
                // Enforce memory limit: silently skip inserts when at cap
                if let Some(limit) = memory_limit
                    && current_count >= limit
                {
                    tracing::debug!(
                        %notebook_id,
                        current_count,
                        limit,
                        "Memory limit reached, skipping insert"
                    );
                    continue;
                }

                memory_repo
                    .create_with_embedding(
                        notebook_id,
                        &memory.content,
                        &memory.memory_type,
                        memory.metadata,
                        memory.salience,
                        &embedding,
                    )
                    .await?;
                current_count += 1;
                applied += 1;
            }
            MemoryAction::Update {
                existing_id,
                new_content,
                new_salience,
                embedding,
            } => {
                // Metadata is intentionally None — updates preserve original metadata.
                // Only content, salience, and embedding are refreshed on dedup merge.
                memory_repo
                    .update(
                        existing_id,
                        Some(new_content),
                        Some(new_salience),
                        None,
                        Some(&embedding),
                    )
                    .await?;
                applied += 1;
            }
            MemoryAction::Skip => {}
        }
    }
    Ok(applied)
}

// ============================================================================
// Response parsing
// ============================================================================

/// Valid memory types from the PRD.
pub(crate) const VALID_MEMORY_TYPES: &[&str] = &[
    "fact",
    "preference",
    "summary",
    "expertise",
    "goal",
    "conversation_summary",
];

/// Parse the raw JSON string from the LLM into a list of [`ExtractedMemory`].
pub(crate) fn parse_extraction_response(raw: &str) -> Result<Vec<ExtractedMemory>, AppError> {
    // Strip markdown code fences that small models sometimes add
    let cleaned = strip_code_fences(raw);

    let v: serde_json::Value = serde_json::from_str(cleaned).map_err(|e| {
        tracing::warn!(error = %e, raw = %raw, "Failed to parse memory extraction JSON");
        AppError::Internal(format!("Memory extraction parse failed: {e}"))
    })?;

    let Some(memories) = v.get("memories").and_then(|m| m.as_array()) else {
        tracing::debug!("No 'memories' array in extraction response");
        return Ok(vec![]);
    };

    let extracted: Vec<ExtractedMemory> = memories
        .iter()
        .filter_map(|m| {
            let content = m.get("content")?.as_str()?.trim().to_string();
            // Cap content at 500 bytes to bound embedding cost and prevent LLM output abuse
            if content.is_empty() || content.len() > 500 {
                return None;
            }

            let memory_type = m.get("memory_type")?.as_str()?;
            if !VALID_MEMORY_TYPES.contains(&memory_type) {
                return None;
            }

            let salience = m
                .get("salience")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.7);
            #[allow(clippy::cast_possible_truncation)]
            let salience = salience as f32;

            // Parse optional context field from LLM output (capped at 200 bytes)
            let context = m
                .get("context")
                .and_then(|c| c.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s.len() <= 200);

            let mut metadata = serde_json::json!({
                "source": "chat_extraction",
            });
            // Store extraction topic from LLM context in metadata
            if let Some(ref ctx) = context {
                metadata["extracted_from_topic"] = serde_json::Value::String(ctx.clone());
            }

            Some(ExtractedMemory {
                content,
                memory_type: memory_type.to_string(),
                salience: salience.clamp(0.0, 1.0),
                metadata,
                context,
            })
        })
        .collect();

    Ok(extracted)
}

/// Strip markdown code fences that the LLM may wrap around JSON output.
pub(crate) fn strip_code_fences(s: &str) -> &str {
    let s = s.trim();
    let s = s.strip_prefix("```json").unwrap_or(s);
    let s = s.strip_prefix("```").unwrap_or(s);
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim()
}
