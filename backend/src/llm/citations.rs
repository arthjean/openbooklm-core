//! Citation extraction from LLM responses.
//!
//! Parses `[N]` markers in Markdown text, skipping code spans and fenced blocks,
//! and maps them to source chunks.

use std::{borrow::Cow, collections::HashSet, sync::LazyLock};

use regex::Regex;

use super::types::{CitableChunk, Citation};

/// Pre-compiled citation regex [N].
static CITATION_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+)\]").expect("valid regex"));

/// Extract citations from LLM response based on [N] markers.
///
/// Skips `[N]` patterns inside inline code (`` `...` ``) and fenced code blocks
/// (`` ```...``` ``), and ignores out-of-bounds references.
pub fn extract_citations(response: &str, context_chunks: &[CitableChunk]) -> Vec<Citation> {
    let code_ranges = find_code_ranges(response);
    let mut citations = Vec::new();
    let mut seen = HashSet::new();
    let mut marker_count = 0;

    for cap in CITATION_REGEX.captures_iter(response) {
        marker_count += 1;

        let full_match = cap.get(0).expect("capture group 0 always exists");
        // Skip citations inside code blocks
        if is_in_code_range(full_match.start(), &code_ranges) {
            continue;
        }

        let Some(index_match) = cap.get(1) else {
            continue;
        };
        let Ok(index) = index_match.as_str().parse::<usize>() else {
            continue;
        };

        // Citations are 1-indexed; [0] is invalid
        if index == 0 {
            tracing::warn!(
                citation_index = 0,
                "Citation [0] is invalid (1-indexed), skipping"
            );
            continue;
        }

        // Convert 1-indexed to 0-indexed
        let chunk_idx = index - 1;

        if seen.contains(&chunk_idx) {
            continue;
        }

        // Skip out-of-bounds citations
        let Some(chunk) = context_chunks.get(chunk_idx) else {
            tracing::warn!(
                citation_index = index,
                available = context_chunks.len(),
                "Citation references non-existent chunk"
            );
            continue;
        };

        seen.insert(chunk_idx);

        // Extract metadata fields for enriched citations (section headers, YouTube timestamps).
        let meta = chunk.metadata.as_ref();
        let section_header = meta
            .and_then(|m| m.get("section_header"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let page_number = meta
            .and_then(|m| m.get("page_number"))
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok());
        let timestamp_start = meta
            .and_then(|m| m.get("timestamp_start"))
            .and_then(|v| v.as_f64());
        let timestamp_end = meta
            .and_then(|m| m.get("timestamp_end"))
            .and_then(|v| v.as_f64());
        let video_id = meta
            .and_then(|m| m.get("video_id"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let citation_url = meta
            .and_then(|m| m.get("citation_url"))
            .and_then(|v| v.as_str())
            .map(String::from);

        citations.push(Citation {
            source_id: chunk.source_id,
            chunk_index: chunk.chunk_index,
            text: truncate_text(&chunk.content, 200).into_owned(),
            relevance_score: chunk.relevance_score,
            section_header,
            page_number,
            timestamp_start,
            timestamp_end,
            video_id,
            citation_url,
        });

        tracing::debug!(
            citation_index = index,
            source_id = %chunk.source_id,
            "Citation matched"
        );
    }

    tracing::info!(
        citations = citations.len(),
        markers = marker_count,
        "Citation extraction complete"
    );

    citations
}

/// Find byte ranges of code spans in a Markdown string.
///
/// Handles fenced code blocks (`` ``` ```) and inline code (`` ` ``).
/// Returns sorted, non-overlapping `(start, end)` ranges.
pub fn find_code_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Fenced code block: ``` at line start (possibly after whitespace)
        if bytes[i] == b'`' && i + 2 < len && bytes[i + 1] == b'`' && bytes[i + 2] == b'`' {
            let start = i;
            // Skip the opening fence and the rest of the line (info string)
            i += 3;
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            // Find the closing fence
            let mut found_close = false;
            while i < len {
                if bytes[i] == b'\n'
                    && i + 3 < len
                    && bytes[i + 1] == b'`'
                    && bytes[i + 2] == b'`'
                    && bytes[i + 3] == b'`'
                {
                    i += 4; // skip \n```
                    // skip rest of closing fence line
                    while i < len && bytes[i] != b'\n' {
                        i += 1;
                    }
                    found_close = true;
                    break;
                }
                i += 1;
            }
            if !found_close {
                i = len; // unclosed fence extends to end
            }
            ranges.push((start, i));
        }
        // Inline code: `...`
        else if bytes[i] == b'`' {
            let start = i;
            i += 1;
            while i < len && bytes[i] != b'`' {
                i += 1;
            }
            if i < len {
                i += 1; // skip closing `
            }
            ranges.push((start, i));
        } else {
            i += 1;
        }
    }

    ranges
}

/// Check if a byte offset falls within any code range.
pub fn is_in_code_range(offset: usize, ranges: &[(usize, usize)]) -> bool {
    ranges
        .iter()
        .any(|&(start, end)| offset >= start && offset < end)
}

/// Truncate text to max characters with ellipsis (UTF-8 safe).
pub fn truncate_text(text: &str, max_chars: usize) -> Cow<'_, str> {
    let char_count = text.chars().count();

    if char_count <= max_chars {
        return Cow::Borrowed(text);
    }

    let truncate_at = max_chars.saturating_sub(3);
    text.char_indices()
        .nth(truncate_at)
        .map(|(idx, _)| Cow::Owned(format!("{}...", &text[..idx])))
        .unwrap_or(Cow::Borrowed(text))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn make_chunk(source_id: &str, content: &str) -> CitableChunk {
        CitableChunk {
            source_id: Uuid::parse_str(source_id).unwrap_or_else(|_| Uuid::new_v4()),
            chunk_index: 0,
            content: content.to_string(),
            relevance_score: 0.9,
            metadata: None,
        }
    }

    #[test]
    fn extract_citations_skips_inline_code() {
        let chunks = vec![
            make_chunk("00000000-0000-0000-0000-000000000001", "chunk zero"),
            make_chunk("00000000-0000-0000-0000-000000000002", "chunk one"),
        ];
        // [1] inside inline code should be skipped, [2] outside should match
        let response = "Use `array[1]` for access [2].";
        let citations = extract_citations(response, &chunks);
        assert_eq!(citations.len(), 1, "Only [2] outside code should be cited");
        assert_eq!(citations[0].chunk_index, 0); // chunk index 1 (0-indexed)
    }

    #[test]
    fn extract_citations_skips_fenced_code() {
        let chunks = vec![
            make_chunk("00000000-0000-0000-0000-000000000001", "chunk zero"),
            make_chunk("00000000-0000-0000-0000-000000000002", "chunk one"),
        ];
        let response = "See below [2]:\n```python\narr[1] = 42\n```\nDone.";
        let citations = extract_citations(response, &chunks);
        assert_eq!(citations.len(), 1);
    }

    #[test]
    fn extract_citations_skips_out_of_bounds() {
        let chunks = vec![make_chunk(
            "00000000-0000-0000-0000-000000000001",
            "only chunk",
        )];
        // [15] is out of bounds (only 1 chunk available)
        let response = "This is true [1] and also [15].";
        let citations = extract_citations(response, &chunks);
        assert_eq!(citations.len(), 1, "Only valid [1] should be cited");
    }

    #[test]
    fn find_code_ranges_inline() {
        let text = "before `code [0] here` after";
        let ranges = find_code_ranges(text);
        assert_eq!(ranges.len(), 1);
        assert!(text[ranges[0].0..ranges[0].1].contains("[0]"));
    }

    #[test]
    fn find_code_ranges_fenced() {
        let text = "before\n```\ncode [1]\n```\nafter [2]";
        let ranges = find_code_ranges(text);
        assert_eq!(ranges.len(), 1);
        assert!(is_in_code_range(text.find("[1]").unwrap(), &ranges));
        assert!(!is_in_code_range(text.find("[2]").unwrap(), &ranges));
    }

    #[test]
    fn extract_citations_rejects_zero_index() {
        let chunks = vec![make_chunk(
            "00000000-0000-0000-0000-000000000001",
            "chunk zero",
        )];
        let response = "See [0]";
        let citations = extract_citations(response, &chunks);
        assert_eq!(citations.len(), 0, "Citation [0] should be rejected");
    }

    #[test]
    fn extract_citations_zero_does_not_map_to_first_chunk() {
        let chunks = vec![
            make_chunk("00000000-0000-0000-0000-000000000001", "first"),
            make_chunk("00000000-0000-0000-0000-000000000002", "second"),
            make_chunk("00000000-0000-0000-0000-000000000003", "third"),
        ];
        let response = "See [0] and [1]";
        let citations = extract_citations(response, &chunks);
        assert_eq!(
            citations.len(),
            1,
            "Only [1] should produce a citation, [0] is invalid"
        );
        assert_eq!(citations[0].source_id, chunks[0].source_id);
    }

    #[test]
    fn truncate_text_short() {
        assert_eq!(truncate_text("hi", 10), "hi");
    }

    #[test]
    fn truncate_text_long() {
        let result = truncate_text("hello world, this is a test", 10);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 13); // 10 chars + "..."
    }
}
