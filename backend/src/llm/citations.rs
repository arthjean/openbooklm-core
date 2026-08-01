//! Citation extraction and verification (US-019).
//!
//! Parses `[N]` markers in Markdown text, skipping code spans and fenced blocks,
//! and maps them to source chunks.
//!
//! # A marker is not a citation
//!
//! A citation is emitted only when its marker resolves to a chunk that was
//! actually retrieved this turn, that chunk carries an index generation, its
//! recorded span and pages describe a passage that can exist, and the quoted
//! passage is a passage of it. Everything else — an index out of range, a `[0]`,
//! a marker inside a code block, an incoherent span, a quote the chunk does not
//! contain — is refused and counted. The count is what makes the failure visible
//! to the grounded-response report instead of silently lowering citation
//! coverage.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

use regex::Regex;

use super::types::{ChunkProvenance, CitableChunk, Citation};

/// Pre-compiled citation regex [N].
static CITATION_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[(\d+)\]").expect("valid regex"));

/// Citations an answer earned, and the markers it did not.
#[derive(Debug, Clone, Default)]
pub struct ExtractedCitations {
    pub citations: Vec<Citation>,
    /// Markers that named something unciteable: out of range, `[0]`, a chunk
    /// with no generation, or a marker written inside code.
    pub rejected: usize,
}

/// Extract citations from an LLM response, discarding the markers it cannot
/// justify.
///
/// Skips `[N]` patterns inside inline code (`` `...` ``) and fenced code blocks
/// (`` ```...``` ``), and refuses out-of-bounds references.
#[must_use]
pub fn extract_citations(response: &str, context_chunks: &[CitableChunk]) -> Vec<Citation> {
    extract_citations_verified(response, context_chunks).citations
}

/// [`extract_citations`], reporting how many markers were refused.
#[must_use]
pub fn extract_citations_verified(
    response: &str,
    context_chunks: &[CitableChunk],
) -> ExtractedCitations {
    extract_citations_with_active_generations(response, context_chunks, None)
}

/// Resolve citations against the active pointers read immediately before
/// emission. A chunk retrieved from a generation that was superseded while the
/// model was answering is stale and is refused.
#[must_use]
pub fn extract_citations_verified_against_active(
    response: &str,
    context_chunks: &[CitableChunk],
    active_generations: &HashMap<uuid::Uuid, uuid::Uuid>,
) -> ExtractedCitations {
    extract_citations_with_active_generations(response, context_chunks, Some(active_generations))
}

fn extract_citations_with_active_generations(
    response: &str,
    context_chunks: &[CitableChunk],
    active_generations: Option<&HashMap<uuid::Uuid, uuid::Uuid>>,
) -> ExtractedCitations {
    let code_ranges = find_code_ranges(response);
    let mut citations = Vec::new();
    let mut seen = HashSet::new();
    let mut marker_count = 0;
    let mut rejected = 0usize;

    for cap in CITATION_REGEX.captures_iter(response) {
        marker_count += 1;

        let full_match = cap.get(0).expect("capture group 0 always exists");
        // A marker inside a code block is code, not a citation. It is still
        // counted: an answer that only "cites" inside a fence has cited nothing.
        if is_in_code_range(full_match.start(), &code_ranges) {
            rejected += 1;
            continue;
        }

        let Some(index_match) = cap.get(1) else {
            rejected += 1;
            continue;
        };
        let Ok(index) = index_match.as_str().parse::<usize>() else {
            rejected += 1;
            continue;
        };

        // Citations are 1-indexed; [0] is invalid
        if index == 0 {
            tracing::warn!(
                citation_index = 0,
                "Citation [0] is invalid (1-indexed), skipping"
            );
            rejected += 1;
            continue;
        }

        // Convert 1-indexed to 0-indexed
        let chunk_idx = index - 1;

        // Skip out-of-bounds citations
        let Some(chunk) = context_chunks.get(chunk_idx) else {
            tracing::warn!(
                citation_index = index,
                available = context_chunks.len(),
                "Citation references non-existent chunk"
            );
            rejected += 1;
            continue;
        };

        // A chunk with no generation was not read from a published index.
        if chunk.generation_id.is_nil() {
            tracing::warn!(
                citation_index = index,
                source_id = %chunk.source_id,
                "Citation resolves to a chunk with no index generation, skipping"
            );
            rejected += 1;
            continue;
        }

        if active_generations.is_some_and(|active| {
            active.get(&chunk.source_id).copied() != Some(chunk.generation_id)
        }) {
            tracing::warn!(
                citation_index = index,
                source_id = %chunk.source_id,
                generation_id = %chunk.generation_id,
                "Citation resolves to a generation that is no longer active, skipping"
            );
            rejected += 1;
            continue;
        }

        // The span and pages the chunker recorded must describe a passage that
        // can exist. One that cannot was not written by this pipeline, and a
        // citation into it would open something the reader cannot verify.
        let provenance = ChunkProvenance::read(chunk.metadata.as_ref());
        if !provenance.is_coherent() {
            tracing::warn!(
                citation_index = index,
                source_id = %chunk.source_id,
                "Citation resolves to a chunk with an incoherent span, skipping"
            );
            rejected += 1;
            continue;
        }

        if active_generations.is_some()
            && !claim_is_supported_by(full_match.start(), response, &chunk.content)
        {
            tracing::warn!(
                citation_index = index,
                source_id = %chunk.source_id,
                "Citation evidence is not linked to the associated claim, skipping"
            );
            rejected += 1;
            continue;
        }

        // Validate every occurrence before deduplicating the public citation.
        // A repeated marker attached to an unsupported claim is a rejection,
        // even when an earlier occurrence of the same marker was valid.
        if !seen.insert(chunk_idx) {
            continue;
        }

        citations.push(Citation::new(
            chunk.source_id,
            chunk.chunk_index,
            truncate_text(&chunk.content, 200).into_owned(),
            chunk.relevance_score,
            provenance,
        ));

        tracing::debug!(
            citation_index = index,
            source_id = %chunk.source_id,
            "Citation matched"
        );
    }

    tracing::info!(
        citations = citations.len(),
        markers = marker_count,
        rejected,
        "Citation extraction complete"
    );

    ExtractedCitations {
        citations,
        rejected,
    }
}

/// Associate a marker with the immediately preceding claim and require a
/// conservative lexical entailment signal from the cited passage.
///
/// This deliberately prefers refusing a paraphrase over publishing an
/// unrelated citation. Numbers and negations are retained as content tokens,
/// which catches the common false-support case where subject words overlap but
/// the value or polarity differs.
#[must_use]
pub fn claim_is_supported_by(marker_start: usize, response: &str, evidence: &str) -> bool {
    let Some(prefix) = response.get(..marker_start) else {
        return false;
    };
    let claim = prefix
        .rsplit(['\n', '.', '!', '?'])
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| c.is_whitespace() || matches!(c, '-' | '*' | ':' | ';'));
    let claim_numbers = numeric_values(claim);
    let evidence_numbers: HashSet<String> = numeric_values(evidence).into_iter().collect();
    if claim_numbers
        .iter()
        .any(|number| !evidence_numbers.contains(number))
    {
        return false;
    }

    let claim_tokens = meaningful_tokens(claim);
    if claim_tokens.is_empty() {
        return false;
    }
    let evidence_tokens: HashSet<String> = meaningful_tokens(evidence).into_iter().collect();

    if claim_tokens
        .iter()
        .filter(|token| is_polarity_token(token))
        .any(|token| !evidence_tokens.contains(token.as_str()))
    {
        return false;
    }

    let matched = claim_tokens
        .iter()
        .filter(|token| evidence_tokens.contains(token.as_str()))
        .count();
    let required = claim_tokens.len().saturating_mul(2).div_ceil(3).max(1);
    matched >= required
}

fn is_polarity_token(token: &str) -> bool {
    matches!(
        token,
        "no" | "not" | "never" | "without" | "aucun" | "jamais" | "pas" | "sans"
    )
}

fn numeric_values(text: &str) -> Vec<String> {
    let mut values = numeric_literal_values(text);
    let tokens: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect();
    let mut index = 0;
    while index < tokens.len() {
        if tokens[index].chars().any(|c| c.is_ascii_digit()) {
            index += 1;
        } else if let Some((value, consumed)) = parse_word_number(&tokens[index..]) {
            values.push(value.to_string());
            index += consumed;
        } else {
            index += 1;
        }
    }
    values
}

fn numeric_literal_values(text: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut run = String::new();
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        if character.is_ascii_digit() {
            run.push(character);
            continue;
        }
        if !run.is_empty()
            && is_numeric_separator(character)
            && chars.peek().is_some_and(char::is_ascii_digit)
        {
            run.push(character);
            continue;
        }
        if run.is_empty()
            && matches!(character, '-' | '+')
            && chars.peek().is_some_and(char::is_ascii_digit)
        {
            run.push(character);
            continue;
        }
        if run.chars().any(|c| c.is_ascii_digit()) {
            runs.extend(normalize_numeric_run(&run));
        }
        run.clear();
    }
    if run.chars().any(|c| c.is_ascii_digit()) {
        runs.extend(normalize_numeric_run(&run));
    }
    runs
}

fn normalize_numeric_run(run: &str) -> Vec<String> {
    let (negative, unsigned) = match run.as_bytes().first() {
        Some(b'-') => (true, &run[1..]),
        Some(b'+') => (false, &run[1..]),
        _ => (false, run),
    };
    let groups: Vec<&str> = unsigned
        .split(is_numeric_separator)
        .filter(|group| !group.is_empty())
        .collect();
    if groups.is_empty() {
        return Vec::new();
    }
    if groups.len() == 1 {
        return vec![canonical_integer(groups[0], negative)];
    }

    let separators: Vec<char> = unsigned
        .chars()
        .filter(|character| is_numeric_separator(*character))
        .collect();
    let is_grouped_integer = groups[0].len() <= 3
        && groups[1..].iter().all(|group| group.len() == 3)
        && separators.len() == groups.len() - 1;
    if is_grouped_integer {
        return vec![canonical_integer(&groups.concat(), negative)];
    }

    if groups.len() == 2 && separators.len() == 1 && matches!(separators[0], '.' | ',') {
        let integer = groups[0].trim_start_matches('0');
        let fraction = groups[1].trim_end_matches('0');
        let integer = if integer.is_empty() { "0" } else { integer };
        if fraction.is_empty() {
            return vec![canonical_integer(integer, negative)];
        }
        let sign = if negative { "-" } else { "" };
        return vec![format!("{sign}{integer}.{fraction}")];
    }

    groups
        .iter()
        .enumerate()
        .map(|(index, group)| canonical_integer(group, negative && index == 0))
        .collect()
}

fn canonical_integer(digits: &str, negative: bool) -> String {
    let digits = digits.trim_start_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    if negative && digits != "0" {
        format!("-{digits}")
    } else {
        digits.to_owned()
    }
}

fn is_numeric_separator(character: char) -> bool {
    matches!(
        character,
        '.' | ',' | '_' | '\'' | '’' | ' ' | '\u{00a0}' | '\u{202f}'
    )
}

fn parse_word_number(tokens: &[String]) -> Option<(u128, usize)> {
    let mut total = 0_u128;
    let mut current = 0_u128;
    let mut consumed = 0;
    let mut saw_number = false;

    while consumed < tokens.len() {
        let token = tokens[consumed].as_str();
        if matches!(token, "and" | "et") && saw_number {
            if tokens
                .get(consumed + 1)
                .is_some_and(|next| is_number_component(next))
            {
                consumed += 1;
                continue;
            }
            break;
        }

        if matches!(token, "dozen" | "douzaine") {
            current = current.max(1).checked_mul(12)?;
            saw_number = true;
            consumed += 1;
            continue;
        }
        if token == "score" {
            current = current.max(1).checked_mul(20)?;
            saw_number = true;
            consumed += 1;
            continue;
        }
        if matches!(token, "hundred" | "cent") {
            current = current.max(1).checked_mul(100)?;
            saw_number = true;
            consumed += 1;
            continue;
        }
        if let Some(scale) = number_scale(token) {
            total = total.checked_add(current.max(1).checked_mul(scale)?)?;
            current = 0;
            saw_number = true;
            consumed += 1;
            continue;
        }

        let Some((atom, atom_tokens)) = number_atom(tokens, consumed) else {
            break;
        };
        current = current.checked_add(atom)?;
        saw_number = true;
        consumed += atom_tokens;
    }

    saw_number.then(|| total.checked_add(current).map(|value| (value, consumed)))?
}

fn number_atom(tokens: &[String], index: usize) -> Option<(u128, usize)> {
    let token = tokens.get(index)?.as_str();
    if token == "quatre"
        && tokens
            .get(index + 1)
            .is_some_and(|next| matches!(next.as_str(), "vingt" | "vingts"))
    {
        return Some((80, 2));
    }
    let value = match token {
        "zero" | "zéro" => 0,
        "one" | "un" => 1,
        "two" | "deux" => 2,
        "three" | "trois" => 3,
        "four" | "quatre" => 4,
        "five" | "cinq" => 5,
        "six" => 6,
        "seven" | "sept" => 7,
        "eight" | "huit" => 8,
        "nine" | "neuf" => 9,
        "ten" | "dix" => 10,
        "eleven" | "onze" => 11,
        "twelve" | "douze" => 12,
        "thirteen" | "treize" => 13,
        "fourteen" | "quatorze" => 14,
        "fifteen" | "quinze" => 15,
        "sixteen" | "seize" => 16,
        "seventeen" => 17,
        "eighteen" => 18,
        "nineteen" => 19,
        "twenty" | "vingt" | "vingts" => 20,
        "thirty" | "trente" => 30,
        "forty" | "quarante" => 40,
        "fifty" | "cinquante" => 50,
        "sixty" | "soixante" => 60,
        "seventy" => 70,
        "eighty" => 80,
        "ninety" => 90,
        _ => return None,
    };
    Some((value, 1))
}

fn number_scale(token: &str) -> Option<u128> {
    match token {
        "thousand" | "mille" => Some(1_000),
        "million" => Some(1_000_000),
        "billion" | "milliard" => Some(1_000_000_000),
        "trillion" => Some(1_000_000_000_000),
        _ => None,
    }
}

fn is_number_component(token: &str) -> bool {
    token.chars().any(|c| c.is_ascii_digit())
        || matches!(
            token,
            "and" | "et" | "dozen" | "douzaine" | "score" | "hundred" | "cent"
        )
        || number_scale(token).is_some()
        || number_atom(&[token.to_owned()], 0).is_some()
}

fn meaningful_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter_map(|raw| {
            let token = raw.to_lowercase();
            ((token.len() >= 2 || token.chars().any(|c| c.is_ascii_digit()))
                && !is_stop_word(&token))
            .then_some(token)
            .filter(|token| !is_number_component(token))
        })
        .collect()
}

fn is_stop_word(token: &str) -> bool {
    matches!(
        token,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "for"
            | "from"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "that"
            | "the"
            | "this"
            | "to"
            | "was"
            | "were"
            | "with"
            | "au"
            | "aux"
            | "ce"
            | "ces"
            | "dans"
            | "de"
            | "des"
            | "du"
            | "et"
            | "est"
            | "la"
            | "le"
            | "les"
            | "par"
            | "pour"
            | "que"
            | "qui"
            | "sur"
            | "une"
    )
}

/// Whether a provider-native citation quotes a passage of the document it names.
///
/// The Anthropic Citations API returns the cited text verbatim, so a quote the
/// document does not contain is a citation pointing at something that was never
/// sent — the native equivalent of an out-of-range marker (US-019 AC-5).
#[must_use]
pub fn quote_belongs_to(document: &str, quoted: &str) -> bool {
    let quoted = quoted.trim();
    !quoted.is_empty() && document.contains(quoted)
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
            generation_id: Uuid::from_u128(7),
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
    fn a_chunk_with_no_generation_cannot_be_cited() {
        let mut chunk = make_chunk("00000000-0000-0000-0000-000000000001", "text");
        chunk.generation_id = Uuid::nil();
        let extracted = extract_citations_verified("Supported [1].", &[chunk]);
        assert!(extracted.citations.is_empty());
        assert_eq!(extracted.rejected, 1);
    }

    #[test]
    fn refused_markers_are_counted_rather_than_ignored() {
        let chunks = vec![make_chunk(
            "00000000-0000-0000-0000-000000000001",
            "only chunk",
        )];
        // A valid marker, an out-of-range one, a `[0]`, and one inside a fence.
        let response = "True [1], also [9] and [0].\n```\narr[3]\n```";
        let extracted = extract_citations_verified(response, &chunks);
        assert_eq!(extracted.citations.len(), 1);
        assert_eq!(extracted.rejected, 3);
    }

    #[test]
    fn a_repeated_marker_is_not_a_rejection() {
        let chunks = vec![make_chunk(
            "00000000-0000-0000-0000-000000000001",
            "only chunk",
        )];
        let extracted = extract_citations_verified("True [1] and again [1].", &chunks);
        assert_eq!(extracted.citations.len(), 1);
        assert_eq!(extracted.rejected, 0);
    }

    /// US-019 AC-3 asks resolution to verify span ownership. A row whose span
    /// ends before it starts, or whose last page precedes its first, was not
    /// written by the chunker, and nothing can be opened at it.
    #[test]
    fn a_chunk_whose_span_cannot_exist_is_not_citable() {
        for broken in [
            serde_json::json!({ "position": 0, "span_start": 900, "span_end": 100 }),
            serde_json::json!({ "position": 0, "page_number": 7, "page_end": 3 }),
            serde_json::json!({ "position": 0, "page_end": 3 }),
        ] {
            let mut chunk = make_chunk("00000000-0000-0000-0000-000000000001", "text");
            chunk.metadata = Some(broken.clone());
            let extracted = extract_citations_verified("Supported [1].", &[chunk]);
            assert!(extracted.citations.is_empty(), "{broken}");
            assert_eq!(extracted.rejected, 1, "{broken}");
        }
    }

    #[test]
    fn a_coherent_span_travels_into_the_citation_as_its_page() {
        let mut chunk = make_chunk("00000000-0000-0000-0000-000000000001", "text");
        chunk.metadata = Some(serde_json::json!({
            "position": 3,
            "page_number": 4,
            "page_end": 5,
            "span_start": 120,
            "span_end": 480,
            "section_header": "Retention",
        }));
        let extracted = extract_citations_verified("Supported [1].", &[chunk]);
        assert_eq!(extracted.rejected, 0);
        assert_eq!(extracted.citations[0].page_number, Some(4));
        assert_eq!(
            extracted.citations[0].section_header.as_deref(),
            Some("Retention")
        );
    }

    /// Legacy rows carry neither span nor page. They stay citable: US-019 adds
    /// provenance, it does not retire the notebooks indexed before it.
    #[test]
    fn a_chunk_with_no_span_at_all_is_still_citable() {
        let chunk = make_chunk("00000000-0000-0000-0000-000000000001", "text");
        let extracted = extract_citations_verified("Supported [1].", &[chunk]);
        assert_eq!(extracted.citations.len(), 1);
        assert_eq!(extracted.rejected, 0);
    }

    #[test]
    fn a_native_quote_must_be_a_passage_of_its_document() {
        assert!(quote_belongs_to("the retry budget is four", "retry budget"));
        assert!(!quote_belongs_to(
            "the retry budget is four",
            "retry budget is five"
        ));
        assert!(!quote_belongs_to("anything", "   "));
    }

    #[test]
    fn production_resolution_requires_current_generation_and_claim_support() {
        let source_id = Uuid::from_u128(11);
        let generation_id = Uuid::from_u128(12);
        let mut chunk = make_chunk(
            &source_id.to_string(),
            "The retry budget is four attempts before the job fails.",
        );
        chunk.generation_id = generation_id;
        let active = HashMap::from([(source_id, generation_id)]);

        let accepted = extract_citations_verified_against_active(
            "The retry budget is four attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert_eq!(accepted.citations.len(), 1);
        assert_eq!(accepted.rejected, 0);

        let unsupported = extract_citations_verified_against_active(
            "The retry budget is five attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert!(unsupported.citations.is_empty());
        assert_eq!(unsupported.rejected, 1);

        let unsupported_digit = extract_citations_verified_against_active(
            "The retry budget is 5 attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert!(unsupported_digit.citations.is_empty());
        assert_eq!(unsupported_digit.rejected, 1);

        let accepted_normalized_number = extract_citations_verified_against_active(
            "The retry budget is 4 attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert_eq!(accepted_normalized_number.citations.len(), 1);
        assert_eq!(accepted_normalized_number.rejected, 0);

        chunk.content = "The retry budget is 1,002 attempts before failure.".to_owned();
        let unsupported_grouped_number = extract_citations_verified_against_active(
            "The retry budget is 1,001 attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert!(unsupported_grouped_number.citations.is_empty());
        assert_eq!(unsupported_grouped_number.rejected, 1);

        chunk.content = "The retry budget is 1,001 attempts before failure.".to_owned();
        let accepted_group_normalization = extract_citations_verified_against_active(
            "The retry budget is 1001 attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert_eq!(accepted_group_normalization.citations.len(), 1);
        assert_eq!(accepted_group_normalization.rejected, 0);

        chunk.content = "The retry budget is thirteen attempts before failure.".to_owned();
        let unsupported_dozen = extract_citations_verified_against_active(
            "The retry budget is a dozen attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert!(unsupported_dozen.citations.is_empty());
        assert_eq!(unsupported_dozen.rejected, 1);

        chunk.content = "The retry budget is eleven attempts before failure.".to_owned();
        let unsupported_large_number = extract_citations_verified_against_active(
            "The retry budget is twelve attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert!(unsupported_large_number.citations.is_empty());
        assert_eq!(unsupported_large_number.rejected, 1);

        chunk.content = "The service stores retry budget policy in a queue.".to_owned();
        let unsupported_long_claim = extract_citations_verified_against_active(
            "The service stores retry budget while database replication permanently changes unrelated ownership semantics [1].",
            &[chunk.clone()],
            &active,
        );
        assert!(unsupported_long_claim.citations.is_empty());
        assert_eq!(unsupported_long_claim.rejected, 1);

        chunk.content = "The retry budget is four attempts before the job fails.".to_owned();
        let repeated_unsupported = extract_citations_verified_against_active(
            "The retry budget is four attempts [1]. It is five attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert_eq!(repeated_unsupported.citations.len(), 1);
        assert_eq!(repeated_unsupported.rejected, 1);

        let wrong_polarity = extract_citations_verified_against_active(
            "The job does not fail after four attempts [1].",
            &[chunk.clone()],
            &active,
        );
        assert!(wrong_polarity.citations.is_empty());
        assert_eq!(wrong_polarity.rejected, 1);

        let stale = extract_citations_verified_against_active(
            "The retry budget is four attempts [1].",
            &[chunk],
            &HashMap::from([(source_id, Uuid::from_u128(13))]),
        );
        assert!(stale.citations.is_empty());
        assert_eq!(stale.rejected, 1);
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
