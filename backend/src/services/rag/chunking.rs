//! RAG chunking: text becomes parents, children, pages and spans.
//!
//! Two passes, using text-splitter with tiktoken for token counting. Pass one
//! splits the source into parent passages, pass two splits each parent into the
//! children that are embedded and retrieved. The parent is what the model reads,
//! the child is what the citation points at.
//!
//! ## Parent-child chunk sizes (small-to-big retrieval)
//!
//! | Type       | Parent (tokens) | Child (tokens) | Child overlap |
//! |------------|-----------------|----------------|---------------|
//! | PDF        | 1024            | 256            | 25            |
//! | Web        | 1024            | 256            | 25            |
//! | Markdown   | 2048 (section)  | 256            | 25            |
//! | Text       | 1024            | 256            | 25            |
//! | DOCX/EPUB  | 2048 (section)  | 256            | 25            |
//!
//! ## Pages come from the extractor, not from a character count (US-019)
//!
//! A paginated source arrives as [`SourceText::paginated`], one string per
//! authoritative page, and cleaning happens *per page* so that joining the
//! cleaned pages yields both the text to split and the exact byte offset where
//! each page starts. A chunk's page is then the page containing its first byte.
//!
//! This replaces a heuristic that divided a character offset by 3,000 and called
//! the quotient a page number. On a source with short pages it drifted by a page
//! every few pages, and a citation that opens the wrong page is worse than one
//! that opens none: the reader checks the wrong paragraph and believes it.
//!
//! ## Spans are exact, and they are recorded before overlap matters
//!
//! Both passes ask the splitter for `chunk_indices`, so every parent and every
//! child carries the byte range it occupies in the cleaned source text. Parents
//! do not overlap, so their ranges partition the source; children overlap by
//! design, and each still carries its own exact range rather than inheriting an
//! approximate one.

use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};

use crate::error::RagError;
use crate::types::{ChunkMetadata, SourceType};

// ============================================================================
// Constants
// ============================================================================

// Parent chunk sizes for parent-child architecture
const PARENT_PDF_CHUNK_SIZE: usize = 1024;
const PARENT_WEB_CHUNK_SIZE: usize = 1024;
const PARENT_MARKDOWN_CHUNK_SIZE: usize = 2048;
const PARENT_TEXT_CHUNK_SIZE: usize = 1024;

/// Child chunk size for parent-child retrieval (tokens).
pub const CHILD_CHUNK_SIZE: usize = 256;
/// Child chunk overlap for parent-child retrieval (tokens).
pub const CHILD_OVERLAP: usize = 25;
/// Parent chunks do not overlap — overlap would embed cross-parent content in children.
const PARENT_OVERLAP: usize = 0;

/// Separator between authoritative pages in the joined source text.
///
/// Two blank lines, as before, so the splitter sees a paragraph boundary where a
/// page ends.
const PAGE_SEPARATOR: &str = "\n\n";

// Compile-time invariant: every parent size must exceed CHILD_CHUNK_SIZE
const _: () = assert!(PARENT_PDF_CHUNK_SIZE > CHILD_CHUNK_SIZE);
const _: () = assert!(PARENT_WEB_CHUNK_SIZE > CHILD_CHUNK_SIZE);
const _: () = assert!(PARENT_MARKDOWN_CHUNK_SIZE > CHILD_CHUNK_SIZE);
const _: () = assert!(PARENT_TEXT_CHUNK_SIZE > CHILD_CHUNK_SIZE);

// ============================================================================
// Source text
// ============================================================================

/// The text to index, with its authoritative page boundaries when it has any.
///
/// A non-paginated source is one page. That is not a special case in the
/// chunker: a single page produces a single page range covering the whole text,
/// and nothing downstream branches on the distinction.
#[derive(Debug, Clone)]
pub struct SourceText {
    pages: Vec<String>,
    paginated: bool,
}

impl SourceText {
    /// Text with no page structure: web, text, Markdown, DOCX, EPUB, YouTube.
    #[must_use]
    pub fn single(text: impl Into<String>) -> Self {
        Self {
            pages: vec![text.into()],
            paginated: false,
        }
    }

    /// Text whose pages the extractor resolved, in order, page 1 first.
    ///
    /// Empty pages are kept: dropping one would shift every page number after
    /// it, which is exactly the failure this type exists to prevent.
    #[must_use]
    pub fn paginated(pages: Vec<String>) -> Self {
        Self {
            pages,
            paginated: true,
        }
    }

    /// Number of pages the extractor reported.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// The full text, pages joined, exactly as it is indexed.
    ///
    /// The only consumer is content-limit validation, which measures what the
    /// user submitted rather than what survives cleaning.
    #[must_use]
    pub fn joined(&self) -> String {
        self.pages.join(PAGE_SEPARATOR)
    }
}

// ============================================================================
// Layout: cleaned text plus page offsets
// ============================================================================

/// The cleaned text the splitter sees, and where each page begins in it.
struct SourceLayout {
    text: String,
    /// Byte offset where page `i + 1` starts. Always starts with 0.
    page_starts: Vec<usize>,
    paginated: bool,
}

impl SourceLayout {
    /// Clean each page, then join. Cleaning per page is what makes the offsets
    /// exact: cleaning the joined text instead would move bytes the page map
    /// had already been computed against.
    fn build(source: &SourceText, source_type: SourceType) -> Self {
        let markdown = uses_markdown_splitter(source_type);
        let mut text = String::new();
        let mut page_starts = Vec::with_capacity(source.pages.len());

        for page in &source.pages {
            let cleaned = if markdown {
                clean_content_markdown(page)
            } else {
                clean_content(page)
            };
            if !text.is_empty() {
                text.push_str(PAGE_SEPARATOR);
            }
            page_starts.push(text.len());
            text.push_str(&cleaned);
        }

        Self {
            text,
            page_starts,
            paginated: source.paginated,
        }
    }

    /// The 1-based page containing `offset`, for a paginated source.
    fn page_for(&self, offset: usize) -> Option<u32> {
        if !self.paginated {
            return None;
        }
        // The last page whose start is at or before the offset.
        let index = self.page_starts.partition_point(|start| *start <= offset);
        u32::try_from(index.max(1)).ok()
    }

    /// The pages a byte range covers, first and last.
    ///
    /// Both, because a range that crosses a page break covers both pages and
    /// naming only the first is a claim the text does not support. `end` is
    /// exclusive, so the last byte is what decides the final page.
    fn pages_for(&self, start: usize, end: usize) -> (Option<u32>, Option<u32>) {
        (
            self.page_for(start),
            self.page_for(end.saturating_sub(1).max(start)),
        )
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Parent chunk size, in [`crate::services::rag::provenance::CHUNK_SIZE_UNIT`],
/// for a source type.
///
/// The single definition of the parent geometry: parent-child chunking reads it
/// to split, and a generation records it as provenance (US-011). Returning
/// `usize` rather than `Option<usize>` is what makes the two agree by
/// construction — there is no arm that could omit a size and leave a
/// fingerprint describing a geometry the splitter did not use.
#[must_use]
pub const fn parent_chunk_size(source_type: SourceType) -> usize {
    match source_type {
        SourceType::Pdf => PARENT_PDF_CHUNK_SIZE,
        SourceType::Web => PARENT_WEB_CHUNK_SIZE,
        SourceType::Text => PARENT_TEXT_CHUNK_SIZE,
        // YouTube transcripts are formatted as timestamped Markdown, so they
        // take the Markdown geometry.
        SourceType::Markdown | SourceType::Docx | SourceType::Epub | SourceType::Youtube => {
            PARENT_MARKDOWN_CHUNK_SIZE
        }
    }
}

/// One retrieval unit: the text that is embedded, and where it came from.
#[derive(Debug, Clone)]
pub struct ChildChunk {
    pub text: String,
    /// Page, section header and exact byte span within the source text.
    pub metadata: ChunkMetadata,
}

/// One context unit: the passage the model reads, and its children.
#[derive(Debug, Clone)]
pub struct ParentChunk {
    pub text: String,
    pub children: Vec<ChildChunk>,
    pub metadata: ChunkMetadata,
}

/// Two-pass parent-child chunking for small-to-big retrieval.
///
/// Pass 1 splits the cleaned source into parent chunks at the per-type parent
/// size, with no overlap. Pass 2 splits each parent into children at
/// [`CHILD_CHUNK_SIZE`] tokens with [`CHILD_OVERLAP`] token overlap.
///
/// Every returned chunk carries its exact byte span in the cleaned source text,
/// and its page when the source is paginated.
///
/// # Errors
/// Returns [`RagError::InvalidChunkConfig`] when a splitter configuration is
/// rejected, which can only happen if the constants above are edited into an
/// inconsistent state.
pub fn chunk_source(
    source: &SourceText,
    source_type: SourceType,
) -> Result<Vec<ParentChunk>, RagError> {
    let layout = SourceLayout::build(source, source_type);
    if layout.text.is_empty() {
        return Ok(Vec::new());
    }

    let markdown = uses_markdown_splitter(source_type);
    let parents = split_indices(
        &layout.text,
        parent_chunk_size(source_type),
        PARENT_OVERLAP,
        markdown,
    )?;

    let mut result = Vec::with_capacity(parents.len());
    let mut headers = HeaderScanner::default();
    let mut skipped_parents = 0u32;

    for (parent_offset, parent_text) in parents {
        let section_header = headers.header_at(&layout.text, parent_offset, parent_text);
        let parent_end = parent_offset + parent_text.len();
        let (first_page, last_page) = layout.pages_for(parent_offset, parent_end);
        let parent_meta = ChunkMetadata {
            section_header: section_header.clone(),
            page_number: first_page,
            page_end: last_page,
            position: u32::try_from(result.len()).unwrap_or(u32::MAX),
            span_start: to_u32(parent_offset),
            span_end: to_u32(parent_end),
            ..ChunkMetadata::default()
        };

        // The parent slice is already cleaned, so the child pass splits it as
        // it stands: re-cleaning would shift the offsets the parent's own span
        // was computed against.
        let children = split_indices(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP, markdown)?;
        if children.is_empty() {
            tracing::warn!(
                parent_len = parent_text.len(),
                "Parent produced no children, skipping"
            );
            skipped_parents += 1;
            continue;
        }

        let children = children
            .into_iter()
            .map(|(child_offset, child_text)| {
                let absolute = parent_offset + child_offset;
                let (first, last) = layout.pages_for(absolute, absolute + child_text.len());
                ChildChunk {
                    text: child_text.to_owned(),
                    metadata: ChunkMetadata {
                        section_header: section_header.clone(),
                        page_number: first,
                        page_end: last,
                        // Overwritten with the flat index when the pipeline
                        // flattens the hierarchy; the parent-relative order is
                        // what matters here.
                        position: 0,
                        span_start: to_u32(absolute),
                        span_end: to_u32(absolute + child_text.len()),
                        ..ChunkMetadata::default()
                    },
                }
            })
            .collect();

        result.push(ParentChunk {
            text: parent_text.to_owned(),
            children,
            metadata: parent_meta,
        });
    }

    tracing::debug!(
        source_type = ?source_type,
        parent_count = result.len(),
        skipped_parents,
        pages = source.page_count(),
        total_children = result.iter().map(|p| p.children.len()).sum::<usize>(),
        "Content chunked with parent-child hierarchy"
    );

    Ok(result)
}

// ============================================================================
// Internal helpers
// ============================================================================

const fn uses_markdown_splitter(source_type: SourceType) -> bool {
    matches!(
        source_type,
        SourceType::Markdown | SourceType::Docx | SourceType::Epub | SourceType::Youtube
    )
}

fn to_u32(value: usize) -> Option<u32> {
    u32::try_from(value).ok()
}

/// Split already-cleaned text, returning each chunk with its byte offset.
fn split_indices(
    text: &str,
    size: usize,
    overlap: usize,
    markdown: bool,
) -> Result<Vec<(usize, &str)>, RagError> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let config = ChunkConfig::new(size).with_overlap(overlap).map_err(|e| {
        tracing::error!(chunk_size = size, overlap, error = %e, "Invalid chunk config");
        RagError::InvalidChunkConfig {
            chunk_size: size,
            overlap,
        }
    })?;

    Ok(if markdown {
        MarkdownSplitter::new(config).chunk_indices(text).collect()
    } else {
        TextSplitter::new(config).chunk_indices(text).collect()
    })
}

/// The closest Markdown heading at or before a chunk, scanned once.
///
/// Parents arrive in document order, so the scan only ever moves forward.
#[derive(Default)]
struct HeaderScanner {
    cursor: usize,
    last: Option<String>,
}

impl HeaderScanner {
    fn header_at(&mut self, text: &str, offset: usize, chunk: &str) -> Option<String> {
        if offset > self.cursor {
            for line in text[self.cursor..offset].lines() {
                if let Some(header) = heading_text(line) {
                    self.last = Some(header);
                }
            }
            self.cursor = offset;
        }
        // A chunk that opens with its own heading owns it.
        if let Some(header) = chunk.lines().next().and_then(heading_text) {
            self.last = Some(header);
        }
        self.last.clone()
    }
}

/// The text of a Markdown ATX heading line, levels 1 to 4.
fn heading_text(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let hashes = trimmed.len() - trimmed.trim_start_matches('#').len();
    if (1..=4).contains(&hashes) && trimmed.as_bytes().get(hashes) == Some(&b' ') {
        Some(trimmed[hashes..].trim().to_owned())
    } else {
        None
    }
}

/// Clean Markdown content before chunking, preserving indentation.
///
/// - Removes null bytes and page-break controls
/// - Normalizes line endings
/// - Collapses 3+ consecutive blank lines to 2
/// - Strips trailing whitespace per line
/// - PRESERVES leading whitespace (critical for code blocks)
fn clean_content_markdown(content: &str) -> String {
    let normalized = normalize(content);

    let mut result = String::with_capacity(normalized.len());
    let mut consecutive_empty = 0u32;

    for line in normalized.lines() {
        let trimmed_end = line.trim_end();
        if trimmed_end.is_empty() {
            consecutive_empty += 1;
            if consecutive_empty <= 2 && !result.is_empty() {
                result.push('\n');
            }
        } else {
            if !result.is_empty() && consecutive_empty == 0 {
                result.push('\n');
            }
            consecutive_empty = 0;
            result.push_str(trimmed_end);
        }
    }

    result.trim_end().to_owned()
}

/// Clean content before chunking.
///
/// - Removes null bytes and page-break controls
/// - Normalizes line endings
/// - Collapses excessive whitespace while preserving paragraph structure
fn clean_content(content: &str) -> String {
    let normalized = normalize(content);

    // Collapse consecutive empty lines into one
    let mut result = String::with_capacity(normalized.len());
    let mut prev_empty = false;

    for line in normalized.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !prev_empty {
                result.push('\n');
                prev_empty = true;
            }
        } else {
            if !result.is_empty() && !prev_empty {
                result.push('\n');
            }
            result.push_str(trimmed);
            prev_empty = false;
        }
    }

    result.trim().to_owned()
}

/// Strip control characters that would corrupt the text or the page map, and
/// normalize line endings.
///
/// The form feed is removed rather than kept as whitespace: it is the page
/// separator the OCR cache uses, and a stray one inside a page would split that
/// page in two on the way back out.
fn normalize(content: &str) -> String {
    content
        .replace(['\0', '\u{000C}'], "")
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parents(text: &str, source_type: SourceType) -> Vec<ParentChunk> {
        chunk_source(&SourceText::single(text), source_type).expect("chunking succeeds")
    }

    #[test]
    fn parent_sizes_per_source_type() {
        assert_eq!(parent_chunk_size(SourceType::Pdf), 1024);
        assert_eq!(parent_chunk_size(SourceType::Web), 1024);
        assert_eq!(parent_chunk_size(SourceType::Text), 1024);
        assert_eq!(parent_chunk_size(SourceType::Markdown), 2048);
        assert_eq!(parent_chunk_size(SourceType::Docx), 2048);
        assert_eq!(parent_chunk_size(SourceType::Epub), 2048);
        assert_eq!(parent_chunk_size(SourceType::Youtube), 2048);
    }

    #[test]
    fn clean_content_removes_control_characters() {
        assert_eq!(clean_content("a\0b\u{000C}c"), "abc");
        assert_eq!(
            clean_content("line1\r\nline2\rline3"),
            "line1\nline2\nline3"
        );
        assert_eq!(clean_content("a\n\n\n\n\nb"), "a\nb");
    }

    #[test]
    fn empty_content_produces_no_chunks() {
        assert!(parents("", SourceType::Text).is_empty());
        assert!(parents("   \n\n  ", SourceType::Text).is_empty());
    }

    #[test]
    fn markdown_keeps_code_block_indentation() {
        let text = "# Example\n\n```rust\nfn main() {\n    let x = 42;\n}\n```";
        let result = parents(text, SourceType::Markdown);
        let joined: String = result.iter().map(|p| p.text.clone()).collect();
        assert!(joined.contains("    let x = 42;"), "{joined}");
    }

    // ====================================================================
    // US-019: spans are exact
    // ====================================================================

    /// Every parent's recorded span is the slice it was cut from. This is the
    /// property every citation ultimately rests on.
    #[test]
    fn every_parent_span_indexes_back_to_its_own_text() {
        let text: String = (0..400)
            .map(|i| format!("Sentence {i} about retention policy and retries. "))
            .collect();
        let source = SourceText::single(&text);
        let layout = SourceLayout::build(&source, SourceType::Text);
        let result = chunk_source(&source, SourceType::Text).expect("chunking");

        assert!(result.len() > 1, "the fixture must produce several parents");
        for parent in &result {
            let start = parent.metadata.span_start.expect("a parent has a span") as usize;
            let end = parent.metadata.span_end.expect("a parent has a span") as usize;
            assert_eq!(
                &layout.text[start..end],
                parent.text,
                "the span must slice back to the parent"
            );
        }
    }

    #[test]
    fn every_child_span_indexes_back_to_its_own_text() {
        let text: String = (0..400)
            .map(|i| format!("Sentence {i} about retention policy and retries. "))
            .collect();
        let source = SourceText::single(&text);
        let layout = SourceLayout::build(&source, SourceType::Text);
        let result = chunk_source(&source, SourceType::Text).expect("chunking");

        for parent in &result {
            for child in &parent.children {
                let start = child.metadata.span_start.expect("a child has a span") as usize;
                let end = child.metadata.span_end.expect("a child has a span") as usize;
                assert_eq!(&layout.text[start..end], child.text);
                // And it lies inside its parent's range.
                assert!(start >= parent.metadata.span_start.expect("parent span") as usize);
                assert!(end <= parent.metadata.span_end.expect("parent span") as usize);
            }
        }
    }

    #[test]
    fn parent_spans_do_not_overlap() {
        let text: String = (0..400)
            .map(|i| format!("Paragraph {i} discusses one topic in some detail. "))
            .collect();
        let result = parents(&text, SourceType::Text);
        let mut previous_end = 0u32;
        for parent in &result {
            let start = parent.metadata.span_start.expect("span");
            assert!(
                start >= previous_end,
                "parents must partition the source, got {start} after {previous_end}"
            );
            previous_end = parent.metadata.span_end.expect("span");
        }
    }

    // ====================================================================
    // US-019: pages come from the extractor
    // ====================================================================

    /// The defect this replaces: a page number computed as
    /// `character_offset / 3000`. With short pages it is wrong from page two
    /// onwards.
    #[test]
    fn a_chunk_reports_the_page_it_was_extracted_from() {
        let pages = vec![
            "ALPHA content on the first page. ".repeat(60),
            "BRAVO content on the second page. ".repeat(60),
            "CHARLIE content on the third page. ".repeat(60),
        ];
        let source = SourceText::paginated(pages);
        let result = chunk_source(&source, SourceType::Pdf).expect("chunking");

        for (needle, expected) in [("ALPHA", 1), ("BRAVO", 2), ("CHARLIE", 3)] {
            // Every chunk that mentions the marker, so a chunk straddling a
            // page break cannot make the assertion accidentally true.
            let ranges: Vec<(u32, u32)> = result
                .iter()
                .flat_map(|p| p.children.iter())
                .filter(|c| c.text.contains(needle))
                .filter_map(|c| Some((c.metadata.page_number?, c.metadata.page_end?)))
                .collect();
            assert!(!ranges.is_empty(), "no chunk carried {needle}");
            assert!(
                ranges
                    .iter()
                    .all(|(first, last)| (*first..=*last).contains(&expected)),
                "{needle} is on page {expected}, chunks reported {ranges:?}"
            );
        }
    }

    /// The page map itself, at the boundaries. A chunk that starts one byte
    /// before a page break belongs to the page it started on.
    #[test]
    fn the_page_map_resolves_every_offset_to_its_own_page() {
        let pages: Vec<String> = (1..=6).map(|i| format!("Page {i} body.")).collect();
        let source = SourceText::paginated(pages);
        let layout = SourceLayout::build(&source, SourceType::Pdf);

        assert_eq!(layout.page_starts.len(), 6);
        for (index, start) in layout.page_starts.iter().enumerate() {
            let page = u32::try_from(index + 1).expect("page");
            assert_eq!(layout.page_for(*start), Some(page), "at the page start");
            assert_eq!(
                layout.page_for(start + 1),
                Some(page),
                "one byte into the page"
            );
            if index > 0 {
                assert_eq!(
                    layout.page_for(start - 1),
                    Some(page - 1),
                    "one byte before the page start still belongs to the previous page"
                );
            }
        }
        assert_eq!(layout.page_for(layout.text.len()), Some(6));
    }

    /// A chunk that crosses a page break reports both pages rather than
    /// claiming the one it happens to start on.
    #[test]
    fn a_range_that_crosses_a_break_reports_both_pages() {
        let pages: Vec<String> = (1..=3).map(|i| format!("Page {i} body.")).collect();
        let source = SourceText::paginated(pages);
        let layout = SourceLayout::build(&source, SourceType::Pdf);

        assert_eq!(layout.pages_for(0, layout.text.len()), (Some(1), Some(3)));
        let second = layout.page_starts[1];
        assert_eq!(layout.pages_for(second, second + 4), (Some(2), Some(2)));
    }

    /// An empty page still consumes its number in the map. Skipping it would
    /// shift every page after it by one, which is how the old heuristic
    /// drifted.
    #[test]
    fn an_empty_page_keeps_its_number_in_the_map() {
        let pages = vec![
            "First page text.".to_owned(),
            String::new(),
            "Third page text with MARKER.".to_owned(),
        ];
        let source = SourceText::paginated(pages);
        let layout = SourceLayout::build(&source, SourceType::Pdf);
        let marker = layout.text.find("MARKER").expect("marker present");
        assert_eq!(layout.page_for(marker), Some(3));
    }

    #[test]
    fn a_non_paginated_source_reports_no_page() {
        let result = parents(&"Some text without pages. ".repeat(50), SourceType::Text);
        assert!(
            result.iter().all(|p| p.metadata.page_number.is_none()
                && p.children.iter().all(|c| c.metadata.page_number.is_none())),
            "a text source has no authoritative pages to report"
        );
    }

    // ====================================================================
    // Section headers
    // ====================================================================

    #[test]
    fn a_chunk_inherits_the_heading_above_it() {
        let text = format!(
            "# Introduction\n\n{}\n\n## Retention\n\n{}",
            "Intro body sentence. ".repeat(200),
            "Retention body sentence MARKER. ".repeat(200)
        );
        let result = parents(&text, SourceType::Markdown);
        let header = result
            .iter()
            .find(|p| p.text.contains("MARKER"))
            .and_then(|p| p.metadata.section_header.clone())
            .expect("a header");
        assert_eq!(header, "Retention");
    }

    #[test]
    fn children_inherit_their_parents_heading() {
        let text = format!("## Policy\n\n{}", "Body sentence. ".repeat(300));
        let result = parents(&text, SourceType::Markdown);
        for parent in &result {
            for child in &parent.children {
                assert_eq!(
                    child.metadata.section_header,
                    parent.metadata.section_header
                );
            }
        }
    }

    #[test]
    fn heading_text_accepts_only_atx_levels_one_to_four() {
        assert_eq!(heading_text("# Title"), Some("Title".to_owned()));
        assert_eq!(heading_text("#### Deep"), Some("Deep".to_owned()));
        assert_eq!(heading_text("##### Too deep"), None);
        assert_eq!(heading_text("#NoSpace"), None);
        assert_eq!(heading_text("plain line"), None);
    }

    // ====================================================================
    // Hierarchy shape
    // ====================================================================

    #[test]
    fn a_large_source_produces_several_parents_each_with_several_children() {
        let text: String = (0..500)
            .map(|i| format!("Sentence number {i} discusses topic {}. ", i % 7))
            .collect();
        let result = parents(&text, SourceType::Text);

        assert!(result.len() > 1);
        assert!(result.iter().any(|p| p.children.len() > 1));
        for parent in &result {
            for child in &parent.children {
                assert!(
                    parent.text.contains(&child.text),
                    "a child must be a substring of its parent"
                );
            }
        }
    }

    #[test]
    fn a_parent_smaller_than_a_child_is_not_subsplit() {
        let result = parents("A short document.", SourceType::Text);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].children.len(), 1);
        assert_eq!(result[0].children[0].text, "A short document.");
    }

    #[test]
    fn parents_are_positioned_in_document_order() {
        let text: String = (0..400)
            .map(|i| format!("Line {i} of the body. "))
            .collect();
        let result = parents(&text, SourceType::Text);
        for (i, parent) in result.iter().enumerate() {
            assert_eq!(
                parent.metadata.position,
                u32::try_from(i).expect("position")
            );
        }
    }
}
