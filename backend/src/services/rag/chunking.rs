//! RAG chunking service for document processing.
//!
//! Handles document chunking for the RAG pipeline using text-splitter
//! with tiktoken for accurate token counting.
//!
//! ## Parent-Child Chunk Sizes (small-to-big retrieval)
//!
//! | Type       | Parent (tokens) | Child (tokens) | Child Overlap |
//! |------------|-----------------|----------------|---------------|
//! | PDF        | 1024            | 256            | 25            |
//! | Web        | 1024            | 256            | 25            |
//! | Markdown   | 2048 (section)  | 256            | 25            |
//! | Text       | 1024            | 256            | 25            |
//! | DOCX/EPUB  | 2048 (section)  | 256            | 25            |
//!
//! ## Legacy Single-Level Sizes (fallback)
//!
//! | Type       | Tokens | Overlap |
//! |------------|--------|---------|
//! | PDF        | 512    | 50      |
//! | Web        | 1024   | 100     |
//! | Markdown   | 1500   | 150     |
//! | Text       | 2048   | 200     |
//! | DOCX/EPUB  | 1500   | 150     |
//!
//! ## Streaming Support
//!
//! For large documents, use `ChunkIterator` or `chunk_content_batched`
//! to avoid loading all chunks into memory at once.

use text_splitter::{ChunkConfig, MarkdownSplitter, TextSplitter};

use crate::error::RagError;
use crate::types::{ChunkMetadata, SourceType};

// ============================================================================
// Constants
// ============================================================================

/// Default batch size for streaming chunk processing.
pub const DEFAULT_CHUNK_BATCH_SIZE: usize = 50;

// Chunk parameters per source type (from CLAUDE.md)
const PDF_CHUNK_SIZE: usize = 512;
const PDF_OVERLAP: usize = 50;
const WEB_CHUNK_SIZE: usize = 1024;
const WEB_OVERLAP: usize = 100;
const MARKDOWN_CHUNK_SIZE: usize = 1500;
const MARKDOWN_OVERLAP: usize = 150;
const TEXT_CHUNK_SIZE: usize = 2048;
const TEXT_OVERLAP: usize = 200;

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

// Compile-time invariant: every parent size must exceed CHILD_CHUNK_SIZE
const _: () = assert!(PARENT_PDF_CHUNK_SIZE > CHILD_CHUNK_SIZE);
const _: () = assert!(PARENT_WEB_CHUNK_SIZE > CHILD_CHUNK_SIZE);
const _: () = assert!(PARENT_MARKDOWN_CHUNK_SIZE > CHILD_CHUNK_SIZE);
const _: () = assert!(PARENT_TEXT_CHUNK_SIZE > CHILD_CHUNK_SIZE);

// ============================================================================
// Types
// ============================================================================

/// Chunk configuration parameters for one splitting pass.
///
/// Parent-child chunking runs two passes, so it builds two of these; the parent
/// geometry itself comes from [`parent_chunk_size`], which is its only
/// definition.
#[derive(Debug, Clone, Copy)]
struct ChunkParams {
    size: usize,
    overlap: usize,
}

impl ChunkParams {
    /// Legacy single-level chunk parameters (fallback when parent-child is disabled).
    const fn for_source_type(source_type: SourceType) -> Self {
        match source_type {
            SourceType::Pdf => Self {
                size: PDF_CHUNK_SIZE,
                overlap: PDF_OVERLAP,
            },
            SourceType::Web => Self {
                size: WEB_CHUNK_SIZE,
                overlap: WEB_OVERLAP,
            },
            // DOCX/EPUB: use markdown-like params since we extract structured content
            SourceType::Markdown | SourceType::Docx | SourceType::Epub => Self {
                size: MARKDOWN_CHUNK_SIZE,
                overlap: MARKDOWN_OVERLAP,
            },
            SourceType::Text => Self {
                size: TEXT_CHUNK_SIZE,
                overlap: TEXT_OVERLAP,
            },
            // YouTube: markdown-like params (transcript formatted as timestamped Markdown)
            SourceType::Youtube => Self {
                size: MARKDOWN_CHUNK_SIZE,
                overlap: MARKDOWN_OVERLAP,
            },
        }
    }

    /// Child (retrieval) chunk parameters for small-to-big retrieval.
    ///
    /// The same for every source type: only the parent geometry varies, and
    /// that lives in [`parent_chunk_size`].
    const fn child_params() -> Self {
        Self {
            size: CHILD_CHUNK_SIZE,
            overlap: CHILD_OVERLAP,
        }
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

/// Chunk content based on source type.
///
/// Uses tiktoken-rs for accurate token counting (GPT-4 compatible).
/// Markdown sources use [`MarkdownSplitter`] which respects headers,
/// code blocks, and list boundaries.
pub fn chunk_content(content: &str, source_type: SourceType) -> Result<Vec<String>, RagError> {
    let params = ChunkParams::for_source_type(source_type);

    let chunks = match source_type {
        SourceType::Markdown | SourceType::Docx | SourceType::Epub | SourceType::Youtube => {
            do_chunk_markdown(content, params)?
        }
        _ => do_chunk(content, params)?,
    };

    tracing::debug!(
        source_type = ?source_type,
        chunk_count = chunks.len(),
        "Content chunked"
    );

    Ok(chunks)
}

/// Chunk content with explicit parameters (for testing or custom use).
pub fn chunk_content_with_params(
    content: &str,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<String>, RagError> {
    do_chunk(
        content,
        ChunkParams {
            size: chunk_size,
            overlap,
        },
    )
}

/// Chunk content and return batches for memory-efficient processing.
///
/// Useful when processing chunks in batches (e.g., for embedding API calls
/// that have a maximum batch size).
pub fn chunk_content_batched(
    content: &str,
    source_type: SourceType,
    batch_size: usize,
) -> Result<Vec<Vec<String>>, RagError> {
    let chunks = chunk_content(content, source_type)?;

    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let batches: Vec<Vec<String>> = chunks.chunks(batch_size).map(<[String]>::to_vec).collect();

    tracing::debug!(
        source_type = ?source_type,
        total_chunks = chunks.len(),
        batch_count = batches.len(),
        batch_size,
        "Content batched"
    );

    Ok(batches)
}

/// Chunk content and extract structured metadata for each chunk.
///
/// Metadata includes:
/// - `section_header`: Closest parent heading (for Markdown/structured docs)
/// - `page_number`: Estimated page (for PDFs, ~3000 chars/page heuristic)
/// - `position`: Sequential index within the source
pub fn chunk_content_with_metadata(
    content: &str,
    source_type: SourceType,
) -> Result<Vec<(String, ChunkMetadata)>, RagError> {
    let chunks = chunk_content(content, source_type)?;
    Ok(attach_metadata(content, chunks, source_type))
}

/// Two-pass parent-child chunking for small-to-big retrieval.
///
/// Pass 1: Split content into parent chunks using larger sizes per source type.
/// Pass 2: Split each parent into child chunks at [`CHILD_CHUNK_SIZE`] tokens
/// with [`CHILD_OVERLAP`] token overlap.
///
/// Returns `(parent_text, child_texts, parent_metadata)` tuples.
/// If a parent fits within [`CHILD_CHUNK_SIZE`], it produces a single child
/// equal to the cleaned parent text — no special handling is needed.
///
/// **Important:** Children are substrings of the *cleaned* parent text (after
/// `clean_content`/`clean_content_markdown`), not of `parent_text` as returned.
/// To verify containment, compare against `clean_content(parent_text)`.
pub fn chunk_content_with_parents(
    content: &str,
    source_type: SourceType,
) -> Result<Vec<(String, Vec<String>, ChunkMetadata)>, RagError> {
    let parent_params = ChunkParams {
        size: parent_chunk_size(source_type),
        overlap: PARENT_OVERLAP,
    };
    let child_params = ChunkParams::child_params();

    // Pass 1: Split into parent chunks
    let parent_chunks = match source_type {
        SourceType::Markdown | SourceType::Docx | SourceType::Epub | SourceType::Youtube => {
            do_chunk_markdown(content, parent_params)?
        }
        _ => do_chunk(content, parent_params)?,
    };

    if parent_chunks.is_empty() {
        return Ok(Vec::new());
    }

    // Attach metadata to parent chunks
    let parents_with_meta = attach_metadata(content, parent_chunks, source_type);

    // Pass 2: Split each parent into child chunks
    let mut result = Vec::new();
    let mut skipped_parents = 0u32;
    for (parent_text, parent_meta) in parents_with_meta {
        let children = match source_type {
            SourceType::Markdown | SourceType::Docx | SourceType::Epub | SourceType::Youtube => {
                do_chunk_markdown(&parent_text, child_params)?
            }
            _ => do_chunk(&parent_text, child_params)?,
        };

        if children.is_empty() {
            tracing::warn!(
                parent_len = parent_text.len(),
                "Parent produced no children, skipping"
            );
            skipped_parents += 1;
            continue;
        }

        result.push((parent_text, children, parent_meta));
    }

    tracing::debug!(
        source_type = ?source_type,
        parent_count = result.len(),
        skipped_parents,
        total_children = result.iter().map(|(_, c, _)| c.len()).sum::<usize>(),
        "Content chunked with parent-child hierarchy"
    );

    Ok(result)
}

/// Process content in streaming batches with a callback.
///
/// Most memory-efficient option for very large documents.
/// Processes each batch immediately via the provided callback.
///
/// # Returns
/// Total number of chunks processed, or first error encountered.
pub fn process_chunks_streaming<F, E>(
    content: &str,
    source_type: SourceType,
    batch_size: usize,
    mut processor: F,
) -> Result<usize, RagError>
where
    F: FnMut(Vec<String>, usize) -> Result<(), E>,
    E: std::fmt::Display,
{
    let chunks = chunk_content(content, source_type)?;

    if chunks.is_empty() {
        return Ok(0);
    }

    let total = chunks.len();

    for (batch_idx, batch) in chunks.chunks(batch_size).enumerate() {
        processor(batch.to_vec(), batch_idx).map_err(|e| {
            tracing::error!(batch_index = batch_idx, error = %e, "Batch processing failed");
            RagError::ChunkingFailed {
                reason: format!("Batch {batch_idx} failed: {e}"),
            }
        })?;
    }

    tracing::debug!(
        source_type = ?source_type,
        total_processed = total,
        batch_count = total.div_ceil(batch_size),
        "Streaming processing complete"
    );

    Ok(total)
}

// ============================================================================
// Iterator
// ============================================================================

/// Yields chunks from a pre-computed vector. Useful for consuming chunks one
/// at a time without exposing the internal `Vec`.
///
/// # Example
/// ```ignore
/// let iter = ChunkIterator::new(content, SourceType::Pdf)?;
/// for chunk in iter {
///     process_chunk(chunk);
/// }
/// ```
pub struct ChunkIterator {
    chunks: std::vec::IntoIter<String>,
    total: usize,
}

impl ChunkIterator {
    /// Create a new chunk iterator for the given content.
    pub fn new(content: &str, source_type: SourceType) -> Result<Self, RagError> {
        let chunks = chunk_content(content, source_type)?;
        let total = chunks.len();
        Ok(Self {
            chunks: chunks.into_iter(),
            total,
        })
    }

    /// Get the total number of chunks.
    #[must_use]
    pub const fn total_chunks(&self) -> usize {
        self.total
    }
}

impl Iterator for ChunkIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chunks.size_hint()
    }
}

impl ExactSizeIterator for ChunkIterator {}

// ============================================================================
// Internal helpers
// ============================================================================

/// Snap a byte offset to the nearest valid UTF-8 char boundary at or before `offset`.
fn snap_to_char_boundary(s: &str, offset: usize) -> usize {
    let offset = offset.min(s.len());
    // Walk backwards (up to 4 bytes for the longest UTF-8 sequence) to find a char boundary.
    let mut pos = offset;
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// Attach positional metadata to chunks based on source type.
///
/// Shared helper used by both `chunk_content_with_metadata` (single-level)
/// and `chunk_content_with_parents` (parent-level metadata).
fn attach_metadata(
    original_content: &str,
    chunks: Vec<String>,
    source_type: SourceType,
) -> Vec<(String, ChunkMetadata)> {
    match source_type {
        SourceType::Markdown | SourceType::Docx | SourceType::Epub => {
            let mut last_header: Option<String> = None;
            let mut search_pos = 0usize;

            chunks
                .into_iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let safe_search_pos = snap_to_char_boundary(original_content, search_pos);
                    if let Some(pos) = original_content[safe_search_pos..]
                        .find(chunk.split('\n').next().unwrap_or(&chunk))
                    {
                        let abs_pos = safe_search_pos + pos;
                        for line in original_content[safe_search_pos..abs_pos].lines() {
                            let trimmed = line.trim();
                            if trimmed.starts_with("# ")
                                || trimmed.starts_with("## ")
                                || trimmed.starts_with("### ")
                                || trimmed.starts_with("#### ")
                            {
                                last_header =
                                    Some(trimmed.trim_start_matches('#').trim().to_string());
                            }
                        }
                        if let Some(first_line) = chunk.lines().next() {
                            let trimmed = first_line.trim();
                            if trimmed.starts_with("# ")
                                || trimmed.starts_with("## ")
                                || trimmed.starts_with("### ")
                                || trimmed.starts_with("#### ")
                            {
                                last_header =
                                    Some(trimmed.trim_start_matches('#').trim().to_string());
                            }
                        }
                        // Advance past the matched first line (char-boundary safe)
                        let first_line = chunk.split('\n').next().unwrap_or(&chunk);
                        search_pos = abs_pos + first_line.len();
                    }

                    let meta = ChunkMetadata {
                        section_header: last_header.clone(),
                        position: u32::try_from(i).unwrap_or(u32::MAX),
                        ..ChunkMetadata::default()
                    };
                    (chunk, meta)
                })
                .collect()
        }
        SourceType::Pdf => {
            const CHARS_PER_PAGE: usize = 3000;
            let mut char_offset = 0usize;

            chunks
                .into_iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let search_from = snap_to_char_boundary(original_content, char_offset);
                    if let Some(pos) = original_content[search_from..]
                        .find(chunk.split('\n').next().unwrap_or(&chunk))
                    {
                        char_offset = search_from + pos;
                    }
                    let page = u32::try_from(char_offset / CHARS_PER_PAGE)
                        .unwrap_or(u32::MAX)
                        .saturating_add(1);

                    let meta = ChunkMetadata {
                        page_number: Some(page),
                        position: u32::try_from(i).unwrap_or(u32::MAX),
                        ..ChunkMetadata::default()
                    };
                    char_offset =
                        snap_to_char_boundary(original_content, char_offset + chunk.len());
                    (chunk, meta)
                })
                .collect()
        }
        _ => chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let meta = ChunkMetadata {
                    position: u32::try_from(i).unwrap_or(u32::MAX),
                    ..ChunkMetadata::default()
                };
                (chunk, meta)
            })
            .collect(),
    }
}

/// Markdown-aware chunking using [`MarkdownSplitter`].
///
/// Respects CommonMark structure: headers as section delimiters,
/// code blocks kept intact, lists grouped when possible.
fn do_chunk_markdown(content: &str, params: ChunkParams) -> Result<Vec<String>, RagError> {
    let cleaned = clean_content_markdown(content);

    if cleaned.is_empty() {
        return Ok(Vec::new());
    }

    let config = ChunkConfig::new(params.size)
        .with_overlap(params.overlap)
        .map_err(|e| {
            tracing::error!(chunk_size = params.size, overlap = params.overlap, error = %e, "Invalid markdown chunk config");
            RagError::InvalidChunkConfig { chunk_size: params.size, overlap: params.overlap }
        })?;

    let splitter = MarkdownSplitter::new(config);
    Ok(splitter.chunks(&cleaned).map(String::from).collect())
}

/// Core chunking logic.
fn do_chunk(content: &str, params: ChunkParams) -> Result<Vec<String>, RagError> {
    let cleaned = clean_content(content);

    if cleaned.is_empty() {
        return Ok(Vec::new());
    }

    let config = ChunkConfig::new(params.size)
        .with_overlap(params.overlap)
        .map_err(|e| {
            tracing::error!(chunk_size = params.size, overlap = params.overlap, error = %e, "Invalid chunk config");
            RagError::InvalidChunkConfig { chunk_size: params.size, overlap: params.overlap }
        })?;

    let splitter = TextSplitter::new(config);
    Ok(splitter.chunks(&cleaned).map(String::from).collect())
}

/// Clean Markdown content before chunking, preserving indentation.
///
/// - Removes null bytes
/// - Normalizes line endings
/// - Collapses 3+ consecutive blank lines to 2
/// - Strips trailing whitespace per line
/// - PRESERVES leading whitespace (critical for code blocks)
fn clean_content_markdown(content: &str) -> String {
    let normalized = content
        .replace('\0', "")
        .replace("\r\n", "\n")
        .replace('\r', "\n");

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
/// - Removes null bytes
/// - Normalizes line endings
/// - Collapses excessive whitespace while preserving paragraph structure
fn clean_content(content: &str) -> String {
    // Remove null bytes and normalize line endings
    let normalized = content
        .replace('\0', "")
        .replace("\r\n", "\n")
        .replace('\r', "\n");

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_params_for_types() {
        // NOTE: These are legacy single-level sizes, only used by the fallback
        // `chunk_content()` path. The active pipeline uses `chunk_content_with_parents()`
        // which uses `parent_chunk_size()` and `child_params()` instead.
        // For Text, the legacy size (2048) exceeds the parent-child parent size (1024)
        // — this is intentional: the legacy path produced larger, single-level chunks
        // while the new architecture uses smaller parents + even smaller children.
        let pdf = ChunkParams::for_source_type(SourceType::Pdf);
        assert_eq!(pdf.size, 512);
        assert_eq!(pdf.overlap, 50);

        let web = ChunkParams::for_source_type(SourceType::Web);
        assert_eq!(web.size, 1024);
        assert_eq!(web.overlap, 100);

        let md = ChunkParams::for_source_type(SourceType::Markdown);
        assert_eq!(md.size, 1500);
        assert_eq!(md.overlap, 150);

        let text = ChunkParams::for_source_type(SourceType::Text);
        assert_eq!(text.size, 2048);
        assert_eq!(text.overlap, 200);

        let docx = ChunkParams::for_source_type(SourceType::Docx);
        assert_eq!(docx.size, 1500);
        assert_eq!(docx.overlap, 150);

        let epub = ChunkParams::for_source_type(SourceType::Epub);
        assert_eq!(epub.size, 1500);
        assert_eq!(epub.overlap, 150);
    }

    #[test]
    fn parent_child_params_for_types() {
        // The child pass is identical for every source type; only the parent
        // geometry varies, and `parent_chunk_size` is its only definition.
        let child = ChunkParams::child_params();
        assert_eq!(child.size, 256);
        assert_eq!(child.overlap, 25);

        assert_eq!(parent_chunk_size(SourceType::Pdf), 1024);
        assert_eq!(parent_chunk_size(SourceType::Web), 1024);
        assert_eq!(parent_chunk_size(SourceType::Markdown), 2048);
        assert_eq!(parent_chunk_size(SourceType::Text), 1024);
        assert_eq!(parent_chunk_size(SourceType::Docx), 2048);
        assert_eq!(parent_chunk_size(SourceType::Epub), 2048);
        assert_eq!(parent_chunk_size(SourceType::Youtube), 2048);
    }

    #[test]
    fn clean_content_removes_nulls() {
        let input = "hello\0world";
        assert_eq!(clean_content(input), "helloworld");

        let input_with_newline = "hello\0\nworld";
        assert_eq!(clean_content(input_with_newline), "hello\nworld");
    }

    #[test]
    fn clean_content_normalizes_line_endings() {
        let input = "line1\r\nline2\rline3";
        assert_eq!(clean_content(input), "line1\nline2\nline3");
    }

    #[test]
    fn clean_content_collapses_empty_lines() {
        let input = "para1\n\n\n\npara2";
        assert_eq!(clean_content(input), "para1\npara2");
    }

    #[test]
    fn chunk_empty_content() {
        let result = chunk_content("", SourceType::Text).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn chunk_iterator_count() {
        let content = "word ".repeat(100);
        let iter = ChunkIterator::new(&content, SourceType::Text).unwrap();
        assert_eq!(iter.len(), iter.total_chunks());
    }

    #[test]
    fn chunk_batched() {
        let content = "word ".repeat(1000);
        let batches = chunk_content_batched(&content, SourceType::Text, 2).unwrap();

        // All batches except possibly the last should have batch_size elements
        for batch in batches.iter().take(batches.len().saturating_sub(1)) {
            assert_eq!(batch.len(), 2);
        }
    }

    #[test]
    fn markdown_chunks_at_header_boundaries() {
        let content = format!(
            "# Section One\n\n{}\n\n# Section Two\n\n{}",
            "First section content. ".repeat(50),
            "Second section content. ".repeat(50),
        );

        let chunks = chunk_content(&content, SourceType::Markdown).unwrap();
        assert!(
            chunks.len() >= 2,
            "Should produce at least 2 chunks for 2 sections"
        );

        // At least one chunk should start with a header
        let has_header_chunk = chunks.iter().any(|c| c.trim_start().starts_with("# "));
        assert!(
            has_header_chunk,
            "At least one chunk should preserve a Markdown header"
        );
    }

    #[test]
    fn markdown_preserves_code_blocks() {
        let code_block = "```rust\nfn main() {\n    println!(\"Hello, world!\");\n}\n```";
        // Pad with enough text to force chunking
        let content = format!(
            "# Introduction\n\n{}\n\n{code_block}\n\n# Conclusion\n\n{}",
            "Intro text. ".repeat(80),
            "Conclusion text. ".repeat(80),
        );

        let chunks = chunk_content(&content, SourceType::Markdown).unwrap();

        // The code block should appear intact in one of the chunks
        let code_preserved = chunks
            .iter()
            .any(|c| c.contains("fn main()") && c.contains("println!"));
        assert!(
            code_preserved,
            "Code block should be preserved intact in a single chunk"
        );
    }

    #[test]
    fn chunk_with_metadata_markdown_headers() {
        let content = format!(
            "# Introduction\n\n{}\n\n## Methods\n\n{}\n\n### Sub-method\n\n{}",
            "Intro paragraph content here. ".repeat(50),
            "Methods section content here. ".repeat(50),
            "Sub-method details here. ".repeat(50),
        );

        let result = chunk_content_with_metadata(&content, SourceType::Markdown).unwrap();
        assert!(!result.is_empty());

        // At least one chunk should have a section_header
        let has_header = result.iter().any(|(_, meta)| meta.section_header.is_some());
        assert!(
            has_header,
            "At least one chunk should have a section_header"
        );

        // All chunks should have sequential positions
        for (i, (_, meta)) in result.iter().enumerate() {
            assert_eq!(meta.position, i as u32);
        }
    }

    #[test]
    fn chunk_with_metadata_pdf_pages() {
        // Create content that spans multiple "pages" (~3000 chars each)
        let content = "A".repeat(9000); // ~3 pages

        let result = chunk_content_with_metadata(&content, SourceType::Pdf).unwrap();
        assert!(!result.is_empty());

        // At least one chunk should have a page_number
        let has_page = result.iter().any(|(_, meta)| meta.page_number.is_some());
        assert!(has_page, "PDF chunks should have page_number");

        // No chunk should have section_header for PDFs
        let has_header = result.iter().any(|(_, meta)| meta.section_header.is_some());
        assert!(!has_header, "PDF chunks should not have section_header");
    }

    #[test]
    fn chunk_with_metadata_text_has_positions() {
        let content = "word ".repeat(500);
        let result = chunk_content_with_metadata(&content, SourceType::Text).unwrap();

        for (i, (_, meta)) in result.iter().enumerate() {
            assert_eq!(meta.position, i as u32);
            assert!(meta.section_header.is_none());
            assert!(meta.page_number.is_none());
        }
    }

    #[test]
    fn clean_content_markdown_preserves_indentation() {
        let input = "# Code Example\n\n```python\ndef hello():\n    print(\"hello\")\n    if True:\n        return 42\n```\n\n\n\n\nMore text here.";
        let cleaned = clean_content_markdown(input);
        // Leading whitespace in code block must be preserved
        assert!(
            cleaned.contains("    print(\"hello\")"),
            "4-space indentation must be preserved: {cleaned}"
        );
        assert!(
            cleaned.contains("        return 42"),
            "8-space indentation must be preserved: {cleaned}"
        );
        // 5 consecutive blank lines collapsed to 2
        assert!(
            !cleaned.contains("\n\n\n"),
            "3+ blank lines should be collapsed to 2: {cleaned}"
        );
    }

    #[test]
    fn markdown_chunk_preserves_code_indentation() {
        let content =
            "# Example\n\n```rust\nfn main() {\n    let x = 42;\n    println!(\"{x}\");\n}\n```";
        let chunks = chunk_content(content, SourceType::Markdown).unwrap();
        let all = chunks.join("\n");
        assert!(
            all.contains("    let x = 42;"),
            "Markdown chunking must preserve code indentation: {all}"
        );
    }

    #[test]
    fn markdown_uses_markdown_splitter() {
        // Verify Markdown uses different chunking than plain text
        let content = "# Header\n\nParagraph content here.";
        let md_chunks = chunk_content(content, SourceType::Markdown).unwrap();
        let text_chunks = chunk_content(content, SourceType::Text).unwrap();

        // Both should produce at least 1 chunk for this small content
        assert!(!md_chunks.is_empty());
        assert!(!text_chunks.is_empty());
    }

    // ── Parent-child chunking tests ─────────────────────────────────────

    #[test]
    fn parent_child_3000_token_text_produces_hierarchy() {
        // ~3000 tokens of text should produce multiple parents (at 1024 each)
        // and multiple children (at 256 each) per parent
        let content =
            "The quick brown fox jumps over the lazy dog near the river bank. ".repeat(500);

        let result = chunk_content_with_parents(&content, SourceType::Text).unwrap();

        // Should produce multiple parents (3000 tokens / 1024 ≈ 3 parents)
        assert!(
            result.len() >= 2,
            "3000-token text should produce at least 2 parents, got {}",
            result.len()
        );

        // Each parent with enough content should have multiple children
        let multi_child_parents = result
            .iter()
            .filter(|(_, children, _)| children.len() > 1)
            .count();
        assert!(
            multi_child_parents >= 1,
            "At least one parent should have multiple children"
        );

        // Verify structure: each tuple has (parent_text, children, metadata)
        for (parent_text, children, meta) in &result {
            assert!(!parent_text.is_empty(), "Parent text should not be empty");
            assert!(!children.is_empty(), "Children should not be empty");
            // Position should be sequential
            assert!(meta.position < result.len() as u32);
        }

        // Total children should be more than total parents
        let total_children: usize = result.iter().map(|(_, c, _)| c.len()).sum();
        assert!(
            total_children > result.len(),
            "Total children ({total_children}) should exceed parent count ({})",
            result.len()
        );
    }

    #[test]
    fn parent_child_children_are_substrings_of_parent() {
        let content =
            "This is a comprehensive document about software engineering practices. ".repeat(400);

        let result = chunk_content_with_parents(&content, SourceType::Text).unwrap();
        assert!(!result.is_empty());

        for (parent_text, children, _) in &result {
            // The cleaned parent text should contain each child
            let cleaned_parent = clean_content(parent_text);
            for child in children {
                assert!(
                    cleaned_parent.contains(child.as_str()),
                    "Child should be a substring of its parent.\nChild: {}\nParent: {}",
                    &child[..child.len().min(80)],
                    &cleaned_parent[..cleaned_parent.len().min(200)]
                );
            }

            // First child should start at the beginning of the parent
            if let Some(first_child) = children.first() {
                assert!(
                    cleaned_parent
                        .starts_with(first_child.split('\n').next().unwrap_or(first_child)),
                    "First child should start at the beginning of the parent"
                );
            }

            // Last child should end at the end of the parent
            if let Some(last_child) = children.last() {
                let last_line = last_child.split('\n').next_back().unwrap_or(last_child);
                assert!(
                    cleaned_parent.ends_with(last_line),
                    "Last child should end at the end of the parent.\nLast child ends: {:?}\nParent ends: {:?}",
                    &last_child[last_child.len().saturating_sub(60)..],
                    &cleaned_parent[cleaned_parent.len().saturating_sub(60)..]
                );
            }
        }
    }

    #[test]
    fn parent_child_small_parent_not_subsplit() {
        // Content well under CHILD_CHUNK_SIZE (256 tokens) → should not be sub-split
        let content = "Short text. ".repeat(15); // ~15 tokens, well under 256

        let result = chunk_content_with_parents(&content, SourceType::Text).unwrap();
        assert_eq!(result.len(), 1, "Small content should produce 1 parent");

        let (parent_text, children, _) = &result[0];
        assert_eq!(children.len(), 1, "Small parent should not be sub-split");
        // The single child should be the cleaned version of the parent
        let cleaned_parent = clean_content(parent_text);
        assert_eq!(
            children[0], cleaned_parent,
            "Single child should equal cleaned parent text"
        );
    }

    #[test]
    fn parent_child_empty_content() {
        let result = chunk_content_with_parents("", SourceType::Text).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn parent_child_pdf_inherits_page_metadata() {
        // Create content spanning multiple pages (~3000 chars per page)
        let content = "A".repeat(9000);

        let result = chunk_content_with_parents(&content, SourceType::Pdf).unwrap();
        assert!(!result.is_empty());

        // PDF parent metadata should have page numbers
        let has_page = result.iter().any(|(_, _, meta)| meta.page_number.is_some());
        assert!(has_page, "PDF parents should have page_number metadata");
    }

    #[test]
    fn parent_child_markdown_inherits_section_header() {
        let content = format!(
            "# Introduction\n\n{}\n\n## Methods\n\n{}\n\n## Results\n\n{}",
            "Introduction content with detailed text. ".repeat(100),
            "Methods section describing approaches. ".repeat(100),
            "Results section with findings. ".repeat(100),
        );

        let result = chunk_content_with_parents(&content, SourceType::Markdown).unwrap();
        assert!(!result.is_empty());

        // At least one parent should have section_header metadata
        let has_header = result
            .iter()
            .any(|(_, _, meta)| meta.section_header.is_some());
        assert!(
            has_header,
            "Markdown parents should have section_header metadata"
        );
    }

    #[test]
    fn parent_child_parent_sizes_correct() {
        // Every parent must be strictly larger than a child, or the second pass
        // could not split it.
        for source_type in [
            SourceType::Pdf,
            SourceType::Web,
            SourceType::Markdown,
            SourceType::Text,
            SourceType::Docx,
            SourceType::Epub,
            SourceType::Youtube,
        ] {
            assert!(
                parent_chunk_size(source_type) > CHILD_CHUNK_SIZE,
                "{source_type:?} parent is not larger than a child"
            );
        }
    }

    #[test]
    fn parent_child_web_type_produces_hierarchy() {
        // Web source type uses the same non-markdown code path as Text/PDF
        // but with its own parent size (1024). Verify it produces hierarchy.
        let content =
            "The quick brown fox jumps over the lazy dog near the river bank. ".repeat(500);

        let result = chunk_content_with_parents(&content, SourceType::Web).unwrap();

        assert!(
            result.len() >= 2,
            "Web: 3000-token text should produce at least 2 parents, got {}",
            result.len()
        );

        let total_children: usize = result.iter().map(|(_, c, _)| c.len()).sum();
        assert!(
            total_children > result.len(),
            "Web: total children ({total_children}) should exceed parent count ({})",
            result.len()
        );
    }
}
