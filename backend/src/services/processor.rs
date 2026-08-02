//! Document processor service.
//!
//! Handles text extraction and normalization for different document types:
//! - PDF extraction via `pdf_extract`
//! - DOCX extraction via `zip` + `quick-xml` (OOXML)
//! - EPUB extraction via `zip` + `scraper` (XHTML chapters)
//! - Plain text normalization
//! - Markdown processing (structure-preserving)

use crate::error::AppError;

// ============================================================================
// Public API
// ============================================================================

/// Extract text content from PDF bytes.
///
/// # Errors
/// Returns `AppError::Internal` if PDF extraction fails.
pub fn extract_pdf_text(pdf_bytes: &[u8]) -> Result<String, AppError> {
    pdf_extract::extract_text_from_mem(pdf_bytes).map_err(|e| {
        let safe: String = e
            .to_string()
            .chars()
            .filter(|c| !c.is_control())
            .take(200)
            .collect();
        AppError::Internal(format!("Failed to extract PDF text: {safe}"))
    })
}

/// Extract text content from PDF bytes, returning one string per page.
///
/// Uses `pdf_extract::extract_text_from_mem_by_pages` which yields per-page
/// text natively from the PDF structure — no form-feed splitting needed.
///
/// # Errors
/// Returns `AppError::Internal` if PDF parsing fails.
pub fn extract_pdf_text_by_pages(pdf_bytes: &[u8]) -> Result<Vec<String>, AppError> {
    pdf_extract::extract_text_from_mem_by_pages(pdf_bytes).map_err(|e| {
        let safe: String = e
            .to_string()
            .chars()
            .filter(|c| !c.is_control())
            .take(200)
            .collect();
        AppError::Internal(format!("Failed to extract PDF text: {safe}"))
    })
}

/// Get the number of pages in a PDF.
///
/// Uses `lopdf` to parse the PDF structure and count pages.
/// Returns `Ok(0)` if the page count cannot be determined (fallback for OCR
/// to process the entire document).
///
/// # Errors
/// Always returns `Ok` — parse failures are logged and default to 0.
pub fn get_pdf_page_count(pdf_bytes: &[u8]) -> Result<usize, AppError> {
    match lopdf::Document::load_mem(pdf_bytes) {
        Ok(doc) => Ok(doc.get_pages().len()),
        Err(e) => {
            let msg: String = e
                .to_string()
                .chars()
                .filter(|c| !c.is_control())
                .take(200)
                .collect();
            tracing::warn!(
                error = msg,
                "Could not determine PDF page count, defaulting to 0"
            );
            Ok(0)
        }
    }
}

/// Detect which PDF pages need OCR based on extracted text sparsity.
///
/// Each element in `page_segments` is the extracted text for one PDF page
/// (as returned by [`extract_pdf_text_by_pages`]). Pages whose trimmed
/// character count falls below `min_text_threshold` are returned as
/// 0-indexed page indices suitable for the Mistral OCR API's `pages` field.
///
/// # Behavior
/// - Empty `page_segments` → returns `[]` (no pages to detect)
/// - Pages with `>= min_text_threshold` trimmed chars are skipped
/// - Capped at 10,000 pages (practical PDF maximum)
#[must_use]
pub fn detect_ocr_pages(page_segments: &[String], min_text_threshold: usize) -> Vec<u32> {
    const MAX_PAGES: usize = 10_000;
    let count = page_segments.len().min(MAX_PAGES);

    (0..count)
        .filter(|&i| page_segments[i].trim().chars().count() < min_text_threshold)
        .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
        .collect()
}

// ============================================================================
// ZIP archive safety
// ============================================================================

/// Maximum ratio of decompressed size to compressed size (100:1).
const MAX_COMPRESSION_RATIO: u64 = 100;

/// Maximum total decompressed size across all archive entries (500 MB).
const MAX_DECOMPRESSED_SIZE: u64 = 500 * 1024 * 1024;

/// Check a ZIP archive for decompression bomb indicators.
///
/// A ZIP bomb is a small compressed file that decompresses to an enormous size,
/// exhausting memory. This check inspects the archive's metadata (without
/// decompressing) to reject archives with suspicious compression ratios or
/// excessive total decompressed sizes.
///
/// **Note:** The metadata-based check uses values from the ZIP central directory
/// which can be attacker-controlled. This serves as a fast pre-filter; the
/// actual decompression protection is enforced by [`read_zip_entry_limited`]
/// which caps bytes read during streaming extraction.
fn check_zip_bomb<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
) -> Result<(), AppError> {
    let mut total_uncompressed: u64 = 0;

    for i in 0..archive.len() {
        let Ok(file) = archive.by_index_raw(i) else {
            continue;
        };
        let compressed = file.compressed_size();
        let uncompressed = file.size();
        total_uncompressed = total_uncompressed.saturating_add(uncompressed);

        // Check per-entry compression ratio
        if compressed == 0 && uncompressed > 0 {
            return Err(AppError::Validation(
                "Suspicious archive: entry claims non-zero size with zero compressed size".into(),
            ));
        }
        if compressed > 0 && uncompressed / compressed > MAX_COMPRESSION_RATIO {
            return Err(AppError::Validation(
                "Suspicious archive: compression ratio too high".into(),
            ));
        }
    }

    if total_uncompressed > MAX_DECOMPRESSED_SIZE {
        return Err(AppError::Validation(
            "Suspicious archive: total decompressed size exceeds limit".into(),
        ));
    }

    Ok(())
}

/// Maximum bytes to read from a single decompressed ZIP entry (100 MB).
/// Enforced during actual streaming decompression as a defense-in-depth
/// against crafted archives with spoofed central directory metadata.
const MAX_ENTRY_DECOMPRESSED_BYTES: u64 = 100 * 1024 * 1024;

/// Read a ZIP entry into a String with a streaming byte limit.
///
/// Unlike `file.size()` (which reads from the attacker-controlled central
/// directory), this enforces the limit during actual decompression via
/// `Read::take()`.
fn read_zip_entry_limited(file: &mut impl std::io::Read, limit: u64) -> Result<String, AppError> {
    use std::io::Read;
    let mut buf = String::new();
    file.take(limit)
        .read_to_string(&mut buf)
        .map_err(|e| AppError::Validation(format!("Failed to read archive entry: {e}")))?;
    Ok(buf)
}

/// Extract text content from DOCX bytes as markdown-like format.
///
/// Parses the OOXML `word/document.xml` inside the ZIP archive, preserving
/// headings (Heading1–6 → `#`–`######`), paragraphs, and list items.
///
/// # Errors
/// Returns `AppError::Validation` for corrupted, password-protected, or invalid DOCX files.
pub fn extract_docx_text(docx_bytes: &[u8]) -> Result<String, AppError> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;
    use zip::ZipArchive;

    let cursor = std::io::Cursor::new(docx_bytes);
    let mut archive = ZipArchive::new(cursor)
        .map_err(|_| AppError::Validation("DOCX file is corrupted or invalid".into()))?;

    check_zip_bomb(&mut archive)?;

    // Check for encryption (password-protected DOCX has EncryptedPackage)
    if archive.by_name("EncryptedPackage").is_ok() {
        return Err(AppError::Validation(
            "This document is password protected".into(),
        ));
    }

    let document_xml = {
        let mut file = archive
            .by_name("word/document.xml")
            .map_err(|_| AppError::Validation("File does not contain a valid document".into()))?;
        read_zip_entry_limited(&mut file, MAX_ENTRY_DECOMPRESSED_BYTES)?
    };

    let mut reader = Reader::from_str(&document_xml);
    let mut output = String::with_capacity(document_xml.len() / 3);
    let mut current_paragraph = String::new();
    let mut current_style = String::new();
    let mut in_paragraph = false;
    let mut in_run = false;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"w:p" => {
                    in_paragraph = true;
                    current_paragraph.clear();
                    current_style.clear();
                }
                b"w:r" => {
                    in_run = true;
                }
                b"w:pStyle" => {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"w:val" {
                            current_style = String::from_utf8_lossy(&attr.value).to_string();
                        }
                    }
                }
                _ => {}
            },
            Ok(Event::Empty(e)) => {
                if e.name().as_ref() == b"w:br" && in_run {
                    current_paragraph.push('\n');
                } else if e.name().as_ref() == b"w:pStyle" {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"w:val" {
                            current_style = String::from_utf8_lossy(&attr.value).to_string();
                        }
                    }
                }
            }
            Ok(Event::Text(e)) => {
                if in_run && in_paragraph {
                    current_paragraph.push_str(
                        &e.decode()
                            .inspect_err(|err| {
                                tracing::warn!(?err, "Failed to decode XML text in DOCX paragraph");
                            })
                            .unwrap_or_default(),
                    );
                }
            }
            // quick-xml emits entities separately from text; without this the
            // paragraph would silently lose every `&amp;` it contains.
            Ok(Event::GeneralRef(e)) => {
                if in_run
                    && in_paragraph
                    && let Some(resolved) = crate::xml::resolve_general_ref(&e)
                {
                    current_paragraph.push_str(&resolved);
                }
            }
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"w:r" => {
                    in_run = false;
                }
                b"w:p" => {
                    in_paragraph = false;
                    let text = current_paragraph.trim().to_string();
                    if !text.is_empty() {
                        let heading_prefix = style_to_heading_prefix(&current_style);
                        if !output.is_empty() {
                            output.push('\n');
                            // Extra newline before headings for readability
                            if heading_prefix.starts_with('#') {
                                output.push('\n');
                            }
                        }
                        output.push_str(heading_prefix);
                        output.push_str(&text);
                    }
                    current_paragraph.clear();
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(AppError::Internal(format!("Failed to parse DOCX XML: {e}")));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(output)
}

/// Extract text content from EPUB bytes as markdown-like format.
///
/// Reads the OPF manifest to determine chapter order, then extracts text from
/// each XHTML chapter using `scraper`. Returns concatenated chapters with
/// `---` separators.
///
/// Also returns metadata (title, author) via the second tuple element.
///
/// # Errors
/// Returns `AppError::Validation` for corrupted, DRM-protected, or invalid EPUB files.
#[allow(clippy::expect_used)] // `body` is a static selector covered by tests.
pub fn extract_epub_text(epub_bytes: &[u8]) -> Result<(String, serde_json::Value), AppError> {
    use scraper::{Html, Selector};
    use zip::ZipArchive;

    // SAFETY: "body" is a valid CSS selector and will never fail to parse
    static BODY_SELECTOR: std::sync::LazyLock<Selector> =
        std::sync::LazyLock::new(|| Selector::parse("body").expect("valid CSS selector"));

    let cursor = std::io::Cursor::new(epub_bytes);
    let mut archive = ZipArchive::new(cursor)
        .map_err(|_| AppError::Validation("EPUB file is corrupted or invalid".into()))?;

    check_zip_bomb(&mut archive)?;

    // Check for DRM
    if archive.by_name("META-INF/encryption.xml").is_ok() {
        return Err(AppError::Validation(
            "This book is DRM-protected and cannot be imported".into(),
        ));
    }

    // Read container.xml to find OPF path
    let opf_path = {
        let mut file = archive.by_name("META-INF/container.xml").map_err(|_| {
            AppError::Validation("Invalid EPUB: META-INF/container.xml missing".into())
        })?;
        let container_xml = read_zip_entry_limited(&mut file, MAX_ENTRY_DECOMPRESSED_BYTES)?;
        extract_opf_path(&container_xml)?
    };

    // Determine the base directory of the OPF file
    let opf_base = opf_path.rfind('/').map(|i| &opf_path[..=i]).unwrap_or("");

    // Read and parse OPF
    let opf_content = {
        let mut file = archive
            .by_name(&opf_path)
            .map_err(|_| AppError::Validation(format!("Invalid EPUB: {opf_path} not found")))?;
        read_zip_entry_limited(&mut file, MAX_ENTRY_DECOMPRESSED_BYTES)?
    };

    let opf = parse_opf(&opf_content)?;

    // Extract text from each chapter in spine order
    let mut chapters = Vec::new();
    let body_selector = &*BODY_SELECTOR;

    // Reuse buffers across iterations to avoid per-chapter allocations
    let mut xhtml = String::new();
    let mut chapter_text = String::new();

    for item_id in &opf.spine_items {
        let Some(href) = opf.manifest.get(item_id.as_str()) else {
            continue;
        };

        // Resolve href relative to OPF base directory
        let full_path = if let Some(stripped) = href.strip_prefix('/') {
            stripped.to_string()
        } else {
            format!("{opf_base}{href}")
        };

        xhtml.clear();
        if let Ok(mut f) = archive.by_name(&full_path) {
            if read_zip_entry_limited(&mut f, MAX_ENTRY_DECOMPRESSED_BYTES)
                .map(|s| xhtml = s)
                .is_err()
            {
                continue;
            }
        } else {
            continue;
        }

        let document = Html::parse_document(&xhtml);
        chapter_text.clear();

        // Extract body content, converting headings to markdown
        for element in document.select(body_selector) {
            for node in element.children() {
                extract_html_node(node, &mut chapter_text);
            }
        }

        let trimmed = chapter_text.trim().to_string();
        if !trimmed.is_empty() {
            chapters.push(trimmed);
        }
    }

    let full_text = chapters.join("\n\n---\n\n");

    let metadata = serde_json::json!({
        "title": opf.title.unwrap_or_default(),
        "author": opf.author.unwrap_or_default(),
    });

    Ok((full_text, metadata))
}

/// Process and clean text content.
///
/// - Trims whitespace from each line
/// - Removes empty lines
/// - Collapses consecutive newlines into one
#[must_use]
pub fn process_text(content: &str) -> String {
    let mut result = String::with_capacity(content.len());
    let mut prev_was_newline = true; // Start true to skip leading newlines

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if !prev_was_newline && !result.is_empty() {
            result.push('\n');
        }
        result.push_str(trimmed);
        prev_was_newline = false;
    }

    result
}

/// Process markdown content (preserves structure).
///
/// Preserves leading whitespace for structural elements:
/// - Headers (`#`)
/// - Lists (`-`, `*`, digits)
/// - Code blocks (`` ` ``)
/// - Blockquotes (`>`)
#[must_use]
pub fn process_markdown(content: &str) -> String {
    content
        .lines()
        .map(|line| {
            if is_markdown_structural(line) {
                line
            } else {
                line.trim()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Map OOXML paragraph style to Markdown heading prefix.
fn style_to_heading_prefix(style: &str) -> &'static str {
    match style {
        s if s.contains("Heading1") || s == "Titre1" || s == "Title" => "# ",
        s if s.contains("Heading2") || s == "Titre2" => "## ",
        s if s.contains("Heading3") || s == "Titre3" => "### ",
        s if s.contains("Heading4") || s == "Titre4" => "#### ",
        s if s.contains("Heading5") || s == "Titre5" => "##### ",
        s if s.contains("Heading6") || s == "Titre6" => "###### ",
        s if s.contains("ListParagraph") || s.contains("ListBullet") => "- ",
        _ => "",
    }
}

/// Check if a line is a markdown structural element that should preserve formatting.
fn is_markdown_structural(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with('#')
        || trimmed.starts_with('-')
        || trimmed.starts_with('*')
        || trimmed.starts_with('`')
        || trimmed.starts_with('>')
        || trimmed.chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// Extract the OPF file path from container.xml.
fn extract_opf_path(container_xml: &str) -> Result<String, AppError> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(container_xml);
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(e)) | Ok(Event::Start(e)) if e.name().as_ref() == b"rootfile" => {
                for attr in e.attributes().flatten() {
                    if attr.key.as_ref() == b"full-path" {
                        return Ok(String::from_utf8_lossy(&attr.value).to_string());
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(AppError::Internal(format!(
                    "Failed to parse container.xml: {e}"
                )));
            }
            _ => {}
        }
        buf.clear();
    }

    Err(AppError::Validation(
        "Invalid EPUB: OPF path not found in container.xml".into(),
    ))
}

/// Parsed OPF data.
struct OpfData {
    spine_items: Vec<String>,
    manifest: std::collections::HashMap<String, String>,
    title: Option<String>,
    author: Option<String>,
}

/// Parse an OPF file to extract spine order, manifest items, title, and author.
fn parse_opf(opf_content: &str) -> Result<OpfData, AppError> {
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    let mut reader = Reader::from_str(opf_content);
    let mut buf = Vec::new();

    let mut manifest: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut spine_items: Vec<String> = Vec::new();
    let mut title: Option<String> = None;
    let mut author: Option<String> = None;

    let mut in_title = false;
    let mut in_creator = false;
    // Reuse buffers across loop iterations to avoid per-element allocations
    let mut item_id = String::new();
    let mut item_href = String::new();
    let mut metadata_buf = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.name().as_ref() {
                b"dc:title" => in_title = true,
                b"dc:creator" => in_creator = true,
                _ => {}
            },
            Ok(Event::Empty(e)) => match e.name().as_ref() {
                b"item" => {
                    item_id.clear();
                    item_href.clear();
                    for attr in e.attributes().flatten() {
                        match attr.key.as_ref() {
                            b"id" => {
                                item_id.push_str(&String::from_utf8_lossy(&attr.value));
                            }
                            b"href" => {
                                item_href.push_str(&String::from_utf8_lossy(&attr.value));
                            }
                            _ => {}
                        }
                    }
                    if !item_id.is_empty() && !item_href.is_empty() {
                        manifest.insert(item_id.clone(), item_href.clone());
                    }
                }
                b"itemref" => {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"idref" {
                            spine_items.push(String::from_utf8_lossy(&attr.value).to_string());
                        }
                    }
                }
                _ => {}
            },
            // Accumulated rather than taken from the first text event: an
            // entity splits the value across several events, so `Tom &amp;
            // Jerry` arrives as three of them and taking the first would store
            // "Tom".
            Ok(Event::Text(e)) => {
                if in_title || in_creator {
                    metadata_buf.push_str(
                        &e.decode()
                            .inspect_err(|err| {
                                tracing::warn!(
                                    ?err,
                                    "Failed to decode XML text in EPUB OPF metadata"
                                );
                            })
                            .unwrap_or_default(),
                    );
                }
            }
            Ok(Event::GeneralRef(e)) => {
                if (in_title || in_creator)
                    && let Some(resolved) = crate::xml::resolve_general_ref(&e)
                {
                    metadata_buf.push_str(&resolved);
                }
            }
            Ok(Event::End(e)) => match e.name().as_ref() {
                b"dc:title" => {
                    let text = metadata_buf.trim();
                    if title.is_none() && !text.is_empty() {
                        title = Some(text.to_string());
                    }
                    metadata_buf.clear();
                    in_title = false;
                }
                b"dc:creator" => {
                    let text = metadata_buf.trim();
                    if author.is_none() && !text.is_empty() {
                        author = Some(text.to_string());
                    }
                    metadata_buf.clear();
                    in_creator = false;
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(AppError::Internal(format!("Failed to parse OPF: {e}")));
            }
            _ => {}
        }
        buf.clear();
    }

    Ok(OpfData {
        spine_items,
        manifest,
        title,
        author,
    })
}

/// Recursively extract text from an HTML node, converting headings to markdown.
fn extract_html_node(node: ego_tree::NodeRef<'_, scraper::Node>, output: &mut String) {
    match node.value() {
        scraper::Node::Text(text) => {
            let t = text.trim();
            if !t.is_empty() {
                if !output.is_empty() && !output.ends_with('\n') && !output.ends_with(' ') {
                    output.push(' ');
                }
                output.push_str(t);
            }
        }
        scraper::Node::Element(elem) => {
            let tag = elem.name();
            let heading_prefix = match tag {
                "h1" => Some("# "),
                "h2" => Some("## "),
                "h3" => Some("### "),
                "h4" => Some("#### "),
                "h5" => Some("##### "),
                "h6" => Some("###### "),
                _ => None,
            };

            if let Some(prefix) = heading_prefix {
                if !output.is_empty() && !output.ends_with('\n') {
                    output.push('\n');
                }
                output.push('\n');
                output.push_str(prefix);
            }

            // Recurse into children
            for child in node.children() {
                extract_html_node(child, output);
            }

            // Add line breaks after block elements
            if let "p" | "div" | "br" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "li" = tag
                && !output.ends_with('\n')
            {
                output.push('\n');
            }
        }
        _ => {
            // Recurse into children for other node types (Document, etc.)
            for child in node.children() {
                extract_html_node(child, output);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a minimal but structurally valid single-page PDF containing
    /// `text`, with a correct cross-reference table.
    ///
    /// Built here rather than checked in as a binary fixture so the bytes are
    /// reviewable, and because a fixture whose provenance nobody remembers is
    /// how a "PDF extraction still works" test quietly stops proving anything.
    fn minimal_pdf(text: &str) -> Vec<u8> {
        let stream = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
             /Resources << /Font << /F1 5 0 R >> >> >>"
                .to_string(),
            format!(
                "<< /Length {} >>\nstream\n{stream}\nendstream",
                stream.len()
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];

        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = Vec::with_capacity(objects.len());
        for (i, body) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.push_str(&format!("{} 0 obj\n{body}\nendobj\n", i + 1));
        }

        let xref_offset = pdf.len();
        pdf.push_str(&format!("xref\n0 {}\n", objects.len() + 1));
        pdf.push_str("0000000000 65535 f \n");
        for offset in &offsets {
            pdf.push_str(&format!("{offset:010} 00000 n \n"));
        }
        pdf.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
            objects.len() + 1
        ));

        pdf.into_bytes()
    }

    /// The guard for the pdf-extract 0.7 -> 0.12 and lopdf 0.34 -> 0.42 jump.
    /// PDF ingestion is the product's main input and nothing else exercises it.
    #[test]
    fn pdf_extraction_returns_page_text() {
        let pdf = minimal_pdf("Reciprocal Rank Fusion uses k of 60");
        let text = extract_pdf_text(&pdf).expect("extraction");
        assert!(text.contains("Reciprocal Rank Fusion"), "got {text:?}");
    }

    #[test]
    fn pdf_page_count_and_per_page_extraction_agree() {
        let pdf = minimal_pdf("one page only");
        assert_eq!(get_pdf_page_count(&pdf).expect("count"), 1);

        let pages = extract_pdf_text_by_pages(&pdf).expect("per-page extraction");
        assert_eq!(pages.len(), 1, "got {pages:?}");
        assert!(pages[0].contains("one page only"), "got {pages:?}");
    }

    /// Build a minimal but real DOCX: a ZIP whose `word/document.xml` is
    /// OOXML. Small enough to read, real enough that the extractor takes its
    /// normal path rather than a test-only one.
    fn docx_with_body(body: &str) -> Vec<u8> {
        use std::io::Write;
        use zip::write::SimpleFileOptions;

        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut cursor);
            zip.start_file("word/document.xml", SimpleFileOptions::default())
                .expect("start file");
            let xml = format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body}</w:body></w:document>"#
            );
            zip.write_all(xml.as_bytes()).expect("write");
            zip.finish().expect("finish");
        }
        cursor.into_inner()
    }

    fn paragraph(text: &str) -> String {
        format!("<w:p><w:r><w:t>{text}</w:t></w:r></w:p>")
    }

    #[test]
    fn docx_extraction_returns_paragraph_text() {
        let docx = docx_with_body(&paragraph("Retrieval fuses two ranked lists."));
        let text = extract_docx_text(&docx).expect("extraction");
        assert!(
            text.contains("Retrieval fuses two ranked lists."),
            "got {text:?}"
        );
    }

    /// The regression guard for the quick-xml 0.38 change: entities stopped
    /// arriving inside `Event::Text`. A reader that ignores `GeneralRef` still
    /// returns plausible text, it just silently loses every `&`, which would
    /// reach the index unnoticed.
    #[test]
    fn docx_extraction_keeps_entities() {
        let docx = docx_with_body(&paragraph("Tom &amp; Jerry &lt;tag&gt; it&#39;s"));
        let text = extract_docx_text(&docx).expect("extraction");
        assert!(text.contains("Tom & Jerry"), "lost `&amp;`: {text:?}");
        assert!(text.contains("<tag>"), "lost `&lt;`/`&gt;`: {text:?}");
        assert!(text.contains("it's"), "lost `&#39;`: {text:?}");
    }

    #[test]
    fn opf_metadata_keeps_entities_and_is_not_truncated_at_the_first_entity() {
        let opf = r#"<?xml version="1.0"?>
<package xmlns:dc="http://purl.org/dc/elements/1.1/">
<metadata>
<dc:title>Tom &amp; Jerry</dc:title>
<dc:creator>Ann &amp; Bob</dc:creator>
</metadata>
<manifest><item id="c1" href="c1.xhtml"/></manifest>
<spine><itemref idref="c1"/></spine>
</package>"#;
        let parsed = parse_opf(opf).expect("parse");
        // Taking the first text event would have stored "Tom".
        assert_eq!(parsed.title.as_deref(), Some("Tom & Jerry"));
        assert_eq!(parsed.author.as_deref(), Some("Ann & Bob"));
        assert_eq!(parsed.spine_items, vec!["c1"]);
    }

    #[test]
    fn process_text_normalizes_whitespace() {
        let input = "  hello  \n  world  ";
        assert_eq!(process_text(input), "hello\nworld");
    }

    #[test]
    fn process_text_removes_empty_lines() {
        let input = "line1\n\n\n\nline2";
        assert_eq!(process_text(input), "line1\nline2");
    }

    #[test]
    fn process_text_handles_empty_input() {
        assert_eq!(process_text(""), "");
        assert_eq!(process_text("   \n\n   "), "");
    }

    #[test]
    fn process_markdown_preserves_headers() {
        let input = "# Header\n  normal text  ";
        assert_eq!(process_markdown(input), "# Header\nnormal text");
    }

    #[test]
    fn process_markdown_preserves_lists() {
        let input = "- item 1\n* item 2\n1. item 3";
        assert_eq!(process_markdown(input), "- item 1\n* item 2\n1. item 3");
    }

    #[test]
    fn process_markdown_preserves_code_blocks() {
        let input = "```rust\nlet x = 1;\n```";
        assert_eq!(process_markdown(input), "```rust\nlet x = 1;\n```");
    }

    #[test]
    fn process_markdown_preserves_blockquotes() {
        let input = "> quoted text";
        assert_eq!(process_markdown(input), "> quoted text");
    }

    #[test]
    fn is_markdown_structural_detection() {
        assert!(is_markdown_structural("# Header"));
        assert!(is_markdown_structural("- list item"));
        assert!(is_markdown_structural("* list item"));
        assert!(is_markdown_structural("1. numbered"));
        assert!(is_markdown_structural("```code"));
        assert!(is_markdown_structural("> quote"));
        assert!(is_markdown_structural("  # indented header"));
        assert!(!is_markdown_structural("normal text"));
    }

    // --- detect_ocr_pages tests ---

    #[test]
    fn detect_ocr_pages_empty_segments_returns_all() {
        let segments: Vec<String> = vec![String::new(); 5];
        assert_eq!(detect_ocr_pages(&segments, 50), vec![0, 1, 2, 3, 4]);

        let whitespace: Vec<String> = vec!["   \n\n  ".into(); 3];
        assert_eq!(detect_ocr_pages(&whitespace, 50), vec![0, 1, 2]);
    }

    #[test]
    fn detect_ocr_pages_full_text_returns_none() {
        let page = "A".repeat(100);
        let segments = vec![page.clone(), page.clone(), page];
        assert_eq!(detect_ocr_pages(&segments, 50), Vec::<u32>::new());
    }

    #[test]
    fn detect_ocr_pages_mixed_returns_sparse_pages() {
        // Pages 0,2,4 have plenty of text; pages 1,3 have <50 chars
        let full = "B".repeat(100);
        let segments = vec![full.clone(), "x".into(), full.clone(), "x".into(), full];
        assert_eq!(detect_ocr_pages(&segments, 50), vec![1, 3]);
    }

    #[test]
    fn detect_ocr_pages_no_segments_returns_empty() {
        assert_eq!(detect_ocr_pages(&[], 50), Vec::<u32>::new());
    }

    #[test]
    fn detect_ocr_pages_zero_threshold_returns_none() {
        // threshold=0 means no segment can have < 0 chars, so no OCR triggered
        let segments = vec!["short".into(), "text".into()];
        assert_eq!(detect_ocr_pages(&segments, 0), Vec::<u32>::new());
    }

    #[test]
    fn detect_ocr_pages_uses_char_count_not_bytes() {
        // Multi-byte chars: 3 CJK characters = 3 chars but 9 bytes
        let segments = vec!["你好世".into()]; // 3 chars, 9 bytes
        // threshold=4: should flag this page (3 chars < 4)
        assert_eq!(detect_ocr_pages(&segments, 4), vec![0]);
        // threshold=3: should NOT flag (3 chars >= 3)
        assert_eq!(detect_ocr_pages(&segments, 3), Vec::<u32>::new());
    }

    #[test]
    fn get_pdf_page_count_invalid_bytes_returns_zero() {
        assert_eq!(get_pdf_page_count(b"not a pdf").unwrap(), 0);
    }

    #[test]
    fn extract_opf_path_from_container() {
        let xml = r#"<?xml version="1.0"?>
        <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
          <rootfiles>
            <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
          </rootfiles>
        </container>"#;
        assert_eq!(extract_opf_path(xml).unwrap(), "OEBPS/content.opf");
    }

    #[test]
    fn check_zip_bomb_rejects_suspicious_ratio() {
        use std::io::{Cursor, Write};
        use zip::ZipWriter;
        use zip::write::SimpleFileOptions;

        // Create a ZIP with a file that has a very high compression ratio.
        // We write a highly compressible payload (all zeros).
        let buf = Vec::new();
        let mut writer = ZipWriter::new(Cursor::new(buf));
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        writer.start_file("bomb.txt", options).unwrap();
        // 10MB of zeros compresses to very little
        let zeros = vec![0u8; 10 * 1024 * 1024];
        writer.write_all(&zeros).unwrap();
        let result = writer.finish().unwrap();
        let bytes = result.into_inner();

        let mut archive = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        // Should be rejected because ratio >> 100:1
        assert!(check_zip_bomb(&mut archive).is_err());
    }

    #[test]
    fn check_zip_bomb_accepts_normal_archive() {
        use std::io::{Cursor, Write};
        use zip::ZipWriter;
        use zip::write::SimpleFileOptions;

        let buf = Vec::new();
        let mut writer = ZipWriter::new(Cursor::new(buf));
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        writer.start_file("normal.txt", options).unwrap();
        writer.write_all(b"Hello, world!").unwrap();
        let result = writer.finish().unwrap();
        let bytes = result.into_inner();

        let mut archive = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        assert!(check_zip_bomb(&mut archive).is_ok());
    }
}
