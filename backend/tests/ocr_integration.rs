#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented
)]

//! Integration tests for OCR support (US-011).
//!
//! Covers the Mistral OCR pipeline end-to-end:
//! - OCR client with wiremock HTTP mocking
//! - Per-page text detection (scanned vs. text-layer PDFs)
//! - Content-hash caching (in-memory mock)
//! - Text merging (native text + OCR markdown)
//!
//! Per-plan OCR limits are a hosted concern and live in
//! `tests/saas_ocr_limits.rs` (US-013).
//!
//! Tests marked `#[ignore]` require a real PostgreSQL database.
//! Run with: `TEST_DATABASE_URL=postgres://... cargo test -- --ignored`
//!
//! Non-ignored tests complete in < 500ms using no external services.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lopdf::dictionary;
use serde_json::json;
use uuid::Uuid;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use openbooklm::clients::{MistralOcrClient, OcrPage, ProviderMetrics};
use openbooklm::repositories::{OcrCacheRepository, RepoResult};
use openbooklm::services::processor::{detect_ocr_pages, get_pdf_page_count};
use openbooklm::services::source_processing::merge_ocr_pages;

// ============================================================================
// Helpers: Minimal PDF generation (lopdf)
// ============================================================================

/// Create a minimal valid PDF with a text content stream.
///
/// Produces a single-page PDF with extractable text, which `pdf_extract` will
/// successfully read — exercising the "no OCR needed" code path.
fn minimal_text_pdf(text: &str) -> Vec<u8> {
    use lopdf::{
        Document, Object, Stream,
        content::{Content, Operation},
    };

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });

    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 12.into()]),
            Operation::new("Td", vec![100.into(), 700.into()]),
            Operation::new("Tj", vec![Object::string_literal(text)]),
            Operation::new("ET", vec![]),
        ],
    };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        "Contents" => content_id,
        "Resources" => dictionary! {
            "Font" => dictionary! { "F1" => font_id }
        },
    });

    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    bytes
}

/// Create a PDF whose page `i` carries `pages[i]` as its only text.
///
/// Used to prove that a chunk's reported page is the page the extractor read it
/// from (US-019 AC-4), so each page's text has to be distinguishable.
fn paged_text_pdf(pages: &[&str]) -> Vec<u8> {
    use lopdf::{
        Document, Object, Stream,
        content::{Content, Operation},
    };

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });

    let mut page_ids = Vec::new();
    for text in pages {
        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 12.into()]),
                Operation::new("Td", vec![72.into(), 720.into()]),
                Operation::new("Tj", vec![Object::string_literal(*text)]),
                Operation::new("ET", vec![]),
            ],
        };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        page_ids.push(doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            "Contents" => content_id,
            "Resources" => dictionary! {
                "Font" => dictionary! { "F1" => font_id }
            },
        }));
    }

    let count = page_ids.len() as i64;
    let kids: Vec<Object> = page_ids.into_iter().map(Object::from).collect();
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count,
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    bytes
}

/// Create a minimal PDF with no text layer (empty page, no content stream).
///
/// This simulates a scanned/image-only PDF where `pdf_extract` returns empty
/// text — exercising the "OCR needed" code path.
fn minimal_image_only_pdf() -> Vec<u8> {
    use lopdf::{Document, Object};

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
        // No "Contents" key → no text layer
    });

    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    bytes
}

/// Create a multi-page PDF where some pages have text and some don't.
fn multi_page_pdf(page_count: usize, text_pages: &[usize]) -> Vec<u8> {
    use lopdf::{
        Document, Object, Stream,
        content::{Content, Operation},
    };

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });

    let mut page_ids = Vec::new();
    for i in 0..page_count {
        let page_id = if text_pages.contains(&i) {
            let text = format!(
                "Page {i} has sufficient text content for detection threshold check - padding"
            );
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new("Tf", vec!["F1".into(), 12.into()]),
                    Operation::new("Td", vec![100.into(), 700.into()]),
                    Operation::new("Tj", vec![Object::string_literal(text)]),
                    Operation::new("ET", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
            doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
                "Contents" => content_id,
                "Resources" => dictionary! {
                    "Font" => dictionary! { "F1" => font_id }
                },
            })
        } else {
            doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
            })
        };
        page_ids.push(page_id);
    }

    let kids: Vec<Object> = page_ids.into_iter().map(Object::from).collect();
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => page_count as i64,
        }),
    );

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).unwrap();
    bytes
}

// ============================================================================
// Mock: In-memory OcrCacheRepository
// ============================================================================

type OcrCacheKey = (String, String);

struct OcrCacheValue {
    source_id: Uuid,
    ocr_text: String,
    pages_processed: i32,
}

#[derive(Default)]
struct InMemoryOcrCache {
    inner: Mutex<HashMap<OcrCacheKey, OcrCacheValue>>,
}

#[async_trait]
impl OcrCacheRepository for InMemoryOcrCache {
    async fn find_by_hash(
        &self,
        source_id: Uuid,
        content_hash: &str,
        model: &str,
    ) -> RepoResult<Option<(String, i32)>> {
        let guard = self.inner.lock().unwrap();
        Ok(guard
            .get(&(content_hash.to_string(), model.to_string()))
            .filter(|value| value.source_id == source_id)
            .map(|value| (value.ocr_text.clone(), value.pages_processed)))
    }

    async fn store(
        &self,
        source_id: Uuid,
        content_hash: &str,
        model: &str,
        ocr_text: &str,
        pages_processed: i32,
    ) -> RepoResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard.retain(|(hash, cache_model), value| {
            value.source_id != source_id || (hash == content_hash && cache_model == model)
        });
        guard
            .entry((content_hash.to_string(), model.to_string()))
            .and_modify(|value| value.source_id = source_id)
            .or_insert_with(|| OcrCacheValue {
                source_id,
                ocr_text: ocr_text.to_string(),
                pages_processed,
            });
        Ok(())
    }

    async fn purge_unowned(&self) -> RepoResult<u64> {
        Ok(0)
    }
}

// ============================================================================
// Helper: create MistralOcrClient pointing at a wiremock server
// ============================================================================

fn make_test_ocr_client(mock_server_uri: &str) -> MistralOcrClient {
    let metrics = Arc::new(ProviderMetrics::new("test_ocr"));
    MistralOcrClient::new(
        "not-a-real-key",
        "mistral-ocr-latest".to_string(),
        50 * 1024 * 1024,
        Duration::from_secs(10),
        metrics,
    )
    .expect("Failed to create test OCR client")
    .with_base_url(mock_server_uri)
}

// ============================================================================
// 1. OCR Client Tests (wiremock)
// ============================================================================

/// Scanned PDF + OCR enabled → text is extracted via OCR.
#[tokio::test]
async fn ocr_client_extracts_text_from_scanned_pdf() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .and(header("Authorization", "Bearer not-a-real-key"))
        .and(body_partial_json(json!({
            "model": "mistral-ocr-latest",
            "include_image_base64": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "pages": [
                { "index": 0, "markdown": "# Invoice\n\nTotal: $100.00" }
            ],
            "usage_info": { "pages_processed": 1 }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = make_test_ocr_client(&mock_server.uri());
    let pdf_bytes = minimal_image_only_pdf();
    let result = client
        .extract_text_from_pdf(&pdf_bytes, None)
        .await
        .expect("OCR should succeed");

    assert_eq!(result.pages_processed, 1);
    assert_eq!(result.pages.len(), 1);
    assert!(result.pages[0].markdown.contains("Invoice"));
    assert!(result.pages[0].markdown.contains("$100.00"));
}

/// Only scanned pages are sent to OCR (lazy OCR via page selection).
#[tokio::test]
async fn ocr_client_sends_only_selected_pages() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .and(body_partial_json(json!({
            "model": "mistral-ocr-latest",
            "pages": [1, 3]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "pages": [
                { "index": 1, "markdown": "OCR page 1" },
                { "index": 3, "markdown": "OCR page 3" }
            ],
            "usage_info": { "pages_processed": 2 }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = make_test_ocr_client(&mock_server.uri());
    // Use a 4-page PDF so page indices [1, 3] correspond to actual pages.
    let pdf_bytes = multi_page_pdf(4, &[0, 2]);
    let result = client
        .extract_text_from_pdf(&pdf_bytes, Some(vec![1, 3]))
        .await
        .expect("OCR should succeed");

    assert_eq!(result.pages_processed, 2);
    assert_eq!(result.pages.len(), 2);
    assert_eq!(result.pages[0].index, 1);
    assert_eq!(result.pages[1].index, 3);
}

/// Mistral API returns an error → client returns error.
#[tokio::test]
async fn ocr_client_handles_api_error() {
    let mock_server = MockServer::start().await;

    // Return 401 for all attempts (client retries on 5xx, not on 4xx).
    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "message": "Invalid API key"
        })))
        .mount(&mock_server)
        .await;

    let client = make_test_ocr_client(&mock_server.uri());
    let pdf_bytes = minimal_image_only_pdf();
    let result = client.extract_text_from_pdf(&pdf_bytes, None).await;

    assert!(result.is_err(), "Should fail on 401");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("401") || err.contains("OCR"),
        "Error should mention status or OCR: {err}"
    );
}

/// Empty PDF bytes → pre-condition error (no API call).
#[tokio::test]
async fn ocr_client_rejects_empty_pdf() {
    let mock_server = MockServer::start().await;

    // Expect zero calls — the client should reject before sending.
    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&mock_server)
        .await;

    let client = make_test_ocr_client(&mock_server.uri());
    let result = client.extract_text_from_pdf(&[], None).await;

    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("empty"), "Error should mention empty: {err}");
}

/// Oversized PDF → pre-condition error (no API call).
#[tokio::test]
async fn ocr_client_rejects_oversized_pdf() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&mock_server)
        .await;

    // Create a client with a very small max to avoid allocating 50MB in tests.
    let metrics = Arc::new(ProviderMetrics::new("test_ocr"));
    let client = MistralOcrClient::new(
        "not-a-real-key",
        "mistral-ocr-latest".to_string(),
        1024, // 1KB max
        Duration::from_secs(10),
        metrics,
    )
    .unwrap()
    .with_base_url(mock_server.uri());

    let pdf_bytes = vec![0u8; 2048]; // 2KB, exceeds 1KB limit
    let result = client.extract_text_from_pdf(&pdf_bytes, None).await;

    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("exceeds"),
        "Error should mention size exceeded: {err}"
    );
}

// ============================================================================
// 2. Per-Page Text Detection Tests
// ============================================================================

/// Text PDF → pdf_extract succeeds → OCR is NOT called.
///
/// When all pages have sufficient text (above threshold), `detect_ocr_pages`
/// returns an empty list, meaning no OCR is needed.
#[test]
fn text_pdf_needs_no_ocr() {
    let page_text = "A".repeat(100);
    let segments = vec![page_text.clone(), page_text.clone(), page_text];
    let ocr_pages = detect_ocr_pages(&segments, 50);
    assert!(
        ocr_pages.is_empty(),
        "Text PDF should not need OCR, got: {ocr_pages:?}"
    );
}

/// Scanned PDF (empty text) → all pages need OCR.
#[test]
fn scanned_pdf_needs_full_ocr() {
    let segments: Vec<String> = vec![String::new(); 5];
    let ocr_pages = detect_ocr_pages(&segments, 50);
    assert_eq!(ocr_pages, vec![0, 1, 2, 3, 4]);
}

/// Mixed PDF → only scanned pages are identified for OCR.
#[test]
fn mixed_pdf_detects_scanned_pages() {
    let full = "B".repeat(100);
    let sparse = "x".to_string(); // < 50 chars
    // Pages: 0=full, 1=sparse, 2=full, 3=sparse, 4=full
    let segments = vec![full.clone(), sparse.clone(), full.clone(), sparse, full];
    let ocr_pages = detect_ocr_pages(&segments, 50);
    assert_eq!(
        ocr_pages,
        vec![1, 3],
        "Only pages with <50 chars should be flagged"
    );
}

/// Page count from a programmatically generated PDF.
#[test]
fn pdf_page_count_valid_pdf() {
    let pdf = minimal_text_pdf("Hello");
    let count = get_pdf_page_count(&pdf).expect("Should parse page count");
    assert_eq!(count, 1, "Single-page PDF should have count 1");
}

/// Page count from a multi-page PDF.
#[test]
fn pdf_page_count_multi_page() {
    let pdf = multi_page_pdf(4, &[0, 2]); // 4 pages, text on pages 0 and 2
    let count = get_pdf_page_count(&pdf).expect("Should parse page count");
    assert_eq!(count, 4, "4-page PDF should have count 4");
}

/// A native PDF's reported pages are the fixture's actual pages (US-019 AC-4).
///
/// The whole chain, offline: a five-page PDF is extracted page by page, the
/// pages reach the chunker as pages, and every chunk reports the page its text
/// was printed on. The heuristic this replaces divided a character offset by
/// 3,000, which put all five markers on page 1.
#[test]
fn native_pdf_chunks_report_the_page_they_were_printed_on() {
    use openbooklm::services::processor::extract_pdf_text_by_pages;
    use openbooklm::services::rag::chunking::{SourceText, chunk_source};
    use openbooklm::types::SourceType;

    let bodies: Vec<String> = (1..=5)
        .map(|i| format!("MARKER{i} is the only marker printed on this page."))
        .collect();
    let refs: Vec<&str> = bodies.iter().map(String::as_str).collect();
    let pdf = paged_text_pdf(&refs);

    let extracted = extract_pdf_text_by_pages(&pdf).expect("extraction succeeds");
    assert_eq!(extracted.len(), 5, "one segment per page");

    let parents = chunk_source(&SourceText::paginated(extracted), SourceType::Pdf)
        .expect("chunking succeeds");
    let children: Vec<_> = parents.iter().flat_map(|p| p.children.iter()).collect();

    for i in 1..=5u32 {
        let marker = format!("MARKER{i}");
        let ranges: Vec<(u32, u32)> = children
            .iter()
            .filter(|c| c.text.contains(&marker))
            .filter_map(|c| Some((c.metadata.page_number?, c.metadata.page_end?)))
            .collect();
        assert!(!ranges.is_empty(), "no chunk carried {marker}");
        assert!(
            ranges
                .iter()
                .all(|(first, last)| (*first..=*last).contains(&i)),
            "{marker} was printed on page {i}, chunks reported {ranges:?}"
        );
    }
}

/// OCR normalization keeps page identity: the merged pages stay pages, and a
/// chunk of the OCR text reports the page the OCR provider gave it.
#[test]
fn ocr_pages_keep_their_provenance_through_chunking() {
    use openbooklm::services::rag::chunking::{SourceText, chunk_source};
    use openbooklm::types::SourceType;

    // Pages 1 and 3 are scanned (empty native text), pages 0, 2, 4 are native.
    let native: Vec<String> = vec![
        "Native page zero content.".into(),
        String::new(),
        "Native page two content.".into(),
        String::new(),
        "Native page four content.".into(),
    ];
    let ocr = vec![
        OcrPage {
            index: 1,
            markdown: "# OCRMARKER1\r\n\r\nScanned page one body.".to_string(),
        },
        OcrPage {
            index: 3,
            markdown: "# OCRMARKER3\r\n\r\nScanned page three body.".to_string(),
        },
    ];

    let merged = merge_ocr_pages(&native, &ocr);
    let parents = chunk_source(&SourceText::paginated(merged), SourceType::Markdown)
        .expect("chunking succeeds");
    let children: Vec<_> = parents.iter().flat_map(|p| p.children.iter()).collect();

    // Line endings were normalized away; the page number was not.
    for (marker, expected) in [("OCRMARKER1", 2u32), ("OCRMARKER3", 4)] {
        let ranges: Vec<(u32, u32)> = children
            .iter()
            .filter(|c| c.text.contains(marker))
            .filter_map(|c| Some((c.metadata.page_number?, c.metadata.page_end?)))
            .collect();
        assert!(!ranges.is_empty(), "no chunk carried {marker}");
        assert!(
            ranges
                .iter()
                .all(|(first, last)| (*first..=*last).contains(&expected)),
            "{marker} belongs to page {expected}, chunks reported {ranges:?}"
        );
    }
    assert!(
        children.iter().all(|c| !c.text.contains('\r')),
        "normalization must remove carriage returns"
    );
}

/// Invalid bytes → page count defaults to 0 (graceful fallback).
#[test]
fn pdf_page_count_invalid_bytes() {
    let count = get_pdf_page_count(b"not a pdf").expect("Should not error");
    assert_eq!(count, 0, "Invalid PDF should fallback to 0 pages");
}

// ============================================================================
// 4. OCR Cache Tests (in-memory mock)
// ============================================================================

/// Cache miss → returns None.
#[tokio::test]
async fn ocr_cache_miss_returns_none() {
    let cache = InMemoryOcrCache::default();
    let result = cache
        .find_by_hash(Uuid::nil(), "abc123", "mistral-ocr-latest")
        .await
        .unwrap();
    assert!(result.is_none(), "Cache miss should return None");
}

/// Store + find → cache hit with correct data.
#[tokio::test]
async fn ocr_cache_hit_returns_stored_text() {
    let cache = InMemoryOcrCache::default();
    cache
        .store(
            Uuid::nil(),
            "abc123",
            "mistral-ocr-latest",
            "# Extracted text",
            3,
        )
        .await
        .unwrap();

    let result = cache
        .find_by_hash(Uuid::nil(), "abc123", "mistral-ocr-latest")
        .await
        .unwrap();
    assert_eq!(
        result,
        Some(("# Extracted text".to_string(), 3)),
        "Cache hit should return stored text and page count"
    );
}

/// Different model → cache miss (model is part of the cache key).
#[tokio::test]
async fn ocr_cache_different_model_is_miss() {
    let cache = InMemoryOcrCache::default();
    cache
        .store(Uuid::nil(), "abc123", "mistral-ocr-latest", "# Text", 1)
        .await
        .unwrap();

    let result = cache
        .find_by_hash(Uuid::nil(), "abc123", "mistral-ocr-v2")
        .await
        .unwrap();
    assert!(result.is_none(), "Different model should be a cache miss");
}

/// Duplicate store is idempotent (first write wins).
#[tokio::test]
async fn ocr_cache_duplicate_store_preserves_first() {
    let cache = InMemoryOcrCache::default();
    cache
        .store(Uuid::nil(), "abc123", "mistral-ocr-latest", "first", 1)
        .await
        .unwrap();
    cache
        .store(Uuid::nil(), "abc123", "mistral-ocr-latest", "second", 2)
        .await
        .unwrap();

    let result = cache
        .find_by_hash(Uuid::nil(), "abc123", "mistral-ocr-latest")
        .await
        .unwrap();
    assert_eq!(
        result,
        Some(("first".to_string(), 1)),
        "First store should win on duplicate"
    );
}

#[tokio::test]
async fn ocr_cache_is_scoped_and_bounded_per_source() {
    let cache = InMemoryOcrCache::default();
    let source_a = Uuid::new_v4();
    let source_b = Uuid::new_v4();

    cache
        .store(source_a, "hash-v1", "model-v1", "first", 1)
        .await
        .unwrap();
    assert!(
        cache
            .find_by_hash(source_b, "hash-v1", "model-v1")
            .await
            .unwrap()
            .is_none()
    );

    cache
        .store(source_a, "hash-v2", "model-v2", "replacement", 2)
        .await
        .unwrap();
    assert!(
        cache
            .find_by_hash(source_a, "hash-v1", "model-v1")
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        cache
            .find_by_hash(source_a, "hash-v2", "model-v2")
            .await
            .unwrap(),
        Some(("replacement".to_owned(), 2))
    );
}

#[tokio::test]
async fn duplicate_ocr_content_transfers_ownership_without_replacing_text() {
    let cache = InMemoryOcrCache::default();
    let source_a = Uuid::new_v4();
    let source_b = Uuid::new_v4();

    cache
        .store(source_a, "shared-hash", "model", "first result", 1)
        .await
        .unwrap();
    cache
        .store(source_b, "shared-hash", "model", "second result", 2)
        .await
        .unwrap();

    assert!(
        cache
            .find_by_hash(source_a, "shared-hash", "model")
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        cache
            .find_by_hash(source_b, "shared-hash", "model")
            .await
            .unwrap(),
        Some(("first result".to_owned(), 1))
    );
}

/// Helper: simulate the pipeline's cache-check-before-API pattern.
///
/// This is a test-only extraction of the logic in `process_source()`:
/// check the OCR cache first, and only call the OCR client on a cache miss.
async fn ocr_with_cache_lookup(
    cache: &dyn OcrCacheRepository,
    client: &MistralOcrClient,
    source_id: Uuid,
    pdf_bytes: &[u8],
    content_hash: &str,
    model: &str,
    pages: Option<Vec<u32>>,
) -> Result<(String, i32), openbooklm::error::AppError> {
    // Check cache first (mirrors source_processing.rs logic)
    if let Some((cached_text, pages_processed)) =
        cache.find_by_hash(source_id, content_hash, model).await?
    {
        return Ok((cached_text, pages_processed));
    }

    // Cache miss → call OCR client
    let result = client.extract_text_from_pdf(pdf_bytes, pages).await?;
    let text = result
        .pages
        .iter()
        .map(|p| p.markdown.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let pages_processed = result.pages_processed as i32;
    cache
        .store(source_id, content_hash, model, &text, pages_processed)
        .await?;
    Ok((text, pages_processed))
}

/// OCR cache hit → returns cached text, API not called.
///
/// Exercises the cache-check-before-API pattern via [`ocr_with_cache_lookup`]:
/// when the cache has a result for the content hash + model, the OCR client
/// is not invoked and the cached text is returned directly. The mock server's
/// `expect(0)` is a genuine guard — the helper function would call the client
/// on a cache miss.
#[tokio::test]
async fn ocr_cache_hit_skips_api_call() {
    let mock_server = MockServer::start().await;

    // Mock should NOT be called — assert with expect(0).
    // This is a genuine guard: `ocr_with_cache_lookup` would call the client
    // on a cache miss, triggering the mock and failing expect(0).
    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "pages": [{ "index": 0, "markdown": "from api" }],
            "usage_info": { "pages_processed": 1 }
        })))
        .expect(0)
        .mount(&mock_server)
        .await;

    let client = make_test_ocr_client(&mock_server.uri());
    let pdf_bytes = minimal_image_only_pdf();

    // Pre-populate cache.
    let cache = InMemoryOcrCache::default();
    let content_hash = "abc123hash";
    let model = "mistral-ocr-latest";
    let cached_text = "# Cached OCR result\n\nExtracted text from cache.";
    cache
        .store(Uuid::nil(), content_hash, model, cached_text, 3)
        .await
        .unwrap();

    // Call the cache-aware helper — should return cached text without calling the API.
    let (text, pages) = ocr_with_cache_lookup(
        &cache,
        &client,
        Uuid::nil(),
        &pdf_bytes,
        content_hash,
        model,
        None,
    )
    .await
    .expect("Cache lookup should succeed");

    assert_eq!(text, cached_text);
    assert_eq!(pages, 3);

    // MockServer's expect(0) fires on drop — if OCR client were called,
    // the mock would record a request and this test would fail.
}

/// OCR cache miss → API is called, result is cached for next lookup.
///
/// The inverse of `ocr_cache_hit_skips_api_call`: verifies that on a cache
/// miss, the OCR client is called exactly once and the result is stored.
#[tokio::test]
async fn ocr_cache_miss_calls_api_and_caches_result() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "pages": [{ "index": 0, "markdown": "# From API" }],
            "usage_info": { "pages_processed": 1 }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let client = make_test_ocr_client(&mock_server.uri());
    let pdf_bytes = minimal_image_only_pdf();
    let cache = InMemoryOcrCache::default();
    let content_hash = "new_hash_never_cached";
    let model = "mistral-ocr-latest";

    // First call: cache miss → API called.
    let (text, pages) = ocr_with_cache_lookup(
        &cache,
        &client,
        Uuid::nil(),
        &pdf_bytes,
        content_hash,
        model,
        None,
    )
    .await
    .expect("OCR should succeed on cache miss");

    assert!(text.contains("# From API"));
    assert_eq!(pages, 1);

    // Verify the result was stored in cache.
    let cached = cache
        .find_by_hash(Uuid::nil(), content_hash, model)
        .await
        .unwrap();
    assert!(cached.is_some(), "Result should be cached after API call");
    assert_eq!(cached.unwrap().0, text);
}

// ============================================================================
// 5. Text Merging Tests (native + OCR)
// ============================================================================

/// Mixed PDF: native text + OCR markdown merged by page index.
#[test]
fn merge_ocr_pages_replaces_sparse_pages() {
    // Per-page segments: pages 0,2,4 have text; pages 1,3 are sparse (empty).
    let segments: Vec<String> = vec![
        "Page 0 text".into(),
        String::new(),
        "Page 2 text".into(),
        String::new(),
        "Page 4 text".into(),
    ];

    let ocr_pages = vec![
        OcrPage {
            index: 1,
            markdown: "# OCR Page 1".to_string(),
        },
        OcrPage {
            index: 3,
            markdown: "# OCR Page 3".to_string(),
        },
    ];

    let merged = merge_ocr_pages(&segments, &ocr_pages);

    // Page identity survives the merge: index i of the result is page i + 1 of
    // the document, which is what a citation resolves against (US-019).
    assert_eq!(merged.len(), 5);
    assert_eq!(merged[0], "Page 0 text", "Native page 0 preserved");
    assert_eq!(merged[1], "# OCR Page 1", "OCR replaced sparse page 1");
    assert_eq!(merged[2], "Page 2 text", "Native page 2 preserved");
    assert_eq!(merged[3], "# OCR Page 3", "OCR replaced sparse page 3");
    assert_eq!(merged[4], "Page 4 text", "Native page 4 preserved");
}

/// Fully scanned PDF: all pages replaced by OCR.
#[test]
fn merge_ocr_pages_replaces_all_empty_pages() {
    let segments: Vec<String> = vec![String::new(); 3];

    let ocr_pages = vec![
        OcrPage {
            index: 0,
            markdown: "# Page 0".to_string(),
        },
        OcrPage {
            index: 1,
            markdown: "# Page 1".to_string(),
        },
        OcrPage {
            index: 2,
            markdown: "# Page 2".to_string(),
        },
    ];

    let merged = merge_ocr_pages(&segments, &ocr_pages);

    assert_eq!(merged, vec!["# Page 0", "# Page 1", "# Page 2"]);
}

/// OCR page index beyond segment count extends the segments vector.
#[test]
fn merge_ocr_pages_extends_for_out_of_range_pages() {
    let segments: Vec<String> = vec!["Page 0 text".into()];
    let ocr_pages = vec![OcrPage {
        index: 2,
        markdown: "# OCR Page 2".to_string(),
    }];

    let merged = merge_ocr_pages(&segments, &ocr_pages);
    assert_eq!(merged.len(), 3, "the gap keeps its page number");
    assert_eq!(merged[0], "Page 0 text", "Native text preserved");
    assert_eq!(merged[1], "", "the skipped page still exists");
    assert_eq!(
        merged[2], "# OCR Page 2",
        "Out-of-range page extended into segments"
    );
}

/// No OCR pages → native text unchanged.
#[test]
fn merge_ocr_pages_no_ocr_preserves_native() {
    let segments: Vec<String> = vec!["Page 0".into(), "Page 1".into()];
    assert_eq!(merge_ocr_pages(&segments, &[]), segments);
}

// ============================================================================
// 7. DB-backed OCR Cache Tests
// ============================================================================

async fn create_ocr_cache_source(db: &sea_orm::DatabaseConnection) -> (Uuid, Uuid) {
    use openbooklm_migration_core::core_track::{CoreMigrator, with_migration_lock};
    use openbooklm_migration_core::{MigratorTrait, validate_core_state};
    use sea_orm::{ConnectionTrait, DatabaseBackend, Statement};

    let state = validate_core_state(db)
        .await
        .expect("validate core migration state for OCR cache test");
    if let Some(remediation) = state.remediation() {
        panic!("unsafe OCR cache test database: {remediation}");
    }
    with_migration_lock(db, async || CoreMigrator::up(db, None).await)
        .await
        .expect("apply core migrations for OCR cache test");

    let account_id = Uuid::new_v4();
    let notebook_id = Uuid::new_v4();
    let source_id = Uuid::new_v4();
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO accounts (id) VALUES ($1)",
        [account_id.into()],
    ))
    .await
    .expect("create OCR test account");
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO notebooks (id, user_id, title) VALUES ($1, $2, 'OCR cache test')",
        [notebook_id.into(), account_id.into()],
    ))
    .await
    .expect("create OCR test notebook");
    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO sources (id, notebook_id, title, source_type, content) \
         VALUES ($1, $2, 'OCR cache test', 'pdf', '')",
        [source_id.into(), notebook_id.into()],
    ))
    .await
    .expect("create OCR test source");
    (account_id, source_id)
}

async fn delete_ocr_cache_fixture(db: &sea_orm::DatabaseConnection, account_id: Uuid) {
    use sea_orm::{ConnectionTrait, DatabaseBackend, Statement};

    db.execute(Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "DELETE FROM accounts WHERE id = $1",
        [account_id.into()],
    ))
    .await
    .expect("delete OCR test fixture");
}

/// Round-trip: store OCR result → find by hash → returns cached text.
///
/// Uses a real PostgreSQL database. Run with:
/// ```bash
/// TEST_DATABASE_URL=postgres://... cargo test -- --ignored
/// ```
#[tokio::test]
#[ignore = "Requires real PostgreSQL database (TEST_DATABASE_URL env var)"]
async fn ocr_cache_db_roundtrip() {
    use openbooklm::core::config::DatabasePoolConfig;
    use openbooklm::repositories::{OcrCacheRepository, SeaOrmOcrCacheRepository};

    let db_url =
        std::env::var("TEST_DATABASE_URL").expect("TEST_DATABASE_URL must be set for this test");
    let db = openbooklm::db::connect(&db_url, &DatabasePoolConfig::default())
        .await
        .expect("Failed to connect to test database");

    let repo = SeaOrmOcrCacheRepository::new(&db);
    let (account_id, source_id) = create_ocr_cache_source(&db).await;
    let hash = format!("test_hash_{}", Uuid::new_v4());
    let model = "mistral-ocr-latest";

    // Cache miss before storing.
    let miss = repo
        .find_by_hash(source_id, &hash, model)
        .await
        .expect("find_by_hash should not error");
    assert!(miss.is_none(), "Should be a cache miss before store");

    // Store.
    repo.store(
        source_id,
        &hash,
        model,
        "# Extracted markdown\n\nSome content.",
        5,
    )
    .await
    .expect("store should succeed");

    // Cache hit after storing.
    let hit = repo
        .find_by_hash(source_id, &hash, model)
        .await
        .expect("find_by_hash should not error");
    assert_eq!(
        hit,
        Some(("# Extracted markdown\n\nSome content.".to_string(), 5)),
        "Should return cached text and page count"
    );

    // Different model → miss.
    let diff_model = repo
        .find_by_hash(source_id, &hash, "mistral-ocr-v2")
        .await
        .expect("find_by_hash should not error");
    assert!(
        diff_model.is_none(),
        "Different model should be a cache miss"
    );
    delete_ocr_cache_fixture(&db, account_id).await;
}

/// Duplicate stores are idempotent and preserve the first OCR payload.
#[tokio::test]
#[ignore = "Requires real PostgreSQL database (TEST_DATABASE_URL env var)"]
async fn ocr_cache_db_duplicate_insert_idempotent() {
    use openbooklm::core::config::DatabasePoolConfig;
    use openbooklm::repositories::{OcrCacheRepository, SeaOrmOcrCacheRepository};

    let db_url =
        std::env::var("TEST_DATABASE_URL").expect("TEST_DATABASE_URL must be set for this test");
    let db = openbooklm::db::connect(&db_url, &DatabasePoolConfig::default())
        .await
        .expect("Failed to connect to test database");

    let repo = SeaOrmOcrCacheRepository::new(&db);
    let (account_id, source_id) = create_ocr_cache_source(&db).await;
    let hash = format!("dup_test_{}", Uuid::new_v4());
    let model = "mistral-ocr-latest";

    // First insert.
    repo.store(source_id, &hash, model, "first text", 1)
        .await
        .expect("first store should succeed");

    // A second store with the same key should not error or replace OCR output.
    repo.store(source_id, &hash, model, "second text", 2)
        .await
        .expect("duplicate store should not error");

    // First value should be preserved.
    let result = repo.find_by_hash(source_id, &hash, model).await.unwrap();
    assert_eq!(
        result,
        Some(("first text".to_string(), 1)),
        "First insert should win on conflict"
    );
    delete_ocr_cache_fixture(&db, account_id).await;
}

// ============================================================================
// 8. Error Propagation
// ============================================================================
//
// Note: these tests verify that API errors propagate correctly through the
// client. Circuit breaker state transitions (Closed → Open → HalfOpen) are
// not tested here because they require either exposing internal state or
// issuing 5+ retryable (5xx) failures with real/paused time, which would
// make the tests slow. The `ResilientExecutor` has its own unit tests for
// state machine correctness in `clients/resilience.rs`.

/// Mistral API returns error → client returns error with status info.
///
/// Uses 422 (non-retryable) to avoid retry backoff delays in tests.
/// Verifies that API failures propagate as source processing errors.
#[tokio::test]
async fn ocr_client_returns_error_on_server_failure() {
    let mock_server = MockServer::start().await;

    // 422 is non-retryable — the client fails immediately without backoff.
    Mock::given(method("POST"))
        .and(path("/v1/ocr"))
        .respond_with(
            ResponseTemplate::new(422).set_body_json(json!({ "error": "Unprocessable content" })),
        )
        .mount(&mock_server)
        .await;

    let client = make_test_ocr_client(&mock_server.uri());
    let pdf_bytes = minimal_image_only_pdf();
    let result = client.extract_text_from_pdf(&pdf_bytes, None).await;

    assert!(result.is_err(), "Should fail on API error");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("422") || err_msg.contains("OCR") || err_msg.contains("failed"),
        "Error should reference the failure: {err_msg}"
    );
}
