//! Shared utility functions for RAG services.
//!
//! Contains helpers used across multiple RAG sub-modules (vector store,
//! search, etc.) to avoid code duplication.

use sha2::{Digest, Sha256};
use std::fmt::Write;

/// Format an embedding vector for pgvector SQL (e.g., `[0.1,0.2,0.3]`).
pub fn format_embedding(embedding: &[f32]) -> String {
    let mut out = String::with_capacity(embedding.len() * 12);
    out.push('[');
    for (i, v) in embedding.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{v}");
    }
    out.push(']');
    out
}

/// Sanitize a query string for use with PostgreSQL `plainto_tsquery`.
///
/// Keeps only alphanumeric characters and whitespace, then normalizes
/// whitespace to single spaces. Prevents SQL injection in full-text queries.
/// Truncate text to a maximum number of characters, appending `"..."` if truncated.
///
/// Character-safe: uses `char_indices` to avoid splitting multi-byte characters.
pub fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let end = text
        .char_indices()
        .nth(max_chars.saturating_sub(3))
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    format!("{}...", &text[..end])
}

/// Sanitize a query string for use with PostgreSQL `plainto_tsquery`.
pub fn sanitize_tsquery(query: &str) -> String {
    query
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compute the SHA-256 hex digest of raw bytes.
///
/// Used for OCR cache keying (hash raw PDF bytes) and chunk deduplication
/// (via [`compute_content_hash`]).
pub fn compute_bytes_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Compute the SHA-256 hex digest of a chunk's content for deduplication.
pub fn compute_content_hash(content: &str) -> String {
    compute_bytes_hash(content.as_bytes())
}

/// Parse a pgvector text representation (e.g., `[0.1,0.2,0.3]`) back to `Vec<f32>`.
///
/// This is the inverse of [`format_embedding`].
pub fn parse_embedding(text: &str) -> Vec<f32> {
    let trimmed = text.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_embedding_basic() {
        let embedding = vec![0.1, 0.2, 0.3];
        let result = format_embedding(&embedding);
        assert_eq!(result, "[0.1,0.2,0.3]");
    }

    #[test]
    fn format_embedding_empty() {
        assert_eq!(format_embedding(&[]), "[]");
    }

    #[test]
    fn format_embedding_single() {
        assert_eq!(format_embedding(&[1.5]), "[1.5]");
    }

    #[test]
    fn sanitize_tsquery_basic() {
        assert_eq!(sanitize_tsquery("hello world"), "hello world");
    }

    #[test]
    fn sanitize_tsquery_removes_special_chars() {
        assert_eq!(sanitize_tsquery("hello; DROP TABLE--"), "hello DROP TABLE");
    }

    #[test]
    fn sanitize_tsquery_normalizes_whitespace() {
        assert_eq!(sanitize_tsquery("  hello   world  "), "hello world");
    }

    #[test]
    fn sanitize_tsquery_empty() {
        assert_eq!(sanitize_tsquery(""), "");
        assert_eq!(sanitize_tsquery("   "), "");
    }

    #[test]
    fn sanitize_tsquery_special_only() {
        assert_eq!(sanitize_tsquery("!@#$%"), "");
    }

    #[test]
    fn truncate_text_short() {
        assert_eq!(truncate_text("hello", 10), "hello");
    }

    #[test]
    fn truncate_text_exact() {
        assert_eq!(truncate_text("hello", 5), "hello");
    }

    #[test]
    fn truncate_text_long() {
        let long = "a".repeat(200);
        let result = truncate_text(&long, 50);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 53);
    }

    #[test]
    fn truncate_text_multibyte() {
        let text = "é".repeat(10); // 2-byte chars
        let result = truncate_text(&text, 5);
        assert!(result.ends_with("..."));
        assert_eq!(result.chars().count(), 5); // 2 chars + "..."
    }

    #[test]
    fn compute_content_hash_consistent() {
        let hash1 = compute_content_hash("hello world");
        let hash2 = compute_content_hash("hello world");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn compute_content_hash_different_input() {
        let hash1 = compute_content_hash("hello world");
        let hash2 = compute_content_hash("hello world!");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn compute_bytes_hash_consistent() {
        let bytes = b"some pdf bytes";
        let hash1 = compute_bytes_hash(bytes);
        let hash2 = compute_bytes_hash(bytes);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64);
    }

    #[test]
    fn compute_bytes_hash_matches_content_hash_for_utf8() {
        // For pure UTF-8 text, bytes hash of .as_bytes() should equal content hash.
        let text = "hello world";
        assert_eq!(
            compute_bytes_hash(text.as_bytes()),
            compute_content_hash(text)
        );
    }

    #[test]
    fn parse_embedding_round_trip() {
        let original = vec![0.1, 0.2, 0.3, -0.5];
        let text = format_embedding(&original);
        let parsed = parse_embedding(&text);
        assert_eq!(original.len(), parsed.len());
        for (a, b) in original.iter().zip(parsed.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {a}, got {b}");
        }
    }

    #[test]
    fn parse_embedding_empty() {
        assert!(parse_embedding("[]").is_empty());
    }

    #[test]
    fn parse_embedding_single() {
        assert_eq!(parse_embedding("[1.5]"), vec![1.5]);
    }
}
