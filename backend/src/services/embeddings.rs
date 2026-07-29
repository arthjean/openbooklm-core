//! Embeddings service for RAG.
//!
//! A thin API over the injected [`EmbeddingProvider`] seam, so source
//! processing and search embed text without knowing which provider is
//! installed (US-020). The hosted client's batching, retry and circuit breaker
//! live behind the trait.

use crate::core::providers::EmbeddingProvider;
use crate::error::AppError;

/// Embedding dimension. Defined by the schema, re-exported here for the call
/// sites that predate the seam.
pub use crate::core::providers::EMBEDDING_DIM;

/// Generate embeddings for documents (for storage).
pub async fn embed_documents(
    embeddings: &dyn EmbeddingProvider,
    texts: &[String],
) -> Result<Vec<Vec<f32>>, AppError> {
    embeddings.embed_documents(texts).await
}

/// Generate embedding for a query (for search).
pub async fn embed_query(
    embeddings: &dyn EmbeddingProvider,
    text: &str,
) -> Result<Vec<f32>, AppError> {
    embeddings.embed_query(text).await
}

/// Build the text to embed for a chunk, prepending context_prefix if available.
///
/// If the chunk has a context prefix (from Contextual Retrieval), the embedded
/// text is `"{context_prefix}\n\n{chunk_content}"`. Otherwise, just the content.
///
/// This ensures the embedding vector captures the contextualized meaning.
#[must_use]
pub fn contextualized_text(content: &str, context_prefix: Option<&str>) -> String {
    match context_prefix {
        Some(prefix) if !prefix.is_empty() => format!("{prefix}\n\n{content}"),
        _ => content.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextualized_text_with_prefix() {
        let text = contextualized_text(
            "Revenue grew 3% in Q3.",
            Some("This chunk is from Tesla's 2024 annual report, discussing Q3 financial results."),
        );
        assert!(text.starts_with("This chunk is from Tesla"));
        assert!(text.contains("\n\n"));
        assert!(text.ends_with("Revenue grew 3% in Q3."));
    }

    #[test]
    fn contextualized_text_without_prefix() {
        let text = contextualized_text("Revenue grew 3% in Q3.", None);
        assert_eq!(text, "Revenue grew 3% in Q3.");
    }

    #[test]
    fn contextualized_text_with_empty_prefix() {
        let text = contextualized_text("Revenue grew 3% in Q3.", Some(""));
        assert_eq!(text, "Revenue grew 3% in Q3.");
    }
}
