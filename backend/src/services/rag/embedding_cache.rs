//! In-memory embedding cache to avoid repeated provider calls.
//!
//! Wraps [`moka::future::Cache`] with a TTL of 30 minutes and a max capacity
//! of 2,000 entries (~8 MB for 1024-dim f32 vectors).
//!
//! ## The key is not the query (US-011)
//!
//! It used to be, and that was wrong in two ways. A HyDE-enabled search embeds a
//! *generated document* derived from the query and stored the result under the
//! query, so a later lookup that wanted the query's own embedding got the HyDE
//! document's. And nothing in the key named the model, so a provider change
//! served vectors from the previous model's space until the TTL expired.
//!
//! The key is therefore `(kind, model fingerprint, text)`:
//! [`QueryEmbeddingKind`] separates the four roles a text can play, and the
//! fingerprint separates vector spaces. Two entries that would collide under one
//! of those dimensions no longer share a slot.
//!
//! No invalidation logic is needed — a query embedding depends only on its key.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Duration;

use moka::future::Cache;

use crate::services::rag::provenance::QueryEmbeddingKind;

/// TTL for cached embeddings (30 minutes).
const CACHE_TTL: Duration = Duration::from_secs(30 * 60);

/// Maximum number of cached embedding vectors.
const CACHE_MAX_CAPACITY: u64 = 2_000;

/// In-memory cache for query embedding vectors.
///
/// Thread-safe (`Send + Sync`) and cheap to clone — designed to live inside
/// `CoreState` and be shared across Axum handlers.
#[derive(Clone)]
pub struct EmbeddingCache {
    inner: Cache<u64, Vec<f32>>,
}

impl EmbeddingCache {
    /// Create a new embedding cache with default TTL and capacity.
    #[must_use]
    pub fn new() -> Self {
        let inner = Cache::builder()
            .max_capacity(CACHE_MAX_CAPACITY)
            .time_to_live(CACHE_TTL)
            .build();
        Self { inner }
    }

    /// Look up a cached embedding.
    ///
    /// `fingerprint` is the embedding provenance fingerprint of the provider
    /// that would produce the vector; see
    /// [`EmbeddingProvenance::fingerprint`](crate::services::rag::provenance::EmbeddingProvenance::fingerprint).
    pub async fn get(
        &self,
        kind: QueryEmbeddingKind,
        fingerprint: &str,
        text: &str,
    ) -> Option<Vec<f32>> {
        self.inner.get(&cache_key(kind, fingerprint, text)).await
    }

    /// Insert an embedding vector under its role and vector space.
    pub async fn insert(
        &self,
        kind: QueryEmbeddingKind,
        fingerprint: &str,
        text: &str,
        embedding: Vec<f32>,
    ) {
        self.inner
            .insert(cache_key(kind, fingerprint, text), embedding)
            .await;
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash `(kind, fingerprint, text)` to a `u64` cache key using `DefaultHasher`
/// (SipHash).
///
/// All three components are hashed, so two entries can only collide by hash
/// collision, never by construction.
fn cache_key(kind: QueryEmbeddingKind, fingerprint: &str, text: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    kind.as_str().hash(&mut hasher);
    fingerprint.hash(&mut hasher);
    text.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const FP: &str = "emb:v1:test:model-a:1024:unit";
    const OTHER_FP: &str = "emb:v1:test:model-b:1024:unit";

    #[tokio::test]
    async fn insert_and_retrieve() {
        let cache = EmbeddingCache::new();
        let embedding = vec![0.1_f32, 0.2, 0.3];

        cache
            .insert(
                QueryEmbeddingKind::Direct,
                FP,
                "hello world",
                embedding.clone(),
            )
            .await;

        assert_eq!(
            cache
                .get(QueryEmbeddingKind::Direct, FP, "hello world")
                .await,
            Some(embedding)
        );
    }

    #[tokio::test]
    async fn miss_returns_none() {
        let cache = EmbeddingCache::new();
        assert_eq!(
            cache
                .get(QueryEmbeddingKind::Direct, FP, "not inserted")
                .await,
            None
        );
    }

    #[tokio::test]
    async fn different_queries_produce_different_keys() {
        let cache = EmbeddingCache::new();

        let emb_a = vec![1.0_f32, 0.0, 0.0];
        let emb_b = vec![0.0_f32, 1.0, 0.0];

        cache
            .insert(QueryEmbeddingKind::Direct, FP, "query alpha", emb_a.clone())
            .await;
        cache
            .insert(QueryEmbeddingKind::Direct, FP, "query beta", emb_b.clone())
            .await;

        assert_eq!(
            cache
                .get(QueryEmbeddingKind::Direct, FP, "query alpha")
                .await,
            Some(emb_a)
        );
        assert_eq!(
            cache
                .get(QueryEmbeddingKind::Direct, FP, "query beta")
                .await,
            Some(emb_b)
        );
    }

    /// A HyDE entry occupies its own slot: nothing is served across namespaces
    /// by the key itself.
    ///
    /// This is a statement about the key, not about the system. Dense retrieval
    /// deliberately asks the HyDE namespace first when HyDE is configured (see
    /// [`lookup_kinds`](crate::services::rag::search) — under HyDE the vector
    /// that path stores *is* the retrieval vector). What this test pins is that
    /// crossing a namespace has to be an explicit decision by a caller, and can
    /// never happen by construction: a lookup that asks for one kind and does
    /// not ask for another gets nothing from the other.
    #[tokio::test]
    async fn a_hyde_entry_occupies_its_own_slot() {
        let cache = EmbeddingCache::new();
        let hyde_vector = vec![0.9_f32, 0.1, 0.0];

        cache
            .insert(
                QueryEmbeddingKind::HydeDocument,
                FP,
                "what is rrf",
                hyde_vector.clone(),
            )
            .await;

        assert_eq!(
            cache
                .get(QueryEmbeddingKind::Direct, FP, "what is rrf")
                .await,
            None,
            "a direct lookup must not be answered by a HyDE-derived vector unless \
             its caller asked for the HyDE namespace too"
        );
        assert_eq!(
            cache
                .get(QueryEmbeddingKind::WorkingMemory, FP, "what is rrf")
                .await,
            None,
            "working memory never asks the HyDE namespace, so it must miss here"
        );
        assert_eq!(
            cache
                .get(QueryEmbeddingKind::HydeDocument, FP, "what is rrf")
                .await,
            Some(hyde_vector)
        );
    }

    #[tokio::test]
    async fn every_kind_has_its_own_slot() {
        let cache = EmbeddingCache::new();
        let kinds = [
            QueryEmbeddingKind::Direct,
            QueryEmbeddingKind::Reformulated,
            QueryEmbeddingKind::HydeDocument,
            QueryEmbeddingKind::WorkingMemory,
        ];

        for (i, kind) in kinds.into_iter().enumerate() {
            cache.insert(kind, FP, "shared text", vec![i as f32]).await;
        }
        for (i, kind) in kinds.into_iter().enumerate() {
            assert_eq!(
                cache.get(kind, FP, "shared text").await,
                Some(vec![i as f32]),
                "{kind:?} must keep its own entry"
            );
        }
    }

    #[tokio::test]
    async fn a_model_change_does_not_serve_the_previous_vector_space() {
        let cache = EmbeddingCache::new();
        let old_vector = vec![0.5_f32; 3];

        cache
            .insert(QueryEmbeddingKind::Direct, FP, "stable query", old_vector)
            .await;

        assert_eq!(
            cache
                .get(QueryEmbeddingKind::Direct, OTHER_FP, "stable query")
                .await,
            None,
            "a different model fingerprint must miss rather than reuse"
        );
    }

    #[test]
    fn key_is_deterministic() {
        assert_eq!(
            cache_key(QueryEmbeddingKind::Direct, FP, "same text"),
            cache_key(QueryEmbeddingKind::Direct, FP, "same text")
        );
    }

    #[test]
    fn key_differs_for_different_text() {
        assert_ne!(
            cache_key(QueryEmbeddingKind::Direct, FP, "text a"),
            cache_key(QueryEmbeddingKind::Direct, FP, "text b")
        );
    }
}
