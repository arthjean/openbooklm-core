//! In-memory embedding cache to avoid repeated Voyage AI calls.
//!
//! Wraps [`moka::future::Cache`] with a TTL of 30 minutes and a max capacity
//! of 2,000 entries (~8 MB for 1024-dim f32 vectors). Cache keys are `u64`
//! hashes of the query text; values are the embedding vectors.
//!
//! No invalidation logic is needed — query embeddings are independent of
//! notebook content (the same query always produces the same embedding).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Duration;

use moka::future::Cache;

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

    /// Look up a cached embedding for the given query text.
    pub async fn get(&self, query: &str) -> Option<Vec<f32>> {
        let key = hash_query(query);
        self.inner.get(&key).await
    }

    /// Insert an embedding vector for the given query text.
    pub async fn insert(&self, query: &str, embedding: Vec<f32>) {
        let key = hash_query(query);
        self.inner.insert(key, embedding).await;
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash a query string to a `u64` cache key using `DefaultHasher` (SipHash).
fn hash_query(query: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    query.hash(&mut hasher);
    hasher.finish()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn insert_and_retrieve() {
        let cache = EmbeddingCache::new();
        let embedding = vec![0.1_f32, 0.2, 0.3];

        cache.insert("hello world", embedding.clone()).await;
        let cached = cache.get("hello world").await;

        assert_eq!(cached, Some(embedding));
    }

    #[tokio::test]
    async fn miss_returns_none() {
        let cache = EmbeddingCache::new();
        assert_eq!(cache.get("not inserted").await, None);
    }

    #[tokio::test]
    async fn different_queries_produce_different_keys() {
        let cache = EmbeddingCache::new();

        let emb_a = vec![1.0_f32, 0.0, 0.0];
        let emb_b = vec![0.0_f32, 1.0, 0.0];

        cache.insert("query alpha", emb_a.clone()).await;
        cache.insert("query beta", emb_b.clone()).await;

        assert_eq!(cache.get("query alpha").await, Some(emb_a));
        assert_eq!(cache.get("query beta").await, Some(emb_b));
    }

    #[test]
    fn hash_is_deterministic() {
        let h1 = hash_query("same text");
        let h2 = hash_query("same text");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_differs_for_different_text() {
        let h1 = hash_query("text a");
        let h2 = hash_query("text b");
        assert_ne!(h1, h2);
    }
}
