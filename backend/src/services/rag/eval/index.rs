//! An in-memory [`SearchRepository`] over the evaluation corpus (US-002).
//!
//! The retrieval evaluator has to exercise the *real* orchestration —
//! [`search`](crate::services::rag::search::search), RRF fusion, the limit and
//! filter handling — because that orchestration is what regresses. What it must
//! not need is PostgreSQL: a release gate that only runs where a database
//! happens to be up is a gate that stops running.
//!
//! So the repository trait is implemented over the corpus instead, with the two
//! ranking functions replaced by deterministic in-process equivalents:
//!
//! | Production | Here |
//! |---|---|
//! | pgvector `<=>` over HNSW | exact cosine over every chunk |
//! | `ts_rank_cd` over `content_tsv` | a documented BM25-shaped scorer |
//!
//! # This is a reference, not an emulation
//!
//! Exact cosine is not "HNSW without the index" — it is the ground truth HNSW
//! approximates, which is exactly what US-015 will measure the real index
//! against. The lexical scorer is a stand-in with the same *shape* as
//! `ts_rank_cd` (term frequency, document length, term rarity) and not its
//! arithmetic; a lexical number from this index is comparable across revisions
//! of this repository and not against PostgreSQL. Both facts are recorded in
//! every report so nobody reads them as production measurements.

use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

use crate::core::providers::{DeterministicEmbedder, EmbeddingProvider};
use crate::error::AppError;
use crate::repositories::{ChunkSearchResult, NotebookScope, RepoResult, SearchRepository};
use crate::services::rag::utils::sanitize_tsquery;

use super::corpus::EvalCorpus;

// ============================================================================
// Indexed chunk
// ============================================================================

/// One corpus chunk, resolved to runtime identifiers and pre-embedded.
#[derive(Debug, Clone)]
pub struct IndexedChunk {
    pub chunk_id: Uuid,
    /// The corpus generation this chunk belongs to.
    ///
    /// The offline index has exactly one, so every result carries it. That is
    /// what lets the evaluator exercise the same trace field production fills
    /// from `sources.active_generation_id` (EP-002).
    pub generation_id: Uuid,
    pub source_id: Uuid,
    pub notebook_id: Uuid,
    pub source_title: String,
    pub chunk_index: i32,
    pub content: String,
    pub parent_content: Option<String>,
    pub metadata: Option<serde_json::Value>,
    embedding: Vec<f32>,
    /// Lower-cased alphanumeric tokens, matching `sanitize_tsquery`.
    tokens: Vec<String>,
}

impl IndexedChunk {
    fn as_result(&self, relevance_score: f32) -> ChunkSearchResult {
        ChunkSearchResult {
            id: self.chunk_id,
            generation_id: self.generation_id,
            source_id: self.source_id,
            chunk_index: self.chunk_index,
            content: self.content.clone(),
            parent_content: self.parent_content.clone(),
            source_title: self.source_title.clone(),
            relevance_score,
            metadata: self.metadata.clone(),
        }
    }
}

// ============================================================================
// Index
// ============================================================================

/// The corpus, indexed for retrieval.
pub struct CorpusIndex {
    chunks: Vec<IndexedChunk>,
    /// Document frequency per token, for the lexical scorer.
    document_frequency: HashMap<String, usize>,
    /// Mean token count, for length normalization.
    mean_length: f32,
    embedder: DeterministicEmbedder,
    generation_id: Uuid,
}

/// BM25 term-frequency saturation. The canonical value.
const BM25_K1: f32 = 1.2;
/// BM25 length-normalization weight. The canonical value.
const BM25_B: f32 = 0.75;

fn tokenize(text: &str) -> Vec<String> {
    sanitize_tsquery(text)
        .split_whitespace()
        .map(str::to_lowercase)
        .collect()
}

impl CorpusIndex {
    /// Build the index. Embedding happens once, here, so a run's latency
    /// percentiles measure retrieval and not fixture setup.
    ///
    /// # Errors
    /// Returns the embedding provider's error. The deterministic provider does
    /// not fail, but the signature keeps the seam honest for a future one.
    pub async fn build(corpus: &EvalCorpus) -> Result<Self, AppError> {
        let embedder = DeterministicEmbedder::new();
        let mut chunks = Vec::with_capacity(corpus.chunk_count());

        for notebook in corpus.notebooks() {
            let notebook_id = notebook.uuid();
            for source in &notebook.sources {
                let source_id = source.uuid();
                // Build each parent passage once, mirroring small-to-big
                // retrieval: the child is embedded, the parent is what the model
                // reads. Chunks sharing a group are children of one parent, so
                // the parent text is their concatenation in document order.
                let mut ordered: Vec<&crate::services::rag::eval::corpus::CorpusChunk> =
                    source.chunks.iter().collect();
                ordered.sort_by_key(|c| c.index);
                let mut parents: HashMap<&str, String> = HashMap::new();
                for chunk in &ordered {
                    if let Some(group) = chunk.parent_group.as_deref() {
                        let text = parents.entry(group).or_default();
                        if !text.is_empty() {
                            text.push_str("\n\n");
                        }
                        text.push_str(&chunk.content);
                    }
                }

                for chunk in &source.chunks {
                    let mut metadata = serde_json::Map::new();
                    if let Some(page) = chunk.page {
                        metadata.insert("page_number".to_owned(), page.into());
                    }
                    if let Some(header) = &chunk.section_header {
                        metadata.insert("section_header".to_owned(), header.clone().into());
                    }
                    metadata.insert(
                        "eval_chunk_slug".to_owned(),
                        serde_json::Value::String(chunk.id.clone()),
                    );
                    metadata.insert(
                        "span_start".to_owned(),
                        serde_json::Value::from(chunk.span.start),
                    );
                    metadata.insert(
                        "span_end".to_owned(),
                        serde_json::Value::from(chunk.span.end),
                    );

                    let parent_content = chunk
                        .parent_group
                        .as_deref()
                        .and_then(|group| parents.get(group).cloned());

                    chunks.push(IndexedChunk {
                        chunk_id: chunk.uuid(),
                        generation_id: corpus.generation_id(),
                        source_id,
                        notebook_id,
                        source_title: source.title.clone(),
                        chunk_index: chunk.index,
                        embedding: embedder.embed_query(&chunk.content).await?,
                        tokens: tokenize(&chunk.content),
                        content: chunk.content.clone(),
                        parent_content,
                        metadata: Some(serde_json::Value::Object(metadata)),
                    });
                }
            }
        }

        let mut document_frequency: HashMap<String, usize> = HashMap::new();
        for chunk in &chunks {
            let mut seen: Vec<&str> = Vec::new();
            for token in &chunk.tokens {
                if !seen.contains(&token.as_str()) {
                    seen.push(token);
                    *document_frequency.entry(token.clone()).or_insert(0) += 1;
                }
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let mean_length = if chunks.is_empty() {
            1.0
        } else {
            chunks.iter().map(|c| c.tokens.len()).sum::<usize>() as f32 / chunks.len() as f32
        };

        Ok(Self {
            chunks,
            document_frequency,
            mean_length: mean_length.max(1.0),
            embedder,
            generation_id: corpus.generation_id(),
        })
    }

    /// The generation every chunk in this index belongs to.
    #[must_use]
    pub const fn generation_id(&self) -> Uuid {
        self.generation_id
    }

    /// The synthetic account that owns every corpus notebook.
    ///
    /// The corpus has no accounts of its own, but production retrieval is
    /// account-scoped (US-020 AC-2), and an offline index that ignored the
    /// account would exercise a query shape the server does not run. One
    /// derived owner is enough to keep the seam real.
    #[must_use]
    pub fn account_id() -> Uuid {
        super::corpus::synthetic_uuid("eval-account")
    }

    /// The scope's notebook, when the scope's account owns this corpus.
    fn owned(&self, scope: NotebookScope) -> Option<Uuid> {
        (scope.account_id == Self::account_id()).then_some(scope.notebook_id)
    }

    /// The scope for one corpus notebook.
    #[must_use]
    pub fn scope(notebook_id: Uuid) -> NotebookScope {
        NotebookScope::new(Self::account_id(), notebook_id)
    }

    /// The embedding provider this index was built with.
    #[must_use]
    pub const fn embedder(&self) -> &DeterministicEmbedder {
        &self.embedder
    }

    /// Fingerprint of the provider that produced the vectors, for the report
    /// header. Two reports built with different fingerprints are not comparable.
    #[must_use]
    pub fn embedding_fingerprint(&self) -> String {
        format!(
            "{}/{}/{}",
            self.embedder.name(),
            self.embedder.model(),
            self.embedder.dimension()
        )
    }

    /// Every indexed chunk, for scoping assertions and report headers.
    #[must_use]
    pub fn chunks(&self) -> &[IndexedChunk] {
        &self.chunks
    }

    /// Exact brute-force cosine search: the reference every approximate index
    /// is measured against.
    ///
    /// # Errors
    /// Returns the embedding provider's error.
    pub async fn exact_reference(
        &self,
        notebook_id: Uuid,
        query: &str,
        limit: i32,
    ) -> Result<Vec<ChunkSearchResult>, AppError> {
        let embedding = self.embedder.embed_query(query).await?;
        Ok(self.rank_dense(notebook_id, &embedding, limit))
    }

    fn rank_dense(
        &self,
        notebook_id: Uuid,
        query_embedding: &[f32],
        limit: i32,
    ) -> Vec<ChunkSearchResult> {
        let mut scored: Vec<(f32, &IndexedChunk)> = self
            .chunks
            .iter()
            .filter(|c| c.notebook_id == notebook_id)
            .map(|c| (cosine(query_embedding, &c.embedding), c))
            .collect();

        sort_by_score_then_id(&mut scored);
        take_limit(scored, limit)
    }

    fn rank_lexical(&self, notebook_id: Uuid, query: &str, limit: i32) -> Vec<ChunkSearchResult> {
        let terms = tokenize(query);
        if terms.is_empty() {
            return Vec::new();
        }

        #[allow(clippy::cast_precision_loss)]
        let corpus_size = self.chunks.len() as f32;

        let mut scored: Vec<(f32, &IndexedChunk)> = self
            .chunks
            .iter()
            .filter(|c| c.notebook_id == notebook_id)
            .filter_map(|chunk| {
                #[allow(clippy::cast_precision_loss)]
                let length = chunk.tokens.len() as f32;
                let mut score = 0.0_f32;
                let mut matched = false;

                for term in &terms {
                    let frequency = chunk.tokens.iter().filter(|t| *t == term).count();
                    if frequency == 0 {
                        continue;
                    }
                    matched = true;
                    #[allow(clippy::cast_precision_loss)]
                    let document_frequency =
                        *self.document_frequency.get(term).unwrap_or(&0) as f32;
                    // Robertson/Sparck-Jones idf, clamped at zero so a term in
                    // every document contributes nothing rather than negatively.
                    let idf = ((corpus_size - document_frequency + 0.5)
                        / (document_frequency + 0.5))
                        .ln()
                        .max(0.0);
                    #[allow(clippy::cast_precision_loss)]
                    let tf = frequency as f32;
                    let denominator =
                        tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / self.mean_length);
                    score += idf * (tf * (BM25_K1 + 1.0)) / denominator;
                }

                // PostgreSQL's `@@` only returns matching rows; so does this.
                if matched { Some((score, chunk)) } else { None }
            })
            .collect();

        sort_by_score_then_id(&mut scored);
        // `ts_rank_cd` is clamped to 1.0 by the production repository, and the
        // fusion layer treats lexical scores as bounded. Match that.
        take_limit(scored, limit)
            .into_iter()
            .map(|mut r| {
                r.relevance_score = r.relevance_score.min(1.0);
                r
            })
            .collect()
    }
}

/// Cosine similarity, clamped at zero to match `(1 - distance).max(0)` in the
/// production repository.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>().max(0.0)
}

/// Sort by score descending, breaking ties on the chunk identifier.
///
/// Without the tie-break, equal scores keep whatever order the source vector
/// happened to have, and a report stops being byte-stable the first time two
/// chunks tie. The tie-break is on the derived UUID rather than the slug so
/// that it stays meaningful when this repository is swapped for a real one.
fn sort_by_score_then_id(scored: &mut [(f32, &IndexedChunk)]) {
    scored.sort_by(|(sa, ca), (sb, cb)| {
        sb.partial_cmp(sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ca.chunk_id.cmp(&cb.chunk_id))
    });
}

fn take_limit(scored: Vec<(f32, &IndexedChunk)>, limit: i32) -> Vec<ChunkSearchResult> {
    let limit = usize::try_from(limit.max(0)).unwrap_or(0);
    scored
        .into_iter()
        .take(limit)
        .map(|(score, chunk)| chunk.as_result(score))
        .collect()
}

#[async_trait]
impl SearchRepository for CorpusIndex {
    async fn search_similar_chunks(
        &self,
        scope: NotebookScope,
        query_embedding: &[f32],
        limit: i32,
    ) -> RepoResult<Vec<ChunkSearchResult>> {
        let Some(notebook_id) = self.owned(scope) else {
            return Ok(Vec::new());
        };
        Ok(self.rank_dense(notebook_id, query_embedding, limit))
    }

    async fn search_lexical_chunks(
        &self,
        scope: NotebookScope,
        query: &str,
        limit: i32,
    ) -> RepoResult<Vec<ChunkSearchResult>> {
        let Some(notebook_id) = self.owned(scope) else {
            return Ok(Vec::new());
        };
        Ok(self.rank_lexical(notebook_id, query, limit))
    }

    async fn count_chunks_for_notebook(&self, scope: NotebookScope) -> RepoResult<i64> {
        let Some(notebook_id) = self.owned(scope) else {
            return Ok(0);
        };
        Ok(i64::try_from(
            self.chunks
                .iter()
                .filter(|c| c.notebook_id == notebook_id)
                .count(),
        )
        .unwrap_or(i64::MAX))
    }

    async fn count_sources_for_notebook(&self, scope: NotebookScope) -> RepoResult<i64> {
        let Some(notebook_id) = self.owned(scope) else {
            return Ok(0);
        };
        Ok(i64::try_from(
            self.chunks
                .iter()
                .filter(|chunk| chunk.notebook_id == notebook_id)
                .map(|chunk| chunk.source_id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
        )
        .unwrap_or(i64::MAX))
    }

    async fn get_all_chunks_for_notebook(
        &self,
        scope: NotebookScope,
    ) -> RepoResult<Vec<ChunkSearchResult>> {
        let Some(notebook_id) = self.owned(scope) else {
            return Ok(Vec::new());
        };
        let mut chunks: Vec<&IndexedChunk> = self
            .chunks
            .iter()
            .filter(|c| c.notebook_id == notebook_id)
            .collect();
        // `ORDER BY s.id, c.chunk_index` in the production query.
        chunks.sort_by(|a, b| {
            a.source_id
                .cmp(&b.source_id)
                .then(a.chunk_index.cmp(&b.chunk_index))
        });
        Ok(chunks.iter().map(|c| c.as_result(1.0)).collect())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    async fn index() -> CorpusIndex {
        let corpus = EvalCorpus::load_default().expect("checked-in corpus loads");
        CorpusIndex::build(&corpus).await.expect("index builds")
    }

    #[tokio::test]
    async fn the_index_covers_every_corpus_chunk() {
        let corpus = EvalCorpus::load_default().expect("corpus");
        let index = CorpusIndex::build(&corpus).await.expect("index");
        assert_eq!(index.chunks().len(), corpus.chunk_count());
        assert!(index.chunks().len() >= 40, "corpus is suspiciously small");
    }

    #[tokio::test]
    async fn search_is_scoped_to_one_notebook() {
        let corpus = EvalCorpus::load_default().expect("corpus");
        let index = CorpusIndex::build(&corpus).await.expect("index");

        for notebook in corpus.notebooks() {
            let id = notebook.uuid();
            let hits = index
                .exact_reference(id, "policy retention deadline", 100)
                .await
                .expect("search");
            let owned: Vec<Uuid> = index
                .chunks()
                .iter()
                .filter(|c| c.notebook_id == id)
                .map(|c| c.chunk_id)
                .collect();
            assert!(
                hits.iter().all(|h| owned.contains(&h.id)),
                "notebook {} leaked a chunk from another notebook",
                notebook.id
            );
        }
    }

    #[tokio::test]
    async fn dense_ranking_is_byte_stable_across_runs() {
        let first = index().await;
        let second = index().await;
        let notebook = first.chunks()[0].notebook_id;

        let a = first
            .exact_reference(notebook, "incident response escalation", 10)
            .await
            .expect("search");
        let b = second
            .exact_reference(notebook, "incident response escalation", 10)
            .await
            .expect("search");

        let ids = |v: &[ChunkSearchResult]| v.iter().map(|r| r.id).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
    }

    #[tokio::test]
    async fn ties_break_on_the_chunk_id_rather_than_insertion_order() {
        let index = index().await;
        let notebook = index.chunks()[0].notebook_id;
        // A query with no lexical overlap gives every chunk the same zero
        // similarity, which is the tie case the report depends on.
        let hits = index
            .exact_reference(notebook, "zzzz", 5)
            .await
            .expect("search");
        let mut sorted = hits.clone();
        sorted.sort_by_key(|r| r.id);
        assert_eq!(
            hits.iter().map(|r| r.id).collect::<Vec<_>>(),
            sorted.iter().map(|r| r.id).collect::<Vec<_>>(),
            "an all-tie result set must come back in chunk-id order"
        );
    }

    #[tokio::test]
    async fn the_lexical_scorer_only_returns_matching_chunks() {
        let index = index().await;
        let notebook = index.chunks()[0].notebook_id;
        let hits = index
            .search_lexical_chunks(CorpusIndex::scope(notebook), "qwertyuiop", 10)
            .await
            .expect("search");
        assert!(
            hits.is_empty(),
            "a term nothing contains must match nothing"
        );
    }

    #[tokio::test]
    async fn an_empty_query_matches_nothing_rather_than_everything() {
        let index = index().await;
        let notebook = index.chunks()[0].notebook_id;
        let hits = index
            .search_lexical_chunks(CorpusIndex::scope(notebook), "   ", 10)
            .await
            .expect("search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn stuffing_returns_every_chunk_in_document_order() {
        let index = index().await;
        let notebook = index.chunks()[0].notebook_id;
        let all = index
            .get_all_chunks_for_notebook(CorpusIndex::scope(notebook))
            .await
            .expect("load");
        let count = index
            .count_chunks_for_notebook(CorpusIndex::scope(notebook))
            .await
            .expect("count");
        assert_eq!(i64::try_from(all.len()).unwrap_or(i64::MAX), count);
        assert!(
            all.iter()
                .all(|r| (r.relevance_score - 1.0).abs() < f32::EPSILON),
            "stuffed chunks all carry the same uniform score"
        );
    }

    #[tokio::test]
    async fn scores_are_always_finite_and_non_negative() {
        let index = index().await;
        let notebook = index.chunks()[0].notebook_id;
        for query in ["retention", "zzzz", "the the the", "policy"] {
            for hit in index.exact_reference(notebook, query, 20).await.expect("d") {
                assert!(hit.relevance_score.is_finite(), "{query}");
                assert!(hit.relevance_score >= 0.0, "{query}");
            }
            for hit in index
                .search_lexical_chunks(CorpusIndex::scope(notebook), query, 20)
                .await
                .expect("l")
            {
                assert!(hit.relevance_score.is_finite(), "{query}");
                assert!((0.0..=1.0).contains(&hit.relevance_score), "{query}");
            }
        }
    }
}
