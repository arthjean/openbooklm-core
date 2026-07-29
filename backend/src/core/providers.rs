//! Embedding and reranking seams (US-020, EP-004).
//!
//! EP-002 inverted three dependencies: identity, entitlements and side effects.
//! It stopped before providers, so `CoreState` still named a commercial vendor
//! by type and retrieval could not run without that vendor's key. This module
//! finishes the inversion.
//!
//! # Two traits, not one
//!
//! `VoyageClient` does both jobs, which made one combined trait the obvious
//! shape. It is the wrong one: the two capabilities have different availability
//! profiles. An installation can embed without reranking — the pipeline already
//! skips reranking below `rerank_min_chunks` and falls back to unreranked
//! results when it fails — and a local embedding model with no cross-encoder is
//! exactly the configuration the self-hosting story wants to allow. Merging them
//! would reproduce, one layer down, the coupling this story removes.
//!
//! # Dimension is part of the contract
//!
//! `chunks.embedding` is `vector(1024)`. Nothing used to check that a provider
//! agreed. A provider returning a different width would either fail deep inside
//! a pgvector insert or, worse, index a chunk that no query can ever match.
//! [`EmbeddingProvider::dimension`] makes the width part of the interface, and
//! the composition roots refuse to start on a mismatch.
//!
//! # The deterministic providers
//!
//! [`DeterministicEmbedder`] and [`DeterministicLlm`] are in the shipped crate,
//! not behind `#[cfg(test)]`, because their two consumers are the test suite and
//! the CI smoke path — and a smoke path that exercises a different binary than
//! the one being released proves less than it appears to. They are selected by
//! one explicit environment variable and never by inference.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::AppError;

/// Width of a stored embedding vector.
///
/// Pinned by the core schema (`chunks.embedding vector(1024)`) and by the HNSW
/// index built over it. Changing it is a migration, not a configuration change.
pub const EMBEDDING_DIM: usize = 1024;

// ============================================================================
// Embedding
// ============================================================================

/// Turns text into vectors the retrieval pipeline can store and search.
///
/// Object-safe on purpose: `CoreState` holds one behind an `Arc<dyn _>` so a
/// composition can supply a hosted client, a local model or a deterministic
/// fixture without the pipeline knowing which.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Short stable identifier, for logs and health output.
    fn name(&self) -> &str;

    /// Model identifier. Recorded in RAG logs, so it must not change silently.
    fn model(&self) -> &str;

    /// Width of the vectors this provider returns.
    ///
    /// Checked against [`EMBEDDING_DIM`] at startup: a provider that disagrees
    /// with the schema cannot serve.
    fn dimension(&self) -> usize;

    /// Embed documents for storage. Order of the output matches the input.
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError>;

    /// Embed a query for search.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, AppError>;

    /// Texts the provider wants per request.
    ///
    /// Ingestion pipelines embedding with storage, so it drives the batching
    /// itself rather than handing the provider the whole corpus. It needs the
    /// provider's preferred shape to do that without either flooding a hosted
    /// API or serialising an in-process one.
    fn batch_size(&self) -> usize {
        1
    }

    /// Batches the provider will accept in flight at once.
    fn concurrency(&self) -> usize {
        1
    }

    /// Embed one batch, already sized by [`Self::batch_size`].
    ///
    /// Distinct from [`Self::embed_documents`] because a provider that rate
    /// limits must account for *this* request, not for a corpus it never sees.
    /// The default is the obvious one for a provider with no such concern.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        self.embed_documents(texts).await
    }
}

/// Shared handle to the configured embedding provider.
pub type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider>;

// ============================================================================
// Reranking
// ============================================================================

/// One reranked document: its position in the input, and its relevance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankedDocument {
    /// Index into the documents passed to [`Reranker::rerank`].
    pub index: usize,
    /// Relevance score. Higher is more relevant; the scale is provider-defined.
    pub relevance_score: f32,
}

/// Reorders retrieved candidates by relevance to the query.
///
/// Optional by construction. Retrieval already degrades to unreranked results
/// when this is absent or fails, so an installation without a cross-encoder is
/// a supported configuration rather than a broken one.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Short stable identifier, for logs and health output.
    fn name(&self) -> &str;

    /// Score `documents` against `query`, returning at most `top_k`.
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: Option<usize>,
    ) -> Result<Vec<RerankedDocument>, AppError>;
}

/// Shared handle to the configured reranker.
pub type SharedReranker = Arc<dyn Reranker>;

// ============================================================================
// Startup validation
// ============================================================================

/// Why a provider cannot serve this schema.
#[derive(Debug, PartialEq, Eq)]
pub struct DimensionMismatch {
    pub provider: String,
    pub model: String,
    pub provider_dimension: usize,
    pub schema_dimension: usize,
}

impl std::fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "embedding provider `{}` (model `{}`) returns {}-dimension vectors, but the schema \
             stores {}. Configure a model of the right width, or migrate the schema — an \
             embedding of the wrong width cannot be indexed and would never match a query.",
            self.provider, self.model, self.provider_dimension, self.schema_dimension
        )
    }
}

impl std::error::Error for DimensionMismatch {}

/// Check a provider against the schema's vector width.
///
/// # Errors
/// Returns [`DimensionMismatch`] when the two disagree.
pub fn check_dimension(provider: &dyn EmbeddingProvider) -> Result<(), DimensionMismatch> {
    if provider.dimension() == EMBEDDING_DIM {
        return Ok(());
    }
    Err(DimensionMismatch {
        provider: provider.name().to_owned(),
        model: provider.model().to_owned(),
        provider_dimension: provider.dimension(),
        schema_dimension: EMBEDDING_DIM,
    })
}

/// Check the vectors a provider actually returned, before they are stored.
///
/// [`check_dimension`] trusts what a provider *declares* at startup. This
/// checks what it *returned*, because the two can disagree: a vendor changing a
/// model's width, a misconfigured model name, or a truncated response all
/// produce vectors the declaration does not describe. Without this the batch
/// fails inside a pgvector insert as an opaque database error, which names
/// neither the provider nor the width.
///
/// # Errors
/// Returns [`RagError::EmbeddingFailed`](crate::error::RagError::EmbeddingFailed)
/// naming the provider, the offending index and both widths.
pub fn check_batch_widths(
    provider: &dyn EmbeddingProvider,
    vectors: &[Vec<f32>],
) -> Result<(), AppError> {
    let expected = provider.dimension();
    for (index, vector) in vectors.iter().enumerate() {
        if vector.len() != expected {
            return Err(crate::error::RagError::EmbeddingFailed {
                reason: format!(
                    "provider `{}` returned a {}-dimension vector at batch index {index}, \
                     expected {expected}",
                    provider.name(),
                    vector.len()
                ),
            }
            .into());
        }
    }
    Ok(())
}

// ============================================================================
// Deterministic providers
// ============================================================================

/// Environment variable that installs the deterministic providers.
///
/// Explicit and never inferred: a server that silently fell back to fake
/// embeddings would index a corpus that looks fine and answers nothing.
pub const DETERMINISTIC_ENV: &str = "OPENBOOKLM_DETERMINISTIC_PROVIDERS";

/// Whether the deterministic providers are requested.
#[must_use]
pub fn deterministic_requested() -> bool {
    std::env::var(DETERMINISTIC_ENV).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// A hashing embedder: no network, no key, same vector on every machine.
///
/// It is a bag-of-words hashing embedder, not noise. Each token is hashed into
/// a bucket and the vector is L2-normalised, so cosine similarity tracks token
/// overlap: a chunk sharing words with the query outranks one that does not.
/// That is the property the smoke path needs — a fixture that always returns
/// the same random vector would pass "an answer arrived" while proving nothing
/// about retrieval.
///
/// It is not a semantic model. Synonyms do not match, and it must never be
/// pointed at a corpus anyone intends to search for real.
#[derive(Debug, Clone)]
pub struct DeterministicEmbedder {
    dimension: usize,
}

impl Default for DeterministicEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl DeterministicEmbedder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            dimension: EMBEDDING_DIM,
        }
    }

    /// Build one with a deliberately wrong width, to exercise the dimension guard.
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self { dimension }
    }

    /// FNV-1a. Chosen for being short enough to read and stable across releases:
    /// `DefaultHasher` explicitly does not promise the same output between Rust
    /// versions, which would make "deterministic" a lie across a toolchain bump.
    fn hash_token(token: &str) -> u64 {
        const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x1000_0000_01b3;

        let mut hash = OFFSET;
        for byte in token.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
        hash
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.dimension];

        for token in text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
        {
            let lowered = token.to_lowercase();
            let hash = Self::hash_token(&lowered);
            let bucket = (hash % self.dimension as u64) as usize;
            // The sign comes from a different bit than the bucket, so two tokens
            // sharing a bucket are as likely to cancel as to reinforce. Without
            // it, every vector would be strictly positive and every pair of
            // documents would look similar.
            let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
            vector[bucket] += sign;
        }

        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        vector
    }
}

#[async_trait]
impl EmbeddingProvider for DeterministicEmbedder {
    fn name(&self) -> &str {
        "deterministic"
    }

    fn model(&self) -> &str {
        "deterministic-hash-v1"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AppError> {
        Ok(texts.iter().map(|t| self.embed(t)).collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, AppError> {
        Ok(self.embed(text))
    }

    fn batch_size(&self) -> usize {
        // Hashing is cheap and synchronous; a large batch just means a longer
        // loop between progress updates.
        128
    }
}

// ============================================================================
// Deterministic LLM
// ============================================================================

/// Opening tag of a retrieved chunk in the RAG system prompt.
///
/// The deterministic provider reads the corpus out of the prompt so its answer
/// depends on what was ingested. A fixture that answered the same thing
/// regardless would let a smoke path pass against an empty index.
///
/// It tracks `format_context_for_llm`, which is the only producer of this
/// shape. `answer_for` degrades to "no source" rather than panicking if the
/// two ever drift, and the smoke path fails on the missing citation — loudly,
/// which is the point.
const SOURCE_OPEN: &str = "<source ";
const SOURCE_CLOSE: &str = "</source>";

/// Longest snippet the deterministic provider quotes back.
const SNIPPET_CHARS: usize = 240;

/// An in-process `LlmProvider` that answers from the retrieved context.
///
/// It does no reasoning. It finds the first retrieved chunk in the system
/// prompt, quotes the opening of it and appends the `[1]` marker that citation
/// extraction resolves. That is enough to exercise the whole chat path —
/// streaming, SSE framing, citation resolution, terminal `done` — with no
/// provider account.
///
/// It is never a fallback. A server reaches it only through
/// [`DETERMINISTIC_ENV`], because an LLM that silently paraphrases its own
/// context would look like a working product to everyone except its users.
#[derive(Debug, Clone, Default)]
pub struct DeterministicLlm;

impl DeterministicLlm {
    /// The sentence the provider streams for `system_prompt`.
    ///
    /// Public so a test can assert the grounding without driving a byte stream.
    #[must_use]
    pub fn answer_for(system_prompt: &str) -> String {
        const NO_SOURCE: &str = "No source was retrieved for this question.";

        let Some(chunk) = first_source_text(system_prompt) else {
            return NO_SOURCE.to_owned();
        };
        if chunk.is_empty() {
            return NO_SOURCE.to_owned();
        }

        let snippet: String = chunk.chars().take(SNIPPET_CHARS).collect();
        format!("According to the source: {} [1]", snippet.trim_end())
    }
}

/// Text of the first `<source>` element in the RAG context, unescaped.
fn first_source_text(system_prompt: &str) -> Option<String> {
    let open = system_prompt.find(SOURCE_OPEN)?;
    // Skip the attribute list: content starts after the tag closes.
    let content_start = open + system_prompt[open..].find('>')? + 1;
    let content_end = content_start + system_prompt[content_start..].find(SOURCE_CLOSE)?;

    let raw = system_prompt[content_start..content_end].trim();
    // `format_context_for_llm` escapes the five XML entities; undo them so the
    // quote reads as the operator wrote it.
    Some(
        raw.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&amp;", "&"),
    )
}

#[async_trait]
impl crate::llm::LlmProvider for DeterministicLlm {
    fn name(&self) -> &str {
        "deterministic"
    }

    fn default_model(&self) -> &str {
        "deterministic-echo-v1"
    }

    fn is_available(&self) -> bool {
        true
    }

    async fn stream_chat(
        &self,
        system_prompt: &str,
        _messages: Vec<crate::llm::LlmMessage>,
        _model: Option<&str>,
        _documents: &[crate::llm::RagDocument],
        _temperature: Option<f32>,
    ) -> Result<crate::llm::ByteStream, AppError> {
        let answer = Self::answer_for(system_prompt);

        // Several frames rather than one: the transport, the citation collector
        // and the SSE adapter all have to survive a chunked response, and a
        // single-frame fake would never exercise that.
        let mut frames: Vec<Result<bytes::Bytes, AppError>> = answer
            .split_inclusive(' ')
            .map(|word| {
                let payload = serde_json::json!({
                    "choices": [{ "delta": { "content": word }, "index": 0 }]
                });
                Ok(bytes::Bytes::from(format!("data: {payload}\n\n")))
            })
            .collect();
        frames.push(Ok(bytes::Bytes::from_static(b"data: [DONE]\n\n")));

        Ok(Box::pin(futures::stream::iter(frames)))
    }

    fn parse_sse_data(&self, data: &str) -> Option<crate::llm::LlmStreamEvent> {
        crate::llm::parse_openai_sse_data(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[tokio::test]
    async fn the_same_text_always_embeds_to_the_same_vector() {
        let first = DeterministicEmbedder::new();
        let second = DeterministicEmbedder::new();
        let a = first.embed_query("reciprocal rank fusion").await.unwrap();
        let b = second.embed_query("reciprocal rank fusion").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn vectors_have_the_schema_width_and_unit_norm() {
        let embedder = DeterministicEmbedder::new();
        let v = embedder.embed_query("hello world").await.unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[tokio::test]
    async fn empty_text_does_not_divide_by_zero() {
        let embedder = DeterministicEmbedder::new();
        let v = embedder.embed_query("   ").await.unwrap();
        assert_eq!(v.len(), EMBEDDING_DIM);
        assert!(v.iter().all(|x| *x == 0.0));
    }

    /// The property the smoke path depends on: overlap ranks above no overlap.
    #[tokio::test]
    async fn overlapping_text_ranks_above_unrelated_text() {
        let embedder = DeterministicEmbedder::new();
        let query = embedder
            .embed_query("what constant does reciprocal rank fusion use")
            .await
            .unwrap();

        let docs = embedder
            .embed_documents(&[
                "Reciprocal Rank Fusion scores each document with a constant k of 60.".to_string(),
                "The kitchen was painted yellow last spring by the previous owner.".to_string(),
            ])
            .await
            .unwrap();

        let relevant = cosine(&query, &docs[0]);
        let unrelated = cosine(&query, &docs[1]);
        assert!(
            relevant > unrelated,
            "relevant {relevant} must outscore unrelated {unrelated}"
        );
    }

    #[tokio::test]
    async fn casing_and_punctuation_do_not_change_the_vector() {
        let embedder = DeterministicEmbedder::new();
        let a = embedder.embed_query("Rank Fusion!").await.unwrap();
        let b = embedder.embed_query("rank, fusion").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn document_order_is_preserved() {
        let embedder = DeterministicEmbedder::new();
        let texts = ["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let batch = embedder.embed_documents(&texts).await.unwrap();
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(batch[i], embedder.embed_query(text).await.unwrap());
        }
    }

    // ------------------------------------------------------------------
    // Deterministic LLM
    // ------------------------------------------------------------------

    /// The real shape, produced by `format_context_for_llm`. A fixture written
    /// against an invented shape is how the first end-to-end run found this
    /// provider answering "no source" over a populated index.
    fn rag_prompt(chunks: &[&str]) -> String {
        let mut s = String::from("You are a helpful assistant.\n\n<sources>\n");
        for (i, c) in chunks.iter().enumerate() {
            s.push_str(&format!(
                "<source index=\"{}\" source_id=\"11111111-1111-4111-8111-111111111111\" \
                 title=\"Notes\">\n{c}\n</source>\n\n",
                i + 1
            ));
        }
        s.push_str("</sources>");
        s
    }

    #[test]
    fn the_answer_quotes_the_first_retrieved_chunk() {
        let prompt = rag_prompt(&[
            "Reciprocal Rank Fusion uses a constant k of 60.",
            "The kitchen was painted yellow.",
        ]);
        let answer = DeterministicLlm::answer_for(&prompt);
        assert!(answer.contains("constant k of 60"), "{answer}");
        assert!(
            !answer.contains("kitchen"),
            "must not reach past chunk 1: {answer}"
        );
        assert!(
            !answer.contains("<source"),
            "must not quote the markup: {answer}"
        );
    }

    #[test]
    fn the_answer_carries_a_resolvable_citation_marker() {
        assert!(DeterministicLlm::answer_for(&rag_prompt(&["Anything at all."])).ends_with("[1]"));
    }

    #[test]
    fn escaped_entities_are_restored_in_the_quote() {
        let prompt = rag_prompt(&["Use &lt;source&gt; tags &amp; nothing else."]);
        let answer = DeterministicLlm::answer_for(&prompt);
        assert!(
            answer.contains("Use <source> tags & nothing else."),
            "{answer}"
        );
    }

    #[test]
    fn an_empty_corpus_produces_no_citation() {
        // No source retrieved means nothing to cite. Emitting `[1]` anyway
        // would make the smoke path pass against an empty index.
        let answer = DeterministicLlm::answer_for("You are a helpful assistant.");
        assert!(!answer.contains("[1]"), "{answer}");
    }

    #[test]
    fn a_present_but_empty_chunk_produces_no_citation() {
        let answer = DeterministicLlm::answer_for(&rag_prompt(&["   "]));
        assert!(!answer.contains("[1]"), "{answer}");
    }

    #[test]
    fn an_unterminated_source_element_produces_no_citation() {
        let answer = DeterministicLlm::answer_for("<sources>\n<source index=\"1\">dangling");
        assert!(!answer.contains("[1]"), "{answer}");
    }

    #[test]
    fn a_long_chunk_is_truncated_rather_than_streamed_whole() {
        let long = "word ".repeat(400);
        let answer = DeterministicLlm::answer_for(&rag_prompt(&[&long]));
        assert!(
            answer.chars().count() < SNIPPET_CHARS + 64,
            "{}",
            answer.len()
        );
    }

    #[tokio::test]
    async fn the_stream_is_chunked_and_terminates() {
        use crate::llm::{LlmProvider, LlmStreamEvent};
        use futures::StreamExt;

        let provider = DeterministicLlm;
        let prompt = rag_prompt(&["Retrieval fuses two lists."]);
        let mut stream = provider
            .stream_chat(&prompt, vec![], None, &[], None)
            .await
            .unwrap();

        let mut text = String::new();
        let mut frames = 0;
        let mut done = false;
        while let Some(frame) = stream.next().await {
            let bytes = frame.unwrap();
            for line in String::from_utf8_lossy(&bytes).lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                match provider.parse_sse_data(data) {
                    Some(LlmStreamEvent::TextDelta { text: delta }) => {
                        frames += 1;
                        text.push_str(&delta);
                    }
                    Some(LlmStreamEvent::Done) => done = true,
                    _ => {}
                }
            }
        }

        assert!(done, "must terminate with Done");
        assert!(
            frames > 1,
            "a single-frame fake would not exercise the transport"
        );
        assert_eq!(text, DeterministicLlm::answer_for(&prompt));
    }

    #[test]
    fn matching_dimension_passes_the_guard() {
        assert!(check_dimension(&DeterministicEmbedder::new()).is_ok());
    }

    #[test]
    fn a_returned_vector_of_the_wrong_width_is_a_typed_error() {
        let embedder = DeterministicEmbedder::new();
        let good = vec![0.0_f32; EMBEDDING_DIM];
        assert!(check_batch_widths(&embedder, std::slice::from_ref(&good)).is_ok());

        // A provider whose declaration is right but whose response is not:
        // this must fail here, not inside the pgvector insert.
        let err = check_batch_widths(&embedder, &[good, vec![0.0_f32; 512]]).unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("512"), "{rendered}");
        assert!(rendered.contains("index 1"), "{rendered}");
        assert!(rendered.contains("deterministic"), "{rendered}");
    }

    #[test]
    fn mismatched_dimension_is_reported_with_both_widths() {
        let err = check_dimension(&DeterministicEmbedder::with_dimension(768)).unwrap_err();
        assert_eq!(err.provider_dimension, 768);
        assert_eq!(err.schema_dimension, EMBEDDING_DIM);
        let rendered = err.to_string();
        assert!(rendered.contains("768"), "{rendered}");
        assert!(rendered.contains("1024"), "{rendered}");
    }
}
