//! Stages 2 to 5: text becomes chunks stored under one building generation.
//!
//! Chunking, embedding reuse, and the embed/store pipeline. Everything here
//! writes only under the generation it is given, and nothing here decides the
//! run's outcome: it either fills a generation or reports why it could not.
//!
//! Batches still commit independently, which used to be the defect this epic
//! exists to fix — an interrupted run left the source with some of its chunks.
//! Under a building generation it is harmless: none of those rows are reachable
//! until the publication transaction, so an interrupted run is simply a
//! generation that never publishes.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use sea_orm::TransactionTrait;
use tracing::info;
use uuid::Uuid;

use crate::entities::source::SourceStatus;
use crate::error::AppError;
use crate::repositories::ChunkRepository;
use crate::services::ingestion_tasks::IngestionTasks;
use crate::services::rag::chunking::ParentChunk;
use crate::services::rag::provenance::ChunkingProvenance;
use crate::services::sources::update_source_status;
use crate::types::ChunkWithContext;

use super::extraction::Extracted;
use super::{IndexOwnership, PipelineFailure, ProcessingDeps, StageContext as _};

/// Upper bound on chunk fan-out, checked before allocation.
const MAX_CHUNKS: usize = 50_000;

/// Chunk `extracted`, then embed and store every chunk under `ownership`.
///
/// Returns the number of chunks the generation was declared to hold. Publication
/// compares that declaration against the rows actually present, which is why it
/// is recorded here and not inferred later.
pub(super) async fn build_chunks(
    deps: &ProcessingDeps,
    tasks: &IngestionTasks,
    ownership: &IndexOwnership,
    extracted: &Extracted,
    source_id: Uuid,
    notebook_id: Uuid,
) -> Result<usize, PipelineFailure> {
    let chunks = chunk_and_enrich(extracted)?;
    let total_chunks = chunks.len();

    // The generation now knows how many chunks it owes, and under which
    // chunking contract. The effective contract is only known here: the PDF+OCR
    // path rewrites the chunking source type, which changes the parent geometry.
    deps.generation_repo
        .record_build_plan(
            ownership.generation_id,
            i32::try_from(total_chunks).unwrap_or(i32::MAX),
            &ChunkingProvenance::current(crate::services::rag::chunking::parent_chunk_size(
                extracted.chunking_source_type,
            )),
        )
        .await
        .stage("Failed to record the build plan")?;

    let reused = compute_embedding_reuse(
        deps.chunk_repo.as_ref(),
        source_id,
        &ownership.provenance.embedding.fingerprint(),
        &chunks,
    )
    .await
    .stage("Failed to query reusable embeddings")?;

    info!(
        total_chunks,
        new_chunks = total_chunks - reused.len(),
        reused_chunks = reused.len(),
        "Chunk deduplication results"
    );

    update_source_status(
        deps.source_repo.as_ref(),
        source_id,
        SourceStatus::Embedding,
        None,
    )
    .await
    .stage("Failed to record embedding status")?;
    deps.broadcaster
        .broadcast_status(notebook_id, source_id, "embedding", None);

    embed_and_store(
        deps,
        tasks,
        ownership.generation_id,
        source_id,
        notebook_id,
        chunks,
        reused,
    )
    .await?;

    Ok(total_chunks)
}

/// Two-pass chunking, flattened, with source-type-specific enrichment applied.
fn chunk_and_enrich(extracted: &Extracted) -> Result<Vec<ChunkWithContext>, PipelineFailure> {
    // Pass 1 splits into parents, pass 2 into children. Children are the
    // retrieval/embedding unit; parents provide LLM context. Both carry their
    // page and their exact byte span (US-019).
    //
    // Contextualization (the former stage 3) is disabled to control cost. See
    // IMPORTANT_FUTUR.md at the repo root to re-enable it.
    let parents = crate::services::rag::chunking::chunk_source(
        &extracted.text,
        extracted.chunking_source_type,
    )
    .stage("Failed to chunk content")?;

    if parents.is_empty() {
        let msg = "No content to process";
        return Err(PipelineFailure::new(msg, AppError::Validation(msg.into())));
    }

    let mut chunks = flatten_parent_child_chunks(&parents).map_err(PipelineFailure::from_error)?;

    // Parse [MM:SS] timestamps out of chunk content so citations can deep-link.
    if let Some(ref video_id) = extracted.youtube_video_id {
        for chunk in &mut chunks {
            let (ts_start, ts_end) =
                crate::clients::youtube::extract_timestamp_range(&chunk.content);
            chunk.metadata.video_id = Some(video_id.clone());
            chunk.metadata.timestamp_start = ts_start;
            chunk.metadata.timestamp_end = ts_end;
            if let Some(start) = ts_start {
                chunk.metadata.citation_url = Some(format!(
                    "https://youtube.com/watch?v={video_id}&t={}",
                    start as u64
                ));
            }
        }
    }

    Ok(chunks)
}

/// One embedded batch on its way to storage.
struct EmbeddedBatch {
    base_index: usize,
    chunks: Vec<ChunkWithContext>,
    embeddings: Vec<Vec<f32>>,
}

/// Producer/consumer pipeline: embed batches, store each under the generation.
///
/// The producer embeds concurrently and sends through a channel of capacity 2,
/// which is the backpressure. The consumer stores each batch in its own
/// transaction on the current task.
async fn embed_and_store(
    deps: &ProcessingDeps,
    tasks: &IngestionTasks,
    generation_id: Uuid,
    source_id: Uuid,
    notebook_id: Uuid,
    chunks: Vec<ChunkWithContext>,
    reused: HashMap<usize, Vec<f32>>,
) -> Result<(), PipelineFailure> {
    let embedder = deps.embeddings.as_ref().ok_or_else(|| {
        PipelineFailure::from_error(AppError::Internal(
            "No embedding provider configured — cannot generate embeddings".into(),
        ))
    })?;

    let total_chunks = chunks.len();
    let batch_size = embedder.batch_size();
    let concurrency = embedder.concurrency();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<EmbeddedBatch>(2);

    let embedder_clone = Arc::clone(embedder);
    let broadcaster = deps.broadcaster.clone();
    let progress_total = u32::try_from(total_chunks).unwrap_or(u32::MAX);
    let progress = Arc::new(AtomicU32::new(0));
    let token = tasks.token();
    let producer_token = token.clone();

    let producer_handle = tasks.spawn(async move {
        use futures::stream::{self, StreamExt, TryStreamExt};

        let batch_work: Vec<(usize, Vec<ChunkWithContext>)> = chunks
            .chunks(batch_size)
            .enumerate()
            .map(|(batch_idx, slice)| (batch_idx * batch_size, slice.to_vec()))
            .collect();

        let result: Result<Vec<()>, AppError> = stream::iter(batch_work)
            .map(|(base_index, batch_chunks)| {
                let tx = tx.clone();
                let embedder = &embedder_clone;
                let reused = &reused;
                let progress = Arc::clone(&progress);
                let broadcaster = &broadcaster;
                let token = producer_token.clone();

                async move {
                    // Admission check. `buffer_unordered` creates each batch
                    // future lazily, so this runs the moment the batch would
                    // have started — which is what makes "zero provider calls
                    // start after cancellation" true rather than approximate.
                    if token.is_cancelled() {
                        return Err(AppError::Internal(
                            "Ingestion cancelled before this batch started".into(),
                        ));
                    }

                    let (mut batch_embeddings, embed_offsets, embed_texts) =
                        split_reused(&batch_chunks, base_index, reused);

                    // Batching shape, rate limiting and usage reconciliation all
                    // belong to the provider.
                    if !embed_texts.is_empty() {
                        let new_embeddings = tokio::select! {
                            biased;
                            () = token.cancelled() => return Err(AppError::Internal(
                                "Ingestion cancelled while embedding".into(),
                            )),
                            result = embedder.embed_batch(&embed_texts) => result?,
                        };
                        // Fail here rather than in the pgvector insert: a
                        // wrong-width vector must never reach a chunk row, and
                        // the error has to name the provider (US-020).
                        crate::core::providers::check_batch_widths(
                            embedder.as_ref(),
                            &new_embeddings,
                        )?;
                        for (offset, emb) in embed_offsets.into_iter().zip(new_embeddings) {
                            batch_embeddings[offset] = emb;
                        }
                    }

                    progress.fetch_add(
                        u32::try_from(batch_chunks.len()).unwrap_or(u32::MAX),
                        Ordering::Relaxed,
                    );
                    broadcaster.broadcast_embedding_progress(
                        notebook_id,
                        source_id,
                        progress.load(Ordering::Relaxed),
                        progress_total,
                    );

                    tx.send(EmbeddedBatch {
                        base_index,
                        chunks: batch_chunks,
                        embeddings: batch_embeddings,
                    })
                    .await
                    .map_err(|_| {
                        AppError::Internal("Pipeline consumer stopped unexpectedly".into())
                    })?;

                    Ok(())
                }
            })
            .buffer_unordered(concurrency)
            .try_collect()
            .await;

        if result.is_err() {
            producer_token.cancel();
        }

        // Drop the sender so the consumer's recv loop terminates.
        drop(tx);

        result?;
        Ok::<(), AppError>(())
    });

    let embed_start = std::time::Instant::now();
    let consumer_error = store_batches(deps, generation_id, source_id, &token, &mut rx).await;

    // The first terminal storage error closes admission immediately, rather
    // than after the outstanding batches have each paid for a provider call
    // whose result nothing will store (US-010).
    if consumer_error.is_some() {
        tasks.cancel();
    }
    // Release the producer. A `send` blocked on a consumer that stopped reading
    // would otherwise hold the task open until the drain deadline aborted it,
    // turning a clean storage failure into an abandoned task. The batches
    // discarded here were embedded but never stored, which is correct: this
    // generation is not going to publish.
    //
    // The producer's own `send` error is not reported. It cannot be the cause of
    // anything — the loop only exits without an error once every sender has
    // already been dropped — and on the error path the storage failure is what
    // the operator needs to see.
    rx.close();
    while rx.recv().await.is_some() {}

    // Awaiting the handle is what keeps the task owned: it is never dropped
    // while the task is still running.
    let producer_result = match producer_handle.await {
        Ok(result) => result,
        Err(e) => Err(AppError::Internal(format!(
            "Pipeline producer task panicked: {e}"
        ))),
    };

    // Storage first: a storage failure is the reason the producer was
    // cancelled, and its error would be the cancellation, not the cause.
    if let Some(e) = consumer_error {
        return Err(PipelineFailure::new(
            format!("Failed to store chunks: {e}"),
            e,
        ));
    }
    if let Err(e) = producer_result {
        return Err(PipelineFailure::new(
            format!("Failed to generate embeddings: {e}"),
            e,
        ));
    }

    info!(
        total_chunks,
        pipeline_ms = embed_start.elapsed().as_millis(),
        "Pipeline embedding+storage completed"
    );
    Ok(())
}

/// Consume embedded batches, storing each in its own transaction.
///
/// Returns the first error, having already rolled its transaction back.
async fn store_batches(
    deps: &ProcessingDeps,
    generation_id: Uuid,
    source_id: Uuid,
    token: &tokio_util::sync::CancellationToken,
    rx: &mut tokio::sync::mpsc::Receiver<EmbeddedBatch>,
) -> Option<AppError> {
    loop {
        let batch = tokio::select! {
            biased;
            () = token.cancelled() => return None,
            batch = rx.recv() => batch,
        };
        let batch = batch?;
        let txn = match deps.db.begin().await {
            Ok(txn) => txn,
            Err(e) => {
                return Some(AppError::Internal(format!(
                    "Failed to begin batch transaction: {e}"
                )));
            }
        };
        if let Err(e) = deps
            .chunk_repo
            .store_chunk_batch(
                generation_id,
                source_id,
                &batch.chunks,
                &batch.embeddings,
                batch.base_index,
                &txn,
            )
            .await
        {
            if let Err(rollback_err) = txn.rollback().await {
                tracing::error!(%source_id, error = %rollback_err, "Failed to rollback batch transaction");
            }
            return Some(e);
        }
        if let Err(e) = txn.commit().await {
            return Some(AppError::Internal(format!("Failed to commit batch: {e}")));
        }
    }
}

/// Split a batch into the embeddings already known and the texts still to embed.
///
/// Returns `(embeddings_with_placeholders, offsets_to_fill, texts_to_embed)`.
/// The placeholders keep the result aligned with `chunks` so a caller can drop
/// fresh vectors into their own positions.
fn split_reused(
    chunks: &[ChunkWithContext],
    base_index: usize,
    reused: &HashMap<usize, Vec<f32>>,
) -> (Vec<Vec<f32>>, Vec<usize>, Vec<String>) {
    let mut embeddings = Vec::with_capacity(chunks.len());
    let mut offsets = Vec::new();
    let mut texts = Vec::new();

    for (offset, chunk) in chunks.iter().enumerate() {
        if let Some(cached) = reused.get(&(base_index + offset)) {
            embeddings.push(cached.clone());
        } else {
            texts.push(crate::services::embeddings::contextualized_text(
                &chunk.content,
                chunk.context_prefix.as_deref(),
            ));
            offsets.push(offset);
            embeddings.push(Vec::new());
        }
    }

    (embeddings, offsets, texts)
}

/// Flatten a parent/child chunk hierarchy into a flat `Vec<ChunkWithContext>`.
///
/// Each child keeps the page and span the chunker resolved for it, and gains
/// its sequential position in the source. Errors when the total exceeds
/// [`MAX_CHUNKS`].
fn flatten_parent_child_chunks(parents: &[ParentChunk]) -> Result<Vec<ChunkWithContext>, AppError> {
    let total_children: usize = parents.iter().map(|p| p.children.len()).sum();
    if total_children > MAX_CHUNKS {
        return Err(AppError::Validation(format!(
            "Document produced too many chunks ({total_children}), maximum is {MAX_CHUNKS}"
        )));
    }

    let mut chunks: Vec<ChunkWithContext> = Vec::with_capacity(total_children);
    for parent in parents {
        let shared_parent: Arc<str> = Arc::from(parent.text.as_str());
        for child in &parent.children {
            let content_hash = crate::services::rag::utils::compute_content_hash(&child.text);
            let metadata = crate::types::ChunkMetadata {
                position: u32::try_from(chunks.len()).unwrap_or(u32::MAX),
                ..child.metadata.clone()
            };
            chunks.push(ChunkWithContext {
                content: child.text.clone(),
                context_prefix: None,
                parent_content: Some(Arc::clone(&shared_parent)),
                metadata,
                content_hash,
            });
        }
    }

    Ok(chunks)
}

/// Which chunks can reuse an existing embedding, by index.
///
/// The reuse key is `(content_hash, embedding_fingerprint)`, not the content
/// hash alone (US-011). The fingerprint half is applied by the repository,
/// which returns nothing at all when the active generation was built by a
/// different provider, model, width or normalization — so a changed fingerprint
/// rebuilds every vector instead of silently mixing two vector spaces.
async fn compute_embedding_reuse(
    chunk_repo: &dyn ChunkRepository,
    source_id: Uuid,
    embedding_fingerprint: &str,
    chunks: &[ChunkWithContext],
) -> Result<HashMap<usize, Vec<f32>>, AppError> {
    let by_hash: HashMap<String, Vec<f32>> = chunk_repo
        .get_reusable_embeddings(source_id, embedding_fingerprint)
        .await?
        .into_iter()
        .filter(|(hash, embedding)| !hash.is_empty() && !embedding.is_empty())
        .collect();

    Ok(chunks
        .iter()
        .enumerate()
        .filter_map(|(i, chunk)| by_hash.get(&chunk.content_hash).map(|e| (i, e.clone())))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SourceType;

    fn chunked(content: &str, source_type: SourceType) -> Vec<ParentChunk> {
        crate::services::rag::chunking::chunk_source(
            &crate::services::rag::chunking::SourceText::single(content),
            source_type,
        )
        .expect("chunking should work")
    }

    #[test]
    fn parent_child_flattening_produces_correct_chunk_with_context() {
        // Large enough to produce multiple parents and children.
        let content =
            "The quick brown fox jumps over the lazy dog near the river bank. ".repeat(500);
        let parents = chunked(&content, SourceType::Text);
        assert!(!parents.is_empty(), "Should produce at least one parent");

        let total_children: usize = parents.iter().map(|p| p.children.len()).sum();
        let chunks = flatten_parent_child_chunks(&parents).expect("flattening should work");

        assert_eq!(chunks.len(), total_children);
        assert!(
            chunks.len() >= 2 * parents.len(),
            "Should have at least 2× more children ({}) than parents ({})",
            chunks.len(),
            parents.len()
        );

        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.parent_content.is_some(),
                "Every child chunk must have parent_content"
            );
            assert_eq!(
                chunk.content_hash,
                crate::services::rag::utils::compute_content_hash(&chunk.content),
                "content_hash must be computed from child text"
            );
            assert_eq!(chunk.metadata.position, u32::try_from(i).expect("position"));
            assert!(
                chunk.context_prefix.is_none(),
                "contextualization is disabled"
            );
        }
    }

    /// Flattening renumbers positions and keeps everything else the chunker
    /// resolved. A span rewritten on the way to storage would point a citation
    /// at the wrong passage (US-019).
    #[test]
    fn flattening_preserves_the_span_the_chunker_resolved() {
        let pages: Vec<String> = (1..=4)
            .map(|i| format!("Page {i} body text about retention. ").repeat(30))
            .collect();
        let parents = crate::services::rag::chunking::chunk_source(
            &crate::services::rag::chunking::SourceText::paginated(pages),
            SourceType::Pdf,
        )
        .expect("chunking");

        let expected: Vec<(Option<u32>, Option<u32>, Option<u32>)> = parents
            .iter()
            .flat_map(|p| p.children.iter())
            .map(|c| {
                (
                    c.metadata.page_number,
                    c.metadata.span_start,
                    c.metadata.span_end,
                )
            })
            .collect();

        let chunks = flatten_parent_child_chunks(&parents).expect("flattening");
        assert_eq!(chunks.len(), expected.len());
        for (chunk, (page, start, end)) in chunks.iter().zip(expected) {
            assert_eq!(chunk.metadata.page_number, page);
            assert_eq!(chunk.metadata.span_start, start);
            assert_eq!(chunk.metadata.span_end, end);
            assert!(page.is_some(), "a PDF chunk reports its page");
            assert!(start.is_some(), "a chunk reports its span");
        }
    }

    #[test]
    fn parent_child_flattening_child_content_differs_from_parent() {
        // Varied content so overlap doesn't produce identical children.
        let content: String = (0..500)
            .map(|i| {
                format!(
                    "Sentence number {i} discusses topic {topic} in depth. ",
                    topic = i % 7
                )
            })
            .collect();

        let parents = chunked(&content, SourceType::Pdf);
        let parent = parents
            .iter()
            .find(|p| p.children.len() > 1)
            .expect("Should have at least one parent with multiple children");

        for child in &parent.children {
            assert!(
                child.text.len() < parent.text.len(),
                "Child ({} bytes) should be shorter than parent ({} bytes)",
                child.text.len(),
                parent.text.len()
            );
        }

        for i in 0..parent.children.len() {
            for j in (i + 1)..parent.children.len() {
                assert_ne!(
                    parent.children[i].text, parent.children[j].text,
                    "Children {i} and {j} should not be identical"
                );
            }
        }
    }

    #[test]
    fn reused_embeddings_keep_their_slot_and_only_new_text_is_embedded() {
        let chunks: Vec<ChunkWithContext> = (0..4)
            .map(|i| ChunkWithContext {
                content: format!("chunk {i}"),
                context_prefix: None,
                parent_content: None,
                metadata: crate::types::ChunkMetadata::default(),
                content_hash: format!("hash-{i}"),
            })
            .collect();

        // Global indices 11 and 13, i.e. offsets 1 and 3 of a batch based at 10.
        let reused = HashMap::from([(11, vec![1.0_f32]), (13, vec![3.0_f32])]);
        let (embeddings, offsets, texts) = split_reused(&chunks, 10, &reused);

        assert_eq!(embeddings.len(), 4, "every chunk keeps a slot");
        assert_eq!(embeddings[1], vec![1.0]);
        assert_eq!(embeddings[3], vec![3.0]);
        assert!(
            embeddings[0].is_empty() && embeddings[2].is_empty(),
            "slots to embed stay empty until the provider answers"
        );
        assert_eq!(offsets, vec![0, 2]);
        assert_eq!(texts, vec!["chunk 0", "chunk 2"]);
    }
}
