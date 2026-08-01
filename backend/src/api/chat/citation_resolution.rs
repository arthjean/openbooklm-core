//! Final citation validation for streamed chat answers.
//!
//! Generation may take long enough for a source to be reprocessed. This module
//! rereads active pointers after the provider finishes and emits public
//! citation events only after generation, span and claim checks pass.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::core::protocol::{ChatEvent, ChatEventStream};
use crate::llm::CitableChunk;
use crate::llm::citations::{
    claim_is_supported_by, extract_citations_verified_against_active, find_code_ranges,
};
use crate::llm::types::ChunkProvenance;
use crate::repositories::{ActiveGenerationLease, SourceRepository};
use crate::services::rag::eval::trace::ReasonCode;
use crate::types::SearchResult;

pub(super) struct CitationResolution<'a> {
    pub uses_native_citations: bool,
    pub native_citations: &'a [crate::llm::NativeCitation],
    pub native_marker_starts: &'a [usize],
    pub doc_citation_map: &'a HashMap<usize, usize>,
    pub rag_documents: &'a [crate::llm::RagDocument],
    pub context_chunks: &'a [SearchResult],
    pub full_response: &'a str,
    pub notebook_id: Uuid,
}

pub(super) async fn resolve_citations(
    out: &ChatEventStream,
    input: &CitationResolution<'_>,
    source_repo: &dyn SourceRepository,
) -> crate::llm::citations::ExtractedCitations {
    let source_ids: HashSet<Uuid> = input
        .context_chunks
        .iter()
        .map(|chunk| chunk.source_id)
        .collect();
    let mut active_generations = HashMap::with_capacity(source_ids.len());
    for source_id in source_ids {
        match source_repo.get_by_id(source_id).await {
            Ok(Some(source)) => {
                if let Some(generation_id) = source.active_generation_id {
                    active_generations.insert(source_id, generation_id);
                }
            }
            Ok(None) => {}
            Err(error) => tracing::warn!(
                notebook_id = %input.notebook_id,
                %source_id,
                error = %error,
                reason = ReasonCode::CitationRejected.as_str(),
                "Citation active-generation lookup failed"
            ),
        }
    }

    if input.uses_native_citations && !input.native_citations.is_empty() {
        resolve_native(out, input, source_repo).await
    } else {
        resolve_prompt_markers(out, input, source_repo, &active_generations).await
    }
}

async fn resolve_native(
    out: &ChatEventStream,
    input: &CitationResolution<'_>,
    source_repo: &dyn SourceRepository,
) -> crate::llm::citations::ExtractedCitations {
    let mut seen_doc_indices = HashSet::new();
    let mut rejected = 0usize;
    let mut mapped = Vec::new();
    let code_ranges = find_code_ranges(input.full_response);
    for (native, marker_start) in input
        .native_citations
        .iter()
        .zip(input.native_marker_starts.iter().copied())
    {
        let Some(document) = input.rag_documents.get(native.document_index) else {
            rejected += 1;
            continue;
        };
        let quote_owned =
            crate::llm::citations::quote_belongs_to(&document.content, &native.cited_text);
        let claim_linked =
            claim_is_supported_by(marker_start, input.full_response, &native.cited_text);
        let marker_outside_code = !code_ranges
            .iter()
            .any(|&(start, end)| marker_start >= start && marker_start < end);
        let provenance = ChunkProvenance::read(document.metadata.as_ref());
        if !quote_owned || !claim_linked || !marker_outside_code || !provenance.is_coherent() {
            rejected += 1;
            tracing::warn!(
                notebook_id = %input.notebook_id,
                document_index = native.document_index,
                source_id = %document.source_id,
                quote_owned,
                claim_linked,
                marker_outside_code,
                coherent_provenance = provenance.is_coherent(),
                reason = ReasonCode::CitationRejected.as_str(),
                "Native citation failed final validation"
            );
            continue;
        }
        let Some(lease) = lock_active_generation(
            source_repo,
            input.notebook_id,
            document.source_id,
            document.generation_id,
        )
        .await
        else {
            rejected += 1;
            continue;
        };

        if !seen_doc_indices.insert(native.document_index) {
            release_lease(lease, input.notebook_id, document.source_id).await;
            continue;
        }
        let citation = crate::llm::Citation::new(
            document.source_id,
            document.chunk_index,
            crate::llm::citations::truncate_text(&native.cited_text, 200).into_owned(),
            document.relevance_score,
            provenance,
        );
        if let Some(number) = input.doc_citation_map.get(&native.document_index) {
            out.emit(ChatEvent::citation(*number, citation.source_id))
                .await;
        }
        release_lease(lease, input.notebook_id, document.source_id).await;
        mapped.push(citation);
    }

    tracing::info!(
        notebook_id = %input.notebook_id,
        native_count = input.native_citations.len(),
        deduped = mapped.len(),
        rejected,
        "Native citation mapping completed"
    );
    crate::llm::citations::ExtractedCitations {
        citations: mapped,
        rejected,
    }
}

async fn resolve_prompt_markers(
    out: &ChatEventStream,
    input: &CitationResolution<'_>,
    source_repo: &dyn SourceRepository,
    active_generations: &HashMap<Uuid, Uuid>,
) -> crate::llm::citations::ExtractedCitations {
    let citable: Vec<CitableChunk> = input
        .context_chunks
        .iter()
        .map(CitableChunk::from)
        .collect();
    let extracted = extract_citations_verified_against_active(
        input.full_response,
        &citable,
        active_generations,
    );
    let mut rejected = extracted.rejected;
    let mut citations = Vec::with_capacity(extracted.citations.len());
    for citation in extracted.citations {
        if let Some((index, chunk)) = input.context_chunks.iter().enumerate().find(|(_, chunk)| {
            chunk.source_id == citation.source_id && chunk.chunk_index == citation.chunk_index
        }) && let Some(lease) = lock_active_generation(
            source_repo,
            input.notebook_id,
            chunk.source_id,
            chunk.generation_id,
        )
        .await
        {
            out.emit(ChatEvent::citation(index + 1, citation.source_id))
                .await;
            release_lease(lease, input.notebook_id, chunk.source_id).await;
            citations.push(citation);
        } else {
            rejected += 1;
        }
    }
    tracing::info!(
        notebook_id = %input.notebook_id,
        context_chunks = input.context_chunks.len(),
        citations = citations.len(),
        rejected,
        response_len = input.full_response.len(),
        "Regex citation extraction completed"
    );
    crate::llm::citations::ExtractedCitations {
        citations,
        rejected,
    }
}

async fn lock_active_generation(
    source_repo: &dyn SourceRepository,
    notebook_id: Uuid,
    source_id: Uuid,
    generation_id: Uuid,
) -> Option<ActiveGenerationLease> {
    match source_repo
        .lock_active_generation(source_id, generation_id)
        .await
    {
        Ok(Some(lease)) => Some(lease),
        Ok(None) => {
            tracing::warn!(
                %notebook_id,
                %source_id,
                %generation_id,
                reason = ReasonCode::CitationRejected.as_str(),
                "Citation generation is no longer active"
            );
            None
        }
        Err(error) => {
            tracing::warn!(
                %notebook_id,
                %source_id,
                %generation_id,
                error = %error,
                reason = ReasonCode::CitationRejected.as_str(),
                "Citation active-generation check failed"
            );
            None
        }
    }
}

async fn release_lease(lease: ActiveGenerationLease, notebook_id: Uuid, source_id: Uuid) {
    if let Err(error) = lease.release().await {
        tracing::warn!(
            %notebook_id,
            %source_id,
            error = %error,
            reason = ReasonCode::CitationRejected.as_str(),
            "Citation generation lease release failed after event enqueue"
        );
    }
}
