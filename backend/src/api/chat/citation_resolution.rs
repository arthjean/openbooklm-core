//! Final citation validation for streamed chat answers.
//!
//! Generation may take long enough for a source to be reprocessed. This module
//! rereads active pointers after the provider finishes and emits public
//! citation events only after generation, span and claim checks pass.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::llm::citations::{
    claim_is_supported_by, extract_citations_verified_against_active, find_code_ranges,
};
use crate::llm::types::ChunkProvenance;
use crate::llm::{CitableChunk, LocatedCitation, NativeCitation};
use crate::repositories::{ActiveGenerationLease, SourceRepository};
use crate::services::rag::eval::trace::ReasonCode;
use crate::types::SearchResult;

pub(super) struct CitationResolution<'a> {
    pub uses_native_citations: bool,
    pub native_citations: &'a [LocatedCitation<NativeCitation>],
    pub doc_citation_map: &'a HashMap<usize, usize>,
    pub rag_documents: &'a [crate::llm::RagDocument],
    pub context_chunks: &'a [SearchResult],
    pub full_response: &'a str,
    pub notebook_id: Uuid,
}

pub(super) struct ResolvedCitations {
    pub citations: Vec<crate::llm::Citation>,
    pub rejected: usize,
    pub event_refs: Vec<(usize, Uuid)>,
    pub lease: Option<ActiveGenerationLease>,
}

struct CitationCandidate {
    citation: crate::llm::Citation,
    event_index: usize,
    generation_id: Uuid,
}

pub(super) async fn resolve_citations(
    input: &CitationResolution<'_>,
    source_repo: &dyn SourceRepository,
) -> ResolvedCitations {
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
        resolve_native(input, source_repo).await
    } else {
        resolve_prompt_markers(input, source_repo, &active_generations).await
    }
}

async fn resolve_native(
    input: &CitationResolution<'_>,
    source_repo: &dyn SourceRepository,
) -> ResolvedCitations {
    let mut seen_doc_indices = HashSet::new();
    let mut rejected = 0usize;
    let mut candidates = Vec::new();
    let code_ranges = find_code_ranges(input.full_response);
    for located in input.native_citations {
        let native = &located.citation;
        let marker_start = located.marker_start;
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
        if !seen_doc_indices.insert(native.document_index) {
            continue;
        }
        let Some(event_index) = input.doc_citation_map.get(&native.document_index).copied() else {
            rejected += 1;
            continue;
        };
        let citation = crate::llm::Citation::new(
            document.source_id,
            document.chunk_index,
            crate::llm::citations::truncate_text(&native.cited_text, 200).into_owned(),
            document.relevance_score,
            provenance,
        );
        candidates.push(CitationCandidate {
            citation,
            event_index,
            generation_id: document.generation_id,
        });
    }

    let resolved = finalize_candidates(source_repo, input.notebook_id, candidates, rejected).await;
    tracing::info!(
        notebook_id = %input.notebook_id,
        native_count = input.native_citations.len(),
        deduped = resolved.citations.len(),
        rejected = resolved.rejected,
        "Native citation mapping completed"
    );
    resolved
}

async fn resolve_prompt_markers(
    input: &CitationResolution<'_>,
    source_repo: &dyn SourceRepository,
    active_generations: &HashMap<Uuid, Uuid>,
) -> ResolvedCitations {
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
    let mut candidates = Vec::with_capacity(extracted.citations.len());
    for located in extracted.citations {
        let citation = located.citation;
        if let Some((index, chunk)) = input.context_chunks.iter().enumerate().find(|(_, chunk)| {
            chunk.source_id == citation.source_id && chunk.chunk_index == citation.chunk_index
        }) {
            candidates.push(CitationCandidate {
                citation,
                event_index: index + 1,
                generation_id: chunk.generation_id,
            });
        } else {
            rejected += 1;
        }
    }
    let resolved = finalize_candidates(source_repo, input.notebook_id, candidates, rejected).await;
    tracing::info!(
        notebook_id = %input.notebook_id,
        context_chunks = input.context_chunks.len(),
        citations = resolved.citations.len(),
        rejected = resolved.rejected,
        response_len = input.full_response.len(),
        "Regex citation extraction completed"
    );
    resolved
}

async fn finalize_candidates(
    source_repo: &dyn SourceRepository,
    notebook_id: Uuid,
    candidates: Vec<CitationCandidate>,
    mut rejected: usize,
) -> ResolvedCitations {
    if candidates.is_empty() {
        return ResolvedCitations {
            citations: Vec::new(),
            rejected,
            event_refs: Vec::new(),
            lease: None,
        };
    }
    let requested: Vec<_> = candidates
        .iter()
        .map(|candidate| (candidate.citation.source_id, candidate.generation_id))
        .collect();
    let lease = match source_repo.lock_active_generations(&requested).await {
        Ok(lease) => lease,
        Err(error) => {
            tracing::warn!(
                %notebook_id,
                error = %error,
                reason = ReasonCode::CitationRejected.as_str(),
                "Citation active-generation lease failed"
            );
            rejected += candidates.len();
            return ResolvedCitations {
                citations: Vec::new(),
                rejected,
                event_refs: Vec::new(),
                lease: None,
            };
        }
    };

    let mut citations = Vec::with_capacity(candidates.len());
    let mut event_refs = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if lease.is_active(candidate.citation.source_id, candidate.generation_id) {
            event_refs.push((candidate.event_index, candidate.citation.source_id));
            citations.push(candidate.citation);
        } else {
            rejected += 1;
            tracing::warn!(
                %notebook_id,
                source_id = %candidate.citation.source_id,
                generation_id = %candidate.generation_id,
                reason = ReasonCode::CitationRejected.as_str(),
                "Citation generation is no longer active"
            );
        }
    }

    if citations.is_empty() {
        if let Err(error) = lease.release().await {
            tracing::warn!(%notebook_id, error = %error, "Empty citation lease release failed");
        }
        ResolvedCitations {
            citations,
            rejected,
            event_refs,
            lease: None,
        }
    } else {
        ResolvedCitations {
            citations,
            rejected,
            event_refs,
            lease: Some(lease),
        }
    }
}
