//! Shared types for LLM providers

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Message role in LLM conversations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
    System,
}

impl Role {
    /// Convert to the string format expected by LLM APIs.
    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for Role {
    fn from(s: &str) -> Self {
        match s {
            "user" => Self::User,
            "assistant" => Self::Assistant,
            "system" => Self::System,
            _ => {
                tracing::warn!(unknown_role = s, "Unknown role string, defaulting to User");
                Self::User
            }
        }
    }
}

impl From<String> for Role {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

/// Teaching mode for pedagogical adaptation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum TeachingMode {
    /// Flash: Quick summary, concise
    Flash,
    /// Deep: Complete exploration, detailed
    #[default]
    Deep,
    /// Quiz: Interactive QCM quiz (one question at a time)
    Quiz,
    /// Glossary: Extract and define key terms
    Glossary,
    /// Summary: Structured summary with sections
    Summary,
    /// Timeline: Chronological events extraction
    Timeline,
}

impl TeachingMode {
    pub const ALL: &[Self] = &[
        Self::Flash,
        Self::Deep,
        Self::Quiz,
        Self::Glossary,
        Self::Summary,
        Self::Timeline,
    ];

    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Flash => "Flash",
            Self::Deep => "Deep",
            Self::Quiz => "Quiz",
            Self::Glossary => "Glossary",
            Self::Summary => "Summary",
            Self::Timeline => "Timeline",
        }
    }

    pub const fn icon(&self) -> &'static str {
        match self {
            Self::Flash => "⚡",
            Self::Deep => "🧠",
            Self::Quiz => "❓",
            Self::Glossary => "📖",
            Self::Summary => "📄",
            Self::Timeline => "🕐",
        }
    }

    pub const fn description(&self) -> &'static str {
        match self {
            Self::Flash => "Quick essential summary",
            Self::Deep => "Complete detailed exploration",
            Self::Quiz => "Interactive multiple-choice quiz",
            Self::Glossary => "Key terms extraction",
            Self::Summary => "Structured summary with sections",
            Self::Timeline => "Chronological events",
        }
    }
}

impl fmt::Display for TeachingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Flash => "flash",
            Self::Deep => "deep",
            Self::Quiz => "quiz",
            Self::Glossary => "glossary",
            Self::Summary => "summary",
            Self::Timeline => "timeline",
        })
    }
}

/// Generic message format for LLM conversations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: Role,
    pub content: String,
}

impl LlmMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    #[inline]
    pub const fn is_user(&self) -> bool {
        matches!(self.role, Role::User)
    }

    #[inline]
    pub const fn is_assistant(&self) -> bool {
        matches!(self.role, Role::Assistant)
    }

    #[inline]
    pub const fn is_system(&self) -> bool {
        matches!(self.role, Role::System)
    }
}

/// Events emitted during LLM streaming.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmStreamEvent {
    /// Text content chunk
    TextDelta { text: String },
    /// Stream completed
    Done,
    /// Error occurred
    Error { message: String },
    /// Native citation from the Anthropic Citations API.
    ///
    /// Emitted inline during streaming when the provider supports native
    /// citations. Carries the exact cited text and document reference.
    NativeCitation { citation: NativeCitation },
}

/// A citation from Anthropic's native Citations API.
///
/// Unlike prompt-based `[N]` markers, native citations are guaranteed
/// to reference documents that were actually provided and cannot be
/// hallucinated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeCitation {
    /// 0-indexed position in the documents array passed to the LLM.
    pub document_index: usize,
    /// Exact text from the source document that was cited.
    pub cited_text: String,
    /// Title of the cited document.
    pub document_title: String,
}

/// A document to pass to the LLM for native citation support.
///
/// Used by providers that support structured document citations (Anthropic).
/// Each document maps to a source chunk from the RAG pipeline.
#[derive(Debug, Clone)]
pub struct RagDocument {
    /// Source UUID for mapping citations back to notebook sources.
    pub source_id: Uuid,
    /// Generation that was active when this document was retrieved.
    pub generation_id: Uuid,
    /// Human-readable document title.
    pub title: String,
    /// Document text content (parent_content or chunk content).
    pub content: String,
    /// Chunk index within the source (for citation DB storage).
    pub chunk_index: i32,
    /// Relevance score from retrieval (for citation metadata).
    pub relevance_score: f32,
    /// Chunk metadata for citation enrichment.
    pub metadata: Option<serde_json::Value>,
}

/// Citation extracted from LLM response.
#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct Citation {
    pub source_id: Uuid,
    pub chunk_index: i32,
    pub text: String,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub section_header: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_start: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_end: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_url: Option<String>,
}

impl Citation {
    /// Assemble a citation from its anchor and the cited chunk's provenance.
    ///
    /// The one place the public citation shape is filled in, so the regex path
    /// and the provider-native path cannot enrich a citation differently.
    #[must_use]
    pub fn new(
        source_id: Uuid,
        chunk_index: i32,
        text: String,
        relevance_score: f32,
        provenance: ChunkProvenance,
    ) -> Self {
        Self {
            source_id,
            chunk_index,
            text,
            relevance_score,
            section_header: provenance.section_header,
            page_number: provenance.page_number,
            timestamp_start: provenance.timestamp_start,
            timestamp_end: provenance.timestamp_end,
            video_id: provenance.video_id,
            citation_url: provenance.citation_url,
        }
    }
}

/// The provenance a citation is built from, read once from a chunk's metadata.
///
/// Chunk metadata travels as JSON because it crosses the database, but
/// [`ChunkMetadata`](crate::types::ChunkMetadata) is its type. Reading it here,
/// typed and once, is what replaced the same six `get("…").and_then(as_str)`
/// probes copied across the regex path, the native-citation path and the
/// evidence renderer: adding a provenance field used to mean adding a seventh
/// probe to each of the three.
///
/// Malformed or absent metadata yields an empty provenance. Citation
/// resolution rejects it because a source/chunk pair without a stable span is
/// not a navigable citation anchor (US-019).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChunkProvenance {
    pub section_header: Option<String>,
    pub page_number: Option<u32>,
    /// Last page the chunk covers, when it runs past its first one (US-019).
    pub page_end: Option<u32>,
    /// Byte range within the source's extracted text. `None` on chunks written
    /// before US-019.
    pub span: Option<(u32, u32)>,
    /// Position within the source, retained separately from the public
    /// citation shape so resolution can prove the metadata owns this chunk.
    pub position: Option<u32>,
    pub timestamp_start: Option<f64>,
    pub timestamp_end: Option<f64>,
    pub video_id: Option<String>,
    pub citation_url: Option<String>,
}

impl ChunkProvenance {
    /// Read provenance out of a chunk's stored metadata.
    #[must_use]
    pub fn read(metadata: Option<&serde_json::Value>) -> Self {
        let position = metadata
            .and_then(|value| value.get("position"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok());
        let Some(parsed) = metadata
            .cloned()
            .and_then(|value| serde_json::from_value::<crate::types::ChunkMetadata>(value).ok())
        else {
            return Self::default();
        };

        Self {
            section_header: parsed.section_header,
            page_number: parsed.page_number,
            page_end: parsed.page_end,
            span: parsed.span_start.zip(parsed.span_end),
            position,
            timestamp_start: parsed.timestamp_start,
            timestamp_end: parsed.timestamp_end,
            video_id: parsed.video_id,
            citation_url: parsed.citation_url,
        }
    }

    /// Whether the recorded span and pages describe a passage that can exist.
    ///
    /// A span must be non-empty and end after it starts; a last page must not
    /// precede the first. Both are written by the chunker in one pass, so a
    /// violation means the row was not produced by it — a hand-edited index, a
    /// truncated write, or a generation from another schema. US-019 AC-3 asks
    /// citation resolution to verify span ownership, and this is the part of
    /// that claim the stored provenance can actually answer.
    #[must_use]
    pub fn is_coherent(&self) -> bool {
        let span_ok = self.span.is_some_and(|(start, end)| start < end);
        let pages_ok = match (self.page_number, self.page_end) {
            (Some(first), Some(last)) => first <= last,
            // A last page with no first page is a page claim with no anchor.
            (None, Some(_)) => false,
            _ => true,
        };
        span_ok && pages_ok
    }

    /// Whether this span can own the exact child passage stored on the chunk.
    ///
    /// The resolver cannot reconstruct an extracted PDF from the public
    /// citation shape, but it can reject metadata whose byte width disagrees
    /// with the child bytes it claims to locate. Ingestion separately proves
    /// that the absolute offsets slice back to those same bytes.
    #[must_use]
    pub fn owns_content(&self, content: &str) -> bool {
        self.span.is_some_and(|(start, end)| {
            start < end
                && usize::try_from(end - start).is_ok_and(|span_len| span_len == content.len())
        }) && self.is_coherent()
    }

    /// Whether the metadata identifies this exact child position and owns its
    /// byte width.
    #[must_use]
    pub fn owns_chunk(&self, chunk_index: i32, content: &str) -> bool {
        u32::try_from(chunk_index).is_ok_and(|index| self.position == Some(index))
            && self.owns_content(content)
    }
}

/// Lightweight chunk type for citation extraction.
///
/// Decouples the `llm` module from the services layer (`SearchResult`).
/// Call sites convert `SearchResult` → `CitableChunk` before passing to
/// [`extract_citations`](crate::llm::citations::extract_citations).
#[derive(Debug, Clone)]
pub struct CitableChunk {
    pub source_id: Uuid,
    /// The index generation this chunk was read from.
    ///
    /// Carried so citation resolution can refuse a chunk with no generation:
    /// every retrieval path joins `sources.active_generation_id`, so a nil
    /// generation here means the chunk did not come from a published index and
    /// must not be presented as evidence (US-019 AC-3).
    pub generation_id: Uuid,
    pub chunk_index: i32,
    pub content: String,
    pub relevance_score: f32,
    /// Chunk metadata for citation enrichment (YouTube timestamps, section headers, etc.).
    pub metadata: Option<serde_json::Value>,
}

/// Information about an LLM provider.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderInfo {
    pub name: String,
    pub default_model: String,
    pub available: bool,
    pub models: Vec<String>,
}
