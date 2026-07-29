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

/// Lightweight chunk type for citation extraction.
///
/// Decouples the `llm` module from the services layer (`SearchResult`).
/// Call sites convert `SearchResult` → `CitableChunk` before passing to
/// [`extract_citations`](crate::llm::citations::extract_citations).
#[derive(Debug, Clone)]
pub struct CitableChunk {
    pub source_id: Uuid,
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
