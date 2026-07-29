//! Repository traits defining data access contracts.
//!
//! Designed for mockability, implementation-agnostic (SeaORM, raw SQL), and single-entity focus.

use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entities::{
    account, account_settings, chat_message, note, notebook, notebook_memory, rag_log, source,
};
use crate::error::AppError;
use crate::services::rag::rag_log::{AggregatedMetrics, RagLogEntry, RagMetrics};
use crate::types::{Citation, SourceType};

/// Common result type for repository operations.
pub type RepoResult<T> = Result<T, AppError>;

/// Pagination parameters for list endpoints.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Pagination {
    pub offset: u64,
    pub limit: u64,
}

impl Default for Pagination {
    fn default() -> Self {
        Self {
            offset: 0,
            limit: 50,
        }
    }
}

/// Paginated response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Paginated<T> {
    pub items: Vec<T>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Notebook
// ─────────────────────────────────────────────────────────────────────────────

/// Notebook with source count for list views (avoids N+1 queries).
#[must_use]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookWithSourceCount {
    pub id: Uuid,
    pub user_id: Uuid,
    pub title: String,
    pub description: Option<String>,
    pub memory_enabled: bool,
    pub is_demo: bool,
    pub suggested_questions: serde_json::Value,
    pub created_at: DateTime<FixedOffset>,
    pub updated_at: DateTime<FixedOffset>,
    pub source_count: i64,
}

#[async_trait]
pub trait NotebookRepository: Send + Sync {
    async fn create(
        &self,
        user_id: Uuid,
        title: String,
        description: Option<String>,
    ) -> RepoResult<notebook::Model>;

    async fn get_by_id(&self, notebook_id: Uuid) -> RepoResult<Option<notebook::Model>>;

    /// Returns error if notebook doesn't exist or user doesn't own it.
    async fn get_for_user(&self, notebook_id: Uuid, user_id: Uuid) -> RepoResult<notebook::Model>;

    async fn list_for_user(&self, user_id: Uuid) -> RepoResult<Vec<notebook::Model>>;

    /// Single query returning notebooks with source counts.
    async fn list_with_source_counts(
        &self,
        user_id: Uuid,
    ) -> RepoResult<Vec<NotebookWithSourceCount>>;

    /// Paginated version of [`list_with_source_counts`](Self::list_with_source_counts).
    async fn list_with_source_counts_paginated(
        &self,
        user_id: Uuid,
        pagination: Pagination,
    ) -> RepoResult<Paginated<NotebookWithSourceCount>> {
        // Default implementation: fetch all and paginate in-memory.
        // Implementations should override with a proper LIMIT/OFFSET query.
        let all = self.list_with_source_counts(user_id).await?;
        let total = all.len() as u64;
        let items = all
            .into_iter()
            .skip(pagination.offset as usize)
            .take(pagination.limit as usize)
            .collect();
        Ok(Paginated {
            items,
            total,
            offset: pagination.offset,
            limit: pagination.limit,
        })
    }

    async fn update(
        &self,
        notebook_id: Uuid,
        user_id: Uuid,
        title: Option<String>,
        description: Option<Option<String>>,
        memory_enabled: Option<bool>,
    ) -> RepoResult<notebook::Model>;

    async fn delete(&self, notebook_id: Uuid, user_id: Uuid) -> RepoResult<()>;

    async fn count_for_user(&self, user_id: Uuid) -> RepoResult<u64>;

    async fn get_with_source_count(
        &self,
        notebook_id: Uuid,
        user_id: Uuid,
    ) -> RepoResult<(notebook::Model, u64)>;

    /// Find the demo notebook for a user (if it exists).
    async fn find_demo_for_user(&self, user_id: Uuid) -> RepoResult<Option<notebook::Model>>;

    /// Update the cached suggested questions for a notebook.
    ///
    /// Internal-only: called from the post-indexation pipeline where
    /// notebook ownership was already verified at the API boundary.
    async fn update_suggested_questions(
        &self,
        notebook_id: Uuid,
        questions: Vec<String>,
    ) -> RepoResult<notebook::Model>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Source
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait SourceRepository: Send + Sync {
    async fn create(
        &self,
        notebook_id: Uuid,
        title: String,
        source_type: SourceType,
        content: String,
        metadata: Option<serde_json::Value>,
    ) -> RepoResult<source::Model>;

    async fn get_by_id(&self, source_id: Uuid) -> RepoResult<Option<source::Model>>;

    /// Returns error if source doesn't exist or user doesn't own it.
    async fn get_for_user(&self, source_id: Uuid, user_id: Uuid) -> RepoResult<source::Model>;

    async fn list_for_notebook(&self, notebook_id: Uuid) -> RepoResult<Vec<source::Model>>;

    /// Paginated source listing for a notebook.
    async fn list_for_notebook_paginated(
        &self,
        notebook_id: Uuid,
        pagination: Pagination,
    ) -> RepoResult<Paginated<source::Model>> {
        let all = self.list_for_notebook(notebook_id).await?;
        let total = all.len() as u64;
        let items = all
            .into_iter()
            .skip(pagination.offset as usize)
            .take(pagination.limit as usize)
            .collect();
        Ok(Paginated {
            items,
            total,
            offset: pagination.offset,
            limit: pagination.limit,
        })
    }

    async fn delete(&self, source_id: Uuid, user_id: Uuid) -> RepoResult<()>;

    async fn update_status(
        &self,
        source_id: Uuid,
        status: source::SourceStatus,
        error_message: Option<String>,
    ) -> RepoResult<source::Model>;

    async fn update_chunk_count(
        &self,
        source_id: Uuid,
        chunk_count: i32,
    ) -> RepoResult<source::Model>;

    async fn count_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64>;

    async fn count_web_sources_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64>;

    async fn get_by_status(&self, status: source::SourceStatus) -> RepoResult<Vec<source::Model>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunk (Vector Store)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a similarity search.
#[must_use]
#[derive(Debug, Clone)]
pub struct ChunkSearchResult {
    pub id: Uuid,
    pub source_id: Uuid,
    pub chunk_index: i32,
    pub content: String,
    pub parent_content: Option<String>,
    pub source_title: String,
    pub relevance_score: f32,
    /// Chunk metadata (JSONB) — section_header, page_number, YouTube timestamps, etc.
    pub metadata: Option<serde_json::Value>,
}

/// Re-export from types for convenience.
pub use crate::types::ChunkWithContext;

#[async_trait]
pub trait ChunkRepository: Send + Sync {
    async fn store_chunks(
        &self,
        source_id: Uuid,
        chunks: &[ChunkWithContext],
        embeddings: &[Vec<f32>],
    ) -> RepoResult<()>;

    /// Returns (chunk_id, chunk_index, content) tuples.
    async fn get_for_source(&self, source_id: Uuid) -> RepoResult<Vec<(Uuid, i32, String)>>;

    /// Returns number of deleted chunks.
    async fn delete_for_source(&self, source_id: Uuid) -> RepoResult<u64>;

    /// Returns a random sample of chunk contents for a notebook.
    async fn sample_chunks_for_notebook(
        &self,
        notebook_id: Uuid,
        limit: i32,
    ) -> RepoResult<Vec<String>>;

    /// Insert a batch of chunks within an existing transaction (no DELETE, no commit).
    ///
    /// Used by the pipeline consumer to incrementally store chunks as embeddings arrive.
    /// `base_chunk_index` is the absolute index of `chunks[0]` in the full source.
    async fn store_chunk_batch(
        &self,
        source_id: Uuid,
        chunks: &[ChunkWithContext],
        embeddings: &[Vec<f32>],
        base_chunk_index: usize,
        txn: &sea_orm::DatabaseTransaction,
    ) -> RepoResult<()>;

    /// Returns `(chunk_index, content_hash, embedding)` tuples for deduplication.
    ///
    /// Used during source reprocessing to identify unchanged chunks whose
    /// embeddings can be reused instead of re-calling Voyage AI.
    async fn get_chunks_with_hashes(
        &self,
        source_id: Uuid,
    ) -> RepoResult<Vec<(i32, String, Vec<f32>)>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Search (raw SQL: pgvector + tsvector)
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait SearchRepository: Send + Sync {
    /// Notebook-scoped vector similarity search using pgvector `<=>` operator.
    async fn search_similar_chunks(
        &self,
        notebook_id: Uuid,
        query_embedding: &[f32],
        limit: i32,
    ) -> RepoResult<Vec<ChunkSearchResult>>;

    /// Notebook-scoped full-text (lexical) search using PostgreSQL tsvector.
    async fn search_lexical_chunks(
        &self,
        notebook_id: Uuid,
        query: &str,
        limit: i32,
    ) -> RepoResult<Vec<ChunkSearchResult>>;

    /// Count total chunks belonging to a notebook.
    ///
    /// Used to decide whether reranking should be applied (skip for small notebooks).
    async fn count_chunks_for_notebook(&self, notebook_id: Uuid) -> RepoResult<i64>;

    /// Load all chunks for a notebook with source titles, ordered by document structure.
    ///
    /// Used by context stuffing to bypass the search pipeline for small notebooks.
    /// All results have `relevance_score = 1.0` (equally relevant when stuffing).
    async fn get_all_chunks_for_notebook(
        &self,
        notebook_id: Uuid,
    ) -> RepoResult<Vec<ChunkSearchResult>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chat
// ─────────────────────────────────────────────────────────────────────────────

/// Paginated chat history result.
#[must_use]
#[derive(Debug, Clone)]
pub struct PaginatedChatHistory {
    pub messages: Vec<chat_message::Model>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
    pub has_more: bool,
}

pub const DEFAULT_CHAT_HISTORY_LIMIT: u64 = 50;
pub const MAX_CHAT_HISTORY_LIMIT: u64 = 200;

#[async_trait]
pub trait ChatRepository: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn create_message(
        &self,
        notebook_id: Uuid,
        role: &str,
        content: &str,
        citations: &[Citation],
        model: Option<&str>,
        agent_id: Option<Uuid>,
        session_id: Option<Uuid>,
    ) -> RepoResult<chat_message::Model>;

    /// Find a single message by ID.
    async fn get_by_id(&self, message_id: Uuid) -> RepoResult<Option<chat_message::Model>>;

    /// Get the most recent message in a notebook (by created_at DESC).
    async fn get_latest_message(
        &self,
        notebook_id: Uuid,
    ) -> RepoResult<Option<chat_message::Model>>;

    /// Get all messages in a notebook up to (and including) the given timestamp.
    ///
    /// Results are ordered chronologically (ascending `created_at`).
    async fn get_conversation_up_to(
        &self,
        notebook_id: Uuid,
        up_to: DateTime<FixedOffset>,
    ) -> RepoResult<Vec<chat_message::Model>>;

    /// Legacy non-paginated history.
    async fn get_history(
        &self,
        notebook_id: Uuid,
        limit: Option<u64>,
    ) -> RepoResult<Vec<chat_message::Model>>;

    async fn get_history_paginated(
        &self,
        notebook_id: Uuid,
        offset: Option<u64>,
        limit: Option<u64>,
    ) -> RepoResult<PaginatedChatHistory>;

    /// Last N messages, ordered chronologically.
    async fn get_recent_history(
        &self,
        notebook_id: Uuid,
        max_messages: u64,
    ) -> RepoResult<Vec<chat_message::Model>>;

    /// Returns number of deleted messages.
    async fn clear_history(&self, notebook_id: Uuid) -> RepoResult<u64>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Note
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait NoteRepository: Send + Sync {
    async fn create(
        &self,
        notebook_id: Uuid,
        title: String,
        content: String,
        original_message_id: Option<Uuid>,
    ) -> RepoResult<note::Model>;

    async fn get_by_id(&self, note_id: Uuid) -> RepoResult<Option<note::Model>>;

    /// Returns error if note doesn't exist or user doesn't own it.
    async fn get_for_user(&self, note_id: Uuid, user_id: Uuid) -> RepoResult<note::Model>;

    async fn list_for_notebook(&self, notebook_id: Uuid) -> RepoResult<Vec<note::Model>>;

    async fn update(
        &self,
        note_id: Uuid,
        user_id: Uuid,
        title: Option<String>,
        content: Option<String>,
    ) -> RepoResult<note::Model>;

    async fn delete(&self, note_id: Uuid, user_id: Uuid) -> RepoResult<()>;

    async fn count_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a memory similarity search.
#[must_use]
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub memory: notebook_memory::Model,
    pub similarity: f32,
}

#[async_trait]
pub trait MemoryRepository: Send + Sync {
    /// Insert a new memory with its embedding vector (raw SQL for VECTOR column).
    async fn create_with_embedding(
        &self,
        notebook_id: Uuid,
        content: &str,
        memory_type: &str,
        metadata: serde_json::Value,
        salience: f32,
        embedding: &[f32],
    ) -> RepoResult<notebook_memory::Model>;

    /// List all memories for a notebook, ordered by salience DESC, created_at DESC.
    async fn list_for_notebook(&self, notebook_id: Uuid)
    -> RepoResult<Vec<notebook_memory::Model>>;

    /// Semantic search: find the most similar memories using cosine distance.
    async fn search_similar(
        &self,
        notebook_id: Uuid,
        query_embedding: &[f32],
        limit: i32,
    ) -> RepoResult<Vec<MemorySearchResult>>;

    /// Batch similarity search: find similar memories for multiple embeddings.
    ///
    /// Implementations should override with a single batched SQL query
    /// (e.g., LATERAL join over unnested embeddings) instead of N round-trips.
    // TODO(perf): SeaOrmMemoryRepository does not override this yet —
    //             still issues N sequential pgvector scans. Implement a
    //             batched LATERAL join query for real N→1 reduction.
    async fn batch_search_similar(
        &self,
        notebook_id: Uuid,
        query_embeddings: &[Vec<f32>],
        limit_per_query: i32,
    ) -> RepoResult<Vec<Vec<MemorySearchResult>>> {
        let mut results = Vec::with_capacity(query_embeddings.len());
        for embedding in query_embeddings {
            results.push(
                self.search_similar(notebook_id, embedding, limit_per_query)
                    .await?,
            );
        }
        Ok(results)
    }

    /// Get a single memory by ID.
    async fn get_by_id(&self, memory_id: Uuid) -> RepoResult<Option<notebook_memory::Model>>;

    /// Update a memory's content, salience, or metadata.
    async fn update(
        &self,
        memory_id: Uuid,
        content: Option<String>,
        salience: Option<f32>,
        metadata: Option<serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> RepoResult<notebook_memory::Model>;

    /// Delete a single memory by ID.
    async fn delete(&self, memory_id: Uuid) -> RepoResult<()>;

    /// Delete all memories for a notebook ("forget all").
    async fn delete_all_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64>;

    /// Count memories for a notebook (used for plan limit enforcement).
    async fn count_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64>;

    /// Count memories of a specific type for a notebook (US-002).
    async fn count_by_type(&self, notebook_id: Uuid, memory_type: &str) -> RepoResult<u64>;

    /// Delete the oldest memories of a specific type for a notebook, keeping at most `keep` (US-002).
    ///
    /// Returns the number of deleted rows.
    async fn delete_oldest_by_type(
        &self,
        notebook_id: Uuid,
        memory_type: &str,
        keep: u64,
    ) -> RepoResult<u64>;

    /// Refresh `updated_at` to prevent temporal decay (US-004).
    async fn touch_memory(&self, memory_id: Uuid) -> RepoResult<()>;

    /// Update only the salience of a memory (US-004 decay).
    async fn update_salience(&self, memory_id: Uuid, new_salience: f32) -> RepoResult<()>;

    /// Delete all non-exempt memories below a salience threshold for a notebook (US-004).
    ///
    /// Memories with `memory_type = "conversation_summary"` are exempt.
    /// Returns the number of deleted rows.
    async fn delete_below_salience(&self, notebook_id: Uuid, threshold: f32) -> RepoResult<u64>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Accounts (public core, US-011)
// ─────────────────────────────────────────────────────────────────────────────

/// The core's ownership root. Carries no identity or commercial data.
#[async_trait]
pub trait AccountRepository: Send + Sync {
    /// Return the account, creating it if the caller's identity adapter has
    /// authenticated an account the core has not seen yet.
    async fn ensure_exists(&self, account_id: Uuid) -> RepoResult<account::Model>;

    async fn find(&self, account_id: Uuid) -> RepoResult<Option<account::Model>>;
}

/// Core preferences only: default provider and default model. Onboarding and
/// campaign state belong to [`SaasAccountSettingsRepository`].
#[async_trait]
pub trait AccountSettingsRepository: Send + Sync {
    /// Gets existing settings or creates defaults if not found.
    async fn get_or_create(&self, account_id: Uuid) -> RepoResult<account_settings::Model>;

    async fn update_defaults(
        &self,
        account_id: Uuid,
        default_provider: Option<String>,
        default_model: Option<String>,
    ) -> RepoResult<account_settings::Model>;
}

// ─────────────────────────────────────────────────────────────────────────────
// RAG Log
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait RagLogRepository: Send + Sync {
    /// Insert a RAG log entry, returning its ID.
    async fn create(&self, entry: &RagLogEntry) -> RepoResult<Uuid>;

    /// Find a RAG log entry by ID.
    async fn get_by_id(&self, id: Uuid) -> RepoResult<Option<rag_log::Model>>;

    /// Find a RAG log entry by ID, filtered by user ownership.
    async fn get_for_user(&self, id: Uuid, user_id: Uuid) -> RepoResult<rag_log::Model>;

    /// Update RAG quality metrics on an existing log entry.
    async fn update_metrics(&self, log_id: Uuid, metrics: &RagMetrics) -> RepoResult<()>;

    /// Update user feedback on a log entry (ownership-checked).
    async fn update_feedback(&self, log_id: Uuid, user_id: Uuid, feedback: &str) -> RepoResult<()>;

    /// Get aggregated RAG metrics for a notebook over the last N days.
    async fn get_notebook_metrics(
        &self,
        notebook_id: Uuid,
        days: i32,
    ) -> RepoResult<AggregatedMetrics>;

    /// Get aggregated RAG metrics for a user across all notebooks.
    async fn get_user_metrics(&self, user_id: Uuid, days: i32) -> RepoResult<AggregatedMetrics>;

    /// Batch lookup: find rag_log entries for a set of chat message IDs (via response_id).
    ///
    /// Returns a map of `response_id -> (log_id, user_feedback)`.
    async fn get_rag_log_ids_for_messages(
        &self,
        message_ids: &[Uuid],
    ) -> RepoResult<HashMap<Uuid, (Uuid, Option<String>)>>;

    /// Find a RAG log entry by response_id (chat message ID).
    async fn get_by_response_id(&self, response_id: Uuid) -> RepoResult<Option<rag_log::Model>>;

    /// Purge RAG logs older than the specified retention period.
    /// Returns the number of deleted rows.
    async fn purge_old_logs(&self, retention_days: i32) -> RepoResult<u64>;

    /// Get distinct source IDs from RAG logs with positive user feedback for a notebook.
    ///
    /// Used by US-009 to boost chunks from preferred sources during retrieval.
    async fn get_preferred_source_ids(&self, notebook_id: Uuid) -> RepoResult<Vec<Uuid>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// OCR Cache
// ─────────────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait OcrCacheRepository: Send + Sync {
    /// Look up cached OCR text by content hash and model.
    ///
    /// Returns `(ocr_text, pages_processed)` on cache hit, `None` on miss.
    async fn find_by_hash(
        &self,
        content_hash: &str,
        model: &str,
    ) -> RepoResult<Option<(String, i32)>>;

    /// Store OCR result in the cache. Uses `ON CONFLICT DO NOTHING` for
    /// race-condition safety (concurrent OCR calls for the same PDF).
    async fn store(
        &self,
        content_hash: &str,
        model: &str,
        ocr_text: &str,
        pages_processed: i32,
    ) -> RepoResult<()>;
}
