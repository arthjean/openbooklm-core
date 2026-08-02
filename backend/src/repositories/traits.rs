//! Repository traits defining data access contracts.
//!
//! Designed for mockability, implementation-agnostic (SeaORM, raw SQL), and single-entity focus.

use std::collections::{HashMap, HashSet};

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

/// Source-row locks held across citation persistence and event enqueue.
///
/// Publication updates the same rows, so it cannot move an active pointer
/// across the final public citation boundary.
#[must_use = "retain the lease through citation persistence and enqueue, then release it"]
pub struct ActiveGenerationLease {
    transaction: sea_orm::DatabaseTransaction,
    active: HashSet<(Uuid, Uuid)>,
}

impl ActiveGenerationLease {
    pub(super) const fn new(
        transaction: sea_orm::DatabaseTransaction,
        active: HashSet<(Uuid, Uuid)>,
    ) -> Self {
        Self {
            transaction,
            active,
        }
    }

    #[must_use]
    pub fn is_active(&self, source_id: Uuid, generation_id: Uuid) -> bool {
        self.active.contains(&(source_id, generation_id))
    }

    pub(crate) const fn transaction(&self) -> &sea_orm::DatabaseTransaction {
        &self.transaction
    }

    pub async fn release(self) -> RepoResult<()> {
        self.transaction.commit().await?;
        Ok(())
    }
}

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

    /// Lock every currently active `(source, generation)` pair in one
    /// transaction and report which requested pairs matched.
    ///
    /// The caller must retain the lease through citation persistence and
    /// non-blocking event enqueue, then release it promptly.
    async fn lock_active_generations(
        &self,
        generations: &[(Uuid, Uuid)],
    ) -> RepoResult<ActiveGenerationLease>;

    async fn lock_active_generation(
        &self,
        source_id: Uuid,
        generation_id: Uuid,
    ) -> RepoResult<Option<ActiveGenerationLease>> {
        let lease = self
            .lock_active_generations(&[(source_id, generation_id)])
            .await?;
        if lease.is_active(source_id, generation_id) {
            Ok(Some(lease))
        } else {
            lease.release().await?;
            Ok(None)
        }
    }

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

    async fn count_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64>;

    async fn count_web_sources_for_notebook(&self, notebook_id: Uuid) -> RepoResult<u64>;

    async fn get_by_status(&self, status: source::SourceStatus) -> RepoResult<Vec<source::Model>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Index generations (EP-002)
// ─────────────────────────────────────────────────────────────────────────────

/// What a successful publication moved.
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct PublicationOutcome {
    pub generation_id: Uuid,
    pub chunk_count: i32,
}

/// The lifecycle of an immutable source index.
///
/// The trait is deliberately narrow: everything it exposes is a transition
/// between the states in
/// [`GenerationState`](crate::entities::source_index_generation::GenerationState),
/// and nothing exposes a way to write the active pointer outside
/// [`publish`](Self::publish) and [`rollback_to_previous`](Self::rollback_to_previous).
#[async_trait]
pub trait GenerationRepository: Send + Sync {
    /// Take ownership of a source's index, or report that someone else has it.
    ///
    /// Returns the new generation id, or `None` when a building generation
    /// already exists — the compare-and-set that makes reprocessing
    /// single-owner (US-009).
    async fn claim(
        &self,
        source_id: Uuid,
        provenance: &crate::services::rag::provenance::GenerationProvenance,
    ) -> RepoResult<Option<Uuid>>;

    /// The id of the source's current building generation, if any.
    async fn find_building(&self, source_id: Uuid) -> RepoResult<Option<Uuid>>;

    /// Declare how many chunks this generation will store, and under which
    /// chunking contract.
    ///
    /// Publication compares the declaration against the rows actually present;
    /// a generation that never declares one cannot be published. The chunking
    /// provenance is recorded here rather than at claim time because the
    /// effective contract is only known after extraction: the PDF+OCR path
    /// rewrites the source type, which changes the chunk geometry.
    async fn record_build_plan(
        &self,
        generation_id: Uuid,
        expected: i32,
        chunking: &crate::services::rag::provenance::ChunkingProvenance,
    ) -> RepoResult<()>;

    /// Validate and publish in one transaction, moving the active pointer.
    async fn publish(
        &self,
        generation_id: Uuid,
        source_id: Uuid,
        expected_dimension: usize,
    ) -> RepoResult<PublicationOutcome>;

    /// Abandon a building generation, leaving the active one untouched.
    async fn mark_failed(
        &self,
        generation_id: Uuid,
        source_id: Uuid,
        reason: &str,
    ) -> RepoResult<()>;

    /// Repoint a source at its previous complete generation in the same
    /// embedding space as the active generation.
    ///
    /// Returns `None` when there is no compatible earlier published generation
    /// to return to. Copies nothing.
    async fn rollback_to_previous(&self, source_id: Uuid) -> RepoResult<Option<Uuid>>;

    /// The ids of every generation of a source, newest first.
    ///
    /// The inspection a reclaim or rollback is checked against: which
    /// generations still exist, in the order they were created.
    async fn list_for_source(&self, source_id: Uuid) -> RepoResult<Vec<Uuid>>;

    /// Delete unreferenced generations older than the retention window.
    ///
    /// Locks the source through selection and deletion, never touches the
    /// active generation or newest compatible rollback target, and reports how
    /// many rows it actually removed.
    async fn reclaim(&self, source_id: Uuid, retention_hours: i32) -> RepoResult<u64>;

    /// Fail building generations abandoned by a process that is gone.
    async fn fail_stale_builds(&self, older_than_secs: i64, reason: &str) -> RepoResult<u64>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Chunk (Vector Store)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a similarity search.
#[must_use]
#[derive(Debug, Clone)]
pub struct ChunkSearchResult {
    pub id: Uuid,
    /// The generation this chunk belongs to.
    ///
    /// Always the source's active generation: every query that produces a
    /// `ChunkSearchResult` joins on the active pointer. Carried through to the
    /// retrieval trace so an operator can tell which index answered a question
    /// (US-004, EP-002).
    pub generation_id: Uuid,
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

/// Chunk access, scoped to index generations (EP-002).
///
/// There is no operation here that deletes a source's chunks. That is the
/// point: FR-03 forbids removing the active index before its replacement is
/// published, so the only way chunks disappear is a generation being reclaimed
/// once nothing references it. Reads resolve through
/// `sources.active_generation_id` and never see a building generation.
#[async_trait]
pub trait ChunkRepository: Send + Sync {
    /// Write a whole generation's chunks in one transaction.
    ///
    /// Convenience over [`store_chunk_batch`](Self::store_chunk_batch) for
    /// callers that already hold every chunk. Same upsert semantics: a position
    /// already present in this generation is overwritten, never duplicated. The
    /// generation must exist and still be building.
    async fn store_chunks(
        &self,
        generation_id: Uuid,
        source_id: Uuid,
        chunks: &[ChunkWithContext],
        embeddings: &[Vec<f32>],
    ) -> RepoResult<()>;

    /// Insert a batch within an existing transaction (no commit).
    ///
    /// `base_chunk_index` is the absolute index of `chunks[0]` in the full
    /// source. Writes are idempotent under
    /// `chunks_generation_chunk_index_unique`: a retried batch overwrites its
    /// own positions rather than creating duplicates.
    async fn store_chunk_batch(
        &self,
        generation_id: Uuid,
        source_id: Uuid,
        chunks: &[ChunkWithContext],
        embeddings: &[Vec<f32>],
        base_chunk_index: usize,
        txn: &sea_orm::DatabaseTransaction,
    ) -> RepoResult<()>;

    /// Returns `(chunk_id, chunk_index, content)` for the source's *active*
    /// generation, ordered by position.
    async fn get_for_source(&self, source_id: Uuid) -> RepoResult<Vec<(Uuid, i32, String)>>;

    /// A random sample of active-generation chunk contents for a notebook.
    async fn sample_chunks_for_notebook(
        &self,
        notebook_id: Uuid,
        limit: i32,
    ) -> RepoResult<Vec<String>>;

    /// Embeddings a new build may reuse, keyed by content hash.
    ///
    /// Only chunks from the source's active generation, and only when that
    /// generation's `embedding_fingerprint` equals the one the new build will
    /// use (US-011): identical text embedded by a different model is a
    /// different vector, and reusing it would mix two vector spaces.
    async fn get_reusable_embeddings(
        &self,
        source_id: Uuid,
        embedding_fingerprint: &str,
    ) -> RepoResult<Vec<(String, Vec<f32>)>>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Search (raw SQL: pgvector + tsvector)
// ─────────────────────────────────────────────────────────────────────────────

/// The account and notebook a retrieval is allowed to read.
///
/// Every search takes ownership as data instead of trusting a check that ran
/// earlier in the request. Between the handler's access check and the SQL there
/// is an embedding call, possibly a reformulation call and a reranker call;
/// ownership can be revoked in that window, and a query scoped only by notebook
/// would still return the content (US-020 AC-2, PRD edge case 8).
///
/// It is a struct rather than two parameters so that no implementation can
/// accept a notebook without an account.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotebookScope {
    pub account_id: Uuid,
    pub notebook_id: Uuid,
}

/// Dense and lexical candidates read from one database snapshot.
///
/// Hybrid retrieval is one logical read. Returning both branches from one
/// repository operation prevents a publication between two independent SQL
/// statements from feeding different generations into RRF.
#[derive(Debug, Default)]
pub struct HybridChunkSearchResult {
    pub dense: Vec<ChunkSearchResult>,
    pub lexical: Vec<ChunkSearchResult>,
}

impl NotebookScope {
    #[must_use]
    pub const fn new(account_id: Uuid, notebook_id: Uuid) -> Self {
        Self {
            account_id,
            notebook_id,
        }
    }
}

#[async_trait]
pub trait SearchRepository: Send + Sync {
    /// Owner-scoped vector similarity search using pgvector `<=>` operator.
    /// Active generations whose vectors do not match `embedding_fingerprint`
    /// are excluded before distance evaluation.
    async fn search_similar_chunks(
        &self,
        scope: NotebookScope,
        query_embedding: &[f32],
        embedding_fingerprint: &str,
        limit: i32,
    ) -> RepoResult<Vec<ChunkSearchResult>>;

    /// Owner-scoped full-text (lexical) search using PostgreSQL tsvector.
    async fn search_lexical_chunks(
        &self,
        scope: NotebookScope,
        query: &str,
        limit: i32,
    ) -> RepoResult<Vec<ChunkSearchResult>>;

    /// Owner-scoped dense and lexical search over one logical snapshot and one
    /// embedding fingerprint.
    ///
    /// In-memory repositories have no concurrent publication boundary, so the
    /// default implementation composes their two deterministic reads. Database
    /// implementations must override this with a shared repeatable-read
    /// transaction.
    async fn search_hybrid_chunks(
        &self,
        scope: NotebookScope,
        query_embedding: &[f32],
        embedding_fingerprint: &str,
        query: &str,
        limit: i32,
    ) -> RepoResult<HybridChunkSearchResult> {
        let dense = self
            .search_similar_chunks(scope, query_embedding, embedding_fingerprint, limit)
            .await?;
        let lexical = self.search_lexical_chunks(scope, query, limit).await?;
        Ok(HybridChunkSearchResult { dense, lexical })
    }

    /// Count the owner's active-generation chunks in a notebook.
    ///
    /// Used to decide whether reranking should be applied (skip for small notebooks).
    async fn count_chunks_for_notebook(&self, scope: NotebookScope) -> RepoResult<i64>;

    /// Count configured sources independently of active index generations.
    async fn count_sources_for_notebook(&self, scope: NotebookScope) -> RepoResult<i64>;

    /// Load all chunks for a notebook with source titles, ordered by document structure.
    ///
    /// Used by context stuffing to bypass the search pipeline for small notebooks.
    /// All results have `relevance_score = 1.0` (equally relevant when stuffing).
    async fn get_all_chunks_for_notebook(
        &self,
        scope: NotebookScope,
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

    /// Persist a message inside a transaction that already pins every source
    /// generation referenced by its citations.
    #[allow(clippy::too_many_arguments)]
    async fn create_message_in_transaction(
        &self,
        transaction: &sea_orm::DatabaseTransaction,
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
