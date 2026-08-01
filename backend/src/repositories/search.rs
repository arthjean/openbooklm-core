//! SeaORM/raw SQL implementation of SearchRepository.
//!
//! Encapsulates all search-related raw SQL queries (pgvector similarity,
//! PostgreSQL full-text search) that cannot be expressed via SeaORM's query builder.
//!
//! ## Active-generation scope (US-008, EP-002)
//!
//! Every query below joins `sources` on **both** `id` and
//! `active_generation_id = chunks.generation_id`. That single extra equality is
//! the whole isolation guarantee: a replacement generation accumulating rows in
//! the same table is unreachable until its publication transaction moves the
//! pointer, and the move is atomic, so no query can observe a mixture. The join
//! also excludes sources with no active generation, which is what an unindexed
//! or failed-first-build source is.
//!
//! It is written as a join predicate rather than a `WHERE` clause on purpose:
//! there is no way to add a filter to one of these queries and forget the
//! generation, because the generation is part of how `sources` is reached.
//!
//! ## Owner scope (US-020)
//!
//! Every query also joins `notebooks` on `user_id`. The handler already checked
//! access, but that check and this query are separated by an embedding call, a
//! possible reformulation call and a reranker call; a notebook whose ownership
//! changed in that window must stop returning content, and the only place that
//! can be true of is the query itself (PRD edge case 8). The scope arrives as a
//! [`NotebookScope`](super::traits::NotebookScope), which cannot be built
//! without an account.

use std::sync::LazyLock;

use async_trait::async_trait;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement, TransactionTrait};

use crate::services::rag::utils::{format_embedding, sanitize_tsquery};

use super::ann::APPROVED_STRATEGY;
use super::traits::{ChunkSearchResult, NotebookScope, RepoResult, SearchRepository};

/// The approved strategy's `SET LOCAL` statements, rendered once.
///
/// The strategy is a constant, so the statements are too. Formatting them per
/// query allocated three strings on the hottest path in the system to produce
/// the same text every time.
static SCAN_PREAMBLE: LazyLock<String> = LazyLock::new(|| APPROVED_STRATEGY.session_preamble());

// ============================================================================
// SQL constants
// ============================================================================

const SEARCH_SIMILAR_SQL: &str = r"
    SELECT c.id, c.generation_id, c.source_id, c.chunk_index, c.content, c.parent_content,
           c.metadata,
           s.title as source_title,
           (c.embedding <=> $1::vector) as distance
    FROM chunks c
    JOIN sources s ON c.source_id = s.id AND s.active_generation_id = c.generation_id
    JOIN notebooks n ON n.id = s.notebook_id AND n.user_id = $3
    WHERE s.notebook_id = $2
    ORDER BY c.embedding <=> $1::vector
    LIMIT $4
";

// The scan strategy and its parameters live in [`super::ann`], where the
// benchmark that selected them and the capability probe that guards them also
// live (US-016).

const SEARCH_LEXICAL_SQL: &str = r"
    SELECT c.id, c.generation_id, c.source_id, c.chunk_index, c.content, c.parent_content,
           c.metadata,
           s.title as source_title,
           ts_rank_cd(c.content_tsv, plainto_tsquery('simple', $1)) as rank
    FROM chunks c
    JOIN sources s ON c.source_id = s.id AND s.active_generation_id = c.generation_id
    JOIN notebooks n ON n.id = s.notebook_id AND n.user_id = $3
    WHERE s.notebook_id = $2
      AND c.content_tsv @@ plainto_tsquery('simple', $1)
    ORDER BY rank DESC
    LIMIT $4
";

const COUNT_CHUNKS_FOR_NOTEBOOK_SQL: &str = r"
    SELECT COUNT(*) as total
    FROM chunks c
    JOIN sources s ON c.source_id = s.id AND s.active_generation_id = c.generation_id
    JOIN notebooks n ON n.id = s.notebook_id AND n.user_id = $2
    WHERE s.notebook_id = $1
";

const GET_ALL_CHUNKS_FOR_NOTEBOOK_SQL: &str = r"
    SELECT c.id, c.generation_id, c.source_id, c.chunk_index, c.content, c.parent_content,
           s.title as source_title
    FROM chunks c
    JOIN sources s ON c.source_id = s.id AND s.active_generation_id = c.generation_id
    JOIN notebooks n ON n.id = s.notebook_id AND n.user_id = $2
    WHERE s.notebook_id = $1
    ORDER BY s.id, c.chunk_index
";

// ============================================================================
// Implementation
// ============================================================================

#[derive(Clone)]
pub struct SeaOrmSearchRepository {
    db: DatabaseConnection,
}

impl SeaOrmSearchRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    #[allow(clippy::unused_self)]
    fn stmt(&self, sql: &str, values: impl IntoIterator<Item = sea_orm::Value>) -> Statement {
        Statement::from_sql_and_values(DbBackend::Postgres, sql, values)
    }
}

#[async_trait]
impl SearchRepository for SeaOrmSearchRepository {
    #[tracing::instrument(skip(self, query_embedding), fields(notebook_id = %scope.notebook_id, %limit))]
    async fn search_similar_chunks(
        &self,
        scope: NotebookScope,
        query_embedding: &[f32],
        limit: i32,
    ) -> RepoResult<Vec<ChunkSearchResult>> {
        let embedding_str = format_embedding(query_embedding);

        // Use a transaction to scope SET LOCAL: the settings revert on commit
        // or rollback, so a scan mode chosen for this query cannot leak into
        // the next borrower of this pooled connection (US-016). All of them go
        // out in one statement, so the scoping costs one round trip and not
        // one per setting.
        let txn = self.db.begin().await?;

        if !SCAN_PREAMBLE.is_empty() {
            txn.execute_unprepared(&SCAN_PREAMBLE).await?;
        }

        let rows = txn
            .query_all(Statement::from_sql_and_values(
                DbBackend::Postgres,
                SEARCH_SIMILAR_SQL,
                [
                    embedding_str.into(),
                    scope.notebook_id.into(),
                    scope.account_id.into(),
                    limit.into(),
                ],
            ))
            .await?;

        txn.commit().await?;

        rows.into_iter()
            .map(|row| {
                let distance: f64 = row.try_get("", "distance")?;
                #[allow(clippy::cast_possible_truncation)]
                Ok(ChunkSearchResult {
                    id: row.try_get("", "id")?,
                    generation_id: row.try_get("", "generation_id")?,
                    source_id: row.try_get("", "source_id")?,
                    chunk_index: row.try_get("", "chunk_index")?,
                    content: row.try_get("", "content")?,
                    parent_content: row.try_get("", "parent_content")?,
                    source_title: row.try_get("", "source_title")?,
                    relevance_score: (1.0 - distance).max(0.0) as f32,
                    metadata: row.try_get("", "metadata").ok(),
                })
            })
            .collect()
    }

    #[tracing::instrument(skip(self), fields(notebook_id = %scope.notebook_id, %limit))]
    async fn search_lexical_chunks(
        &self,
        scope: NotebookScope,
        query: &str,
        limit: i32,
    ) -> RepoResult<Vec<ChunkSearchResult>> {
        let sanitized = sanitize_tsquery(query);
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        let rows = self
            .db
            .query_all(self.stmt(
                SEARCH_LEXICAL_SQL,
                [
                    sanitized.into(),
                    scope.notebook_id.into(),
                    scope.account_id.into(),
                    limit.into(),
                ],
            ))
            .await?;

        rows.into_iter()
            .map(|row| {
                let rank: f32 = row.try_get("", "rank")?;
                Ok(ChunkSearchResult {
                    id: row.try_get("", "id")?,
                    generation_id: row.try_get("", "generation_id")?,
                    source_id: row.try_get("", "source_id")?,
                    chunk_index: row.try_get("", "chunk_index")?,
                    content: row.try_get("", "content")?,
                    parent_content: row.try_get("", "parent_content")?,
                    source_title: row.try_get("", "source_title")?,
                    relevance_score: rank.min(1.0),
                    metadata: row.try_get("", "metadata").ok(),
                })
            })
            .collect()
    }

    #[tracing::instrument(skip(self), fields(notebook_id = %scope.notebook_id))]
    async fn count_chunks_for_notebook(&self, scope: NotebookScope) -> RepoResult<i64> {
        let rows = self
            .db
            .query_all(self.stmt(
                COUNT_CHUNKS_FOR_NOTEBOOK_SQL,
                [scope.notebook_id.into(), scope.account_id.into()],
            ))
            .await?;

        let total: i64 = rows
            .first()
            .and_then(|r| r.try_get::<i64>("", "total").ok())
            .unwrap_or(0);

        Ok(total)
    }

    #[tracing::instrument(skip(self), fields(notebook_id = %scope.notebook_id))]
    async fn get_all_chunks_for_notebook(
        &self,
        scope: NotebookScope,
    ) -> RepoResult<Vec<ChunkSearchResult>> {
        let rows = self
            .db
            .query_all(self.stmt(
                GET_ALL_CHUNKS_FOR_NOTEBOOK_SQL,
                [scope.notebook_id.into(), scope.account_id.into()],
            ))
            .await?;

        rows.into_iter()
            .map(|row| {
                Ok(ChunkSearchResult {
                    id: row.try_get("", "id")?,
                    generation_id: row.try_get("", "generation_id")?,
                    source_id: row.try_get("", "source_id")?,
                    chunk_index: row.try_get("", "chunk_index")?,
                    content: row.try_get("", "content")?,
                    parent_content: row.try_get("", "parent_content")?,
                    source_title: row.try_get("", "source_title")?,
                    relevance_score: 1.0,
                    metadata: None,
                })
            })
            .collect()
    }
}
