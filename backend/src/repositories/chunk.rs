//! SeaORM/raw SQL implementation of ChunkRepository for pgvector.
//!
//! Uses raw SQL because SeaORM doesn't natively support vector types and similarity operators.

use std::fmt::Write;

use async_trait::async_trait;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement, TransactionTrait};
use uuid::Uuid;

use crate::error::RagError;
use crate::services::rag::utils::format_embedding;
use crate::types::ChunkWithContext;

use super::traits::{ChunkRepository, RepoResult};

// SQL queries as constants for readability and reuse
const DELETE_CHUNKS_SQL: &str = "DELETE FROM chunks WHERE source_id = $1";

/// Number of bound parameter placeholders per row in the batch INSERT.
/// 9 bound params: id, source_id, chunk_index, content, context_prefix, parent_content,
/// metadata, embedding, content_hash. (created_at uses the SQL literal NOW().)
const PARAMS_PER_ROW: usize = 9;

/// Default batch size for multi-row INSERT.
/// 500 rows × 9 params = 4,500 parameters (well under PostgreSQL's 65,535 limit).
const BATCH_INSERT_SIZE: usize = 500;

const GET_CHUNKS_SQL: &str = r"
    SELECT id, chunk_index, content FROM chunks WHERE source_id = $1 ORDER BY chunk_index
";

const SAMPLE_CHUNKS_SQL: &str = r"
    SELECT c.content
    FROM chunks c
    JOIN sources s ON c.source_id = s.id
    WHERE s.notebook_id = $1
    ORDER BY RANDOM()
    LIMIT $2
";

const GET_CHUNKS_WITH_HASHES_SQL: &str = r"
    SELECT chunk_index, content_hash, embedding::text
    FROM chunks WHERE source_id = $1
    ORDER BY chunk_index
";

/// Insert a batch of (chunk, embedding) pairs into the `chunks` table using
/// the provided connection (transaction or bare connection).
///
/// `base_chunk_index` is the absolute index of `chunks[0]` within the full
/// source — used as the stored `chunk_index` value for each row.
async fn insert_chunk_rows(
    conn: &impl ConnectionTrait,
    source_id: Uuid,
    chunks: &[ChunkWithContext],
    embeddings: &[Vec<f32>],
    base_chunk_index: usize,
) -> RepoResult<()> {
    // Split into sub-batches of BATCH_INSERT_SIZE to stay under the PG parameter limit.
    for sub_batch in chunks
        .iter()
        .zip(embeddings)
        .enumerate()
        .collect::<Vec<_>>()
        .chunks(BATCH_INSERT_SIZE)
    {
        let row_count = sub_batch.len();

        let mut sql = String::from(
            "INSERT INTO chunks (id, source_id, chunk_index, content, context_prefix, parent_content, metadata, embedding, content_hash, created_at) VALUES ",
        );
        let mut values: Vec<sea_orm::Value> = Vec::with_capacity(row_count * PARAMS_PER_ROW);

        for (row_idx, &(offset, (chunk, embedding))) in sub_batch.iter().enumerate() {
            if row_idx > 0 {
                sql.push_str(", ");
            }
            let base = row_idx * PARAMS_PER_ROW;
            write!(
                sql,
                "(${}, ${}, ${}, ${}, ${}, ${}, ${}::jsonb, ${}::vector, ${}, NOW())",
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6,
                base + 7,
                base + 8,
                base + 9,
            )
            .expect("write to String");

            let chunk_index = base_chunk_index + offset;
            let context_prefix_val: sea_orm::Value = match &chunk.context_prefix {
                Some(prefix) => prefix.clone().into(),
                None => sea_orm::Value::String(None),
            };
            let parent_content_val: sea_orm::Value = match &chunk.parent_content {
                Some(parent) => String::from(&**parent).into(),
                None => sea_orm::Value::String(None),
            };
            let metadata_json = serde_json::to_string(&chunk.metadata).unwrap_or_else(|e| {
                tracing::warn!(
                    %source_id,
                    chunk_index,
                    error = %e,
                    "Failed to serialize chunk metadata, using empty object"
                );
                "{}".to_string()
            });

            values.extend([
                Uuid::new_v4().into(),
                source_id.into(),
                i32::try_from(chunk_index).unwrap_or(i32::MAX).into(),
                chunk.content.clone().into(),
                context_prefix_val,
                parent_content_val,
                metadata_json.into(),
                format_embedding(embedding).into(),
                chunk.content_hash.clone().into(),
            ]);
        }

        debug_assert_eq!(
            values.len(),
            row_count * PARAMS_PER_ROW,
            "BUG: values count ({}) != row_count ({}) * PARAMS_PER_ROW ({})",
            values.len(),
            row_count,
            PARAMS_PER_ROW,
        );

        conn.execute(Statement::from_sql_and_values(
            DbBackend::Postgres,
            &sql,
            values,
        ))
        .await?;
    }

    Ok(())
}

#[derive(Clone)]
pub struct SeaOrmChunkRepository {
    db: DatabaseConnection,
}

impl SeaOrmChunkRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    #[allow(clippy::unused_self)]
    fn stmt(&self, sql: &str, values: impl IntoIterator<Item = sea_orm::Value>) -> Statement {
        Statement::from_sql_and_values(DbBackend::Postgres, sql, values)
    }
}

#[async_trait]
impl ChunkRepository for SeaOrmChunkRepository {
    #[tracing::instrument(skip(self, chunks, embeddings), fields(%source_id, chunk_count = chunks.len()))]
    async fn store_chunks(
        &self,
        source_id: Uuid,
        chunks: &[ChunkWithContext],
        embeddings: &[Vec<f32>],
    ) -> RepoResult<()> {
        if chunks.len() != embeddings.len() {
            return Err(RagError::VectorStoreFailed {
                source_id: source_id.to_string(),
                reason: format!(
                    "Chunks/embeddings mismatch: {} chunks, {} embeddings",
                    chunks.len(),
                    embeddings.len()
                ),
            }
            .into());
        }

        if chunks.is_empty() {
            return Ok(());
        }

        // Wrap DELETE + INSERTs in a transaction so a failure mid-insert
        // doesn't leave the source with partial chunks.
        let txn = self.db.begin().await?;

        // Delete existing chunks (for reprocessing)
        txn.execute(self.stmt(DELETE_CHUNKS_SQL, [source_id.into()]))
            .await?;

        insert_chunk_rows(&txn, source_id, chunks, embeddings, 0).await?;

        txn.commit().await?;
        tracing::debug!(%source_id, count = chunks.len(), "Stored chunks with embeddings");
        Ok(())
    }

    #[tracing::instrument(skip(self, chunks, embeddings, txn), fields(%source_id, chunk_count = chunks.len(), base_chunk_index))]
    async fn store_chunk_batch(
        &self,
        source_id: Uuid,
        chunks: &[ChunkWithContext],
        embeddings: &[Vec<f32>],
        base_chunk_index: usize,
        txn: &sea_orm::DatabaseTransaction,
    ) -> RepoResult<()> {
        if chunks.len() != embeddings.len() {
            return Err(RagError::VectorStoreFailed {
                source_id: source_id.to_string(),
                reason: format!(
                    "Chunks/embeddings mismatch: {} chunks, {} embeddings",
                    chunks.len(),
                    embeddings.len()
                ),
            }
            .into());
        }

        if chunks.is_empty() {
            return Ok(());
        }

        insert_chunk_rows(txn, source_id, chunks, embeddings, base_chunk_index).await
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn get_for_source(&self, source_id: Uuid) -> RepoResult<Vec<(Uuid, i32, String)>> {
        let rows = self
            .db
            .query_all(self.stmt(GET_CHUNKS_SQL, [source_id.into()]))
            .await?;

        rows.into_iter()
            .map(|row| {
                Ok((
                    row.try_get("", "id")?,
                    row.try_get("", "chunk_index")?,
                    row.try_get("", "content")?,
                ))
            })
            .collect::<Result<_, sea_orm::DbErr>>()
            .map_err(Into::into)
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn get_chunks_with_hashes(
        &self,
        source_id: Uuid,
    ) -> RepoResult<Vec<(i32, String, Vec<f32>)>> {
        let rows = self
            .db
            .query_all(self.stmt(GET_CHUNKS_WITH_HASHES_SQL, [source_id.into()]))
            .await?;

        rows.into_iter()
            .map(|row| {
                let chunk_index: i32 = row.try_get("", "chunk_index")?;
                let content_hash: String = row.try_get("", "content_hash")?;
                let embedding_text: String = row.try_get("", "embedding")?;
                let embedding = crate::services::rag::utils::parse_embedding(&embedding_text);
                Ok((chunk_index, content_hash, embedding))
            })
            .collect::<Result<_, sea_orm::DbErr>>()
            .map_err(Into::into)
    }

    #[tracing::instrument(skip(self), fields(%source_id))]
    async fn delete_for_source(&self, source_id: Uuid) -> RepoResult<u64> {
        Ok(self
            .db
            .execute(self.stmt(DELETE_CHUNKS_SQL, [source_id.into()]))
            .await?
            .rows_affected())
    }

    #[tracing::instrument(skip(self), fields(%notebook_id, %limit))]
    async fn sample_chunks_for_notebook(
        &self,
        notebook_id: Uuid,
        limit: i32,
    ) -> RepoResult<Vec<String>> {
        let rows = self
            .db
            .query_all(self.stmt(SAMPLE_CHUNKS_SQL, [notebook_id.into(), limit.into()]))
            .await?;

        rows.into_iter()
            .map(|row| row.try_get("", "content"))
            .collect::<Result<_, sea_orm::DbErr>>()
            .map_err(Into::into)
    }
}
