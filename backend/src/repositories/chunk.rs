//! SeaORM/raw SQL implementation of ChunkRepository for pgvector.
//!
//! Uses raw SQL because SeaORM doesn't natively support vector types and similarity operators.
//!
//! Every read joins `sources.active_generation_id` (EP-002). A chunk that
//! belongs to a building or superseded generation exists in the table and is
//! invisible to every query here — that is what makes a replacement build
//! harmless while it runs.

use std::fmt::Write;

use async_trait::async_trait;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement, TransactionTrait};
use uuid::Uuid;

use crate::error::RagError;
use crate::services::rag::utils::format_embedding;
use crate::types::ChunkWithContext;

use super::traits::{ChunkRepository, RepoResult};

// SQL queries as constants for readability and reuse

/// A writer holds a shared row lock for its whole batch transaction.
/// Publication takes the conflicting update lock before validation, so no
/// chunk can arrive between validation and the active-pointer move.
const LOCK_BUILDING_GENERATION_SQL: &str = r"
    SELECT id
    FROM source_index_generations
    WHERE id = $1 AND source_id = $2 AND state = 'building'
    FOR SHARE
";

/// Number of bound parameter placeholders per row in the batch INSERT.
/// 10 bound params: id, generation_id, source_id, chunk_index, content,
/// context_prefix, parent_content, metadata, embedding, content_hash.
/// (created_at uses the SQL literal NOW().)
const PARAMS_PER_ROW: usize = 10;

/// Default batch size for multi-row INSERT.
/// 500 rows × 10 params = 5,000 parameters (well under PostgreSQL's 65,535 limit).
const BATCH_INSERT_SIZE: usize = 500;

const GET_CHUNKS_SQL: &str = r"
    SELECT c.id, c.chunk_index, c.content
    FROM chunks c
    JOIN sources s ON s.id = c.source_id AND s.active_generation_id = c.generation_id
    WHERE c.source_id = $1
    ORDER BY c.chunk_index
";

const SAMPLE_CHUNKS_SQL: &str = r"
    SELECT c.content
    FROM chunks c
    JOIN sources s ON s.id = c.source_id AND s.active_generation_id = c.generation_id
    WHERE s.notebook_id = $1
    ORDER BY RANDOM()
    LIMIT $2
";

/// Embeddings eligible for reuse: active generation, matching fingerprint.
///
/// The fingerprint predicate is on the *generation*, not on the chunk: a chunk
/// carries no provenance of its own, it inherits the generation's, and that is
/// the whole reason generations exist.
const GET_REUSABLE_EMBEDDINGS_SQL: &str = r"
    SELECT c.content_hash, c.embedding::text AS embedding
    FROM chunks c
    JOIN sources s ON s.id = c.source_id AND s.active_generation_id = c.generation_id
    JOIN source_index_generations g ON g.id = c.generation_id
    WHERE c.source_id = $1
      AND g.embedding_fingerprint = $2
      AND c.content_hash <> ''
      AND c.embedding IS NOT NULL
";

/// Insert a batch of (chunk, embedding) pairs into the `chunks` table using
/// the provided connection (transaction or bare connection).
///
/// `base_chunk_index` is the absolute index of `chunks[0]` within the full
/// source — used as the stored `chunk_index` value for each row.
///
/// `ON CONFLICT (generation_id, chunk_index)` is what makes a retried batch
/// idempotent: the same position inside the same generation is overwritten with
/// the same content rather than duplicated. Position is the identity here;
/// the row's own `id` is not, which is why it is excluded from the update.
async fn insert_chunk_rows(
    conn: &impl ConnectionTrait,
    generation_id: Uuid,
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
            "INSERT INTO chunks (id, generation_id, source_id, chunk_index, content, context_prefix, parent_content, metadata, embedding, content_hash, created_at) VALUES ",
        );
        let mut values: Vec<sea_orm::Value> = Vec::with_capacity(row_count * PARAMS_PER_ROW);

        for (row_idx, &(offset, (chunk, embedding))) in sub_batch.iter().enumerate() {
            if row_idx > 0 {
                sql.push_str(", ");
            }
            let base = row_idx * PARAMS_PER_ROW;
            write!(
                sql,
                "(${}, ${}, ${}, ${}, ${}, ${}, ${}, ${}::jsonb, ${}::vector, ${}, NOW())",
                base + 1,
                base + 2,
                base + 3,
                base + 4,
                base + 5,
                base + 6,
                base + 7,
                base + 8,
                base + 9,
                base + 10,
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
                generation_id.into(),
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

        sql.push_str(
            " ON CONFLICT (generation_id, chunk_index) DO UPDATE SET \
             content = EXCLUDED.content, \
             context_prefix = EXCLUDED.context_prefix, \
             parent_content = EXCLUDED.parent_content, \
             metadata = EXCLUDED.metadata, \
             embedding = EXCLUDED.embedding, \
             content_hash = EXCLUDED.content_hash",
        );

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

async fn lock_building_generation(
    conn: &impl ConnectionTrait,
    generation_id: Uuid,
    source_id: Uuid,
) -> RepoResult<()> {
    let row = conn
        .query_one(Statement::from_sql_and_values(
            DbBackend::Postgres,
            LOCK_BUILDING_GENERATION_SQL,
            [generation_id.into(), source_id.into()],
        ))
        .await?;
    if row.is_none() {
        return Err(RagError::VectorStoreFailed {
            source_id: source_id.to_string(),
            reason: "index generation is not owned by this source or is no longer building"
                .to_owned(),
        }
        .into());
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

/// Reject a chunk/embedding pairing that cannot be stored.
fn check_pairing(source_id: Uuid, chunks: usize, embeddings: usize) -> RepoResult<()> {
    if chunks != embeddings {
        return Err(RagError::VectorStoreFailed {
            source_id: source_id.to_string(),
            reason: format!("Chunks/embeddings mismatch: {chunks} chunks, {embeddings} embeddings"),
        }
        .into());
    }
    Ok(())
}

#[async_trait]
impl ChunkRepository for SeaOrmChunkRepository {
    #[tracing::instrument(skip(self, chunks, embeddings), fields(%generation_id, %source_id, chunk_count = chunks.len()))]
    async fn store_chunks(
        &self,
        generation_id: Uuid,
        source_id: Uuid,
        chunks: &[ChunkWithContext],
        embeddings: &[Vec<f32>],
    ) -> RepoResult<()> {
        check_pairing(source_id, chunks.len(), embeddings.len())?;
        if chunks.is_empty() {
            return Ok(());
        }

        let txn = self.db.begin().await?;
        lock_building_generation(&txn, generation_id, source_id).await?;
        insert_chunk_rows(&txn, generation_id, source_id, chunks, embeddings, 0).await?;
        txn.commit().await?;
        tracing::debug!(%source_id, count = chunks.len(), "Stored chunks with embeddings");
        Ok(())
    }

    #[tracing::instrument(skip(self, chunks, embeddings, txn), fields(%generation_id, %source_id, chunk_count = chunks.len(), base_chunk_index))]
    async fn store_chunk_batch(
        &self,
        generation_id: Uuid,
        source_id: Uuid,
        chunks: &[ChunkWithContext],
        embeddings: &[Vec<f32>],
        base_chunk_index: usize,
        txn: &sea_orm::DatabaseTransaction,
    ) -> RepoResult<()> {
        check_pairing(source_id, chunks.len(), embeddings.len())?;
        if chunks.is_empty() {
            return Ok(());
        }

        lock_building_generation(txn, generation_id, source_id).await?;
        insert_chunk_rows(
            txn,
            generation_id,
            source_id,
            chunks,
            embeddings,
            base_chunk_index,
        )
        .await
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

    #[tracing::instrument(skip(self), fields(%source_id, %embedding_fingerprint))]
    async fn get_reusable_embeddings(
        &self,
        source_id: Uuid,
        embedding_fingerprint: &str,
    ) -> RepoResult<Vec<(String, Vec<f32>)>> {
        let rows = self
            .db
            .query_all(self.stmt(
                GET_REUSABLE_EMBEDDINGS_SQL,
                [source_id.into(), embedding_fingerprint.into()],
            ))
            .await?;

        rows.into_iter()
            .map(|row| {
                let content_hash: String = row.try_get("", "content_hash")?;
                let embedding_text: String = row.try_get("", "embedding")?;
                let embedding = crate::services::rag::utils::parse_embedding(&embedding_text);
                Ok((content_hash, embedding))
            })
            .collect::<Result<_, sea_orm::DbErr>>()
            .map_err(Into::into)
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
