//! SeaORM/raw SQL implementation of OcrCacheRepository.
//!
//! Source-owned content-hash caching for OCR results.

use async_trait::async_trait;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement};

use super::traits::{OcrCacheRepository, RepoResult};

// ============================================================================
// SQL constants
// ============================================================================

const FIND_BY_HASH_SQL: &str = r"
    SELECT ocr_text, pages_processed
    FROM ocr_cache
    WHERE source_id = $1 AND content_hash = $2 AND model = $3
    LIMIT 1
";

const INSERT_SQL: &str = r"
    WITH evicted AS (
        DELETE FROM ocr_cache
        WHERE source_id = $1 AND (content_hash <> $2 OR model <> $3)
    )
    INSERT INTO ocr_cache (source_id, content_hash, model, ocr_text, pages_processed)
    VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (content_hash, model) DO UPDATE
        SET source_id = EXCLUDED.source_id
";

const PURGE_UNOWNED_SQL: &str = "DELETE FROM ocr_cache WHERE source_id IS NULL";

// ============================================================================
// Implementation
// ============================================================================

#[derive(Clone)]
pub struct SeaOrmOcrCacheRepository {
    db: DatabaseConnection,
}

impl SeaOrmOcrCacheRepository {
    pub fn new(db: &DatabaseConnection) -> Self {
        Self { db: db.clone() }
    }

    #[allow(clippy::unused_self)]
    fn stmt(&self, sql: &str, values: impl IntoIterator<Item = sea_orm::Value>) -> Statement {
        Statement::from_sql_and_values(DbBackend::Postgres, sql, values)
    }
}

#[async_trait]
impl OcrCacheRepository for SeaOrmOcrCacheRepository {
    #[tracing::instrument(skip(self), fields(%source_id, %content_hash, %model))]
    async fn find_by_hash(
        &self,
        source_id: uuid::Uuid,
        content_hash: &str,
        model: &str,
    ) -> RepoResult<Option<(String, i32)>> {
        let row = self
            .db
            .query_one(self.stmt(
                FIND_BY_HASH_SQL,
                [source_id.into(), content_hash.into(), model.into()],
            ))
            .await?;

        match row {
            Some(r) => {
                let ocr_text: String = r.try_get("", "ocr_text")?;
                let pages_processed: i32 = r.try_get("", "pages_processed")?;
                Ok(Some((ocr_text, pages_processed)))
            }
            None => Ok(None),
        }
    }

    #[tracing::instrument(
        skip(self, ocr_text),
        fields(%source_id, %content_hash, %model, %pages_processed)
    )]
    async fn store(
        &self,
        source_id: uuid::Uuid,
        content_hash: &str,
        model: &str,
        ocr_text: &str,
        pages_processed: i32,
    ) -> RepoResult<()> {
        self.db
            .execute(self.stmt(
                INSERT_SQL,
                [
                    source_id.into(),
                    content_hash.into(),
                    model.into(),
                    ocr_text.into(),
                    pages_processed.into(),
                ],
            ))
            .await?;
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn purge_unowned(&self) -> RepoResult<u64> {
        Ok(self
            .db
            .execute(self.stmt(PURGE_UNOWNED_SQL, []))
            .await?
            .rows_affected())
    }
}
