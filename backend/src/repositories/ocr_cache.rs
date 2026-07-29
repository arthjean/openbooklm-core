//! SeaORM/raw SQL implementation of OcrCacheRepository.
//!
//! Content-hash caching for OCR results. Keyed by (SHA-256 hash of PDF bytes, model).

use async_trait::async_trait;
use sea_orm::{ConnectionTrait, DatabaseConnection, DbBackend, Statement};

use super::traits::{OcrCacheRepository, RepoResult};

// ============================================================================
// SQL constants
// ============================================================================

const FIND_BY_HASH_SQL: &str = r"
    SELECT ocr_text, pages_processed
    FROM ocr_cache
    WHERE content_hash = $1 AND model = $2
    LIMIT 1
";

const INSERT_SQL: &str = r"
    INSERT INTO ocr_cache (content_hash, model, ocr_text, pages_processed)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT (content_hash, model) DO NOTHING
";

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
    #[tracing::instrument(skip(self), fields(%content_hash, %model))]
    async fn find_by_hash(
        &self,
        content_hash: &str,
        model: &str,
    ) -> RepoResult<Option<(String, i32)>> {
        let row = self
            .db
            .query_one(self.stmt(FIND_BY_HASH_SQL, [content_hash.into(), model.into()]))
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

    #[tracing::instrument(skip(self, ocr_text), fields(%content_hash, %model, %pages_processed))]
    async fn store(
        &self,
        content_hash: &str,
        model: &str,
        ocr_text: &str,
        pages_processed: i32,
    ) -> RepoResult<()> {
        self.db
            .execute(self.stmt(
                INSERT_SQL,
                [
                    content_hash.into(),
                    model.into(),
                    ocr_text.into(),
                    pages_processed.into(),
                ],
            ))
            .await?;
        Ok(())
    }
}
