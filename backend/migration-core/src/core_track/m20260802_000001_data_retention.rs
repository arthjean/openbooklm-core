//! Source-owned OCR cache entries.
//!
//! The legacy cache keyed full OCR text only by `(content_hash, model)`, so the
//! derived document survived deletion of every source that produced it. Add an
//! optional owner first: current writers always provide it and source deletion
//! cascades, while an older binary can still write a nullable legacy entry
//! during a rolling deployment. The cache is derived data, so existing unowned
//! rows are discarded rather than retained without a deletion boundary.

use sea_orm_migration::prelude::*;
use sea_orm_migration::sea_orm::ConnectionTrait;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared(
                r"
                ALTER TABLE ocr_cache
                    ADD COLUMN IF NOT EXISTS source_id UUID
                        REFERENCES sources(id) ON DELETE CASCADE;

                DELETE FROM ocr_cache WHERE source_id IS NULL;

                CREATE UNIQUE INDEX IF NOT EXISTS ocr_cache_source_unique
                    ON ocr_cache(source_id)
                    WHERE source_id IS NOT NULL;
                ",
            )
            .await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared(
                r"
                DROP INDEX IF EXISTS ocr_cache_source_unique;
                ALTER TABLE ocr_cache DROP COLUMN IF EXISTS source_id;
                ",
            )
            .await?;
        Ok(())
    }
}
