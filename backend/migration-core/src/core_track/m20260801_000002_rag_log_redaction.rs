//! Structural redaction for RAG interaction logs (US-004).
//!
//! The retrieval trace already carried query hashes, but the legacy `rag_logs`
//! table still retained raw query, reformulation and HyDE text. New code writes
//! hashes only. The trigger is the write-compatibility boundary: an older
//! binary may still name the legacy columns, but PostgreSQL discards their
//! values before the row reaches storage.

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
                ALTER TABLE rag_logs
                    ADD COLUMN IF NOT EXISTS query_hash TEXT,
                    ADD COLUMN IF NOT EXISTS reformulated_query_hash TEXT;

                UPDATE rag_logs
                   SET query = '',
                       reformulated_query = NULL,
                       hyde_document = NULL,
                       query_hash = COALESCE(query_hash, '');

                ALTER TABLE rag_logs ALTER COLUMN query SET DEFAULT '';
                ALTER TABLE rag_logs ALTER COLUMN query_hash SET DEFAULT '';
                ALTER TABLE rag_logs ALTER COLUMN query_hash SET NOT NULL;

                CREATE OR REPLACE FUNCTION scrub_rag_log_raw_text()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    NEW.query := '';
                    NEW.reformulated_query := NULL;
                    NEW.hyde_document := NULL;
                    RETURN NEW;
                END;
                $$;

                DROP TRIGGER IF EXISTS rag_logs_scrub_raw_text ON rag_logs;
                CREATE TRIGGER rag_logs_scrub_raw_text
                    BEFORE INSERT OR UPDATE OF query, reformulated_query, hyde_document
                    ON rag_logs
                    FOR EACH ROW
                    EXECUTE FUNCTION scrub_rag_log_raw_text();
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
                DROP TRIGGER IF EXISTS rag_logs_scrub_raw_text ON rag_logs;
                DROP FUNCTION IF EXISTS scrub_rag_log_raw_text();
                ALTER TABLE rag_logs ALTER COLUMN query DROP DEFAULT;
                ALTER TABLE rag_logs DROP COLUMN IF EXISTS reformulated_query_hash;
                ALTER TABLE rag_logs DROP COLUMN IF EXISTS query_hash;
                ",
            )
            .await?;
        Ok(())
    }
}
