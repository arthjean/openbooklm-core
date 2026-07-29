//! Public core schema baseline (US-012, EP-003).
//!
//! The complete core schema for a **new** installation, in one migration. It is
//! not a replay of the legacy history: that history interleaves core tables
//! with Clerk, Stripe and campaign tables, and a public edition must not carry
//! them.
//!
//! An existing hosted database never runs this file. It is marked as satisfied
//! by the bridge, which records equivalence rather than recreating tables. The
//! two paths therefore differ in one documented way: here `notebooks.user_id`
//! and `rag_logs.user_id` reference `accounts(id)`, while a bridged legacy
//! database still references `users(id)`. Both hold the same UUIDs; repointing
//! the constraint is a contraction, deferred to a later PRD.
//!
//! Every future core schema change appends a migration after this one.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // pgvector is a hard prerequisite: retrieval is the product.
        db.execute_unprepared("CREATE EXTENSION IF NOT EXISTS vector;")
            .await?;

        db.execute_unprepared(
            r"
            CREATE TABLE IF NOT EXISTS accounts (
                id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS account_settings (
                account_id       UUID PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
                default_provider TEXT NOT NULL,
                default_model    TEXT NOT NULL,
                updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS notebooks (
                id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id             UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                title               VARCHAR(255) NOT NULL,
                description         TEXT,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                memory_enabled      BOOLEAN NOT NULL DEFAULT true,
                is_demo             BOOLEAN NOT NULL DEFAULT false,
                suggested_questions JSONB NOT NULL DEFAULT '[]'::jsonb
            );
            CREATE INDEX IF NOT EXISTS idx_notebooks_user_id ON notebooks(user_id);
            CREATE UNIQUE INDEX IF NOT EXISTS notebooks_user_demo_unique
                ON notebooks(user_id) WHERE (is_demo = true);

            CREATE TABLE IF NOT EXISTS sources (
                id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                notebook_id   UUID NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
                title         VARCHAR(255) NOT NULL,
                source_type   VARCHAR(50) NOT NULL,
                content       TEXT NOT NULL,
                metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
                chunk_count   INTEGER NOT NULL DEFAULT 0,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status        VARCHAR(20) NOT NULL DEFAULT 'pending',
                error_message TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_sources_notebook_id ON sources(notebook_id);
            CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);

            CREATE TABLE IF NOT EXISTS chunks (
                id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_id      UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                chunk_index    INTEGER NOT NULL,
                content        TEXT NOT NULL,
                created_at     TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                embedding      vector(1024),
                content_tsv    tsvector,
                context_prefix TEXT,
                metadata       JSONB DEFAULT '{}'::jsonb,
                content_hash   TEXT NOT NULL DEFAULT '',
                parent_content TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(source_id, content_hash);

            CREATE TABLE IF NOT EXISTS notes (
                id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                notebook_id         UUID NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
                title               VARCHAR(255) NOT NULL,
                content             TEXT NOT NULL,
                original_message_id UUID,
                created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_notes_notebook_id ON notes(notebook_id);

            CREATE TABLE IF NOT EXISTS chat_messages (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                notebook_id UUID NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
                role        VARCHAR(20) NOT NULL,
                content     TEXT NOT NULL,
                citations   JSONB NOT NULL DEFAULT '[]'::jsonb,
                model       VARCHAR(100),
                created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                agent_id    UUID,
                session_id  UUID
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_notebook_id ON chat_messages(notebook_id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_notebook_created
                ON chat_messages(notebook_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                ON chat_messages(session_id) WHERE (session_id IS NOT NULL);

            CREATE TABLE IF NOT EXISTS notebook_memories (
                id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                notebook_id UUID NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
                content     TEXT NOT NULL,
                memory_type VARCHAR(50) NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
                salience    REAL NOT NULL DEFAULT 1.0,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                embedding   vector(1024)
            );
            CREATE INDEX IF NOT EXISTS idx_notebook_memories_notebook_id
                ON notebook_memories(notebook_id);

            CREATE TABLE IF NOT EXISTS rag_logs (
                id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                notebook_id         UUID NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
                user_id             UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
                query               TEXT NOT NULL,
                reformulated_query  TEXT,
                hyde_document       TEXT,
                chunks_retrieved    JSONB DEFAULT '[]'::jsonb,
                response_id         UUID REFERENCES chat_messages(id) ON DELETE SET NULL,
                retrieval_score_avg REAL,
                context_relevance   REAL,
                answer_faithfulness REAL,
                answer_relevance    REAL,
                user_feedback       VARCHAR(20),
                created_at          TIMESTAMPTZ DEFAULT now(),
                model               VARCHAR(100),
                provider            VARCHAR(50)
            );
            CREATE INDEX IF NOT EXISTS idx_rag_logs_notebook ON rag_logs(notebook_id);
            CREATE INDEX IF NOT EXISTS idx_rag_logs_user ON rag_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_rag_logs_created ON rag_logs(created_at);

            CREATE TABLE IF NOT EXISTS ocr_cache (
                id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content_hash    VARCHAR(64) NOT NULL,
                model           VARCHAR(64) NOT NULL,
                ocr_text        TEXT NOT NULL,
                pages_processed INTEGER NOT NULL DEFAULT 0,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ocr_cache_hash_model
                ON ocr_cache(content_hash, model);
            ",
        )
        .await?;

        // Lexical half of hybrid search. `simple` rather than a language
        // configuration: the corpus is multilingual and stemming one language
        // degrades the others.
        db.execute_unprepared(
            r"
            CREATE OR REPLACE FUNCTION chunks_content_tsv_trigger() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                NEW.content_tsv := to_tsvector('simple',
                    COALESCE(NEW.context_prefix, '') || ' ' || COALESCE(NEW.content, '')
                );
                RETURN NEW;
            END;
            $$;

            DROP TRIGGER IF EXISTS chunks_content_tsv_update ON chunks;
            CREATE TRIGGER chunks_content_tsv_update
                BEFORE INSERT OR UPDATE OF content, context_prefix ON chunks
                FOR EACH ROW EXECUTE FUNCTION chunks_content_tsv_trigger();

            CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING gin(content_tsv);
            ",
        )
        .await?;

        // Dense half. HNSW parameters match the tuned legacy index; changing
        // them changes recall, so they are pinned here rather than defaulted.
        db.execute_unprepared(
            r"
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
                USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
            CREATE INDEX IF NOT EXISTS idx_notebook_memories_embedding ON notebook_memories
                USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);
            ",
        )
        .await?;

        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Reversible for a fresh install only. Never run this against a bridged
        // hosted database: the bridge recorded the baseline as satisfied without
        // creating these tables, so dropping them here would destroy production
        // data this migration never created. See docs/migrations.md.
        manager
            .get_connection()
            .execute_unprepared(
                r"
                DROP TRIGGER IF EXISTS chunks_content_tsv_update ON chunks;
                DROP FUNCTION IF EXISTS chunks_content_tsv_trigger();
                DROP TABLE IF EXISTS ocr_cache;
                DROP TABLE IF EXISTS rag_logs;
                DROP TABLE IF EXISTS notebook_memories;
                DROP TABLE IF EXISTS chat_messages;
                DROP TABLE IF EXISTS notes;
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS sources;
                DROP TABLE IF EXISTS notebooks;
                DROP TABLE IF EXISTS account_settings;
                DROP TABLE IF EXISTS accounts;
                ",
            )
            .await?;
        Ok(())
    }
}
