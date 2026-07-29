//! Repository layer for data access abstraction.
//!
//! Implements the repository pattern to abstract database access from business
//! logic. Every table named here belongs to the core schema; the hosted
//! product owns its own repositories, outside this module (US-013).

mod account;
mod chat;
mod chunk;
mod memory;
mod note;
mod notebook;
mod ocr_cache;
mod rag_log;
mod search;
mod source;
mod traits;

// Re-export traits and types
pub use traits::{
    AccountRepository, AccountSettingsRepository, ChatRepository, ChunkRepository,
    ChunkSearchResult, ChunkWithContext, DEFAULT_CHAT_HISTORY_LIMIT, MAX_CHAT_HISTORY_LIMIT,
    MemoryRepository, MemorySearchResult, NoteRepository, NotebookRepository,
    NotebookWithSourceCount, OcrCacheRepository, PaginatedChatHistory, RagLogRepository,
    RepoResult, SearchRepository, SourceRepository,
};

// Re-export SeaORM implementations
pub use account::{SeaOrmAccountRepository, SeaOrmAccountSettingsRepository};
pub use chat::SeaOrmChatRepository;
pub use chunk::SeaOrmChunkRepository;
pub use memory::SeaOrmMemoryRepository;
pub use note::SeaOrmNoteRepository;
pub use notebook::SeaOrmNotebookRepository;
pub use ocr_cache::SeaOrmOcrCacheRepository;
pub use rag_log::SeaOrmRagLogRepository;
pub use search::SeaOrmSearchRepository;
pub use source::SeaOrmSourceRepository;
