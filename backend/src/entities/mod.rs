//! SeaORM models for the core schema.
//!
//! Every table here exists in a self-hosted installation: the account that owns
//! the data, its core preferences, and the notebook, source, chunk, note, chat,
//! memory, RAG-log and OCR-cache tables the product is made of.
//!
//! Identity, subscription, usage and campaign models belong to the hosted
//! composition and are not part of this schema. This module never references
//! them (US-013).

/// Generates a `Related<$entity>` impl for the current module's `Entity`.
///
/// # Usage
/// ```ignore
/// impl_related!(super::account::Entity, Relation::Account);
/// impl_related!(super::source::Entity, Relation::Sources);
/// ```
macro_rules! impl_related {
    ($entity:path, $relation:expr) => {
        impl Related<$entity> for Entity {
            fn to() -> RelationDef {
                $relation.def()
            }
        }
    };
}

pub(crate) use impl_related;

pub mod account;
pub mod account_settings;
pub mod chat_message;
pub mod chunk;
pub mod note;
pub mod notebook;
pub mod notebook_memory;
pub mod ocr_cache;
pub mod rag_log;
pub mod source;

pub use account::Entity as Account;
pub use account_settings::Entity as AccountSettings;
pub use chat_message::Entity as ChatMessage;
pub use chunk::Entity as Chunk;
pub use note::Entity as Note;
pub use notebook::Entity as Notebook;
pub use notebook_memory::Entity as NotebookMemory;
pub use ocr_cache::Entity as OcrCache;
pub use rag_log::Entity as RagLog;
pub use source::Entity as Source;
