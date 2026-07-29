//! Vector store types and re-exports.
//!
//! All raw SQL search queries have been moved to [`crate::repositories::search`]
//! (`SeaOrmSearchRepository`). This module retains only re-exports for backward
//! compatibility.

// Re-export from canonical locations.
pub use crate::repositories::ChunkSearchResult;
pub use crate::types::{ChunkMetadata, ChunkWithContext};
