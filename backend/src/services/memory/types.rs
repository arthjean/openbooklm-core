//! Memory types: `ExtractedMemory`, `MemoryAction`, `DecayResult`.

use uuid::Uuid;

/// A memory fact extracted from a chat exchange, before embedding & storage.
#[derive(Debug, Clone)]
pub struct ExtractedMemory {
    pub content: String,
    pub memory_type: String,
    pub salience: f32,
    pub metadata: serde_json::Value,
    /// Context from the LLM extraction — explains why this memory was extracted.
    pub context: Option<String>,
}

/// What to do with an extracted memory after deduplication check.
#[derive(Debug)]
pub enum MemoryAction {
    /// Insert as a new memory.
    Insert {
        memory: ExtractedMemory,
        embedding: Vec<f32>,
    },
    /// Update an existing memory (similarity > 0.9).
    Update {
        existing_id: Uuid,
        new_content: String,
        new_salience: f32,
        embedding: Vec<f32>,
    },
    /// Skip — near-exact duplicate (similarity > 0.97).
    Skip,
}

/// Result of running temporal decay on a notebook's memories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecayResult {
    /// Number of memories whose salience was reduced (includes those
    /// subsequently deleted by the threshold check).
    pub decayed_count: usize,
    pub deleted_count: usize,
}
