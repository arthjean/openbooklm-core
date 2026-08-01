//! Semantic provenance for index generations (US-011, EP-002).
//!
//! An embedding is only comparable to another embedding produced by the same
//! provider, model, width and normalization; a chunk position only means the
//! same thing under the same chunking schema, size unit and sizer. Those two
//! facts are what a *fingerprint* records, and every generation stores both.
//!
//! Reuse is keyed on `(content_hash, embedding_fingerprint)` rather than on the
//! content hash alone. That is the whole point: identical text embedded by a
//! different model is a different vector, and reusing the old one silently
//! mixes two vector spaces inside one index.
//!
//! Fingerprints are strings on purpose. They are stored, compared for equality
//! and printed in diagnostics — never parsed back.

use std::fmt::Write as _;

use crate::error::AppError;

/// Version of the chunking contract: parent/child structure, sizes, overlap.
///
/// Bump this whenever a change alters which text lands in which chunk. A bumped
/// version yields a new fingerprint, which forces a rebuild instead of letting
/// old positions be reused under new semantics.
pub const CHUNKING_SCHEMA_VERSION: i32 = 1;

/// The unit [`crate::services::rag::chunking`] measures chunk sizes in.
///
/// `text-splitter` is driven by a `tiktoken` tokenizer, so the constants are
/// token counts, not characters. US-011 requires the declared unit and the
/// measured unit to agree; `chunking::tests` asserts it.
pub const CHUNK_SIZE_UNIT: &str = "tokens";

/// The tokenizer those token counts come from.
///
/// `cl100k_base` is the encoding `tiktoken-rs` gives the splitter. Naming it is
/// what makes "256 tokens" a checkable claim rather than a number.
pub const CHUNK_SIZER_IDENTITY: &str = "tiktoken:cl100k_base";

/// Prefix marking provenance that was never recorded.
///
/// Chunks that predate generations are backfilled with this. It never equals a
/// live fingerprint, so the first reprocess rebuilds rather than reusing
/// embeddings whose model is unknown.
pub const LEGACY_PREFIX: &str = "legacy:";

/// How a provider's vectors are scaled.
///
/// The core does not renormalize what a provider returns, so this records the
/// provider's own contract rather than a transformation applied here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// The provider documents unit-length vectors (cosine distance is a dot product).
    Unit,
    /// The provider makes no such guarantee.
    Unknown,
}

impl Normalization {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unit => "unit",
            Self::Unknown => "unknown",
        }
    }
}

/// Everything that decides whether two vectors live in the same space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingProvenance {
    pub provider: String,
    pub model: String,
    pub dimension: usize,
    pub normalization: Normalization,
}

impl EmbeddingProvenance {
    /// Read provenance off a live provider.
    ///
    /// [`Normalization::Unit`] is asserted, not discovered: every provider the
    /// core ships documents unit-length vectors, and the trait has no method to
    /// ask. Adding one is the change a non-unit provider requires — until then
    /// this constant is what the fingerprint means, and the variant exists so
    /// that a caller constructing provenance by hand (a test double with
    /// arbitrary vectors, say) can record the truth instead of this default.
    #[must_use]
    pub fn from_provider(provider: &dyn crate::core::providers::EmbeddingProvider) -> Self {
        Self {
            provider: provider.name().to_owned(),
            model: provider.model().to_owned(),
            dimension: provider.dimension(),
            normalization: Normalization::Unit,
        }
    }

    /// The stored, compared form.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        format!(
            "emb:v1:{}:{}:{}:{}",
            self.provider,
            self.model,
            self.dimension,
            self.normalization.as_str()
        )
    }
}

/// Everything that decides whether two chunk positions mean the same thing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkingProvenance {
    pub schema_version: i32,
    pub size_unit: &'static str,
    pub sizer_identity: &'static str,
    /// Parent/child sizes and overlap, in `size_unit`.
    pub parent_size: usize,
    pub child_size: usize,
    pub child_overlap: usize,
}

impl ChunkingProvenance {
    /// The chunking contract this build compiles.
    #[must_use]
    pub fn current(parent_size: usize) -> Self {
        Self {
            schema_version: CHUNKING_SCHEMA_VERSION,
            size_unit: CHUNK_SIZE_UNIT,
            sizer_identity: CHUNK_SIZER_IDENTITY,
            parent_size,
            child_size: crate::services::rag::chunking::CHILD_CHUNK_SIZE,
            child_overlap: crate::services::rag::chunking::CHILD_OVERLAP,
        }
    }

    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut out = String::with_capacity(64);
        let _ = write!(
            out,
            "chunk:v{}:{}:{}:{}/{}+{}",
            self.schema_version,
            self.size_unit,
            self.sizer_identity,
            self.parent_size,
            self.child_size,
            self.child_overlap
        );
        out
    }
}

/// The provenance one generation is built under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationProvenance {
    pub embedding: EmbeddingProvenance,
    pub chunking: ChunkingProvenance,
}

impl GenerationProvenance {
    /// Reject provenance a generation cannot be published under.
    ///
    /// Called before the building generation is created rather than at
    /// publication: a generation whose provenance is unusable should never
    /// consume provider budget in the first place (US-011).
    ///
    /// # Errors
    /// Returns [`AppError::Internal`] when the provider reports an empty name or
    /// model, a zero width, or a width that disagrees with the schema.
    pub fn validate(&self) -> Result<(), AppError> {
        let e = &self.embedding;
        if e.provider.trim().is_empty() || e.model.trim().is_empty() {
            return Err(AppError::Internal(format!(
                "embedding provenance is incomplete (provider={:?}, model={:?}) — \
                 a generation cannot record which vectors it holds",
                e.provider, e.model
            )));
        }
        if e.dimension == 0 {
            return Err(AppError::Internal(format!(
                "embedding provider `{}` reports a width of 0",
                e.provider
            )));
        }
        if e.dimension != crate::core::providers::EMBEDDING_DIM {
            return Err(AppError::Internal(format!(
                "embedding provider `{}` (model `{}`) reports width {}, but the schema stores \
                 vector({}) — indexing would mix incompatible vector spaces",
                e.provider,
                e.model,
                e.dimension,
                crate::core::providers::EMBEDDING_DIM
            )));
        }
        if e.fingerprint().starts_with(LEGACY_PREFIX)
            || self.chunking.fingerprint().starts_with(LEGACY_PREFIX)
        {
            return Err(AppError::Internal(
                "refusing to build a generation under `legacy` provenance: that marker exists \
                 only for chunks backfilled by m20260801_000001_index_generations"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Cache namespaces for query embeddings (US-011).
///
/// A direct query, its reformulation, a HyDE document and a working-memory
/// lookup are four different texts asking four different questions. Hashing
/// them into one keyspace lets a reformulation answer a later direct query, so
/// the namespace is part of the key, alongside the model fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryEmbeddingKind {
    Direct,
    Reformulated,
    HydeDocument,
    WorkingMemory,
}

impl QueryEmbeddingKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Reformulated => "reformulated",
            Self::HydeDocument => "hyde",
            Self::WorkingMemory => "memory",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> EmbeddingProvenance {
        EmbeddingProvenance {
            provider: "voyage".into(),
            model: "voyage-context-3".into(),
            dimension: crate::core::providers::EMBEDDING_DIM,
            normalization: Normalization::Unit,
        }
    }

    #[test]
    fn embedding_fingerprint_separates_models() {
        let a = provenance();
        let mut b = provenance();
        b.model = "voyage-3-lite".into();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn embedding_fingerprint_separates_normalization() {
        let a = provenance();
        let mut b = provenance();
        b.normalization = Normalization::Unknown;
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn embedding_fingerprint_is_stable() {
        assert_eq!(
            provenance().fingerprint(),
            "emb:v1:voyage:voyage-context-3:1024:unit"
        );
    }

    #[test]
    fn chunking_fingerprint_separates_sizes() {
        let a = ChunkingProvenance::current(1024);
        let b = ChunkingProvenance::current(2048);
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn chunking_fingerprint_names_its_unit_and_sizer() {
        let fingerprint = ChunkingProvenance::current(1024).fingerprint();
        assert!(fingerprint.contains(CHUNK_SIZE_UNIT), "{fingerprint}");
        assert!(fingerprint.contains(CHUNK_SIZER_IDENTITY), "{fingerprint}");
    }

    #[test]
    fn validate_rejects_width_mismatch() {
        let mut embedding = provenance();
        embedding.dimension = 768;
        let err = GenerationProvenance {
            embedding,
            chunking: ChunkingProvenance::current(1024),
        }
        .validate()
        .expect_err("768 must not pass against a vector(1024) schema");
        assert!(err.to_string().contains("768"), "{err}");
    }

    #[test]
    fn validate_rejects_missing_identity() {
        let mut embedding = provenance();
        embedding.model = "  ".into();
        let err = GenerationProvenance {
            embedding,
            chunking: ChunkingProvenance::current(1024),
        }
        .validate()
        .expect_err("a blank model is not provenance");
        assert!(err.to_string().contains("incomplete"), "{err}");
    }

    #[test]
    fn validate_accepts_current_provenance() {
        GenerationProvenance {
            embedding: provenance(),
            chunking: ChunkingProvenance::current(1024),
        }
        .validate()
        .expect("the shipped configuration must be publishable");
    }

    #[test]
    fn query_embedding_namespaces_are_distinct() {
        let all = [
            QueryEmbeddingKind::Direct,
            QueryEmbeddingKind::Reformulated,
            QueryEmbeddingKind::HydeDocument,
            QueryEmbeddingKind::WorkingMemory,
        ];
        let mut seen = std::collections::HashSet::new();
        for kind in all {
            assert!(seen.insert(kind.as_str()), "duplicate namespace {kind:?}");
        }
    }
}
