//! Filtered approximate nearest-neighbour strategy (US-015, US-016).
//!
//! # The problem this exists for
//!
//! A dense retrieval query filters by notebook and orders by vector distance.
//! PostgreSQL runs the HNSW index scan first and applies the filter to what
//! comes out, so a notebook holding 0.1% of the corpus can exhaust the index
//! scan before ten of its own rows have been seen. The query then returns four
//! rows instead of ten and reports success: the API cannot tell an underfilled
//! approximate scan from a notebook that genuinely has four relevant passages.
//!
//! pgvector 0.8.0 added iterative scans, which resume the index scan until the
//! requested number of *filtered* rows is found or a scan budget is spent. The
//! [`ScanStrategy`] here names the modes; US-015's benchmark measures them; the
//! search repository applies the one the benchmark approved.
//!
//! # Why the capability is probed rather than assumed
//!
//! `SET LOCAL hnsw.iterative_scan` on a build that does not know the setting is
//! an error, not a no-op. A deployment on pgvector 0.7 would therefore fail
//! every dense query at run time with an opaque message. [`VectorCapabilities`]
//! asks the database what it supports once, at startup, so the diagnostic names
//! the required version instead.
//!
//! # Why this lives in the repository layer
//!
//! A scan strategy is how this layer reads vectors from PostgreSQL, not a
//! retrieval policy. Keeping it beside the query it parameterises is also what
//! stops the repository from importing `crate::services`, which the layering
//! rule in [`crate::types`] forbids in that direction.

use sea_orm::{ConnectionTrait, DbBackend, Statement, TransactionTrait};

use crate::error::AppError;

/// Minimum pgvector version the approved filtered-scan strategy requires.
///
/// 0.8.0 is where `hnsw.iterative_scan` appeared. Deployments below it are
/// refused at startup rather than silently degraded, because the degradation is
/// invisible: it looks like a notebook with less evidence.
pub const MIN_PGVECTOR_VERSION: (u32, u32) = (0, 8);

/// The strategy US-015's benchmark approved, and the one dense retrieval runs.
///
/// Selected from `contracts/baseline/ann/filtered-hnsw.json`: the cheapest
/// strategy meeting Recall@10 >= 0.95, top-k fill >= 0.99 and P95 <= 300 ms at
/// 100%, 10%, 1% and 0.1% filter selectivity on the reference environment.
/// `strict_order` rather than `relaxed_order` because the API and the fusion
/// layer both consume dense results in distance order, and relaxed order gives
/// that up for throughput this workload does not need.
///
/// Changing this constant requires re-running the benchmark and updating the
/// report in the same change: it is a measured value, not a tuning knob.
pub const APPROVED_STRATEGY: ScanStrategy = ScanStrategy::HnswIterative {
    mode: IterativeMode::StrictOrder,
    ef_search: 100,
    max_scan_tuples: 20_000,
};

/// How an iterative scan orders what it resumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterativeMode {
    /// Rows come back in exact distance order. The ordering the API and the
    /// fusion layer depend on.
    StrictOrder,
    /// Rows may come back slightly out of distance order in exchange for
    /// throughput.
    RelaxedOrder,
}

impl IterativeMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StrictOrder => "strict_order",
            Self::RelaxedOrder => "relaxed_order",
        }
    }
}

/// One way of running a filtered dense search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanStrategy {
    /// Sequential scan with exact distances. The ground truth every
    /// approximate strategy is measured against, never a production strategy:
    /// it reads every row.
    Exact,
    /// A single HNSW scan with `ef_search` candidates, filtered afterwards.
    Hnsw { ef_search: i32 },
    /// An HNSW scan that resumes until enough filtered rows are found or
    /// `max_scan_tuples` are examined.
    HnswIterative {
        mode: IterativeMode,
        ef_search: i32,
        max_scan_tuples: i32,
    },
}

impl ScanStrategy {
    /// A stable label for reports and telemetry.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Exact => "exact".to_owned(),
            Self::Hnsw { ef_search } => format!("hnsw_ef{ef_search}"),
            Self::HnswIterative {
                mode,
                ef_search,
                max_scan_tuples,
            } => format!("hnsw_{}_ef{ef_search}_max{max_scan_tuples}", mode.as_str()),
        }
    }

    /// Every setting this strategy needs, as one statement.
    ///
    /// Sent in a single round trip rather than one per setting: three `SET
    /// LOCAL` statements executed separately cost three round trips on the
    /// hottest query in the system, to transport three constants.
    ///
    /// Empty when the strategy changes nothing, in which case the caller must
    /// send nothing at all.
    #[must_use]
    pub fn session_preamble(self) -> String {
        self.session_settings().join("; ")
    }

    /// The `SET LOCAL` statements this strategy needs, one per entry.
    ///
    /// `LOCAL` is what keeps a per-query setting from leaking into a pooled
    /// connection: it is reverted when the transaction ends, so the next
    /// borrower of that connection cannot inherit a scan mode chosen for
    /// someone else's query (US-016).
    ///
    /// Retrieval sends [`session_preamble`](Self::session_preamble); this
    /// itemised form is what the benchmark report records, so a reader can see
    /// exactly which settings produced a measurement.
    #[must_use]
    pub fn session_settings(self) -> Vec<String> {
        match self {
            // Nothing. Exactness is a property of the SQL the benchmark writes
            // (`ORDER BY (embedding <=> v) + 0`, which no vector index can
            // serve), not of a planner switch. Disabling `enable_indexscan` for
            // one query would be a session-wide planner change riding on a
            // pooled connection, and a measurement harness whose reference
            // measurement can contaminate the strategies it is comparing
            // measures nothing.
            Self::Exact => Vec::new(),
            Self::Hnsw { ef_search } => vec![format!("SET LOCAL hnsw.ef_search = {ef_search}")],
            Self::HnswIterative {
                mode,
                ef_search,
                max_scan_tuples,
            } => vec![
                format!("SET LOCAL hnsw.ef_search = {ef_search}"),
                format!("SET LOCAL hnsw.iterative_scan = {}", mode.as_str()),
                format!("SET LOCAL hnsw.max_scan_tuples = {max_scan_tuples}"),
            ],
        }
    }

    /// Whether the ordering must be computed exactly rather than through the
    /// vector index.
    ///
    /// The caller expresses this in SQL. See [`session_settings`](Self::session_settings)
    /// for why it is not a planner setting.
    #[must_use]
    pub const fn is_exact_reference(self) -> bool {
        matches!(self, Self::Exact)
    }

    /// Whether rows come back in exact distance order.
    ///
    /// A hard requirement rather than a preference: the API returns dense
    /// results in distance order and reciprocal rank fusion consumes their
    /// *rank*, so a strategy that reorders rows for throughput silently
    /// changes what fusion computes. `relaxed_order` buys throughput this
    /// workload does not need at that price, which is why the benchmark
    /// rejects it instead of ranking it (US-016 AC-3).
    #[must_use]
    pub const fn preserves_distance_order(self) -> bool {
        !matches!(
            self,
            Self::HnswIterative {
                mode: IterativeMode::RelaxedOrder,
                ..
            }
        )
    }

    /// Relative cost, cheapest first.
    ///
    /// "The cheapest strategy that clears every threshold" has to be decided by
    /// a property of the strategy. Deciding it by the order strategies were
    /// pushed into a list makes the recommendation depend on the shape of a
    /// loop, which no reader of the report can see.
    ///
    /// A single scan is cheapest; an iterative scan costs more the more tuples
    /// it may examine; exact search reads the whole table and is never
    /// recommended, only used as the reference.
    #[must_use]
    pub const fn cost_rank(self) -> i64 {
        match self {
            Self::Hnsw { .. } => 0,
            Self::HnswIterative {
                max_scan_tuples, ..
            } => 1 + max_scan_tuples as i64,
            Self::Exact => i64::MAX,
        }
    }

    /// Whether this strategy needs a pgvector build that knows iterative scans.
    #[must_use]
    pub const fn needs_iterative_scan(self) -> bool {
        matches!(self, Self::HnswIterative { .. })
    }
}

/// What the deployed pgvector build offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorCapabilities {
    /// `extversion` from `pg_extension`, verbatim.
    pub extension_version: String,
    /// Parsed `(major, minor)`, or `None` when the version string is not the
    /// usual `major.minor.patch`.
    pub version: Option<(u32, u32)>,
    /// Whether `hnsw.iterative_scan` exists in this build.
    pub iterative_scan: bool,
}

impl VectorCapabilities {
    /// Ask the database what it supports.
    ///
    /// Everything runs inside one transaction, for two reasons. A pooled
    /// connection would otherwise answer the three questions from up to three
    /// different backends, and pgvector registers its `hnsw.*` settings when
    /// its library is first loaded *into a backend* — so a connection that has
    /// never evaluated a vector expression reports the settings as unknown even
    /// on a build that has them. The transaction pins one backend, and the
    /// warm-up expression makes it load the library before the settings are
    /// read.
    ///
    /// The version comparison is the fallback rather than the primary check: a
    /// distribution that backports the setting is described by what it can
    /// actually do, not by its version string.
    ///
    /// # Errors
    /// Returns the database error when the extension cannot be queried at all.
    pub async fn probe<C: ConnectionTrait + TransactionTrait>(db: &C) -> Result<Self, AppError> {
        let txn = db.begin().await?;

        // Loads pgvector into this backend, which registers `hnsw.*`.
        txn.query_one(Statement::from_string(
            DbBackend::Postgres,
            "SELECT '[1,0]'::vector <=> '[0,1]'::vector AS warmup",
        ))
        .await?;

        let row = txn
            .query_one(Statement::from_string(
                DbBackend::Postgres,
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
            ))
            .await?;

        let extension_version: String = match row {
            Some(row) => row.try_get("", "extversion")?,
            None => {
                return Err(AppError::Internal(
                    "the pgvector extension is not installed; dense retrieval cannot run".into(),
                ));
            }
        };

        // `current_setting(name, true)` returns NULL for an unknown GUC instead
        // of raising, which is exactly the probe this needs.
        let probe = txn
            .query_one(Statement::from_string(
                DbBackend::Postgres,
                "SELECT current_setting('hnsw.iterative_scan', true) AS mode",
            ))
            .await?;
        let setting_present = probe
            .and_then(|row| row.try_get::<Option<String>>("", "mode").ok())
            .flatten()
            .is_some();

        txn.commit().await?;

        let version = parse_version(&extension_version);
        let iterative_scan = setting_present || version.is_some_and(|v| v >= MIN_PGVECTOR_VERSION);

        Ok(Self {
            version,
            extension_version,
            iterative_scan,
        })
    }

    /// An actionable error when this build cannot run `strategy`.
    ///
    /// Returns `Ok(())` when it can. The message names the required version and
    /// what was found, because "invalid value for parameter" from PostgreSQL
    /// tells an operator nothing about what to install.
    ///
    /// # Errors
    /// Returns [`AppError::Internal`] describing the missing capability.
    pub fn ensure_supports(&self, strategy: ScanStrategy) -> Result<(), AppError> {
        if !strategy.needs_iterative_scan() || self.iterative_scan {
            return Ok(());
        }
        let (major, minor) = MIN_PGVECTOR_VERSION;
        Err(AppError::Internal(format!(
            "filtered dense retrieval requires pgvector >= {major}.{minor} for \
             hnsw.iterative_scan; this database reports {}. Upgrade the \
             extension (ALTER EXTENSION vector UPDATE) or pin the image \
             documented in docs/architecture/filtered-ann.md.",
            self.extension_version
        )))
    }
}

/// Parse `major.minor[.patch]` into `(major, minor)`.
fn parse_version(raw: &str) -> Option<(u32, u32)> {
    let mut parts = raw.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_labels_are_stable_and_carry_their_parameters() {
        assert_eq!(ScanStrategy::Exact.label(), "exact");
        assert_eq!(ScanStrategy::Hnsw { ef_search: 100 }.label(), "hnsw_ef100");
        assert_eq!(
            ScanStrategy::HnswIterative {
                mode: IterativeMode::StrictOrder,
                ef_search: 100,
                max_scan_tuples: 20_000,
            }
            .label(),
            "hnsw_strict_order_ef100_max20000"
        );
    }

    #[test]
    fn the_exact_reference_changes_no_session_state() {
        assert!(ScanStrategy::Exact.session_settings().is_empty());
        assert!(ScanStrategy::Exact.is_exact_reference());
        assert!(!ScanStrategy::Hnsw { ef_search: 100 }.is_exact_reference());
    }

    #[test]
    fn the_preamble_carries_every_setting_in_one_statement() {
        let strategy = ScanStrategy::HnswIterative {
            mode: IterativeMode::StrictOrder,
            ef_search: 100,
            max_scan_tuples: 20_000,
        };
        assert_eq!(
            strategy.session_preamble(),
            "SET LOCAL hnsw.ef_search = 100; \
             SET LOCAL hnsw.iterative_scan = strict_order; \
             SET LOCAL hnsw.max_scan_tuples = 20000"
        );
        assert!(
            ScanStrategy::Exact.session_preamble().is_empty(),
            "a strategy that changes nothing must send nothing"
        );
    }

    #[test]
    fn relaxed_order_is_refused_rather_than_ranked() {
        // The fusion layer consumes dense rank, so a strategy that reorders
        // rows is not a cheaper option, it is a different result set.
        assert!(
            !ScanStrategy::HnswIterative {
                mode: IterativeMode::RelaxedOrder,
                ef_search: 100,
                max_scan_tuples: 20_000,
            }
            .preserves_distance_order()
        );
        assert!(APPROVED_STRATEGY.preserves_distance_order());
        assert!(ScanStrategy::Hnsw { ef_search: 100 }.preserves_distance_order());
        assert!(ScanStrategy::Exact.preserves_distance_order());
    }

    #[test]
    fn cost_orders_a_single_scan_below_an_iterative_one_below_exact() {
        let single = ScanStrategy::Hnsw { ef_search: 100 };
        let iterative_small = ScanStrategy::HnswIterative {
            mode: IterativeMode::StrictOrder,
            ef_search: 100,
            max_scan_tuples: 20_000,
        };
        let iterative_large = ScanStrategy::HnswIterative {
            mode: IterativeMode::StrictOrder,
            ef_search: 100,
            max_scan_tuples: 100_000,
        };
        assert!(single.cost_rank() < iterative_small.cost_rank());
        assert!(iterative_small.cost_rank() < iterative_large.cost_rank());
        assert!(iterative_large.cost_rank() < ScanStrategy::Exact.cost_rank());
    }

    #[test]
    fn every_setting_is_transaction_local() {
        for strategy in [
            ScanStrategy::Exact,
            ScanStrategy::Hnsw { ef_search: 100 },
            ScanStrategy::HnswIterative {
                mode: IterativeMode::RelaxedOrder,
                ef_search: 40,
                max_scan_tuples: 1_000,
            },
        ] {
            for statement in strategy.session_settings() {
                assert!(
                    statement.starts_with("SET LOCAL "),
                    "a pooled connection must not inherit `{statement}`"
                );
            }
        }
    }

    #[test]
    fn an_iterative_strategy_is_refused_on_a_build_without_the_setting() {
        let old = VectorCapabilities {
            extension_version: "0.7.4".to_owned(),
            version: Some((0, 7)),
            iterative_scan: false,
        };

        // A plain HNSW scan needs nothing new.
        assert!(
            old.ensure_supports(ScanStrategy::Hnsw { ef_search: 100 })
                .is_ok()
        );

        let error = old
            .ensure_supports(ScanStrategy::HnswIterative {
                mode: IterativeMode::StrictOrder,
                ef_search: 100,
                max_scan_tuples: 20_000,
            })
            .expect_err("0.7 cannot run an iterative scan");
        let message = error.to_string();
        assert!(
            message.contains("0.8"),
            "the error must name the requirement: {message}"
        );
        assert!(message.contains("0.7.4"), "and what was found: {message}");
    }

    #[test]
    fn a_capable_build_accepts_every_strategy() {
        let current = VectorCapabilities {
            extension_version: "0.8.5".to_owned(),
            version: Some((0, 8)),
            iterative_scan: true,
        };
        for strategy in [
            ScanStrategy::Exact,
            ScanStrategy::Hnsw { ef_search: 100 },
            ScanStrategy::HnswIterative {
                mode: IterativeMode::StrictOrder,
                ef_search: 100,
                max_scan_tuples: 20_000,
            },
        ] {
            assert!(current.ensure_supports(strategy).is_ok());
        }
    }

    #[test]
    fn version_parsing_survives_an_unexpected_string() {
        assert_eq!(parse_version("0.8.5"), Some((0, 8)));
        assert_eq!(parse_version("0.8"), Some((0, 8)));
        assert_eq!(parse_version("unreleased"), None);
        assert_eq!(parse_version(""), None);
    }
}
