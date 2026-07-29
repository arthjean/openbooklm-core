//! Source events broadcasting for real-time SSE updates.
//!
//! Provides a broadcast channel system for source processing events.
//! Each notebook has its own channel with an event ID counter and replay buffer
//! for reliable reconnection. Clients subscribe to receive real-time updates
//! when sources change status, and can replay missed events via `Last-Event-ID`.

use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use uuid::Uuid;

/// Broadcast channel capacity per notebook.
const CHANNEL_CAPACITY: usize = 100;

/// Maximum number of events kept in the replay buffer per notebook.
const REPLAY_BUFFER_CAPACITY: usize = 200;

/// Default interval between cleanup sweeps (seconds).
/// Configurable via `SSE_CLEANUP_INTERVAL_SECS` env var.
pub const DEFAULT_CLEANUP_INTERVAL_SECS: u64 = 60;

/// Default staleness threshold: channels with no subscribers AND no events
/// for this long are removed. Configurable via `SSE_STALE_CHANNEL_SECS` env var.
pub const DEFAULT_STALE_CHANNEL_SECS: u64 = 300;

/// Default maximum channel age (seconds). Channels older than this are
/// force-removed regardless of subscriber count.
/// Configurable via `SSE_MAX_CHANNEL_AGE_SECS` env var.
pub const DEFAULT_MAX_CHANNEL_AGE_SECS: u64 = 3600;

/// Configuration for SSE broadcaster cleanup behavior.
#[derive(Debug, Clone)]
pub struct SseCleanupConfig {
    /// Interval between cleanup sweeps.
    pub cleanup_interval: Duration,
    /// Channels with no subscribers and no events for this long are removed.
    pub stale_threshold: Duration,
    /// Maximum channel age — channels older than this are force-removed.
    pub max_channel_age: Duration,
}

impl Default for SseCleanupConfig {
    fn default() -> Self {
        Self {
            cleanup_interval: Duration::from_secs(DEFAULT_CLEANUP_INTERVAL_SECS),
            stale_threshold: Duration::from_secs(DEFAULT_STALE_CHANNEL_SECS),
            max_channel_age: Duration::from_secs(DEFAULT_MAX_CHANNEL_AGE_SECS),
        }
    }
}

impl SseCleanupConfig {
    /// Load configuration from environment variables with defaults.
    pub fn from_env() -> Self {
        Self {
            cleanup_interval: Duration::from_secs(
                std::env::var("SSE_CLEANUP_INTERVAL_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_CLEANUP_INTERVAL_SECS),
            ),
            stale_threshold: Duration::from_secs(
                std::env::var("SSE_STALE_CHANNEL_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_STALE_CHANNEL_SECS),
            ),
            max_channel_age: Duration::from_secs(
                std::env::var("SSE_MAX_CHANNEL_AGE_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(DEFAULT_MAX_CHANNEL_AGE_SECS),
            ),
        }
    }
}

// ============================================================================
// Event Types
// ============================================================================
//
// The types themselves live in `core::protocol::source`, the public contract
// (US-009). They are re-exported here so this module keeps its role as the
// source-event entry point.

pub use crate::core::protocol::source::{EmbeddingProgress, SourceEvent, SourceStatusData};

// ============================================================================
// Notebook Channel
// ============================================================================

/// Per-notebook broadcast channel with event IDs and a replay buffer.
struct NotebookChannel {
    sender: broadcast::Sender<(u64, SourceEvent)>,
    next_event_id: AtomicU64,
    replay_buffer: Mutex<VecDeque<(u64, SourceEvent)>>,
    last_event_time: Mutex<Instant>,
    created_at: Instant,
}

impl NotebookChannel {
    fn new() -> Self {
        let (sender, _) = broadcast::channel(CHANNEL_CAPACITY);
        Self {
            sender,
            next_event_id: AtomicU64::new(1),
            replay_buffer: Mutex::new(VecDeque::with_capacity(REPLAY_BUFFER_CAPACITY)),
            last_event_time: Mutex::new(Instant::now()),
            created_at: Instant::now(),
        }
    }
}

// ============================================================================
// Broadcaster
// ============================================================================

/// Manager for source event broadcast channels.
///
/// Each notebook has its own broadcast channel with monotonic event IDs and a
/// bounded replay buffer. Channels are lazily created when the first subscriber
/// connects or the first event is sent.
#[derive(Clone)]
pub struct SourceEventBroadcaster {
    channels: Arc<DashMap<Uuid, NotebookChannel>>,
    cleanup_config: SseCleanupConfig,
}

impl Default for SourceEventBroadcaster {
    fn default() -> Self {
        Self {
            channels: Arc::new(DashMap::new()),
            cleanup_config: SseCleanupConfig::default(),
        }
    }
}

impl SourceEventBroadcaster {
    /// Create a new broadcaster with default cleanup config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new broadcaster with custom cleanup configuration.
    #[must_use]
    pub fn with_cleanup_config(cleanup_config: SseCleanupConfig) -> Self {
        Self {
            channels: Arc::new(DashMap::new()),
            cleanup_config,
        }
    }

    /// Subscribe to events for a specific notebook.
    ///
    /// Returns a receiver that will receive all future `(event_id, event)` pairs.
    ///
    /// # Security
    ///
    /// This method performs **no authorization check**. Callers must verify that
    /// the requesting user has access to the notebook before calling this method
    /// (e.g., via `verify_notebook_access` in the HTTP handler layer).
    #[must_use]
    pub fn subscribe(&self, notebook_id: Uuid) -> broadcast::Receiver<(u64, SourceEvent)> {
        let channel = self.get_or_create_channel(notebook_id);
        channel.sender.subscribe()
    }

    /// Broadcast an event for a notebook.
    ///
    /// Creates the channel eagerly if it doesn't exist, so events are always
    /// buffered for replay even before any subscriber connects. Assigns a
    /// monotonic event ID and stores in the replay buffer.
    #[allow(clippy::needless_pass_by_value)] // event is cloned internally for broadcast
    #[allow(clippy::significant_drop_tightening)] // DashMap entry must be held for consistent broadcast
    pub fn broadcast(&self, notebook_id: Uuid, event: SourceEvent) -> Option<u64> {
        let channel = self.get_or_create_channel(notebook_id);

        // Relaxed ordering is safe here because the DashMap entry provides
        // synchronization: all threads see the same AtomicU64 instance
        let event_id = channel.next_event_id.fetch_add(1, Relaxed);

        // Store in replay buffer
        {
            let mut buf = channel.replay_buffer.lock();
            if buf.len() >= REPLAY_BUFFER_CAPACITY {
                buf.pop_front();
            }
            buf.push_back((event_id, event.clone()));
        }

        // Update timestamp
        *channel.last_event_time.lock() = Instant::now();

        let receiver_count = channel.sender.receiver_count();
        if channel.sender.send((event_id, event.clone())).is_err() {
            tracing::debug!(%notebook_id, "No active SSE subscribers");
            return Some(event_id);
        }

        tracing::debug!(
            %notebook_id,
            source_id = ?event.source_id(),
            event_type = event.event_type(),
            event_id,
            receiver_count,
            "Broadcast SSE event"
        );

        Some(event_id)
    }

    /// Replay buffered events after the given event ID.
    ///
    /// Returns all events with `id > last_event_id`, in order.
    ///
    /// # Security
    ///
    /// This method performs **no authorization check**. Callers must verify that
    /// the requesting user has access to the notebook before calling this method.
    #[must_use]
    pub fn replay_since(&self, notebook_id: Uuid, last_event_id: u64) -> Vec<(u64, SourceEvent)> {
        let Some(channel) = self.channels.get(&notebook_id) else {
            return Vec::new();
        };

        let buf = channel.replay_buffer.lock();
        buf.iter()
            .filter(|(id, _)| *id > last_event_id)
            .cloned()
            .collect()
    }

    /// Broadcast a status change event.
    pub fn broadcast_status(
        &self,
        notebook_id: Uuid,
        source_id: Uuid,
        status: &str,
        error_message: Option<String>,
    ) {
        self.broadcast(
            notebook_id,
            SourceEvent::status(source_id, status, error_message),
        );
    }

    /// Broadcast a ready event.
    pub fn broadcast_ready(&self, notebook_id: Uuid, source_id: Uuid, chunk_count: i32) {
        self.broadcast(notebook_id, SourceEvent::ready(source_id, chunk_count));
    }

    /// Broadcast a ready event with degraded services information.
    pub fn broadcast_ready_degraded(
        &self,
        notebook_id: Uuid,
        source_id: Uuid,
        chunk_count: i32,
        degraded_services: Vec<String>,
    ) {
        self.broadcast(
            notebook_id,
            SourceEvent::ready_degraded(source_id, chunk_count, degraded_services),
        );
    }

    /// Broadcast embedding progress for a source.
    pub fn broadcast_embedding_progress(
        &self,
        notebook_id: Uuid,
        source_id: Uuid,
        chunks_done: u32,
        chunks_total: u32,
    ) {
        self.broadcast(
            notebook_id,
            SourceEvent::status_with_progress(source_id, "embedding", chunks_done, chunks_total),
        );
    }

    /// Broadcast an error event.
    pub fn broadcast_error(&self, notebook_id: Uuid, source_id: Uuid, message: &str) {
        self.broadcast(notebook_id, SourceEvent::error(source_id, message));
    }

    /// Broadcast that OCR extraction has started.
    pub fn broadcast_ocr_started(&self, notebook_id: Uuid, source_id: Uuid, total_pages: u32) {
        self.broadcast(
            notebook_id,
            SourceEvent::ocr_started(source_id, total_pages),
        );
    }

    /// Broadcast OCR page progress.
    ///
    /// Not yet called — the Mistral OCR client issues a single batch API call.
    /// This is scaffolding for future batched OCR where pages are processed
    /// in groups with progress emitted between each batch.
    pub fn broadcast_ocr_progress(
        &self,
        notebook_id: Uuid,
        source_id: Uuid,
        current_page: u32,
        total_pages: u32,
    ) {
        self.broadcast(
            notebook_id,
            SourceEvent::ocr_progress(source_id, current_page, total_pages),
        );
    }

    /// Broadcast that OCR extraction completed.
    pub fn broadcast_ocr_completed(
        &self,
        notebook_id: Uuid,
        source_id: Uuid,
        pages_processed: u32,
    ) {
        self.broadcast(
            notebook_id,
            SourceEvent::ocr_completed(source_id, pages_processed),
        );
    }

    /// Broadcast that OCR result was served from cache.
    pub fn broadcast_ocr_cache_hit(&self, notebook_id: Uuid, source_id: Uuid) {
        self.broadcast(notebook_id, SourceEvent::ocr_cache_hit(source_id));
    }

    /// Remove channel for a notebook (for resource cleanup).
    pub fn cleanup(&self, notebook_id: Uuid) {
        self.channels.remove(&notebook_id);
    }

    /// Get the number of active channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get the total number of active receivers across all channels.
    #[must_use]
    pub fn total_receivers(&self) -> usize {
        self.channels
            .iter()
            .map(|entry| entry.sender.receiver_count())
            .sum()
    }

    /// Remove channels that meet any of these criteria:
    /// 1. Zero active receivers (immediately, no staleness wait)
    /// 2. Stale: no subscribers AND no events for `stale_threshold`
    /// 3. Exceeded max channel age (force-removed regardless of subscribers)
    ///
    /// Returns the number of channels removed.
    pub fn cleanup_stale_channels(&self) -> usize {
        let now = Instant::now();
        let mut removed = 0;

        self.channels.retain(|_notebook_id, channel| {
            let has_subscribers = channel.sender.receiver_count() > 0;
            let channel_age = now.duration_since(channel.created_at);

            // Force-remove channels that exceeded max age
            if channel_age > self.cleanup_config.max_channel_age {
                removed += 1;
                return false;
            }

            // Remove channels with zero active receivers immediately
            if !has_subscribers {
                let last_event = *channel.last_event_time.lock();
                let is_stale = now.duration_since(last_event) > self.cleanup_config.stale_threshold;

                if is_stale {
                    removed += 1;
                    return false;
                }
            }

            true
        });

        removed
    }

    /// Spawn a background task that periodically removes stale channels.
    pub fn start_cleanup_task(&self) {
        let broadcaster = self.clone();
        let interval = broadcaster.cleanup_config.cleanup_interval;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let removed = broadcaster.cleanup_stale_channels();
                tracing::debug!(
                    removed,
                    remaining = broadcaster.channel_count(),
                    total_receivers = broadcaster.total_receivers(),
                    "SSE cleanup cycle completed"
                );
                if removed > 0 {
                    tracing::info!(
                        removed,
                        remaining = broadcaster.channel_count(),
                        "Cleaned up stale SSE channels"
                    );
                }
            }
        });
    }

    fn get_or_create_channel(
        &self,
        notebook_id: Uuid,
    ) -> dashmap::mapref::one::Ref<'_, Uuid, NotebookChannel> {
        // Use the entry ref directly instead of dropping and re-getting
        // to avoid race with cleanup_stale_channels
        self.channels
            .entry(notebook_id)
            .or_insert_with(NotebookChannel::new)
            .downgrade()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_constructors() {
        let id = Uuid::new_v4();

        let status = SourceEvent::status(id, "processing", None);
        assert_eq!(status.source_id(), Some(id));
        assert_eq!(status.event_type(), "source:status");

        let ready = SourceEvent::ready(id, 42);
        assert_eq!(ready.source_id(), Some(id));
        assert_eq!(ready.event_type(), "source:ready");

        let error = SourceEvent::error(id, "failed");
        assert_eq!(error.source_id(), Some(id));
        assert_eq!(error.event_type(), "source:error");
    }

    #[tokio::test]
    async fn broadcast_to_subscriber() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();

        let mut rx = broadcaster.subscribe(notebook_id);
        broadcaster.broadcast_ready(notebook_id, source_id, 10);

        let (event_id, event) = rx.recv().await.unwrap();
        assert_eq!(event_id, 1);
        assert_eq!(event.source_id(), Some(source_id));
    }

    #[test]
    fn channel_count() {
        let broadcaster = SourceEventBroadcaster::new();
        assert_eq!(broadcaster.channel_count(), 0);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let _ = broadcaster.subscribe(id1);
        assert_eq!(broadcaster.channel_count(), 1);

        let _ = broadcaster.subscribe(id2);
        assert_eq!(broadcaster.channel_count(), 2);

        broadcaster.cleanup(id1);
        assert_eq!(broadcaster.channel_count(), 1);
    }

    #[test]
    fn broadcast_assigns_monotonic_event_ids() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();

        // Create channel by subscribing
        let _rx = broadcaster.subscribe(notebook_id);

        let id1 = broadcaster.broadcast(
            notebook_id,
            SourceEvent::status(source_id, "processing", None),
        );
        let id2 = broadcaster.broadcast(
            notebook_id,
            SourceEvent::status(source_id, "embedding", None),
        );
        let id3 = broadcaster.broadcast(notebook_id, SourceEvent::ready(source_id, 10));

        assert_eq!(id1, Some(1));
        assert_eq!(id2, Some(2));
        assert_eq!(id3, Some(3));
    }

    #[test]
    fn replay_since_returns_events_after_id() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();

        let _rx = broadcaster.subscribe(notebook_id);

        broadcaster.broadcast_status(notebook_id, source_id, "processing", None);
        broadcaster.broadcast_status(notebook_id, source_id, "contextualizing", None);
        broadcaster.broadcast_status(notebook_id, source_id, "embedding", None);
        broadcaster.broadcast_ready(notebook_id, source_id, 10);

        // Replay after event 2 → should get events 3 and 4
        let replayed = broadcaster.replay_since(notebook_id, 2);
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].0, 3);
        assert_eq!(replayed[1].0, 4);
    }

    #[test]
    fn replay_since_empty_for_unknown_notebook() {
        let broadcaster = SourceEventBroadcaster::new();
        let replayed = broadcaster.replay_since(Uuid::new_v4(), 0);
        assert!(replayed.is_empty());
    }

    #[test]
    fn replay_since_returns_all_when_last_id_is_zero() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();

        let _rx = broadcaster.subscribe(notebook_id);

        broadcaster.broadcast_status(notebook_id, source_id, "processing", None);
        broadcaster.broadcast_ready(notebook_id, source_id, 5);

        let replayed = broadcaster.replay_since(notebook_id, 0);
        assert_eq!(replayed.len(), 2);
    }

    #[test]
    fn replay_buffer_evicts_oldest_when_full() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();

        let _rx = broadcaster.subscribe(notebook_id);

        // Fill beyond REPLAY_BUFFER_CAPACITY
        for i in 0..(REPLAY_BUFFER_CAPACITY + 50) {
            broadcaster.broadcast(
                notebook_id,
                SourceEvent::status(source_id, format!("step-{i}"), None),
            );
        }

        let replayed = broadcaster.replay_since(notebook_id, 0);
        assert_eq!(replayed.len(), REPLAY_BUFFER_CAPACITY);

        // Oldest event should have been evicted; first event ID should be 51
        assert_eq!(replayed[0].0, 51);
        // Last event ID should be 250
        assert_eq!(
            replayed[REPLAY_BUFFER_CAPACITY - 1].0,
            (REPLAY_BUFFER_CAPACITY + 50) as u64
        );
    }

    #[test]
    fn broadcast_creates_channel_eagerly() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        assert_eq!(broadcaster.channel_count(), 0);

        let result = broadcaster.broadcast(notebook_id, SourceEvent::ready(Uuid::new_v4(), 1));
        assert!(result.is_some(), "Broadcast should create channel eagerly");
        assert_eq!(broadcaster.channel_count(), 1);
    }

    #[test]
    fn replay_works_for_events_before_subscriber() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();

        // Broadcast 3 events before any subscriber connects
        broadcaster.broadcast_status(notebook_id, source_id, "processing", None);
        broadcaster.broadcast_status(notebook_id, source_id, "embedding", None);
        broadcaster.broadcast_ready(notebook_id, source_id, 10);

        // Late subscriber connects — replay should return all 3 events
        let replayed = broadcaster.replay_since(notebook_id, 0);
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].0, 1);
        assert_eq!(replayed[2].0, 3);
    }

    #[test]
    fn cleanup_stale_channels_removes_idle_channels() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();

        // Create a channel (subscribe creates it), then drop the receiver
        {
            let _rx = broadcaster.subscribe(notebook_id);
        }
        assert_eq!(broadcaster.channel_count(), 1);

        // Manually set the last_event_time to be old enough
        if let Some(channel) = broadcaster.channels.get(&notebook_id) {
            *channel.last_event_time.lock() =
                Instant::now() - Duration::from_secs(DEFAULT_STALE_CHANNEL_SECS + 1);
        }

        let removed = broadcaster.cleanup_stale_channels();
        assert_eq!(removed, 1);
        assert_eq!(broadcaster.channel_count(), 0);
    }

    #[test]
    fn cleanup_keeps_channels_with_subscribers() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();

        let _rx = broadcaster.subscribe(notebook_id);
        assert_eq!(broadcaster.channel_count(), 1);

        // Even if time is old, channel has a subscriber → keep it
        if let Some(channel) = broadcaster.channels.get(&notebook_id) {
            *channel.last_event_time.lock() =
                Instant::now() - Duration::from_secs(DEFAULT_STALE_CHANNEL_SECS + 1);
        }

        let removed = broadcaster.cleanup_stale_channels();
        assert_eq!(removed, 0);
        assert_eq!(broadcaster.channel_count(), 1);
    }

    #[test]
    fn cleanup_keeps_recent_channels_without_subscribers() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();

        // Create channel and drop subscriber
        {
            let _rx = broadcaster.subscribe(notebook_id);
        }
        assert_eq!(broadcaster.channel_count(), 1);

        // last_event_time is recent (just created) → keep it
        let removed = broadcaster.cleanup_stale_channels();
        assert_eq!(removed, 0);
        assert_eq!(broadcaster.channel_count(), 1);
    }

    #[test]
    fn cleanup_force_removes_channels_exceeding_max_age() {
        let config = SseCleanupConfig {
            max_channel_age: Duration::from_secs(10),
            ..Default::default()
        };
        let broadcaster = SourceEventBroadcaster::with_cleanup_config(config);
        let notebook_id = Uuid::new_v4();

        // Channel with an active subscriber
        let _rx = broadcaster.subscribe(notebook_id);
        assert_eq!(broadcaster.channel_count(), 1);

        // Manually backdate the channel's created_at to exceed max age
        if let Some(channel) = broadcaster.channels.get(&notebook_id) {
            // SAFETY: created_at is private, so we use a workaround: remove + re-insert
            // Instead, we just test with a very short max_channel_age config
            let _ = channel; // hold ref
        }
        // Backdate created_at by mutating via DashMap
        if let Some(mut channel) = broadcaster.channels.get_mut(&notebook_id) {
            channel.created_at = Instant::now() - Duration::from_secs(11);
        }

        // Even though there's an active subscriber, max age forces removal
        let removed = broadcaster.cleanup_stale_channels();
        assert_eq!(removed, 1);
        assert_eq!(broadcaster.channel_count(), 0);
    }

    #[test]
    fn cleanup_keeps_young_channels_with_subscribers() {
        let config = SseCleanupConfig {
            max_channel_age: Duration::from_secs(3600),
            ..Default::default()
        };
        let broadcaster = SourceEventBroadcaster::with_cleanup_config(config);
        let notebook_id = Uuid::new_v4();

        let _rx = broadcaster.subscribe(notebook_id);
        assert_eq!(broadcaster.channel_count(), 1);

        // Channel is young and has subscriber → keep it
        let removed = broadcaster.cleanup_stale_channels();
        assert_eq!(removed, 0);
        assert_eq!(broadcaster.channel_count(), 1);
    }

    #[test]
    fn total_receivers_counts_across_channels() {
        let broadcaster = SourceEventBroadcaster::new();
        assert_eq!(broadcaster.total_receivers(), 0);

        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let _rx1a = broadcaster.subscribe(id1);
        let _rx1b = broadcaster.subscribe(id1);
        let _rx2 = broadcaster.subscribe(id2);

        assert_eq!(broadcaster.total_receivers(), 3);
        assert_eq!(broadcaster.channel_count(), 2);
    }

    #[test]
    fn total_receivers_decreases_on_drop() {
        let broadcaster = SourceEventBroadcaster::new();
        let id = Uuid::new_v4();

        let rx1 = broadcaster.subscribe(id);
        let _rx2 = broadcaster.subscribe(id);
        assert_eq!(broadcaster.total_receivers(), 2);

        drop(rx1);
        assert_eq!(broadcaster.total_receivers(), 1);
    }

    #[test]
    fn with_cleanup_config_uses_custom_values() {
        let config = SseCleanupConfig {
            cleanup_interval: Duration::from_secs(30),
            stale_threshold: Duration::from_secs(120),
            max_channel_age: Duration::from_secs(1800),
        };
        let broadcaster = SourceEventBroadcaster::with_cleanup_config(config);

        assert_eq!(
            broadcaster.cleanup_config.cleanup_interval,
            Duration::from_secs(30)
        );
        assert_eq!(
            broadcaster.cleanup_config.stale_threshold,
            Duration::from_secs(120)
        );
        assert_eq!(
            broadcaster.cleanup_config.max_channel_age,
            Duration::from_secs(1800)
        );
    }

    #[test]
    fn sse_cleanup_config_default_values() {
        let config = SseCleanupConfig::default();
        assert_eq!(
            config.cleanup_interval,
            Duration::from_secs(DEFAULT_CLEANUP_INTERVAL_SECS)
        );
        assert_eq!(
            config.stale_threshold,
            Duration::from_secs(DEFAULT_STALE_CHANNEL_SECS)
        );
        assert_eq!(
            config.max_channel_age,
            Duration::from_secs(DEFAULT_MAX_CHANNEL_AGE_SECS)
        );
    }

    // --- US-012: Graceful degradation SSE events ---

    #[test]
    fn ready_event_always_writes_degraded_services() {
        let event = SourceEvent::ready(Uuid::new_v4(), 10);
        let json = serde_json::to_value(&event).unwrap();
        // US-009 removed `skip_serializing_if`: the derive now matches the wire
        // form clients have always parsed, where the key is always present.
        let data = json.get("data").unwrap();
        assert_eq!(
            data.get("degraded_services"),
            Some(&serde_json::json!([])),
            "Empty degraded_services must still be written: {json}"
        );
    }

    #[test]
    fn ready_degraded_event_includes_services() {
        let event =
            SourceEvent::ready_degraded(Uuid::new_v4(), 10, vec!["contextualization".to_string()]);
        let json = serde_json::to_value(&event).unwrap();
        let data = json.get("data").unwrap();
        let services = data
            .get("degraded_services")
            .expect("degraded_services should be present");
        assert_eq!(services.as_array().unwrap().len(), 1);
        assert_eq!(services[0], "contextualization");
    }

    #[test]
    fn ready_degraded_event_with_multiple_services() {
        let event = SourceEvent::ready_degraded(
            Uuid::new_v4(),
            5,
            vec!["contextualization".to_string(), "reranking".to_string()],
        );
        let json = serde_json::to_value(&event).unwrap();
        let data = json.get("data").unwrap();
        let services = data.get("degraded_services").unwrap().as_array().unwrap();
        assert_eq!(services.len(), 2);
        assert_eq!(services[0], "contextualization");
        assert_eq!(services[1], "reranking");
    }

    #[tokio::test]
    async fn broadcast_ready_degraded_delivers_event() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();

        let mut rx = broadcaster.subscribe(notebook_id);
        broadcaster.broadcast_ready_degraded(
            notebook_id,
            source_id,
            10,
            vec!["contextualization".to_string()],
        );

        let (event_id, event) = rx.recv().await.unwrap();
        assert_eq!(event_id, 1);
        assert_eq!(event.source_id(), Some(source_id));
        assert_eq!(event.event_type(), "source:ready");
        // Verify the degraded_services field is present in serialized form
        let json = serde_json::to_value(&event).unwrap();
        let data = json.get("data").unwrap();
        assert!(data.get("degraded_services").is_some());
    }

    // --- US-008: OCR SSE events ---

    #[test]
    fn ocr_event_constructors() {
        let id = Uuid::new_v4();

        let started = SourceEvent::ocr_started(id, 12);
        assert_eq!(started.source_id(), Some(id));
        assert_eq!(started.event_type(), "source:ocr_started");

        let progress = SourceEvent::ocr_progress(id, 5, 12);
        assert_eq!(progress.source_id(), Some(id));
        assert_eq!(progress.event_type(), "source:ocr_progress");

        let completed = SourceEvent::ocr_completed(id, 12);
        assert_eq!(completed.source_id(), Some(id));
        assert_eq!(completed.event_type(), "source:ocr_completed");

        let cache_hit = SourceEvent::ocr_cache_hit(id);
        assert_eq!(cache_hit.source_id(), Some(id));
        assert_eq!(cache_hit.event_type(), "source:ocr_cache_hit");
    }

    #[test]
    fn ocr_started_serialization() {
        let event = SourceEvent::ocr_started(Uuid::nil(), 20);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["event"], "source:ocr_started");
        assert_eq!(json["data"]["total_pages"], 20);
        assert!(json["data"]["source_id"].is_string());
    }

    #[test]
    fn ocr_progress_serialization() {
        let event = SourceEvent::ocr_progress(Uuid::nil(), 7, 20);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["event"], "source:ocr_progress");
        assert_eq!(json["data"]["current_page"], 7);
        assert_eq!(json["data"]["total_pages"], 20);
        assert!(json["data"]["source_id"].is_string());
    }

    #[test]
    fn ocr_completed_serialization() {
        let event = SourceEvent::ocr_completed(Uuid::nil(), 15);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["event"], "source:ocr_completed");
        assert_eq!(json["data"]["pages_processed"], 15);
        assert!(json["data"]["source_id"].is_string());
    }

    #[test]
    fn ocr_cache_hit_serialization() {
        let event = SourceEvent::ocr_cache_hit(Uuid::nil());
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["event"], "source:ocr_cache_hit");
        assert!(json["data"]["source_id"].is_string());
    }

    #[tokio::test]
    async fn broadcast_ocr_events_deliver() {
        let broadcaster = SourceEventBroadcaster::new();
        let notebook_id = Uuid::new_v4();
        let source_id = Uuid::new_v4();

        let mut rx = broadcaster.subscribe(notebook_id);

        broadcaster.broadcast_ocr_started(notebook_id, source_id, 10);
        let (_, event) = rx.recv().await.unwrap();
        assert_eq!(event.event_type(), "source:ocr_started");

        broadcaster.broadcast_ocr_progress(notebook_id, source_id, 3, 10);
        let (_, event) = rx.recv().await.unwrap();
        assert_eq!(event.event_type(), "source:ocr_progress");

        broadcaster.broadcast_ocr_completed(notebook_id, source_id, 10);
        let (_, event) = rx.recv().await.unwrap();
        assert_eq!(event.event_type(), "source:ocr_completed");

        broadcaster.broadcast_ocr_cache_hit(notebook_id, source_id);
        let (_, event) = rx.recv().await.unwrap();
        assert_eq!(event.event_type(), "source:ocr_cache_hit");
    }
}
