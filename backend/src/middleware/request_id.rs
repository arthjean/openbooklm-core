//! Request ID middleware with structured request/response logging.
//!
//! Generates or propagates X-Request-ID headers and emits structured log
//! events for every HTTP request. Each request is wrapped in a tracing span
//! containing `request_id`, `method`, `path`, and `user_id` (recorded by
//! auth middleware for authenticated requests).
//!
//! Response logging is level-gated by status code:
//! - **INFO** for 2xx/3xx
//! - **WARN** for 4xx
//! - **ERROR** for 5xx
//!
//! Health check endpoints (`/health`, `/health/detailed`) are excluded from
//! logging to reduce noise.

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::Request,
    http::{HeaderValue, header, header::HeaderName},
    middleware::Next,
    response::Response,
};
use tracing::Instrument;
use uuid::Uuid;

use crate::types::RequestContext;

/// Header name for request ID.
pub static X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Middleware that generates/propagates X-Request-ID and logs structured request data.
///
/// The tracing span includes a `user_id` placeholder field that the auth
/// middleware fills in via `tracing::Span::current().record()`.
pub async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    let path = request.uri().path().to_owned();
    let skip_logging = is_health_check(&path);

    // Extract existing or generate new request ID
    let request_id: Arc<str> = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .map(Arc::from)
        .unwrap_or_else(|| Arc::from(Uuid::new_v4().to_string()));

    // Pre-compute header value (infallible for valid UUID/existing header)
    let header_value = HeaderValue::from_str(&request_id)
        .unwrap_or_else(|_| HeaderValue::from_static("invalid-request-id"));

    // Inject into request headers and extensions
    request
        .headers_mut()
        .insert(X_REQUEST_ID.clone(), header_value.clone());
    request
        .extensions_mut()
        .insert(RequestId(Arc::clone(&request_id)));

    // Create tracing span with placeholders for fields recorded downstream:
    // - user_id: recorded by auth_middleware
    // - provider/notebook_id: recorded by stream_llm_response
    let span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %request.method(),
        path = %path,
        user_id = tracing::field::Empty,
        provider = tracing::field::Empty,
        notebook_id = tracing::field::Empty,
    );

    let start = Instant::now();

    // Propagate request ID via task-local storage so downstream services
    // and HTTP clients can include it in outgoing requests and log events.
    let ctx = RequestContext::new(Arc::clone(&request_id));

    // Run request within span + request context scope
    let mut response = ctx
        .scope(
            async {
                let response = next.run(request).await;

                if !skip_logging {
                    let status = response.status().as_u16();
                    let duration_ms =
                        u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
                    let response_size = response
                        .headers()
                        .get(header::CONTENT_LENGTH)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok());

                    log_response(status, duration_ms, response_size);
                }

                response
            }
            .instrument(span),
        )
        .await;

    response
        .headers_mut()
        .insert(X_REQUEST_ID.clone(), header_value);

    response
}

/// Emit a structured log event at the appropriate level for the HTTP status code.
fn log_response(status: u16, duration_ms: u64, response_size: Option<u64>) {
    macro_rules! log_at {
        ($level:ident, $msg:expr) => {
            tracing::$level!(status, duration_ms, response_size = response_size, $msg)
        };
    }

    match status {
        200..=399 => log_at!(info, "Request completed"),
        400..=499 => log_at!(warn, "Client error"),
        _ => log_at!(error, "Server error"),
    }
}

/// Health check paths excluded from request logging.
fn is_health_check(path: &str) -> bool {
    path == "/health" || path == "/health/detailed"
}

/// Extension type for accessing request ID in handlers.
#[derive(Clone, Debug)]
pub struct RequestId(pub Arc<str>);

impl std::ops::Deref for RequestId {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Display for RequestId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::registry::LookupSpan;

    // ── Test infrastructure ─────────────────────────────────────────────

    /// Newtype wrapper for span fields stored in span extensions.
    /// Using a newtype prevents collisions with other layers' extension data.
    struct StoredFields(Vec<(String, String)>);

    /// An event captured by [`SpanFieldCaptureLayer`] with its full span
    /// context (fields from all ancestor spans).
    #[derive(Debug)]
    struct CapturedEvent {
        message: String,
        /// Fields collected from all spans in the event's scope (leaf → root).
        span_fields: Vec<(String, String)>,
    }

    /// Tracing layer that captures events together with the fields from all
    /// ancestor spans.  Used to verify that `request_id` (and other span
    /// fields) propagate from the root `request` span to events emitted
    /// anywhere in the call stack.
    struct SpanFieldCaptureLayer {
        events: Arc<Mutex<Vec<CapturedEvent>>>,
    }

    /// Visitor that records every span/event field as a `(name, value)` pair.
    struct FieldVisitor(Vec<(String, String)>);

    impl tracing::field::Visit for FieldVisitor {
        fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
            self.0.push((field.name().to_owned(), value.to_owned()));
        }

        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            self.0.push((field.name().to_owned(), format!("{value:?}")));
        }
    }

    impl<S> tracing_subscriber::Layer<S> for SpanFieldCaptureLayer
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_new_span(
            &self,
            attrs: &tracing::span::Attributes<'_>,
            id: &tracing::span::Id,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = FieldVisitor(Vec::new());
            attrs.record(&mut visitor);
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(StoredFields(visitor.0));
            }
        }

        fn on_record(
            &self,
            id: &tracing::span::Id,
            values: &tracing::span::Record<'_>,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = FieldVisitor(Vec::new());
            values.record(&mut visitor);
            if let Some(span) = ctx.span(id) {
                let mut extensions = span.extensions_mut();
                if let Some(stored) = extensions.get_mut::<StoredFields>() {
                    stored.0.extend(visitor.0);
                } else {
                    extensions.insert(StoredFields(visitor.0));
                }
            }
        }

        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            // Extract the event's message field.
            struct MsgVisitor(String);
            impl tracing::field::Visit for MsgVisitor {
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.0 = format!("{value:?}");
                    }
                }
            }
            let mut msg = MsgVisitor(String::new());
            event.record(&mut msg);

            // Walk all ancestor spans and collect their fields.
            let mut span_fields = Vec::new();
            if let Some(scope) = ctx.event_scope(event) {
                for span in scope {
                    if let Some(fields) = span.extensions().get::<StoredFields>() {
                        span_fields.extend(fields.0.iter().cloned());
                    }
                }
            }

            self.events.lock().unwrap().push(CapturedEvent {
                message: msg.0,
                span_fields,
            });
        }
    }

    /// Helper: assert that a captured event contains a span field with the
    /// given name and value.
    fn assert_has_span_field(event: &CapturedEvent, key: &str, value: &str) {
        assert!(
            event
                .span_fields
                .iter()
                .any(|(k, v)| k == key && v == value),
            "Event '{}' missing span field {key}={value}. Span fields: {:?}",
            event.message,
            event.span_fields
        );
    }

    // ── Tests ───────────────────────────────────────────────────────────

    /// Verifies that `request_id` and dynamically recorded `user_id` from the
    /// root `request` span are visible in the span context of all events
    /// emitted within the request lifecycle.
    ///
    /// This simulates the full middleware chain:
    /// 1. `request_id_middleware` creates a span with `request_id` + empty `user_id`
    /// 2. `auth_middleware` records `user_id` via `Span::current().record()`
    /// 3. Service/repository code emits log events within the span
    ///
    /// The test proves that both static (`request_id`) and dynamically
    /// recorded (`user_id`) fields propagate to all descendant events.
    #[test]
    fn request_id_and_user_id_propagate_to_service_events() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(SpanFieldCaptureLayer {
            events: Arc::clone(&captured),
        });

        tracing::subscriber::with_default(subscriber, || {
            // 1. request_id_middleware creates the root request span
            let span = tracing::info_span!(
                "request",
                request_id = "req-abc-123",
                method = "GET",
                path = "/api/notebooks",
                user_id = tracing::field::Empty,
            );
            let _guard = span.enter();

            // 2. auth_middleware dynamically records the user ID
            tracing::Span::current().record("user_id", "user-456");

            // 3. Service-layer event
            tracing::info!("Loading notebook list");

            // 4. Repository-layer event
            tracing::info!("Querying notebooks from database");
        });

        let events = captured.lock().unwrap();
        assert_eq!(events.len(), 2, "Expected 2 captured events");

        // Both events must see request_id from the root span
        for event in events.iter() {
            assert_has_span_field(event, "request_id", "req-abc-123");
        }

        // Both events must see the dynamically recorded user_id
        for event in events.iter() {
            assert_has_span_field(event, "user_id", "user-456");
        }
    }

    /// Verifies that `request_id` propagates through nested child spans,
    /// simulating the API → service → repository span hierarchy that
    /// `#[instrument]` annotations create.
    ///
    /// Events emitted at any nesting depth must include `request_id` from the
    /// root `request` span, plus fields from their immediate and ancestor spans.
    #[test]
    fn nested_child_spans_inherit_parent_request_id() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(SpanFieldCaptureLayer {
            events: Arc::clone(&captured),
        });

        tracing::subscriber::with_default(subscriber, || {
            // Root: request_id_middleware span
            let request_span = tracing::info_span!(
                "request",
                request_id = "req-xyz-789",
                method = "POST",
                path = "/api/notebooks/abc/chat",
                user_id = tracing::field::Empty,
            );
            let _req = request_span.enter();
            tracing::Span::current().record("user_id", "user-789");

            // Child: service-layer span (simulates #[instrument])
            let service_span = tracing::info_span!("process_chat", notebook_id = "nb-abc");
            let _svc = service_span.enter();
            tracing::info!("Starting RAG retrieval");

            // Grandchild: repository-layer span
            let repo_span = tracing::info_span!("vector_search", source_count = 5u32);
            let _repo = repo_span.enter();
            tracing::info!("Vector search complete");
        });

        let events = captured.lock().unwrap();
        assert_eq!(events.len(), 2, "Expected 2 captured events");

        // Both events must have request_id from the root span
        for event in events.iter() {
            assert_has_span_field(event, "request_id", "req-xyz-789");
            assert_has_span_field(event, "user_id", "user-789");
        }

        // The service-layer event should have notebook_id from its span
        assert_has_span_field(&events[0], "notebook_id", "nb-abc");

        // The repo-layer event should have fields from both parent spans
        assert_has_span_field(&events[1], "notebook_id", "nb-abc");
        assert_has_span_field(&events[1], "source_count", "5");
    }
}
