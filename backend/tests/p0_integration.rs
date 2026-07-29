//! Integration tests for P0 fixes (US-017).
//!
//! Verifies the critical core P0 fixes:
//! - **US-002**: Unknown source types return 400 Validation error
//! - **US-003**: Chat pagination clamps to MAX_CHAT_HISTORY_LIMIT
//!
//! The atomic-deletion and webhook body-limit cases moved to
//! `tests/saas_p0_integration.rs` (US-013): both need hosted provisioning or a
//! hosted route, neither of which the public core has.
//!
//! Every test here completes in < 100ms using no external services.

use axum::{http::StatusCode, response::IntoResponse};

// ============================================================================
// US-003: Chat History Pagination Clamping
// ============================================================================

/// Verifies that pagination helper clamps `limit` to MAX_CHAT_HISTORY_LIMIT (200)
/// when a client provides a higher value, preventing DoS via excessive DB load.
#[test]
fn chat_pagination_clamps_limit_to_max() {
    use openbooklm::repositories::{DEFAULT_CHAT_HISTORY_LIMIT, MAX_CHAT_HISTORY_LIMIT};
    use openbooklm::validation::validate_pagination;

    // Client requests limit=999 -> clamped to 200
    let (offset, limit) = validate_pagination(
        Some(0),
        Some(999),
        DEFAULT_CHAT_HISTORY_LIMIT,
        MAX_CHAT_HISTORY_LIMIT,
    );
    assert_eq!(limit, MAX_CHAT_HISTORY_LIMIT);
    assert_eq!(offset, 0);

    // Client requests exactly MAX -> unchanged
    let (_, limit) = validate_pagination(
        None,
        Some(MAX_CHAT_HISTORY_LIMIT),
        DEFAULT_CHAT_HISTORY_LIMIT,
        MAX_CHAT_HISTORY_LIMIT,
    );
    assert_eq!(limit, MAX_CHAT_HISTORY_LIMIT);

    // Client omits limit -> gets default (50)
    let (_, limit) = validate_pagination(
        None,
        None,
        DEFAULT_CHAT_HISTORY_LIMIT,
        MAX_CHAT_HISTORY_LIMIT,
    );
    assert_eq!(limit, DEFAULT_CHAT_HISTORY_LIMIT);

    // Client requests limit=u64::MAX -> clamped to 200
    let (_, limit) = validate_pagination(
        None,
        Some(u64::MAX),
        DEFAULT_CHAT_HISTORY_LIMIT,
        MAX_CHAT_HISTORY_LIMIT,
    );
    assert_eq!(limit, MAX_CHAT_HISTORY_LIMIT);
}

// ============================================================================
// US-002: Unknown Source Type Returns 400 Validation Error
// ============================================================================

/// Verifies that unknown source types produce AppError::Validation which maps
/// to HTTP 400 Bad Request — not a silent fallback to Text.
#[test]
fn unknown_source_type_returns_400_validation_error() {
    use openbooklm::error::AppError;
    use openbooklm::types::SourceType;

    let invalid_types = ["pdff", "html", "", "csv", "json", "   ", "XML", "yaml"];

    for input in invalid_types {
        let result = SourceType::try_from(input);
        assert!(result.is_err(), "Expected error for \"{input}\" but got Ok");

        let err = result.unwrap_err();

        // Verify it's a Validation error (not Internal/Database/etc.)
        assert!(
            matches!(&err, AppError::Validation(_)),
            "Expected AppError::Validation for \"{input}\", got: {err:?}"
        );

        // Verify it produces HTTP 400
        let response = err.into_response();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "Unknown source type \"{input}\" should produce 400, got {}",
            response.status()
        );
    }
}

/// Verifies that valid source types still succeed (regression guard).
#[test]
fn valid_source_types_still_accepted() {
    use openbooklm::types::SourceType;

    let valid = [
        "pdf", "PDF", "text", "txt", "markdown", "md", "web", "url", "docx", "epub",
    ];

    for input in valid {
        assert!(
            SourceType::try_from(input).is_ok(),
            "Valid source type \"{input}\" should be accepted"
        );
    }
}
