//! Centralized input validation for all API handlers.
//!
//! All validation functions follow the pattern:
//! `fn validate_X(input) -> Result<(), AppError>` returning `AppError::Validation`.
//!
//! Clamping functions (for DoS prevention) silently constrain values to safe bounds
//! rather than returning errors.

use std::net::{IpAddr, Ipv4Addr};

use url::Url;

use crate::error::AppError;

// ============================================================================
// String length constants
// ============================================================================

/// Maximum title length (notebooks, sources, notes, agents).
pub const MAX_TITLE_LENGTH: usize = 255;

/// Maximum system prompt length (10,000 chars).
pub const MAX_SYSTEM_PROMPT_LENGTH: usize = 10_000;

/// Maximum description length (1,000 chars).
pub const MAX_DESCRIPTION_LENGTH: usize = 1_000;

/// Maximum chat message length (10,000 chars).
pub const MAX_MESSAGE_LENGTH: usize = 10_000;

/// Valid LLM provider names.
pub const VALID_PROVIDERS: &[&str] = &["mistral", "anthropic", "openai"];

// ============================================================================
// SSRF constants
// ============================================================================

/// Blocked cloud metadata hosts for SSRF protection (domain names only).
const BLOCKED_METADATA_HOSTS: &[&str] = &["metadata.google.internal", "metadata.goog"];

/// AWS/GCP/Azure metadata endpoint IP — compared directly instead of via `to_string()`.
const METADATA_IP: Ipv4Addr = Ipv4Addr::new(169, 254, 169, 254);

// ============================================================================
// Generic string validation
// ============================================================================

/// Validate a string field is non-empty (after trimming) and within max length.
pub fn validate_string(s: &str, max_len: usize, field_name: &str) -> Result<(), AppError> {
    if s.trim().is_empty() {
        return Err(AppError::Validation(format!(
            "{field_name} cannot be empty"
        )));
    }
    if s.len() > max_len {
        return Err(AppError::Validation(format!(
            "{field_name} must be {max_len} characters or less"
        )));
    }
    Ok(())
}

// ============================================================================
// Domain-specific validators
// ============================================================================

/// Validate title is non-empty and within length limits.
pub fn validate_title(title: &str) -> Result<(), AppError> {
    validate_string(title, MAX_TITLE_LENGTH, "Title")
}

/// Validate system prompt is non-empty and within length limits.
pub fn validate_system_prompt(prompt: &str) -> Result<(), AppError> {
    validate_string(prompt, MAX_SYSTEM_PROMPT_LENGTH, "System prompt")
}

/// Validate description is non-empty and within length limits.
pub fn validate_description(desc: &str) -> Result<(), AppError> {
    validate_string(desc, MAX_DESCRIPTION_LENGTH, "Description")
}

/// Validate content is non-empty.
pub fn validate_content(content: &str) -> Result<(), AppError> {
    validate_string(content, usize::MAX, "Content")
}

/// Validate a chat message: non-empty after trimming, within length limits.
///
/// Returns the trimmed message on success.
pub fn validate_message(message: &str) -> Result<&str, AppError> {
    let trimmed = message.trim();
    if trimmed.is_empty() {
        return Err(AppError::Validation("Message cannot be empty".into()));
    }
    if trimmed.len() > MAX_MESSAGE_LENGTH {
        return Err(AppError::Validation(format!(
            "Message must be {MAX_MESSAGE_LENGTH} characters or less"
        )));
    }
    Ok(trimmed)
}

/// Validate a search query: non-empty after trimming, within configurable max length.
///
/// Returns the trimmed query on success.
pub fn validate_search_query(query: &str, max_length: usize) -> Result<&str, AppError> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return Err(AppError::Validation("Search query cannot be empty".into()));
    }
    if trimmed.len() > max_length {
        return Err(AppError::Validation(format!(
            "Search query must be {max_length} characters or less"
        )));
    }
    Ok(trimmed)
}

/// Validate an LLM provider name is one of the known providers.
pub fn validate_provider(provider: &str) -> Result<(), AppError> {
    if !VALID_PROVIDERS.contains(&provider) {
        return Err(crate::error::SettingsError::InvalidProvider {
            provider: provider.to_string(),
        }
        .into());
    }
    Ok(())
}

// ============================================================================
// Pagination clamping
// ============================================================================

/// Clamp pagination parameters to server-enforced bounds.
///
/// Returns `(offset, limit)` where `limit` is clamped to `max_limit`.
/// `offset` defaults to `0` and `limit` defaults to `default_limit`.
pub fn validate_pagination(
    offset: Option<u64>,
    limit: Option<u64>,
    default_limit: u64,
    max_limit: u64,
) -> (u64, u64) {
    let offset = offset.unwrap_or(0);
    let limit = limit.unwrap_or(default_limit).min(max_limit);
    (offset, limit)
}

// ============================================================================
// SSRF validation
// ============================================================================

/// Validate URL for SSRF protection — first-pass hostname check.
///
/// Performs pre-resolution hostname validation to block obvious SSRF targets:
/// private IPs, localhost, link-local addresses, and cloud metadata endpoints.
///
/// DNS rebinding is not covered here — all fetching is delegated to the Firecrawl
/// server-side API, which applies its own post-resolution IP checks.
///
/// **SECURITY NOTE (residual risk):** If Firecrawl is ever self-hosted or a
/// second direct-fetch path is added (e.g. `reqwest::get(url)`), DNS rebinding
/// protection must be implemented at the application level via post-resolution
/// IP validation. Do NOT add direct URL fetching without this safeguard.
pub fn validate_url_for_ssrf(url_str: &str) -> Result<(), AppError> {
    let url = Url::parse(url_str)
        .map_err(|e| AppError::Validation(format!("Invalid URL format: {e}")))?;

    if !matches!(url.scheme(), "http" | "https") {
        return Err(AppError::Validation(format!(
            "URL scheme '{}' not allowed. Only http/https permitted.",
            url.scheme()
        )));
    }

    let host = url
        .host()
        .ok_or_else(|| AppError::Validation("URL must have a host".into()))?;

    match &host {
        url::Host::Domain(domain) => {
            if matches!(*domain, "localhost") {
                return Err(AppError::Validation("URLs to localhost not allowed".into()));
            }
            let host_str = url
                .host_str()
                .ok_or_else(|| AppError::Validation("URL must have a host".into()))?;
            if BLOCKED_METADATA_HOSTS.contains(&host_str) {
                return Err(AppError::Validation(
                    "URLs to cloud metadata endpoints not allowed".into(),
                ));
            }
        }
        url::Host::Ipv4(v4) => {
            let ip = IpAddr::V4(*v4);
            if is_private_or_reserved_ip(ip) {
                return Err(AppError::Validation(
                    "URLs to private/reserved IPs not allowed".into(),
                ));
            }
            // Check IPv4 against the cloud metadata IP directly (no String allocation)
            if *v4 == METADATA_IP {
                return Err(AppError::Validation(
                    "URLs to cloud metadata endpoints not allowed".into(),
                ));
            }
        }
        url::Host::Ipv6(v6) => {
            let ip = IpAddr::V6(*v6);
            if is_private_or_reserved_ip(ip) {
                return Err(AppError::Validation(
                    "URLs to private/reserved IPs not allowed".into(),
                ));
            }
        }
    }

    Ok(())
}

/// Check if an IP address is private or reserved.
pub(crate) fn is_private_or_reserved_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                // Unique Local Addresses (fc00::/7) — IPv6 equivalent of RFC 1918
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // Link-local (fe80::/10)
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                // Teredo tunneling (2001:0000::/32) — embeds arbitrary IPv4
                || (v6.segments()[0] == 0x2001 && v6.segments()[1] == 0x0000)
                // 6to4 (2002::/16) — embeds arbitrary IPv4
                || v6.segments()[0] == 0x2002
                // IPv4-mapped IPv6 (::ffff:0:0/96) — check the mapped IPv4
                || matches!(v6.to_ipv4_mapped(), Some(v4) if is_private_or_reserved_ip(IpAddr::V4(v4)))
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ====================================================================
    // validate_string
    // ====================================================================

    #[test]
    fn validate_string_rejects_empty() {
        assert!(validate_string("", 100, "Field").is_err());
        assert!(validate_string("   ", 100, "Field").is_err());
    }

    #[test]
    fn validate_string_rejects_too_long() {
        let long = "a".repeat(101);
        assert!(validate_string(&long, 100, "Field").is_err());
    }

    #[test]
    fn validate_string_accepts_valid() {
        assert!(validate_string("hello", 100, "Field").is_ok());
        assert!(validate_string(&"a".repeat(100), 100, "Field").is_ok());
    }

    #[test]
    fn validate_string_error_messages_include_field_name() {
        let err = validate_string("", 100, "MyField").unwrap_err();
        assert!(err.to_string().contains("MyField"));

        let err = validate_string(&"a".repeat(11), 10, "MyField").unwrap_err();
        assert!(err.to_string().contains("MyField"));
    }

    // ====================================================================
    // validate_title
    // ====================================================================

    #[test]
    fn validate_title_rejects_empty() {
        assert!(validate_title("").is_err());
        assert!(validate_title("   ").is_err());
    }

    #[test]
    fn validate_title_rejects_too_long() {
        let long_title = "a".repeat(256);
        assert!(validate_title(&long_title).is_err());
    }

    #[test]
    fn validate_title_accepts_valid() {
        assert!(validate_title("My Notebook").is_ok());
        assert!(validate_title(&"a".repeat(255)).is_ok());
    }

    // ====================================================================
    // validate_system_prompt
    // ====================================================================

    #[test]
    fn validate_system_prompt_rejects_empty() {
        assert!(validate_system_prompt("").is_err());
        assert!(validate_system_prompt("   ").is_err());
    }

    #[test]
    fn validate_system_prompt_rejects_too_long() {
        let long = "a".repeat(10_001);
        assert!(validate_system_prompt(&long).is_err());
    }

    #[test]
    fn validate_system_prompt_accepts_valid() {
        assert!(validate_system_prompt("You are a helpful assistant.").is_ok());
        assert!(validate_system_prompt(&"a".repeat(10_000)).is_ok());
    }

    // ====================================================================
    // validate_description
    // ====================================================================

    #[test]
    fn validate_description_rejects_empty() {
        assert!(validate_description("").is_err());
        assert!(validate_description("   ").is_err());
    }

    #[test]
    fn validate_description_rejects_too_long() {
        let long = "a".repeat(1_001);
        assert!(validate_description(&long).is_err());
    }

    #[test]
    fn validate_description_accepts_valid() {
        assert!(validate_description("A short description").is_ok());
        assert!(validate_description(&"a".repeat(1_000)).is_ok());
    }

    // ====================================================================
    // validate_content
    // ====================================================================

    #[test]
    fn validate_content_rejects_empty() {
        assert!(validate_content("").is_err());
        assert!(validate_content("   ").is_err());
    }

    #[test]
    fn validate_content_accepts_valid() {
        assert!(validate_content("Some content here").is_ok());
        assert!(validate_content(&"a".repeat(100_000)).is_ok());
    }

    // ====================================================================
    // validate_message
    // ====================================================================

    #[test]
    fn validate_message_rejects_empty() {
        assert!(validate_message("").is_err());
        assert!(validate_message("   ").is_err());
    }

    #[test]
    fn validate_message_rejects_too_long() {
        let long = "a".repeat(MAX_MESSAGE_LENGTH + 1);
        assert!(validate_message(&long).is_err());
    }

    #[test]
    fn validate_message_accepts_valid() {
        assert!(validate_message("Hello").is_ok());
        assert!(validate_message(&"a".repeat(MAX_MESSAGE_LENGTH)).is_ok());
    }

    #[test]
    fn validate_message_returns_trimmed() {
        assert_eq!(validate_message("  hello  ").unwrap(), "hello");
    }

    // ====================================================================
    // validate_search_query
    // ====================================================================

    #[test]
    fn validate_search_query_rejects_empty() {
        assert!(validate_search_query("", 500).is_err());
        assert!(validate_search_query("   ", 500).is_err());
    }

    #[test]
    fn validate_search_query_rejects_too_long() {
        let long = "a".repeat(501);
        assert!(validate_search_query(&long, 500).is_err());
    }

    #[test]
    fn validate_search_query_accepts_valid() {
        assert!(validate_search_query("hello", 500).is_ok());
        assert!(validate_search_query(&"a".repeat(500), 500).is_ok());
    }

    #[test]
    fn validate_search_query_returns_trimmed() {
        assert_eq!(validate_search_query("  hello  ", 500).unwrap(), "hello");
    }

    // ====================================================================
    // validate_provider
    // ====================================================================

    #[test]
    fn validate_provider_accepts_valid() {
        for provider in VALID_PROVIDERS {
            assert!(validate_provider(provider).is_ok());
        }
    }

    #[test]
    fn validate_provider_rejects_unknown() {
        assert!(validate_provider("unknown").is_err());
        assert!(validate_provider("").is_err());
        assert!(validate_provider("Mistral").is_err()); // case-sensitive
    }

    // ====================================================================
    // validate_pagination
    // ====================================================================

    #[test]
    fn pagination_defaults() {
        let (offset, limit) = validate_pagination(None, None, 50, 200);
        assert_eq!(offset, 0);
        assert_eq!(limit, 50);
    }

    #[test]
    fn pagination_clamps_to_max() {
        let (_, limit) = validate_pagination(None, Some(500), 50, 200);
        assert_eq!(limit, 200);
    }

    #[test]
    fn pagination_preserves_valid_values() {
        let (offset, limit) = validate_pagination(Some(42), Some(25), 50, 200);
        assert_eq!(offset, 42);
        assert_eq!(limit, 25);
    }

    #[test]
    fn pagination_extreme_limit_clamped() {
        let (_, limit) = validate_pagination(Some(0), Some(u64::MAX), 50, 200);
        assert_eq!(limit, 200);
    }

    // ====================================================================
    // SSRF validation
    // ====================================================================

    #[test]
    fn ssrf_blocks_localhost() {
        assert!(validate_url_for_ssrf("http://localhost/admin").is_err());
        assert!(validate_url_for_ssrf("http://127.0.0.1/admin").is_err());
    }

    #[test]
    fn ssrf_blocks_private_ips() {
        assert!(validate_url_for_ssrf("http://192.168.1.1/").is_err());
        assert!(validate_url_for_ssrf("http://10.0.0.1/").is_err());
    }

    #[test]
    fn ssrf_blocks_metadata_endpoints() {
        assert!(validate_url_for_ssrf("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(validate_url_for_ssrf("http://metadata.google.internal/").is_err());
    }

    #[test]
    fn ssrf_allows_valid_urls() {
        assert!(validate_url_for_ssrf("https://example.com").is_ok());
        assert!(validate_url_for_ssrf("https://docs.google.com/document/d/123").is_ok());
    }

    #[test]
    fn ssrf_rejects_non_http_schemes() {
        assert!(validate_url_for_ssrf("ftp://files.example.com/doc").is_err());
        assert!(validate_url_for_ssrf("file:///etc/passwd").is_err());
    }

    #[test]
    fn private_ip_detection() {
        assert!(is_private_or_reserved_ip("10.0.0.1".parse().unwrap()));
        assert!(is_private_or_reserved_ip("172.16.0.1".parse().unwrap()));
        assert!(is_private_or_reserved_ip("192.168.0.1".parse().unwrap()));
        assert!(is_private_or_reserved_ip("127.0.0.1".parse().unwrap()));
        assert!(is_private_or_reserved_ip("::1".parse().unwrap()));
        assert!(!is_private_or_reserved_ip("8.8.8.8".parse().unwrap()));
    }

    #[test]
    fn ipv6_unique_local_blocked() {
        assert!(is_private_or_reserved_ip("fc00::1".parse().unwrap()));
        assert!(is_private_or_reserved_ip(
            "fd12:3456:789a::1".parse().unwrap()
        ));
    }

    #[test]
    fn ipv6_link_local_blocked() {
        assert!(is_private_or_reserved_ip("fe80::1".parse().unwrap()));
    }

    #[test]
    fn ipv4_mapped_ipv6_blocked() {
        // ::ffff:192.168.1.1 maps to a private IPv4
        assert!(is_private_or_reserved_ip(
            "::ffff:192.168.1.1".parse().unwrap()
        ));
        assert!(is_private_or_reserved_ip(
            "::ffff:127.0.0.1".parse().unwrap()
        ));
        assert!(is_private_or_reserved_ip(
            "::ffff:10.0.0.1".parse().unwrap()
        ));
    }

    #[test]
    fn ipv4_mapped_ipv6_public_allowed() {
        assert!(!is_private_or_reserved_ip(
            "::ffff:8.8.8.8".parse().unwrap()
        ));
    }

    #[test]
    fn ipv6_teredo_blocked() {
        // Teredo tunneling: 2001:0000::/32
        assert!(is_private_or_reserved_ip(
            "2001:0000:4136:e378:8000:63bf:3fff:fdd2".parse().unwrap()
        ));
    }

    #[test]
    fn ipv6_6to4_blocked() {
        // 6to4: 2002::/16
        assert!(is_private_or_reserved_ip(
            "2002:c0a8:0101::1".parse().unwrap()
        ));
    }

    #[test]
    fn ipv6_public_allowed() {
        assert!(!is_private_or_reserved_ip(
            "2607:f8b0:4004:800::200e".parse().unwrap()
        ));
        // 2001:db8::/32 is documentation, but 2001:200::/23 is a real allocation
        // Our Teredo check only blocks 2001:0000::/32 specifically
        assert!(!is_private_or_reserved_ip("2001:0200::1".parse().unwrap()));
    }

    #[test]
    fn ssrf_blocks_ipv6_private_urls() {
        assert!(validate_url_for_ssrf("http://[fc00::1]/admin").is_err());
        assert!(validate_url_for_ssrf("http://[fe80::1]/admin").is_err());
        assert!(validate_url_for_ssrf("http://[::ffff:192.168.1.1]/admin").is_err());
    }
}
