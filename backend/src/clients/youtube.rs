//! YouTube transcript client using the InnerTube API directly.
//!
//! Fetches video transcripts and metadata via YouTube's internal InnerTube
//! `/youtubei/v1/player` endpoint with an ANDROID client context. This is the
//! same approach used by Python's `youtube-transcript-api`, `yt-dlp`, LangChain,
//! and LlamaIndex.
//!
//! Key design: a **single `reqwest::Client`** handles both the InnerTube POST
//! (to discover caption tracks) and the transcript GET (to fetch content).
//! YouTube validates that the caption URL is fetched by the same client session
//! that discovered it — using separate clients causes empty responses.
//!
//! The ANDROID client context avoids YouTube's POT (Proof of Origin Token)
//! requirement for caption URLs, which the WEB client would trigger.

use std::sync::Arc;
use std::time::Duration;

use url::Url;

use crate::core::config::CoreConfig;
use crate::error::{AppError, SourceError};

use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use super::metrics::{ClientMetrics, ProviderMetrics};
use super::resilience::ResilientExecutor;
use super::retry::RetryConfig;

const PROVIDER_NAME: &str = "youtube";
const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// InnerTube API endpoint for player data (captions + video details).
const INNERTUBE_PLAYER_URL: &str = "https://www.youtube.com/youtubei/v1/player";

/// Default InnerTube API key for the ANDROID client.
///
/// **Publication classification: PUBLIC_UPSTREAM_MATERIAL.** This value is not a
/// StriveX credential. It ships inside the public YouTube Android application,
/// is extracted from YouTube's public JS, and is hardcoded identically by every
/// major transcript library. It grants no account access and is not rotatable by
/// this project. Recorded as finding F-003 in `docs/security/publication-audit.md`
/// and fingerprinted in `scripts/secret-scan-allowlist.json`.
///
/// Override via the `YOUTUBE_INNERTUBE_API_KEY` env var when upstream changes it,
/// so operators can follow a rotation without redeploying.
const DEFAULT_INNERTUBE_API_KEY: &str = "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8";

/// ANDROID client version — matches `youtube-transcript-api` `_settings.py`.
const ANDROID_CLIENT_VERSION: &str = "20.10.38";

/// User-Agent for the ANDROID client context.
const ANDROID_USER_AGENT: &str = "com.google.android.youtube/20.10.38 (Linux; U; Android 11) gzip";

// ============================================================================
// Public result types
// ============================================================================

/// Transcript extraction result with metadata.
#[derive(Debug, Clone)]
pub struct YouTubeTranscriptResult {
    pub snippets: Vec<TranscriptSnippet>,
    pub video_id: String,
    pub language: String,
    pub language_code: String,
    pub is_generated: bool,
}

/// A single transcript snippet with timestamp.
#[derive(Debug, Clone)]
pub struct TranscriptSnippet {
    pub text: String,
    pub start: f64,
    pub duration: f64,
}

/// Video metadata.
#[derive(Debug, Clone)]
pub struct YouTubeVideoMetadata {
    pub title: String,
    pub author: String,
    pub duration_seconds: u32,
    pub channel_id: String,
    pub thumbnail_url: Option<String>,
}

// ============================================================================
// Client
// ============================================================================

/// YouTube transcript client with resilience patterns.
///
/// Uses the InnerTube API directly with an ANDROID client context.
/// A single `reqwest::Client` is shared across all requests to maintain
/// session consistency for caption URL fetching.
#[derive(Debug, Clone)]
pub struct YouTubeClient {
    http: reqwest::Client,
    api_key: Arc<str>,
    resilience: ResilientExecutor,
}

impl YouTubeClient {
    /// Create from application config.
    pub fn from_config(config: &CoreConfig, metrics: &ClientMetrics) -> Option<Self> {
        let _proxy_url = config.youtube_proxy_url.as_deref();
        let api_key: Arc<str> = config
            .youtube_innertube_api_key
            .as_deref()
            .unwrap_or(DEFAULT_INNERTUBE_API_KEY)
            .into();

        match Self::new(api_key, metrics.provider(PROVIDER_NAME)) {
            Ok(client) => Some(client),
            Err(e) => {
                tracing::warn!(error = %e, "YouTube client initialization failed");
                None
            }
        }
    }

    /// Create a new YouTube client with an ANDROID-configured HTTP client.
    fn new(api_key: Arc<str>, metrics: Arc<ProviderMetrics>) -> Result<Self, SourceError> {
        let http = reqwest::Client::builder()
            .user_agent(ANDROID_USER_AGENT)
            .default_headers({
                let mut h = reqwest::header::HeaderMap::new();
                h.insert(
                    reqwest::header::ACCEPT_LANGUAGE,
                    reqwest::header::HeaderValue::from_static("en-US"),
                );
                h
            })
            .build()
            .map_err(|e| SourceError::ProcessingFailed {
                reason: format!("Failed to create YouTube HTTP client: {e}"),
            })?;

        let retry_config = RetryConfig::new(3)
            .with_initial_delay(Duration::from_secs(1))
            .with_max_delay(Duration::from_secs(30));

        let circuit_breaker = Arc::new(CircuitBreaker::new(
            PROVIDER_NAME,
            CircuitBreakerConfig::new(5)
                .with_open_duration(Duration::from_secs(60))
                .with_success_threshold(2),
        ));

        Ok(Self {
            http,
            api_key,
            resilience: ResilientExecutor::new(
                PROVIDER_NAME,
                retry_config,
                circuit_breaker,
                metrics,
            )
            .with_timeout_secs(DEFAULT_TIMEOUT_SECS),
        })
    }

    /// Fetch transcript for a video.
    ///
    /// Language fallback: tries `[locale, "en"]` first, then any available.
    /// Locale must be a valid ISO 639-1 code (2 lowercase ASCII letters).
    pub async fn fetch_transcript(
        &self,
        video_id: &str,
        locale: &str,
    ) -> Result<YouTubeTranscriptResult, AppError> {
        if !is_valid_locale(locale) {
            return Err(AppError::Validation(format!(
                "Invalid locale code: {locale}"
            )));
        }

        let video_id = video_id.to_string();
        let locale = locale.to_string();
        let http = self.http.clone();
        let api_key = self.api_key.clone();

        self.resilience
            .execute(
                |retry_secs| {
                    AppError::from(SourceError::ProcessingFailed {
                        reason: format!(
                            "YouTube service unavailable. Retry after {retry_secs} seconds."
                        ),
                    })
                },
                move || {
                    let http = http.clone();
                    let vid = video_id.clone();
                    let loc = locale.clone();
                    let key = api_key.clone();
                    async move { fetch_transcript_inner(&http, &vid, &loc, &key).await }
                },
            )
            .await
    }

    /// Fetch video metadata (title, author, duration).
    pub async fn fetch_video_details(
        &self,
        video_id: &str,
    ) -> Result<YouTubeVideoMetadata, AppError> {
        let resp = innertube_player(&self.http, video_id, &self.api_key)
            .await
            .map_err(|e| e.0)?;

        let details = resp
            .get("videoDetails")
            .ok_or_else(|| source_err("Video details not found in response"))?;

        Ok(YouTubeVideoMetadata {
            title: json_str(details, "title").unwrap_or_default(),
            author: json_str(details, "author").unwrap_or_default(),
            duration_seconds: json_str(details, "lengthSeconds")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            channel_id: json_str(details, "channelId").unwrap_or_default(),
            thumbnail_url: details
                .pointer("/thumbnail/thumbnails/0/url")
                .and_then(|v| v.as_str())
                .map(String::from),
        })
    }

    /// Fetch video title and author via YouTube's oEmbed API (lightweight).
    ///
    /// Used for the title auto-fill UX — fast and doesn't count against
    /// the InnerTube circuit breaker.
    pub async fn fetch_oembed_title(video_url: &str) -> Result<(String, String), AppError> {
        let encoded_url: String = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("url", video_url)
            .append_pair("format", "json")
            .finish();
        let oembed_url = format!("https://www.youtube.com/oembed?{encoded_url}");

        let resp = reqwest::get(&oembed_url).await.map_err(|e| {
            AppError::from(SourceError::ProcessingFailed {
                reason: format!("Failed to fetch YouTube oEmbed: {e}"),
            })
        })?;

        if !resp.status().is_success() {
            return Err(AppError::from(SourceError::ProcessingFailed {
                reason: "Video not found or not embeddable".into(),
            }));
        }

        #[derive(serde::Deserialize)]
        struct OEmbedResponse {
            title: String,
            author_name: String,
        }

        let data: OEmbedResponse = resp.json().await.map_err(|e| {
            AppError::from(SourceError::ProcessingFailed {
                reason: format!("Failed to parse oEmbed response: {e}"),
            })
        })?;

        Ok((data.title, data.author_name))
    }
}

super::impl_client_resilience_methods!(YouTubeClient, PROVIDER_NAME);

// ============================================================================
// InnerTube API
// ============================================================================

/// Build the InnerTube ANDROID client request body.
fn innertube_body(video_id: &str) -> serde_json::Value {
    serde_json::json!({
        "context": {
            "client": {
                "clientName": "ANDROID",
                "clientVersion": ANDROID_CLIENT_VERSION,
                "androidSdkVersion": 30,
                "hl": "en",
                "timeZone": "UTC",
                "utcOffsetMinutes": 0
            }
        },
        "videoId": video_id
    })
}

/// POST to InnerTube `/player` and return the JSON response.
///
/// Checks playability status and returns classified errors.
async fn innertube_player(
    http: &reqwest::Client,
    video_id: &str,
    api_key: &str,
) -> super::resilience::HttpResult<serde_json::Value> {
    let url = format!("{INNERTUBE_PLAYER_URL}?key={api_key}&prettyPrint=false");

    let resp = http
        .post(&url)
        .json(&innertube_body(video_id))
        .send()
        .await
        .map_err(|e| {
            (
                source_err(&format!("InnerTube request failed: {e}")),
                None,
                true,
            )
        })?;

    let status = resp.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err((
            source_err("YouTube rate limit reached. Please try again later."),
            Some(429),
            true,
        ));
    }
    if !status.is_success() {
        return Err((
            source_err(&format!("YouTube returned status {status}")),
            Some(status.as_u16()),
            status.is_server_error(),
        ));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| {
        (
            source_err(&format!("Failed to parse InnerTube response: {e}")),
            None,
            false,
        )
    })?;

    // Check playability status for user-facing errors
    check_playability(&data)?;

    Ok(data)
}

/// Check playability status and return classified errors.
fn check_playability(data: &serde_json::Value) -> super::resilience::HttpResult<()> {
    let status = data
        .pointer("/playabilityStatus/status")
        .and_then(|v| v.as_str())
        .unwrap_or("OK");

    let reason = data
        .pointer("/playabilityStatus/reason")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match status {
        "OK" | "LIVE_STREAM_OFFLINE" => Ok(()),
        "LOGIN_REQUIRED" if reason.contains("age") => Err((
            source_err("This video requires age verification and cannot be processed."),
            None,
            false,
        )),
        "LOGIN_REQUIRED" => Err((
            source_err("This video requires authentication and cannot be processed."),
            None,
            false,
        )),
        "UNPLAYABLE" | "ERROR" => Err((
            source_err("This video is not accessible (private or deleted)."),
            None,
            false,
        )),
        _ => Err((
            source_err(&format!("Video unavailable: {status} — {reason}")),
            None,
            false,
        )),
    }
}

// ============================================================================
// Transcript fetching (inner logic)
// ============================================================================

/// Fetch transcript via InnerTube + caption URL.
///
/// Three-tier language fallback: user locale → English → any available.
async fn fetch_transcript_inner(
    http: &reqwest::Client,
    video_id: &str,
    locale: &str,
    api_key: &str,
) -> super::resilience::HttpResult<YouTubeTranscriptResult> {
    // Step 1: InnerTube POST — get caption tracks
    let data = innertube_player(http, video_id, api_key).await?;

    let tracks = data
        .pointer("/captions/playerCaptionsTracklistRenderer/captionTracks")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            (
                source_err("This video has no subtitles available. Try a different video."),
                None,
                false,
            )
        })?;

    if tracks.is_empty() {
        return Err((
            source_err("This video has no subtitles available. Try a different video."),
            None,
            false,
        ));
    }

    // Step 2: Pick best caption track (language fallback)
    let preferred: Vec<&str> = if locale == "en" {
        vec!["en"]
    } else {
        vec![locale, "en"]
    };

    let track = pick_caption_track(tracks, &preferred);
    let (base_url, lang_code, lang_name, is_generated) = extract_track_info(track)?;

    tracing::info!(video_id, language = lang_code, "Selected caption track");

    // Step 3: Check for POT requirement
    if base_url.contains("&exp=xpe") {
        return Err((
            source_err(
                "This video requires browser authentication for captions. Try a different video.",
            ),
            None,
            false,
        ));
    }

    // Step 4: Fetch transcript content (same client — critical for auth)
    let snippets = fetch_transcript_content(http, &base_url)
        .await
        .map_err(|reason| (source_err(&reason), None, true))?;

    Ok(YouTubeTranscriptResult {
        snippets,
        video_id: video_id.to_string(),
        language: lang_name,
        language_code: lang_code,
        is_generated,
    })
}

/// Pick the best caption track from InnerTube response.
///
/// Priority: preferred languages (manual > auto) → any manual → any auto.
fn pick_caption_track<'a>(
    tracks: &'a [serde_json::Value],
    preferred: &[&str],
) -> &'a serde_json::Value {
    // Try preferred languages first (manual captions prioritized)
    for lang in preferred {
        // Manual first
        if let Some(t) = tracks.iter().find(|t| {
            json_str(t, "languageCode").as_deref() == Some(lang)
                && json_str(t, "kind").as_deref() != Some("asr")
        }) {
            return t;
        }
        // Then auto-generated
        if let Some(t) = tracks
            .iter()
            .find(|t| json_str(t, "languageCode").as_deref() == Some(lang))
        {
            return t;
        }
    }

    // Fallback: first manual track, then first auto track
    let available: Vec<&str> = tracks
        .iter()
        .filter_map(|t| t.get("languageCode")?.as_str())
        .collect();
    tracing::info!(
        available = ?available,
        "No transcript in preferred languages, falling back to first available"
    );

    tracks
        .iter()
        .find(|t| json_str(t, "kind").as_deref() != Some("asr"))
        .unwrap_or(&tracks[0])
}

/// Extract track info: (base_url, language_code, language_name, is_generated).
fn extract_track_info(
    track: &serde_json::Value,
) -> super::resilience::HttpResult<(String, String, String, bool)> {
    let base_url = json_str(track, "baseUrl")
        .ok_or_else(|| (source_err("Caption track has no URL"), None, false))?;

    let lang_code = json_str(track, "languageCode").unwrap_or_else(|| "und".into());
    let is_generated = json_str(track, "kind").as_deref() == Some("asr");

    // Extract language name from `name.runs[0].text` or `name.simpleText`
    let lang_name = track
        .pointer("/name/runs/0/text")
        .or_else(|| track.pointer("/name/simpleText"))
        .and_then(|v| v.as_str())
        .unwrap_or(&lang_code)
        .to_string();

    Ok((base_url, lang_code, lang_name, is_generated))
}

/// Fetch and parse transcript content from a caption URL.
///
/// Strategy: strip `&fmt=srv3` (get legacy XML) → fallback to `&fmt=json3`.
async fn fetch_transcript_content(
    http: &reqwest::Client,
    base_url: &str,
) -> Result<Vec<TranscriptSnippet>, String> {
    // Strategy 1: Legacy XML (strip fmt=srv3, same as Python's youtube-transcript-api)
    let xml_url = strip_fmt_param(base_url);
    match fetch_body(http, &xml_url).await {
        Ok(body) if !body.is_empty() => {
            if let Ok(snippets) = parse_transcript_xml(&body)
                && !snippets.is_empty()
            {
                return Ok(snippets);
            }
        }
        _ => {}
    }

    // Strategy 2: JSON3 (most reliable for auto-generated captions)
    let json_url = set_fmt_param(base_url, "json3");
    let body = fetch_body(http, &json_url).await?;
    parse_transcript_json3(&body)
}

/// Fetch a URL body using the shared client.
async fn fetch_body(http: &reqwest::Client, url: &str) -> Result<String, String> {
    let resp = http
        .get(url)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("YouTube returned status {}", resp.status()));
    }

    resp.text()
        .await
        .map_err(|e| format!("Failed to read response body: {e}"))
}

// ============================================================================
// Transcript parsing
// ============================================================================

/// Parse legacy XML transcript (`<transcript><text start="..." dur="...">...`).
fn parse_transcript_xml(xml: &str) -> Result<Vec<TranscriptSnippet>, String> {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    let mut reader = Reader::from_str(xml);
    let mut snippets = Vec::new();
    let mut start = 0.0_f64;
    let mut duration = 0.0_f64;
    let mut in_text = false;
    let mut buf = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) if e.name().as_ref() == b"text" => {
                start = 0.0;
                duration = 0.0;
                buf.clear();
                for attr in e.attributes().flatten() {
                    match attr.key.as_ref() {
                        b"start" => {
                            start = std::str::from_utf8(&attr.value)
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0);
                        }
                        b"dur" => {
                            duration = std::str::from_utf8(&attr.value)
                                .unwrap_or("0")
                                .parse()
                                .unwrap_or(0.0);
                        }
                        _ => {}
                    }
                }
                in_text = true;
            }
            Ok(Event::Text(e)) if in_text => {
                if let Ok(decoded) = e.decode() {
                    buf.push_str(&decoded);
                }
            }
            // Transcripts are dense in `&#39;`; dropping these would strip
            // every apostrophe from the captions.
            Ok(Event::GeneralRef(e)) if in_text => {
                if let Some(resolved) = crate::xml::resolve_general_ref(&e) {
                    buf.push_str(&resolved);
                }
            }
            Ok(Event::End(e)) if e.name().as_ref() == b"text" => {
                let trimmed = buf.trim();
                if !trimmed.is_empty() {
                    snippets.push(TranscriptSnippet {
                        text: trimmed.to_string(),
                        start,
                        duration,
                    });
                }
                in_text = false;
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {e}")),
            _ => {}
        }
    }

    Ok(snippets)
}

/// Parse JSON3 transcript (`{"events": [{"tStartMs": ..., "segs": [...]}]}`).
fn parse_transcript_json3(json: &str) -> Result<Vec<TranscriptSnippet>, String> {
    #[derive(serde::Deserialize)]
    struct Response {
        events: Vec<Event>,
    }
    #[derive(serde::Deserialize)]
    struct Event {
        #[serde(rename = "tStartMs", default)]
        t_start_ms: u64,
        #[serde(rename = "dDurationMs", default)]
        d_duration_ms: u64,
        #[serde(default)]
        segs: Vec<Seg>,
    }
    #[derive(serde::Deserialize)]
    struct Seg {
        #[serde(default)]
        utf8: String,
    }

    let data: Response =
        serde_json::from_str(json).map_err(|e| format!("JSON3 parse failed: {e}"))?;

    let snippets: Vec<TranscriptSnippet> = data
        .events
        .into_iter()
        .filter(|e| !e.segs.is_empty())
        .map(|e| {
            let text: String = e.segs.iter().map(|s| s.utf8.as_str()).collect();
            TranscriptSnippet {
                text: text.trim().to_string(),
                start: e.t_start_ms as f64 / 1000.0,
                duration: e.d_duration_ms as f64 / 1000.0,
            }
        })
        .filter(|s| !s.text.is_empty())
        .collect();

    if snippets.is_empty() {
        return Err("No transcript text found in JSON3 response".into());
    }

    Ok(snippets)
}

// ============================================================================
// URL helpers
// ============================================================================

/// Strip the `&fmt=...` parameter from a caption URL.
fn strip_fmt_param(url: &str) -> String {
    if let Some(pos) = url.find("&fmt=") {
        let end = url[pos + 5..].find('&').map_or(url.len(), |i| pos + 5 + i);
        format!("{}{}", &url[..pos], &url[end..])
    } else {
        url.to_string()
    }
}

/// Set the `fmt` parameter to a specific value in a caption URL.
fn set_fmt_param(url: &str, fmt: &str) -> String {
    let stripped = strip_fmt_param(url);
    if stripped.contains('?') {
        format!("{stripped}&fmt={fmt}")
    } else {
        format!("{stripped}?fmt={fmt}")
    }
}

// ============================================================================
// Error helpers
// ============================================================================

/// Shorthand for creating a `SourceError::ProcessingFailed` `AppError`.
fn source_err(reason: &str) -> AppError {
    AppError::from(SourceError::ProcessingFailed {
        reason: reason.to_string(),
    })
}

/// Extract a string field from a JSON value.
fn json_str(value: &serde_json::Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(String::from)
}

// ============================================================================
// URL Validation and Video ID Extraction
// ============================================================================

/// Extract the YouTube video ID from a URL.
///
/// Supports:
/// - `youtube.com/watch?v=ID`
/// - `youtu.be/ID`
/// - `youtube.com/embed/ID`
/// - `youtube.com/shorts/ID`
/// - `youtube.com/live/ID`
/// - `m.youtube.com/watch?v=ID`
/// - URLs with additional query params (`&t=123`, `&list=...`)
pub fn extract_youtube_video_id(url_str: &str) -> Result<String, AppError> {
    let parsed =
        Url::parse(url_str).map_err(|_| AppError::Validation("Invalid URL format".into()))?;

    let host = parsed.host_str().unwrap_or("");

    // Normalize host — strip `www.` and `m.` prefixes
    let normalized_host = host
        .strip_prefix("www.")
        .or_else(|| host.strip_prefix("m."))
        .unwrap_or(host);

    let video_id = match normalized_host {
        "youtube.com" => extract_from_youtube_com(&parsed),
        "youtu.be" => extract_from_youtu_be(&parsed),
        _ => None,
    };

    match video_id {
        Some(id) if is_valid_video_id(&id) => Ok(id),
        Some(_) => Err(AppError::Validation(
            "Invalid YouTube video ID format".into(),
        )),
        None => Err(AppError::Validation(
            "Invalid YouTube URL. Please enter a valid youtube.com or youtu.be link.".into(),
        )),
    }
}

/// Extract video ID from youtube.com URLs.
fn extract_from_youtube_com(url: &Url) -> Option<String> {
    let path = url.path();

    // /watch?v=ID
    if path == "/watch" || path.starts_with("/watch") {
        return url
            .query_pairs()
            .find(|(key, _)| key == "v")
            .map(|(_, value)| value.into_owned());
    }

    // /embed/ID, /shorts/ID, /live/ID
    let prefixes = ["/embed/", "/shorts/", "/live/"];
    for prefix in prefixes {
        if let Some(rest) = path.strip_prefix(prefix) {
            let id = rest.split('/').next().unwrap_or(rest);
            if !id.is_empty() {
                return Some(id.to_string());
            }
        }
    }

    None
}

/// Extract video ID from youtu.be/ID short URLs.
fn extract_from_youtu_be(url: &Url) -> Option<String> {
    let path = url.path().strip_prefix('/')?;
    let id = path.split('/').next().unwrap_or(path);
    if id.is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}

/// Validate locale code: must be exactly 2 lowercase ASCII letters (ISO 639-1).
fn is_valid_locale(locale: &str) -> bool {
    locale.len() == 2 && locale.bytes().all(|b| b.is_ascii_lowercase())
}

/// Validate video ID format: 11 characters, alphanumeric + `-` + `_`.
fn is_valid_video_id(id: &str) -> bool {
    id.len() == 11
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

// ============================================================================
// Transcript Formatting
// ============================================================================

/// Format transcript snippets as timestamped Markdown.
///
/// Groups snippets into ~60-second sections with `## [MM:SS - MM:SS]` headers.
/// Each snippet line is prefixed with `[MM:SS]`.
pub fn format_transcript_as_markdown(snippets: &[TranscriptSnippet], title: &str) -> String {
    if snippets.is_empty() {
        return String::new();
    }

    let mut output = String::with_capacity(snippets.len() * 80);

    if !title.is_empty() {
        let safe_title = title
            .replace('\\', "\\\\")
            .replace('#', "\\#")
            .replace('[', "\\[")
            .replace(']', "\\]")
            .replace('`', "\\`");
        output.push_str("# ");
        output.push_str(&safe_title);
        output.push_str("\n\n");
    }

    let section_duration = 60.0;
    let mut section_start = snippets[0].start;
    let mut section_end = section_start + section_duration;

    output.push_str(&format!(
        "## [{}]\n\n",
        format_section_range(section_start, section_end.min(last_snippet_end(snippets))),
    ));

    for snippet in snippets {
        if snippet.start >= section_end {
            section_start = snippet.start;
            section_end = section_start + section_duration;
            output.push_str(&format!(
                "\n## [{}]\n\n",
                format_section_range(section_start, section_end.min(last_snippet_end(snippets))),
            ));
        }

        output.push_str(&format!(
            "[{}] {}\n",
            format_timestamp(snippet.start),
            snippet.text.trim(),
        ));
    }

    output
}

fn format_timestamp(seconds: f64) -> String {
    let total_secs = seconds as u64;
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    format!("{mins:02}:{secs:02}")
}

fn format_section_range(start: f64, end: f64) -> String {
    format!("{} - {}", format_timestamp(start), format_timestamp(end))
}

fn last_snippet_end(snippets: &[TranscriptSnippet]) -> f64 {
    snippets.last().map(|s| s.start + s.duration).unwrap_or(0.0)
}

/// Extract the timestamp range from a chunk of transcript text.
///
/// Parses `[MM:SS]` prefixes from each line and returns the earliest start
/// and latest end timestamps found. Returns `(None, None)` if no timestamps.
pub fn extract_timestamp_range(content: &str) -> (Option<f64>, Option<f64>) {
    let mut min_start: Option<f64> = None;
    let mut max_end: Option<f64> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(ts) = parse_timestamp_prefix(trimmed) {
            match min_start {
                None => min_start = Some(ts),
                Some(current) if ts < current => min_start = Some(ts),
                _ => {}
            }
            match max_end {
                None => max_end = Some(ts),
                Some(current) if ts > current => max_end = Some(ts),
                _ => {}
            }
        }
    }

    (min_start, max_end)
}

fn parse_timestamp_prefix(line: &str) -> Option<f64> {
    let rest = line.strip_prefix('[')?;
    let end = rest.find(']')?;
    let ts_str = &rest[..end];
    let (mins_str, secs_str) = ts_str.split_once(':')?;
    let mins: f64 = mins_str.trim().parse().ok()?;
    let secs: f64 = secs_str.trim().parse().ok()?;
    Some(mins * 60.0 + secs)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── URL extraction tests ────────────────────────────────────────────

    #[test]
    fn extract_standard_watch_url() {
        let id = extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn extract_watch_url_with_extra_params() {
        let id =
            extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn extract_short_url() {
        let id = extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn extract_embed_url() {
        let id = extract_youtube_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn extract_shorts_url() {
        let id = extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn extract_mobile_url() {
        let id = extract_youtube_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn extract_live_url() {
        let id = extract_youtube_video_id("https://www.youtube.com/live/dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn extract_no_www_url() {
        let id = extract_youtube_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ").unwrap();
        assert_eq!(id, "dQw4w9WgXcQ");
    }

    #[test]
    fn reject_non_youtube_url() {
        assert!(extract_youtube_video_id("https://example.com/watch?v=dQw4w9WgXcQ").is_err());
    }

    #[test]
    fn reject_missing_video_id() {
        assert!(extract_youtube_video_id("https://www.youtube.com/watch").is_err());
    }

    #[test]
    fn reject_invalid_video_id_length() {
        assert!(extract_youtube_video_id("https://www.youtube.com/watch?v=short").is_err());
    }

    #[test]
    fn reject_invalid_url() {
        assert!(extract_youtube_video_id("not a url").is_err());
    }

    // ── Transcript formatting tests ─────────────────────────────────────

    #[test]
    fn format_empty_transcript() {
        assert!(format_transcript_as_markdown(&[], "Test").is_empty());
    }

    #[test]
    fn format_single_snippet() {
        let snippets = vec![TranscriptSnippet {
            text: "Hello world".into(),
            start: 0.0,
            duration: 5.0,
        }];
        let result = format_transcript_as_markdown(&snippets, "Video");
        assert!(result.contains("# Video"));
        assert!(result.contains("[00:00]"));
        assert!(result.contains("Hello world"));
    }

    #[test]
    fn format_creates_section_headers() {
        let snippets = vec![
            TranscriptSnippet {
                text: "Start".into(),
                start: 0.0,
                duration: 5.0,
            },
            TranscriptSnippet {
                text: "Middle".into(),
                start: 65.0,
                duration: 5.0,
            },
        ];
        let result = format_transcript_as_markdown(&snippets, "");
        assert_eq!(result.matches("## [").count(), 2);
    }

    #[test]
    fn format_timestamp_formatting() {
        assert_eq!(format_timestamp(0.0), "00:00");
        assert_eq!(format_timestamp(65.5), "01:05");
        assert_eq!(format_timestamp(3661.0), "61:01");
    }

    // ── Video ID validation tests ───────────────────────────────────────

    #[test]
    fn valid_video_id() {
        assert!(is_valid_video_id("dQw4w9WgXcQ"));
        assert!(is_valid_video_id("abc-_123ABC"));
    }

    #[test]
    fn invalid_video_id_wrong_length() {
        assert!(!is_valid_video_id("short"));
        assert!(!is_valid_video_id("waytoolongforavideoid"));
    }

    #[test]
    fn invalid_video_id_bad_chars() {
        assert!(!is_valid_video_id("abc!@#$%^&()"));
    }

    // ── Timestamp range extraction tests ────────────────────────────────

    #[test]
    fn extract_timestamps_from_chunk() {
        let (start, end) = extract_timestamp_range("[00:00] Hello\n[00:30] World\n[01:05] End");
        assert_eq!(start, Some(0.0));
        assert_eq!(end, Some(65.0));
    }

    #[test]
    fn extract_timestamps_with_section_headers() {
        let (start, end) =
            extract_timestamp_range("## [00:00 - 01:00]\n\n[00:00] Hello\n[00:45] There");
        assert_eq!(start, Some(0.0));
        assert_eq!(end, Some(45.0));
    }

    #[test]
    fn extract_timestamps_empty_content() {
        let (start, end) = extract_timestamp_range("No timestamps here");
        assert!(start.is_none());
        assert!(end.is_none());
    }

    // ── Caption URL normalization tests ──────────────────────────────────

    #[test]
    fn strip_fmt_removes_srv3() {
        let url = "https://www.youtube.com/api/timedtext?v=ID&lang=fr&fmt=srv3&key=yt8";
        let stripped = strip_fmt_param(url);
        assert!(!stripped.contains("fmt="));
        assert!(stripped.contains("&key=yt8"));
    }

    #[test]
    fn strip_fmt_no_op_when_missing() {
        let url = "https://www.youtube.com/api/timedtext?v=ID&lang=fr";
        assert_eq!(strip_fmt_param(url), url);
    }

    #[test]
    fn strip_fmt_at_end() {
        let stripped = strip_fmt_param("https://www.youtube.com/api/timedtext?v=ID&fmt=srv3");
        assert!(!stripped.contains("fmt="));
    }

    #[test]
    fn set_fmt_json3() {
        let url = "https://www.youtube.com/api/timedtext?v=ID&fmt=srv3&key=yt8";
        let result = set_fmt_param(url, "json3");
        assert!(result.contains("&fmt=json3"));
        assert!(!result.contains("srv3"));
    }

    // ── InnerTube body tests ────────────────────────────────────────────

    #[test]
    fn innertube_body_structure() {
        let body = innertube_body("dQw4w9WgXcQ");
        assert_eq!(body["videoId"], "dQw4w9WgXcQ");
        assert_eq!(body["context"]["client"]["clientName"], "ANDROID");
        assert_eq!(
            body["context"]["client"]["clientVersion"],
            ANDROID_CLIENT_VERSION
        );
    }

    // ── XML parsing tests ───────────────────────────────────────────────

    #[test]
    fn parse_xml_basic() {
        let xml = r#"<transcript>
            <text start="0.5" dur="2.3">Hello world</text>
            <text start="2.8" dur="1.9">Second line</text>
        </transcript>"#;
        let snippets = parse_transcript_xml(xml).unwrap();
        assert_eq!(snippets.len(), 2);
        assert_eq!(snippets[0].text, "Hello world");
        assert!((snippets[0].start - 0.5).abs() < 0.01);
        assert!((snippets[0].duration - 2.3).abs() < 0.01);
    }

    #[test]
    fn parse_xml_with_entities() {
        let xml = r#"<transcript>
            <text start="0" dur="1">I can&#39;t &amp; won&#39;t</text>
        </transcript>"#;
        let snippets = parse_transcript_xml(xml).unwrap();
        assert_eq!(snippets[0].text, "I can't & won't");
    }

    #[test]
    fn parse_xml_empty() {
        let snippets = parse_transcript_xml("<transcript></transcript>").unwrap();
        assert!(snippets.is_empty());
    }

    // ── JSON3 parsing tests ─────────────────────────────────────────────

    #[test]
    fn parse_json3_basic() {
        let json = r#"{"events":[
            {"tStartMs":500,"dDurationMs":2300,"segs":[{"utf8":"Hello "},{"utf8":"world"}]},
            {"tStartMs":2800,"dDurationMs":1900,"segs":[{"utf8":"Second line"}]}
        ]}"#;
        let snippets = parse_transcript_json3(json).unwrap();
        assert_eq!(snippets.len(), 2);
        assert_eq!(snippets[0].text, "Hello world");
        assert!((snippets[0].start - 0.5).abs() < 0.01);
        assert!((snippets[0].duration - 2.3).abs() < 0.01);
    }

    #[test]
    fn parse_json3_skips_empty_events() {
        let json = r#"{"events":[
            {"tStartMs":0,"dDurationMs":1000},
            {"tStartMs":1000,"dDurationMs":2000,"segs":[{"utf8":"Hello"}]},
            {"tStartMs":3000,"dDurationMs":1000,"segs":[{"utf8":"  "}]}
        ]}"#;
        let snippets = parse_transcript_json3(json).unwrap();
        assert_eq!(snippets.len(), 1);
        assert_eq!(snippets[0].text, "Hello");
    }

    // ── Playability check tests ─────────────────────────────────────────

    #[test]
    fn playability_ok() {
        let data = serde_json::json!({"playabilityStatus": {"status": "OK"}});
        assert!(check_playability(&data).is_ok());
    }

    #[test]
    fn playability_error() {
        let data = serde_json::json!({"playabilityStatus": {"status": "ERROR", "reason": "gone"}});
        assert!(check_playability(&data).is_err());
    }

    #[test]
    fn playability_age_restricted() {
        let data = serde_json::json!({"playabilityStatus": {
            "status": "LOGIN_REQUIRED",
            "reason": "Sign in to confirm your age"
        }});
        let err = check_playability(&data).unwrap_err();
        assert!(err.0.to_string().contains("age verification"));
    }
}
