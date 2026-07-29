//! OpenAI-compatible SSE parsing.
//!
//! Shared parser for streaming responses from OpenAI and Mistral,
//! which both use the same SSE chunk format.

use super::types::LlmStreamEvent;

/// Shared SSE parser for OpenAI-compatible streaming responses.
///
/// Used by OpenAI and Mistral clients which both use the same SSE format.
/// Handles `[DONE]` sentinel, content deltas, and finish reasons.
pub fn parse_openai_sse_data(data: &str) -> Option<LlmStreamEvent> {
    if data == "[DONE]" {
        return Some(LlmStreamEvent::Done);
    }

    let chunk: OpenAiStreamChunk = serde_json::from_str(data).ok()?;
    let choice = chunk.choices.first()?;

    if let Some(content) = &choice.delta.content {
        return Some(LlmStreamEvent::TextDelta {
            text: content.clone(),
        });
    }

    choice.finish_reason.as_ref().map(|_| LlmStreamEvent::Done)
}

/// OpenAI-compatible SSE stream chunk (shared across providers).
#[derive(Debug, serde::Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_openai_sse_done() {
        assert_eq!(parse_openai_sse_data("[DONE]"), Some(LlmStreamEvent::Done));
    }

    #[test]
    fn parse_openai_sse_text_delta() {
        let data = r#"{"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}"#;
        assert_eq!(
            parse_openai_sse_data(data),
            Some(LlmStreamEvent::TextDelta {
                text: "Hello".to_string()
            })
        );
    }

    #[test]
    fn parse_openai_sse_empty_delta() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":null}]}"#;
        assert_eq!(parse_openai_sse_data(data), None);
    }

    #[test]
    fn parse_openai_sse_finish_reason() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        assert_eq!(parse_openai_sse_data(data), Some(LlmStreamEvent::Done));
    }

    #[test]
    fn parse_openai_sse_malformed_json() {
        assert_eq!(parse_openai_sse_data("not json"), None);
    }
}
