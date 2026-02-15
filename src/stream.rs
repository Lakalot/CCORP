//! Stream processing utilities for SSE event translation.

use crate::domain::errors::{ErrorBody, ErrorEnvelope, ErrorSource};

pub(crate) struct ToolCallDelta {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

pub(crate) struct ParsedChunk {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallDelta>,
    pub finish_reason: Option<String>,
}

/// Build an SSE `event: error` frame using the canonical `ErrorEnvelope` structure.
///
/// Reuses `domain::errors` types so stream and non-stream error shapes stay in sync.
pub(crate) fn stream_error_event(source: ErrorSource, message: &str, request_id: &str) -> String {
    let envelope = ErrorEnvelope {
        error_type: "error".to_string(),
        error: ErrorBody {
            source,
            message: message.to_string(),
            request_id: request_id.to_string(),
        },
    };
    let payload =
        serde_json::to_string(&envelope).expect("ErrorEnvelope serialization should never fail");
    format!("event: error\ndata: {payload}\n\n")
}

pub(crate) fn parse_openai_stream_chunk(data: &str) -> Result<Option<ParsedChunk>, ()> {
    let value: serde_json::Value = serde_json::from_str(data).map_err(|_| ())?;
    let Some(choices) = value.get("choices").and_then(serde_json::Value::as_array) else {
        return Ok(None);
    };
    let Some(choice) = choices.first() else {
        return Ok(None);
    };

    let content = choice
        .get("delta")
        .and_then(|delta| delta.get("content"))
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string);
    let finish_reason = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
        .map(ToString::to_string);

    let mut tool_calls = Vec::new();
    if let Some(tc_array) = choice
        .get("delta")
        .and_then(|d| d.get("tool_calls"))
        .and_then(serde_json::Value::as_array)
    {
        for tc in tc_array {
            tool_calls.push(ToolCallDelta {
                id: tc.get("id").and_then(|v| v.as_str()).map(String::from),
                name: tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|v| v.as_str())
                    .map(String::from),
                arguments: tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .and_then(|v| v.as_str())
                    .map(String::from),
            });
        }
    }

    Ok(Some(ParsedChunk {
        content,
        tool_calls,
        finish_reason,
    }))
}

pub(crate) fn map_stop_reason(finish_reason: &str) -> &str {
    match finish_reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_openai_stream_chunk_ignores_non_choice_payloads() {
        let parsed =
            parse_openai_stream_chunk(r#"{"type":"ping"}"#).expect("payload should be valid json");
        assert!(parsed.is_none());
    }

    #[test]
    fn parse_openai_stream_chunk_extracts_content_and_finish_reason() {
        let parsed = parse_openai_stream_chunk(
            r#"{"id":"s","model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"tool_calls"}]}"#,
        )
        .expect("payload should parse")
        .expect("choice payload should be extracted");
        assert_eq!(parsed.content.as_deref(), Some("hi"));
        assert_eq!(parsed.finish_reason.as_deref(), Some("tool_calls"));
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parse_openai_stream_chunk_extracts_tool_call_deltas() {
        let parsed = parse_openai_stream_chunk(
            r#"{"id":"s","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}"#,
        )
        .expect("payload should parse")
        .expect("choice payload should be extracted");
        assert!(parsed.content.is_none());
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(parsed.tool_calls[0].name.as_deref(), Some("lookup"));
        assert_eq!(parsed.tool_calls[0].arguments.as_deref(), Some("{}"));
    }

    #[test]
    fn stream_error_event_uses_canonical_envelope() {
        let event = stream_error_event(ErrorSource::Upstream, "test error", "req-123");
        assert!(event.starts_with("event: error\ndata: "));
        assert!(event.ends_with("\n\n"));
        let data = event
            .strip_prefix("event: error\ndata: ")
            .unwrap()
            .strip_suffix("\n\n")
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(data).expect("should be valid JSON");
        assert_eq!(parsed["type"], "error");
        assert_eq!(parsed["error"]["source"], "upstream");
        assert_eq!(parsed["error"]["message"], "test error");
        assert_eq!(parsed["error"]["request_id"], "req-123");
    }

    #[test]
    fn map_stop_reason_maps_known_values() {
        assert_eq!(map_stop_reason("tool_calls"), "tool_use");
        assert_eq!(map_stop_reason("length"), "max_tokens");
        assert_eq!(map_stop_reason("stop"), "end_turn");
        assert_eq!(map_stop_reason("unknown_future_value"), "end_turn");
    }
}
