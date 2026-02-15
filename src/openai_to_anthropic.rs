use crate::models::*;
use serde_json::json;

pub fn format_openai_to_anthropic(resp: OpenAIResponse) -> Result<AnthropicResponse, String> {
    let choice = resp.choices.first();
    let mut content = Vec::new();

    if let Some(choice) = choice {
        if let Some(text) = &choice.message.content {
            content.push(json!({ "type": "text", "text": text }));
        }

        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                let input =
                    serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments)
                        .map_err(|e| {
                            format!(
                                "Invalid tool_call arguments JSON for '{}': {e}",
                                tool_call.function.name
                            )
                        })?;
                content.push(json!({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": input,
                }));
            }
        }
    }

    let stop_reason = match choice.map(|c| c.finish_reason.as_str()) {
        Some("tool_calls") => "tool_use",
        Some("length") => "max_tokens",
        _ => "end_turn",
    }
    .to_string();

    let id = if resp.id.starts_with("msg_") {
        resp.id
    } else {
        format!("msg_{}", resp.id)
    };

    let usage = resp.usage.map(|u| AnthropicUsage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
    });

    Ok(AnthropicResponse {
        id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        stop_reason,
        stop_sequence: None,
        model: resp.model,
        usage,
    })
}

#[cfg(test)]
mod tests {
    use super::format_openai_to_anthropic;
    use crate::models::{
        OpenAIChoice, OpenAIFunction, OpenAIMessage, OpenAIResponse, OpenAIToolCall, OpenAIUsage,
    };

    fn sample_choice(finish_reason: &str) -> OpenAIChoice {
        OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: "assistant".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: finish_reason.to_string(),
        }
    }

    fn sample_choice_with_tool_call(arguments: &str) -> OpenAIChoice {
        OpenAIChoice {
            index: 0,
            message: OpenAIMessage {
                role: "assistant".to_string(),
                content: Some("Hello".to_string()),
                tool_calls: Some(vec![OpenAIToolCall {
                    id: "tool-1".to_string(),
                    tool_type: "function".to_string(),
                    function: OpenAIFunction {
                        name: "lookup".to_string(),
                        arguments: arguments.to_string(),
                    },
                }]),
                tool_call_id: None,
            },
            finish_reason: "tool_calls".to_string(),
        }
    }

    #[test]
    fn mapper_handles_empty_choices_without_panic() {
        let response = OpenAIResponse {
            id: "chatcmpl-empty".to_string(),
            choices: vec![],
            model: "openai-test-model".to_string(),
            usage: None,
        };

        let mapped = format_openai_to_anthropic(response).expect("mapping should succeed");

        assert_eq!(mapped.response_type, "message");
        assert_eq!(mapped.role, "assistant");
        assert!(mapped.content.is_empty());
        assert_eq!(mapped.stop_reason, "end_turn");
        assert!(mapped.usage.is_none());
    }

    #[test]
    fn mapper_is_deterministic_for_equivalent_inputs() {
        let first = OpenAIResponse {
            id: "chatcmpl-test".to_string(),
            choices: vec![sample_choice_with_tool_call("{\"query\":\"weather\"}")],
            model: "openai-test-model".to_string(),
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };
        let second = OpenAIResponse {
            id: "chatcmpl-test".to_string(),
            choices: vec![sample_choice_with_tool_call("{\"query\":\"weather\"}")],
            model: "openai-test-model".to_string(),
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };

        let mapped_first = format_openai_to_anthropic(first).expect("mapping should succeed");
        let mapped_second = format_openai_to_anthropic(second).expect("mapping should succeed");

        assert_eq!(
            serde_json::to_value(mapped_first).expect("first mapping should serialize"),
            serde_json::to_value(mapped_second).expect("second mapping should serialize")
        );
    }

    #[test]
    fn stop_reason_maps_tool_calls_to_tool_use() {
        let response = OpenAIResponse {
            id: "chatcmpl-test".to_string(),
            choices: vec![sample_choice("tool_calls")],
            model: "test".to_string(),
            usage: None,
        };
        assert_eq!(
            format_openai_to_anthropic(response)
                .expect("mapping should succeed")
                .stop_reason,
            "tool_use"
        );
    }

    #[test]
    fn stop_reason_maps_length_to_max_tokens() {
        let response = OpenAIResponse {
            id: "chatcmpl-test".to_string(),
            choices: vec![sample_choice("length")],
            model: "test".to_string(),
            usage: None,
        };
        assert_eq!(
            format_openai_to_anthropic(response)
                .expect("mapping should succeed")
                .stop_reason,
            "max_tokens"
        );
    }

    #[test]
    fn stop_reason_maps_stop_to_end_turn() {
        let response = OpenAIResponse {
            id: "chatcmpl-test".to_string(),
            choices: vec![sample_choice("stop")],
            model: "test".to_string(),
            usage: None,
        };
        assert_eq!(
            format_openai_to_anthropic(response)
                .expect("mapping should succeed")
                .stop_reason,
            "end_turn"
        );
    }

    #[test]
    fn id_is_normalized_with_msg_prefix() {
        let response = OpenAIResponse {
            id: "chatcmpl-abc123".to_string(),
            choices: vec![sample_choice("stop")],
            model: "test".to_string(),
            usage: None,
        };
        let mapped = format_openai_to_anthropic(response).expect("mapping should succeed");
        assert!(
            mapped.id.starts_with("msg_"),
            "ID should start with msg_ prefix, got: {}",
            mapped.id
        );
    }

    #[test]
    fn id_preserves_existing_msg_prefix() {
        let response = OpenAIResponse {
            id: "msg_already-prefixed".to_string(),
            choices: vec![sample_choice("stop")],
            model: "test".to_string(),
            usage: None,
        };
        let mapped = format_openai_to_anthropic(response).expect("mapping should succeed");
        assert_eq!(mapped.id, "msg_already-prefixed");
    }

    #[test]
    fn usage_is_mapped_from_openai_to_anthropic() {
        let response = OpenAIResponse {
            id: "chatcmpl-test".to_string(),
            choices: vec![sample_choice("stop")],
            model: "test".to_string(),
            usage: Some(OpenAIUsage {
                prompt_tokens: 42,
                completion_tokens: 100,
                total_tokens: 142,
            }),
        };
        let mapped = format_openai_to_anthropic(response).expect("mapping should succeed");
        let usage = mapped.usage.expect("usage should be present");
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 100);
    }

    #[test]
    fn tool_calls_with_null_content_produce_only_tool_use_blocks() {
        let response = OpenAIResponse {
            id: "chatcmpl-tool-only".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: OpenAIFunction {
                            name: "weather".to_string(),
                            arguments: r#"{"city":"Paris"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: "tool_calls".to_string(),
            }],
            model: "test".to_string(),
            usage: None,
        };

        let mapped = format_openai_to_anthropic(response).expect("mapping should succeed");
        assert_eq!(mapped.stop_reason, "tool_use");
        assert_eq!(
            mapped.content.len(),
            1,
            "only tool_use block, no text block"
        );
        assert_eq!(
            mapped.content[0].get("type").and_then(|v| v.as_str()),
            Some("tool_use")
        );
    }

    #[test]
    fn malformed_tool_call_arguments_returns_error() {
        let response = OpenAIResponse {
            id: "chatcmpl-bad-args".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![OpenAIToolCall {
                        id: "call_1".to_string(),
                        tool_type: "function".to_string(),
                        function: OpenAIFunction {
                            name: "weather".to_string(),
                            arguments: "{bad-json}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: "tool_calls".to_string(),
            }],
            model: "test".to_string(),
            usage: None,
        };

        let err = format_openai_to_anthropic(response).expect_err("should fail on bad arguments");
        assert!(
            err.contains("Invalid tool_call arguments JSON"),
            "error should mention invalid arguments: {err}"
        );
    }

    #[test]
    fn tool_calls_are_mapped_to_tool_use_with_stable_fields() {
        let response = OpenAIResponse {
            id: "chatcmpl-tool".to_string(),
            choices: vec![sample_choice_with_tool_call(r#"{"city":"Paris"}"#)],
            model: "test".to_string(),
            usage: None,
        };

        let mapped = format_openai_to_anthropic(response).expect("mapping should succeed");
        assert_eq!(mapped.stop_reason, "tool_use");
        assert_eq!(mapped.content.len(), 2);

        let tool_block = mapped
            .content
            .iter()
            .find(|block| block.get("type").and_then(|v| v.as_str()) == Some("tool_use"))
            .expect("tool_use block should exist");
        assert_eq!(
            tool_block.get("id").and_then(|v| v.as_str()),
            Some("tool-1")
        );
        assert_eq!(
            tool_block.get("name").and_then(|v| v.as_str()),
            Some("lookup")
        );
        assert_eq!(
            tool_block
                .get("input")
                .and_then(|v| v.get("city"))
                .and_then(|v| v.as_str()),
            Some("Paris")
        );
    }
}
