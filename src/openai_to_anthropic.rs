use crate::models::*;
use serde_json::json;

pub fn format_openai_to_anthropic(resp: OpenAIResponse) -> AnthropicResponse {
    let choice = resp.choices.first();
    let mut content = Vec::new();

    if let Some(choice) = choice {
        if let Some(text) = &choice.message.content {
            content.push(json!({ "type": "text", "text": text }));
        }

        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                content.push(json!({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments).unwrap_or(json!({})),
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

    AnthropicResponse {
        id,
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        stop_reason,
        stop_sequence: None,
        model: resp.model,
        usage,
    }
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

        let mapped = format_openai_to_anthropic(response);

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

        let mapped_first = format_openai_to_anthropic(first);
        let mapped_second = format_openai_to_anthropic(second);

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
        assert_eq!(format_openai_to_anthropic(response).stop_reason, "tool_use");
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
            format_openai_to_anthropic(response).stop_reason,
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
        assert_eq!(format_openai_to_anthropic(response).stop_reason, "end_turn");
    }

    #[test]
    fn id_is_normalized_with_msg_prefix() {
        let response = OpenAIResponse {
            id: "chatcmpl-abc123".to_string(),
            choices: vec![sample_choice("stop")],
            model: "test".to_string(),
            usage: None,
        };
        let mapped = format_openai_to_anthropic(response);
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
        let mapped = format_openai_to_anthropic(response);
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
        let mapped = format_openai_to_anthropic(response);
        let usage = mapped.usage.expect("usage should be present");
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 100);
    }
}
