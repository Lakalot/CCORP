use crate::config::Config;
use crate::models::*;
use serde_json::json;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationError {
    message: String,
}

impl TranslationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub fn map_model(anthropic_model: &str, settings: &Config) -> String {
    if anthropic_model.contains("haiku") {
        settings.model_haiku.clone()
    } else if anthropic_model.contains("sonnet") {
        settings.model_sonnet.clone()
    } else if anthropic_model.contains("opus") {
        settings.model_opus.clone()
    } else {
        anthropic_model.to_string()
    }
}

pub fn format_anthropic_to_openai(
    req: AnthropicRequest,
    settings: &Config,
) -> Result<OpenAIRequest, TranslationError> {
    let mut openapi_messages = Vec::new();

    if let Some(system) = req.system {
        let system_text = parse_system_content(&system)?;
        if !system_text.is_empty() {
            openapi_messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(system_text),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    for message in req.messages {
        match message.role.as_str() {
            "user" => {
                if let Some(content_array) = message.content.as_array() {
                    let mut user_text = String::new();
                    for content in content_array {
                        let block_type = content
                            .get("type")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| TranslationError::new("Missing content block type"))?;
                        match block_type {
                            "text" => {
                                let text =
                                    content.get("text").and_then(|v| v.as_str()).ok_or_else(
                                        || TranslationError::new("Invalid text content block"),
                                    )?;
                                user_text.push_str(text);
                            }
                            "tool_result" => {
                                let tool_use_id = content
                                    .get("tool_use_id")
                                    .and_then(|v| v.as_str())
                                    .ok_or_else(|| {
                                        TranslationError::new(
                                            "Invalid tool_result block: missing tool_use_id",
                                        )
                                    })?;
                                let result_content = content.get("content").ok_or_else(|| {
                                    TranslationError::new(
                                        "Invalid tool_result block: missing content",
                                    )
                                })?;
                                let content_str = if let Some(s) = result_content.as_str() {
                                    s.to_string()
                                } else {
                                    result_content.to_string()
                                };
                                openapi_messages.push(OpenAIMessage {
                                    role: "tool".to_string(),
                                    content: Some(content_str),
                                    tool_call_id: Some(tool_use_id.to_string()),
                                    tool_calls: None,
                                });
                            }
                            _ => {
                                return Err(TranslationError::new(format!(
                                    "Unsupported user content block type: {block_type}"
                                )));
                            }
                        }
                    }
                    if !user_text.is_empty() {
                        openapi_messages.push(OpenAIMessage {
                            role: "user".to_string(),
                            content: Some(user_text),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                } else if let Some(content_str) = message.content.as_str() {
                    openapi_messages.push(OpenAIMessage {
                        role: "user".to_string(),
                        content: Some(content_str.to_string()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                } else {
                    return Err(TranslationError::new("Invalid user message content"));
                }
            }
            "assistant" => {
                let mut assistant_message = OpenAIMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: None,
                    tool_call_id: None,
                };
                let mut tool_calls = Vec::new();
                if let Some(content_array) = message.content.as_array() {
                    let mut assistant_text = String::new();
                    for content in content_array {
                        let block_type = content
                            .get("type")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| TranslationError::new("Missing content block type"))?;
                        match block_type {
                            "text" => {
                                let text =
                                    content.get("text").and_then(|v| v.as_str()).ok_or_else(
                                        || TranslationError::new("Invalid text content block"),
                                    )?;
                                assistant_text.push_str(text);
                            }
                            "tool_use" => {
                                let tool_id = content
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .ok_or_else(|| {
                                        TranslationError::new("Invalid tool_use block: missing id")
                                    })?;
                                let tool_name = content
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .ok_or_else(|| {
                                        TranslationError::new(
                                            "Invalid tool_use block: missing name",
                                        )
                                    })?;
                                let input = content.get("input").ok_or_else(|| {
                                    TranslationError::new("Invalid tool_use block: missing input")
                                })?;
                                tool_calls.push(OpenAIToolCall {
                                    id: tool_id.to_string(),
                                    tool_type: "function".to_string(),
                                    function: OpenAIFunction {
                                        name: tool_name.to_string(),
                                        arguments: input.to_string(),
                                    },
                                });
                            }
                            _ => {
                                return Err(TranslationError::new(format!(
                                    "Unsupported assistant content block type: {block_type}"
                                )));
                            }
                        }
                    }
                    if !assistant_text.is_empty() {
                        assistant_message.content = Some(assistant_text);
                    }
                }
                if !tool_calls.is_empty() {
                    assistant_message.tool_calls = Some(tool_calls);
                }
                openapi_messages.push(assistant_message);
            }
            "system" => {
                if let Some(content_str) = message.content.as_str() {
                    openapi_messages.push(OpenAIMessage {
                        role: "system".to_string(),
                        content: Some(content_str.to_string()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                } else {
                    return Err(TranslationError::new("Invalid system message content"));
                }
            }
            other => {
                return Err(TranslationError::new(format!(
                    "Unsupported message role: {other}"
                )));
            }
        }
    }

    let mut tools = None;
    if let Some(anthropic_tools) = req.tools {
        tools = Some(
            anthropic_tools
                .into_iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": t["input_schema"],
                        }
                    })
                })
                .collect(),
        );
    }

    Ok(OpenAIRequest {
        model: map_model(&req.model, settings),
        messages: openapi_messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        stop: req.stop_sequences,
        stream: req.stream,
        tools,
    })
}

fn parse_system_content(system: &serde_json::Value) -> Result<String, TranslationError> {
    if let Some(system_str) = system.as_str() {
        return Ok(system_str.to_string());
    }

    if let Some(blocks) = system.as_array() {
        let mut text = String::new();
        for block in blocks {
            let block_type = block
                .get("type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| TranslationError::new("Missing system block type"))?;
            match block_type {
                "text" => {
                    let value = block
                        .get("text")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| TranslationError::new("Invalid system text block"))?;
                    text.push_str(value);
                }
                _ => {
                    return Err(TranslationError::new(format!(
                        "Unsupported system block type: {block_type}"
                    )));
                }
            }
        }
        return Ok(text);
    }

    Err(TranslationError::new("Invalid system content"))
}

#[cfg(test)]
mod tests {
    use super::{format_anthropic_to_openai, map_model};
    use crate::{config::Config, models::AnthropicRequest};
    use serde_json::json;

    fn test_config() -> Config {
        Config {
            port: 3332,
            base_url: "http://localhost".to_string(),
            api_key: "dummy".to_string(),
            inbound_api_key: "dummy".to_string(),
            model_haiku: "mapped-haiku".to_string(),
            model_sonnet: "mapped-sonnet".to_string(),
            model_opus: "mapped-opus".to_string(),
        }
    }

    #[test]
    fn map_model_uses_alias_mapping_from_config() {
        let cfg = test_config();
        assert_eq!(map_model("claude-3-5-haiku-latest", &cfg), "mapped-haiku");
        assert_eq!(map_model("claude-sonnet-4", &cfg), "mapped-sonnet");
        assert_eq!(map_model("claude-opus-4", &cfg), "mapped-opus");
        assert_eq!(map_model("custom-model", &cfg), "custom-model");
    }

    #[test]
    fn format_maps_system_text_and_stream_flag() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4",
            "system": "You are concise.",
            "messages": [
                {"role":"user","content":"Hello"}
            ],
            "stream": true
        }))
        .expect("request should deserialize");

        let mapped = format_anthropic_to_openai(req, &cfg).expect("translation should succeed");
        assert_eq!(mapped.model, "mapped-sonnet");
        assert_eq!(mapped.stream, Some(true));
        assert_eq!(mapped.messages.len(), 2);
        assert_eq!(mapped.messages[0].role, "system");
        assert_eq!(
            mapped.messages[0].content.as_deref(),
            Some("You are concise.")
        );
        assert_eq!(mapped.messages[1].role, "user");
        assert_eq!(mapped.messages[1].content.as_deref(), Some("Hello"));
    }

    #[test]
    fn format_maps_tool_use_and_tool_result_blocks() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model":"claude-haiku-3-5",
            "messages":[
                {"role":"assistant","content":[
                    {"type":"text","text":"Checking..."},
                    {"type":"tool_use","id":"toolu_1","name":"weather","input":{"city":"Paris"}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_1","content":{"temp_c":12}}
                ]}
            ]
        }))
        .expect("request should deserialize");

        let mapped = format_anthropic_to_openai(req, &cfg).expect("translation should succeed");
        assert_eq!(mapped.model, "mapped-haiku");
        assert_eq!(mapped.messages.len(), 2);

        let assistant = &mapped.messages[0];
        assert_eq!(assistant.role, "assistant");
        assert_eq!(assistant.content.as_deref(), Some("Checking..."));
        let tool_calls = assistant
            .tool_calls
            .as_ref()
            .expect("tool_calls should be present");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "toolu_1");
        assert_eq!(tool_calls[0].function.name, "weather");
        assert_eq!(tool_calls[0].function.arguments, r#"{"city":"Paris"}"#);

        let tool = &mapped.messages[1];
        assert_eq!(tool.role, "tool");
        assert_eq!(tool.tool_call_id.as_deref(), Some("toolu_1"));
        assert_eq!(tool.content.as_deref(), Some(r#"{"temp_c":12}"#));
    }

    #[test]
    fn format_rejects_unknown_blocks_with_validation_error() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model":"claude-opus-4",
            "messages":[
                {"role":"user","content":[
                    {"type":"text","text":"A"},
                    {"type":"unknown_block","foo":"bar"},
                    {"type":"text","text":"B"}
                ]}
            ]
        }))
        .expect("request should deserialize");

        let err = format_anthropic_to_openai(req, &cfg).expect_err("translation should fail");
        assert_eq!(
            err.message(),
            "Unsupported user content block type: unknown_block"
        );
    }

    #[test]
    fn format_supports_system_array_blocks() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4",
            "system": [
                {"type":"text","text":"You are "},
                {"type":"text","text":"strict."}
            ],
            "messages": [{"role":"user","content":"Hi"}]
        }))
        .expect("request should deserialize");

        let mapped = format_anthropic_to_openai(req, &cfg).expect("translation should succeed");
        assert_eq!(mapped.messages[0].role, "system");
        assert_eq!(
            mapped.messages[0].content.as_deref(),
            Some("You are strict.")
        );
    }

    #[test]
    fn format_rejects_unsupported_system_block_type() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4",
            "system": [{"type":"image","data":"abc"}],
            "messages": [{"role":"user","content":"Hi"}]
        }))
        .expect("request should deserialize");

        let err = format_anthropic_to_openai(req, &cfg).expect_err("translation should fail");
        assert_eq!(err.message(), "Unsupported system block type: image");
    }

    #[test]
    fn format_maps_max_tokens_top_p_and_stop_sequences() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop_sequences": ["END", "STOP"],
            "messages": [{"role":"user","content":"Hello"}]
        }))
        .expect("request should deserialize");

        let mapped = format_anthropic_to_openai(req, &cfg).expect("translation should succeed");
        assert_eq!(mapped.max_tokens, Some(1024));
        assert_eq!(mapped.temperature, Some(0.7));
        assert_eq!(mapped.top_p, Some(0.9));
        assert_eq!(
            mapped.stop,
            Some(vec!["END".to_string(), "STOP".to_string()])
        );
    }

    #[test]
    fn format_preserves_stream_false() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4",
            "messages": [{"role":"user","content":"Hello"}],
            "stream": false
        }))
        .expect("request should deserialize");

        let mapped = format_anthropic_to_openai(req, &cfg).expect("translation should succeed");
        assert_eq!(mapped.stream, Some(false));
    }

    #[test]
    fn format_omits_optional_fields_when_absent() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4",
            "messages": [{"role":"user","content":"Hello"}]
        }))
        .expect("request should deserialize");

        let mapped = format_anthropic_to_openai(req, &cfg).expect("translation should succeed");
        assert_eq!(mapped.max_tokens, None);
        assert_eq!(mapped.temperature, None);
        assert_eq!(mapped.top_p, None);
        assert_eq!(mapped.stop, None);
        assert_eq!(mapped.stream, None);
        assert!(mapped.tools.is_none());
    }

    #[test]
    fn format_tool_result_string_content_not_double_quoted() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model":"claude-sonnet-4",
            "messages":[
                {"role":"assistant","content":[
                    {"type":"tool_use","id":"toolu_1","name":"weather","input":{}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"}
                ]}
            ]
        }))
        .expect("request should deserialize");

        let mapped = format_anthropic_to_openai(req, &cfg).expect("translation should succeed");
        let tool_msg = &mapped.messages[1];
        assert_eq!(tool_msg.role, "tool");
        assert_eq!(tool_msg.content.as_deref(), Some("sunny"));
    }

    #[test]
    fn format_maps_tools_to_openai_function_schema() {
        let cfg = test_config();
        let req: AnthropicRequest = serde_json::from_value(json!({
            "model":"claude-sonnet-4",
            "messages":[{"role":"user","content":"Hi"}],
            "tools":[
                {
                    "name":"weather_lookup",
                    "description":"Get weather",
                    "input_schema":{
                        "type":"object",
                        "properties":{"city":{"type":"string"}},
                        "required":["city"]
                    }
                }
            ]
        }))
        .expect("request should deserialize");

        let mapped = format_anthropic_to_openai(req, &cfg).expect("translation should succeed");
        let tools = mapped.tools.expect("tools should exist");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "weather_lookup");
        assert_eq!(tools[0]["function"]["description"], "Get weather");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["city"]["type"],
            "string"
        );
    }
}
