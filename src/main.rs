mod anthropic_to_openai;
mod application;
mod config;
mod domain;
mod infrastructure;
mod interfaces;
mod models;
mod openai_to_anthropic;
mod openrouter;
mod stream;
mod switch_model;

use axum::{
    Extension, Router,
    body::Body,
    extract::{Json, Request, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use config::Config;
use domain::errors::ErrorSource;
use futures_util::stream::StreamExt;
use interfaces::error_response;
use models::AnthropicRequest;
use reqwest::Client;
use serde_json::json;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use stream::{map_stop_reason, parse_openai_stream_chunk, stream_error_event};
use tokio::sync::RwLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RwLock<Config>>,
    pub logging_path: Arc<Option<String>>,
    pub client: Client,
    pub inbound_api_key: Arc<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct RequestId(pub(crate) String);

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

const REQUEST_TIMEOUT_SECS: u64 = 60;
/// Default stream mode when the `stream` field is omitted: non-streaming.
const DEFAULT_STREAM_MODE: bool = false;

#[tokio::main]
async fn main() {
    init_tracing();

    let args: Vec<String> = std::env::args().collect();
    let logging_path = match parse_logging_path(&args) {
        Ok(path) => path,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(1);
        }
    };

    let settings = match Config::try_from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    println!("Using the following model mappings:");
    println!("- Haiku: {}", settings.model_haiku);
    println!("- Sonnet: {}", settings.model_sonnet);
    println!("- Opus: {}", settings.model_opus);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], settings.port));

    let inbound_key = settings.inbound_api_key.clone();
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .build()
        .expect("HTTP client should build");
    let state = AppState {
        config: Arc::new(RwLock::new(settings)),
        logging_path: Arc::new(logging_path),
        client,
        inbound_api_key: Arc::new(inbound_key),
    };

    let app = build_router(state);

    println!("listening on {addr}");

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("Unable to bind listener: {error}");
            std::process::exit(1);
        }
    };

    if let Err(error) = axum::serve(listener, app).await {
        eprintln!("Server error: {error}");
        std::process::exit(1);
    }
}

fn init_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ccor=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

fn parse_logging_path(args: &[String]) -> Result<Option<String>, String> {
    if let Some(index) = args.iter().position(|arg| arg == "--logging") {
        if let Some(path) = args.get(index + 1) {
            std::fs::create_dir_all(path)
                .map_err(|error| format!("Failed to create logging directory: {error}"))?;
            println!("Logging requests and responses to: {path}");
            Ok(Some(path.clone()))
        } else {
            Err("--logging flag requires a path argument".to_string())
        }
    } else {
        Ok(None)
    }
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/messages", post(messages_handler))
        .route(
            "/switch-model",
            get(switch_model::switch_model_get).post(switch_model::switch_model_post),
        )
        .layer(middleware::from_fn(timeout_middleware))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(middleware::from_fn(trace_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state)
}

async fn messages_handler(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    payload: Result<Json<AnthropicRequest>, JsonRejection>,
) -> Response {
    let payload = match payload {
        Ok(Json(payload)) => payload,
        Err(_) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                ErrorSource::LocalValidation,
                "Malformed request payload",
                request_id.0,
            )
            .into_response();
        }
    };

    // Resolve stream mode at the request boundary, before translation.
    let is_stream = resolve_stream_mode(payload.stream);

    let settings_guard = state.config.read().await;
    let openai_request =
        match anthropic_to_openai::format_anthropic_to_openai(payload, &settings_guard) {
            Ok(request) => request,
            Err(error) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    ErrorSource::LocalValidation,
                    format!("Invalid Anthropic payload: {}", error.message()),
                    request_id.0,
                )
                .into_response();
            }
        };

    if let Ok(request_json) = serde_json::to_string_pretty(&openai_request) {
        log_payload(&state.logging_path, "request", &request_json);
    }

    let base_url = settings_guard.base_url.clone();
    let upstream_api_key = settings_guard.api_key.clone();
    drop(settings_guard);

    tracing::info!(
        request_id = %request_id.0,
        route = "/v1/messages",
        provider = "openrouter",
        model = %openai_request.model,
        "dispatching request upstream"
    );

    if is_stream {
        forward_stream(
            state.client.clone(),
            base_url,
            upstream_api_key,
            openai_request,
            Arc::clone(&state.logging_path),
            request_id.0,
        )
    } else {
        forward_non_stream(
            &state.client,
            &base_url,
            &upstream_api_key,
            &openai_request,
            &state.logging_path,
            request_id.0,
        )
        .await
    }
}

fn resolve_stream_mode(stream: Option<bool>) -> bool {
    stream.unwrap_or(DEFAULT_STREAM_MODE)
}

/// Constant-time comparison to prevent timing attacks on API key validation.
/// Iterates over the longer input to avoid leaking expected key length.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let max_len = a.len().max(b.len());
    let len_mismatch = if a.len() == b.len() { 0u8 } else { 1u8 };
    let mut acc = 0u8;
    for i in 0..max_len {
        let byte_a = a.get(i).copied().unwrap_or(0);
        let byte_b = b.get(i).copied().unwrap_or(0);
        acc |= byte_a ^ byte_b;
    }
    (acc | len_mismatch) == 0
}

fn validate_inbound_api_key(
    headers: &HeaderMap,
    expected_api_key: &str,
    request_id: &str,
) -> Result<(), Box<Response>> {
    match headers.get("x-api-key") {
        None => Err(Box::new(
            error_response(
                StatusCode::UNAUTHORIZED,
                ErrorSource::LocalAuth,
                "Missing x-api-key header",
                request_id,
            )
            .into_response(),
        )),
        Some(value) => match value.to_str() {
            Ok(text)
                if !text.is_empty()
                    && constant_time_eq(text.as_bytes(), expected_api_key.as_bytes()) =>
            {
                Ok(())
            }
            Err(_) => Err(Box::new(
                error_response(
                    StatusCode::UNAUTHORIZED,
                    ErrorSource::LocalAuth,
                    "Invalid x-api-key",
                    request_id,
                )
                .into_response(),
            )),
            Ok(_) => Err(Box::new(
                error_response(
                    StatusCode::UNAUTHORIZED,
                    ErrorSource::LocalAuth,
                    "Invalid x-api-key",
                    request_id,
                )
                .into_response(),
            )),
        },
    }
}

async fn forward_non_stream(
    client: &Client,
    base_url: &str,
    api_key: &str,
    openai_request: &models::OpenAIRequest,
    logging_path: &Option<String>,
    request_id: String,
) -> Response {
    let res = client
        .post(format!("{base_url}/chat/completions"))
        .bearer_auth(api_key)
        .json(openai_request)
        .send()
        .await;

    let Ok(res) = res else {
        return error_response(
            StatusCode::BAD_GATEWAY,
            ErrorSource::Upstream,
            "Failed to contact upstream provider",
            request_id,
        )
        .into_response();
    };

    if !res.status().is_success() {
        return error_response(
            StatusCode::BAD_GATEWAY,
            ErrorSource::Upstream,
            "Upstream provider returned an error",
            request_id,
        )
        .into_response();
    }

    let openai_response: models::OpenAIResponse = match res.json().await {
        Ok(response) => response,
        Err(_) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                ErrorSource::Upstream,
                "Upstream response could not be parsed",
                request_id,
            )
            .into_response();
        }
    };
    let anthropic_response = match openai_to_anthropic::format_openai_to_anthropic(openai_response)
    {
        Ok(response) => response,
        Err(error) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                ErrorSource::Upstream,
                format!("Upstream response translation failed: {error}"),
                request_id,
            )
            .into_response();
        }
    };

    if let Ok(response_json) = serde_json::to_string_pretty(&anthropic_response) {
        log_payload(logging_path, "response", &response_json);
    }

    (StatusCode::OK, Json(anthropic_response)).into_response()
}

fn forward_stream(
    client: Client,
    base_url: String,
    api_key: String,
    openai_request: models::OpenAIRequest,
    logging_path: Arc<Option<String>>,
    request_id: String,
) -> Response {
    let stream = async_stream::stream! {
        let res = client
            .post(format!("{base_url}/chat/completions"))
            .bearer_auth(&api_key)
            .json(&openai_request)
            .send()
            .await;

        let Ok(res) = res else {
            tracing::error!(
                request_id = %request_id,
                "upstream stream connection failed"
            );
            let event = stream_error_event(
                ErrorSource::Upstream,
                "Failed to contact upstream provider",
                &request_id,
            );
            log_payload(&logging_path, "stream-error", &event);
            yield Ok::<_, axum::Error>(event.into_bytes());
            return;
        };

        if !res.status().is_success() {
            let status = res.status();
            let upstream_body = res.text().await.unwrap_or_default();
            let log_body: String = upstream_body.chars().take(512).collect();
            tracing::error!(
                request_id = %request_id,
                upstream_status = %status,
                "upstream stream request failed: {}", log_body
            );
            let message = if status == StatusCode::TOO_MANY_REQUESTS {
                "Upstream provider rate-limited"
            } else {
                "Upstream provider returned an error"
            };
            let event = stream_error_event(
                ErrorSource::Upstream,
                message,
                &request_id,
            );
            log_payload(&logging_path, "stream-error", &event);
            yield Ok::<_, axum::Error>(event.into_bytes());
            return;
        }

        let msg_id = format!("msg_{}", &request_id);
        let message_start = json!({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": openai_request.model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            }
        });
        let sse_event = format!("event: message_start\ndata: {message_start}\n\n");
        yield Ok::<_, axum::Error>(sse_event.into_bytes());

        let mut byte_stream = res.bytes_stream();
        let mut buffer = String::new();
        let mut content_block_open = false;
        let mut tool_block_open = false;
        let mut block_index: u32 = 0;
        let mut final_stop_reason: Option<String> = None;
        let mut saw_done_frame = false;

        while let Some(item) = byte_stream.next().await {
            let Ok(chunk) = item else {
                tracing::warn!(
                    request_id = %request_id,
                    "upstream stream chunk read failed"
                );
                if content_block_open || tool_block_open {
                    let content_block_stop = json!({
                        "type": "content_block_stop",
                        "index": block_index
                    });
                    let sse_event = format!("event: content_block_stop\ndata: {content_block_stop}\n\n");
                    yield Ok::<_, axum::Error>(sse_event.into_bytes());
                }
                let event = stream_error_event(
                    ErrorSource::Upstream,
                    "Streaming response interrupted",
                    &request_id,
                );
                log_payload(&logging_path, "stream-error", &event);
                yield Ok::<_, axum::Error>(event.into_bytes());
                return;
            };
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            while let Some(pos) = buffer.find("\n\n") {
                let frame = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                let frame = frame.trim();
                if frame.is_empty() {
                    continue;
                }

                if let Some(data) = frame.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        saw_done_frame = true;
                        break;
                    }
                    let parsed_chunk = match parse_openai_stream_chunk(data) {
                        Ok(parsed) => parsed,
                        Err(_) => {
                            tracing::warn!(
                                request_id = %request_id,
                                "upstream stream chunk could not be parsed"
                            );
                            if content_block_open || tool_block_open {
                                let content_block_stop = json!({
                                    "type": "content_block_stop",
                                    "index": block_index
                                });
                                let sse_event = format!("event: content_block_stop\ndata: {content_block_stop}\n\n");
                                yield Ok::<_, axum::Error>(sse_event.into_bytes());
                            }
                            let event = stream_error_event(
                                ErrorSource::Upstream,
                                "Upstream stream chunk could not be parsed",
                                &request_id,
                            );
                            log_payload(&logging_path, "stream-error", &event);
                            yield Ok::<_, axum::Error>(event.into_bytes());
                            return;
                        }
                    };

                    if let Some(chunk_data) = parsed_chunk {
                        // Handle tool call deltas
                        for tc in &chunk_data.tool_calls {
                            if tc.id.is_some() || tc.name.is_some() {
                                // New tool call — close any open block first
                                if content_block_open {
                                    let stop = json!({"type": "content_block_stop", "index": block_index});
                                    let sse_event = format!("event: content_block_stop\ndata: {stop}\n\n");
                                    yield Ok::<_, axum::Error>(sse_event.into_bytes());
                                    block_index += 1;
                                    content_block_open = false;
                                }
                                if tool_block_open {
                                    let stop = json!({"type": "content_block_stop", "index": block_index});
                                    let sse_event = format!("event: content_block_stop\ndata: {stop}\n\n");
                                    yield Ok::<_, axum::Error>(sse_event.into_bytes());
                                    block_index += 1;
                                }

                                let start = json!({
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tc.id.as_deref().unwrap_or(""),
                                        "name": tc.name.as_deref().unwrap_or(""),
                                        "input": {}
                                    }
                                });
                                let sse_event = format!("event: content_block_start\ndata: {start}\n\n");
                                yield Ok::<_, axum::Error>(sse_event.into_bytes());
                                tool_block_open = true;
                            }

                            if let Some(args) = &tc.arguments
                                && !args.is_empty()
                                && tool_block_open
                            {
                                let delta = json!({
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args
                                    }
                                });
                                let sse_event = format!("event: content_block_delta\ndata: {delta}\n\n");
                                yield Ok::<_, axum::Error>(sse_event.into_bytes());
                            }
                        }

                        // Handle text content
                        if let Some(content) = chunk_data.content {
                            // Close tool block if switching to text
                            if tool_block_open {
                                let stop = json!({"type": "content_block_stop", "index": block_index});
                                let sse_event = format!("event: content_block_stop\ndata: {stop}\n\n");
                                yield Ok::<_, axum::Error>(sse_event.into_bytes());
                                block_index += 1;
                                tool_block_open = false;
                            }

                            if !content_block_open {
                                let content_block_start = json!({
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {
                                        "type": "text",
                                        "text": ""
                                    }
                                });
                                let sse_event = format!("event: content_block_start\ndata: {content_block_start}\n\n");
                                yield Ok::<_, axum::Error>(sse_event.into_bytes());
                                content_block_open = true;
                            }

                            let anthropic_stream_event = json!({
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {
                                    "type": "text_delta",
                                    "text": content
                                }
                            });
                            let sse_event = format!("event: content_block_delta\ndata: {anthropic_stream_event}\n\n");
                            yield Ok::<_, axum::Error>(sse_event.into_bytes());
                        }

                        if let Some(reason) = chunk_data.finish_reason
                            && !reason.is_empty()
                        {
                            final_stop_reason = Some(map_stop_reason(&reason).to_string());
                        }
                    }
                }
            }
        }

        if content_block_open || tool_block_open {
            let content_block_stop = json!({
                "type": "content_block_stop",
                "index": block_index
            });
            let sse_event = format!("event: content_block_stop\ndata: {content_block_stop}\n\n");
            yield Ok::<_, axum::Error>(sse_event.into_bytes());
        }

        if !saw_done_frame {
            tracing::warn!(
                request_id = %request_id,
                "upstream stream ended without [DONE] frame"
            );
            let event = stream_error_event(
                ErrorSource::Upstream,
                "Streaming response interrupted",
                &request_id,
            );
            log_payload(&logging_path, "stream-error", &event);
            yield Ok::<_, axum::Error>(event.into_bytes());
            return;
        }

        if let Some(stop_reason) = final_stop_reason {
            let message_delta = json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": null
                },
                "usage": {
                    "output_tokens": 0
                }
            });
            let sse_event = format!("event: message_delta\ndata: {message_delta}\n\n");
            yield Ok::<_, axum::Error>(sse_event.into_bytes());
        }

        let message_stop = json!({
            "type": "message_stop"
        });
        let sse_event = format!("event: message_stop\ndata: {message_stop}\n\n");
        yield Ok::<_, axum::Error>(sse_event.into_bytes());
    };

    let body = Body::from_stream(stream);

    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .unwrap()
}

fn log_payload(logging_path: &Option<String>, suffix: &str, content: &str) {
    if let Some(dir) = logging_path.as_ref() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let file_path = format!("{dir}/{timestamp}-{suffix}.json");
        let _ = std::fs::write(file_path, content);
    }
}

async fn request_id_middleware(mut request: Request, next: Next) -> Response {
    let request_id = next_request_id();
    request
        .extensions_mut()
        .insert(RequestId(request_id.clone()));

    let mut response = next.run(request).await;
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}

async fn trace_middleware(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .map(|value| value.0.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let anthropic_version = request
        .headers()
        .get("anthropic-version")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("none");
    let headers = redact_sensitive_headers(request.headers());
    tracing::info!(%request_id, %method, %path, %anthropic_version, ?headers, "incoming request");

    let response = next.run(request).await;
    tracing::info!(
        %request_id,
        status = %response.status(),
        "request completed"
    );
    response
}

async fn timeout_middleware(request: Request, next: Next) -> Response {
    let request_id = request
        .extensions()
        .get::<RequestId>()
        .map(|value| value.0.clone())
        .unwrap_or_else(|| "unknown".to_string());
    match tokio::time::timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS), next.run(request)).await {
        Ok(response) => response,
        Err(_) => error_response(
            StatusCode::GATEWAY_TIMEOUT,
            ErrorSource::RoutingExhausted,
            "Request timed out",
            request_id,
        )
        .into_response(),
    }
}

async fn auth_middleware(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let requires_auth = matches!(request.uri().path(), "/v1/messages" | "/switch-model");

    if requires_auth {
        let request_id = request
            .extensions()
            .get::<RequestId>()
            .map(|value| value.0.clone())
            .unwrap_or_else(|| "unknown".to_string());

        if let Err(response) =
            validate_inbound_api_key(request.headers(), &state.inbound_api_key, &request_id)
        {
            return *response;
        }
    }

    next.run(request).await
}

fn next_request_id() -> String {
    let counter = REQUEST_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("req-{millis}-{counter}")
}

fn redact_sensitive_headers(headers: &HeaderMap) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            let lower_name = name.as_str().to_ascii_lowercase();
            let redacted = is_sensitive_header_name(&lower_name);
            let display_value = if redacted {
                "<redacted>".to_string()
            } else {
                value.to_str().unwrap_or("<binary>").to_string()
            };
            (name.to_string(), display_value)
        })
        .collect()
}

fn is_sensitive_header_name(lower_name: &str) -> bool {
    matches!(
        lower_name,
        "authorization" | "x-api-key" | "proxy-authorization" | "cookie" | "set-cookie"
    ) || lower_name.contains("secret")
        || lower_name.contains("password")
        || lower_name.contains("credential")
        || (lower_name.contains("token") && !lower_name.contains("ratelimit"))
        || (lower_name.ends_with("-key")
            && !lower_name.contains("idempotency")
            && !lower_name.contains("cache"))
}

#[cfg(test)]
mod tests {
    use super::{
        AppState, build_router, next_request_id, redact_sensitive_headers, resolve_stream_mode,
    };
    use crate::config::Config;
    use axum::{
        Json, Router,
        body::Body,
        http::{HeaderMap, HeaderValue, Request, Response as HttpResponse, StatusCode, header},
        response::Response,
        routing::post,
    };
    use reqwest::Client;
    use serde_json::Value;
    use serde_json::json;
    use std::{sync::Arc, time::Duration};
    use tokio::sync::{RwLock, oneshot};
    use tower::util::ServiceExt;

    /// Wrapper that aborts the mock server task on drop for clean test teardown.
    struct MockServer(tokio::task::JoinHandle<()>);

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.0.abort();
        }
    }

    #[test]
    fn request_id_is_generated_with_expected_prefix() {
        let request_id = next_request_id();
        assert!(request_id.starts_with("req-"));
    }

    #[test]
    fn sensitive_headers_are_redacted() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("secret"));
        headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
        headers.insert("x-auth-token", HeaderValue::from_static("abc123"));
        headers.insert("cookie", HeaderValue::from_static("session=secret"));
        headers.insert("x-normal-header", HeaderValue::from_static("visible"));

        let redacted = redact_sensitive_headers(&headers);

        assert!(
            redacted
                .iter()
                .any(|(name, value)| name == "x-api-key" && value == "<redacted>")
        );
        assert!(
            redacted
                .iter()
                .any(|(name, value)| name == "x-normal-header" && value == "visible")
        );
        assert!(
            redacted
                .iter()
                .any(|(name, value)| name == "authorization" && value == "<redacted>")
        );
        assert!(
            redacted
                .iter()
                .any(|(name, value)| name == "x-auth-token" && value == "<redacted>")
        );
        assert!(
            redacted
                .iter()
                .any(|(name, value)| name == "cookie" && value == "<redacted>")
        );
    }

    #[tokio::test]
    async fn router_adds_request_id_header_on_404() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/not-found")
                    .body(Body::empty())
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(response.headers().contains_key("x-request-id"));
    }

    #[test]
    fn stream_mode_defaults_to_non_stream_when_flag_omitted() {
        assert!(!resolve_stream_mode(None));
    }

    #[test]
    fn stream_mode_uses_explicit_true_flag() {
        assert!(resolve_stream_mode(Some(true)));
    }

    #[test]
    fn stream_mode_uses_explicit_false_flag() {
        assert!(!resolve_stream_mode(Some(false)));
    }

    enum ApiKeyMode {
        Missing,
        Plain(&'static str),
    }

    enum UpstreamMode {
        None,
        Success,
        ConnectivityFailure,
        HttpFailure,
    }

    struct ContractCase {
        name: &'static str,
        payload: &'static str,
        api_key: ApiKeyMode,
        upstream_mode: UpstreamMode,
        expected_status: StatusCode,
        expected_source: Option<&'static str>,
        expected_message: Option<&'static str>,
    }

    #[tokio::test]
    async fn epic1_contract_matrix_is_table_driven_and_correlated() {
        let cases = vec![
            ContractCase {
                name: "non_stream_happy_path",
                payload: r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hello"}]}"#,
                api_key: ApiKeyMode::Plain("dummy"),
                upstream_mode: UpstreamMode::Success,
                expected_status: StatusCode::OK,
                expected_source: None,
                expected_message: None,
            },
            ContractCase {
                name: "missing_key",
                payload: r#"{"model":"claude-sonnet-4","messages":[]}"#,
                api_key: ApiKeyMode::Missing,
                upstream_mode: UpstreamMode::None,
                expected_status: StatusCode::UNAUTHORIZED,
                expected_source: Some("local_auth"),
                expected_message: Some("Missing x-api-key header"),
            },
            ContractCase {
                name: "invalid_key",
                payload: r#"{"model":"claude-sonnet-4","messages":[]}"#,
                api_key: ApiKeyMode::Plain("wrong-key"),
                upstream_mode: UpstreamMode::None,
                expected_status: StatusCode::UNAUTHORIZED,
                expected_source: Some("local_auth"),
                expected_message: Some("Invalid x-api-key"),
            },
            ContractCase {
                name: "malformed_payload",
                payload: r#"{"model":"claude-sonnet-4","messages":[}"#,
                api_key: ApiKeyMode::Plain("dummy"),
                upstream_mode: UpstreamMode::None,
                expected_status: StatusCode::BAD_REQUEST,
                expected_source: Some("local_validation"),
                expected_message: Some("Malformed request payload"),
            },
            ContractCase {
                name: "upstream_connectivity_failure_mapping",
                payload: r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hello"}]}"#,
                api_key: ApiKeyMode::Plain("dummy"),
                upstream_mode: UpstreamMode::ConnectivityFailure,
                expected_status: StatusCode::BAD_GATEWAY,
                expected_source: Some("upstream"),
                expected_message: Some("Failed to contact upstream provider"),
            },
            ContractCase {
                name: "upstream_http_failure_mapping",
                payload: r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hello"}]}"#,
                api_key: ApiKeyMode::Plain("dummy"),
                upstream_mode: UpstreamMode::HttpFailure,
                expected_status: StatusCode::BAD_GATEWAY,
                expected_source: Some("upstream"),
                expected_message: Some("Upstream provider returned an error"),
            },
        ];

        for case in cases {
            let mut mock_handle: Option<MockServer> = None;
            let app = match case.upstream_mode {
                UpstreamMode::None => build_router(test_state()),
                UpstreamMode::Success => {
                    let (base_url, handle) = spawn_mock_openai_server().await;
                    mock_handle = Some(handle);
                    build_router(test_state_with_base_url(base_url))
                }
                UpstreamMode::ConnectivityFailure => {
                    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                        .await
                        .expect("listener bind should succeed");
                    let addr = listener
                        .local_addr()
                        .expect("listener should have a local address");
                    drop(listener);
                    build_router(test_state_with_base_url(format!("http://{}", addr)))
                }
                UpstreamMode::HttpFailure => {
                    let (base_url, handle) = spawn_mock_openai_error_server().await;
                    mock_handle = Some(handle);
                    build_router(test_state_with_base_url(base_url))
                }
            };

            let mut request_builder = Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json");

            request_builder = match case.api_key {
                ApiKeyMode::Missing => request_builder,
                ApiKeyMode::Plain(value) => request_builder.header("x-api-key", value),
            };

            let response = app
                .oneshot(
                    request_builder
                        .body(Body::from(case.payload))
                        .expect("request build should succeed"),
                )
                .await
                .unwrap_or_else(|_| panic!("request should complete for {}", case.name));

            assert_eq!(
                response.status(),
                case.expected_status,
                "status mismatch in {}",
                case.name
            );

            let request_id_header = response
                .headers()
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                .unwrap_or_else(|| panic!("x-request-id should be present in {}", case.name))
                .to_string();
            assert!(
                request_id_header.starts_with("req-"),
                "x-request-id must use req- prefix in {}",
                case.name
            );

            let body = parse_json_body(response).await;
            if let Some(source) = case.expected_source {
                assert_eq!(
                    body["type"], "error",
                    "error envelope type drift in {}",
                    case.name
                );
                let error = body
                    .get("error")
                    .and_then(Value::as_object)
                    .unwrap_or_else(|| panic!("error body missing in {}", case.name));
                assert_eq!(
                    error.get("source").and_then(Value::as_str),
                    Some(source),
                    "error source drift in {}",
                    case.name
                );
                assert_eq!(
                    error.get("request_id").and_then(Value::as_str),
                    Some(request_id_header.as_str()),
                    "request_id mismatch in {}",
                    case.name
                );
                if let Some(message) = case.expected_message {
                    assert_eq!(
                        error.get("message").and_then(Value::as_str),
                        Some(message),
                        "error message drift in {}",
                        case.name
                    );
                }
            } else {
                assert_eq!(
                    body.get("type").and_then(Value::as_str),
                    Some("message"),
                    "non-stream response type drift in {}",
                    case.name
                );
                assert_eq!(
                    body.get("role").and_then(Value::as_str),
                    Some("assistant"),
                    "role drift in {}",
                    case.name
                );
                assert!(
                    body.get("id")
                        .and_then(Value::as_str)
                        .is_some_and(|id| id.starts_with("msg_")),
                    "id must start with msg_ in {}",
                    case.name
                );
                assert!(
                    body.get("content").is_some_and(Value::is_array),
                    "content must be array in {}",
                    case.name
                );
                assert!(
                    body.get("model").is_some_and(Value::is_string),
                    "model must be string in {}",
                    case.name
                );
                assert!(
                    body.get("usage").is_some_and(Value::is_object),
                    "usage must be object in {}",
                    case.name
                );
                assert!(
                    body.get("error").is_none(),
                    "success response must not include error envelope in {}",
                    case.name
                );
            }

            drop(mock_handle);
        }
    }

    #[tokio::test]
    async fn startup_smoke_self_host_is_runnable_with_minimal_state() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let app = build_router(test_state());
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await;
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let response = Client::new()
            .get(format!("http://{}/not-found", addr))
            .send()
            .await
            .expect("smoke request should reach self-hosted server");

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert!(
            response.headers().contains_key("x-request-id"),
            "startup smoke should preserve request-id middleware wiring"
        );

        let _ = shutdown_tx.send(());
        let _ = server.await;
    }

    #[tokio::test]
    async fn messages_local_auth_error_shape_is_stable_and_correlated() {
        let app = build_router(test_state());
        let payload = r#"{"model":"claude-sonnet-4","messages":[]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = parse_json_body(response).await;
        assert_eq!(body.as_object().expect("body should be object").len(), 2);
        assert_eq!(
            body.get("type").and_then(Value::as_str),
            Some("error"),
            "envelope should have top-level type: error"
        );
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");

        assert_eq!(error.len(), 3);
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("local_auth")
        );
        assert_eq!(
            error.get("request_id").and_then(Value::as_str),
            Some(request_id_header.as_str())
        );
    }

    #[tokio::test]
    async fn messages_invalid_api_key_header_returns_local_auth_error() {
        let app = build_router(test_state());
        let payload = r#"{"model":"claude-sonnet-4","messages":[]}"#;
        let invalid_header = HeaderValue::from_bytes(&[0x80]).expect("header value should build");

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", invalid_header)
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(response.headers().contains_key("x-request-id"));
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        assert!(body_text.contains("\"source\":\"local_auth\""));
        assert!(body_text.contains("\"message\":\"Invalid x-api-key\""));
        assert!(body_text.contains("\"type\":\"error\""));
        assert!(!body_text.contains("dummy"));
        assert!(body_text.contains("\"request_id\":\"req-"));
    }

    #[tokio::test]
    async fn messages_wrong_api_key_returns_local_auth_error() {
        let app = build_router(test_state());
        let payload = r#"{"model":"claude-sonnet-4","messages":[]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "wrong-key")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        assert!(body_text.contains("\"source\":\"local_auth\""));
        assert!(body_text.contains("\"message\":\"Invalid x-api-key\""));
        assert!(!body_text.contains("wrong-key"));
    }

    #[tokio::test]
    async fn messages_local_validation_error_shape_is_stable_and_correlated() {
        let app = build_router(test_state());
        let payload = r#"{"model":"claude-sonnet-4","messages":[}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = parse_json_body(response).await;
        assert_eq!(body.as_object().expect("body should be object").len(), 2);
        assert_eq!(
            body.get("type").and_then(Value::as_str),
            Some("error"),
            "envelope should have top-level type: error"
        );
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");

        assert_eq!(error.len(), 3);
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("local_validation")
        );
        assert_eq!(
            error.get("request_id").and_then(Value::as_str),
            Some(request_id_header.as_str())
        );
    }

    #[tokio::test]
    async fn messages_missing_required_fields_returns_local_validation_error() {
        let app = build_router(test_state());
        let payload = r#"{"messages":[]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(response.headers().contains_key("x-request-id"));
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        assert!(body_text.contains("\"source\":\"local_validation\""));
        assert!(body_text.contains("\"message\":\"Malformed request payload\""));
        assert!(body_text.contains("\"request_id\":\"req-"));
    }

    #[tokio::test]
    async fn messages_unknown_block_returns_local_validation_error() {
        let app = build_router(test_state());
        let payload = r#"{
            "model":"claude-sonnet-4",
            "messages":[{"role":"user","content":[{"type":"unknown_block","foo":"bar"}]}]
        }"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(response.headers().contains_key("x-request-id"));
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        assert!(body_text.contains("\"source\":\"local_validation\""));
        assert!(body_text.contains("\"message\":\"Invalid Anthropic payload: Unsupported user content block type: unknown_block\""));
        assert!(body_text.contains("\"request_id\":\"req-"));
    }

    #[tokio::test]
    async fn messages_malformed_tool_definition_returns_local_validation_error() {
        let app = build_router(test_state());
        let payload = r#"{
            "model":"claude-sonnet-4",
            "messages":[{"role":"user","content":"Hello"}],
            "tools":[{"description":"missing name","input_schema":{"type":"object"}}]
        }"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = parse_json_body(response).await;
        assert_eq!(
            body.get("type").and_then(Value::as_str),
            Some("error"),
            "envelope must have top-level type: error"
        );
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("local_validation")
        );
        assert_eq!(
            error.get("message").and_then(Value::as_str),
            Some("Invalid Anthropic payload: Invalid tool definition: missing name")
        );
        assert!(
            error
                .get("request_id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with("req-")),
            "request_id must be present and use req- prefix"
        );
    }

    #[tokio::test]
    async fn messages_stream_malformed_tool_definition_returns_local_validation_error() {
        let app = build_router(test_state());
        let payload = r#"{
            "model":"claude-sonnet-4",
            "stream":true,
            "messages":[{"role":"user","content":"Hello"}],
            "tools":[{"description":"missing name","input_schema":{"type":"object"}}]
        }"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = parse_json_body(response).await;
        assert_eq!(
            body.get("type").and_then(Value::as_str),
            Some("error"),
            "envelope must have top-level type: error"
        );
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("local_validation")
        );
        assert_eq!(
            error.get("message").and_then(Value::as_str),
            Some("Invalid Anthropic payload: Invalid tool definition: missing name")
        );
        assert!(
            error
                .get("request_id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with("req-")),
            "request_id must be present and use req- prefix"
        );
    }

    #[tokio::test]
    async fn messages_unknown_tool_result_id_returns_local_validation_error() {
        let app = build_router(test_state());
        let payload = r#"{
            "model":"claude-sonnet-4",
            "messages":[
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"missing","content":{"ok":true}}]}
            ]
        }"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = parse_json_body(response).await;
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("local_validation")
        );
        assert_eq!(
            error.get("message").and_then(Value::as_str),
            Some(
                "Invalid Anthropic payload: Invalid tool_result block: unknown tool_use_id missing"
            )
        );
    }

    #[tokio::test]
    async fn messages_valid_minimal_payload_translates_and_returns_ok() {
        let (base_url, _server) = spawn_mock_openai_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload =
            r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().contains_key("x-request-id"));
        let body = parse_json_body(response).await;
        assert_eq!(body["type"], "message");
        assert_eq!(body["role"], "assistant");
        assert_eq!(body["stop_reason"], "end_turn");
        assert!(
            body["id"]
                .as_str()
                .expect("id should be a string")
                .starts_with("msg_"),
            "response id should start with msg_ prefix"
        );
        assert!(body["content"].is_array(), "content should be an array");
        assert!(body["model"].is_string(), "model should be present");
        let usage = body
            .get("usage")
            .expect("usage should be present in response");
        assert_eq!(usage["input_tokens"], 10);
        assert_eq!(usage["output_tokens"], 20);
    }

    #[tokio::test]
    async fn messages_non_stream_tools_are_mapped_end_to_end() {
        let (base_url, _server) = spawn_mock_openai_tool_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload = r#"{
            "model":"claude-sonnet-4",
            "stream":false,
            "messages":[{"role":"user","content":"Use a tool"}],
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
        }"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        let body = parse_json_body(response).await;
        assert_eq!(body.get("type").and_then(Value::as_str), Some("message"));
        assert_eq!(
            body.get("stop_reason").and_then(Value::as_str),
            Some("tool_use")
        );
        let content_blocks = body
            .get("content")
            .and_then(Value::as_array)
            .expect("content should be array");
        let tool_use = content_blocks
            .iter()
            .find(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
            .expect("tool_use block should be present");
        assert_eq!(
            tool_use.get("id").and_then(Value::as_str),
            Some("call_weather")
        );
        assert_eq!(
            tool_use.get("name").and_then(Value::as_str),
            Some("weather_lookup")
        );
        assert_eq!(
            tool_use
                .get("input")
                .and_then(Value::as_object)
                .and_then(|obj| obj.get("city"))
                .and_then(Value::as_str),
            Some("Paris")
        );
    }

    #[tokio::test]
    async fn messages_stream_true_routes_to_sse_pipeline() {
        let (base_url, _server) = spawn_mock_openai_stream_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload = r#"{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert!(response.headers().contains_key("x-request-id"));
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        let events = parse_sse_events(&body_text);
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            event_names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(
            events[2]
                .1
                .get("delta")
                .and_then(|v| v.get("text"))
                .and_then(Value::as_str),
            Some("STREAM_OK")
        );
        let message_start_data = &events[0].1;
        let message_obj = message_start_data
            .get("message")
            .expect("message_start must contain message object");
        assert_eq!(
            message_obj.get("type").and_then(Value::as_str),
            Some("message"),
            "message.type must be 'message'"
        );
        assert_eq!(
            message_obj.get("role").and_then(Value::as_str),
            Some("assistant"),
            "message.role must be 'assistant'"
        );
        assert!(
            message_obj
                .get("id")
                .and_then(Value::as_str)
                .is_some_and(|id| id.starts_with("msg_")),
            "message.id must start with msg_ prefix"
        );
        assert!(
            message_obj.get("model").is_some_and(Value::is_string),
            "message.model must be present"
        );
        assert_eq!(
            events[4]
                .1
                .get("delta")
                .and_then(|v| v.get("stop_reason"))
                .and_then(Value::as_str),
            Some("end_turn")
        );
    }

    #[tokio::test]
    async fn messages_stream_malformed_chunk_returns_error_event_with_request_correlation() {
        let (base_url, _server) = spawn_mock_openai_malformed_stream_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload = r#"{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        let events = parse_sse_events(&body_text);
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

        assert_eq!(
            event_names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "error"
            ]
        );
        assert_stream_error_is_canonical_and_terminal(&events, &request_id_header, "upstream");
        assert!(
            !event_names.contains(&"message_stop"),
            "terminal success event must not be emitted on stream parse failure"
        );
    }

    #[tokio::test]
    async fn messages_stream_truncated_upstream_emits_error_event_with_request_correlation() {
        let (base_url, _server) = spawn_mock_openai_truncated_stream_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload = r#"{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        let events = parse_sse_events(&body_text);
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

        assert_eq!(
            event_names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "error"
            ]
        );
        assert_stream_error_is_canonical_and_terminal(&events, &request_id_header, "upstream");
        let error_payload = events
            .last()
            .expect("error event must exist")
            .1
            .get("error")
            .cloned()
            .expect("error body must exist");
        assert_eq!(
            error_payload.get("message").and_then(Value::as_str),
            Some("Streaming response interrupted")
        );
        assert!(
            !event_names.contains(&"message_stop"),
            "terminal success event must not be emitted on truncated upstream stream"
        );
    }

    #[tokio::test]
    async fn messages_stream_unknown_events_and_tool_deltas_are_handled_gracefully() {
        let (base_url, _server) = spawn_mock_openai_stream_with_unknown_event_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload = r#"{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        let events = parse_sse_events(&body_text);
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

        // tool_calls delta produces tool_use block, then text content produces text block
        assert_eq!(
            event_names,
            vec![
                "message_start",
                "content_block_start", // tool_use (index 0)
                "content_block_delta", // input_json_delta
                "content_block_stop",  // close tool block
                "content_block_start", // text (index 1)
                "content_block_delta", // text_delta STREAM_OK
                "content_block_stop",  // close text block
                "message_delta",
                "message_stop"
            ]
        );
        assert!(
            !event_names.contains(&"error"),
            "unknown stream event payloads should not crash the parser"
        );

        // Verify tool_use content block
        let first_block_start = &events[1].1;
        assert_eq!(
            first_block_start
                .get("content_block")
                .and_then(|cb| cb.get("type"))
                .and_then(Value::as_str),
            Some("tool_use"),
            "first content block should be tool_use"
        );
        assert_eq!(
            first_block_start
                .get("content_block")
                .and_then(|cb| cb.get("id"))
                .and_then(Value::as_str),
            Some("call_1"),
            "tool_use block should have correct id"
        );
        assert_eq!(
            first_block_start
                .get("content_block")
                .and_then(|cb| cb.get("name"))
                .and_then(Value::as_str),
            Some("lookup"),
            "tool_use block should have correct name"
        );

        // Verify input_json_delta
        let tool_delta = &events[2].1;
        assert_eq!(
            tool_delta
                .get("delta")
                .and_then(|d| d.get("type"))
                .and_then(Value::as_str),
            Some("input_json_delta"),
            "tool delta should be input_json_delta type"
        );

        // Verify text block has index 1 (incremented)
        let text_block_start = &events[4].1;
        assert_eq!(
            text_block_start.get("index").and_then(Value::as_u64),
            Some(1),
            "text block should have index 1 after tool block"
        );

        // Verify message_delta stop_reason
        let message_delta = events
            .iter()
            .find(|(name, _)| name == "message_delta")
            .map(|(_, data)| data)
            .expect("message_delta event should exist");
        assert_eq!(
            message_delta
                .get("delta")
                .and_then(Value::as_object)
                .and_then(|delta| delta.get("stop_reason"))
                .and_then(Value::as_str),
            Some("tool_use")
        );
    }

    #[tokio::test]
    async fn messages_stream_false_routes_to_non_stream_pipeline() {
        let (base_url, _server) = spawn_mock_openai_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload = r#"{"model":"claude-sonnet-4","stream":false,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_ne!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert!(
            response.headers().contains_key("x-request-id"),
            "x-request-id must be present for stream=false path"
        );
        let body = parse_json_body(response).await;
        assert_eq!(body.get("type").and_then(Value::as_str), Some("message"));
        assert_eq!(body.get("role").and_then(Value::as_str), Some("assistant"));
    }

    #[tokio::test]
    async fn messages_invalid_stream_type_returns_local_validation_error() {
        let app = build_router(test_state());
        let payload = r#"{"model":"claude-sonnet-4","stream":"yes","messages":[]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = parse_json_body(response).await;
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("local_validation")
        );
        assert_eq!(
            error.get("message").and_then(Value::as_str),
            Some("Malformed request payload")
        );
        assert_eq!(
            error.get("request_id").and_then(Value::as_str),
            Some(request_id_header.as_str())
        );
    }

    #[tokio::test]
    async fn messages_stream_integer_returns_local_validation_error() {
        let app = build_router(test_state());
        let payload = r#"{"model":"claude-sonnet-4","stream":0,"messages":[]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = parse_json_body(response).await;
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("local_validation")
        );
    }

    #[tokio::test]
    async fn messages_stream_string_false_returns_local_validation_error() {
        let app = build_router(test_state());
        let payload = r#"{"model":"claude-sonnet-4","stream":"false","messages":[]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = parse_json_body(response).await;
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("local_validation")
        );
    }

    #[tokio::test]
    async fn messages_unparseable_upstream_payload_returns_upstream_error() {
        let (base_url, _server) = spawn_mock_openai_invalid_json_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload =
            r#"{"model":"claude-sonnet-4","messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = parse_json_body(response).await;
        let error = body
            .get("error")
            .and_then(Value::as_object)
            .expect("error envelope should exist");

        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some("upstream")
        );
        assert_eq!(
            error.get("message").and_then(Value::as_str),
            Some("Upstream response could not be parsed")
        );
        assert_eq!(
            error.get("request_id").and_then(Value::as_str),
            Some(request_id_header.as_str())
        );
    }

    #[tokio::test]
    async fn switch_model_missing_api_key_returns_local_auth_error() {
        let app = build_router(test_state());

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/switch-model")
                    .body(Body::empty())
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(response.headers().contains_key("x-request-id"));
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        assert!(body_text.contains("\"source\":\"local_auth\""));
    }

    #[tokio::test]
    async fn messages_stream_upstream_unreachable_emits_error_event() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        drop(listener);
        let app = build_router(test_state_with_base_url(format!("http://{}", addr)));
        let payload = r#"{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        let events = parse_sse_events(&body_text);
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

        assert_eq!(event_names, vec!["error"]);
        assert_stream_error_is_canonical_and_terminal(&events, &request_id_header, "upstream");
        let error_payload = events[0]
            .1
            .get("error")
            .cloned()
            .expect("error body must exist");
        assert_eq!(
            error_payload.get("message").and_then(Value::as_str),
            Some("Failed to contact upstream provider")
        );
    }

    #[tokio::test]
    async fn messages_stream_upstream_http_failure_emits_error_event() {
        let (base_url, _server) = spawn_mock_openai_error_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload = r#"{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        let events = parse_sse_events(&body_text);
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

        assert_eq!(event_names, vec!["error"]);
        assert_stream_error_is_canonical_and_terminal(&events, &request_id_header, "upstream");
        let error_payload = events[0]
            .1
            .get("error")
            .cloned()
            .expect("error body must exist");
        assert_eq!(
            error_payload.get("message").and_then(Value::as_str),
            Some("Upstream provider returned an error")
        );
    }

    #[tokio::test]
    async fn messages_stream_upstream_rate_limited_emits_error_event() {
        let (base_url, _server) = spawn_mock_openai_rate_limited_server().await;
        let app = build_router(test_state_with_base_url(base_url));
        let payload = r#"{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        let events = parse_sse_events(&body_text);
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

        assert_eq!(event_names, vec!["error"]);
        assert_stream_error_is_canonical_and_terminal(&events, &request_id_header, "upstream");
        let error_payload = events[0]
            .1
            .get("error")
            .cloned()
            .expect("error body must exist");
        assert_eq!(
            error_payload.get("message").and_then(Value::as_str),
            Some("Upstream provider rate-limited")
        );
    }

    async fn spawn_mock_openai_server() -> (String, MockServer) {
        async fn chat_completions_handler() -> (StatusCode, Json<serde_json::Value>) {
            let response = json!({
                "id": "chatcmpl-test",
                "model": "openai-test-model",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello from upstream"
                    }
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            });
            (StatusCode::OK, Json(response))
        }

        let app = Router::new().route("/chat/completions", post(chat_completions_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn spawn_mock_openai_tool_server() -> (String, MockServer) {
        async fn chat_completions_tool_handler(
            Json(body): Json<Value>,
        ) -> (StatusCode, Json<Value>) {
            let tools = body
                .get("tools")
                .and_then(Value::as_array)
                .expect("tools should exist in upstream request");
            let first_tool = tools.first().expect("at least one tool should exist");
            assert_eq!(
                first_tool.get("type").and_then(Value::as_str),
                Some("function")
            );
            assert_eq!(
                first_tool
                    .get("function")
                    .and_then(Value::as_object)
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str),
                Some("weather_lookup")
            );

            let response = json!({
                "id": "chatcmpl-tools",
                "model": "openai-test-model",
                "choices": [{
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "weather_lookup",
                                "arguments": "{\"city\":\"Paris\"}"
                            }
                        }]
                    }
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            });
            (StatusCode::OK, Json(response))
        }

        let app = Router::new().route("/chat/completions", post(chat_completions_tool_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn spawn_mock_openai_error_server() -> (String, MockServer) {
        async fn chat_completions_error_handler() -> (StatusCode, &'static str) {
            (StatusCode::INTERNAL_SERVER_ERROR, "boom")
        }

        let app = Router::new().route("/chat/completions", post(chat_completions_error_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn spawn_mock_openai_rate_limited_server() -> (String, MockServer) {
        async fn chat_completions_rate_limited_handler() -> (StatusCode, &'static str) {
            (StatusCode::TOO_MANY_REQUESTS, "rate-limited")
        }

        let app = Router::new().route(
            "/chat/completions",
            post(chat_completions_rate_limited_handler),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn spawn_mock_openai_invalid_json_server() -> (String, MockServer) {
        async fn chat_completions_invalid_json_handler() -> (StatusCode, &'static str) {
            (StatusCode::OK, "this is not json")
        }

        let app = Router::new().route(
            "/chat/completions",
            post(chat_completions_invalid_json_handler),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn spawn_mock_openai_stream_server() -> (String, MockServer) {
        async fn chat_completions_stream_handler(Json(body): Json<Value>) -> HttpResponse<Body> {
            assert_eq!(
                body.get("stream").and_then(Value::as_bool),
                Some(true),
                "upstream request must contain stream: true"
            );
            let payload = concat!(
                "data: {\"id\":\"chatcmpl-stream\",\"model\":\"openai-test-model\",",
                "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",",
                "\"content\":\"STREAM_OK\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n"
            );
            HttpResponse::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload))
                .expect("response build should succeed")
        }

        let app = Router::new().route("/chat/completions", post(chat_completions_stream_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn spawn_mock_openai_malformed_stream_server() -> (String, MockServer) {
        async fn chat_completions_stream_handler(Json(body): Json<Value>) -> HttpResponse<Body> {
            assert_eq!(
                body.get("stream").and_then(Value::as_bool),
                Some(true),
                "upstream request must contain stream: true"
            );
            let payload = concat!(
                "data: {\"id\":\"chatcmpl-stream\",\"model\":\"openai-test-model\",",
                "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",",
                "\"content\":\"STREAM_OK\"},\"finish_reason\":null}]}\n\n",
                "data: {bad-json}\n\n"
            );
            HttpResponse::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload))
                .expect("response build should succeed")
        }

        let app = Router::new().route("/chat/completions", post(chat_completions_stream_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn spawn_mock_openai_stream_with_unknown_event_server() -> (String, MockServer) {
        async fn chat_completions_stream_handler(Json(body): Json<Value>) -> HttpResponse<Body> {
            assert_eq!(
                body.get("stream").and_then(Value::as_bool),
                Some(true),
                "upstream request must contain stream: true"
            );
            let payload = concat!(
                "event: ping\ndata: {\"type\":\"ping\"}\n\n",
                "data: {\"id\":\"chatcmpl-stream\",\"model\":\"openai-test-model\",",
                "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"id\":\"call_1\",",
                "\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]},",
                "\"finish_reason\":null}]}\n\n",
                "data: {\"id\":\"chatcmpl-stream\",\"model\":\"openai-test-model\",",
                "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"STREAM_OK\"},",
                "\"finish_reason\":\"tool_calls\"}]}\n\n",
                "data: [DONE]\n\n"
            );
            HttpResponse::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload))
                .expect("response build should succeed")
        }

        let app = Router::new().route("/chat/completions", post(chat_completions_stream_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn spawn_mock_openai_truncated_stream_server() -> (String, MockServer) {
        async fn chat_completions_stream_handler(Json(body): Json<Value>) -> HttpResponse<Body> {
            assert_eq!(
                body.get("stream").and_then(Value::as_bool),
                Some(true),
                "upstream request must contain stream: true"
            );
            let payload = concat!(
                "data: {\"id\":\"chatcmpl-stream\",\"model\":\"openai-test-model\",",
                "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",",
                "\"content\":\"STREAM_OK\"},\"finish_reason\":null}]}\n\n"
            );
            HttpResponse::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload))
                .expect("response build should succeed")
        }

        let app = Router::new().route("/chat/completions", post(chat_completions_stream_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    async fn parse_json_body(response: Response) -> Value {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        serde_json::from_slice::<Value>(&body).expect("body should be valid json")
    }

    fn parse_sse_events(body_text: &str) -> Vec<(String, Value)> {
        body_text
            .split("\n\n")
            .filter_map(|frame| {
                let trimmed = frame.trim();
                if trimmed.is_empty() {
                    return None;
                }
                if trimmed == "data: [DONE]" {
                    return None;
                }
                let mut event_name: Option<String> = None;
                let mut data_payload: Option<Value> = None;
                let mut has_data_line = false;
                for line in trimmed.lines() {
                    if let Some(name) = line.strip_prefix("event: ") {
                        event_name = Some(name.to_string());
                    } else if let Some(data) = line.strip_prefix("data: ") {
                        has_data_line = true;
                        data_payload = serde_json::from_str::<Value>(data).ok();
                    }
                }
                if event_name.is_some() || has_data_line {
                    let name = event_name.unwrap_or_else(|| {
                        panic!("SSE frame has data but no event name: {trimmed}")
                    });
                    let payload = data_payload.unwrap_or_else(|| {
                        panic!("SSE frame '{name}' has unparseable data: {trimmed}")
                    });
                    Some((name, payload))
                } else {
                    None
                }
            })
            .collect()
    }

    fn assert_stream_error_is_canonical_and_terminal(
        events: &[(String, Value)],
        request_id: &str,
        expected_source: &str,
    ) {
        let (event_name, payload) = events.last().expect("error event must exist");
        assert_eq!(event_name, "error", "last stream event must be error");
        assert_eq!(
            payload.get("type").and_then(Value::as_str),
            Some("error"),
            "top-level stream payload must be type=error"
        );
        let error = payload.get("error").expect("error body must exist");
        assert_eq!(
            error.get("source").and_then(Value::as_str),
            Some(expected_source)
        );
        assert_eq!(
            error.get("request_id").and_then(Value::as_str),
            Some(request_id)
        );
        assert!(
            error
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|m| !m.is_empty()),
            "error.message must be a non-empty string"
        );
    }

    async fn spawn_mock_openai_stalled_stream_server() -> (String, MockServer) {
        async fn chat_completions_stream_handler(Json(body): Json<Value>) -> HttpResponse<Body> {
            assert_eq!(
                body.get("stream").and_then(Value::as_bool),
                Some(true),
                "upstream request must contain stream: true"
            );
            let stream = async_stream::stream! {
                let chunk = concat!(
                    "data: {\"id\":\"chatcmpl-stream\",\"model\":\"openai-test-model\",",
                    "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},",
                    "\"finish_reason\":null}]}\n\n"
                );
                yield Ok::<_, std::io::Error>(chunk.to_string());
                // Stall: sleep longer than the client's read_timeout
                tokio::time::sleep(Duration::from_secs(30)).await;
                yield Ok::<_, std::io::Error>("data: [DONE]\n\n".to_string());
            };
            HttpResponse::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(stream))
                .expect("response build should succeed")
        }

        let app = Router::new().route("/chat/completions", post(chat_completions_stream_handler));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("listener bind should succeed");
        let addr = listener
            .local_addr()
            .expect("listener should have a local address");
        let handle = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        (format!("http://{}", addr), MockServer(handle))
    }

    #[tokio::test]
    async fn messages_stream_stalled_upstream_emits_error_event() {
        let (base_url, _server) = spawn_mock_openai_stalled_stream_server().await;
        let app = build_router(test_state_with_read_timeout(
            base_url,
            Duration::from_secs(1),
        ));
        let payload = r#"{"model":"claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"Hello"}]}"#;

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .header("x-api-key", "dummy")
                    .body(Body::from(payload))
                    .expect("request build should succeed"),
            )
            .await
            .expect("request should complete");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        let request_id_header = response
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id should be present")
            .to_string();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        let events = parse_sse_events(&body_text);
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

        assert!(
            event_names.contains(&"error"),
            "stalled stream must produce an error event"
        );
        assert!(
            !event_names.contains(&"message_stop"),
            "terminal success event must not be emitted on stalled stream"
        );
        assert_stream_error_is_canonical_and_terminal(&events, &request_id_header, "upstream");
        let error_payload = events
            .last()
            .expect("error event must exist")
            .1
            .get("error")
            .cloned()
            .expect("error body must exist");
        assert_eq!(
            error_payload.get("message").and_then(Value::as_str),
            Some("Streaming response interrupted")
        );
    }

    fn test_state() -> AppState {
        test_state_with_base_url("https://openrouter.ai/api/v1".to_string())
    }

    fn test_state_with_base_url(base_url: String) -> AppState {
        let config = Config {
            port: 3332,
            base_url,
            api_key: "upstream-provider-key".to_string(),
            inbound_api_key: "dummy".to_string(),
            model_haiku: "haiku".to_string(),
            model_sonnet: "sonnet".to_string(),
            model_opus: "opus".to_string(),
        };
        AppState {
            config: Arc::new(RwLock::new(config)),
            logging_path: Arc::new(None),
            client: Client::new(),
            inbound_api_key: Arc::new("dummy".to_string()),
        }
    }

    fn test_state_with_read_timeout(base_url: String, timeout: Duration) -> AppState {
        let config = Config {
            port: 3332,
            base_url,
            api_key: "upstream-provider-key".to_string(),
            inbound_api_key: "dummy".to_string(),
            model_haiku: "haiku".to_string(),
            model_sonnet: "sonnet".to_string(),
            model_opus: "opus".to_string(),
        };
        let client = Client::builder()
            .read_timeout(timeout)
            .build()
            .expect("client should build");
        AppState {
            config: Arc::new(RwLock::new(config)),
            logging_path: Arc::new(None),
            client,
            inbound_api_key: Arc::new("dummy".to_string()),
        }
    }
}
