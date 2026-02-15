mod anthropic_to_openai;
mod application;
mod config;
mod domain;
mod infrastructure;
mod interfaces;
mod models;
mod openai_to_anthropic;
mod openrouter;
mod switch_model;

use axum::{
    Extension, Router,
    body::Body,
    extract::{Json, Request, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use config::Config;
use domain::errors::ErrorSource;
use futures_util::stream::StreamExt;
use interfaces::error_response;
use models::{AnthropicRequest, OpenAIStreamResponse};
use reqwest::Client;
use serde_json::json;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<RwLock<Config>>,
    pub logging_path: Arc<Option<String>>,
}

#[derive(Clone, Debug)]
struct RequestId(String);

static REQUEST_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

const REQUEST_TIMEOUT_SECS: u64 = 60;

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

    let state = AppState {
        config: Arc::new(RwLock::new(settings)),
        logging_path: Arc::new(logging_path),
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
        .layer(middleware::from_fn(auth_placeholder_middleware))
        .layer(middleware::from_fn(trace_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        .with_state(state)
}

async fn messages_handler(
    State(state): State<AppState>,
    Extension(request_id): Extension<RequestId>,
    headers: HeaderMap,
    Json(payload): Json<AnthropicRequest>,
) -> Response {
    let settings_guard = state.config.read().await;
    let openai_request = anthropic_to_openai::format_anthropic_to_openai(payload, &settings_guard);

    if let Some(path) = state.logging_path.as_ref() {
        let timestamp = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let request_path = format!("{path}/{timestamp}-request.json");
        if let Ok(request_json) = serde_json::to_string_pretty(&openai_request) {
            let _ = std::fs::write(request_path, request_json);
        }
    }
    let client = Client::new();
    let api_key = match headers.get("x-api-key") {
        None => {
            return error_response(
                StatusCode::UNAUTHORIZED,
                ErrorSource::LocalAuth,
                "Missing x-api-key header",
                request_id.0,
            )
            .into_response();
        }
        Some(value) => match value.to_str() {
            Ok(text) => text.to_string(),
            Err(_) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    ErrorSource::LocalValidation,
                    "Invalid x-api-key header format",
                    request_id.0,
                )
                .into_response();
            }
        },
    };

    if openai_request.stream.unwrap_or(false) {
        let base_url = settings_guard.base_url.clone();
        let request_id_value = request_id.0.clone();
        drop(settings_guard);
        let stream = async_stream::stream! {
            let res = client
                .post(format!(
                    "{base_url}/chat/completions",
                ))
                .bearer_auth(api_key)
                .json(&openai_request)
                .send()
                .await;

            let Ok(res) = res else {
                let event = stream_error_event(
                    ErrorSource::Upstream,
                    "Failed to contact upstream provider",
                    &request_id_value,
                );
                yield Ok::<_, axum::Error>(event.into_bytes());
                return;
            };

            if !res.status().is_success() {
                let upstream_body = res.text().await.unwrap_or_default();
                tracing::error!("OpenRouter request failed: {}", upstream_body);
                let event = stream_error_event(
                    ErrorSource::Upstream,
                    "Upstream provider returned an error",
                    &request_id_value,
                );
                yield Ok::<_, axum::Error>(event.into_bytes());
                return;
            }

            let mut stream = res.bytes_stream();

            let mut full_response = String::new();
            while let Some(item) = stream.next().await {
                let Ok(chunk) = item else {
                    let event = stream_error_event(
                        ErrorSource::Upstream,
                        "Streaming response interrupted",
                        &request_id_value,
                    );
                    yield Ok::<_, axum::Error>(event.into_bytes());
                    return;
                };
                full_response.push_str(&String::from_utf8_lossy(&chunk));
                let chunk_str = String::from_utf8_lossy(&chunk);
                for line in chunk_str.split("

") {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            break;
                        }
                        if let Ok(stream_res) = serde_json::from_str::<OpenAIStreamResponse>(data) {
                            let choice = &stream_res.choices[0];
                            if let Some(content) = &choice.delta.content {
                                let anthropic_stream_event = json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": content
                                    }
                                });
                                let sse_event = format!("event: content_block_delta
data: {anthropic_stream_event}

");
                                yield Ok::<_, axum::Error>(sse_event.into_bytes());
                            }
                        }
                    }
                }
            }

            let message_stop = json!({
                "type": "message_stop"
            });
            let sse_event = format!("event: message_stop
data: {message_stop}

");
            yield Ok::<_, axum::Error>(sse_event.into_bytes());

            if let Some(path) = state.logging_path.as_ref() {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                let response_path = format!("{path}/{timestamp}-response.json");
                let _ = std::fs::write(response_path, full_response);
            }
        };

        let body = Body::from_stream(stream);

        Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .body(body)
            .unwrap()
    } else {
        let res = client
            .post(format!("{}/chat/completions", settings_guard.base_url))
            .bearer_auth(api_key)
            .json(&openai_request)
            .send()
            .await;

        let Ok(res) = res else {
            return error_response(
                StatusCode::BAD_GATEWAY,
                ErrorSource::Upstream,
                "Failed to contact upstream provider",
                request_id.0,
            )
            .into_response();
        };

        if !res.status().is_success() {
            return error_response(
                StatusCode::BAD_GATEWAY,
                ErrorSource::Upstream,
                "Upstream provider returned an error",
                request_id.0,
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
                    request_id.0,
                )
                .into_response();
            }
        };
        let anthropic_response =
            openai_to_anthropic::format_openai_to_anthropic(openai_response.clone());

        if let Some(path) = state.logging_path.as_ref() {
            let timestamp = std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();
            let response_path = format!("{path}/{timestamp}-response.json");
            if let Ok(response_json) = serde_json::to_string_pretty(&anthropic_response) {
                let _ = std::fs::write(response_path, response_json);
            }
        }

        (StatusCode::OK, Json(anthropic_response)).into_response()
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
    let headers = redact_sensitive_headers(request.headers());
    tracing::info!(%request_id, %method, %path, ?headers, "incoming request");

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

async fn auth_placeholder_middleware(request: Request, next: Next) -> Response {
    // Reserved insertion point for Story 1.4 x-api-key enforcement policy.
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
            let redacted = matches!(
                lower_name.as_str(),
                "authorization" | "x-api-key" | "proxy-authorization" | "cookie" | "set-cookie"
            );
            let display_value = if redacted {
                "<redacted>".to_string()
            } else {
                value.to_str().unwrap_or("<binary>").to_string()
            };
            (name.to_string(), display_value)
        })
        .collect()
}

fn stream_error_event(source: ErrorSource, message: &str, request_id: &str) -> String {
    let payload = json!({
        "type": "error",
        "error": {
            "source": source,
            "message": message,
            "request_id": request_id
        }
    });
    format!("event: error\ndata: {payload}\n\n")
}

#[cfg(test)]
mod tests {
    use super::{AppState, build_router, next_request_id, redact_sensitive_headers};
    use crate::config::Config;
    use axum::{
        body::Body,
        http::{HeaderMap, HeaderValue, Request, StatusCode},
    };
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tower::util::ServiceExt;

    #[test]
    fn request_id_is_generated_with_expected_prefix() {
        let request_id = next_request_id();
        assert!(request_id.starts_with("req-"));
    }

    #[test]
    fn sensitive_headers_are_redacted() {
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", HeaderValue::from_static("secret"));
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

    #[tokio::test]
    async fn messages_missing_api_key_returns_local_auth_error() {
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
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should be readable");
        let body_text = String::from_utf8(body.to_vec()).expect("body should be utf8");
        assert!(body_text.contains("\"source\":\"local_auth\""));
        assert!(body_text.contains("\"request_id\":\"req-"));
    }

    fn test_state() -> AppState {
        let config = Config {
            port: 3332,
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: "dummy".to_string(),
            model_haiku: "haiku".to_string(),
            model_sonnet: "sonnet".to_string(),
            model_opus: "opus".to_string(),
        };
        AppState {
            config: Arc::new(RwLock::new(config)),
            logging_path: Arc::new(None),
        }
    }
}
