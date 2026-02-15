//! Interface-layer components reserved for HTTP/CLI adapters.

use axum::{Json, http::StatusCode, response::IntoResponse};

use crate::domain::errors::{ErrorBody, ErrorEnvelope, ErrorSource};

pub fn error_response(
    status: StatusCode,
    source: ErrorSource,
    message: impl Into<String>,
    request_id: impl Into<String>,
) -> impl IntoResponse {
    let body = ErrorEnvelope {
        error: ErrorBody {
            source,
            message: message.into(),
            request_id: request_id.into(),
        },
    };

    (status, Json(body))
}
