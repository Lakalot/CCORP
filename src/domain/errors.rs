use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorSource {
    LocalAuth,
    LocalValidation,
    Upstream,
    RoutingExhausted,
    #[allow(dead_code)] // Reserved for future stories per architecture error taxonomy
    Internal,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub source: ErrorSource,
    pub message: String,
    pub request_id: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}
