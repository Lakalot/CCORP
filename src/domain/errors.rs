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
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: ErrorBody,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_envelope_serializes_with_type_field() {
        let envelope = ErrorEnvelope {
            error_type: "error".to_string(),
            error: ErrorBody {
                source: ErrorSource::LocalAuth,
                message: "test".to_string(),
                request_id: "req-1".to_string(),
            },
        };
        let json = serde_json::to_value(&envelope).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["source"], "local_auth");
        assert_eq!(json["error"]["message"], "test");
        assert_eq!(json["error"]["request_id"], "req-1");
    }

    #[test]
    fn all_error_sources_serialize_to_snake_case() {
        let sources = vec![
            (ErrorSource::LocalAuth, "local_auth"),
            (ErrorSource::LocalValidation, "local_validation"),
            (ErrorSource::Upstream, "upstream"),
            (ErrorSource::RoutingExhausted, "routing_exhausted"),
            (ErrorSource::Internal, "internal"),
        ];
        for (source, expected) in sources {
            let json = serde_json::to_value(source).unwrap();
            assert_eq!(json.as_str().unwrap(), expected);
        }
    }
}
