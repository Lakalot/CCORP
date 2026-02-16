use dotenvy::dotenv;
use serde::Deserialize;
use serde::Serialize;
use std::env;
use std::fmt;
use std::fs;

use crate::domain::route_policy::{RoutePolicyConfig, validate_and_sort_routes};
use std::collections::HashSet;

/// TOML configuration structure
#[derive(Deserialize, Serialize)]
struct JsonConfig {
    port: u16,
    models: ModelConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    route_policy: Option<RoutePolicyConfig>,
}

#[derive(Deserialize, Serialize)]
struct ModelConfig {
    haiku: String,
    sonnet: String,
    opus: String,
}

/// Runtime configuration loaded from environment variables.
#[derive(Clone, Debug)]
pub struct Config {
    /// The port to listen on
    pub port: u16,
    /// Base URL for the OpenRouter API (e.g., https://openrouter.ai/api/v1)
    pub base_url: String,
    /// API key for authenticating with OpenRouter
    pub api_key: String,
    /// API key for authenticating inbound client requests (CCORP_API_KEY, falls back to api_key)
    pub inbound_api_key: String,
    /// Override model name for Claude 3.5 Haiku
    pub model_haiku: String,
    /// Override model name for Claude Sonnet 4
    pub model_sonnet: String,
    /// Override model name for Claude Opus 4
    pub model_opus: String,
    /// Optional deterministic route policy for upstream model/provider selection.
    pub route_policy: Option<RoutePolicyConfig>,
}

impl Config {
    /// Known upstream providers for route policy validation.
    pub fn known_providers(&self) -> HashSet<String> {
        HashSet::from(["openrouter".to_string()])
    }

    /// Known model aliases for route policy validation.
    pub fn known_models(&self) -> HashSet<String> {
        HashSet::from([
            self.model_haiku.clone(),
            self.model_sonnet.clone(),
            self.model_opus.clone(),
        ])
    }

    /// Load configuration from `config.json` and `.env` file.
    pub fn from_env() -> Self {
        Self::try_from_env().unwrap_or_else(|err| panic!("Configuration error: {err}"))
    }

    /// Deterministic runtime configuration loading from `.env` and `config.json`.
    pub fn try_from_env() -> Result<Self, ConfigError> {
        dotenv().ok();
        let api_key = env::var("OPENROUTER_API_KEY").ok();
        let inbound_api_key = env::var("CCORP_API_KEY").ok();
        let config_contents =
            fs::read_to_string("config.json").map_err(|_| ConfigError::ConfigFileRead)?;
        Self::try_from_sources(api_key, inbound_api_key, &config_contents)
    }

    /// Build runtime config from explicit sources. Used by startup and tests.
    pub(crate) fn try_from_sources(
        api_key: Option<String>,
        inbound_api_key: Option<String>,
        config_contents: &str,
    ) -> Result<Self, ConfigError> {
        let api_key = api_key
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty())
            .ok_or(ConfigError::MissingApiKey)?;

        let inbound_api_key = inbound_api_key
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty())
            .unwrap_or_else(|| api_key.clone());

        let config: JsonConfig =
            serde_json::from_str(config_contents).map_err(|_| ConfigError::ConfigParse)?;

        let result = Config {
            port: config.port,
            base_url: default_openrouter_base_url(),
            api_key,
            inbound_api_key,
            model_haiku: config.models.haiku,
            model_sonnet: config.models.sonnet,
            model_opus: config.models.opus,
            route_policy: config.route_policy,
        };

        if let Some(ref policy) = result.route_policy {
            validate_and_sort_routes(policy, &result.known_providers(), &result.known_models())
                .map_err(|e| ConfigError::InvalidRoutePolicy(e.message().to_string()))?;
        }

        Ok(result)
    }

    /// Write configuration to `config.json` (excluding secrets like api_key).
    /// Uses atomic write (temp file + rename) to prevent corruption.
    pub fn write(&self) -> Result<(), ConfigError> {
        let config_out = JsonConfig {
            port: self.port,
            models: ModelConfig {
                haiku: self.model_haiku.clone(),
                sonnet: self.model_sonnet.clone(),
                opus: self.model_opus.clone(),
            },
            route_policy: self.route_policy.clone(),
        };

        let json_string =
            serde_json::to_string_pretty(&config_out).map_err(|_| ConfigError::ConfigSerialize)?;

        let tmp_path = "config.json.tmp";
        fs::write(tmp_path, json_string).map_err(|_| ConfigError::ConfigFileWrite)?;
        fs::rename(tmp_path, "config.json").map_err(|_| ConfigError::ConfigFileWrite)?;
        Ok(())
    }
}

fn default_openrouter_base_url() -> String {
    "https://openrouter.ai/api/v1".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    MissingApiKey,
    ConfigFileRead,
    ConfigParse,
    ConfigSerialize,
    ConfigFileWrite,
    InvalidRoutePolicy(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::MissingApiKey => {
                write!(f, "OPENROUTER_API_KEY must be set via .env or environment")
            }
            ConfigError::ConfigFileRead => write!(f, "Could not read config.json file"),
            ConfigError::ConfigParse => write!(f, "Could not parse config.json file"),
            ConfigError::ConfigSerialize => {
                write!(f, "Could not serialize configuration")
            }
            ConfigError::ConfigFileWrite => write!(f, "Could not write config.json file"),
            ConfigError::InvalidRoutePolicy(msg) => {
                write!(f, "Invalid route policy: {msg}")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::{Config, ConfigError};

    #[test]
    fn rejects_invalid_json_config() {
        let result = Config::try_from_sources(Some("k".to_string()), None, "{");
        assert!(result.is_err());
    }

    #[test]
    fn rejects_missing_api_key() {
        let config = r#"{"port":3000,"models":{"haiku":"h","sonnet":"s","opus":"o"}}"#;
        let result = Config::try_from_sources(None, None, config);
        assert!(result.is_err());
    }

    #[test]
    fn accepts_valid_sources() {
        let config = r#"{"port":3000,"models":{"haiku":"h","sonnet":"s","opus":"o"}}"#;
        let result = Config::try_from_sources(Some("secret".to_string()), None, config);
        assert!(result.is_ok());
    }

    #[test]
    fn startup_smoke_accepts_minimal_required_sources() {
        let config = r#"{"port":3000,"models":{"haiku":"h","sonnet":"s","opus":"o"}}"#;
        let loaded = Config::try_from_sources(Some("secret".to_string()), None, config)
            .expect("minimal startup configuration should load");
        assert_eq!(loaded.port, 3000);
        assert_eq!(loaded.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(loaded.api_key, "secret");
        assert_eq!(
            loaded.inbound_api_key, "secret",
            "CCORP_API_KEY should default to OPENROUTER_API_KEY when omitted"
        );
    }

    #[test]
    fn startup_smoke_missing_api_key_fails_fast_with_actionable_diagnostic() {
        let config = r#"{"port":3000,"models":{"haiku":"h","sonnet":"s","opus":"o"}}"#;
        let error = Config::try_from_sources(None, None, config)
            .expect_err("startup should fail fast when OPENROUTER_API_KEY is absent");
        assert_eq!(error, ConfigError::MissingApiKey);
        assert!(
            error.to_string().contains("OPENROUTER_API_KEY"),
            "diagnostic should explain which environment variable is required"
        );
    }

    #[test]
    fn rejects_invalid_route_policy_at_load_time() {
        let config = r#"{"port":3000,"models":{"haiku":"h","sonnet":"s","opus":"o"},"route_policy":{"routes":[{"provider":"openrouter","model":"s","priority":1},{"provider":"openrouter","model":"h","priority":1}]}}"#;
        let error = Config::try_from_sources(Some("secret".to_string()), None, config)
            .expect_err("invalid route policy should fail at load time");
        assert!(matches!(error, ConfigError::InvalidRoutePolicy(_)));
        assert!(error.to_string().contains("Duplicate priority"));
    }

    #[test]
    fn startup_smoke_invalid_config_fails_fast_with_actionable_diagnostic() {
        let error = Config::try_from_sources(Some("secret".to_string()), None, "{")
            .expect_err("startup should fail fast on malformed config.json");
        assert_eq!(error, ConfigError::ConfigParse);
        assert_eq!(error.to_string(), "Could not parse config.json file");
    }
}
