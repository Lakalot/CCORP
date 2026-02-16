use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutePolicyConfig {
    pub routes: Vec<RouteTargetConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteTargetConfig {
    pub provider: String,
    pub model: String,
    pub priority: u32,
}

pub type ResolvedRoute = RouteTargetConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutePolicyError {
    message: String,
}

impl RoutePolicyError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for RoutePolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RoutePolicyError {}

pub fn validate_and_sort_routes(
    policy: &RoutePolicyConfig,
    known_providers: &HashSet<String>,
    known_models: &HashSet<String>,
) -> Result<Vec<ResolvedRoute>, RoutePolicyError> {
    if policy.routes.is_empty() {
        return Err(RoutePolicyError::new(
            "route_policy.routes must not be empty",
        ));
    }

    let mut seen_priority = HashMap::<u32, &RouteTargetConfig>::new();
    let mut seen_target = HashSet::<(String, String)>::new();
    let mut normalized = Vec::with_capacity(policy.routes.len());

    for route in &policy.routes {
        if route.priority == 0 {
            return Err(RoutePolicyError::new(
                "route_policy priority must start at 1",
            ));
        }

        if !known_providers.contains(&route.provider) {
            return Err(RoutePolicyError::new(format!(
                "Unknown provider '{}' in route_policy",
                route.provider
            )));
        }

        if !known_models.contains(&route.model) {
            return Err(RoutePolicyError::new(format!(
                "Unknown model '{}' in route_policy",
                route.model
            )));
        }

        if let Some(existing) = seen_priority.insert(route.priority, route) {
            return Err(RoutePolicyError::new(format!(
                "Duplicate priority {} in route_policy ({}:{}, {}:{})",
                route.priority, existing.provider, existing.model, route.provider, route.model
            )));
        }

        let key = (route.provider.clone(), route.model.clone());
        if !seen_target.insert(key.clone()) {
            return Err(RoutePolicyError::new(format!(
                "Ambiguous route_policy: duplicate route target {}:{}",
                key.0, key.1
            )));
        }

        normalized.push(route.clone());
    }

    normalized.sort_by_key(|route| route.priority);

    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::{RoutePolicyConfig, RouteTargetConfig, validate_and_sort_routes};
    use std::collections::HashSet;

    fn known_providers() -> HashSet<String> {
        HashSet::from([String::from("openrouter")])
    }

    fn known_models() -> HashSet<String> {
        HashSet::from([
            String::from("haiku"),
            String::from("sonnet"),
            String::from("opus"),
        ])
    }

    #[test]
    fn sorts_routes_by_priority_deterministically() {
        let policy = RoutePolicyConfig {
            routes: vec![
                RouteTargetConfig {
                    provider: "openrouter".to_string(),
                    model: "opus".to_string(),
                    priority: 3,
                },
                RouteTargetConfig {
                    provider: "openrouter".to_string(),
                    model: "haiku".to_string(),
                    priority: 2,
                },
                RouteTargetConfig {
                    provider: "openrouter".to_string(),
                    model: "sonnet".to_string(),
                    priority: 1,
                },
            ],
        };

        let ordered = validate_and_sort_routes(&policy, &known_providers(), &known_models())
            .expect("policy should be valid");
        assert_eq!(ordered[0].model, "sonnet");
        assert_eq!(ordered[1].model, "haiku");
        assert_eq!(ordered[2].model, "opus");
    }

    #[test]
    fn rejects_duplicate_priorities() {
        let policy = RoutePolicyConfig {
            routes: vec![
                RouteTargetConfig {
                    provider: "openrouter".to_string(),
                    model: "sonnet".to_string(),
                    priority: 1,
                },
                RouteTargetConfig {
                    provider: "openrouter".to_string(),
                    model: "haiku".to_string(),
                    priority: 1,
                },
            ],
        };

        let err = validate_and_sort_routes(&policy, &known_providers(), &known_models())
            .expect_err("duplicate priorities must fail");
        assert!(err.message().contains("Duplicate priority"));
    }

    #[test]
    fn rejects_empty_policy() {
        let policy = RoutePolicyConfig { routes: vec![] };

        let err = validate_and_sort_routes(&policy, &known_providers(), &known_models())
            .expect_err("empty policy must fail");
        assert_eq!(err.message(), "route_policy.routes must not be empty");
    }

    #[test]
    fn rejects_unknown_provider() {
        let policy = RoutePolicyConfig {
            routes: vec![RouteTargetConfig {
                provider: "other-provider".to_string(),
                model: "sonnet".to_string(),
                priority: 1,
            }],
        };

        let err = validate_and_sort_routes(&policy, &known_providers(), &known_models())
            .expect_err("unknown provider must fail");
        assert!(err.message().contains("Unknown provider"));
    }

    #[test]
    fn rejects_unknown_model() {
        let policy = RoutePolicyConfig {
            routes: vec![RouteTargetConfig {
                provider: "openrouter".to_string(),
                model: "unknown".to_string(),
                priority: 1,
            }],
        };

        let err = validate_and_sort_routes(&policy, &known_providers(), &known_models())
            .expect_err("unknown model must fail");
        assert!(err.message().contains("Unknown model"));
    }

    #[test]
    fn rejects_duplicate_targets() {
        let policy = RoutePolicyConfig {
            routes: vec![
                RouteTargetConfig {
                    provider: "openrouter".to_string(),
                    model: "sonnet".to_string(),
                    priority: 1,
                },
                RouteTargetConfig {
                    provider: "openrouter".to_string(),
                    model: "sonnet".to_string(),
                    priority: 2,
                },
            ],
        };

        let err = validate_and_sort_routes(&policy, &known_providers(), &known_models())
            .expect_err("duplicate targets must fail");
        assert!(err.message().contains("Ambiguous route_policy"));
    }
}
