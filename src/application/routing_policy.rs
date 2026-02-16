use crate::{
    config::Config,
    domain::route_policy::{ResolvedRoute, RoutePolicyError, validate_and_sort_routes},
};

pub fn resolve_primary_route(
    config: &Config,
    mapped_request_model: &str,
) -> Result<ResolvedRoute, RoutePolicyError> {
    if let Some(policy) = &config.route_policy {
        let ordered =
            validate_and_sort_routes(policy, &config.known_providers(), &config.known_models())?;
        return ordered
            .into_iter()
            .next()
            .ok_or_else(|| RoutePolicyError::new("route_policy.routes must not be empty"));
    }

    Ok(ResolvedRoute {
        provider: "openrouter".to_string(),
        model: mapped_request_model.to_string(),
        priority: 1,
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        application::routing_policy::resolve_primary_route,
        config::Config,
        domain::route_policy::{RoutePolicyConfig, RouteTargetConfig},
    };

    fn config_with_policy(policy: Option<RoutePolicyConfig>) -> Config {
        Config {
            port: 3332,
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: "dummy".to_string(),
            inbound_api_key: "dummy".to_string(),
            model_haiku: "haiku".to_string(),
            model_sonnet: "sonnet".to_string(),
            model_opus: "opus".to_string(),
            route_policy: policy,
        }
    }

    #[test]
    fn defaults_to_mapped_model_without_policy() {
        let cfg = config_with_policy(None);
        let primary = resolve_primary_route(&cfg, "sonnet").expect("route should resolve");
        assert_eq!(primary.provider, "openrouter");
        assert_eq!(primary.model, "sonnet");
        assert_eq!(primary.priority, 1);
    }

    #[test]
    fn uses_policy_primary_when_present() {
        let cfg = config_with_policy(Some(RoutePolicyConfig {
            routes: vec![
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
        }));
        let primary = resolve_primary_route(&cfg, "opus").expect("route should resolve");
        assert_eq!(primary.model, "sonnet");
        assert_eq!(primary.priority, 1);
    }

    #[test]
    fn rejects_invalid_policy() {
        let cfg = config_with_policy(Some(RoutePolicyConfig {
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
        }));
        let err = resolve_primary_route(&cfg, "opus").expect_err("policy should fail validation");
        assert!(err.message().contains("Duplicate priority"));
    }
}
