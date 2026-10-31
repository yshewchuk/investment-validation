"""Source-free validation of the fixed nonnegative candidate schema/model pair."""


def magnitude_candidate_family(recipe):
    """Select a closed family before any training import; not full recipe validation."""
    message = "candidate requires a supported recipe schema/model pair"
    if type(recipe) is not dict:
        raise ValueError(message)
    schema = recipe.get("schema_version")
    if type(schema) is not str:
        raise ValueError(message)
    if schema == "size_causal_comparator.v1":
        return "size_v1_4"
    model = recipe.get("model")
    if type(model) is not str:
        raise ValueError(message)
    if schema == "magnitude_elasticnet_recipe.v1" and model == "elasticnet":
        return "elasticnet"
    if schema == "magnitude_ridge_hgb_recipe.v1" and model in ("history_ridge", "history_hgb"):
        return "ridge_hgb"
    if schema == "magnitude_ridge_hgb_recipe.v2" and model == "history_hgb":
        return "ridge_hgb"
    raise ValueError(message)
