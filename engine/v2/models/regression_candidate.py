"""Source-free closed three-family regression recipes with an explicit target domain."""
from .magnitude_elasticnet_recipe import regression_elasticnet_recipe
from .regression_elasticnet_spline_recipe import regression_elasticnet_spline_recipe
from .regression_mlp_recipe import regression_mlp_recipe
from .regression_ridge_hgb_recipe import regression_recipe

_V1_HGB_KEYS = {"max_leaf_nodes"}
_EARLY_HGB_KEYS = {"max_leaf_nodes", "learning_rate", "min_samples_leaf", "l2_regularization"}
_HGB_OPTIONAL_KEYS = {"loss", "max_iter", "quantile"}


def _keys_ok(family, keys):
    if family == "ridge":
        return keys == {"alpha"}
    if family == "elasticnet":
        return keys == {"alpha", "l1_ratio"} or keys == {
            "alpha", "l1_ratio", "spline_knots"}
    if family == "mlp":
        return keys == {"profile"}
    if family == "hgb":
        return keys == _V1_HGB_KEYS or (
            _EARLY_HGB_KEYS <= keys <= _EARLY_HGB_KEYS | _HGB_OPTIONAL_KEYS)
    return False


def regression_candidate_recipe(family, parameters, *, target_domain, columns, feature_dictionary_hash):
    """Assemble existing policy; this grants no source, selection or fitting authority."""
    if (type(family) is not str or type(parameters) is not dict
            or not _keys_ok(family, set(parameters))):
        raise ValueError("regression candidate requires a supported family and exact parameter keys")
    if family == "elasticnet":
        if "spline_knots" in parameters:
            knots = parameters["spline_knots"]
            if type(knots) is not int or not (knots == 0 or 4 <= knots <= 6):
                raise ValueError("regression ElasticNet spline_knots requires 0 or 4 through 6")
            if knots == 0:
                return regression_elasticnet_recipe(target_domain=target_domain,
                    columns=columns, feature_dictionary_hash=feature_dictionary_hash,
                    alpha=parameters["alpha"], l1_ratio=parameters["l1_ratio"])
            return regression_elasticnet_spline_recipe(target_domain=target_domain,
                columns=columns, feature_dictionary_hash=feature_dictionary_hash,
                alpha=parameters["alpha"], l1_ratio=parameters["l1_ratio"], n_knots=knots)
        return regression_elasticnet_recipe(target_domain=target_domain, columns=columns,
            feature_dictionary_hash=feature_dictionary_hash, **parameters)
    if family == "mlp":
        return regression_mlp_recipe(target_domain=target_domain, columns=columns,
            feature_dictionary_hash=feature_dictionary_hash, profile=parameters["profile"])
    return regression_recipe(target_domain=target_domain,
        model="history_ridge" if family == "ridge" else "history_hgb", columns=columns,
        feature_dictionary_hash=feature_dictionary_hash, **parameters)
