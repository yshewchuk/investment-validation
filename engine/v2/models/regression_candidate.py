"""Source-free closed three-family regression recipes with an explicit target domain."""
from .magnitude_elasticnet_recipe import regression_elasticnet_recipe
from .regression_ridge_hgb_recipe import regression_recipe

_KEYS = {"ridge": {"alpha"}, "elasticnet": {"alpha", "l1_ratio"}, "hgb": {"max_leaf_nodes"}}


def regression_candidate_recipe(family, parameters, *, target_domain, columns, feature_dictionary_hash):
    """Assemble existing policy; this grants no source, selection or fitting authority."""
    if (type(family) is not str or family not in _KEYS or type(parameters) is not dict
            or set(parameters) != _KEYS[family]):
        raise ValueError("regression candidate requires a supported family and exact parameter keys")
    if family == "elasticnet":
        return regression_elasticnet_recipe(target_domain=target_domain, columns=columns,
            feature_dictionary_hash=feature_dictionary_hash, **parameters)
    return regression_recipe(target_domain=target_domain,
        model="history_ridge" if family == "ridge" else "history_hgb", columns=columns,
        feature_dictionary_hash=feature_dictionary_hash, **parameters)
