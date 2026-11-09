"""Source-free family recipe assembly, not candidate admission or a search plan."""
from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe

__all__ = ["candidate_recipe"]

_KEYS = {"ridge": {"alpha"}, "elasticnet": {"alpha", "l1_ratio"},
         "hgb": {"max_leaf_nodes"}, "mlp": {"profile"}}
_HGB_LEAF = {"max_leaf_nodes"}
_HGB_BASE = {"learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}
_HGB_OPTIONAL = {"loss", "max_iter", "quantile"}


def _hgb_keys(parameters):
    keys = set(parameters)
    return (_HGB_LEAF <= keys <= _HGB_LEAF | _HGB_OPTIONAL
            or _HGB_BASE <= keys <= _HGB_BASE | _HGB_OPTIONAL)


def candidate_recipe(family, parameters, *, columns, feature_dictionary_hash) -> dict:
    """Return a closed family recipe; columns and hashes prove no source admission."""
    if (type(family) is not str or family not in _KEYS or type(parameters) is not dict
            or (not _hgb_keys(parameters) if family == "hgb"
                else set(parameters) != _KEYS[family])):
        raise ValueError("candidate requires a supported family and exact parameter keys")
    if family == "elasticnet":
        return elasticnet_recipe(columns=columns, feature_dictionary_hash=feature_dictionary_hash,
                                 **parameters)
    if family == "mlp":
        return mlp_recipe(columns=columns, feature_dictionary_hash=feature_dictionary_hash,
                          **parameters)
    return ridge_hgb_recipe(model="history_ridge" if family == "ridge" else "history_hgb",
                           columns=columns, feature_dictionary_hash=feature_dictionary_hash,
                           **parameters)
