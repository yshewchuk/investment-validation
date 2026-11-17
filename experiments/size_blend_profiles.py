"""Fixed SIZE blend profiles: pure declarations, no I/O."""
import json

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail

FEATURE_SET = "HMOSXLP"
AGGREGATION = "equal_mean"
DEFINED_FROM = "interrupted_v12_inner_2023_2024"

EN = ("elasticnet", {"alpha": 0.03, "l1_ratio": 0.75, "spline_knots": 6})
HGB_ABS = ("hgb", {"learning_rate": 0.03, "l2_regularization": 10.0, "loss": "absolute_error",
                   "max_iter": 3000, "max_leaf_nodes": 15, "min_samples_leaf": 300})
HGB_SQ = ("hgb", {"learning_rate": 0.03, "l2_regularization": 10.0, "loss": "squared_error",
                  "max_iter": 3000, "max_leaf_nodes": 15, "min_samples_leaf": 1000})
MLP = ("mlp", {"profile": "chrono15_relu64x32_l2_1"})
PROFILES = {
    "size_b1_v1": (EN, HGB_ABS, HGB_SQ, MLP),
    "size_b2_v1": (HGB_ABS, HGB_SQ, MLP),
}


def validate_blends(value):
    """Refuse anything but a sorted, unique, nonempty list of known profile names."""
    if (type(value) is not list or not value
            or not all(type(name) is str and name in PROFILES for name in value)
            or value != sorted(set(value))):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "stage-2 blends must be a sorted unique nonempty list of known profiles")
    return list(value)


def resolve_blends(names, candidate_specs):
    """Map each profile to the base candidate spec ids of its components, in component order."""
    resolved = {}
    for name in names:
        ids = []
        for family, parameters in PROFILES[name]:
            matches = [spec["candidate_spec_id"] for spec in candidate_specs
                       if spec["family"] == family
                       and spec["feature_recipe"]["subset"] == FEATURE_SET
                       and spec["parameters"] == parameters]
            if not matches:
                raise fail("INVALID_EXPERIMENT_SPEC",
                           f"stage-2 blend {name} has a component with no compiled candidate")
            if len(matches) > 1:
                raise fail("INVALID_EXPERIMENT_SPEC",
                           f"stage-2 blend {name} has an ambiguous component")
            ids.append(matches[0])
        resolved[name] = ids
    return resolved


def distinct_component_count(names):
    """Number of distinct (family, parameters) components across the named profiles."""
    return len({(family, json.dumps(parameters, sort_keys=True))
                for name in names for family, parameters in PROFILES[name]})


def blend_records(resolved, candidate_id_by_spec):
    """Sealed plan records, one per blend, sorted by name."""
    records = []
    for name in sorted(resolved):
        profile = [{"family": family, "parameters": dict(parameters)}
                   for family, parameters in PROFILES[name]]
        component_ids = sorted(candidate_id_by_spec[spec_id] for spec_id in resolved[name])
        blend_id = content_hash({"profile": profile, "aggregation": AGGREGATION,
                                 "component_candidate_ids": component_ids})
        records.append({"name": name, "profile": profile, "aggregation": AGGREGATION,
                        "component_candidate_ids": component_ids, "blend_id": blend_id,
                        "defined_from": DEFINED_FROM, "inner_scores_optimistic": True})
    return records
