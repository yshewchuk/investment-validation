"""Pure v3/v4 stage-2 request validation with bounded request-declared linear grids."""
import json
import math

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments import magnitude_search_request as v2

__all__ = ["compile_request"]

_MAX_AXIS_LENGTH = 32
_MAX_V4_CANDIDATES = 1000
_BOUNDS = {
    ("ridge", "alpha"): (1e-6, 1000.0),
    ("elasticnet", "alpha"): (1e-6, 1000.0),
    ("elasticnet", "l1_ratio"): (1e-6, 1.0),
    ("hgb", "learning_rate"): (1e-3, 1.0),
    ("hgb", "max_leaf_nodes"): (2, 64),
    ("hgb", "min_samples_leaf"): (1, 1000),
    ("hgb", "l2_regularization"): (0, 1000),
}
_FAMILIES = {
    "ridge": {"alpha"},
    "elasticnet": {"alpha", "l1_ratio"},
    "hgb": {"max_leaf_nodes"},
}
_FAMILIES_V4 = {
    "ridge": {"alpha"},
    "elasticnet": {"alpha", "l1_ratio"},
    "hgb": {"learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"},
}


_SELECTION_METRICS = {"rmse": 0.6, "mae": 0.3, "large_move_underprediction": 0.1}
_SELECTION_LEADS = {"1": 1.0, "3": 0.0, "5": 0.0}


def _selection(request, horizons):
    if "selection" not in request:
        return
    v2._require(1 in horizons, "stage-2 blended selection requires lead-1 horizon")
    policy = request["selection"]
    v2._require(v2._shape(policy, {"metric_weights", "lead_weights",
        "large_move_threshold_pct", "tie_band_floor"}),
        "stage-2 selection policy has an invalid shape")
    metrics, leads = policy["metric_weights"], policy["lead_weights"]
    v2._require(v2._shape(metrics, set(_SELECTION_METRICS))
        and all(type(metrics[key]) in (int, float) and metrics[key] == expected
                and math.isfinite(metrics[key]) for key, expected in _SELECTION_METRICS.items()),
        "stage-2 selection metric weights differ from the registered policy")
    v2._require(v2._shape(leads, set(_SELECTION_LEADS))
        and all(type(leads[key]) in (int, float) and leads[key] == expected
                and math.isfinite(leads[key]) for key, expected in _SELECTION_LEADS.items()),
        "stage-2 selection lead weights differ from the registered policy")
    threshold, floor = policy["large_move_threshold_pct"], policy["tie_band_floor"]
    v2._require(type(threshold) in (int, float) and threshold == 10.0 and math.isfinite(threshold)
        and type(floor) in (int, float) and floor == 0.005 and math.isfinite(floor),
        "stage-2 selection threshold or tie-band floor differs from the registered policy")
    request["selection"] = {"metric_weights": dict(_SELECTION_METRICS),
        "lead_weights": dict(_SELECTION_LEADS), "large_move_threshold_pct": 10.0,
        "tie_band_floor": 0.005}


def _finite_positive(value):
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _finite_nonnegative(value):
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _space(request):
    v2._require(request["feature_sets"] == v2._SETS,
                "stage-2 requires exactly ordered H/HM/HMO feature sets")
    space = request["space"]
    version = request["schema_version"]
    families = _FAMILIES_V4 if version == "magnitude_search_request.v4" else _FAMILIES
    v2._require(v2._shape(space, families),
                "stage-2 requires the closed three-family grid")
    canonical = {}
    for family, axes in families.items():
        supplied = space[family]
        v2._require(v2._shape(supplied, axes),
                    "stage-2 grid axes differ from the closed policy")
        canonical[family] = {}
        for axis in sorted(axes):
            values = supplied[axis]
            v2._require(type(values) is list and 1 <= len(values) <= _MAX_AXIS_LENGTH,
                        "stage-2 grid axis length must be between 1 and 32")
            if family == "hgb" and version != "magnitude_search_request.v4":
                v2._require(values == [7, 15],
                            "stage-2 HGB max_leaf_nodes grid remains fixed at [7, 15]")
                canonical[family][axis] = [7, 15]
                continue
            if family == "hgb" and axis in ("max_leaf_nodes", "min_samples_leaf"):
                v2._require(all(type(value) is int for value in values),
                            "stage-2 HGB integer grid values must be integers")
                canonical[family][axis] = [int(value) for value in values]
            elif family == "hgb" and axis == "l2_regularization":
                v2._require(all(_finite_nonnegative(value) for value in values),
                            "stage-2 grid values must be finite nonnegative numbers")
                canonical[family][axis] = [float(value) for value in values]
            else:
                v2._require(all(_finite_positive(value) for value in values),
                            "stage-2 grid values must be finite positive numbers")
                canonical[family][axis] = [float(value) for value in values]
            v2._require(values == sorted(values) and len(set(values)) == len(values),
                        "stage-2 grid axes must be strictly increasing")
            lower, upper = _BOUNDS[(family, axis)]
            v2._require(all(lower <= value <= upper for value in values),
                        "stage-2 grid values exceed their allowed bounds")
    request["space"] = canonical


def _counts(request, horizons):
    outer, inner = request["outer_years"], request["inner_years"]
    v2._require(v2._years(outer) and v2._shape(inner, outer),
                "stage-2 requires ordered outer years and exact inner-year keys")
    v2._require(all(v2._years(inner[year]) and inner[year][-1] < year for year in outer),
                "stage-2 inner years must be ordered and strictly earlier than their outer year")
    count = len(v2._SETS) * sum(math.prod(len(values) for values in axes.values())
                                for axes in request["space"].values())
    if request["schema_version"] == "magnitude_search_request.v4":
        v2._require(count <= _MAX_V4_CANDIDATES,
                    "stage-2 v4 grid exceeds the 1000-candidate ceiling")
    inners, outers = sum(map(len, inner.values())), len(outer)
    lead_count = len(horizons)
    fits = lead_count * (count * inners + outers)
    baseline_logical_fits = lead_count * outers
    baseline_component_fits = 2 * lead_count * outers
    v2._require(v2._positive(request["max_candidates"]) and request["max_candidates"] >= count
                and v2._positive(request["max_fits"]) and request["max_fits"] >= fits,
                "stage-2 candidate or search-fit cap is insufficient")
    return {"candidates": count, "inner_folds": inners, "outer_folds": outers,
        "search_fits": fits, "baseline_logical_fits": baseline_logical_fits,
        "baseline_component_fits": baseline_component_fits,
        "total_logical_fits": fits + baseline_logical_fits,
        "total_component_fits": fits + baseline_component_fits}


def compile_request(request, *, horizons=(5, 3, 1)):
    """Return canonical declarations/counts; no source/fold verification or I/O."""
    horizons = v2.validate_horizons(horizons)
    try:
        v2._plain(request)
        request = json.loads(json.dumps(request, allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "stage-2 request requires plain finite JSON") from None
    version = request.get("schema_version") if type(request) is dict else None
    request_keys = set(request) if type(request) is dict else set()
    v2._require((version == "magnitude_search_request.v3"
                 and request_keys in (v2._KEYS, v2._KEYS | {"selection"}))
                or (version == "magnitude_search_request.v4"
                    and request_keys == v2._KEYS | {"selection"}),
                "stage-2 request requires its closed v3/v4 schema and v4 selection policy")
    _selection(request, horizons)
    v2._references(request)
    _space(request)
    counts = _counts(request, horizons)
    v2._comparison(request["comparison"], counts)
    v2._limits(request["limits"])
    request_id = content_hash(request)
    result = {"schema_version": "magnitude_search_schedule.v2", "request": request,
        "request_id": request_id, "horizons": list(horizons),
        "candidate_specs": v2._specifications(request, request_id),
        "schedule": [{"outer_year": year, "inner_years": request["inner_years"][year]}
                     for year in request["outer_years"]], "counts": counts}
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
