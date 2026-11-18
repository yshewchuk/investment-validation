"""Pure v3-v9 validation with bounded grids, fixed SIZE and named neural profiles."""
import json
import math

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_panel_qualified_table import recipe as qualified_panel_recipe
from experiments import magnitude_search_request as v2
from experiments.magnitude_search_refits import outer_refit_bound
from experiments.magnitude_search_request_schema import (
    allows_hgb_axes, allows_neural, allows_size_family, request_version,
    uses_chrono_neural_profiles, uses_extended_neural_profiles)
from experiments.size_blend_profiles import (
    distinct_component_count, resolve_blends, validate_blends)

__all__ = ["compile_request", "outer_units", "spec_request_id"]

_MAX_AXIS_LENGTH = 32
_MAX_V4_CANDIDATES = 1000
_BOUNDS = {
    ("ridge", "alpha"): (1e-6, 10000.0),
    ("elasticnet", "alpha"): (1e-6, 1000.0),
    ("elasticnet", "l1_ratio"): (1e-6, 1.0),
    ("hgb", "learning_rate"): (1e-3, 1.0),
    ("hgb", "max_leaf_nodes"): (2, 64),
    ("hgb", "min_samples_leaf"): (1, 1000),
    ("hgb", "l2_regularization"): (0, 1000),
    ("hgb", "max_iter"): (100, 5000),
}
_HGB_V9_OPTIONAL = {"loss", "max_iter", "quantile"}
_HGB_V9_DEFAULTS = {"loss": ["squared_error"], "max_iter": [500]}
_SPLINE_SETS = {"HMOSX", "HMOSXL", "HMOSXLP"}
_SPLINE_KNOTS = (0, 4, 5, 6)
_HGB_LOSSES = ("absolute_error", "gamma", "poisson", "quantile", "squared_error")
_SIGNED_HGB_LOSSES = tuple(loss for loss in _HGB_LOSSES if loss not in ("gamma", "poisson"))


def hgb_losses(target_domain="nonnegative"):
    """Return the losses admitted for a target domain."""
    if target_domain == "nonnegative":
        return _HGB_LOSSES
    if target_domain == "signed":
        return _SIGNED_HGB_LOSSES
    raise ValueError("HGB losses require a signed or nonnegative target domain")


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
_FAMILIES_V5 = {**_FAMILIES_V4, "size_v1_4": set()}
_FAMILIES_V6 = {**_FAMILIES_V5, "mlp": {"profile"}}
_MLP_PROFILES = {"relu32_l2_0p01", "relu64x32_l2_0p01", "relu64x32_l2_0p1"}
_MLP_PROFILES_V7 = _MLP_PROFILES | {
    "es_relu32_l2_0p01", "es_relu32_l2_1", "es_relu64x32_l2_0p01",
    "es_relu64x32_l2_1", "fixed200_relu64x32_l2_1", "fixed200_relu64x32_l2_10",
    "fixed200_relu64x32_l2_100"}
_MLP_PROFILES_V8 = _MLP_PROFILES_V7 | {
    "chrono15_relu32_l2_0p01", "chrono15_relu32_l2_1",
    "chrono15_relu64x32_l2_0p01", "chrono15_relu64x32_l2_1",
    "chrono10_relu64x32_l2_0p01", "chrono10_relu64x32_l2_1"}
_MLP_PROFILES_V13 = _MLP_PROFILES_V8 | {"chrono15_relu128x64_l2_1", "chrono15_relu128x64x32_l2_1"}


_SELECTION_METRICS = {"rmse": 0.6, "mae": 0.3, "large_move_underprediction": 0.1}
_SELECTION_LEADS = {"1": 1.0, "3": 0.0, "5": 0.0}


def _hash_recipe(version):
    if version >= 12:
        return lambda subset: qualified_panel_recipe("HMOSXLP")
    if version >= 11:
        return lambda subset: qualified_panel_recipe("HMOSX")
    if version >= 10:
        return lambda subset: qualified_panel_recipe("HMOS")
    return qualified_panel_recipe if version >= 9 else v2.panel_recipe


def _winner_rules(request, policy, version):
    if version < 14 or "winner_rules" not in policy:
        return
    rules = policy["winner_rules"]
    v2._require(type(rules) is list and rules == ["overall", "rmse", "mae"],
                "stage-2 selection winner_rules must be exactly overall, rmse, mae")
    request["selection"]["winner_rules"] = ["overall", "rmse", "mae"]


def _extra_refits(request, policy):
    if "extra_refits" not in policy:
        return
    value = policy["extra_refits"]
    v2._require(type(value) is list and bool(value)
                and all(v2._hash(item) for item in value)
                and value == sorted(set(value)),
                "stage-2 selection extra_refits must be a sorted unique list of sha256 candidate ids")
    request["selection"]["extra_refits"] = list(value)


def _selection(request, horizons, version):
    if "selection" not in request:
        return
    v2._require(1 in horizons, "stage-2 blended selection requires lead-1 horizon")
    policy = request["selection"]
    keys = {"metric_weights", "lead_weights", "large_move_threshold_pct", "tie_band_floor"}
    if version >= 14 and type(policy) is dict:
        v2._require(not ("extra_refits" in policy and "winner_rules" not in policy),
                    "stage-2 selection extra_refits requires winner_rules")
    v2._require(v2._shape(policy, keys) if version < 14
                else type(policy) is dict and set(policy) in (
                    keys, keys | {"winner_rules"}, keys | {"winner_rules", "extra_refits"}),
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
    _winner_rules(request, policy, version)
    _extra_refits(request, policy)


def _blends(request, blends):
    blends = validate_blends(blends)
    v2._require("selection" in request and "winner_rules" in request["selection"],
                "stage-2 blends require selection.winner_rules")
    request["blends"] = list(blends)


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


def _log_target(value, canonical):
    v2._require(type(value) is dict and bool(value),
                "stage-2 log_target must be a nonempty object")
    v2._require(set(value) <= {"elasticnet", "mlp"},
                "stage-2 log_target names an unknown family")
    result = {}
    for family in sorted(value):
        v2._require(family in canonical,
                    "stage-2 log_target requires its base family grid")
        base = canonical[family]
        supplied = value[family]
        v2._require(type(supplied) is dict and set(supplied) == set(base),
                    "stage-2 log_target axes differ from the base grid")
        family_grid = {}
        for axis in sorted(base):
            values = supplied[axis]
            v2._require(type(values) is list and bool(values),
                        "stage-2 log_target axis must be a nonempty list")
            if axis == "profile":
                v2._require(all(type(item) is str for item in values)
                            and values == sorted(set(values)),
                            "stage-2 log_target MLP profiles must be unique"
                            " and lexicographically ordered")
                v2._require(all(item in base[axis] for item in values),
                            "stage-2 log_target profile is not in the base grid")
                family_grid[axis] = list(values)
            elif axis == "spline_knots":
                v2._require(all(type(item) is int for item in values),
                            "stage-2 log_target spline_knots values must be integers")
                v2._require(values == sorted(values) and len(set(values)) == len(values),
                            "stage-2 log_target axes must be strictly increasing")
                v2._require(all(item in base[axis] for item in values),
                            "stage-2 log_target spline_knots is not in the base grid")
                family_grid[axis] = list(values)
            else:
                v2._require(all(type(item) in (int, float) for item in values),
                            "stage-2 log_target grid values must be numbers")
                values = [float(item) for item in values]
                v2._require(values == sorted(values) and len(set(values)) == len(values),
                            "stage-2 log_target axes must be strictly increasing")
                v2._require(all(item in base[axis] for item in values),
                            "stage-2 log_target grid value is not in the base grid")
                family_grid[axis] = values
        result[family] = family_grid
    return result


def _space(request):
    version = request_version(request["schema_version"])
    allowed_sets = (*v2._SETS, "MO-", "M",
        *(["HMOS"] if version >= 10 else []), *(["HMOSX"] if version >= 11 else []),
        *(["HMOSXL", "HMOSXLP"] if version >= 12 else []))
    feature_sets = request["feature_sets"]
    valid_sets = (type(feature_sets) is list and bool(feature_sets)
        and all(type(subset) is str and subset in allowed_sets for subset in feature_sets)
        and feature_sets == [subset for subset in allowed_sets if subset in feature_sets])
    v2._require(valid_sets if version >= 9 else feature_sets == v2._SETS,
                (f"stage-2 v{version} requires a nonempty ordered subset of H/HM/HMO/MO-/M"
                 + ("/HMOS" if version >= 10 else "")
                 + ("/HMOSX" if version >= 11 else "")
                 + ("/HMOSXL/HMOSXLP" if version >= 12 else ""))
                if version >= 9 else "stage-2 requires exactly ordered H/HM/HMO feature sets")
    space = request["space"]
    version = request["schema_version"]
    version_number = request_version(version)
    has_log_target = False
    if version_number >= 14 and type(space) is dict and "log_target" in space:
        log_target = space.pop("log_target")
        has_log_target = True
    families = (_FAMILIES_V6 if allows_neural(version_number)
                else _FAMILIES_V5 if allows_size_family(version_number)
                else _FAMILIES_V4 if allows_hgb_axes(version_number) else _FAMILIES)
    v2._require((type(space) is dict and set(space) in (set(families), set(families) - {"ridge"}))
                if version_number >= 13 else
                (type(space) is dict and set(space) == set(families)) if version_number >= 9
                else v2._shape(space, families),
                "stage-2 requires the closed family grid")
    canonical = {}
    for family, axes in families.items():
        if family not in space:
            continue
        supplied = space[family]
        if version_number >= 9 and family == "hgb":
            declared = set(supplied) if type(supplied) is dict else None
            v2._require(declared is not None and axes <= declared
                        and declared <= axes | _HGB_V9_OPTIONAL,
                        "stage-2 grid axes differ from the closed policy")
            if declared - axes:
                supplied = {**supplied}
                for optional, default in _HGB_V9_DEFAULTS.items():
                    supplied.setdefault(optional, list(default))
                axes = axes | {optional for optional in _HGB_V9_OPTIONAL
                               if optional in supplied}
            loss_values = supplied.get("loss")
            if "quantile" in supplied or (type(loss_values) is list
                    and "quantile" in loss_values):
                v2._require(type(loss_values) is list and loss_values == ["quantile"],
                    "stage-2 HGB quantile loss must be the only declared HGB loss")
                v2._require("quantile" in supplied,
                    "stage-2 HGB quantile loss requires a quantile axis")
        else:
            if (version_number >= 13 and family == "elasticnet"
                    and type(supplied) is dict and "spline_knots" in supplied):
                axes = axes | {"spline_knots"}
            v2._require(v2._shape(supplied, axes),
                        "stage-2 grid axes differ from the closed policy")
        canonical[family] = {}
        for axis in sorted(axes):
            values = supplied[axis]
            v2._require(type(values) is list and 1 <= len(values) <= _MAX_AXIS_LENGTH,
                        "stage-2 grid axis length must be between 1 and 32")
            if family == "mlp":
                parsed_version = request_version(version)
                allowed = (_MLP_PROFILES_V13 if parsed_version >= 13
                           else _MLP_PROFILES_V8 if uses_chrono_neural_profiles(parsed_version)
                           else _MLP_PROFILES_V7 if uses_extended_neural_profiles(parsed_version)
                           else _MLP_PROFILES)
                v2._require(all(type(value) is str and value in allowed
                    for value in values)
                    and values == sorted(set(values)),
                    "stage-2 MLP profiles must be known, unique and lexicographically ordered")
                canonical[family][axis] = list(values)
                continue
            if family == "hgb" and not allows_hgb_axes(request_version(version)):
                v2._require(values == [7, 15],
                            "stage-2 HGB max_leaf_nodes grid remains fixed at [7, 15]")
                canonical[family][axis] = [7, 15]
                continue
            if family == "hgb" and axis == "loss":
                v2._require(all(type(value) is str and value in hgb_losses()
                    for value in values) and values == sorted(set(values)),
                    "stage-2 HGB loss values must be known, unique and lexicographically ordered")
                canonical[family][axis] = list(values)
                continue
            if family == "hgb" and axis == "quantile":
                v2._require(all(type(value) in (int, float) and 0 < value < 1
                    for value in values),
                    "stage-2 HGB quantile values must be numbers strictly between 0 and 1")
                v2._require(values == sorted(values) and len(set(values)) == len(values),
                            "stage-2 grid axes must be strictly increasing")
                canonical[family][axis] = [float(value) for value in values]
                continue
            if family == "elasticnet" and axis == "spline_knots":
                v2._require(all(type(value) is int and value in _SPLINE_KNOTS
                    for value in values),
                    "stage-2 spline_knots values must be 0 or integers 4..6")
                v2._require(values == sorted(values) and len(set(values)) == len(values),
                            "stage-2 grid axes must be strictly increasing")
                v2._require(max(values) == 0 or all(subset in _SPLINE_SETS
                    for subset in request["feature_sets"]),
                    "stage-2 spline_knots > 0 requires feature sets that contain all"
                    " spline source columns (HMOSX, HMOSXL, HMOSXLP)")
                canonical[family][axis] = list(values)
                continue
            if family == "hgb" and axis in ("max_leaf_nodes", "min_samples_leaf", "max_iter"):
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
    if has_log_target:
        canonical["log_target"] = _log_target(log_target, canonical)
    request["space"] = canonical


def outer_units(request):
    """design section 3: winner refits (+ extra_refits) plus, when blends are declared, their components and the best-single refit"""
    return (outer_refit_bound(request.get("selection"))
            + (distinct_component_count(request["blends"]) + 1 if "blends" in request else 0))


def _counts(request, horizons):
    outer, inner = request["outer_years"], request["inner_years"]
    v2._require(v2._years(outer) and v2._shape(inner, outer),
                "stage-2 requires ordered outer years and exact inner-year keys")
    v2._require(all(v2._years(inner[year]) and inner[year][-1] < year for year in outer),
                "stage-2 inner years must be ordered and strictly earlier than their outer year")
    size = (allows_size_family(request_version(request["schema_version"]))
            and "HMO" in request["feature_sets"])
    count = int(size) + len(request["feature_sets"]) * sum(
        math.prod(len(values) for values in axes.values())
        for family, axes in request["space"].items()
        if family not in ("size_v1_4", "log_target"))
    count += len(request["feature_sets"]) * sum(
        math.prod(len(values) for values in axes.values())
        for axes in request["space"].get("log_target", {}).values())
    if allows_hgb_axes(request_version(request["schema_version"])):
        v2._require(count <= _MAX_V4_CANDIDATES,
                    "stage-2 v6 and later grid exceeds the 1000-candidate ceiling"
                    if allows_neural(request_version(request["schema_version"]))
                    else "stage-2 v4/v5 grid exceeds the 1000-candidate ceiling")
    inners, outers = sum(map(len, inner.values())), len(outer)
    lead_count = len(horizons)
    units = outer_units(request)
    fits = lead_count * (count * inners + units * outers)
    baseline_logical_fits = lead_count * outers
    baseline_component_fits = 2 * lead_count * outers
    extra_components = lead_count * (inners + outers) if size else 0
    v2._require(v2._positive(request["max_candidates"]) and request["max_candidates"] >= count
                and v2._positive(request["max_fits"]) and request["max_fits"] >= fits,
                "stage-2 candidate or search-fit cap is insufficient")
    return {"candidates": count, "inner_folds": inners, "outer_folds": outers,
        "search_fits": fits, "baseline_logical_fits": baseline_logical_fits,
        "baseline_component_fits": baseline_component_fits,
        "total_logical_fits": fits + baseline_logical_fits,
        "total_component_fits": fits + extra_components + baseline_component_fits}


def spec_request_id(request):
    """Request id used for candidate specs; ignores selection.extra_refits.

    Candidate ids must not depend on extra_refits (they are the ids extras name).
    """
    selection = request.get("selection")
    if type(selection) is not dict or "extra_refits" not in selection:
        return content_hash(request)
    stripped = {**request, "selection": {k: v for k, v in selection.items() if k != "extra_refits"}}
    return content_hash(stripped)


def compile_request(request, *, horizons=(5, 3, 1)):
    """Return canonical declarations/counts; no source/fold verification or I/O."""
    horizons = v2.validate_horizons(horizons)
    try:
        v2._plain(request)
        request = json.loads(json.dumps(request, allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "stage-2 request requires plain finite JSON") from None
    version = request.get("schema_version") if type(request) is dict else None
    has_blends = (type(request) is dict and "blends" in request
                  and request_version(version) is not None and request_version(version) >= 14)
    if has_blends:
        blends = request.pop("blends")
    request_keys = set(request) if type(request) is dict else set()
    v2._require((version == "magnitude_search_request.v3"
                 and request_keys in (v2._KEYS, v2._KEYS | {"selection"},
                                      v2._KEYS | {"workers"},
                                      v2._KEYS | {"selection", "workers"}))
                or (allows_hgb_axes(request_version(version))
                    and request_keys in (v2._KEYS | {"selection"},
                                         v2._KEYS | {"selection", "workers"})),
                "stage-2 request requires its closed v6 and later schema and selection policy"
                if allows_neural(request_version(version))
                else "stage-2 request requires its closed v3/v4/v5 schema and selection policy")
    v2.validate_workers(request)
    request.pop("workers", None)
    _selection(request, horizons, request_version(version))
    if has_blends:
        _blends(request, blends)
    v2._references(request, recipe=_hash_recipe(request_version(version)))
    _space(request)
    counts = _counts(request, horizons)
    if has_blends:
        counts["blends"] = len(request["blends"])
    v2._comparison(request["comparison"], counts)
    v2._limits(request["limits"])
    request_id = content_hash(request)
    specs = [s for s in v2._specifications(request, spec_request_id(request),
        recipe=qualified_panel_recipe if request_version(version) >= 9 else v2.panel_recipe)
        if s["family"] != "size_v1_4" or s["feature_recipe"]["subset"] == "HMO"]
    if has_blends:
        resolve_blends(request["blends"], specs)
    result = {"schema_version": "magnitude_search_schedule.v2", "request": request,
        "request_id": request_id, "horizons": list(horizons), "candidate_specs": specs,
        "schedule": [{"outer_year": year, "inner_years": request["inner_years"][year]}
                     for year in request["outer_years"]], "counts": counts}
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
