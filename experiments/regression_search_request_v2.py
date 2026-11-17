"""regression_search_request.v2 compiler: SIZE-parity feature sets over the v1 families (P1 skeleton)."""
import json
import math
import re

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_panel_qualified_table import recipe as qualified_recipe
from experiments import magnitude_search_request as base
from experiments.magnitude_search_request_v3 import (
    _BOUNDS, _MAX_AXIS_LENGTH, _MAX_V4_CANDIDATES, _finite_positive, hgb_losses)
from experiments.size_panel_request import validate_panel_request

__all__ = ["compile_request"]

SCHEMA = "regression_search_request.v2"
_SETS = ["HMO", "HMOS", "HMOSX", "HMOSXL", "HMOSXLP"]
_HGB_REQUIRED = ("l2_regularization", "learning_rate", "max_leaf_nodes", "min_samples_leaf")
_HGB_OPTIONAL = ("loss", "max_iter", "quantile")
_HGB_DEFAULTS = {"loss": ["squared_error"], "max_iter": [500]}
_FAMILIES = {"ridge": ("alpha",), "elasticnet": ("alpha", "l1_ratio"), "hgb": _HGB_REQUIRED}
_SPLINE_KNOTS = (0, 4, 5, 6)
_SPLINE_SETS = ("HMOSX", "HMOSXL", "HMOSXLP")
_KEYS = base._KEYS | {"panel_request", "selection"}
_METRICS = {"rmse": 0.67, "mae": 0.33}
_LEADS = {"1": 1.0, "3": 0.0, "5": 0.0}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _finite_nonnegative(value):
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _selection(request):
    policy = request["selection"]
    base._require(base._shape(policy, {"metric_weights", "lead_weights", "tie_band_floor"}),
                  "stage-2 selection policy has an invalid shape")
    metrics, leads = policy["metric_weights"], policy["lead_weights"]
    base._require(base._shape(metrics, set(_METRICS))
                  and all(_number(metrics[key]) and metrics[key] == expected
                          for key, expected in _METRICS.items())
                  and base._shape(leads, set(_LEADS))
                  and all(_number(leads[key]) and leads[key] == expected
                          for key, expected in _LEADS.items()),
                  "stage-2 selection weights differ from the registered policy")
    floor = policy["tie_band_floor"]
    base._require(_number(floor) and floor >= 0,
                  "stage-2 selection tie-band floor must be a finite nonnegative number")
    request["selection"] = {"metric_weights": {"rmse": 0.67, "mae": 0.33},
                            "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
                            "tie_band_floor": float(floor)}


def _space(request):
    space = request["space"]
    families = set(_FAMILIES)
    base._require(type(space) is dict and set(space) in (families, families | {"mlp"}),
                  "stage-2 grid axes differ from the closed policy")
    canonical = {}
    for family in sorted(space):
        if family == "mlp":
            supplied = space[family]
            base._require(type(supplied) is dict and set(supplied) == {"profile"},
                          "stage-2 grid axes differ from the closed policy")
            values = supplied["profile"]
            base._require(type(values) is list and 1 <= len(values) <= _MAX_AXIS_LENGTH
                          and all(type(value) is str for value in values)
                          and values == sorted(set(values)),
                          "stage-2 mlp profile values must be known, unique and"
                          " lexicographically ordered")
            for profile in values:
                try:
                    mlp_recipe(columns=["x"],
                               feature_dictionary_hash=request["feature_dictionary_hash"],
                               profile=profile)
                except ValueError:
                    base._require(False, "stage-2 mlp profile values must be known, unique"
                                         " and lexicographically ordered")
            canonical[family] = {"profile": list(values)}
            continue
        required = _FAMILIES[family]
        supplied = space[family]
        if family == "hgb":
            declared = set(supplied) if type(supplied) is dict else set()
            base._require(type(supplied) is dict and set(required) <= declared
                          and declared <= set(required) | set(_HGB_OPTIONAL),
                          "stage-2 grid axes differ from the closed policy")
            supplied = {**supplied}
            for optional, default in _HGB_DEFAULTS.items():
                supplied.setdefault(optional, list(default))
            loss_values = supplied.get("loss")
            if "quantile" in declared or (type(loss_values) is list
                                          and "quantile" in loss_values):
                base._require(type(loss_values) is list and loss_values == ["quantile"],
                    "stage-2 HGB quantile loss must be the only declared HGB loss")
                base._require("quantile" in declared,
                    "stage-2 HGB quantile loss requires a quantile axis")
            axes = sorted(declared | set(_HGB_DEFAULTS))
        elif family == "elasticnet":
            declared = set(supplied) if type(supplied) is dict else set()
            base._require(type(supplied) is dict and set(required) <= declared
                          and declared <= set(required) | {"spline_knots"},
                          "stage-2 grid axes differ from the closed policy")
            supplied = {**supplied}
            axes = sorted(declared)
        else:
            base._require(base._shape(supplied, required),
                          "stage-2 grid axes differ from the closed policy")
            axes = sorted(required)
        canonical[family] = {}
        for axis in axes:
            values = supplied[axis]
            base._require(type(values) is list and 1 <= len(values) <= _MAX_AXIS_LENGTH,
                          "stage-2 grid axis length must be between 1 and 32")
            if family == "elasticnet" and axis == "spline_knots":
                base._require(all(type(value) is int and value in _SPLINE_KNOTS
                                  for value in values)
                              and values == sorted(set(values)),
                              "stage-2 spline_knots values must be ascending 0 or 4..6")
                if values == [0]:
                    continue
                if any(value > 0 for value in values):
                    base._require(all(subset in _SPLINE_SETS
                                      for subset in request["feature_sets"]),
                                  "stage-2 spline_knots above 0 need every feature set to"
                                  " hold the spline source columns")
                canonical[family][axis] = [int(value) for value in values]
                continue
            if family == "hgb" and axis == "loss":
                base._require(not any(type(value) is str and value in ("gamma", "poisson")
                                      for value in values),
                    "stage-2 regression v2 HGB loss gamma and poisson do not support a signed target")
                base._require(all(type(value) is str and value in hgb_losses("signed")
                                  for value in values) and values == sorted(set(values)),
                    "stage-2 HGB loss values must be known, unique and lexicographically ordered")
                canonical[family][axis] = list(values)
                continue
            if family == "hgb" and axis == "quantile":
                base._require(all(type(value) in (int, float) and 0 < value < 1
                                  for value in values),
                    "stage-2 HGB quantile values must be numbers strictly between 0 and 1")
                base._require(values == sorted(values) and len(set(values)) == len(values),
                              "stage-2 grid axes must be strictly increasing")
                canonical[family][axis] = [float(value) for value in values]
                continue
            integer_axis = family == "hgb" and axis in (
                "max_leaf_nodes", "min_samples_leaf", "max_iter")
            if integer_axis:
                base._require(all(type(value) is int for value in values),
                              "stage-2 grid values must be finite positive numbers"
                              " (integers for HGB count axes)")
                canonical[family][axis] = [int(value) for value in values]
            elif family == "hgb" and axis == "l2_regularization":
                base._require(all(_finite_nonnegative(value) for value in values),
                              "stage-2 grid values must be finite nonnegative numbers")
                canonical[family][axis] = [float(value) for value in values]
            else:
                base._require(all(_finite_positive(value) for value in values),
                              "stage-2 grid values must be finite positive numbers")
                canonical[family][axis] = [float(value) for value in values]
            base._require(values == sorted(values) and len(set(values)) == len(values),
                          "stage-2 grid axes must be strictly increasing")
            lower, upper = _BOUNDS[(family, axis)]
            base._require(all(lower <= value <= upper for value in values),
                          "stage-2 grid values exceed their allowed bounds")
    request["space"] = canonical


def _counts(request, lead):
    outer, inner = request["outer_years"], request["inner_years"]
    base._require(base._years(outer) and base._shape(inner, outer),
                  "stage-2 requires ordered outer years and exact inner-year keys")
    base._require(all(base._years(inner[year]) and inner[year][-1] < year for year in outer),
                  "stage-2 inner years must be ordered and strictly earlier than their outer year")
    count = len(request["feature_sets"]) * sum(
        math.prod(len(values) for values in axes.values()) for axes in request["space"].values())
    base._require(count <= _MAX_V4_CANDIDATES,
                  "stage-2 regression v2 grid exceeds the 1000-candidate ceiling")
    inners, outers = sum(map(len, inner.values())), len(outer)
    fits = len(lead) * (count * inners + outers)
    base._require(base._positive(request["max_candidates"]) and request["max_candidates"] >= count
                  and base._positive(request["max_fits"]) and request["max_fits"] >= fits,
                  "stage-2 candidate or search-fit cap is insufficient")
    baseline = len(lead) * outers
    return {"candidates": count, "inner_folds": inners, "outer_folds": outers,
            "search_fits": fits, "baseline_logical_fits": baseline,
            "baseline_component_fits": baseline, "total_logical_fits": fits + baseline,
            "total_component_fits": fits + baseline}


def compile_request(request, *, horizons=None):
    """Return canonical declarations/counts; no source/fold verification or I/O."""
    try:
        base._plain(request)
        request = json.loads(json.dumps(request, allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "stage-2 request requires plain finite JSON") from None
    base._require(type(request) is dict and request.get("schema_version") == SCHEMA
                  and set(request) in (_KEYS, _KEYS | {"workers"}),
                  "stage-2 request requires its closed regression v2 schema")
    if "workers" in request:
        workers = request["workers"]
        base._require(type(workers) is int and 1 <= workers <= 8,
                      "stage-2 workers requires an integer from 1 through 8")
        request.pop("workers")
    panel = validate_panel_request(request["panel_request"])
    if horizons is None:
        lead = [h for h in (5, 3, 1) if h in panel["horizons"]]
    else:
        lead = list(base.validate_horizons(horizons))
        base._require(set(lead) == set(panel["horizons"]),
                      "stage-2 horizons must match the panel request")
    base._require(1 in lead, "stage-2 blended selection requires lead-1 horizon")
    for value in (request["feature_snapshot_id"], request["history_snapshot_id"],
                  request["panel_request"]["price_history_snapshot"]):
        base._require(type(value) is str and re.fullmatch(r"snap_[0-9a-f]{32}", value),
                      "stage-2 snapshot IDs require native immutable ID syntax")
    base._references({**request, "schema_version": "magnitude_search_request.v2"},
                     recipe=lambda subset: qualified_recipe("HMOSXLP"))
    _selection(request)
    feature_sets = request["feature_sets"]
    base._require(type(feature_sets) is list and bool(feature_sets)
                  and all(type(subset) is str and subset in _SETS for subset in feature_sets)
                  and feature_sets == [subset for subset in _SETS if subset in feature_sets],
                  "stage-2 regression v2 requires a nonempty ordered subset of"
                  " HMO/HMOS/HMOSX/HMOSXL/HMOSXLP")
    _space(request)
    counts = _counts(request, lead)
    base._comparison(request["comparison"], counts, regression=True)
    base._require(request["comparison"]["max_component_fits"] >= counts["total_component_fits"],
                  "stage-2 comparison component cap is insufficient")
    base._limits(request["limits"])
    base._require(panel["feature_snapshot_id"] == request["feature_snapshot_id"]
                  and panel["history_snapshot"] == request["history_snapshot_id"],
                  "regression panel snapshots must match request IDs")
    base._require(request["limits"]["child"]["memory_bytes"]
                  <= request["limits"]["parent"]["memory_bytes"],
                  "child memory must not exceed inherited parent limit")
    request_id = content_hash(request)
    result = {"schema_version": "regression_search_schedule.v1", "request": request,
              "request_id": request_id, "horizons": lead,
              "candidate_specs": base._specifications(request, request_id, qualified_recipe),
              "schedule": [{"outer_year": year, "inner_years": request["inner_years"][year]}
                           for year in request["outer_years"]], "counts": counts}
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
