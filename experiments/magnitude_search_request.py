"""Pure stage-2 specifications/counts, never native admission or executable plans."""
import itertools
import json
import math
import re
import sys

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_panel_qualified_table import recipe as qualified_recipe
from engine.v2.research.magnitude_panel_table import recipe as panel_recipe
from experiments.magnitude_search_execution import validate_execution_limits
from experiments.size_panel_request import validate_panel_request

__all__ = ["compile_request", "validate_workers"]

_KEYS = {"schema_version", "feature_snapshot_id", "history_snapshot_id",
    "feature_dictionary_hash", "eligibility_policy_ref", "feature_sets", "space",
    "outer_years", "inner_years", "max_candidates", "max_fits", "comparison", "limits"}
_SPACE = {"ridge": {"alpha": [0.1, 1.0, 10.0]},
    "elasticnet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.25, 0.75]},
    "hgb": {"max_leaf_nodes": [7, 15]}}
_SETS = ["H", "HM", "HMO"]


def _require(condition, message):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", message)


def _plain(value):
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError
        for item in value.values():
            _plain(item)
    elif type(value) is list:
        for item in value:
            _plain(item)
    elif type(value) not in (str, int, float, bool, type(None)) or (
            type(value) is float and not math.isfinite(value)):
        raise ValueError


def _shape(value, keys):
    return type(value) is dict and set(value) == set(keys)


def _hash(value):
    return type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _positive(value):
    return type(value) is int and value > 0


def validate_workers(request):
    """Return the declared worker count (default 1); only v2/v3 requests may declare it."""
    if type(request) is not dict or "workers" not in request:
        return 1
    _require(request.get("schema_version") in ("magnitude_search_request.v2", "magnitude_search_request.v3"),
             "stage-2 workers require a v2 or v3 search request")
    _require(type(request["workers"]) is int and 1 <= request["workers"] <= 8,
             "stage-2 workers requires an integer from 1 through 8")
    return request["workers"]


def _references(request, recipe=panel_recipe):
    snapshots = [request["feature_snapshot_id"], request["history_snapshot_id"]]
    if request["schema_version"] == "regression_search_request.v1":
        snapshots.append(request["panel_request"]["price_history_snapshot"])
    for value in snapshots:
        _require(type(value) is str and re.fullmatch(r"snap_[0-9a-f]{32}", value) is not None,
                 "stage-2 snapshot IDs require native immutable ID syntax")
    _require(request["feature_dictionary_hash"] == recipe("HMO")["dictionary_hash"],
             "stage-2 feature dictionary differs from the supported panel dictionary")
    ref = request["eligibility_policy_ref"]
    policy = eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)
    _require(_shape(ref, {"schema_ref", "plan_id", "policy_id"})
             and ref["schema_ref"] == "group_eligibility_plan.v1" and _hash(ref["plan_id"])
             and ref["policy_id"] == policy["policy_id"],
             "stage-2 eligibility reference requires the declared 30-day policy")


def _space(request):
    _require(request["feature_sets"] == _SETS,
             "stage-2 requires exactly ordered H/HM/HMO feature sets")
    space = request["space"]
    _require(_shape(space, _SPACE), "stage-2 requires the closed three-family grid")
    for family, axes in _SPACE.items():
        _require(_shape(space[family], axes), "stage-2 grid axes differ from the closed policy")
        for axis, expected in axes.items():
            values = space[family][axis]
            _require(type(values) is list and len(values) == len(expected)
                     and all(type(v) in (int, float) and v == e for v, e in zip(values, expected)),
                     "stage-2 grid values differ from the closed policy")
    request["space"] = json.loads(json.dumps(_SPACE))


def _years(values):
    return (type(values) is list and bool(values)
        and all(type(y) is str and re.fullmatch(r"[0-9]{4}", y) is not None
                and 2 <= int(y) <= 9998 for y in values)
        and values == sorted(set(values)))


def _counts(request):
    outer, inner = request["outer_years"], request["inner_years"]
    _require(_years(outer) and _shape(inner, outer),
             "stage-2 requires ordered outer years and exact inner-year keys")
    _require(all(_years(inner[year]) and inner[year][-1] < year for year in outer),
             "stage-2 inner years must be ordered and strictly earlier than their outer year")
    count = len(_SETS) * sum(math.prod(map(len, axes.values())) for axes in _SPACE.values())
    inners, outers = sum(map(len, inner.values())), len(outer)
    fits = 3 * (count * inners + outers)
    _require(_positive(request["max_candidates"]) and request["max_candidates"] >= count
             and _positive(request["max_fits"]) and request["max_fits"] >= fits,
             "stage-2 candidate or search-fit cap is insufficient")
    components = (3 if request["schema_version"] == "regression_search_request.v1" else 6) * outers
    return {"candidates": count, "inner_folds": inners, "outer_folds": outers,
        "search_fits": fits, "baseline_logical_fits": 3 * outers,
        "baseline_component_fits": components, "total_logical_fits": fits + 3 * outers,
        "total_component_fits": fits + components}


def _comparison(value, counts, *, regression=False):
    reference, sanity = (("iv_crush_causal_refit", "training_mean") if regression
                         else ("size_v1_4_causal_refit", "history_mean"))
    _require(_shape(value, {"reference", "sanity_reference", "max_logical_fits", "max_component_fits"})
             and value["reference"] == reference and value["sanity_reference"] == sanity,
             "regression comparison requires iv_crush_causal_refit and training_mean references" if regression
             else "stage-2 comparison requires the fixed causal legacy and history-mean references")
    _require(all(_positive(value[cap]) and value[cap] >= counts[count] for cap, count in (
        ("max_logical_fits", "baseline_logical_fits"), ("max_component_fits", "baseline_component_fits"))),
        "stage-2 comparison fit cap is insufficient")


def _limits(value):
    _require(_shape(value, {"child", "parent", "build", "job"}),
             "stage-2 requires explicit child, parent, build and job limits")
    validate_execution_limits(value["child"])
    for name, keys in (("parent", {"memory_bytes", "wall_seconds"}),
                       ("build", {"memory_bytes", "wall_seconds"}), ("job", {"wall_seconds"})):
        limits = value[name]
        _require(_shape(limits, keys)
                 and all(_positive(v) and v < sys.maxsize for v in limits.values()),
                 "stage-2 parent/build/job limits require explicit bounded positive integers")


def _specifications(request, request_id, recipe=panel_recipe):
    specs = []
    for subset in request["feature_sets"]:
        feature_recipe = recipe(subset)
        for family, axes in request["space"].items():
            names = sorted(axes)
            for values in itertools.product(*(axes[name] for name in names)):
                spec = {"family": family, "feature_recipe": feature_recipe,
                        "parameters": dict(zip(names, values))}
                specs.append({**spec, "candidate_spec_id": content_hash({"request_id": request_id, **spec})})
    return sorted(specs, key=lambda spec: spec["candidate_spec_id"])


def compile_request(request):
    """Return canonical declarations/counts; no source/fold verification or I/O."""
    try:
        _plain(request)
        request = json.loads(json.dumps(request, allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "stage-2 request requires plain finite JSON") from None
    regression = type(request) is dict and request.get("schema_version") == "regression_search_request.v1"
    if regression:
        _require(_shape(request, _KEYS | {"panel_request"}),
                 "stage-2 request requires its closed v2 schema")
        validate_workers(request)
        validate_panel_request(request["panel_request"])
    else:
        _require((_shape(request, _KEYS) or _shape(request, _KEYS | {"workers"}))
                 and request["schema_version"] == "magnitude_search_request.v2",
                 "stage-2 request requires its closed v2 schema")
        validate_workers(request)
        request.pop("workers", None)
    recipe = qualified_recipe if regression else panel_recipe
    _references(request, recipe)
    _space(request)
    counts = _counts(request)
    _comparison(request["comparison"], counts, regression=regression)
    _limits(request["limits"])
    if regression:
        panel = request["panel_request"]
        _require(panel["feature_snapshot_id"] == request["feature_snapshot_id"]
                 and panel["history_snapshot"] == request["history_snapshot_id"],
                 "regression panel snapshots must match request IDs")
        _require(request["limits"]["child"]["memory_bytes"] <= request["limits"]["parent"]["memory_bytes"],
                 "child memory must not exceed inherited parent limit")
    request_id = content_hash(request)
    result = {"schema_version": "regression_search_schedule.v1" if regression else "magnitude_search_schedule.v2", "request": request,
        "request_id": request_id, "horizons": [5, 3, 1],
        "candidate_specs": _specifications(request, request_id, recipe),
        "schedule": [{"outer_year": year, "inner_years": request["inner_years"][year]}
                     for year in request["outer_years"]], "counts": counts}
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
