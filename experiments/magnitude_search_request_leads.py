"""Qualified v2 schedule compiler with request-bound lead subsets."""
import json
import math

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments import magnitude_search_request as base

__all__ = ["compile_request"]


def _counts(request, horizons):
    outer, inner = request["outer_years"], request["inner_years"]
    base._require(base._years(outer) and base._shape(inner, outer),
                  "stage-2 requires ordered outer years and exact inner-year keys")
    base._require(all(base._years(inner[year]) and inner[year][-1] < year for year in outer),
                  "stage-2 inner years must be ordered and strictly earlier than their outer year")
    count = len(base._SETS) * sum(math.prod(map(len, axes.values())) for axes in base._SPACE.values())
    inners, outers = sum(map(len, inner.values())), len(outer)
    lead_count = len(horizons)
    fits = lead_count * (count * inners + outers)
    baseline_logical_fits = lead_count * outers
    baseline_component_fits = 2 * lead_count * outers
    base._require(base._positive(request["max_candidates"]) and request["max_candidates"] >= count
                  and base._positive(request["max_fits"]) and request["max_fits"] >= fits,
                  "stage-2 candidate or search-fit cap is insufficient")
    return {"candidates": count, "inner_folds": inners, "outer_folds": outers,
        "search_fits": fits, "baseline_logical_fits": baseline_logical_fits,
        "baseline_component_fits": baseline_component_fits,
        "total_logical_fits": fits + baseline_logical_fits,
        "total_component_fits": fits + baseline_component_fits}


def compile_request(request, *, horizons=(5, 3, 1)):
    """Compile v2 grids without binding the lead choice into request identity."""
    horizons = base.validate_horizons(horizons)
    try:
        base._plain(request)
        request = json.loads(json.dumps(request, allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "stage-2 request requires plain finite JSON") from None
    base._require((base._shape(request, base._KEYS)
                   or base._shape(request, base._KEYS | {"workers"}))
                  and request["schema_version"] == "magnitude_search_request.v2",
                  "stage-2 request requires its closed v2 schema")
    base.validate_workers(request)
    request.pop("workers", None)
    base._references(request)
    base._space(request)
    counts = _counts(request, horizons)
    base._comparison(request["comparison"], counts)
    base._limits(request["limits"])
    request_id = content_hash(request)
    result = {"schema_version": "magnitude_search_schedule.v2", "request": request,
        "request_id": request_id, "horizons": list(horizons),
        "candidate_specs": base._specifications(request, request_id),
        "schedule": [{"outer_year": year, "inner_years": request["inner_years"][year]}
                     for year in request["outer_years"]], "counts": counts}
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))
