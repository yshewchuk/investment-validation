"""Pure blend outer forecast averaging: completed component refits become one blend fold result."""
import math

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments.magnitude_report import magnitude_metrics


def _conflict():
    return fail("EXPERIMENT_IDENTITY_CONFLICT", "blend component rows do not align on events and targets")


def average_forecasts(forecast_lists, column="predicted_absolute_move_pct"):
    """forecast_lists: one row list per component, in sorted component-id order.
    Rows key on (event_id, horizon_sessions); every list must have the same keys with no duplicates,
    and equal absolute_move_pct where rows carry it, else refuse EXPERIMENT_IDENTITY_CONFLICT.
    Returns the first list's rows in order with `column` replaced by the fsum mean."""
    if not forecast_lists:
        raise _conflict()
    maps = []
    for rows in forecast_lists:
        by_key = {}
        for row in rows:
            key = (row["event_id"], row["horizon_sessions"])
            if key in by_key:
                raise _conflict()
            by_key[key] = row
        maps.append(by_key)
    keys = set(maps[0])
    if any(set(by_key) != keys for by_key in maps[1:]):
        raise _conflict()
    averaged = []
    for row in forecast_lists[0]:
        key = (row["event_id"], row["horizon_sessions"])
        if any(by_key[key].get("absolute_move_pct") != row.get("absolute_move_pct") for by_key in maps[1:]):
            raise _conflict()
        values = [by_key[key][column] for by_key in maps]
        averaged.append({**row, column: math.fsum(values) / len(forecast_lists)})
    return averaged


def blend_outer_result(plan, blend, year, component_results, component_forecasts):
    """component_results: completed collect fold results in sorted component-id order.
    component_forecasts: each component's full stored forecast rows, same order and length."""
    ids = sorted(blend["component_candidate_ids"])
    if not component_results or len(component_results) != len(ids):
        raise _conflict()
    if len(component_forecasts) != len(component_results):
        raise _conflict()
    first = component_results[0]
    if any(result["event_count"] != first["event_count"] for result in component_results):
        raise _conflict()
    predictions = average_forecasts([result["predictions"] for result in component_results])
    body = {
        "schema_version": "magnitude_search_outer_blend_result.v1",
        "plan_id": plan["plan_id"],
        "outer_year": year,
        "candidate": {"candidate_id": blend["blend_id"], "kind": "blend", "blend_name": blend["name"],
                      "component_candidate_ids": ids},
        "component_refits": [{"candidate_id": candidate_id, "result_id": result["result_id"]}
                             for candidate_id, result in zip(ids, component_results)],
        "forecasts_id": content_hash(average_forecasts(component_forecasts)),
        "evaluation_inputs": first["evaluation_inputs"],
        "event_count": first["event_count"],
        "forecast_count": len(predictions),
        "predictions": predictions,
        "metrics": magnitude_metrics(predictions),
        "per_horizon": [{"horizon_sessions": horizon, **magnitude_metrics([
            row for row in predictions if row["horizon_sessions"] == horizon])} for horizon in plan["horizons"]],
        "prospective_candidate_id": first["prospective_candidate_id"],
        "fit_attempt_count": sum(result["fit_attempt_count"] for result in component_results),
        "evaluation_authority": "selected_outer_fold_only",
        "final_read_authority": "none",
        "holdout_results": first["holdout_results"],
    }
    return {**body, "result_id": content_hash(body)}
