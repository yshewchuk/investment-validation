"""Closed IV policy for the shared durable comparison journal; no second executor."""
import math
from statistics import mean

from engine.v2.foundation import canonical_json
from engine.v2.ops.errors import fail
from experiments.iv_crush_comparator import COLUMNS, fit_evidence, recipe
from experiments.iv_crush_plan import target
from experiments.iv_crush_search import prepare_plan as prepare_plan
from experiments.regression_search_binding import fold_inputs as fold_inputs


def _equal(actual, expected):
    if canonical_json(actual) != canonical_json(expected):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "IV comparison differs from admitted inputs")


def execution(plan):
    """Only the installed signed recipe is eligible for the fixed child dispatch."""
    _equal(plan.get("target"), target())
    _equal(plan.get("comparison"), recipe())
    return {"schema_version": "iv_crush_comparison_execution.v1", "features": list(COLUMNS),
            "fit_recipe": plan["comparison"], "target": target()}


def validate_result(result, inputs, fold, plan, *, fits=None):
    """Reconstruct exact coverage/accounting from causal arrays, never evaluation labels."""
    _equal(sorted(result) if isinstance(result, dict) else None, ["event", "fit_details", "forecasts"])
    _equal(result["event"], "succeeded")
    expected, details = [], []
    baseline = mean(row["crush_pct_iv30"] for row in inputs["training"])
    for fit in fit_evidence(inputs, fold, plan) if fits is None else fits:
        horizon, fitted = fit["horizon_sessions"], fit["fit_eligible"]
        details.append({"horizon_sessions": horizon, "training_event_ids": fit["training_event_ids"],
                        "logical_fits": int(fitted), "component_fits": int(fitted)})
        eligible = set(fit["evaluation_event_ids"])
        for event in fold["evaluation_event_ids"]:
            reason = ("unavailable_input" if event not in eligible else
                      "insufficient_training" if not fitted else None)
            expected.append({"event_id": event, "horizon_sessions": horizon,
                             "abstention": reason, "training_mean": baseline})
    _equal(result["fit_details"], details)
    _equal(type(result["forecasts"]) is list, True)
    _equal(len(result["forecasts"]), len(expected))
    expected.sort(key=lambda row: (row["event_id"], row["horizon_sessions"]))
    for row, wanted in zip(result["forecasts"], expected):
        field = "predicted_crush_pct_iv30"
        _equal(sorted(row) if isinstance(row, dict) else None, sorted([*wanted, field]))
        _equal({key: value for key, value in row.items() if key != field}, wanted)
        value = row[field]
        if ((wanted["abstention"] is not None and value is not None)
                or (wanted["abstention"] is None
                    and (type(value) not in (int, float) or not math.isfinite(value)))):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "invalid signed IV comparison forecast")
