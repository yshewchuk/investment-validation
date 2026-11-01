"""Signed regression evidence and unchanged magnitude adapters, entirely synthetic."""
from fractions import Fraction
import math

import numpy as np
import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.research.regression_evidence import error_metrics, normalize_forecasts
from experiments.magnitude_report import magnitude_metrics
from experiments.magnitude_search_evidence import normalize_forecasts as magnitude_forecasts

FOLD = {"evaluation_event_ids": ["b", "a"]}
HORIZONS = [5, 3, 1]
COLUMN = "predicted_crush_pct_iv30"


def rows(column=COLUMN, value=-30):
    return [{"event_id": event, "horizon_sessions": h, column: value}
            for event in FOLD["evaluation_event_ids"] for h in HORIZONS]


def normalize(values, **kwargs):
    return normalize_forecasts(values, FOLD, HORIZONS,
        **{**dict(prediction_column=COLUMN, target_domain="signed"), **kwargs})


@pytest.mark.parametrize("value", [-120, -1, 0, 2, Fraction(-1, 3), np.float64(-4)])
def test_signed_exact_complete_coverage_is_detached_canonical(value):
    supplied = rows(value=value)
    result = normalize(supplied)
    assert [(r["event_id"], r["horizon_sessions"]) for r in result] == [
        (e, h) for e in ("a", "b") for h in (1, 3, 5)]
    assert all(type(r[COLUMN]) is float and r[COLUMN] == float(value) for r in result)
    result[0][COLUMN] = 999
    assert all(r[COLUMN] == value for r in supplied)


@pytest.mark.parametrize("value", [True, None, "-2", float("nan"), float("inf"), -float("inf")])
def test_invalid_signed_values_refuse(value):
    with pytest.raises((ValueError, TypeError)):
        normalize(rows(value=value))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "unknown_event", "unknown_horizon", "bool_horizon"])
def test_coverage_and_shape_fail_closed(mutation):
    supplied = rows()
    if mutation == "missing":
        supplied.pop()
    elif mutation == "duplicate":
        supplied[-1] = supplied[0].copy()
    elif mutation == "extra":
        supplied[0]["target"] = 0
    elif mutation == "unknown_event":
        supplied[0]["event_id"] = "other"
    else:
        supplied[0]["horizon_sessions"] = True if mutation == "bool_horizon" else 2
    with pytest.raises(ValueError):
        normalize(supplied)


@pytest.mark.parametrize("value", [Fraction(-1, 3), np.float64(-1)])
def test_strict_json_refuses_non_json_numeric_types(value):
    with pytest.raises(ValueError):
        normalize(rows(value=value), strict_json=True)


def test_allow_abstentions_true_allows_partial_but_refuses_bad_shape_or_empty():
    supplied = rows()
    supplied[0][COLUMN] = None
    result = normalize(supplied, allow_abstentions=True)
    assert [(r["event_id"], r["horizon_sessions"]) for r in result] == [
        ("a", 1), ("a", 3), ("a", 5), ("b", 1), ("b", 3)]

    missing = rows()
    missing.pop()
    with pytest.raises(ValueError):
        normalize(missing, allow_abstentions=True)

    duplicate = rows()
    duplicate[-1] = duplicate[0].copy()
    with pytest.raises(ValueError):
        normalize(duplicate, allow_abstentions=True)

    with pytest.raises(ValueError):
        normalize(rows(value=None), allow_abstentions=True)


@pytest.mark.parametrize("policy", [dict(target_domain="magnitude"), dict(target_domain=True),
    dict(prediction_column=""), dict(prediction_column="event_id"), dict(prediction_column="horizon_sessions")])
def test_invalid_explicit_policy_refuses(policy):
    with pytest.raises(ValueError):
        normalize([], **policy)


def test_nonnegative_adapter_keeps_old_domain_and_column():
    supplied = rows(column="predicted_absolute_move_pct", value=Fraction(1, 3))
    expected = normalize_forecasts(supplied, FOLD, HORIZONS,
        prediction_column="predicted_absolute_move_pct", target_domain="nonnegative")
    assert magnitude_forecasts(supplied, FOLD, HORIZONS) == expected
    with pytest.raises(ValueError):
        magnitude_forecasts(rows(column="predicted_absolute_move_pct"), FOLD, HORIZONS)
    with pytest.raises(ValueError):
        normalize(rows(), target_domain="nonnegative")


@pytest.mark.parametrize("errors", [[-3, 0, 4], [0, 0], [1e300, -1e300], [1e-300, -1e-300]])
def test_equal_weight_metrics_and_magnitude_wrapper_parity(errors):
    result = error_metrics(iter(errors))
    expected = {"count": len(errors), "rmse": math.hypot(*errors) / math.sqrt(len(errors)),
                "mae": sum(abs(x) / len(errors) for x in errors)}
    assert result == pytest.approx(expected)
    assert result == magnitude_metrics([{"predicted_absolute_move_pct": max(e, 0),
                                         "absolute_move_pct": max(-e, 0)} for e in errors])


@pytest.mark.parametrize("errors", [[], [True], [None], ["1"], [float("nan")], [float("inf")], [10**1000]])
def test_bad_residuals_refuse_without_metrics(errors):
    with pytest.raises(ValueError):
        error_metrics(errors)


def test_numpy_residuals_do_not_overflow_in_native_dtype():
    result = error_metrics([np.int64(-(2**63)), np.int64(0)])
    assert result == pytest.approx({"count": 2, "rmse": 2**63 / math.sqrt(2), "mae": 2**62})
    assert error_metrics([np.float32(-3), np.float32(4)]) == pytest.approx(
        {"count": 2, "rmse": math.sqrt(12.5), "mae": 3.5})


def test_magnitude_failure_envelope_preserved():
    for values, message in (([], "require scored rows"),
                           ([{"predicted_absolute_move_pct": float("inf"), "absolute_move_pct": 0}], "must be finite")):
        with pytest.raises(OpsError, match=message) as caught:
            magnitude_metrics(values)
        assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
