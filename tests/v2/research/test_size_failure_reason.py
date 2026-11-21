from engine.v2.ops.errors import fail
from engine.v2.research.horizon_fitted import _validate_forecasts
from experiments.magnitude_failure_reason import classify
from experiments.magnitude_search_outer import _timed_shape
from experiments.size_search_report import failure_counts


def test_classify_convergence_from_plain_value_error():
    assert classify(ValueError("ElasticNet convergence failed; no retry permitted")) == "nonconvergence"


def test_classify_convergence_from_ops_error():
    error = fail("EXPERIMENT_VARIANT_FAILED", "ElasticNet convergence failed; no retry permitted")
    assert classify(error) == "nonconvergence"


def test_classify_forecast_validation_failure():
    assert classify(ValueError("forecast validation failed")) == "invalid_prediction"


def test_classify_never_returns_caller_text():
    result = classify(ValueError("secret caller text"))
    assert result == "value_error"
    assert "secret" not in result


def test_classify_arithmetic_and_other():
    assert classify(ZeroDivisionError()) == "arithmetic_error"
    assert classify(RuntimeError("boom")) == "other"


def test_validate_forecasts_rejects_nan_with_policy_message():
    try:
        _validate_forecasts([float("nan")], 1, "nonnegative")
    except ValueError as error:
        assert str(error) == "forecast validation failed"
    else:
        raise AssertionError("NaN forecast was accepted")


def test_timed_shape_accepts_failed_reason_when_untimed():
    assert _timed_shape({"event": "failed", "failure_reason": "nonconvergence"}, {"event"}, untimed=True)


def test_timed_shape_rejects_unregistered_reason():
    assert not _timed_shape({"event": "failed", "failure_reason": "bogus"}, {"event"}, untimed=True)


def test_timed_shape_rejects_reason_on_succeeded_event():
    result = {"event": "succeeded", "forecasts": [], "failure_reason": "nonconvergence"}
    assert not _timed_shape(result, {"event", "forecasts"}, untimed=True)


def test_failure_counts_groups_failed_rows_and_marks_legacy_unrecorded():
    rows = [
        {"status": "failed", "failure_reason": "nonconvergence"},
        {"status": "failed", "failure_reason": "nonconvergence"},
        {"status": "failed", "failure_reason": None},
        {"status": "succeeded", "failure_reason": None},
    ]
    assert failure_counts(rows) == {"nonconvergence": 2, "unrecorded": 1}
