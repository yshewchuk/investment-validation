from __future__ import annotations

import numpy as np
import pytest

from engine.v2.ops.errors import OpsError
from experiments.size_width_errmodel import (
    MIN_EVENTS,
    calibrate,
    fake_forecast,
    fit,
    paired,
    t1_markdown,
)


def refusal(callable_):
    try:
        callable_()
    except OpsError as exc:
        assert exc.code == "EXPERIMENT_IDENTITY_CONFLICT"
        return exc.problem.details["reason"]
    raise AssertionError("expected a refusal")


def _world(n=4000):
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(1.0, 0.5, n))
    truth = {"L": 0.3, "b": 0.7, "s": 0.2, "mean_log_y": float(np.mean(np.log(y)))}
    f = fake_forecast(truth, y, rng.normal(size=n))
    model = fit(f, y)
    z = rng.normal(size=(20, n))
    return model, f, y, z


def _calibrations():
    model, f, y, z = _world()
    noisy = dict(model)
    noisy["s"] = 3 * model["s"]
    return model, calibrate(model, f, y, z), calibrate(noisy, f, y, z)


def test_paired_drops_nonpositive_and_counts_each_row_once():
    forecast = [1, 0, -1, np.nan, np.inf, 2, 3, 4, -1, 2]
    label = [1, 1, 1, 1, 1, 0, -1, 2, -1, np.nan]
    mask, dropped = paired(forecast, label)
    assert np.array_equal(mask, [True, False, False, False, False, False, False, True, False, False])
    assert dropped == {"nonpositive_forecast": 5, "nonpositive_label": 3}


def test_paired_rejects_length_mismatch():
    with pytest.raises(ValueError):
        paired([1.0, 2.0], [1.0])


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(0)
    n = 2000
    y = np.exp(rng.normal(1.0, 0.5, n))
    cy = np.log(y) - np.mean(np.log(y))
    f = np.exp(0.3 + 0.7 * cy + 0.2 * rng.normal(size=n))
    model = fit(f, y)
    assert abs(model["L"] - 0.3) < 0.02
    assert abs(model["b"] - 0.7) < 0.03
    assert abs(model["s"] - 0.2) < 0.02
    assert model["n"] == n
    assert model["dropped"] == {"nonpositive_forecast": 0, "nonpositive_label": 0}


def test_fit_counts_drops():
    rng = np.random.default_rng(1)
    n = 2000
    y = np.exp(rng.normal(1.0, 0.5, n))
    f = np.exp(0.3 + 0.7 * (np.log(y) - np.mean(np.log(y))) + 0.2 * rng.normal(size=n))
    f = np.concatenate([f, [0.0, -1.0, np.nan, -1.0], [1.0, 2.0]])
    y = np.concatenate([y, [1.0, 1.0, 1.0, -1.0], [0.0, -3.0]])
    model = fit(f, y)
    assert model["n"] == n
    assert model["dropped"] == {"nonpositive_forecast": 4, "nonpositive_label": 2}


def test_fit_refuses_insufficient_inputs():
    rng = np.random.default_rng(2)
    y = np.exp(rng.normal(1.0, 0.5, 299))
    f = np.exp(rng.normal(1.0, 0.2, 299))
    assert refusal(lambda: fit(f, y)) == "m0_inputs_insufficient"
    assert MIN_EVENTS == 300
    y_flat = np.full(400, 2.0)
    f_flat = np.exp(rng.normal(1.0, 0.2, 400))
    assert refusal(lambda: fit(f_flat, y_flat)) == "m0_inputs_insufficient"


def test_fake_forecast_positive_shape_preserving_and_exact_without_noise():
    rng = np.random.default_rng(3)
    y = np.exp(rng.normal(size=50))
    model = {"L": 0.3, "b": 0.7, "s": 0.2, "mean_log_y": 0.2}
    out = fake_forecast(model, y, rng.normal(size=50))
    assert out.shape == (50,)
    assert np.all(out > 0)
    model["s"] = 0.0
    exact = fake_forecast(model, y, rng.normal(size=50))
    assert np.allclose(exact, np.exp(0.3 + 0.7 * (np.log(y) - 0.2)))
    assert fake_forecast(model, y, rng.normal(size=(5, 50))).shape == (5, 50)


def test_calibrate_passes_when_forecasts_come_from_same_model():
    model, f, y, z = _world()
    cal = calibrate(model, f, y, z)
    assert cal["ok"] is True
    assert cal["reason"] is None
    assert abs(cal["rmse_rel"]) <= 0.05
    assert abs(cal["mae_rel"]) <= 0.05


def test_calibrate_fails_when_fake_is_noisier():
    model, f, y, z = _world()
    noisy = dict(model)
    noisy["s"] = 3 * model["s"]
    cal = calibrate(noisy, f, y, z)
    assert cal["ok"] is False
    assert cal["reason"] == "fake_not_calibrated"


def test_calibrate_ignores_dropped_rows_and_rejects_bad_z():
    model, f, y, z = _world()
    rng = np.random.default_rng(4)
    base = calibrate(model, f, y, z)
    f2 = np.concatenate([f, [0.0, 2.0, -1.0]])
    y2 = np.concatenate([y, [1.0, 0.0, 3.0]])
    z2 = np.concatenate([z, rng.normal(size=(20, 3))], axis=1)
    with_drops = calibrate(model, f2, y2, z2)
    for key in ("rmse_m0", "rmse_fake", "mae_m0", "mae_fake", "rmse_rel", "mae_rel"):
        assert with_drops[key] == pytest.approx(base[key])
    assert with_drops["ok"] == base["ok"]
    bad_nan = z.copy()
    bad_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        calibrate(model, f, y, bad_nan)
    with pytest.raises(ValueError):
        calibrate(model, f, y, z[:, :-1])
    with pytest.raises(ValueError):
        calibrate(model, f, y, z[0])


def test_t1_markdown_reports_both_calibration_outcomes():
    model, cal_pass, cal_fail = _calibrations()
    md = t1_markdown(model, cal_pass)
    assert "### T1 error model (M0, 2023-24 inner)" in md
    assert "| L | b | s | n |" in md
    assert "Dropped: nonpositive_forecast=0, nonpositive_label=0" in md
    assert "Calibration: PASS" in md
    assert md.endswith("\n")
    md_fail = t1_markdown(model, cal_fail)
    assert "Calibration: FAIL (fake_not_calibrated)" in md_fail
