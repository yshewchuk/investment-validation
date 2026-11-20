"""EXP-188 SY2: hand-built checks of the synthetic evaluation (I1, X1-X7, slopes, stop ids, determinism)."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_synthetic_eval as ev

EVENTS = [f"E{i}" for i in range(8)]
DATES = pd.to_datetime(["2024-01-15", "2024-01-20", "2024-04-15", "2024-04-20",
                        "2024-07-15", "2024-07-20", "2024-10-15", "2024-10-20"])
TICKERS = {e: ("T1" if i % 2 == 0 else "T2") for i, e in enumerate(EVENTS)}
ARM = {"spread_ladder": [0.2, 0.6], "level_gap_min_s": 0.4, "inner_years": [2023, 2024]}
RESAMPLES = 200
META = {
    "O": {"family": "oracle", "params": {}, "rmse": 0.5, "mae": 0.3},
    "M02": {"family": "median_noise", "params": {"s": 0.2}, "rmse": 1.2, "mae": 0.6},
    "M06": {"family": "median_noise", "params": {"s": 0.6}, "rmse": 1.8, "mae": 0.9},
    "N02": {"family": "mean_noise", "params": {"s": 0.2}, "rmse": 1.2, "mae": 0.6},
    "N06": {"family": "mean_noise", "params": {"s": 0.6}, "rmse": 1.8, "mae": 0.9},
    "SH": {"family": "shrinkage", "params": {"L": 1.0}, "rmse": 2.0, "mae": 1.0},
    "U1": {"family": "level_shift", "params": {"shift": 0.1}, "rmse": 1.0, "mae": 0.5},
    "U2": {"family": "level_shift", "params": {"shift": -0.1}, "rmse": 1.0, "mae": 0.5},
}
DEFAULT = {"O": 3.0, "M02": 2.0, "M06": 2.0, "N02": 2.0, "N06": 1.0, "SH": 1.5, "U1": 1.0, "U2": 1.0}


def frame_of(over=None, *, width=None, pred=None, status="taken"):
    """Decision rows for every model on short-vol units S1, S2 and straddle unit R_L; P&L per event from ``over``."""
    pnl, frames = {**DEFAULT, **(over or {})}, []
    w = np.ones(8) if width is None else np.asarray(width, dtype=float)
    p = np.full(8, 0.05) if pred is None else np.asarray(pred, dtype=float)  # constant overall, so constant per quarter
    for mid, value in pnl.items():
        for book, unit in (("short_vol", "S1"), ("short_vol", "S2"), ("straddle", "R_L")):
            short = book == "short_vol"
            frames.append(pd.DataFrame({
                "model_id": mid, "book": book, "arm_or_rule": "as_is" if short else unit,
                "structure": unit if short else None, "event_id": EVENTS, "event_date": DATES,
                "status": status, "pnl_dollars": np.full(8, float(value)), "width": w, "pred_move": p}))
    return pd.concat(frames, ignore_index=True)


def run(over=None, *, width=None, pred=None, meta=None, replay_equal=None):
    return ev.evaluate(frame_of(over, width=width, pred=pred), META if meta is None else meta, TICKERS, ARM,
                       replay_equal=replay_equal, resamples=RESAMPLES)


def row(out, id_, scope):
    return next(r for r in out["x"] if r[0] == id_ and r[1] == scope)


def test_i1_fires_on_identical_pnl_and_passes_otherwise():
    assert row(run({k: 1.0 for k in DEFAULT}), "I1", "S1")[2] == "fail"
    assert row(run(), "I1", "S1")[2] == "pass"


def test_x1_passes_with_a_dominant_oracle():
    assert row(run(), "X1", "S1")[2:] == ["pass", "failing=0 of 7"]


def test_x1_fails_when_another_predictor_beats_the_oracle():
    out = run({"SH": 5.0})
    assert row(out, "X1", "S1")[2] == "fail"
    assert "X1 S1" in out["stop"]


def test_x2_clear_directional_and_failed():
    assert row(run(), "X2", "median_noise/S1")[2] == "Clear"
    directional = run({"M02": 1.0, "M06": 2.0}, meta={**META, "M06": {**META["M06"], "rmse": 1.0}})
    assert row(directional, "X2", "median_noise/S1")[2] == "Directional"
    failed = run({"M02": 1.0, "M06": 2.0})
    assert row(failed, "X2", "median_noise/S1")[2] == "Failed"
    assert "X2 median_noise/S1" in failed["stop"]


def test_x3_classifies_the_median_minus_mean_contrast():
    out = run()
    assert row(out, "X3", "s=0.6")[2] == "clear"
    assert not [r for r in out["x"] if r[0] == "X3" and r[1] == "s=0.2"]  # below level_gap_min_s


def test_x4_with_one_straddle_unit_is_not_testable():
    assert row(run(), "X4", "s=0.6")[2] == "not_testable"


def test_x5_agrees_disagrees_and_zero_only_agrees_with_zero():
    assert row(run({"U1": 2.0, "U2": 1.0}), "X5", "S1")[2] == "pass"
    assert row(run({"U1": 1.0, "U2": 2.0}), "X5", "S1")[2] == "fail"
    assert row(run({"U1": 1.0, "U2": 1.0, "M06": 1.0, "N06": 1.0}), "X5", "S1")[2] == "pass"
    assert row(run(), "X5", "S1")[2] == "fail"  # zero shift gap against a positive level gap


def test_x6_passes_with_constant_width_within_each_quarter():
    out = run(width=[1, 1, 2, 2, 3, 3, 4, 4])
    assert row(out, "X6", "S1")[2] == "pass"
    assert row(out, "X6", "R_L")[2:] == ["pass", "pred_var=0 taken=8"]


def test_x6_straddle_passes_with_pred_move_constant_within_each_quarter():
    out = run(pred=[0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4])
    assert row(out, "X6", "R_L")[2:] == ["pass", "pred_var=0 taken=8"]
    assert "X6 R_L" not in out["stop"]


def test_x6_straddle_fails_on_pred_move_variance_within_a_quarter_and_stops():
    out = run(pred=[0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4])
    assert row(out, "X6", "R_L")[2] == "fail"
    assert "X6 R_L" in out["stop"]


def test_x6_straddle_fails_with_a_nan_pred_move_before_the_variance_test():
    out = run(pred=[0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, np.nan])
    assert row(out, "X6", "R_L")[2:] == ["fail", "missing_pred=1 taken=8"]
    assert "X6 R_L" in out["stop"]


def test_x6_fails_on_within_quarter_width_variance():
    out = run(width=[1, 2, 2, 2, 3, 3, 4, 4])
    assert row(out, "X6", "S1")[2] == "fail"
    assert "X6 S1" in out["stop"]


def test_x6_passes_with_complete_constant_widths():
    assert row(run(width=[2] * 8), "X6", "S1")[2:] == ["pass", "var=0 taken=8"]


def test_x6_fails_when_every_width_is_missing():
    out = run(width=[np.nan] * 8)
    assert row(out, "X6", "S1")[2:] == ["fail", "missing_widths=8 taken=8"]
    assert "X6 S1" in out["stop"]


def test_x6_fails_on_one_missing_width_before_the_variance_test():
    out = run(width=[1, 1, 2, 2, 3, 3, 4, np.nan])
    assert row(out, "X6", "S1")[2:] == ["fail", "missing_widths=1 taken=8"]
    assert "X6 S1" in out["stop"]


def test_x7_values_and_stop_membership():
    assert row(run(replay_equal=True), "X7", "replay")[2] == "pass"
    assert row(run(replay_equal=None), "X7", "replay")[2] == "not_run"
    failed = run(replay_equal=False)
    assert row(failed, "X7", "replay")[2] == "fail"
    assert "X7 replay" in failed["stop"]


def test_stop_holds_gating_rows_only():
    out = run(replay_equal=False)
    assert out["stop"] == ["X7 replay"]  # X5 fails here but never stops the run
    assert row(out, "X5", "S1")[2] == "fail"


def test_identical_pnl_fails_i1_and_x1_and_only_x_ids_stop():
    out = run({k: 1.0 for k in DEFAULT})
    assert row(out, "I1", "S1")[2] == "fail" and row(out, "X1", "S1")[2] == "fail"
    assert "X1 S1" in out["stop"] and "I1 S1" not in out["stop"]
    assert all(s.startswith("X") for s in out["stop"])


def test_slope_matches_polyfit_and_its_interval_brackets_it():
    meta = {**META, "M02": {**META["M02"], "rmse": 1.0}, "M06": {**META["M06"], "rmse": 2.0}}
    out = run({"M02": 1.0, "M06": 3.0}, meta=meta)
    slope = next(r for r in out["slopes"] if r[:3] == ["median_noise", "S1", "rmse"])
    assert slope[3] == pytest.approx(np.polyfit([1.0, 2.0], [1.0, 3.0], 1)[0])
    assert slope[3] == pytest.approx(2.0)
    assert slope[4] - 1e-9 <= slope[3] <= slope[5] + 1e-9


def test_evaluation_is_deterministic_and_json_safe():
    first, second = run(), run()
    assert first["hash"] == second["hash"]
    assert first["x"] == second["x"] and first["slopes"] == second["slopes"]
    json.dumps(first)
    moved = frame_of()
    at = (moved["model_id"] == "M06") & (moved["event_id"] == "E3") & (moved["structure"] == "S1")
    moved.loc[at, "pnl_dollars"] += 0.5  # one P&L cell
    assert ev.evaluate(moved, META, TICKERS, ARM, resamples=RESAMPLES)["hash"] != first["hash"]


def _s1(frame):
    return frame[(frame["book"] == "short_vol") & (frame["structure"] == "S1")]


def test_a_missing_cell_raises_and_a_nan_pnl_counts_zero():
    d = frame_of()
    at = (d["model_id"] == "M06") & (d["event_id"] == "E3") & (d["structure"] == "S1")
    with pytest.raises(ValueError):
        ev.evaluate(d[~at], META, TICKERS, ARM, resamples=RESAMPLES)
    nan, zero = d.copy(), d.copy()
    nan.loc[at, "pnl_dollars"] = np.nan
    zero.loc[at, "pnl_dollars"] = 0.0
    pd.testing.assert_series_equal(ev._wide(_s1(nan)).mean(), ev._wide(_s1(zero)).mean())  # the NaN cell is 0
    assert ev.evaluate(nan, META, TICKERS, ARM, resamples=RESAMPLES)["hash"] == ev.evaluate(zero, META, TICKERS, ARM, resamples=RESAMPLES)["hash"]


def test_population_drops_an_event_undetermined_under_any_model_or_unit():
    d = frame_of()
    at = (d["model_id"] == "M02") & (d["event_id"] == "E0") & (d["structure"] == "S1")
    d.loc[at, "status"] = "undetermined"
    assert len(ev.population(d)) == len(d) - 8 * 3  # E0's rows under all 8 models and 3 units leave E
    assert "E0" not in set(ev.population(d)["event_id"])
    assert ev.evaluate(d, META, TICKERS, ARM, resamples=RESAMPLES)["hash"] == ev.evaluate(d[d["event_id"] != "E0"], META, TICKERS, ARM, resamples=RESAMPLES)["hash"]


def test_means_and_contrast_rows_hand_checked():
    out = run()
    assert len(out["means"]) == 8 * 3 and all(len(r) == 10 for r in out["means"])
    # M06 on S1: 8 taken events at a constant 2.0 per event; rmse and mae from META
    assert ["M06", "median_noise", '{"s": 0.6}', "S1", 8, 8, 0, 2.0, 1.8, 0.9] in out["means"]
    # median M06 (2.0) minus mean N06 (1.0) is +1 on every event, so the mean is 1.0 and every quarter (4 of 4) is positive;
    # a constant series gives degenerate bootstrap intervals at 1.0
    assert ["S1", "s=0.6", 1.0, "4/0/0", 1.0, 1.0, 1.0, 1.0] in out["contrasts"]
    assert all(len(r) == 8 for r in out["contrasts"])
