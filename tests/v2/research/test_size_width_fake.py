import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from experiments import size_width_errmodel as errmodel
from experiments import size_width_fake as fake
from experiments import size_width_tuning as tuning
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sizing import GRID
from tests.v2.research.test_size_pnl_shortvol import chunk_of, events_for
from tests.v2.research.test_size_width_sweep import _small_window

DRAW = 200
PRIOR = [f"E{i}" for i in range(6)]
CUR = [f"E{i}" for i in range(6, 12)]
MODEL = {"L": 1.2, "b": 0.7, "s": 0.3, "mean_log_y": 1.0}


def _recipe_z(event_id, d):
    """The documented seed recipe, written out again here rather than imported."""
    seed = int.from_bytes(hashlib.sha256(f"{event_id}|exp189-v1|{d}".encode()).digest()[:8], "big")
    return np.random.default_rng(seed).standard_normal()


def _world(n=4000, seed=11):
    """Synthetic M0: forecasts drawn from the same log-error process the fake uses, so the fitted model is calibrated."""
    rng = np.random.default_rng(seed)
    index = pd.Index([f"E{i}" for i in range(n)], name="event_id")
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    y = np.exp(rng.normal(np.log(4.0), 0.3, n))
    truth = {"L": float(np.log(4.0)), "b": 0.8, "s": 0.25, "mean_log_y": float(np.mean(np.log(y)))}
    pred = errmodel.fake_forecast(truth, y, rng.standard_normal(n))
    table = pd.DataFrame({"event_date": dates, "pred_move": pred, "pred_crush": rng.normal(-15, 3, n)}, index=index)
    labels = pd.DataFrame({"abs_move": y, "crush_pct_iv30": table["pred_crush"].to_numpy() + rng.normal(0, 5, n)},
                          index=index)
    return table, labels, errmodel.fit(pred, y)


def _months():
    chunk = chunk_of(*[(e, {}) for e in PRIOR + CUR])
    prior = events_for(*PRIOR, date="2024-05-10")
    cur = events_for(*CUR, date="2024-06-10")
    sessions = {e: (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-25")) for e in PRIOR}
    sessions.update({e: (pd.Timestamp("2024-06-11"), pd.Timestamp("2024-06-21")) for e in CUR})
    return [(chunk, prior, False), (chunk, cur, True)], sessions


def _per_draw(peaks=None, trades=None):
    """One draw: every structure at return 0.01 on every grid k, except ``peaks`` ({name: {k: ret}}); 150 trades each unless
    ``trades`` maps a name to another count."""
    peaks, trades = peaks or {}, trades or {}
    return {name: {k: {"ret": peaks.get(name, {}).get(k, 0.01), "se": 0.0, "trades": trades.get(name, 150)}
                   for k in GRID} for name in NAMES}


# ---- seeds ----------------------------------------------------------------

def test_draw_z_reproduces_and_follows_the_documented_recipe():
    ids = ["E0", "E1", "E2"]
    first = fake.draw_z(ids, 0)
    np.testing.assert_array_equal(first, fake.draw_z(ids, 0))
    np.testing.assert_array_equal(first, [_recipe_z(e, 0) for e in ids])
    assert fake.draw_seed("E7", 3) == int.from_bytes(hashlib.sha256(b"E7|exp189-v1|3").digest()[:8], "big")


def test_z_differs_between_draws_and_events_and_not_with_other_events():
    a = fake.draw_z(["E0", "E1"], 0)
    assert not np.array_equal(a, fake.draw_z(["E0", "E1"], 1))
    assert a[0] != a[1]
    assert fake.draw_z(["E1"], 0)[0] == a[1]
    assert fake.draw_z(["E1", "E0", "E9"], 0)[0] == a[1]
    assert fake.z_matrix(["E0", "E1", "E2"], n=4).shape == (4, 3)


# ---- fake forecasts -------------------------------------------------------

def test_fake_pred_is_the_model_formula_with_nan_for_bad_labels_and_str_index():
    ids = ["E0", "E1", "E2", "E9"]
    labels = pd.DataFrame({"abs_move": [2.5, 0.0, -1.0]}, index=["E0", "E1", "E2"])
    got = fake.fake_pred(MODEL, ids, labels, 2)
    z = _recipe_z("E0", 2)
    expected = np.exp(MODEL["L"] + MODEL["b"] * (np.log(2.5) - MODEL["mean_log_y"]) + MODEL["s"] * z)
    assert got.index.tolist() == ids and all(isinstance(i, str) for i in got.index)
    assert got.iloc[0] == pytest.approx(expected)
    assert got.iloc[1:].isna().all()


# ---- tuning C -------------------------------------------------------------

def test_tune_c_pools_the_draws_by_hand_and_takes_the_argmax():
    per_draw = [_per_draw({"CND-PS": {1.5: r}, "TWIN-P": {1.0: 0.12}}) for r in (0.08, 0.10, 0.12)]
    pooled = fake.pool_draws(per_draw)
    assert pooled["CND-PS"][1.5]["ret"] == pytest.approx(0.10)
    assert pooled["CND-PS"][1.5]["se"] == pytest.approx(0.02 / math.sqrt(3))
    assert pooled["CND-PS"][1.5]["trades"] == 450
    body = fake.tune_c(per_draw)
    assert body["CND-PS"]["k"] == 1.5 and body["CND-PS"]["flags"] == []
    assert body["TWIN-P"]["k"] == 1.0 and body["TWIN-P"]["flags"] == []
    assert body["CND-PS"]["members"] == ["CND-PS"]
    assert body["CND-PS"]["scores"]["1.5"]["not_scored"] is False


def test_pool_draws_single_draw_has_zero_se():
    assert fake.pool_draws([_per_draw()])["CND-PS"][1.0]["se"] == 0.0


def test_thin_structure_is_k_defaulted_and_not_scored():
    per_draw = [_per_draw({"BFLY-P": {2.0: 0.5}}, trades={"BFLY-P": 50}) for _ in range(3)]
    body = fake.tune_c(per_draw)
    assert body["BFLY-P"]["k"] == 1.0 and body["BFLY-P"]["flags"] == ["k_defaulted"]
    assert all(s["not_scored"] for s in body["BFLY-P"]["scores"].values())


def test_argmax_at_the_top_edge_is_k_at_edge_with_an_interior_k():
    per_draw = [_per_draw({"RAMP7": {3.0: 0.20, 1.25: 0.15}}) for _ in range(3)]
    body = fake.tune_c(per_draw)
    assert body["RAMP7"]["k"] == 1.25 and body["RAMP7"]["flags"] == ["k_at_edge"]


# ---- end to end -----------------------------------------------------------

def test_run_fake_writes_tuning_c_that_frozen_reads_and_is_reproducible(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    res = fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert res["status"] == "ok"
    assert (tmp_path / "receipts" / "tuning_C.json").exists()
    factors = tuning.frozen(tmp_path, "C", "hash-1")
    assert set(factors) == set(NAMES) and all(v in GRID for v in factors.values())
    with pytest.raises(Refusal) as caught:
        tuning.frozen(tmp_path, "C", "hash-2")
    assert caught.value.details["reason"] == "tuning_receipt_changed"
    again = fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert again["body"] == res["body"]


def test_uncalibrated_fake_writes_no_receipt_and_runs_no_draws(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    monkeypatch.setattr(fake, "calibration", lambda *a, **k: {"ok": False, "reason": "fake_not_calibrated"})
    monkeypatch.setattr(fake, "run_draw", lambda *a, **k: pytest.fail("no draws when not calibrated"))
    res = fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert res["status"] == "fake_not_calibrated"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()
    with pytest.raises(Refusal) as caught:
        tuning.frozen(tmp_path, "C", "hash-1")
    assert caught.value.details["reason"] == "tuning_receipt_missing"


def test_draws_differ_and_one_draw_is_reproducible(monkeypatch):
    _small_window(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    d0 = fake.run_draw(months, table, labels, model, 0, sessions, draws=DRAW)
    assert d0 == fake.run_draw(months, table, labels, model, 0, sessions, draws=DRAW)
    assert d0 != fake.run_draw(months, table, labels, model, 1, sessions, draws=DRAW)


def test_anchor_scores_every_structure_and_k_and_t4_marks_thin_cells(monkeypatch):
    _small_window(monkeypatch)
    table, labels, _ = _world()
    months, sessions = _months()
    res = fake.run_anchor(months, table, labels, sessions, draws=DRAW)
    assert set(res) == set(NAMES)
    for name in NAMES:
        assert set(res[name]) == set(GRID)
    assert set(res["CND-PS"][1.0]) == {"ret", "se", "trades"}
    md = fake.t4_markdown(res)
    assert "### T4 anchor (perfect forecast, 2023-24, descriptive only)" in md
    assert all(name in md for name in NAMES)
    assert "n/a" in md


# ---- arm C shape in the tuning module -------------------------------------

def test_units_c_is_b_and_valid_checks_the_arm_c_shape():
    assert tuning.units("C") == tuning.units("B")
    body = {n: {"k": 1.0, "flags": [], "members": [n], "scores": {}} for n in NAMES}
    receipt = {"experiment": "EXP-189", "arm": "C", "window": "2023H2-2024", "body": body}
    assert tuning._valid("C", receipt)
    missing = {n: v for n, v in body.items() if n != NAMES[0]}
    assert not tuning._valid("C", {**receipt, "body": missing})
