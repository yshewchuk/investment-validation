import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from engine.pnl_sim import DRAWS
from experiments import size_pnl_shortvol as sv
from experiments import size_width_account as account
from experiments import size_width_errmodel as errmodel
from experiments import size_pnl_receipts as receipts
from experiments import size_width_fake as fake
from experiments import size_width_tuning as tuning
from experiments.size_pnl_pool import build_pool, pool_history
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sizing import GRID
from tests.v2.research.test_size_pnl_shortvol import chunk_of, events_for
from tests.v2.research.test_size_width_sweep import _small_window

DRAW = 200
PRIOR = [f"E{i}" for i in range(6)]
CUR = [f"E{i}" for i in range(6, 12)]
MODEL = {"L": 1.2, "b": 0.7, "s": 0.3, "mean_log_y": 1.0}
C_MODEL = {**MODEL, "n": 4000}  # a fake model a C receipt may carry: n is the fit's event count, at least MIN_EVENTS
ORACLE_CASES = [("CND-PS", 0.75), ("CND-PS", 1.0), ("CND-PS", 1.5), ("TWIN-P", 1.5)]


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
    available = (dates + pd.Timedelta(days=1)).strftime("%Y-%m-%d").to_numpy()
    labels = pd.DataFrame({"abs_move": y, "crush_pct_iv30": table["pred_crush"].to_numpy() + rng.normal(0, 5, n),
                           "available": available}, index=index)
    return table, labels, errmodel.fit(pred, y)


def _months():
    chunk = chunk_of(*[(e, {}) for e in PRIOR + CUR])
    prior = events_for(*PRIOR, date="2024-05-10")
    cur = events_for(*CUR, date="2024-06-10")
    sessions = {e: (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-25")) for e in PRIOR}
    sessions.update({e: (pd.Timestamp("2024-06-11"), pd.Timestamp("2024-06-21")) for e in CUR})
    return [(chunk, prior, False), (chunk, cur, True)], sessions


def _small_draws(monkeypatch):
    """Protocol N is tuning.FAKE_DRAWS (20); the tests that run a 2-draw stage patch the constant to 2."""
    monkeypatch.setattr(tuning, "FAKE_DRAWS", 2)


@pytest.fixture(autouse=True)
def _small_sim_draws(monkeypatch):
    """The small runs simulate DRAW draws, not the protocol DRAWS (4000), so the protocol simulation count is patched to DRAW here."""
    monkeypatch.setattr(tuning, "FAKE_SIM_DRAWS", DRAW)


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


def test_z_matrix_holds_the_recipe_value_of_each_draw_and_event():
    ids = ["E0", "E1", "E2"]
    z = fake.z_matrix(ids, n=4)
    assert z.shape == (4, 3)
    for d in range(4):
        for j, e in enumerate(ids):
            assert z[d, j] == _recipe_z(e, d)


def test_z_matrix_accepts_an_iterator_and_matches_the_list():
    ids = ["E0", "E1", "E2"]
    np.testing.assert_array_equal(fake.z_matrix(iter(ids), 3), fake.z_matrix(ids, 3))


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


def test_fake_pred_accepts_an_iterator_and_matches_the_list():
    ids = ["E0", "E1", "E2", "E9"]
    labels = pd.DataFrame({"abs_move": [2.5, 0.0, -1.0]}, index=["E0", "E1", "E2"])
    pd.testing.assert_series_equal(fake.fake_pred(MODEL, iter(ids), labels, 2), fake.fake_pred(MODEL, ids, labels, 2))


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


def test_choose_scores_exactly_300_trades_and_not_299():
    def score(ret, trades):
        return {"ret": ret, "se": 0.0, "trades": trades}
    assert tuning.choose({1.5: score(0.2, 300)}) == {"k": 1.5, "flags": []}
    assert tuning.choose({1.5: score(0.2, 299)}) == {"k": 1.0, "flags": ["k_defaulted"]}


def test_tune_c_min_trades_boundary_at_300_pooled_trades():
    at_300 = [_per_draw({"CND-PS": {1.5: 0.2}}, trades={"CND-PS": t}) for t in (100, 100, 100)]
    body = fake.tune_c(at_300)
    assert body["CND-PS"]["scores"]["1.5"]["trades"] == 300
    assert body["CND-PS"]["scores"]["1.5"]["not_scored"] is False
    assert body["CND-PS"]["k"] == 1.5 and body["CND-PS"]["flags"] == []

    at_299 = [_per_draw({"CND-PS": {1.5: 0.2}}, trades={"CND-PS": t}) for t in (100, 100, 99)]
    body = fake.tune_c(at_299)
    assert body["CND-PS"]["scores"]["1.5"]["trades"] == 299
    assert body["CND-PS"]["scores"]["1.5"]["not_scored"] is True
    assert body["CND-PS"]["k"] == 1.0 and body["CND-PS"]["flags"] == ["k_defaulted"]


# ---- end to end -----------------------------------------------------------

def test_run_fake_writes_tuning_c_that_frozen_reads_and_is_reproducible(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    _small_draws(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    res = fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert res["status"] == "ok"
    assert (tmp_path / "receipts" / "tuning_C.json").exists()
    factors = tuning.frozen(tmp_path, "C", "hash-1")
    assert set(factors) == set(NAMES) and all(v in GRID for v in factors.values())
    with pytest.raises(Refusal) as caught:
        tuning.frozen(tmp_path, "C", "hash-2")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "tuning_receipt_changed"
    again = fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert again["body"] == res["body"]
    receipt = json.loads((tmp_path / "receipts" / "tuning_C.json").read_text(encoding="utf-8"))
    meta = receipt["fake"]
    assert meta["N"] == 2 and meta["seed_tag"] == "exp189-v1"
    assert meta["model"]["L"] == pytest.approx(model["L"])
    assert meta["calibration"]["ok"] is True
    assert receipt["arm"] == "C" and receipt["inputs_hash"] == "hash-1" and receipt["window"] == "2023H2-2024"
    assert receipt["body"] == json.loads(json.dumps(res["body"]))
    for key in ("rmse_rel", "mae_rel"):
        assert meta["calibration"][key] == pytest.approx(res["calibration"][key], abs=0.05)


def test_run_fake_and_run_draw_accept_generators_and_match_the_list(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    _small_draws(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    as_list = fake.run_fake(months, table, labels, model, sessions, tmp_path / "list", "hash-1", n=2, draws=DRAW)
    as_gen = fake.run_fake(
        (m for m in months), table, labels, model, sessions, tmp_path / "gen", "hash-1", n=2, draws=DRAW)
    assert as_gen["body"] == as_list["body"]
    assert as_list["pooled"]["CND-PS"][1.0]["trades"] > 0
    assert as_gen["pooled"]["CND-PS"][1.0]["trades"] == as_list["pooled"]["CND-PS"][1.0]["trades"]

    one = fake.run_draw(months, table, labels, model, 0, sessions, draws=DRAW, names=NAMES)
    assert fake.run_draw(months, table, labels, model, 0, sessions, draws=DRAW, names=iter(NAMES)) == one


def test_real_calibration_failure_omits_c_and_writes_no_receipt(tmp_path, monkeypatch):
    _small_draws(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    res = fake.run_fake(months, table, labels, {**model, "s": 3.0}, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert res["status"] == "fake_not_calibrated"
    assert res["calibration"]["reason"] == "fake_not_calibrated" and res["calibration"]["ok"] is False
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


def test_uncalibrated_fake_writes_no_receipt_and_runs_no_draws(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    _small_draws(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    monkeypatch.setattr(fake, "calibration", lambda *a, **k: {"ok": False, "reason": "fake_not_calibrated"})
    monkeypatch.setattr(fake, "run_draw", lambda *a, **k: pytest.fail("no draws when not calibrated"))
    res = fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert res["status"] == "fake_not_calibrated"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()
    with pytest.raises(Refusal) as caught:
        tuning.frozen(tmp_path, "C", "hash-1")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
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


def _direct(months, table, labels, pred, sessions, name="CND-PS", k=1.0):
    """(trades, total_return) of the last month's taken set, by the module functions directly: no fake._run, no Sweep.
    Each month's pool is built by hand: rows with event_date before the month start, labels whose ``available`` string is
    before the month-start ISO string. Follows Sweep's scaled-events route at a fixed k: each month is simulated once on the
    unscaled forecast events, scored on the events with pred_move x k (NaN stays NaN), then decided on the whole history
    and priced at mid marks."""
    history, chunk_last, scaled_last = [], None, None
    for chunk, events, _output in months:
        start = pd.to_datetime(events["event_date"]).dt.to_period("M").min().start_time
        rows_before = table[pd.to_datetime(table["event_date"]) < start]
        labels_before = labels[labels["available"] < start.date().isoformat()]
        pool = build_pool(rows_before.assign(pred_move=rows_before.index.astype(str).map(pred)), labels_before)
        forecast = events.assign(pred_move=events["event_id"].astype(str).map(pred))
        scaled_events = forecast.assign(pred_move=forecast["pred_move"] * k)
        sims = sv.simulate(chunk, forecast, pool, draws=DRAW)
        history.append(sv.score_events(chunk, scaled_events, sims, name))
        chunk_last, scaled_last = chunk, scaled_events
    dec = sv.decide(pd.concat(history, ignore_index=True))
    last_ids = set(scaled_last["event_id"].astype(str))
    taken = dec[dec["event_id"].astype(str).isin(last_ids) & (dec["structure"] == name) & (dec["status"] == "taken")]
    mid = sv.score(chunk_last, scaled_last, taken, name, 0.5, mid_marks=True)
    sec = sv.secured(chunk_last, scaled_last, taken, name)
    rows = mid.merge(sec, on="event_id")
    rows["entry_date"] = rows["event_id"].map(lambda e: sessions[e][0])
    rows["exit_date"] = rows["event_id"].map(lambda e: sessions[e][1])
    rows = rows.rename(columns={"exit_value": "exit_value_mid"})
    rows["exit_value"] = rows["exit_value_mid"]
    return len(taken), float(account.simulate_account(rows, "mid")["total_return"])


def _hand_draw_pred(model, table, labels, d):
    """Draw ``d``'s forecasts by hand from the documented formula; NaN where the label is not > 0."""
    y = labels["abs_move"].reindex(table.index).to_numpy(dtype=float)
    ok = np.isfinite(y) & (y > 0)
    z = np.array([_recipe_z(str(e), d) for e in table.index])
    f = np.exp(model["L"] + model["b"] * (np.log(np.where(ok, y, 1.0)) - model["mean_log_y"]) + model["s"] * z)
    return pd.Series(np.where(ok, f, np.nan), index=table.index.astype(str))


@pytest.mark.parametrize("name,k", ORACLE_CASES)
def test_run_anchor_matches_the_direct_route(name, k, monkeypatch):
    _small_window(monkeypatch)
    table, labels, _ = _world()
    months, sessions = _months()
    actual = labels["abs_move"].reindex(table.index).to_numpy(dtype=float)
    ok = np.isfinite(actual) & (actual > 0)
    pred = pd.Series(np.where(ok, actual, np.nan), index=table.index.astype(str))

    res = fake.run_anchor(months, table, labels, sessions, draws=DRAW)
    n_direct, ret_direct = _direct(months, table, labels, pred, sessions, name=name, k=k)
    if (name, k) == ("CND-PS", 1.0):
        assert n_direct > 0  # the check is not vacuous
    assert res[name][k]["trades"] == n_direct
    assert res[name][k]["ret"] == pytest.approx(ret_direct)

    hist = pool_history(table.assign(pred_move=pred), labels)
    err = hist.loc[hist["pred_abs_move"].notna(), "err_move"]
    assert len(err) > 0 and (err == 0.0).all()


@pytest.mark.parametrize("d,name,k", [(0, *case) for case in ORACLE_CASES] + [(1, "CND-PS", 1.0)])
def test_run_draw_matches_the_direct_route_with_a_hand_computed_fake_forecast(d, name, k, monkeypatch):
    _small_window(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    pred = _hand_draw_pred(model, table, labels, d)

    res = fake.run_draw(months, table, labels, model, d, sessions, draws=DRAW)
    n_direct, ret_direct = _direct(months, table, labels, pred, sessions, name=name, k=k)
    if (d, name, k) == (0, "CND-PS", 1.0):
        assert n_direct > 0  # the check is not vacuous
    assert res[name][k]["trades"] == n_direct
    assert res[name][k]["ret"] == pytest.approx(ret_direct)


def test_run_fake_pools_two_draws_into_cnd_ps_scores_at_each_k(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    _small_draws(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    res = fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert res["status"] == "ok"
    for k in (0.75, 1.0, 1.5):
        draws = [_direct(months, table, labels, _hand_draw_pred(model, table, labels, d), sessions, name="CND-PS", k=k)
                 for d in (0, 1)]
        rets = np.array([ret for _trades, ret in draws], dtype=float)
        score = res["body"]["CND-PS"]["scores"][str(k)]
        assert score["ret"] == pytest.approx(float(np.mean(rets)))
        assert score["trades"] == sum(trades for trades, _ret in draws)
        assert score["se"] == pytest.approx(float(np.std(rets, ddof=1)) / math.sqrt(2))


def test_t4_markdown_renders_300_trades_and_marks_299_n_a():
    scores = {"CND-PS": {1.0: {"ret": 0.0123, "se": 0.0, "trades": 300},
                         1.5: {"ret": 0.0123, "se": 0.0, "trades": 299}}}
    md = fake.t4_markdown(scores)
    assert "| CND-PS | 1.23 | n/a |" in md


# ---- arm C shape in the tuning module -------------------------------------

def _c_frozen(root, receipt):
    """Write ``receipt`` as the tuning_C receipt under ``root`` and read it back through tuning.frozen."""
    receipts.write_receipt(root / "receipts" / "tuning_C.json", receipt)
    return tuning.frozen(root, "C", "h")


def test_units_c_is_b_and_frozen_checks_the_arm_c_shape(tmp_path):
    assert tuning.units("C") == tuning.units("B")
    body = {n: {"k": 1.0, "flags": [], "members": [n], "scores": {}} for n in NAMES}
    fake_meta = {"N": 20, "seed_tag": "exp189-v1", "model": C_MODEL, "calibration": {"ok": True}, "sim_draws": DRAW}
    receipt = {"experiment": "EXP-189", "arm": "C", "inputs_hash": "h", "window": "2023H2-2024", "body": body,
               "fake": fake_meta}
    assert set(_c_frozen(tmp_path / "valid", receipt)) == set(NAMES)
    bad = {
        "calibration_failed": {**receipt, "fake": {**fake_meta, "calibration": {"ok": False}}},
        "no_fake": {k: v for k, v in receipt.items() if k != "fake"},
        "fake_empty": {**receipt, "fake": {}},
        "N_2": {**receipt, "fake": {**fake_meta, "N": 2}},
        "N_missing": {**receipt, "fake": {k: v for k, v in fake_meta.items() if k != "N"}},
        "N_true": {**receipt, "fake": {**fake_meta, "N": True}},
        "missing_unit": {**receipt, "body": {n: v for n, v in body.items() if n != NAMES[0]}},
    }
    for name, change in bad.items():
        with pytest.raises(Refusal) as caught:
            _c_frozen(tmp_path / name, change)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT", name
        assert caught.value.details["reason"] == "tuning_receipt_changed", name


def test_write_tuning_c_is_private_and_not_a_public_entry():
    assert not hasattr(fake, "write_tuning_c")


def test_write_tuning_c_refuses_an_uncalibrated_fake_and_writes_nothing(tmp_path, monkeypatch):
    _small_draws(monkeypatch)
    with pytest.raises(Refusal) as caught:
        fake._write_tuning_c(tmp_path, {}, "h", C_MODEL, {"ok": False, "reason": "fake_not_calibrated"}, 2, DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "fake_not_calibrated"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


def test_run_fake_refuses_a_reduced_draw_count_before_calibration_and_writes_nothing(tmp_path, monkeypatch):
    table, labels, model = _world()
    months, sessions = _months()
    monkeypatch.setattr(fake, "calibration", lambda *a, **k: pytest.fail("no calibration for a non-protocol N"))
    monkeypatch.setattr(fake, "run_draw", lambda *a, **k: pytest.fail("no draws for a non-protocol N"))
    with pytest.raises(Refusal) as caught:
        fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "fake_draws_not_protocol"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


def test_write_tuning_c_refuses_a_reduced_draw_count_with_an_ok_calibration_and_writes_nothing(tmp_path):
    with pytest.raises(Refusal) as caught:
        fake._write_tuning_c(tmp_path, {}, "h", C_MODEL, {"ok": True}, 2, DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "fake_draws_not_protocol"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


BAD_MODELS = {
    "no_n": MODEL,
    "n_10": {**MODEL, "n": 10},
    "L_none": {**C_MODEL, "L": None},
}


@pytest.mark.parametrize("name", list(BAD_MODELS))
def test_write_tuning_c_refuses_a_malformed_fake_model_and_writes_nothing(name, tmp_path):
    with pytest.raises(Refusal) as caught:
        fake._write_tuning_c(tmp_path, {}, "h", BAD_MODELS[name], {"ok": True}, 20, DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "fake_model_invalid"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


GOOD_FAKE_MODEL = {"L": 1.0, "b": 0.8, "s": 0.2, "mean_log_y": 1.0, "n": 300}


def test_fake_model_ok_accepts_the_good_model_and_s_zero():
    assert tuning.fake_model_ok(GOOD_FAKE_MODEL) is True
    assert tuning.fake_model_ok({**GOOD_FAKE_MODEL, "s": 0.0}) is True


@pytest.mark.parametrize("model", [
    {**GOOD_FAKE_MODEL, "n": 299},
    {**GOOD_FAKE_MODEL, "n": True},
    {**GOOD_FAKE_MODEL, "n": 300.0},
    {**GOOD_FAKE_MODEL, "s": -0.1},
    {**GOOD_FAKE_MODEL, "s": None},
    {**GOOD_FAKE_MODEL, "L": True},
    {**GOOD_FAKE_MODEL, "L": float("inf")},
    {**GOOD_FAKE_MODEL, "b": float("nan")},
    {**GOOD_FAKE_MODEL, "mean_log_y": "1.0"},
    *[{k: v for k, v in GOOD_FAKE_MODEL.items() if k != key} for key in ("L", "b", "s", "mean_log_y", "n")],
    None,
    [1.0, 0.8, 0.2, 1.0, 300],
], ids=lambda m: repr(m)[:60])
def test_fake_model_ok_refuses_each_malformed_model(model):
    assert tuning.fake_model_ok(model) is False


@pytest.mark.parametrize("name", list(BAD_MODELS))
def test_run_fake_refuses_a_malformed_fake_model_before_any_draw(name, tmp_path, monkeypatch):
    table, labels, _ = _world()
    months, sessions = _months()
    monkeypatch.setattr(fake, "calibration", lambda *a, **k: pytest.fail("no calibration for a malformed model"))
    monkeypatch.setattr(fake, "run_draw", lambda *a, **k: pytest.fail("no draws for a malformed model"))
    with pytest.raises(Refusal) as caught:
        fake.run_fake(months, table, labels, BAD_MODELS[name], sessions, tmp_path, "hash-1", n=20, draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "fake_model_invalid"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


def test_frozen_refuses_a_c_receipt_written_with_a_reduced_draw_count(tmp_path):
    body = {n: {"k": 1.0, "flags": [], "members": [n], "scores": {}} for n in NAMES}
    receipts.write_receipt(tmp_path / "receipts" / "tuning_C.json", {
        "experiment": "EXP-189", "arm": "C", "inputs_hash": "h", "window": "2023H2-2024", "body": body,
        "fake": {"N": 2, "seed_tag": "exp189-v1", "model": C_MODEL, "calibration": {"ok": True}}})
    with pytest.raises(Refusal) as caught:
        tuning.frozen(tmp_path, "C", "h")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "tuning_receipt_changed"


def test_frozen_refuses_a_stored_arm_c_receipt_whose_calibration_failed(tmp_path):
    body = {n: {"k": 1.0, "flags": [], "members": [n], "scores": {}} for n in NAMES}
    receipts.write_receipt(tmp_path / "receipts" / "tuning_C.json", {
        "experiment": "EXP-189", "arm": "C", "inputs_hash": "h", "window": "2023H2-2024", "body": body,
        "fake": {"N": 2, "seed_tag": "exp189-v1", "model": C_MODEL, "calibration": {"ok": False}}})
    with pytest.raises(Refusal) as caught:
        tuning.frozen(tmp_path, "C", "h")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "tuning_receipt_changed"


@pytest.mark.parametrize("n", [20.0, np.int64(20), True])
def test_run_fake_refuses_a_non_int_protocol_n_before_any_work(n, tmp_path, monkeypatch):
    table, labels, model = _world()
    months, sessions = _months()
    monkeypatch.setattr(fake, "calibration", lambda *a, **k: pytest.fail("no calibration for a non-int N"))
    monkeypatch.setattr(fake, "run_draw", lambda *a, **k: pytest.fail("no draws for a non-int N"))
    with pytest.raises(Refusal) as caught:
        fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=n, draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "fake_draws_not_protocol"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


def test_c_receipt_with_wrong_seed_missing_or_bad_model_is_refused_and_a_valid_one_accepted(tmp_path):
    body = {n: {"k": 1.0, "flags": [], "members": [n], "scores": {}} for n in NAMES}
    valid = {"N": 20, "seed_tag": "exp189-v1", "model": C_MODEL, "calibration": {"ok": True}, "sim_draws": DRAW}
    bad = {
        "seed_tag_other": {**valid, "seed_tag": "other"},
        "model_missing": {k: v for k, v in valid.items() if k != "model"},
        "model_L_null": {**valid, "model": {**C_MODEL, "L": None}},  # NaN is not strict JSON, so the receipt carries null
        "model_n_10": {**valid, "model": {**C_MODEL, "n": 10}},
    }
    for name, fake_meta in bad.items():
        receipt = {"experiment": "EXP-189", "arm": "C", "inputs_hash": "h", "window": "2023H2-2024", "body": body,
                   "fake": fake_meta}
        with pytest.raises(Refusal) as caught:
            _c_frozen(tmp_path / name, receipt)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT", name
        assert caught.value.details["reason"] == "tuning_receipt_changed", name
    receipt = {"experiment": "EXP-189", "arm": "C", "inputs_hash": "h", "window": "2023H2-2024", "body": body,
               "fake": valid}
    assert set(_c_frozen(tmp_path / "valid", receipt)) == set(NAMES)


def test_run_fake_refuses_a_non_protocol_sim_draw_count_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "FAKE_SIM_DRAWS", DRAWS)
    table, labels, model = _world()
    months, sessions = _months()
    monkeypatch.setattr(fake, "calibration", lambda *a, **k: pytest.fail("no calibration for a non-protocol sim draw count"))
    monkeypatch.setattr(fake, "run_draw", lambda *a, **k: pytest.fail("no draws for a non-protocol sim draw count"))
    with pytest.raises(Refusal) as caught:
        fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "fake_sim_draws_not_protocol"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


@pytest.mark.parametrize("sim_draws", [DRAW, None, True], ids=["200", "missing", "true"])
def test_frozen_refuses_a_c_receipt_whose_sim_draws_is_not_the_protocol(sim_draws, tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "FAKE_SIM_DRAWS", DRAWS)
    body = {n: {"k": 1.0, "flags": [], "members": [n], "scores": {}} for n in NAMES}
    fake_meta = {"N": 20, "seed_tag": "exp189-v1", "model": C_MODEL, "calibration": {"ok": True}}
    if sim_draws is not None:
        fake_meta["sim_draws"] = sim_draws
    with pytest.raises(Refusal) as caught:
        _c_frozen(tmp_path, {"experiment": "EXP-189", "arm": "C", "inputs_hash": "h", "window": "2023H2-2024",
                             "body": body, "fake": fake_meta})
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "tuning_receipt_changed"


def test_frozen_accepts_a_c_receipt_whose_sim_draws_is_the_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "FAKE_SIM_DRAWS", DRAWS)
    body = {n: {"k": 1.0, "flags": [], "members": [n], "scores": {}} for n in NAMES}
    receipt = {"experiment": "EXP-189", "arm": "C", "inputs_hash": "h", "window": "2023H2-2024", "body": body,
               "fake": {"N": 20, "seed_tag": "exp189-v1", "model": C_MODEL, "calibration": {"ok": True},
                        "sim_draws": DRAWS}}
    assert set(_c_frozen(tmp_path, receipt)) == set(NAMES)


@pytest.mark.parametrize("empty", [[], iter([])], ids=["list", "generator"])
def test_run_fake_refuses_no_months_before_calibration_and_writes_nothing(empty, tmp_path, monkeypatch):
    table, labels, model = _world()
    _, sessions = _months()
    monkeypatch.setattr(fake, "calibration", lambda *a, **k: pytest.fail("no calibration without months"))
    monkeypatch.setattr(fake, "run_draw", lambda *a, **k: pytest.fail("no draws without months"))
    with pytest.raises(Refusal) as caught:
        fake.run_fake(empty, table, labels, model, sessions, tmp_path, "hash-1", draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "months_empty"
    assert not (tmp_path / "receipts" / "tuning_C.json").exists()


def test_run_fake_stores_the_sim_draw_count_in_the_receipt(tmp_path, monkeypatch):
    _small_window(monkeypatch)
    _small_draws(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    res = fake.run_fake(months, table, labels, model, sessions, tmp_path, "hash-1", n=2, draws=DRAW)
    assert res["status"] == "ok"
    receipt = json.loads((tmp_path / "receipts" / "tuning_C.json").read_text(encoding="utf-8"))
    assert receipt["fake"]["sim_draws"] == DRAW


# ---- causal per-month pools -----------------------------------------------

def test_anchor_with_no_label_printed_before_any_month_takes_no_trades(monkeypatch):
    _small_window(monkeypatch)
    table, labels, _ = _world()
    months, sessions = _months()
    none_printed = labels.assign(available="2099-01-01")
    res = fake.run_anchor(months, table, none_printed, sessions, draws=DRAW)
    assert all(res[name][k]["trades"] == 0 for name in NAMES for k in GRID)
    normal = fake.run_anchor(months, table, labels, sessions, draws=DRAW)
    assert normal["CND-PS"][1.0]["trades"] > 0


def test_label_printed_after_a_month_start_is_excluded_from_that_month(monkeypatch):
    _small_window(monkeypatch)
    table, labels, _ = _world()
    months, sessions = _months()
    # Every label is printed on 2024-05-15: June's pool (start 2024-06-01) holds them, May's (start 2024-05-01) holds none.
    late = labels.assign(available="2024-05-15")
    res = fake.run_anchor(months, table, late, sessions, draws=DRAW)
    assert set(res) == set(NAMES)
    for name in NAMES:
        assert set(res[name]) == set(GRID)
    actual = late["abs_move"].reindex(table.index).to_numpy(dtype=float)
    ok = np.isfinite(actual) & (actual > 0)
    pred = pd.Series(np.where(ok, actual, np.nan), index=table.index.astype(str))
    n_direct, ret_direct = _direct(months, table, late, pred, sessions, name="CND-PS", k=1.0)
    assert res["CND-PS"][1.0]["trades"] == n_direct
    assert res["CND-PS"][1.0]["ret"] == pytest.approx(ret_direct)


def test_labels_without_an_available_column_refuse_in_run_draw_and_run_anchor(monkeypatch):
    _small_window(monkeypatch)
    table, labels, model = _world()
    months, sessions = _months()
    bare = labels.drop(columns=["available"])
    with pytest.raises(Refusal) as caught:
        fake.run_draw(months, table, bare, model, 0, sessions, draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "labels_availability_missing"
    with pytest.raises(Refusal) as caught:
        fake.run_anchor(months, table, bare, sessions, draws=DRAW)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "labels_availability_missing"


# ---- empty pool slices and all-NaN forecasts (pandas 3 str-dtype map) -------

def test_at_keeps_float64_for_empty_and_all_nan_forecasts_and_empty_pool_history_survives():
    table, labels, _ = _world()
    table, labels = table.iloc[:12], labels.iloc[:12]
    pred = pd.Series(np.full(len(table), np.nan), index=[str(e) for e in table.index])
    empty = fake._at(pred, table.index[:0])
    assert empty.dtype == np.float64 and empty.shape == (0,)
    allnan = fake._at(pred, table.index)
    assert allnan.dtype == np.float64 and allnan.shape == (len(table),) and np.isnan(allnan).all()
    out = pool_history(table.iloc[:0].assign(pred_move=fake._at(pred, table.index[:0])), labels)
    assert out.empty


def _smoke_world(nan_labels=False):
    """Twelve rows E0..E11 with every event dated inside or after the first month (no row before 2024-05-01)."""
    table, labels, _ = _world()
    table, labels = table.iloc[:12], labels.iloc[:12]
    dates = ["2024-05-10"] * 6 + ["2024-06-10"] * 6
    table = table.assign(event_date=pd.to_datetime(dates))
    if nan_labels:
        labels = labels.assign(abs_move=np.nan)
    return table, labels


@pytest.mark.parametrize("kind", ["anchor", "draw"])
@pytest.mark.parametrize("nan_labels", [False, True], ids=["finite", "all_nan"])
def test_smoke_world_with_empty_first_pool_slice_survives_with_no_trades(kind, nan_labels, monkeypatch):
    _small_window(monkeypatch)
    table, labels = _smoke_world(nan_labels)
    months, sessions = _months()
    if kind == "anchor":
        res = fake.run_anchor(months, table, labels, sessions, draws=DRAW)
    else:
        res = fake.run_draw(months, table, labels, MODEL, 0, sessions, draws=DRAW)
    assert set(res) == set(NAMES)
    assert all(set(res[name]) == set(GRID) for name in NAMES)
    assert all(res[name][k]["trades"] == 0 for name in NAMES for k in GRID)


def test_at_returns_finite_forecasts_at_their_own_ids():
    pred = pd.Series([1.5, np.nan, 3.5], index=["1", "2", "3"])
    out = fake._at(pred, pd.Index([3, 1, 2, 9]))  # integer ids, other order, one absent
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [3.5, 1.5, np.nan, np.nan])


def test_anchor_sweep_continues_past_an_empty_pool_slice_with_forecasts_at_their_ids(monkeypatch):
    _small_window(monkeypatch)
    table, labels = _smoke_world()
    months, sessions = _months()
    seen = []
    orig = fake.Sweep.add_month

    def spy(self, chunk, events, *args, **kwargs):
        seen.append(events)
        return orig(self, chunk, events, *args, **kwargs)

    monkeypatch.setattr(fake.Sweep, "add_month", spy)
    fake.run_anchor(months, table, labels, sessions, draws=DRAW)
    assert len(seen) == len(months) == 2
    for events in seen:
        assert len(events) > 0
        got = events["pred_move"].to_numpy(dtype=float)
        want = labels["abs_move"].reindex(events["event_id"].astype(str)).to_numpy(dtype=float)
        np.testing.assert_array_equal(got, want)
        assert np.isfinite(got).all()
