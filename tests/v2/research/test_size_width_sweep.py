import numpy as np
import pandas as pd
import pytest

from engine.forecast_sizing import forecast_params
from engine.pnl_sim import DRAWS
from experiments import size_pnl_shortvol as sv
from experiments.size_pnl_pool import build_pool
from experiments.size_width_sizing import GRID, k_params
from experiments.size_width_sweep import _TAKEN, E_ARM, WINDOWS, Sweep, feed, in_window, pick_e, scaled
from tests.v2.research.test_size_pnl_shortvol import _tables, chunk_of, events_for

DRAW = 200
NAME = "CND-PS"
WINDOW = (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-07-01"))
PRIOR = [f"P{i}" for i in range(6)]
CUR = [f"C{i}" for i in range(6)]


def _two_months():
    chunk = chunk_of(*[(e, {}) for e in PRIOR + CUR])
    prior = events_for(*PRIOR, date="2024-05-10")
    cur = events_for(*CUR, date="2024-06-10")
    sessions = {e: (pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-25")) for e in PRIOR}
    sessions.update({e: (pd.Timestamp("2024-06-11"), pd.Timestamp("2024-06-21")) for e in CUR})
    return chunk, prior, cur, sessions


def _score(eid, k, value, status="scored", reason=None):
    return {"event_id": eid, "event_date": pd.Timestamp("2024-06-10"), "structure": NAME, "width": 0.04,
            "entry_cost": 3.0, "rel_spread": 0.1, "mcap_usd": 20e9, "exp_pnl_sim": value, "status": status,
            "reason": reason, "alpha": 0.5}


def _small_window(monkeypatch):
    real = sv.trailing_cutoff
    monkeypatch.setattr(sv, "trailing_cutoff",
                        lambda history, as_of, **kw: real(history, as_of, **{**kw, "min_window": 1}))


# ---- scaling and parity ---------------------------------------------------

def test_scaled_multiplies_pred_move_keeps_nan_and_takes_an_array():
    ev = pd.DataFrame({"event_id": ["a", "b"], "pred_move": [2.0, np.nan]})
    out = scaled(ev, 1.5)
    assert out["pred_move"].iloc[0] == pytest.approx(3.0)
    assert np.isnan(out["pred_move"].iloc[1])
    arr = scaled(ev, np.array([2.0, 3.0]))
    assert arr["pred_move"].iloc[0] == pytest.approx(4.0)
    assert np.isnan(arr["pred_move"].iloc[1])
    assert ev["pred_move"].iloc[0] == 2.0  # the input is not mutated


@pytest.mark.parametrize("name", sv.NAMES)
def test_forecast_at_k_times_pred_matches_k_params_at_pred(name):
    for k in GRID:
        for pred in (1.0, 2.0, 4.0, 6.0, 9.0, 30.0):
            scaled_params = forecast_params(name, pred * k)
            direct = k_params(name, pred, k)
            assert (scaled_params is None) == (direct is None), (name, k, pred)
            if direct is not None:
                assert scaled_params["width_moneyness"] == pytest.approx(direct["width_moneyness"], rel=0, abs=1e-12)


# ---- arm E pick -----------------------------------------------------------

def test_pick_e_takes_the_best_k_then_nearest_one_then_smaller():
    values = {  # event -> {k: exp_pnl_sim}
        "a": {0.75: 1.0, 1.0: 1.0, 1.25: 1.0, 1.5: 9.0},   # best at 1.5
        "b": {0.75: 5.0, 1.0: 1.0, 1.25: 5.0, 1.5: 1.0},   # tie 0.75 / 1.25 -> 0.75
        "c": {0.75: 2.0, 1.0: 4.0, 1.25: 2.0, 1.5: 4.0},   # tie 1.0 / 1.5 -> 1.0
    }
    ks = (0.75, 1.0, 1.25, 1.5)
    by_k = {}
    for k in ks:
        rows = [_score(e, k, values[e][k]) for e in values]
        rows.append(_score("d", k, np.nan, status="undetermined", reason="thin_history"))
        by_k[k] = pd.DataFrame(rows, columns=sv.SCORE_COLS)
    out = pick_e(by_k, ks)
    chosen = dict(zip(out["event_id"], out["k"]))
    assert chosen == {"a": 1.5, "b": 0.75, "c": 1.0, "d": 1.0}
    d = out[out["event_id"] == "d"].iloc[0]
    assert d["status"] == "undetermined" and d["reason"] == "thin_history"


# ---- the sweep itself -----------------------------------------------------

def test_add_month_at_k_one_takes_what_experiment_188_takes(monkeypatch):
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep = Sweep()
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)

    rows = sweep.rows([(NAME, 1.0)], WINDOW)
    assert not rows.empty
    assert list(rows.columns) == _TAKEN
    assert (rows["k"] == 1.0).all()
    assert (rows["entry_cost"] > 0).all()
    assert (rows["entry_date"] == pd.Timestamp("2024-06-11")).all()
    assert (rows["exit_date"] == pd.Timestamp("2024-06-21")).all()

    sims_p = sv.simulate(chunk, prior, pool, draws=DRAW)
    sims_c = sv.simulate(chunk, cur, pool, draws=DRAW)
    hist = pd.concat([sv.score_events(chunk, prior, sims_p, NAME), sv.score_events(chunk, cur, sims_c, NAME)],
                     ignore_index=True)
    dec = sv.decide(hist)
    expected = set(dec.loc[(dec["status"] == "taken") & dec["event_id"].isin(CUR), "event_id"])
    assert set(rows["event_id"]) == expected


def test_burn_in_month_adds_history_and_no_taken_rows(monkeypatch):
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, _cur, sessions = _two_months()
    sweep = Sweep()
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    assert sweep.taken == {}
    assert sweep.scored
    assert len(sweep.scored[(NAME, 1.0)][0]) == len(prior)


def test_e_arm_rows_carry_grid_k_and_curves_cover_every_arm(monkeypatch):
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep = Sweep()
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)

    scored = sweep.scored_rows(NAME, E_ARM, WINDOW)
    assert not scored.empty
    assert set(scored["k"]).issubset(set(GRID))
    taken = sweep.rows([(NAME, E_ARM)], WINDOW)
    assert set(taken["k"]).issubset(set(GRID))
    curves = sweep.curves(NAME, WINDOW)
    assert set(curves) == {str(k) for k in GRID} | {"E"}


def test_event_absent_from_the_chunk_is_dropped(monkeypatch):
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    cur_plus_ghost = pd.concat([cur, events_for("GHOST", date="2024-06-10")], ignore_index=True)
    sweep = Sweep()
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    sweep.add_month(chunk, cur_plus_ghost, pool, output=True, sessions=sessions, draws=DRAW)
    for frames in [*sweep.scored.values(), *sweep.taken.values()]:
        for frame in frames:
            assert "GHOST" not in set(frame["event_id"])


def _row(eid, date, value, *, status="scored", k=None):
    row = {"event_id": eid, "event_date": pd.Timestamp(date), "structure": NAME, "exp_pnl_sim": value,
           "status": status, "reason": None}
    if k is not None:
        row["k"] = k
    return row


def test_curves_are_window_means_and_none_for_an_empty_window():
    sweep = Sweep(ks=(1.0, 1.5), names=(NAME,))
    sweep.scored[(NAME, 1.0)] = [pd.DataFrame([_row("a", "2024-03-01", 0.1), _row("b", "2024-03-02", 0.3),
                                               _row("u", "2024-03-03", np.nan, status="undetermined")])]
    sweep.scored[(NAME, 1.5)] = [pd.DataFrame([_row("a", "2024-03-01", 0.2), _row("b", "2024-03-02", 0.4),
                                               _row("c", "2024-03-03", 0.6)])]
    sweep.scored[(NAME, E_ARM)] = [pd.DataFrame([_row("a", "2024-03-01", 0.3, k=1.0), _row("b", "2024-03-02", 0.6, k=1.5)])]
    assert sweep.curves(NAME, WINDOWS["2024"]) == pytest.approx({"1.0": 0.2, "1.5": 0.4, "E": 0.45})
    assert sweep.curves(NAME, WINDOWS["2025"]) == {"1.0": None, "1.5": None, "E": None}


def test_in_window_is_half_open():
    f = pd.DataFrame([_row("start", "2024-01-01", 0.0), _row("inside", "2024-12-31", 0.0), _row("end", "2025-01-01", 0.0)])
    assert set(in_window(f, WINDOWS["2024"])["event_id"]) == {"start", "inside"}


def test_declined_counts_include_the_decision_gate_outcomes(monkeypatch):
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep = Sweep()
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)

    sims_p = sv.simulate(chunk, prior, pool, draws=DRAW)
    sims_c = sv.simulate(chunk, cur, pool, draws=DRAW)
    hist = pd.concat([sv.score_events(chunk, prior, sims_p, NAME), sv.score_events(chunk, cur, sims_c, NAME)],
                     ignore_index=True)
    dec = sv.decide(hist)
    gone = dec[dec["event_id"].isin(CUR) & (dec["structure"] == NAME) & (dec["status"] != "taken")]
    expected = {str(r): int(c) for r, c in gone["reason"].fillna("none").value_counts().items()}
    assert any(reason in expected for reason in ("thin_history", "below_cutoff"))
    assert sweep.declined[(NAME, 1.0)] == expected


class _Recorder:
    def __init__(self):
        self.calls = []

    def add_month(self, chunk, events, pool, *, output, sessions, draws=DRAWS):
        self.calls.append((chunk, output, draws))


def test_feed_stops_before_until_and_sets_output_from_output_from():
    months = [
        {"period": pd.Timestamp("2024-01-01"), "chunk": "c1", "events": None, "pool": None, "sessions": {}},
        {"period": pd.Timestamp("2024-02-01"), "chunk": "c2", "events": None, "pool": None, "sessions": {}},
        {"period": pd.Timestamp("2024-03-01"), "chunk": "c3", "events": None, "pool": None, "sessions": {}, "draws": 50},
    ]
    rec = _Recorder()
    feed(rec, months, output_from=pd.Timestamp("2024-02-01"), until=pd.Timestamp("2024-03-01"))
    assert [(c, o) for c, o, _d in rec.calls] == [("c1", False), ("c2", True)]
    assert [d for _c, _o, d in rec.calls] == [DRAWS, DRAWS]

    rec = _Recorder()
    feed(rec, months, output_from=pd.Timestamp("2024-02-01"))
    assert rec.calls[-1] == ("c3", True, 50)
