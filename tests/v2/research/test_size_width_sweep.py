import types

import numpy as np
import pandas as pd
import pytest

from engine.forecast_sizing import SHORT_VOL_OUTER, forecast_params
from engine.structures import BUY, STRUCTURES
from experiments import size_pnl_shortvol as sv
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_straddle import lots
from experiments.size_width_sizing import GRID, k_params
from experiments.size_width_sweep import RET_COLS, _TAKEN, E_ARM, WINDOWS, Sweep, admit_k, in_window, pick_e, scaled
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


PRED = 6.0  # events_for's default pred_move


def test_taken_row_and_scored_row_match_a_hand_derived_condor(monkeypatch):
    # CND-PS: up1/dn1 SELL at atm -/+ w*spot, up2/dn2 BUY at atm -/+ 2w*spot, w = k*pred/100/divisor, atm 100, strikes snapped
    # by hand (the integer ladder is exact here). Entry mid 1 + 0.01*(K-100)^2; every exit quote is bid = ask = 1.0.
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep = Sweep()
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)

    def mid(strike):
        return 1.0 + 0.01 * (strike - 100) ** 2

    def hand_legs(k):
        """(side_sign, qty, strike) of each leg: the legs of the structure, strikes by hand."""
        w = k * PRED / 100.0 / SHORT_VOL_OUTER[NAME]
        step = {"atm": 0, "up1": 1, "dn1": -1, "up2": 2, "dn2": -2}  # atm is the zero-quantity reference leg
        return [(1.0 if leg.side == BUY else -1.0, float(leg.qty), round(100 + step[leg.name] * w * 100))
                for leg in STRUCTURES[NAME](width_moneyness=w).legs]

    cost1 = sum(sign * qty * mid(strike) for sign, qty, strike in hand_legs(1.0))  # 1.36 - 1.09 - 1.09 + 1.36
    cost2 = sum(sign * qty * mid(strike) for sign, qty, strike in hand_legs(2.0))  # 2.44 - 1.36 - 1.36 + 2.44
    assert cost1 == pytest.approx(0.54)
    assert cost2 == pytest.approx(2.16)
    assert cost2 - cost1 == pytest.approx(1.62)

    row = sweep.rows([(NAME, 1.0)], WINDOW).iloc[0]
    # each leg closes at its listed-quote 1.0: a long leg is sold for +1.0, a short leg is bought back for -1.0
    exit_mid = sum(sign * qty * 1.0 for sign, qty, _strike in hand_legs(1.0))  # two longs +1 each, two shorts -1 each: 0 for the condor
    assert row["entry_cost"] == pytest.approx(cost1)
    assert row["exit_value_mid"] == pytest.approx(exit_mid)
    assert row["pnl_dollars_mid"] == pytest.approx((exit_mid - cost1) * 100 * lots(cost1))

    scored_k2 = sweep.scored[(NAME, 2.0)][-1]
    hit = scored_k2[scored_k2["event_id"] == row["event_id"]].iloc[0]
    assert hit["status"] == "scored"
    assert hit["width"] == pytest.approx(2.0 * PRED / 100.0 / SHORT_VOL_OUTER[NAME])
    assert hit["entry_cost"] == pytest.approx(cost2)


@pytest.mark.parametrize("kwargs", [{"ks": ()}, {"ks": (1.0, 1.0)}, {"ks": (99.0,)}, {"ks": (True,)}, {"ks": (1.1,)},
                                    {"names": ()}, {"names": (NAME, NAME)}, {"names": ("NOT-A-STRUCTURE",)}])
def test_sweep_refuses_an_ungridded_or_repeated_admission(kwargs):
    with pytest.raises(ValueError):
        Sweep(**kwargs)


def test_admit_k_accepts_grid_reals_and_refuses_everything_else():
    assert admit_k(1) == 1.0 and admit_k(1.0) == 1.0 and admit_k(2.5) == 2.5
    for bad in (True, 99.0, "1.0", None):
        with pytest.raises(ValueError):
            admit_k(bad)


def test_rows_refuses_a_taken_row_with_no_exit_date():
    row = dict.fromkeys(_TAKEN)
    row.update({"event_id": "nat", "event_date": pd.Timestamp("2024-12-02"), "structure": NAME, "k": 1.0,
                "entry_date": pd.Timestamp("2024-12-02"), "exit_date": pd.NaT})
    sweep = Sweep()
    sweep.taken[(NAME, 1.0)] = [pd.DataFrame([row], columns=_TAKEN)]
    with pytest.raises(ValueError, match="exit dates"):
        sweep.rows([(NAME, 1.0)], WINDOWS["2024"])


def test_rows_keep_only_trades_that_exit_inside_the_window():
    def taken(eid, event, exit_):
        row = dict.fromkeys(_TAKEN)
        row.update({"event_id": eid, "event_date": pd.Timestamp(event), "structure": NAME, "k": 1.0,
                    "entry_date": pd.Timestamp(event), "exit_date": pd.Timestamp(exit_)})
        return row

    sweep = Sweep()
    sweep.taken[(NAME, 1.0)] = [pd.DataFrame([taken("late", "2024-12-20", "2025-01-10"),
                                              taken("in", "2024-12-02", "2024-12-20")], columns=_TAKEN)]
    assert list(sweep.rows([(NAME, 1.0)], WINDOWS["2024"])["event_id"]) == ["in"]
    assert sweep.rows([(NAME, 1.0)], WINDOWS["2025"]).empty  # both event dates are in 2024


# ---- pricing seam, ladder columns, fresh rows -----------------------------

def _stub_pricing(log):
    """A pricing seam that records each call in ``log`` and returns minimal frames; C0 has a non-positive entry cost."""
    def simulate(chunk, events, pool, draws=DRAW):
        log.append(("simulate",))
        return {str(e): "sim" for e in events["event_id"]}

    def score_events(chunk, ev, sims, name, alpha=0.5):
        log.append(("score_events", name, alpha))
        return pd.DataFrame([_score(e, 1.0, 0.1) for e in ev["event_id"]], columns=sv.SCORE_COLS)

    def decide(hist):
        log.append(("decide",))
        return hist.assign(status="taken", reason=None)

    def score(chunk, ev, mine, name, alpha, mid_marks=False, repriced=False):
        log.append(("score", alpha, mid_marks, repriced))
        eids = list(mine["event_id"])
        return pd.DataFrame({"event_id": eids, "structure": name, "entry_cost": [-1.0 if e == "C0" else 2.0 for e in eids],
                             "exit_value": 1.0, "rung": 0, "lots": 1, "pnl_per_share": [alpha * 3.0 - 0.5 for _ in eids],
                             "pnl_dollars": 100.0})

    def secured(chunk, ev, mine, name):
        log.append(("secured",))
        return pd.DataFrame({"event_id": list(mine["event_id"]), "secured_per_contract": 5.0})

    return types.SimpleNamespace(simulate=simulate, score_events=score_events, decide=decide, score=score, secured=secured)


def test_pricing_seam_is_used(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("sv.simulate must not run when a pricing seam is given")
    monkeypatch.setattr(sv, "simulate", refuse)
    chunk, _prior, cur, sessions = _two_months()
    log = []
    sweep = Sweep(ks=(1.0,), names=(NAME,), pricing=_stub_pricing(log))
    sweep.add_month(chunk, cur, None, output=True, sessions=sessions)
    assert {c[0] for c in log} == {"simulate", "score_events", "decide", "score", "secured"}
    assert set(sweep.taken[(NAME, 1.0)][0]["event_id"]) == set(CUR)
    assert Sweep().pricing is sv


def test_ladder_columns():
    chunk, _prior, cur, sessions = _two_months()
    log = []
    sweep = Sweep(ks=(1.0,), names=(NAME,), pricing=_stub_pricing(log), ladder=True)
    sweep.add_month(chunk, cur, None, output=True, sessions=sessions)
    taken = sweep.taken[(NAME, 1.0)][0]
    assert list(taken.columns) == _TAKEN + RET_COLS
    assert RET_COLS == ["ret_a0.00", "ret_a0.25", "ret_a0.50", "ret_a0.75", "ret_a1.00"]
    by_id = taken.set_index("event_id")
    assert np.isnan(by_id.loc["C0", "ret_a1.00"])
    assert by_id.loc["C1", "ret_a1.00"] == pytest.approx((1.0 * 3.0 - 0.5) / 2.0)
    assert by_id.loc["C1", "ret_a0.00"] == pytest.approx(-0.5 / 2.0)
    assert ("score", 1.0, True, True) in log

    plain = Sweep(ks=(1.0,), names=(NAME,), pricing=_stub_pricing([]))
    plain.add_month(chunk, cur, None, output=True, sessions=sessions)
    assert list(plain.taken[(NAME, 1.0)][0].columns) == _TAKEN


def test_fresh_declined_rows_are_the_month_declined_decisions_with_arm_strings(monkeypatch):
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep = Sweep(names=(NAME,))
    sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    burn = sweep.declined_rows()
    assert burn.empty and list(burn.columns) == ["event_id", "event_date", "structure", "status", "reason", "arm"]
    sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)
    got = sweep.declined_rows()
    assert not got.empty and (got["status"] != "taken").all() and set(got["event_id"]) <= set(CUR)
    assert all(isinstance(a, str) for a in got["arm"]) and "1.0" in set(got["arm"])
    assert (got[got["arm"] == "1.0"]["structure"] == NAME).all()


def test_add_month_returns_fresh_rows(monkeypatch):
    _small_window(monkeypatch)
    pool = build_pool(*_tables(), floor=None)
    chunk, prior, cur, sessions = _two_months()
    sweep = Sweep(names=(NAME,))
    burn = sweep.add_month(chunk, prior, pool, output=False, sessions=sessions, draws=DRAW)
    assert burn.empty and list(burn.columns) == _TAKEN + ["arm"]
    fresh = sweep.add_month(chunk, cur, pool, output=True, sessions=sessions, draws=DRAW)
    assert {"1.0", "E"}.issubset(set(fresh["arm"]))
    for arm in [*sweep.ks, E_ARM]:
        n = sum(len(f) for f in sweep.taken.get((NAME, arm), []))
        assert (fresh["arm"] == str(arm)).sum() == n


