"""Tests for the F reads of the accounts, objective, ladder, non-trade counts, fallback and T8/T9 (EXP-189 A1.2-A1.6, W7b2).

Synthetic only. Expected values come from the margin simulator, hand-built frames and a CurrentFills copy, never from the
function under test. Fixtures come from the F rows, F fills and F quotes tests.
"""
import json
import types

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_account as base
from experiments import size_pnl_shortvol as sv
from experiments import size_width_account as account
from experiments import size_width_f_counts as counts
from experiments import size_width_f_fills as ff
from experiments import size_width_fallback as fallback
from experiments import size_width_pricing as seam
from experiments import size_width_tuning as tuning
from experiments.fair_value_fills import VARIANTS
from experiments.size_pnl_straddle import lots
from experiments.size_width_deviation import DeviationSweep
from experiments.size_width_sweep import E_ARM, INNER, RET_COLS, Sweep
from experiments.size_width_success import LADDER
from tests.v2.research.test_size_width_f_fills import EVENTS, NAME, SIMS, _tight_frames
from tests.v2.research.test_size_width_f_quotes import DEFAULT_PLAN, _setup
from tests.v2.research.test_size_width_f_rows import SESS, WINDOW, _all_taken, _month

START = 200_000.0
POLICY = {**base.POLICY, "start_equity": START}
F_SKIP = ff.ENTRY_REASONS[0]


def _f_trades():
    """Three hand-built F trades: F, mid and harsh give different returns, and trade A's F and mid debits give different lots."""
    return pd.DataFrame({"event_id": ["A", "B", "C"],
                         "entry_date": pd.to_datetime(["2024-01-02", "2024-02-01", "2024-04-01"]),
                         "exit_date": pd.to_datetime(["2024-02-01", "2024-03-01", "2024-05-01"]),
                         "entry_cost": [2.0, 1.0, 3.0], "exit_value": [3.0, 0.9, 3.4],
                         "entry_cost_mid": [1.0, 1.2, 2.0], "exit_value_mid": [1.0, 1.0, 2.5],
                         "entry_cost_harsh": [1.2, 1.4, 2.2], "exit_value_harsh": [0.8, 0.8, 2.3],
                         "secured_per_contract": [500.0, 500.0, 500.0]})


def _plain(rows, entry, exit_):
    """A plain (non-F) frame of one view's own columns, for the margin simulator directly."""
    return pd.DataFrame({"entry_date": rows["entry_date"], "exit_date": rows["exit_date"], "entry_cost": rows[entry],
                         "exit_value": rows[exit_], "secured_per_contract": rows["secured_per_contract"]})


def _total_return(frame):
    return base.margin.simulate(frame, **POLICY).attrs["final_equity"] / START - 1.0


def _quarter_expected(frame, quarters):
    t = base.margin.simulate(frame, **POLICY)
    done = t[t["funded"]]
    pnl = done.assign(q=pd.PeriodIndex(pd.to_datetime(done["exit_date"]), freq="Q")).groupby("q")["pnl_usd"].sum().to_dict()
    equity, out = START, []
    for q in quarters:
        out.append(float(pnl.get(q, 0.0)) / equity)
        equity += float(pnl.get(q, 0.0))
    return out


def test_views_fund_each_at_its_own_debit_and_match_plain_frames():
    rows = _f_trades()
    assert account.deciding_mark(rows) == "primary"
    assert lots(2.0) != lots(1.0)  # trade A's lots differ between its F and mid debits
    views = {"primary": ("entry_cost", "exit_value"), "mid": ("entry_cost_mid", "exit_value_mid"),
             "harsh": ("entry_cost_harsh", "exit_value_harsh")}
    got = {m: account.simulate_account(rows, m)["total_return"] for m in views}
    for m, (entry, exit_) in views.items():
        assert got[m] == pytest.approx(_total_return(_plain(rows, entry, exit_)), rel=1e-12)
    assert len({round(v, 12) for v in got.values()}) == 3


def test_unit_score_and_quarter_returns_follow_f():
    rows, quarters = _f_trades(), tuning._quarters(INNER)
    plain = _plain(rows, "entry_cost", "exit_value")
    assert tuning.unit_score(rows, INNER)["ret"] == pytest.approx(_total_return(plain), rel=1e-12)
    assert account.quarter_returns(rows, quarters) == pytest.approx(_quarter_expected(plain, quarters), rel=1e-12)


def test_non_f_rows_still_walk_exit_value_mid_and_refuse_harsh():
    rows = _f_trades()[["event_id", "entry_date", "exit_date", "entry_cost", "exit_value", "exit_value_mid",
                        "secured_per_contract"]]
    quarters = tuning._quarters(INNER)
    assert account.deciding_mark(rows) == "mid"
    assert account.simulate_account(rows, "mid")["total_return"] == pytest.approx(
        _total_return(_plain(rows, "entry_cost", "exit_value_mid")), rel=1e-12)
    assert account.simulate_account(rows, "primary")["total_return"] == pytest.approx(
        _total_return(_plain(rows, "entry_cost", "exit_value")), rel=1e-12)
    assert account.quarter_returns(rows, quarters) == pytest.approx(
        _quarter_expected(_plain(rows, "entry_cost", "exit_value_mid"), quarters), rel=1e-12)
    with pytest.raises(ValueError):
        account.simulate_account(rows, "harsh")
    with pytest.raises(ValueError):
        account.simulate_account(rows.iloc[:0], "harsh")  # an unknown view is refused even for no rows


class _Stub:
    """A sweep stand-in: one frame per k; ``rows`` returns the frame of the first pair's k."""

    def __init__(self, frames):
        self.ks, self.frames = tuple(frames), frames

    def rows(self, pairs, window):
        return self.frames[pairs[0][1]]


def _one(entry, exit_, entry_mid, exit_mid):
    """One trade inside the inner window: F's debit and proceeds, and the mid pair."""
    return pd.DataFrame({"event_id": ["X"], "entry_date": pd.to_datetime(["2024-02-01"]),
                         "exit_date": pd.to_datetime(["2024-03-01"]), "entry_cost": [entry], "exit_value": [exit_],
                         "entry_cost_mid": [entry_mid], "exit_value_mid": [exit_mid], "secured_per_contract": [500.0]})


def test_tune_picks_the_k_on_the_f_marks_and_the_mid_pick_on_the_mid(monkeypatch):
    monkeypatch.setattr(tuning, "MIN_TRADES", 1)
    # k 1.0 is best on F (+50%, mid flat); k 1.5 is best on mid (+80%, F flat)
    f_frames = {1.0: _one(2.0, 3.0, 1.0, 1.0), 1.5: _one(2.0, 2.0, 1.0, 1.8)}
    assert tuning.tune(_Stub(f_frames), "A", INNER)["all"]["k"] == 1.0
    mid_frames = {k: _plain(f, "entry_cost_mid", "exit_value_mid").rename(columns={"exit_value": "exit_value_mid"})
                  for k, f in f_frames.items()}
    assert tuning.tune(_Stub(mid_frames), "A", INNER)["all"]["k"] == 1.5


def _f_sweep(quotes, **kw):
    pricing = seam.load(fills=ff.FFills(quotes))
    pricing.decide = _all_taken
    return Sweep(ks=(1.0,), names=(NAME,), pricing=pricing, **kw)


def _f_taken(quotes, chunk):
    """The F-scored taken rows of EVENTS at k = 1.0, as the sweep's own decisions are."""
    scored = seam.load(fills=ff.FFills(quotes)).score_events(chunk, EVENTS, SIMS, NAME, alpha=0.5)
    mine = _all_taken(scored)
    return mine[mine["status"] == "taken"]


def test_ladder_columns_are_priced_per_alpha_by_the_current_fills_copy(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = _f_sweep(fq, ladder=True, choices=True)
    out = _month(sweep, chunk)
    mine, copy = _f_taken(fq, chunk), seam.load()
    got = out[out["arm"] == "1.0"].set_index("event_id")
    for a, col in zip(LADDER, RET_COLS):
        lad = copy.score(chunk, EVENTS, mine, NAME, a, mid_marks=True, repriced=True).set_index("event_id")
        cost = pd.to_numeric(lad["entry_cost"], errors="coerce")
        want = (pd.to_numeric(lad["pnl_per_share"], errors="coerce") / cost).where(cost > 0)
        for eid in got.index:
            assert got.loc[eid, col] == pytest.approx(want[eid], rel=1e-9)
    assert not np.allclose(got[RET_COLS[0]], got[RET_COLS[-1]])  # the ladder is priced at each alpha, not once
    choices = pd.concat(sweep.fresh_choices, ignore_index=True)
    ce = choices[choices["status"] == "taken"].set_index("event_id")
    for a in (0.0, 1.0):
        lad = copy.score(chunk, EVENTS, mine, NAME, a, mid_marks=True, repriced=True).set_index("event_id")
        for eid in ce.index:
            assert ce.loc[eid, f"entry_a{a:.2f}"] == pytest.approx(lad.loc[eid, "entry_cost"], rel=1e-9)
    assert not np.allclose(ce["entry_a0.00"], ce["entry_a1.00"])


def test_a_nan_ladder_return_is_counted_as_nonpositive_debit(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = _f_sweep(fq, ladder=True)
    _month(sweep, chunk)
    frame = sweep.taken[(NAME, 1.0)][-1]
    frame.iloc[0, frame.columns.get_loc("ret_a0.00")] = np.nan
    res = tuning.evaluate(sweep, {n: 1.0 for n in sv.NAMES}, WINDOW)
    assert res["account"]["nonpositive_debit"] == {f"{a:.2f}": (1 if a == 0.0 else 0) for a in LADDER}


def _cand(rows):
    return pd.DataFrame(rows, columns=["event_id", "event_date", "reason"])


def test_nontrades_count_by_event_date_and_compare_each_arm_with_production():
    scored = {(NAME, 1.0): [_cand([("E0", "2024-01-05", None), ("E1", "2024-01-06", F_SKIP), ("E2", "2023-06-01", None)])],
              (NAME, 0.5): [_cand([("E0", "2024-01-05", F_SKIP), ("E1", "2024-01-06", F_SKIP)])],
              (NAME, E_ARM): [_cand([("E0", "2024-01-05", None), ("E1", "2024-01-06", F_SKIP)])]}
    declined = {(NAME, 1.0): [_cand([("E1", "2024-01-06", F_SKIP), ("E3", "2024-01-07", np.nan),
                                     ("E2", "2023-06-01", "below_cutoff")])]}
    sweep = types.SimpleNamespace(scored=scored, declined_frames=declined, ks=(0.5, 1.0))
    prod = counts.nontrades(sweep, NAME, 1.0, WINDOW)
    assert prod == {"candidates": 2, "f_skipped": 1, "reasons": {F_SKIP: 1, "none": 1},
                    "only_arm_f_skipped": 0, "only_production_f_skipped": 0}
    half = counts.nontrades(sweep, NAME, 0.5, WINDOW)
    assert half == {"candidates": 2, "f_skipped": 2, "reasons": {}, "only_arm_f_skipped": 1, "only_production_f_skipped": 0}
    e = counts.nontrades(sweep, NAME, E_ARM, WINDOW)
    assert e == {"candidates": 2, "f_skipped": 1, "reasons": {}, "only_arm_f_skipped": 0, "only_production_f_skipped": 0,
                 "any_k_f_skipped_share": 1.0}
    assert "any_k_f_skipped_share" not in half and "any_k_f_skipped_share" not in prod


def test_nontrades_are_none_without_production_or_with_a_missing_grid_k():
    scored = {(NAME, 0.5): [_cand([("E0", "2024-01-05", F_SKIP)])], (NAME, E_ARM): [_cand([("E0", "2024-01-05", None)])]}
    no_prod = types.SimpleNamespace(scored=scored, declined_frames={}, ks=(0.5,))
    got = counts.nontrades(no_prod, NAME, 0.5, WINDOW)
    assert got["only_arm_f_skipped"] is None and got["only_production_f_skipped"] is None
    no_k = types.SimpleNamespace(scored=scored, declined_frames={}, ks=(0.5, 1.0))
    assert counts.nontrades(no_k, NAME, E_ARM, WINDOW)["any_k_f_skipped_share"] is None


def test_evaluate_on_an_f_sweep_carries_the_views_and_is_json_safe(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = _f_sweep(fq)
    _month(sweep, chunk)
    res = tuning.evaluate(sweep, {n: 1.0 for n in sv.NAMES}, WINDOW)
    assert set(res["account"]["views"]) == {"F", "mid", "harsh", *VARIANTS}
    assert set(res["structures"][NAME]["views"]) == {"F", "mid", "harsh"}
    assert res["account"]["views"]["F"]["total_return"] == pytest.approx(res["account"]["total_return"], rel=1e-12)
    assert res["structures"][NAME]["nontrades"]["candidates"] == 2
    assert "nonpositive_debit" not in res["account"]

    def expect(rows, v):
        entry, exit_ = ("entry_cost", "exit_value") if v == "F" else (f"entry_cost_{v}", f"exit_value_{v}")
        return _total_return(_plain(rows, entry, exit_)), float((rows[exit_] / rows[entry] - 1.0).mean())

    acct_rows = sweep.rows([(n, 1.0) for n in sv.NAMES], WINDOW)
    for v in ("F", "mid", "harsh", *VARIANTS):
        tr, mr = expect(acct_rows, v)
        assert res["account"]["views"][v]["total_return"] == pytest.approx(tr, rel=1e-12)
        assert res["account"]["views"][v]["mean_ret"] == pytest.approx(mr, rel=1e-12)
    rows = sweep.rows([(NAME, 1.0)], WINDOW)
    for v in ("F", "mid", "harsh"):
        tr, mr = expect(rows, v)
        assert res["structures"][NAME]["views"][v]["total_return"] == pytest.approx(tr, rel=1e-12)
        assert res["structures"][NAME]["views"][v]["mean_ret"] == pytest.approx(mr, rel=1e-12)
    json.dumps(res, allow_nan=False)


def test_evaluate_on_a_non_f_sweep_has_no_f_keys(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = Sweep(ks=(1.0,), names=(NAME,), pricing=seam.load())
    sweep.add_month(chunk, EVENTS, None, output=True, sessions=SESS, sims=SIMS)
    res = tuning.evaluate(sweep, {n: 1.0 for n in sv.NAMES}, WINDOW)
    assert "views" not in res["account"] and "nonpositive_debit" not in res["account"]
    assert "views" not in res["structures"][NAME] and "nontrades" not in res["structures"][NAME]


def test_t8_ret_reads_f_on_an_f_sweep(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    pricing = seam.load(fills=ff.FFills(fq))
    pricing.decide = _all_taken
    sweep = DeviationSweep(ks=(1.0,), names=(NAME,), pricing=pricing)
    _month(sweep, chunk)
    got = {d["event_id"]: d["ret"] for d in sweep.deviation}
    rows = sweep.taken[(NAME, 1.0)][-1]
    assert len(got) == len(rows) == 2
    for r in rows.to_dict("records"):
        eid = str(r["event_id"])
        assert got[eid] == pytest.approx(r["exit_value"] / r["entry_cost"] - 1.0, rel=1e-12)
        assert not np.isclose(got[eid], r["exit_value_mid"] / r["entry_cost_mid"] - 1.0)


def test_t9_trades_and_stats_read_f_on_an_f_sweep(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = _f_sweep(fq)
    _month(sweep, chunk)
    trades = fallback._trades(sweep, lambda name: 1.0, WINDOW)
    assert len(trades) == 2
    assert trades["ret"].tolist() == pytest.approx((trades["exit_value"] / trades["entry_cost"] - 1.0).tolist(), rel=1e-12)
    stats = fallback._stats(trades, "taken")
    assert stats["taken_pnl"] == pytest.approx(trades["pnl_dollars"].mean(), rel=1e-12)
    assert stats["taken_ret"] == pytest.approx(trades["ret"].mean(), rel=1e-12)
    empty = fallback._trades(sweep, lambda name: 0.5, WINDOW)
    assert fallback._stats(empty, "taken") == {"taken_trades": 0, "taken_ret": None, "taken_pnl": None}


def test_fallback_accounts_carry_f_only_on_an_f_fallback(monkeypatch):
    def fake(sweep, by, window, production=None):
        acc = {"total_return": 0.1 if sweep.name == "fb" else 0.2, "views": {"F": {"total_return": 0.3}}}
        return {"account": acc, "structures": {n: {"nontrades": {"from": sweep.name, "production": getattr(production, "name", None)}} for n in sv.NAMES}}

    monkeypatch.setattr(fallback, "evaluate", fake)
    fb = types.SimpleNamespace(name="fb", f=True, factors={n: 1.0 for n in sv.NAMES})
    other = types.SimpleNamespace(name="base", f=True)
    got = fallback.accounts(fb, other, {"jun": WINDOW})["jun"]
    assert got["fallback"] == 0.1 and got["base"] == 0.2
    assert got["f"]["fallback"]["views"]["F"]["total_return"] == 0.3
    assert got["f"]["fallback"]["nontrades"][NAME] == {"from": "fb", "production": "base"}
    assert got["f"]["base"] == {"views": {"F": {"total_return": 0.3}},
                                "nontrades": {n: {"from": "base", "production": None} for n in sv.NAMES}}
    fb.f = False
    assert "f" not in fallback.accounts(fb, other, {"jun": WINDOW})["jun"]


def _stub(name, scored, declined, ks):
    """An F sweep stand-in with no taken rows (flat accounts); the non-trade counts read ``scored`` and ``declined``."""
    empty = _f_trades().iloc[:0]
    return types.SimpleNamespace(name=name, f=True, ladder=False, ks=ks, scored=scored, declined_frames=declined,
                                 factors={n: 1.0 for n in sv.NAMES}, rows=lambda pairs, window: empty)


def test_fallback_nontrades_reach_the_receipted_summary_with_base_as_production():
    """Real tuning.evaluate: the fallback's and base's non-trade counts survive accounts(), and the production is base, not fb."""
    fb = _stub(NAME, {(NAME, E_ARM): [_cand([("E0", "2024-01-05", None), ("E1", "2024-01-06", F_SKIP), ("E4", "2024-01-08", None)])]},
               {(NAME, E_ARM): [_cand([("D1", "2024-01-10", "below_cutoff"), ("D2", "2024-01-11", np.nan)])]}, (1.0,))
    base = _stub("base", {(NAME, 1.0): [_cand([("E0", "2024-01-05", None), ("E2", "2024-01-06", F_SKIP),
                                              ("E3", "2024-01-07", F_SKIP), ("E4", "2024-01-08", None)])]},
                 {(NAME, 1.0): [_cand([("D9", "2024-01-09", "below_cutoff")])]}, (1.0,))
    got = fallback.accounts(fb, base, {"jun": WINDOW})["jun"]
    assert got["f"]["fallback"]["nontrades"][NAME] == {"candidates": 3, "f_skipped": 1, "reasons": {"below_cutoff": 1, "none": 1},
                                                       "only_arm_f_skipped": 1, "only_production_f_skipped": 2, "any_k_f_skipped_share": None}
    assert got["f"]["base"]["nontrades"][NAME] == {"candidates": 4, "f_skipped": 2, "reasons": {"below_cutoff": 1},
                                                   "only_arm_f_skipped": 0, "only_production_f_skipped": 0}
    assert set(got["f"]["fallback"]["nontrades"]) == set(sv.NAMES)
    json.dumps(got, allow_nan=False)
