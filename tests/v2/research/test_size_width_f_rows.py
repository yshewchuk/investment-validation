"""Tests for the F-primary taken rows (experiments/size_width_f_rows.py) through a real Sweep on synthetic chains.

Expected values are computed here from the stored quotes, the F leg frames and the lots rule, never from the module
under test. Fixtures and helpers come from the F quote and F fills tests.
"""
import math

import numpy as np
import pandas as pd
import pytest

from engine.portfolio import CAPITAL_PER_TRADE, CONTRACT_MULTIPLIER
from experiments import fair_value_fills as fvf
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_shortvol as sv
from experiments import size_width_f_fills as ff
from experiments import size_width_f_rows as fr
from experiments import size_width_pricing as seam
from experiments.size_pnl_straddle import EXIT_UNQUOTED
from experiments.size_width_sweep import _TAKEN, Sweep
from tests.v2.research.test_size_width_f_fills import (
    EVENTS, NAME, SIMS, StubQuotes, _debits, _exit_status, _proceeds, _tight_frames, _trade)
from tests.v2.research.test_size_width_f_quotes import (
    DEFAULT_PLAN, ENTRY, EXIT, _legs_frame, _listing, _setup, _ts)

EVENT_INDEX = {"E0": 0, "E1": 1}
SESS = {e: (_ts(ENTRY), _ts(EXIT)) for e in EVENT_INDEX}
WINDOW = (_ts("2024-01-01"), _ts("2025-01-01"))


def _all_taken(scored):
    """Stands in for decide: every scored row is taken."""
    out = scored.copy()
    ok = (out["status"] == "scored").to_numpy()
    out.loc[ok, "status"] = "taken"
    out.loc[ok, "reason"] = None
    return out


def _e0_taken(scored):
    """E0 taken, E1 declined, so the month has a declined row beside the taken ones."""
    out = _all_taken(scored)
    later = (out["event_id"] == "E1").to_numpy()
    out.loc[later, "status"] = "not_taken"
    out.loc[later, "reason"] = "below_cutoff"
    return out


def _sweep(quotes, *, choices=False, decide=_all_taken):
    pricing = seam.load(fills=ff.FFills(quotes))
    pricing.decide = decide
    return Sweep(ks=(1.0,), names=(NAME,), pricing=pricing, choices=choices)


def _month(sweep, chunk, output=True):
    return sweep.add_month(chunk, EVENTS, None, output=output, sessions=SESS, sims=SIMS)


def _no_oi_frames():
    """Tight quotes with open interest unknown on the entry date, so the thin-OI terms differ across the variants."""
    frames = _tight_frames()
    for ticker in ("AAA", "BBB"):
        key = (ticker, _ts(ENTRY))
        frames[key] = frames[key].assign(open_interest=np.nan)
    return frames


def _traded(chunk, i):
    trade, _ = _trade(chunk, i)
    return [leg for leg in trade.legs if leg["qty"] > 0]


def _quote(chunk, idx, when):
    return float(chunk[f"bid_{when}"][idx]), float(chunk[f"ask_{when}"][idx])


def _entry_debit(chunk, legs, harsh):
    """Per-share debit at the stored entry quotes: mid for both sides, or buy at the ask and sell at the bid (harsh)."""
    total = 0.0
    for leg in legs:
        bid, ask = _quote(chunk, leg["idx"], "e")
        buy, sell = (ask, bid) if harsh else ((bid + ask) / 2, (bid + ask) / 2)
        total += leg["qty"] * (buy if leg["side"] == "buy" else -sell)
    return total


def _exit_value(chunk, legs, harsh):
    """Per-share value of closing the legs at the stored exit quotes: a buy sells at the bid (harsh) or mid, a sell buys
    at the ask (harsh) or mid."""
    total = 0.0
    for leg in legs:
        bid, ask = _quote(chunk, leg["idx"], "x")
        mid = (bid + ask) / 2
        if leg["side"] == "buy":
            total += leg["qty"] * (bid if harsh else mid)
        else:
            total -= leg["qty"] * (ask if harsh else mid)
    return total


def _lots(debit):
    return max(1, math.floor(CAPITAL_PER_TRADE / (debit * CONTRACT_MULTIPLIER)))


def _pnl(exit_value, debit):
    return (exit_value - debit) * CONTRACT_MULTIPLIER * _lots(debit)


def _ladder(chunk):
    """CurrentFills score of the ladder at alpha 0.5, indexed by event: the rung the ladder gives each event."""
    scored = seam.PRICING.score_events(chunk, EVENTS, SIMS, NAME, alpha=0.5)
    return seam.PRICING.score(chunk, EVENTS, scored.assign(status="taken"), NAME, 0.5).set_index("event_id")


def test_f_sweep_returns_taken_then_f_columns_then_arm(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = _sweep(fq)
    assert sweep.f is True
    out = _month(sweep, chunk)
    assert list(out.columns) == _TAKEN + fr.F_COLS + ["arm"]
    assert set(out["arm"]) == {"1.0", "E"}
    assert len(out) == 4 and set(out["event_id"]) == set(EVENT_INDEX)


def test_empty_month_and_empty_rows_carry_the_taken_columns(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = _sweep(fq)
    assert list(_month(sweep, chunk, output=False).columns) == _TAKEN + fr.F_COLS + ["arm"]
    rows = sweep.rows([(NAME, 1.0)], WINDOW)
    assert rows.empty and list(rows.columns) == sweep.taken_cols == _TAKEN + fr.F_COLS


@pytest.mark.parametrize("make", [lambda: sv, lambda: seam.load()], ids=["shortvol", "current-fills"])
def test_non_f_pricing_adds_no_column(tmp_path, make):
    _, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = Sweep(ks=(1.0,), names=(NAME,), pricing=make())
    assert sweep.f is False
    assert list(_month(sweep, chunk).columns) == _TAKEN + ["arm"]
    assert list(sweep.rows([(NAME, 1.0)], WINDOW).columns) == _TAKEN


def test_primary_columns_are_the_f_debit_proceeds_and_their_pnl(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    out = _month(_sweep(fq), chunk)
    assert len(out) == 4
    for row in out.to_dict("records"):
        traded = _traded(chunk, EVENT_INDEX[row["event_id"]])
        i = EVENT_INDEX[row["event_id"]]
        debit = _debits(_legs_frame(fq, chunk, traded, i=i))[2]
        proceeds = _proceeds(_legs_frame(fq, chunk, traded, i=i, at_exit=True))
        assert row["entry_cost"] == pytest.approx(debit, abs=1e-9)
        assert row["exit_value"] == pytest.approx(proceeds, abs=1e-9)
        assert row["pnl_dollars"] == pytest.approx(_pnl(proceeds, debit), rel=1e-9)


def test_mid_and_harsh_views_price_the_stored_quotes_and_pair_with_their_own_debits(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    out = _month(_sweep(fq), chunk)
    ladder = _ladder(chunk)
    assert len(out) == 4
    for row in out.to_dict("records"):
        traded = _traded(chunk, EVENT_INDEX[row["event_id"]])
        mid_in, harsh_in = _entry_debit(chunk, traded, False), _entry_debit(chunk, traded, True)
        mid_out, harsh_out = _exit_value(chunk, traded, False), _exit_value(chunk, traded, True)
        assert row["entry_cost_mid"] == pytest.approx(mid_in, abs=1e-9)
        assert row["exit_value_mid"] == pytest.approx(mid_out, abs=1e-9)
        assert row["pnl_dollars_mid"] == pytest.approx(_pnl(mid_out, mid_in), rel=1e-9)
        assert row["entry_cost_harsh"] == pytest.approx(harsh_in, abs=1e-9)
        assert row["exit_value_harsh"] == pytest.approx(harsh_out, abs=1e-9)
        assert row["pnl_dollars_harsh"] == pytest.approx(_pnl(harsh_out, harsh_in), rel=1e-9)
        assert row["rung_ladder"] == ladder.loc[row["event_id"], "rung"]
        assert not np.isclose(row["entry_cost_mid"], row["entry_cost"])  # F's debit is not the mid's


def test_sensitivities_are_per_variant_with_their_own_debit_and_lots(tmp_path):
    fq, chunk = _setup(tmp_path, _no_oi_frames(), DEFAULT_PLAN)
    out = _month(_sweep(fq), chunk)
    assert len(out) == 4
    for row in out.to_dict("records"):
        i = EVENT_INDEX[row["event_id"]]
        traded = _traded(chunk, i)
        entry = fvf.structure_entry(_legs_frame(fq, chunk, traded, i=i))
        exit_ = fvf.structure_exit(_legs_frame(fq, chunk, traded, i=i, at_exit=True))
        assert not np.isclose(entry.sensitivity_debits["unknown0"], entry.sensitivity_debits["unknown1"])
        for v in fvf.VARIANTS:
            debit, proceeds = entry.sensitivity_debits[v], exit_.sensitivity_proceeds[v]
            assert row[f"entry_cost_{v}"] == pytest.approx(debit, abs=1e-9)
            assert row[f"exit_value_{v}"] == pytest.approx(proceeds, abs=1e-9)
            assert row[f"pnl_dollars_{v}"] == pytest.approx(_pnl(proceeds, debit), rel=1e-9)


def test_f_unquoted_exit_zeroes_primary_and_sensitivities_but_not_mid_or_harsh(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    out = _month(_sweep(StubQuotes(fq, _exit_status("unpriceable"))), chunk)
    ladder = _ladder(chunk)
    assert len(out) == 4
    for row in out.to_dict("records"):
        i = EVENT_INDEX[row["event_id"]]
        traded = _traded(chunk, i)
        debit = _debits(_legs_frame(fq, chunk, traded, i=i))[2]
        assert row["rung"] == EXIT_UNQUOTED
        assert row["exit_value"] == 0.0 and row["pnl_dollars"] == pytest.approx(_pnl(0.0, debit), rel=1e-9)
        assert row["rung_ladder"] == ladder.loc[row["event_id"], "rung"] != EXIT_UNQUOTED
        mid_out, harsh_out = _exit_value(chunk, traded, False), _exit_value(chunk, traded, True)
        assert mid_out != 0.0
        assert row["exit_value_mid"] == pytest.approx(mid_out, abs=1e-9)
        assert row["exit_value_harsh"] == pytest.approx(harsh_out, abs=1e-9)
        sens = fvf.structure_entry(_legs_frame(fq, chunk, traded, i=i)).sensitivity_debits
        for v in fvf.VARIANTS:
            assert row[f"exit_value_{v}"] == 0.0
            assert row[f"pnl_dollars_{v}"] == pytest.approx(_pnl(0.0, sens[v]), rel=1e-9)


def test_ladder_rung3_zeroes_primary_and_every_sensitivity_exit(tmp_path):
    _, probe = _setup(tmp_path / "probe", _tight_frames(), DEFAULT_PLAN)
    strike = float(probe["strike"][_traded(probe, 0)[0]["idx"]])
    frames = _tight_frames()
    frames[("AAA", _ts(EXIT))] = _listing("AAA", EXIT, skip=[("P", strike)])  # E0 loses a traded exit contract
    fq, chunk = _setup(tmp_path / "rung3", frames, DEFAULT_PLAN)
    out = _month(_sweep(fq), chunk)
    ladder = _ladder(chunk)
    assert ladder.loc["E0", "rung"] == EXIT_UNQUOTED  # the ladder's own rung 3 for E0
    row = out[(out["event_id"] == "E0") & (out["arm"] == "1.0")].iloc[0]
    traded = _traded(chunk, 0)
    debit = _debits(_legs_frame(fq, chunk, traded, i=0))[2]
    assert row["rung"] == EXIT_UNQUOTED and row["rung_ladder"] == EXIT_UNQUOTED
    assert row["exit_value"] == 0.0 and row["pnl_dollars"] == pytest.approx(_pnl(0.0, debit), rel=1e-9)
    assert row["exit_value_mid"] == 0.0 and row["exit_value_harsh"] == 0.0
    sens = fvf.structure_entry(_legs_frame(fq, chunk, traded, i=0)).sensitivity_debits
    for v in fvf.VARIANTS:
        assert row[f"exit_value_{v}"] == 0.0
        assert row[f"pnl_dollars_{v}"] == pytest.approx(_pnl(0.0, sens[v]), rel=1e-9)


def test_exit_flags_reach_the_taken_rows(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    out = _month(_sweep(StubQuotes(fq, _exit_status("rate_unknown"))), chunk)
    assert out["exit_fair_mid"].dtype == bool and out["exit_negative"].dtype == bool
    assert out["exit_fair_mid"].all()


def test_arm_e_choice_frame_carries_the_current_fills_mid_exit(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    sweep = _sweep(fq, choices=True)
    _month(sweep, chunk)
    choices = pd.concat(sweep.fresh_choices, ignore_index=True)
    taken = choices[choices["status"] == "taken"].set_index("event_id")
    assert set(taken.index) == set(EVENT_INDEX)
    for eid, i in EVENT_INDEX.items():
        assert taken.loc[eid, "exit_value_mid"] == pytest.approx(_exit_value(chunk, _traded(chunk, i), False), abs=1e-9)


def test_f_taken_and_declined_rows_write_as_decisions(tmp_path):
    fq, chunk = _setup(tmp_path / "chunk", _tight_frames(), DEFAULT_PLAN)
    sweep = _sweep(fq, decide=_e0_taken)
    taken = _month(sweep, chunk)
    declined = sweep.declined_rows()
    assert len(taken) == 2 and len(declined) == 2
    taken = taken.assign(model_id="M0", book="short_vol", arm_or_rule=taken["arm"], status="taken", reason=None)
    declined = declined.assign(model_id="M0", book="short_vol", arm_or_rule=lambda f: f["arm"])
    _, written = decisions._write_decisions(tmp_path / "m.parquet", [taken, declined])
    assert written == 4
    back = pd.read_parquet(tmp_path / "m.parquet")
    assert back["exit_fair_mid"].notna().sum() == 2
