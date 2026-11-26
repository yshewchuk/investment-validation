"""Tests for the F fills object (experiments/size_width_f_fills.py): constants, entry and exit rules, reuse and the seam.

Synthetic chains only: the quote adapter's fixtures feed a real FQuotes and a ladder chunk. Expected values are computed
here from the rules, never from the module under test.
"""
import math
import types

import numpy as np
import pandas as pd
import pytest

from engine.fills import MIN_MEANINGFUL_COST, FillModel
from experiments import fair_value_fills as fvf
from experiments import size_pnl_shortvol as sv
from experiments import size_width_f_fills as ff
from experiments import size_width_pricing as seam
from experiments.fair_value_inputs import FairValueRefusal
from experiments.size_pnl_pool import structure_value
from experiments.size_pnl_straddle import EXIT_UNQUOTED
from experiments.size_width_fills import CurrentFills, Exit, Skip, Trade
from tests.v2.research.test_size_width_f_quotes import (
    DEFAULT_PLAN, ENTRY, EXIT, PRED, STRIKES, _base, _frames, _idx, _legs_frame, _listing, _mid, _reader, _set, _setup,
    _ts)

NAME = "CND-PS"
TIGHT = {(right, k): (_mid(k) - 0.005, _mid(k) + 0.005) for k in STRIKES for right in ("P", "C")}
SIMS_GRID = np.arange(60.0, 141.0)
SIMS = {e: (SIMS_GRID, 0.001 * (SIMS_GRID - 100.0) ** 2) for e in ("E0", "E1")}
EVENTS = pd.DataFrame({"event_id": ["E0", "E1"], "event_date": [_ts(ENTRY) + pd.Timedelta(days=1)] * 2,
                       "pred_move": [PRED, PRED], "mcap_usd": [1e10, 1e10]})


def _tight(ticker):
    """Entry-date chain with quotes 0.005 either side of mid, so the F guard passes for CND-PS."""
    return _listing(ticker, ENTRY, quotes=TIGHT)


def _tight_frames():
    frames = _frames()
    for ticker in ("AAA", "BBB"):
        frames[(ticker, _ts(ENTRY))] = _tight(ticker)
    return frames


def _trade(chunk, i, alpha=0.5):
    """(Trade, priced) for event ``i``'s CND-PS at ``alpha``, resolved as score_events resolves it."""
    priced, legs, reason = sv.resolve_priced(chunk, i, NAME, PRED, FillModel(alpha))
    assert reason is None
    return Trade(chunk, i, str(chunk["event_id"][i]), NAME, tuple(legs)), priced


def _pair(chunk):
    """Event E0 as a two-leg trade with no reference leg: buy the 100 put, sell the 95 put, one contract each."""
    legs = ({"name": "long", "strike": 100.0, "side": "buy", "qty": 1.0, "idx": _idx(chunk, 0, "P", 100.0)},
            {"name": "short", "strike": 95.0, "side": "sell", "qty": 1.0, "idx": _idx(chunk, 0, "P", 95.0)})
    return Trade(chunk, 0, str(chunk["event_id"][0]), NAME, legs)


class StubQuotes:
    """A real FQuotes that records each legs() call as (i, exit, [leg idx]), then lets ``edit(frame, exit)`` overwrite
    columns of the leg frame it returns."""

    def __init__(self, quotes, edit=None):
        self.quotes, self.edit, self.calls = quotes, edit, []

    def legs(self, chunk, i, resolved, *, exit=False):
        self.calls.append((i, exit, [leg["idx"] for leg in resolved]))
        out = self.quotes.legs(chunk, i, resolved, exit=exit)
        return out if self.edit is None else self.edit(out, exit)


def _count(stub, exit):
    return sum(1 for _, flag, _ in stub.calls if flag is exit)


def _entry_edit(statuses=(), **cols):
    """Edit for entry calls only: the first rows take ``statuses`` (the rest stay ok), then ``cols`` are assigned."""
    def edit(frame, exit):
        if exit:
            return frame
        frame = frame.assign(status=list(statuses) + ["ok"] * (len(frame) - len(statuses)))
        return frame.assign(**cols)
    return edit


def _exit_status(*statuses):
    """Edit for exit calls only: the first traded rows take ``statuses`` (the rest stay ok)."""
    def edit(frame, exit):
        if not exit:
            return frame
        return frame.assign(status=list(statuses) + ["ok"] * (len(frame) - len(statuses)))
    return edit


def _pair_frame(values):
    """The two-leg entry frame the guard reads: row 0 the buy, row 1 the sell, quantity 1 each."""
    return pd.DataFrame(values).assign(side=["buy", "sell"], qty=[1.0, 1.0])


def _debits(frame):
    """Per-share (debit_mid, debit_fair, debit_F, spread_sum) of the positive legs: buys minus sells."""
    f = frame[frame["qty"] > 0]
    qty, buy = f["qty"].to_numpy(float), (f["side"] == "buy").to_numpy()
    mid = ((f["eff_bid"] + f["eff_ask"]) / 2).to_numpy(float)
    fair = f["V"].to_numpy(float)
    spread = (f["eff_ask"] - f["eff_bid"]).to_numpy(float)
    return (
        float(np.sum(qty * np.where(buy, mid, -mid))),
        float(np.sum(qty * np.where(buy, fair, -fair))),
        float(np.sum(qty * np.where(buy, f["fill_buy"].to_numpy(float), -f["fill_sell"].to_numpy(float)))),
        float(np.sum(qty * spread)),
    )


def _guards(frame):
    """Which of the three guard conditions of the rules hold for the entry leg frame."""
    mid, fair, debit_f, spread = _debits(frame)
    return {
        "not_a_debit": mid <= MIN_MEANINGFUL_COST or fair <= MIN_MEANINGFUL_COST,
        "fill_over_fair": debit_f - fair > 0.25 * fair,
        "spread_over_debit": spread > debit_f,
    }


def _fallback_fill(row, fill):
    """The V = mid fill of a fallback leg. ``fill`` is the side of the fill ("buy" or "sell"), not of the leg."""
    mid = (row["eff_bid"] + row["eff_ask"]) / 2
    thin = row["thin_size_buy"] if fill == "buy" else row["thin_size_sell"]
    c = min(1.0, 0.25 + 0.50 * min(row["s"] / 0.50, 1.0) + 0.15 * thin + 0.10 * row["thin_oi"])
    if fill == "buy":
        return mid + c * (row["eff_ask"] - mid)
    return mid - c * (mid - row["eff_bid"])


def _proceeds(frame, fallback=()):
    """Exit proceeds per share of the positive legs: an opening buy closes at a sell fill, an opening sell at a buy
    fill. Positions in ``fallback`` take the V = mid fill; the other legs use their stored fills."""
    total = 0.0
    f = frame[frame["qty"] > 0].reset_index(drop=True)
    for pos, row in f.iterrows():
        fill = "sell" if row["side"] == "buy" else "buy"
        if pos in fallback:
            px = _fallback_fill(row, fill)
        else:
            px = row["fill_sell"] if fill == "sell" else row["fill_buy"]
        total += row["qty"] * (px if fill == "sell" else -px)
    return float(total)


FAR = {"fill_buy": lambda f: f["V"] + 10.0, "fill_sell": lambda f: f["V"] - 10.0}  # debit_F exceeds debit_fair by 10 per leg
NAD_ONLY = dict(eff_bid=[0.49, 0.49], eff_ask=[0.51, 0.51], V=[1.0, 0.5], fill_buy=[1.0, 1.0], fill_sell=[0.5, 0.5])
FOF_ONLY = dict(eff_bid=[0.99, 0.49], eff_ask=[1.01, 0.51], V=[1.0, 0.5], fill_buy=[1.0, 1.0], fill_sell=[0.0, 0.0])
SOD_ONLY = dict(eff_bid=[0.5, 0.49], eff_ask=[1.5, 0.51], V=[1.0, 0.5], fill_buy=[1.0, 1.0], fill_sell=[0.5, 0.5])
NAD_AND_FOF = dict(eff_bid=[0.49, 0.49], eff_ask=[0.51, 0.51], V=[1.0, 0.5], fill_buy=[1.0, 1.0], fill_sell=[0.0, 0.0])
FOF_AND_SOD = dict(eff_bid=[0.5, 0.49], eff_ask=[1.5, 0.51], V=[1.0, 0.5], fill_buy=[1.0, 1.0], fill_sell=[0.0, 0.0])
AT_MIN = dict(eff_bid=[MIN_MEANINGFUL_COST, 0.0], eff_ask=[MIN_MEANINGFUL_COST, 0.0], V=[1.0, 0.0],
              fill_buy=[1.0, 1.0], fill_sell=[0.0, 0.0])

STATUS_CASES = [
    pytest.param(("unpriceable",), "f_unpriceable", False, id="unpriceable"),
    pytest.param(("rate_unknown",), "f_rate_unknown", False, id="rate-unknown"),
    pytest.param(("no_fair_value",), "f_no_fair_value", False, id="no-fair-value"),
    pytest.param(("no_fair_value", "unpriceable"), "f_unpriceable", False, id="no-fair-value-then-unpriceable"),
    pytest.param(("no_fair_value", "rate_unknown"), "f_rate_unknown", False, id="no-fair-value-then-rate-unknown"),
    pytest.param(("rate_unknown", "unpriceable"), "f_unpriceable", False, id="rate-unknown-then-unpriceable"),
    pytest.param(("no_fair_value",), "f_no_fair_value", True, id="status-beats-fills-far-above-v"),
]

GUARD_CASES = [
    pytest.param(NAD_ONLY, {"not_a_debit"}, "f_not_a_debit", id="only-not-a-debit"),
    pytest.param(FOF_ONLY, {"fill_over_fair"}, "f_fill_over_fair", id="only-fill-over-fair"),
    pytest.param(SOD_ONLY, {"spread_over_debit"}, "f_spread_over_debit", id="only-spread-over-debit"),
    pytest.param(NAD_AND_FOF, {"not_a_debit", "fill_over_fair"}, "f_not_a_debit", id="not-a-debit-beats-fill-over-fair"),
    pytest.param(FOF_AND_SOD, {"fill_over_fair", "spread_over_debit"}, "f_fill_over_fair",
                 id="fill-over-fair-beats-spread-over-debit"),
    pytest.param(AT_MIN, {"not_a_debit"}, "f_not_a_debit", id="debit-mid-exactly-the-minimum"),
]


def test_reason_tuple_and_exit_flags_are_the_documented_constants():
    assert ff.ENTRY_REASONS == ("f_unpriceable", "f_rate_unknown", "f_no_fair_value", "f_not_a_debit",
                                "f_fill_over_fair", "f_spread_over_debit")
    assert ff.FFills.exit_flags == ff.EXIT_FLAGS == ("exit_fair_mid", "exit_negative")
    assert not set(ff.EXIT_FLAGS) & set(sv._OUT_COLS)


def test_accepted_entry_returns_the_f_debit_and_the_sv_relative_spread(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    trade, priced = _trade(chunk, 0)
    traded = [leg for leg in trade.legs if leg["qty"] > 0]
    _, fair, debit_f, _ = _debits(_legs_frame(fq, chunk, traded))
    assert debit_f > fair > 0
    got = ff.FFills(fq).entry(trade, priced)
    assert not isinstance(got, Skip) and got is not None
    assert got[0] == pytest.approx(debit_f)
    assert got[1] == sv.entry(priced)[1]


def test_entry_debit_is_alpha_free_while_the_sv_cost_moves_with_alpha(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    trade, _ = _trade(chunk, 0)
    traded = [leg for leg in trade.legs if leg["qty"] > 0]
    debit_f = _debits(_legs_frame(fq, chunk, traded))[2]
    costs, got = set(), {}
    for alpha in (0.0, 0.5, 1.0):
        t, p = _trade(chunk, 0, alpha)
        costs.add(sv.entry(p)[0])
        got[alpha] = ff.FFills(fq).entry(t, p)
    assert len(costs) == 3  # the sv mark moves with alpha
    for alpha in (0.0, 0.5, 1.0):
        assert not isinstance(got[alpha], Skip)
        assert got[alpha][0] == pytest.approx(debit_f)


def test_accepted_entry_and_exit_match_hand_computed_literals(tmp_path):
    """Buy one put filled at 1.04 (V 1.00, quote 0.90/1.10), sell one filled at 0.48 (V 0.50, quote 0.45/0.55).

    debit_mid = 1.00 - 0.50 = 0.50; debit_fair = 0.50; debit_F = 1.04 - 0.48 = 0.56 (0.06 over fair, under the
    0.125 limit); spread_sum = 0.20 + 0.10 = 0.30 <= 0.56: accepted at 0.56. Closing sells the long at 0.96 and buys
    the short back at 0.52: proceeds_F = 0.96 - 0.52 = 0.44.
    """
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    _, priced = _trade(chunk, 0)
    cols = dict(eff_bid=[0.90, 0.45], eff_ask=[1.10, 0.55], V=[1.00, 0.50], fill_buy=[1.04, 0.52], fill_sell=[0.96, 0.48])

    def edit(frame, exit):
        return frame.assign(status=["ok", "ok"], **cols)

    model = ff.FFills(StubQuotes(fq, edit))
    trade = _pair(chunk)
    cost, rel = model.entry(trade, priced)
    assert cost == pytest.approx(0.56, abs=1e-12)
    assert rel == sv.entry(priced)[1]
    got = model.exit_value(trade, list(trade.legs), 0.5)
    assert got.value == pytest.approx(0.44, abs=1e-12)
    assert got == Exit(got.value, frozenset(), False)


def test_quantities_weight_the_debit_and_the_proceeds(tmp_path):
    """Two long puts and one short: debit_F = 2 * 1.04 - 0.48 = 1.60 (fair 1.50, spread sum 0.50: accepted);
    proceeds_F = 2 * 0.96 - 0.52 = 1.40."""
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    _, priced = _trade(chunk, 0)
    cols = dict(eff_bid=[0.90, 0.45], eff_ask=[1.10, 0.55], V=[1.00, 0.50], fill_buy=[1.04, 0.52], fill_sell=[0.96, 0.48])

    def edit(frame, exit):
        return frame.assign(status=["ok", "ok"], **cols)

    pair = _pair(chunk)
    trade = pair._replace(legs=({**pair.legs[0], "qty": 2.0}, pair.legs[1]))
    model = ff.FFills(StubQuotes(fq, edit))
    assert model.entry(trade, priced)[0] == pytest.approx(1.60, abs=1e-12)
    assert model.exit_value(trade, list(trade.legs), 0.5).value == pytest.approx(1.40, abs=1e-12)


def test_fair_mid_exit_and_negative_exit_match_hand_computed_literals(tmp_path):
    """The long leg has no rate at exit: V = mid = 1.00 of 0.80/1.20, s = 0.40, thin size 1, thin OI 0.5, so
    c = 0.25 + 0.50 * 0.8 + 0.15 + 0.05 = 0.85 and it sells at 1.00 - 0.85 * 0.20 = 0.83 (its stored fill 5.0 is
    ignored). The short leg is bought back at its stored 1.50: proceeds_F = 0.83 - 1.50 = -0.67, kept and flagged."""
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    cols = dict(eff_bid=[0.80, 1.40], eff_ask=[1.20, 1.60], V=[1.10, 1.50], s=[0.40, 0.10], thin_size_buy=[0.0, 0.0],
                thin_size_sell=[1.0, 0.0], thin_oi=[0.5, 0.0], unknown_bid_size=[False, False],
                unknown_ask_size=[False, False], unknown_oi=[True, False], fill_buy=[5.0, 1.50], fill_sell=[5.0, 1.45])

    def edit(frame, exit):
        return frame.assign(status=["rate_unknown", "ok"], **cols)

    trade = _pair(chunk)
    got = ff.FFills(StubQuotes(fq, edit)).exit_value(trade, list(trade.legs), 0.5)
    assert got.value == pytest.approx(-0.67, abs=1e-12)
    assert got == Exit(got.value, frozenset({"exit_fair_mid", "exit_negative"}), False)


@pytest.mark.parametrize("statuses, reason, far", STATUS_CASES)
def test_status_reason_is_named_before_the_guard(tmp_path, statuses, reason, far):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade, priced = _trade(chunk, 0)
    if far:  # the guard columns would also fail on their own
        traded = [leg for leg in trade.legs if leg["qty"] > 0]
        assert any(_guards(_legs_frame(fq, chunk, traded).assign(**FAR)).values())
    stub = StubQuotes(fq, _entry_edit(statuses, **(FAR if far else {})))
    assert ff.FFills(stub).entry(trade, priced) == Skip(reason)


@pytest.mark.parametrize("values, holds, reason", GUARD_CASES)
def test_guard_conditions_run_in_order_and_the_first_that_holds_is_named(tmp_path, values, holds, reason):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    _, priced = _trade(chunk, 0)
    assert {name for name, hit in _guards(_pair_frame(values)).items() if hit} == holds
    got = ff.FFills(StubQuotes(fq, _entry_edit(**values))).entry(_pair(chunk), priced)
    assert got == Skip(reason)


def test_reference_leg_of_quantity_zero_is_never_read(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade, priced = _trade(chunk, 0)
    assert any(leg["qty"] == 0 for leg in trade.legs)
    stub = StubQuotes(fq)
    ff.FFills(stub).entry(trade, priced)
    assert stub.calls == [(0, False, [leg["idx"] for leg in trade.legs if leg["qty"] > 0])]


def test_sv_entry_refusal_comes_first_and_no_quotes_are_read(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade, _ = _trade(chunk, 0)
    stub = StubQuotes(fq)
    priced = types.SimpleNamespace(cost=float("nan"), legs=[])
    assert ff.FFills(stub).entry(trade, priced) is None
    assert stub.calls == []


def test_entry_never_reads_exit_date_rows(tmp_path):
    frames = _tight_frames()
    real = _reader(_base(frames))

    def read(expiries):
        if (expiries["obs_date"] == _ts(EXIT)).any():
            raise AssertionError("entry asked for exit-date rows")
        return real(expiries)

    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN, read=read)
    stub = StubQuotes(fq)
    model = ff.FFills(stub)
    for i in (0, 1):
        trade, priced = _trade(chunk, i)
        got = model.entry(trade, priced)
        assert got is not None and not isinstance(got, Skip)
        model.entry_result(trade)
    assert stub.calls and all(not flag for _, flag, _ in stub.calls)


def test_exit_with_every_leg_ok_is_the_stored_proceeds_for_any_alpha_and_mid(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade, _ = _trade(chunk, 0)
    traded = [leg for leg in trade.legs if leg["qty"] > 0]
    want = _proceeds(_legs_frame(fq, chunk, traded, at_exit=True))
    for alpha in (0.0, 0.5):
        for mid in (False, True):
            got = ff.FFills(fq).exit_value(trade, traded, alpha, mid=mid)
            assert got.value == pytest.approx(want)
            assert got.flags == frozenset() and got.unquoted is False


@pytest.mark.parametrize("status", ["rate_unknown", "no_fair_value"])
def test_exit_fair_mid_leg_takes_v_at_mid_and_flags_the_exit(tmp_path, status):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade, _ = _trade(chunk, 0)
    traded = [leg for leg in trade.legs if leg["qty"] > 0]
    frame = _legs_frame(fq, chunk, traded, at_exit=True)
    got = ff.FFills(StubQuotes(fq, _exit_status(status))).exit_value(trade, traded, 0.5)
    assert got.value == pytest.approx(_proceeds(frame, fallback=(0,)))
    want = _proceeds(frame, fallback=(0,))
    assert got.flags == frozenset({"exit_fair_mid"} | ({"exit_negative"} if want < 0 else set()))
    assert got.unquoted is False
    assert abs(got.value - _proceeds(frame)) > 1e-9  # the fallback is exercised, not a no-op


def test_exit_below_zero_is_kept_and_flagged_exit_negative(tmp_path):
    frames = _frames()
    frames[("AAA", _ts(EXIT))] = _listing("AAA", EXIT, quotes={("P", 95.0): (49.9, 50.0)})
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN)
    trade = _pair(chunk)
    traded = list(trade.legs)
    want = _proceeds(_legs_frame(fq, chunk, traded, at_exit=True))
    got = ff.FFills(fq).exit_value(trade, traded, 0.5)
    assert want < 0
    assert got.value == pytest.approx(want)
    assert got.flags == frozenset({"exit_negative"}) and got.unquoted is False


def test_exit_leg_not_listed_on_the_exit_date_is_unquoted_with_no_flags(tmp_path):
    frames = _frames()
    frames[("AAA", _ts(EXIT))] = _listing("AAA", EXIT, skip=[("P", 95.0)])
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN)
    trade = _pair(chunk)
    assert ff.FFills(fq).exit_value(trade, list(trade.legs), 0.5) == Exit(0.0, frozenset(), True)


def test_unpriceable_leg_beside_a_fallback_leg_is_unquoted_with_no_flags(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade = _pair(chunk)
    stub = StubQuotes(fq, _exit_status("unpriceable", "rate_unknown"))
    assert ff.FFills(stub).exit_value(trade, list(trade.legs), 0.5) == Exit(0.0, frozenset(), True)


def test_entry_and_exit_results_are_the_structure_results_of_the_positive_legs(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade, _ = _trade(chunk, 0)
    traded = [leg for leg in trade.legs if leg["qty"] > 0]
    model = ff.FFills(fq)
    assert model.entry_result(trade) == fvf.structure_entry(_legs_frame(fq, chunk, traded))
    assert model.exit_result(trade) == fvf.structure_exit(_legs_frame(fq, chunk, traded, at_exit=True))
    assert set(model.entry_result(trade).sensitivity_debits) == set(fvf.VARIANTS)


def test_results_are_kept_per_chunk_and_recomputed_for_a_new_chunk(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    stub = StubQuotes(fq)
    model = ff.FFills(stub)
    t0, p0 = _trade(chunk, 0)
    t1, p1 = _trade(chunk, 1)
    for _ in range(3):
        model.entry(t0, p0)
    assert _count(stub, False) == 1
    model.entry(t1, p1)  # another event index on the same chunk
    assert _count(stub, False) == 2
    model.entry(t0, p0)  # the chunk's memo keeps both events
    assert _count(stub, False) == 2
    pair = _pair(chunk)  # other resolved legs of the same event are a separate result
    assert model.entry_result(pair) != model.entry_result(t0)
    assert _count(stub, False) == 3
    model.entry_result(pair)
    assert _count(stub, False) == 3
    model.entry(t0._replace(chunk=dict(chunk)), p0)  # a new chunk object drops the memo
    assert _count(stub, False) == 4
    model.entry(t0, p0)  # back to the first chunk: its results were dropped, so computed again
    assert _count(stub, False) == 5

    traded = [leg for leg in t0.legs if leg["qty"] > 0]
    for alpha in (0.0, 0.5, 0.5):
        model.exit_value(t0, traded, alpha)
    assert _count(stub, True) == 1
    model.exit_value(t0._replace(chunk=dict(chunk)), traded, 0.5)
    assert _count(stub, True) == 2


def test_loading_an_f_copy_leaves_the_real_module_alone(tmp_path):
    fq, _ = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    real_entry = sv.entry
    copy = seam.load(fills=ff.FFills(fq))
    assert type(copy.fills) is ff.FFills
    assert type(seam.PRICING.fills) is CurrentFills
    assert sv.entry is real_entry


def test_score_events_prices_accepted_events_at_the_f_debit_whatever_alpha(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    scored = {alpha: seam.load(fills=ff.FFills(fq)).score_events(chunk, EVENTS, SIMS, NAME, alpha=alpha)
              for alpha in (0.5, 0.0)}
    for eid, i in (("E0", 0), ("E1", 1)):
        row = scored[0.5].loc[scored[0.5]["event_id"] == eid].iloc[0]
        trade, _ = _trade(chunk, i)
        traded = [leg for leg in trade.legs if leg["qty"] > 0]
        assert row["status"] == "scored"
        assert row["entry_cost"] == pytest.approx(_debits(_legs_frame(fq, chunk, traded, i=i))[2])
        assert np.isfinite(row["exp_pnl_sim"])
    np.testing.assert_array_equal(scored[0.5]["entry_cost"].to_numpy(), scored[0.0]["entry_cost"].to_numpy())


def test_skipped_event_is_undetermined_with_its_f_reason_and_the_other_scores(tmp_path):
    frames = _frames()
    frames[("AAA", _ts(ENTRY))] = _tight("AAA")  # BBB keeps the default wide quotes: a fill-over-fair skip
    fq, chunk = _setup(tmp_path, frames, DEFAULT_PLAN)
    rows = seam.load(fills=ff.FFills(fq)).score_events(chunk, EVENTS, SIMS, NAME, alpha=0.5).set_index("event_id")
    assert rows.loc["E0", "status"] == "scored"
    skipped = rows.loc["E1"]
    assert skipped["reason"] == "f_fill_over_fair" and skipped["status"] == "undetermined"
    assert np.isnan(skipped["exp_pnl_sim"]) and np.isnan(skipped["entry_cost"])


def test_score_marks_exits_at_f_proceeds_with_bool_flag_columns(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    model = seam.load(fills=ff.FFills(fq))
    scored = model.score_events(chunk, EVENTS, SIMS, NAME, alpha=0.5)
    rows = model.score(chunk, EVENTS, scored.assign(status="taken"), NAME, 0.5)
    assert rows["exit_fair_mid"].dtype == bool and rows["exit_negative"].dtype == bool
    by = rows.set_index("event_id")
    for eid, i in (("E0", 0), ("E1", 1)):
        trade, _ = _trade(chunk, i)
        traded = [leg for leg in trade.legs if leg["qty"] > 0]
        want = _proceeds(_legs_frame(fq, chunk, traded, i=i, at_exit=True))
        assert by.loc[eid, "exit_value"] == pytest.approx(want)
        assert by.loc[eid, "pnl_per_share"] == pytest.approx(by.loc[eid, "exit_value"] - by.loc[eid, "entry_cost"])


def test_score_path_matches_hand_set_fills_and_an_independent_simulated_return(tmp_path):
    """Hand-set fills on the score path: every leg has V = _mid(strike) (the fixture's mid); entry buys fill at
    V + 0.01 and sells at V - 0.01; exit sells fill at V - 0.02 and buys at V + 0.02, except E1 whose exit buys fill at
    V + 5 and whose first exit leg has no rate. Expected costs, proceeds and simulated returns are computed here from
    the strikes, sides and quantities of the resolved legs and the sims, never from a production fill column."""
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    stub = StubQuotes(fq)

    def edit(frame, exit):
        v = frame["strike"].map(_mid).to_numpy(float)
        step, i = (0.02 if exit else 0.01), stub.calls[-1][0]
        frame = frame.assign(V=v, eff_bid=v - 0.005, eff_ask=v + 0.005, fill_buy=v + step, fill_sell=v - step)
        if exit and i == 1:
            frame = frame.assign(fill_buy=v + 5.0, status=["rate_unknown"] + ["ok"] * (len(frame) - 1))
        return frame

    stub.edit = edit
    model = seam.load(fills=ff.FFills(stub))
    scored = model.score_events(chunk, EVENTS, SIMS, NAME, alpha=0.5).set_index("event_id")
    rows = model.score(chunk, EVENTS, scored.reset_index().assign(status="taken"), NAME, 0.5).set_index("event_id")
    for eid, i in (("E0", 0), ("E1", 1)):
        trade, _ = _trade(chunk, i)
        traded = [leg for leg in trade.legs if leg["qty"] > 0]
        assert {leg["side"] for leg in traded} == {"buy", "sell"}
        cost = sum(leg["qty"] * (_mid(leg["strike"]) + 0.01 if leg["side"] == "buy" else -(_mid(leg["strike"]) - 0.01))
                   for leg in traded)
        assert scored.loc[eid, "status"] == "scored"
        assert scored.loc[eid, "entry_cost"] == pytest.approx(cost, abs=1e-12)
        closes = [{"strike": leg["strike"], "side": "sell" if leg["side"] == "buy" else "buy", "qty": leg["qty"]}
                  for leg in traded]
        grid, means = SIMS[eid]
        want = (structure_value(means, grid, closes) - cost) / cost
        assert scored.loc[eid, "exp_pnl_sim"] == pytest.approx(want, rel=1e-12)
        assert rows.loc[eid, "entry_cost"] == pytest.approx(cost, abs=1e-12)
    trade, _ = _trade(chunk, 0)
    traded = [leg for leg in trade.legs if leg["qty"] > 0]
    proceeds = sum(leg["qty"] * (_mid(leg["strike"]) - 0.02 if leg["side"] == "buy" else -(_mid(leg["strike"]) + 0.02))
                   for leg in traded)
    assert rows.loc["E0", "exit_value"] == pytest.approx(proceeds, abs=1e-12)
    assert rows.loc["E0", "pnl_per_share"] == pytest.approx(proceeds - scored.loc["E0", "entry_cost"], abs=1e-12)
    assert not rows.loc["E0", "exit_fair_mid"] and not rows.loc["E0", "exit_negative"]
    assert rows.loc["E1", "exit_fair_mid"] and rows.loc["E1", "exit_negative"]
    assert rows.loc["E1", "exit_value"] < 0


def test_f_unpriceable_exit_is_unquoted_for_its_event_only(tmp_path):
    fq, chunk = _setup(tmp_path, _tight_frames(), DEFAULT_PLAN)
    stub = StubQuotes(fq)

    def edit(frame, exit):  # E0 (chunk index 0) only: its first traded exit leg is unpriceable
        if exit and stub.calls[-1][0] == 0:
            return frame.assign(status=["unpriceable"] + ["ok"] * (len(frame) - 1))
        return frame

    stub.edit = edit
    model = seam.load(fills=ff.FFills(stub))
    taken = model.score_events(chunk, EVENTS, SIMS, NAME, alpha=0.5).assign(status="taken")
    rows = model.score(chunk, EVENTS, taken, NAME, 0.5).set_index("event_id")
    assert rows.loc["E0", "rung"] == EXIT_UNQUOTED and rows.loc["E0", "exit_value"] == 0.0
    assert not rows.loc["E0", ["exit_fair_mid", "exit_negative"]].any()
    assert rows.loc["E1", "rung"] != EXIT_UNQUOTED
    dropped = model.score(chunk, EVENTS, taken, NAME, 0.5, drop_unquoted=True)
    assert list(dropped["event_id"]) == ["E1"]


def test_a_changed_stored_entry_bid_is_refused_by_score_events(tmp_path):
    frames = _tight_frames()
    _, chunk = _setup(tmp_path / "chunk", frames, DEFAULT_PLAN)
    trade, _ = _trade(chunk, 0)
    leg = next(leg for leg in trade.legs if leg["qty"] > 0)
    right = "P" if chunk["right"][leg["idx"]] == 0 else "C"
    strike = float(chunk["strike"][leg["idx"]])
    snap = _set(_base(frames), "AAA", ENTRY, right, strike, bid=_mid(strike) - 0.3)
    fq, chunk = _setup(tmp_path / "snap", frames, DEFAULT_PLAN, snap=snap)
    model = seam.load(fills=ff.FFills(fq))
    with pytest.raises(FairValueRefusal) as exc:
        model.score_events(chunk, EVENTS, SIMS, NAME, alpha=0.5)
    assert exc.value.details["reason"] == "f_quote_mismatch"
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_exit_value_at_v_is_the_positive_legs_at_v_signed_by_side(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade = _pair(chunk)  # buy the 100 put, sell the 95 put, one each: no concession, so V - V
    frame = fq.legs(chunk, 0, list(trade.legs), exit=True)
    want = float(frame["V"].iloc[0] - frame["V"].iloc[1])
    assert ff.FFills(fq).exit_value_at_v(trade) == pytest.approx(want, abs=1e-12)


def test_exit_value_at_v_weights_a_fallback_leg_at_its_effective_mid_not_v(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    pair = _pair(chunk)
    trade = pair._replace(legs=({**pair.legs[0], "qty": 2.0}, pair.legs[1]))  # long 2 (fallback), short 1
    frame = fq.legs(chunk, 0, list(trade.legs), exit=True)
    mid = (frame["eff_bid"].iloc[0] + frame["eff_ask"].iloc[0]) / 2
    got = ff.FFills(StubQuotes(fq, _exit_status("rate_unknown"))).exit_value_at_v(trade)
    assert got == pytest.approx(2 * mid - frame["V"].iloc[1], abs=1e-12)
    assert abs(got - (2 * frame["V"].iloc[0] - frame["V"].iloc[1])) > 1e-9  # the fallback is exercised, not a no-op


def test_exit_value_at_v_weights_a_short_fallback_leg_at_its_effective_mid_not_v(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    pair = _pair(chunk)
    trade = pair._replace(legs=(pair.legs[0], {**pair.legs[1], "qty": 2.0}))  # long 1 (V), short 2 (fallback)
    frame = fq.legs(chunk, 0, list(trade.legs), exit=True)
    v_long, v_short = float(frame["V"].iloc[0]), float(frame["V"].iloc[1])
    mid_short = float(frame["eff_bid"].iloc[1] + frame["eff_ask"].iloc[1]) / 2
    got = ff.FFills(StubQuotes(fq, _exit_status("ok", "rate_unknown"))).exit_value_at_v(trade)
    assert got == pytest.approx(v_long - 2 * mid_short, abs=1e-12)
    assert abs(got - (v_long - 2 * v_short)) > 1e-9  # the short's V is not its effective mid, so using V would differ


def test_exit_value_at_v_never_reads_a_zero_quantity_reference_leg(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade, _ = _trade(chunk, 0)
    traded = [leg for leg in trade.legs if leg["qty"] > 0]
    assert len(traded) < len(trade.legs)
    stub = StubQuotes(fq)
    got = ff.FFills(stub).exit_value_at_v(trade)
    positive = [leg["idx"] for leg in traded]
    assert stub.calls and all(call == (0, True, positive) for call in stub.calls)
    frame = fq.legs(chunk, 0, traded, exit=True)
    want = sum(leg["qty"] * (1.0 if leg["side"] == "buy" else -1.0) * frame["V"].iloc[pos] for pos, leg in enumerate(traded))
    assert got == pytest.approx(float(want), abs=1e-9)


def test_exit_value_at_v_is_nan_when_a_leg_is_unpriceable(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    got = ff.FFills(StubQuotes(fq, _exit_status("unpriceable"))).exit_value_at_v(_pair(chunk))
    assert math.isnan(got)


def _invalid_side_trade(chunk):
    """_pair with the positive buy leg's side set to "hold": not a side F can close."""
    pair = _pair(chunk)
    return pair._replace(legs=({**pair.legs[0], "side": "hold"}, pair.legs[1]))


def _zero_qty_trade(chunk):
    """_pair with every leg at quantity 0: no positive-quantity leg is traded."""
    pair = _pair(chunk)
    return pair._replace(legs=tuple({**leg, "qty": 0.0} for leg in pair.legs))


def _empty_legs_trade(chunk):
    return _pair(chunk)._replace(legs=())


@pytest.mark.parametrize("build", [_invalid_side_trade, _zero_qty_trade, _empty_legs_trade],
                         ids=["invalid_side", "zero_qty", "empty_legs"])
def test_exit_result_and_exit_value_at_v_refuse_alike(tmp_path, build):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    trade = build(chunk)
    with pytest.raises(fvf.FairValueRefusal) as via_result:
        ff.FFills(fq).exit_result(trade)
    with pytest.raises(fvf.FairValueRefusal) as via_value:
        ff.FFills(fq).exit_value_at_v(trade)
    assert type(via_value.value) is type(via_result.value)
    assert via_value.value.code == via_result.value.code
    assert via_value.value.details["reason"] == via_result.value.details["reason"] == "leg_invalid"
    assert str(via_value.value) == str(via_result.value)


def test_was_priced_is_false_before_the_entry_result_true_after_and_false_for_a_new_chunk(tmp_path):
    fq, chunk = _setup(tmp_path, _frames(), DEFAULT_PLAN)
    model = ff.FFills(fq)
    trade, _ = _trade(chunk, 0)
    assert model.was_priced(trade) is False
    model.entry_result(trade)
    assert model.was_priced(trade) is True
    assert model.was_priced(trade._replace(chunk=dict(chunk))) is False  # a new chunk object drops the results
