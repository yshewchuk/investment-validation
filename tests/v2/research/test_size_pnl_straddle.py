"""EXP-188 S7: the straddle under production's G_PROD gate. Synthetic fixtures only; no repository."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.fills import FillModel
from engine.portfolio import CAPITAL_PER_TRADE, CONTRACT_MULTIPLIER
from engine.score import GATE_MCAP_FLOOR, Scorer
from engine.structures import CALL, PUT, StructurePrice
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_gate_agreement as ga
from experiments import size_pnl_ladder as lad
from experiments import size_pnl_readers as rd
from experiments import size_pnl_straddle as sd
from experiments.size_pnl_gate_fold import REGISTERED_THRESHOLD

NAN = float("nan")
GATE = lambda ev: ("taken", None)
NOT_TAKEN = lambda ev: ("not_taken", "below_gate_threshold")


def pair(k, *, call_e=(2.0, 2.2), put_e=(1.8, 2.0), call_x=(1.0, 1.4, 1), put_x=(0.6, 1.0, 1)):
    """Put and call at strike ``k``: entry (bid, ask), exit (bid, ask, rung)."""
    return [(0, k, *put_e, *put_x), (1, k, *call_e, *call_x)]


def rec(eid, spot, legs):
    cols = list(zip(*legs))
    return {"event_id": eid, "expiry": np.datetime64("2024-03-15"), "spot_entry": spot, "spot_exit": spot,
            "dte_entry": 30, "dte_exit": 29, "right": cols[0], "strike": cols[1], "bid_e": cols[2],
            "ask_e": cols[3], "bid_x": cols[4], "ask_x": cols[5], "exit_rung": cols[6]}


def one(eid="E", spot=100.0, **kw):
    return lad._pack([rec(eid, spot, pair(100, **kw))], {})


def good(*eids):
    """Chunk with one healthy ATM straddle per event id (entry relative spread ~0.10)."""
    return lad._pack([rec(e, 100.0, pair(100)) for e in eids], {})


def ev(eid, edge, *, date="2024-06-10", mcap=20e9, implied=5.0, pred=None):
    """Event row whose forecast edge is ``edge`` (pred - implied). NaN ``implied`` or ``pred`` means missing."""
    return {"event_id": eid, "event_date": date, "pred_abs_move": 5.0 + edge if pred is None else pred,
            "implied_move": implied, "mcap_usd": mcap}


def legs(chunk, i=0):
    """(call_idx, put_idx) that ``resolve`` picks for event ``i`` (None, None on a refusal)."""
    return sd.resolve(chunk, i)[1:3]


def only(*taken):
    """A gate stub that takes the listed event ids and refuses the rest."""
    return lambda event: ("taken", None) if event["event_id"] in taken else ("not_taken", "below_gate_threshold")


# ---- resolve ----------------------------------------------------------------

def test_resolve_nearest_call_strike_chosen():
    chunk = lad._pack([rec("E", 101.0, pair(95) + pair(100) + pair(105))], {})
    priced, c, p, reason = sd.resolve(chunk, 0)
    assert reason is None and isinstance(priced, StructurePrice)
    assert chunk["strike"][c] == 100 and chunk["right"][c] == 1
    assert chunk["strike"][p] == 100 and chunk["right"][p] == 0


def test_resolve_tie_goes_to_lower_strike_on_an_ascending_ladder():
    chunk = lad._pack([rec("E", 102.5, pair(100) + pair(105))], {})
    c, _ = legs(chunk)
    assert chunk["strike"][c] == 100


def test_call_without_put_at_the_atm_strike_refuses_without_shifting():
    # Healthy pairs at 95 and 110 do not rescue an ATM strike (100) that has a call but no put.
    call_only = [(1, 100, 2.0, 2.2, 1.0, 1.4, 1)]
    chunk = lad._pack([rec("E", 100.0, pair(95) + pair(110) + call_only)], {})
    assert sd.resolve(chunk, 0) == (None, None, None, "no_atm_pair")


def test_atm_put_only_at_nearest_call_strike_is_no_atm_pair():
    # Full pair at 95, call-only at 100: the ATM strike is the nearest call (100), which has no put.
    chunk = lad._pack([rec("E", 100.0, pair(95) + [(1, 100, 2.0, 2.2, 1.0, 1.4, 1)])], {})
    assert sd.resolve(chunk, 0)[3] == "no_atm_pair"


def test_put_only_event_is_no_atm_pair():
    chunk = lad._pack([rec("po", 100.0, [(0, 100, 1.8, 2.0, 0.6, 1.0, 1)])], {})
    assert sd.resolve(chunk, 0)[3] == "no_atm_pair"


def test_non_finite_spot_is_no_atm_pair():
    chunk = lad._pack([rec("F", NAN, pair(100))], {})
    assert sd.resolve(chunk, 0) == (None, None, None, "no_atm_pair")


@pytest.mark.parametrize("kw", [{"put_e": (2.2, 2.0)}, {"call_e": (NAN, 2.2)}], ids=["crossed", "nan_bid"])
def test_crossed_or_nan_bid_is_entry_quote_unusable_and_never_taken(kw):
    chunk = one(**kw)
    assert sd.resolve(chunk, 0) == (None, None, None, "entry_quote_unusable")
    assert sd.entry_state(chunk, 0)["reason"] == "entry_quote_unusable"
    out = sd.decide(pd.DataFrame([ev("E", 1.0)]), chunk, GATE)
    assert out.loc[0, "status"] == "undetermined" and out.loc[0, "reason"] == "entry_quote_unusable"


@pytest.mark.parametrize("kw, cost", [({"put_e": (0.0, 0.5)}, 2.1 + 0.25), ({"call_e": (2.0, 2.0)}, 2.0 + 1.9)],
                         ids=["zero_bid_put", "locked_call"])
def test_zero_bid_and_locked_entry_quotes_are_priced_at_mid(kw, cost):
    chunk = one(**kw)
    priced, c, p, reason = sd.resolve(chunk, 0)
    assert reason is None and priced.cost == pytest.approx(cost)
    assert sd.entry_state(chunk, 0)["status"] == "ok"
    out = sd.decide(pd.DataFrame([ev("E", 1.0)]), chunk, GATE)
    assert out.loc[0, "status"] == "taken" and pd.isna(out.loc[0, "reason"])


@pytest.mark.parametrize("alpha", [0.5, 0.25])
def test_resolve_cost_is_the_fill_model_buy_sum_on_three_events(alpha):
    chunk = lad._pack([rec("A", 100.0, pair(100)),
                       rec("B", 97.0, pair(95) + pair(100, call_e=(2.4, 2.6))),
                       rec("C", 210.0, pair(210, put_e=(3.0, 3.4), call_e=(4.1, 4.5)))], {})
    fill = FillModel(alpha)
    for i in range(3):
        _, c, p, reason = sd.resolve(chunk, i, alpha)
        assert reason is None
        expected = sum(float(fill.buy(chunk["bid_e"][k], chunk["ask_e"][k])) for k in (c, p))
        assert sd.resolve(chunk, i, alpha)[0].cost == pytest.approx(expected)


def test_chain_snapshot_puts_only_holds_only_puts():
    chunk = good("E")
    assert list(sd.chain_snapshot(chunk, 0, puts_only=True).rows["right"]) == [PUT]
    rows = sd.chain_snapshot(chunk, 0).rows
    assert list(rows["right"]) == [PUT, CALL] and list(rows["strike"]) == [100, 100]


# ---- entry_state ----------------------------------------------------------

def test_entry_healthy_reports_spread():
    st = sd.entry_state(one(), 0)
    assert st["status"] == "ok" and st["reason"] is None
    assert st["rel_spread"] == pytest.approx((0.2 / 2.1 + 0.2 / 1.9) / 2)
    assert st["priced"].cost == pytest.approx(2.1 + 1.9)


def test_entry_no_atm_pair_is_undetermined():
    calls_only = lad._pack([rec("E", 100.0, [(1, 100, 2.0, 2.2, 1.0, 1.4, 1)])], {})
    st = sd.entry_state(calls_only, 0)
    assert st["status"] == "undetermined" and st["reason"] == "no_atm_pair"


# ---- exit_value -----------------------------------------------------------

def test_exit_rung1_at_half_is_mid_sum():
    chunk = one()
    c, p = legs(chunk)
    value, label = sd.exit_value(chunk, c, p, 0.5)
    assert value == pytest.approx(1.2 + 0.8) and label == "rung1"


def test_exit_rung2_uses_alpha_zero():
    chunk = one(call_x=(0.0, 0.5, 2), put_x=(0.0, 0.4, 2))
    c, p = legs(chunk)
    assert sd.exit_value(chunk, c, p, 0.5) == (0.0, "rung2")
    assert sd.exit_value(chunk, c, p, 1.0) == (0.0, "rung2")
    mixed = one(call_x=(0.0, 0.5, 2))
    c, p = legs(mixed)
    value, label = sd.exit_value(mixed, c, p, 0.5)
    assert value == pytest.approx(0.8) and label == "rung2"


def test_exit_one_rung3_leg_is_worst_case_even_with_rung1_other_leg():
    chunk = one(call_x=(NAN, NAN, 3))
    c, p = legs(chunk)
    assert sd.exit_value(chunk, c, p, 0.5) == (0.0, sd.EXIT_UNQUOTED)


def test_exit_locked_quote_priced_as_rung1():
    chunk = one(call_x=(1.0, 1.0, 1))
    c, p = legs(chunk)
    value, label = sd.exit_value(chunk, c, p, 0.5)
    assert value == pytest.approx(1.0 + 0.8) and label == "rung1"


@pytest.mark.parametrize("alpha, msg", [(2.0, r"alpha must lie in \[0, 1\], got 2\.0"),
                                        (NAN, r"alpha must be finite, got nan")])
def test_exit_value_bad_alpha_raises_on_rung2_and_rung3(alpha, msg):
    for chunk in (one(call_x=(0.0, 0.5, 2), put_x=(0.0, 0.4, 2)),  # all rung 2
                  one(call_x=(NAN, NAN, 3))):                      # rung 3
        c, p = legs(chunk)
        with pytest.raises(ValueError, match=msg):
            sd.exit_value(chunk, c, p, alpha)


# ---- lots -----------------------------------------------------------------

def test_lots_floor_floor_of_one_and_nonfinite():
    assert sd.lots(3.0) == math.floor(CAPITAL_PER_TRADE / (3.0 * CONTRACT_MULTIPLIER))
    assert sd.lots(4.0) == 25
    assert sd.lots(0.70) == 142  # 10000 / 70 = 142.86 -> 142; round() would give 143
    assert sd.lots(150.0) == 1  # expensive straddle still takes one lot
    for bad in (NAN, math.inf, 0.0, -1.0):
        assert sd.lots(bad) == 1


def test_lots_exact_division_does_not_round_down():
    # CAPITAL_PER_TRADE / (cost * 100) is an exact integer here; float error must not floor it to one less.
    assert sd.lots(2.5) == 40   # 10000 / 250
    assert sd.lots(1.25) == 80  # 10000 / 125


# ---- decide ---------------------------------------------------------------

def test_gate_taken_and_not_taken_pass_through():
    events = pd.DataFrame([ev("hi", 1.0), ev("lo", 0.5)])
    out = sd.decide(events, good("hi", "lo"), only("hi")).set_index("event_id")
    assert out.loc["hi", "status"] == "taken" and pd.isna(out.loc["hi", "reason"])
    assert out.loc["lo", "status"] == "not_taken" and out.loc["lo", "reason"] == "below_gate_threshold"


def test_decide_requires_a_gate_callable():
    with pytest.raises(ValueError, match="G_PROD needs a gate callable"):
        sd.decide(pd.DataFrame([ev("hi", 1.0)]), good("hi"), None)


def test_decision_columns_have_no_cutoff():
    out = sd.decide(pd.DataFrame([ev("hi", 1.0)]), good("hi"), GATE)
    assert list(out.columns) == sd.DECISION_COLS and "cutoff" not in out.columns


def test_rel_spread_recorded_from_chunk():
    out = sd.decide(pd.DataFrame([ev("hi", 1.0)]), good("hi"), GATE)
    assert out.loc[0, "rel_spread"] == pytest.approx((0.2 / 2.1 + 0.2 / 1.9) / 2)


def test_missing_implied_is_missing_forecast_and_bad_entry_unusable():
    events = pd.DataFrame([ev("m", 1.0, implied=NAN), ev("u", 1.0)])
    chunk = lad._pack([rec("m", 100.0, pair(100)), rec("u", 100.0, pair(100, put_e=(2.2, 2.0)))], {})
    out = sd.decide(events, chunk, GATE).set_index("event_id")
    assert out.loc["m", "reason"] == "missing_forecast" and out.loc["m", "status"] == "undetermined"
    assert out.loc["u", "reason"] == "entry_quote_unusable" and out.loc["u", "status"] == "undetermined"


def test_missing_pred_is_missing_forecast():
    out = sd.decide(pd.DataFrame([ev("p", 1.0, pred=NAN)]), good("p"), GATE)
    assert out.loc[0, "status"] == "undetermined" and out.loc[0, "reason"] == "missing_forecast"


@pytest.mark.parametrize("null", [None, pd.NA])
@pytest.mark.parametrize("field", ["pred_abs_move", "implied_move"])
def test_null_forecast_input_is_missing_forecast(field, null):
    event = dict(ev("n", 1.0), **{field: null})
    out = sd.decide(pd.DataFrame([event]), good("n"), GATE)
    assert out.loc[0, "status"] == "undetermined" and out.loc[0, "reason"] == "missing_forecast"


@pytest.mark.parametrize("null", [None, pd.NA])
def test_missing_forecast_beats_null_mcap(null):
    event = dict(ev("b", 1.0, implied=NAN), mcap_usd=null)
    out = sd.decide(pd.DataFrame([event]), good("b"), GATE)
    assert out.loc[0, "reason"] == "missing_forecast"


def test_event_absent_from_chunk_is_no_ladder_chunk():
    out = sd.decide(pd.DataFrame([ev("ghost", 1.0)]), good("other"), GATE)
    assert out.loc[0, "status"] == "undetermined" and out.loc[0, "reason"] == "no_ladder_chunk"


def test_put_only_event_decides_as_no_atm_pair():
    chunk = lad._pack([rec("po", 100.0, [(0, 100, 1.8, 2.0, 0.6, 1.0, 1)])], {})
    out = sd.decide(pd.DataFrame([ev("po", 1.0)]), chunk, GATE)
    assert out.loc[0, "status"] == "undetermined" and out.loc[0, "reason"] == "no_atm_pair"


def test_spread_exactly_max_is_not_filtered():
    chunk = lad._pack([rec("s", 100.0, pair(100, put_e=(7.0, 9.0), call_e=(7.0, 9.0)))], {})
    out = sd.decide(pd.DataFrame([ev("s", 1.0)]), chunk, GATE)
    assert out.loc[0, "rel_spread"] == 0.25
    assert out.loc[0, "status"] == "taken" and pd.isna(out.loc[0, "reason"])


def test_mcap_exactly_floor_is_not_filtered():
    out = sd.decide(pd.DataFrame([ev("m", 1.0, mcap=GATE_MCAP_FLOOR)]), good("m"), GATE)
    assert out.loc[0, "status"] == "taken"


def test_universe_filters_are_not_taken():
    chunk = lad._pack([rec("small", 100.0, pair(100))], {})
    events = pd.DataFrame([ev("small", 1.0, mcap=0.5e9)])
    out = sd.decide(events, chunk, GATE).set_index("event_id")
    assert set(out.index) == {"small"}
    assert out.loc["small", "status"] == "not_taken" and out.loc["small", "reason"] == "universe_filter"


def test_in_universe_is_production_gate_floor_only():
    assert sd.in_universe(GATE_MCAP_FLOOR) and sd.in_universe(GATE_MCAP_FLOOR * 1.5)
    assert not sd.in_universe(0.5e9)
    for null in (NAN, None, pd.NA):
        assert sd.in_universe(null)  # production passes a missing market cap; the gate withholds on it


@pytest.mark.parametrize("mcap, expected", [(GATE_MCAP_FLOOR, True), (GATE_MCAP_FLOOR - 1, False), (0.5e9, False),
                                            (5e9, True), (0.0, True), (-1.0, True), (NAN, True)])
def test_in_universe_matches_production_gate_domain(mcap, expected):
    assert sd.in_universe(mcap) is expected


def test_in_universe_delegates_to_production_gate_domain(monkeypatch):
    seen = []

    def stub(self, request, features):
        seen.append(features)
        return False

    monkeypatch.setattr(Scorer, "_gate_in_domain", stub)
    assert sd.in_universe(5e9) is False
    assert len(seen) == 1 and list(seen[0].columns) == ["mcap_log"] and len(seen[0]) == 1


@pytest.mark.parametrize("verdict", [("undetermined", "gate_input_nan"), ("taken", None)])
@pytest.mark.parametrize("null", [None, pd.NA, NAN])
def test_null_mcap_reaches_the_gate_and_takes_its_verdict(null, verdict):
    calls = []

    def gate(event):
        calls.append(event["event_id"])
        return verdict

    out = sd.decide(pd.DataFrame([dict(ev("m", 1.0), mcap_usd=null)]), good("m"), gate)
    assert calls == ["m"]
    assert out.loc[0, "status"] == verdict[0]
    if verdict[1] is None:
        assert pd.isna(out.loc[0, "reason"])
    else:
        assert out.loc[0, "reason"] == verdict[1]
    assert pd.isna(out.loc[0, "mcap_usd"])


def test_decide_records_mcap_usd_on_taken_and_universe_rows():
    assert sd.DECISION_COLS.index("mcap_usd") == sd.DECISION_COLS.index("rel_spread") + 1
    events = pd.DataFrame([ev("t", 1.0, mcap=5e9), ev("s", 1.0, mcap=0.5e9)])
    out = sd.decide(events, good("t", "s"), GATE).set_index("event_id")
    assert out.loc["t", "status"] == "taken" and out.loc["t", "mcap_usd"] == 5e9
    assert out.loc["s", "reason"] == "universe_filter" and out.loc["s", "mcap_usd"] == 0.5e9


def test_five_billion_name_is_in_the_universe():
    # Under the old $10B short-vol floor this name was filtered; production's STR-THRU universe takes it.
    out = sd.decide(pd.DataFrame([ev("five", 1.0, mcap=5e9)]), good("five"), GATE)
    assert out.loc[0, "status"] == "taken" and pd.isna(out.loc[0, "reason"])


def test_wide_spread_name_in_universe_is_taken():
    # Rel spread 0.67: production applies no spread term, so the straddle takes it; rel_spread is still recorded.
    chunk = lad._pack([rec("wide", 100.0, pair(100, put_e=(1.0, 2.0), call_e=(1.0, 2.0)))], {})
    out = sd.decide(pd.DataFrame([ev("wide", 1.0, mcap=5e9)]), chunk, GATE)
    assert out.loc[0, "rel_spread"] > 0.25
    assert out.loc[0, "status"] == "taken" and pd.isna(out.loc[0, "reason"])


def _bad_quote_chunk(eid):
    """Spot 10, ATM call mid 2.0 and put mid 1.5: entry cost 3.5 is 35% of spot, above production's 30% ceiling."""
    return lad._pack([rec(eid, 10.0, pair(10, call_e=(1.5, 2.5), put_e=(1.0, 2.0)))], {})


def test_bad_quote_straddle_is_not_taken_and_the_gate_is_not_called():
    calls = []

    def gate(event):
        calls.append(event["event_id"])
        return "taken", None

    out = sd.decide(pd.DataFrame([ev("bq", 1.0)]), _bad_quote_chunk("bq"), gate)
    assert calls == []
    assert out.loc[0, "status"] == "not_taken" and out.loc[0, "reason"] == "bad_quote"


def test_bad_quote_exactly_30_percent_is_not_refused():
    # Mids 1.5 + 1.5 = 3.0 at spot 10 is exactly 30.0%, which production allows (strictly greater is refused).
    chunk = lad._pack([rec("eq", 10.0, pair(10, call_e=(1.0, 2.0), put_e=(1.0, 2.0)))], {})
    out = sd.decide(pd.DataFrame([ev("eq", 1.0)]), chunk, GATE)
    assert out.loc[0, "status"] == "taken" and pd.isna(out.loc[0, "reason"])


def test_normal_chunk_still_reaches_the_gate_callable():
    calls = []

    def gate(event):
        calls.append(event["event_id"])
        return "taken", None

    out = sd.decide(pd.DataFrame([ev("g", 1.0)]), good("g"), gate)
    assert calls == ["g"]
    assert out.loc[0, "status"] == "taken" and pd.isna(out.loc[0, "reason"])


def test_trade_counts_keep_undetermined_separate():
    events = pd.DataFrame([ev("u", 1.0, implied=NAN), ev("lo", 0.5), ev("hi", 1.0)])
    counts = sd.trade_counts(sd.decide(events, good("u", "lo", "hi"), only("hi")))
    assert counts == {"taken": 1, "not_taken": 1, "undetermined": 1}


def test_decide_reads_no_exit_quotes():
    events = pd.DataFrame([ev("hi", 1.0)])
    chunk = good("hi")
    with_exit = sd.decide(events, chunk, GATE)
    for key in ("bid_x", "ask_x", "exit_rung"):
        del chunk[key]
    without_exit = sd.decide(events, chunk, GATE)
    pd.testing.assert_frame_equal(with_exit, without_exit)
    assert without_exit.loc[0, "status"] == "taken"


# ---- score, rung_shares, trade_counts -------------------------------------

def _scoring_fixture():
    chunk = lad._pack([rec("A", 100.0, pair(100)),
                       rec("B", 100.0, pair(100, call_x=(NAN, NAN, 3)))], {})
    events = pd.DataFrame([ev("A", 1.0), ev("B", 1.0)])
    return chunk, events, sd.decide(events, chunk, GATE)


def test_score_pnl_dollars_and_unquoted_rows():
    chunk, events, decisions = _scoring_fixture()
    scored = sd.score(chunk, events, decisions, 0.5).set_index("event_id")
    # Entry 4.0 at mid; A exits at 2.0 (rung 1), B at worst-case 0.0 (rung 3).
    assert scored.loc["A", "pnl_per_share"] == pytest.approx(-2.0)
    assert scored.loc["A", "lots"] == 25
    assert scored.loc["A", "pnl_dollars"] == pytest.approx(-2.0 * 100 * 25)
    assert scored.loc["B", "rung"] == sd.EXIT_UNQUOTED
    for row in scored.itertuples():
        assert row.pnl_dollars == pytest.approx((row.exit_value - row.entry_cost) * 100 * row.lots)
    kept = sd.score(chunk, events, decisions, 0.5, drop_unquoted=True)
    assert list(kept["event_id"]) == ["A"]


def test_rung_shares_sum_to_one_and_empty_is_dict():
    chunk, events, decisions = _scoring_fixture()
    shares = sd.rung_shares(sd.score(chunk, events, decisions, 0.5))
    assert shares == {"rung1": 0.5, sd.EXIT_UNQUOTED: 0.5}
    assert sum(shares.values()) == pytest.approx(1.0)
    none = sd.decide(events, chunk, NOT_TAKEN)
    assert sd.rung_shares(sd.score(chunk, events, none, 0.5)) == {}


def test_score_missing_event_raises():
    chunk = lad._pack([rec("A", 100.0, pair(100))], {})
    events = pd.DataFrame([ev("Z", 1.0)])
    forced = sd.decide(events, chunk, GATE).assign(status="taken")  # defence-in-depth: taken but absent
    with pytest.raises(ValueError, match=r"taken event 'Z' is missing from the chunk or events"):
        sd.score(chunk, events, forced, 0.5)


def test_score_taken_event_absent_from_events_raises():
    chunk = lad._pack([rec("Z", 100.0, pair(100))], {})
    events = pd.DataFrame([ev("A", 1.0)])  # Z is in the chunk but not in events
    forced = sd.decide(pd.DataFrame([ev("Z", 1.0)]), chunk, GATE).assign(status="taken")
    with pytest.raises(ValueError, match=r"taken event 'Z' is missing from the chunk or events"):
        sd.score(chunk, events, forced, 0.5)


def test_score_taken_event_without_legs_raises():
    chunk = lad._pack([rec("P", 100.0, [(0, 100, 1.8, 2.0, 0.6, 1.0, 1)])], {})
    events = pd.DataFrame([ev("P", 1.0)])
    forced = sd.decide(events, chunk, GATE).assign(status="taken")  # defence-in-depth: taken, no ATM pair
    with pytest.raises(ValueError, match=r"taken event 'P' has no ATM straddle"):
        sd.score(chunk, events, forced, 0.5)


@pytest.mark.parametrize("alpha, msg", [(2.0, r"alpha must lie in \[0, 1\], got 2\.0"),
                                        (NAN, r"alpha must be finite, got nan")])
def test_score_bad_alpha_raises_with_no_taken_trades(alpha, msg):
    chunk = lad._pack([rec("A", 100.0, pair(100))], {})
    events = pd.DataFrame([ev("A", 1.0)])
    none_taken = sd.decide(events, chunk, GATE).assign(status="not_taken")
    with pytest.raises(ValueError, match=msg):
        sd.score(chunk, events, none_taken, alpha)


# ---- real ladder path: missing exit chain ----------------------------------

EXPIRY = "2024-03-15"


def _chain(rows, day):
    frame = pd.DataFrame(rows, columns=["expiry", "dte", "right", "strike", "bid", "ask"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    frame.insert(0, "ticker", "E")
    frame["obs_date"], frame["spot"] = pd.Timestamp(day), 100.0
    return frame


def _ladder_chunk(exit_rows):
    """Chunk from ``lad.ladder_chunks`` for one BMO event on 2024-03-14 (entry 03-13, exit 03-15), with the
    ATM-healthy entry quotes of ``good()``. ``exit_rows=None`` means the exit chain is absent."""
    strikes = (90, 100, 110)
    entry = [(EXPIRY, 2, "P", k, 1.8, 2.0) for k in strikes] + [(EXPIRY, 2, "C", k, 2.0, 2.2) for k in strikes]
    groups = {("E", pd.Timestamp("2024-03-13")): _chain(entry, "2024-03-13")}
    if exit_rows is not None:
        groups[("E", pd.Timestamp("2024-03-15"))] = _chain(exit_rows, "2024-03-15")
    admitted = rd.Admitted({"events": pd.DataFrame({"event_id": ["E"], "event_date": pd.to_datetime(["2024-03-14"])}),
                            "excluded": {}})
    plan = pd.DataFrame([{"event_id": "E", "ticker": "E", "event_date": "2024-03-14", "session": "BMO",
                          "entry_date": "2024-03-13", "exit_date": "2024-03-15"}])
    index = SimpleNamespace(get=lambda t, d: groups.get((str(t), pd.Timestamp(d).normalize())))
    [(_, chunk)] = lad.ladder_chunks(admitted, plan, lambda keys: index)
    return chunk


def test_missing_exit_chain_decides_as_covered_and_scores_worst_case():
    events = pd.DataFrame([ev("E", 1.0)])
    exit_rows = [(EXPIRY, 0, r, k, 0.8, 1.0) for r in "PC" for k in (90, 100, 110)]
    covered, missing = _ladder_chunk(exit_rows), _ladder_chunk(None)
    dec_cov = sd.decide(events, covered, GATE).loc[0]
    dec_miss = sd.decide(events, missing, GATE).loc[0]
    assert dec_miss["status"] == "taken" and dec_cov["status"] == "taken"
    assert dec_miss["rel_spread"] == dec_cov["rel_spread"]
    scored = sd.score(missing, events, sd.decide(events, missing, GATE), 0.5).set_index("event_id")
    assert scored.loc["E", "exit_value"] == 0.0
    assert scored.loc["E", "rung"] == sd.EXIT_UNQUOTED


# ---- Amendment 1 sensitivity: rung-2 legs marked at alpha (mid_marks) ----

def test_mid_marks_values_a_rung2_leg_at_alpha_and_keeps_the_label():
    chunk = one(call_x=(0.0, 0.5, 2))  # call rung 2 (bid 0.0), put rung 1 (mid 0.8)
    c, p = legs(chunk)
    assert sd.exit_value(chunk, c, p, 0.5) == (pytest.approx(0.8), "rung2")
    value, label = sd.exit_value(chunk, c, p, 0.5, True)
    assert value == pytest.approx(0.25 + 0.8) and label == "rung2"  # the call sells at its mid, 0.25


def test_mid_marks_keeps_rung3_and_equals_the_primary_when_rung1():
    chunk = one(call_x=(NAN, NAN, 3))
    c, p = legs(chunk)
    assert sd.exit_value(chunk, c, p, 0.5, True) == (0.0, sd.EXIT_UNQUOTED)
    healthy = one()
    c, p = legs(healthy)
    assert sd.exit_value(healthy, c, p, 0.5, True) == sd.exit_value(healthy, c, p, 0.5)


def test_mid_marks_score_keeps_the_taken_set_and_changes_only_the_rung2_row():
    chunk = lad._pack([rec("A", 100.0, pair(100, call_x=(0.0, 0.5, 2))),
                       rec("B", 100.0, pair(100, call_x=(NAN, NAN, 3)))], {})
    events = pd.DataFrame([ev("A", 1.0), ev("B", 1.0)])
    decisions = sd.decide(events, chunk, GATE)
    primary = sd.score(chunk, events, decisions, 0.5).set_index("event_id")
    pd.testing.assert_frame_equal(sd.score(chunk, events, decisions, 0.5, mid_marks=False).set_index("event_id"), primary)
    mid = sd.score(chunk, events, decisions, 0.5, mid_marks=True).set_index("event_id")
    assert list(mid.index) == list(primary.index) == ["A", "B"]
    assert mid.loc["A", "exit_value"] == pytest.approx(primary.loc["A", "exit_value"] + 0.25)
    assert mid.loc["A", "pnl_dollars"] == pytest.approx(primary.loc["A", "pnl_dollars"] + 0.25 * 100 * primary.loc["A", "lots"])
    assert mid.loc["B", "exit_value"] == 0.0 and mid.loc["B", "rung"] == sd.EXIT_UNQUOTED


# ---- M0 gate agreement section (Amendment 6, merged A6-3a) -----------------

def _m0_rows():
    """M0's ten determined events: four taken by a score-based G_PROD stub, six refused below the registered threshold."""
    ids = [f"G{k}" for k in range(10)]
    scores = {e: (REGISTERED_THRESHOLD + 0.01 if k < 4 else REGISTERED_THRESHOLD - 0.01) for k, e in enumerate(ids)}

    def gate(event):
        ret = scores[event["event_id"]]
        return ("taken", None) if ret >= REGISTERED_THRESHOLD else ("not_taken", "below_gate_threshold")

    dec = sd.decide(pd.DataFrame([ev(e, 3.0) for e in ids]), good(*ids), gate)
    m0 = dec.assign(gate_score=dec["event_id"].map(scores))
    kept = m0[(m0["status"] == "taken") | (m0["reason"] == "below_gate_threshold")].assign(entry_cost=4.0, exit_value_mid=4.4)
    return kept, set(kept.loc[kept["status"] == "taken", "event_id"])


def test_m0_gate_section_renders_at_the_bar():
    kept, taken = _m0_rows()
    assert len(taken) == 4
    text = ga.section(kept, {e: 0.01 for e in taken})  # stored set equals M0's taken set: Jaccard 1.0
    assert ga.HEADING in text and "| bar |" in text


def test_m0_gate_section_refuses_a_disjoint_stored_set_below_the_bar():
    kept, _ = _m0_rows()
    disjoint = {"G5": 0.01, "G6": 0.02}  # covered, but refused by M0: Jaccard 0
    with pytest.raises(OpsError) as caught:
        ga.section(kept, disjoint)
    assert caught.value.problem.details["reason"] == "gate_agreement_below_bar"
    assert "not production-faithful" in ga.section(kept, disjoint, accepted_gap="x")
