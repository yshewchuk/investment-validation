"""EXP-188 S5: short-vol scoring. Synthetic ladders and a stub pool only; no repository."""
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from engine.entry_rules import MAX_REL_SPREAD, MCAP_FLOOR
from engine.fills import FillModel
from engine.forecast_sizing import FORECAST_SIZED, forecast_params
from engine.pnl_sim import expected_pnl
from engine.portfolio import CONTRACT_MULTIPLIER
from engine.structures import CALL, PUT, STRUCTURES
from experiments import size_pnl_ladder as lad
from experiments import size_pnl_shortvol as sv
from experiments.size_pnl_pool import build_pool
from experiments.size_pnl_straddle import EXIT_UNQUOTED, lots

NAN = float("nan")
DRAW = 200
PRED = 6.0
NAMES = sv.NAMES


def quote(k):
    """Entry (bid, ask) at strike ``k``: a convex mid of 1 + 0.01*(k-100)^2 with a 0.2 spread."""
    mid = 1.0 + 0.01 * (k - 100) ** 2
    return mid - 0.1, mid + 0.1


def rec(eid, spot, legs):
    cols = list(zip(*legs))
    return {"event_id": eid, "expiry": np.datetime64("2024-03-15"), "spot_entry": spot, "spot_exit": spot,
            "dte_entry": 30, "dte_exit": 29, "right": cols[0], "strike": cols[1], "bid_e": cols[2],
            "ask_e": cols[3], "bid_x": cols[4], "ask_x": cols[5], "exit_rung": cols[6]}


def put_legs(*, strikes=range(80, 121), entry=quote, exit_=None):
    """Put rows across ``strikes``. ``exit_`` maps strike -> (bid, ask, rung); default is locked and rung 1."""
    exit_ = exit_ or {}
    legs = []
    for k in strikes:
        bid, ask = entry(k)
        xb, xa, rung = exit_.get(k, (1.0, 1.0, 1))
        legs.append((0, float(k), bid, ask, xb, xa, rung))
    return legs


def chunk_of(*events, entry=quote, strikes=range(80, 121)):
    """Chunk with one put ladder per (event_id, exit_map) pair, spot 100."""
    return lad._pack([rec(eid, 100.0, put_legs(strikes=strikes, entry=entry, exit_=ex))
                      for eid, ex in events], {})


def events_for(*eids, pred=PRED, mcap=20e9, date="2024-06-10"):
    return pd.DataFrame([{"event_id": e, "event_date": pd.Timestamp(date), "pred_move": pred,
                          "pred_crush": -15.0, "pre_iv30": 40.0, "mcap_usd": mcap, "dte_exit": 9}
                         for e in eids])


def taken(chunk, *eids, name="CND-PS", alpha=0.5, c=1.0):
    """Taken decisions for ``eids``, each carrying the entry cost the module prices at ``alpha`` and ``c``.

    An event absent from the chunk, or with no entry cost, gets NaN, so ``score`` refuses it on its own checks.
    """
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    costs = []
    for eid in eids:
        cost = NAN
        if eid in pos:
            legs, _ = sv.resolve(chunk, pos[eid], name, PRED, c)
            entered = None if legs is None else sv.entry_legs(chunk, legs, FillModel(alpha))
            if entered is not None:
                cost = entered[0]
        costs.append(cost)
    return pd.DataFrame({"event_id": list(eids), "status": ["taken"] * len(eids),
                         "structure": [name] * len(eids), "entry_cost": costs})


def _tables(seed=7, n=600):
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 1096, n), unit="D")
    pred_move = rng.uniform(2, 8, n)
    pred_crush = rng.normal(-15, 3, n)
    index = pd.Index([f"E{i}" for i in range(n)], name="event_id")
    table = pd.DataFrame({"event_date": dates, "pred_move": pred_move, "pred_crush": pred_crush}, index=index)
    labels = pd.DataFrame({"abs_move": pred_move + rng.normal(0, 1.5, n),
                           "crush_pct_iv30": pred_crush + rng.normal(0, 5, n)}, index=index)
    return table, labels


@pytest.fixture
def pool():
    return build_pool(*_tables(), floor=None)


def run(chunk, events, pool, name="CND-PS"):
    return sv.score_events(chunk, events, sv.simulate(chunk, events, pool, draws=DRAW), name)


def scored_row(eid, *, pnl=0.5, mcap=20e9, rel=0.1, date="2024-06-10", name="TWIN-P"):
    return {"event_id": eid, "event_date": pd.Timestamp(date), "structure": name, "width": 0.04,
            "entry_cost": 3.0, "rel_spread": rel, "mcap_usd": mcap, "exp_pnl_sim": pnl,
            "status": "scored", "reason": None}


def scored_frame(*rows):
    return pd.DataFrame(list(rows), columns=sv.SCORE_COLS)


def prior_scores(n=120, date="2024-01-15"):
    """n scored rows in the window before a June 2024 decision; pnl 0.00 .. (n-1)/100."""
    return [scored_row(f"P{i}", pnl=i / 100.0, date=date) for i in range(n)]


def flipped(structure):
    """The same legs with every side reversed: a short-the-wings, long-the-body structure."""
    legs = tuple(dataclasses.replace(leg, side=sv.SELL if leg.side == sv.BUY else sv.BUY)
                 for leg in structure.legs)
    return dataclasses.replace(structure, legs=legs)


# ---- names and pinning ----------------------------------------------------

def test_names_are_the_seven_sized_structures():
    assert NAMES == tuple(sorted(FORECAST_SIZED)) and len(NAMES) == 7


@pytest.mark.parametrize("name", NAMES)
def test_pinned_passes_for_every_name(name):
    assert sv.pinned(name).name == name


def test_pinned_refuses_moved_exit(monkeypatch):
    real = STRUCTURES["CND-PS"]
    monkeypatch.setitem(STRUCTURES, "CND-PS", lambda: dataclasses.replace(real(), exit_offset=2))
    with pytest.raises(sv.Refusal) as caught:
        sv.pinned("CND-PS")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    details = caught.value.details
    assert details["reason"] == "factory_default_mismatch"
    assert details["structure"] == "CND-PS" and details["field"] == "exit_offset"


def test_pinned_refuses_moved_entry(monkeypatch):
    real = STRUCTURES["CND-PS"]
    monkeypatch.setitem(STRUCTURES, "CND-PS", lambda: dataclasses.replace(real(), entry_offset=1, exit_offset=2))
    with pytest.raises(sv.Refusal) as caught:
        sv.pinned("CND-PS")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    details = caught.value.details
    assert details["reason"] == "factory_default_mismatch"
    assert details["structure"] == "CND-PS" and details["field"] == "entry_offset"


def test_score_refuses_a_drifted_factory(monkeypatch):
    real = STRUCTURES["CND-PS"]
    monkeypatch.setitem(STRUCTURES, "CND-PS", lambda: dataclasses.replace(real(), exit_offset=2))
    chunk = chunk_of(("A", {}))
    with pytest.raises(sv.Refusal) as caught:
        sv.score(chunk, events_for("A"), taken(chunk), "CND-PS", 0.5)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "factory_default_mismatch"


@pytest.mark.parametrize("index", [0, 1])
def test_pinned_refuses_any_call_leg(monkeypatch, index):
    # Index 0 is the short anchor leg (side SELL) and index 1 a long wing: each refuses on its own.
    real = STRUCTURES["BFLY-P"]()
    legs = list(real.legs)
    legs[index] = dataclasses.replace(legs[index], right=CALL)
    monkeypatch.setitem(STRUCTURES, "BFLY-P", lambda: dataclasses.replace(real, legs=tuple(legs)))
    with pytest.raises(sv.Refusal) as caught:
        sv.pinned("BFLY-P")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "call_leg_refused"


@pytest.mark.parametrize("name", NAMES)
def test_no_call_leg_in_any_of_the_seven(name):
    assert all(leg.right == PUT for leg in STRUCTURES[name]().legs)
    legs, reason = sv.resolve(chunk_of(("A", {})), 0, name, PRED)
    assert reason is None and all(leg["strike"] >= 80 for leg in legs)


# ---- resolve: width, refusals --------------------------------------------

def test_twin_p_width_is_forecast_over_divisor():
    assert forecast_params("TWIN-P", PRED)["width_moneyness"] == pytest.approx(0.04)
    legs, reason = sv.resolve(chunk_of(("A", {})), 0, "TWIN-P", PRED)
    assert reason is None
    assert sorted({leg["strike"] for leg in legs}) == [84.0, 92.0, 96.0, 100.0, 104.0, 108.0, 116.0]


def test_c_divides_the_forecast_before_sizing():
    chunk = chunk_of(("A", {}))
    legs, reason = sv.resolve(chunk, 0, "TWIN-P", PRED, c=1.5)
    assert reason is None
    assert forecast_params("TWIN-P", PRED / 1.5) == pytest.approx({"width_moneyness": 0.04 / 1.5})
    base, _ = sv.resolve(chunk, 0, "TWIN-P", PRED)
    strikes = sorted(leg["strike"] for leg in legs)
    assert strikes != sorted(leg["strike"] for leg in base)
    # resolve builds its factory from forecast_params(name, pred / c) at c=1, so this is that factory's ladder.
    factory, _ = sv.resolve(chunk, 0, "TWIN-P", PRED / 1.5)
    assert strikes == sorted(leg["strike"] for leg in factory)


def test_out_of_range_forecast_declines():
    chunk = chunk_of(("A", {}))
    for pred in (0.1, 200.0, NAN):
        assert sv.resolve(chunk, 0, "TWIN-P", pred) == (None, "width_declined")


def test_resolve_refuses_a_call_leg_factory(monkeypatch):
    # Same call-leg fixture as test_pinned_refuses_any_call_leg, on the short anchor leg; resolve must refuse it too.
    real = STRUCTURES["BFLY-P"]
    def call_leg_factory(**kw):
        spec = real(**kw)
        legs = list(spec.legs)
        legs[0] = dataclasses.replace(legs[0], right=CALL)
        return dataclasses.replace(spec, legs=tuple(legs))
    monkeypatch.setitem(STRUCTURES, "BFLY-P", call_leg_factory)
    with pytest.raises(sv.Refusal) as caught:
        sv.resolve(chunk_of(("A", {})), 0, "BFLY-P", PRED)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "call_leg_refused"


@pytest.mark.parametrize("c", [0.0, -1.0, NAN, math.inf])
def test_bad_c_refuses(c):
    with pytest.raises(ValueError, match="c must be finite and > 0"):
        sv.resolve(chunk_of(("A", {})), 0, "TWIN-P", PRED, c=c)


def test_coarse_ladder_is_too_coarse():
    # Strikes every 10: CND-PS's inner (103) and outer (106) both snap to 110.
    chunk = chunk_of(("A", {}), strikes=range(60, 141, 10))
    assert sv.resolve(chunk, 0, "CND-PS", PRED) == (None, "ladder_too_coarse")


def test_missing_mirror_strike_is_unresolved():
    chunk = chunk_of(("A", {}), strikes=[k for k in range(80, 121) if k != 84])
    assert sv.resolve(chunk, 0, "TWIN-P", PRED) == (None, "structure_unresolved")


def test_negative_floor_refuses(monkeypatch):
    real = STRUCTURES["CND-PS"]
    monkeypatch.setitem(STRUCTURES, "CND-PS", lambda **kw: flipped(real(**kw)))
    assert sv.resolve(chunk_of(("A", {})), 0, "CND-PS", PRED) == (None, "negative_floor")


def test_unflipped_condor_has_a_floor():
    legs, reason = sv.resolve(chunk_of(("A", {})), 0, "CND-PS", PRED)
    assert reason is None
    assert sorted(leg["strike"] for leg in legs if leg["qty"] > 0) == [94.0, 97.0, 103.0, 106.0]


def test_mirrored_leg_takes_the_listed_strike_and_scores(pool):
    # BFLY-P's anchor is the listed 100.1 (bracket below spot 100.5) and its wings sit at +/-6.1 about it, so the
    # upper wing is the listed 106.2 and the mirror is 2*100.1 - 106.2 = 93.99999999999999 in float, not 94.0.
    spot = 100.5
    pred = 100.0 * 6.1 / spot  # BFLY-P's width is pred/100 (outer 1.0), so width * spot = 6.1
    strikes = sorted({float(k) for k in range(80, 121)} | {100.1, 106.2})
    chunk = lad._pack([rec("A", spot, put_legs(strikes=strikes))], {})
    legs, reason = sv.resolve(chunk, 0, "BFLY-P", pred)
    assert reason is None
    listed = set(chunk["strike"].tolist())
    assert {leg["strike"] for leg in legs} <= listed
    assert {leg["strike"] for leg in legs} == {94.0, 100.1, 106.2}
    out = run(chunk, events_for("A", pred=pred), pool, name="BFLY-P").iloc[0]
    assert out["status"] == "scored" and math.isfinite(out["exp_pnl_sim"])


# ---- simulate and the S4 equivalence --------------------------------------

def test_simulate_then_score_equals_expected_pnl(pool):
    chunk, events = chunk_of(("A", {})), events_for("A")
    out = run(chunk, events, pool, name="TWIN-P").iloc[0]
    assert out["status"] == "scored"
    legs, _ = sv.resolve(chunk, 0, "TWIN-P", PRED)
    exits = [{"strike": leg["strike"], "side": "sell" if leg["side"] == "buy" else "buy", "qty": leg["qty"]}
             for leg in legs if leg["qty"] > 0]
    direct = expected_pnl(exit_legs=exits, spot=100.0, entry_cost=out["entry_cost"], pre_iv30=40.0,
                          pred_abs_move=PRED, pred_iv_crush=-15.0, dte_exit=9,
                          event_date=pd.Timestamp("2024-06-10"), pool=pool, key="A", draws=DRAW)
    assert out["exp_pnl_sim"] == pytest.approx(direct["exp_pnl_sim"], abs=1e-9)


def test_unsimulable_spot_is_unsimulable(pool):
    chunk = lad._pack([rec("N", NAN, put_legs())], {})
    sims = sv.simulate(chunk, events_for("N"), pool, draws=DRAW)
    assert sims == {"N": None}
    out = sv.score_events(chunk, events_for("N"), sims, "CND-PS")
    assert out.loc[0, "status"] == "undetermined" and out.loc[0, "reason"] == "unsimulable"


def test_score_events_entry_quotes_and_cost(pool):
    out = run(chunk_of(("A", {})), events_for("A"), pool).iloc[0]
    assert out["entry_cost"] == pytest.approx(2 * 1.36 - 2 * 1.09)  # BUY 94, 106 minus SELL 97, 103
    assert out["width"] == pytest.approx(0.03)
    legs, _ = sv.resolve(chunk_of(("A", {})), 0, "CND-PS", PRED)  # every leg, references included
    mids = [1.0 + 0.01 * (leg["strike"] - 100) ** 2 for leg in legs]  # the 0.2 spread over the mid, per leg
    assert out["rel_spread"] == pytest.approx(np.mean([0.2 / mid for mid in mids]))


def test_score_events_reasons(pool):
    nan_leg = chunk_of(("A", {}), entry=lambda k: (NAN, NAN) if k == 94 else quote(k))
    assert run(nan_leg, events_for("A"), pool).loc[0, "reason"] == "entry_quote_unusable"
    flat = chunk_of(("A", {}), entry=lambda k: (1.0, 2.0))  # buys and sells cancel to exactly 0
    assert run(flat, events_for("A"), pool).loc[0, "reason"] == "entry_not_debit"


def test_debit_of_a_nanodollar_is_not_a_debit(pool):
    # Mids are 1.5 everywhere except the BUY at 94 (mid 1.5 + 1e-9): the entry debit is 1e-9 a share, under MIN_DEBIT.
    thin = chunk_of(("A", {}), entry=lambda k: (1.0, 2.0 + 2e-9) if k == 94 else (1.0, 2.0))
    out = run(thin, events_for("A"), pool).iloc[0]
    assert out["entry_cost"] == pytest.approx(1e-9, abs=1e-12) and out["reason"] == "entry_not_debit"


def test_score_refuses_a_nanodollar_debit():
    thin = chunk_of(("A", {}), entry=lambda k: (1.0, 2.0 + 2e-9) if k == 94 else (1.0, 2.0))
    with pytest.raises(ValueError, match="entry debit below"):
        sv.score(thin, events_for("A"), taken(thin, "A"), "CND-PS", 0.5)


def test_score_events_refuses_bad_alpha_and_c(pool):
    chunk, events = chunk_of(("A", {})), events_for("A")
    sims = sv.simulate(chunk, events, pool, draws=DRAW)
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
        sv.score_events(chunk, events, sims, "CND-PS", alpha=2.0)
    with pytest.raises(ValueError, match="c must be finite"):
        sv.score_events(chunk, events, sims, "CND-PS", c=0.0)


# ---- decide ---------------------------------------------------------------

def test_thin_history_with_fewer_than_100_prior_scores():
    frame = scored_frame(scored_row("hi", pnl=1.0), *prior_scores(99))
    out = sv.decide(frame).set_index("event_id")
    assert out.loc["hi", "status"] == "undetermined" and out.loc["hi", "reason"] == "thin_history"


def test_taken_and_below_cutoff_with_enough_history():
    frame = scored_frame(scored_row("hi", pnl=1.0), scored_row("lo", pnl=0.5), *prior_scores())
    out = sv.decide(frame).set_index("event_id")
    assert out.loc["hi", "cutoff"] == pytest.approx(0.952)
    assert out.loc["hi", "status"] == "taken" and pd.isna(out.loc["hi", "reason"])
    assert out.loc["lo", "status"] == "not_taken" and out.loc["lo", "reason"] == "below_cutoff"


def test_universe_filters_and_boundaries():
    frame = scored_frame(scored_row("small", pnl=1.0, mcap=MCAP_FLOOR - 1), scored_row("wide", pnl=1.0, rel=0.3),
                         scored_row("floor", mcap=MCAP_FLOOR), scored_row("edge", rel=MAX_REL_SPREAD),
                         *prior_scores())
    out = sv.decide(frame).set_index("event_id")
    for eid in ("small", "wide"):
        assert out.loc[eid, "status"] == "not_taken" and out.loc[eid, "reason"] == "universe_filter"
    for eid in ("floor", "edge"):  # exactly on the boundary is not filtered
        assert out.loc[eid, "reason"] != "universe_filter"
        assert out.loc[eid, "status"] in ("taken", "not_taken") and out.loc[eid, "cutoff"] == pytest.approx(0.952)


def test_universe_failure_with_usable_cutoff_is_universe_filter():
    frame = scored_frame(scored_row("wide", pnl=1.0, rel=MAX_REL_SPREAD + 0.01), *prior_scores())
    out = sv.decide(frame).set_index("event_id")
    assert out.loc["wide", "status"] == "not_taken" and out.loc["wide", "reason"] == "universe_filter"
    assert out.loc["wide", "cutoff"] == pytest.approx(0.952)


def test_below_cutoff_with_passing_universe_is_below_cutoff():
    frame = scored_frame(scored_row("lo", pnl=0.5, mcap=MCAP_FLOOR, rel=MAX_REL_SPREAD), *prior_scores())
    out = sv.decide(frame).set_index("event_id")
    assert out.loc["lo", "status"] == "not_taken" and out.loc["lo", "reason"] == "below_cutoff"


def test_nan_spread_with_thin_history_is_thin_history():
    frame = scored_frame(scored_row("s", rel=NAN), *prior_scores(99))
    out = sv.decide(frame).set_index("event_id")
    assert out.loc["s", "status"] == "undetermined" and out.loc["s", "reason"] == "thin_history"


def test_missing_mcap_and_missing_spread_are_undetermined():
    frame = scored_frame(scored_row("m", mcap=NAN), scored_row("s", rel=NAN),
                         scored_row("ms", mcap=NAN, rel=0.3), *prior_scores())
    out = sv.decide(frame).set_index("event_id")
    assert out.loc["m", "reason"] == "missing_mcap" and out.loc["m", "status"] == "undetermined"
    assert out.loc["s", "reason"] == "missing_spread" and out.loc["s", "status"] == "undetermined"
    assert out.loc["ms", "reason"] == "missing_mcap" and out.loc["ms", "status"] == "undetermined"


def test_decide_does_not_mutate_input():
    frame = scored_frame(scored_row("hi", pnl=1.0), *prior_scores())
    before = frame.copy()
    sv.decide(frame)
    pd.testing.assert_frame_equal(frame, before)


# ---- score: marks, rungs, lots --------------------------------------------

def test_rung1_marks_and_lots():
    chunk = chunk_of(("A", {94: (2.0, 2.0, 1), 106: (2.0, 2.0, 1)}))
    scored = sv.score(chunk, events_for("A"), taken(chunk, "A"), "CND-PS", 0.5)
    row = scored.iloc[0]
    assert row["rung"] == "rung1" and row["exit_value"] == pytest.approx(4.0 - 2.0)
    assert row["entry_cost"] == pytest.approx(0.54)
    assert row["lots"] == lots(0.54) == 185
    assert row["pnl_per_share"] == pytest.approx(2.0 - 0.54)
    assert row["pnl_dollars"] == pytest.approx((2.0 - 0.54) * CONTRACT_MULTIPLIER * 185)


def test_rung2_leg_marks_at_zero_alpha_and_rung1_leg_uses_alpha():
    chunk = chunk_of(("A", {94: (1.0, 2.0, 1), 106: (1.0, 2.0, 2)}))
    rung2 = sv.score(chunk, events_for("A"), taken(chunk, "A"), "CND-PS", 0.5).iloc[0]
    assert rung2["rung"] == "rung2" and rung2["exit_value"] == pytest.approx(1.5 + 1.0 - 2.0)
    # Moving alpha moves only the rung-1 leg (94): it sells at 2.0; the rung-2 leg stays on its bid at 1.0.
    assert sv.score(chunk, events_for("A"), taken(chunk, "A", alpha=1.0), "CND-PS", 1.0).iloc[0]["exit_value"] == \
        pytest.approx(2.0 + 1.0 - 2.0)


def test_rung3_marks_zero_and_drop_unquoted_removes_it():
    chunk = chunk_of(("A", {106: (NAN, NAN, 3)}), ("B", {}))
    events, decisions = events_for("A", "B"), taken(chunk, "A", "B")
    scored = sv.score(chunk, events, decisions, "CND-PS", 0.5).set_index("event_id")
    assert scored.loc["A", "exit_value"] == 0.0 and scored.loc["A", "rung"] == EXIT_UNQUOTED
    assert scored.loc["B", "rung"] == "rung1"
    kept = sv.score(chunk, events, decisions, "CND-PS", 0.5, drop_unquoted=True)
    assert list(kept["event_id"]) == ["B"]


def test_rung_shares_and_empty():
    chunk = chunk_of(("A", {}), ("B", {106: (NAN, NAN, 3)}))
    scored = sv.score(chunk, events_for("A", "B"), taken(chunk, "A", "B"), "CND-PS", 0.5)
    assert sv.rung_shares(scored) == {"rung1": 0.5, EXIT_UNQUOTED: 0.5}
    assert sv.rung_shares(scored.iloc[:0]) == {}


def test_score_taken_event_absent_from_chunk_raises():
    chunk = chunk_of(("A", {}))
    with pytest.raises(ValueError, match=r"taken event 'Z' is missing from the chunk or events"):
        sv.score(chunk, events_for("A"), taken(chunk, "Z"), "CND-PS", 0.5)


def test_score_refuses_invalid_alpha_even_with_nothing_taken():
    chunk = chunk_of(("A", {}))
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\], got 2\.0"):
        sv.score(chunk, events_for("A"), taken(chunk), "CND-PS", 2.0)


def test_score_ignores_a_taken_row_for_another_structure():
    chunk, events = chunk_of(("A", {})), events_for("A")
    assert sv.score(chunk, events, taken(chunk, "A", name="TWIN-P"), "CND-PS", 0.5).empty


def test_score_needs_a_structure_column():
    chunk, events = chunk_of(("A", {})), events_for("A")
    bare = pd.DataFrame({"event_id": ["A"], "status": ["taken"]})
    with pytest.raises(ValueError, match="structure column"):
        sv.score(chunk, events, bare, "CND-PS", 0.5)


def test_score_binds_to_the_pricing_alpha(pool):
    chunk, events = chunk_of(("A", {})), events_for("A")
    priced = run(chunk, events, pool).assign(status="taken")  # priced at alpha 0.5, c 1.0
    with pytest.raises(ValueError, match="decision was not priced at this alpha and c"):
        sv.score(chunk, events, priced, "CND-PS", 0.25)
    same = sv.score(chunk, events, priced, "CND-PS", 0.5)
    assert len(same) == 1 and same.iloc[0]["entry_cost"] == pytest.approx(priced.iloc[0]["entry_cost"])


def test_score_needs_an_entry_cost_column():
    chunk, events = chunk_of(("A", {})), events_for("A")
    bare = pd.DataFrame({"event_id": ["A"], "status": ["taken"], "structure": ["CND-PS"]})
    with pytest.raises(ValueError, match="entry_cost column"):
        sv.score(chunk, events, bare, "CND-PS", 0.5)


def test_score_refuses_a_nan_entry_cost():
    chunk, events = chunk_of(("A", {})), events_for("A")
    decisions = taken(chunk, "A").assign(entry_cost=NAN)
    with pytest.raises(ValueError, match="decision was not priced at this alpha and c"):
        sv.score(chunk, events, decisions, "CND-PS", 0.5)
