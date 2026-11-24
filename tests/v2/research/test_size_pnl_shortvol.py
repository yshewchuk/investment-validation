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
from engine.portfolio import CAPITAL_PER_TRADE, CONTRACT_MULTIPLIER
from engine.score import _mean_relative_spread
from engine.structures import BUY, CALL, PUT, STRUCTURES, ChainSnapshot, ExpirySelector, price_structure
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


def taken(chunk, *eids, name="CND-PS", alpha=0.5):
    """Taken decisions for ``eids``, each carrying the entry cost the module prices at ``alpha``.

    An event absent from the chunk, or with no entry cost, gets NaN, so ``score`` refuses it on its own checks.
    """
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    costs = []
    for eid in eids:
        cost = NAN
        if eid in pos:
            priced, legs, _ = sv.resolve_priced(chunk, pos[eid], name, PRED, FillModel(alpha))
            entered = None if legs is None else sv.entry(priced)
            if entered is not None:
                cost = entered[0]
        costs.append(cost)
    return pd.DataFrame({"event_id": list(eids), "status": ["taken"] * len(eids),
                         "structure": [name] * len(eids), "entry_cost": costs,
                         "alpha": [alpha] * len(eids)})


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
            "status": "scored", "reason": None, "alpha": 0.5}


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


def test_pinned_refuses_a_leg_expiry_off_the_ladder_selector(monkeypatch):
    real = STRUCTURES["TWIN-P"]
    off = ExpirySelector(kind="first_post_event", max_dte=30)
    monkeypatch.setitem(STRUCTURES, "TWIN-P",
                        lambda: dataclasses.replace(real(), legs=tuple(dataclasses.replace(leg, expiry=off)
                                                                       for leg in real().legs)))
    with pytest.raises(sv.Refusal) as caught:
        sv.pinned("TWIN-P")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.details["reason"] == "factory_default_mismatch"
    assert caught.value.details["structure"] == "TWIN-P"
    assert caught.value.details["field"] == real().legs[0].name


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

def test_twin_p_width_is_production_rule_at_k_1():
    assert forecast_params("TWIN-P", PRED)["width_moneyness"] == pytest.approx(0.04)
    legs, reason = sv.resolve(chunk_of(("A", {})), 0, "TWIN-P", PRED)
    assert reason is None
    assert sorted({leg["strike"] for leg in legs}) == [84.0, 92.0, 96.0, 100.0, 104.0, 108.0, 116.0]


def test_width_params_is_production_forecast_params():
    # Amendment 6: short vol is as-is only, so the width is production's rule at k=1 with no divisor.
    assert sv._width_params("BFLY-P", 0.05) == forecast_params("BFLY-P", 0.05)


def test_score_events_rejects_the_removed_c_keyword():
    chunk, events = chunk_of(("A", {})), events_for("A")
    sims = {}
    with pytest.raises(TypeError):
        sv.score_events(chunk, events, sims, "CND-PS", alpha=0.5, c=1.0)


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


def test_coarse_ladder_is_too_coarse():
    # Strikes every 10: CND-PS's inner (103) and outer (106) both snap to 110.
    chunk = chunk_of(("A", {}), strikes=range(60, 141, 10))
    assert sv.resolve(chunk, 0, "CND-PS", PRED) == (None, "ladder_too_coarse")


def test_missing_mirror_strike_is_unresolved():
    chunk = chunk_of(("A", {}), strikes=[k for k in range(80, 121) if k != 84])
    assert sv.resolve(chunk, 0, "TWIN-P", PRED) == (None, "structure_unresolved")


def test_flipped_condor_resolves_without_a_floor_drop(monkeypatch):
    # The EXP-133 zero floor is gone: a short-the-wings condor resolves like any other.
    real = STRUCTURES["CND-PS"]
    monkeypatch.setitem(STRUCTURES, "CND-PS", lambda **kw: flipped(real(**kw)))
    legs, reason = sv.resolve(chunk_of(("A", {})), 0, "CND-PS", PRED)
    assert reason is None and len(legs) > 0
    # Pin the flipped orientation: resolved (side, qty) equals the flipped spec's, for the legs with qty > 0.
    want = [(leg.side, leg.qty) for leg in flipped(real()).legs if leg.qty > 0]
    got = [(leg["side"], leg["qty"]) for leg in legs if leg["qty"] > 0]
    assert got == want and len(got) > 0


def test_flipped_condor_is_scored_with_a_hand_oracle(monkeypatch):
    # Flipped CND-PS SELLS 94 and 106 and BUYS 97 and 103 at entry. At alpha 0.5 (mid) the chunk's mids give
    # cost = 2*1.09 - 2*1.36 = -0.54 a share: not a debit, so the row is undetermined as entry_not_debit, not scored.
    real = STRUCTURES["CND-PS"]
    monkeypatch.setitem(STRUCTURES, "CND-PS", lambda **kw: flipped(real(**kw)))
    chunk = chunk_of(("A", {}))
    grid = np.arange(80.0, 121.0)
    means = 0.001 * (grid - 80.0) ** 2
    out = sv.score_events(chunk, events_for("A"), {"A": (grid, means)}, "CND-PS").iloc[0]
    assert out["entry_cost"] == pytest.approx(2 * 1.09 - 2 * 1.36)  # -0.54
    assert out["status"] == "undetermined" and out["reason"] == "entry_not_debit"
    assert math.isnan(out["exp_pnl_sim"])


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
    _priced, legs, _ = sv.resolve_priced(chunk_of(("A", {})), 0, "CND-PS", PRED, FillModel(0.5))  # every leg, refs incl.
    mids = [1.0 + 0.01 * (leg["strike"] - 100) ** 2 for leg in legs]  # the 0.2 spread over the mid, per leg
    assert out["rel_spread"] == pytest.approx(np.mean([0.2 / mid for mid in mids]))


def test_score_events_reasons(pool):
    nan_leg = chunk_of(("A", {}), entry=lambda k: (NAN, NAN) if k == 94 else quote(k))
    assert run(nan_leg, events_for("A"), pool).loc[0, "reason"] == "entry_quote_unusable"
    flat = chunk_of(("A", {}), entry=lambda k: (1.0, 2.0))  # buys and sells cancel to exactly 0
    out = run(flat, events_for("A"), pool).iloc[0]
    assert out["reason"] == "entry_not_debit" and out["status"] == "undetermined" and math.isnan(out["exp_pnl_sim"])


def test_score_events_negative_cost_is_not_a_debit(pool):
    # Sells (97, 103) mid 2.5 each exceed buys (94, 106) mid 1.5 each: the entry cost is -2.0 a share.
    neg = chunk_of(("A", {}), entry=lambda k: (2.0, 3.0) if k in (97, 103) else (1.0, 2.0))
    out = run(neg, events_for("A"), pool).iloc[0]
    assert out["entry_cost"] == pytest.approx(-2.0)
    assert out["reason"] == "entry_not_debit" and out["status"] == "undetermined" and math.isnan(out["exp_pnl_sim"])


def test_debit_of_a_nanodollar_is_simulated():
    # Mids are 1.5 everywhere except the BUY at 94 (mid 1.5 + 1e-9): the entry debit is 1e-9 a share, and any positive debit is simulated.
    thin = chunk_of(("A", {}), entry=lambda k: (1.0, 2.0 + 2e-9) if k == 94 else (1.0, 2.0))
    # Hand-supplied simulation on the chunk's strike grid 80..120. A quadratic is used rather than a line: a line is
    # symmetric about the condor's centre (100), so the BUY and SELL terms would cancel and the oracle would be -1 for any sign.
    grid = np.arange(80.0, 121.0)
    means = 0.001 * (grid - 80.0) ** 2
    sims = {"A": (grid, means)}
    out = sv.score_events(thin, events_for("A"), sims, "CND-PS").iloc[0]
    assert out["status"] == "scored" and math.isfinite(out["exp_pnl_sim"]) and pd.isna(out["reason"])
    # Independent oracle, by hand from the literals. CND-PS traded legs, one contract each (as in the cost assertion
    # 2*1.36 - 2*1.09): BUY 94 and 106, SELL 97 and 103. A BUY leg closes at +mean, a SELL leg at -mean.
    def mean_at(k):
        return 0.001 * (k - 80.0) ** 2
    value = mean_at(94) + mean_at(106) - mean_at(97) - mean_at(103)  # 0.196 + 0.676 - 0.289 - 0.529 = 0.054
    cost = 1e-9
    assert out["entry_cost"] == pytest.approx(cost, abs=1e-12)
    assert out["exp_pnl_sim"] == pytest.approx((value - cost) / cost, rel=1e-3)


def test_score_accepts_a_nanodollar_debit():
    thin = chunk_of(("A", {}), entry=lambda k: (1.0, 2.0 + 2e-9) if k == 94 else (1.0, 2.0))
    scored = sv.score(thin, events_for("A"), taken(thin, "A"), "CND-PS", 0.5)
    assert len(scored) == 1 and scored.iloc[0]["entry_cost"] == pytest.approx(1e-9, abs=1e-12)
    row = scored.iloc[0]
    # Exit quotes default to locked at 1.0 on rung 1 (put_legs), so every leg marks at 1.0 under any alpha.
    # One contract each: BUY 94 and 106 at +1.0 each, SELL 97 and 103 at -1.0 each, so exit = 1 + 1 - 1 - 1 = 0.0.
    assert row["rung"] == "rung1"
    assert row["exit_value"] == pytest.approx(0.0, abs=1e-9)
    assert row["pnl_per_share"] < 0
    assert row["pnl_per_share"] == pytest.approx(-1e-9, rel=1e-3)  # exit 0.0 less the 1e-9 debit
    # Lots from the capital rule by hand, on the row's own entry cost (about 1e-9, so the count is ~1e11 and float
    # noise in the cost matters: use the cost the row carries, not the literal).
    assert row["lots"] == math.floor(CAPITAL_PER_TRADE / (row["entry_cost"] * 100))


def test_score_refuses_a_non_debit_entry():
    # Sells (97, 103) mid 2.5 exceed buys (94, 106) mid 1.5: the entry cost is -2.0 a share, and score refuses it.
    neg = chunk_of(("A", {}), entry=lambda k: (2.0, 3.0) if k in (97, 103) else (1.0, 2.0))
    decisions = taken(neg, "A")  # priced at alpha 0.5 from the chunk, so the bound check passes and the debit check fires
    assert decisions.loc[0, "entry_cost"] == pytest.approx(-2.0) and decisions.loc[0, "alpha"] == 0.5
    with pytest.raises(ValueError, match="not a debit"):
        sv.score(neg, events_for("A"), decisions, "CND-PS", 0.5)


def test_score_events_refuses_bad_alpha(pool):
    chunk, events = chunk_of(("A", {})), events_for("A")
    sims = sv.simulate(chunk, events, pool, draws=DRAW)
    with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
        sv.score_events(chunk, events, sims, "CND-PS", alpha=2.0)


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
    priced = run(chunk, events, pool).assign(status="taken")  # priced at alpha 0.5
    with pytest.raises(ValueError, match="decision was not priced at this alpha"):
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
    with pytest.raises(ValueError, match="decision was not priced at this alpha"):
        sv.score(chunk, events, decisions, "CND-PS", 0.5)


def test_score_needs_an_alpha_column():
    chunk, events = chunk_of(("A", {})), events_for("A")
    decisions = taken(chunk, "A").drop(columns="alpha")
    with pytest.raises(ValueError, match="alpha column"):
        sv.score(chunk, events, decisions, "CND-PS", 0.5)


# ---- Amendment 1 sensitivity: rung-2 legs marked at alpha (mid_marks) ----

def test_mid_marks_keeps_the_taken_set_and_changes_only_the_rung2_marks():
    chunk = chunk_of(("A", {94: (1.0, 2.0, 2)}), ("B", {}))
    events, decisions = events_for("A", "B"), taken(chunk, "A", "B")
    primary = sv.score(chunk, events, decisions, "CND-PS", 0.5)
    pd.testing.assert_frame_equal(primary, sv.score(chunk, events, decisions, "CND-PS", 0.5, mid_marks=False))  # default unchanged
    mid = sv.score(chunk, events, decisions, "CND-PS", 0.5, mid_marks=True).set_index("event_id")
    primary = primary.set_index("event_id")
    assert list(mid.index) == list(primary.index) == ["A", "B"]
    # Event A, by hand. CND-PS legs (one contract each): BUY 94, SELL 97, SELL 103, BUY 106. Only 94 is rung 2 (bid 1.0, ask 2.0);
    # every other leg has bid = ask = 1.0 at exit, so any alpha marks it at 1.0.
    # Entry at alpha 0.5 (mid): 1.36 - 1.09 - 1.09 + 1.36 = 0.54. lots = floor(10000 / (0.54 * 100)) = 185.
    # Primary: the BUY 94 closes by selling at its bid (alpha 0): +1.0; exit = 1.0 - 1.0 - 1.0 + 1.0 = 0.0.
    # Mid: the BUY 94 closes by selling at bid + 0.5 * (ask - bid) = 1.5; exit = 1.5 - 1.0 - 1.0 + 1.0 = 0.5.
    # pnl_per_share = exit - 0.54; pnl_dollars = pnl_per_share * 100 * 185.
    assert primary.loc["A", "rung"] == "rung2" and mid.loc["A", "rung"] == "rung2"
    assert primary.loc["A", "entry_cost"] == pytest.approx(0.54) and mid.loc["A", "entry_cost"] == pytest.approx(0.54)
    assert primary.loc["A", "exit_value"] == pytest.approx(0.0)
    assert mid.loc["A", "exit_value"] == pytest.approx(0.5)
    assert primary.loc["A", "pnl_per_share"] == pytest.approx(-0.54)
    assert mid.loc["A", "pnl_per_share"] == pytest.approx(-0.04)
    assert primary.loc["A", "pnl_dollars"] == pytest.approx(-9990.0)  # -0.54 * 100 * 185
    assert mid.loc["A", "pnl_dollars"] == pytest.approx(-740.0)  # -0.04 * 100 * 185
    assert mid.loc["B", "exit_value"] == pytest.approx(primary.loc["B", "exit_value"])  # rung 1 is marked the same under both


def test_mid_marks_equals_the_primary_when_every_leg_is_rung1():
    chunk, events = chunk_of(("A", {})), events_for("A")
    decisions = taken(chunk, "A")
    pd.testing.assert_frame_equal(sv.score(chunk, events, decisions, "CND-PS", 0.5, mid_marks=True),
                                  sv.score(chunk, events, decisions, "CND-PS", 0.5))


def test_mid_marks_keeps_a_rung3_row_at_zero_with_its_label():
    chunk, events = chunk_of(("A", {106: (NAN, NAN, 3)})), events_for("A")
    decisions = taken(chunk, "A")
    for mid_marks in (False, True):
        row = sv.score(chunk, events, decisions, "CND-PS", 0.5, mid_marks=mid_marks).iloc[0]
        assert row["exit_value"] == 0.0 and row["rung"] == EXIT_UNQUOTED


# ---- secured cash (Amendment 2) -------------------------------------------

def test_secured_matches_the_sold_put_legs_and_ignores_score_lots():
    from experiments import size_pnl_account as acct

    chunk = chunk_of(("A", {94: (2.0, 2.0, 1), 106: (2.0, 2.0, 1)}))
    out = sv.secured(chunk, events_for("A"), taken(chunk, "A"), "CND-PS")
    legs, _ = sv.resolve(chunk, 0, "CND-PS", PRED)
    expected = sum(leg["qty"] * leg["strike"] * 100 for leg in legs if leg["side"] == "sell" and leg["qty"] > 0)
    assert len(out) == 1 and out.iloc[0]["event_id"] == "A"
    assert out.iloc[0]["secured_per_contract"] == pytest.approx(expected) and expected > 0
    assert out.iloc[0]["secured_per_contract"] == pytest.approx(acct.secured_per_contract(legs))


def test_secured_returns_only_taken_rows():
    chunk = chunk_of(("A", {}))
    decisions = taken(chunk, "A").assign(status="not_taken")
    assert len(sv.secured(chunk, events_for("A"), decisions, "CND-PS")) == 0


# ---- production pricing: chain snapshot, resolve, cost and spread (A6-2a) ---

def test_chain_snapshot_carries_only_this_events_puts():
    chunk = chunk_of(("A", {}), ("B", {}))
    snap = sv.chain_snapshot(chunk, 1)
    assert isinstance(snap, ChainSnapshot) and snap.session is None
    assert set(snap.rows["right"]) == {"P"} and list(snap.rows["strike"]) == [float(k) for k in range(80, 121)]
    assert snap.spot == 100.0 and snap.event_date == pd.Timestamp("2024-03-15")
    assert (snap.rows["expiry"] == pd.Timestamp("2024-03-15")).all() and (snap.rows["dte"] == 30).all()


@pytest.mark.parametrize("name", NAMES)
def test_resolve_legs_are_price_structures_legs(name):
    chunk = chunk_of(("A", {}))
    legs, reason = sv.resolve(chunk, 0, name, PRED)
    priced = price_structure(STRUCTURES[name](**forecast_params(name, PRED)), sv.chain_snapshot(chunk, 0),
                             FillModel(0.5))
    assert reason is None and len(legs) == len(priced.legs)
    for leg, want in zip(legs, priced.legs):
        assert (leg["name"], leg["side"], leg["qty"]) == (want.name, want.side, want.qty)
        assert leg["strike"] == pytest.approx(want.strike)


def test_cost_and_spread_are_production_on_all_resolved_legs():
    # Reference legs (qty 0) get a 0.6 spread and traded legs 0.2, so the reference legs move the mean.
    name = next(n for n in NAMES if any(leg.qty == 0 for leg in sv.resolve_priced(
        chunk_of(("A", {})), 0, n, PRED, FillModel(0.5))[0].legs))
    refs = [leg.strike for leg in sv.resolve_priced(chunk_of(("A", {})), 0, name, PRED, FillModel(0.5))[0].legs
            if leg.qty == 0]

    def wide(k):
        mid = 1.0 + 0.01 * (k - 100) ** 2
        half = 0.3 if any(np.isclose(k, r) for r in refs) else 0.1
        return mid - half, mid + half

    priced, legs, reason = sv.resolve_priced(chunk_of(("A", {}), entry=wide), 0, name, PRED, FillModel(0.5))
    assert reason is None and legs is not None
    cost, rel = sv.entry(priced)
    assert cost == pytest.approx(priced.cost)
    assert rel == pytest.approx(_mean_relative_spread(priced))
    traded = [leg for leg in priced.legs if leg.qty > 0]
    assert rel != pytest.approx(np.mean([(leg.ask - leg.bid) / (0.5 * (leg.bid + leg.ask)) for leg in traded]))


def test_crossed_entry_quote_on_a_traded_strike_is_unusable_not_raised(pool):
    # FillModel raises ValueError on bid > ask; resolve_priced maps that to a reason, not an exception.
    chunk = chunk_of(("A", {}), entry=lambda k: (2.0, 1.0) if k == 94 else quote(k))
    assert sv.resolve_priced(chunk, 0, "CND-PS", PRED, FillModel(0.5)) == (None, None, "entry_quote_unusable")
    assert run(chunk, events_for("A"), pool).loc[0, "reason"] == "entry_quote_unusable"


