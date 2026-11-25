"""EXP-188 pricing twin (experiments/size_pnl_pricing.py): the SHORTVOL and STRADDLE copies must agree with the originals
bit for bit, and the driver's scoring must not depend on which pricer is wired. Synthetic ladders and a stub pool only."""
import numpy as np
import pandas as pd
import pytest

from engine.chain_index import price_structure_indexed
from engine.fills import FillModel
from engine.structures import PUT, price_structure
from experiments import size_pnl_driver as driver
from experiments import size_pnl_equivalence as equivalence
from experiments import size_pnl_gprod as gprod
from experiments import size_pnl_ladder as lad
from experiments import size_pnl_pricing as pricing
from experiments import size_pnl_shortvol as sv
from experiments import size_pnl_straddle as sd
from experiments.size_pnl_pool import build_pool
from tests.v2.research.test_size_pnl_shortvol import DRAW, PRED, _tables, chunk_of, events_for, put_legs, quote, rec
from tests.v2.research.test_size_pnl_straddle import pair

NAN = float("nan")
ALPHAS = (0.0, 0.5, 1.0)
HIST = 110  # warm-up events: trailing_cutoff needs min_window=100 scored rows before the targets; 10 more as a margin
HIST_DATE, TARGET_DATE = "2024-01-15", "2024-06-10"
TARGETS = ["T0", "T1", "T2", "C0", "M0", "X0", "W0"]
MID = "prod"
JAN, JUN = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01")


def tight(k):
    """Entry (bid, ask) at strike k: a 0.04 spread, inside MAX_REL_SPREAD."""
    mid = 1.0 + 0.01 * (k - 100) ** 2
    return mid - 0.02, mid + 0.02


def wide(k):
    """Entry (bid, ask) at strike k: an 0.8 spread, outside MAX_REL_SPREAD (decide filters it)."""
    mid = 1.0 + 0.01 * (k - 100) ** 2
    return mid - 0.4, mid + 0.4


def crossed(k):
    return (2.0, 1.0) if k == 94 else tight(k)


def _atm_call():
    """The ATM call row (right=1, tight quotes, plain exit) that STR-THRU needs. Short vol reads puts only, so it changes no short-vol result."""
    return [(1, 100.0, *tight(100), 1.0, 1.0, 1)]


def _ladders():
    """Short-vol ladders: normal history, then targets for each refusal and exit case. Every event also carries the ATM call."""
    recs = [rec(f"H{j}", 100.0, put_legs(entry=tight) + _atm_call()) for j in range(HIST)]
    recs += [rec("T0", 100.0, put_legs(entry=tight) + _atm_call()),
             rec("T1", 100.0, put_legs(entry=tight, exit_={94: (1.0, 2.0, 2)}) + _atm_call()),  # rung 2 on a traded strike
             rec("T2", 100.0, put_legs(entry=tight, exit_={106: (NAN, NAN, 3)}) + _atm_call()),  # rung 3 on a traded strike
             rec("C0", 100.0, put_legs(strikes=range(60, 141, 10), entry=tight) + _atm_call()),  # ladder_too_coarse
             rec("M0", 100.0, put_legs(strikes=[k for k in range(80, 121) if k != 84], entry=tight) + _atm_call()),  # no 84 mirror
             rec("X0", 100.0, put_legs(entry=crossed) + _atm_call()),  # entry_quote_unusable
             rec("W0", 100.0, put_legs(entry=wide) + _atm_call())]
    return lad._pack(recs, {})


def _events():
    """Driver-shaped events: the short-vol columns plus the straddle gate's forecast columns (edge = pred_abs_move - implied_move)."""
    hist = events_for(*[f"H{j}" for j in range(HIST)], date=HIST_DATE)
    frame = pd.concat([hist, events_for(*TARGETS, date=TARGET_DATE)], ignore_index=True)
    return frame.assign(pred_abs_move=PRED, implied_move=5.0)


def _same(got, want):
    pd.testing.assert_frame_equal(got, want, check_exact=True)


@pytest.fixture(scope="module")
def pool():
    return build_pool(*_tables(), floor=None)


# ---- a. wiring ----------------------------------------------------------------

def test_the_twin_is_wired_into_every_experiment_caller():
    assert driver.shortvol is pricing.SHORTVOL and driver.straddle is pricing.STRADDLE
    assert equivalence.shortvol is pricing.SHORTVOL and gprod.straddle is pricing.STRADDLE
    assert pricing.SHORTVOL is not sv and pricing.STRADDLE is not sd
    assert pricing.SHORTVOL.price_structure is price_structure_indexed
    assert pricing.STRADDLE.price_structure is price_structure_indexed
    assert sv.price_structure is price_structure and sd.price_structure is price_structure
    assert pricing.SHORTVOL.Refusal is sv.Refusal and driver.Refusal is sv.Refusal


# ---- b. short vol ---------------------------------------------------------------

def test_short_vol_twin_equals_the_original_on_every_name_and_alpha(pool):
    chunk, events = _ladders(), _events()
    targets = events[events["event_id"].isin(TARGETS)]
    pos = {str(e): i for i, e in enumerate(chunk["event_id"])}
    tpos = [pos[t] for t in TARGETS]
    sims = sv.simulate(chunk, events, pool, draws=DRAW)
    reasons, rungs, scored, taken = set(), set(), 0, 0
    for name in sv.NAMES:
        for alpha in ALPHAS:
            # Full history only at alpha 0.5, where decide needs the trailing bar. score_events is one row per event, so the
            # target rows at the other alphas are the rows the full history would give for those events.
            full = alpha == 0.5
            want = sv.score_events(chunk, events if full else targets, sims, name, alpha=alpha)
            _same(pricing.SHORTVOL.score_events(chunk, events if full else targets, sims, name, alpha=alpha), want)
            reasons |= set(want["reason"].dropna())
            scored += int((want["status"] == "scored").sum())
            sets = []
            if full:
                decided = sv.decide(want)
                _same(pricing.SHORTVOL.decide(want), decided)
                taken += int((decided["status"] == "taken").sum())
                sets.append(decided[decided["status"] == "taken"])
            for j in tpos:
                fill = FillModel(alpha)
                assert pricing.SHORTVOL.resolve_priced(chunk, j, name, PRED, fill) == sv.resolve_priced(chunk, j, name, PRED, fill)
            # The decided taken rows, and every scored target as taken (so every exit case is marked, whatever decide says).
            on_targets = want[want["event_id"].isin(TARGETS)]
            sets.append(on_targets[on_targets["status"] == "scored"].assign(status="taken"))
            for rows in sets:
                for kw in ({}, {"mid_marks": True}, {"repriced": True}):
                    got = pricing.SHORTVOL.score(chunk, events, rows, name, alpha, **kw)
                    _same(got, sv.score(chunk, events, rows, name, alpha, **kw))
                    rungs |= set(got["rung"])
                _same(pricing.SHORTVOL.secured(chunk, events, rows, name), sv.secured(chunk, events, rows, name))
    assert {"ladder_too_coarse", "structure_unresolved", "entry_quote_unusable"} <= reasons
    assert {"rung1", "rung2", sd.EXIT_UNQUOTED} <= rungs
    assert scored > 0 and taken > 0


# ---- c. straddle ----------------------------------------------------------------

TAKEN = {"S0", "S1", "S2"}


def _gate(event):
    """A gate that takes three events and refuses the rest, so decide returns a mix."""
    return ("taken", None) if event["event_id"] in TAKEN else ("not_taken", "below_gate_threshold")


def _straddle_chunk():
    """ATM straddles: healthy, rung-2 and rung-3 exits, a crossed put, a call without a put, and a non-finite spot."""
    recs = [rec("S0", 100.0, pair(100)),
            rec("S1", 100.0, pair(100, call_x=(1.0, 1.4, 2), put_x=(0.6, 1.0, 2))),
            rec("S2", 100.0, pair(100, put_x=(NAN, NAN, 3))),
            rec("S3", 100.0, pair(100, put_e=(2.2, 2.0))),
            rec("S4", 100.0, [(1, 100, 2.0, 2.2, 1.0, 1.4, 1)]),
            rec("S5", NAN, pair(100)),
            rec("S6", 100.0, pair(100))]
    return lad._pack(recs, {})


def _straddle_events(ids):
    return pd.DataFrame({"event_id": ids, "event_date": pd.Timestamp(TARGET_DATE), "pred_abs_move": 6.0,
                         "implied_move": 5.0, "mcap_usd": 20e9})


def test_straddle_twin_equals_the_original_on_resolve_decide_and_score():
    chunk, ids = _straddle_chunk(), [f"S{j}" for j in range(7)]
    events = _straddle_events(ids)
    reasons, rungs = set(), set()
    for j in range(len(ids)):
        for alpha in ALPHAS:
            want = sd.resolve(chunk, j, alpha)
            assert pricing.STRADDLE.resolve(chunk, j, alpha) == want
            reasons.add(want[3])
    decided = sd.decide(events, chunk, _gate)
    _same(pricing.STRADDLE.decide(events, chunk, _gate), decided)
    taken = int((decided["status"] == "taken").sum())
    for alpha in ALPHAS:
        for mid in (False, True):
            for drop in (False, True):
                want = sd.score(chunk, events, decided, alpha, drop_unquoted=drop, mid_marks=mid)
                _same(pricing.STRADDLE.score(chunk, events, decided, alpha, drop_unquoted=drop, mid_marks=mid), want)
                rungs |= set(want["rung"])
    assert {"no_atm_pair", "entry_quote_unusable"} <= reasons
    assert {"rung1", "rung2", sd.EXIT_UNQUOTED} <= rungs
    assert taken > 0


# ---- d. cache -------------------------------------------------------------------

def test_short_vol_snapshot_is_reused_per_event_and_fresh_for_a_new_chunk():
    chunk = chunk_of(("A", {}), ("B", {}))
    first = pricing.SHORTVOL.chain_snapshot(chunk, 1)
    assert pricing.SHORTVOL.chain_snapshot(chunk, 1) is first
    moved = chunk_of(("A", {}), ("B", {}), entry=lambda k: (quote(k)[0] + 0.5, quote(k)[1] + 0.5))
    fresh = pricing.SHORTVOL.chain_snapshot(moved, 1)
    assert fresh is not first and pricing.SHORTVOL.chain_snapshot(moved, 1) is fresh
    _same(fresh.rows, sv.chain_snapshot(moved, 1).rows)  # the new quotes, not the cached old ones
    assert not np.array_equal(fresh.rows["bid"].to_numpy(), first.rows["bid"].to_numpy())


def test_straddle_puts_only_and_full_snapshots_are_distinct():
    chunk = lad._pack([rec("A", 100.0, pair(100))], {})
    full = pricing.STRADDLE.chain_snapshot(chunk, 0)
    puts = pricing.STRADDLE.chain_snapshot(chunk, 0, puts_only=True)
    assert full is not puts
    assert pricing.STRADDLE.chain_snapshot(chunk, 0, puts_only=True) is puts
    assert pricing.STRADDLE.chain_snapshot(chunk, 0) is full
    assert len(full.rows) == 2 and set(puts.rows["right"]) == {PUT} and len(puts.rows) == 1


# ---- e. the driver's per-month scoring, real pricing on both wirings ------------

def _verdict(event):
    """A month gate that takes the even-numbered events and refuses the odd ones (every id ends in a digit)."""
    return ("taken", None) if int(str(event["event_id"])[-1]) % 2 == 0 else ("not_taken", "below_gate_threshold")


def _month_frames(chunk, table, labels, history, targets):
    """The driver's per-month path, as _outer and _inner_decisions run it on each chunk: Pass A (_month_scores), then
    Pass B (_month). Warm-up January only adds history; June emits. Returns the emitted decision frames."""
    sessions = {str(e): ("2024-06-09", "2024-06-11") for e in list(history["event_id"]) + list(targets["event_id"])}

    def gate(events, chunk_):
        return driver.straddle.decide(events, chunk_, _verdict)

    st, frames = driver._model_state(), []
    for frame, start_m, emit in ((history, JAN, JUN), (targets, JUN, JAN)):
        scored, bad = driver._month_scores(MID, table, frame, chunk, labels, start_m, st)
        frames += driver._month(MID, frame, scored, bad, chunk, start_m, st, emit_from=emit, sessions=sessions, gate=gate)
    return frames


def test_driver_month_scoring_with_the_twin_equals_the_original_wiring(monkeypatch):
    # A full driver.run needs store-shaped sources (ladder chunks from load_index, bound model tables, production and
    # labels). The synthetic fixtures in test_size_pnl_driver*.py stub the pool and the simulation (build_pool returns
    # None), so the real pricer cannot run through them. This compares the per-chunk scoring path that run uses
    # (driver._month_scores and driver._month, size_pnl_driver.py lines 420-461) with real pricing on both sides.
    # n=1500 so January's pool (table rows before 2024-01-01, floor 2023-01-01) clears MIN_POOL; with n=600 it had 198 rows
    # and every warm-up event was dropped as thin_history.
    table, labels = _tables(n=1500)
    labels = labels.assign(available=(table["event_date"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"))
    chunk, events = _ladders(), _events()
    history, targets = events.iloc[:HIST], events.iloc[HIST:]
    real, calls, straddle_calls = price_structure_indexed, [], []

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    def straddle_spy(*args, **kwargs):
        straddle_calls.append(1)
        return real(*args, **kwargs)

    # The originals (sv, sd) are not touched by these spies: the original run below wires driver.shortvol/straddle to them.
    monkeypatch.setattr(pricing.SHORTVOL, "price_structure", spy)
    monkeypatch.setattr(pricing.STRADDLE, "price_structure", straddle_spy)
    wired = _month_frames(chunk, table, labels, history, targets)
    assert calls, "the indexed pricer was never called in the short-vol wired run"
    assert straddle_calls, "the indexed pricer was never called in the straddle wired run"
    monkeypatch.setattr(driver, "shortvol", sv)
    monkeypatch.setattr(driver, "straddle", sd)
    monkeypatch.setattr(driver.equivalence, "shortvol", sv)
    monkeypatch.setattr(driver.gprod, "straddle", sd)
    original = _month_frames(chunk, table, labels, history, targets)
    assert wired and len(wired) == len(original)
    for got, want in zip(wired, original):
        _same(got, want)
    rows = pd.concat(wired, ignore_index=True)
    assert not ((rows["book"] == "short_vol") & (rows["reason"] == "thin_history")).any(), "warm-up January was not simulated"
    short = rows[(rows["book"] == "short_vol") & (rows["status"] == "taken")]
    assert np.isfinite(short["entry_cost"].astype(float).to_numpy()).any()
    assert ((rows["book"] == "short_vol") & (rows["status"] == "not_taken")).any()
    assert ((rows["book"] == "straddle") & (rows["status"] == "taken")).any()
