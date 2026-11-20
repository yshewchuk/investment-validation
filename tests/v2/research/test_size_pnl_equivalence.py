"""EXP-188 S9c: the fast-path equivalence preflight over a real ladder chunk (synthetic only)."""
import numpy as np
import pandas as pd
import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_pnl_equivalence as eq
from experiments import size_pnl_ladder as lad
from experiments import size_pnl_pool as pool_mod
from experiments import size_pnl_shortvol as sv
from experiments.size_pnl_pool import build_pool

DRAW = 200
PRED = 6.0
EIDS = ("E1", "E2", "E3", "E4")


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


@pytest.fixture
def case(pool):
    chunk = chunk_of(*((e, {}) for e in EIDS))
    events = events_for(*EIDS)
    return chunk, events, sv.simulate(chunk, events, pool, draws=DRAW)


def _legs(item):
    return [(leg["strike"], leg["side"], leg["qty"]) for leg in item["legs"]]


def test_sample_is_order_independent(case):
    chunk, events, sims = case
    shuffled = events.sample(frac=1.0, random_state=3)
    first = eq.build_sample(chunk, events, sims, alpha=0.5)
    second = eq.build_sample(chunk, shuffled, sims, alpha=0.5)
    assert len(first) >= 2
    assert [i["key"] for i in first] == [i["key"] for i in second]
    assert [_legs(i) for i in first] == [_legs(i) for i in second]
    assert [i["entry_cost"] for i in first] == [i["entry_cost"] for i in second]


def test_sample_spans_more_than_one_structure(case):
    chunk, events, sims = case
    sample = eq.build_sample(chunk, events, sims, alpha=0.5)
    assert len(sample) >= 2
    assert len({tuple(_legs(i)) for i in sample}) > 1


def test_sample_respects_size(case):
    chunk, events, sims = case
    full = eq.build_sample(chunk, events, sims, alpha=0.5)
    assert len(eq.build_sample(chunk, events, sims, alpha=0.5, size=1)) == 1
    assert len(eq.build_sample(chunk, events, sims, alpha=0.5, size=2)) == min(2, len(full))


def test_check_chunk_agrees_with_expected_pnl(case, pool):
    chunk, events, sims = case
    out = eq.check_chunk(chunk, events, sims, pool, alpha=0.5)
    assert out["checked"] >= 1
    assert out["max_abs_diff"] <= 1e-9


def test_check_chunk_is_none_for_an_empty_sample(case, pool):
    chunk, events, _sims = case
    assert eq.check_chunk(chunk, events, {}, pool, alpha=0.5) is None


def test_corrupted_fast_path_refuses(case, pool, monkeypatch):
    chunk, events, sims = case
    real = pool_mod.put_means

    def scaled(*args, **kwargs):
        means = real(*args, **kwargs)
        return None if means is None else means * 1.001

    monkeypatch.setattr(pool_mod, "put_means", scaled)
    with pytest.raises(OpsError) as caught:
        eq.check_chunk(chunk, events, sims, pool, alpha=0.5)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.details["reason"] == "fast_path_mismatch"


def test_planted_difference_is_reported_as_max_abs_diff(case, pool, monkeypatch):
    """A 3e-10 offset on the fast path's structure_pnl is under the 1e-9 tolerance: it passes and is reported exactly."""
    chunk, events, sims = case
    real = pool_mod.structure_pnl

    def shifted(*args, **kwargs):
        value = real(*args, **kwargs)
        return None if value is None else value + 3e-10

    monkeypatch.setattr(pool_mod, "structure_pnl", shifted)
    out = eq.check_chunk(chunk, events, sims, pool, alpha=0.5)
    assert out["max_abs_diff"] == pytest.approx(3e-10, abs=1e-12)
    assert out["checked"] == len(eq.build_sample(chunk, events, sims, alpha=0.5))
