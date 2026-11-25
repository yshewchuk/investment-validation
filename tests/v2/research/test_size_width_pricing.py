"""EXP-189 pricing seam: the chain-index copy of size_pnl_shortvol must agree with the real module, case by case. Synthetic chunks only."""
import numpy as np
import pandas as pd
import pytest

from engine.chain_index import price_structure_indexed
from engine.fills import FillModel
from engine.forecast_sizing import forecast_params
from engine.structures import STRUCTURES, price_structure
from experiments import size_pnl_ladder as lad
from experiments import size_pnl_shortvol as sv
from experiments import size_width_pricing as seam
from tests.v2.research.test_size_pnl_shortvol import events_for, rec

FORECASTS = (1.0, 3.0, 5.0, 12.0, 40.0, float("nan"))
ALPHAS = (0.0, 0.5, 1.0)
EVENT_IDS = [f"E{i}" for i in range(8)]
SIMS_GRID = np.arange(60.0, 141.0)
SIMS = {e: (SIMS_GRID, 0.001 * (SIMS_GRID - 100.0) ** 2) for e in EVENT_IDS}
SIMS["E1"] = None  # an unsimulable event


def _legs(rng, spot, strikes, *, calls=False, crossed=(), duplicate=False):
    """Put rows across ``strikes`` (call rows too when ``calls``; a strike in ``crossed`` has bid above ask; a duplicate strike row when set)."""
    legs = []
    for k in strikes:
        mid = 1.0 + 0.01 * (k - spot) ** 2 + float(rng.uniform(0.0, 0.05))
        bid, ask = (mid + 0.1, mid - 0.1) if k in crossed else (mid - 0.1, mid + 0.1)
        exit_row = (mid - 0.1, mid + 0.1, 1 + int(rng.integers(0, 2)))
        legs.append((0, float(k), bid, ask, *exit_row))
        if calls:
            legs.append((1, float(k), bid, ask, *exit_row))
    if duplicate:
        legs.append((0, float(list(strikes)[len(strikes) // 2]), 5.0, 5.2, 5.0, 5.2, 1))
    return legs


def _chunk():
    """Eight events: dense ladders at several spots, a coarse ladder, unsorted strikes, a duplicate strike row, a crossed entry
    quote on the 94 strike, and mixed calls and puts."""
    rng = np.random.default_rng(11)
    dense, coarse = range(80, 121), range(60, 141, 10)
    shuffled = list(rng.permutation(np.arange(80.0, 121.0)))
    specs = [("E0", 100.0, dict(strikes=dense)), ("E1", 95.0, dict(strikes=dense)),
             ("E2", 103.7, dict(strikes=dense, calls=True)), ("E3", 100.0, dict(strikes=coarse)),
             ("E4", 100.0, dict(strikes=shuffled)), ("E5", 100.0, dict(strikes=dense, duplicate=True)),
             ("E6", 100.0, dict(strikes=dense, crossed=(94.0,))), ("E7", 95.0, dict(strikes=coarse, calls=True))]
    return lad._pack([rec(eid, spot, _legs(rng, spot, **kw)) for eid, spot, kw in specs], {})


@pytest.fixture(scope="module")
def chunk():
    return _chunk()


def test_the_real_module_is_not_mutated_and_the_copy_is_its_own_instance():
    assert sv.price_structure is price_structure
    assert seam.PRICING is not sv
    assert seam.PRICING.price_structure is price_structure_indexed
    assert seam.PRICING.Refusal is sv.Refusal


def test_chain_snapshot_is_reused_per_event_of_a_chunk_and_fresh_for_a_new_chunk(chunk):
    first = seam.PRICING.chain_snapshot(chunk, 0)
    assert seam.PRICING.chain_snapshot(chunk, 0) is first
    other = _chunk()
    fresh = seam.PRICING.chain_snapshot(other, 0)
    assert fresh is not first and seam.PRICING.chain_snapshot(other, 0) is fresh


def test_resolve_priced_equals_the_real_module_on_every_case(chunk):
    reasons, priced_count = set(), 0
    for i in range(len(EVENT_IDS)):
        for name in sv.NAMES:
            for pred in FORECASTS:
                for alpha in ALPHAS:
                    fill = FillModel(alpha)
                    want = sv.resolve_priced(chunk, i, name, pred, fill)
                    assert seam.PRICING.resolve_priced(chunk, i, name, pred, fill) == want
                    reasons.add(want[2])
                    if want[2] is None:
                        priced_count += 1
                        spec = STRUCTURES[name](**forecast_params(name, pred))
                        assert want[0] == price_structure(spec, sv.chain_snapshot(chunk, i), fill)
    assert {None, "ladder_too_coarse"} <= reasons
    assert reasons - {None, "ladder_too_coarse"}, reasons
    assert priced_count > 0


@pytest.mark.parametrize("name", sv.NAMES)
def test_score_events_score_and_secured_equal_the_real_module(chunk, name):
    events = events_for(*EVENT_IDS, pred=5.0)
    for alpha in ALPHAS:
        want = sv.score_events(chunk, events, SIMS, name, alpha=alpha)
        pd.testing.assert_frame_equal(seam.PRICING.score_events(chunk, events, SIMS, name, alpha=alpha), want, check_exact=True)
        taken_rows = want[want["status"] == "scored"].assign(status="taken")
        if taken_rows.empty:
            continue
        pd.testing.assert_frame_equal(seam.PRICING.score(chunk, events, taken_rows, name, alpha),
                                      sv.score(chunk, events, taken_rows, name, alpha), check_exact=True)
        pd.testing.assert_frame_equal(seam.PRICING.secured(chunk, events, taken_rows, name),
                                      sv.secured(chunk, events, taken_rows, name), check_exact=True)
