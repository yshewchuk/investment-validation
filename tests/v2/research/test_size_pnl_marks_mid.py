"""EXP-188 Amendment 1 sensitivity (audit row 17): short-vol mid_marks against production's structure_return on the same quotes."""
import pytest

from engine.fills import FillModel
from engine.forecast_sizing import forecast_params
from engine.structures import STRUCTURES, price_structure, structure_return
from experiments import size_pnl_shortvol as sv
from tests.v2.research.test_size_pnl_shortvol import PRED, chunk_of, events_for, taken

NAME = "CND-PS"


def production_pnl(chunk, i):
    """Production's P&L of CND-PS on event ``i``, per share.

    structure_return's ``pnl`` is exit_value - cost, with both priced per share, the same unit as ``pnl_per_share``.
    Entry is priced at FillModel(0.5) on the entry quotes; the exit is priced closing=True at FillModel(0.5), pinned to the
    entry contracts, on a snapshot built from the exit quotes.
    """
    spec = STRUCTURES[NAME](**forecast_params(NAME, PRED))
    entry = price_structure(spec, sv.chain_snapshot(chunk, i), FillModel(0.5))
    exit_chunk = {**chunk, "bid_e": chunk["bid_x"], "ask_e": chunk["ask_x"]}  # chain_snapshot reads bid_e/ask_e: give it the exit quotes
    exit_ = price_structure(spec, sv.chain_snapshot(exit_chunk, i), FillModel(0.5), pin=entry.legs, closing=True)
    return structure_return(entry, exit_)["pnl"]


def test_mid_marks_pnl_equals_production_structure_return_on_a_rung2_event():
    chunk, events = chunk_of(("A", {94: (1.0, 2.0, 2)})), events_for("A")
    mid = sv.score(chunk, events, taken(chunk, "A"), NAME, 0.5, mid_marks=True).iloc[0]
    assert mid["rung"] == "rung2"
    assert mid["pnl_per_share"] == pytest.approx(production_pnl(chunk, 0), abs=1e-9)


def test_primary_differs_from_production_on_a_rung2_event():
    chunk, events = chunk_of(("A", {94: (1.0, 2.0, 2)})), events_for("A")
    primary = sv.score(chunk, events, taken(chunk, "A"), NAME, 0.5).iloc[0]
    assert primary["rung"] == "rung2"
    assert primary["pnl_per_share"] != pytest.approx(production_pnl(chunk, 0), abs=1e-9)


def test_primary_and_mid_equal_production_on_an_all_rung1_event():
    chunk, events = chunk_of(("B", {})), events_for("B")
    decisions, expected = taken(chunk, "B"), production_pnl(chunk, 0)
    for mid_marks in (False, True):
        row = sv.score(chunk, events, decisions, NAME, 0.5, mid_marks=mid_marks).iloc[0]
        assert row["rung"] == "rung1" and row["pnl_per_share"] == pytest.approx(expected, abs=1e-9)
