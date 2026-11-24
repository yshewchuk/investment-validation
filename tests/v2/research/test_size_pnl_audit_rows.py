"""EXP-188 Amendment 6 audit rows 11 and 12: the ledger book sizes each trade by ``lots``, and print windows are calendar anchors."""
import pandas as pd
import pytest

from engine.calendar import AMC, BMO, TradingCalendar
from engine.portfolio import build_book
from engine.structures import STRUCTURES
from experiments import size_pnl_shortvol as sv
from experiments.size_pnl_straddle import lots
from experiments.size_pnl_tables import _windows

COSTS = [0.5, 1.37, 2.0, 25.0, 0.0]


def _prediction(i, cost):
    """A canonical-shaped ledger prediction: one trade per ticker, gate passed, the entry cost in intended_prices."""
    return {"row_id": f"R{i}", "as_of": "2025-01-02", "event_date": f"2025-02-{10 + i:02d}", "ticker": f"T{i}",
            "strategy": "short_vol", "score": {"gate_pass": True},
            "intended_prices": {"entry_cost": cost, "quote_date": "2025-01-02"}}


# ---- row 11: the book's contracts are lots(entry cost) -----------------------

def test_book_contracts_are_the_lots_of_each_entry_cost():
    book = build_book(predictions=[_prediction(i, cost) for i, cost in enumerate(COSTS)], outcomes=[])
    got = dict(zip(book["ticker"], book["contracts"]))
    assert len(book) == len(COSTS)
    for i, cost in enumerate(COSTS):
        assert got[f"T{i}"] == lots(cost)
    assert got["T4"] == lots(0) == 1  # a zero cost takes one lot


# ---- row 12: print windows are the calendar's anchors --------------------------

# Weekdays only (no holidays), so the trading day before a Monday is the Friday.
CAL = TradingCalendar(pd.bdate_range("2025-01-01", "2025-03-31"))


@pytest.mark.parametrize("sess, day, entry, exit_", [
    (AMC, "2025-01-08", "2025-01-08", "2025-01-09"),  # Wednesday AMC: entry on the event day, exit the next day
    (BMO, "2025-01-15", "2025-01-14", "2025-01-15"),  # Wednesday BMO: entry the day before, exit on the event day
    (AMC, "2025-01-13", "2025-01-13", "2025-01-14"),  # Monday AMC
    (BMO, "2025-01-20", "2025-01-17", "2025-01-20"),  # Monday BMO: entry is the prior Friday
])
def test_print_windows_are_the_calendar_anchors(sess, day, entry, exit_):
    got = _windows(CAL, pd.Timestamp(day), sess)
    assert got == (CAL.last_pre_print(day, sess), CAL.first_post_print(day, sess), None)
    assert got[0] == pd.Timestamp(entry) and got[1] == pd.Timestamp(exit_)


@pytest.mark.parametrize("name", sv.NAMES)
def test_every_short_vol_structure_enters_on_the_event_day_and_exits_the_next(name):
    spec = STRUCTURES[name]()
    assert spec.entry_offset == 0 and spec.exit_offset == 1
