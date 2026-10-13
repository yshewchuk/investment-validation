"""``engine/calendar.py`` ``market_close`` / ``early_close_days``.

The session-end rules the price-refresh coverage buffer reads: 16:00
America/New_York normally, 13:00 on the three NYSE early-close patterns
(Black Friday; Dec 24 as a weekday trading day; Jul 3 only when Jul 4 falls
Tue-Fri), always returned UTC-aware so they compare against a UTC
``fetched_at`` stamp.
"""
from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from engine.calendar import early_close_days, market_close, us_market_holidays

TS = pd.Timestamp


def test_a_normal_day_closes_at_1600_et_in_utc():
    # 2026-09-14 is EDT (UTC-4): 16:00 -> 20:00Z.
    assert market_close("2026-09-14") == TS("2026-09-14T20:00:00+00:00")
    # 2026-01-05 is EST (UTC-5): 16:00 -> 21:00Z. The zone name carries DST.
    assert market_close("2026-01-05") == TS("2026-01-05T21:00:00+00:00")
    assert market_close(pd.Timestamp("2026-09-14")).tzinfo == timezone.utc


@pytest.mark.parametrize("day,close_utc", [
    ("2026-11-27", "18:00:00"),  # Black Friday, EST
    ("2026-12-24", "18:00:00"),  # Dec 24 weekday, EST
    ("2025-07-03", "17:00:00"),  # Jul 3 with Jul 4 on a Friday, EDT
])
def test_an_early_close_day_closes_at_1300_et(day, close_utc):
    assert TS(day) in early_close_days(TS(day).year)
    assert market_close(day) == TS(f"{day}T{close_utc}+00:00")


def test_the_three_early_close_rules_by_year():
    # 2026: Jul 4 is a Saturday -> the holiday is OBSERVED on Fri Jul 3
    # (market closed, no early close), so only Black Friday and Dec 24.
    assert early_close_days(2026) == {TS("2026-11-27"), TS("2026-12-24")}
    assert TS("2026-07-03") in us_market_holidays(2026)
    # 2025: Jul 4 is a Friday -> Jul 3 early-closes, plus Black Friday, Dec 24.
    assert early_close_days(2025) == {TS("2025-07-03"), TS("2025-11-28"), TS("2025-12-24")}
    # 2021: Jul 4 a Sunday (observed Mon Jul 5) and Dec 25 a Saturday
    # (observed Fri Dec 24) knock out both candidate days; only Black Friday.
    assert early_close_days(2021) == {TS("2021-11-26")}
    # 2016: Jul 4 a Monday leaves Jul 3 (a Sunday) a non-trading weekend day,
    # and Dec 24 fell on a Saturday — Black Friday again alone.
    assert early_close_days(2016) == {TS("2016-11-25")}


@pytest.mark.parametrize("year", range(2006, 2031))
def test_every_early_close_day_is_a_weekday_non_holiday_among_the_three_rules(year):
    holidays = us_market_holidays(year)
    thanksgiving = next(d for d in holidays if d.month == 11 and d.weekday() == 3)
    allowed = {thanksgiving + pd.Timedelta(days=1),  # Black Friday
               TS(year=year, month=12, day=24), TS(year=year, month=7, day=3)}
    days = early_close_days(year)
    assert days, "Black Friday is a trading day every year"
    assert days <= allowed
    for d in days:
        assert d.weekday() < 5, f"{d.date()} is a weekend"
        assert d not in holidays, f"{d.date()} is a market holiday"
