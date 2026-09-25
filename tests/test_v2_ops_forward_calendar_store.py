"""Spec s4b Change 5: the native forward-calendar store's pure planning.

A fake ``Fetcher``-shaped stub and hand-built ``RefreshUnit`` fixtures -- no
network, no catalog. The session tie-break is compared against the untouched
legacy ``engine.calendar.SESSION_PRIORITY`` module as an oracle.
"""
from __future__ import annotations

import pandas as pd

from engine.calendar import SESSION_PRIORITY
from engine.v2.contracts import SnapshotRef
from engine.v2.ops.forward_calendar_store import (
    SESSION_BY_TIME,
    date_units,
    horizon_dates,
    plan_forward_calendar,
    resolve_session_claims,
)
from engine.v2.ops.incremental_data import classify_response


def _snapshot(snapshot_id="snap-parent"):
    return SnapshotRef(
        snapshot_id=snapshot_id, manifest_hash="sha256:" + "a" * 64,
        parent_snapshot_id=None, table_versions={}, calendar_version="cal-v1",
        source_priority_version="priority-v1", finality_receipt_refs=(),
        knowledge_mode_by_table={})


class FakeFetcher:
    """A ``Fetcher``-shaped counter; no network, no cache."""

    def __init__(self):
        self.calls = 0

    def fetch(self, source, endpoint, params=None, *, live=False, note=""):
        self.calls += 1
        return None


def _cached(unit):
    return classify_response(200, unit.expected_keys, returned_keys=unit.expected_keys,
                             request_id=unit.request_id, receipt_ref="cache:" + unit.request_id,
                             cache_hit=True)


def test_claims_carry_session_priority_same_as_legacy():
    """Two conflicting forward claims: Nasdaq says BMO, yfinance says AMC.
    The winner is what the untouched legacy SESSION_PRIORITY oracle picks."""
    claims = {"nasdaq": "BMO", "yfinance": "AMC"}
    expected_src = next(name for name in SESSION_PRIORITY if claims.get(name))
    assert resolve_session_claims(claims) == ("AMC", expected_src)

    # ORATS, once the event is past, still beats both forward sources.
    assert resolve_session_claims({"orats": "BMO", "yfinance": "AMC",
                                   "nasdaq": "AMC"}) == ("BMO", "orats")
    # Unknown-or-missing sessions fall through in the same order.
    assert resolve_session_claims({"nasdaq": "BMO"}) == ("BMO", "nasdaq")
    assert resolve_session_claims({}) == (None, None)
    assert set(SESSION_BY_TIME.values()) == {"BMO", "AMC"}


def test_horizon_dates_falls_back_to_weekdays_without_a_calendar():
    days = horizon_dates(pd.Timestamp("2026-09-18"), 7)
    assert [str(day.date()) for day in days] == [
        "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24",
        "2026-09-25"]
    fake = type("Calendar", (), {"days": [pd.Timestamp("2026-09-21")]})()
    assert [str(day.date()) for day in horizon_dates("2026-09-18", 7, calendar=fake)] == [
        "2026-09-21"]


def test_provider_calls_go_through_the_shared_budget():
    dates = [pd.Timestamp("2026-09-21"), pd.Timestamp("2026-09-22"),
             pd.Timestamp("2026-09-23")]
    units = date_units(dates)
    nasdaq_cached = {units[0].request_id: _cached(units[0])}

    nasdaq, yfinance = plan_forward_calendar(
        _snapshot(), dates, ("AAPL",), cached_nasdaq=nasdaq_cached, cached_yfinance={})

    assert nasdaq.provider_calls == len(nasdaq.fetch_units) * nasdaq.max_attempts == 2 * 3
    assert nasdaq.provider_account == "nasdaq"
    assert yfinance.provider_calls == len(yfinance.fetch_units) * yfinance.max_attempts == 1 * 3
    assert yfinance.provider_account == "yfinance"


def test_no_provider_call_when_all_cached():
    """Every unit has a satisfying cached outcome: zero reserved calls and
    the fake fetcher is never touched (cache-first, not fetch-then-discard)."""
    dates = [pd.Timestamp("2026-09-21"), pd.Timestamp("2026-09-22"),
             pd.Timestamp("2026-09-23")]
    units = date_units(dates)
    cached = {unit.request_id: _cached(unit) for unit in units}

    nasdaq, _yfinance = plan_forward_calendar(
        _snapshot(), dates, (), cached_nasdaq=cached, cached_yfinance={})
    assert nasdaq.provider_calls == 0
    assert nasdaq.fetch_units == ()

    fake = FakeFetcher()
    for unit in nasdaq.fetch_units:
        fake.fetch("nasdaq", "calendar/earnings", {"date": unit.expected_keys[0]})
    assert fake.calls == 0
