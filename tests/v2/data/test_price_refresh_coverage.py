"""``load_fetch_coverage`` and the coverage-keyed daily skip in ``plan_refresh``.

The 2026-09-16 defect: a daily ticker was skipped because SOME fetch carried a
``fetched_at`` dated today, even when that fetch's body ended at yesterday's
close (a pre-market run). These tests write into a real ``Fetcher`` over a
throwaway root through a fake yfinance adapter that returns a CSV body per
ticker, so the coverage read goes through the genuine gzip sidecar layout; the
fetch's arrival stamp is frozen per write so "stamped today, last bar Friday"
is an actual fixture rather than a comment.
"""
from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from engine.data import fetch as fetch_mod
from engine.data.fetch import Fetcher
from engine.data.pulls import price_refresh as pr
from engine.data.sources.base import Response
from engine.data.throttle import SourceConfig, Throttle

SESSION = "2026-09-14"  # a Monday; the prior trading day is Friday 2026-09-11
PRIOR_SESSION = "2026-09-11"


def _events(rows: list[tuple[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"ticker": pd.Series(dtype=str),
                             "event_date": pd.Series(dtype="datetime64[ns]")})
    tickers, dates = zip(*rows)
    return pd.DataFrame({"ticker": list(tickers), "event_date": pd.to_datetime(list(dates))})


def _csv_bytes(bars: dict[str, float]) -> bytes:
    lines = ["Date,Close"]
    for d, close in sorted(bars.items()):
        lines.append(f"{d},{close}")
    return ("\n".join(lines) + "\n").encode()


class _CsvYFAdapter:
    """Replays one scripted CSV body per ticker (``Date,Close``)."""

    name = "yfinance"

    def __init__(self, bodies: dict[str, bytes]):
        self.bodies = bodies
        self.requested: list[str] = []

    def request(self, endpoint, params, timeout):
        ticker = params.get("ticker")
        self.requested.append(ticker)
        body = self.bodies[ticker]
        return Response(200, body, {}, f"https://example/{ticker}")

    def quota_from(self, response):
        return None

    def is_auth_failure(self, response):
        return False


def _frozen(day: date):
    """``date``/``datetime`` substitutes whose ``today()``/``now()`` are ``day``.

    ``Fetcher._persist`` stamps ``fetched_at`` from ``datetime.now(timezone.utc)``
    and ``cache_key`` takes the day from ``date.today()``, so patching both in
    ``engine.data.fetch`` makes one write look like it arrived at 13:30 UTC on
    ``day`` — pre-market, which is exactly the fetch that must not count.
    """
    moment = datetime(day.year, day.month, day.day, 13, 30, tzinfo=timezone.utc)

    class _Day(date):
        @classmethod
        def today(cls):
            return day

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment if tz is None else moment.astimezone(tz)

    return _Day, _Now


@pytest.fixture
def fetch_root(tmp_path):
    return tmp_path / "fetch"


def _fetch(fetch_root, bodies, tickers, *, on, monkeypatch):
    """One real ``Fetcher`` write per ticker, all stamped on ``on``."""
    adapter = _CsvYFAdapter(bodies)
    fetcher = Fetcher(fetch_root, throttle=Throttle(
        {"yfinance": SourceConfig("yfinance", 0.0, 0.0, 2, 10.0)}, sleep_fn=lambda s: None),
        adapters={"yfinance": adapter})
    day_cls, now_cls = _frozen(on)
    monkeypatch.setattr(fetch_mod, "date", day_cls)
    monkeypatch.setattr(fetch_mod, "datetime", now_cls)
    for ticker in tickers:
        assert pr.fetch_one(fetcher, ticker)["ok"] is True
    return adapter


def _write_raw_fetch(fetch_root, ticker, *, key, fetched_at, body_bytes, compressed=True):
    """A sidecar pair written directly — the shape for an unreadable body."""
    directory = Path(fetch_root) / "yfinance" / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    raw = gzip.compress(body_bytes, mtime=0) if compressed else body_bytes
    (directory / f"{key}.body.gz").write_bytes(raw)
    (directory / f"{key}.meta.json").write_text(json.dumps({
        "source": "yfinance", "endpoint": "history", "key": key,
        "params": {"ticker": ticker, "period": "max"},
        "fetched_at": fetched_at, "status": 200}))


# --------------------------------------------------------------------------
# what a fetch covers is its bars, not its arrival stamp
# --------------------------------------------------------------------------


def test_a_same_day_fetch_ending_at_the_prior_session_is_not_coverage(fetch_root, monkeypatch):
    """The defect itself: bytes arrived Monday, last bar is Friday's close."""
    _fetch(fetch_root, {"AAA": _csv_bytes({PRIOR_SESSION: 100.0})}, ["AAA"],
           on=date(2026, 9, 14), monkeypatch=monkeypatch)

    coverage = pr.load_fetch_coverage(SESSION, fetch_root=fetch_root)
    history = pr.load_fetch_history(fetch_root=fetch_root)
    assert coverage == {"AAA": [date(2026, 9, 11)]}
    assert history == {"AAA": [date(2026, 9, 14)]}, "fetched_at still says today"

    plan = pr.plan_refresh(SESSION, events=_events([("AAA", SESSION)]), price_universe=[],
                           fetch_history=history, fetch_coverage=coverage)
    assert plan["daily"] == ["AAA"]
    assert plan["skipped_already_fetched"] == []


def test_a_fetch_holding_the_session_bar_is_coverage(fetch_root, monkeypatch):
    _fetch(fetch_root, {"AAA": _csv_bytes({PRIOR_SESSION: 100.0, SESSION: 101.0})}, ["AAA"],
           on=date(2026, 9, 14), monkeypatch=monkeypatch)

    coverage = pr.load_fetch_coverage(SESSION, fetch_root=fetch_root)
    assert coverage == {"AAA": [date(2026, 9, 14)]}

    plan = pr.plan_refresh(SESSION, events=_events([("AAA", SESSION)]), price_universe=[],
                           fetch_coverage=coverage)
    assert plan["daily"] == []
    assert plan["skipped_already_fetched"] == ["AAA"]


def test_the_newest_bar_wins_across_a_ticker_s_fetches(fetch_root, monkeypatch):
    """One entry per fetch, each carrying its own body date: a ticker fetched
    Sunday holding Friday's close and again Monday holding Monday's close has
    both dates, and any of them reaching the requirement skips it."""
    _fetch(fetch_root, {"AAA": _csv_bytes({PRIOR_SESSION: 100.0})}, ["AAA"],
           on=date(2026, 9, 13), monkeypatch=monkeypatch)
    _fetch(fetch_root, {"AAA": _csv_bytes({SESSION: 101.0})}, ["AAA"],
           on=date(2026, 9, 14), monkeypatch=monkeypatch)

    coverage = pr.load_fetch_coverage(SESSION, fetch_root=fetch_root)
    assert sorted(coverage["AAA"]) == [date(2026, 9, 11), date(2026, 9, 14)]
    plan = pr.plan_refresh(SESSION, events=_events([("AAA", SESSION)]), price_universe=[],
                           fetch_coverage=coverage)
    assert plan["skipped_already_fetched"] == ["AAA"]


def test_the_body_of_a_fetch_stamp_too_old_to_hold_the_bar_is_never_opened(
        fetch_root, monkeypatch):
    """Cheap prefilter only: ``fetched_at`` older than ``session - 1 day``
    cannot contain the session's bar, so it contributes nothing."""
    _write_raw_fetch(fetch_root, "AAA", key="stalefr", fetched_at="2026-09-11T20:00:00+00:00",
                     body_bytes=_csv_bytes({SESSION: 101.0}))
    assert pr.load_fetch_coverage(SESSION, fetch_root=fetch_root) == {}


# --------------------------------------------------------------------------
# unreadable means not covered
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,body,compressed", [
    ("corruptgz", b"this is not a gzip stream at all", False),
    ("emptybody", b"", True),
    ("headeronly", b"Date,Close\n", True),
    ("nodates", b"Date,Close\nnot-a-date,100.0\n", True),
])
def test_an_unreadable_fetch_contributes_nothing(fetch_root, key, body, compressed):
    _write_raw_fetch(fetch_root, "BAD", key=key, fetched_at=f"{SESSION}T20:00:00+00:00",
                     body_bytes=body, compressed=compressed)

    coverage = pr.load_fetch_coverage(SESSION, fetch_root=fetch_root)
    assert coverage == {}
    plan = pr.plan_refresh(SESSION, events=_events([("BAD", SESSION)]), price_universe=[],
                           fetch_coverage=coverage)
    assert plan["daily"] == ["BAD"]


def test_a_ticker_with_no_entry_at_all_is_daily(fetch_root):
    assert pr.load_fetch_coverage(SESSION, fetch_root=fetch_root) == {}
    plan = pr.plan_refresh(SESSION, events=_events([("NEWCO", SESSION)]), price_universe=[],
                           always_daily=("SPY",))
    assert plan["daily"] == ["NEWCO", "SPY"]


# --------------------------------------------------------------------------
# the partition, and the holiday session
# --------------------------------------------------------------------------


def test_a_mixed_board_partitions_into_exactly_one_bucket_each(fetch_root, monkeypatch):
    bodies = {
        "COVERED": _csv_bytes({SESSION: 101.0}),          # skip: holds today's bar
        "PREMARKET": _csv_bytes({PRIOR_SESSION: 100.0}),  # daily: today's bytes, Friday's bars
        "STALE": _csv_bytes({"2026-09-04": 99.0}),        # daily: old body
    }
    _fetch(fetch_root, bodies, ["COVERED", "PREMARKET", "STALE"],
           on=date(2026, 9, 14), monkeypatch=monkeypatch)
    _write_raw_fetch(fetch_root, "CORRUPT", key="corrupt",
                     fetched_at=f"{SESSION}T20:00:00+00:00", body_bytes=b"nope", compressed=False)
    coverage = pr.load_fetch_coverage(SESSION, fetch_root=fetch_root)
    events = _events([("COVERED", SESSION), ("PREMARKET", SESSION), ("STALE", SESSION),
                      ("CORRUPT", SESSION), ("UNSEEN", SESSION)])

    plan = pr.plan_refresh(SESSION, events=events,
                           price_universe=["MONTHLY_NEW", "MONTHLY_DONE"],
                           fetch_history={"MONTHLY_DONE": [date(2026, 9, 2)]},
                           fetch_coverage=coverage)

    assert plan["daily"] == ["CORRUPT", "PREMARKET", "STALE", "UNSEEN"]
    assert plan["skipped_already_fetched"] == ["COVERED", "MONTHLY_DONE"]
    assert plan["monthly"] == ["MONTHLY_NEW"]
    buckets = plan["daily"] + plan["monthly"] + plan["skipped_already_fetched"]
    assert sorted(buckets) == ["CORRUPT", "COVERED", "MONTHLY_DONE", "MONTHLY_NEW",
                               "PREMARKET", "STALE", "UNSEEN"]
    assert len(buckets) == len(set(buckets))


def test_a_holiday_session_requires_the_last_trading_day_not_the_holiday():
    assert pr._required_daily_bar_date("2026-09-07") == pd.Timestamp("2026-09-04")
    assert pr._required_daily_bar_date(SESSION) == pd.Timestamp(SESSION)


def test_a_holiday_session_is_skipped_by_a_fetch_holding_the_prior_friday(
        fetch_root, monkeypatch):
    """2026-09-07 is Labor Day; the required bar is Friday 2026-09-04, and a
    fetch stamped on the holiday weekend that holds it covers the session."""
    holiday = "2026-09-07"
    _fetch(fetch_root, {"AAA": _csv_bytes({"2026-09-03": 99.0, "2026-09-04": 100.0})}, ["AAA"],
           on=date(2026, 9, 6), monkeypatch=monkeypatch)
    coverage = pr.load_fetch_coverage(holiday, fetch_root=fetch_root)
    assert coverage == {"AAA": [date(2026, 9, 4)]}

    plan = pr.plan_refresh(holiday, events=_events([("AAA", holiday)]),
                           price_universe=["MONTHLY"],
                           fetch_history={"MONTHLY": [date(2026, 9, 1)]},
                           fetch_coverage=coverage)
    assert plan["skipped_already_fetched"] == ["AAA", "MONTHLY"]
    assert plan["daily"] == []
    assert plan["monthly"] == []
