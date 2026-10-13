"""Close-buffered daily coverage: ``load_fetch_times`` and the stamp-keyed skip.

A daily ticker's cached fetch covers the session only if its ``fetched_at``
(UTC-aware) is AFTER the close of the required trading day — the latest
trading day on or before the session, the regular 16:00 America/New_York
close for every trading day (``price_refresh._regular_market_close``) — plus
``COVERAGE_CLOSE_BUFFER`` (30 minutes). On the ~3 NYSE half-days a year a
fetch made before 16:30 ET may be refetched (accepted
harmless extra refetch). A pre-market fetch lands on the session's date but
predates its close (the 2026-10-12 defect), so it never excuses the session.
Bodies are never parsed on this path: capture is responsible for failing
loudly. A bad post-close response is not retried until a run targets a session
whose close (plus the buffer) is later than that fetch; that run refetches it.
Store-level fixtures write
into a real ``Fetcher`` over a throwaway root through a fake yfinance adapter,
stamping arrival by freezing the clock or writing the sidecar meta directly.
"""
from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from engine.data import fetch as fetch_mod
from engine.data.fetch import Fetcher, cache_key
from engine.data.pulls import price_refresh as pr
from engine.data.sources.base import Response
from engine.data.throttle import SourceConfig, Throttle

SESSION = "2026-09-14"  # a Monday; EDT close 20:00Z, cutoff close+30m = 20:30Z
PREMARKET_STAMP = f"{SESSION}T13:30:00+00:00"
POST_CLOSE_STAMP = f"{SESSION}T21:00:00+00:00"


def _events(rows: list[tuple[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"ticker": pd.Series(dtype=str),
                             "event_date": pd.Series(dtype="datetime64[ns]")})
    tickers, dates = zip(*rows)
    return pd.DataFrame({"ticker": list(tickers), "event_date": pd.to_datetime(list(dates))})


def _plan(times: dict) -> dict:
    return pr.plan_refresh(SESSION, events=_events([("AAA", SESSION)]),
                           price_universe=[], fetch_times=times)


class _QueueAdapter:
    """Replays a QUEUE of scripted bodies per ticker; an unexpected extra
    call raises, so "the refetch really hit the network" is a real assertion."""

    name = "yfinance"

    def __init__(self, queues: dict[str, list[bytes]]):
        self.queues = {t: list(q) for t, q in queues.items()}
        self.requests: list[str] = []

    def request(self, endpoint, params, timeout):
        ticker = params.get("ticker")
        self.requests.append(ticker)
        return Response(200, self.queues[ticker].pop(0), {}, f"https://example/{ticker}")

    def quota_from(self, response):
        return None

    def is_auth_failure(self, response):
        return False


def _fetcher(fetch_root, adapter):
    return Fetcher(fetch_root, throttle=Throttle(
        {"yfinance": SourceConfig("yfinance", 0.0, 0.0, 2, 10.0)}, sleep_fn=lambda s: None),
        adapters={"yfinance": adapter})


def _freeze(monkeypatch, moment: datetime):
    """Pin ``Fetcher._persist``'s ``fetched_at`` and ``cache_key``'s day to
    ``moment`` (UTC), so one write looks like it arrived at exactly that time."""

    class _Day(date):
        @classmethod
        def today(cls):
            return moment.date()

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment if tz is None else moment.astimezone(tz)

    monkeypatch.setattr(fetch_mod, "date", _Day)
    monkeypatch.setattr(fetch_mod, "datetime", _Now)


def _write_raw_fetch(fetch_root, ticker, *, key, fetched_at, period="max"):
    """A sidecar pair written directly — the shape for stamps the loader must
    see but never a body it must read (coverage never opens bodies)."""
    directory = Path(fetch_root) / "yfinance" / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{key}.body.gz").write_bytes(gzip.compress(b"Date,Close\n2026-01-02,1\n"))
    meta = {"source": "yfinance", "endpoint": "history", "key": key,
            "params": {"ticker": ticker, "period": period}, "status": 200}
    if fetched_at is not None:
        meta["fetched_at"] = fetched_at
    (directory / f"{key}.meta.json").write_text(json.dumps(meta))


@pytest.fixture
def fetch_root(tmp_path):
    return tmp_path / "fetch"


# --------------------------------------------------------------------------
# plan_refresh: the stamp's side of close+buffer decides the skip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stamp,expect_daily", [
    (PREMARKET_STAMP, True),                        # session day, pre-market
    (f"{SESSION}T19:59:00+00:00", True),            # just before the close
    (f"{SESSION}T20:29:00+00:00", True),            # just before close+buffer
    (f"{SESSION}T20:31:00+00:00", False),           # just after close+buffer
    (POST_CLOSE_STAMP, False),
])
def test_a_stamp_at_or_before_close_plus_buffer_never_excuses_the_session(
        stamp, expect_daily):
    plan = _plan({"AAA": [pd.Timestamp(stamp)]})
    assert ("AAA" in plan["daily"]) is expect_daily
    assert ("AAA" in plan["skipped_already_fetched"]) is not expect_daily


def test_coverage_is_strictly_after_close_plus_buffer():
    """The comparison is a strict ``>``: a fetch stamped EXACTLY at the
    cutoff (16:00 ET close + 30 min = 20:30Z on this EDT session) is not
    coverage and refetches; one microsecond past it excuses the session."""
    at_cutoff = _plan({"AAA": [pd.Timestamp(f"{SESSION}T20:30:00+00:00")]})
    assert at_cutoff["daily"] == ["AAA"]
    assert at_cutoff["skipped_already_fetched"] == []
    one_microsecond_later = _plan(
        {"AAA": [pd.Timestamp(f"{SESSION}T20:30:00.000001+00:00")]})
    assert one_microsecond_later["daily"] == []
    assert one_microsecond_later["skipped_already_fetched"] == ["AAA"]


@pytest.mark.parametrize("value", ["garbage", None, pd.Timestamp("2026-09-14 21:00")])
def test_a_stamp_that_is_not_a_tz_aware_instant_is_ignored_never_raises(value):
    """Only a tz-aware ``pd.Timestamp``/``datetime`` has a side of the
    cutoff: a garbage string, ``None``, or a naive timestamp is no time at
    all, so it is ignored rather than raising, and a ticker whose only
    stamp is such a value stays on the daily list — unknown is never
    coverage."""
    plan = _plan({"AAA": [value]})
    assert plan["daily"] == ["AAA"]
    assert plan["skipped_already_fetched"] == []


def test_a_half_day_fetch_before_the_1630_et_cutoff_is_refetched():
    """2026-11-27 is Black Friday: a 13:45 ET fetch predates the plain 16:30 ET
    cutoff, so it is refetched — the accepted harmless extra refetch."""
    plan = pr.plan_refresh("2026-11-27", events=_events([("AAA", "2026-11-27")]),
                           price_universe=[],
                           fetch_times={"AAA": [pd.Timestamp("2026-11-27T18:45:00+00:00")]})
    assert plan["daily"] == ["AAA"]


def test_the_coverage_cutoff_follows_the_new_york_zone_in_winter():
    """The zone, not a fixed UTC hour: 2026-12-14 is a Monday in EST, so the
    regular 16:00 ET close is 21:00Z and the cutoff 21:30Z. A 20:45Z stamp
    (15:45 ET, before the close) refetches even though it would clear the
    20:30Z EDT cutoff of a summer session; 21:45Z (16:45 ET) excuses the
    session."""
    winter_events = _events([("AAA", "2026-12-14")])
    pre_close = pr.plan_refresh("2026-12-14", events=winter_events,
                                price_universe=[],
                                fetch_times={"AAA": [pd.Timestamp("2026-12-14T20:45:00+00:00")]})
    assert pre_close["daily"] == ["AAA"]
    assert pre_close["skipped_already_fetched"] == []
    post_close = pr.plan_refresh("2026-12-14", events=winter_events,
                                 price_universe=[],
                                 fetch_times={"AAA": [pd.Timestamp("2026-12-14T21:45:00+00:00")]})
    assert post_close["skipped_already_fetched"] == ["AAA"]
    assert post_close["daily"] == []


def test_a_monday_holiday_session_is_covered_by_friday_post_close():
    """Session 2026-09-07 is Labor Day; the required trading day is Friday
    2026-09-04, whose 20:00Z close + buffer a 21:00Z Friday stamp clears."""
    times = {"AAA": [pd.Timestamp("2026-09-04T21:00:00+00:00")]}
    plan = pr.plan_refresh("2026-09-07", events=_events([("AAA", "2026-09-04")]),
                           price_universe=[], fetch_times=times)
    assert plan["skipped_already_fetched"] == ["AAA"]
    # The same stamp does NOT cover Tuesday: its required day is Monday.
    plan = pr.plan_refresh("2026-09-08", events=_events([("AAA", "2026-09-04")]),
                           price_universe=[], fetch_times=times)
    assert plan["daily"] == ["AAA"]


def test_coverage_is_daily_only_monthly_keeps_the_month_key():
    """An off-board ticker still resumes monthly on FETCHED-AT DATES: a stamp
    any day of the session's month skips it; the daily path never reads it."""
    times = {"ZZZ": [pd.Timestamp("2026-09-01T13:30:00+00:00")]}
    plan = pr.plan_refresh(SESSION, events=_events([]), price_universe=["ZZZ"],
                           fetch_history={"ZZZ": [date(2026, 9, 1)]}, fetch_times=times)
    assert plan["skipped_already_fetched"] == ["ZZZ"]


# --------------------------------------------------------------------------
# load_fetch_times — stamps only, never bodies
# --------------------------------------------------------------------------


def test_load_fetch_times_keeps_utc_aware_stamps_and_filters_like_history(fetch_root):
    _write_raw_fetch(fetch_root, "AAA", key="k-aware",
                     fetched_at="2026-09-14T13:30:00+00:00")
    _write_raw_fetch(fetch_root, "AAA", key="k-naive", fetched_at="2026-09-14T21:00:00")
    _write_raw_fetch(fetch_root, "AAA", key="k-other-period", period="1mo",
                     fetched_at="2026-09-14T22:00:00+00:00")
    _write_raw_fetch(fetch_root, "BBB", key="k-missing-stamp", fetched_at=None)
    _write_raw_fetch(fetch_root, "CCC", key="k-unparseable", fetched_at="not-a-stamp")

    times = pr.load_fetch_times(fetch_root=fetch_root)

    assert times == {"AAA": [pd.Timestamp("2026-09-14T13:30:00+00:00"),
                             pd.Timestamp("2026-09-14T21:00:00+00:00")]}
    assert all(t.tzinfo is not None for t in times["AAA"])
    assert "BBB" not in times and "CCC" not in times, \
        "a missing or unparseable stamp contributes nothing"


def test_a_ticker_whose_only_stamps_are_unreadable_is_daily(fetch_root):
    """Unknown is never coverage: it can only lengthen the daily list."""
    _write_raw_fetch(fetch_root, "AAA", key="k-missing", fetched_at=None)
    _write_raw_fetch(fetch_root, "AAA", key="k-garbage", fetched_at="yesterday")
    plan = _plan(pr.load_fetch_times(fetch_root=fetch_root))
    assert plan["daily"] == ["AAA"]


def test_a_post_close_stamp_covers_even_when_the_body_is_garbage(fetch_root):
    """The accepted trade-off, pinned: coverage never opens the body, so a
    truncated response stamped after the close counts as covered — capture
    is responsible for failing loudly. A bad post-close response is not
    retried until a run targets a session whose close (plus the buffer) is
    later than that fetch; that run refetches it."""
    directory = Path(fetch_root) / "yfinance" / "gb"
    directory.mkdir(parents=True)
    (directory / "gb.body.gz").write_bytes(b"this is not even gzip")
    (directory / "gb.meta.json").write_text(json.dumps({
        "source": "yfinance", "endpoint": "history", "key": "gb",
        "params": {"ticker": "AAA", "period": "max"},
        "fetched_at": POST_CLOSE_STAMP, "status": 200}))
    plan = _plan(pr.load_fetch_times(fetch_root=fetch_root))
    assert plan["skipped_already_fetched"] == ["AAA"]


# --------------------------------------------------------------------------
# production entry point: one unreadable stamp never aborts the plan
# --------------------------------------------------------------------------


def test_load_fetch_history_returns_the_good_date_and_skips_the_bad_stamp(fetch_root):
    """One garbage ``fetched_at`` for a ticker must not abort monthly
    planning: ``load_fetch_history`` keeps the ticker's valid date and does
    not raise."""
    _write_raw_fetch(fetch_root, "AAA", key="k-good", fetched_at=POST_CLOSE_STAMP)
    _write_raw_fetch(fetch_root, "AAA", key="k-garbage", fetched_at="garbage")
    assert pr.load_fetch_history(fetch_root=fetch_root) == {"AAA": [date(2026, 9, 14)]}


def test_invoke_price_refresh_planned_path_survives_an_unparseable_stamp(tmp_path):
    """``load_fetch_history`` runs BEFORE ``load_fetch_times`` on the real
    path, so a garbage ``fetched_at`` there once aborted planning outright.
    Through the production entry point the bad-stamp ticker must instead
    contribute nothing (monthly/daily refetch), while the valid post-close
    ticker is skipped."""
    from engine.v2.ops.legacy_adapter import invoke_price_refresh

    source_root = tmp_path / "source"
    fetch_root = source_root / "data" / "raw" / "fetch"
    _write_raw_fetch(fetch_root, "BADSTAMP", key="k-garbage-stamp", fetched_at="garbage")
    _write_raw_fetch(fetch_root, "GOODSTAMP", key="k-post-close", fetched_at=POST_CLOSE_STAMP)

    result = invoke_price_refresh(SESSION, dry_run=True,
                                  planned_tickers=("BADSTAMP", "GOODSTAMP"),
                                  source_root=source_root)

    plan = result["plan"]
    assert "BADSTAMP" in plan["daily"], "unknown is never coverage: it refetches"
    assert "GOODSTAMP" in plan["skipped_already_fetched"]
    assert "BADSTAMP" not in plan["skipped_already_fetched"]
    assert plan["monthly"] == []
    assert result["report"] is None, "dry run fetches nothing"


@pytest.mark.parametrize("fetched_at", ["NaT", ""])
def test_invoke_price_refresh_planned_path_plans_daily_on_a_nat_or_empty_stamp(
        tmp_path, fetched_at):
    """``pd.Timestamp("NaT")`` parses WITHOUT raising and hands back NaT, and
    NaT is no date and no side of any cutoff — an empty stamp likewise
    carries no time. Both are unparseable: the entry contributes nothing to
    either loader (neither may crash on it), and through the production
    entry point the ticker is planned daily rather than raising."""
    from engine.v2.ops.legacy_adapter import invoke_price_refresh

    source_root = tmp_path / "source"
    fetch_root = source_root / "data" / "raw" / "fetch"
    _write_raw_fetch(fetch_root, "NATSTAMP",
                     key=f"k-blank-stamp-{fetched_at or 'empty'}",
                     fetched_at=fetched_at)
    _write_raw_fetch(fetch_root, "GOODSTAMP", key="k-post-close",
                     fetched_at=POST_CLOSE_STAMP)

    result = invoke_price_refresh(SESSION, dry_run=True,
                                  planned_tickers=("NATSTAMP", "GOODSTAMP"),
                                  source_root=source_root)

    plan = result["plan"]
    assert "NATSTAMP" in plan["daily"], f"{fetched_at!r} is unparseable: refetch"
    assert "NATSTAMP" not in plan["skipped_already_fetched"]
    assert "GOODSTAMP" in plan["skipped_already_fetched"]
    assert plan["monthly"] == []


# --------------------------------------------------------------------------
# execution level: real Fetcher, fake adapter, the refetch token end to end
# --------------------------------------------------------------------------


def test_run_refresh_refetches_past_a_cached_premarket_body(fetch_root, monkeypatch):
    """A pre-market fetch stamps today but predates today's close, so the
    plan keeps the ticker daily; ``run_refresh``'s token turns that plan into
    a SECOND real adapter call (the stale body must not be served out of the
    same-day cache), and the post-close stamp then excuses the next run."""
    adapter = _QueueAdapter({"AAA": [b"Date,Close\n2026-09-11,100\n",
                                     b"Date,Close\n2026-09-11,100\n2026-09-14,101\n"]})
    fetcher = _fetcher(fetch_root, adapter)

    _freeze(monkeypatch, datetime(2026, 9, 14, 13, 30, tzinfo=timezone.utc))
    assert pr.fetch_one(fetcher, "AAA")["ok"]
    first_times = pr.load_fetch_times(fetch_root=fetch_root)
    plan = _plan(first_times)
    assert plan["daily"] == ["AAA"], "the pre-market stamp is not coverage"

    _freeze(monkeypatch, datetime(2026, 9, 14, 20, 45, tzinfo=timezone.utc))
    report = pr.run_refresh(plan, fetcher)
    assert report["fetched"]["daily"] == ["AAA"]
    assert adapter.requests == ["AAA", "AAA"], "the token forced a real network call"

    times = pr.load_fetch_times(fetch_root=fetch_root)
    assert len(times["AAA"]) == 2, "append-only: the stale entry is kept beside the fresh"
    assert _plan(times)["skipped_already_fetched"] == ["AAA"]


def test_cache_key_refetch_token_is_distinct_and_optional():
    params = {"ticker": "AAA", "period": "max"}
    plain = cache_key("yfinance", "history", params, live=True)
    assert cache_key("yfinance", "history", params, live=True, refetch_token=None) == plain
    keys = {plain,
            cache_key("yfinance", "history", params, live=True, refetch_token="t1"),
            cache_key("yfinance", "history", params, live=True, refetch_token="t2")}
    assert len(keys) == 3

    tokened = cache_key("yfinance", "history", params, live=True, refetch_token="t1")
    assert cache_key("yfinance", "history", params,
                     live=True, refetch_token="t1") == tokened, \
        "same source/endpoint/params/token must give the identical key"
    other_params = {"ticker": "BBB", "period": "max"}
    assert cache_key("yfinance", "history", other_params,
                     live=True, refetch_token="t1") != tokened, \
        "a different request with the same token is a different key"


def test_a_tokenless_repeat_is_a_cache_hit_while_a_token_misses(fetch_root, monkeypatch):
    _freeze(monkeypatch, datetime(2026, 9, 14, 13, 30, tzinfo=timezone.utc))
    adapter = _QueueAdapter({"AAA": [b"Date,Close\n2026-09-11,100\n",
                                     b"Date,Close\n2026-09-14,101\n"]})
    fetcher = _fetcher(fetch_root, adapter)

    pr.fetch_one(fetcher, "AAA")
    pr.fetch_one(fetcher, "AAA")
    assert adapter.requests == ["AAA"], "the plain live=True entry is a same-day cache hit"

    pr.fetch_one(fetcher, "AAA", refetch_token="post-close")
    assert adapter.requests == ["AAA", "AAA"], "a distinct token is a guaranteed miss"
    rec = fetcher.fetch("yfinance", "history", {"ticker": "AAA", "period": "max"},
                        live=True, note="price-refresh")
    assert b"2026-09-11,100" in rec.body and rec.from_cache, \
        "the untokenized fetch still serves the stale body, kept untouched"
