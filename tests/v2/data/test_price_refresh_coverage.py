"""``load_fetch_coverage`` and the coverage-keyed daily skip in ``plan_refresh``.

The 2026-09-16 defect: a daily ticker was skipped because SOME fetch carried a
``fetched_at`` dated today, even when that fetch's body ended at yesterday's
close (a pre-market run). These tests write into a real ``Fetcher`` over a
throwaway root through a fake yfinance adapter that returns a CSV body per
ticker, so the coverage read goes through the genuine gzip sidecar layout; the
fetch's arrival stamp is frozen per write so "stamped today, last bar Friday"
is an actual fixture rather than a comment.

Which rows of a body count as valid bars is capture's rule, not a second one
written for this loader: the tests inject the same wrapper
``legacy_adapter.invoke_price_refresh`` passes in production
(``engine.v2.data.price_download_sources.read_tier1_body``'s ``date`` column),
so every fixture body here is a real 7-column Tier-1 body and a row capture
would drop (null date, null ``Adj Close``) is dropped by the loader too.
"""
from __future__ import annotations

import gzip
import json
import zlib
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from engine.data import fetch as fetch_mod
from engine.data.fetch import Fetcher, cache_key
from engine.data.pulls import price_refresh as pr
from engine.data.sources.base import Response
from engine.data.throttle import SourceConfig, Throttle
from engine.v2.data.errors import DataError
from engine.v2.data.price_download_sources import read_tier1_body

SESSION = "2026-09-14"  # a Monday; the prior trading day is Friday 2026-09-11
PRIOR_SESSION = "2026-09-11"


def _events(rows: list[tuple[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"ticker": pd.Series(dtype=str),
                             "event_date": pd.Series(dtype="datetime64[ns]")})
    tickers, dates = zip(*rows)
    return pd.DataFrame({"ticker": list(tickers), "event_date": pd.to_datetime(list(dates))})


def _read_bars(body: bytes):
    """The production wrapper: capture's reader, its valid bar dates."""
    return read_tier1_body(body)["date"]


def _tier1_bytes(rows: list[tuple[str, float, float | None]], *,
                 with_adj_close: bool = True) -> bytes:
    """A real Tier-1 body: ``rows`` is ``[(date, close, adj_close), ...]``, with
    ``Open``/``High``/``Low``/``Close`` all ``close`` and ``Volume`` fixed.

    An ``adj_close`` of ``None`` writes an EMPTY ``Adj Close`` field (capture's
    reader drops that row); ``with_adj_close=False`` drops the column from the
    header entirely (capture's reader refuses that body as a contract
    mismatch)."""
    columns = ["Date", "Open", "High", "Low", "Close"]
    if with_adj_close:
        columns.append("Adj Close")
    columns.append("Volume")
    lines = [",".join(columns)]
    for d, close, adj in sorted(rows):
        fields = [d, close, close, close, close]
        if with_adj_close:
            fields.append("" if adj is None else adj)
        fields.append(1000)
        lines.append(",".join(str(f) for f in fields))
    return ("\n".join(lines) + "\n").encode()


def _csv_bytes(bars: dict[str, float]) -> bytes:
    """Every bar carrying a real ``Adj Close`` — the ordinary Tier-1 shape."""
    return _tier1_bytes([(d, close, close) for d, close in bars.items()])


class _CsvYFAdapter:
    """Replays one scripted CSV body per ticker (a real 7-column Tier-1 body)."""

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


class _ScriptedYFAdapter:
    """Replays a QUEUE of scripted CSV bodies per ticker.

    Same shape as ``tests/v2/data/test_price_refresh.py``'s ``_FakeYFAdapter``
    (a scripted :class:`Response` per request, recording every request), but
    successive requests for one ticker walk the queue: the pre-market body,
    then the post-close body. An unexpected extra call raises rather than
    silently replaying, so "exactly two network calls" is a real assertion.
    """

    name = "yfinance"

    def __init__(self, queues: dict[str, list[bytes]]):
        self.queues = {t: list(q) for t, q in queues.items()}
        self.requests: list[str] = []

    def request(self, endpoint, params, timeout):
        ticker = params.get("ticker")
        self.requests.append(ticker)
        body = self.queues[ticker].pop(0)
        return Response(200, body, {}, f"https://example/{ticker}")

    def quota_from(self, response):
        return None

    def is_auth_failure(self, response):
        return False


def _throttled_fetcher(fetch_root, adapter):
    """A real ``Fetcher`` over a throwaway root with the fake adapter under it."""
    return Fetcher(fetch_root, throttle=Throttle(
        {"yfinance": SourceConfig("yfinance", 0.0, 0.0, 2, 10.0)}, sleep_fn=lambda s: None),
        adapters={"yfinance": adapter})


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


def _fetch(fetch_root, bodies, tickers, *, on, monkeypatch, refetch_token=None):
    """One real ``Fetcher`` write per ticker, all stamped on ``on``."""
    adapter = _CsvYFAdapter(bodies)
    fetcher = Fetcher(fetch_root, throttle=Throttle(
        {"yfinance": SourceConfig("yfinance", 0.0, 0.0, 2, 10.0)}, sleep_fn=lambda s: None),
        adapters={"yfinance": adapter})
    day_cls, now_cls = _frozen(on)
    monkeypatch.setattr(fetch_mod, "date", day_cls)
    monkeypatch.setattr(fetch_mod, "datetime", now_cls)
    for ticker in tickers:
        assert pr.fetch_one(fetcher, ticker, refetch_token=refetch_token)["ok"] is True
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


def test_the_reader_is_injected_because_legacy_has_no_validity_rule_of_its_own():
    """Coverage validity is capture's rule, not a second one written here:
    ``engine/data`` may not import ``engine.v2`` (``checks/import_layers.py``),
    so ``load_fetch_coverage`` takes the reader as a required parameter and has
    no CSV/row parsing of its own to fall back on. A call without ``read_bars``
    is a TypeError, not a silent reimplementation of the rule."""
    import inspect

    with pytest.raises(TypeError):
        pr.load_fetch_coverage(SESSION)
    assert "read_bars" in inspect.signature(pr.load_fetch_coverage).parameters


# --------------------------------------------------------------------------
# what a fetch covers is its bars, not its arrival stamp
# --------------------------------------------------------------------------


def test_a_same_day_fetch_ending_at_the_prior_session_is_not_coverage(fetch_root, monkeypatch):
    """The defect itself: bytes arrived Monday, last bar is Friday's close."""
    _fetch(fetch_root, {"AAA": _csv_bytes({PRIOR_SESSION: 100.0})}, ["AAA"],
           on=date(2026, 9, 14), monkeypatch=monkeypatch)

    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
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

    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
    assert coverage == {"AAA": [date(2026, 9, 14)]}

    plan = pr.plan_refresh(SESSION, events=_events([("AAA", SESSION)]), price_universe=[],
                           fetch_coverage=coverage)
    assert plan["daily"] == []
    assert plan["skipped_already_fetched"] == ["AAA"]


def test_the_newest_bar_wins_across_a_ticker_s_fetches(fetch_root, monkeypatch):
    """One entry per fetch, each carrying its own body date: a Monday
    pre-market fetch holding Friday's close and a tokenized Monday refetch
    holding Monday's close give the plan both candidate dates, and any of
    them reaching the requirement skips it. (A Sunday-stamped Friday body is
    no longer even read for a full Monday — the prefilter floor is the
    required bar date now, see the weekend tests below.)"""
    _fetch(fetch_root, {"AAA": _csv_bytes({PRIOR_SESSION: 100.0})}, ["AAA"],
           on=date(2026, 9, 14), monkeypatch=monkeypatch)
    _fetch(fetch_root, {"AAA": _csv_bytes({SESSION: 101.0})}, ["AAA"],
           on=date(2026, 9, 14), monkeypatch=monkeypatch, refetch_token="post-close")

    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
    assert sorted(coverage["AAA"]) == [date(2026, 9, 11), date(2026, 9, 14)]
    plan = pr.plan_refresh(SESSION, events=_events([("AAA", SESSION)]), price_universe=[],
                           fetch_coverage=coverage)
    assert plan["skipped_already_fetched"] == ["AAA"]


def test_the_body_of_a_fetch_stamp_too_old_to_hold_the_bar_is_never_opened(
        fetch_root, monkeypatch):
    """Cheap prefilter only: a ``fetched_at`` before the required bar date
    (here: Friday stamped, full Monday session requiring Monday's own bar)
    cannot contain that bar, so the body is skipped and contributes nothing."""
    _write_raw_fetch(fetch_root, "AAA", key="stalefr", fetched_at="2026-09-11T20:00:00+00:00",
                     body_bytes=_csv_bytes({SESSION: 101.0}))
    assert pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root) == {}


# --------------------------------------------------------------------------
# unreadable means not covered
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,body,compressed", [
    ("corruptgz", b"this is not a gzip stream at all", False),
    ("emptybody", b"", True),
    ("headeronly", b"Date,Open,High,Low,Close,Adj Close,Volume\n", True),
    ("nodates", _tier1_bytes([("not-a-date", 100.0, 100.0)]), True),
    # capture's own refusals: a body it will not accept contributes nothing.
    ("noadjcolumn", _tier1_bytes([(PRIOR_SESSION, 100.0, 100.0)], with_adj_close=False), True),
    ("allnanadj", _tier1_bytes([(PRIOR_SESSION, 100.0, None), (SESSION, 101.0, None)]), True),
])
def test_an_unreadable_fetch_contributes_nothing(fetch_root, key, body, compressed):
    """Unknown is never coverage, and capture's own refusals are one of its
    shapes: an undecodable body, an empty one, one with no usable dates, and
    the two bodies ``read_tier1_body`` refuses typed (no ``Adj Close`` column;
    an ``Adj Close`` that parses entirely NaN) all contribute nothing."""
    _write_raw_fetch(fetch_root, "BAD", key=key, fetched_at=f"{SESSION}T20:00:00+00:00",
                     body_bytes=body, compressed=compressed)

    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
    assert coverage == {}
    plan = pr.plan_refresh(SESSION, events=_events([("BAD", SESSION)]), price_universe=[],
                           fetch_coverage=coverage)
    assert plan["daily"] == ["BAD"]


def test_a_body_missing_the_adj_close_column_is_not_coverage_and_does_not_raise(fetch_root):
    """The loader swallows capture's typed refusal rather than re-deriving its
    own rule: ``read_tier1_body`` raises ``CONTRACT_MISMATCH`` on this body —
    proved directly above — and the coverage read turns that into "this fetch
    covers nothing", which only ever lengthens the daily list."""
    body = _tier1_bytes([(PRIOR_SESSION, 100.0, 100.0), (SESSION, 101.0, 101.0)],
                        with_adj_close=False)
    with pytest.raises(DataError) as refusal:
        _read_bars(body)
    assert refusal.value.code == "CONTRACT_MISMATCH"

    _write_raw_fetch(fetch_root, "BAD", key="no-adj-column",
                     fetched_at=f"{SESSION}T20:00:00+00:00", body_bytes=body)
    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
    assert coverage == {}
    plan = pr.plan_refresh(SESSION, events=_events([("BAD", SESSION)]), price_universe=[],
                           fetch_coverage=coverage)
    assert plan["daily"] == ["BAD"]


def test_a_latest_row_with_an_empty_adj_close_is_not_the_session_bar(fetch_root):
    """Capture's rule decides coverage, per row: the session bar is physically
    the body's last row, but with no ``Adj Close`` it is a row capture would
    never store, so that fetch reaches only the prior session and the ticker
    stays planned daily — while the identical body whose last row does carry an
    ``Adj Close`` covers the session and skips."""
    _write_raw_fetch(fetch_root, "NOADJ", key="latest-row-no-adj",
                     fetched_at=f"{SESSION}T20:00:00+00:00",
                     body_bytes=_tier1_bytes([(PRIOR_SESSION, 100.0, 100.0),
                                              (SESSION, 101.0, None)]))
    _write_raw_fetch(fetch_root, "WITHADJ", key="latest-row-with-adj",
                     fetched_at=f"{SESSION}T20:00:00+00:00",
                     body_bytes=_tier1_bytes([(PRIOR_SESSION, 100.0, 100.0),
                                              (SESSION, 101.0, 101.0)]))

    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
    assert coverage == {"NOADJ": [date(2026, 9, 11)], "WITHADJ": [date(2026, 9, 14)]}

    plan = pr.plan_refresh(SESSION,
                           events=_events([("NOADJ", SESSION), ("WITHADJ", SESSION)]),
                           price_universe=[], fetch_coverage=coverage)
    assert plan["daily"] == ["NOADJ"]
    assert plan["skipped_already_fetched"] == ["WITHADJ"]


def test_a_ticker_with_no_entry_at_all_is_daily(fetch_root):
    assert pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root) == {}
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
    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
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
    coverage = pr.load_fetch_coverage(holiday, read_bars=_read_bars, fetch_root=fetch_root)
    assert coverage == {"AAA": [date(2026, 9, 4)]}

    plan = pr.plan_refresh(holiday, events=_events([("AAA", holiday)]),
                           price_universe=["MONTHLY"],
                           fetch_history={"MONTHLY": [date(2026, 9, 1)]},
                           fetch_coverage=coverage)
    assert plan["skipped_already_fetched"] == ["AAA", "MONTHLY"]
    assert plan["daily"] == []
    assert plan["monthly"] == []


# --------------------------------------------------------------------------
# the prefilter floor is the required BAR date, not session - 1 day
# --------------------------------------------------------------------------


def test_a_friday_stamped_fetch_is_read_for_the_weekend_and_holiday_sessions(fetch_root):
    """Sat 09-05, Sun 09-06 and the Labor-Day Monday 09-07 all require
    Friday 2026-09-04's close, so a fetch stamped that very Friday is read
    for every one of them — its age relative to the session is not what the
    floor measures."""
    _write_raw_fetch(fetch_root, "AAA", key="friday-stamp",
                     fetched_at="2026-09-04T20:00:00+00:00",
                     body_bytes=_csv_bytes({"2026-09-03": 99.0, "2026-09-04": 100.0}))
    for session in ("2026-09-05", "2026-09-06", "2026-09-07"):
        coverage = pr.load_fetch_coverage(session, read_bars=_read_bars, fetch_root=fetch_root)
        assert coverage == {"AAA": [date(2026, 9, 4)]}, session
        plan = pr.plan_refresh(session, events=_events([]), price_universe=[],
                               always_daily=("AAA",), fetch_coverage=coverage)
        assert plan["skipped_already_fetched"] == ["AAA"], session
        assert plan["daily"] == [], session


def test_a_fetch_stamped_before_the_required_day_is_never_read(fetch_root):
    """Thursday 09-03's bytes cannot hold the Friday bar the 09-05..09-07
    sessions require. The body even *contains* 09-04 (physically impossible,
    the point of the fixture): were it opened it would count as coverage, so
    an empty result with the reader never reached proves the prefilter skipped
    it before opening. The reader is rigged to explode on a call, which only
    fails the test when it is actually reached — a loader-wide swallow of
    capture's refusals (``test_an_unreadable_fetch_contributes_nothing``) is
    precisely why the spy records the call itself."""
    _write_raw_fetch(fetch_root, "ZZZ", key="thursday-stamp",
                     fetched_at="2026-09-03T21:00:00+00:00",
                     body_bytes=_csv_bytes({"2026-09-04": 100.0}))

    reached = []

    def _no_body_reads(body: bytes):
        reached.append(body)
        raise AssertionError("a fetch before the required bar date must not open its body")

    coverage = pr.load_fetch_coverage("2026-09-07", read_bars=_no_body_reads,
                                      fetch_root=fetch_root)
    assert reached == []
    assert coverage == {}
    plan = pr.plan_refresh("2026-09-07", events=_events([]), price_universe=[],
                           always_daily=("ZZZ",), fetch_coverage=coverage)
    assert plan["daily"] == ["ZZZ"]


# --------------------------------------------------------------------------
# a corrupt gzip body is a contributing zero, not a crash
# --------------------------------------------------------------------------


def test_a_corrupt_gzip_body_raising_zlib_error_contributes_nothing(fetch_root):
    """Valid gzip header, garbage deflate stream: ``entry.body()`` dies with
    ``zlib.error`` inside the read, and the ticker simply goes on today's
    daily list instead of taking the coverage load down with it."""
    corrupt = gzip.compress(_csv_bytes({SESSION: 101.0}))[:10] + b"\xff\xff\xff\xff no deflate"
    with pytest.raises(zlib.error):
        gzip.decompress(corrupt)
    _write_raw_fetch(fetch_root, "BAD", key="zlib-error",
                     fetched_at=f"{SESSION}T20:00:00+00:00",
                     body_bytes=corrupt, compressed=False)

    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
    assert coverage == {}
    plan = pr.plan_refresh(SESSION, events=_events([("BAD", SESSION)]), price_universe=[],
                           fetch_coverage=coverage)
    assert plan["daily"] == ["BAD"]


# --------------------------------------------------------------------------
# bar-date parsing edges in plan_refresh
# --------------------------------------------------------------------------


def test_tz_aware_bar_dates_parse_to_their_trade_date(fetch_root):
    """yfinance writes per-row UTC offsets that flip at DST; ``2026-10-09
    00:00:00-04:00`` is 04:00Z and must still name Friday 2026-10-09 — the
    required bar of the Sunday 10-11 session its Friday stamp is read for.
    ``read_tier1_body`` does the UTC parse, the loader's normalize is what
    drops the intraday part, so the pairing is what this pins."""
    _write_raw_fetch(fetch_root, "AAA", key="tz-aware-bar",
                     fetched_at="2026-10-09T20:00:00+00:00",
                     body_bytes=_tier1_bytes([("2026-10-09 00:00:00-04:00", 100.0, 100.0)]))
    coverage = pr.load_fetch_coverage("2026-10-11", read_bars=_read_bars, fetch_root=fetch_root)
    assert coverage == {"AAA": [date(2026, 10, 9)]}


def test_coverage_newer_than_the_required_bar_date_still_skips():
    """>=, not ==: a body whose newest bar is PAST the required date still
    covers the session. This test fails if ``>= required`` ever becomes
    ``== required``."""
    plan = pr.plan_refresh("2026-10-12", events=_events([("AAA", "2026-10-12")]),
                           price_universe=[], fetch_coverage={"AAA": [date(2026, 10, 13)]})
    assert plan["skipped_already_fetched"] == ["AAA"]
    assert plan["daily"] == []


# --------------------------------------------------------------------------
# refetch_token: a known-stale cached answer must not answer the call
# --------------------------------------------------------------------------


def test_cache_key_refetch_token_is_distinct_and_optional():
    params = {"ticker": "AAPL", "period": "max"}
    live_base = cache_key("yfinance", "history", params, live=True)
    assert live_base == cache_key("yfinance", "history", params, live=True)
    assert cache_key("yfinance", "history", params, live=True, refetch_token=None) == live_base

    tok1 = cache_key("yfinance", "history", params, live=True, refetch_token="nonce-1")
    tok2 = cache_key("yfinance", "history", params, live=True, refetch_token="nonce-2")
    assert tok1 != tok2
    assert len({live_base, tok1, tok2}) == 3

    plain = cache_key("yfinance", "history", params)
    assert plain == cache_key("yfinance", "history", params)
    assert cache_key("yfinance", "history", params, refetch_token="nonce-1") != plain


def test_fetcher_fetch_with_token_does_not_read_the_cached_entry(fetch_root):
    body1 = _csv_bytes({PRIOR_SESSION: 100.0})
    body2 = _csv_bytes({SESSION: 101.0})
    adapter = _ScriptedYFAdapter({"AAA": [body1, body2]})
    fetcher = _throttled_fetcher(fetch_root, adapter)
    params = {"ticker": "AAA", "period": "max"}

    first = fetcher.fetch("yfinance", "history", params, live=True)
    assert first.from_cache is False
    calls = fetcher.network_calls

    plain_hit = fetcher.fetch("yfinance", "history", params, live=True)
    assert plain_hit.from_cache is True
    assert plain_hit.body == body1
    assert fetcher.network_calls == calls, "the untokenized retry stays a cache hit"

    refetched = fetcher.fetch("yfinance", "history", params, live=True, refetch_token="nonce-1")
    assert refetched.from_cache is False, "a token must not read the entry it made stale"
    assert refetched.body == body2
    assert refetched.key != first.key
    assert fetcher.network_calls == calls + 1

    reuse = fetcher.fetch("yfinance", "history", params, live=True, refetch_token="nonce-1")
    assert reuse.from_cache is True, "the same token re-reads the entry it created"


def test_daily_refresh_refetches_past_todays_cached_premarket_body(fetch_root, monkeypatch):
    """Execution-level walk of the 2026-10-12 defect's fix: a pre-market body
    stamped today caches the plain ``live=True`` key, planning still demands
    the session bar, and ``run_refresh``'s daily token makes the second
    network call genuinely happen — adapter, not cache — leaving BOTH bodies
    on disk (append-only) so the next plan skips the ticker."""
    premarket = _csv_bytes({PRIOR_SESSION: 100.0})
    with_session_bar = _csv_bytes({PRIOR_SESSION: 100.0, SESSION: 101.0})
    adapter = _ScriptedYFAdapter({"AAA": [premarket, with_session_bar]})
    fetcher = _throttled_fetcher(fetch_root, adapter)
    day_cls, now_cls = _frozen(date(2026, 9, 14))
    monkeypatch.setattr(fetch_mod, "date", day_cls)
    monkeypatch.setattr(fetch_mod, "datetime", now_cls)

    assert pr.fetch_one(fetcher, "AAA")["ok"] is True
    assert adapter.requests == ["AAA"]
    assert fetcher.network_calls == 1

    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
    history = pr.load_fetch_history(fetch_root=fetch_root)
    plan = pr.plan_refresh(SESSION, events=_events([("AAA", SESSION)]), price_universe=[],
                           fetch_history=history, fetch_coverage=coverage)
    assert plan["daily"] == ["AAA"], "stamped today, holds only Friday's bar"
    assert plan["skipped_already_fetched"] == []

    report = pr.run_refresh(plan, fetcher)
    assert report["failed"] == []
    assert report["fetched"]["daily"] == ["AAA"]
    assert adapter.requests == ["AAA", "AAA"], "the daily refetch reached the adapter again"
    assert fetcher.network_calls == 2

    coverage = pr.load_fetch_coverage(SESSION, read_bars=_read_bars, fetch_root=fetch_root)
    assert max(coverage["AAA"]) >= date(2026, 9, 14)
    next_plan = pr.plan_refresh(SESSION, events=_events([("AAA", SESSION)]),
                                price_universe=[], fetch_coverage=coverage)
    assert next_plan["daily"] == []
    assert next_plan["skipped_already_fetched"] == ["AAA"]

    entries = fetch_mod.iter_cached("yfinance", "history", root=fetch_root)
    assert len(entries) == 2 and len({e.key for e in entries}) == 2, (
        "append-only: the stale pre-market body is kept beside the fresh one")
