"""The scheduled full-history yfinance price re-downloader.

``plan_refresh`` is pure (no I/O), so most of this is data-in/data-out. The
network side (``run_refresh``/``fetch_one``) is exercised against a fake
yfinance adapter, same pattern as ``tests/test_fetch.py``'s ``FakeAdapter`` --
no network, no real credential, tmp roots only.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine import paths
from engine.data import fetch as fetch_mod
from engine.data.fetch import Fetcher, cache_key
from engine.data.pulls import price_refresh as pr
from engine.data.sources.base import Response
from engine.data.throttle import SourceConfig, Throttle


def _events(rows: list[tuple[str, str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({"ticker": pd.Series(dtype=str),
                             "event_date": pd.Series(dtype="datetime64[ns]")})
    tickers, dates = zip(*rows)
    return pd.DataFrame({"ticker": list(tickers), "event_date": pd.to_datetime(list(dates))})


class _FakeYFAdapter:
    """Records requested tickers and replays a scripted status per ticker."""

    name = "yfinance"

    def __init__(self, statuses: dict[str, int] | None = None):
        self.statuses = statuses or {}
        self.requested: list[str] = []

    def request(self, endpoint, params, timeout):
        ticker = params.get("ticker")
        self.requested.append(ticker)
        status = self.statuses.get(ticker, 200)
        body = b"Date,Close\n2026-01-02,100.0\n" if status == 200 else b""
        return Response(status, body, {}, f"https://example/{ticker}")

    def quota_from(self, response):
        return None

    def is_auth_failure(self, response):
        return False


def _fetcher(tmp_path, statuses: dict[str, int] | None = None):
    adapter = _FakeYFAdapter(statuses)
    throttle = Throttle(
        {"yfinance": SourceConfig("yfinance", 0.0, 0.0, 2, 10.0)},
        sleep_fn=lambda s: None,
    )
    return Fetcher(tmp_path / "fetch", throttle=throttle, adapters={"yfinance": adapter}), adapter


# --------------------------------------------------------------------------
# window edges
# --------------------------------------------------------------------------


class TestWindowEdges:
    def test_event_on_session_day_is_in_window(self):
        d = pd.Timestamp("2026-09-14")
        assert pr.in_daily_window(d, d) is True

    def test_event_at_exactly_horizon_days_is_in_window(self):
        d = pd.Timestamp("2026-09-14")
        assert pr.in_daily_window(d, d + pd.Timedelta(days=35)) is True

    def test_event_one_day_past_horizon_is_excluded(self):
        d = pd.Timestamp("2026-09-14")
        assert pr.in_daily_window(d, d + pd.Timedelta(days=36)) is False

    def test_fifth_trading_session_after_a_holiday_is_included(self):
        # 2026-11-23 (Mon) -> Thanksgiving 2026-11-26 (Thu) is a scheduled
        # NYSE holiday (engine.calendar.us_market_holidays); the 5th trading
        # session after is 2026-12-01, independently verified below without
        # reusing price_refresh's own trading-day arithmetic.
        event = pd.Timestamp("2026-11-23")
        fifth = _trading_days_after_oracle(event, 5)
        assert fifth == pd.Timestamp("2026-12-01")
        assert pr.in_daily_window(fifth, event) is True

    def test_sixth_trading_session_after_a_holiday_is_excluded(self):
        event = pd.Timestamp("2026-11-23")
        sixth = _trading_days_after_oracle(event, 6)
        assert sixth == pd.Timestamp("2026-12-02")
        assert pr.in_daily_window(sixth, event) is False


def _trading_days_after_oracle(start, n: int) -> pd.Timestamp:
    """Day-by-day walk skipping weekends and NYSE holidays -- an
    implementation independent of ``engine.calendar.projected_trading_days``,
    which is what ``price_refresh`` itself uses, so this is a real
    cross-check rather than a tautology."""
    from engine.calendar import us_market_holidays

    d = pd.Timestamp(start).normalize()
    holidays: set[pd.Timestamp] = set()
    seen_years: set[int] = set()
    count = 0
    while count < n:
        d += pd.Timedelta(days=1)
        if d.year not in seen_years:
            holidays |= us_market_holidays(d.year)
            seen_years.add(d.year)
        if d.weekday() < 5 and d not in holidays:
            count += 1
    return d


# --------------------------------------------------------------------------
# selection rules
# --------------------------------------------------------------------------


class TestSelectionRules:
    def test_board_ticker_outside_its_window_falls_back_to_monthly(self):
        session = "2026-09-14"
        events = _events([("AAA", "2026-11-01")])  # far outside the 35-day horizon
        plan = pr.plan_refresh(session, events=events, price_universe=[])
        assert "AAA" in plan["monthly"]
        assert "AAA" not in plan["daily"]

    def test_non_board_tickers_go_monthly(self):
        session = "2026-09-14"
        plan = pr.plan_refresh(session, events=_events([]), price_universe=["ZZZ"])
        assert plan["monthly"] == ["ZZZ"]
        assert plan["daily"] == []

    def test_board_ticker_in_window_is_daily(self):
        session = "2026-09-14"
        events = _events([("AAA", "2026-09-20")])
        plan = pr.plan_refresh(session, events=events, price_universe=[])
        assert plan["daily"] == ["AAA"]

    def test_a_ticker_already_fetched_this_month_is_skipped(self):
        session = "2026-09-14"
        history = {"ZZZ": [date(2026, 9, 1)]}
        plan = pr.plan_refresh(session, events=_events([]), price_universe=["ZZZ"],
                               fetch_history=history)
        assert plan["skipped_already_fetched"] == ["ZZZ"]
        assert plan["monthly"] == []

    def test_a_daily_ticker_already_fetched_today_is_skipped(self):
        session = "2026-09-14"
        events = _events([("AAA", "2026-09-14")])
        history = {"AAA": [date(2026, 9, 14)]}
        plan = pr.plan_refresh(session, events=events, price_universe=[], fetch_history=history)
        assert plan["skipped_already_fetched"] == ["AAA"]
        assert plan["daily"] == []

    def test_a_daily_ticker_fetched_a_different_day_this_month_is_not_skipped(self):
        """Daily resumability is keyed on the exact session, not the month --
        a fetch from three days ago does not excuse today's print-window pull."""
        session = "2026-09-14"
        events = _events([("AAA", "2026-09-14")])
        history = {"AAA": [date(2026, 9, 11)]}
        plan = pr.plan_refresh(session, events=events, price_universe=[], fetch_history=history)
        assert plan["daily"] == ["AAA"]
        assert plan["skipped_already_fetched"] == []

    def test_a_ticker_only_in_events_not_in_price_universe_is_still_covered(self):
        session = "2026-09-14"
        events = _events([("NEWCO", "2026-09-14")])
        plan = pr.plan_refresh(session, events=events, price_universe=[])
        assert plan["daily"] == ["NEWCO"]

    def test_every_universe_ticker_lands_in_exactly_one_bucket(self):
        session = "2026-09-14"
        events = _events([("AAA", "2026-09-14"), ("BBB", "2027-01-01")])
        history = {"CCC": [date(2026, 9, 1)]}
        plan = pr.plan_refresh(session, events=events, price_universe=["BBB", "CCC", "DDD"],
                               fetch_history=history)
        buckets = plan["daily"] + plan["monthly"] + plan["skipped_already_fetched"]
        assert sorted(buckets) == ["AAA", "BBB", "CCC", "DDD"]
        assert len(buckets) == len(set(buckets))


# --------------------------------------------------------------------------
# Tier-1 keys
# --------------------------------------------------------------------------


class TestKeys:
    def test_two_different_days_produce_two_distinct_keys_and_bodies(self, monkeypatch, tmp_path):
        fetcher, adapter = _fetcher(tmp_path)

        class _Day1(date):
            @classmethod
            def today(cls):
                return date(2026, 9, 14)

        monkeypatch.setattr(fetch_mod, "date", _Day1)
        out1 = pr.fetch_one(fetcher, "AAPL")
        key1 = cache_key("yfinance", "history", {"ticker": "AAPL", "period": "max"}, live=True)

        class _Day2(date):
            @classmethod
            def today(cls):
                return date(2026, 9, 15)

        monkeypatch.setattr(fetch_mod, "date", _Day2)
        out2 = pr.fetch_one(fetcher, "AAPL")
        key2 = cache_key("yfinance", "history", {"ticker": "AAPL", "period": "max"}, live=True)

        assert out1 == {"ticker": "AAPL", "ok": True}
        assert out2 == {"ticker": "AAPL", "ok": True}
        assert key1 != key2
        assert fetcher.body_path("yfinance", key1).exists()
        assert fetcher.body_path("yfinance", key2).exists()
        assert adapter.requested == ["AAPL", "AAPL"]

        undated_key = cache_key("yfinance", "history", {"ticker": "AAPL", "period": "max"})
        assert not fetcher.body_path("yfinance", undated_key).exists()


# --------------------------------------------------------------------------
# failures
# --------------------------------------------------------------------------


class TestFailures:
    def test_a_failing_ticker_is_reported_and_the_run_continues(self, tmp_path):
        fetcher, _ = _fetcher(tmp_path, statuses={"BAD": 404})
        plan = {"session": "2026-09-14", "daily": ["AAA", "BAD", "CCC"],
                "monthly": [], "skipped_already_fetched": []}

        report = pr.run_refresh(plan, fetcher)

        assert sorted(report["fetched"]["daily"]) == ["AAA", "CCC"]
        assert report["failed"] == [
            {"ticker": "BAD", "group": "daily", "error_class": "FetchError"}
        ]
        assert report["counts"] == {
            "daily": 3, "monthly": 0, "skipped_already_fetched": 0,
            "fetched": 2, "failed": 1,
        }

    def test_a_failed_fetch_carries_no_payload(self, tmp_path):
        fetcher, _ = _fetcher(tmp_path, statuses={"BAD": 404})
        plan = {"session": "2026-09-14", "daily": ["BAD"], "monthly": [],
                "skipped_already_fetched": []}

        report = pr.run_refresh(plan, fetcher)

        failure = report["failed"][0]
        assert set(failure) == {"ticker", "group", "error_class"}

    def test_a_failed_ticker_stays_in_the_next_plan(self, tmp_path):
        """404s are not persisted (engine/data/sources/base.py), so a failed
        fetch leaves no Tier-1 meta -- the next plan must see it as never
        fetched, not silently drop it."""
        fetcher, _ = _fetcher(tmp_path, statuses={"BAD": 404})
        plan = {"session": "2026-09-14", "daily": [], "monthly": ["BAD"],
                "skipped_already_fetched": []}
        pr.run_refresh(plan, fetcher)

        history = pr.load_fetch_history(fetch_root=tmp_path / "fetch")
        assert "BAD" not in history

        next_plan = pr.plan_refresh("2026-09-14", events=_events([]), price_universe=["BAD"],
                                    fetch_history=history)
        assert "BAD" in next_plan["monthly"]
        assert "BAD" not in next_plan["skipped_already_fetched"]


# --------------------------------------------------------------------------
# dry-run
# --------------------------------------------------------------------------


class TestDryRun:
    def test_plan_refresh_never_touches_a_fetcher(self):
        """Structural: plan_refresh takes no fetcher/network handle at all, so
        a CLI --dry-run that only calls this makes zero fetch calls by
        construction."""
        import inspect

        params = inspect.signature(pr.plan_refresh).parameters
        assert "fetcher" not in params

    def test_dry_run_cli_command_never_constructs_a_fetcher(self, tmp_path, monkeypatch):
        from engine.v2.ops import cli

        monkeypatch.setattr(pr, "load_events", lambda: _events([("AAA", "2026-09-14")]))
        monkeypatch.setattr(pr, "load_price_universe", lambda: {"ZZZ"})
        monkeypatch.setattr(pr, "load_fetch_history", lambda: {})

        def _boom(*a, **k):
            raise AssertionError("Fetcher must not be constructed on --dry-run")

        monkeypatch.setattr(fetch_mod, "Fetcher", _boom)

        args = cli.parser().parse_args(["price-refresh", "--session", "2026-09-14", "--dry-run"])
        document = cli.price_refresh_command(args, tmp_path / "ops")

        assert document["dry_run"] is True
        assert document["counts"]["daily"] == 1
        assert not (tmp_path / "ops").exists()


# --------------------------------------------------------------------------
# grandfathered guard
# --------------------------------------------------------------------------


class TestGrandfatheredGuard:
    def test_raw_yf_is_refused_as_a_write_target(self):
        with pytest.raises(PermissionError):
            paths.assert_writable(paths.RAW_YF / "px_AAPL.csv")

    def test_fetcher_default_root_is_not_the_grandfathered_px_tree(self):
        assert paths.RAW_FETCH != paths.RAW_YF
        with pytest.raises(ValueError):
            paths.RAW_YF.relative_to(paths.RAW_FETCH)

    def test_load_price_universe_never_writes_to_its_yf_dir(self, tmp_path):
        yf_dir = tmp_path / "yf"
        yf_dir.mkdir()
        (yf_dir / "px_AAPL.csv").write_text("date,close_raw\n2026-01-01,1.0\n")
        before = sorted(yf_dir.iterdir())

        universe = pr.load_price_universe(yf_dir=yf_dir, fetch_root=tmp_path / "fetch")

        assert "AAPL" in universe
        assert sorted(yf_dir.iterdir()) == before
