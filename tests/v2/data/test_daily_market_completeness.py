"""Market-wide daily snapshots must be latest-first and complete.

Two adjacent failures let an under-filled ORATS market-wide response become the
surviving row for a date: the fetch index was ordered by opaque cache key, and a
full rebuild had no coverage guard before it replaced ``daily_market``. These
tests pin the fetch-time precedence and the pre-writer refusal.
"""
from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from engine.data import rebuild, store
from engine.data.normalize import n_daily

DATES = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
LOW_DATE = "2024-01-03"
TICKERS = [f"T{i:02d}" for i in range(10)]

#: Five names lack the low date, so a full build sees only half the market.
MISSING_ON_LOW = frozenset(TICKERS[i] for i in (0, 2, 3, 4, 5))


def _write_market_wide(root, key, *, fetched_at, spot):
    directory = root / "orats" / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": [{"ticker": "AAA", "tradeDate": "2024-01-02", "stockPrice": spot}]
    }
    with gzip.open(directory / f"{key}.body.gz", "wb") as fh:
        fh.write(json.dumps(payload).encode())
    (directory / f"{key}.meta.json").write_text(
        json.dumps(
            {
                "key": key,
                "source": "orats",
                "endpoint": "hist/summaries",
                "params": {"tradeDate": "2024-01-02"},
                "status": 200,
                "fetched_at": fetched_at,
            }
        )
    )


def _frame(ticker, missing_on_low=()):
    dates = [day for day in DATES if not (ticker in missing_on_low and day == LOW_DATE)]
    parsed = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": parsed.astype("datetime64[ns]"),
            "year": parsed.year.astype("int64"),
            "spot": 100.0,
        }
    )


def _install_daily_source(monkeypatch, *, missing_on_low=()):
    def fake_list_tickers(root=None, *, include_fetch=True):
        return list(TICKERS)

    def fake_normalize_ticker(ticker, **_):
        return _frame(ticker, missing_on_low), {"ticker": ticker}

    monkeypatch.setattr(n_daily, "list_tickers", fake_list_tickers)
    monkeypatch.setattr(n_daily, "normalize_ticker", fake_normalize_ticker)


def test_latest_fetched_market_wide_snapshot_wins(tmp_path, monkeypatch):
    """The newer fetch wins even when its cache key sorts before the older one."""
    from engine import paths

    root = tmp_path / "fetch"
    monkeypatch.setattr(paths, "RAW_FETCH", root)
    _write_market_wide(
        root, "aa" + "0" * 62, fetched_at="2026-09-02T12:00:00+00:00", spot=10.0
    )
    _write_market_wide(
        root, "zz" + "1" * 62, fetched_at="2026-09-01T12:00:00+00:00", spot=20.0
    )

    n_daily.fetch_daily_index.cache_clear()
    try:
        frame, _ = n_daily.normalize_ticker(
            "AAA", summaries_dir=tmp_path / "absent", cores_dir=tmp_path / "absent"
        )
    finally:
        n_daily.fetch_daily_index.cache_clear()

    assert len(frame) == 1
    assert frame["spot"].iloc[0] == pytest.approx(10.0)


def test_low_date_refuses_before_writer_and_preserves_prior_table(tmp_root, monkeypatch):
    prior = pd.DataFrame(
        {
            "ticker": ["OLD"],
            "date": pd.to_datetime(["2024-01-02"]).astype("datetime64[ns]"),
            "year": [2024],
            "spot": [42.0],
        }
    )
    store.write_table(prior, "daily_market")
    before = store.read_table("daily_market")
    before_hash = store.table_stats("daily_market").content_hash

    _install_daily_source(monkeypatch, missing_on_low=MISSING_ON_LOW)

    class _WriterMustNotBeConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("writer constructed before the preflight")

    monkeypatch.setattr(store, "PartitionedWriter", _WriterMustNotBeConstructed)

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.build_daily_table()

    error = excinfo.value
    assert error.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert error.date == LOW_DATE
    assert error.count == 5
    assert error.previous_count == 10
    assert error.next_count == 10
    assert error.threshold == pytest.approx(7.5)
    assert LOW_DATE in error.reason

    pd.testing.assert_frame_equal(store.read_table("daily_market"), before)
    assert store.table_stats("daily_market").content_hash == before_hash


def test_complete_full_build_passes(tmp_root, monkeypatch):
    _install_daily_source(monkeypatch)

    report = rebuild.build_daily_table()

    expected = len(TICKERS) * len(DATES)
    assert report["tickers"] == len(TICKERS)
    assert report["skipped"] == 0
    assert report["rows"] == expected
    assert len(store.read_table("daily_market")) == expected


def test_sample_build_skips_the_guard(tmp_root, monkeypatch):
    """The same low-date fixture is fine when sampling, because it is partial."""
    _install_daily_source(monkeypatch, missing_on_low=MISSING_ON_LOW)

    report = rebuild.build_daily_table(sample=2)

    # T00 lacks the low date; T01 has all four.
    assert report["tickers"] == 2
    assert report["rows"] == 7
    assert len(store.read_table("daily_market")) == 7
