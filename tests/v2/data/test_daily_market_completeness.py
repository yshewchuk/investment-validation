"""Market-wide daily snapshots must be ordered oldest-first and complete.

Two adjacent failures let an under-filled ORATS market-wide response become the
surviving row for a date: the fetch index was ordered by opaque cache key, and a
full rebuild had no coverage guard before it replaced ``daily_market``. These
tests pin the fetch-time precedence and the pre-writer refusal.
"""
from __future__ import annotations

import gzip
import json
import os

import pandas as pd
import pytest

from engine.data import rebuild, store
from engine.data.fetch import FetchError, iter_cached
from engine.data.normalize import n_daily
from engine.data.validate import validate_daily

DATES = ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
LOW_DATE = "2024-01-03"
TICKERS = [f"T{i:02d}" for i in range(10)]

#: Three names lack the low date, so a full build sees only 7 of 10 — below the
#: 75% floor but above a mere 50% rule.
MISSING_ON_LOW = frozenset(TICKERS[i] for i in (0, 2, 3))


def _write_market_wide(
    root,
    key,
    *,
    fetched_at,
    spot=None,
    trade_date="2024-01-02",
    rows=None,
    endpoint="hist/summaries",
    status=200,
    params=None,
):
    directory = root / "orats" / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [{"ticker": "AAA", "tradeDate": trade_date, "stockPrice": spot}]
    payload = {"data": rows}
    with gzip.open(directory / f"{key}.body.gz", "wb") as fh:
        fh.write(json.dumps(payload).encode())
    (directory / f"{key}.meta.json").write_text(
        json.dumps(
            {
                "key": key,
                "source": "orats",
                "endpoint": endpoint,
                "params": params if params is not None else {"tradeDate": trade_date},
                "status": status,
                "fetched_at": fetched_at,
            }
        )
    )


def _frame(
    ticker,
    missing_date=None,
    missing_tickers=(),
    invalid_date=None,
    invalid_tickers=(),
):
    dates = [
        day
        for day in DATES
        if not (ticker in missing_tickers and day == missing_date)
    ]
    parsed = pd.to_datetime(dates)
    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "date": parsed.astype("datetime64[ns]"),
            "year": parsed.year.astype("int64"),
            "spot": 100.0,
        }
    )
    if ticker in invalid_tickers and invalid_date is not None:
        frame.loc[frame["date"] == pd.Timestamp(invalid_date), "spot"] = -1.0
    return frame


def _install_daily_source(
    monkeypatch,
    *,
    missing_date=None,
    missing_tickers=(),
    invalid_date=None,
    invalid_tickers=(),
):
    def fake_list_tickers(root=None, *, include_fetch=True):
        return list(TICKERS)

    def fake_normalize_ticker(ticker, **_):
        return (
            _frame(
                ticker,
                missing_date,
                missing_tickers,
                invalid_date,
                invalid_tickers,
            ),
            {"ticker": ticker},
        )

    monkeypatch.setattr(n_daily, "list_tickers", fake_list_tickers)
    monkeypatch.setattr(n_daily, "normalize_ticker", fake_normalize_ticker)


def test_latest_fetched_market_wide_snapshot_wins(tmp_path, monkeypatch):
    """The newer fetch wins; its cache key sorts before the older response."""
    from engine import paths

    root = tmp_path / "fetch"
    monkeypatch.setattr(paths, "RAW_FETCH", root)
    newer_key = "aa" + "1" * 62
    older_key = "zz" + "0" * 62
    assert newer_key < older_key
    _write_market_wide(
        root, older_key, fetched_at="2026-09-01T12:00:00+00:00", spot=20.0
    )
    _write_market_wide(
        root, newer_key, fetched_at="2026-09-02T12:00:00+00:00", spot=10.0
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


def test_an_oversized_sidecar_is_refused_before_parsing(tmp_path, monkeypatch):
    from engine import paths

    monkeypatch.setattr(paths, "RAW_FETCH", tmp_path)
    monkeypatch.setattr(paths, "RAW_ORATS_SUMMARIES", tmp_path / "absent")
    directory = tmp_path / "orats" / "aa"
    directory.mkdir(parents=True)
    meta = directory / ("aa" + "0" * 62 + ".meta.json")
    meta.write_text("{not json " + "x" * (70 * 1024))

    n_daily.fetch_daily_index.cache_clear()
    try:
        with pytest.raises(FetchError, match="over the 65536-byte metadata cap"):
            rebuild._daily_tickers(None)
        with pytest.raises(FetchError, match="over the 65536-byte metadata cap"):
            n_daily.list_tickers()
    finally:
        n_daily.fetch_daily_index.cache_clear()

    # The unbounded default keeps the old skip-bad-sidecar behaviour.
    assert iter_cached("orats", root=tmp_path) == []


def test_bounded_reader_refuses_a_symlinked_sidecar(tmp_path):
    directory = tmp_path / "orats" / "aa"
    directory.mkdir(parents=True)
    target = directory / "target.meta.json"
    target.write_text("{}")
    link = directory / ("aa" + "0" * 62 + ".meta.json")
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(FetchError, match="metadata sidecar"):
        iter_cached("orats", root=tmp_path, max_meta_bytes=64 * 1024)


def test_bounded_reader_does_not_hang_on_a_fifo(tmp_path):
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("FIFOs or O_NONBLOCK are unavailable")
    directory = tmp_path / "orats" / "bb"
    directory.mkdir(parents=True)
    fifo = directory / ("bb" + "0" * 62 + ".meta.json")
    try:
        os.mkfifo(fifo)
    except (OSError, NotImplementedError):
        pytest.skip("FIFOs are unavailable")

    with pytest.raises(FetchError, match="not a regular file"):
        iter_cached("orats", root=tmp_path, max_meta_bytes=64 * 1024)


@pytest.mark.parametrize("cap", [True, False, -1, 1.5, "64"])
def test_metadata_cap_argument_must_be_a_nonnegative_int(tmp_path, cap):
    with pytest.raises(ValueError, match="nonnegative integer"):
        iter_cached("orats", root=tmp_path, max_meta_bytes=cap)


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

    _install_daily_source(
        monkeypatch, missing_date=LOW_DATE, missing_tickers=MISSING_ON_LOW
    )

    class _WriterMustNotBeConstructed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("writer constructed before the preflight")

    monkeypatch.setattr(store, "PartitionedWriter", _WriterMustNotBeConstructed)

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.build_daily_table()

    error = excinfo.value
    assert error.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert error.date == LOW_DATE
    assert error.count == 7
    assert error.previous_count == 10
    assert error.next_count == 10
    assert error.threshold == pytest.approx(7.5)
    assert error.details["date"] == LOW_DATE
    assert error.details["count"] == 7
    assert error.details["previous_count"] == 10
    assert error.details["next_count"] == 10
    assert error.details["threshold"] == pytest.approx(7.5)
    assert error.details["reason"] == error.reason
    assert LOW_DATE in error.reason
    assert error.reason == (
        "daily_market has 7 rows on 2024-01-03, below 75% of the median "
        "of its adjacent published dates (10 and 10); threshold 7.5"
    )
    assert str(error) == f"{error.code}: {error.reason}"

    pd.testing.assert_frame_equal(store.read_table("daily_market"), before)
    assert store.table_stats("daily_market").content_hash == before_hash


def test_first_date_refuses_against_its_only_neighbour(tmp_root, monkeypatch):
    first = DATES[0]
    _install_daily_source(
        monkeypatch, missing_date=first, missing_tickers=MISSING_ON_LOW
    )

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.build_daily_table()

    error = excinfo.value
    assert error.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert error.date == first
    assert error.count == 7
    assert error.previous_count is None
    assert error.next_count == 10
    assert error.threshold == pytest.approx(7.5)
    assert error.details["previous_count"] is None
    assert error.details["next_count"] == 10
    assert "only adjacent published date (10)" in error.reason


def test_last_date_refuses_against_its_only_neighbour(tmp_root, monkeypatch):
    last = DATES[-1]
    _install_daily_source(
        monkeypatch, missing_date=last, missing_tickers=MISSING_ON_LOW
    )

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.build_daily_table()

    error = excinfo.value
    assert error.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert error.date == last
    assert error.count == 7
    assert error.previous_count == 10
    assert error.next_count is None
    assert error.threshold == pytest.approx(7.5)
    assert error.details["previous_count"] == 10
    assert error.details["next_count"] is None
    assert "only adjacent published date (10)" in error.reason


def test_invalid_rows_on_low_date_are_excluded_before_the_count(
    tmp_root, monkeypatch
):
    rows = pd.concat(
        [
            _frame(ticker, invalid_date=LOW_DATE, invalid_tickers=MISSING_ON_LOW)
            for ticker in TICKERS
        ],
        ignore_index=True,
    )
    low_rows = rows["date"] == pd.Timestamp(LOW_DATE)
    assert low_rows.sum() == len(TICKERS)
    clean, _ = validate_daily(rows)
    assert (clean["date"] == pd.Timestamp(LOW_DATE)).sum() == 7

    _install_daily_source(
        monkeypatch, invalid_date=LOW_DATE, invalid_tickers=MISSING_ON_LOW
    )

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.build_daily_table()

    error = excinfo.value
    assert error.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert error.date == LOW_DATE
    assert error.count == 7
    assert error.threshold == pytest.approx(7.5)


def test_sample_zero_is_a_full_build_and_refuses_the_gap(tmp_root, monkeypatch):
    _install_daily_source(
        monkeypatch, missing_date=LOW_DATE, missing_tickers=MISSING_ON_LOW
    )

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.build_daily_table(sample=0)

    assert excinfo.value.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert excinfo.value.date == LOW_DATE
    assert excinfo.value.count == 7


def test_the_public_builder_has_no_preflight_switch(tmp_root, monkeypatch):
    """A caller cannot opt out of the guard through the public signature."""
    _install_daily_source(
        monkeypatch, missing_date=LOW_DATE, missing_tickers=MISSING_ON_LOW
    )

    with pytest.raises(TypeError):
        rebuild.build_daily_table(preflight=False)


def test_an_empty_ticker_inventory_refuses_instead_of_writing_zero_rows(
    tmp_root, monkeypatch
):
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
    monkeypatch.setattr(
        n_daily, "list_tickers", lambda root=None, *, include_fetch=True: []
    )

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.build_daily_table()

    error = excinfo.value
    assert error.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert error.date is None
    assert error.count == 0
    assert error.previous_count is None
    assert error.next_count is None
    assert error.threshold == 0.0
    assert error.details["tickers"] == 0
    assert error.details["dates"] == 0
    assert "no source tickers" in error.reason
    pd.testing.assert_frame_equal(store.read_table("daily_market"), before)
    assert store.table_stats("daily_market").content_hash == before_hash


def test_empty_market_wide_snapshot_is_a_zero_count_date(tmp_root, monkeypatch):
    from engine import paths

    root = tmp_root / "fetch"
    monkeypatch.setattr(paths, "RAW_FETCH", root)
    _write_market_wide(
        root,
        "aa" + "0" * 62,
        fetched_at="2026-09-02T12:00:00+00:00",
        trade_date="2024-01-08",
        rows=[],
    )
    _install_daily_source(monkeypatch)

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.build_daily_table()

    error = excinfo.value
    assert error.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert error.date == "2024-01-08"
    assert error.count == 0
    assert error.previous_count == 10
    assert error.next_count is None
    assert error.threshold == pytest.approx(7.5)


def test_market_wide_candidates_need_success_and_no_ticker(tmp_root, monkeypatch):
    from engine import paths

    root = tmp_root / "fetch"
    monkeypatch.setattr(paths, "RAW_FETCH", root)
    _write_market_wide(
        root,
        "aa" + "0" * 62,
        fetched_at="2026-09-02T12:00:00+00:00",
        trade_date="2024-01-08",
    )
    _write_market_wide(
        root,
        "bb" + "1" * 62,
        fetched_at="2026-09-02T12:00:00+00:00",
        trade_date="2024-01-09",
        endpoint="hist/cores",
    )
    _write_market_wide(
        root,
        "cc" + "2" * 62,
        fetched_at="2026-09-02T12:00:00+00:00",
        trade_date="2024-01-10",
        status=500,
    )
    _write_market_wide(
        root,
        "dd" + "3" * 62,
        fetched_at="2026-09-02T12:00:00+00:00",
        trade_date="2024-01-11",
        params={"ticker": "AAA", "tradeDate": "2024-01-11"},
    )
    _write_market_wide(
        root,
        "ee" + "4" * 62,
        fetched_at="2026-09-02T12:00:00+00:00",
        trade_date="2024-01-12",
        endpoint="hist/strikes",
    )

    dates = rebuild._market_wide_snapshot_dates()
    assert [pd.Timestamp(day).date().isoformat() for day in dates] == [
        "2024-01-08",
        "2024-01-09",
    ]


def test_incomplete_daily_preflight_prevents_the_events_builder(
    tmp_root, monkeypatch
):
    from engine import paths

    _install_daily_source(
        monkeypatch, missing_date=LOW_DATE, missing_tickers=MISSING_ON_LOW
    )
    paths.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    paths.SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    paths.MANIFEST.write_text("prior manifest")
    paths.SNAPSHOT_FILE.write_text('{"snapshot": "prior"}')
    called = []
    monkeypatch.setattr(
        rebuild, "build_events_table", lambda *a, **k: called.append("events")
    )

    with pytest.raises(rebuild.DailyMarketCompletenessError) as excinfo:
        rebuild.rebuild(tables=("events", "daily"), finalize=False)

    assert excinfo.value.code == "DAILY_MARKET_DATE_INCOMPLETE"
    assert called == []
    assert paths.MANIFEST.read_text() == "prior manifest"
    assert paths.SNAPSHOT_FILE.read_text() == '{"snapshot": "prior"}'


def test_rebuild_skips_the_guard_when_daily_is_not_selected(
    tmp_root, monkeypatch
):
    _install_daily_source(
        monkeypatch, missing_date=LOW_DATE, missing_tickers=MISSING_ON_LOW
    )

    result = rebuild.rebuild(tables=("securities",), finalize=False)

    assert result.reports["securities"]["rows"] == 0


def test_rebuild_sample_skips_the_guard(tmp_root, monkeypatch):
    _install_daily_source(
        monkeypatch, missing_date=LOW_DATE, missing_tickers=MISSING_ON_LOW
    )

    result = rebuild.rebuild(tables=("daily",), sample=2, finalize=False)

    assert result.reports["daily"]["tickers"] == 2
    assert result.reports["daily"]["rows"] == 7


def test_the_preflight_does_not_quarantine_before_the_write_pass(
    tmp_root, monkeypatch
):
    from engine import paths
    from engine.data import validate

    _install_daily_source(
        monkeypatch, invalid_date=LOW_DATE, invalid_tickers={"T00"}
    )

    calls = []
    real_quarantine = validate.quarantine

    def record_quarantine(source_file, reason, details=None, *, root=None):
        calls.append((source_file, reason, details, root))
        return real_quarantine(source_file, reason, details, root=root)

    monkeypatch.setattr(validate, "quarantine", record_quarantine)

    report = rebuild.build_daily_table()

    # T00's invalid row is below the floor's impact, so the build succeeds; only
    # the write pass calls the quarantine helper, exactly once, with a source
    # path. The counting pass passes no source path and must add no call.
    assert report["rows"] == len(TICKERS) * len(DATES) - 1
    assert len(calls) == 1
    assert calls[0][0] == "orats/summaries/T00.json.gz"
    assert calls[0][3] is None
    assert "daily row validation" in calls[0][1]
    assert len(list(paths.QUARANTINE.glob("*.flag.json"))) == 1


def test_complete_full_build_passes(tmp_root, monkeypatch):
    _install_daily_source(monkeypatch)

    report = rebuild.build_daily_table()

    expected = len(TICKERS) * len(DATES)
    assert report["tickers"] == len(TICKERS)
    assert report["skipped"] == 0
    assert report["rows"] == expected
    table = store.read_table("daily_market")
    assert len(table) == expected
    assert set(table["ticker"]) == set(TICKERS)
    assert (table["spot"] > 0).all()
    assert table["spot"].eq(100.0).all()


def test_sample_build_skips_the_guard(tmp_root, monkeypatch):
    """The same low-date fixture is fine when sampling, because it is partial."""
    _install_daily_source(
        monkeypatch, missing_date=LOW_DATE, missing_tickers=MISSING_ON_LOW
    )

    report = rebuild.build_daily_table(sample=2)

    # T00 lacks the low date; T01 has all four.
    assert report["tickers"] == 2
    assert report["rows"] == 7
    table = store.read_table("daily_market")
    assert len(table) == 7
    assert set(table["ticker"]) == {"T00", "T01"}
    assert (table["spot"] > 0).all()
    assert table["spot"].eq(100.0).all()
