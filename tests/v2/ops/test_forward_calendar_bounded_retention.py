"""Bounded retained rows for the forward-calendar pinned reads (issue #362).

Synthetic pinned snapshots only. These tests keep the real repository path,
real shared-reader leases and real row accounting, while proving that
``daily_sessions`` and ``_existing_index`` retain at most their configured scan
guard even when the pinned source tables are larger than that guard.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from engine.v2.contracts import TableContract
from engine.v2.data.computed_moves import native_trading_calendar
from engine.v2.data.legacy_mapping import build_legacy_mapping
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, from_document
from engine.v2.ops import forward_calendar_store, pinned_partition_reader
from tests.ops_support import catalog
from tests.test_v2_ops_forward_calendar_store import _seeded_parent

_DAILY_DATES = (
    datetime(2026, 1, 2),
    datetime(2026, 3, 16),
    datetime(2026, 6, 15),
    datetime(2026, 7, 6),
    datetime(2026, 11, 18),
    datetime(2026, 12, 31),
)
_DAILY_SESSIONS = tuple(pd.Timestamp(day).normalize() for day in _DAILY_DATES)
_HORIZON_END = pd.Timestamp("2027-02-01")

_PRESENT_EVENT_KEY = ("AAA", "2025-01-15")
_EVENT_KEYS = {
    _PRESENT_EVENT_KEY,
    ("E003", "2025-01-04"),
    ("E999", "2025-02-02"),
}
_PRESENT_EVENT_KEYS = {
    _PRESENT_EVENT_KEY,
    ("E003", "2025-01-04"),
}

_BASE_EVENT_ROW = {
    "event_id": "event-template",
    "ticker": "ZZZ",
    "event_date": datetime(2025, 1, 1),
    "year": 2025,
    "session": "AMC",
    "session_src": None,
    "annc_tod": None,
    "src_orats": True,
    "src_oquants": False,
    "src_nasdaq": False,
    "src_yfinance": False,
    "date_agree": True,
    "date_conflict": False,
    "updated_at": None,
    "event_cluster_id": None,
    "claim_count": None,
    "reconciliation": None,
}


def _scope(tmp_path, name: str):
    root = tmp_path / name
    root.mkdir()
    conn, clock, _supervisor = catalog(root)
    return conn, clock, ArtifactStore(root / "objects")


def _capture_accounts(monkeypatch):
    real_counter = pinned_partition_reader.RetainedRowCount
    accounts: list[pinned_partition_reader.RetainedRowCount] = []

    def _fresh_counter():
        account = real_counter()
        accounts.append(account)
        return account

    monkeypatch.setattr(pinned_partition_reader, "RetainedRowCount", _fresh_counter)
    return accounts


def _daily_rows(ticker_count: int) -> tuple[dict, ...]:
    daily_contract = from_document(
        TableContract, build_legacy_mapping()["tables"]["daily_market"])
    defaults = {column.name: None for column in daily_contract.columns if column.nullable}
    rows = []
    for index in range(ticker_count):
        ticker = f"T{index:03d}"
        for day_index, day in enumerate(_DAILY_DATES):
            implied_move = (
                None
                if day_index % 3 == 0
                else round(1.0 + 0.25 * day_index + 0.1 * index, 2)
            )
            rows.append({
                **defaults,
                "ticker": ticker,
                "date": day,
                "year": 2026,
                "implied_move": implied_move,
            })
    return tuple(rows)


def _extra_event_rows(count: int) -> tuple[dict, ...]:
    rows = []
    for index in range(count):
        row = dict(_BASE_EVENT_ROW)
        row.update(
            event_id=f"extra-{index:04d}",
            ticker=f"E{index:03d}",
            event_date=datetime(2025, 1, 1 + index % 28),
        )
        rows.append(row)
    return tuple(rows)


def _event_key(row) -> tuple[str, str]:
    return (str(row["ticker"]), str(row["event_date"])[:10])


def _earnings_columns(snapshot):
    table_name = forward_calendar_store.TABLE_NAME
    contract = next(item for item in snapshot.contracts if item.table_name == table_name)
    return tuple(column.name for column in contract.columns)


def _all_event_rows(repository, snapshot) -> dict[tuple[str, str], dict]:
    table_name = forward_calendar_store.TABLE_NAME
    columns = _earnings_columns(snapshot)
    rows = {}
    for lease in forward_calendar_store._scan_rows(
            repository, snapshot.snapshot, table_name, columns):
        with lease as batch:
            for row in batch:
                rows[_event_key(row)] = dict(row)
    return rows


def test_daily_sessions_retention_bounds_rows_without_changing_calendar(tmp_path, monkeypatch):
    """A session-only daily scan bounds retained rows independently of tickers."""
    accounts = _capture_accounts(monkeypatch)
    monkeypatch.setattr(forward_calendar_store, "MAX_SCAN_ROWS", 4)

    frames = {}
    peaks = []
    for ticker_count in (3, 12):
        conn, clock, store = _scope(tmp_path, f"daily-sessions-{ticker_count}")
        parent = _seeded_parent(
            conn,
            store,
            clock,
            scope=f"fwd-cal-daily-sessions-{ticker_count}",
            daily_market_rows=_daily_rows(ticker_count),
        )
        repository = Repository(conn, store)

        before = len(accounts)
        sessions = forward_calendar_store.daily_sessions(repository, parent.snapshot)
        [session_account] = accounts[before:]
        frame = sessions["sessions"]

        assert list(frame.columns) == ["date"]
        assert list(frame["date"]) == list(_DAILY_SESSIONS)
        assert session_account.peak_rows <= 4
        assert session_account.live_rows == 0

        grouped = forward_calendar_store.daily_by_ticker(repository, parent.snapshot)
        sessions_calendar = native_trading_calendar(
            sessions, horizon_end=_HORIZON_END).days
        grouped_calendar = native_trading_calendar(
            grouped, horizon_end=_HORIZON_END).days
        assert sessions_calendar == grouped_calendar

        frames[ticker_count] = frame
        peaks.append(session_account.peak_rows)
        conn.close()

    pd.testing.assert_frame_equal(frames[3], frames[12])
    assert peaks[0] == peaks[1] <= 4

    conn, clock, store = _scope(tmp_path, "daily-sessions-empty")
    parent = _seeded_parent(
        conn,
        store,
        clock,
        scope="fwd-cal-daily-sessions-empty",
        daily_market_rows=(),
    )
    repository = Repository(conn, store)
    before = len(accounts)
    assert forward_calendar_store.daily_sessions(repository, parent.snapshot) == {}
    [empty_account] = accounts[before:]
    assert empty_account.peak_rows == 0
    assert empty_account.live_rows == 0
    conn.close()


def test_existing_index_retention_is_bounded_and_exact(tmp_path, monkeypatch):
    """Filtered event rows stay bounded while preserving the full-scan values."""
    accounts = _capture_accounts(monkeypatch)
    monkeypatch.setattr(forward_calendar_store, "MAX_SCAN_ROWS", 5)

    for row_count in (40, 120):
        conn, clock, store = _scope(tmp_path, f"events-{row_count}")
        parent = _seeded_parent(
            conn,
            store,
            clock,
            scope=f"fwd-cal-events-{row_count}",
            extra_event_rows=_extra_event_rows(row_count),
        )
        repository = Repository(conn, store)

        before = len(accounts)
        existing = forward_calendar_store._existing_index(
            repository, parent, _EVENT_KEYS)
        [existing_account] = accounts[before:]

        all_rows = _all_event_rows(repository, parent)
        assert existing == {
            key: dict(row)
            for key, row in all_rows.items()
            if key in _EVENT_KEYS
        }
        assert set(existing) == _PRESENT_EVENT_KEYS
        assert existing_account.peak_rows <= 5
        assert existing_account.live_rows == 0
        conn.close()


def test_merged_row_is_identical_from_filtered_or_unfiltered_rows(tmp_path, monkeypatch):
    """The merged event row is independent of the existing-index filter path."""
    monkeypatch.setattr(forward_calendar_store, "MAX_SCAN_ROWS", 5)
    conn, clock, store = _scope(tmp_path, "events-merge")
    parent = _seeded_parent(
        conn,
        store,
        clock,
        scope="fwd-cal-events-merge",
        extra_event_rows=_extra_event_rows(40),
    )
    repository = Repository(conn, store)

    ticker, day = _PRESENT_EVENT_KEY
    claims = {"nasdaq": "AMC", "yfinance": "BMO"}
    updated_at = "2026-09-18T00:00:00"

    filtered = forward_calendar_store._existing_index(
        repository, parent, {_PRESENT_EVENT_KEY})[_PRESENT_EVENT_KEY]
    full = _all_event_rows(repository, parent)[_PRESENT_EVENT_KEY]

    merged_filtered = forward_calendar_store._merged_row(
        filtered, ticker, day, dict(claims), updated_at=updated_at)
    merged_full = forward_calendar_store._merged_row(
        full, ticker, day, dict(claims), updated_at=updated_at)

    assert merged_filtered == merged_full
    assert merged_filtered["updated_at"] == updated_at
    assert merged_filtered["session"] == "BMO"
    assert merged_filtered["session_src"] == "yfinance"
    assert merged_filtered["src_nasdaq"] is True
    conn.close()
