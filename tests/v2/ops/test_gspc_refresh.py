"""Legacy ^GSPC CSV refresh: synthetic bodies only, no network, no repo data.

The fetcher is a stub whose ``.fetch`` returns a record with ``status`` and
``body`` like a ``RawRecord``; the target is always a ``tmp_path`` file. The
invoke-level wiring through ``legacy_adapter.invoke_price_refresh`` is covered
at the bottom of this file with every seam faked.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.data.sources.base import CredentialRotated, FetchError
from engine.data.pulls.gspc_refresh import HEADER_LINES, refresh_gspc_daily

DATES = ["2026-08-25", "2026-08-26", "2026-08-27"]


def _closes(count: int) -> list[float]:
    return [100.0 + i for i in range(count)]


def _body(dates: list[str]) -> bytes:
    """A Tier-1-shaped yfinance body: Date,Close,High,Low,Open,Volume."""
    lines = ["Date,Close,High,Low,Open,Volume"]
    for i, d in enumerate(dates):
        c = _closes(len(dates))[i]
        lines.append(f"{d},{c},{c + 1},{c - 1},{c},{1000 + i}")
    return ("\n".join(lines) + "\n").encode()


def _legacy_text(dates: list[str]) -> str:
    """The whole legacy file for ``dates``: three header lines, then rows."""
    closes = _closes(len(dates))
    lines = list(HEADER_LINES)
    for i, d in enumerate(dates):
        c = closes[i]
        lines.append(f"{d},{c},{c},{c + 1},{c - 1},{c},{1000 + i}\n")
    return "".join(lines)


def _write_legacy(path, dates: list[str]) -> bytes:
    """Seed a stored file in the legacy layout; return its exact bytes."""
    path.write_text(_legacy_text(dates))
    return path.read_bytes()


def _rows_body(rows: list[tuple[str, float]]) -> bytes:
    """A yfinance body with EXACT ``(date, close)`` pairs, OHLC around the close."""
    lines = ["Date,Close,High,Low,Open,Volume"]
    for i, (d, c) in enumerate(rows):
        lines.append(f"{d},{c},{c + 1},{c - 1},{c},{1000 + i}")
    return ("\n".join(lines) + "\n").encode()


def _write_legacy_rows(path, rows: list[tuple[str, float]]) -> bytes:
    """Seed the legacy file with EXACT ``(date, close)`` pairs; return its bytes."""
    lines = list(HEADER_LINES)
    for i, (d, c) in enumerate(rows):
        lines.append(f"{d},{c},{c},{c + 1},{c - 1},{c},{1000 + i}\n")
    path.write_text("".join(lines))
    return path.read_bytes()


class StubFetcher:
    """Returns one canned record (or raises one canned error) per call."""

    def __init__(self, *, status: int = 200, body: bytes = b"", error: Exception | None = None):
        self.status, self.body, self.error = status, body, error
        self.calls: list[tuple] = []

    def fetch(self, source, endpoint, params, *, live, note, refetch_token=None):
        self.calls.append((source, endpoint, params, live, note, refetch_token))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(status=self.status, body=self.body)


def test_success_writes_exact_legacy_layout(tmp_path):
    """The written file is the three header lines plus positional rows."""
    path = tmp_path / "gspc_daily.csv"
    fetcher = StubFetcher(body=_body(DATES))

    result = refresh_gspc_daily(fetcher, path, minimum_last_date=DATES[-1],
                                refetch_token="run-1")

    assert result == {"ok": True, "last_date": "2026-08-27", "rows": 3}
    assert fetcher.calls == [("yfinance", "history",
                              {"ticker": "^GSPC", "period": "max"},
                              True, "gspc-refresh", "run-1")]
    assert path.read_bytes() == _legacy_text(DATES).encode("utf-8")
    text = path.read_text()
    assert text.startswith("Price,Adj Close,Close,High,Low,Open,Volume\n"
                           "Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC\n"
                           "Date,,,,,,\n")
    raw = pd.read_csv(path, skiprows=3, header=None)
    closes = _closes(len(DATES))
    assert [str(v) for v in raw[0]] == DATES
    assert list(raw[1]) == closes  # adj == close: the body has no Adj Close
    assert list(raw[2]) == closes
    assert list(raw[3]) == [c + 1 for c in closes]
    assert list(raw[4]) == [c - 1 for c in closes]
    assert list(raw[5]) == closes
    assert [int(v) for v in raw[6]] == [1000 + i for i in range(len(DATES))]


def test_stale_source_leaves_file_untouched(tmp_path):
    """A source ending before the minimum is refused; bytes and mtime stay."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES[:-1])
    mtime = path.stat().st_mtime_ns

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path,
                                minimum_last_date="2026-08-28")

    assert result == {"ok": False, "reason": "stale_source"}
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime


def test_stale_source_skipping_cutoff_with_later_max_is_refused(tmp_path):
    """A history missing the cutoff is refused even when its last date is later."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES[:-1])
    mtime = path.stat().st_mtime_ns
    fetcher = StubFetcher(body=_body(["2026-08-25", "2026-08-26", "2026-08-28"]))

    result = refresh_gspc_daily(fetcher, path, minimum_last_date="2026-08-27")

    assert result == {"ok": False, "reason": "stale_source"}
    assert len(fetcher.calls) == 1
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime


def test_invalid_minimum_last_date_is_refused(tmp_path):
    """A minimum_last_date that is not a date is refused before writing."""
    path = tmp_path / "gspc_daily.csv"
    fetcher = StubFetcher(body=_body(DATES))

    result = refresh_gspc_daily(fetcher, path, minimum_last_date="not-a-date")

    assert result == {"ok": False, "reason": "invalid_cutoff"}
    assert fetcher.calls == []
    assert not path.exists()


def test_timezone_aware_minimum_last_date_is_refused(tmp_path):
    """Timezone-aware cutoffs return a refusal and leave the stored series unchanged."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES)
    fetcher = StubFetcher(body=_body(DATES))

    result = refresh_gspc_daily(
        fetcher, path,
        minimum_last_date=pd.Timestamp("2026-08-27 12:00:00", tz="America/Toronto"),
    )

    assert result == {"ok": False, "reason": "invalid_cutoff"}
    assert fetcher.calls == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("cutoff", [20261021, True])
def test_numeric_minimum_last_date_is_refused(tmp_path, cutoff):
    """Numeric cutoffs are refused before fetching or writing."""
    path = tmp_path / "gspc_daily.csv"
    fetcher = StubFetcher(body=_body(DATES))

    result = refresh_gspc_daily(fetcher, path, minimum_last_date=cutoff)

    assert result == {"ok": False, "reason": "invalid_cutoff"}
    assert fetcher.calls == []
    assert not path.exists()


def test_identical_second_call_leaves_file_untouched(tmp_path):
    """Same bytes => no rewrite, so the panel's mtime-keyed cache stays warm."""
    path = tmp_path / "gspc_daily.csv"
    refresh_gspc_daily(StubFetcher(body=_body(DATES)), path, refetch_token="run-1")
    before = path.read_bytes()
    mtime = path.stat().st_mtime_ns

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path, refetch_token="run-2")

    assert result == {"ok": True, "last_date": "2026-08-27", "rows": 3}
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("failure", [FetchError("synthetic outage"), OSError("down"),
                                     ValueError("bad")])
def test_fetch_errors_refuse_without_touching_the_file(tmp_path, failure):
    """An exception is a refusal, never a partial or missing write."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES)

    result = refresh_gspc_daily(StubFetcher(error=failure), path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "fetch_failed"}
    assert path.read_bytes() == before


def test_non_200_status_refuses_without_touching_the_file(tmp_path):
    """A stubbed 500 record is refused by the status guard, file unchanged."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES)

    result = refresh_gspc_daily(StubFetcher(status=500, body=_body(DATES)), path,
                               refetch_token="run-1")

    assert result == {"ok": False, "reason": "fetch_failed"}
    assert path.read_bytes() == before


def test_garbage_body_is_unparseable(tmp_path):
    """A body with no Close column never reaches the file."""
    path = tmp_path / "gspc_daily.csv"

    result = refresh_gspc_daily(StubFetcher(body=b"this is not a price file\n"), path,
                               refetch_token="run-1")

    assert result == {"ok": False, "reason": "unparseable"}
    assert not path.exists()


def test_oversize_fresh_body_is_unparseable_before_decoding(tmp_path, monkeypatch):
    """A body above the ceiling is refused, never parsed into a frame."""
    import engine.data.pulls.gspc_refresh as gspc_module

    path = tmp_path / "gspc_daily.csv"
    monkeypatch.setattr(gspc_module, "MAX_FRESH_BYTES", 16)

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path,
                                refetch_token="run-1")

    assert result == {"ok": False, "reason": "unparseable"}
    assert not path.exists()


def test_fewer_rows_than_stored_is_refused_as_shrunk(tmp_path):
    """A smaller answer may not chop history off the series."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, ["2026-07-31", *DATES])

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "shrunk"}
    assert path.read_bytes() == before


def test_older_last_date_than_stored_is_refused(tmp_path):
    """An answer ending before the stored end moves the clock backwards."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, ["2026-08-25", "2026-08-26", "2026-08-27"])

    result = refresh_gspc_daily(
        StubFetcher(body=_body(["2026-08-24", "2026-08-25", "2026-08-26"])),
        path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "older_than_existing"}
    assert path.read_bytes() == before


def test_same_row_count_with_later_first_date_is_shrunk(tmp_path):
    """Same length but a later start chopped the earliest session off the front."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, ["2026-08-24", "2026-08-25", "2026-08-26"])

    result = refresh_gspc_daily(
        StubFetcher(body=_body(["2026-08-25", "2026-08-26", "2026-08-27"])),
        path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "shrunk"}
    assert path.read_bytes() == before


def test_dropped_mid_series_date_is_shrunk(tmp_path):
    """Equal count, equal first and last dates, yet a stored date vanished."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, ["2026-08-24", "2026-08-26", "2026-08-27"])

    result = refresh_gspc_daily(
        StubFetcher(body=_body(["2026-08-24", "2026-08-25", "2026-08-27"])),
        path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "shrunk"}
    assert path.read_bytes() == before


def test_pre_2006_rows_are_cut_to_the_stored_start(tmp_path):
    """A 1927-deep period=max body is accepted, written from the stored start."""
    path = tmp_path / "gspc_daily.csv"
    stored = ["2006-01-03", "2006-01-04", *DATES]
    _write_legacy(path, stored)
    body_dates = ["1927-01-04", "1986-07-08", "2005-12-30", *stored, "2026-08-28"]

    result = refresh_gspc_daily(StubFetcher(body=_body(body_dates)), path,
                                refetch_token="run-1")

    assert result == {"ok": True, "last_date": "2026-08-28", "rows": 6}
    text = path.read_text()
    assert "1927" not in text and "1986" not in text and "2005-12-30" not in text
    raw = pd.read_csv(path, skiprows=3, header=None)
    assert [str(v) for v in raw[0]] == [*stored, "2026-08-28"]


def test_catchup_rewrite_writes_the_exact_bytes(tmp_path):
    """Stored file plus three extra later dates: the whole rewritten file is exact."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES)
    caught_up = [*DATES, "2026-08-28", "2026-08-29", "2026-08-30"]

    result = refresh_gspc_daily(StubFetcher(body=_body(caught_up)), path,
                                refetch_token="run-1")

    assert result == {"ok": True, "last_date": "2026-08-30", "rows": 6}
    assert path.read_bytes() == _legacy_text(caught_up).encode("utf-8")
    assert path.read_bytes() != before


#: Eight stored dates: the first three are older than the tail (their Close
#: must survive a refresh), the last five are the correctable tail
#: (``TAIL_CORRECTABLE_ROWS``).
TAIL_DATES = [f"2026-08-{day:02d}" for day in range(20, 28)]


def test_head_close_mismatch_refuses_as_history_changed(tmp_path):
    """A fresh answer that rewrites a stored Close outside the last five
    stored rows is a DIFFERENT history, not a catch-up: refused, the file
    untouched."""
    path = tmp_path / "gspc_daily.csv"
    stored = [(d, 100.0 + i) for i, d in enumerate(TAIL_DATES)]
    before = _write_legacy_rows(path, stored)
    changed = list(stored)
    changed[0] = (changed[0][0], 100.0 + 1e-3)  # rel ~1e-5, far outside 1e-9

    result = refresh_gspc_daily(StubFetcher(body=_rows_body(changed)), path,
                                refetch_token="run-1")

    assert result == {"ok": False, "reason": "history_changed"}
    assert path.read_bytes() == before


def test_sixth_from_last_close_mismatch_is_history_changed(tmp_path):
    """The tail is EXACTLY the last five stored rows: the 6th-from-last row
    (the head's newest) is still guarded."""
    path = tmp_path / "gspc_daily.csv"
    stored = [(d, 100.0 + i) for i, d in enumerate(TAIL_DATES)]
    before = _write_legacy_rows(path, stored)
    changed = list(stored)
    changed[-6] = (changed[-6][0], changed[-6][1] + 1e-6)  # rel ~1e-8 > 1e-9

    result = refresh_gspc_daily(StubFetcher(body=_rows_body(changed)), path,
                                refetch_token="run-1")

    assert result == {"ok": False, "reason": "history_changed"}
    assert path.read_bytes() == before


def test_tail_close_mismatch_is_accepted_and_the_fresh_value_is_written(tmp_path):
    """Mismatches inside the last five stored rows are the corrections this
    pull exists to make: accepted, and the written file carries the fresh
    value. A head row inside the rel 1e-9 tolerance is likewise accepted and
    rewritten with the fresh value."""
    path = tmp_path / "gspc_daily.csv"
    stored = [(d, 100.0 + i) for i, d in enumerate(TAIL_DATES)]
    before = _write_legacy_rows(path, stored)
    corrected = list(stored)
    corrected[-5] = (corrected[-5][0], 555.0)                     # 5th from last
    corrected[-1] = (corrected[-1][0], corrected[-1][1] + 1e-10)  # tail
    corrected[1] = (corrected[1][0], 101.0 + 1e-10)  # head, rel ~1e-12 <= 1e-9

    result = refresh_gspc_daily(StubFetcher(body=_rows_body(corrected)), path,
                                refetch_token="run-1")

    assert result == {"ok": True, "last_date": "2026-08-27", "rows": 8}
    raw = pd.read_csv(path, skiprows=3, header=None)
    assert list(raw[2])[-5] == 555.0
    assert list(raw[2])[-1] == pytest.approx(107.0 + 1e-10)
    assert list(raw[2])[1] == pytest.approx(101.0 + 1e-10)
    assert path.read_bytes() != before


def test_existing_directory_is_unreadable(tmp_path):
    """A directory at the target path is never a stored series; refused, untouched."""
    path = tmp_path / "gspc_daily.csv"
    path.mkdir()

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "unreadable"}
    assert path.is_dir()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo unavailable")
def test_fifo_at_the_target_is_unreadable(tmp_path):
    """A FIFO fed a valid stored series still fails the S_ISREG check.

    The bytes go in through an ``O_RDWR`` open (which never blocks on a
    FIFO) and the refresh opens ``O_NONBLOCK``, so neither side can hang.
    Without the regular-file check the guarded read would consume the
    buffered rows, pass every guard and replace the FIFO with a real file.
    """
    path = tmp_path / "gspc_daily.csv"
    os.mkfifo(path)
    fd = os.open(str(path), os.O_RDWR | os.O_NONBLOCK)
    try:
        os.write(fd, _legacy_text(DATES).encode("utf-8"))
    finally:
        os.close(fd)

    result = refresh_gspc_daily(StubFetcher(body=_body([*DATES, "2026-08-28"])),
                                path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "unreadable"}
    assert stat.S_ISFIFO(os.stat(path).st_mode)
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "gspc_daily.csv", "gspc_daily.csv.lock"]


def test_oversized_existing_is_unreadable(tmp_path, monkeypatch):
    """A regular file above the size ceiling is refused rather than slurped in."""
    import engine.data.pulls.gspc_refresh as gspc_module

    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES)
    monkeypatch.setattr(gspc_module, "MAX_EXISTING_BYTES", 16)

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "unreadable"}
    assert path.read_bytes() == before


def test_first_write_when_file_is_absent(tmp_path):
    """With no stored file there is nothing to guard against; it is created."""
    path = tmp_path / "gspc_daily.csv"

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path, refetch_token="run-1")

    assert result["ok"] is True and result["rows"] == 3
    assert path.read_bytes() == _legacy_text(DATES).encode("utf-8")


def test_malformed_path_is_unreadable_before_fetching():
    """A path that is not even a path is refused without spending a fetch."""
    fetcher = StubFetcher(body=_body(DATES))

    result = refresh_gspc_daily(fetcher, None, refetch_token="run-1")

    assert result == {"ok": False, "reason": "unreadable"}
    assert fetcher.calls == []


def test_is_grandfathered_allows_only_the_real_staging_siblings():
    """The carve-out names are exactly ``.lock`` and ``.<digits>.<hex32>.tmp``."""
    from engine import paths

    hex32 = "0123456789abcdef" * 2
    lock = paths.GSPC_DAILY.with_name("gspc_daily.csv.lock")
    tmp = paths.GSPC_DAILY.with_name(f"gspc_daily.csv.{os.getpid()}.{hex32}.tmp")
    evil = paths.GSPC_DAILY.with_name("gspc_daily.csv.evil")

    assert paths.is_grandfathered(lock) is False
    assert paths.is_grandfathered(tmp) is False
    assert paths.is_grandfathered(evil) is True
    with pytest.raises(PermissionError, match="grandfathered"):
        paths.assert_writable(evil)


def test_credential_rotated_propagates_and_leaves_the_file_untouched(tmp_path):
    """A whole-run credential condition is never swallowed into a refusal."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES)
    error = CredentialRotated("rotated")

    with pytest.raises(CredentialRotated) as raised:
        refresh_gspc_daily(StubFetcher(error=error), path, refetch_token="run-1")

    # The very object the fetcher raised, unchanged -- not a re-wrap.
    assert raised.value is error
    assert str(raised.value) == "rotated"
    assert path.read_bytes() == before


def test_assert_writable_permission_error_is_unwritable(tmp_path, monkeypatch):
    """A refusal inside the staging guard loop is the unwritable reason.

    Only the ``.tmp`` sibling is denied, so the lock guard and the target
    guard pass and the refusal lands on the second ``assert_writable`` call
    of the ``for guard in (path, tmp)`` loop -- never a raised error.
    """
    import engine.paths as paths_module

    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES)
    guards: list[str] = []

    def deny(p):
        guards.append(str(p))
        if str(p).endswith(".tmp"):
            raise PermissionError(f"refusing to write: {p}")

    monkeypatch.setattr(paths_module, "assert_writable", deny)

    result = refresh_gspc_daily(StubFetcher(body=_body([*DATES, "2026-08-28"])),
                                path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "unwritable"}
    assert path.read_bytes() == before
    assert len(guards) == 3
    assert guards[:2] == [str(path) + ".lock", str(path)]
    assert guards[2].endswith(".tmp")
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "gspc_daily.csv", "gspc_daily.csv.lock"]


def test_replace_failure_is_unwritable_and_leaves_no_stray_files(tmp_path, monkeypatch):
    """An ``os.replace`` OSError refuses with the bytes intact, and the temp
    sibling is cleaned up; only the ``.lock`` sibling may remain."""
    path = tmp_path / "gspc_daily.csv"
    before = _write_legacy(path, DATES)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)

    result = refresh_gspc_daily(StubFetcher(body=_body([*DATES, "2026-08-28"])), path,
                                refetch_token="run-1")

    assert result == {"ok": False, "reason": "unwritable"}
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [
        "gspc_daily.csv", "gspc_daily.csv.lock"]


def test_adj_close_column_is_used_for_the_adj_slot(tmp_path):
    """With a real Adj Close in the body, column 2 carries it, not Close."""
    path = tmp_path / "gspc_daily.csv"
    closes = _closes(len(DATES))
    lines = ["Date,Adj Close,Close,High,Low,Open,Volume"]
    for i, d in enumerate(DATES):
        c = closes[i]
        lines.append(f"{d},{c * 0.5},{c},{c + 1},{c - 1},{c},{1000 + i}")
    body = ("\n".join(lines) + "\n").encode()

    result = refresh_gspc_daily(StubFetcher(body=body), path, refetch_token="run-1")

    assert result["ok"] is True
    raw = pd.read_csv(path, skiprows=3, header=None)
    assert list(raw[1]) == [c * 0.5 for c in closes]  # adj == Adj Close
    assert list(raw[2]) == closes  # close stays the raw Close series


def test_non_finite_close_dropped_and_inf_volume_does_not_raise(tmp_path):
    """inf Close loses its row; inf volume renders as 0, never an int64 crash."""
    path = tmp_path / "gspc_daily.csv"
    body = (b"Date,Close,High,Low,Open,Volume\n"
            b"2026-08-25,100.0,101,99,100,inf\n"
            b"2026-08-26,inf,102,100,101,1001\n"
            b"2026-08-27,102.0,103,101,102,1002\n")

    result = refresh_gspc_daily(StubFetcher(body=body), path, refetch_token="run-1")

    assert result == {"ok": True, "last_date": "2026-08-27", "rows": 2}
    raw = pd.read_csv(path, skiprows=3, header=None)
    assert [str(v) for v in raw[0]] == ["2026-08-25", "2026-08-27"]
    assert [int(v) for v in raw[6]] == [0, 1002]


def test_symlink_target_is_unreadable(tmp_path):
    """A symlink at the target is not a stored series; refused, real bytes intact."""
    real = tmp_path / "stored.csv"
    real.write_text(_legacy_text(DATES))
    path = tmp_path / "gspc_daily.csv"
    path.symlink_to(real)

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path,
                                refetch_token="run-1")

    assert result == {"ok": False, "reason": "unreadable"}
    assert real.read_text() == _legacy_text(DATES)
    assert path.is_symlink()


def test_invoke_price_refresh_real_run_returns_gspc_daily(tmp_path, monkeypatch):
    """The adapter wires one Fetcher into both crossings and reports the pull.

    Every seam is faked -- plan/run/Fetcher/refresh -- so this exercises only
    the adapter's own crossing: the new ``gspc_daily`` key next to the
    unchanged ``plan``/``report`` keys, one Fetcher shared by both calls, the
    target under the supplied ``source_root``, and a unique refetch token per
    run (the ^GSPC cache key never varies on its own).
    """
    import engine.data.fetch as fetch_module
    import engine.data.pulls.price_refresh as price_refresh_module
    import engine.data.pulls.gspc_refresh as gspc_module

    from engine.v2.ops.legacy_adapter import invoke_price_refresh

    source_root = tmp_path / "selected-source"
    plan = {"session": "2026-10-20", "daily": ["AAA"], "monthly": [],
            "skipped_already_fetched": []}
    report = {"counts": {"fetched": 1}}
    outcome = {"ok": True, "last_date": "2026-10-20", "rows": 1}
    constructed: list = []
    runs: list = []
    calls: list = []

    class FakeFetcher:
        def __init__(self, *, root=None):
            self.root = root
            constructed.append(self)

    monkeypatch.setattr(fetch_module, "Fetcher", FakeFetcher)
    monkeypatch.setattr(price_refresh_module, "plan_refresh",
                        lambda session, **kwargs: plan)
    monkeypatch.setattr(price_refresh_module, "run_refresh",
                        lambda planned, fetcher: runs.append((planned, fetcher)) or report)

    def fake_refresh(fetcher, path, *, refetch_token):
        calls.append((fetcher, Path(path), refetch_token))
        return dict(outcome)

    monkeypatch.setattr(gspc_module, "refresh_gspc_daily", fake_refresh)

    result = invoke_price_refresh("2026-10-20", planned_tickers=("AAA",),
                                  source_root=source_root)

    assert result == {"plan": plan, "report": report, "gspc_daily": outcome}
    assert runs == [(plan, constructed[0])]
    assert constructed[0].root == source_root / "data" / "raw" / "fetch"
    fetcher, path, token = calls[0]
    assert fetcher is constructed[0]
    assert path == (source_root / "earnings_predictions" / "data" / "raw"
                    / "polygon" / "gspc_daily.csv")
    assert path.is_relative_to(source_root)
    assert isinstance(token, str) and token

    second = invoke_price_refresh("2026-10-20", planned_tickers=("AAA",),
                                  source_root=source_root)
    assert second["gspc_daily"] == outcome
    assert calls[1][2] != token


def test_invoke_price_refresh_source_root_without_planned_tickers_refuses(tmp_path):
    """The mirror of the planned-path check: a bare source root is a bad request."""
    from engine.v2.ops.errors import OpsError
    from engine.v2.ops.legacy_adapter import invoke_price_refresh

    with pytest.raises(OpsError) as error:
        invoke_price_refresh("2026-10-20", source_root=tmp_path / "selected")

    assert error.value.code == "INVALID_REQUEST"
    assert error.value.problem.message == "source_root requires planned tickers"


def test_invoke_price_refresh_blank_source_root_refuses(tmp_path):
    """A blank (whitespace-only) source root is as bad as a missing one."""
    from engine.v2.ops.errors import OpsError
    from engine.v2.ops.legacy_adapter import invoke_price_refresh

    with pytest.raises(OpsError) as error:
        invoke_price_refresh("2026-10-20", planned_tickers=("AAA",),
                             source_root="   ")

    assert error.value.code == "INVALID_REQUEST"
    assert error.value.problem.message == "source_root must not be blank"


def test_cli_price_refresh_real_run_report_includes_gspc_daily(tmp_path, monkeypatch):
    """The CLI carries the ^GSPC outcome into both the saved JSON and the return.

    Only ``invoke_price_refresh`` is faked (returning plan/report/gspc_daily);
    the command's own real-run path -- write the saved report, then hand back
    the dict -- runs for real under ``tmp_path``.
    """
    from engine.v2.ops import cli

    plan = {"session": "2026-10-20", "daily": ["AAA"], "monthly": [],
            "skipped_already_fetched": []}
    report = {"counts": {"fetched": 1}}
    outcome = {"ok": True, "last_date": "2026-10-20", "rows": 1}
    monkeypatch.setattr(
        "engine.v2.ops.legacy_adapter.invoke_price_refresh",
        lambda *args, **kwargs: {"plan": plan, "report": report, "gspc_daily": outcome},
    )
    args = SimpleNamespace(session="2026-10-20", dry_run=False,
                           tickers=None, source_root=None)
    root = tmp_path / "ops-root"

    result = cli.price_refresh_command(args, root)

    assert result["gspc_daily"] == outcome
    saved = json.loads((root / "price_refresh" / "2026-10-20.json").read_text())
    assert saved["gspc_daily"] == outcome
