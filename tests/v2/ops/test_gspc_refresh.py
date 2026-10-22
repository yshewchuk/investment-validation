"""Legacy ^GSPC CSV refresh: synthetic bodies only, no network, no repo data.

The fetcher is a stub whose ``.fetch`` returns a record with ``status`` and
``body`` like a ``RawRecord``; the target is always a ``tmp_path`` file. The
invoke-level wiring through ``legacy_adapter.invoke_price_refresh`` is covered
at the bottom of this file with every seam faked.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.data.sources.base import FetchError
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

    result = refresh_gspc_daily(fetcher, path, refetch_token="run-1")

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


def test_existing_directory_is_unreadable(tmp_path):
    """A directory at the target path is never a stored series; refused, untouched."""
    path = tmp_path / "gspc_daily.csv"
    path.mkdir()

    result = refresh_gspc_daily(StubFetcher(body=_body(DATES)), path, refetch_token="run-1")

    assert result == {"ok": False, "reason": "unreadable"}
    assert path.is_dir()


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
    assert path.exists()


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
