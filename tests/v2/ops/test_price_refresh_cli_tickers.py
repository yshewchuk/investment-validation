"""``ops price-refresh --tickers``/``--source-root`` contract at the CLI.

Only the provider-facing ``Fetcher`` and the legacy discovery loaders are
faked. Planning is real; the source tree lives under pytest's temporary
directory and dry runs never construct a provider client or write disk.
"""
from __future__ import annotations

import contextlib
import json
from contextlib import redirect_stdout
from io import StringIO

import pytest

from engine.data import fetch
from engine.v2.ops import cli

SESSION = "2026-09-11"


def run(args, root):
    """Invoke ``cli.main`` for ``price-refresh`` and parse the printed JSON."""
    document = json.loads(_invoke(args, root))
    return document


def _invoke(args, root):
    """Return ``main``'s raw printed document (the tests only need the bytes)."""
    buffer = StringIO()
    with redirect_stdout(buffer):
        cli.main(["--root", str(root), "price-refresh", *args])
    return buffer.getvalue()


def test_tickers_without_source_root_is_invalid_request(tmp_path, monkeypatch):
    """``--tickers`` without ``--source-root`` refuses before any provider work."""
    def unexpected_fetcher(*args, **kwargs):
        pytest.fail("Fetcher constructed")

    monkeypatch.setattr(fetch, "Fetcher", unexpected_fetcher)

    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        result = cli.main(["--root", str(tmp_path / "ops-root"), "price-refresh",
                           "--session", SESSION, "--dry-run", "--tickers", "AAPL"])
    printed = buffer.getvalue()
    assert result == 2
    assert json.loads(printed)["code"] == "INVALID_REQUEST"


def test_source_root_without_tickers_is_invalid_request(tmp_path, monkeypatch):
    """``--source-root`` without ``--tickers`` refuses before any provider work."""
    def unexpected_fetcher(*args, **kwargs):
        pytest.fail("Fetcher constructed")

    monkeypatch.setattr(fetch, "Fetcher", unexpected_fetcher)

    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        result = cli.main(["--root", str(tmp_path / "ops-root"), "price-refresh",
                           "--session", SESSION, "--dry-run",
                           "--source-root", str(tmp_path)])
    printed = buffer.getvalue()
    assert result == 2
    assert json.loads(printed)["code"] == "INVALID_REQUEST"


def test_tickers_file_dry_run_plans_without_writing(tmp_path, monkeypatch):
    """A ``@file`` watchlist plans exactly those names plus SPY, no side effects."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    tickers_file = tmp_path / "watchlist.txt"
    tickers_file.write_text("ZZNEW\nAAPL\n")

    def unexpected_fetcher(*args, **kwargs):
        pytest.fail("Fetcher constructed")

    monkeypatch.setattr(fetch, "Fetcher", unexpected_fetcher)
    monkeypatch.setattr("engine.data.pulls.price_refresh.load_events",
                        lambda: pytest.fail("legacy events universe read"))
    monkeypatch.setattr("engine.data.pulls.price_refresh.load_price_universe",
                        lambda: pytest.fail("legacy inventory universe read"))

    document = run(["--session", SESSION, "--dry-run",
                    "--tickers", f"@{tickers_file}",
                    "--source-root", str(source_root)],
                   tmp_path / "ops-root")

    assert document["session"] == SESSION
    assert document["dry_run"] is True
    assert document["counts"] == {"daily": 3, "monthly": 0,
                                  "skipped_already_fetched": 0}
    assert not (source_root / "data").exists()


def test_tickers_dry_run_counts_names_plus_calendar(tmp_path, monkeypatch, capsys):
    """Asserts the daily count (given names plus SPY)."""
    source_root = tmp_path / "source"
    source_root.mkdir()

    def unexpected_fetcher(*args, **kwargs):
        pytest.fail("Fetcher constructed")

    monkeypatch.setattr(fetch, "Fetcher", unexpected_fetcher)
    monkeypatch.setattr("engine.data.pulls.price_refresh.load_events",
                        lambda: pytest.fail("legacy events universe read"))
    monkeypatch.setattr("engine.data.pulls.price_refresh.load_price_universe",
                        lambda: pytest.fail("legacy inventory universe read"))

    document = run(["--session", SESSION, "--dry-run",
                    "--tickers", "ZZNEW,AAPL",
                    "--source-root", str(source_root)],
                   tmp_path / "ops-root")

    assert document["session"] == SESSION
    assert document["dry_run"] is True
    assert document["counts"] == {"daily": 3, "monthly": 0,
                                  "skipped_already_fetched": 0}
    assert not (source_root / "data").exists()


def test_empty_tickers_file_is_invalid_request(tmp_path, monkeypatch):
    """An empty ``@file`` watchlist refuses before any provider work."""
    def unexpected_fetcher(*args, **kwargs):
        pytest.fail("Fetcher constructed")

    monkeypatch.setattr(fetch, "Fetcher", unexpected_fetcher)

    tickers_file = tmp_path / "empty.txt"
    tickers_file.write_text("")

    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        result = cli.main(["--root", str(tmp_path / "ops-root"), "price-refresh",
                           "--session", SESSION, "--dry-run",
                           "--tickers", "@" + str(tickers_file),
                           "--source-root", str(tmp_path / "src")])
    printed = buffer.getvalue()
    assert result == 2
    assert json.loads(printed)["code"] == "INVALID_REQUEST"


def test_without_tickers_keeps_legacy_universe(tmp_path, monkeypatch):
    """With no ``--tickers``, the CLI still routes through the legacy adapter."""
    calls = []

    def record_invoke(args, kwargs):
        calls.append((args, kwargs))
        return {"plan": {"session": SESSION, "daily": [], "monthly": [],
                         "skipped_already_fetched": []},
                "report": None}

    monkeypatch.setattr(
        "engine.v2.ops.legacy_adapter.invoke_price_refresh",
        lambda *args, **kwargs: record_invoke(args, kwargs),
    )

    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        result = cli.main(["--root", str(tmp_path / "ops-root"), "price-refresh",
                           "--session", SESSION, "--dry-run"])
    assert result == 0
    assert calls[0][0] == (SESSION,)
    assert calls[0][1] == {"dry_run": True}