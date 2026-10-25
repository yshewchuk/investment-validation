"""The ``REGIME_TICKER`` (^GSPC), the single source for the nine ``spy_*``
panel regime features, is a declared ``price_history`` dependency: capture
ingests the legacy ``gspc_daily.csv`` (``source_kind="legacy_gspc_csv"``) as
one more ticker with the same required-ticker / tombstone / no-drop-history
rules, and ``SOURCE_NOT_FOUND`` refuses before any write when a required
^GSPC is absent from both the sources and the prior dataset version. The
fixture csv's Adj Close/Close/High/Low/Open columns are deliberately all
distinct (a capture reading any column but Close fails), and both stored
closes must equal Close exactly (^GSPC is never dividend-adjusted). Synthetic
px trees and a synthetic three-header gspc csv only -- no network, no real
market data.
"""
from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from engine.data.pulls.gspc_refresh import HEADER_LINES
from engine.v2.data.price_history_table import (
    CALENDAR_TICKER,
    REGIME_TICKER,
    REQUIRED_CAPTURE_TICKERS,
)
from engine.v2.ops.errors import OpsError
from engine.v2.ops.price_history_store import _GSPC_MAX_BYTES, capture
from tests.v2.ops.test_calendar_ticker_price_history import _write_sources
from tests.v2.ops.test_v2_ops_price_history import (
    _base_snapshot,
    _scan_price_history,
)

_GSPC_DATES = ("2024-01-02", "2024-01-03", "2024-01-04")
_GSPC_CLOSES = (4700.913, 4742.83, 4756.01)  # round_trip-sensitive on purpose


def _gspc_path(source_root):
    return source_root / "earnings_predictions" / "data" / "raw" / "polygon" / "gspc_daily.csv"


def _write_gspc_rows(source_root, rows):
    directory = _gspc_path(source_root).parent
    directory.mkdir(parents=True, exist_ok=True)
    lines = list(HEADER_LINES) + [row + "\n" for row in rows]
    _gspc_path(source_root).write_text("".join(lines))


def _write_gspc(source_root, *, closes=_GSPC_CLOSES, dates=_GSPC_DATES) -> None:
    """The legacy layout: three header lines, then positional
    ``date,adj,close,high,low,open,volume`` rows. Every price column is
    distinct (adj = close + 100, high = close + 5, low = close - 3,
    open = close - 1), so a capture reading any column other than Close for
    the stored closes fails loudly."""
    _write_gspc_rows(source_root, [
        f"{day},{close + 100.0},{close},{close + 5.0},{close - 3.0},"
        f"{close - 1.0},{close * 1e9:.0f}"
        for day, close in zip(dates, closes, strict=True)])


def test_capture_ingests_gspc_daily_csv_as_the_regime_ticker(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    _write_gspc(source_root)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["failed_tickers"] == 0
    rows = _scan_price_history(conn, store, report["result_snapshot_id"],
                               ticker=REGIME_TICKER)
    assert not rows.empty
    assert sorted(rows["date"]) == list(_GSPC_DATES)
    assert set(rows["source_kind"]) == {"legacy_gspc_csv"}
    assert not rows["deleted"].any()
    by_date = rows.set_index("date")
    for i, day in enumerate(_GSPC_DATES):
        # ^GSPC has no dividend adjustment: close_adj == close_raw == Close
        # EXACTLY (the distinct Adj Close column must never reach the table),
        # high_raw == High exactly.
        assert by_date.loc[day, "close_raw"] == _GSPC_CLOSES[i]
        assert by_date.loc[day, "close_adj"] == _GSPC_CLOSES[i]
        assert by_date.loc[day, "high_raw"] == _GSPC_CLOSES[i] + 5.0


def test_gspc_source_hash_is_the_content_hash_of_the_bytes_read(tmp_path):
    """Hash, ``retrieved_at`` and rows all describe ONE read: the stored
    ``source_hash`` is exactly the sha256 of the file's bytes."""
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    _write_gspc(source_root)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    rows = _scan_price_history(conn, store, report["result_snapshot_id"],
                               ticker=REGIME_TICKER)
    raw = _gspc_path(source_root).read_bytes()
    assert set(rows["source_hash"]) == {hashlib.sha256(raw).hexdigest()}


def test_gspc_null_close_row_is_dropped(tmp_path):
    """A non-numeric ``Close`` coerces to NaN and the row is dropped (the
    legacy ``_gspc_series`` tolerated null closes); ``high_raw`` may stay
    NaN -- the 2024-01-02 row carries no High and is kept."""
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=False)
    _write_gspc_rows(source_root, [
        "2024-01-02,4800.913,4700.913,,4697.913,4699.913,4700913000000",
        "2024-01-03,4842.83,,4747.83,4739.83,4741.83,4742830000000",
        "2024-01-04,4856.01,4756.01,4761.01,4753.01,4755.01,4756010000000",
    ])
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["failed_tickers"] == 0
    rows = _scan_price_history(conn, store, report["result_snapshot_id"],
                               ticker=REGIME_TICKER)
    assert sorted(rows["date"]) == ["2024-01-02", "2024-01-04"]
    kept = rows[rows["date"] == "2024-01-02"].iloc[0]
    assert kept["close_raw"] == 4700.913 and kept["close_adj"] == 4700.913
    assert pd.isna(kept["high_raw"])


def test_oversize_gspc_file_refuses_like_a_malformed_csv(tmp_path):
    """A file over the 32 MiB capture cap refuses on ``fstat`` BEFORE its
    bytes are read: required, it is not a usable ticker (``SOURCE_NOT_FOUND``,
    nothing committed); unrequired, it fails only its own retrieval, reported
    as a parse error exactly like a malformed px csv."""
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    path = _gspc_path(source_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.truncate(_GSPC_MAX_BYTES + 1)
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=(REGIME_TICKER,))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()
    assert head["snapshot_id"] == base.snapshot_id

    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert [(f["ticker"], f["error"]) for f in report["failures"]] == \
        [(REGIME_TICKER, "ValueError")]


def test_malformed_required_regime_ticker_file_fails_like_a_malformed_px_csv(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    _gspc_path(source_root).parent.mkdir(parents=True, exist_ok=True)
    _gspc_path(source_root).write_bytes(b"not,a,price\nfile\n")
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=(REGIME_TICKER,))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": [REGIME_TICKER]}
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()
    assert head["snapshot_id"] == base.snapshot_id  # nothing was committed

    # Not required: the malformed file fails only its own ticker's capture,
    # exactly as a malformed px csv does, and the run still commits the rest.
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert [(f["ticker"], f["error"]) for f in report["failures"]] == \
        [(REGIME_TICKER, "EmptyDataError")]
    assert report["failed_tickers"] == 1
    assert "result_snapshot_id" in report


def test_missing_required_regime_ticker_refuses_source_not_found(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)  # SPY present, ^GSPC absent
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=REQUIRED_CAPTURE_TICKERS)
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": [REGIME_TICKER]}
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()
    assert head["snapshot_id"] == base.snapshot_id  # nothing was committed


def test_malformed_gspc_falls_back_to_the_prior_stored_history(tmp_path):
    """A malformed ``gspc_daily.csv`` never destroys an already-captured
    ^GSPC history: the failed retrieval changes nothing, the ticker stays
    usable through the prior dataset version (the required check reads its
    live rows), and the run commits the prior rows forward."""
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    _write_gspc(source_root)
    first = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                    required_tickers=REQUIRED_CAPTURE_TICKERS)
    before = _scan_price_history(conn, store, first["result_snapshot_id"],
                                 ticker=REGIME_TICKER)

    _gspc_path(source_root).write_bytes(b"not,a,price\nfile\n")
    second = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                     required_tickers=REQUIRED_CAPTURE_TICKERS)
    assert [(f["ticker"], f["error"]) for f in second["failures"]] == \
        [(REGIME_TICKER, "EmptyDataError")]  # the parse still fails and is reported
    assert "result_snapshot_id" in second  # but the prior history keeps the run committable
    after = _scan_price_history(conn, store, second["result_snapshot_id"],
                                ticker=REGIME_TICKER)
    cols = ["date", "close_adj", "close_raw", "high_raw", "retrieved_at", "deleted",
            "source_hash"]
    assert after[cols].sort_values("date").reset_index(drop=True).equals(
        before[cols].sort_values("date").reset_index(drop=True))
    assert not after["deleted"].any()


def test_required_capture_tickers_is_shared_by_cli_and_nightly_trigger():
    """``ops price-history capture`` (cli) and the nightly trigger pass the
    SAME shared tuple ``(CALENDAR_TICKER, REGIME_TICKER)`` as the capture's
    required tickers -- imported, never restated at either call site."""
    import inspect

    from engine.v2.ops import cli, nightly_trigger

    assert REQUIRED_CAPTURE_TICKERS == (CALENDAR_TICKER, REGIME_TICKER)
    for fn in (cli.price_history_command, nightly_trigger._refresh_plan_prices):
        source = inspect.getsource(fn)
        assert "required_tickers=REQUIRED_CAPTURE_TICKERS" in source
