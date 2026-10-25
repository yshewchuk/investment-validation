"""The ``REGIME_TICKER`` (^GSPC), the single source for the nine ``spy_*``
panel regime features, is a declared ``price_history`` dependency: capture
ingests the legacy ``gspc_daily.csv`` (``source_kind="legacy_gspc_csv"``) as
one more ticker with the same required-ticker / tombstone / no-drop-history
rules, and ``SOURCE_NOT_FOUND`` refuses before any write when a required
^GSPC is absent from both the sources and the prior dataset version.
Synthetic px trees and a synthetic three-header gspc csv only -- no network,
no real market data.
"""
from __future__ import annotations

import pytest

from engine.data.pulls.gspc_refresh import HEADER_LINES
from engine.v2.data.price_history_table import CALENDAR_TICKER, REGIME_TICKER
from engine.v2.ops.errors import OpsError
from engine.v2.ops.price_history_store import capture
from tests.v2.ops.test_calendar_ticker_price_history import _write_sources
from tests.v2.ops.test_v2_ops_price_history import (
    _base_snapshot,
    _scan_price_history,
)

_GSPC_DATES = ("2024-01-02", "2024-01-03", "2024-01-04")
_GSPC_CLOSES = (4700.913, 4742.83, 4756.01)  # round_trip-sensitive on purpose


def _write_gspc(source_root, *, closes=_GSPC_CLOSES, dates=_GSPC_DATES) -> None:
    """The legacy layout: three header lines, then positional
    ``date,adj,close,high,low,open,volume`` rows (high = close + 5.0)."""
    directory = source_root / "earnings_predictions" / "data" / "raw" / "polygon"
    directory.mkdir(parents=True, exist_ok=True)
    lines = list(HEADER_LINES)
    for day, close in zip(dates, closes, strict=True):
        lines.append(f"{day},{close},{close},{close + 5.0},{close - 3.0},"
                     f"{close - 1.0},{close * 1e9:.0f}\n")
    (directory / "gspc_daily.csv").write_text("".join(lines))


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
        # ^GSPC has no dividend adjustment: close_adj == close_raw == Close,
        # high_raw == High.
        assert by_date.loc[day, "close_raw"] == pytest.approx(_GSPC_CLOSES[i])
        assert by_date.loc[day, "close_adj"] == by_date.loc[day, "close_raw"]
        assert by_date.loc[day, "high_raw"] == pytest.approx(_GSPC_CLOSES[i] + 5.0)


def test_missing_required_regime_ticker_refuses_source_not_found(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)  # SPY present, ^GSPC absent
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=(CALENDAR_TICKER, REGIME_TICKER))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    assert exc.value.problem.details == {"tickers": [REGIME_TICKER]}
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()
    assert head["snapshot_id"] == base.snapshot_id  # nothing was committed


def test_malformed_required_regime_ticker_file_fails_like_a_malformed_px_csv(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_sources(source_root, with_spy=True)
    gspc_dir = source_root / "earnings_predictions" / "data" / "raw" / "polygon"
    gspc_dir.mkdir(parents=True, exist_ok=True)
    (gspc_dir / "gspc_daily.csv").write_bytes(b"not,a,price\nfile\n")
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock,
                required_tickers=(REGIME_TICKER,))
    assert exc.value.problem.code == "SOURCE_NOT_FOUND"
    head = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
                        ).fetchone()
    assert head["snapshot_id"] == base.snapshot_id

    # Not required: the malformed file fails only its own ticker's capture,
    # exactly as a malformed px csv does, and the run still commits the rest.
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert [(f["ticker"], f["error"]) for f in report["failures"]] == \
        [(REGIME_TICKER, "EmptyDataError")]
    assert report["failed_tickers"] == 1
    assert "result_snapshot_id" in report
