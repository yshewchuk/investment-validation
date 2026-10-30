"""Price version identity digests, verification and bounded per-ticker memory."""
import tracemalloc
from datetime import date, timedelta

import numpy as np
import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash
from engine.v2.ops import native_registration
from engine.v2.research import _magnitude_panel_sources as sources
from engine.v2.research import magnitude_panel_qualified as builder


def _rows(n, ticker, bump=None):
    rows = [{"date": (date(2000, 1, 3) + timedelta(days=i)).isoformat(),
             "retrieved_at": "2026-10-01T00:00:00+00:00", "deleted": False,
             "source_hash": f"sha256:{ticker}{i}", "capture_id": f"cap{i}"}
            for i in range(n)]
    if isinstance(bump, int):
        rows[bump]["source_hash"] = "sha256:revised"
    return rows


def _patch(monkeypatch, rows_by_ticker):
    monkeypatch.setattr(sources, "price_dates",
        lambda repo, snap, ticker, limits: rows_by_ticker[ticker])


def test_digests_equal_for_equal_inputs(monkeypatch):
    rows = {"A": _rows(5, "A"), "B": _rows(3, "B"), "^GSPC": _rows(7, "^GSPC")}
    _patch(monkeypatch, rows)
    first = builder._price_inputs(None, None, ["A", "B"], {})
    second = builder._price_inputs(None, None, ["A", "B"], {})
    assert first[0] == second[0]
    digests, dates = first
    assert set(digests) == {"A", "B", "^GSPC"}
    for ticker in ("A", "B", "^GSPC"):
        assert digests[ticker] == {"rows": len(rows[ticker]),
                                   "digest": content_hash(rows[ticker])}
    assert isinstance(dates["A"], np.ndarray)
    assert dates["A"].dtype == np.dtype("datetime64[ns]")
    assert len(dates["A"]) == 5


def test_digest_changes_when_one_version_row_changes(monkeypatch):
    rows = {"A": _rows(5, "A"), "B": _rows(4, "B"), "^GSPC": _rows(6, "^GSPC")}
    _patch(monkeypatch, rows)
    baseline, _ = builder._price_inputs(None, None, ["A", "B"], {})
    revised = {"A": _rows(5, "A", bump=3), "B": rows["B"], "^GSPC": rows["^GSPC"]}
    _patch(monkeypatch, revised)
    changed, _ = builder._price_inputs(None, None, ["A", "B"], {})
    assert changed["A"]["digest"] != baseline["A"]["digest"]
    assert changed["A"]["rows"] == baseline["A"]["rows"] == 5
    assert changed["B"] == baseline["B"]
    assert changed["^GSPC"] == baseline["^GSPC"]


def test_versions_verify_digest(monkeypatch):
    rows = {"A": _rows(4, "A"), "^GSPC": _rows(6, "^GSPC")}
    _patch(monkeypatch, rows)
    digests, _ = builder._price_inputs(None, None, ["A"], {})
    context = {"request": {"limits": {}}, "price_row_digests": digests}
    versions = builder._versions(None, None, context, "A")
    assert set(versions) == {row["date"] for row in rows["A"]}
    assert all(versions[row["date"]] == row for row in rows["A"])
    _patch(monkeypatch, {"A": _rows(4, "A", bump=1), "^GSPC": rows["^GSPC"]})
    with pytest.raises(DataError) as raised:
        builder._versions(None, None, context, "A")
    assert raised.value.code == "IDENTITY_CONFLICT"
    assert "IDENTITY_CONFLICT" in str(raised.value)
    _patch(monkeypatch, {"A": _rows(3, "A"), "^GSPC": rows["^GSPC"]})
    with pytest.raises(DataError) as raised:
        builder._versions(None, None, context, "A")
    assert raised.value.code == "IDENTITY_CONFLICT"


def test_identity_peak_memory_independent_of_total_rows(monkeypatch):
    ROWS = 1500

    def price_dates(repository, snapshot, ticker, limits):
        return _rows(ROWS, ticker)

    monkeypatch.setattr(sources, "price_dates", price_dates)

    def peak(tickers):
        tracemalloc.start()
        try:
            builder._price_inputs(None, None, [f"T{i}" for i in range(tickers)], {})
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    peak_10 = peak(10)
    peak_120 = peak(120)
    # The v3 inline metadata held ~1.3 KB of Python objects per row, i.e.
    # >200 MB at this scale; identity must stay bounded per ticker.
    assert peak_120 < 3 * peak_10
    assert peak_120 < 8 * 1024 * 1024


def test_registration_accepts_v3_and_v4_contexts():
    assert native_registration._SIZE_PANEL_CONTEXTS == (
        "magnitude_panel_context.v3", "magnitude_panel_context.v4")
