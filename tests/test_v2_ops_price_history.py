"""``price_history`` capture -- the Tier-2, catalog-backed ops-layer store
(SEND-BACK 2026-09-14 rework of the original private-registry design).

A synthetic ``securities`` table stands in for "the rest of the shadow
root": :func:`capture` must carry its dataset version forward UNCHANGED
alongside a freshly-advanced ``price_history`` one, in the SAME committed
snapshot, reusing catalog rows rather than re-scanning anything (SEND-BACK
requirement 1's central question). Real sqlite catalog + ``ArtifactStore``,
synthetic px/Tier-1 fixtures under a throwaway ``source_root`` -- no network,
no legacy writes.
"""
from __future__ import annotations

import gzip
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from engine.v2.data import reference_catalog
from engine.v2.data.errors import DataError
from engine.v2.data.legacy_materialization import px_series_tickers
from engine.v2.data.price_history_table import PRICE_HISTORY_TABLE_NAME
from engine.v2.data.reference_catalog import ReferenceInput
from engine.v2.data.repository import Repository
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.price_history_store import capture
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

_SEC = contract_for("securities")
_SEC_REF = contract_ref_for(_SEC)

#: A test-only maximal cutoff for direct ``materialize_price_series`` calls
#: below (these tests exercise that function on its own, not through a
#: planned ``LegacyMaterializationRequest``, so there is no job cutoff to
#: thread through) -- NOT the deleted production
#: ``PRICE_SERIES_MATERIALIZATION_CEILING`` constant (SEND-BACK 2026-09-14
#: item 2: materialization is no longer maximal-ceiling by default; the real
#: per-job cutoff now lives on ``LegacyMaterializationRequest.
#: observation_ceiling``, see ``tests/test_v2_data_legacy_materialization.py``).
_FAR_FUTURE_CEILING = "9999-12-31T23:59:59.000000Z"


def _securities_row(ticker: str, year: int) -> dict:
    return dict(ticker=ticker, year=year, first_date=None, last_date=None, mcap_usd=1.5e9,
               mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
               n_obs=250, src="orats")


def _base_snapshot(tmp_path, *, scope="shadow", with_reference_inputs=True):
    """A committed snapshot whose only table is ``securities`` -- "the rest
    of the shadow root" a price_history capture must extend, not re-scan."""
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(store, _SEC, _SEC_REF, [_securities_row("AAPL", 2024)], "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC}, scope=scope,
                         receipt_id="base-r1")
    if with_reference_inputs:
        ref = ReferenceInput(kind="calendar", legacy_path="calendar.csv", object_id="art_cal",
                             content_hash="sha256:" + "cd" * 32, byte_size=5)
        with transaction(conn):
            reference_catalog.insert_reference_inputs(conn, "base-r1", [ref])
    return conn, clock, store, snap


# --------------------------------------------------------------------------
# fixtures: legacy px csv tree / Tier-1 fetch cache, under a throwaway source_root
# --------------------------------------------------------------------------


def _px_dir(source_root: Path) -> Path:
    d = Path(source_root) / "earnings_predictions" / "data" / "raw" / "yfinance"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_px(source_root: Path, ticker: str, rows: dict) -> Path:
    """rows: {date: close_adj}."""
    path = _px_dir(source_root) / f"px_{ticker}.csv"
    frame = pd.DataFrame([{"date": d, "close_adj": v, "close_raw": v, "high_raw": v}
                          for d, v in sorted(rows.items())])
    frame.to_csv(path, index=False)
    return path


def _epoch(iso_timestamp: str) -> float:
    return datetime.fromisoformat(iso_timestamp).timestamp()


def _yfinance_csv_bytes(rows: dict) -> bytes:
    lines = ["Date,Open,High,Low,Close,Volume"]
    for d, close in sorted(rows.items()):
        lines.append(f"{d},{close},{close},{close},{close},1000")
    return ("\n".join(lines) + "\n").encode()


def _write_tier1(source_root: Path, ticker: str, rows: dict, *, key: str,
                 fetched_at: str = "2024-01-04T00:00:00+00:00") -> None:
    directory = Path(source_root) / "data" / "raw" / "fetch" / "yfinance" / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    body = _yfinance_csv_bytes(rows)
    (directory / f"{key}.body.gz").write_bytes(gzip.compress(body, mtime=0))
    (directory / f"{key}.meta.json").write_text(json.dumps({
        "source": "yfinance", "endpoint": "history", "key": key,
        "params": {"ticker": ticker, "period": "max"},
        "fetched_at": fetched_at, "status": 200}))


# --------------------------------------------------------------------------
# capture -> commit a new generation, reusing "securities" unchanged
# --------------------------------------------------------------------------


def test_capture_first_run_adds_rows_and_commits_a_new_generation_reusing_other_tables(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0, "2024-01-02": 101.0})

    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["by_outcome"] == {"added": 1}
    assert report["rows_added"] == 2
    assert "receipt_id" in report and "result_snapshot_id" in report

    repository = Repository(conn, store)
    new_snapshot = repository.resolve(report["result_snapshot_id"])
    assert PRICE_HISTORY_TABLE_NAME in new_snapshot.table_versions
    # "securities" carried forward UNCHANGED -- same dataset_version_id as
    # the base snapshot, reused rather than re-derived (SEND-BACK requirement
    # 1's reuse-without-re-scan question).
    assert (new_snapshot.table_versions["securities"].dataset_version_id
           == base.table_versions["securities"].dataset_version_id)
    assert new_snapshot.snapshot_id != base.snapshot_id


def test_capture_changed_value_advances_price_history_without_touching_old_snapshot(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0})
    first = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)

    _write_px(source_root, "AAPL", {"2024-01-01": 105.0})  # value changed
    second = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert second["by_outcome"] == {"added": 1}

    repository = Repository(conn, store)
    first_snapshot = repository.resolve(first["result_snapshot_id"])
    second_snapshot = repository.resolve(second["result_snapshot_id"])
    assert (first_snapshot.table_versions[PRICE_HISTORY_TABLE_NAME].dataset_version_id
           != second_snapshot.table_versions[PRICE_HISTORY_TABLE_NAME].dataset_version_id)
    # The FIRST snapshot still resolves exactly as it did -- a later capture
    # never rewrites what a past scoring saw.
    still_resolves = repository.resolve(first["result_snapshot_id"])
    assert still_resolves == first_snapshot


def test_capture_duplicate_source_hash_is_a_no_op(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0})
    capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["by_outcome"] == {"duplicate_source_hash": 1}
    assert report["rows_added"] == 0


def test_capture_anti_backdating_refuses_when_source_timestamp_is_not_later(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_tier1(source_root, "AAPL", {"2024-01-01": 100.0}, key="k1",
                fetched_at="2024-06-05T00:00:00+00:00")
    capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    # A second, DIFFERENT retrieval (different bytes -> different source_hash)
    # whose fetched_at is EARLIER than what was already captured.
    _write_tier1(source_root, "AAPL", {"2024-01-01": 999.0}, key="k2",
                fetched_at="2024-01-01T00:00:00+00:00")
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    # Every entry under source_root is re-scanned each run: k1 (already
    # captured) comes back as a no-op, and the new, EARLIER k2 is refused.
    assert report["by_outcome"] == {"error": 1, "duplicate_source_hash": 1}


def test_capture_merges_px_and_every_tier1_retrieval_for_the_same_ticker(tmp_path):
    """SEND-BACK 2026-09-14 item 1: a ticker with a px file plus the undated
    Tier-1 entry plus two dated re-downloads captures FOUR retrievals, in
    ascending retrieved_at order across BOTH sources -- px precedence is a
    READ rule (never applied at capture time; see the module docstring), so
    a px file no longer excludes any Tier-1 retrieval. A rerun is a no-op.
    """
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_tier1(source_root, "AAPL", {"2024-01-01": 100.0}, key="k1",
                fetched_at="2024-01-01T00:00:00+00:00")
    px_path = _write_px(source_root, "AAPL", {"2024-01-01": 100.5})
    os.utime(px_path, (_epoch("2024-02-01T00:00:00+00:00"),) * 2)
    _write_tier1(source_root, "AAPL", {"2024-01-01": 101.0}, key="k2",
                fetched_at="2024-03-01T00:00:00+00:00")
    _write_tier1(source_root, "AAPL", {"2024-01-01": 101.5}, key="k3",
                fetched_at="2024-04-01T00:00:00+00:00")

    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["by_outcome"] == {"added": 4}

    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])
    from engine.v2.contracts import DataQuery, KeyPredicate
    dvr = snapshot.table_versions[PRICE_HISTORY_TABLE_NAME]
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=dvr.table_contract_ref,
                      columns=("close_adj", "retrieved_at", "source_kind"),
                      key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAPL",)),),
                      order_by=("ticker", "date", "retrieved_at"), max_batch_rows=100, max_result_rows=100)
    rows = [r for batch in repository.scan(query, table_name=PRICE_HISTORY_TABLE_NAME)
           for r in batch.to_pylist()]
    assert [r["retrieved_at"][:10] for r in rows] == \
        ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
    assert [r["source_kind"] for r in rows] == \
        ["tier1_fetch", "legacy_px_csv", "tier1_fetch", "tier1_fetch"]
    assert [r["close_adj"] for r in rows] == [100.0, 100.5, 101.0, 101.5]

    rerun = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert rerun["by_outcome"] == {"duplicate_source_hash": 4}
    assert rerun["rows_added"] == 0


def test_capture_dry_run_writes_nothing_but_reports_true_counts(tmp_path):
    conn, clock, store, base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", dry_run=True, clock=clock)
    assert report["dry_run"] is True
    assert report["by_outcome"] == {"added": 1}
    assert "receipt_id" not in report
    row = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'").fetchone()
    assert row["snapshot_id"] == base.snapshot_id  # head unmoved
    assert conn.execute("SELECT COUNT(*) FROM data_price_captures").fetchone()[0] == 0


def test_capture_refuses_when_scope_has_no_existing_head(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0})
    with pytest.raises(OpsError) as exc:
        capture(conn, store, source_root, root=tmp_path, scope="nonexistent", clock=clock)
    assert exc.value.code == "SNAPSHOT_NOT_READY"


# --------------------------------------------------------------------------
# multi-retrieval capture (coordinator addition, 2026-09-14)
# --------------------------------------------------------------------------


def test_capture_multiple_dated_tier1_retrievals_captured_in_fetched_at_order_when_shuffled(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    # Presented to the filesystem/glob in SHUFFLED order (key names sort k3 < ka < kb,
    # unrelated to fetched_at order) -- capture must still apply them oldest-first.
    _write_tier1(source_root, "AAPL", {"2024-01-01": 100.0}, key="kb",
                fetched_at="2024-03-01T00:00:00+00:00")
    _write_tier1(source_root, "AAPL", {"2024-01-01": 101.0}, key="k3",
                fetched_at="2024-01-01T00:00:00+00:00")
    _write_tier1(source_root, "AAPL", {"2024-01-01": 102.0}, key="ka",
                fetched_at="2024-02-01T00:00:00+00:00")
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["by_outcome"] == {"added": 3}

    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])
    from engine.v2.contracts import DataQuery, KeyPredicate
    dvr = snapshot.table_versions[PRICE_HISTORY_TABLE_NAME]
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=dvr.table_contract_ref,
                      columns=("close_adj", "retrieved_at"),
                      key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAPL",)),),
                      order_by=("ticker", "date", "retrieved_at"), max_batch_rows=100, max_result_rows=100)
    rows = [r for batch in repository.scan(query, table_name=PRICE_HISTORY_TABLE_NAME)
           for r in batch.to_pylist()]
    # Three versions of the SAME date, in ascending retrieved_at (fetched_at) order.
    assert [r["retrieved_at"][:10] for r in rows] == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert [r["close_adj"] for r in rows] == [101.0, 102.0, 100.0]


def test_capture_rerun_after_multiple_retrievals_is_a_no_op(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_tier1(source_root, "AAPL", {"2024-01-01": 100.0}, key="k1",
                fetched_at="2024-01-01T00:00:00+00:00")
    _write_tier1(source_root, "AAPL", {"2024-01-01": 101.0}, key="k2",
                fetched_at="2024-02-01T00:00:00+00:00")
    first = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert first["by_outcome"] == {"added": 2}
    second = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert second["by_outcome"] == {"duplicate_source_hash": 2}
    assert second["rows_added"] == 0


# --------------------------------------------------------------------------
# reference inputs copied forward into the new receipt (SEND-BACK requirement 1)
# --------------------------------------------------------------------------


def test_reference_inputs_copied_forward_to_the_new_receipt(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    inputs = reference_catalog.reference_inputs_for_receipt(conn, receipt_id=report["receipt_id"])
    assert [i.kind for i in inputs] == ["calendar"]
    assert inputs[0].legacy_path == "calendar.csv"


def test_pin_snapshot_inputs_can_resolve_the_new_generation(tmp_path):
    """The exact contract ``snapshot_planning.pin_snapshot_inputs`` needs
    (task brief 2026-09-14, second message): a committed receipt for the new
    head snapshot, with reference inputs recorded under it."""
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    found = reference_catalog.committed_receipt_for_snapshot(
        conn, scope="shadow", snapshot_id=report["result_snapshot_id"])
    assert found == report["receipt_id"]


# --------------------------------------------------------------------------
# wiring: legacy_score snapshot materialization writes px_<T>.csv (requirement 3)
# --------------------------------------------------------------------------


@pytest.mark.skip(reason="full legacy_adapter.materialize fixture (every "
                         "LEGACY_SCORE_READ_PLAN_V1 table + pinned legacy SNAPSHOT/registry "
                         "refs) belongs to tests/test_v2_data_legacy_materialization.py's own "
                         "existing end-to-end fixture; this file only proves get_price_series/"
                         "materialize_price_series's own contract (see "
                         "test_v2_data_price_downloads.py's get_price_series tests and "
                         "test_materialize_price_series_writes_px_files below).")
def test_materialized_tree_has_px_files_and_the_scorer_reader_loads_them(tmp_path):
    pass


def test_px_series_tickers_is_direct_union_evidence(tmp_path):
    class _Req:
        direct_scope = {"tickers": ["AAPL", "SPY"]}
        evidence_scope = {"tickers": ["AAPL", "MSFT"]}
    assert px_series_tickers(_Req()) == ("AAPL", "MSFT", "SPY")


def test_materialize_price_series_writes_px_files_and_parse_equality_holds(tmp_path):
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0, "2024-01-02": 101.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])

    dest = tmp_path / "materialized"
    written = materialize_price_series(repository, snapshot, dest, tickers=("AAPL",),
                                       observation_ceiling=_FAR_FUTURE_CEILING)
    assert written["AAPL"].is_file()
    from engine.v2.data import price_download_sources
    readback = price_download_sources.read_legacy_px_csv(written["AAPL"])
    assert list(readback["close_adj"]) == [100.0, 101.0]


def test_materialize_price_series_refuses_typed_when_a_ticker_has_no_history(tmp_path):
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])
    with pytest.raises(DataError) as exc:
        materialize_price_series(repository, snapshot, tmp_path / "out", tickers=("ZZZZ",),
                                 observation_ceiling=_FAR_FUTURE_CEILING)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_appending_a_changed_capture_after_pinning_leaves_materialized_output_unchanged(tmp_path):
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0})
    pinned = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    pinned_snapshot = repository.resolve(pinned["result_snapshot_id"])
    dest = tmp_path / "materialized"
    materialize_price_series(repository, pinned_snapshot, dest, tickers=("AAPL",),
                             observation_ceiling=_FAR_FUTURE_CEILING)
    from engine.v2.data import price_download_sources
    before = price_download_sources.read_legacy_px_csv(dest / "earnings_predictions" / "data" /
                                                       "raw" / "yfinance" / "px_AAPL.csv")

    # A later, changed capture advances the HEAD -- the pinned snapshot must
    # still resolve and materialize identically.
    _write_px(source_root, "AAPL", {"2024-01-01": 999.0})
    capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)

    repository2 = Repository(conn, store)
    still_pinned = repository2.resolve(pinned["result_snapshot_id"])
    dest2 = tmp_path / "materialized2"
    materialize_price_series(repository2, still_pinned, dest2, tickers=("AAPL",),
                             observation_ceiling=_FAR_FUTURE_CEILING)
    after = price_download_sources.read_legacy_px_csv(dest2 / "earnings_predictions" / "data" /
                                                      "raw" / "yfinance" / "px_AAPL.csv")
    assert list(before["close_adj"]) == list(after["close_adj"]) == [100.0]


# --------------------------------------------------------------------------
# materialization is point-in-time to the JOB's own cutoff (SEND-BACK
# 2026-09-14 item 2) -- not a maximal ceiling that sees every retrieval ever
# captured, even when a later one sits in the SAME pinned snapshot version.
# --------------------------------------------------------------------------


def test_materialize_price_series_respects_the_jobs_own_cutoff_not_a_maximal_one(tmp_path):
    """Retrievals A (fetched 2024-01-01, before a job's cutoff) and B
    (fetched 2024-06-01, after it) both land in the SAME pinned snapshot (one
    capture run sees both Tier-1 entries at once). A job whose own cutoff
    sits between them must materialize A's value; a job with a later cutoff
    -- still the SAME pinned snapshot, never a re-resolve -- gets B's."""
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_tier1(source_root, "AAPL", {"2024-01-01": 100.0}, key="a",
                fetched_at="2024-01-01T00:00:00+00:00")
    _write_tier1(source_root, "AAPL", {"2024-01-01": 200.0}, key="b",
                fetched_at="2024-06-01T00:00:00+00:00")
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["by_outcome"] == {"added": 2}
    repository = Repository(conn, store)
    pinned_snapshot = repository.resolve(report["result_snapshot_id"])
    from engine.v2.data import price_download_sources

    between_a_and_b = materialize_price_series(
        repository, pinned_snapshot, tmp_path / "between", tickers=("AAPL",),
        observation_ceiling="2024-03-01T00:00:00Z")
    before = price_download_sources.read_legacy_px_csv(
        between_a_and_b["AAPL"])
    assert list(before["close_adj"]) == [100.0]

    after_b = materialize_price_series(
        repository, pinned_snapshot, tmp_path / "after", tickers=("AAPL",),
        observation_ceiling="2024-12-31T23:59:59Z")
    later = price_download_sources.read_legacy_px_csv(after_b["AAPL"])
    assert list(later["close_adj"]) == [200.0]
