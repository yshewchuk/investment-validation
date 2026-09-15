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
import random
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import DATA_FAILURE_CODES
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


def _yfinance_csv_bytes(rows: dict, *, adj_close: dict | None = None) -> bytes:
    """rows: {date: close} used for Open/High/Low/Close/high_raw (and
    close_raw, since Close is unadjusted). ``adj_close`` (default: same as
    ``rows``) sets ``Adj Close`` -- the dividend-adjusted series
    ``read_tier1_body`` maps to ``close_adj`` -- per date, for tests that need
    it to differ from ``Close``.
    """
    adj_close = adj_close if adj_close is not None else rows
    lines = ["Date,Open,High,Low,Close,Adj Close,Volume"]
    for d, close in sorted(rows.items()):
        lines.append(f"{d},{close},{close},{close},{close},{adj_close[d]},1000")
    return ("\n".join(lines) + "\n").encode()


def _write_tier1(source_root: Path, ticker: str, rows: dict, *, key: str,
                 fetched_at: str = "2024-01-04T00:00:00+00:00",
                 adj_close: dict | None = None) -> None:
    directory = Path(source_root) / "data" / "raw" / "fetch" / "yfinance" / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    body = _yfinance_csv_bytes(rows, adj_close=adj_close)
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
# Tier-1 parsing fix (2026-09-15): close_raw/high_raw are real values now,
# not forced NaN, and diff_retrieval only stores what actually changed.
# --------------------------------------------------------------------------


def test_capture_stores_exactly_the_changed_close_adj_row_when_raw_and_high_match(tmp_path):
    """px carries close_raw/high_raw natively; a later Tier-1 retrieval whose
    Close/High match px's close_raw/high_raw on every date but whose
    Adj Close differs on exactly one (a genuine restatement, the FDS shape
    from the verified defect) must diff to exactly ONE new row -- proof that
    close_raw/high_raw no longer force every Tier-1 row to look "changed"
    the way the pre-fix all-NaN parse did.
    """
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    px_path = _write_px(source_root, "AAPL",
                        {"2024-01-01": 100.0, "2024-01-02": 101.0, "2024-01-03": 102.0})
    os.utime(px_path, (_epoch("2024-01-01T00:00:00+00:00"),) * 2)
    first = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert first["by_outcome"] == {"added": 1}
    assert first["rows_added"] == 3

    _write_tier1(source_root, "AAPL",
                {"2024-01-01": 100.0, "2024-01-02": 101.0, "2024-01-03": 102.0}, key="k1",
                fetched_at="2024-02-01T00:00:00+00:00",
                adj_close={"2024-01-01": 100.0, "2024-01-02": 105.5, "2024-01-03": 102.0})
    second = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    # The unchanged px entry re-scans as a no-op; the tier1 entry contributes
    # exactly the one row whose close_adj actually changed.
    assert second["by_outcome"] == {"added": 1, "duplicate_source_hash": 1}
    assert second["rows_added"] == 1

    repository = Repository(conn, store)
    from engine.v2.contracts import DataQuery, KeyPredicate
    snapshot = repository.resolve(second["result_snapshot_id"])
    dvr = snapshot.table_versions[PRICE_HISTORY_TABLE_NAME]
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=dvr.table_contract_ref,
                      columns=("date", "close_adj", "retrieved_at"),
                      key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAPL",)),),
                      order_by=("ticker", "date", "retrieved_at"), max_batch_rows=100, max_result_rows=100)
    rows = [r for batch in repository.scan(query, table_name=PRICE_HISTORY_TABLE_NAME)
           for r in batch.to_pylist()]
    changed = [r for r in rows if r["retrieved_at"][:10] == "2024-02-01"]
    assert len(changed) == 1
    assert changed[0]["date"] == "2024-01-02"
    assert changed[0]["close_adj"] == 105.5


def test_overlap_report_counts_per_column_and_max_relative_diff(tmp_path):
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    px_frame = pd.DataFrame([
        {"date": "2024-01-01", "close_adj": 100.0, "close_raw": 100.0, "high_raw": 100.5},
        {"date": "2024-01-02", "close_adj": 101.0, "close_raw": 101.0, "high_raw": 101.5},
        {"date": "2024-01-03", "close_adj": 102.0, "close_raw": 102.0, "high_raw": 102.5},
    ])
    px_frame.to_csv(_px_dir(source_root) / "px_AAPL.csv", index=False)

    # date 1: matches on every column. date 2: close_raw (Close) mismatches
    # only. date 3: high_raw (High) AND close_adj (Adj Close) mismatch.
    lines = ["Date,Open,High,Low,Close,Adj Close,Volume",
            "2024-01-01,100.0,100.5,100.0,100.0,100.0,1000",
            "2024-01-02,105.0,101.5,101.0,105.0,101.0,1000",
            "2024-01-03,102.0,110.0,102.0,102.0,99.0,1000"]
    body = ("\n".join(lines) + "\n").encode()
    key = "kov1"
    directory = Path(source_root) / "data" / "raw" / "fetch" / "yfinance" / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{key}.body.gz").write_bytes(gzip.compress(body, mtime=0))
    (directory / f"{key}.meta.json").write_text(json.dumps({
        "source": "yfinance", "endpoint": "history", "key": key,
        "params": {"ticker": "AAPL", "period": "max"},
        "fetched_at": "2024-02-01T00:00:00+00:00", "status": 200}))

    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", dry_run=True, clock=clock)
    overlap = report["overlap_by_column"]
    assert report["overlapping_tickers"] == 1
    assert overlap["close_raw"] == {"differing_dates": 1, "max_relative_diff": pytest.approx(4 / 101.0)}
    assert overlap["high_raw"] == {"differing_dates": 1, "max_relative_diff": pytest.approx(7.5 / 102.5)}
    assert overlap["close_adj"] == {"differing_dates": 1, "max_relative_diff": pytest.approx(3 / 102.0)}


def test_capture_contract_bump_recaptures_cleanly_and_old_snapshot_is_unchanged(tmp_path):
    """Stands in for the real 2026-09-15 transition: a ``price_history.v1``-
    pinned snapshot already holds a wrongly-parsed row (close_raw/high_raw
    NaN, the exact pre-fix shape) for AAPL. ``capture()`` always targets the
    CURRENT ``PRICE_HISTORY_CONTRACT`` (now ``price_history.v2``), so it must
    recapture AAPL fresh under the new contract -- never refused as
    ``duplicate_source_hash`` even though the retrieval's raw bytes (and so
    ``source_hash``) could otherwise collide with something already recorded
    -- while the OLD snapshot keeps resolving its own, untouched
    ``price_history.v1`` dataset version.
    """
    import dataclasses

    from engine.v2.contracts import DataQuery, KeyPredicate
    from engine.v2.data import manifests as manifests_mod
    from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT

    old_base = dataclasses.replace(PRICE_HISTORY_CONTRACT, contract_id="price_history.v1",
                                   definition_hash="sha256:" + "0" * 64)
    old_contract = dataclasses.replace(
        old_base, definition_hash=manifests_mod.table_contract_hash(old_base))
    old_ref = contract_ref_for(old_contract)
    assert old_contract.contract_id != PRICE_HISTORY_CONTRACT.contract_id

    conn, clock, store = catalog_and_store(tmp_path)
    sec_record = publish_and_inspect(store, _SEC, _SEC_REF, [_securities_row("AAPL", 2024)], "2024")
    wrong_row = dict(ticker="AAPL", date="2024-01-01", close_adj=999.0, close_raw=None,
                     high_raw=None, retrieved_at="2024-01-01T00:00:00Z", deleted=False,
                     source_kind="tier1_fetch", source_hash="pre-fix-hash", capture_id="pre-fix-cap")
    ph_record = publish_and_inspect(store, old_contract, old_ref, [wrong_row], "AAPL")
    base = commit_tables(conn, clock, {"securities": [sec_record], PRICE_HISTORY_TABLE_NAME: [ph_record]},
                         {"securities": _SEC, PRICE_HISTORY_TABLE_NAME: old_contract}, scope="shadow",
                         receipt_id="base-r1")
    ref = ReferenceInput(kind="calendar", legacy_path="calendar.csv", object_id="art_cal",
                         content_hash="sha256:" + "cd" * 32, byte_size=5)
    with transaction(conn):
        reference_catalog.insert_reference_inputs(conn, "base-r1", [ref])
    assert (base.table_versions[PRICE_HISTORY_TABLE_NAME].table_contract_ref.contract_id
           == "price_history.v1")

    source_root = tmp_path / "legacy"
    _write_tier1(source_root, "AAPL", {"2024-01-01": 100.0}, key="k1",
                fetched_at="2024-02-01T00:00:00+00:00", adj_close={"2024-01-01": 99.5})

    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    # Never duplicate_source_hash: the new contract's own chain started empty.
    assert report["by_outcome"] == {"added": 1}

    repository = Repository(conn, store)
    new_snapshot = repository.resolve(report["result_snapshot_id"])
    new_dvr = new_snapshot.table_versions[PRICE_HISTORY_TABLE_NAME]
    assert new_dvr.table_contract_ref.contract_id == PRICE_HISTORY_CONTRACT.contract_id

    query = DataQuery(snapshot_id=new_snapshot.snapshot_id, table_contract_ref=new_dvr.table_contract_ref,
                      columns=("close_adj", "close_raw", "high_raw"),
                      key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAPL",)),),
                      order_by=("ticker", "date", "retrieved_at"), max_batch_rows=100, max_result_rows=100)
    rows = [r for batch in repository.scan(query, table_name=PRICE_HISTORY_TABLE_NAME)
           for r in batch.to_pylist()]
    assert rows == [{"close_adj": 99.5, "close_raw": 100.0, "high_raw": 100.0}]

    # The OLD snapshot is untouched: still price_history.v1, same wrong values.
    old_snapshot = repository.resolve(base.snapshot_id)
    old_dvr = old_snapshot.table_versions[PRICE_HISTORY_TABLE_NAME]
    assert old_dvr.table_contract_ref.contract_id == "price_history.v1"
    assert old_dvr == base.table_versions[PRICE_HISTORY_TABLE_NAME]


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


class _CountingRepository:
    """Wraps a real ``Repository``, counting ``.scan()`` and
    ``.fragment_records()`` calls separately -- proves
    :func:`tickers_with_price_history` (task brief 2026-09-15 send-back)
    answers existence from catalog metadata (``fragment_records``) ONCE per
    pinned table version, for however many tickers are asked about
    together, and never falls back to a ``Repository.scan`` of actual row
    content (the first implementation of this primitive did exactly that
    and hit ``RESULT_LIMIT_EXCEEDED`` against real data -- see the
    function's own docstring). Delegates every other attribute to the
    wrapped real repository."""

    def __init__(self, inner):
        self._inner = inner
        self.scan_calls = 0
        self.fragment_records_calls = 0

    def scan(self, *args, **kwargs):
        self.scan_calls += 1
        return self._inner.scan(*args, **kwargs)

    def fragment_records(self, *args, **kwargs):
        self.fragment_records_calls += 1
        return self._inner.fragment_records(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_tickers_with_price_history_is_one_metadata_lookup_for_many_tickers(tmp_path):
    """task brief 2026-09-15 send-back: existence is answered from ONE
    ``fragment_records`` catalog lookup for N tickers together (never a
    per-ticker loop, and never a ``Repository.scan`` of row content at
    all), and a repeat call for the SAME pinned table version -- any ticker
    subset -- is a cache hit (no additional lookup)."""
    from engine.v2.data import price_history_query
    from engine.v2.data.price_history_query import tickers_with_price_history

    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    present_tickers = ["AAA", "BBB", "CCC"]
    for ticker in present_tickers:
        _write_px(source_root, ticker, {"2024-01-01": 100.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    real_repository = Repository(conn, store)
    snapshot = real_repository.resolve(report["result_snapshot_id"])

    # A fresh cache state: the in-process LRU is module-global, so a prior
    # test's entry must never make this assertion pass for the wrong reason.
    price_history_query._EXISTENCE_CACHE.clear()

    wrapped = _CountingRepository(real_repository)
    requested = tuple(present_tickers) + ("YYY", "ZZZ")  # two genuinely absent tickers too
    present = tickers_with_price_history(wrapped, snapshot, requested)
    assert present == frozenset(present_tickers)
    assert wrapped.fragment_records_calls == 1
    assert wrapped.scan_calls == 0  # never falls back to a row-content scan

    # Same pinned table version, a DIFFERENT ticker subset: still a cache hit.
    again = tickers_with_price_history(wrapped, snapshot, ("AAA", "YYY"))
    assert again == frozenset({"AAA"})
    assert wrapped.fragment_records_calls == 1
    assert wrapped.scan_calls == 0

    # has_price_history (kept as a thin single-ticker convenience) goes
    # through the same primitive/cache -- no additional lookup.
    assert price_history_query.has_price_history(wrapped, snapshot, "AAA") is True
    assert wrapped.fragment_records_calls == 1
    assert price_history_query.has_price_history(wrapped, snapshot, "YYY") is False
    assert wrapped.fragment_records_calls == 1


def test_materialize_price_series_writes_px_files_and_parse_equality_holds(tmp_path):
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0, "2024-01-02": 101.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])

    dest = tmp_path / "materialized"
    result = materialize_price_series(repository, snapshot, dest, tickers=("AAPL",),
                                      observation_ceiling=_FAR_FUTURE_CEILING)
    written = result.written
    assert result.px_absent == ()
    assert written["AAPL"].is_file()
    from engine.v2.data import price_download_sources
    readback = price_download_sources.read_legacy_px_csv(written["AAPL"])
    assert list(readback["close_adj"]) == [100.0, 101.0]


def test_materialize_price_series_writes_non_nan_columns_from_tier1_source(tmp_path):
    """A Tier-1-only ticker (no px file): close_raw/high_raw must be real,
    non-NaN values in the materialized px csv -- the 2026-09-15 fix; before
    it, every Tier-1-sourced ticker's materialized close_raw/high_raw were
    NaN (``read_tier1_body`` never parsed them)."""
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_tier1(source_root, "AAPL", {"2024-01-01": 100.0, "2024-01-02": 101.0}, key="k1",
                fetched_at="2024-01-01T00:00:00+00:00",
                adj_close={"2024-01-01": 99.0, "2024-01-02": 100.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])

    dest = tmp_path / "materialized"
    written = materialize_price_series(repository, snapshot, dest, tickers=("AAPL",),
                                       observation_ceiling=_FAR_FUTURE_CEILING).written
    from engine.v2.data import price_download_sources
    readback = price_download_sources.read_legacy_px_csv(written["AAPL"])
    assert list(readback["close_adj"]) == [99.0, 100.0]
    assert list(readback["close_raw"]) == [100.0, 101.0]
    assert list(readback["high_raw"]) == [100.0, 101.0]
    assert not readback["close_raw"].isna().any()
    assert not readback["high_raw"].isna().any()


def test_materialize_price_series_skips_and_records_a_ticker_with_no_history(tmp_path):
    """A ticker with NO price_history rows at all is skipped, not refused
    (task brief 2026-09-15: mirror legacy's absence semantics, panel.py
    446-560) -- no ``px_<T>.csv`` is written for it, it is named in
    ``px_absent``, and every ticker that DOES have history is still written
    and readback-verified normally."""
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0, "2024-01-02": 101.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])

    dest = tmp_path / "out"
    result = materialize_price_series(repository, snapshot, dest, tickers=("AAPL", "ZZZZ"),
                                      observation_ceiling=_FAR_FUTURE_CEILING)
    assert result.px_absent == ("ZZZZ",)
    assert "ZZZZ" not in result.written
    assert not (dest / "earnings_predictions" / "data" / "raw" / "yfinance" / "px_ZZZZ.csv").exists()
    assert result.written["AAPL"].is_file()
    from engine.v2.data import price_download_sources
    readback = price_download_sources.read_legacy_px_csv(result.written["AAPL"])
    assert list(readback["close_adj"]) == [100.0, 101.0]


def test_materialize_price_series_still_refuses_typed_for_a_readback_mismatch(tmp_path, monkeypatch):
    """Requirement 2: absence is the ONLY case that skips. A ticker that DOES
    have price_history rows but whose materialized file fails the readback
    round trip must still refuse ``CONTRACT_MISMATCH`` typed, exactly as
    before -- it must not be swallowed by the new skip-on-absence path."""
    from engine.v2.data import price_download_sources
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    _write_px(source_root, "AAPL", {"2024-01-01": 100.0, "2024-01-02": 101.0})
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])

    real_read = price_download_sources.read_legacy_px_csv

    def _corrupted_read(path):
        frame = real_read(path)
        frame = frame.copy()
        frame["close_adj"] = frame["close_adj"] + 1.0
        return frame

    monkeypatch.setattr(price_download_sources, "read_legacy_px_csv", _corrupted_read)
    with pytest.raises(DataError) as exc:
        materialize_price_series(repository, snapshot, tmp_path / "out", tickers=("AAPL",),
                                 observation_ceiling=_FAR_FUTURE_CEILING)
    assert exc.value.code == "CONTRACT_MISMATCH"


def _awkward_close_adj_values(n=300, seed=20260915):
    """Same construction as test_v2_data_price_downloads.py's
    ``_awkward_float64s`` -- doubles needing the full 17 significant digits
    for an exact round trip often enough to reproduce the real snapshot's
    behavior (143 of 144 real tickers hit at least one such row)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        bits = rng.getrandbits(64)
        x = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if x == x and 0.0001 < abs(x) < 100000:
            out.append(x)
    return out


def test_verify_price_readback_mismatch_refuses_typed_registered_code():
    """Regression for the 2026-09-15 defect: ``_verify_price_readback`` used
    to raise the unregistered literal ``VALIDATION_FAILED``, so
    ``errors.make_problem`` threw a bare, untyped ``ValueError`` the moment
    a real mismatch occurred (worker.stderr on att_7eaea005.../
    att_6059778f...) instead of the typed ``DataError`` every other refusal
    in this package raises. It must now come back as a registered code a
    caller can branch on."""
    from engine.v2.data.legacy_materialization import _verify_price_readback
    written = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "close_adj": [100.0, 101.0]})
    readback = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "close_adj": [100.0, 101.5]})
    with pytest.raises(DataError) as exc:
        _verify_price_readback("AAPL", written, readback)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.code in DATA_FAILURE_CODES


def test_materialize_price_series_round_trips_awkward_float_close_adj(tmp_path):
    """End-to-end through capture -> materialize_price_series ->
    ``_verify_price_readback`` with real awkward doubles as close_adj: must
    not raise (the 2026-09-15 defect made this fail on almost every real
    ticker in the shadow snapshot), and the materialized file must read
    back the exact values captured."""
    from engine.v2.data.legacy_materialization import materialize_price_series
    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    values = _awkward_close_adj_values()
    # one distinct calendar date per value (12 * 28 = 336 slots, 300 values: no collisions)
    by_date = {f"2024-{((i // 28) % 12) + 1:02d}-{(i % 28) + 1:02d}": v
              for i, v in enumerate(values)}
    _write_px(source_root, "AAPL", by_date)
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    repository = Repository(conn, store)
    snapshot = repository.resolve(report["result_snapshot_id"])

    dest = tmp_path / "materialized"
    written = materialize_price_series(repository, snapshot, dest, tickers=("AAPL",),
                                       observation_ceiling=_FAR_FUTURE_CEILING).written
    from engine.v2.data import price_download_sources
    readback = price_download_sources.read_legacy_px_csv(written["AAPL"])
    expected = [by_date[d] for d in sorted(by_date)]
    assert list(readback["close_adj"]) == expected


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
        observation_ceiling="2024-03-01T00:00:00Z").written
    before = price_download_sources.read_legacy_px_csv(
        between_a_and_b["AAPL"])
    assert list(before["close_adj"]) == [100.0]

    after_b = materialize_price_series(
        repository, pinned_snapshot, tmp_path / "after", tickers=("AAPL",),
        observation_ceiling="2024-12-31T23:59:59Z").written
    later = price_download_sources.read_legacy_px_csv(after_b["AAPL"])
    assert list(later["close_adj"]) == [200.0]


# --------------------------------------------------------------------------
# leakage guard: close_adj is only safe for SCALE-INVARIANT features within
# one retrieval's own view (price_history.py's module docstring, "Leakage
# invariant for close_adj", added 2026-09-15). A later retrieval that
# restates every earlier close by one constant factor k (exactly what a new
# dividend/split does to yfinance's Adj Close) must leave a ratio-of-closes
# feature computed from dates strictly BEFORE the restated boundary
# unchanged, because k cancels in the ratio -- real-data measured on FDS
# (the ticker the verified Tier-1 parsing defect flagged as a genuine
# restatement): 3.8e-3 relative close_adj difference, <=1.4e-5pp feature
# difference.
# --------------------------------------------------------------------------


def _runup_style_features(closes, idx: int) -> dict:
    """The exact arithmetic of ``engine.data.features.panel.add_runup_features``
    (panel.py:556-573) for one anchor index: ``dist_high``/``dist_ema`` need
    ``idx >= 252`` (rolling(252, min_periods=120).max() / ewm(span=252)), and
    ``ret5``/``ret10``/``ret20`` need ``idx >= 20``. Each is a ratio of two
    closes -- scale-invariant by construction -- computed directly rather
    than through ``add_runup_features`` itself, which needs a full legacy
    events frame and Tier-1 px-dir wiring this test has no reason to build.
    """
    series = pd.Series(closes)
    ema252 = series.ewm(span=252, adjust=False).mean().to_numpy()
    high252 = series.rolling(252, min_periods=120).max().to_numpy()
    out = {}
    if idx >= 252 and np.isfinite(high252[idx]) and np.isfinite(ema252[idx]) and ema252[idx] > 0:
        out["dist_high"] = (closes[idx] / high252[idx] - 1.0) * 100
        out["dist_ema"] = (closes[idx] / ema252[idx] - 1.0) * 100
    if idx >= 20 and closes[idx - 20] > 0:
        out["ret20"] = (closes[idx] / closes[idx - 20] - 1.0) * 100
        out["ret10"] = (closes[idx] / closes[idx - 10] - 1.0) * 100
        out["ret5"] = (closes[idx] / closes[idx - 5] - 1.0) * 100
    return out


def test_a_later_restatement_leaves_scale_invariant_runup_features_unchanged(tmp_path):
    """Retrieval 1 (px, retrieved T1): 300 synthetic old dates. Retrieval 2
    (Tier-1, retrieved T2 > T1): the SAME 300 dates with close_adj rescaled
    by k=0.99 (a later dividend/split restating the whole history it covers,
    same shape a real yfinance re-pull produces), plus 5 brand-new appended
    dates. Both retrievals land in ONE pinned snapshot (a real capture -> `
    materialize_price_series` round trip, never a bare DataFrame check).
    Materializing at a cutoff BEFORE T2 sees only the k=1 view; materializing
    at a cutoff AFTER T2 sees the k=0.99 view for the same old dates. A
    ratio-of-closes feature anchored on the LAST old date (whose entire
    252-day lookback lies inside the uniformly-rescaled old region) must
    come out identical between the two views, within 1e-9 relative -- proof
    that the rescale factor cancels exactly through the real storage/as-of
    path, not merely in the abstract arithmetic.
    """
    from engine.v2.data import price_download_sources
    from engine.v2.data.legacy_materialization import materialize_price_series

    n_old, n_new, k = 300, 5, 0.99
    rng = np.random.default_rng(20260915)
    old_closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, size=n_old))
    new_closes = old_closes[-1] * np.cumprod(1 + rng.normal(0, 0.01, size=n_new))
    old_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2023-01-02", periods=n_old)]
    new_dates = [d.strftime("%Y-%m-%d")
                for d in pd.date_range(old_dates[-1], periods=n_new + 1)[1:]]

    conn, clock, store, _base = _base_snapshot(tmp_path)
    source_root = tmp_path / "legacy"
    px_path = _write_px(source_root, "ZLK", dict(zip(old_dates, old_closes)))
    os.utime(px_path, (_epoch("2024-01-01T00:00:00+00:00"),) * 2)
    capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)

    restated = dict(zip(old_dates, old_closes * k)) | dict(zip(new_dates, new_closes))
    _write_tier1(source_root, "ZLK", restated, key="k1", fetched_at="2024-06-01T00:00:00+00:00")
    report = capture(conn, store, source_root, root=tmp_path, scope="shadow", clock=clock)
    assert report["rows_added"] == n_old + n_new  # every old date restated + n_new brand new

    repository = Repository(conn, store)
    pinned = repository.resolve(report["result_snapshot_id"])

    before = materialize_price_series(repository, pinned, tmp_path / "before", tickers=("ZLK",),
                                      observation_ceiling="2024-03-01T00:00:00Z").written
    after = materialize_price_series(repository, pinned, tmp_path / "after", tickers=("ZLK",),
                                     observation_ceiling="2024-12-31T23:59:59Z").written
    before_closes = price_download_sources.read_legacy_px_csv(before["ZLK"])["close_adj"].to_numpy()
    after_closes = price_download_sources.read_legacy_px_csv(after["ZLK"])["close_adj"].to_numpy()
    assert len(before_closes) == n_old
    assert len(after_closes) == n_old + n_new
    # The restatement actually took effect (else this test would prove nothing).
    assert not np.allclose(before_closes, after_closes[:n_old])

    anchor_idx = n_old - 1  # the last old date -- its whole 252-day lookback is old-region
    before_features = _runup_style_features(before_closes, anchor_idx)
    after_features = _runup_style_features(after_closes, anchor_idx)
    assert set(before_features) == {"dist_high", "dist_ema", "ret5", "ret10", "ret20"}
    assert set(before_features) == set(after_features)
    for name, b in before_features.items():
        a = after_features[name]
        assert abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0), (name, a, b)
