"""Tier-0 tests for the v2 reconcile path (``engine.v2.research.reconcile_trades``).

The parent snapshot carries canonical engine rows, non-canonical simulated
rows and a live record. Reconciliation must tombstone exactly the
non-canonical simulated rows as a new ``trades`` dataset version, leave
everything else byte-identical, record the snapshot it read, and honour an
explicit ``snapshot_id`` after the head has moved. The legacy
``tools/reconcile_trades.py`` and its mutable table are not touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data.normalize import n_trades as legacy_n_trades  # noqa: E402
from engine.v2.contracts.data import DatasetManifest  # noqa: E402
from engine.v2.data import catalog, manifests  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import _snapshot, reconcile_trades  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)

_TRADES_COLUMNS = (
    "trade_id", "kind", "strategy", "variant", "ticker", "event_id",
    "event_date", "year", "legs", "entry_date", "exit_date", "strike",
    "expiry", "fill_alpha", "entry_cost", "exit_value", "ret", "provenance",
)


def _event_row(event_id: str, ticker: str = "TEST") -> dict:
    return {
        "event_id": event_id, "ticker": ticker,
        "event_date": pd.Timestamp(event_id.rsplit("_", 1)[-1]), "year": 2024,
        "session": "AMC", "session_src": "test", "annc_tod": None,
        "src_orats": True, "src_oquants": True, "src_nasdaq": False,
        "src_yfinance": False, "date_agree": True, "date_conflict": False,
        "updated_at": None, "event_cluster_id": None, "claim_count": 1,
        "reconciliation": "test",
    }


def _trade_row(trade_id, event_id, *, kind="sim", provenance="engine.replay",
               fill_alpha=0.0, entry_cost=3.8, exit_value=5.0, year=2024) -> dict:
    event_date = pd.Timestamp("2024-05-02")
    return {
        "trade_id": trade_id, "kind": kind, "strategy": "STR-THRU",
        "variant": "e+0_x+1", "ticker": "TEST", "event_id": event_id,
        "event_date": event_date, "year": year, "legs": "{}",
        "entry_date": event_date, "exit_date": pd.Timestamp("2024-05-03"),
        "strike": 100.0, "expiry": pd.Timestamp("2024-05-03"),
        "fill_alpha": fill_alpha, "entry_cost": entry_cost,
        "exit_value": exit_value, "ret": (exit_value - entry_cost) / entry_cost,
        "provenance": provenance,
    }


def _publish(store, name, rows):
    contract = contract_for(name)
    record = publish_and_inspect(store, contract, contract_ref_for(contract),
                                 rows, "2024")
    return contract, record


def _commit_all(conn, clock, store, *, trades_rows, event_rows, receipt_id,
                expected_head=None, generation=0):
    """One real snapshot over ``trades`` and ``earnings_events``."""
    contracts, tables, all_records, all_objects = {}, {}, [], []
    for name, rows in (
        ("trades", sorted(trades_rows, key=lambda row: str(row["trade_id"]))),
        ("earnings_events", sorted(event_rows, key=lambda row: str(row["event_id"]))),
    ):
        contract, record = _publish(store, name, rows)
        contracts[name], tables[name] = contract, [record]
        all_records.extend(tables[name])
        all_objects.append(record.object_ref)
    table_manifests: dict[str, DatasetManifest] = {}
    for name, records in tables.items():
        table_manifests[name] = manifests.dataset_manifest(
            contract_ref_for(contracts[name]), records,
            knowledge_mode="reconstructed", coverage_receipt_refs=(fake_hash("cov"),),
            availability_evidence_refs=())
    snap = manifests.snapshot_ref(
        table_manifests, calendar_version="cal.v1",
        source_priority_version="prio.v1", finality_receipt_refs=(fake_hash("fin"),),
    )
    receipt = catalog.commit_snapshot(
        conn, scope="shadow", request_hash=fake_hash(f"{receipt_id}-request"),
        contracts=list(contracts.values()), objects=all_objects, records=all_records,
        manifests=list(table_manifests.values()), snapshot=snap,
        expected_head_snapshot_id=expected_head, expected_head_generation=generation,
        receipt_id=receipt_id, attempt_id="att-1", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store,
    )
    return receipt.snapshot_ref


def _mixed_trades() -> list[dict]:
    return [
        _trade_row("canon:1", "TEST_2024-05-02"),
        _trade_row("gone:1", "GONE_2024-01-01"),
        _trade_row("gone:legacy", "GONE_2024-01-01", provenance="legacy:S2"),
        _trade_row("live:1", "GONE_2024-01-01", kind="live", provenance="live:book"),
    ]


def _read_trades(repository, snapshot) -> pd.DataFrame:
    return _snapshot.read_table(repository, snapshot, "trades", _TRADES_COLUMNS)


def test_reconcile_removes_only_non_canonical_rows(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    parent = _commit_all(conn, clock, store, trades_rows=_mixed_trades(),
                         event_rows=[_event_row("TEST_2024-05-02")], receipt_id="r1")
    repository = Repository(conn, store)

    outcome = reconcile_trades.run(repository, scope="shadow",
                                   reports_dir=tmp_path / "reports", stamp="t1")
    assert outcome["committed"] is True
    assert outcome["outcome"] == "changed"
    assert outcome["snapshot_id"] == parent.snapshot_id
    assert outcome["rows_removed"] == 2
    assert outcome["removed_trade_ids"] == ["gone:1", "gone:legacy"]

    committed = repository.resolve_pinned("shadow")
    assert committed.snapshot_id != parent.snapshot_id
    ids = set(_read_trades(repository, committed)["trade_id"].astype(str))
    assert ids == {"canon:1", "live:1"}
    conn.close()


def test_reconcile_leaves_canonical_rows_untouched(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    parent = _commit_all(conn, clock, store, trades_rows=_mixed_trades(),
                         event_rows=[_event_row("TEST_2024-05-02")], receipt_id="r1")
    repository = Repository(conn, store)
    before = _read_trades(repository, parent)

    reconcile_trades.run(repository, scope="shadow", dry_run=False,
                         reports_dir=None, stamp="t1")

    after = _read_trades(repository, repository.resolve_pinned("shadow"))
    keep = before[before["trade_id"].isin(["canon:1", "live:1"])]
    got = after[after["trade_id"].isin(["canon:1", "live:1"])]
    pd.testing.assert_frame_equal(
        got.sort_values("trade_id").reset_index(drop=True),
        keep.sort_values("trade_id").reset_index(drop=True),
    )
    conn.close()


def test_reconcile_records_the_pinned_snapshot_id(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    parent = _commit_all(conn, clock, store, trades_rows=_mixed_trades(),
                         event_rows=[_event_row("TEST_2024-05-02")], receipt_id="r1")
    repository = Repository(conn, store)

    outcome = reconcile_trades.run(repository, scope="shadow",
                                   reports_dir=tmp_path / "reports", stamp="t1")
    assert outcome["snapshot_id"] == parent.snapshot_id
    committed = repository.resolve(outcome["committed_snapshot_id"])
    assert committed.parent_snapshot_id == parent.snapshot_id
    report = json.loads(Path(outcome["path"]).read_text())
    assert report["snapshot_id"] == parent.snapshot_id
    assert report["rows_removed"] == 2
    conn.close()


def test_reconcile_explicit_snapshot_id_ignores_a_moved_head(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    first = _commit_all(
        conn, clock, store,
        trades_rows=[_trade_row("r1", "LATER_2024-06-01"),
                     _trade_row("r2", "TEST_2024-05-02")],
        event_rows=[_event_row("TEST_2024-05-02")], receipt_id="r1",
    )
    second = _commit_all(
        conn, clock, store,
        trades_rows=[_trade_row("r1", "LATER_2024-06-01"),
                     _trade_row("r2", "TEST_2024-05-02"),
                     _trade_row("r3", "GONE_2024-01-01")],
        event_rows=[_event_row("TEST_2024-05-02"), _event_row("LATER_2024-06-01")],
        receipt_id="r2", expected_head=first.snapshot_id, generation=1,
    )
    assert second.snapshot_id != first.snapshot_id
    repository = Repository(conn, store)

    pinned = reconcile_trades.run(repository, scope="shadow",
                                  snapshot_id=first.snapshot_id, dry_run=True,
                                  reports_dir=None)
    assert pinned["snapshot_id"] == first.snapshot_id
    assert pinned["removed_trade_ids"] == ["r1"]

    head = reconcile_trades.run(repository, scope="shadow", dry_run=True,
                                reports_dir=None)
    assert head["snapshot_id"] == second.snapshot_id
    assert head["removed_trade_ids"] == ["r3"]
    assert repository.resolve_pinned("shadow").snapshot_id == second.snapshot_id
    conn.close()


def test_reconcile_refuses_a_stale_expected_head_rather_than_overwriting(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    first = _commit_all(conn, clock, store, trades_rows=_mixed_trades(),
                        event_rows=[_event_row("TEST_2024-05-02")], receipt_id="r1")
    second = _commit_all(conn, clock, store,
                         trades_rows=_mixed_trades() + [_trade_row("canon:2", "TEST_2024-05-02")],
                         event_rows=[_event_row("TEST_2024-05-02")], receipt_id="r2",
                         expected_head=first.snapshot_id, generation=1)
    repository = Repository(conn, store)

    with pytest.raises(DataError) as err:
        reconcile_trades.run(repository, scope="shadow", snapshot_id=first.snapshot_id,
                             reports_dir=None)
    assert err.value.code == "SNAPSHOT_CONFLICT"
    assert repository.resolve_pinned("shadow").snapshot_id == second.snapshot_id
    conn.close()


def test_moved_filter_matches_legacy_on_same_frames():
    trades = pd.DataFrame(_mixed_trades())
    events = pd.DataFrame([_event_row("TEST_2024-05-02")])
    new, new_report = reconcile_trades.filter_to_canonical_events(trades.copy(), events)
    old, old_report = legacy_n_trades.filter_to_canonical_events(trades.copy(), events)
    pd.testing.assert_frame_equal(new, old)
    assert new_report == old_report
