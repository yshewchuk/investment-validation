"""Tier-0 test for the v2 trades write path (``generic_incremental``).

The parent snapshot carries a ``trades`` table with a legacy row, an
other-strategy engine row, and two STR-THRU engine rows. A rebuild of STR-THRU
must tombstone the STR-THRU row the rebuild no longer produces, correct the
one it does, and leave everything else exactly where it was; a commit against
a head another writer already advanced must refuse, not overwrite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import DatasetManifest  # noqa: E402
from engine.v2.data import catalog, manifests  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import _build_run, _plan, _snapshot, build_trades  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)
from tests.test_v2_research_replay import (  # noqa: E402
    _calendar,
    _chain_rows,
    _event_rows,
)

_TRADES_COLUMNS = (
    "trade_id", "kind", "strategy", "variant", "ticker", "event_id",
    "event_date", "year", "legs", "entry_date", "exit_date", "strike",
    "expiry", "fill_alpha", "entry_cost", "exit_value", "ret", "provenance",
)


def _trade_row(trade_id, strategy, provenance, *, fill_alpha=0.0, entry_cost=3.8,
               exit_value=5.0, year=2024) -> dict:
    event_date = pd.Timestamp("2024-05-02")
    return {
        "trade_id": trade_id, "kind": "sim", "strategy": strategy,
        "variant": "e+0_x+1", "ticker": "TEST", "event_id": "TEST_2024-05-02",
        "event_date": event_date, "year": year, "legs": "{}",
        "entry_date": event_date, "exit_date": pd.Timestamp("2024-05-03"),
        "strike": 100.0, "expiry": pd.Timestamp("2024-05-03"),
        "fill_alpha": fill_alpha, "entry_cost": entry_cost,
        "exit_value": exit_value, "ret": (exit_value - entry_cost) / entry_cost,
        "provenance": provenance,
    }


def _publish(store, name, rows, partition_key):
    contract = contract_for(name)
    record = publish_and_inspect(store, contract, contract_ref_for(contract),
                                 rows, partition_key)
    return contract, record


def _commit_all(conn, clock, store, *, trades_rows, receipt_id,
                expected_head=None, generation=0):
    """One real snapshot over option_chains, earnings_events and trades."""
    contracts, tables, all_records, all_objects = {}, {}, [], []
    for name, rows, partition in (
        ("option_chains", _chain_rows(), "2024"),
        ("earnings_events", _event_rows(), "2024"),
        ("trades", sorted(trades_rows, key=lambda row: str(row["trade_id"])), "2024"),
    ):
        contract, record = _publish(store, name, rows, partition)
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


def _initial_trades() -> list[dict]:
    return [
        _trade_row("legacy:1", "S2", "legacy:S2"),
        _trade_row("STR-THRU:e+0_x+1:TEST:20240502:a0", "STR-THRU", "engine.replay"),
        _trade_row("STR-THRU:e+0_x+1:OLD:20240101:a0", "STR-THRU", "engine.replay"),
        _trade_row("CAL-P:e+0_x+1:TEST:20240502:a0", "CAL-P", "engine.replay"),
    ]


def test_rebuild_tombstones_only_the_rebuilt_strategys_vanished_rows(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    parent = _commit_all(conn, clock, store, trades_rows=_initial_trades(),
                         receipt_id="r1")
    monkeypatch.setattr(_plan, "trading_calendar", lambda: _calendar())
    repository = Repository(conn, store)

    outcome = _build_run.run(
        repository, strategies=["STR-THRU"], reports_dir=tmp_path / "reports",
        stamp="t1",
    )
    assert outcome["committed"] is True
    assert outcome["outcome"] == "changed"
    assert outcome["rows"] == 5

    committed = repository.resolve_pinned("shadow")
    assert committed.snapshot_id != parent.snapshot_id
    frame = _snapshot.read_table(repository, committed, "trades", _TRADES_COLUMNS)
    ids = set(frame["trade_id"].astype(str))

    assert "legacy:1" in ids
    assert "CAL-P:e+0_x+1:TEST:20240502:a0" in ids
    assert "STR-THRU:e+0_x+1:OLD:20240101:a0" not in ids

    rebuilt = frame[frame["provenance"] == build_trades.PROVENANCE]
    assert len(rebuilt) == 5
    expected = {f"STR-THRU:e+0_x+1:TEST:20240502:a{alpha}"
                for alpha in (0, 25, 50, 75, 100)}
    assert set(rebuilt["trade_id"].astype(str)) == expected
    conn.close()


def test_rebuild_refuses_a_stale_expected_head_rather_than_overwriting(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    parent = _commit_all(conn, clock, store, trades_rows=_initial_trades(),
                         receipt_id="r1")
    monkeypatch.setattr(_plan, "trading_calendar", lambda: _calendar())
    repository = Repository(conn, store)

    _build_run.run(repository, strategies=["STR-THRU"],
                   reports_dir=tmp_path / "reports", stamp="t1")
    advanced = repository.resolve_pinned("shadow")

    with pytest.raises(DataError) as err:
        _build_run.run(
            repository, strategies=["STR-THRU"],
            snapshot_id=parent.snapshot_id, reports_dir=tmp_path / "stale",
            stamp="stale",
        )
    assert err.value.code == "SNAPSHOT_CONFLICT"
    assert repository.resolve_pinned("shadow").snapshot_id == advanced.snapshot_id
    conn.close()


def test_rebuild_dry_run_writes_nothing(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    parent = _commit_all(conn, clock, store, trades_rows=_initial_trades(),
                         receipt_id="r1")
    monkeypatch.setattr(_plan, "trading_calendar", lambda: _calendar())
    repository = Repository(conn, store)

    outcome = _build_run.run(repository, strategies=["STR-THRU"],
                             reports_dir=None, dry_run=True)
    assert outcome["committed"] is False
    assert repository.resolve_pinned("shadow").snapshot_id == parent.snapshot_id
    conn.close()
