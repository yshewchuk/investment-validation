"""Synthetic tests for the pinned original-panel COPY identity helper.

Catalog + ArtifactStore built with the shared ``tests.data_scan_support``
helpers (real published Parquet objects, real ``FragmentRecord``s, real
``commit_snapshot``). The helper returns a direct dict whose every key carries
the ``panel_copy_`` prefix. No private fixtures, no live provider, no fitting.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pyarrow as pa
import pytest

from engine.v2.foundation import CONTENT_HASH_PREFIX
from tests.data_scan_support import (catalog_and_store, commit_tables, contract_for,
                                     contract_ref_for, fake_hash, hand_built_record, to_bytes)
from tools.phase5_pinned_panel import COPY_PREFIX, PANEL_TABLE, PinnedPanelContext, verify_panel_copy

IDENTITY_KEYS = {"panel_copy_mode", "panel_copy_snapshot_id", "panel_copy_dataset_version_id",
                 "panel_copy_object_id", "panel_copy_sha256"}


def panel_table(rows: list[tuple[str, str, float]]) -> pa.Table:
    return pa.table({
        "ticker": pa.array([r[0] for r in rows], type=pa.string()),
        "date": pa.array([r[1] for r in rows], type=pa.string()),
        "momentum": pa.array([r[2] for r in rows], type=pa.float64()),
    })


ROWS = [("AAA", "2024-01-02", 0.5), ("MMM", "2024-06-30", 1.5)]
ALL_ONE = [(panel_table(ROWS), "all", ("AAA", "2024-01-02"), ("MMM", "2024-06-30"))]


def build_pinned(tmp_path, tables_and_keys):
    """One committed snapshot over ``feature_panel`` fragments; returns
    ``(context, store, snapshot_ref, records, source_bytes)``. The context uses
    the store's real root and the catalog filename the connection reported."""
    conn, clock, store = catalog_and_store(tmp_path)
    contract = contract_for(PANEL_TABLE)
    contract_ref = contract_ref_for(contract)
    records, source_bytes = [], None
    for table, partition_key, key_min, key_max in tables_and_keys:
        record = hand_built_record(store, contract, contract_ref, table,
                                   partition_key=partition_key, row_count=table.num_rows,
                                   primary_key_min=key_min, primary_key_max=key_max,
                                   logical_label=fake_hash(f"panel-{partition_key}"))
        records.append(record)
        if source_bytes is None:
            source_bytes = to_bytes(table)
    snapshot = commit_tables(conn, clock, {PANEL_TABLE: records}, {PANEL_TABLE: contract},
                             store=store)
    catalog_path = Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()
    conn.close()
    context = PinnedPanelContext(catalog_path=catalog_path, snapshot_id=snapshot.snapshot_id,
                                 artifact_root=store.root)
    return context, store, snapshot, records, source_bytes


def test_real_original_object_returns_direct_prefixed_identities(tmp_path):
    context, _store, snapshot, records, source_bytes = build_pinned(tmp_path, ALL_ONE)
    identity = verify_panel_copy(context)
    assert set(identity) == IDENTITY_KEYS
    assert all(key.startswith(COPY_PREFIX) for key in identity)
    assert identity["panel_copy_mode"] == "COPY"
    assert identity["panel_copy_snapshot_id"] == snapshot.snapshot_id
    assert identity["panel_copy_dataset_version_id"] == snapshot.table_versions[PANEL_TABLE].dataset_version_id
    assert identity["panel_copy_object_id"] == records[0].object_ref.object_id
    assert identity["panel_copy_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert records[0].object_ref.content_hash == CONTENT_HASH_PREFIX + identity["panel_copy_sha256"]


def test_expected_hash_accepted_prefixed_or_bare(tmp_path):
    context, _store, _snap, _records, source_bytes = build_pinned(tmp_path, ALL_ONE)
    raw = hashlib.sha256(source_bytes).hexdigest()
    assert verify_panel_copy(context, raw) == verify_panel_copy(context, CONTENT_HASH_PREFIX + raw)


def test_expected_hash_mismatch_refused_without_path(tmp_path):
    context, _store, _snap, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    with pytest.raises(ValueError) as excinfo:
        verify_panel_copy(context, "0" * 64)
    message = str(excinfo.value)
    assert message.startswith(COPY_PREFIX)
    assert str(tmp_path) not in message


def test_missing_snapshot_refused(tmp_path):
    context, _store, _snap, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    missing = PinnedPanelContext(catalog_path=context.catalog_path,
                                 snapshot_id="snap_missing", artifact_root=context.artifact_root)
    with pytest.raises(ValueError, match=COPY_PREFIX):
        verify_panel_copy(missing)


def test_unreadable_catalog_refused(tmp_path):
    context = PinnedPanelContext(catalog_path=tmp_path / "nope.sqlite",
                                 snapshot_id="snap_any", artifact_root=tmp_path / "store")
    with pytest.raises(ValueError, match=COPY_PREFIX):
        verify_panel_copy(context)


def test_readable_but_empty_catalog_refused_without_path(tmp_path):
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).close()
    context = PinnedPanelContext(catalog_path=empty, snapshot_id="snap_any",
                                 artifact_root=tmp_path / "store")
    with pytest.raises(ValueError) as excinfo:
        verify_panel_copy(context)
    message = str(excinfo.value)
    assert message.startswith(COPY_PREFIX)
    assert str(tmp_path) not in message


def test_multi_fragment_panel_refused(tmp_path):
    context, _store, _snap, _records, _bytes = build_pinned(
        tmp_path, [
            (panel_table(ROWS[:1]), "a", ("AAA", "2024-01-02"), ("MMM", "2024-06-30")),
            (panel_table([("ZZZ", "2024-12-31", 2.0)]), "b", ("NNN", "2024-07-01"),
             ("ZZZ", "2024-12-31")),
        ])
    with pytest.raises(ValueError, match=COPY_PREFIX):
        verify_panel_copy(context)


def test_corrupt_object_bytes_refused(tmp_path):
    context, store, _snap, records, _bytes = build_pinned(tmp_path, ALL_ONE)
    digest = records[0].object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    object_path = store.root / "objects" / digest[:2] / digest
    object_path.write_bytes(b"substituted, not the original panel bytes at all")
    with pytest.raises(ValueError, match=COPY_PREFIX):
        verify_panel_copy(context)


def test_global_live_panel_path_is_ignored(tmp_path, monkeypatch):
    """Verification is anchored to the pinned catalog/objects only: pointing the
    live global panel path at unrelated bytes cannot change the COPY identity."""
    context, _store, _snap, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    before = verify_panel_copy(context)
    decoy = tmp_path / "decoy-panel.parquet"
    monkeypatch.setattr("engine.paths.PANEL", decoy)
    decoy.write_bytes(b"global panel rewrite")
    (tmp_path / "panel.parquet").write_bytes(b"another global panel rewrite")
    assert verify_panel_copy(context) == before


def test_catalog_path_with_uri_ambiguous_characters_verifies(tmp_path):
    context, _store, snapshot, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    expected = verify_panel_copy(context)
    relocated = tmp_path / "snap shots" / "odd?#1.sqlite"
    relocated.parent.mkdir(parents=True)
    context.catalog_path.rename(relocated)
    catalog_bytes = relocated.read_bytes()
    moved = PinnedPanelContext(catalog_path=relocated, snapshot_id=snapshot.snapshot_id,
                               artifact_root=context.artifact_root)
    assert verify_panel_copy(moved) == expected
    assert relocated.read_bytes() == catalog_bytes  # read-only: source never rewritten


def test_verification_publishes_nothing(tmp_path):
    context, store, _snap, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    objects_dir = store.root / "objects"
    before = sorted(p.relative_to(store.root) for p in objects_dir.rglob("*") if p.is_file())
    verify_panel_copy(context)
    verify_panel_copy(context)
    after = sorted(p.relative_to(store.root) for p in objects_dir.rglob("*") if p.is_file())
    assert before == after
