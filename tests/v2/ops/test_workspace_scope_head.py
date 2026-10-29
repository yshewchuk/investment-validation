"""Workspace native scope head over the pinned snapshot, with reference inputs."""
from __future__ import annotations

import sqlite3

import pytest

from tests.data_scan_support import fake_hash
from tests.v2.ops.test_research_workspace import (  # noqa: F401
    _create,
    _details,
    _state,
    prod,
)

from engine.v2.data import catalog, errors as data_errors
from engine.v2.data import reference_catalog
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.ops.workspace import head as head_module
from engine.v2.ops.workspace.head import commit_scope_head
from engine.v2.ops.workspace.catalog_rows import copy_closure_rows

SCOPE = "size-scope"


def _add_inputs(prod, n=2):
    conn = sqlite3.connect(prod.catalog)
    receipt = conn.execute(
        "SELECT commit_receipt_ref FROM data_snapshots WHERE snapshot_id = ?",
        (prod.snap.snapshot_id,)).fetchone()[0]
    for i in range(n):
        conn.execute(
            "INSERT INTO data_import_reference_inputs (receipt_id, legacy_path, kind, "
            "object_id, content_hash, byte_size, fold) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (receipt, f"ref/{i}.json", "calendar", f"obj{i}", fake_hash(f"c{i}"), 10 + i, ""))
    conn.commit()
    conn.close()
    return receipt


def _heads(ws):
    return ws.conn.execute("SELECT COUNT(*) FROM data_snapshot_heads").fetchone()[0]


def _inputs(conn, receipt_id):
    return [tuple(r) for r in conn.execute(
        "SELECT legacy_path, kind, object_id, content_hash, byte_size FROM "
        "data_import_reference_inputs WHERE receipt_id = ? ORDER BY legacy_path",
        (receipt_id,))]


def test_head_resolves_to_pinned_snapshot_with_reference_inputs(prod):
    original = _add_inputs(prod)
    with _create(prod) as ws:
        result = commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert result["snapshot_id"] == prod.snap.snapshot_id
        assert result["manifest_hash"] == prod.snap.manifest_hash
        assert result["generation"] == 1
        assert result["reference_inputs"] == 2
        pointer = ws.conn.execute(
            "SELECT commit_receipt_ref FROM data_snapshots WHERE snapshot_id = ?",
            (prod.snap.snapshot_id,)).fetchone()[0]
        assert pointer == original
        assert tuple(ws.conn.execute(
            "SELECT result_snapshot_id, status FROM data_import_receipts WHERE receipt_id = ?",
            (pointer,)).fetchone()) == (prod.snap.snapshot_id, "committed")
        head = Repository(ws.conn).resolve_pinned(SCOPE)
        assert head.snapshot_id == prod.snap.snapshot_id
        assert head.manifest_hash == prod.snap.manifest_hash
        ro = sqlite3.connect(prod.catalog.as_uri() + "?mode=ro", uri=True)
        expected = _inputs(ro, original)
        ro.close()
        assert len(expected) == 2
        assert _inputs(ws.conn, result["receipt_id"]) == expected
        assert len(reference_catalog.reference_inputs_for_receipt(
            ws.conn, receipt_id=result["receipt_id"])) == 2


def test_rerun_is_idempotent(prod):
    _add_inputs(prod)
    with _create(prod) as ws:
        first = commit_scope_head(ws, SCOPE, clock=prod.clock)
        counts = [ws.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in (
            "data_import_receipts", "data_import_reference_inputs", "data_snapshots")]
        second = commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert second == first
        assert counts == [ws.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in (
            "data_import_receipts", "data_import_reference_inputs", "data_snapshots")]
        assert _heads(ws) == 1


def test_no_reference_inputs_refuses_and_leaves_no_head(prod):
    with _create(prod) as ws:
        with pytest.raises(data_errors.DataError) as caught:
            commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert caught.value.problem.code == "SNAPSHOT_NOT_READY"
        assert _heads(ws) == 0


def test_failed_commit_leaves_no_head_and_rerun_converges(prod, monkeypatch):
    _add_inputs(prod)
    with _create(prod) as ws:
        real = reference_catalog.insert_reference_inputs

        def boom(*args, **kwargs):
            raise RuntimeError("injected")

        monkeypatch.setattr(reference_catalog, "insert_reference_inputs", boom)
        with pytest.raises(OpsError) as caught:
            commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert _details(caught) == ("WORKSPACE_REFUSED", "storage_failure")
        assert _heads(ws) == 0
        assert ws.conn.execute(
            "SELECT COUNT(*) FROM data_import_receipts WHERE receipt_id LIKE 'size-head-%'"
        ).fetchone()[0] == 0
        monkeypatch.setattr(reference_catalog, "insert_reference_inputs", real)
        result = commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert result["reference_inputs"] == 2
        assert _heads(ws) == 1


def test_invalid_scope_refuses_without_writing(prod):
    _add_inputs(prod)
    with _create(prod) as ws:
        with pytest.raises(OpsError) as caught:
            commit_scope_head(ws, "", clock=prod.clock)
        assert _details(caught) == ("WORKSPACE_REFUSED", "scope_invalid")
        assert ws.conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0


def test_production_is_unchanged(prod):
    _add_inputs(prod)
    before = _state(prod)
    with _create(prod) as ws:
        commit_scope_head(ws, SCOPE, clock=prod.clock)
    assert _state(prod) == before


def test_reference_inputs_over_cap_refuses_without_head(prod, monkeypatch):
    _add_inputs(prod)
    monkeypatch.setattr(head_module, "_MAX_REFERENCE_INPUTS", 1)
    with _create(prod) as ws:
        with pytest.raises(OpsError) as caught:
            commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert _details(caught) == ("WORKSPACE_REFUSED", "reference_inputs_over_cap")
        assert _heads(ws) == 0


def test_oversized_reference_inputs_bytes_refuse(prod, monkeypatch):
    _add_inputs(prod)
    monkeypatch.setattr(head_module, "_MAX_REFERENCE_BYTES", 10)
    with _create(prod) as ws:
        with pytest.raises(OpsError) as caught:
            commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert _details(caught) == ("WORKSPACE_REFUSED", "reference_inputs_over_cap")
        assert _heads(ws) == 0


def test_replay_after_head_moved_refuses_conflict(prod):
    _add_inputs(prod)
    with _create(prod) as ws:
        result = commit_scope_head(ws, SCOPE, clock=prod.clock)
        catalog.move_head(
            ws.conn, scope=SCOPE, to_snapshot_id=prod.snap.parent_snapshot_id,
            expected_snapshot_id=prod.snap.snapshot_id, expected_generation=1,
            receipt_ref=result["receipt_id"], clock=prod.clock)
        with pytest.raises(data_errors.DataError) as caught:
            commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert caught.value.problem.code == "SNAPSHOT_CONFLICT"
        assert ws.conn.execute(
            "SELECT snapshot_id FROM data_snapshot_heads WHERE scope = ?", (SCOPE,)
        ).fetchone()[0] == prod.snap.parent_snapshot_id


def test_conflicting_existing_receipt_row_refuses(prod):
    original = _add_inputs(prod)
    with _create(prod) as ws:
        copy_closure_rows(ws)
        ro = sqlite3.connect(prod.catalog.as_uri() + "?mode=ro", uri=True)
        row = list(ro.execute(
            "SELECT * FROM data_import_receipts WHERE receipt_id = ?", (original,)).fetchone())
        ro.close()
        row[1] = row[1] + "-conflict"
        ws.conn.execute(
            "INSERT INTO data_import_receipts VALUES (%s)" % ",".join("?" * len(row)), row)
        ws.conn.commit()
        with pytest.raises(OpsError) as caught:
            commit_scope_head(ws, SCOPE, clock=prod.clock)
        assert _details(caught) == ("WORKSPACE_REFUSED", "row_mismatch")
        assert _heads(ws) == 0
