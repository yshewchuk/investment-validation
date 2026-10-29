"""Workspace closure catalog rows: bounded copy from the read-only source."""
from __future__ import annotations

import hashlib
import sqlite3
from types import SimpleNamespace

import pytest

from tests.data_scan_support import catalog_and_store
from tests.v2.ops.test_research_workspace import (  # noqa: F401
    _commit,
    _create,
    _details,
    prod,
)

from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.ops.workspace import catalog_rows
from engine.v2.ops.workspace.catalog_rows import copy_closure_rows


def _production_fragments(prod, snapshot_id):
    conn = sqlite3.connect(prod.catalog.as_uri() + "?mode=ro", uri=True)
    try:
        version = conn.execute(
            "SELECT dataset_version_id FROM data_snapshot_tables WHERE snapshot_id = ?",
            (snapshot_id,)).fetchone()[0]
        fragments = {row[0] for row in conn.execute(
            "SELECT fragment_id FROM data_version_fragments WHERE dataset_version_id = ?",
            (version,))}
        objects = conn.execute(
            "SELECT COUNT(DISTINCT object_id) FROM data_fragments WHERE fragment_id IN "
            "(SELECT fragment_id FROM data_version_fragments WHERE dataset_version_id = ?)",
            (version,)).fetchone()[0]
        return version, fragments, objects
    finally:
        conn.close()


def _production_contract(prod):
    conn = sqlite3.connect(prod.catalog.as_uri() + "?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT * FROM data_contracts LIMIT 1")
        columns = [column[0] for column in cursor.description]
        return dict(zip(columns, cursor.fetchone()))
    finally:
        conn.close()


def _insert_contract(ws, prod, contract_id, definition_hash):
    row = _production_contract(prod)
    row["contract_id"] = contract_id
    row["definition_hash"] = definition_hash
    columns = sorted(row)
    ws.conn.execute(
        "INSERT INTO data_contracts (%s) VALUES (%s)"
        % (", ".join(columns), ",".join("?" * len(columns))),
        [row[column] for column in columns])
    ws.conn.commit()


@pytest.fixture
def prod3(tmp_path):
    root = tmp_path / "prod3"
    root.mkdir()
    conn, clock, store = catalog_and_store(root)
    first, _ = _commit(conn, clock, store, "AAA", "r1", None, 0)
    second, _ = _commit(conn, clock, store, "BBB", "r2", first.snapshot_id, 1)
    snap, record = _commit(conn, clock, store, "CCC", "r3", second.snapshot_id, 2)
    conn.close()
    digest = record.object_ref.content_hash.removeprefix("sha256:")
    return SimpleNamespace(
        root=root, catalog=root / "catalog.sqlite", store=root / "store", snap=snap,
        clock=clock, tmp=tmp_path,
        object=root / "store" / "objects" / digest[:2] / digest)


def test_copy_brings_chain_and_manifest_hash(prod):
    with _create(prod) as ws:
        result = copy_closure_rows(ws)
        assert result["manifest_hash"] == prod.snap.manifest_hash
        assert result["chain"] == [prod.snap.parent_snapshot_id]
        assert result["rows"]["data_snapshots"] == 2
        snapshot_ids = {row[0] for row in ws.conn.execute(
            "SELECT snapshot_id FROM data_snapshots")}
        assert snapshot_ids == {prod.snap.snapshot_id, prod.snap.parent_snapshot_id}
        table_snapshot_ids = {row[0] for row in ws.conn.execute(
            "SELECT snapshot_id FROM data_snapshot_tables")}
        assert table_snapshot_ids == {prod.snap.snapshot_id}
        resolved = Repository(ws.conn).resolve_full(prod.snap.snapshot_id)
        assert resolved.snapshot.manifest_hash == prod.snap.manifest_hash
        assert ws.conn.execute("SELECT COUNT(*) FROM data_snapshot_heads").fetchone()[0] == 0


def test_chain_is_parent_first(prod3):
    second_id = prod3.snap.parent_snapshot_id
    conn = sqlite3.connect(prod3.catalog.as_uri() + "?mode=ro", uri=True)
    try:
        first_id = conn.execute(
            "SELECT parent_snapshot_id FROM data_snapshots WHERE snapshot_id = ?",
            (second_id,)).fetchone()[0]
    finally:
        conn.close()
    with _create(prod3) as ws:
        result = copy_closure_rows(ws)
    assert result["chain"] == [second_id, first_id]


def test_copy_is_exactly_the_closure(prod):
    pinned_version, pinned_fragments, pinned_objects = _production_fragments(
        prod, prod.snap.snapshot_id)
    ancestor_version, ancestor_fragments, _ = _production_fragments(
        prod, prod.snap.parent_snapshot_id)
    with _create(prod) as ws:
        copy_closure_rows(ws)
        assert ws.conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0] == pinned_objects
        assert {row[0] for row in ws.conn.execute(
            "SELECT fragment_id FROM data_fragments")} == pinned_fragments
        assert {row[0] for row in ws.conn.execute(
            "SELECT DISTINCT dataset_version_id FROM data_version_fragments")} == {pinned_version}
        assert ancestor_version != pinned_version
        assert not ancestor_fragments & pinned_fragments
        assert ws.conn.execute(
            "SELECT COUNT(*) FROM data_fragments WHERE fragment_id IN (%s)"
            % ",".join("?" * len(ancestor_fragments)),
            tuple(ancestor_fragments)).fetchone()[0] == 0


def test_copy_is_idempotent(prod):
    with _create(prod) as ws:
        first = copy_closure_rows(ws)
        counts = {table: ws.conn.execute(
            "SELECT COUNT(*) FROM %s" % table).fetchone()[0] for table in first["rows"]}
        second = copy_closure_rows(ws)
        assert second == first
        assert first["rows"]["data_snapshots"] == 2
        assert {table: ws.conn.execute(
            "SELECT COUNT(*) FROM %s" % table).fetchone()[0] for table in first["rows"]} == counts


def test_workspace_in_transaction_refuses(prod):
    with _create(prod) as ws:
        ws.conn.execute("BEGIN")
        try:
            with pytest.raises(OpsError) as caught:
                copy_closure_rows(ws)
            assert _details(caught) == ("WORKSPACE_REFUSED", "workspace_invalid")
        finally:
            ws.conn.rollback()


def test_row_cap_refuses(prod, monkeypatch):
    monkeypatch.setattr(catalog_rows, "_MAX_ROWS", 1)
    with _create(prod) as ws:
        with pytest.raises(OpsError) as caught:
            copy_closure_rows(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "copy_rows_over_cap")
        assert ws.conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0


def test_byte_cap_refuses(prod, monkeypatch):
    monkeypatch.setattr(catalog_rows, "_MAX_BYTES", 1)
    with _create(prod) as ws:
        with pytest.raises(OpsError) as caught:
            copy_closure_rows(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "copy_bytes_over_cap")


def test_unexpected_extra_row_refuses_and_rolls_back(prod):
    with _create(prod) as ws:
        _insert_contract(ws, prod, "other", "sha256:" + "0" * 64)
        with pytest.raises(OpsError) as caught:
            copy_closure_rows(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "unexpected_rows")
        assert ws.conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0


def test_conflicting_existing_row_refuses(prod):
    contract_id = _production_contract(prod)["contract_id"]
    with _create(prod) as ws:
        try:
            _insert_contract(ws, prod, contract_id, "sha256:" + "1" * 64)
        except (sqlite3.IntegrityError, sqlite3.OperationalError):
            pytest.skip("workspace triggers block inserting a conflicting row")
        with pytest.raises(OpsError) as caught:
            copy_closure_rows(ws)
        assert _details(caught) == ("WORKSPACE_REFUSED", "row_mismatch")
        assert ws.conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0


def test_production_is_unchanged(prod):
    before = hashlib.sha256(prod.catalog.read_bytes()).hexdigest()
    with _create(prod) as ws:
        copy_closure_rows(ws)
    assert hashlib.sha256(prod.catalog.read_bytes()).hexdigest() == before


def test_no_object_files_copied(prod):
    with _create(prod) as ws:
        before = sorted(path.relative_to(ws.store.root)
                        for path in ws.store.root.rglob("*"))
        copy_closure_rows(ws)
        after = sorted(path.relative_to(ws.store.root)
                       for path in ws.store.root.rglob("*"))
        assert after == before
