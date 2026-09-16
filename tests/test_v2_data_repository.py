"""D10: exact snapshot resolution — phase-2 guide §8.1, §12.

``Repository.resolve`` never touches object bytes on disk (that is scan's job,
P2-4, out of scope here) — it rebuilds identity purely from catalog rows, so
every fixture in this file uses the same synthetic (non-materialized)
``FragmentRecord``s ``tests/test_v2_data_manifests.py`` already builds, reused
per the task brief rather than re-derived. ``resolve_snapshot_head`` is the
one function here that does touch a real store: it publishes the resolved
``SnapshotRef`` document as a Phase 1 artifact.

The "corrupt membership" and "resolve never reads data_snapshot_heads" cases
work on a **consistent copy** of the catalog file (a fresh ``sqlite3``
``backup()``, never a raw file copy under WAL), so the corruption never
touches the connection other tests in this file still use.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data.catalog import commit_snapshot  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.catalog import connect as ops_connect  # noqa: E402
from engine.v2.ops.snapshots import resolve_snapshot_head  # noqa: E402
from tests.ops_support import FakeClock  # noqa: E402
from tests.test_v2_data_commit import (  # noqa: E402
    _commit,
    _hash,
    _manifest_and_snapshot,
    _noop_fence,
)
from tests.test_v2_data_manifests import _SEC_CONTRACT, _record_for  # noqa: E402


def _catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    return conn, clock


def _consistent_copy(src_path: Path, dst_path: Path) -> None:
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
    finally:
        src.close()
        dst.close()


# --------------------------------------------------------------------------
# unknown id, corrupt membership, never reads data_snapshot_heads
# --------------------------------------------------------------------------


def test_resolve_unknown_snapshot_id_not_found(tmp_path):
    conn, _clock = _catalog(tmp_path)
    with pytest.raises(DataError) as err:
        Repository(conn).resolve("snap_" + "0" * 32)
    assert err.value.code == "SNAPSHOT_NOT_FOUND"


def test_resolve_corrupt_membership_is_manifest_corrupt(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")
    conn.close()

    copy_path = tmp_path / "corrupt.sqlite"
    _consistent_copy(tmp_path / "catalog.sqlite", copy_path)
    corrupt = ops_connect(copy_path)
    corrupt.execute("DROP TRIGGER data_version_fragments_no_delete")
    corrupt.execute("DROP TRIGGER data_version_fragments_no_update")
    dsv_id = corrupt.execute(
        "SELECT dataset_version_id FROM data_snapshot_tables WHERE snapshot_id = ?",
        (snap.snapshot_id,)).fetchone()[0]
    corrupt.execute("DELETE FROM data_version_fragments WHERE dataset_version_id = ?", (dsv_id,))

    with pytest.raises(DataError) as err:
        Repository(corrupt).resolve(snap.snapshot_id)
    assert err.value.code == "MANIFEST_CORRUPT"
    corrupt.close()


def test_resolve_never_reads_data_snapshot_heads(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")
    conn.close()

    copy_path = tmp_path / "no_heads.sqlite"
    _consistent_copy(tmp_path / "catalog.sqlite", copy_path)
    stripped = ops_connect(copy_path)
    stripped.execute("DROP TABLE data_snapshot_heads")

    resolved = Repository(stripped).resolve(snap.snapshot_id)
    assert resolved.snapshot_id == snap.snapshot_id
    assert resolved.manifest_hash == snap.manifest_hash
    stripped.close()


# --------------------------------------------------------------------------
# a fixed reader while another connection advances the head
# --------------------------------------------------------------------------


def test_resolved_ref_fixed_while_another_connection_advances_the_head(tmp_path):
    conn, clock = _catalog(tmp_path)
    record_a = _record_for("2024")
    _, snap_a = _commit(conn, clock, [record_a], receipt_id="ra", scope="shadow")
    resolved_before = Repository(conn).resolve(snap_a.snapshot_id)

    store = ArtifactStore(tmp_path / "artifacts")
    ref_before = resolve_snapshot_head(conn, store, "shadow", clock=clock)
    content_before = json.loads(store.read_verified(ref_before).decode())
    assert content_before["snapshot_id"] == snap_a.snapshot_id

    conn2 = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    record_b = _record_for("2025")
    manifest_ab, snap_b = _manifest_and_snapshot([record_a, record_b])
    commit_snapshot(
        conn2, scope="shadow", request_hash=_hash("b-request"), contracts=[_SEC_CONTRACT],
        objects=[record_a.object_ref, record_b.object_ref], records=[record_a, record_b],
        manifests=[manifest_ab], snapshot=snap_b, expected_head_snapshot_id=snap_a.snapshot_id,
        expected_head_generation=1, receipt_id="rb", attempt_id="att-1", fence=1,
        fence_check=_noop_fence, clock=clock)

    # A's already-resolved ref is unchanged, and re-resolving A is identical.
    resolved_after = Repository(conn).resolve(snap_a.snapshot_id)
    assert resolved_after == resolved_before

    # resolve_snapshot_head now returns B's artifact.
    ref_after = resolve_snapshot_head(conn, store, "shadow", clock=clock)
    assert ref_after != ref_before
    content_after = json.loads(store.read_verified(ref_after).decode())
    assert content_after["snapshot_id"] == snap_b.snapshot_id
    conn2.close()


def test_resolve_snapshot_head_repeat_calls_same_ref_over_an_unchanged_head(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _commit(conn, clock, [record], receipt_id="r1", scope="shadow")
    store = ArtifactStore(tmp_path / "artifacts")

    ref1 = resolve_snapshot_head(conn, store, "shadow", clock=clock)
    ref2 = resolve_snapshot_head(conn, store, "shadow", clock=clock)
    assert ref1 == ref2


def test_resolve_snapshot_head_no_committed_head_raises(tmp_path):
    conn, _clock = _catalog(tmp_path)
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises(DataError) as err:
        resolve_snapshot_head(conn, store, "shadow", clock=_clock)
    assert err.value.code == "SNAPSHOT_NOT_READY"


# --------------------------------------------------------------------------
# Coverage ratchet fix (2026-09-15): resolve_full/latest_dataset_version/
# table_contract/fragment_records and the private per-row corruption checks
# were entirely untested under the Phase 2 fixed suite even though
# Repository is already in scope here (D10).
# --------------------------------------------------------------------------


def test_read_only_refuses_a_nested_transaction(tmp_path):
    conn, _clock = _catalog(tmp_path)
    conn.execute("BEGIN")
    with pytest.raises(RuntimeError):
        Repository(conn).resolve("snap_" + "0" * 32)
    conn.execute("ROLLBACK")


def test_private_row_lookups_refuse_unknown_ids(tmp_path):
    """Direct calls into Repository's own per-row corruption checks --
    ``resolve``/``resolve_full`` reach these only through a fully corrupted
    catalog, so this exercises each refusal in isolation instead."""
    conn, _clock = _catalog(tmp_path)
    repo = Repository(conn)
    with pytest.raises(DataError) as err:
        repo._manifest(conn, "dsv_" + "0" * 32)
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        repo._contract_ref(conn, "contract_" + "0" * 32)
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        repo._full_contract(conn, "contract_" + "0" * 32)
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        repo._record(conn, {"object_id": "obj_" + "0" * 32}, _SEC_CONTRACT)
    assert err.value.code == "MANIFEST_CORRUPT"


def test_resolve_full_happy_path_and_unknown_id(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")

    resolved = Repository(conn).resolve_full(snap.snapshot_id)
    assert resolved.snapshot == Repository(conn).resolve(snap.snapshot_id)
    assert len(resolved.records) == 1
    assert resolved.records[0].fragment_id == record.fragment_id
    assert len(resolved.contracts) == 1
    assert len(resolved.objects) == 1

    with pytest.raises(DataError) as err:
        Repository(conn).resolve_full("snap_" + "0" * 32)
    assert err.value.code == "SNAPSHOT_NOT_FOUND"


def test_resolve_and_resolve_full_refuse_a_tampered_snapshot_row(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")
    conn.close()

    copy_path = tmp_path / "tampered.sqlite"
    _consistent_copy(tmp_path / "catalog.sqlite", copy_path)
    tampered = ops_connect(copy_path)
    tampered.execute("DROP TRIGGER data_snapshots_no_delete")
    tampered.execute("DROP TRIGGER data_snapshots_no_update")
    tampered.execute("UPDATE data_snapshots SET manifest_hash = ? WHERE snapshot_id = ?",
                     ("sha256:" + "0" * 64, snap.snapshot_id))

    with pytest.raises(DataError) as err:
        Repository(tampered).resolve(snap.snapshot_id)
    assert err.value.code == "MANIFEST_CORRUPT"

    with pytest.raises(DataError) as err:
        Repository(tampered).resolve_full(snap.snapshot_id)
    assert err.value.code == "MANIFEST_CORRUPT"
    tampered.close()


def test_latest_dataset_version_empty_and_populated(tmp_path):
    conn, clock = _catalog(tmp_path)
    repo = Repository(conn)
    manifest, records = repo.latest_dataset_version(_SEC_CONTRACT.contract_id)
    assert manifest is None
    assert records == ()

    record = _record_for("2024")
    _commit(conn, clock, [record], receipt_id="r1")
    manifest, records = repo.latest_dataset_version(_SEC_CONTRACT.contract_id)
    assert manifest is not None
    assert len(records) == 1
    assert records[0].fragment_id == record.fragment_id


def test_table_contract_and_fragment_records_refuse_unknown_table(tmp_path):
    conn, clock = _catalog(tmp_path)
    record = _record_for("2024")
    _, snap = _commit(conn, clock, [record], receipt_id="r1")
    repo = Repository(conn)

    contract = repo.table_contract(snap, "securities")
    assert contract.contract_id == _SEC_CONTRACT.contract_id
    records = repo.fragment_records(snap, "securities")
    assert len(records) == 1

    with pytest.raises(DataError) as err:
        repo.table_contract(snap, "not_a_real_table")
    assert err.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError) as err:
        repo.fragment_records(snap, "not_a_real_table")
    assert err.value.code == "CONTRACT_MISMATCH"


