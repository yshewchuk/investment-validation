"""D11: a crash at every §7.3 commit boundary leaves the old or the new
complete snapshot — never a mixed one — and a fault-free retry completes it.

One ``_FaultSwitch`` instance is armed with exactly one target point name and
passed as the ``fault`` callable to every layer that accepts one:
``foundation.ArtifactStore`` (its own ``"copied"``/``"linked"`` points, fired
after the object's bytes are fsynced and after it is durably linked into the
store), ``objects.publish_legacy_file``/``inspect_fragment`` (the remaining
object-side points), and ``catalog.commit_snapshot`` (the ten catalog-side
points, task brief decision 2). A mismatched point name is a silent no-op, so
the same switch is safe to wire through every call, including snapshot A's
own fault-free setup.

Fifteen points in total: five object-side (``before_copy``, ``during_copy``,
``copied`` — after object fsync — ``linked`` — after object publication —
``during_inspection``) plus the ten named catalog-side points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import ArtifactRef  # noqa: E402
from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.catalog import commit_snapshot  # noqa: E402
from engine.v2.data.objects import (  # noqa: E402
    PARQUET_FRAGMENT_SCHEMA_REF,
    inspect_fragment,
    publish_legacy_file,
)
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import CONTENT_HASH_PREFIX, ArtifactStore  # noqa: E402
from tests.test_v2_data_commit import (  # noqa: E402
    _catalog,
    _hash,
    _manifest_and_snapshot,
    _noop_fence,
)
from tests.test_v2_data_manifests import (  # noqa: E402
    _SEC_CONTRACT,
    _SEC_REF,
    IRH_A,
    RECEIPT_A,
    _securities_rows,
    _table_from_rows,
    _to_bytes,
)
from tests.test_v2_data_objects import _write_source  # noqa: E402

OBJECT_SIDE_POINTS = ("before_copy", "during_copy", "copied", "linked", "during_inspection")
CATALOG_SIDE_POINTS = ("before_transaction", "after_contracts", "after_objects", "after_fragments",
                       "after_dataset_versions", "after_memberships", "after_snapshot",
                       "after_snapshot_tables", "before_head_update", "before_commit")
FAULT_POINTS = OBJECT_SIDE_POINTS + CATALOG_SIDE_POINTS


class _Crash(Exception):
    pass


class _FaultSwitch:
    """One callable, reused across every fault hook; fires only for its own point."""

    def __init__(self) -> None:
        self.point: str | None = None

    def __call__(self, name: str) -> None:
        if name == self.point:
            raise _Crash(name)


def _prepare_record(store: ArtifactStore, source_root: Path, year: int, switch):
    data = _to_bytes(_table_from_rows(_SEC_CONTRACT, _securities_rows(year)))
    file_ref = _write_source(source_root, f"securities/{year}.parquet", data)
    obj = publish_legacy_file(store, f"att-{year}", source_root, file_ref, fault=switch)
    inspection = inspect_fragment(store, obj, _SEC_CONTRACT, _SEC_REF, str(year), fault=switch)
    return manifests.fragment_record(inspection, _SEC_REF, input_receipt_refs=(RECEIPT_A,),
                                     import_request_hash=IRH_A)


def _artifact_ref(object_id: str, content_hash: str, byte_size: int) -> ArtifactRef:
    digest = content_hash.removeprefix(CONTENT_HASH_PREFIX)
    return ArtifactRef(artifact_id=object_id, content_hash=content_hash,
                       schema_ref=PARQUET_FRAGMENT_SCHEMA_REF, byte_size=byte_size,
                       storage_key=f"objects/{digest[:2]}/{digest}")


def _assert_every_cataloged_object_verifies(conn, store: ArtifactStore) -> None:
    for row in conn.execute("SELECT object_id, content_hash, byte_size FROM data_objects"):
        store.verify(_artifact_ref(row["object_id"], row["content_hash"], row["byte_size"]))


def _assert_no_orphan_snapshot_tables(conn) -> None:
    orphans = conn.execute(
        "SELECT 1 FROM data_snapshot_tables st LEFT JOIN data_snapshots s "
        "ON s.snapshot_id = st.snapshot_id WHERE s.snapshot_id IS NULL").fetchall()
    assert orphans == []


def _assert_every_snapshot_resolves(conn) -> None:
    for (snapshot_id,) in conn.execute("SELECT snapshot_id FROM data_snapshots").fetchall():
        Repository(conn).resolve(snapshot_id)


def _commit(conn, clock, contracts, objects, records, manifests_, snapshot, *, receipt_id,
           attempt_id, expected_head_snapshot_id, expected_head_generation, fault=None):
    return commit_snapshot(
        conn, scope="shadow", request_hash=_hash(f"{receipt_id}-request"), contracts=contracts,
        objects=objects, records=records, manifests=manifests_, snapshot=snapshot,
        expected_head_snapshot_id=expected_head_snapshot_id,
        expected_head_generation=expected_head_generation, receipt_id=receipt_id,
        attempt_id=attempt_id, fence=1, fence_check=_noop_fence, clock=clock, fault=fault)


@pytest.mark.parametrize("point", FAULT_POINTS)
def test_fault_at_every_commit_boundary_leaves_old_or_new_complete(tmp_path, point):
    conn, clock = _catalog(tmp_path)
    switch = _FaultSwitch()
    store = ArtifactStore(tmp_path / "objects", fault=switch)
    source_root = tmp_path / "source"

    record_a = _prepare_record(store, source_root, 2024, switch)
    manifest_a, snap_a = _manifest_and_snapshot([record_a])
    _commit(conn, clock, [_SEC_CONTRACT], [record_a.object_ref], [record_a], [manifest_a], snap_a,
           receipt_id="ra", attempt_id="att-a", expected_head_snapshot_id=None,
           expected_head_generation=0)

    switch.point = point
    with pytest.raises(_Crash):
        record_b = _prepare_record(store, source_root, 2025, switch)
        manifest_ab, snap_b = _manifest_and_snapshot([record_a, record_b])
        _commit(conn, clock, [_SEC_CONTRACT], [record_a.object_ref, record_b.object_ref],
               [record_a, record_b], [manifest_ab], snap_b, receipt_id="rb", attempt_id="att-b",
               expected_head_snapshot_id=snap_a.snapshot_id, expected_head_generation=1,
               fault=switch)
    switch.point = None

    # The head resolves to complete A; nothing mixed or half-written survives.
    head = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()
    assert head["snapshot_id"] == snap_a.snapshot_id
    assert head["generation"] == 1
    resolved_a = Repository(conn).resolve(snap_a.snapshot_id)
    assert resolved_a.snapshot_id == snap_a.snapshot_id
    _assert_every_cataloged_object_verifies(conn, store)
    _assert_no_orphan_snapshot_tables(conn)
    _assert_every_snapshot_resolves(conn)

    # A fault-free retry commits B; the head now resolves to complete B.
    record_b2 = _prepare_record(store, source_root, 2025, switch)
    manifest_ab2, snap_b2 = _manifest_and_snapshot([record_a, record_b2])
    _commit(conn, clock, [_SEC_CONTRACT], [record_a.object_ref, record_b2.object_ref],
           [record_a, record_b2], [manifest_ab2], snap_b2, receipt_id="rb", attempt_id="att-b",
           expected_head_snapshot_id=snap_a.snapshot_id, expected_head_generation=1)
    head2 = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()
    assert head2["snapshot_id"] == snap_b2.snapshot_id
    assert head2["generation"] == 2
    resolved_b = Repository(conn).resolve(snap_b2.snapshot_id)
    assert resolved_b.snapshot_id == snap_b2.snapshot_id
    _assert_every_cataloged_object_verifies(conn, store)
    _assert_no_orphan_snapshot_tables(conn)
    _assert_every_snapshot_resolves(conn)
