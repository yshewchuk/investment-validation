#!/usr/bin/env python3
"""D11 fault matrix: a crash at every §7.3 commit boundary, on a synthetic root.

Reuses ``tests/test_v2_data_atomicity.py``'s own fixture helpers (the exact
same ``_FaultSwitch``/``_prepare_record``/``_commit`` machinery that file's
parametrized test already proves against every point) rather than re-deriving
a second synthetic legacy root here. For each of the fifteen §7.3 points: a
real snapshot A commits cleanly, then a real snapshot B's commit is crashed at
that point, and this records whether the catalog head still resolves to A
("old_head") and whether every cataloged object still verifies.

No real data, no legacy tree, no network: the catalog and object store are
built fresh under a private temp directory per point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.foundation import ArtifactStore  # noqa: E402
from tests.test_v2_data_atomicity import (  # noqa: E402
    FAULT_POINTS,
    _assert_every_cataloged_object_verifies,
    _assert_every_snapshot_resolves,
    _assert_no_orphan_snapshot_tables,
    _commit,
    _Crash,
    _FaultSwitch,
    _prepare_record,
)
from tests.test_v2_data_commit import _catalog, _manifest_and_snapshot  # noqa: E402
from tests.test_v2_data_manifests import _SEC_CONTRACT  # noqa: E402

SCHEMA_VERSION = "fault_matrix.v1.0"


def _verified_objects(conn, store) -> bool:
    try:
        _assert_every_cataloged_object_verifies(conn, store)
        _assert_no_orphan_snapshot_tables(conn)
        _assert_every_snapshot_resolves(conn)
        return True
    except AssertionError:
        return False


def _run_point(point_root: Path, point: str) -> dict:
    point_root.mkdir(parents=True, exist_ok=True)
    conn, clock = _catalog(point_root)
    switch = _FaultSwitch()
    store = ArtifactStore(point_root / "objects", fault=switch)
    source_root = point_root / "source"

    record_a = _prepare_record(store, source_root, 2024, switch)
    manifest_a, snap_a = _manifest_and_snapshot([record_a])
    _commit(conn, clock, [_SEC_CONTRACT], [record_a.object_ref], [record_a], [manifest_a], snap_a,
           receipt_id="ra", attempt_id="att-a", expected_head_snapshot_id=None,
           expected_head_generation=0)

    switch.point = point
    crashed = False
    try:
        record_b = _prepare_record(store, source_root, 2025, switch)
        manifest_ab, snap_b = _manifest_and_snapshot([record_a, record_b])
        _commit(conn, clock, [_SEC_CONTRACT], [record_a.object_ref, record_b.object_ref],
               [record_a, record_b], [manifest_ab], snap_b, receipt_id="rb", attempt_id="att-b",
               expected_head_snapshot_id=snap_a.snapshot_id, expected_head_generation=1,
               fault=switch)
    except _Crash:
        crashed = True
    switch.point = None

    if not crashed:
        raise RuntimeError(f"fault point {point!r} never fired — name/hook drift")
    head = conn.execute(
        "SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'").fetchone()
    outcome = "old_head" if head["snapshot_id"] == snap_a.snapshot_id else "new_head"
    return {"point": point, "outcome": outcome, "verified_objects": _verified_objects(conn, store)}


def build(scratch_root: Path | None = None) -> list[dict]:
    if scratch_root is None:
        with tempfile.TemporaryDirectory(prefix="phase2-fault-matrix-") as scratch:
            return [_run_point(Path(scratch) / point, point) for point in FAULT_POINTS]
    return [_run_point(scratch_root / point, point) for point in FAULT_POINTS]


def publish(rows: list[dict], artifact_root: Path) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(rows, indent=2, sort_keys=True).encode()
    path = artifact_root / "fault_matrix.json"
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = build()
    ref = publish(rows, args.artifact_root)
    print(json.dumps({**ref, "points": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
