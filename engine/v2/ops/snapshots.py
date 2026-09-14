"""Exact head resolution pinned as an artifact, and the real-fence commit wrapper — §8.1, §7.3.

``resolve_snapshot_head`` is the ops-side companion to
``engine.v2.data.repository.Repository.resolve``: it reads exactly one row
from ``data_snapshot_heads``, resolves that snapshot id through the data
layer (never trusting the head row itself as evidence — ``Repository.resolve``
re-verifies every manifest hash from immutable rows), publishes the complete,
verified ``SnapshotRef`` as a content-addressed Phase 1 artifact, and
registers it — the same publish-then-register shape ``ops/plans.py::save_plan``
already uses. A caller places the returned ``ArtifactRef.artifact_id`` in
``JobSpec.input_refs`` to pin a job to one immutable document instead of a
live, movable head; that job-planning step belongs to ``ops/nightly.py``
(another slice), not here. Publishing is content-addressed and
``register_artifact`` is itself idempotent (same content, same id, no second
row), so repeat calls over an unchanged head return the identical
``ArtifactRef``.

``commit_snapshot_for_attempt`` is ``engine.v2.data.catalog.commit_snapshot``
with the real Phase 1 ``verify_fence`` supplied as its injected
``fence_check`` — the data layer cannot import ops, so ``commit_snapshot``
takes a fence-check callable instead (task brief decision 2).
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence

from engine.v2.contracts import (
    ArtifactRef,
    DatasetManifest,
    FragmentRecord,
    ObjectRef,
    SnapshotRef,
    TableContract,
)
from engine.v2.data.catalog import commit_snapshot
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, Clock
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.lifecycle import verify_fence

__all__ = ["commit_snapshot_for_attempt", "resolve_snapshot_head"]

SNAPSHOT_REF_SCHEMA_REF = "snapshot_ref.v1.0"


def resolve_snapshot_head(conn: sqlite3.Connection, store: ArtifactStore, scope: str, *,
                          clock: Clock) -> ArtifactRef:
    """Read one head, resolve it exactly, publish and register the ``SnapshotRef``."""
    row = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope = ?",
                       (scope,)).fetchone()
    if row is None:
        raise data_fail("SNAPSHOT_NOT_READY", "scope has no committed head", details={"scope": scope})
    snapshot = Repository(conn).resolve(row["snapshot_id"])
    ref = store.publish_bytes(dumps(snapshot).encode("utf-8"), schema_ref=SNAPSHOT_REF_SCHEMA_REF)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def commit_snapshot_for_attempt(conn: sqlite3.Connection, store: ArtifactStore, *, scope: str,
                                request_hash: str, contracts: Sequence[TableContract],
                                objects: Sequence[ObjectRef], records: Sequence[FragmentRecord],
                                manifests: Sequence[DatasetManifest], snapshot: SnapshotRef,
                                expected_head_snapshot_id: str | None, expected_head_generation: int,
                                receipt_id: str, attempt_id: str, fence: int, clock: Clock,
                                fault: Callable[[str], None] | None = None,
                                record_references: Callable[[sqlite3.Connection, str], None]
                                | None = None):
    """``commit_snapshot`` under the real Phase 1 fence, not a test double.

    ``store`` lets pre-transaction verification re-stream a multi-fragment
    partition's objects (``manifests.verify_partition_hashes``).
    ``record_references`` inserts the import's reference inputs inside the
    commit transaction (``engine.v2.data.reference_catalog``).
    """
    return commit_snapshot(
        conn, scope=scope, request_hash=request_hash, contracts=contracts, objects=objects,
        records=records, manifests=manifests, snapshot=snapshot,
        expected_head_snapshot_id=expected_head_snapshot_id,
        expected_head_generation=expected_head_generation, receipt_id=receipt_id,
        attempt_id=attempt_id, fence=fence,
        fence_check=lambda c: verify_fence(c, attempt_id, fence, clock.now()), clock=clock,
        fault=fault, store=store, record_references=record_references)
