"""Create the workspace catalog's native scope head over the pinned snapshot."""
from __future__ import annotations

import sys
import time

from engine.v2.data import errors as data_errors
from engine.v2.data import reference_catalog
from engine.v2.data.catalog import commit_snapshot
from engine.v2.foundation import content_hash
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.workspace.catalog_rows import copy_closure_rows
from engine.v2.ops.workspace.scope import validate_size_scope_request

_MAX_REFERENCE_INPUTS = 10_000
_MAX_REFERENCE_BYTES = 16 << 20


def _refuse(reason: str, **details):
    return fail("WORKSPACE_REFUSED", "research workspace scope head refused",
                details={"reason": reason, **details})


def _log(message: str) -> None:
    print("[workspace] " + message, file=sys.stderr, flush=True)


def _head_check(scope: str, snapshot_id: str):
    """Run first inside the commit transaction, so a receipt replay cannot hide a moved head."""
    def check(conn) -> None:
        row = conn.execute("SELECT snapshot_id FROM data_snapshot_heads WHERE scope = ?",
                           (scope,)).fetchone()
        if row is not None and row[0] != snapshot_id:
            raise data_errors.fail("SNAPSHOT_CONFLICT", "scope head is not the pinned snapshot",
                                   details={"scope": scope, "snapshot_id": snapshot_id})
    return check


def _copy_original_receipt(workspace, receipt_id: str, snapshot_id: str) -> None:
    """Copy the snapshot's own import receipt so ``commit_receipt_ref`` resolves in the workspace."""
    source = workspace._source_conn.execute(
        "SELECT * FROM data_import_receipts WHERE receipt_id = ?", (receipt_id,)).fetchone()
    if source is None or tuple(source)[4] != snapshot_id or tuple(source)[5] != "committed":
        raise _refuse("receipt_missing", snapshot_id=snapshot_id)
    with transaction(workspace.conn):
        workspace.conn.execute(
            "INSERT OR IGNORE INTO data_import_receipts VALUES (%s)" % ",".join("?" * len(source)),
            tuple(source))
        stored = workspace.conn.execute(
            "SELECT * FROM data_import_receipts WHERE receipt_id = ?", (receipt_id,)).fetchone()
        if tuple(stored) != tuple(source):
            raise _refuse("row_mismatch", table="data_import_receipts")


def commit_scope_head(workspace, scope: str, *, clock) -> dict:
    """Head ``scope`` at the pinned snapshot with its receipt's reference inputs, in one commit."""
    started = time.monotonic()
    validate_size_scope_request(workspace, scope)
    try:
        _log(f"stage=head start scope={scope}")
        copy_closure_rows(workspace)
        pinned = workspace.source_snapshot()
        sid, manifest_hash = pinned.snapshot_id, pinned.manifest_hash
        row = workspace.conn.execute(
            "SELECT commit_receipt_ref FROM data_snapshots WHERE snapshot_id = ?",
            (sid,)).fetchone()
        if row is None or not row[0]:
            raise _refuse("receipt_missing", snapshot_id=sid)
        count, size = workspace._source_conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(length(legacy_path) + length(kind) + "
            "length(object_id) + length(content_hash)), 0) FROM (SELECT * FROM "
            "data_import_reference_inputs WHERE receipt_id = ? LIMIT ?)",
            (row[0], _MAX_REFERENCE_INPUTS + 1)).fetchone()
        if count > _MAX_REFERENCE_INPUTS or size > _MAX_REFERENCE_BYTES:
            raise _refuse("reference_inputs_over_cap", receipt_id=row[0])
        inputs = reference_catalog.reference_inputs_for_receipt(
            workspace._source_conn, receipt_id=row[0])
        resolved = workspace.source.resolve_full(sid)
        if (resolved.snapshot.snapshot_id != sid
                or resolved.snapshot.manifest_hash != manifest_hash):
            raise _refuse("pin_mismatch", snapshot_id=sid)
        _copy_original_receipt(workspace, row[0], sid)
        identity = {"workspace_scope_head": scope, "snapshot_id": sid,
                    "manifest_hash": manifest_hash}
        request_hash = content_hash(identity)
        receipt_id = "size-head-" + request_hash.split(":")[-1][:32]
        receipt = commit_snapshot(
            workspace.conn, scope=scope, request_hash=request_hash,
            contracts=resolved.contracts, objects=resolved.objects,
            records=resolved.records, manifests=list(resolved.table_manifests.values()),
            snapshot=resolved.snapshot, expected_head_snapshot_id=None,
            expected_head_generation=0, receipt_id=receipt_id, attempt_id=receipt_id,
            fence=1, fence_check=_head_check(scope, sid), clock=clock, store=workspace.store,
            record_references=lambda conn, rid: reference_catalog.insert_reference_inputs(
                conn, rid, inputs),
            audit_partitions=False)
        _log(f"stage=head done scope={scope} snapshot={sid} reference_inputs={len(inputs)} "
             f"{time.monotonic() - started:.2f}s elapsed")
        return {"scope": scope, "snapshot_id": sid, "manifest_hash": manifest_hash,
                "receipt_id": receipt_id,
                "generation": receipt.resulting_head_generation,
                "reference_inputs": len(inputs)}
    except (OpsError, data_errors.DataError):
        raise
    except Exception as exc:
        raise _refuse("storage_failure", cause=type(exc).__name__) from exc
