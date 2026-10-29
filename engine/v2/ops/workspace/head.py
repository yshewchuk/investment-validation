"""Create the workspace catalog's native scope head over the pinned snapshot."""
from __future__ import annotations

import sys
import time

from engine.v2.data import errors as data_errors
from engine.v2.data import reference_catalog
from engine.v2.data.catalog import commit_snapshot
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.workspace.catalog_rows import copy_closure_rows
from engine.v2.ops.workspace.scope import validate_size_scope_request


def _refuse(reason: str, **details):
    return fail("WORKSPACE_REFUSED", "research workspace scope head refused",
                details={"reason": reason, **details})


def _log(message: str) -> None:
    print("[workspace] " + message, file=sys.stderr, flush=True)


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
        inputs = reference_catalog.reference_inputs_for_receipt(
            workspace._source_conn, receipt_id=row[0])
        resolved = workspace.source.resolve_full(sid)
        if (resolved.snapshot.snapshot_id != sid
                or resolved.snapshot.manifest_hash != manifest_hash):
            raise _refuse("pin_mismatch", snapshot_id=sid)
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
            fence=1, fence_check=lambda _conn: None, clock=clock, store=workspace.store,
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
