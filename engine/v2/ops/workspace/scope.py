"""Workspace-local scope view over the workspace's immutable snapshot pin."""
from __future__ import annotations

from engine.v2.data import errors as data_errors
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail

SCOPE_HEAD_VERSION = "research_workspace_scope_head.v1.0"
_SCOPE_PREFIX = "size-scope-head:"


def _refuse(reason: str, **details):
    return fail("WORKSPACE_REFUSED", "research workspace scope refused",
                details={"reason": reason, **details})


class WorkspaceScopeRepository:
    """Research repository view whose one logical scope is bound to a workspace pin."""

    def __init__(self, workspace, scope: str, snapshot_id: str, manifest_hash: str,
                 binding_id: str):
        self._workspace = workspace
        self._scope = scope
        self._snapshot_id = snapshot_id
        self._manifest_hash = manifest_hash
        self.binding_id = binding_id

    def resolve_pinned(self, scope: str):
        """Resolve only the scope recorded in this view, and recheck the workspace pin."""
        if type(scope) is not str or scope != self._scope:
            raise _refuse("scope_mismatch")
        snapshot = self._workspace.source_snapshot()
        if (snapshot.snapshot_id != self._snapshot_id
                or snapshot.manifest_hash != self._manifest_hash):
            raise data_errors.fail(
                "SNAPSHOT_CLOSURE_BROKEN", "workspace scope pin changed",
                details={"reason": "pin_mismatch", "snapshot_id": self._snapshot_id})
        return snapshot

    def __getattr__(self, name: str):
        """Delegate the source's existing allowlisted read methods only."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._workspace.source, name)


def size_scope_repository(workspace, scope: str, *, clock) -> WorkspaceScopeRepository:
    """Persist and return a scope binding to this workspace's pinned source snapshot.

    The binding is a workspace read pin, not a native catalog head: the isolated
    workspace catalog has no production ``data_snapshots`` rows to reference.
    """
    if (type(scope) is not str or not scope or scope != scope.strip()
            or len(scope) > 100 or "\x00" in scope):
        raise _refuse("scope_invalid")
    if (workspace is None or not callable(getattr(workspace, "source_snapshot", None))
            or not callable(getattr(workspace, "record_read_pin", None))
            or not getattr(workspace, "snapshot_id", None)
            or not getattr(workspace, "manifest_hash", None)):
        raise _refuse("workspace_invalid")

    snapshot = workspace.source_snapshot()
    if (snapshot.snapshot_id != workspace.snapshot_id
            or snapshot.manifest_hash != workspace.manifest_hash):
        raise data_errors.fail(
            "SNAPSHOT_CLOSURE_BROKEN", "workspace scope pin changed",
            details={"reason": "pin_mismatch", "snapshot_id": workspace.snapshot_id})
    fields = {"schema_version": SCOPE_HEAD_VERSION, "scope": scope,
              "snapshot_id": snapshot.snapshot_id, "manifest_hash": snapshot.manifest_hash}
    binding_id = content_hash(fields)
    document = {**fields, "binding_id": binding_id}
    name = _SCOPE_PREFIX + scope
    try:
        workspace.record_read_pin(name, document, clock=clock)
    except (OpsError, data_errors.DataError):
        raise
    except Exception as exc:
        raise _refuse("storage_failure", cause=type(exc).__name__) from exc
    return WorkspaceScopeRepository(workspace, scope, snapshot.snapshot_id,
                                    snapshot.manifest_hash, binding_id)