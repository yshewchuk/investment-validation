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


def _resolve_snapshot(workspace):
    """Keep unexpected workspace read/storage failures inside the typed boundary."""
    try:
        return workspace.source_snapshot()
    except (OpsError, data_errors.DataError):
        raise
    except Exception as exc:
        raise _refuse("storage_failure", cause=type(exc).__name__) from exc


class WorkspaceScopeRepository:
    """Research repository view whose one logical scope is bound to a workspace pin."""

    def __init__(self, workspace, scope: str, snapshot_id: str, manifest_hash: str,
                 binding_id: str, read_pin_ref):
        self._workspace = workspace
        self._scope = scope
        self._snapshot_id = snapshot_id
        self._manifest_hash = manifest_hash
        self.binding_id = binding_id
        self.read_pin_ref = read_pin_ref

    def resolve_pinned(self, scope: str):
        """Resolve only this scope and keep it bound to the captured workspace pin."""
        if type(scope) is not str or scope != self._scope:
            raise _refuse("scope_mismatch")
        snapshot = _resolve_snapshot(self._workspace)
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


def _validate_scope(scope: str) -> None:
    if (type(scope) is not str or not scope or scope != scope.strip()
            or len(scope) > 100 or "\x00" in scope):
        raise _refuse("scope_invalid")
    try:
        scope.encode("utf-8")
    except UnicodeEncodeError:
        raise _refuse("scope_invalid") from None


def _valid_workspace(workspace) -> bool:
    try:
        return bool(
            workspace is not None
            and callable(getattr(workspace, "source_snapshot", None))
            and callable(getattr(workspace, "record_read_pin", None))
            and callable(getattr(getattr(workspace, "source", None), "resolve", None))
            and getattr(workspace, "snapshot_id", None)
            and getattr(workspace, "manifest_hash", None))
    except Exception:
        return False


def validate_size_scope_request(workspace, scope: str) -> None:
    _validate_scope(scope)
    if not _valid_workspace(workspace):
        raise _refuse("workspace_invalid")


def resolve_size_scope_snapshot(workspace, scope: str):
    """Validate a workspace scope before resolving its pinned snapshot; write no read pin."""
    validate_size_scope_request(workspace, scope)
    return _resolve_snapshot(workspace)


def size_scope_repository(workspace, scope: str, *, clock) -> WorkspaceScopeRepository:
    """Persist and return a scope binding to this workspace's pinned source snapshot.

    The binding is a workspace read pin, not a native catalog head: the isolated
    workspace catalog has no production ``data_snapshots`` rows to reference.
    """
    # The shared resolver validates before source access and keeps storage errors typed.
    snapshot = resolve_size_scope_snapshot(workspace, scope)
    fields = {"schema_version": SCOPE_HEAD_VERSION, "scope": scope,
              "snapshot_id": snapshot.snapshot_id, "manifest_hash": snapshot.manifest_hash}
    binding_id = content_hash(fields)
    document = {**fields, "binding_id": binding_id}
    name = _SCOPE_PREFIX + scope
    try:
        read_pin_ref = workspace.record_read_pin(name, document, clock=clock)
    except (OpsError, data_errors.DataError):
        raise
    except Exception as exc:
        raise _refuse("storage_failure", cause=type(exc).__name__) from exc
    return WorkspaceScopeRepository(workspace, scope, snapshot.snapshot_id,
                                    snapshot.manifest_hash, binding_id, read_pin_ref)