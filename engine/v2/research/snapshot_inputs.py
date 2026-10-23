"""Snapshot-reference shape and exact committed-identity admission for research."""
from engine.v2.contracts.data import SnapshotRef
from engine.v2.data import errors


def snapshot_reference(snapshot):
    """Validate native reference shape only; this does not prove commitment."""
    if (type(snapshot) is not SnapshotRef or type(snapshot.snapshot_id) is not str
            or not snapshot.snapshot_id.strip()):
        raise errors.fail("SNAPSHOT_UNRESOLVED", "an exact native snapshot reference is required")
    return snapshot


def admit_snapshot(repository, snapshot):
    """Resolve and compare the full reference; preserve native integrity errors."""
    snapshot_reference(snapshot)
    try:
        resolved = repository.resolve(snapshot.snapshot_id)
    except errors.DataError as exc:
        if exc.code != "SNAPSHOT_NOT_FOUND":
            raise
        raise errors.fail("SNAPSHOT_UNRESOLVED", "the requested snapshot cannot be resolved") from None
    if resolved != snapshot:
        raise errors.fail("SNAPSHOT_UNRESOLVED", "snapshot differs from its committed identity")
    return snapshot
