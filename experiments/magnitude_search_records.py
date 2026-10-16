"""Immutable search-owned journal using the existing catalog/artifact primitives."""
import json

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import ArtifactError, canonical_json, from_document, to_document
from engine.v2.foundation.artifacts import safe_relative_path
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.migrations import Migration
from experiments.magnitude_search_errors import artifact_failure

OWNER = "magnitude_search"
SCHEMA = "magnitude_search_record.v1"
MAX_BYTES = 16 * 1024 * 1024
MIGRATIONS = (Migration(version=1, name="immutable_trial_records", statements=(
    """CREATE TABLE magnitude_search_records (
        run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
        slot TEXT NOT NULL, ref_json TEXT NOT NULL, PRIMARY KEY(run_id, slot)
    ) STRICT""",
    """CREATE TRIGGER magnitude_search_no_update BEFORE UPDATE ON magnitude_search_records
    BEGIN SELECT RAISE(ABORT, 'magnitude search records are immutable'); END""",
    """CREATE TRIGGER magnitude_search_no_delete BEFORE DELETE ON magnitude_search_records
    BEGIN SELECT RAISE(ABORT, 'magnitude search records are immutable'); END""",
    """CREATE TRIGGER magnitude_search_no_replace BEFORE INSERT ON magnitude_search_records
    WHEN EXISTS (SELECT 1 FROM magnitude_search_records WHERE run_id=NEW.run_id AND slot=NEW.slot)
    BEGIN SELECT RAISE(ABORT, 'magnitude search records are immutable'); END""",
)),)


def _artifact_failure(error, action, *, unsafe_is_conflict=True):
    message = "search journal artifact could not be " + action
    if isinstance(error, OSError):
        error = ArtifactError("IO_ERROR", message)
    return artifact_failure(error, integrity_message=message, resource_message=message,
                            unsafe_is_conflict=unsafe_is_conflict)


def _reference(conn, run_id, slot):
    row = conn.execute("SELECT CASE WHEN length(CAST(ref_json AS BLOB))<=4096 THEN ref_json END "
                       "FROM magnitude_search_records WHERE run_id=? AND slot=?", (run_id, slot)).fetchone()
    if row is None:
        return None
    try:
        ref = from_document(ArtifactRef, json.loads(row[0]))
        if ref.schema_ref != SCHEMA or not 0 < ref.byte_size <= MAX_BYTES:
            raise ValueError
        safe_relative_path(ref.storage_key)
        return ref
    except (ArtifactError, ValueError, TypeError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal reference is invalid") from None


def read_record(conn, store, run_id, slot):
    ref = _reference(conn, run_id, slot)
    if ref is None:
        return None
    try:
        payload = store.read_verified(ref)
    except (ArtifactError, OSError) as error:
        raise _artifact_failure(error, "verified", unsafe_is_conflict=False) from None
    try:
        record = json.loads(payload)
        if (set(record) != {"schema_version", "run_id", "slot", "body"}
                or record["schema_version"] != SCHEMA or record["run_id"] != run_id or record["slot"] != slot):
            raise ValueError
        return record["body"]
    except (ValueError, TypeError, KeyError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal binding is invalid") from None


def append_record(conn, store, run_id, slot, body, *, replay=True, absent=()):
    """Publish then serialize the unique association; never do artifact I/O under the lock."""
    previous = read_record(conn, store, run_id, slot)
    if previous is not None:
        if not replay or canonical_json(previous) != canonical_json(body):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed")
        return previous
    payload = canonical_json({"schema_version": SCHEMA, "run_id": run_id, "slot": slot, "body": body}).encode()
    if len(payload) > MAX_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "search journal record exceeds its byte limit")
    try:
        ref = store.publish_bytes(payload, schema_ref=SCHEMA)
    except (ArtifactError, OSError) as error:
        raise _artifact_failure(error, "published") from None
    with transaction(conn):
        if any(_reference(conn, run_id, other) is not None for other in absent):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial phase is already committed")
        previous_ref = _reference(conn, run_id, slot)
        if previous_ref is not None:
            if not replay or previous_ref != ref:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed")
        else:
            conn.execute("INSERT INTO magnitude_search_records VALUES (?,?,?)",
                         (run_id, slot, canonical_json(to_document(ref))))
    return json.loads(payload)["body"]
