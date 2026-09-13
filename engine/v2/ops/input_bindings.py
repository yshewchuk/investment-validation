"""Launch-time resolution of ``job_``-ID input bindings, recorded durably.

A worker's ``input_bindings`` parameter maps a staging-relative name to
either a direct artifact ID (admitted only via ``spec.input_refs``) or a
``job_<id>#<output_name>`` reference to a declared dependency's committed
output. Resolution turns the second form into a concrete artifact ID exactly
once, at launch, against the parent's *committed* state; the coordinator
later trusts the row this module records instead of re-deriving it (P2-5/B1a).
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.v2.foundation import content_hash
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import fail

__all__ = ["ResolvedBinding", "record_resolved_bindings", "recorded_bindings",
           "resolve_bindings", "resolved_inputs_hash"]


@dataclass(frozen=True)
class ResolvedBinding:
    name: str
    binding: str
    artifact_id: str
    content_hash: str


def _resolve_job_binding(conn, spec, binding):
    dependency, separator, output_name = binding.partition("#")
    if not separator or not output_name:
        raise fail("VALIDATION_FAILED", "job-id input binding has no explicit output name",
                   details={"binding": binding})
    if dependency not in spec.dependency_job_ids:
        raise fail("INPUT_CHANGED", "input binding names an undeclared dependency",
                   details={"binding": binding})
    job_row = conn.execute("SELECT state FROM jobs WHERE job_id=?", (dependency,)).fetchone()
    if job_row is None or job_row[0] != "succeeded":
        raise fail("INPUT_CHANGED", "input binding's parent job has not succeeded",
                   details={"binding": binding})
    attempt_row = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id=? AND state='succeeded' "
        "ORDER BY attempt_number DESC LIMIT 1", (dependency,)).fetchone()
    if attempt_row is None:
        raise fail("INPUT_CHANGED", "input binding's parent has no succeeded attempt",
                   details={"binding": binding})
    output_row = conn.execute(
        "SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? AND name=?",
        (attempt_row[0], output_name)).fetchone()
    if output_row is None:
        raise fail("INPUT_CHANGED", "input binding names an output its parent did not produce",
                   details={"binding": binding})
    return output_row[0]


def resolve_bindings(conn, store, spec):
    """Resolve every ``input_bindings`` entry to its committed artifact.

    Pure read: nothing is recorded and nothing is copied into staging. Raises
    ``OpsError`` (``INPUT_CHANGED`` or ``VALIDATION_FAILED``) on the first
    binding that is not admitted; never a bare ``ValueError``.
    """
    bindings = spec.parameters.get("input_bindings") or {}
    admitted_refs = set(spec.input_refs)
    resolved = {}
    for name, raw_binding in sorted(bindings.items()):
        binding = str(raw_binding)
        if binding.startswith("job_"):
            artifact_id = _resolve_job_binding(conn, spec, binding)
        elif binding in admitted_refs:
            artifact_id = binding
        else:
            raise fail("INPUT_CHANGED", "input binding names an artifact outside input_refs",
                       details={"name": name})
        ref = artifact(conn, store, artifact_id)
        resolved[name] = ResolvedBinding(name=name, binding=binding,
                                         artifact_id=ref.artifact_id, content_hash=ref.content_hash)
    return resolved


def record_resolved_bindings(conn, attempt_id, resolved):
    """Persist resolved bindings; must run inside the caller's open transaction."""
    if not conn.in_transaction:
        raise ValueError("recording input bindings requires an open transaction")
    for item in resolved.values():
        conn.execute("INSERT INTO attempt_input_bindings VALUES (?,?,?,?,?)",
                     (attempt_id, item.name, item.binding, item.artifact_id, item.content_hash))


def recorded_bindings(conn, attempt_id):
    """The durable resolution recorded for this attempt at launch, by name."""
    rows = conn.execute(
        "SELECT name, binding, artifact_id, content_hash FROM attempt_input_bindings "
        "WHERE attempt_id=? ORDER BY name", (attempt_id,)).fetchall()
    return {row[0]: ResolvedBinding(name=row[0], binding=row[1], artifact_id=row[2],
                                    content_hash=row[3]) for row in rows}


def resolved_inputs_hash(spec, resolved):
    """A cache-identity ``inputs`` value covering admitted refs and bindings.

    ``resolved`` may be either the live result of :func:`resolve_bindings` or
    the durable rows from :func:`recorded_bindings` — both map name to a
    :class:`ResolvedBinding`. A parent whose committed output artifact
    changes changes this hash, so a stale checkpoint is never reused.
    """
    pairs = sorted((item.name, item.artifact_id) for item in resolved.values())
    return content_hash({"input_refs": sorted(spec.input_refs), "bindings": pairs})


def resolve_and_record(conn, store, claim):
    """Resolve every binding and durably record it before anything is staged."""
    resolved = resolve_bindings(conn, store, claim.spec)
    with transaction(conn):
        record_resolved_bindings(conn, claim.attempt_id, resolved)
    return resolved
