"""One loader for stored cohort inputs: v2 sharded manifests and the v1 whole-cohort adapter.

Contract: experiments/ARCHITECTURE.md, "Sharded cohort inputs" (#991 fix 4).
"""
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments.magnitude_search_shards import (
    LIST_MEMBERS,
    MANIFEST_SCHEMA,
    iter_section,
    verify_manifest,
)

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"


def _position_key(row):
    return (row["position"],)


def _section_rows(store, member, entry):
    """Stream one section; keyless sections store ``{position, row}`` and are unwrapped."""
    if member["key"] is not None:
        return list(iter_section(store, member["section"], entry, member["key"]))
    rows = []
    for item in iter_section(store, member["section"], entry, _position_key):
        if set(item) != {"position", "row"} or item["position"] != len(rows):
            raise fail(_CONFLICT, "search cohort section positions are invalid")
        rows.append(item["row"])
    return rows


def _place(inputs, path, value):
    *parents, leaf = path.split(".")
    target = inputs
    for name in parents:
        target = target.setdefault(name, {})
        if not isinstance(target, dict):
            raise fail(_CONFLICT, "search cohort manifest does not fit its sections")
    if leaf in target:
        raise fail(_CONFLICT, "search cohort manifest does not fit its sections")
    target[leaf] = value


def load_inputs(store, record, expected_id):
    """Return the cohort inputs of a stored record, bound to ``expected_id``.

    A manifest-schema record is verified, then its sections are streamed shard by
    shard onto the inline members, which are flat scalars keyed by dotted path. Any
    other dict is a v1 whole cohort whose id is its content hash. Both yield the same
    inputs shape; any mismatch is a conflict.
    """
    if not isinstance(record, dict):
        raise fail(_CONFLICT, "search cohort record is invalid")
    if record.get("schema_version") != MANIFEST_SCHEMA:
        if content_hash(record) != expected_id:
            raise fail(_CONFLICT, "search cohort differs from its binding")
        return record
    verify_manifest(record, expected_id)
    sections = record["sections"]
    if set(sections) != {member["section"] for member in LIST_MEMBERS}:
        raise fail(_CONFLICT, "search cohort manifest sections are invalid")
    inputs = {}
    for path, value in record["inline"].items():
        _place(inputs, path, value)
    for member in LIST_MEMBERS:
        _place(inputs, member["section"], _section_rows(store, member, sections[member["section"]]))
    return inputs
