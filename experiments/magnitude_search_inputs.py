"""One loader for stored cohort inputs: v2 sharded manifests and the v1 whole-cohort adapter.

Contract: experiments/ARCHITECTURE.md, "Sharded cohort inputs" (#991 fix 4).
"""
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments.magnitude_search_shards import (
    MANIFEST_SCHEMA,
    build_manifest,
    iter_section,
    list_members,
    manifest_id,
    verify_manifest,
    write_section,
)

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"


def _position_key(row):
    return (row["position"],)


def _iter_rows(store, member, entry):
    """Stream one section's verified rows, holding one shard at a time; keyless sections are unwrapped."""
    if member["key"] is not None:
        yield from iter_section(store, member["section"], entry, member["key"])
        return
    count = 0
    for item in iter_section(store, member["section"], entry, _position_key):
        if set(item) != {"position", "row"} or item["position"] != count:
            raise fail(_CONFLICT, "search cohort section positions are invalid")
        count += 1
        yield item["row"]


def _section_rows(store, member, entry):
    return list(_iter_rows(store, member, entry))


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


def _admit_manifest(record, expected_id):
    """Verify a manifest record against its id and its section/inline shape; return its sections."""
    verify_manifest(record, expected_id)
    sections = record["sections"]
    members = list_members(record["inline"].get("preflight.schema_version"))
    if set(sections) != {member["section"] for member in members}:
        raise fail(_CONFLICT, "search cohort manifest sections are invalid")
    shape = {}
    for path, value in record["inline"].items():
        _place(shape, path, value)
    for member in members:
        _place(shape, member["section"], None)
    return sections


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
    sections = _admit_manifest(record, expected_id)
    inputs = {}
    for path, value in record["inline"].items():
        _place(inputs, path, value)
    for member in list_members(record["inline"].get("preflight.schema_version")):
        _place(inputs, member["section"], _section_rows(store, member, sections[member["section"]]))
    return inputs


def load_fit_inputs(store, manifest, expected_id, evaluation_event_ids, horizons, feature_columns, *,
                    target_column="absolute_move_pct"):
    """Stream stored training/feature rows for fixed fitting or admission replay.

    Verifies the manifest against ``expected_id`` and reads only its ``training`` and
    ``features`` sections, shard by shard. Returns ``{"features", "training"}``: training rows
    reduced to (event_id, target_column); feature rows kept for training or evaluation
    events at the selected horizons, reduced to (event_id, context_id, horizon_sessions,
    *feature_columns), in stored order. A manifest, hash, shard or inline-path mismatch is
    EXPERIMENT_IDENTITY_CONFLICT (retryable store errors propagate); a row missing a
    requested column raises KeyError. The magnitude fit child reports MemoryError as
    resource_limit and other loader failures as unavailable; library callers retain the error.
    Both loaders share ``_admit_manifest``; projection arguments must be
    lists or tuples (INVALID_EXPERIMENT_SPEC otherwise).
    """
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise fail(_CONFLICT, "search cohort record is invalid")
    if (not isinstance(evaluation_event_ids, (list, tuple)) or not isinstance(horizons, (list, tuple))
            or not isinstance(feature_columns, (list, tuple))):
        raise fail("INVALID_EXPERIMENT_SPEC", "search fit projection arguments are invalid")
    sections = _admit_manifest(manifest, expected_id)
    schema = manifest["inline"].get("preflight.schema_version")
    expected_target = "crush_pct_iv30" if schema == "research_iv_crush_preflight.v1" else "absolute_move_pct"
    if target_column != expected_target or manifest["inline"].get("target.column", "absolute_move_pct") != target_column:
        raise fail(_CONFLICT, "search fit target differs from its cohort")
    members = {member["section"]: member for member in list_members(schema)}
    training = [{name: row[name] for name in ("event_id", target_column)}
                for row in _iter_rows(store, members["training"], sections["training"])]
    events = {row["event_id"] for row in training} | set(evaluation_event_ids)
    wanted = set(horizons)
    columns = ("event_id", "context_id", "horizon_sessions", *feature_columns)
    features = [{name: row[name] for name in columns}
                for row in _iter_rows(store, members["features"], sections["features"])
                if row["event_id"] in events and row["horizon_sessions"] in wanted]
    return {"features": features, "training": training}


def _path_value(inputs, path):
    value = inputs
    for name in path.split("."):
        value = value[name]
    return value


def _inline_scalars(value, prefix, skip, out):
    for name, item in value.items():
        path = prefix + name
        if path in skip:
            continue
        if isinstance(item, dict):
            _inline_scalars(item, path + ".", skip, out)
        else:
            out[path] = item


def publish_inputs(store, inputs):
    """Shard ``inputs`` into the store; return ``(manifest, inputs_id)``.

    The inverse of ``load_inputs``: list-valued members become sections (keyless ones as
    ``{position, row}`` rows), every other leaf is an inline scalar by dotted path. Shards
    are published here; the caller appends the manifest, which is the commit point.
    """
    sections = {}
    for member in list_members(inputs.get("preflight", {}).get("schema_version")):
        path = member["section"]
        rows = _path_value(inputs, path)
        key = member["key"]
        if key is None:
            rows = ({"position": index, "row": item} for index, item in enumerate(rows))
            key = _position_key
        sections[path] = write_section(store, path, rows, key)
    inline = {}
    _inline_scalars(inputs, "", set(sections), inline)
    manifest = build_manifest(inline, sections)
    return manifest, manifest_id(manifest)
