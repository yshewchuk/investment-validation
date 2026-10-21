"""Deterministic sharded cohort inputs: writer, manifest hash and verified streaming reader.

Contract: experiments/ARCHITECTURE.md, "Sharded cohort inputs" (#991 fix 4).
"""
import json

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    NONFINITE_KEY,
    canonical_json,
    content_hash,
    from_document,
    to_document,
    untag_nonfinite,
)
from engine.v2.foundation.artifacts import safe_relative_path
from engine.v2.ops.errors import OpsError, fail
from experiments.magnitude_search_errors import artifact_failure

SHARD_SCHEMA = "magnitude_search_shard.v1"
MANIFEST_SCHEMA = "magnitude_search_cohort_manifest.v1"
TARGET_BYTES = 4 * 1024 * 1024
MAX_SHARD_BYTES = 8 * 1024 * 1024
MAX_SHARDS = 8192

_INVALID_SPEC = "INVALID_EXPERIMENT_SPEC"
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"


def _feature_key(row):
    return (row["event_id"], row["horizon_sessions"])


LIST_MEMBERS = (
    {
        "section": "features",
        "rows": ("event_id", "context_id", "horizon_sessions", "feature columns"),
        "key": _feature_key,
        "note": "stored rows keep the (event_id, context_id, horizon_sessions) scan order "
                "of one context",
    },
    {
        "section": "training",
        "rows": ("event_id", "absolute_move_pct"),
        "key": None,
        "problem": "no strictly increasing key: rows follow the (event_date, event_id) admitted "
                   "order but carry no event_date",
    },
    {
        "section": "preflight.event_ids",
        "rows": ("event_id",),
        "key": None,
        "problem": "no strictly increasing key: bare event ids follow the "
                   "(event_date, event_id) admitted order",
    },
    {
        "section": "preflight.planned_inputs",
        "rows": ("event_id", "fragment_id", "object_id", "content_hash"),
        "key": None,
        "problem": "no strictly increasing key: rows follow the (event_date, event_id) admitted "
                   "order but carry no event_date",
    },
    {
        "section": "training_inputs.candidate_event_ids",
        "rows": ("event_id",),
        "key": None,
        "problem": "no strictly increasing key: bare event ids follow the "
                   "(event_date, event_id) admitted order",
    },
    {
        "section": "training_inputs.qualified_event_ids",
        "rows": ("event_id",),
        "key": None,
        "problem": "no strictly increasing key: candidate subset keeps the "
                   "(event_date, event_id) admitted order",
    },
    {
        "section": "training_inputs.read_inputs",
        "rows": ("event_id", "fragment_id", "object_id", "content_hash"),
        "key": None,
        "problem": "no strictly increasing key: rows follow the (event_date, event_id) admitted "
                   "order but carry no event_date",
    },
    {
        "section": "training_inputs.omitted_labels",
        "rows": ("event_id", "reason"),
        "key": None,
        "problem": "no strictly increasing key: omitted rows keep the plan's "
                   "(event_date, event_id) order",
    },
)


def _artifact_failure(error, action, *, unsafe_is_conflict=True):
    message = "search cohort shard could not be " + action
    if isinstance(error, OSError):
        error = ArtifactError("IO_ERROR", message)
    return artifact_failure(error, integrity_message=message, resource_message=message,
                            unsafe_is_conflict=unsafe_is_conflict)


def _empty_size(section):
    document = {"schema_version": SHARD_SCHEMA, "section": section, "rows": []}
    return len(canonical_json(document).encode())


def _scan_reserved_keys(value):
    """Refuse a row that carries the non-finite tag key as literal data.

    ``canonical_json`` tags a non-finite float as ``{"__nonfinite__": ...}``
    and ``untag_nonfinite`` decodes that shape, so a row dict with that exact
    key would be ambiguous on the read side. Recurse over dicts' keys and
    lists'/tuples' items, bounded by the RecursionError mapping in
    ``_row_payload``.
    """
    if isinstance(value, dict):
        if NONFINITE_KEY in value:
            raise fail(_INVALID_SPEC, "search cohort row uses a reserved key")
        for item in value.values():
            _scan_reserved_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan_reserved_keys(item)


def _row_payload(row):
    try:
        _scan_reserved_keys(row)
        return canonical_json(row)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise fail(_INVALID_SPEC, "search cohort row is not canonical JSON") from None


def _less(left, right):
    try:
        return left < right
    except TypeError:
        return False


def _sort_key(key, row, previous):
    try:
        value = key(row)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail(_INVALID_SPEC, "search cohort section key is invalid") from None
    if not isinstance(value, tuple) or not all(type(part) in (str, int) for part in value):
        raise fail(_INVALID_SPEC, "search cohort section key is invalid")
    if previous is not None and not _less(previous, value):
        raise fail(_INVALID_SPEC, "search cohort section keys must strictly increase")
    return value


def _publish_shard(store, section, batch, first, last):
    document = {"schema_version": SHARD_SCHEMA, "section": section, "rows": batch}
    payload = canonical_json(document).encode()
    try:
        ref = store.publish_bytes(payload, schema_ref=SHARD_SCHEMA)
    except (ArtifactError, OSError) as error:
        raise _artifact_failure(error, "published") from None
    return {"ref": to_document(ref), "rows": len(batch),
            "first_key": list(first), "last_key": list(last)}


def _flush(store, section, shards, batch, first, last):
    if len(shards) >= MAX_SHARDS:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "search cohort section exceeds its shard limit")
    shards.append(_publish_shard(store, section, batch, first, last))


def write_section(store, section, rows, key):
    """Shard one section in its admitted order; equal rows publish equal shards."""
    batch, first, last, previous, count = [], None, None, None, 0
    shards, size = [], _empty_size(section)
    for row in rows:
        payload = _row_payload(row)
        width = len(payload.encode())
        snapshot = json.loads(payload)
        row_key = _sort_key(key, snapshot, previous)
        if batch and size + width + 1 > TARGET_BYTES:
            _flush(store, section, shards, batch, first, last)
            batch, size, first = [], _empty_size(section), None
        size += width if not batch else width + 1
        if size > MAX_SHARD_BYTES:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "search cohort shard exceeds its byte limit")
        batch.append(snapshot)
        first = row_key if first is None else first
        last = row_key
        previous = row_key
        count += 1
    if batch:
        _flush(store, section, shards, batch, first, last)
    return {"row_count": count, "shards": shards}


def _builder_shard_total(sections):
    try:
        return sum(len(entry["shards"]) for entry in sections.values())
    except (AttributeError, KeyError, TypeError):
        raise fail(_INVALID_SPEC, "search cohort manifest is invalid") from None


def _strict_values(manifest):
    """Admit only values stdlib JSON can store without non-finite literals.

    ``canonical_json``/``content_hash`` accept far more than strict JSON: a
    non-finite float becomes a ``{"__nonfinite__": ...}`` tag and any other
    object is stringified, so shape checks and the hash binding alone would
    pass a manifest ``build_manifest`` refuses to build. Builder and verifier
    run this one admission so written and verified manifests match.
    """
    json.dumps(manifest, allow_nan=False)


def build_manifest(inline, sections):
    """Bind inline scalars and section entries into one cohort manifest."""
    if _builder_shard_total(sections) > MAX_SHARDS:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "search cohort manifest exceeds its shard limit")
    manifest = {"schema_version": MANIFEST_SCHEMA, "inline": inline, "sections": sections}
    try:
        _strict_values(manifest)
        manifest = json.loads(canonical_json(manifest))
        _manifest_shape(manifest)
    except (OpsError, TypeError, ValueError, OverflowError, RecursionError):
        raise fail(_INVALID_SPEC, "search cohort manifest is invalid") from None
    return manifest


def manifest_id(manifest):
    """The cohort inputs_id: the content hash of the assembled manifest."""
    return content_hash(manifest)


def _section_binding(entry):
    if (not isinstance(entry, dict) or set(entry) != {"row_count", "shards"}
            or type(entry["row_count"]) is not int or entry["row_count"] < 0
            or not isinstance(entry["shards"], list)):
        raise fail(_CONFLICT, "search cohort section binding is invalid")
    if len(entry["shards"]) > MAX_SHARDS:
        raise fail(_CONFLICT, "search cohort section exceeds its shard limit")
    if (entry["row_count"] == 0) != (not entry["shards"]):
        raise fail(_CONFLICT, "search cohort section count is invalid")
    return entry["row_count"], entry["shards"]


def _stored_key(value):
    if not isinstance(value, list) or not all(type(part) in (str, int) for part in value):
        raise fail(_CONFLICT, "search cohort shard binding is invalid")
    return tuple(value)


def _shard_binding(shard, previous_last):
    if not isinstance(shard, dict) or set(shard) != {"ref", "rows", "first_key", "last_key"}:
        raise fail(_CONFLICT, "search cohort shard binding is invalid")
    try:
        ref = from_document(ArtifactRef, shard["ref"])
        if ref.schema_ref != SHARD_SCHEMA or not 0 < ref.byte_size <= MAX_SHARD_BYTES:
            raise ValueError
        safe_relative_path(ref.storage_key)
    except (ArtifactError, ValueError, TypeError, RecursionError):
        raise fail(_CONFLICT, "search cohort shard reference is invalid") from None
    if type(shard["rows"]) is not int or shard["rows"] <= 0:
        raise fail(_CONFLICT, "search cohort shard binding is invalid")
    first, last = _stored_key(shard["first_key"]), _stored_key(shard["last_key"])
    if previous_last is not None and not _less(previous_last, first):
        raise fail(_CONFLICT, "search cohort shard boundaries are out of order")
    return ref, shard["rows"], first, last


def _bindings(entry):
    row_count, shards = _section_binding(entry)
    bindings, previous_last = [], None
    for shard in shards:
        binding = _shard_binding(shard, previous_last)
        bindings.append(binding)
        previous_last = binding[3]
    if sum(binding[1] for binding in bindings) != row_count:
        raise fail(_CONFLICT, "search cohort section count is invalid")
    return row_count, bindings


def _payload_rows(payload, section, expected):
    try:
        document = json.loads(payload)
        if (not isinstance(document, dict) or set(document) != {"schema_version", "section", "rows"}
                or document["schema_version"] != SHARD_SCHEMA or document["section"] != section
                or not isinstance(document["rows"], list)):
            raise ValueError
    except (ValueError, TypeError, RecursionError):
        raise fail(_CONFLICT, "search cohort shard binding is invalid") from None
    if len(document["rows"]) != expected:
        raise fail(_CONFLICT, "search cohort shard binding is invalid")
    return untag_nonfinite(document["rows"])


def _verified_keys(rows, key, first, last):
    try:
        keys = [key(row) for row in rows]
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail(_CONFLICT, "search cohort shard binding is invalid") from None
    if any(not isinstance(value, tuple) or not all(type(part) in (str, int) for part in value)
           for value in keys):
        raise fail(_CONFLICT, "search cohort shard binding is invalid")
    if keys[0] != first or keys[-1] != last:
        raise fail(_CONFLICT, "search cohort shard boundary keys differ")
    for earlier, later in zip(keys, keys[1:]):
        if not _less(earlier, later):
            raise fail(_CONFLICT, "search cohort shard keys are out of order")
    return keys


def iter_section(store, section, entry, key):
    """Stream one section's verified rows in order, holding one shard at a time."""
    _, bindings = _bindings(entry)
    for ref, expected, first, last in bindings:
        try:
            payload = store.read_verified(ref)
        except (ArtifactError, OSError) as error:
            raise _artifact_failure(error, "read", unsafe_is_conflict=False) from None
        rows = _payload_rows(payload, section, expected)
        _verified_keys(rows, key, first, last)
        yield from rows


def _scalar_inline(inline):
    return all(value is None or type(value) in (str, int, float, bool)
               for value in inline.values())


def _manifest_shape(manifest):
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "inline", "sections"}
            or manifest["schema_version"] != MANIFEST_SCHEMA
            or not isinstance(manifest["inline"], dict)
            or not _scalar_inline(manifest["inline"])
            or not isinstance(manifest["sections"], dict)):
        raise fail(_CONFLICT, "search cohort manifest is invalid")
    for entry in manifest["sections"].values():
        _bindings(entry)
    if sum(len(entry["shards"]) for entry in manifest["sections"].values()) > MAX_SHARDS:
        raise fail(_CONFLICT, "search cohort manifest exceeds its shard limit")


def verify_manifest(manifest, expected_id):
    """Shape-check a manifest and bind it to the expected inputs_id."""
    _manifest_shape(manifest)
    try:
        _strict_values(manifest)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise fail(_CONFLICT, "search cohort manifest is invalid") from None
    if manifest_id(manifest) != expected_id:
        raise fail(_CONFLICT, "search cohort manifest differs from its binding")
    return manifest