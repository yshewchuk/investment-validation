"""Bounded CAS sections for native SIZE registration v1.7."""

import hashlib

from engine.v2.foundation.canonical import iter_canonical_json
from engine.v2.ops.errors import OpsError, fail
from experiments.magnitude_search_shards import iter_section, write_section

_PLACEHOLDER = "$size_artifact"
_ARTIFACT_KEYS = frozenset({"events", "target_events", "event_ids", "fragments",
                            "fragment_ids", "source_fragments", "history_fragments",
                            "price_fragments", "fragment_refs", "metadata", "fragment_list",
                            "target_metadata"})

MAX_SIZE_ARTIFACT_BYTES = 1024 * 1024 * 1024


def size_registration_artifact_bytes(compact_document):
    """Sum every referenced shard byte_size before any object reads."""
    if not isinstance(compact_document, dict):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration envelope is invalid")
    artifacts = compact_document.get("size_artifacts")
    if not isinstance(artifacts, dict):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration artifact map is invalid")
    total = 0
    for identifier, descriptor in artifacts.items():
        if (not isinstance(identifier, str) or not isinstance(descriptor, dict)
                or set(descriptor) != {"kind", "section", "entry"}
                or descriptor["kind"] not in ("list", "object", "map")
                or descriptor["section"] != "size_registration.v1.7:" + identifier
                or not isinstance(descriptor["entry"], dict)
                or not isinstance(descriptor["entry"].get("shards"), list)):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration artifact reference is invalid")
        for shard in descriptor["entry"]["shards"]:
            if not isinstance(shard, dict) or not isinstance(shard.get("ref"), dict):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration artifact reference is invalid")
            byte_size = shard["ref"].get("byte_size")
            if type(byte_size) is not int or byte_size < 0:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration artifact reference is invalid")
            total += byte_size
    if total > MAX_SIZE_ARTIFACT_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "SIZE registration artifact byte limit exceeded")
    return total


def _position(row):
    return (row["position"],)


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _publish(store, rows, kind, identifier):
    section = "size_registration.v1.7:" + identifier
    entry = write_section(store, section, rows, _position)
    return {_PLACEHOLDER: identifier}, {"kind": kind, "section": section, "entry": entry}


def _artifact_identifier(value, kind):
    digest = hashlib.sha256()
    digest.update(kind.encode("utf-8"))
    for chunk in iter_canonical_json(value):
        digest.update(chunk.encode("utf-8"))
    return "h" + digest.hexdigest()


def pack_size_registration(store, document):
    """Return a compact reference document and the original logical source document."""
    artifacts = {}
    digest_to_descriptor = {}

    def store_value(value, kind):
        identifier = _artifact_identifier(value, kind)
        cached = digest_to_descriptor.get(identifier)
        if cached is not None:
            return {_PLACEHOLDER: cached[0]}
        if kind == "list":
            rows = ({"position": index, "row": walk(item)} for index, item in enumerate(value))
        elif kind == "map":
            rows = ({"position": index, "row": [name, walk(value[name], name)]}
                    for index, name in enumerate(sorted(value)))
        else:
            rows = iter(({"position": 0, "row": _plain(value)},))
        placeholder, descriptor = _publish(store, rows, kind, identifier)
        digest_to_descriptor[identifier] = (identifier, descriptor)
        artifacts[identifier] = descriptor
        return placeholder

    def walk(value, key=None):
        if isinstance(value, dict):
            if key in ("metadata", "target_metadata", "confirmed_metadata", "price_row_metadata") or (
                    isinstance(key, str) and key.endswith("metadata")):
                return store_value(value, "map")
            return {name: walk(item, name) for name, item in value.items()}
        if isinstance(value, list):
            if (len(value) >= 64 or key in _ARTIFACT_KEYS
                    or isinstance(key, str) and (key.endswith("fragments") or key.endswith("fragment_ids"))):
                return store_value(value, "list")
            return [walk(item) for item in value]
        return value

    compact = walk(document)
    compact["size_artifacts"] = artifacts
    return compact, document


def hydrate_size_registration(store, compact_document, *, verified_rows=None):
    """Verify referenced CAS shards and return the logical registration object."""
    size_registration_artifact_bytes(compact_document)
    artifacts = compact_document.get("size_artifacts")
    seen = set()

    def walk(value):
        if isinstance(value, dict) and set(value) == {_PLACEHOLDER}:
            identifier = value[_PLACEHOLDER]
            descriptor = artifacts.get(identifier)
            if (not isinstance(identifier, str) or not isinstance(descriptor, dict)
                    or set(descriptor) != {"kind", "section", "entry"}
                    or descriptor["kind"] not in ("list", "object", "map")
                    or descriptor["section"] != "size_registration.v1.7:" + identifier):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration artifact reference is invalid")
            seen.add(identifier)
            try:
                values = []
                for index, row in enumerate(iter_section(store, descriptor["section"], descriptor["entry"], _position, verified_rows=verified_rows)):
                    if (not isinstance(row, dict) or set(row) != {"position", "row"}
                            or row["position"] != index):
                        raise ValueError
                    values.append(walk(row["row"]))
            except (OpsError, ValueError, TypeError, KeyError, RecursionError):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration artifact failed verification") from None
            if descriptor["kind"] == "list":
                reconstructed = values
            elif descriptor["kind"] == "map":
                mapping = {}
                for entry in values:
                    if (not isinstance(entry, list) or len(entry) != 2
                            or not isinstance(entry[0], str) or entry[0] in mapping):
                        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration mapping artifact is invalid")
                    mapping[entry[0]] = entry[1]
                reconstructed = mapping
            else:
                if len(values) != 1:
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration object artifact is invalid")
                reconstructed = values[0]
            if _artifact_identifier(reconstructed, descriptor["kind"]) != identifier:
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration artifact identity mismatch")
            return reconstructed
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items() if key != "size_artifacts"}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    result = walk(compact_document)
    if seen != set(artifacts):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "SIZE registration has unreferenced artifacts")
    return result
