"""Shard cut, manifest binding, verified reader and inventory coverage."""
import json
from contextlib import contextmanager

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import ArtifactStore, canonical_json, from_document, to_document
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_shards as shards


def row(event_id, value=0):
    return {"event_id": event_id, "horizon_sessions": 1, "value": value}


def key(item):
    return (item["event_id"], item["horizon_sessions"])


def rows(count):
    return [row(f"E{index:03d}", index) for index in range(count)]


class Once:
    def __init__(self, items):
        self.items = items
        self.calls = 0

    def __iter__(self):
        self.calls += 1
        return iter(self.items)


@contextmanager
def refusal(code, retryable):
    with pytest.raises(OpsError) as caught:
        yield
    assert caught.value.problem.code == code
    assert caught.value.problem.retryable is retryable


def reference(entry):
    return from_document(ArtifactRef, entry["ref"])


def publish(store, document):
    ref = store.publish_bytes(canonical_json(document).encode(), schema_ref=shards.SHARD_SCHEMA)
    return to_document(ref)


def shard_entry(store, batch, *, section="features", schema=shards.SHARD_SCHEMA):
    document = {"schema_version": schema, "section": section, "rows": batch}
    return {"ref": publish(store, document), "rows": len(batch),
            "first_key": list(key(batch[0])), "last_key": list(key(batch[-1]))}


def section_entry(*items):
    return {"row_count": sum(item["rows"] for item in items), "shards": list(items)}


def counted_publisher(monkeypatch):
    published = []
    original = ArtifactStore.publish_bytes

    def counted(self, data, *, schema_ref):
        published.append(data)
        return original(self, data, schema_ref=schema_ref)

    monkeypatch.setattr(ArtifactStore, "publish_bytes", counted)
    return published


def test_write_is_deterministic_across_batch_iterator_types(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    batch = rows(6)
    first = shards.write_section(store, "features", batch, key)
    once = Once(batch)
    alternatives = (
        shards.write_section(store, "features", (item for item in batch), key),
        shards.write_section(store, "features", iter(batch), key),
        shards.write_section(store, "features", once, key),
    )
    assert alternatives[0] == alternatives[1] == alternatives[2] == first
    assert once.calls == 1
    inline = {"stored_inputs_id": "sha256:x"}
    manifest = shards.build_manifest(inline, {"features": first})
    assert shards.build_manifest(inline, {"features": first}) == manifest
    assert shards.manifest_id(manifest) == shards.manifest_id(dict(manifest))


def test_cut_at_target_and_size_equals_computed_size(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "TARGET_BYTES", 200)
    monkeypatch.setattr(shards, "MAX_SHARD_BYTES", 400)
    store = ArtifactStore(tmp_path / "store")
    batch = rows(9)
    written = shards.write_section(store, "features", batch, key)
    assert written["row_count"] == len(batch)
    assert len(written["shards"]) > 1
    for entry in written["shards"]:
        payload = store.read_verified(reference(entry))
        document = json.loads(payload)
        assert len(payload) == entry["ref"]["byte_size"] == len(canonical_json(document).encode())
        assert entry["rows"] == len(document["rows"])
        assert entry["first_key"] == list(key(document["rows"][0]))
        assert entry["last_key"] == list(key(document["rows"][-1]))
        assert entry["ref"]["byte_size"] <= 200


def test_round_trip_order_for_zero_one_and_many_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "TARGET_BYTES", 150)
    store = ArtifactStore(tmp_path / "store")
    empty = shards.write_section(store, "features", [], key)
    assert empty == {"row_count": 0, "shards": []}
    assert list(shards.iter_section(store, "features", empty, key)) == []
    one = rows(1)
    written = shards.write_section(store, "features", one, key)
    assert list(shards.iter_section(store, "features", written, key)) == one
    many = rows(30)
    written = shards.write_section(store, "features", many, key)
    assert len(written["shards"]) > 1
    assert list(shards.iter_section(store, "features", written, key)) == many


def test_oversized_row_refuses_without_publishing(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    published = counted_publisher(monkeypatch)
    monkeypatch.setattr(shards, "TARGET_BYTES", 256)
    monkeypatch.setattr(shards, "MAX_SHARD_BYTES", 256)
    with refusal("RESOURCE_LIMIT_EXCEEDED", False):
        shards.write_section(store, "features", [row("E" * 400)], key)
    assert published == []


def test_shard_count_bound_refuses_before_extra_publish(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    published = counted_publisher(monkeypatch)
    monkeypatch.setattr(shards, "TARGET_BYTES", 150)
    monkeypatch.setattr(shards, "MAX_SHARD_BYTES", 1000)
    monkeypatch.setattr(shards, "MAX_SHARDS", 2)
    with refusal("RESOURCE_LIMIT_EXCEEDED", False):
        shards.write_section(store, "features", rows(12), key)
    assert len(published) == 2


def test_duplicate_and_out_of_order_keys_refuse_while_writing(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    with refusal("INVALID_EXPERIMENT_SPEC", False):
        shards.write_section(store, "features", [row("E002"), row("E001")], key)
    with refusal("INVALID_EXPERIMENT_SPEC", False):
        shards.write_section(store, "features", [row("E001"), row("E001")], key)


def test_build_manifest_shard_budget(monkeypatch):
    monkeypatch.setattr(shards, "MAX_SHARDS", 2)
    shard = {"ref": {}, "rows": 1, "first_key": [0], "last_key": [1]}
    assert shards.build_manifest({}, {"features": {"row_count": 2, "shards": [shard, shard]}})
    with refusal("RESOURCE_LIMIT_EXCEEDED", False):
        shards.build_manifest({}, {"features": {"row_count": 2, "shards": [shard, shard]},
                                   "training": {"row_count": 1, "shards": [shard]}})


def test_verify_manifest_shape_and_hash():
    manifest = shards.build_manifest({"stored_inputs_id": "sha256:x"}, {})
    assert shards.verify_manifest(manifest, shards.manifest_id(manifest)) == manifest
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        shards.verify_manifest(manifest, "sha256:" + "0" * 64)
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        shards.verify_manifest({**manifest, "extra": 1}, shards.manifest_id(manifest))
    broken = {**manifest, "sections": {"features": {"row_count": True, "shards": []}}}
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        shards.verify_manifest(broken, shards.manifest_id(broken))


def _tampered(store):
    entry = shards.write_section(store, "features", rows(1), key)
    path = store.root / reference(entry["shards"][0]).storage_key
    size = path.stat().st_size
    path.chmod(0o644)
    path.write_bytes(b"x" * size)
    return entry


def _wrong_schema(store):
    return section_entry(shard_entry(store, rows(1), schema="other.v1"))


def _wrong_section(store):
    return section_entry(shard_entry(store, rows(1), section="training"))


def _wrong_rows_count(store):
    entry = shards.write_section(store, "features", rows(1), key)
    return {"row_count": 1, "shards": [{**entry["shards"][0], "rows": 2}]}


def _wrong_first_key(store):
    entry = shards.write_section(store, "features", rows(1), key)
    return {"row_count": 1, "shards": [{**entry["shards"][0], "first_key": ["E000"]}]}


def _wrong_last_key(store):
    entry = shards.write_section(store, "features", rows(1), key)
    return {"row_count": 1, "shards": [{**entry["shards"][0], "last_key": ["E999"]}]}


def _decreasing_keys(store):
    return section_entry(shard_entry(store, [row("E002"), row("E001")]))


def _overlapping_shards(store):
    first = shards.write_section(store, "features", [row("E001"), row("E002")], key)
    second = shards.write_section(store, "features", [row("E002"), row("E003")], key)
    return {"row_count": 4, "shards": [first["shards"][0], second["shards"][0]]}


def _duplicate_keys(store):
    return section_entry(shard_entry(store, [row("E001"), row("E001")]))


def _nonzero_count_with_empty_list(store):
    return {"row_count": 2, "shards": []}


def _shard_for_empty_section(store):
    return {"row_count": 0, "shards": [shard_entry(store, rows(1))]}


def _unsafe_reference(store):
    entry = shards.write_section(store, "features", rows(1), key)
    shard = entry["shards"][0]
    return {"row_count": 1, "shards": [{**shard, "ref": {**shard["ref"], "storage_key": "../x"}}]}


def _malformed_reference(store):
    entry = shards.write_section(store, "features", rows(1), key)
    return {"row_count": 1, "shards": [{**entry["shards"][0], "ref": {"storage_key": "x"}}]}


def _oversized_reference(store):
    entry = shards.write_section(store, "features", rows(1), key)
    shard = entry["shards"][0]
    ref = {**shard["ref"], "byte_size": shards.MAX_SHARD_BYTES + 1}
    return {"row_count": 1, "shards": [{**shard, "ref": ref}]}


CASES = (_tampered, _wrong_schema, _wrong_section, _wrong_rows_count, _wrong_first_key, _wrong_last_key,
         _decreasing_keys, _overlapping_shards, _duplicate_keys, _nonzero_count_with_empty_list,
         _shard_for_empty_section, _unsafe_reference, _malformed_reference, _oversized_reference)


@pytest.mark.parametrize("case", CASES, ids=[item.__name__ for item in CASES])
def test_reader_refusals_are_identity_conflicts(tmp_path, case):
    store = ArtifactStore(tmp_path / "store")
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        list(shards.iter_section(store, "features", case(store), key))


def test_malformed_reference_refuses_before_any_read(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    entry = shards.write_section(store, "features", rows(1), key)
    shard = entry["shards"][0]
    broken = {"row_count": 1, "shards": [{**shard, "ref": {**shard["ref"], "storage_key": "/x"}}]}
    monkeypatch.setattr(ArtifactStore, "read_verified",
                        lambda *args: pytest.fail("artifact access before syntax admission"))
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        list(shards.iter_section(store, "features", broken, key))


def _storage_gone(self, ref):
    raise OSError("storage disappeared")


def test_missing_shard_and_oserror_are_retryable(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    batch = rows(1)
    entry = shards.write_section(store, "features", batch, key)
    (store.root / reference(entry["shards"][0]).storage_key).unlink()
    with refusal("RESOURCE_UNAVAILABLE", True):
        list(shards.iter_section(store, "features", entry, key))
    second = ArtifactStore(tmp_path / "second")
    entry = shards.write_section(second, "features", batch, key)
    monkeypatch.setattr(ArtifactStore, "read_verified", _storage_gone)
    with refusal("RESOURCE_UNAVAILABLE", True):
        list(shards.iter_section(second, "features", entry, key))


def test_reader_streams_one_shard_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "TARGET_BYTES", 150)
    store = ArtifactStore(tmp_path / "store")
    batch = rows(6)
    entry = shards.write_section(store, "features", batch, key)
    assert len(entry["shards"]) > 1
    reads = []
    original = ArtifactStore.read_verified

    def counted(self, ref):
        reads.append(ref.storage_key)
        return original(self, ref)

    monkeypatch.setattr(ArtifactStore, "read_verified", counted)
    reader = shards.iter_section(store, "features", entry, key)
    assert iter(reader) is reader
    assert next(reader) == batch[0]
    assert len(reads) == 1
    assert list(reader) == batch[1:]
    assert len(reads) == len(entry["shards"])


def test_inventory_sections_are_unique_and_keyed_or_flagged():
    members = shards.LIST_MEMBERS
    assert len({member["section"] for member in members}) == len(members)
    for member in members:
        assert member["rows"]
        if member.get("problem"):
            assert member["key"] is None
        else:
            assert callable(member["key"])