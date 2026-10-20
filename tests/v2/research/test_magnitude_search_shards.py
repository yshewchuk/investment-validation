"""Shard cut, manifest binding, verified reader and inventory coverage."""
import json
import math
from contextlib import contextmanager

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
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


def expected_partition(section, batch, target):
    """Shard row counts write_section yields for a batch of known row widths."""
    empty = len(canonical_json({"schema_version": shards.SHARD_SCHEMA,
                                "section": section, "rows": []}).encode())
    partitions, size, count = [], empty, 0
    for item in batch:
        width = len(canonical_json(item).encode())
        if count and size + width + 1 > target:
            partitions.append(count)
            size, count = empty, 0
        size += width if not count else width + 1
        count += 1
    if count:
        partitions.append(count)
    return partitions


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
    assert [entry["rows"] for entry in first["shards"]] == [len(batch)]
    assert [item for entry in first["shards"]
            for item in json.loads(store.read_verified(reference(entry)))["rows"]] == batch
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
    assert [entry["rows"] for entry in written["shards"]] == expected_partition("features", batch, 200)
    recovered = []
    for entry in written["shards"]:
        payload = store.read_verified(reference(entry))
        document = json.loads(payload)
        assert len(payload) == entry["ref"]["byte_size"] == len(canonical_json(document).encode())
        assert entry["rows"] == len(document["rows"])
        assert entry["first_key"] == list(key(document["rows"][0]))
        assert entry["last_key"] == list(key(document["rows"][-1]))
        assert entry["ref"]["byte_size"] <= 200
        recovered.extend(document["rows"])
    assert recovered == batch
    changed = list(batch)
    changed[0] = row(batch[0]["event_id"], batch[0]["value"] + 1)
    altered = shards.write_section(store, "features", changed, key)
    assert shards.manifest_id(shards.build_manifest({}, {"features": altered})) != \
        shards.manifest_id(shards.build_manifest({}, {"features": written}))


def test_round_trip_order_for_zero_one_and_many_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "TARGET_BYTES", 150)
    store = ArtifactStore(tmp_path / "store")
    empty = shards.write_section(store, "features", [], key)
    assert empty == {"row_count": 0, "shards": []}
    assert list(shards.iter_section(store, "features", empty, key)) == []
    one = rows(1)
    written = shards.write_section(store, "features", one, key)
    assert [entry["rows"] for entry in written["shards"]] == [1]
    assert list(shards.iter_section(store, "features", written, key)) == one
    many = rows(30)
    written = shards.write_section(store, "features", many, key)
    assert len(written["shards"]) > 1
    assert [entry["rows"] for entry in written["shards"]] == expected_partition("features", many, 150)
    recovered = []
    for entry in written["shards"]:
        document = json.loads(store.read_verified(reference(entry)))
        assert entry["ref"]["byte_size"] == len(canonical_json(document).encode())
        recovered.extend(document["rows"])
    assert recovered == many
    assert recovered == list(shards.iter_section(store, "features", written, key))


def test_reader_decodes_nonfinite_feature_values(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    batch = [{"event_id": "E000", "horizon_sessions": 1, "value": float("nan")},
             {"event_id": "E001", "horizon_sessions": 1, "value": float("inf")}]
    written = shards.write_section(store, "features", batch, key)
    recovered = list(shards.iter_section(store, "features", written, key))
    assert [item["event_id"] for item in recovered] == ["E000", "E001"]
    assert math.isnan(recovered[0]["value"])
    assert recovered[1]["value"] == float("inf")


def test_oversized_row_refuses_without_publishing(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    published = counted_publisher(monkeypatch)
    monkeypatch.setattr(shards, "TARGET_BYTES", 256)
    monkeypatch.setattr(shards, "MAX_SHARD_BYTES", 256)
    with refusal("RESOURCE_LIMIT_EXCEEDED", False):
        shards.write_section(store, "features", [row("E" * 400)], key)
    assert published == []


def test_non_canonical_row_refuses_while_writing(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    with refusal("INVALID_EXPERIMENT_SPEC", False):
        shards.write_section(store, "features", [{"k": object()}], key)


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


def test_build_manifest_shard_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "MAX_SHARDS", 2)
    store = ArtifactStore(tmp_path / "store")
    first = shards.write_section(store, "features", [row("E001")], key)
    second = shards.write_section(store, "features", [row("E002")], key)
    shard_list = first["shards"] + second["shards"]
    sections = {"features": {"row_count": 2, "shards": shard_list}}
    assert shards.build_manifest({}, sections) == {
        "schema_version": shards.MANIFEST_SCHEMA, "inline": {}, "sections": sections}
    with refusal("RESOURCE_LIMIT_EXCEEDED", False):
        shards.build_manifest({}, {"features": {"row_count": 2, "shards": shard_list},
                                   "training": {"row_count": 1, "shards": first["shards"]}})


def test_verify_manifest_shard_budget(monkeypatch):
    monkeypatch.setattr(shards, "MAX_SHARDS", 2)
    shard = {"ref": {}, "rows": 1, "first_key": [0], "last_key": [1]}
    manifest = {"schema_version": shards.MANIFEST_SCHEMA, "inline": {},
                "sections": {"features": {"row_count": 3, "shards": [shard, shard, shard]}}}
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        shards.verify_manifest(manifest, shards.manifest_id(manifest))


def _two_shard_section(store, section):
    first = shards.write_section(store, section, [row("E000"), row("E001")], key)
    second = shards.write_section(store, section, [row("E002"), row("E003")], key)
    return {"row_count": 4, "shards": first["shards"] + second["shards"]}


def test_verify_manifest_cross_section_shard_total(tmp_path, monkeypatch):
    monkeypatch.setattr(shards, "MAX_SHARDS", 8192)
    store = ArtifactStore(tmp_path / "store")
    features = _two_shard_section(store, "features")
    training = _two_shard_section(store, "training")
    assert [len(features["shards"]), len(training["shards"])] == [2, 2]
    manifest = {"schema_version": shards.MANIFEST_SCHEMA, "inline": {},
                "sections": {"features": features, "training": training}}
    expected_id = shards.manifest_id(manifest)
    monkeypatch.setattr(shards, "MAX_SHARDS", 2)
    with pytest.raises(OpsError) as caught:
        shards.verify_manifest(manifest, expected_id)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert "shard limit" in caught.value.problem.message


def test_verify_manifest_section_count_binding(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    entry = shards.write_section(store, "features", rows(1), key)
    too_many = {"schema_version": shards.MANIFEST_SCHEMA, "inline": {},
                "sections": {"features": {"row_count": 2, "shards": entry["shards"]}}}
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        shards.verify_manifest(too_many, shards.manifest_id(too_many))
    empty_with_shards = {"schema_version": shards.MANIFEST_SCHEMA, "inline": {},
                         "sections": {"features": {"row_count": 0, "shards": entry["shards"]}}}
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        shards.verify_manifest(empty_with_shards, shards.manifest_id(empty_with_shards))


def _collecting_reads(monkeypatch):
    reads = []
    original = ArtifactStore.read_verified

    def counted(self, ref):
        reads.append(ref)
        return original(self, ref)

    monkeypatch.setattr(ArtifactStore, "read_verified", counted)
    return reads


def test_section_total_refuses_before_any_read(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    entry = shards.write_section(store, "features", rows(2), key)
    broken = {"row_count": 3, "shards": entry["shards"]}
    reads = _collecting_reads(monkeypatch)
    collected = []
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        for item in shards.iter_section(store, "features", broken, key):
            collected.append(item)
    assert collected == []
    assert reads == []


def test_later_zero_row_shard_refuses_before_any_read(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    entry = shards.write_section(store, "features", rows(1), key)
    shard = entry["shards"][0]
    broken = {"row_count": 1, "shards": [shard, {**shard, "rows": 0}]}
    reads = _collecting_reads(monkeypatch)
    collected = []
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        for item in shards.iter_section(store, "features", broken, key):
            collected.append(item)
    assert collected == []
    assert reads == []


def test_section_shard_budget_refuses_before_any_read(tmp_path, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    monkeypatch.setattr(shards, "MAX_SHARDS", 2)
    shard = {"ref": {}, "rows": 1, "first_key": [0], "last_key": [1]}
    entry = {"row_count": 3, "shards": [shard, shard, shard]}
    reads = _collecting_reads(monkeypatch)
    collected = []
    with refusal("EXPERIMENT_IDENTITY_CONFLICT", False):
        for item in shards.iter_section(store, "features", entry, key):
            collected.append(item)
    assert collected == []
    assert reads == []


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
    expected = {
        "features": ("event_id", "context_id", "horizon_sessions", "feature columns"),
        "training": ("event_id", "absolute_move_pct"),
        "preflight.event_ids": ("event_id",),
        "preflight.planned_inputs": ("event_id", "fragment_id", "object_id", "content_hash"),
        "training_inputs.candidate_event_ids": ("event_id",),
        "training_inputs.qualified_event_ids": ("event_id",),
        "training_inputs.read_inputs": ("event_id", "fragment_id", "object_id", "content_hash"),
        "training_inputs.omitted_labels": ("event_id", "reason"),
    }
    members = shards.LIST_MEMBERS
    assert [member["section"] for member in members] == list(expected)
    for member in members:
        assert member["rows"] == expected[member["section"]]
        if member["section"] == "features":
            assert callable(member["key"])
            assert "problem" not in member
        else:
            assert member["key"] is None
            assert "no strictly increasing key" in member["problem"]


def test_manifest_id_tracks_inline_and_content():
    first = shards.build_manifest({"stored_inputs_id": "sha256:x"}, {})
    second = shards.build_manifest({"stored_inputs_id": "sha256:y"}, {})
    assert shards.manifest_id(first) == content_hash(first)
    assert shards.manifest_id(second) == content_hash(second)
    assert shards.manifest_id(first) != shards.manifest_id(second)


def test_write_snapshots_rows_against_generator_mutation(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    yielded = []

    def mutating():
        shared = {"event_id": "E000", "horizon_sessions": 1, "value": 0}
        for index in range(6):
            shared["event_id"] = f"E{index:03d}"
            shared["value"] = index
            yielded.append(dict(shared))
            yield shared

    written = shards.write_section(store, "features", mutating(), key)
    recovered = list(shards.iter_section(store, "features", written, key))
    assert recovered == rows(6)
    assert recovered == yielded
    assert written["row_count"] == 6
    assert [entry["rows"] for entry in written["shards"]] == [6]


def test_wire_constants_are_literal():
    assert shards.SHARD_SCHEMA == "magnitude_search_shard.v1"
    assert shards.MANIFEST_SCHEMA == "magnitude_search_cohort_manifest.v1"
    assert shards.TARGET_BYTES == 4 * 1024 * 1024
    assert shards.MAX_SHARD_BYTES == 8 * 1024 * 1024
    assert shards.MAX_SHARDS == 8192


def test_features_key_returns_pair_and_refuses_missing_field():
    feature = next(member for member in shards.LIST_MEMBERS if member["section"] == "features")
    assert feature["key"]({"event_id": "E007", "horizon_sessions": 3}) == ("E007", 3)
    with pytest.raises(KeyError):
        feature["key"]({"event_id": "E007"})


@pytest.mark.parametrize("inline, sections", [
    (None, {}),
    ({}, None),
    ({}, {"s": 5}),
    ({}, {"features": {"row_count": 1, "shards": [{"bad": 1}]}}),
    ({}, {"features": {"row_count": 1, "shards": "nope"}}),
    ({"x": object()}, {}),
], ids=["inline_none", "sections_none", "section_not_dict", "shard_not_dict", "shards_not_list",
        "non_canonical_inline"])
def test_build_manifest_invalid_inputs_are_invalid_spec(inline, sections):
    with refusal("INVALID_EXPERIMENT_SPEC", False):
        shards.build_manifest(inline, sections)


def test_build_manifest_snapshots_caller_inputs(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    entry = shards.write_section(store, "features", rows(2), key)
    inline = {"stored_inputs_id": "sha256:x"}
    sections = {"features": entry}
    manifest = shards.build_manifest(inline, sections)
    before = json.loads(canonical_json(manifest))
    expected_id = shards.manifest_id(manifest)
    inline["stored_inputs_id"] = "sha256:y"
    sections["features"] = {}
    entry["row_count"] = 99
    entry["shards"][0]["rows"] = 99
    assert manifest == before
    assert shards.manifest_id(manifest) == expected_id


class RaisingStore:
    def __init__(self, error):
        self.error = error

    def publish_bytes(self, data, *, schema_ref):
        raise self.error

    def read_verified(self, ref):
        raise self.error


@pytest.mark.parametrize("error, code, retryable", [
    (ArtifactError("INTEGRITY_FAILED", "torn object"), "EXPERIMENT_IDENTITY_CONFLICT", False),
    (OSError("storage gone"), "RESOURCE_UNAVAILABLE", True),
], ids=["integrity_failed", "oserror"])
def test_publish_failures_map_to_shared_classification(error, code, retryable):
    with refusal(code, retryable):
        shards.write_section(RaisingStore(error), "features", rows(1), key)


def test_read_unsafe_path_is_retryable_resource_unavailable(tmp_path):
    real = ArtifactStore(tmp_path / "store")
    entry = shards.write_section(real, "features", rows(1), key)
    with refusal("RESOURCE_UNAVAILABLE", True):
        list(shards.iter_section(
            RaisingStore(ArtifactError("UNSAFE_PATH", "traversing path")), "features", entry, key))
