"""Bounded immutable conversion reuse never replaces per-call identity checks."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest

from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact
from engine.v2.scoring import native_residuals as module


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(module, "_DRIVER_ARRAY_CACHE", {})


def artifact(index=0, *, buckets=True, empty=False):
    return make_driver_residual_pool_artifact(
        role="size", model_id=f"model-{index}", fold="2025-01-01",
        flat_residuals=() if empty else (1.0, 2.0, 3.0, 4.0),
        buckets={"edges": (-np.inf, 0.0, np.inf), "pools": ((1.0, 2.0), (3.0,)), "min_pool": 2} if buckets else None,
        deciles=2, min_pool=2, lineage=Lineage(data=(DataDependency(table="synthetic"),)))


def block(value):
    return {module.RESIDUAL_ARTIFACTS_FIELD: {"driver": value},
            module.RESIDUAL_KEYS_FIELD: {"driver": {
                "role": value.role, "model_id": value.model_id, "fold": value.fold,
                "content_hash": value.content_hash}}}


def lookup(value, prediction=-1.0):
    pool, flag = module.driver_pool_from_artifact(block(value), "driver", prediction)
    assert flag is None
    return pool, flag


@pytest.mark.parametrize("prediction,expected", [(-1.0, [1., 2.]), (0.0, [1., 2., 3., 4.]),
    (10.0, [1., 2., 3., 4.]), (None, [1., 2., 3., 4.]), (np.nan, [1., 2., 3., 4.])])
def test_cold_warm_values_fallback_and_sampling_match(prediction, expected):
    value = artifact()
    cold = None
    for _ in range(2):
        pool, flag = lookup(value, prediction)
        if cold is not None:
            assert pool is cold
        cold = pool
        assert flag is None
        np.testing.assert_array_equal(pool, expected)
        np.testing.assert_array_equal(np.random.default_rng(7).choice(pool, 20),
                                      np.random.default_rng(7).choice(expected, 20))


def test_warm_hit_does_not_convert_and_arrays_cannot_be_made_writable(monkeypatch):
    value = artifact()
    cold, _ = lookup(value)
    monkeypatch.setattr(module, "_immutable_array", lambda *a: pytest.fail("warm hit converted"))
    monkeypatch.setattr(module, "_deeply_frozen", lambda *a: pytest.fail("warm hit rescanned input"))
    warm, _ = lookup(value)
    assert cold is warm
    buckets, flat, _, retained = module._DRIVER_ARRAY_CACHE[id(value)]
    assert retained is value
    for array in (flat, buckets["edges"], *buckets["pools"]):
        with pytest.raises(ValueError):
            array[0] = 99
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("field,bad", [("role", "wrong"), ("model_id", "wrong"),
    ("fold", "2024-01-01"), ("content_hash", "wrong")])
def test_every_warm_call_rechecks_expected_identity(field, bad, monkeypatch):
    value = artifact()
    lookup(value)
    request = block(value)
    request[module.RESIDUAL_KEYS_FIELD]["driver"][field] = bad
    monkeypatch.setattr(module, "_driver_arrays", lambda *a: pytest.fail("mismatch consulted cache"))
    assert module.driver_pool_from_artifact(request, "driver", -1) == (None, "MODEL_NOT_READY")


@pytest.mark.parametrize("missing", ["artifact", "key", "key_part"])
def test_warm_cache_cannot_supply_missing_authority(missing):
    value = artifact()
    lookup(value)
    request = block(value)
    if missing == "artifact":
        request[module.RESIDUAL_ARTIFACTS_FIELD].clear()
    elif missing == "key":
        request[module.RESIDUAL_KEYS_FIELD].clear()
    else:
        del request[module.RESIDUAL_KEYS_FIELD]["driver"]["role"]
    assert module.driver_pool_from_artifact(request, "driver", -1) == (None, "MODEL_NOT_READY")


def test_new_content_does_not_reuse_old_arrays():
    first = artifact()
    second = make_driver_residual_pool_artifact(role=first.role, model_id=first.model_id, fold=first.fold,
        flat_residuals=(9.0,), buckets=None, deciles=2, min_pool=2, lineage=first.lineage)
    lookup(first)
    np.testing.assert_array_equal(lookup(second)[0], [9.0])
    request = block(second)
    request[module.RESIDUAL_KEYS_FIELD]["driver"]["content_hash"] = first.content_hash
    assert module.driver_pool_from_artifact(request, "driver", -1) == (None, "MODEL_NOT_READY")


def test_fifo_entry_cap_and_rebuild(monkeypatch):
    values = [artifact(i) for i in range(5)]
    for value in values[:4]:
        lookup(value)
    lookup(values[0])  # A hit must not promote FIFO order into LRU order.
    lookup(values[4])
    assert list(module._DRIVER_ARRAY_CACHE) == [id(value) for value in values[1:]]
    original, calls = module._immutable_array, []
    def convert(value):
        calls.append(1)
        return original(value)
    monkeypatch.setattr(module, "_immutable_array", convert)
    np.testing.assert_array_equal(lookup(values[0])[0], [1.0, 2.0])
    assert len(calls) == 4
    assert len(module._DRIVER_ARRAY_CACHE) == 4


def test_byte_cap_eviction_and_oversized_nonretention(monkeypatch):
    # One artifact has 4 flat + 3 edge + 3 bucket floats = 80 bytes.
    monkeypatch.setattr(module, "_DRIVER_ARRAY_CACHE_BYTES", 80)
    first, second = artifact(1), artifact(2)
    lookup(first)
    assert module._DRIVER_ARRAY_CACHE[id(first)][2] == 80
    lookup(second)
    assert list(module._DRIVER_ARRAY_CACHE) == [id(second)]
    monkeypatch.setattr(module, "_DRIVER_ARRAY_CACHE_BYTES", 79)
    module._DRIVER_ARRAY_CACHE.clear()
    for _ in range(2):
        np.testing.assert_array_equal(lookup(first)[0], [1.0, 2.0])
        assert module._DRIVER_ARRAY_CACHE == {}


def test_unbucketed_empty_and_unpinned_requests_keep_behavior():
    value = artifact(buckets=False)
    request = block(value)
    del request[module.RESIDUAL_KEYS_FIELD]["driver"]["content_hash"]
    pool, flag = module.driver_pool_from_artifact(request, "driver", 0)
    assert flag is None
    np.testing.assert_array_equal(pool, [1., 2., 3., 4.])
    empty = block(artifact(buckets=False, empty=True))
    assert module.driver_pool_from_artifact(empty, "driver", -1) == (None, "MISSING_MODEL_RESIDUALS")


def test_concurrent_misses_remain_bounded(monkeypatch):
    monkeypatch.setattr(module, "_DRIVER_ARRAY_CACHE_BYTES", 160)
    values = [make_driver_residual_pool_artifact(role="size", model_id=f"model-{i}", fold=None,
        flat_residuals=(float(i),) * 10, buckets=None, deciles=1, min_pool=1, lineage=Lineage())
        for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lookup, values * 3))
    for value, (pool, flag) in zip(values * 3, results, strict=True):
        assert flag is None
        np.testing.assert_array_equal(pool, value.flat_residuals)
    assert len(module._DRIVER_ARRAY_CACHE) == 2
    assert sum(entry[2] for entry in module._DRIVER_ARRAY_CACHE.values()) == 160


def test_corrupted_serialized_pool_cannot_hit_warm_original_hash():
    from engine.v2.models.residual_artifact import residual_artifact_from_document

    original = artifact(buckets=False)
    lookup(original)
    document = original.payload()
    document["flat_residuals"][0] = 999.0
    changed = residual_artifact_from_document(document)
    assert changed.content_hash != original.content_hash
    request = block(changed)
    request[module.RESIDUAL_KEYS_FIELD]["driver"]["content_hash"] = original.content_hash
    assert module.driver_pool_from_artifact(request, "driver", 0) == (None, "MODEL_NOT_READY")
    np.testing.assert_array_equal(lookup(original, None)[0], [1., 2., 3., 4.])


@pytest.mark.parametrize("warm_original", [False, True])
def test_direct_artifacts_with_same_declared_hash_keep_their_own_contents(warm_original):
    original = artifact(buckets=False)
    forged = replace(original, flat_residuals=(999.0,))
    assert forged.content_hash == original.content_hash
    if warm_original:
        lookup(original)
    np.testing.assert_array_equal(lookup(forged)[0], [999.0])
    np.testing.assert_array_equal(lookup(forged)[0], [999.0])
    np.testing.assert_array_equal(lookup(original)[0], original.flat_residuals)


def test_distinct_equal_objects_convert_independently_and_same_object_reuses(monkeypatch):
    original, equal = artifact(), artifact()
    convert, calls = module._immutable_array, []
    def track(values):
        calls.append(values)
        return convert(values)
    monkeypatch.setattr(module, "_immutable_array", track)
    first, _ = lookup(original)
    second, _ = lookup(equal)
    assert first is not second
    assert len(calls) == 8
    assert lookup(original)[0] is first
    assert lookup(equal)[0] is second
    assert len(calls) == 8


def test_shallow_frozen_mutable_input_is_never_retained():
    original = artifact(buckets=False)
    mutable = replace(original, flat_residuals=list(original.flat_residuals))
    first, _ = lookup(mutable)
    np.testing.assert_array_equal(first, original.flat_residuals)
    assert not module._DRIVER_ARRAY_CACHE
    mutable.flat_residuals[0] = 999.0
    np.testing.assert_array_equal(lookup(mutable)[0], [999.0, 2.0, 3.0, 4.0])
    assert not module._DRIVER_ARRAY_CACHE
    np.testing.assert_array_equal(first, original.flat_residuals)


def test_default_numeric_byte_and_entry_caps_are_the_published_contract():
    assert module._DRIVER_ARRAY_CACHE_BYTES == 16 * 1024 * 1024
    assert module._DRIVER_ARRAY_CACHE_LIMIT == 4


@pytest.mark.parametrize("field", ["oversized", "flat_residuals", "bucket_edges", "bucket_pools", "nested_pool"])
def test_bypassed_inputs_preserve_residents_and_observe_mutable_values(monkeypatch, field):
    monkeypatch.setattr(module, "_DRIVER_ARRAY_CACHE_BYTES", 80)
    resident = artifact(1)
    retained, flag = lookup(resident)
    assert flag is None
    entry = module._DRIVER_ARRAY_CACHE[id(resident)]
    candidate = artifact(2)
    if field == "oversized":
        candidate = replace(candidate, flat_residuals=candidate.flat_residuals + (5.0,))
    elif field == "nested_pool":
        candidate = replace(candidate, bucket_pools=(list(candidate.bucket_pools[0]), candidate.bucket_pools[1]))
    else:
        candidate = replace(candidate, **{field: list(getattr(candidate, field))})
    for changed in (False, True):
        if changed:
            if field == "flat_residuals":
                candidate.flat_residuals[0] = 99.0
            elif field == "bucket_edges":
                candidate.bucket_edges[1] = -2.0
            elif field == "bucket_pools":
                candidate.bucket_pools[0] = (99.0, 98.0)
            elif field == "nested_pool":
                candidate.bucket_pools[0][0] = 99.0
        prediction = None if field == "flat_residuals" else -1.0
        pool, flag = lookup(candidate, prediction)
        expected = [1.0, 2.0]
        if field == "flat_residuals":
            expected = [99.0 if changed else 1.0, 2.0, 3.0, 4.0]
        elif changed and field == "bucket_edges":
            expected = [1.0, 2.0, 3.0, 4.0]
        elif changed and field == "bucket_pools":
            expected = [99.0, 98.0]
        elif changed and field == "nested_pool":
            expected = [99.0, 2.0]
        assert flag is None
        np.testing.assert_array_equal(pool, expected)
        assert list(module._DRIVER_ARRAY_CACHE) == [id(resident)]
        assert module._DRIVER_ARRAY_CACHE[id(resident)] is entry
        assert lookup(resident)[0] is retained


def test_overlapping_same_object_misses_reuse_winner_without_evicting_peers(monkeypatch):
    peers = [artifact(i) for i in range(3)]
    resident_arrays = [lookup(peer)[0] for peer in peers]
    entries = dict(module._DRIVER_ARRAY_CACHE)
    target = artifact(4, buckets=False)
    barrier = Barrier(2, timeout=5)
    convert = module._immutable_array
    def overlapping(values):
        # Both callers have already missed; neither may insert until both
        # private conversions exist. This forces the under-lock winner path.
        result = convert(values)
        barrier.wait()
        return result
    monkeypatch.setattr(module, "_immutable_array", overlapping)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lookup, target) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    (first, flag1), (second, flag2) = results
    assert flag1 is flag2 is None
    np.testing.assert_array_equal(first, [1.0, 2.0, 3.0, 4.0])
    assert first is second
    assert list(module._DRIVER_ARRAY_CACHE) == [*(id(peer) for peer in peers), id(target)]
    assert sum(entry[2] for entry in module._DRIVER_ARRAY_CACHE.values()) == 272
    for peer, array in zip(peers, resident_arrays, strict=True):
        assert module._DRIVER_ARRAY_CACHE[id(peer)] is entries[id(peer)]
        assert lookup(peer)[0] is array
    assert lookup(target)[0] is first
