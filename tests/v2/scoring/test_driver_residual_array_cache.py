"""Bounded immutable conversion reuse never replaces per-call identity checks."""
from concurrent.futures import ThreadPoolExecutor

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
    return module.driver_pool_from_artifact(block(value), "driver", prediction)


@pytest.mark.parametrize("prediction,expected", [(-1.0, [1., 2.]), (0.0, [1., 2., 3., 4.]),
    (10.0, [1., 2., 3., 4.]), (None, [1., 2., 3., 4.]), (np.nan, [1., 2., 3., 4.])])
def test_cold_warm_values_fallback_and_sampling_match(prediction, expected):
    value = artifact()
    for _ in range(2):
        pool, flag = lookup(value, prediction)
        assert flag is None
        np.testing.assert_array_equal(pool, expected)
        np.testing.assert_array_equal(np.random.default_rng(7).choice(pool, 20),
                                      np.random.default_rng(7).choice(expected, 20))


def test_warm_hit_does_not_convert_and_arrays_cannot_be_made_writable(monkeypatch):
    value = artifact()
    cold, _ = lookup(value)
    monkeypatch.setattr(module, "_immutable_array", lambda *a: pytest.fail("warm hit converted"))
    warm, _ = lookup(value)
    assert cold is warm
    buckets, flat, _ = module._DRIVER_ARRAY_CACHE[value.content_hash]
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
    for value in values:
        lookup(value)
    assert list(module._DRIVER_ARRAY_CACHE) == [value.content_hash for value in values[1:]]
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
    assert module._DRIVER_ARRAY_CACHE[first.content_hash][2] == 80
    lookup(second)
    assert list(module._DRIVER_ARRAY_CACHE) == [second.content_hash]
    monkeypatch.setattr(module, "_DRIVER_ARRAY_CACHE_BYTES", 79)
    module._DRIVER_ARRAY_CACHE.clear()
    for _ in range(2):
        np.testing.assert_array_equal(lookup(first)[0], [1.0, 2.0])
        assert module._DRIVER_ARRAY_CACHE == {}


def test_unbucketed_empty_and_unpinned_requests_keep_behavior():
    value = artifact(buckets=False)
    request = block(value)
    del request[module.RESIDUAL_KEYS_FIELD]["driver"]["content_hash"]
    np.testing.assert_array_equal(module.driver_pool_from_artifact(request, "driver", 0)[0], [1., 2., 3., 4.])
    assert lookup(artifact(buckets=False, empty=True)) == (None, "MISSING_MODEL_RESIDUALS")


def test_concurrent_misses_remain_bounded(monkeypatch):
    monkeypatch.setattr(module, "_DRIVER_ARRAY_CACHE_BYTES", 160)
    values = [artifact(i) for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lookup, values * 3))
    for pool, flag in results:
        assert flag is None
        np.testing.assert_array_equal(pool, [1.0, 2.0])
    assert len(module._DRIVER_ARRAY_CACHE) <= 2
    assert sum(entry[2] for entry in module._DRIVER_ARRAY_CACHE.values()) <= 160


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
