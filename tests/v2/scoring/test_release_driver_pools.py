"""Driver residual pool member resolution in ``release_bindings`` (issue #479).

Every case uses a synthetic temporary release only: no real data, no fixed
paths, no production thresholds. The member's bytes are verified against the
catalog before decoding, so a declared-but-broken member refuses all-or-nothing
with the exact member id and a path-free message.
"""
import json
import unittest.mock
from pathlib import Path
from types import MappingProxyType

import pytest

from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import Lineage
from engine.v2.models.residual_artifact import (
    make_driver_residual_pool_artifact,
    make_paired_residual_pool_artifact,
)
from engine.v2.scoring.release_bindings import ModelNotReady, resolve_release_binding
from tests.test_v2_scoring_release_bindings import (
    _assert_no_leak,
    _dep_root,
    _obj,
    _row,
    _sha,
    _stage_and_promote,
    _write_catalog,
    _write_object,
)

_MEMBER = "driver_residual_pool:size"


def _driver(role="size", model_id="m-size"):
    artifact = make_driver_residual_pool_artifact(
        role=role, model_id=model_id, fold=None, flat_residuals=(0.1, -0.2, 0.3),
        buckets=None, deciles=10, min_pool=2, lineage=Lineage())
    return artifact, serialize_frozen_state(artifact)


def _staged_object(tmp_path, payload, content_hash):
    path = _write_object(_dep_root(tmp_path), payload)
    return _obj(path, content_hash)


def test_valid_member_loads_immutably_and_repeatably(tmp_path):
    _stage_and_promote(tmp_path)
    artifact, payload = _driver()
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_staged_object(tmp_path, payload, artifact.content_hash)])])

    binding = resolve_release_binding(tmp_path)

    assert binding.driver_residual_artifacts == {"size": artifact}
    assert isinstance(binding.driver_residual_artifacts, MappingProxyType)
    with pytest.raises(TypeError):
        binding.driver_residual_artifacts["size"] = artifact

    again = resolve_release_binding(tmp_path)
    assert again.driver_residual_artifacts == binding.driver_residual_artifacts


def test_no_member_yields_an_empty_mapping(tmp_path):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path, rows=[])

    binding = resolve_release_binding(tmp_path)

    assert dict(binding.driver_residual_artifacts) == {}
    assert isinstance(binding.driver_residual_artifacts, MappingProxyType)


def test_non_staged_member_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    artifact, payload = _driver()
    _write_catalog(tmp_path, rows=[_row(
        _MEMBER, [_staged_object(tmp_path, payload, artifact.content_hash)], status="PENDING")])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_empty_member_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_malformed_object_reference_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_obj("objects/x", None)])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_object_path_escape_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    artifact, payload = _driver()
    (tmp_path / "escape.json").write_bytes(payload)
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_obj("../escape.json", artifact.content_hash)])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_object_path_with_embedded_nul_refuses(tmp_path):
    """A hash-valid catalog member whose object path contains an embedded NUL
    byte makes the real ``Path.resolve`` raise a ``ValueError``; the resolver
    translates that to the exact path-free ``ModelNotReady`` naming the
    member, chaining the original ``ValueError`` and writing no file at the
    invalid path. The escape refusal for a resolved path stays separate."""
    _stage_and_promote(tmp_path)
    artifact, _payload = _driver()
    _write_catalog(tmp_path, rows=[_row(
        _MEMBER, [_obj("objects/\x00artifact", artifact.content_hash)])])

    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)

    assert error.value.member_id == _MEMBER
    assert error.value.detail == "driver residual pool: object path is invalid"
    assert isinstance(error.value.__cause__, ValueError)
    _assert_no_leak(tmp_path, error.value)


def test_absent_object_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    missing = "objects/" + "0" * 64
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_obj(missing, _sha(b"x"))])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_unreadable_object_refuses_without_leaking_path(tmp_path):
    _stage_and_promote(tmp_path)
    artifact, payload = _driver()
    staged = _staged_object(tmp_path, payload, artifact.content_hash)
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [staged])])
    object_path = _dep_root(tmp_path) / staged["path"]
    real_read = Path.read_bytes

    def failing_read(self):
        if self == object_path:
            raise OSError("simulated read failure")
        return real_read(self)

    with unittest.mock.patch.object(Path, "read_bytes", failing_read):
        with pytest.raises(ModelNotReady) as error:
            resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    assert "simulated read failure" not in error.value.detail
    _assert_no_leak(tmp_path, error.value)


def test_declared_hash_mismatch_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    _artifact, payload = _driver()
    _write_catalog(tmp_path, rows=[_row(
        _MEMBER, [_staged_object(tmp_path, payload, _sha(b"another document"))])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_member_bytes_disagreeing_with_its_own_hash_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    artifact, payload = _driver()
    padded = payload + b" "
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_staged_object(tmp_path, padded, _sha(padded))])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    assert error.value.detail == "driver residual pool disagrees with its own hash"
    _assert_no_leak(tmp_path, error.value)


def test_non_json_member_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    junk = b"not valid json"
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_staged_object(tmp_path, junk, _sha(junk))])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_unknown_schema_member_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    junk = json.dumps({"schema_version": "not_a_residual_pool.v9.9"}).encode()
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_staged_object(tmp_path, junk, _sha(junk))])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_paired_artifact_in_the_driver_slot_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    paired = make_paired_residual_pool_artifact(
        move_model_id="m-size", crush_model_id="m-crush", cutoff=None,
        rows=[["2026-01-01", "AAA", 1.0, 0.1, 0.2]], lineage=Lineage())
    payload = serialize_frozen_state(paired)
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_staged_object(tmp_path, payload, paired.content_hash)])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_artifact_role_mismatch_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    artifact, payload = _driver(role="implied_t1")
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_staged_object(tmp_path, payload, artifact.content_hash)])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_duplicate_member_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    artifact, payload = _driver()
    obj = _staged_object(tmp_path, payload, artifact.content_hash)
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [obj]), _row(_MEMBER, [obj])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    _assert_no_leak(tmp_path, error.value)


def test_non_mapping_lineage_document_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    _artifact, payload = _driver()
    document = json.loads(payload)
    document["lineage"] = ["not-a-mapping"]
    mutated = json.dumps(document).encode()
    _write_catalog(tmp_path, rows=[_row(
        _MEMBER, [_staged_object(tmp_path, mutated, _sha(mutated))])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    assert error.value.detail == "driver residual pool could not be verified or loaded"
    assert isinstance(error.value.__cause__, AttributeError)
    _assert_no_leak(tmp_path, error.value)


def test_overflowing_flat_residual_refuses(tmp_path):
    _stage_and_promote(tmp_path)
    _artifact, payload = _driver()
    document = json.loads(payload)
    document["flat_residuals"] = [10**400, 0.1, 0.3]
    mutated = json.dumps(document).encode()
    _write_catalog(tmp_path, rows=[_row(
        _MEMBER, [_staged_object(tmp_path, mutated, _sha(mutated))])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    assert error.value.detail == "driver residual pool could not be verified or loaded"
    assert isinstance(error.value.__cause__, OverflowError)
    _assert_no_leak(tmp_path, error.value)


def test_integer_string_digit_limit_member_refuses(tmp_path):
    """A hash-valid member whose JSON number exceeds the interpreter's integer
    string digit limit refuses as not valid JSON, caused by a bare ValueError
    from json's number parser -- never escapes as an uncaught refusal."""
    _stage_and_promote(tmp_path)
    big = b"[" + b"1" * 10000 + b"]"
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_staged_object(tmp_path, big, _sha(big))])])
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    assert error.value.detail == "driver residual pool is not valid JSON"
    assert isinstance(error.value.__cause__, ValueError)
    assert "digit" in str(error.value.__cause__).casefold()
    _assert_no_leak(tmp_path, error.value)


def test_member_json_recursion_error_refuses(tmp_path):
    """A hash-valid member whose JSON document makes ``json.loads`` hit the
    interpreter recursion limit refuses as not valid JSON, caused by a
    RecursionError out of the member decode -- never escapes as an uncaught
    refusal. The patch replaces the shared ``json.loads`` attribute for the
    duration of this test, but raises only for the exact member payload and
    delegates catalog and manifest inputs to the saved real loader."""
    _stage_and_promote(tmp_path)
    artifact, payload = _driver()
    _write_catalog(tmp_path, rows=[_row(_MEMBER, [_staged_object(tmp_path, payload, artifact.content_hash)])])
    real_loads = json.loads

    def poisoned_loads(arg, *args, **kwargs):
        if arg == payload:
            raise RecursionError("json document nesting exceeds the interpreter limit")
        return real_loads(arg, *args, **kwargs)

    with unittest.mock.patch(
            "engine.v2.scoring.release_bindings.json.loads", poisoned_loads):
        with pytest.raises(ModelNotReady) as error:
            resolve_release_binding(tmp_path)
    assert error.value.member_id == _MEMBER
    assert error.value.detail == "driver residual pool is not valid JSON"
    assert isinstance(error.value.__cause__, RecursionError)
    _assert_no_leak(tmp_path, error.value)