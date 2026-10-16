"""Synthetic release-owned chooser state; no providers or stored market data."""
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from engine.v2.models.admissible_table import make_admissible_depth_table
from engine.v2.models import deployment, frozen_state_io
from engine.v2.models.chooser_analog_pool import make_chooser_analog_pool_artifact
from engine.v2.models.frozen_state import FrozenStateError, serialize_frozen_state
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.scoring import release_bindings
from engine.v2.scoring.release_bindings import ModelNotReady, resolve_release_binding
from tests.v2.scoring.test_v2_scoring_release_bindings import (
    _assert_no_leak,
    _dep_root,
    _happy_catalog,
    _obj,
    _row,
    _sha,
    _stage_and_promote,
    _write_catalog,
    _write_object,
)

_POOL = "chooser_analog_pool"
_TABLE = "admissible_table:dyn_sv"
_FIELDS = {_POOL: "chooser_analog_artifacts", _TABLE: "chooser_admissible_artifacts"}


def _pool(*, pool_id="synthetic-pool", cutoff="2026-02-01", pnl=7.0):
    return make_chooser_analog_pool_artifact(
        pool_id=pool_id, cutoff=cutoff,
        strategies={"TWIN-P": [
            ["2026-01-20", pnl, 1.0, 2.0, 3.0, 4.0, 5.0],
            ["2026-01-10", -2.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]}, lineage=Lineage(data=(DataDependency(
            table="synthetic-candidates", end_exclusive="2026-02-01"),)))


def _table(*, table_id="synthetic-table", version="synthetic-v1", fallback=4.0):
    return make_admissible_depth_table(
        table_id=table_id, version=version,
        breakpoints=((2.0, 3.0), (5.0, 8.0)), fallback=fallback,
        provenance="synthetic calibration", lineage=Lineage())


def _artifact(member):
    return _pool() if member == _POOL else _table()


def _stored(root, artifact):
    payload = serialize_frozen_state(artifact)
    return _obj(_write_object(_dep_root(root), payload), artifact.content_hash,
                name="untrusted-name-is-not-the-key")


def _catalog(root, member, *, objects=None, status="STAGED"):
    if objects is None:
        objects = [_stored(root, _artifact(member))]
    row = _row(member, [])
    row.update(objects=objects, status=status)
    _write_catalog(root, rows=[row])


def _assert_refused(root, member):
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(root)
    assert error.value.code == "MODEL_NOT_READY"
    assert error.value.member_id == member
    _assert_no_leak(root, error.value)
    return error.value


def test_loads_all_keys_in_catalog_order_with_verified_immutable_values(tmp_path):
    _stage_and_promote(tmp_path)
    pools = (_pool(cutoff=None), _pool())
    tables = (_table(version="synthetic-v2"), _table())
    _write_catalog(tmp_path, rows=[
        _row(_POOL, [_stored(tmp_path, artifact) for artifact in pools]),
        _row(_TABLE, [_stored(tmp_path, artifact) for artifact in tables]),
    ])

    binding = resolve_release_binding(tmp_path)

    assert isinstance(binding.chooser_analog_artifacts, tuple)
    assert isinstance(binding.chooser_admissible_artifacts, tuple)
    assert [a.key for a in binding.chooser_analog_artifacts] == [
        ("synthetic-pool", None), ("synthetic-pool", "2026-02-01")]
    assert [a.key for a in binding.chooser_admissible_artifacts] == [
        ("synthetic-table", "synthetic-v2"), ("synthetic-table", "synthetic-v1")]
    pool = binding.chooser_analog_artifacts[1]
    assert pool.rows_for("TWIN-P") == (
        ("2026-01-20", 7.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        ("2026-01-10", -2.0, 6.0, 7.0, 8.0, 9.0, 10.0))
    assert pool.rows_for("BFLY-P") is None
    assert pool.lineage.data[0].table == "synthetic-candidates"
    assert pool.lineage.data[0].end_exclusive == "2026-02-01"
    table = binding.chooser_admissible_artifacts[1]
    assert table.breakpoints == ((2.0, 3.0), (5.0, 8.0))
    assert table.fallback == 4.0
    assert table.provenance == "synthetic calibration"
    assert [a.content_hash for a in binding.chooser_analog_artifacts] == [
        _sha(serialize_frozen_state(a)) for a in pools]
    assert [a.content_hash for a in binding.chooser_admissible_artifacts] == [
        _sha(serialize_frozen_state(a)) for a in tables]
    with pytest.raises(FrozenInstanceError):
        binding.chooser_analog_artifacts = ()
    with pytest.raises(FrozenInstanceError):
        table.fallback = 99.0
    assert resolve_release_binding(tmp_path) == binding


@pytest.mark.parametrize("member,artifacts,expected_keys", [
    (_POOL, (_pool(pool_id="pool-b", pnl=17.0), _pool(pool_id="pool-a", pnl=3.0)),
     [("pool-b", "2026-02-01"), ("pool-a", "2026-02-01")]),
    (_TABLE, (_table(table_id="table-b", fallback=9.0), _table(table_id="table-a")),
     [("table-b", "synthetic-v1"), ("table-a", "synthetic-v1")]),
], ids=["distinct-pool-ids", "distinct-table-ids"])
@pytest.mark.parametrize("separate_rows", [False, True])
def test_distinct_ids_sharing_cutoff_or_version_are_not_duplicates(
        tmp_path, member, artifacts, expected_keys, separate_rows):
    """Full identity matters: comparing only key[1] rejects these valid states."""
    _stage_and_promote(tmp_path)
    objects = [_stored(tmp_path, artifact) for artifact in artifacts]
    rows = ([_row(member, [obj]) for obj in objects] if separate_rows
            else [_row(member, objects)])
    _write_catalog(tmp_path, rows=rows)

    binding = resolve_release_binding(tmp_path)

    loaded = getattr(binding, _FIELDS[member])
    assert [artifact.key for artifact in loaded] == expected_keys
    # Literal identity/order/value expectations catch omission, sorting or
    # replacement with the other object's content as well as false refusal.
    if member == _POOL:
        assert [artifact.rows_for("TWIN-P")[0][1] for artifact in loaded] == [17.0, 3.0]
        assert binding.chooser_admissible_artifacts == ()
    else:
        assert [artifact.fallback for artifact in loaded] == [9.0, 4.0]
        assert binding.chooser_analog_artifacts == ()


@pytest.mark.parametrize("member", [_POOL, _TABLE, None])
def test_absence_is_independent_and_preserves_other_release_state(tmp_path, member):
    _stage_and_promote(tmp_path)
    _happy_catalog(tmp_path)
    catalog = json.loads((tmp_path / "phase5_release.json").read_bytes())
    if member is not None:
        catalog["members"].append(_row(member, [_stored(tmp_path, _artifact(member))]))
    _write_catalog(tmp_path, rows=catalog["members"])

    binding = resolve_release_binding(tmp_path)

    for expected_member, field in _FIELDS.items():
        assert len(getattr(binding, field)) == int(expected_member == member)
    assert set(binding.payoff_artifacts) == {"STR-THRU"}
    assert set(binding.recalibration_artifacts) == {"STR-THRU"}
    assert set(binding.analog_artifacts) == {"STR-THRU"}
    assert set(binding.model_identity) == {"gate:STR-THRU"}


def test_constructor_copies_mutable_sequences(tmp_path):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path)
    expected_pools = (_pool(pool_id="pool-b", pnl=17.0), _pool(pool_id="pool-a", pnl=3.0))
    expected_tables = (_table(table_id="table-b", fallback=9.0), _table(table_id="table-a"))
    pools, tables = list(expected_pools), list(expected_tables)
    binding = replace(resolve_release_binding(tmp_path),
                      chooser_analog_artifacts=pools, chooser_admissible_artifacts=tables)
    pools.clear()
    tables.clear()
    assert binding.chooser_analog_artifacts == expected_pools
    assert binding.chooser_admissible_artifacts == expected_tables


def test_empty_pool_and_unrelated_member_ids_preserve_existing_contract(tmp_path):
    _stage_and_promote(tmp_path)
    empty = make_chooser_analog_pool_artifact(
        pool_id="empty-pool", cutoff=None, strategies={}, lineage=Lineage())
    _write_catalog(tmp_path, rows=[
        _row(_POOL, [_stored(tmp_path, empty)]),
        _row(_POOL + ":unrelated", [], status="MISSING"),
        _row(_TABLE + ":unrelated", [], status="MISSING"),
    ])
    binding = resolve_release_binding(tmp_path)
    assert binding.chooser_analog_artifacts[0].strategies == ()
    assert binding.chooser_analog_artifacts[0].key == ("empty-pool", None)
    assert binding.chooser_admissible_artifacts == ()


@pytest.mark.parametrize("member", [_POOL, _TABLE])
@pytest.mark.parametrize("status", ["MISSING", "PENDING", None])
def test_declared_nonstaged_state_refuses_whole_release(tmp_path, member, status):
    _stage_and_promote(tmp_path)
    _catalog(tmp_path, member, status=status)
    _assert_refused(tmp_path, member)


@pytest.mark.parametrize("member", [_POOL, _TABLE])
@pytest.mark.parametrize("objects", [[], {}, "object", [None], [{}], [
    {"path": "objects/x", "content_hash": None}]])
def test_bad_object_declaration_refuses(tmp_path, member, objects):
    _stage_and_promote(tmp_path)
    _catalog(tmp_path, member, objects=objects)
    _assert_refused(tmp_path, member)


@pytest.mark.parametrize("member", [_POOL, _TABLE])
@pytest.mark.parametrize("location", ["missing", "escaped", "nul", "symlink-loop", "unreadable"])
def test_unusable_object_path_refuses_without_leaking(tmp_path, monkeypatch, member, location):
    _stage_and_promote(tmp_path)
    artifact = _artifact(member)
    obj = _stored(tmp_path, artifact)
    target = _dep_root(tmp_path) / obj["path"]
    if location == "missing":
        target.unlink()
    elif location == "escaped":
        obj["path"] = "../outside.json"
        (tmp_path / "outside.json").write_bytes(serialize_frozen_state(artifact))
    elif location == "nul":
        obj["path"] = "objects/\x00invalid"
    elif location == "symlink-loop":
        loop = _dep_root(tmp_path) / "loop"
        loop.symlink_to("loop")
        obj["path"] = "loop"
    else:
        real_open = frozen_state_io.os.open

        def unreadable(path, flags, *args, **kwargs):
            if Path(path) == target:
                raise OSError("private-file-details")
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(frozen_state_io.os, "open", unreadable)
    _catalog(tmp_path, member, objects=[obj])
    error = _assert_refused(tmp_path, member)
    assert "private-file-details" not in str(error)


@pytest.mark.parametrize("member", [_POOL, _TABLE])
@pytest.mark.parametrize("payload", [b"not json", b"\xff", b"[]", b"{}", b"[" * 2000])
def test_hash_valid_malformed_bytes_refuse(tmp_path, member, payload):
    _stage_and_promote(tmp_path)
    obj = _obj(_write_object(_dep_root(tmp_path), payload), _sha(payload))
    _catalog(tmp_path, member, objects=[obj])
    _assert_refused(tmp_path, member)


@pytest.mark.parametrize("member", [_POOL, _TABLE])
@pytest.mark.parametrize("defect", ["lineage", "wrong-type", "decoded-hash", "byte-hash"])
def test_corrupt_artifact_refuses_at_verified_type_boundary(tmp_path, member, defect):
    _stage_and_promote(tmp_path)
    artifact = _artifact(member)
    payload = serialize_frozen_state(artifact)
    if defect == "lineage":
        document = artifact.payload()
        document["lineage"] = ["not a mapping"]
        payload = json.dumps(document).encode()
    elif defect == "wrong-type":
        payload = serialize_frozen_state(_table() if member == _POOL else _pool())
    elif defect == "decoded-hash":
        payload += b" "  # Byte hash agrees, decoded canonical artifact hash does not.
    obj = _obj(_write_object(_dep_root(tmp_path), payload), _sha(payload))
    if defect == "byte-hash":
        obj["content_hash"] = _sha(b"another object")
    _catalog(tmp_path, member, objects=[obj])
    _assert_refused(tmp_path, member)


@pytest.mark.parametrize("field,value", [("k", float("inf")), ("strategies", [1])])
def test_pool_decoder_edge_cases_are_typed_refusals(tmp_path, field, value):
    _stage_and_promote(tmp_path)
    document = _pool().payload()
    document[field] = value
    payload = json.dumps(document).encode()
    _catalog(tmp_path, _POOL, objects=[
        _obj(_write_object(_dep_root(tmp_path), payload), _sha(payload))])
    _assert_refused(tmp_path, _POOL)


@pytest.mark.parametrize("member", [_POOL, _TABLE])
@pytest.mark.parametrize("error_type", [RecursionError, RuntimeError])
def test_loader_runtime_failures_are_path_free_on_every_python_version(
        tmp_path, monkeypatch, member, error_type):
    # Python versions differ in their parser recursion and symlink-loop
    # exceptions; inject at the real decoder after byte verification too.
    _stage_and_promote(tmp_path)
    _catalog(tmp_path, member)

    def failed_decode(document):
        raise error_type("private-decoder-details")

    monkeypatch.setattr("engine.v2.models.frozen_state._from_document", failed_decode)
    error = _assert_refused(tmp_path, member)
    assert isinstance(error.__cause__, error_type)
    assert "private-decoder-details" not in str(error)


@pytest.mark.parametrize("member", [_POOL, _TABLE])
@pytest.mark.parametrize("separate_rows", [False, True])
@pytest.mark.parametrize("same_bytes", [False, True])
def test_duplicate_semantic_keys_refuse_without_overwrite(tmp_path, member, separate_rows, same_bytes):
    _stage_and_promote(tmp_path)
    first = _artifact(member)
    second = first if same_bytes else (_pool(pnl=99.0) if member == _POOL else _table(fallback=9.0))
    objects = [_stored(tmp_path, a) for a in (first, second)]
    rows = ([_row(member, [obj]) for obj in objects] if separate_rows
            else [_row(member, objects)])
    _write_catalog(tmp_path, rows=rows)
    error = _assert_refused(tmp_path, member)
    assert error.detail == "ambiguous chooser state key"


@pytest.mark.parametrize("member", [_POOL, _TABLE])
def test_no_loader_cache_survives_resolution_or_hides_second_reference(tmp_path, member):
    _stage_and_promote(tmp_path)
    obj = _stored(tmp_path, _artifact(member))
    _catalog(tmp_path, member, objects=[obj])
    resolve_release_binding(tmp_path)
    target = _dep_root(tmp_path) / obj["path"]
    original = target.read_bytes()
    target.write_bytes(b"broken after first load")
    _assert_refused(tmp_path, member)
    target.write_bytes(original)
    _catalog(tmp_path, member, objects=[obj, {**obj, "path": "objects/absent-copy"}])
    error = _assert_refused(tmp_path, member)
    assert error.detail == "chooser state could not be verified or loaded"


def test_release_local_catalog_wins_and_failure_returns_no_partial_binding(tmp_path):
    _stage_and_promote(tmp_path)
    _catalog(tmp_path, _POOL)
    root_catalog = tmp_path / "phase5_release.json"
    local_catalog = _dep_root(tmp_path) / "releases" / "r1" / "phase5_release.json"
    local_catalog.write_bytes(root_catalog.read_bytes())
    _catalog(tmp_path, _TABLE)
    binding = resolve_release_binding(tmp_path)
    assert len(binding.chooser_analog_artifacts) == 1
    assert binding.chooser_admissible_artifacts == ()
    broken = [_row(_POOL, [_stored(tmp_path, _pool())]), _row(_TABLE, [])]
    _write_catalog(tmp_path, rows=broken)
    local_catalog.write_bytes(root_catalog.read_bytes())
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    _assert_refused(tmp_path, _TABLE)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before  # The entire release read remains side-effect free.


def test_promotion_and_rollback_expose_only_current_release_chooser_state(tmp_path):
    for release_id, table in (("r1", _table()), ("r2", _table(version="synthetic-v2"))):
        _stage_and_promote(tmp_path, release_id=release_id)
        _write_catalog(tmp_path, release_id=release_id,
                       rows=[_row(_TABLE, [_stored(tmp_path, table)])])
        local = _dep_root(tmp_path) / "releases" / release_id / "phase5_release.json"
        local.write_bytes((tmp_path / "phase5_release.json").read_bytes())
    binding = resolve_release_binding(tmp_path)
    assert binding.release_id == "r2"
    assert binding.chooser_admissible_artifacts[0].version == "synthetic-v2"
    deployment.rollback(_dep_root(tmp_path))
    restored = resolve_release_binding(tmp_path)
    assert restored.release_id == "r1"
    assert restored.chooser_admissible_artifacts[0].version == "synthetic-v1"


@pytest.mark.parametrize("member", [_POOL, _TABLE])
@pytest.mark.parametrize("headroom", [-1, 0, 1])
def test_chooser_object_byte_boundary(tmp_path, monkeypatch, member, headroom):
    _stage_and_promote(tmp_path)
    artifact = _artifact(member)
    _catalog(tmp_path, member, objects=[_stored(tmp_path, artifact)])
    limit = len(serialize_frozen_state(artifact)) + headroom
    monkeypatch.setattr(release_bindings, "_CHOOSER_MAX_ARTIFACT_BYTES", limit)
    if headroom < 0:
        error = _assert_refused(tmp_path, member)
        assert isinstance(error.__cause__, FrozenStateError)
        assert str(error.__cause__) == "frozen state exceeds byte limit"
    else:
        binding = resolve_release_binding(tmp_path)
        assert getattr(binding, _FIELDS[member]) == (artifact,)


@pytest.mark.parametrize("member", [_POOL, _TABLE])
def test_shipped_32_mib_limit_refuses_sparse_oversize_object(tmp_path, member):
    assert release_bindings._CHOOSER_MAX_ARTIFACT_BYTES == 33_554_432
    _stage_and_promote(tmp_path)
    obj = _stored(tmp_path, _artifact(member))
    with (_dep_root(tmp_path) / obj["path"]).open("wb") as stream:
        stream.truncate(33_554_433)
    _catalog(tmp_path, member, objects=[obj])
    error = _assert_refused(tmp_path, member)
    assert isinstance(error.__cause__, FrozenStateError)
    assert str(error.__cause__) == "frozen state exceeds byte limit"


def test_chooser_limit_does_not_bound_absent_families_or_other_state(tmp_path, monkeypatch):
    _stage_and_promote(tmp_path)
    _happy_catalog(tmp_path)
    monkeypatch.setattr(release_bindings, "_CHOOSER_MAX_ARTIFACT_BYTES", 1)
    binding = resolve_release_binding(tmp_path)
    assert binding.chooser_analog_artifacts == binding.chooser_admissible_artifacts == ()
    assert set(binding.payoff_artifacts) == set(binding.analog_artifacts) == {"STR-THRU"}
    assert set(binding.recalibration_artifacts) == {"STR-THRU"}


@pytest.mark.parametrize("member", [_POOL, _TABLE])
def test_second_reference_cannot_use_cache_to_hide_oversize_bytes(tmp_path, monkeypatch, member):
    _stage_and_promote(tmp_path)
    artifact = _artifact(member)
    obj = _stored(tmp_path, artifact)
    payload = serialize_frozen_state(artifact)
    second_path = _write_object(_dep_root(tmp_path), payload + b" ")
    _catalog(tmp_path, member, objects=[obj, {**obj, "path": second_path}])
    monkeypatch.setattr(release_bindings, "_CHOOSER_MAX_ARTIFACT_BYTES", len(payload))
    error = _assert_refused(tmp_path, member)
    assert isinstance(error.__cause__, FrozenStateError)
    assert str(error.__cause__) == "frozen state exceeds byte limit"


@pytest.mark.parametrize("member", [_POOL, _TABLE])
def test_growth_during_chooser_read_refuses_whole_release(tmp_path, monkeypatch, member):
    _stage_and_promote(tmp_path)
    artifact = _artifact(member)
    obj = _stored(tmp_path, artifact)
    target = _dep_root(tmp_path) / obj["path"]
    _catalog(tmp_path, member, objects=[obj])
    monkeypatch.setattr(release_bindings, "_CHOOSER_MAX_ARTIFACT_BYTES",
                        len(serialize_frozen_state(artifact)))
    real_fstat = frozen_state_io.os.fstat

    def grow_after_fstat(fd):
        info = real_fstat(fd)
        with target.open("ab") as stream:
            stream.write(b"growth")
        return info

    monkeypatch.setattr(frozen_state_io.os, "fstat", grow_after_fstat)
    error = _assert_refused(tmp_path, member)
    assert isinstance(error.__cause__, FrozenStateError)
    assert str(error.__cause__) == "frozen state exceeds byte limit"
