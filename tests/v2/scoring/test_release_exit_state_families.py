"""Synthetic paired-residual-pool and trailing-cutoff release state; no providers or stored market data."""
import json
from dataclasses import FrozenInstanceError

import pytest

from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.residual_artifact import make_paired_residual_pool_artifact
from engine.v2.models.trailing_cutoff_artifact import make_trailing_cutoff_artifact
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

_PAIRED = "paired_residual_pool"
_TRAILING = "trailing_pnl_cutoff"
_FIELDS = {_PAIRED: "paired_residual_artifacts", _TRAILING: "trailing_cutoff_artifacts"}
_LABELS = {_PAIRED: "paired residual pool", _TRAILING: "trailing cutoff"}


def _paired(*, move_model_id="m-move", crush_model_id="m-crush", cutoff="2026-02-01", value=1.0):
    return make_paired_residual_pool_artifact(
        move_model_id=move_model_id, crush_model_id=crush_model_id, cutoff=cutoff,
        rows=[["2026-01-01", "AAA", 1.0, value, 0.2]], lineage=Lineage())


def _trailing(*, history_id="features.pnl_sim_history", month="2026-03-01", cutoff=0.5):
    return make_trailing_cutoff_artifact(
        month=month, cutoff=cutoff, history_id=history_id,
        lineage=Lineage(data=(DataDependency(
            table="synthetic.pnl-history", end_exclusive=month),)))


def _artifact(member, **kwargs):
    return _paired(**kwargs) if member == _PAIRED else _trailing(**kwargs)


def _wrong_type(member):
    return _trailing() if member == _PAIRED else _paired()


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


@pytest.mark.parametrize("member", [None, _PAIRED, _TRAILING])
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


@pytest.mark.parametrize("member,artifacts,expected_keys", [
    (_PAIRED, (_paired(cutoff="2026-03-01", value=17.0), _paired()),
     [("m-move", "m-crush", "2026-03-01"), ("m-move", "m-crush", "2026-02-01")]),
    (_TRAILING, (_trailing(month="2026-04-01", cutoff=9.0), _trailing()),
     [("features.pnl_sim_history", "2026-04-01"), ("features.pnl_sim_history", "2026-03-01")]),
], ids=["paired-residual-pool", "trailing-cutoff"])
def test_distinct_keys_load_in_catalog_order_verified_and_frozen(
        tmp_path, member, artifacts, expected_keys):
    _stage_and_promote(tmp_path)
    _write_catalog(tmp_path, rows=[
        _row(member, [_stored(tmp_path, artifact) for artifact in artifacts])])

    binding = resolve_release_binding(tmp_path)

    loaded = getattr(binding, _FIELDS[member])
    assert isinstance(loaded, tuple)
    assert [a.key for a in loaded] == expected_keys
    assert loaded == artifacts
    assert [a.content_hash for a in loaded] == [
        _sha(serialize_frozen_state(a)) for a in artifacts]
    other = _FIELDS[_TRAILING if member == _PAIRED else _PAIRED]
    assert getattr(binding, other) == ()
    with pytest.raises(FrozenInstanceError):
        setattr(binding, _FIELDS[member], ())
    assert resolve_release_binding(tmp_path) == binding


@pytest.mark.parametrize("member", [_PAIRED, _TRAILING])
@pytest.mark.parametrize("same_bytes", [False, True])
def test_duplicate_semantic_keys_refuse_without_overwrite(tmp_path, member, same_bytes):
    _stage_and_promote(tmp_path)
    first = _artifact(member)
    second = first if same_bytes else (
        _paired(value=99.0) if member == _PAIRED else _trailing(cutoff=9.0))
    _catalog(tmp_path, member, objects=[
        _stored(tmp_path, first), _stored(tmp_path, second)])
    error = _assert_refused(tmp_path, member)
    assert "ambiguous" in error.detail
    assert _LABELS[member] in error.detail


@pytest.mark.parametrize("member", [_PAIRED, _TRAILING])
def test_other_family_artifact_under_member_refuses(tmp_path, member):
    _stage_and_promote(tmp_path)
    _catalog(tmp_path, member, objects=[_stored(tmp_path, _wrong_type(member))])
    error = _assert_refused(tmp_path, member)
    assert "wrong artifact type" in error.detail


@pytest.mark.parametrize("member", [_PAIRED, _TRAILING])
def test_tampered_object_refuses_naming_member_without_path_leak(tmp_path, member):
    _stage_and_promote(tmp_path)
    artifact = _artifact(member)
    obj = _obj(_write_object(_dep_root(tmp_path), serialize_frozen_state(artifact)),
               _sha(b"another document"))
    _catalog(tmp_path, member, objects=[obj])
    _assert_refused(tmp_path, member)


@pytest.mark.parametrize("member", [_PAIRED, _TRAILING])
@pytest.mark.parametrize("status", ["MISSING", "PENDING", None])
def test_declared_nonstaged_state_refuses_whole_release(tmp_path, member, status):
    _stage_and_promote(tmp_path)
    _catalog(tmp_path, member, status=status)
    _assert_refused(tmp_path, member)
