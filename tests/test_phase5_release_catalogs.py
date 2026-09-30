"""Catalog survival using the actual release and frozen-state producers."""
import json
import os
from pathlib import Path

import pytest

from checks import phase5_release as layout
from engine.v2.foundation import content_hash
from engine.v2.models import deployment
from engine.v2.models.analog_artifact import make_board_analog_pool_artifact
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import Lineage
from engine.v2.models.payoff_artifact import make_payoff_line_artifact, serialize_payoff_artifact
from engine.v2.models.recalibration_artifact import (
    make_recalibration_map_artifact, serialize_recalibration_artifact,
)
from engine.v2.scoring.release_bindings import ModelNotReady, resolve_release_binding
from tests.test_checks_phase5_acceptance import _models
from tools import phase5_prepare_release as prep
from tools.phase5_calibration_keys import staged_keys


def _local(root, rid="A"):
    return root / "deployment" / "releases" / rid / layout.MANIFEST_NAME


def _root(root):
    return root / layout.MANIFEST_NAME


def _stage(root, rid="A", alpha=0.55, incumbent=None):
    # Same producer shapes as test_v2_scoring_release_bindings; catalogs are
    # emitted by write_release, never invented as a legacy-format fixture.
    payoff = make_payoff_line_artifact(
        {"n": 2, "intercept": 0.1, "slope": 0.2, "resid_sd": 0.01, "r": 0.5,
         "residuals": [0.01, -0.01]},
        strategy="STR-THRU", driver="driver_prediction", alpha=alpha)
    recal = make_recalibration_map_artifact(
        {"n": 3, "base_rate": 0.4, "x_thresholds": [0.3, 0.7], "y_thresholds": [0.35, 0.65]},
        strategy="STR-THRU", alpha=alpha, min_pairs=2)
    analog = make_board_analog_pool_artifact(
        strategy="STR-THRU", alpha=alpha, cutoff=None, population_edges=[0.0, 1.0],
        causal_edges=None, lineage=Lineage(),
        rows=[["row-1", "2026-01-01", "2026-01-05", 0.02, 0.03, None, None, None]])
    payloads = {
        "payoff_line:STR-THRU": {"payoff": serialize_payoff_artifact(payoff)},
        "recalibration_map:STR-THRU": {"recalibration": serialize_recalibration_artifact(recal)},
        "board_analog_matcher": {"analog": serialize_frozen_state(analog)},
    }
    release, inventory, models = _models(rid, keys=(("gate", "STR-THRU"),))
    states = [s for s in prep.build_states(payloads) if s.spec.member_id in payloads]
    path = prep.write_release(root, release, inventory, models, states, incumbent=incumbent)
    return path, (payoff.content_hash, recal.content_hash, analog.content_hash)


def _hashes(root):
    binding = resolve_release_binding(root)
    return tuple(group["STR-THRU"][0].content_hash for group in (
        binding.payoff_artifacts, binding.recalibration_artifacts, binding.analog_artifacts))


def _rewrite(path, **changes):
    body = json.loads(path.read_bytes())
    body.update(changes)
    body.pop("manifest_hash")
    body["manifest_hash"] = content_hash(body)
    path.write_text(json.dumps(body))
    return body


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("copy_incumbent", [False, True])
def test_real_rollback_restores_all_three_state_families(tmp_path, legacy, copy_incumbent):
    source = tmp_path / "source"
    _, hashes_a = _stage(source)
    deployment.promote(layout.deployment_root(source), "A")
    assert _hashes(source) == hashes_a
    if legacy:
        _local(source).unlink()
    root = tmp_path / "destination" if copy_incumbent else source
    incumbent = None
    if copy_incumbent:
        incumbent = source / "custom-store-name"
        layout.deployment_root(source).rename(incumbent)
    _, hashes_b = _stage(root, "B", 0.65, incumbent)
    assert layout.read_manifest(root)["release_id"] == "B"
    assert staged_keys(root)["payoff_line:STR-THRU"] == {("STR-THRU", 0.65, None)}
    assert _hashes(root) == hashes_a  # Staging B must not switch scoring off A.
    dep = layout.deployment_root(root)
    deployment.promote(dep, "B")
    assert _hashes(root) == hashes_b
    assert all(a != b for a, b in zip(hashes_a, hashes_b))
    deployment.rollback(dep)
    assert resolve_release_binding(root).release_id == "A"
    assert _hashes(root) == hashes_a
    assert _local(root, "A").is_file() and _local(root, "B").is_file()
    assert layout.read_manifest(root)["release_id"] == "B"


def test_producer_hash_oracle_rejects_planted_b_result_after_rollback(tmp_path, monkeypatch):
    _, expected_a = _stage(tmp_path)
    dep = layout.deployment_root(tmp_path)
    deployment.promote(dep, "A")
    _stage(tmp_path, "B", 0.65)
    deployment.promote(dep, "B")
    wrong = resolve_release_binding(tmp_path)
    deployment.rollback(dep)
    assert _hashes(tmp_path) == expected_a
    monkeypatch.setattr(__name__ + ".resolve_release_binding", lambda root: wrong)
    with pytest.raises(AssertionError):
        assert _hashes(tmp_path) == expected_a


@pytest.mark.parametrize("damage", ["json", "schema", "hash", "id", "directory", "dangling", "fifo"])
def test_bad_local_never_falls_back_to_valid_root(tmp_path, damage):
    _stage(tmp_path)
    deployment.promote(layout.deployment_root(tmp_path), "A")
    local = _local(tmp_path)
    if damage == "json":
        local.write_bytes(b"{")
    elif damage == "schema":
        _rewrite(local, schema_version="unsupported")
    elif damage == "hash":
        body = json.loads(local.read_bytes())
        body["manifest_hash"] = "incorrect"
        local.write_text(json.dumps(body))
    elif damage == "id":
        _rewrite(local, release_id="B")
    else:
        local.unlink()
        if damage == "directory":
            local.mkdir()
        elif damage == "fifo":
            os.mkfifo(local)
        else:
            local.symlink_to(tmp_path / "missing")
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert str(tmp_path) not in str(error.value)
    with pytest.raises(layout.ReleaseLayoutError):
        layout.read_manifest(tmp_path)
    with pytest.raises(layout.ReleaseLayoutError):
        layout.preserve_legacy_manifest(tmp_path)
    destination = tmp_path / "copy" / "deployment"
    with pytest.raises(layout.ReleaseLayoutError):
        prep._copy_incumbent(layout.deployment_root(tmp_path), destination)
    assert not destination.exists()


def test_unreadable_local_and_read_race_are_not_absence(tmp_path, monkeypatch):
    _stage(tmp_path)
    deployment.promote(layout.deployment_root(tmp_path), "A")
    real_read = Path.read_bytes
    for exception in (PermissionError, FileNotFoundError):
        def unreadable(path):
            if path == _local(tmp_path):
                raise exception("private path")
            return real_read(path)
        monkeypatch.setattr(Path, "read_bytes", unreadable)
        with pytest.raises(ModelNotReady) as error:
            resolve_release_binding(tmp_path)
        assert "private path" not in str(error.value)
        with pytest.raises(layout.ReleaseLayoutError):
            layout.read_manifest(tmp_path)


def test_legacy_fallback_requires_matching_id_and_local_needs_no_root(tmp_path):
    _stage(tmp_path)
    deployment.promote(layout.deployment_root(tmp_path), "A")
    expected = _hashes(tmp_path)
    original = _root(tmp_path).read_bytes()
    _root(tmp_path).unlink()
    assert _hashes(tmp_path) == expected
    _root(tmp_path).write_bytes(original)
    _local(tmp_path).unlink()
    assert _hashes(tmp_path) == expected
    assert layout.read_manifest(tmp_path)["release_id"] == "A"
    assert not _local(tmp_path).exists()  # Readers do not migrate.
    _rewrite(_root(tmp_path), release_id="B")
    with pytest.raises(ModelNotReady):
        resolve_release_binding(tmp_path)


@pytest.mark.parametrize("damage", ["json", "unsafe", "nonstring", "unstaged", "modelhash", "modelid"])
def test_invalid_legacy_migration_refuses(tmp_path, damage):
    _stage(tmp_path)
    _local(tmp_path).unlink()
    if damage == "json":
        _root(tmp_path).write_bytes(b"{")
    elif damage in ("unsafe", "nonstring", "unstaged"):
        _rewrite(_root(tmp_path), release_id={
            "unsafe": "../escape", "nonstring": 42, "unstaged": "missing"}[damage])
    else:
        model = deployment._manifest_path(layout.deployment_root(tmp_path), "A")
        body = json.loads(model.read_bytes())
        if damage == "modelhash":
            body["release_hash"] = "incorrect"
        else:
            body["release"]["release_id"] = "B"
        model.write_text(json.dumps(body))
    before = _root(tmp_path).read_bytes()
    with pytest.raises(layout.ReleaseLayoutError):
        layout.preserve_legacy_manifest(tmp_path)
    assert not _local(tmp_path).exists()
    assert _root(tmp_path).read_bytes() == before
    with pytest.raises(layout.ReleaseLayoutError):
        prep._copy_incumbent(layout.deployment_root(tmp_path), tmp_path / "copy" / "deployment")
    assert not (tmp_path / "copy").exists()


def test_copy_preflight_is_read_only_and_failed_destination_is_retryable(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _stage(source)
    legacy = _root(source).read_bytes()
    _local(source).unlink()
    store = source / "custom-store-name"
    layout.deployment_root(source).rename(store)
    destination = tmp_path / "copy" / "deployment"
    _root(source).write_bytes(b"{")
    with pytest.raises(layout.ReleaseLayoutError):
        prep._copy_incumbent(store, destination)
    assert not destination.exists()
    _root(source).write_bytes(legacy)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    real_write = deployment._atomic_write_bytes

    def forbid_source_write(path, data):
        assert source not in path.parents
        return real_write(path, data)

    monkeypatch.setattr(deployment, "_atomic_write_bytes", forbid_source_write)
    prep._copy_incumbent(store, destination)
    assert _local(destination.parent).read_bytes() == legacy
    assert {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("copy_incumbent", [False, True])
def test_existing_local_authority_survives_stale_root(tmp_path, copy_incumbent):
    _stage(tmp_path)
    old = _root(tmp_path).read_bytes()
    _stage(tmp_path, alpha=0.75)  # Same-ID staging still permits an explicit rewrite.
    newest = _local(tmp_path).read_bytes()
    assert newest != old
    _root(tmp_path).write_bytes(old)
    assert layout.read_manifest(tmp_path) == json.loads(newest)
    destination = tmp_path / "copy" if copy_incumbent else tmp_path
    _stage(destination, "B", 0.65,
           layout.deployment_root(tmp_path) if copy_incumbent else None)
    assert _local(destination).read_bytes() == newest


def test_absent_legacy_does_not_invent_catalog_on_copy(tmp_path):
    _stage(tmp_path)
    _local(tmp_path).unlink()
    _root(tmp_path).unlink()
    destination = tmp_path / "copy" / "deployment"
    prep._copy_incumbent(layout.deployment_root(tmp_path), destination)
    assert not _local(destination.parent).exists()


def test_candidate_root_is_validated_even_with_good_local(tmp_path):
    _stage(tmp_path)
    _root(tmp_path).write_bytes(b"{")
    with pytest.raises(layout.ReleaseLayoutError):
        layout.read_manifest(tmp_path)
    with pytest.raises(layout.ReleaseLayoutError):
        _stage(tmp_path, "B")


def test_interrupted_root_publication_keeps_complete_files_and_retry_works(tmp_path, monkeypatch):
    _stage(tmp_path)
    dep = layout.deployment_root(tmp_path)
    deployment.promote(dep, "A")
    old = _root(tmp_path).read_bytes()
    _local(tmp_path).unlink()  # The old legacy catalog must be preserved first.
    pointer = (dep / "DEPLOYED").read_bytes()
    history = {p.name: p.read_bytes() for p in (dep / "history").iterdir()}
    real_replace = deployment.os.replace
    def interrupted(source, dest):
        if dest == _root(tmp_path):
            assert _local(tmp_path).read_bytes() == old
            assert json.loads(_local(tmp_path, "B").read_bytes())["release_id"] == "B"
            raise OSError("simulated publication interruption")
        return real_replace(source, dest)
    with monkeypatch.context() as patch:
        patch.setattr(deployment.os, "replace", interrupted)
        with pytest.raises(OSError):
            _stage(tmp_path, "B", 0.65)
    assert _root(tmp_path).read_bytes() == old
    assert (dep / "DEPLOYED").read_bytes() == pointer
    assert {p.name: p.read_bytes() for p in (dep / "history").iterdir()} == history
    assert resolve_release_binding(tmp_path).release_id == "A"
    assert _stage(tmp_path, "B", 0.65)[0] == _local(tmp_path, "B")
    assert layout.read_manifest(tmp_path)["release_id"] == "B"
