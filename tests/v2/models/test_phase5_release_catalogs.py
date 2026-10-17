"""Catalog survival using the actual release and frozen-state producers."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from checks import phase5_acceptance as acceptance
from engine.v2.foundation import content_hash
from engine.v2.models import deployment
from engine.v2.models import release_layout as layout
from engine.v2.models import release_objects
from engine.v2.models.analog_artifact import make_board_analog_pool_artifact
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import Lineage
from engine.v2.models.payoff_artifact import make_payoff_line_artifact, serialize_payoff_artifact
from engine.v2.models.recalibration_artifact import (
    make_recalibration_map_artifact, serialize_recalibration_artifact,
)
from engine.v2.scoring.release_bindings import ModelNotReady, resolve_release_binding
from tests.v2.models.test_checks_phase5_acceptance import _models
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
    deployment.mark_staging_succeeded(layout.deployment_root(root), rid)
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


def test_unreadable_local_catalog_is_not_absence(tmp_path, monkeypatch):
    _stage(tmp_path)
    deployment.promote(layout.deployment_root(tmp_path), "A")
    real_open = os.open
    # A catalog missing at open is absence (atomic open); an unreadable one is an error.
    def unreadable(path, *args, **kwargs):
        if Path(path) == _local(tmp_path):
            raise PermissionError("private path")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(os, "open", unreadable)
    with pytest.raises(ModelNotReady) as error:
        resolve_release_binding(tmp_path)
    assert "private path" not in str(error.value)
    with pytest.raises(layout.ReleaseLayoutError):
        layout.read_manifest(tmp_path)


def test_directory_or_symlink_local_catalog_is_not_absence(tmp_path):
    _stage(tmp_path)
    deployment.promote(layout.deployment_root(tmp_path), "A")
    local = _local(tmp_path)
    real = local.read_bytes()
    local.unlink()
    local.mkdir()
    with pytest.raises(layout.ReleaseLayoutError):
        layout.read_manifest(tmp_path)
    local.rmdir()
    target = tmp_path / "target.json"
    target.write_bytes(real)
    local.symlink_to(target)
    with pytest.raises(layout.ReleaseLayoutError):
        layout.read_manifest(tmp_path)


def test_catalog_size_bound_is_exact(tmp_path, monkeypatch):
    _stage(tmp_path)
    size = _local(tmp_path).stat().st_size
    monkeypatch.setattr(layout, "MAX_CATALOG_BYTES", size)
    body = layout.read_catalog(_local(tmp_path))
    assert body["release_id"] == "A"
    monkeypatch.setattr(layout, "MAX_CATALOG_BYTES", size - 1)
    with pytest.raises(layout.ReleaseLayoutError, match="could not be read or validated"):
        layout.read_catalog(_local(tmp_path))


def test_catalog_cap_constant_and_regular_file_guard(tmp_path):
    assert layout.MAX_CATALOG_BYTES == 64 * 1024 * 1024
    with pytest.raises(layout.ReleaseLayoutError, match="not a bounded regular file"):
        release_objects.read_regular_file(tmp_path, 16)
    assert release_objects.read_regular_file(tmp_path / "absent", 16) is None
    small = tmp_path / "small.bin"
    small.write_bytes(b"x" * 8)
    assert release_objects.read_regular_file(small, 8) == b"x" * 8
    with pytest.raises(layout.ReleaseLayoutError, match="not a bounded regular file"):
        release_objects.read_regular_file(small, 7)
    for bad in (-1, -2, None, 1.5, True):
        with pytest.raises(layout.ReleaseLayoutError, match="non-negative integer"):
            release_objects.read_regular_file(tmp_path / "absent", bad)
    with pytest.raises(layout.ReleaseLayoutError, match="not a bounded regular file"):
        release_objects.read_regular_file(small, 0)
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert release_objects.read_regular_file(empty, 0) == b""


def test_read_regular_file_requests_at_most_limit_plus_one(tmp_path, monkeypatch):
    path = tmp_path / "data.bin"
    path.write_bytes(b"x" * 20)
    sizes = []
    real_fdopen = release_objects.os.fdopen

    class Spy:
        def __init__(self, handle):
            self._handle = handle

        def read(self, size=-1):
            sizes.append(size)
            return self._handle.read(size)

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    monkeypatch.setattr(release_objects.os, "fdopen", lambda *a, **k: Spy(real_fdopen(*a, **k)))
    assert release_objects.read_regular_file(path, 20) == b"x" * 20
    assert sizes == [21]


def test_read_regular_file_refuses_fifo_without_blocking(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no mkfifo")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(layout.ReleaseLayoutError, match="not a bounded regular file"):
        release_objects.read_regular_file(fifo, 16)


def test_read_regular_file_refusals_do_not_leak_descriptors(tmp_path):
    def open_fds():
        return len(os.listdir("/proc/self/fd"))

    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("no /proc/self/fd")
    directory = tmp_path / "dir"
    directory.mkdir()
    fifo = tmp_path / "fifo"
    if hasattr(os, "mkfifo"):
        os.mkfifo(fifo)
    before = open_fds()
    for _ in range(20):
        with pytest.raises(layout.ReleaseLayoutError):
            release_objects.read_regular_file(directory, 16)
        if fifo.exists():
            with pytest.raises(layout.ReleaseLayoutError):
                release_objects.read_regular_file(fifo, 16)
    assert open_fds() == before


def test_write_object_accepts_empty_payload(tmp_path):
    digest, rel = layout.write_object(tmp_path, b"")
    assert (layout.deployment_root(tmp_path) / rel).read_bytes() == b""
    assert digest == "sha256:" + hashlib.sha256(b"").hexdigest()


def test_write_object_refuses_existing_larger_object(tmp_path):
    digest, rel = layout.write_object(tmp_path, b"abc")
    dest = layout.deployment_root(tmp_path) / rel
    dest.write_bytes(b"abcdef")
    with pytest.raises(layout.ObjectVerificationError):
        layout.write_object(tmp_path, b"abc")
    assert dest.read_bytes() == b"abcdef"


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
    real_write = deployment.atomic_write_bytes

    def forbid_source_write(path, data):
        assert source not in path.parents
        return real_write(path, data)

    monkeypatch.setattr(deployment, "atomic_write_bytes", forbid_source_write)
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


def test_reprepare_failed_acceptance_invalidates_prior_approval(tmp_path):
    _stage(tmp_path)
    dep = layout.deployment_root(tmp_path)
    deployment.promote(dep, "A")
    manifest = deployment._manifest_path(dep, "A")
    staged_model_manifest = manifest.read_bytes()
    pointer = (dep / "DEPLOYED").read_bytes()
    history = {p.name: p.read_bytes() for p in (dep / "history").glob("*.json")}
    release, inventory, models = _models("A", keys=(("gate", "STR-THRU"),))
    states = prep.build_states({})
    prep.write_release(tmp_path, release, inventory, models, states)
    assert manifest.read_bytes() == staged_model_manifest
    findings = acceptance._Findings()
    specs = tuple(s for s in layout.STATE_SPECS if s.member_id == "payoff_line:STR-THRU")
    rows, _loaded = acceptance._state_member_rows(
        tmp_path, layout.read_manifest(tmp_path), findings, specs)
    row = next(item for item in rows if item["member_id"] == "payoff_line:STR-THRU")
    assert row["status"] == layout.MISSING
    assert row["verdict"] == acceptance.MEMBER_MISSING
    assert acceptance.MEMBER_MISSING in {item["code"] for item in findings.rows}
    with pytest.raises(deployment.StagingNotSuccessful) as error:
        deployment.promote(dep, "A")
    assert error.value.retryable is False
    assert (dep / "DEPLOYED").read_bytes() == pointer
    assert {p.name: p.read_bytes() for p in (dep / "history").glob("*.json")} == history
    assert not deployment._staging_status_path(dep, "A").exists()
