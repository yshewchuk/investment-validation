"""Bootstrap assembly tests (#1102 slice B); carry tests live in test_state_stage.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.v2.models import release_layout as phase5_release
from engine.v2.models import inventory as inventory_module
from engine.v2.ops.errors import OpsError
from engine.v2.ops.state_stage import (
    run_state_stage_worker,
    state_stage_parameter_problems,
    state_stage_plan,
)
from tools import phase5_prepare_release

from tests.v2.ops.test_state_stage import (
    REPO_ROOT,
    _fixture,
    _params,
    _spec,
    _stage_prior,
    _STATES,
)

_SNAPSHOT = "a" * 64


def _bootstrap_states(cutoff="2026-10-01"):
    """Every bound state as a file name -> (member, object name)."""
    states = {}
    for spec in phase5_release.STATE_SPECS:
        member = spec.member_id
        if member == "admissible_table:dyn_sv":
            continue
        if member in ("paired_residual_pool", "board_analog_matcher",
                      "chooser_analog_pool"):
            name = f"x|y|{cutoff}"
        elif member == "trailing_pnl_cutoff":
            name = f"hist|{cutoff[:7]}"
        else:
            name = f"{member}|1"
        states[f"{member.replace(':', '_')}.json"] = (member, name)
    return states


def _mock_bootstrap_inputs(monkeypatch, tmp_path, states=None, prior_names=None,
                           mode="bootstrap", prior_release_id="prior1", drift=(),
                           inventory=None):
    """Patch inventory, panel and the payload readers for a bootstrap run."""
    if prior_release_id is not None and inventory is None:
        inventory, _body = _stage_prior(tmp_path, prior_names=prior_names)
    states = _bootstrap_states() if states is None else states
    staging = tmp_path / "staging"
    staging.mkdir(exist_ok=True)
    for name in states:
        (staging / name).write_bytes(f"state-{name}".encode())

    def fake_classify(root, names):
        return ([ (Path(root) / name, f"state-{name}".encode()) for name in names ], [])

    def fake_frozen(paths, max_bytes=None):
        from tools import phase5_state_stage_job

        assert max_bytes == phase5_state_stage_job.MAX_STATE_BYTES
        name = Path(paths[0]).name
        member, object_name = states[name]
        return {member: {object_name: Path(paths[0]).read_bytes()}}

    def fake_model_release(inv, resolve=None):
        release, _inventory, payloads = _fixture(
            inv.release_id, deployment_id=inv.deployment_id)
        return release, payloads

    monkeypatch.setattr("tools.phase5_state_bootstrap._classify_bound_files", fake_classify)
    monkeypatch.setattr(phase5_prepare_release, "frozen_state_payloads", fake_frozen)
    monkeypatch.setattr(phase5_prepare_release, "artifact_file_payloads", lambda files: {})
    monkeypatch.setattr(phase5_prepare_release, "default_admissible_table",
                        lambda: {"admissible_table:dyn_sv": {"t|1": b"table"}})
    monkeypatch.setattr(phase5_prepare_release, "tier4_fold_payloads", lambda *a, **k: {})
    monkeypatch.setattr(phase5_prepare_release, "fold_policies",
                        lambda found, snapshot: (None, {}))
    class _FakePreflight:
        digest = "dummy"

    monkeypatch.setattr(phase5_prepare_release, "_run_preflights",
                        lambda release, states, pinned_panel, sources: (
                            sources,
                            {name: _FakePreflight() for build in states
                             for name in build.payloads},
                            {}))
    monkeypatch.setattr(phase5_prepare_release, "_destination_digest",
                        lambda path: "dummy")
    monkeypatch.setattr(phase5_prepare_release, "model_release", fake_model_release)
    if inventory is not None:
        monkeypatch.setattr(inventory_module, "current_release_inventory",
                            lambda *a, **k: (inventory, drift))
    monkeypatch.setattr("engine.v2.models.pinned_panel.verify_panel_copy",
                        lambda ctx: {"panel_copy_sha256": _SNAPSHOT})
    monkeypatch.setattr(phase5_prepare_release, "verify_panel_copy",
                        lambda ctx: {"panel_copy_sha256": _SNAPSHOT})
    parameters = {"expected_ids": ["phase5_state_stage"], "release_root": str(tmp_path),
                  "prior_release_id": prior_release_id, "release_id": "cand1",
                  "cutoff": "2026-10-01",
                  "state_files": sorted(states),
                  "input_bindings": {**{name: f"job_x#training/{name}" for name in states},
                                     "legacy_manifest.json": "artifact_manifest"},
                  "mode": mode,
                  "bootstrap": {"tier4_dir": "/tmp/tier4", "panel_catalog": "/tmp/panel",
                                "panel_snapshot_id": "snap",
                                "panel_artifact_root": "/tmp/artifact"}}
    return parameters, staging


def test_bootstrap_requires_the_mapping_and_exact_keys():
    assert state_stage_parameter_problems(
        None, _params(mode="bootstrap", bootstrap=None)) == [
            "bootstrap must be a mapping with tier4_dir, panel_catalog, "
            "panel_snapshot_id and panel_artifact_root"]
    assert state_stage_parameter_problems(
        None, _params(mode="bootstrap", bootstrap={"tier4_dir": "/t"})) == [
            "bootstrap keys must be exactly tier4_dir, panel_catalog, "
            "panel_snapshot_id and panel_artifact_root"]
    assert state_stage_parameter_problems(
        None, _params(mode="bootstrap", bootstrap={"tier4_dir": "/t", "panel_catalog": "/c",
                                                   "panel_snapshot_id": "s",
                                                   "panel_artifact_root": ""})) == [
            "bootstrap values must be non-empty strings"]


def test_bootstrap_mapping_is_rejected_outside_bootstrap():
    assert state_stage_parameter_problems(
        None, _params(bootstrap={"tier4_dir": "/t", "panel_catalog": "/c",
                                 "panel_snapshot_id": "s", "panel_artifact_root": "/a"})) == [
            "bootstrap must be present only when mode is bootstrap"]


def test_carry_plan_omits_bootstrap_and_keeps_the_spec_hash(tmp_path):
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01",
                            state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                            source_root=REPO_ROOT)
    assert "bootstrap" not in plan["parameters"]
    spec_hash = plan["spec_hash"]
    for key in ("expected_ids", "release_root", "prior_release_id", "release_id",
                "cutoff", "state_files", "input_bindings", "mode"):
        assert key in plan["parameters"]
    plan2 = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                             release_id="cand1", cutoff="2026-10-01",
                             state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                             source_root=REPO_ROOT)
    assert plan2["spec_hash"] == spec_hash


def test_bootstrap_plan_passes_the_mapping(tmp_path):
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01",
                            state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                            source_root=REPO_ROOT, mode="bootstrap",
                            bootstrap={"tier4_dir": "/t", "panel_catalog": "/c",
                                       "panel_snapshot_id": "s",
                                       "panel_artifact_root": "/a"})
    assert plan["parameters"]["bootstrap"] == {
        "tier4_dir": "/t", "panel_catalog": "/c",
        "panel_snapshot_id": "s", "panel_artifact_root": "/a"}


def test_bootstrap_plan_refuses_a_bad_mapping(tmp_path):
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                         release_id="cand1", cutoff="2026-10-01",
                         state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                         source_root=REPO_ROOT, mode="bootstrap", bootstrap={"x": 1})
    assert err.value.code == "INVALID_REQUEST"


def test_none_prior_stages_a_candidate_catalog_only(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory)
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    catalog = phase5_release.catalog_path(tmp_path, "cand1")
    assert catalog.exists()
    root_manifest = tmp_path / phase5_release.MANIFEST_NAME
    if root_manifest.exists():
        assert json.loads(root_manifest.read_bytes())["release_id"] == "prior1"
    body = json.loads(catalog.read_bytes())
    assert body["sources"]["pool_cutoff"] == "2026-10-01"
    assert body["sources"]["carried_from"] is None
    summary = json.loads((staging / "stage_result.json").read_text())
    assert summary["n_carried"] == 0
    assert summary["release_id"] == "cand1"


def test_no_pool_catalog_requires_matching_bindings(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    assert phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_bindings_differ_refuses_without_writing(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    before = sorted(path.name for path in (phase5_release.deployment_root(tmp_path)
                                           / "objects").iterdir())

    def broken(inv, resolve=None):
        release, _inventory, payloads = _fixture(
            inv.release_id, deployment_id=inv.deployment_id, binding_id="b2")
        return release, payloads

    monkeypatch.setattr(phase5_prepare_release, "model_release", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == ("VALIDATION_FAILED: candidate bindings differ "
                              "from the incumbent's")
    assert err.value.problem.details == {"code": "BINDINGS_DIFFER"}
    after = sorted(path.name for path in (phase5_release.deployment_root(tmp_path)
                                          / "objects").iterdir())
    assert after == before


def test_empty_member_refuses_before_writing(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, states=_STATES, prior_release_id=None, inventory=inventory)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release has an empty member"
    assert err.value.problem.details["code"] == "EMPTY_MEMBER"
    assert "members" in err.value.problem.details


def test_cutoff_mismatch_of_a_paired_object_refuses(tmp_path, monkeypatch):
    states = _bootstrap_states()
    states["paired_residual_pool.json"] = ("paired_residual_pool", "mv|cr|2026-10-02")
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path, states=states)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == ("VALIDATION_FAILED: a pool object cutoff differs "
                              "from the session cutoff")
    assert err.value.problem.details == {"members": ["paired_residual_pool"]}


def test_damaged_prior_still_refuses(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").write_bytes(b"{not json")
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: prior release could not be read"
    assert err.value.problem.details == {"code": "PRIOR_MISMATCH"}


def test_usable_prior_still_refuses(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: prior release already has a pool catalog"
    assert err.value.problem.details == {"code": "PRIOR_USABLE"}


def test_id_conflict_on_a_different_pre_existing_catalog(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    from engine.v2.models import deployment

    release, inventory, payloads = _fixture("cand1")
    deployment.stage_release(phase5_release.deployment_root(tmp_path), release,
                             inventory, payloads)
    body = phase5_release.manifest_body(
        "cand1", "d1", [phase5_release.member_row(
            _spec("paired_residual_pool"), phase5_release.STAGED, [], "other")],
        {"pool_cutoff": "2020-01-01", "carried_from": None})
    phase5_release.write_manifest(tmp_path, body, candidate_only=True)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == {"code": "ID_CONFLICT"}


def test_rerun_is_idempotent(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    assert run_state_stage_worker(parameters, staging)["completed_ids"] == [
        "phase5_state_stage"]
    first = phase5_release.catalog_path(tmp_path, "cand1").read_bytes()
    run_state_stage_worker(parameters, staging)
    assert phase5_release.catalog_path(tmp_path, "cand1").read_bytes() == first


def test_pinned_panel_failure_echoes_no_exception_text(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    secret = "secret-internal-detail"

    def broken(ctx):
        raise ValueError(secret)

    monkeypatch.setattr("engine.v2.models.pinned_panel.verify_panel_copy", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert secret not in str(err.value)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"


def test_write_manifest_candidate_only_requires_identical_content(tmp_path):
    from engine.v2.models import deployment

    release, inventory, payloads = _fixture("cand1")
    deployment.stage_release(phase5_release.deployment_root(tmp_path), release,
                             inventory, payloads)
    body = phase5_release.manifest_body(
        "cand1", "d1", [phase5_release.member_row(
            _spec("paired_residual_pool"), phase5_release.STAGED, [], "x")],
        {"pool_cutoff": "2026-10-01", "carried_from": None})
    phase5_release.write_manifest(tmp_path, body, candidate_only=True)
    assert not (tmp_path / phase5_release.MANIFEST_NAME).exists()
    assert phase5_release.catalog_path(tmp_path, "cand1").exists()
    phase5_release.write_manifest(tmp_path, body, candidate_only=True)
    other = phase5_release.manifest_body(
        "cand1", "d1", [phase5_release.member_row(
            _spec("paired_residual_pool"), phase5_release.STAGED, [], "y")],
        {"pool_cutoff": "2026-10-01", "carried_from": None})
    with pytest.raises(phase5_release.CatalogConflict):
        phase5_release.write_manifest(tmp_path, other, candidate_only=True)


def test_artifact_file_payloads_refuses_an_unknown_schema(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps({"schema_version": "mystery.v1"}))
    from tools.phase5_state_stage_job import StageRefused

    with pytest.raises(StageRefused) as err:
        phase5_prepare_release.artifact_file_payloads([path])
    assert str(err.value) == "a bound state file could not be loaded"

@pytest.mark.parametrize("vary,value", [
    ("model_id", "m2"),
    ("feature_order", ("y",)),
    ("content_hash", "f" * 64),
])
def test_bindings_differ_by_field_refuses(tmp_path, monkeypatch, vary, value):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    import dataclasses

    def broken(inv, resolve=None):
        release, _inventory, payloads = _fixture(
            inv.release_id, deployment_id=inv.deployment_id)
        binding = release.bindings[0]
        if vary == "model_id":
            binding = dataclasses.replace(binding, model_id=value)
        elif vary == "feature_order":
            binding = dataclasses.replace(binding, feature_order=value)
        elif vary == "content_hash":
            member = dataclasses.replace(binding.members[0], content_hash=value)
            binding = dataclasses.replace(binding, members=(member,))
        release = dataclasses.replace(release, bindings=(binding,))
        return release, payloads

    monkeypatch.setattr(phase5_prepare_release, "model_release", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == ("VALIDATION_FAILED: candidate bindings differ "
                              "from the incumbent's")
    assert err.value.problem.details == {"code": "BINDINGS_DIFFER"}


def test_bootstrap_plan_spec_hash_changes_with_bootstrap_value(tmp_path):
    def plan(**overrides):
        bootstrap = {"tier4_dir": "/t", "panel_catalog": "/c",
                     "panel_snapshot_id": "s", "panel_artifact_root": "/a"}
        bootstrap.update(overrides)
        return state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                                release_id="cand1", cutoff="2026-10-01",
                                state_refs={"a.json": "j#o"},
                                manifest_ref="artifact_manifest",
                                source_root=REPO_ROOT, mode="bootstrap",
                                bootstrap=bootstrap)
    base = plan()["spec_hash"]
    for key in ("tier4_dir", "panel_catalog", "panel_snapshot_id",
                "panel_artifact_root"):
        assert plan(**{key: "changed"})["spec_hash"] != base


def test_unresolvable_incumbent_refuses_with_prior_mismatch(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    from engine.v2.models import deployment

    def broken_resolve(*args, **kwargs):
        raise deployment.DeploymentError("unresolvable")

    monkeypatch.setattr(deployment, "resolve_release", broken_resolve)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: prior release could not be read"
    assert err.value.problem.details == {"code": "PRIOR_MISMATCH"}


def test_symlink_bound_file_refused_with_fixed_message(tmp_path):
    from tools.phase5_state_bootstrap import _classify_bound_files
    from tools.phase5_state_stage_job import StageRefused

    real = tmp_path / "real.json"
    real.write_bytes(b'{"schema_version": "paired_residual_pool.v1.0"}')
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(StageRefused) as err:
        _classify_bound_files(tmp_path, ["link.json"])
    assert str(err.value) == "a bound state file could not be loaded"


def test_fifo_bound_file_refused_with_fixed_message(tmp_path):
    import os

    from tools.phase5_state_bootstrap import _classify_bound_files
    from tools.phase5_state_stage_job import StageRefused

    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(StageRefused) as err:
        _classify_bound_files(tmp_path, ["fifo.json"])
    assert str(err.value) == "a bound state file could not be loaded"


def test_oversize_payoff_artifact_refused(tmp_path, monkeypatch):
    from tools.phase5_state_bootstrap import _classify_bound_files
    from tools.phase5_state_stage_job import StageRefused
    import tools.phase5_state_bootstrap as bootstrap_module

    monkeypatch.setattr(bootstrap_module, "MAX_STATE_BYTES", 10)
    path = tmp_path / "payoff.json"
    path.write_bytes(b'{"schema_version": "payoff_line.v1", "extra": "x"}')
    with pytest.raises(StageRefused) as err:
        _classify_bound_files(tmp_path, ["payoff.json"])
    assert str(err.value) == "a bound state file could not be loaded"


def test_admissible_depth_table_classified_frozen(tmp_path):
    from tools.phase5_state_bootstrap import _classify_bound_files

    path = tmp_path / "admissible.json"
    path.write_text(json.dumps({"schema_version": "admissible_depth_table.v1.0"}))
    frozen, artifacts = _classify_bound_files(tmp_path, ["admissible.json"])
    assert len(frozen) == 1
    assert frozen[0][0] == path
    assert artifacts == []


def test_id_conflict_pre_check_fires_before_write(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    candidate = phase5_release.catalog_path(tmp_path, "cand1")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(b"{not json")

    calls = []
    original_write = phase5_prepare_release.write_release
    monkeypatch.setattr(phase5_prepare_release, "write_release",
                        lambda *a, **k: calls.append((a, k)) or original_write(*a, **k))
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == {"code": "ID_CONFLICT"}
    assert calls == []


def test_matching_candidate_cutoff_does_not_conflict_in_pre_check(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    first = run_state_stage_worker(parameters, staging)
    assert first["completed_ids"] == ["phase5_state_stage"]

    calls = []
    original_write = phase5_prepare_release.write_release
    monkeypatch.setattr(phase5_prepare_release, "write_release",
                        lambda *a, **k: calls.append((a, k)) or original_write(*a, **k))
    second = run_state_stage_worker(parameters, staging)
    assert second["completed_ids"] == ["phase5_state_stage"]
    assert calls
