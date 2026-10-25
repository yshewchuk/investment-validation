from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.v2.models import release_layout as phase5_release
from checks import phase5_release_assembly as assembly
from engine.v2.foundation import ArtifactError
from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
    deployment,
)
from engine.v2.models import inventory as inventory_module
from engine.v2.models.deployment import DeploymentError
from engine.v2.ops import stages
from engine.v2.ops.errors import OpsError
from engine.v2.ops.state_stage import (
    StateStageParameters,
    run_state_stage_worker,
    state_stage_job_kind,
    state_stage_parameter_problems,
    state_stage_plan,
)
from tools import phase5_prepare_release

from tests.v2.ops.state_stage_support import (
    BOOTSTRAP as _BOOTSTRAP,
    REPO_ROOT,
    STATES as _STATES,
    UNRELATED_MEMBER_ID,
    fixture as _fixture,
    params as _params,
    spec as _spec,
    stage_prior as _stage_prior,
)


def test_valid_parameters_have_no_problems():
    assert state_stage_parameter_problems(None, _params()) == []


def test_bindings_without_the_manifest_key_are_reported():
    bindings = {name: f"job_x#training/{name}" for name in _STATES}
    assert state_stage_parameter_problems(
        None, _params(input_bindings=bindings)) == [
            "input_bindings keys must equal state_files plus legacy_manifest.json"]


def test_bindings_with_an_extra_manifest_key_are_reported():
    bindings = {**{name: f"job_x#training/{name}" for name in _STATES},
                "legacy_manifest.json": "artifact_manifest", "extra.json": "j#o"}
    assert state_stage_parameter_problems(
        None, _params(input_bindings=bindings)) == [
            "input_bindings keys must equal state_files plus legacy_manifest.json"]


@pytest.mark.parametrize("manifest_ref", [None, "", 5])
def test_manifest_binding_must_be_a_non_empty_string(manifest_ref):
    bindings = {**{name: f"job_x#training/{name}" for name in _STATES},
                "legacy_manifest.json": manifest_ref}
    assert state_stage_parameter_problems(
        None, _params(input_bindings=bindings)) == [
            "input_bindings['legacy_manifest.json'] must be a non-empty string"]


@pytest.mark.parametrize("input_bindings", [
    ["a.json", "legacy_manifest.json"], "a.json", None])
def test_non_mapping_input_bindings_are_reported_not_raised(input_bindings):
    params = _params(state_files=("a.json",), input_bindings=input_bindings)
    assert state_stage_parameter_problems(None, params) == [
        "input_bindings keys must equal state_files plus legacy_manifest.json"]


def test_state_files_must_not_list_the_manifest():
    params = _params(state_files=tuple(sorted(_STATES)) + ("legacy_manifest.json",),
                     input_bindings={**{name: "j#o" for name in _STATES},
                                     "legacy_manifest.json": "artifact_manifest"})
    assert state_stage_parameter_problems(None, params) == [
        "state_files must not list legacy_manifest.json"]


@pytest.mark.parametrize("changes", [
    {"expected_ids": ("training",)}, {"release_root": ""}, {"release_id": "prior1"},
    {"release_id": "../x"}, {"cutoff": "2026-9-1"}, {"cutoff": "bad"},
    {"input_bindings": None}, {"input_bindings": {"paired.json": "j#o"}},
])
def test_invalid_parameters_are_reported(changes):
    assert state_stage_parameter_problems(None, _params(**changes))


def test_kind_is_registered_and_leases_the_pointer():
    kind = stages.registry().get("phase5_state_stage")
    assert kind.worker == "phase5_state_stage"
    assert kind.resource_classes == frozenset({"projection"})
    assert kind.effects == ("staged",)
    assert kind.checkpoint_contract == "phase5_state_stage_result.v1.0"
    assert kind.namespaces == frozenset({"shadow", "smoke"})
    assert kind.store_domains == (("deployment_pointer", "write"), ("legacy_store", "read"))
    assert ("legacy_store", "read") in state_stage_job_kind().store_domains
    from engine.v2.ops.submission import RetryPolicy

    assert state_stage_job_kind().retry == RetryPolicy("none", 1)


def test_plan_pins_files_refs_and_absolute_root(tmp_path):
    refs = {name: f"job_x#training/{name}" for name in _STATES}
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01", state_refs=refs,
                            manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert plan["kind"] == "phase5_state_stage"
    assert plan["parameters"]["release_root"] == str(tmp_path.resolve())
    assert plan["input_refs"] == [*sorted(refs.values()), "artifact_manifest"]
    assert plan["parameters"]["state_files"] == tuple(sorted(refs))
    assert plan["parameters"]["input_bindings"]["legacy_manifest.json"] == "artifact_manifest"
    assert plan["mode"] == "shadow" and plan["effects"] == ["staged"]
    assert plan["resource_class"] == "projection"
    assert plan["schema_version"] == "operations_plan.v1.0"
    assert plan["blocked_prerequisites"] == []


def test_plan_uses_the_projection_scratch_for_the_legacy_read_set(tmp_path):
    from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named

    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01",
                            state_refs={"a.json": "j#o"},
                            manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert plan["resource_class"] == "projection"
    assert profile_named(DEFAULT_POLICY, "projection").scratch_bytes >= 2 * 1024**3


def test_plan_refuses_invalid_request(tmp_path):
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root=str(tmp_path), prior_release_id="p", release_id="p",
                         cutoff="2026-10-01", state_refs={"a.json": "j#o"},
                         manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert err.value.code == "INVALID_REQUEST"
    assert err.value.problem.message == "state stage plan is invalid"
    assert err.value.problem.details == {
        "problems": ["release_id must differ from prior_release_id"]}


@pytest.mark.parametrize("manifest_ref", [None, ""])
def test_plan_without_a_manifest_ref_is_an_invalid_request(tmp_path, manifest_ref):
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                         release_id="cand1", cutoff="2026-10-01",
                         state_refs={"a.json": "j#o"}, manifest_ref=manifest_ref,
                         source_root=REPO_ROOT)
    assert err.value.code == "INVALID_REQUEST"
    assert err.value.problem.message == "state stage plan is invalid"
    assert err.value.problem.details == {"problems": [
        "input_bindings['legacy_manifest.json'] must be a non-empty string"]}


def _wire(tmp_path, monkeypatch, states=None, drift=(), prior_names=None, mode=None,
          prior_release_id="prior1"):
    """Stage a prior release, patch the state classifier and the inventory, return worker args."""
    inventory, _body = _stage_prior(tmp_path, prior_names=prior_names)
    states = _STATES if states is None else states
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in states:
        (staging / name).write_bytes(f"state-{name}".encode())

    def fake_payloads(paths, max_bytes=None):
        from tools import phase5_state_stage_job

        assert max_bytes == phase5_state_stage_job.MAX_STATE_BYTES
        name = Path(paths[0]).name
        member, object_name = states[name]
        return {member: {object_name: Path(paths[0]).read_bytes()}}

    monkeypatch.setattr(phase5_prepare_release, "frozen_state_payloads", fake_payloads)
    monkeypatch.setattr(inventory_module, "current_release_inventory", lambda *a, **k: (inventory, drift))
    parameters = {"expected_ids": ["phase5_state_stage"], "release_root": str(tmp_path),
                  "prior_release_id": prior_release_id, "release_id": "cand1",
                  "cutoff": "2026-10-01",
                  "state_files": sorted(states),
                  "input_bindings": {**{name: f"job_x#training/{name}" for name in states},
                                     "legacy_manifest.json": "artifact_manifest"}}
    if mode is not None:
        parameters["mode"] = mode
        if mode == "bootstrap":
            parameters["bootstrap"] = dict(_BOOTSTRAP)
    return parameters, staging


def test_worker_stages_candidate_carrying_the_unadvanced_members(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    deployed = phase5_release.deployment_root(tmp_path) / "DEPLOYED"
    assert not deployed.exists()
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    assert result["no_work"] is False
    assert not deployed.exists()
    assert result["outputs"] == [{"name": "stage_result", "path": "stage_result.json",
                                  "schema": "phase5_state_stage_result.v1.0"}]
    summary = json.loads((staging / "stage_result.json").read_text())
    assert summary["n_new"] == 3 and summary["n_carried"] == 7
    assert summary["release_id"] == "cand1" and summary["prior_release_id"] == "prior1"
    assert summary["cutoff"] == "2026-10-01"
    assert summary["schema_version"] == "phase5_state_stage_result.v1.0"
    body = json.loads(phase5_release.catalog_path(tmp_path, "cand1").read_bytes())
    assert body["sources"]["pool_cutoff"] == "2026-10-01"
    assert body["sources"]["carried_from"] == "prior1"
    prior = json.loads(phase5_release.catalog_path(tmp_path, "prior1").read_bytes())
    rows = {row["member_id"]: row for row in body["members"]}
    prior_rows = {row["member_id"]: row for row in prior["members"]}
    for member_id in ("driver_residual_pool:size", "driver_residual_pool:implied_t1",
                      "driver_residual_pool:runup_move", "chooser_analog_pool"):
        assert rows[member_id]["objects"] == prior_rows[member_id]["objects"]
    names = {obj["name"] for obj in rows["board_analog_matcher"]["objects"]}
    assert names == {"prior.bin", "STR-THRU|0.5000|2026-10-01"}


def test_successful_stage_publishes_the_staging_success_record(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    assert run_state_stage_worker(parameters, staging)["completed_ids"] == ["phase5_state_stage"]
    status = (phase5_release.deployment_root(tmp_path) / "releases" / "cand1"
              / "staging-status.json")
    assert json.loads(status.read_text())["state"] == "succeeded"


def test_light_check_failure_leaves_no_staging_success_record(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    _tamper_catalog(monkeypatch, _drop_board_objects)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release failed the light checks"
    status = (phase5_release.deployment_root(tmp_path) / "releases" / "cand1"
              / "staging-status.json")
    assert not status.exists()


def test_a_mark_failure_refuses_with_a_fixed_message(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)

    def broken(root, release_id):
        raise DeploymentError("boom")

    monkeypatch.setattr(deployment, "mark_staging_succeeded", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.code == "VALIDATION_FAILED"
    assert str(err.value) == "VALIDATION_FAILED: staging success could not be recorded"
    status = (phase5_release.deployment_root(tmp_path) / "releases" / "cand1"
              / "staging-status.json")
    assert not status.exists()


def test_a_staged_candidate_promotes_without_a_staging_refusal(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    assert run_state_stage_worker(parameters, staging)["completed_ids"] == ["phase5_state_stage"]
    store = phase5_release.deployment_root(tmp_path)
    state = deployment.promote(store, "cand1")
    assert state.release_id == "cand1"
    assert deployment.current_pointer(store).release_id == "cand1"


def test_worker_refuses_a_missing_advancing_member(tmp_path, monkeypatch):
    states = {"paired.json": _STATES["paired.json"]}
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.code == "VALIDATION_FAILED"
    assert not (staging / "stage_result.json").exists()
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()
    assert err.value.problem.details == {"members": ["paired_residual_pool"]}
    assert str(err.value) == ("VALIDATION_FAILED: state files are not exactly "
                              "the advancing members")


def test_worker_refuses_a_state_file_beyond_the_advancing_members(tmp_path, monkeypatch):
    states = {**_STATES, "pool.json": ("driver_residual_pool:size", "prior|2026-10-01")}
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.code == "VALIDATION_FAILED"
    assert not (staging / "stage_result.json").exists()
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()
    assert err.value.problem.details == {"members": ["board_analog_matcher",
                                                     "driver_residual_pool:size",
                                                     "paired_residual_pool",
                                                     "trailing_pnl_cutoff"]}
    assert str(err.value) == ("VALIDATION_FAILED: state files are not exactly "
                              "the advancing members")


def test_worker_refuses_inventory_drift_before_any_candidate(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch, drift=("MISSING_ARTIFACT_FILE",))
    store = phase5_release.deployment_root(tmp_path) / "objects"
    before = sorted(path.name for path in store.iterdir())
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.problem.details == {"issues": 1}
    assert str(err.value) == "VALIDATION_FAILED: release inventory drifted"
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()
    assert sorted(path.name for path in store.iterdir()) == before


def test_worker_refuses_a_cutoff_that_does_not_advance(tmp_path, monkeypatch):
    states = {name: (member, obj.replace("2026-10-01", "2026-09-01"))
              for name, (member, obj) in _STATES.items()}
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    parameters["cutoff"] = "2026-09-01"
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.problem.details == {"code": "CUTOFF_NOT_ADVANCED"}
    assert str(err.value) == "VALIDATION_FAILED: release assembly refused"
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


@pytest.mark.parametrize("files,expected", [
    ((), "state_files must be a non-empty tuple without duplicates"),
    (("a.json", "a.json"), "state_files must be a non-empty tuple without duplicates"),
    (("a.txt",), "state_files must be plain .json names, not summaries"),
    (("../x.json",), "state_files must be plain .json names, not summaries"),
    (("a/b.json",), "state_files must be plain .json names, not summaries"),
    (("x.summary.json",), "state_files must be plain .json names, not summaries"),
])
def test_state_file_shape_problems_name_the_defect(files, expected):
    params = _params(state_files=files,
                     input_bindings={**{name: "j#o" for name in files},
                                     "legacy_manifest.json": "artifact_manifest"})
    assert state_stage_parameter_problems(None, params) == [expected]


def test_trailing_objects_are_stamped_and_paired_objects_use_their_name_cutoff(tmp_path, monkeypatch):
    states = dict(_STATES)
    states["trailing.json"] = ("trailing_pnl_cutoff", "hist|2026-09")
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    assert run_state_stage_worker(parameters, staging)["completed_ids"] == ["phase5_state_stage"]
    body = json.loads(phase5_release.catalog_path(tmp_path, "cand1").read_bytes())
    assert body["sources"]["pool_cutoff"] == "2026-10-01"


def test_paired_object_with_other_name_cutoff_is_refused(tmp_path, monkeypatch):
    states = dict(_STATES)
    states["paired.json"] = ("paired_residual_pool", "mv|cr|2026-10-02")
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.problem.details == {"code": "CUTOFF_MISMATCH"}
    assert str(err.value) == "VALIDATION_FAILED: release assembly refused"


def _tamper_catalog(monkeypatch, mutate):
    from checks import phase5_release_assembly as assembly_module

    real = assembly_module.carry_forward_release

    def wrapped(root, **kwargs):
        path = real(root, **kwargs)
        body = json.loads(path.read_bytes())
        mutate(body)
        body = phase5_release.manifest_body(
            body["release_id"], body["deployment_id"], body["members"], body["sources"])
        path.write_text(json.dumps(body))
        return path

    monkeypatch.setattr(assembly_module, "carry_forward_release", wrapped)


def _wrong_pool_cutoff(body):
    body["sources"]["pool_cutoff"] = "2000-01-01"


def _wrong_carried_from(body):
    body["sources"]["carried_from"] = "other"


def _drop_board_objects(body):
    for row in body["members"]:
        if row["member_id"] == "board_analog_matcher":
            row["objects"] = []


@pytest.mark.parametrize("mutate", [_wrong_pool_cutoff, _wrong_carried_from, _drop_board_objects])
def test_light_checks_refuse_a_candidate_that_does_not_match(tmp_path, monkeypatch, mutate):
    parameters, staging = _wire(tmp_path, monkeypatch)
    _tamper_catalog(monkeypatch, mutate)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release failed the light checks"
    assert not (staging / "stage_result.json").exists()


def test_plan_resolves_the_production_release_root(tmp_path, monkeypatch):
    from engine.v2.models import deployment

    monkeypatch.setattr(deployment, "production_release_root", lambda: tmp_path)
    plan = state_stage_plan(release_root="", prior_release_id="prior1", release_id="cand1",
                            cutoff="2026-10-01", state_refs={"a.json": "j#o"},
                            manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert plan["parameters"]["release_root"] == str(tmp_path.resolve())


def test_plan_without_any_release_root_is_an_invalid_request(monkeypatch):
    from engine.v2.models import deployment

    def missing():
        raise deployment.MissingReleaseRoot()

    monkeypatch.setattr(deployment, "production_release_root", missing)
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root="", prior_release_id="prior1", release_id="cand1",
                         cutoff="2026-10-01", state_refs={"a.json": "j#o"},
                         manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert err.value.code == "INVALID_REQUEST"
    assert err.value.problem.message == (
        "no release root given and no production release root is configured")


def test_worker_refuses_a_symlinked_state_file(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    target = staging / "paired.json"
    real = tmp_path / "real.json"
    real.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(real)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a bound state file is not a bounded regular file"


def test_real_loader_refuses_oversize_symlink_and_fifo(tmp_path):
    import os

    from engine.v2.models.frozen_state_io import BoundedStateReadError

    big = tmp_path / "big.json"
    big.write_bytes(b"x" * 10)
    with pytest.raises(BoundedStateReadError):
        phase5_prepare_release.frozen_state_payloads([big], max_bytes=9)
    link = tmp_path / "link.json"
    link.symlink_to(big)
    with pytest.raises(OSError):
        phase5_prepare_release.frozen_state_payloads([link], max_bytes=100)
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(BoundedStateReadError):
        phase5_prepare_release.frozen_state_payloads([fifo], max_bytes=100)


def test_light_checks_refuse_an_oversize_catalog(tmp_path, monkeypatch):
    from tools import phase5_state_stage_job as job

    path = phase5_release.catalog_path(tmp_path, "cand1")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{" + b" " * 20 + b"}")
    monkeypatch.setattr(phase5_release, "MAX_CATALOG_BYTES", 5)
    parameters = {"release_id": "cand1", "cutoff": "2026-10-01", "prior_release_id": "prior1"}
    with pytest.raises(job.StageRefused) as err:
        job.light_checks(tmp_path, parameters, set())
    assert str(err.value) == "the candidate catalog is not a bounded regular file"


def test_worker_maps_a_wrong_typed_state_file_to_a_typed_refusal(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)

    def wrong_types(paths, max_bytes=None):
        raise KeyError("role")

    monkeypatch.setattr(phase5_prepare_release, "frozen_state_payloads", wrong_types)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a bound state file could not be loaded"


def test_worker_refuses_an_oversize_state_file(tmp_path, monkeypatch):
    from tools import phase5_state_stage_job

    parameters, staging = _wire(tmp_path, monkeypatch)
    monkeypatch.setattr(phase5_state_stage_job, "MAX_STATE_BYTES", 1)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a bound state file is not a bounded regular file"


def test_worker_refuses_a_duplicate_state_object(tmp_path, monkeypatch):
    states = dict(_STATES)
    states["board.json"] = _STATES["paired.json"]
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a state object appears twice"
    assert err.value.problem.details == {"member": "paired_residual_pool"}


def test_worker_refuses_a_corrupt_state_file(tmp_path, monkeypatch):
    from engine.v2.models.frozen_state import FrozenStateError

    parameters, staging = _wire(tmp_path, monkeypatch)

    def corrupt(paths, max_bytes=None):
        raise FrozenStateError("bad")

    monkeypatch.setattr(phase5_prepare_release, "frozen_state_payloads", corrupt)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a bound state file could not be loaded"


@pytest.mark.parametrize("exc_factory", [
    lambda: ValueError("corrupt registry"),
    lambda: ArtifactError("UNSAFE_PATH", "bad"),
    lambda: DeploymentError("bad"),
])
def test_worker_refuses_an_inventory_that_cannot_be_built(tmp_path, monkeypatch, exc_factory):
    parameters, staging = _wire(tmp_path, monkeypatch)

    def broken(*args, **kwargs):
        raise exc_factory()

    monkeypatch.setattr(inventory_module, "current_release_inventory", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: release inventory could not be built"
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def _corrupt_staged_manifest(monkeypatch):
    from checks import phase5_release_assembly as assembly_module

    real = assembly_module.carry_forward_release

    def wrapped(root, **kwargs):
        path = real(root, **kwargs)
        manifest = phase5_release.staged_manifest_path(
            phase5_release.deployment_root(root), kwargs["release_id"])
        manifest.write_text("{}")
        return path

    monkeypatch.setattr(assembly_module, "carry_forward_release", wrapped)


def test_light_checks_refuse_a_corrupted_staged_manifest(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    _corrupt_staged_manifest(monkeypatch)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release failed the light checks"


def test_light_checks_refuse_a_catalog_with_no_members(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    _tamper_catalog(monkeypatch, lambda body: body.__setitem__("members", []))
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release failed the light checks"


def test_a_resupplied_object_replaces_the_prior_object_of_the_same_name(tmp_path, monkeypatch):
    from checks import phase5_release_assembly as assembly_module

    names = {member_id: "prior.bin" for member_id in (
        *assembly_module.POOL_MEMBER_IDS, assembly_module.CHOOSER_MEMBER_ID, UNRELATED_MEMBER_ID)}
    names["trailing_pnl_cutoff"] = "hist|2026-10"
    states = dict(_STATES)
    states["trailing.json"] = ("trailing_pnl_cutoff", "hist|2026-10")
    parameters, staging = _wire(tmp_path, monkeypatch, states=states, prior_names=names)
    run_state_stage_worker(parameters, staging)
    body = json.loads(phase5_release.catalog_path(tmp_path, "cand1").read_bytes())
    rows = {row["member_id"]: row for row in body["members"]}
    new_hash = phase5_release.sha256_bytes((staging / "trailing.json").read_bytes())
    assert [(o["name"], o["content_hash"]) for o in rows["trailing_pnl_cutoff"]["objects"]] == [
        ("hist|2026-10", new_hash)]
    assert {o["name"] for o in rows["board_analog_matcher"]["objects"]} == {
        "prior.bin", "STR-THRU|0.5000|2026-10-01"}
    assert [o["name"] for o in rows["driver_residual_pool:size"]["objects"]] == ["prior.bin"]


def test_a_trailing_month_after_the_cutoff_is_refused(tmp_path, monkeypatch):
    states = dict(_STATES)
    states["trailing.json"] = ("trailing_pnl_cutoff", "hist|2026-11")
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.problem.details == {"months": ["hist|2026-11"]}
    assert str(err.value) == "VALIDATION_FAILED: a trailing object month starts after cutoff"
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def _tamper_raw(monkeypatch, mutate):
    from checks import phase5_release_assembly as assembly_module

    real = assembly_module.carry_forward_release

    def wrapped(root, **kwargs):
        path = real(root, **kwargs)
        body = json.loads(path.read_bytes())
        mutate(body)
        path.write_text(json.dumps(body))
        return path

    monkeypatch.setattr(assembly_module, "carry_forward_release", wrapped)


def test_light_checks_refuse_a_stale_manifest_hash(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    _tamper_raw(monkeypatch, lambda body: body["sources"].__setitem__("note", "x"))
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release failed the light checks"


def test_light_checks_refuse_a_catalog_for_another_release(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    _tamper_catalog(monkeypatch, lambda body: body.__setitem__("release_id", "other"))
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release failed the light checks"


def test_worker_dispatch_routes_phase5_state_stage(monkeypatch):
    from engine.v2.ops import state_stage, worker

    seen = {}

    def fake(parameters, root):
        seen["args"] = (parameters, root)
        return {"ok": True}

    monkeypatch.setattr(state_stage, "run_state_stage_worker", fake)
    assert worker.dispatch("phase5_state_stage", {"p": 1}, "/r") == {"ok": True}
    assert seen["args"] == ({"p": 1}, "/r")


def test_registered_kind_validates_and_plan_hash_matches(tmp_path):
    from engine.v2.foundation import content_hash

    kind = stages.registry().get("phase5_state_stage")
    assert kind.validate(None, _params()) == []
    assert kind.validate(None, _params(cutoff="bad"))
    refs = {name: f"job_x#training/{name}" for name in _STATES}
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01", state_refs=refs,
                            manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert plan["spec_hash"] == content_hash(plan["parameters"])
    assert plan["implementation_ref"] and plan["environment_ref"]


def test_plan_hashes_pin_the_source_root_and_environment(tmp_path, monkeypatch):
    from dataclasses import replace

    from engine.v2.foundation import content_hash
    from engine.v2.ops import state_stage
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
    from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named

    projection = replace(profile_named(DEFAULT_POLICY, "projection"), thread_count=3)
    monkeypatch.setattr(state_stage, "DEFAULT_POLICY", replace(
        DEFAULT_POLICY, profiles=tuple(projection if profile.name == "projection" else profile
                                       for profile in DEFAULT_POLICY.profiles)))
    refs = {name: f"job_x#training/{name}" for name in _STATES}
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01", state_refs=refs,
                            manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert plan["implementation_ref"] == content_hash(worker_source_manifest(REPO_ROOT))
    assert plan["environment_ref"] == content_hash(environment_identity(3))


def test_plan_environment_ref_falls_back_to_the_pinned_profile_cpu_count(tmp_path, monkeypatch):
    from dataclasses import replace

    from engine.v2.foundation import content_hash
    from engine.v2.ops import state_stage
    from engine.v2.ops.fingerprints import environment_identity
    from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named

    projection = replace(profile_named(DEFAULT_POLICY, "projection"), thread_count=0, cpu_count=5)
    monkeypatch.setattr(state_stage, "DEFAULT_POLICY", replace(
        DEFAULT_POLICY, profiles=tuple(projection if profile.name == "projection" else profile
                                       for profile in DEFAULT_POLICY.profiles)))
    refs = {name: f"job_x#training/{name}" for name in _STATES}
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01", state_refs=refs,
                            manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert plan["environment_ref"] == content_hash(environment_identity(5))


def test_plan_refusal_publishes_the_exact_problems(tmp_path):
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root=str(tmp_path), prior_release_id="p", release_id="p",
                         cutoff="2026-10-01", state_refs={"a.json": "j#o"},
                         manifest_ref="artifact_manifest", source_root=REPO_ROOT)
    assert err.value.problem.message == "state stage plan is invalid"
    assert err.value.problem.details == {"problems": ["release_id must differ from prior_release_id"]}


def test_one_bad_file_among_good_ones_is_reported():
    params = _params(state_files=("a.json", "x.txt"),
                     input_bindings={"a.json": "j#o", "x.txt": "j#o",
                                     "legacy_manifest.json": "artifact_manifest"})
    assert "state_files must be plain .json names, not summaries" in state_stage_parameter_problems(
        None, params)


def test_worker_refuses_a_missing_bound_file(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    (staging / "paired.json").unlink()
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.code == "VALIDATION_FAILED"
    assert str(err.value) == "VALIDATION_FAILED: a bound state file could not be read"


def test_a_trailing_month_that_is_not_a_date_is_refused(tmp_path, monkeypatch):
    states = dict(_STATES)
    states["trailing.json"] = ("trailing_pnl_cutoff", "hist|abc")
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.code == "VALIDATION_FAILED"
    assert err.value.problem.details == {"months": ["hist|abc"]}
    assert str(err.value) == "VALIDATION_FAILED: a trailing object month is not a date"


def test_state_file_one_byte_over_the_cap_is_refused(tmp_path, monkeypatch):
    from tools import phase5_state_stage_job

    parameters, staging = _wire(tmp_path, monkeypatch)
    largest = max((staging / name).stat().st_size for name in _STATES)
    monkeypatch.setattr(phase5_state_stage_job, "MAX_STATE_BYTES", largest - 1)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a bound state file is not a bounded regular file"


@pytest.mark.parametrize("changes,problem", [
    ({"expected_ids": ("training",)}, "expected_ids must be exactly ('phase5_state_stage',)"),
    ({"release_root": ""}, "release_root must be a non-empty string"),
    ({"release_id": "../x"}, "prior_release_id and release_id must be safe non-empty ids"),
    ({"release_id": ""}, "prior_release_id and release_id must be safe non-empty ids"),
    ({"release_id": 5}, "prior_release_id and release_id must be safe non-empty ids"),
    ({"release_id": ".hidden"}, "prior_release_id and release_id must be safe non-empty ids"),
    ({"release_id": "a\\b"}, "prior_release_id and release_id must be safe non-empty ids"),
    ({"release_id": "a\x00b"}, "prior_release_id and release_id must be safe non-empty ids"),
    ({"release_id": "prior1"}, "release_id must differ from prior_release_id"),
    ({"cutoff": "2026-9-1"}, "cutoff must be an extended ISO date (YYYY-MM-DD)"),
    ({"input_bindings": None},
     "input_bindings keys must equal state_files plus legacy_manifest.json"),
])
def test_each_invalid_parameter_reports_its_exact_problem(changes, problem):
    assert state_stage_parameter_problems(None, _params(**changes)) == [problem]


def test_plan_hashes_follow_the_source_root(tmp_path, monkeypatch):
    from engine.v2.foundation import content_hash
    from engine.v2.ops import fingerprints

    seen = []

    def manifest(root):
        seen.append(root)
        return {"root": str(root)}

    monkeypatch.setattr(fingerprints, "worker_source_manifest", manifest)
    refs = {name: f"job_x#training/{name}" for name in _STATES}
    first, second = tmp_path / "a", tmp_path / "b"
    plans = [state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                              release_id="cand1", cutoff="2026-10-01", state_refs=refs,
                              manifest_ref="artifact_manifest", source_root=root)
             for root in (first, second)]
    assert seen == [first, second]
    assert plans[0]["implementation_ref"] == content_hash({"root": str(first)})
    assert plans[1]["implementation_ref"] == content_hash({"root": str(second)})
    assert plans[0]["environment_ref"] == plans[1]["environment_ref"]
    assert plans[0]["spec_hash"] == content_hash(plans[0]["parameters"])


def test_light_checks_refuse_a_vanished_catalog_with_the_exact_message(tmp_path, monkeypatch):
    from checks import phase5_release_assembly as assembly_module

    real = assembly_module.carry_forward_release

    def wrapped(root, **kwargs):
        path = real(root, **kwargs)
        path.unlink()
        return path

    parameters, staging = _wire(tmp_path, monkeypatch)
    monkeypatch.setattr(assembly_module, "carry_forward_release", wrapped)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: the candidate catalog could not be read"


def test_worker_refuses_a_missing_prior_release(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: prior release could not be read"
    assert err.value.problem.details == {"code": "PRIOR_MISMATCH"}
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_worker_refuses_a_malformed_prior_catalog(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    path = phase5_release.catalog_path(tmp_path, "prior1")
    body = json.loads(path.read_bytes())
    body["members"] = []
    path.write_text(json.dumps(body))
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: prior release could not be read"
    assert err.value.problem.details == {"code": "PRIOR_MISMATCH"}
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_object_write_failure_is_a_typed_refusal(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)

    def broken(release_root, payload):
        raise OSError("disk full")

    monkeypatch.setattr(phase5_release, "write_object", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate objects could not be written"
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_carry_failure_is_a_typed_publish_refusal(tmp_path, monkeypatch):
    from checks import phase5_release_assembly as assembly_module

    def broken(root, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(assembly_module, "carry_forward_release", broken)
    parameters, staging = _wire(tmp_path, monkeypatch)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"


def test_state_file_of_exactly_the_cap_is_accepted(tmp_path, monkeypatch):
    from tools import phase5_state_stage_job

    parameters, staging = _wire(tmp_path, monkeypatch)
    largest = max((staging / name).stat().st_size for name in _STATES)
    monkeypatch.setattr(phase5_state_stage_job, "MAX_STATE_BYTES", largest)
    assert run_state_stage_worker(parameters, staging)["completed_ids"] == ["phase5_state_stage"]


def test_state_cap_is_pinned():
    from tools import phase5_state_stage_job

    assert phase5_state_stage_job.MAX_STATE_BYTES == 512 * 1024 * 1024


def test_worker_refuses_an_invalid_cutoff_before_staging(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    parameters["cutoff"] = "not-a-date"
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: state stage parameters are invalid"
    assert err.value.problem.details == {
        "problems": ["cutoff must be an extended ISO date (YYYY-MM-DD)"]}
    assert not (staging / "stage_result.json").exists()
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_worker_refuses_a_traversing_state_file_before_staging(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    parameters["state_files"] = [*parameters["state_files"], "../evil.json"]
    parameters["input_bindings"] = {
        **{name: f"job_x#training/{name}" for name in parameters["state_files"]},
        "legacy_manifest.json": "artifact_manifest"}
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: state stage parameters are invalid"
    assert err.value.problem.details == {
        "problems": ["state_files must be plain .json names, not summaries"]}
    assert not (staging / "stage_result.json").exists()
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_worker_refuses_a_parameters_dict_missing_a_field(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    del parameters["release_id"]
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: state stage parameters are invalid"
    assert err.value.problem.details == {
        "problems": ["parameters are missing a field or have the wrong type"]}
    assert not (staging / "stage_result.json").exists()
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_bootstrap_mode_accepts_a_null_prior():
    assert state_stage_parameter_problems(
        None, _params(mode="bootstrap", prior_release_id=None)) == []


def test_bootstrap_with_an_unsafe_non_null_prior_is_reported():
    assert state_stage_parameter_problems(
        None, _params(mode="bootstrap", prior_release_id="../x")) == [
            "prior_release_id and release_id must be safe non-empty ids"]


def test_bootstrap_with_prior_equal_to_release_is_reported():
    assert state_stage_parameter_problems(
        None, _params(mode="bootstrap", prior_release_id="cand1")) == [
            "release_id must differ from prior_release_id"]


def test_unknown_mode_is_reported():
    assert state_stage_parameter_problems(
        None, _params(mode="sideways")) == ["mode must be 'carry' or 'bootstrap'"]


def test_carry_with_a_null_prior_is_reported():
    assert state_stage_parameter_problems(None, _params(prior_release_id=None)) == [
        "prior_release_id and release_id must be safe non-empty ids"]


def test_bootstrap_with_a_null_prior_checks_only_the_release_id():
    assert state_stage_parameter_problems(
        None, _params(mode="bootstrap", prior_release_id=None, release_id="../x")) == [
            "release_id must be a safe non-empty id"]


def test_plan_passes_the_bootstrap_mode_through(tmp_path):
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id=None,
                            release_id="cand1", cutoff="2026-10-01",
                            state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                            source_root=REPO_ROOT, mode="bootstrap",
                            bootstrap=dict(_BOOTSTRAP))
    assert plan["parameters"]["mode"] == "bootstrap"
    assert plan["parameters"]["bootstrap"] == _BOOTSTRAP


def _store_names(root):
    store = phase5_release.deployment_root(root) / "objects"
    return sorted(path.name for path in store.iterdir()) if store.exists() else []


def _assert_bootstrap_refusal(tmp_path, staging, parameters, message, code):
    before = _store_names(tmp_path)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == f"VALIDATION_FAILED: {message}"
    assert err.value.problem.details == {"code": code}
    assert _store_names(tmp_path) == before
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()
    assert not (staging / "stage_result.json").exists()


def test_bootstrap_refuses_a_damaged_prior_before_any_write(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch, mode="bootstrap")
    phase5_release.catalog_path(tmp_path, "prior1").write_bytes(b"{not json")
    monkeypatch.setattr("checks.phase5_release_assembly.prior_carry_set",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("carry path reached")))
    _assert_bootstrap_refusal(tmp_path, staging, parameters,
                              "prior release could not be read", "PRIOR_MISMATCH")


def test_bootstrap_refuses_a_usable_prior(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch, mode="bootstrap")
    _assert_bootstrap_refusal(tmp_path, staging, parameters,
                              "prior release already has a pool catalog", "PRIOR_USABLE")


def test_light_checks_accept_a_null_carried_from_in_bootstrap(tmp_path, monkeypatch):
    from tools import phase5_state_stage_job as job

    parameters, staging = _wire(tmp_path, monkeypatch)
    assert run_state_stage_worker(parameters, staging)["completed_ids"] == ["phase5_state_stage"]
    path = phase5_release.catalog_path(tmp_path, "cand1")
    body = json.loads(path.read_bytes())
    body["sources"]["carried_from"] = None
    path.write_text(json.dumps(phase5_release.manifest_body(
        body["release_id"], body["deployment_id"], body["members"], body["sources"])))
    job.light_checks(tmp_path, {**parameters, "mode": "bootstrap"}, set())


def test_light_checks_refuse_a_non_null_carried_from_in_bootstrap(tmp_path, monkeypatch):
    from tools import phase5_state_stage_job as job

    parameters, staging = _wire(tmp_path, monkeypatch)
    assert run_state_stage_worker(parameters, staging)["completed_ids"] == ["phase5_state_stage"]
    with pytest.raises(job.StageRefused) as err:
        job.light_checks(tmp_path, {**parameters, "mode": "bootstrap"}, set())
    assert str(err.value) == "candidate release failed the light checks"
