from __future__ import annotations

import json
from pathlib import Path

import pytest

from checks import phase5_release
from checks import phase5_release_assembly as assembly
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

REPO_ROOT = Path(__file__).resolve().parents[3]
UNRELATED_MEMBER_ID = "payoff_line:STR-THRU"


def _fixture(release_id, deployment_id="d1", binding_id="b1"):
    payload = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
                          "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}]},
                         sort_keys=True).encode()
    member_hash = phase5_release.sha256_bytes(payload)
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(binding_id=binding_id, model_id="m1", role="size", strategy_id="*",
                           decision_clock_id="entry-close", adapter="json-linear.v1",
                           feature_order=("x",), output_names=("prediction",), members=(member,))
    release = ModelRelease(release_id=release_id, deployment_id=deployment_id, bindings=(binding,))
    inv_member = ArtifactInventoryMember(member_id="m1:estimator", kind="estimator",
                                         artifact_ref="artifact://m1", content_hash=member_hash)
    artifact = ModelArtifactInventory(artifact_id="m1", role="size", strategy_ids=("*",),
                                      compatible_clock_ids=("entry-close",),
                                      target_contract_ref="return.v1", ordered_features=("x",),
                                      members=(inv_member,))
    inv_binding = ReleaseBinding(role="size", strategy_id="*", clock_id="entry-close",
                                 artifact_id="m1", ordered_features=("x",),
                                 required_member_kinds=("estimator",))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id=deployment_id, known_clock_ids=("entry-close",),
        artifacts=(artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",))
    return release, inventory, {member_hash: payload}


def _spec(member_id):
    return next(spec for spec in phase5_release.STATE_SPECS if spec.member_id == member_id)


def _stage_prior(root, release_id="prior1", deployment_id="d1", binding_id="b1",
                 prior_names=None, prior_sources=None):
    release, inventory, payloads = _fixture(release_id, deployment_id=deployment_id,
                                            binding_id=binding_id)
    deployment.stage_release(phase5_release.deployment_root(root), release, inventory, payloads)
    rows = []
    for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID, UNRELATED_MEMBER_ID):
        payload = f"prior-{member_id}".encode()
        content_hash, rel = phase5_release.write_object(root, payload)
        name = "prior.bin" if prior_names is None else prior_names[member_id]
        obj = {"name": name, "path": rel, "content_hash": content_hash, "bytes": len(payload)}
        rows.append(phase5_release.member_row(_spec(member_id), phase5_release.STAGED, [obj], "prior"))
    sources = {"pool_cutoff": "2026-09-01"} if prior_sources is None else prior_sources
    body = phase5_release.manifest_body(release_id, deployment_id, rows, sources)
    phase5_release.write_manifest(root, body)
    return inventory, body


_STATES = {
    "paired.json": ("paired_residual_pool", "mv|cr|2026-10-01"),
    "board.json": ("board_analog_matcher", "STR-THRU|0.5000|2026-10-01"),
    "trailing.json": ("trailing_pnl_cutoff", "hist|2026-10-01"),
}


def _params(**changes):
    values = dict(expected_ids=("phase5_state_stage",), release_root="/r", prior_release_id="prior1",
                  release_id="cand1", cutoff="2026-10-01", state_files=tuple(sorted(_STATES)),
                  input_bindings={name: f"job_x#training/{name}" for name in _STATES})
    values.update(changes)
    return StateStageParameters(**values)


def test_valid_parameters_have_no_problems():
    assert state_stage_parameter_problems(None, _params()) == []


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
    assert kind.resource_classes == frozenset({"delivery"})
    assert kind.effects == ("staged",)
    assert kind.checkpoint_contract == "phase5_state_stage_result.v1.0"
    assert kind.namespaces == frozenset({"shadow", "smoke"})
    assert kind.store_domains == (("deployment_pointer", "write"),)
    assert state_stage_job_kind().retry.max_attempts == 1


def test_plan_pins_files_refs_and_absolute_root(tmp_path):
    refs = {name: f"job_x#training/{name}" for name in _STATES}
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01", state_refs=refs,
                            source_root=REPO_ROOT)
    assert plan["kind"] == "phase5_state_stage"
    assert plan["parameters"]["release_root"] == str(tmp_path.resolve())
    assert plan["input_refs"] == sorted(refs.values())
    assert plan["parameters"]["state_files"] == tuple(sorted(refs))
    assert plan["mode"] == "shadow" and plan["effects"] == ["staged"]
    assert plan["resource_class"] == "delivery"
    assert plan["schema_version"] == "operations_plan.v1.0"
    assert plan["blocked_prerequisites"] == []


def test_plan_refuses_invalid_request(tmp_path):
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root=str(tmp_path), prior_release_id="p", release_id="p",
                         cutoff="2026-10-01", state_refs={"a.json": "j#o"},
                         source_root=REPO_ROOT)
    assert err.value.code == "INVALID_REQUEST"


def _wire(tmp_path, monkeypatch, states=None, drift=()):
    """Stage a prior release, patch the state classifier and the inventory, return worker args."""
    inventory, _body = _stage_prior(tmp_path)
    states = _STATES if states is None else states
    staging = tmp_path / "staging"
    staging.mkdir()
    for name in states:
        (staging / name).write_bytes(f"state-{name}".encode())

    def fake_payloads(paths):
        name = Path(paths[0]).name
        member, object_name = states[name]
        return {member: {object_name: Path(paths[0]).read_bytes()}}

    monkeypatch.setattr(phase5_prepare_release, "frozen_state_payloads", fake_payloads)
    monkeypatch.setattr(inventory_module, "current_release_inventory", lambda *a, **k: (inventory, drift))
    parameters = {"expected_ids": ["phase5_state_stage"], "release_root": str(tmp_path),
                  "prior_release_id": "prior1", "release_id": "cand1", "cutoff": "2026-10-01",
                  "state_files": sorted(states)}
    return parameters, staging


def test_worker_stages_candidate_carrying_the_unadvanced_members(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    assert result["no_work"] is False
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


def test_worker_refuses_a_missing_advancing_member(tmp_path, monkeypatch):
    states = {"paired.json": _STATES["paired.json"]}
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.code == "VALIDATION_FAILED"
    assert not (staging / "stage_result.json").exists()
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()
    assert err.value.problem.details == {"members": ["paired_residual_pool"]}


def test_worker_refuses_inventory_drift_before_any_candidate(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch, drift=("MISSING_ARTIFACT_FILE",))
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.problem.details == {"issues": 1}
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_worker_refuses_a_cutoff_that_does_not_advance(tmp_path, monkeypatch):
    states = {name: (member, obj.replace("2026-10-01", "2026-09-01"))
              for name, (member, obj) in _STATES.items()}
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    parameters["cutoff"] = "2026-09-01"
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.problem.details == {"code": "CUTOFF_NOT_ADVANCED"}
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


@pytest.mark.parametrize("files,text", [
    ((), "non-empty tuple"), (("a.json", "a.json"), "without duplicates"),
    (("a.txt",), "plain .json"), (("../x.json",), "plain .json"), (("a/b.json",), "plain .json"),
    (("x.summary.json",), "plain .json"),
])
def test_state_file_shape_problems_name_the_defect(files, text):
    params = _params(state_files=files, input_bindings={name: "j#o" for name in files})
    problems = state_stage_parameter_problems(None, params)
    assert any(text in problem for problem in problems)


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
    assert err.value.code == "VALIDATION_FAILED"
    assert "light checks" in str(err.value)
    assert not (staging / "stage_result.json").exists()


def test_plan_resolves_the_production_release_root(tmp_path, monkeypatch):
    from engine.v2.models import deployment

    monkeypatch.setattr(deployment, "production_release_root", lambda: tmp_path)
    plan = state_stage_plan(release_root="", prior_release_id="prior1", release_id="cand1",
                            cutoff="2026-10-01", state_refs={"a.json": "j#o"},
                            source_root=REPO_ROOT)
    assert plan["parameters"]["release_root"] == str(tmp_path.resolve())


def test_plan_without_any_release_root_is_an_invalid_request(monkeypatch):
    from engine.v2.models import deployment

    def missing():
        raise deployment.MissingReleaseRoot()

    monkeypatch.setattr(deployment, "production_release_root", missing)
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root="", prior_release_id="prior1", release_id="cand1",
                         cutoff="2026-10-01", state_refs={"a.json": "j#o"},
                         source_root=REPO_ROOT)
    assert err.value.code == "INVALID_REQUEST"


def test_worker_refuses_a_symlinked_state_file(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    target = staging / "paired.json"
    real = tmp_path / "real.json"
    real.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(real)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert err.value.code == "VALIDATION_FAILED"
    assert "bounded regular file" in str(err.value)


def test_worker_refuses_an_oversize_state_file(tmp_path, monkeypatch):
    from tools import phase5_state_stage_job

    parameters, staging = _wire(tmp_path, monkeypatch)
    monkeypatch.setattr(phase5_state_stage_job, "MAX_STATE_BYTES", 1)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert "bounded regular file" in str(err.value)


def test_worker_refuses_a_duplicate_state_object(tmp_path, monkeypatch):
    states = dict(_STATES)
    states["board.json"] = _STATES["paired.json"]
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert "appears twice" in str(err.value)
    assert err.value.problem.details == {"member": "paired_residual_pool"}


def test_worker_refuses_a_corrupt_state_file(tmp_path, monkeypatch):
    from engine.v2.models.frozen_state import FrozenStateError

    parameters, staging = _wire(tmp_path, monkeypatch)

    def corrupt(paths):
        raise FrozenStateError("bad")

    monkeypatch.setattr(phase5_prepare_release, "frozen_state_payloads", corrupt)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert "could not be loaded" in str(err.value)


def test_worker_refuses_an_inventory_that_cannot_be_built(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)

    def broken(*args, **kwargs):
        raise ValueError("corrupt registry")

    monkeypatch.setattr(inventory_module, "current_release_inventory", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert "inventory could not be built" in str(err.value)
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
    assert err.value.code == "VALIDATION_FAILED"
    assert "light checks" in str(err.value)


def test_light_checks_refuse_a_catalog_with_a_bad_schema(tmp_path, monkeypatch):
    parameters, staging = _wire(tmp_path, monkeypatch)
    _tamper_catalog(monkeypatch, lambda body: body.__setitem__("members", []))
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert "light checks" in str(err.value)


def test_a_resupplied_object_replaces_the_prior_object_of_the_same_name(tmp_path, monkeypatch):
    states = dict(_STATES)
    states["trailing.json"] = ("trailing_pnl_cutoff", "prior.bin")
    parameters, staging = _wire(tmp_path, monkeypatch, states=states)
    run_state_stage_worker(parameters, staging)
    body = json.loads(phase5_release.catalog_path(tmp_path, "cand1").read_bytes())
    rows = {row["member_id"]: row for row in body["members"]}
    new_hash = phase5_release.sha256_bytes((staging / "trailing.json").read_bytes())
    assert [(o["name"], o["content_hash"]) for o in rows["trailing_pnl_cutoff"]["objects"]] == [
        ("prior.bin", new_hash)]
    assert {o["name"] for o in rows["board_analog_matcher"]["objects"]} == {
        "prior.bin", "STR-THRU|0.5000|2026-10-01"}
    assert [o["name"] for o in rows["driver_residual_pool:size"]["objects"]] == ["prior.bin"]
