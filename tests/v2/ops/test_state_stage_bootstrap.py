"""Bootstrap assembly tests (#1102 slice B); carry tests live in test_state_stage.py."""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.models import deployment
from engine.v2.models import inventory as inventory_module
from engine.v2.models import release_layout as phase5_release
from engine.v2.models.payoff_artifact import (
    PAYOFF_LINE_ARTIFACT_V1,
    make_payoff_line_artifact,
    serialize_payoff_artifact,
)
from engine.v2.models.recalibration_artifact import serialize_recalibration_artifact
from engine.v2.models.training.recalibration import build_recalibration_map_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.state_stage import (
    StateStageParameters,
    run_state_stage_worker,
    state_stage_parameter_problems,
    state_stage_plan,
)
from tools import phase5_prepare_release

from tests.v2.ops.state_stage_support import (
    REPO_ROOT,
    STATES as _STATES,
    fixture as _fixture,
    params as _params,
    spec as _spec,
    stage_prior as _stage_prior,
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


def _expected_bootstrap_objects(states):
    """Member ids and object hashes the mocked bootstrap payloads must yield."""
    hashes = {phase5_release.sha256_bytes(f"state-{name}".encode()) for name in states}
    hashes.add(phase5_release.sha256_bytes(b"table"))
    members = {member for member, _name in states.values()}
    members.add("admissible_table:dyn_sv")
    return members, hashes


def _catalog_contents(root, release_id="cand1"):
    body = json.loads(phase5_release.catalog_path(root, release_id).read_bytes())
    members = {row["member_id"] for row in body["members"]}
    hashes = {obj["content_hash"] for row in body["members"] for obj in row["objects"]}
    return members, hashes


def _catalog_pairs(root, release_id="cand1"):
    body = json.loads(phase5_release.catalog_path(root, release_id).read_bytes())
    return {(row["member_id"], obj["name"])
            for row in body["members"] for obj in row["objects"]}


def _expected_bootstrap_pairs(states):
    pairs = {pair for pair in states.values()}
    pairs.add(("admissible_table:dyn_sv", "t|1"))
    return pairs


_LINE_FIT = {
    "intercept": 0.05, "slope": 0.004, "resid_sd": 0.01, "n": 250,
    "r": 0.42, "residuals": [0.01, -0.02, 0.03, -0.01],
}


def _artifact_pairs(n=900, seed=5):
    rng = np.random.default_rng(seed)
    raw = np.round(rng.uniform(0.0, 1.0, n), 2)
    frame = pd.DataFrame({
        "strategy": np.where(rng.random(n) < 0.8, "STR-THRU", "STR-RUNUP"),
        "fill_alpha": np.where(rng.random(n) < 0.8, 0.5, 0.25),
        "event_id": [f"E{i}" for i in range(n)],
        "ticker": "AAA",
        "event_date": pd.Timestamp("2019-01-01"),
        "exit_date": pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 800, n), "D"),
        "raw_win": raw,
        "outcome": (rng.uniform(0, 1, n) < 0.2 + 0.5 * raw).astype(float),
    })
    frame.loc[:6, "raw_win"] = np.nan
    frame.loc[7:9, "outcome"] = np.nan
    return frame


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

    monkeypatch.setattr("tools.phase5_state_bootstrap.classify_bound_files", fake_classify)
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
    expected_parameters = {
        "expected_ids": ("phase5_state_stage",),
        "release_root": str(tmp_path.resolve()),
        "prior_release_id": "prior1",
        "release_id": "cand1",
        "cutoff": "2026-10-01",
        "state_files": ("a.json",),
        "input_bindings": {"a.json": "j#o", "legacy_manifest.json": "artifact_manifest"},
        "mode": "carry",
    }
    assert plan["parameters"] == expected_parameters
    spec_hash = plan["spec_hash"]
    assert spec_hash == content_hash(expected_parameters)
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
    assert plan["spec_hash"] == content_hash(plan["parameters"])


def test_bootstrap_plan_copies_the_caller_mapping(tmp_path):
    bootstrap = {"tier4_dir": "/t", "panel_catalog": "/c",
                 "panel_snapshot_id": "s", "panel_artifact_root": "/a"}
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01",
                            state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                            source_root=REPO_ROOT, mode="bootstrap", bootstrap=bootstrap)
    spec_hash = plan["spec_hash"]
    bootstrap["tier4_dir"] = "/changed"
    bootstrap["extra"] = "x"
    assert plan["parameters"]["bootstrap"] == {
        "tier4_dir": "/t", "panel_catalog": "/c",
        "panel_snapshot_id": "s", "panel_artifact_root": "/a"}
    assert plan["spec_hash"] == spec_hash == content_hash(plan["parameters"])


def test_bootstrap_plan_refuses_a_bad_mapping(tmp_path):
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                         release_id="cand1", cutoff="2026-10-01",
                         state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                         source_root=REPO_ROOT, mode="bootstrap", bootstrap={"x": 1})
    assert err.value.code == "INVALID_REQUEST"
    assert err.value.problem.details == {"problems": [
        "bootstrap keys must be exactly tier4_dir, panel_catalog, "
        "panel_snapshot_id and panel_artifact_root"]}


def test_none_prior_stages_a_candidate_catalog_only(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    (tmp_path / phase5_release.MANIFEST_NAME).unlink(missing_ok=True)
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory)
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    catalog = phase5_release.catalog_path(tmp_path, "cand1")
    assert catalog.exists()
    assert not (tmp_path / phase5_release.MANIFEST_NAME).exists()
    body = json.loads(catalog.read_bytes())
    assert body["sources"]["pool_cutoff"] == "2026-10-01"
    assert body["sources"]["carried_from"] is None
    summary = json.loads((staging / "stage_result.json").read_text())
    expected_members, expected_hashes = _expected_bootstrap_objects(_bootstrap_states())
    assert summary["n_new"] == len(expected_hashes)
    assert summary["n_carried"] == 0
    assert summary["release_id"] == "cand1"
    assert _catalog_contents(tmp_path) == (expected_members, expected_hashes)
    assert _catalog_pairs(tmp_path) == _expected_bootstrap_pairs(_bootstrap_states())


def test_none_prior_refuses_when_a_release_is_deployed(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    (tmp_path / phase5_release.MANIFEST_NAME).unlink(missing_ok=True)
    monkeypatch.setattr(deployment, "current_pointer", lambda root: object())
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a release is already deployed"
    assert err.value.problem.details == {"code": "PRIOR_MISMATCH"}
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


def test_none_prior_refuses_when_pointer_is_unreadable(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    (tmp_path / phase5_release.MANIFEST_NAME).unlink(missing_ok=True)

    def broken(root):
        raise deployment.DeploymentError("boom")

    monkeypatch.setattr(deployment, "current_pointer", broken)
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: prior release could not be read"
    assert err.value.problem.details == {"code": "PRIOR_MISMATCH"}


def test_no_pool_catalog_requires_matching_bindings(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    assert phase5_release.catalog_path(tmp_path, "cand1").exists()
    summary = json.loads((staging / "stage_result.json").read_text())
    expected_members, expected_hashes = _expected_bootstrap_objects(_bootstrap_states())
    assert summary["n_new"] == len(expected_hashes)
    assert _catalog_contents(tmp_path) == (expected_members, expected_hashes)
    assert _catalog_pairs(tmp_path) == _expected_bootstrap_pairs(_bootstrap_states())


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
    assert err.value.problem.details == {
        "code": "EMPTY_MEMBER",
        "members": [
            "chooser_analog_pool", "driver_residual_pool:implied_t1",
            "driver_residual_pool:runup_move", "driver_residual_pool:size",
            "payoff_line:STR-RUNUP", "payoff_line:STR-THRU",
            "payoff_surface:STR-RUNUP", "recalibration_map:STR-RUNUP",
            "recalibration_map:STR-THRU", "tier4_folds:implied_t1",
            "tier4_folds:iv_crush", "tier4_folds:runup_move",
            "tier4_folds:size",
        ],
    }
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()
    assert not phase5_release.staged_manifest_path(
        phase5_release.deployment_root(tmp_path), "cand1").exists()


@pytest.mark.parametrize("member", [
    "paired_residual_pool", "board_analog_matcher", "chooser_analog_pool",
])
def test_pool_object_cutoff_mismatch_refuses(tmp_path, monkeypatch, member):
    states = _bootstrap_states()
    states[f"{member}.json"] = (member, "x|y|2026-10-02")
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path, states=states)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == ("VALIDATION_FAILED: a pool object cutoff differs "
                              "from the session cutoff")
    assert err.value.problem.details == {"members": [member]}


def test_trailing_month_after_cutoff_refuses(tmp_path, monkeypatch):
    states = _bootstrap_states()
    states["trailing_pnl_cutoff.json"] = ("trailing_pnl_cutoff", "hist|2026-11")
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path, states=states)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a trailing object month starts after cutoff"
    assert err.value.problem.details == {"months": ["hist|2026-11"]}


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


def test_id_conflict_on_a_catalog_with_non_dict_sources(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    release, inventory, payloads = _fixture("cand1")
    deployment.stage_release(phase5_release.deployment_root(tmp_path), release,
                             inventory, payloads)
    body = {
        "schema_version": phase5_release.PHASE5_RELEASE_SCHEMA,
        "release_id": "cand1",
        "deployment_id": "d1",
        "members": [],
        "sources": ["not a dict"],
    }
    body["manifest_hash"] = content_hash(body)
    phase5_release.write_catalog(phase5_release.catalog_path(tmp_path, "cand1"), body)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == {"code": "ID_CONFLICT"}


def test_rerun_is_idempotent(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    assert run_state_stage_worker(parameters, staging)["completed_ids"] == [
        "phase5_state_stage"]
    expected_members, expected_hashes = _expected_bootstrap_objects(_bootstrap_states())
    summary = json.loads((staging / "stage_result.json").read_text())
    assert summary["n_new"] == len(expected_hashes)
    assert _catalog_contents(tmp_path) == (expected_members, expected_hashes)
    assert _catalog_pairs(tmp_path) == _expected_bootstrap_pairs(_bootstrap_states())
    first = phase5_release.catalog_path(tmp_path, "cand1").read_bytes()
    run_state_stage_worker(parameters, staging)
    assert phase5_release.catalog_path(tmp_path, "cand1").read_bytes() == first


def test_bootstrap_conflict_restores_original_staging_success(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    first = run_state_stage_worker(parameters, staging)
    assert first["completed_ids"] == ["phase5_state_stage"]

    store = phase5_release.deployment_root(tmp_path)
    status_path = store / "releases" / "cand1" / "staging-status.json"
    original_record = status_path.read_bytes()
    original_manifest = phase5_release.staged_manifest_path(store, "cand1").read_bytes()

    monkeypatch.setattr(
        phase5_prepare_release, "_run_preflights",
        lambda release, states, pinned_panel, sources: (
            {**dict(sources or {}), "panel_copy_sha256": "b" * 64},
            {name: types.SimpleNamespace(digest="dummy")
             for build in states for name in build.payloads},
            {}))

    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == {"code": "ID_CONFLICT"}

    assert status_path.read_bytes() == original_record
    assert phase5_release.staged_manifest_path(store, "cand1").read_bytes() == original_manifest
    from checks.phase5_release_assembly import PRIOR_USABLE, classify_prior
    assert classify_prior(tmp_path, "cand1").kind == PRIOR_USABLE


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
    from tools.phase5_prepare_release import PrepareRefused

    with pytest.raises(PrepareRefused) as err:
        phase5_prepare_release.artifact_file_payloads([path])
    assert str(err.value) == "a bound state file could not be loaded"


def test_artifact_file_payloads_refuses_invalid_tuple_data(tmp_path, monkeypatch):
    from tools.phase5_prepare_release import PrepareRefused

    monkeypatch.setattr(phase5_prepare_release, "MAX_STATE_BYTES", 10)
    path = tmp_path / "payoff.json"

    with pytest.raises(PrepareRefused) as err:
        phase5_prepare_release.artifact_file_payloads([(path, b"x" * 11)])
    assert str(err.value) == "a bound state file could not be loaded"

    with pytest.raises(PrepareRefused) as err:
        phase5_prepare_release.artifact_file_payloads([(path, "not bytes")])
    assert str(err.value) == "a bound state file could not be loaded"


def test_artifact_file_payloads_refuses_a_missing_file(tmp_path):
    from tools.phase5_prepare_release import PrepareRefused

    missing = tmp_path / "missing.json"
    with pytest.raises(PrepareRefused) as err:
        phase5_prepare_release.artifact_file_payloads([missing])
    assert str(err.value) == "a bound state file could not be loaded"


def test_prepare_release_main_refuses_a_missing_bound_state_file(tmp_path, monkeypatch, capsys):
    def broken(args, pinned_panel=None):
        phase5_prepare_release.artifact_file_payloads([tmp_path / "missing.json"])

    monkeypatch.setattr(phase5_prepare_release, "_real_inputs", broken)
    rc = phase5_prepare_release.main(
        ["--release-id", "cand1", "--out", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refused: PrepareRefused: a bound state file could not be loaded" in err


@pytest.mark.parametrize("vary,value", [
    ("model_id", "m2"),
    ("feature_order", ("y",)),
    ("content_hash", "f" * 64),
    ("adapter", "other"),
    ("decision_clock_id", "other"),
    ("strategy_id", "other"),
    ("output_names", ("other",)),
    ("role", "other"),
    ("binding_id", "b2"),
    ("name", "changed-member"),
])
def test_bindings_differ_by_field_refuses(tmp_path, monkeypatch, vary, value):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    import dataclasses

    def broken(inv, resolve=None):
        release, _inventory, payloads = _fixture(
            inv.release_id, deployment_id=inv.deployment_id)
        binding = release.bindings[0]
        if vary == "content_hash":
            member = dataclasses.replace(binding.members[0], content_hash=value)
            binding = dataclasses.replace(binding, members=(member,))
        elif vary == "name":
            member = dataclasses.replace(binding.members[0], name=value)
            binding = dataclasses.replace(binding, members=(member,))
        else:
            binding = dataclasses.replace(binding, **{vary: value})
        release = dataclasses.replace(release, bindings=(binding,))
        return release, payloads

    monkeypatch.setattr(phase5_prepare_release, "model_release", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == ("VALIDATION_FAILED: candidate bindings differ "
                              "from the incumbent's")
    assert err.value.problem.details == {"code": "BINDINGS_DIFFER"}


@pytest.mark.parametrize("key", [
    "tier4_dir", "panel_catalog", "panel_snapshot_id", "panel_artifact_root",
])
def test_bootstrap_plan_spec_hash_changes_with_bootstrap_value(tmp_path, key):
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
    from tools.phase5_state_bootstrap import classify_bound_files
    from tools.phase5_state_stage_job import StageRefused

    real = tmp_path / "real.json"
    real.write_bytes(b'{"schema_version": "paired_residual_pool.v1.0"}')
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(StageRefused) as err:
        classify_bound_files(tmp_path, ["link.json"])
    assert str(err.value) == "a bound state file could not be loaded"


def test_fifo_bound_file_refused_with_fixed_message(tmp_path):
    import os

    from tools.phase5_state_bootstrap import classify_bound_files
    from tools.phase5_state_stage_job import StageRefused

    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(StageRefused) as err:
        classify_bound_files(tmp_path, ["fifo.json"])
    assert str(err.value) == "a bound state file could not be loaded"


def test_oversize_payoff_artifact_refused(tmp_path, monkeypatch):
    from tools.phase5_state_bootstrap import classify_bound_files
    from tools.phase5_state_stage_job import StageRefused
    import tools.phase5_state_bootstrap as bootstrap_module

    monkeypatch.setattr(bootstrap_module, "MAX_STATE_BYTES", 10)
    path = tmp_path / "payoff.json"
    document = {"schema_version": PAYOFF_LINE_ARTIFACT_V1, "extra": "x"}
    path.write_bytes(json.dumps(document).encode())
    with pytest.raises(StageRefused) as err:
        classify_bound_files(tmp_path, ["payoff.json"])
    assert str(err.value) == "a bound state file could not be loaded"


def test_bound_artifacts_classify_and_load(tmp_path):
    from tools.phase5_state_bootstrap import classify_bound_files

    cutoff = "2020-09-01"
    payoff = make_payoff_line_artifact(
        _LINE_FIT, strategy="STR-THRU", driver="abs_move", alpha=0.5, cutoff=cutoff,
    )
    recalibration = build_recalibration_map_artifact(
        _artifact_pairs(), strategy="STR-THRU", alpha=0.5, before=cutoff,
    )
    (tmp_path / "payoff.json").write_bytes(serialize_payoff_artifact(payoff))
    (tmp_path / "recalibration.json").write_bytes(
        serialize_recalibration_artifact(recalibration))
    frozen, artifacts = classify_bound_files(
        tmp_path, ["payoff.json", "recalibration.json"])
    assert frozen == []
    assert [path.name for path, _data in artifacts] == ["payoff.json", "recalibration.json"]
    payloads = phase5_prepare_release.artifact_file_payloads(artifacts)
    assert set(payloads) == {"payoff_line:STR-THRU", "recalibration_map:STR-THRU"}
    expected = "STR-THRU|0.5|2020-09-01"
    assert set(payloads["payoff_line:STR-THRU"]) == {expected}
    assert set(payloads["recalibration_map:STR-THRU"]) == {expected}


def test_admissible_depth_table_classified_frozen(tmp_path):
    from tools.phase5_state_bootstrap import classify_bound_files

    path = tmp_path / "admissible.json"
    path.write_text(json.dumps({"schema_version": "admissible_depth_table.v1.0"}))
    frozen, artifacts = classify_bound_files(tmp_path, ["admissible.json"])
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
    expected_members, expected_hashes = _expected_bootstrap_objects(_bootstrap_states())
    summary = json.loads((staging / "stage_result.json").read_text())
    assert summary["n_new"] == len(expected_hashes)
    assert _catalog_contents(tmp_path) == (expected_members, expected_hashes)
    assert _catalog_pairs(tmp_path) == _expected_bootstrap_pairs(_bootstrap_states())

    calls = []
    original_write = phase5_prepare_release.write_release
    monkeypatch.setattr(phase5_prepare_release, "write_release",
                        lambda *a, **k: calls.append((a, k)) or original_write(*a, **k))
    second = run_state_stage_worker(parameters, staging)
    assert second["completed_ids"] == ["phase5_state_stage"]
    assert calls


def test_inventory_deployment_id_mismatch_refuses(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path)
    import dataclasses

    inventory = dataclasses.replace(inventory, deployment_id="other")
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, inventory=inventory)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == ("VALIDATION_FAILED: candidate bindings differ "
                              "from the incumbent's")
    assert err.value.problem.details == {"code": "BINDINGS_DIFFER"}


def test_tier4_fold_payloads_filters_by_snapshot(tmp_path):
    from tools.phase5_prepare_release import tier4_fold_payloads

    snap = "a" * 64
    matching = tmp_path / f"model1_202610_{snap[:12]}.joblib"
    matching.write_bytes(b"fold-data")
    non_matching = tmp_path / f"model1_202610_{'b' * 12}.joblib"
    non_matching.write_bytes(b"other-data")
    result = tier4_fold_payloads(tmp_path, {"role": "model1"}, snap, None)
    folds = result.get("tier4_folds:role", {})
    assert set(folds.keys()) == {matching.name}
    assert folds[matching.name] == matching.read_bytes()
    assert non_matching.read_bytes() not in folds.values()


def test_id_conflict_on_matching_cutoff_but_different_objects(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    candidate = phase5_release.catalog_path(tmp_path, "cand1")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    obj = {"name": "other", "path": "objects/other",
           "content_hash": "sha256:" + "0" * 64, "bytes": 1}
    body = phase5_release.manifest_body(
        "cand1", "d1", [phase5_release.member_row(
            _spec("paired_residual_pool"), phase5_release.STAGED, [obj], "other")],
        {"pool_cutoff": "2026-10-01", "carried_from": None})
    phase5_release.write_catalog(candidate, body)

    calls = []
    original_write = phase5_prepare_release.write_release
    monkeypatch.setattr(phase5_prepare_release, "write_release",
                        lambda *a, **k: calls.append((a, k)) or original_write(*a, **k))
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == {"code": "ID_CONFLICT"}
    assert calls == []
    assert not phase5_release.staged_manifest_path(
        phase5_release.deployment_root(tmp_path), "cand1").exists()


def test_duplicate_admissible_object_refuses(tmp_path, monkeypatch):
    states = dict(_STATES)
    states["admissible.json"] = ("admissible_table:dyn_sv", "t|1")
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, states=states)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a state object appears twice"
    assert err.value.problem.details == {"member": "admissible_table:dyn_sv"}


def test_staging_success_failure_refuses(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    def broken(*args, **kwargs):
        raise deployment.DeploymentError("secret-internal-detail")

    monkeypatch.setattr(deployment, "mark_staging_succeeded", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: staging success could not be recorded"
    assert err.value.problem.details == {}
    assert "secret-internal-detail" not in str(err.value)


@pytest.mark.parametrize("error", [
    phase5_prepare_release.PrepareRefused("secret-internal-detail"),
    OSError("secret-internal-detail"),
])
def test_write_release_failure_refuses(tmp_path, monkeypatch, error):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(phase5_prepare_release, "write_release", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == {}
    assert "secret-internal-detail" not in str(err.value)


@pytest.mark.parametrize("code,expected_details", [
    ("RELEASE_ID_REUSED", {"code": "ID_CONFLICT"}),
    ("MANIFEST_UNREADABLE", {"code": "ID_CONFLICT"}),
    ("DUPLICATE_BINDING", {}),
])
def test_write_release_staging_refused_maps_codes(tmp_path, monkeypatch, code,
                                                   expected_details):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    def broken(*args, **kwargs):
        raise deployment.StagingRefused((deployment.ReleaseIssue(
            path="$.releases[cand1]", code=code, detail="secret-internal-detail"),))

    monkeypatch.setattr(phase5_prepare_release, "write_release", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == expected_details
    assert "secret-internal-detail" not in str(err.value)


def test_inventory_build_failure_refuses(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    def broken(*args, **kwargs):
        raise OSError("secret-internal-detail")

    monkeypatch.setattr(inventory_module, "current_release_inventory", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: release inventory could not be built"
    assert err.value.problem.details == {}
    assert "secret-internal-detail" not in str(err.value)


def test_readback_not_usable_refuses(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    import checks.phase5_release_assembly as assembly_module

    real = assembly_module.classify_prior

    def fake(root, release_id):
        if release_id == "cand1":
            return types.SimpleNamespace(kind="OTHER", reason="not usable")
        return real(root, release_id)

    monkeypatch.setattr(assembly_module, "classify_prior", fake)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release failed the light checks"
    assert err.value.problem.details == {}


def test_tier4_fold_payload_is_in_catalog(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    (tmp_path / phase5_release.MANIFEST_NAME).unlink(missing_ok=True)
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory)
    monkeypatch.setattr(
        phase5_prepare_release, "tier4_fold_payloads",
        lambda *a, **k: {"tier4_folds:size": {"fold-object.joblib": b"fold-bytes"}})
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    body = json.loads(phase5_release.catalog_path(tmp_path, "cand1").read_bytes())
    row = next(row for row in body["members"]
               if row["member_id"] == "tier4_folds:size")
    entry = next(obj for obj in row["objects"] if obj["name"] == "fold-object.joblib")
    stored = (phase5_release.deployment_root(tmp_path) / entry["path"]).read_bytes()
    assert stored == b"fold-bytes"


def test_pinned_panel_identity_is_recorded_in_catalog(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    (tmp_path / phase5_release.MANIFEST_NAME).unlink(missing_ok=True)
    identity = {"panel_copy_sha256": _SNAPSHOT}
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory)
    fake = types.SimpleNamespace(digest="dummy")
    monkeypatch.setattr(
        phase5_prepare_release, "_run_preflights",
        lambda release, states, pinned_panel, sources: (
            {**sources, **identity},
            {name: fake for build in states for name in build.payloads},
            {}))
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    body = json.loads(phase5_release.catalog_path(tmp_path, "cand1").read_bytes())
    assert body["sources"]["panel_copy_sha256"] == _SNAPSHOT


def test_registry_drift_refuses_without_writing(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    (tmp_path / phase5_release.MANIFEST_NAME).unlink(missing_ok=True)
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory,
        drift=("MISSING_ARTIFACT_FILE",))
    store = phase5_release.deployment_root(tmp_path) / "objects"
    before = sorted(path.name for path in store.iterdir())
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: release inventory drifted"
    assert err.value.problem.details == {"issues": 1}
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()
    assert sorted(path.name for path in store.iterdir()) == before


def test_artifact_payload_bytes_match_bound_file(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    (tmp_path / phase5_release.MANIFEST_NAME).unlink(missing_ok=True)
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory)
    artifact_name = "payoff_line_STR-THRU.json"
    bound = (staging / artifact_name).read_bytes()

    def classify(root, names):
        frozen, artifacts = [], []
        for name in names:
            item = (Path(root) / name, f"state-{name}".encode())
            (artifacts if name == artifact_name else frozen).append(item)
        return frozen, artifacts

    monkeypatch.setattr("tools.phase5_state_bootstrap.classify_bound_files", classify)
    monkeypatch.setattr(phase5_prepare_release, "artifact_file_payloads",
                        lambda files: {"payoff_line:STR-THRU": {"bound.bin": bound}})
    result = run_state_stage_worker(parameters, staging)
    assert result["completed_ids"] == ["phase5_state_stage"]
    body = json.loads(phase5_release.catalog_path(tmp_path, "cand1").read_bytes())
    row = next(row for row in body["members"]
               if row["member_id"] == "payoff_line:STR-THRU")
    entry = next(obj for obj in row["objects"] if obj["name"] == "bound.bin")
    stored = (phase5_release.deployment_root(tmp_path) / entry["path"]).read_bytes()
    assert stored == bound
    assert phase5_release.sha256_bytes(stored) == entry["content_hash"]


def test_frozen_state_replaced_between_classification_and_load(tmp_path, monkeypatch):
    inventory, _body = _stage_prior(tmp_path, prior_names=None)
    (tmp_path / phase5_release.MANIFEST_NAME).unlink(missing_ok=True)
    parameters, staging = _mock_bootstrap_inputs(
        monkeypatch, tmp_path, prior_release_id=None, inventory=inventory)
    states = _bootstrap_states()

    def replaced(paths, max_bytes=None):
        name = Path(paths[0]).name
        member, object_name = states[name]
        return {member: {object_name: b"replaced-bytes"}}

    monkeypatch.setattr(phase5_prepare_release, "frozen_state_payloads", replaced)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: a bound state file could not be loaded"
    assert not phase5_release.catalog_path(tmp_path, "cand1").exists()


@pytest.mark.parametrize("bootstrap", [
    [("tier4_dir", "/t"), ("panel_catalog", "/c"), ("panel_snapshot_id", "s"),
     ("panel_artifact_root", "/a")],
    "bootstrap",
    42,
])
def test_plan_refuses_a_non_mapping_bootstrap_as_invalid_request(tmp_path, bootstrap):
    with pytest.raises(OpsError) as err:
        state_stage_plan(release_root=str(tmp_path), prior_release_id=None,
                         release_id="cand1", cutoff="2026-10-01",
                         state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                         source_root=REPO_ROOT, mode="bootstrap", bootstrap=bootstrap)
    assert err.value.code == "INVALID_REQUEST"
    assert err.value.problem.details == {"problems": [
        "bootstrap must be a mapping with tier4_dir, panel_catalog, "
        "panel_snapshot_id and panel_artifact_root"]}


def test_bootstrap_plan_parameters_decode_with_the_real_decoder(tmp_path):
    from engine.v2.foundation import from_document

    bootstrap = {"tier4_dir": "/t", "panel_catalog": "/c",
                 "panel_snapshot_id": "s", "panel_artifact_root": "/a"}
    plan = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                            release_id="cand1", cutoff="2026-10-01",
                            state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                            source_root=REPO_ROOT, mode="bootstrap", bootstrap=bootstrap)
    decoded = from_document(StateStageParameters,
                            json.loads(json.dumps(plan["parameters"])))
    assert isinstance(decoded, StateStageParameters)
    assert decoded.bootstrap == bootstrap

    carry = state_stage_plan(release_root=str(tmp_path), prior_release_id="prior1",
                             release_id="cand1", cutoff="2026-10-01",
                             state_refs={"a.json": "j#o"}, manifest_ref="artifact_manifest",
                             source_root=REPO_ROOT)
    decoded_carry = from_document(StateStageParameters,
                                  json.loads(json.dumps(carry["parameters"])))
    assert decoded_carry.bootstrap is None


@pytest.mark.parametrize("target_name,attribute,error_name,message", [
    ("inventory", "current_release_inventory", "RegistryError",
     "release inventory could not be built"),
    ("prepare", "fold_policies", "Tier4Error",
     "candidate release could not be prepared"),
    ("prepare", "tier4_fold_payloads", "Tier4Error",
     "candidate release could not be prepared"),
])
def test_registry_and_producer_failures_refuse(tmp_path, monkeypatch, target_name,
                                               attribute, error_name, message):
    from engine.data.features.tier4 import Tier4Error
    from engine.models.registry import RegistryError

    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    target = inventory_module if target_name == "inventory" else phase5_prepare_release
    errors = {"RegistryError": RegistryError, "Tier4Error": Tier4Error}

    def broken(*args, **kwargs):
        raise errors[error_name]("secret-internal-detail")

    monkeypatch.setattr(target, attribute, broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == f"VALIDATION_FAILED: {message}"
    assert err.value.problem.details == {}
    assert "secret-internal-detail" not in str(err.value)


def test_unknown_bound_member_refuses(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()

    def unknown(paths, max_bytes=None):
        path = Path(paths[0])
        return {"unknown_member": {path.name: path.read_bytes()}}

    monkeypatch.setattr(phase5_prepare_release, "frozen_state_payloads", unknown)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: state files name a member outside the catalog"
    assert err.value.problem.details == {"members": ["unknown_member"]}


def test_id_conflict_on_a_different_pool_cutoff(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    run_state_stage_worker(parameters, staging)
    catalog = phase5_release.catalog_path(tmp_path, "cand1")
    body = json.loads(catalog.read_bytes())
    body["sources"]["pool_cutoff"] = "2020-01-01"
    catalog.write_bytes(json.dumps(body).encode())
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == {"code": "ID_CONFLICT"}


def test_bootstrap_conflict_restore_failure_refuses(tmp_path, monkeypatch):
    parameters, staging = _mock_bootstrap_inputs(monkeypatch, tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    first = run_state_stage_worker(parameters, staging)
    assert first["completed_ids"] == ["phase5_state_stage"]

    monkeypatch.setattr(
        phase5_prepare_release, "_run_preflights",
        lambda release, states, pinned_panel, sources: (
            {**dict(sources or {}), "panel_copy_sha256": "b" * 64},
            {name: types.SimpleNamespace(digest="dummy")
             for build in states for name in build.payloads},
            {}))

    def broken(*args, **kwargs):
        raise deployment.DeploymentError("secret-internal-detail")

    monkeypatch.setattr(deployment, "restore_staging_success_record", broken)
    with pytest.raises(OpsError) as err:
        run_state_stage_worker(parameters, staging)
    assert str(err.value) == "VALIDATION_FAILED: candidate release could not be published"
    assert err.value.problem.details == {}
    assert "secret-internal-detail" not in str(err.value)
