"""S3a-2 typed plan and request builders for the retrain chain."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import engine.v2.ops.retrain_candidate as retrain_candidate
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.retrain_candidate import (
    STATE_FILES,
    accept_plan,
    accept_request,
    candidate_id,
    check_plan,
    check_request,
    promote_plan,
    promote_request,
    retrain_cutoff,
    stage_plan,
    stage_request,
    step_key,
    train_plan,
    train_request,
)
from engine.v2.ops.retrain_check import (
    ACCEPTANCE_FILE,
    phase5_accept_job_kind,
    retrain_check_job_kind,
)
from engine.v2.ops.state_stage import state_stage_job_kind
from engine.v2.ops.submission import job_id_for
from engine.v2.ops.training import promote_job_kind, training_job_kind

MONTH = "2026-10"
SESSION = "2026-11-02"
BASE = "rel1"
HASH = "sha256:" + "a" * 64
IMPL = "impl"
ENV = "env"
ROOT = "/release"


def _features():
    plan = build_nightly_plan(Path(__file__).resolve().parents[3], "2026-09-18")
    for request in build_legacy_job_requests(
        plan, tickers=("FAKE",), year_start=2025, year_end=2026,
    ):
        if request.job.kind == "legacy_features":
            job = replace(request.job, input_refs=("legacy-manifest-ref",))
            return replace(request, job=job)
    raise AssertionError("no legacy_features request")


def _train_plan():
    return train_plan(
        month=MONTH, features_request=_features(),
        implementation_ref=IMPL, environment_ref=ENV,
    )


def _chain():
    features = _features()
    train_plan_value = train_plan(
        month=MONTH, features_request=features,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    train = train_request(plan=train_plan_value)
    stage_plan_value = stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, implementation_ref=IMPL, environment_ref=ENV,
    )
    stage = stage_request(plan=stage_plan_value)
    accept_plan_value = accept_plan(
        month=MONTH, session=SESSION, release_root=ROOT, stage_request=stage,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    accept = accept_request(plan=accept_plan_value)
    check_plan_value = check_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        accept_request=accept, implementation_ref=IMPL, environment_ref=ENV,
    )
    check = check_request(plan=check_plan_value)
    promote_plan_value = promote_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        check_request=check, implementation_ref=IMPL, environment_ref=ENV,
    )
    promote = promote_request(plan=promote_plan_value)
    plans = (train_plan_value, stage_plan_value, accept_plan_value,
             check_plan_value, promote_plan_value)
    return features, plans, (train, stage, accept, check, promote)


def _invalid(field, call):
    with pytest.raises(OpsError) as excinfo:
        call()
    assert excinfo.value.problem.code == "INVALID_REQUEST"
    assert excinfo.value.problem.details["field"] == field


def test_candidate_id_and_cutoff_are_canonical():
    assert candidate_id(MONTH, SESSION) == "pools-retrain-2026-10-2026-11-02"
    assert retrain_cutoff(MONTH) == "2026-11-01"
    assert retrain_cutoff("2026-12") == "2027-01-01"
    assert retrain_candidate._safe_id(candidate_id(MONTH, SESSION))


def test_invalid_inputs_are_rejected():
    _invalid("month", lambda: candidate_id("2026-13", SESSION))
    _invalid("session", lambda: candidate_id(MONTH, "2026-02-30"))
    _invalid("month", lambda: retrain_cutoff("2026-1"))
    _invalid("month", lambda: retrain_cutoff("2026-13"))
    _invalid("month", lambda: retrain_cutoff("9999-12"))
    plan = _train_plan()
    _invalid("step", lambda: step_key("bogus", plan))
    _invalid("plan", lambda: step_key("train", None))
    missing = dict(plan)
    missing.pop("kind")
    _invalid("plan", lambda: step_key("train", missing))
    _invalid("spec_hash", lambda: step_key("train", {**plan, "spec_hash": HASH}))
    _invalid("spec_hash", lambda: step_key(
        "train", {**plan, "spec_hash": "sha256:" + "A" * 64},
    ))
    _invalid("base", lambda: stage_plan(
        month=MONTH, session=SESSION, base=".hidden", release_root=ROOT,
        features_request=_features(), implementation_ref=IMPL, environment_ref=ENV,
    ))
    _invalid("base", lambda: stage_plan(
        month=MONTH, session=SESSION, base="a/b", release_root=ROOT,
        features_request=_features(), implementation_ref=IMPL, environment_ref=ENV,
    ))
    _invalid("release_root", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root="  ",
        features_request=_features(), implementation_ref=IMPL, environment_ref=ENV,
    ))
    assert not retrain_candidate._safe_id("")
    assert not retrain_candidate._safe_id(".hidden")
    assert not retrain_candidate._safe_id("a/b")
    assert not retrain_candidate._safe_id("a\\b")
    assert not retrain_candidate._safe_id("a\0b")


def test_step_key_is_canonical():
    plan = _train_plan()
    key = step_key("train", plan)
    assert key.startswith("retrain-train:")
    assert len(key) == len("retrain-train:") + 24
    assert key == step_key("train", plan)
    assert key != step_key("train", plan, parents=("parent-1",))
    changed = {**plan, "parameters": {**plan["parameters"], "month": "2026-11"}}
    changed["spec_hash"] = content_hash(changed["parameters"])
    assert step_key("train", changed) != key


def test_train_plan_uses_monthly_pools_and_legacy_manifest():
    features = _features()
    manifest = features.job.input_refs[0]
    plan = train_plan(
        month=MONTH, features_request=features,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    params = plan["parameters"]
    assert plan["kind"] == "training"
    assert plan["spec_hash"] == content_hash(params)
    assert params["mode"] == "monthly_pools"
    assert params["input_mode"] == "legacy"
    assert params["month"] == MONTH
    assert tuple(params["cutoffs"]) == (retrain_cutoff(MONTH),)
    assert params["input_bindings"] == {"legacy_manifest.json": manifest}
    assert plan["input_refs"] == [manifest]
    assert plan["dependency_job_ids"] == [
        job_id_for(features.namespace, features.idempotency_key)
    ]


def test_stage_plan_binds_train_outputs_and_legacy_manifest():
    features = _features()
    manifest = features.job.input_refs[0]
    train_plan_value = train_plan(
        month=MONTH, features_request=features,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    train_id = job_id_for("shadow", step_key("train", train_plan_value))
    plan = stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, implementation_ref=IMPL, environment_ref=ENV,
    )
    params = plan["parameters"]
    refs = {name: f"job_{train_id}#training/{name}" for name in STATE_FILES}
    assert plan["kind"] == "phase5_state_stage"
    assert plan["spec_hash"] == content_hash(params)
    assert tuple(params["expected_ids"]) == ("phase5_state_stage",)
    assert params["input_bindings"] == {**refs, "legacy_manifest.json": manifest}
    assert set(params["state_files"]) == set(STATE_FILES)
    assert params["mode"] == "monthly"
    assert params["cutoff"] == retrain_cutoff(MONTH)
    assert params["prior_release_id"] == BASE
    assert params["release_id"] == candidate_id(MONTH, SESSION)
    assert plan["input_refs"] == [refs[name] for name in sorted(refs)] + [manifest]
    assert plan["dependency_job_ids"] == [train_id]


def test_wrong_upstream_kinds_are_rejected():
    features, _plans, requests = _chain()
    _train, stage, accept, check, _promote = requests
    wrong_features = replace(features, job=replace(features.job, kind="training"))
    _invalid("features_request", lambda: train_plan(
        month=MONTH, features_request=wrong_features,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_stage = replace(stage, job=replace(stage.job, kind="training"))
    _invalid("stage_request", lambda: accept_plan(
        month=MONTH, session=SESSION, release_root=ROOT, stage_request=wrong_stage,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_accept = replace(accept, job=replace(accept.job, kind="training"))
    _invalid("accept_request", lambda: check_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        accept_request=wrong_accept, implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_check = replace(check, job=replace(check.job, kind="training"))
    _invalid("check_request", lambda: promote_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        check_request=wrong_check, implementation_ref=IMPL, environment_ref=ENV,
    ))


def test_all_five_requests_chain_and_bind_artifacts():
    features, plans, requests = _chain()
    train, stage, accept, check, promote = requests
    feature_id = job_id_for(features.namespace, features.idempotency_key)
    stage_id = job_id_for("shadow", stage.idempotency_key)
    accept_id = job_id_for("shadow", accept.idempotency_key)
    check_id = job_id_for("shadow", check.idempotency_key)

    assert [r.job.kind for r in requests] == [
        "training", "phase5_state_stage", "phase5_accept", "retrain_check", "models_promote",
    ]
    assert [r.idempotency_key for r in requests] == [
        step_key(step, plan)
        for step, plan in zip(("train", "stage", "accept", "check", "promote"), plans)
    ]
    for request, plan in zip(requests, plans):
        assert len(request.idempotency_key) <= 128
        assert plan["spec_hash"] == content_hash(plan["parameters"])
        for ref in request.job.input_refs:
            assert not ref.startswith("job_")
            assert "#" not in ref
            assert "/" not in ref
    assert any(ref.startswith("job_") for ref in plans[1]["input_refs"])
    assert any(ref.startswith("job_") for ref in plans[3]["input_refs"])
    assert train.job.dependency_job_ids == (feature_id,)
    assert job_id_for("shadow", train.idempotency_key) in stage.job.dependency_job_ids
    assert stage_id in accept.job.dependency_job_ids
    assert accept_id in check.job.dependency_job_ids
    assert check_id in promote.job.dependency_job_ids
    assert check.job.parameters["input_bindings"] == {
        ACCEPTANCE_FILE: f"job_{accept_id}#acceptance"
    }
    assert promote.job.parameters["expected_previous_release_id"] == BASE
    assert promote.job.parameters["release_id"] == candidate_id(MONTH, SESSION)
    assert retrain_candidate._safe_id(candidate_id(MONTH, SESSION))

    kinds = (training_job_kind, state_stage_job_kind, phase5_accept_job_kind,
             retrain_check_job_kind, promote_job_kind)
    for request, plan, kind_fn in zip(requests, plans, kinds):
        kind = kind_fn()
        assert request.job.kind == kind.name
        assert request.namespace == "shadow"
        assert request.job.output_namespace == "shadow"
        assert request.job.implementation_ref == IMPL
        assert request.job.environment_ref == ENV
        assert request.job.resource_class == plan["resource_class"]
        assert request.job.spec_hash == plan["spec_hash"]
        assert request.job.retry_policy_ref == kind.retry.name
        assert request.job.checkpoint_contract_ref == kind.checkpoint_contract


def test_request_parents_are_merged_and_deduplicated():
    features = _features()
    plan = train_plan(month=MONTH, features_request=features,
                      implementation_ref=IMPL, environment_ref=ENV)
    feature_id = job_id_for(features.namespace, features.idempotency_key)
    request = train_request(
        plan=plan, parents=(feature_id, "parent-2"),
    )
    assert request.job.dependency_job_ids == (feature_id, "parent-2")
    assert request.idempotency_key != train_request(plan=plan).idempotency_key


def test_d1_pool_only_semantics():
    assert STATE_FILES == (
        "driver_residual_pool__size.json",
        "driver_residual_pool__implied_t1.json",
        "driver_residual_pool__runup_move.json",
        "chooser_analog_pool.json",
    )
    features = _features()
    train_plan_value = train_plan(
        month=MONTH, features_request=features,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    train_params = train_plan_value["parameters"]
    assert train_params["mode"] == "monthly_pools"
    assert train_params["month"] == MONTH
    assert tuple(train_params["cutoffs"]) == (retrain_cutoff(MONTH),)
    stage_plan_value = stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, implementation_ref=IMPL, environment_ref=ENV,
    )
    bindings = stage_plan_value["parameters"]["input_bindings"]
    assert set(bindings) == set(STATE_FILES) | {"legacy_manifest.json"}
    assert all(bindings[name].endswith(f"#training/{name}") for name in STATE_FILES)
    doc = retrain_candidate.__doc__
    assert "re-frozen" in doc
    assert "champion-carried" in doc
    assert "dated" in doc
    assert retrain_candidate._safe_id(candidate_id(MONTH, SESSION))
