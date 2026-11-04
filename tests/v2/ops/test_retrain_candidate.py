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


def _restage():
    base = _features()
    job = replace(
        base.job,
        kind="retrain_step",
        parameters={"step": "restage"},
        input_refs=(),
        dependency_job_ids=(),
    )
    return replace(base, namespace="shadow", idempotency_key="restage-key", job=job)


def _train_plan():
    return train_plan(
        month=MONTH, features_request=_features(),
        implementation_ref=IMPL, environment_ref=ENV,
    )


def _train_request():
    return train_request(plan=_train_plan())


def _chain():
    features = _features()
    restage = _restage()
    train_plan_value = train_plan(
        month=MONTH, features_request=features,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    train = train_request(plan=train_plan_value, parents=(restage,))
    stage_plan_value = stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=train,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    stage = stage_request(plan=stage_plan_value)
    accept_plan_value = accept_plan(
        month=MONTH, session=SESSION, release_root=ROOT, stage_request=stage,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    accept = accept_request(plan=accept_plan_value)
    check_plan_value = check_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        stage_request=stage, accept_request=accept,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    check = check_request(plan=check_plan_value)
    promote_plan_value = promote_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        stage_request=stage, accept_request=accept, check_request=check,
        implementation_ref=IMPL, environment_ref=ENV,
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
    malformed_plan = {**plan, "dependency_job_ids": None}
    _invalid("plan", lambda: step_key("train", malformed_plan))
    _invalid("spec_hash", lambda: step_key("train", {**plan, "spec_hash": HASH}))
    _invalid("spec_hash", lambda: step_key(
        "train", {**plan, "spec_hash": "sha256:" + "A" * 64},
    ))
    _invalid("spec_hash", lambda: step_key(
        "train", {**plan, "parameters": {"bad": object()}},
    ))
    _invalid("base", lambda: stage_plan(
        month=MONTH, session=SESSION, base=".hidden", release_root=ROOT,
        features_request=_features(), training_request=_train_request(),
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    _invalid("base", lambda: stage_plan(
        month=MONTH, session=SESSION, base="a/b", release_root=ROOT,
        features_request=_features(), training_request=_train_request(),
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    _invalid("release_root", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root="  ",
        features_request=_features(), training_request=_train_request(),
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    _invalid("release_root", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root="release",
        features_request=_features(), training_request=_train_request(),
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    assert not retrain_candidate._safe_id("")
    assert not retrain_candidate._safe_id(".hidden")
    assert not retrain_candidate._safe_id("a/b")
    assert not retrain_candidate._safe_id("a\\b")
    assert not retrain_candidate._safe_id("a\0b")
    assert not retrain_candidate._safe_root("release")
    assert not retrain_candidate._safe_root("  ")
    assert not retrain_candidate._safe_root("/release\0")
    assert retrain_candidate._safe_root(ROOT)


def test_step_key_is_canonical():
    plan = _train_plan()
    key = step_key("train", plan)
    assert key.startswith("retrain-train:")
    assert len(key) == len("retrain-train:") + 24
    assert key == step_key("train", plan)
    fingerprint = {
        "kind": plan["kind"],
        "spec_hash": plan["spec_hash"],
        "parameters": plan["parameters"],
        "implementation_ref": plan["implementation_ref"],
        "environment_ref": plan["environment_ref"],
        "resource_class": plan["resource_class"],
        "input_refs": list(plan["input_refs"]),
        "dependency_job_ids": list(plan["dependency_job_ids"]),
    }
    expected = "retrain-train:" + content_hash(fingerprint).split(":", 1)[1][:24]
    assert key == expected
    changed = {**plan, "parameters": {**plan["parameters"], "month": "2026-11"}}
    changed["spec_hash"] = content_hash(changed["parameters"])
    assert step_key("train", changed) != key
    for field, value in (
        ("implementation_ref", "impl2"),
        ("environment_ref", "env2"),
        ("resource_class", "other"),
        ("input_refs", ["other"]),
        ("dependency_job_ids", ["other"]),
    ):
        assert step_key("train", {**plan, field: value}) != key
    assert step_key("train", plan, parents=("parent-1",)) != key


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
    restage = _restage()
    train_plan_value = train_plan(
        month=MONTH, features_request=features,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    train = train_request(plan=train_plan_value, parents=(restage,))
    train_id = job_id_for(train.namespace, train.idempotency_key)
    feature_id = job_id_for(features.namespace, features.idempotency_key)
    assert feature_id in train.job.dependency_job_ids
    assert train_id != job_id_for("shadow", step_key("train", train_plan_value))
    plan = stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=train,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    params = plan["parameters"]
    refs = {name: f"{train_id}#training/{name}" for name in STATE_FILES}
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
    wrong_train = replace(
        _train_request(), job=replace(_train_request().job, kind="phase5_state_stage"),
    )
    _invalid("training_request", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=wrong_train,
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
        stage_request=stage, accept_request=wrong_accept,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_check = replace(check, job=replace(check.job, kind="training"))
    _invalid("check_request", lambda: promote_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        stage_request=stage, accept_request=accept, check_request=wrong_check,
        implementation_ref=IMPL, environment_ref=ENV,
    ))


def test_wrong_candidates_and_equal_base_are_rejected():
    _features_, _plans, requests = _chain()
    _train, stage, accept, check, _promote = requests
    candidate = candidate_id(MONTH, SESSION)
    wrong_stage = replace(stage, job=replace(
        stage.job, parameters={**stage.job.parameters, "release_id": "other"},
    ))
    _invalid("stage_request", lambda: accept_plan(
        month=MONTH, session=SESSION, release_root=ROOT, stage_request=wrong_stage,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_mode = replace(stage, job=replace(
        stage.job, parameters={**stage.job.parameters, "mode": "daily"},
    ))
    _invalid("stage_request", lambda: accept_plan(
        month=MONTH, session=SESSION, release_root=ROOT, stage_request=wrong_mode,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_accept = replace(accept, job=replace(
        accept.job, parameters={**accept.job.parameters, "candidate_release_id": "other"},
    ))
    _invalid("accept_request", lambda: check_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        stage_request=stage, accept_request=wrong_accept,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_check = replace(check, job=replace(
        check.job, parameters={**check.job.parameters, "candidate_release_id": "other"},
    ))
    _invalid("check_request", lambda: promote_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        stage_request=stage, accept_request=accept, check_request=wrong_check,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_incumbent = replace(check, job=replace(
        check.job, parameters={**check.job.parameters, "incumbent_release_id": "other"},
    ))
    _invalid("check_request", lambda: promote_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        stage_request=stage, accept_request=accept, check_request=wrong_incumbent,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    _invalid("base", lambda: check_plan(
        month=MONTH, session=SESSION, base=candidate, release_root=ROOT,
        stage_request=stage, accept_request=accept,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    _invalid("base", lambda: promote_plan(
        month=MONTH, session=SESSION, base=candidate, release_root=ROOT,
        stage_request=stage, accept_request=accept, check_request=check,
        implementation_ref=IMPL, environment_ref=ENV,
    ))


def test_stage_plan_rejects_equal_base_and_mismatched_training_bindings():
    features = _features()
    candidate = candidate_id(MONTH, SESSION)
    _invalid("base", lambda: stage_plan(
        month=MONTH, session=SESSION, base=candidate, release_root=ROOT,
        features_request=features, training_request=_train_request(),
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    train = _train_request()
    wrong_cutoffs = replace(train, job=replace(
        train.job, parameters={**train.job.parameters, "cutoffs": ["2026-12-01"]},
    ))
    _invalid("training_request", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=wrong_cutoffs,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_bindings = replace(train, job=replace(
        train.job, parameters={
            **train.job.parameters,
            "input_bindings": {"legacy_manifest.json": "other"},
        },
    ))
    _invalid("training_request", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=wrong_bindings,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_inputs = replace(train, job=replace(
        train.job, input_refs=("other-manifest",),
    ))
    _invalid("training_request", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=wrong_inputs,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    wrong_output = replace(train, job=replace(
        train.job, output_namespace="other",
    ))
    _invalid("training_request", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=wrong_output,
        implementation_ref=IMPL, environment_ref=ENV,
    ))
    feature_id = job_id_for(features.namespace, features.idempotency_key)
    wrong_deps = replace(train, job=replace(
        train.job,
        dependency_job_ids=tuple(
            dep for dep in train.job.dependency_job_ids if dep != feature_id
        ),
    ))
    _invalid("training_request", lambda: stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=wrong_deps,
        implementation_ref=IMPL, environment_ref=ENV,
    ))


def test_stage_request_mismatches_are_rejected_in_check_and_promote():
    _features_, _plans, requests = _chain()
    _train, stage, accept, check, _promote = requests
    wrong_prior = replace(stage, job=replace(
        stage.job, parameters={**stage.job.parameters, "prior_release_id": "other"},
    ))
    wrong_root = replace(stage, job=replace(
        stage.job, parameters={**stage.job.parameters, "release_root": "/other"},
    ))
    wrong_namespace = replace(stage, namespace="other")
    wrong_mode = replace(stage, job=replace(
        stage.job, parameters={**stage.job.parameters, "mode": "daily"},
    ))
    for wrong in (wrong_prior, wrong_root, wrong_namespace, wrong_mode):
        _invalid("stage_request", lambda wrong=wrong: check_plan(
            month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
            stage_request=wrong, accept_request=accept,
            implementation_ref=IMPL, environment_ref=ENV,
        ))
        _invalid("stage_request", lambda wrong=wrong: promote_plan(
            month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
            stage_request=wrong, accept_request=accept, check_request=check,
            implementation_ref=IMPL, environment_ref=ENV,
        ))


def test_upstream_request_root_namespace_and_dependency_mismatches_are_rejected():
    _features_, _plans, requests = _chain()
    _train, stage, accept, check, _promote = requests
    wrong_accept_root = replace(accept, job=replace(
        accept.job, parameters={**accept.job.parameters, "release_root": "/other"},
    ))
    wrong_accept_namespace = replace(accept, namespace="other")
    wrong_accept_deps = replace(accept, job=replace(
        accept.job, dependency_job_ids=(),
    ))
    wrong_accept_output = replace(accept, job=replace(
        accept.job, output_namespace="other",
    ))
    for wrong in (wrong_accept_root, wrong_accept_namespace, wrong_accept_deps,
                  wrong_accept_output):
        _invalid("accept_request", lambda wrong=wrong: check_plan(
            month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
            stage_request=stage, accept_request=wrong,
            implementation_ref=IMPL, environment_ref=ENV,
        ))
    for wrong in (wrong_accept_deps, wrong_accept_namespace, wrong_accept_output):
        _invalid("accept_request", lambda wrong=wrong: promote_plan(
            month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
            stage_request=stage, accept_request=wrong, check_request=check,
            implementation_ref=IMPL, environment_ref=ENV,
        ))
    wrong_check_root = replace(check, job=replace(
        check.job, parameters={**check.job.parameters, "release_root": "/other"},
    ))
    wrong_check_namespace = replace(check, namespace="other")
    wrong_check_output = replace(check, job=replace(
        check.job, output_namespace="other",
    ))
    wrong_check_deps = replace(check, job=replace(
        check.job, dependency_job_ids=("other",),
    ))
    for wrong in (wrong_check_root, wrong_check_namespace, wrong_check_output,
                  wrong_check_deps):
        _invalid("check_request", lambda wrong=wrong: promote_plan(
            month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
            stage_request=stage, accept_request=accept, check_request=wrong,
            implementation_ref=IMPL, environment_ref=ENV,
        ))


def test_all_five_requests_chain_and_bind_artifacts():
    features, plans, requests = _chain()
    train, stage, accept, check, promote = requests
    restage = _restage()
    manifest = features.job.input_refs[0]
    feature_id = job_id_for(features.namespace, features.idempotency_key)
    restage_id = job_id_for(restage.namespace, restage.idempotency_key)
    train_id = job_id_for("shadow", train.idempotency_key)
    stage_id = job_id_for("shadow", stage.idempotency_key)
    accept_id = job_id_for("shadow", accept.idempotency_key)
    check_id = job_id_for("shadow", check.idempotency_key)

    assert [r.job.kind for r in requests] == [
        "training", "phase5_state_stage", "phase5_accept", "retrain_check", "models_promote",
    ]
    assert [r.idempotency_key for r in requests] == [
        step_key("train", plans[0], (restage_id,)),
        step_key("stage", plans[1]),
        step_key("accept", plans[2]),
        step_key("check", plans[3]),
        step_key("promote", plans[4]),
    ]
    assert [plan["mode"] for plan in plans] == ["shadow"] * len(plans)
    assert [tuple(plan["parameters"]["expected_ids"]) for plan in plans] == [
        ("training",),
        ("phase5_state_stage",),
        ("phase5_accept",),
        ("retrain_check",),
        ("models_promote",),
    ]
    for request, plan in zip(requests, plans):
        assert len(request.idempotency_key) <= 128
        assert request.principal == "operator"
        assert plan["spec_hash"] == content_hash(plan["parameters"])
        for ref in request.job.input_refs:
            assert not ref.startswith("job_")
            assert "#" not in ref
            assert "/" not in ref
    assert train.job.dependency_job_ids == (feature_id, restage_id)
    assert stage.job.dependency_job_ids == (train_id,)
    assert accept.job.dependency_job_ids == (stage_id,)
    assert check.job.dependency_job_ids == (accept_id,)
    assert promote.job.dependency_job_ids == (check_id,)
    stage_params = stage.job.parameters
    assert stage.namespace == "shadow"
    assert stage_params["release_id"] == candidate_id(MONTH, SESSION)
    assert stage_params["prior_release_id"] == BASE
    assert stage_params["release_root"] == ROOT
    assert stage_params["mode"] == "monthly"
    for request in (accept, check, promote):
        assert request.namespace == "shadow"
        assert request.job.parameters["release_root"] == ROOT
    assert accept.job.parameters["candidate_release_id"] == stage_params["release_id"]
    assert check.job.parameters["candidate_release_id"] == stage_params["release_id"]
    assert check.job.parameters["incumbent_release_id"] == stage_params["prior_release_id"]
    assert promote.job.parameters["release_id"] == stage_params["release_id"]
    assert promote.job.parameters["expected_previous_release_id"] == (
        stage_params["prior_release_id"]
    )
    assert tuple(train.job.input_refs) == (manifest,)
    assert tuple(stage.job.input_refs) == (manifest,)
    assert tuple(check.job.input_refs) == ()
    assert any(ref.startswith("job_") for ref in plans[1]["input_refs"])
    assert any(ref.startswith("job_") for ref in plans[3]["input_refs"])
    assert stage.job.parameters["input_bindings"] == {
        **{name: f"{train_id}#training/{name}" for name in STATE_FILES},
        "legacy_manifest.json": manifest,
    }
    assert check.job.parameters["input_bindings"] == {
        ACCEPTANCE_FILE: f"{accept_id}#acceptance"
    }
    assert promote.job.parameters["expected_previous_release_id"] == BASE
    assert promote.job.parameters["release_id"] == candidate_id(MONTH, SESSION)
    assert plans[4]["kind"] == "models_promote"
    assert promote.job.kind == promote_job_kind().name == plans[4]["kind"]
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


def test_request_parameters_are_detached_from_plan():
    _features_, plans, requests = _chain()
    train = requests[0]
    original = train.job.parameters["month"]
    plans[0]["parameters"]["month"] = "1999-01"
    assert train.job.parameters["month"] == original
    original_cutoffs = tuple(train.job.parameters["cutoffs"])
    cutoffs = plans[0]["parameters"]["cutoffs"]
    if isinstance(cutoffs, list):
        cutoffs[0] = "1999-01"
    else:
        plans[0]["parameters"]["cutoffs"] = ("1999-01",)
    assert tuple(train.job.parameters["cutoffs"]) == original_cutoffs


def test_request_parents_are_merged_and_deduplicated():
    features = _features()
    plan = train_plan(month=MONTH, features_request=features,
                      implementation_ref=IMPL, environment_ref=ENV)
    feature_id = job_id_for(features.namespace, features.idempotency_key)
    restage = _restage()
    restage_id = job_id_for(restage.namespace, restage.idempotency_key)
    request = train_request(plan=plan, parents=(restage, restage))
    assert request.job.dependency_job_ids == (feature_id, restage_id)
    assert request.idempotency_key != train_request(plan=plan).idempotency_key
    assert request.idempotency_key == train_request(
        plan=plan, parents=(restage,),
    ).idempotency_key
    _invalid("parents", lambda: train_request(plan=plan, parents=("parent-2",)))
    wrong_restage = replace(restage, job=replace(restage.job, kind="training"))
    _invalid("parents", lambda: train_request(plan=plan, parents=(wrong_restage,)))
    _invalid("parents", lambda: stage_request(plan=_chain()[1][1], parents=(restage,)))


def test_request_rejects_unsubmittable_plans():
    plan = _train_plan()
    for field, value in (
        ("schema_version", "operations_plan.v2.0"),
        ("mode", "apply"),
        ("blocked_prerequisites", ["missing"]),
        ("effects", ["applied"]),
    ):
        _invalid("plan", lambda field=field, value=value: train_request(
            plan={**plan, field: value},
        ))
    _invalid("plan", lambda: train_request(
        plan={**plan, "kind": "phase5_state_stage"},
    ))
    _invalid("input_refs", lambda: train_request(
        plan={**plan, "input_refs": [1]},
    ))
    del_plan = {**plan}
    del del_plan["input_refs"]
    _invalid("input_refs", lambda: train_request(plan=del_plan))


def test_manifest_ref_rejects_job_output_ref():
    features = _features()
    wrong = replace(features, job=replace(features.job, input_refs=("job_abc",)))
    _invalid("features_request", lambda: train_plan(
        month=MONTH, features_request=wrong,
        implementation_ref=IMPL, environment_ref=ENV,
    ))


def test_request_parents_generator_matches_tuple():
    features = _features()
    plan = train_plan(month=MONTH, features_request=features,
                      implementation_ref=IMPL, environment_ref=ENV)
    feature_id = job_id_for(features.namespace, features.idempotency_key)
    restage = _restage()
    restage_id = job_id_for(restage.namespace, restage.idempotency_key)
    from_generator = train_request(plan=plan, parents=(p for p in (restage,)))
    from_tuple = train_request(plan=plan, parents=(restage,))
    assert from_generator.job.dependency_job_ids == (feature_id, restage_id)
    assert from_generator.job.dependency_job_ids == from_tuple.job.dependency_job_ids
    assert from_generator.idempotency_key == from_tuple.idempotency_key
    _invalid("parents", lambda: train_request(plan=plan, parents=None))


def test_promote_rejects_check_with_wrong_acceptance_binding():
    _features_, _plans, requests = _chain()
    _train, stage, accept, check, _promote = requests
    wrong_binding = replace(check, job=replace(
        check.job, parameters={
            **check.job.parameters,
            "input_bindings": {ACCEPTANCE_FILE: "job_other#acceptance"},
        },
    ))
    _invalid("check_request", lambda: promote_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        stage_request=stage, accept_request=accept, check_request=wrong_binding,
        implementation_ref=IMPL, environment_ref=ENV,
    ))


def test_d1_pool_only_semantics():
    assert STATE_FILES == (
        "driver_residual_pool__size.json",
        "driver_residual_pool__implied_t1.json",
        "driver_residual_pool__runup_move.json",
        "chooser_analog_pool.json",
    )
    features = _features()
    manifest = features.job.input_refs[0]
    train_plan_value = train_plan(
        month=MONTH, features_request=features,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    train_params = train_plan_value["parameters"]
    assert train_params["mode"] == "monthly_pools"
    assert train_params["month"] == MONTH
    assert tuple(train_params["cutoffs"]) == (retrain_cutoff(MONTH),)
    assert train_params["input_bindings"] == {"legacy_manifest.json": manifest}
    restage = _restage()
    train = train_request(plan=train_plan_value, parents=(restage,))
    train_id = job_id_for(train.namespace, train.idempotency_key)
    stage_plan_value = stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, training_request=train,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    bindings = stage_plan_value["parameters"]["input_bindings"]
    expected = {name: f"{train_id}#training/{name}" for name in STATE_FILES}
    expected["legacy_manifest.json"] = manifest
    assert bindings == expected
    assert tuple(stage_plan_value["parameters"]["state_files"]) == tuple(sorted(STATE_FILES))
    assert stage_plan_value["parameters"]["cutoff"] == retrain_cutoff(MONTH)
    assert retrain_candidate._safe_id(candidate_id(MONTH, SESSION))
