"""S3a-2 typed plan and request builders for the retrain chain."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import engine.v2.ops.retrain_candidate as retrain_candidate
from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.retrain_check import ACCEPTANCE_FILE
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
from engine.v2.ops.submission import job_id_for

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
            job = replace(request.job, input_refs=("artifact://legacy-manifest",))
            return replace(request, job=job)
    raise AssertionError("no legacy_features request")


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
    _invalid("step", lambda: step_key("bogus", SESSION, MONTH, BASE, HASH))
    _invalid("session", lambda: step_key("train", "2026-13-01", MONTH, BASE, HASH))
    _invalid("month", lambda: step_key("train", SESSION, "2026-13", BASE, HASH))
    _invalid("base", lambda: step_key("train", SESSION, MONTH, ".hidden", HASH))
    _invalid("base", lambda: step_key("train", SESSION, MONTH, "a/b", HASH))
    _invalid("spec_hash", lambda: step_key("train", SESSION, MONTH, BASE, "sha256:" + "A" * 64))
    _invalid("spec_hash", lambda: step_key("train", SESSION, MONTH, BASE, "sha256:" + "a" * 63))
    assert not retrain_candidate._safe_id("")
    assert not retrain_candidate._safe_id(".hidden")
    assert not retrain_candidate._safe_id("a/b")
    assert not retrain_candidate._safe_id("a\\b")
    assert not retrain_candidate._safe_id("a\0b")


def test_step_key_is_canonical():
    assert step_key("train", SESSION, MONTH, BASE, HASH) == (
        "retrain-train:2026-11-02:2026-10:rel1:" + HASH
    )


def test_train_plan_uses_monthly_pools_and_legacy_manifest():
    features = _features()
    manifest = features.job.input_refs[0]
    plan = train_plan(
        month=MONTH, features_request=features,
        implementation_ref=IMPL, environment_ref=ENV,
    )
    params = plan["parameters"]
    assert plan["kind"] == "training"
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
    train_id = job_id_for(
        "shadow", step_key("train", SESSION, MONTH, BASE, train_plan_value["spec_hash"]),
    )
    plan = stage_plan(
        month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
        features_request=features, implementation_ref=IMPL, environment_ref=ENV,
    )
    params = plan["parameters"]
    refs = {name: f"job_{train_id}#training/{name}" for name in STATE_FILES}
    assert plan["kind"] == "phase5_state_stage"
    assert tuple(params["expected_ids"]) == ("phase5_state_stage",)
    assert params["input_bindings"] == {**refs, "legacy_manifest.json": manifest}
    assert set(params["state_files"]) == set(STATE_FILES)
    assert params["mode"] == "monthly"
    assert params["cutoff"] == retrain_cutoff(MONTH)
    assert params["prior_release_id"] == BASE
    assert params["release_id"] == candidate_id(MONTH, SESSION)
    assert plan["input_refs"] == [refs[name] for name in sorted(refs)] + [manifest]
    assert plan["dependency_job_ids"] == [train_id]


def test_all_five_requests_chain_and_bind_artifacts():
    features = _features()
    feature_id = job_id_for(features.namespace, features.idempotency_key)

    train = train_request(
        plan=train_plan(month=MONTH, features_request=features,
                        implementation_ref=IMPL, environment_ref=ENV),
        session=SESSION, month=MONTH, base=BASE,
    )
    stage = stage_request(
        plan=stage_plan(month=MONTH, session=SESSION, base=BASE,
                        release_root=ROOT, features_request=features,
                        implementation_ref=IMPL, environment_ref=ENV),
        session=SESSION, month=MONTH, base=BASE,
    )
    stage_id = job_id_for("shadow", stage.idempotency_key)
    accept = accept_request(
        plan=accept_plan(month=MONTH, session=SESSION, release_root=ROOT,
                         stage_request=stage, implementation_ref=IMPL,
                         environment_ref=ENV),
        session=SESSION, month=MONTH, base=BASE,
    )
    accept_id = job_id_for("shadow", accept.idempotency_key)
    check = check_request(
        plan=check_plan(month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
                        accept_request=accept, implementation_ref=IMPL,
                        environment_ref=ENV),
        session=SESSION, month=MONTH, base=BASE,
    )
    check_id = job_id_for("shadow", check.idempotency_key)
    promote = promote_request(
        plan=promote_plan(month=MONTH, session=SESSION, base=BASE, release_root=ROOT,
                          check_request=check, implementation_ref=IMPL,
                          environment_ref=ENV),
        session=SESSION, month=MONTH, base=BASE,
    )

    requests = (train, stage, accept, check, promote)
    assert [r.job.kind for r in requests] == [
        "training", "phase5_state_stage", "phase5_accept", "retrain_check", "models_promote",
    ]
    assert [r.idempotency_key for r in requests] == [
        step_key(step, SESSION, MONTH, BASE, r.job.spec_hash)
        for step, r in zip(("train", "stage", "accept", "check", "promote"), requests)
    ]
    assert train.job.dependency_job_ids == (feature_id,)
    assert job_id_for("shadow", train.idempotency_key) in stage.job.dependency_job_ids
    assert stage_id in accept.job.dependency_job_ids
    assert accept_id in check.job.dependency_job_ids
    assert check_id in promote.job.dependency_job_ids
    assert check.job.parameters["input_bindings"] == {
        ACCEPTANCE_FILE: f"job_{accept_id}#{ACCEPTANCE_FILE}"
    }
    assert promote.job.parameters["expected_previous_release_id"] == BASE
    assert promote.job.parameters["release_id"] == candidate_id(MONTH, SESSION)
    assert retrain_candidate._safe_id(candidate_id(MONTH, SESSION))


def test_request_parents_are_merged_and_deduplicated():
    features = _features()
    plan = train_plan(month=MONTH, features_request=features,
                      implementation_ref=IMPL, environment_ref=ENV)
    feature_id = job_id_for(features.namespace, features.idempotency_key)
    request = train_request(
        plan=plan, session=SESSION, month=MONTH, base=BASE,
        parents=(feature_id, "parent-2"),
    )
    assert request.job.dependency_job_ids == (feature_id, "parent-2")


def test_d1_pool_only_semantics():
    assert STATE_FILES == (
        "driver_residual_pool__size.json",
        "driver_residual_pool__implied_t1.json",
        "driver_residual_pool__runup_move.json",
        "chooser_analog_pool.json",
    )
    doc = retrain_candidate.__doc__
    assert "re-frozen" in doc
    assert "champion-carried" in doc
    assert "dated" in doc
