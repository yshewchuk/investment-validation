"""Pure candidate identity, cutoff, and typed step plan/request builders for the monthly retrain chain.

The three driver pools are S1 re-frozen and champion-carried; the chooser pool is dated.
Candidate identity and cutoff derive deterministically, never inspecting state.
"""
import re
from datetime import date, timedelta
from pathlib import Path

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import content_hash, from_document, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.state_stage import (
    StateStageParameters,
    state_stage_job_kind,
    state_stage_parameter_problems,
)
from engine.v2.ops.submission import job_id_for
from engine.v2.ops.training import (
    TrainingParameters,
    training_job_kind,
    training_parameter_problems,
)

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_SESSION = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


def _safe_id(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and not value.startswith(".")
            and not any(char in value for char in ("/", "\\", "\0")))


def _safe_root(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and "\0" not in value
            and Path(value).is_absolute())


def _need(condition, field):
    if not condition:
        raise fail("INVALID_REQUEST", "invalid retrain builder input", details={"field": field})


def _date_ok(value: object, pattern: re.Pattern[str], suffix: str = "") -> bool:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        return False
    text = value + suffix
    try:
        return date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False


def candidate_id(month: str, session: str) -> str:
    _need(_date_ok(month, _MONTH, "-01"), "month")
    _need(_date_ok(session, _SESSION), "session")
    result = f"pools-retrain-{month}-{session}"
    _need(_safe_id(result), "candidate_id")
    return result


def retrain_cutoff(month: str) -> str:
    _need(_date_ok(month, _MONTH, "-01"), "month")
    start = date(int(month[:4]), int(month[5:]), 1)
    try:
        return (start + timedelta(days=32)).replace(day=1).isoformat()
    except (ValueError, OverflowError):
        _need(False, "month")


STEPS = ("train", "stage")
STATE_FILES = ("driver_residual_pool__size.json", "driver_residual_pool__implied_t1.json",
               "driver_residual_pool__runup_move.json", "chooser_analog_pool.json")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


_PLAN_FIELDS = ("kind", "spec_hash", "parameters", "implementation_ref", "environment_ref",
                "resource_class", "input_refs", "dependency_job_ids")


def step_key(step: str, plan, parents=()) -> str:
    _need(step in STEPS, "step")
    _need(isinstance(plan, dict), "plan")
    _need(all(field in plan for field in _PLAN_FIELDS), "plan")
    spec_hash = plan["spec_hash"]
    _need(isinstance(spec_hash, str) and _HASH.fullmatch(spec_hash), "spec_hash")
    try:
        _need(spec_hash == content_hash(plan["parameters"]), "spec_hash")
        deps = tuple(dict.fromkeys((*plan["dependency_job_ids"], *parents)))
        fingerprint = {
            "kind": plan["kind"],
            "spec_hash": spec_hash,
            "parameters": plan["parameters"],
            "implementation_ref": plan["implementation_ref"],
            "environment_ref": plan["environment_ref"],
            "resource_class": plan["resource_class"],
            "input_refs": list(plan["input_refs"]),
            "dependency_job_ids": list(deps),
        }
        digest = content_hash(fingerprint)
    except TypeError:
        _need(False, "plan")
    return f"retrain-{step}:{digest.split(':', 1)[1][:24]}"


def _request_job_id(request, field, expected_kind):
    _need(isinstance(request, SubmitRequest), field)
    _need(request.job.kind == expected_kind, field)
    return job_id_for(request.namespace, request.idempotency_key)


def _manifest_ref(features_request):
    _need(isinstance(features_request, SubmitRequest), "features_request")
    _need(features_request.job.kind == "legacy_features", "features_request")
    refs = features_request.job.input_refs
    _need(isinstance(refs, (list, tuple)) and len(refs) == 1
          and isinstance(refs[0], str) and refs[0], "features_request")
    _need(not refs[0].startswith("job_"), "features_request")
    return refs[0], job_id_for(features_request.namespace, features_request.idempotency_key)


def _plan(kind, params, refs, resource, implementation_ref, environment_ref, deps=()):
    parameters = to_document(params)
    return {"schema_version": "operations_plan.v1.0", "kind": kind, "mode": "shadow",
            "effects": ["staged"], "parameters": parameters, "input_refs": list(refs),
            "blocked_prerequisites": [], "spec_hash": content_hash(parameters),
            "implementation_ref": implementation_ref, "environment_ref": environment_ref,
            "resource_class": resource, "dependency_job_ids": list(deps)}


def train_plan(*, month, features_request, implementation_ref, environment_ref):
    manifest, feature_id = _manifest_ref(features_request)
    params = TrainingParameters(expected_ids=("training",), mode="monthly_pools", month=month,
                                cutoffs=(retrain_cutoff(month),), input_mode="legacy",
                                input_bindings={"legacy_manifest.json": manifest})
    return _plan("training", params, (manifest,), "experiment_heavy", implementation_ref,
                 environment_ref, (feature_id,))


def stage_plan(*, month, session, base, release_root, features_request, training_request,
               implementation_ref, environment_ref):
    _need(_safe_id(base), "base")
    _need(_safe_root(release_root), "release_root")
    candidate = candidate_id(month, session)
    _need(base != candidate, "base")
    manifest, feature_id = _manifest_ref(features_request)
    train_id = _request_job_id(training_request, "training_request", "training")
    _need(training_request.namespace == "shadow", "training_request")
    _need(training_request.job.output_namespace == "shadow", "training_request")
    train_params = training_request.job.parameters
    _need(isinstance(train_params, dict), "training_request")
    train_refs = training_request.job.input_refs
    train_deps = training_request.job.dependency_job_ids
    _need(isinstance(train_refs, (list, tuple)), "training_request")
    _need(isinstance(train_deps, (list, tuple)), "training_request")
    try:
        train_params = from_document(TrainingParameters, train_params)
    except (KeyError, TypeError, ValueError):
        _need(False, "training_request")
    _need(not training_parameter_problems(None, train_params), "training_request")
    _need(train_params.mode == "monthly_pools", "training_request")
    _need(train_params.month == month, "training_request")
    _need(train_params.cutoffs == (retrain_cutoff(month),), "training_request")
    _need(train_params.input_bindings == {"legacy_manifest.json": manifest},
          "training_request")
    _need(tuple(train_refs) == (manifest,), "training_request")
    _need(feature_id in train_deps, "training_request")
    refs = {name: f"{train_id}#training/{name}" for name in STATE_FILES}
    bindings = {**refs, "legacy_manifest.json": manifest}
    params = StateStageParameters(expected_ids=("phase5_state_stage",), release_root=release_root,
        prior_release_id=base, release_id=candidate, cutoff=retrain_cutoff(month),
        state_files=tuple(sorted(STATE_FILES)), input_bindings=bindings, mode="monthly")
    problems = state_stage_parameter_problems(None, params)
    if problems:
        raise fail("INVALID_REQUEST", "state stage plan is invalid",
                   details={"problems": list(problems)})
    return _plan("phase5_state_stage", params, (*[refs[n] for n in sorted(refs)], manifest),
                 "projection", implementation_ref, environment_ref, (train_id,))


_REQUEST_KINDS = {"train": training_job_kind, "stage": state_stage_job_kind}
_EXPECTED_PLAN_KINDS = {"train": "training", "stage": "phase5_state_stage"}


def _request(step, plan, parents=()):
    _need(isinstance(plan, dict), "plan")
    kind = _REQUEST_KINDS[step]()
    _need(plan.get("kind") == _EXPECTED_PLAN_KINDS[step], "plan")
    _need(plan["kind"] == kind.name, "plan")
    _need(plan.get("schema_version") == "operations_plan.v1.0", "plan")
    _need(plan.get("mode") == "shadow", "plan")
    _need(plan.get("blocked_prerequisites") == [], "plan")
    _need(plan.get("effects") == ["staged"], "plan")
    plan_refs = plan.get("input_refs")
    _need(isinstance(plan_refs, (list, tuple))
          and all(isinstance(ref, str) for ref in plan_refs), "input_refs")
    try:
        parents = tuple(parents)
    except TypeError:
        _need(False, "parents")
    _need(all(isinstance(parent, SubmitRequest) for parent in parents), "parents")
    if step == "train":
        _need(all(parent.namespace == "shadow" and parent.job.kind == "retrain_step"
                  for parent in parents), "parents")
    else:
        _need(not parents, "parents")
    parent_ids = tuple(job_id_for(parent.namespace, parent.idempotency_key)
                       for parent in parents)
    key = step_key(step, plan, parent_ids)
    deps = tuple(dict.fromkeys((*plan["dependency_job_ids"], *parent_ids)))
    refs = tuple(ref for ref in plan_refs if not ref.startswith("job_"))
    job = JobSpec(kind=kind.name, implementation_ref=plan["implementation_ref"],
        spec_hash=plan["spec_hash"], parameters=to_document(plan["parameters"]),
        input_refs=refs, dependency_job_ids=deps,
        output_namespace="shadow", resource_class=plan["resource_class"],
        retry_policy_ref=kind.retry.name, checkpoint_contract_ref=kind.checkpoint_contract,
        environment_ref=plan["environment_ref"])
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job)


def train_request(*, plan, parents=()):
    return _request("train", plan, parents)


def stage_request(*, plan, parents=()):
    return _request("stage", plan, parents)

