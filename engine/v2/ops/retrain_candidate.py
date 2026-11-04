"""Pure candidate identity, cutoff, and typed step plan/request builders for the monthly retrain chain (S3a).

The three driver pools are S1 re-frozen and champion-carried; the chooser pool is dated.
Candidate identity and cutoff derive deterministically, never inspecting state.
"""
import re
from datetime import date, timedelta

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import content_hash, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.retrain_check import (
    ACCEPTANCE_FILE,
    Phase5AcceptParameters,
    RetrainCheckParameters,
    phase5_accept_job_kind,
    retrain_check_job_kind,
)
from engine.v2.ops.state_stage import (
    StateStageParameters,
    state_stage_job_kind,
    state_stage_parameter_problems,
)
from engine.v2.ops.submission import job_id_for
from engine.v2.ops.training import (
    PromoteParameters,
    TrainingParameters,
    promote_job_kind,
    training_job_kind,
)

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_SESSION = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


def _safe_id(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and not value.startswith(".")
            and not any(char in value for char in ("/", "\\", "\0")))


def _safe_root(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\0" not in value


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


STEPS = ("train", "stage", "accept", "check", "promote")
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
    _need(isinstance(digest, str) and _HASH.fullmatch(digest), "spec_hash")
    return f"retrain-{step}:{digest.split(':', 1)[1][:24]}"


def _request_job_id(request, field, expected_kind):
    _need(isinstance(request, SubmitRequest), field)
    _need(request.job.kind == expected_kind, field)
    return job_id_for(request.namespace, request.idempotency_key)


def _manifest_ref(features_request):
    _need(isinstance(features_request, SubmitRequest), "features_request")
    _need(features_request.job.kind == "legacy_features", "features_request")
    refs = features_request.job.input_refs
    _need(len(refs) == 1 and isinstance(refs[0], str) and refs[0], "features_request")
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


def stage_plan(*, month, session, base, release_root, features_request,
               implementation_ref, environment_ref):
    _need(_safe_id(base), "base")
    _need(_safe_root(release_root), "release_root")
    candidate = candidate_id(month, session)
    manifest, _ = _manifest_ref(features_request)
    train_plan_value = train_plan(month=month, features_request=features_request,
                                  implementation_ref=implementation_ref,
                                  environment_ref=environment_ref)
    train_id = job_id_for("shadow", step_key("train", train_plan_value))
    refs = {name: f"job_{train_id}#training/{name}" for name in STATE_FILES}
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


def accept_plan(*, month, session, release_root, stage_request,
                implementation_ref, environment_ref):
    _need(_safe_root(release_root), "release_root")
    params = Phase5AcceptParameters(expected_ids=("phase5_accept",), release_root=release_root,
                                    candidate_release_id=candidate_id(month, session))
    return _plan("phase5_accept", params, (), "validation", implementation_ref, environment_ref,
                 (_request_job_id(stage_request, "stage_request", "phase5_state_stage"),))


def check_plan(*, month, session, base, release_root, accept_request,
               implementation_ref, environment_ref):
    _need(_safe_id(base), "base")
    _need(_safe_root(release_root), "release_root")
    accept_id = _request_job_id(accept_request, "accept_request", "phase5_accept")
    ref = f"job_{accept_id}#acceptance"
    params = RetrainCheckParameters(expected_ids=("retrain_check",), release_root=release_root,
        candidate_release_id=candidate_id(month, session), incumbent_release_id=base,
        input_bindings={ACCEPTANCE_FILE: ref})
    return _plan("retrain_check", params, (ref,), "validation", implementation_ref,
                 environment_ref, (accept_id,))


def promote_plan(*, month, session, base, release_root, check_request,
                 implementation_ref, environment_ref):
    _need(_safe_id(base), "base")
    _need(_safe_root(release_root), "release_root")
    params = PromoteParameters(expected_ids=("models_promote",), release_root=release_root,
        release_id=candidate_id(month, session), expected_previous_release_id=base)
    return _plan("promote", params, (), "delivery", implementation_ref, environment_ref,
                 (_request_job_id(check_request, "check_request", "retrain_check"),))


_REQUEST_KINDS = {"train": training_job_kind, "stage": state_stage_job_kind,
                  "accept": phase5_accept_job_kind, "check": retrain_check_job_kind,
                  "promote": promote_job_kind}


def _request(step, plan, parents=()):
    kind = _REQUEST_KINDS[step]()
    key = step_key(step, plan, parents)
    deps = tuple(dict.fromkeys((*plan["dependency_job_ids"], *parents)))
    refs = tuple(ref for ref in plan["input_refs"] if not ref.startswith("job_"))
    job = JobSpec(kind=kind.name, implementation_ref=plan["implementation_ref"],
        spec_hash=plan["spec_hash"], parameters=plan["parameters"],
        input_refs=refs, dependency_job_ids=deps,
        output_namespace="shadow", resource_class=plan["resource_class"],
        retry_policy_ref=kind.retry.name, checkpoint_contract_ref=kind.checkpoint_contract,
        environment_ref=plan["environment_ref"])
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job)


def train_request(*, plan, parents=()):
    return _request("train", plan, parents)


def stage_request(*, plan, parents=()):
    return _request("stage", plan, parents)


def accept_request(*, plan, parents=()):
    return _request("accept", plan, parents)


def check_request(*, plan, parents=()):
    return _request("check", plan, parents)


def promote_request(*, plan, parents=()):
    return _request("promote", plan, parents)

