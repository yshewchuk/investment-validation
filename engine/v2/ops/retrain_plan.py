"""Plan-built retrain chain (#805 slice 2; models ARCHITECTURE section 2.3, plan shape).

Jobs of kind ``retrain_step``: ``restage`` (always planned when the plan carries a
``retrain`` section) then, when a month is due, ``train, stage, accept, check, promote``.
``score`` depends on the last job. Slice 2 implements no step: ``restage`` records
``restage: unchecked`` and every chain step fails with ``RETRAIN_UNAVAILABLE`` (worker
side, ``worker._dispatch_retrain_step``); slice 6 replaces the stub.
"""
from __future__ import annotations

from dataclasses import replace

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import to_document
from engine.v2.ops.stages import RetrainStepParameters
from engine.v2.ops.submission import job_id_for

KIND = "retrain_step"


def step_key(step: str, session: str, month: str, base: str, plan_hash: str) -> str:
    """``retrain-restage:{session}:{plan_hash}`` or ``retrain-{step}:{session}:{M}:{base}:{plan_hash}`` (R6).

    ``plan_hash`` is the plan generation token (16 hex characters), so a same-session replan
    with a changed manifest gets fresh jobs."""
    if step == "restage":
        return f"retrain-restage:{session}:{plan_hash}"
    return f"retrain-{step}:{session}:{month}:{base}:{plan_hash}"


def _request(step, key, session, month, base, parents, implementation_ref, runtime, contract):
    parameters = to_document(RetrainStepParameters(
        expected_ids=(key,), session=session, step=step, month=month, base=base))
    job = JobSpec(kind=KIND, implementation_ref=implementation_ref, spec_hash=None,
                  parameters=parameters, input_refs=(),
                  dependency_job_ids=tuple(parents), output_namespace="shadow",
                  retry_policy_ref="bounded", checkpoint_contract_ref=contract, **runtime)
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job)


def _typed_retrain_request(kind_name, key, parameters, parents, implementation_ref,
                           validation_env, source_plan=None, input_refs=()):
    from engine.v2.foundation import content_hash, to_document
    from engine.v2.ops.retrain_check import phase5_accept_job_kind, retrain_check_job_kind
    from engine.v2.ops.state_stage import state_stage_job_kind
    from engine.v2.ops.training import promote_job_kind, training_job_kind
    specs = {
        "training": (training_job_kind, "bounded"),
        "phase5_state_stage": (state_stage_job_kind, "none"),
        "phase5_accept": (phase5_accept_job_kind, "none"),
        "retrain_check": (retrain_check_job_kind, "none"),
        "models_promote": (promote_job_kind, "bounded"),
    }
    factory, retry_ref = specs[kind_name]
    kind = factory()
    source = source_plan or {}
    params = to_document(parameters)
    job = JobSpec(
        kind=kind_name, implementation_ref=source.get("implementation_ref", implementation_ref),
        spec_hash=source.get("spec_hash", content_hash(params)), parameters=params,
        input_refs=tuple(ref for ref in (input_refs or source.get("input_refs", ()))
                         if "#" not in ref),
        dependency_job_ids=tuple(parents), output_namespace="shadow",
        retry_policy_ref=retry_ref, checkpoint_contract_ref=kind.checkpoint_contract,
        resource_class=source.get("resource_class", "validation"),
        environment_ref=source.get("environment_ref", validation_env))
    return SubmitRequest(namespace="shadow", idempotency_key=key,
                         principal="operator", job=job)


def _due_retrain_requests(session, month, base, ph, keys, restage_key,
                          implementation_ref, requests):
    from engine.v2.foundation import content_hash, to_document
    from engine.v2.models import deployment
    from engine.v2.ops.experiments import default_checkout_root
    from engine.v2.ops.fingerprints import environment_identity
    from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
    from engine.v2.ops.retrain_candidate import candidate_id, retrain_cutoff
    from engine.v2.ops.retrain_check import Phase5AcceptParameters, RetrainCheckParameters
    from engine.v2.ops.state_stage import state_stage_plan
    from engine.v2.ops.training import promote_plan, training_plan
    from tools.phase5_training_job import state_file_name
    cutoff, candidate = retrain_cutoff(month), candidate_id(month, session)
    features_id = keys["features"]
    features_request = next(
        request for request in requests
        if job_id_for("shadow", request.idempotency_key) == features_id)
    manifest_ref = features_request.job.parameters.get("input_bindings", {}).get(
        "legacy_manifest.json")
    if not manifest_ref:
        from engine.v2.ops.errors import fail
        raise fail("INVALID_REQUEST", "due retrain requires a pinned legacy manifest")
    train_plan = training_plan(mode="monthly_pools", month=month, cutoffs=(cutoff,),
                               manifest_ref=manifest_ref)
    train_key = step_key("train", session, month, base, ph)
    train_id = job_id_for("shadow", train_key)
    members = ("driver_residual_pool:size", "driver_residual_pool:implied_t1",
               "driver_residual_pool:runup_move", "chooser_analog_pool")
    state_refs = {state_file_name(member):
                  f"{train_id}#training/{state_file_name(member)}" for member in members}
    stage_plan = state_stage_plan(
        release_root=None, prior_release_id=base, release_id=candidate, cutoff=cutoff,
        state_refs=state_refs, manifest_ref=manifest_ref,
        source_root=default_checkout_root(), mode="monthly")
    release_root = stage_plan["parameters"]["release_root"]
    deployment_root = str(deployment.production_deployment_root())
    validation = profile_named(DEFAULT_POLICY, "validation")
    validation_env = content_hash(environment_identity(
        validation.thread_count or validation.cpu_count))
    accept_key = step_key("accept", session, month, base, ph)
    accept_id = job_id_for("shadow", accept_key)
    acceptance = to_document(Phase5AcceptParameters(
        expected_ids=("phase5_accept",), release_root=release_root,
        candidate_release_id=candidate))
    check_key = step_key("check", session, month, base, ph)
    acceptance_ref = f"{accept_id}#acceptance"
    check = to_document(RetrainCheckParameters(
        expected_ids=("retrain_check",), release_root=release_root,
        candidate_release_id=candidate, incumbent_release_id=base,
        input_bindings={"acceptance.json": acceptance_ref}))
    promote_key = step_key("promote", session, month, base, ph)
    promote = promote_plan(release_root=deployment_root, release_id=candidate,
                           expected_previous_release_id=base)
    stage_key = step_key("stage", session, month, base, ph)
    stage_id = job_id_for("shadow", stage_key)
    check_id = job_id_for("shadow", check_key)
    return [
        _typed_retrain_request("training", train_key, train_plan["parameters"],
                               (job_id_for("shadow", restage_key), features_id),
                               implementation_ref, validation_env, train_plan),
        _typed_retrain_request("phase5_state_stage", stage_key,
                               stage_plan["parameters"], (train_id,),
                               implementation_ref, validation_env, stage_plan),
        _typed_retrain_request("phase5_accept", accept_key, acceptance, (stage_id,),
                               implementation_ref, validation_env),
        _typed_retrain_request("retrain_check", check_key, check, (accept_id,),
                               implementation_ref, validation_env),
        _typed_retrain_request("models_promote", promote_key, promote["parameters"],
                               (check_id,), implementation_ref, validation_env, promote),
    ]


def attach_retrain_chain(plan, requests, keys, implementation_ref, runtime, contract):
    section = plan.get("retrain")
    if section is None:
        return requests
    from engine.v2.ops.plans import check_retrain_section
    check_retrain_section(section)
    session = plan["session"]
    ph = (keys["score"].rsplit("_", 1)[-1] if "score" in keys
          else str(plan["plan_hash"]).rsplit(":", 1)[-1])[:16]
    refresh = (keys["refresh"],) if "refresh" in keys else ()
    restage_key = step_key("restage", session, "", "", ph)
    chain = [_request("restage", restage_key, session, "", "", refresh,
                      implementation_ref, runtime, contract)]
    last_key = restage_key
    if section.get("state") == "due":
        month, base = section["month"], section["base"]
        chain.extend(_due_retrain_requests(session, month, base, ph, keys,
                                           restage_key, implementation_ref, requests))
        last_key = step_key("promote", session, month, base, ph)
    out, inserted = [], False
    for request in requests:
        if request.job.kind == "legacy_score" and not inserted:
            out.extend(chain)
            inserted = True
            request = replace(request, job=replace(
                request.job, dependency_job_ids=tuple(request.job.dependency_job_ids)
                + (job_id_for("shadow", last_key),)))
        out.append(request)
    return tuple(out if inserted else chain + out)
