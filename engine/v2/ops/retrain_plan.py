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
from engine.v2.ops.retrain_due import STEPS
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


def attach_retrain_chain(plan, requests, keys, implementation_ref, runtime, contract):
    """Insert the restage (+ chain) requests before ``score`` and make ``score`` depend on
    the last of them. A plan without a ``retrain`` section is returned unchanged.
    ``runtime`` is the caller's ``resource_class``/``environment_ref`` dict for the kind
    and ``contract`` its checkpoint contract (passed in so this module does not import
    ``nightly``)."""
    section = plan.get("retrain")
    if section is None:
        return requests
    from engine.v2.ops.plans import check_retrain_section
    check_retrain_section(section)
    session = plan["session"]
    # The plan generation: the score stage's job id embeds the scope hash (decision clock,
    # implementation, manifest) that every ordinary stage key carries; plan_hash alone does not
    # change on an unchanged-input replan. Without a score stage, fall back to the plan hash.
    ph = (keys["score"].rsplit("_", 1)[-1] if "score" in keys
          else str(plan["plan_hash"]).rsplit(":", 1)[-1])[:16]
    refresh = (keys["refresh"],) if "refresh" in keys else ()
    restage_key = step_key("restage", session, "", "", ph)
    chain = [_request("restage", restage_key, session, "", "", refresh,
                      implementation_ref, runtime, contract)]
    last = job_id_for("shadow", restage_key)
    if section.get("state") == "due":
        for step in STEPS:
            key = step_key(step, session, section["month"], section["base"], ph)
            chain.append(_request(step, key, session, section["month"], section["base"],
                                  (last,), implementation_ref, runtime, contract))
            last = job_id_for("shadow", key)
    out = []
    inserted = False
    for request in requests:
        if request.job.kind == "legacy_score" and not inserted:
            out.extend(chain)
            inserted = True
            request = replace(request, job=replace(
                request.job, dependency_job_ids=tuple(request.job.dependency_job_ids) + (last,)))
        out.append(request)
    if not inserted:
        out = chain + out
    return tuple(out)
