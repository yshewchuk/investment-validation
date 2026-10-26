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


def step_key(step: str, session: str, month: str = "", base: str = "") -> str:
    """``retrain-restage:{session}`` or ``retrain-{step}:{session}:{M}:{base}`` (R6)."""
    if step == "restage":
        return f"retrain-restage:{session}"
    return f"retrain-{step}:{session}:{month}:{base}"


def _request(step, key, session, month, base, parents, implementation_ref, environment_ref):
    from engine.v2.ops.nightly import _checkpoint_contract, _sidecar_runtime_identity

    parameters = to_document(RetrainStepParameters(
        expected_ids=(key,), session=session, step=step, month=month, base=base))
    runtime = _sidecar_runtime_identity(KIND)
    if environment_ref:
        runtime["environment_ref"] = environment_ref
    job = JobSpec(kind=KIND, implementation_ref=implementation_ref, spec_hash=None,
                  parameters=parameters, input_refs=(),
                  dependency_job_ids=tuple(parents), output_namespace="shadow",
                  retry_policy_ref="bounded", checkpoint_contract_ref=_checkpoint_contract(KIND),
                  **runtime)
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job)


def attach_retrain_chain(plan, requests, keys, implementation_ref, environment_ref):
    """Insert the restage (+ chain) requests before ``score`` and make ``score`` depend on
    the last of them. A plan without a ``retrain`` section is returned unchanged."""
    section = plan.get("retrain")
    if not section:
        return requests
    session = plan["session"]
    refresh = (keys["refresh"],) if "refresh" in keys else ()
    restage_key = step_key("restage", session)
    chain = [_request("restage", restage_key, session, "", "", refresh,
                      implementation_ref, environment_ref)]
    last = job_id_for("shadow", restage_key)
    if section.get("state") == "due":
        for step in STEPS:
            key = step_key(step, session, section["month"], section["base"])
            chain.append(_request(step, key, session, section["month"], section["base"],
                                  (last,), implementation_ref, environment_ref))
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
