"""Preregister and record one reconstructed-history RUNUP exit comparison."""
from pathlib import Path

from threadpoolctl import threadpool_limits

from engine.v2.contracts.data import SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.foundation import canonical_json
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import ExperimentSpec, resolve_experiment_plan
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments.runup_exit_report import HORIZON, compare_runup_exits, validate_request

RUNNER = "experiments/v2_runup_study.py"
ROOT = Path(__file__).resolve().parents[1]


def comparison_request(spec, event_ids):
    """Admit the closed, deterministic one-candidate contract without I/O."""
    if not isinstance(spec, ExperimentSpec):
        raise fail("INVALID_EXPERIMENT_SPEC", "RUNUP requires an experiment specification")
    plan = resolve_experiment_plan(spec)
    economics = plan.as_document()["economic_params"]
    recipe = economics.get("exit", {})
    if (plan.runner != RUNNER or plan.price_source != "option_chains"
            or spec.input_files != () or plan.seed != 0 or plan.folds != ()
            or set(economics) != {"fill", "exit"}
            or recipe.get("kind") != "target_stop" or recipe.get("trading_days") != HORIZON
            or "threshold_units" not in recipe):
        raise fail("INVALID_EXPERIMENT_SPEC", "RUNUP requires its explicit single target/stop policy")
    return validate_request({"event_ids": event_ids, "fill_alpha": economics["fill"],
        "threshold_units": recipe["threshold_units"], "arms": [{"arm_id": plan.arms[0],
        "target_pnl": recipe["target_pnl"], "stop_pnl": recipe["stop_pnl"]}]})


def _prepare(repository, spec, request, *, snapshot, scope, as_of_month, expected=None):
    if not isinstance(snapshot, SnapshotRef):
        raise fail("INVALID_EXPERIMENT_SPEC", "RUNUP requires an explicit snapshot reference")
    return native.prepare_native_registration(repository, spec, code_root=ROOT,
        snapshot=snapshot, scope=scope, as_of_month=as_of_month,
        event_ids=request["event_ids"], expected=expected)


def register_runup(conn, store, repository, spec, *, snapshot, scope, as_of_month, event_ids):
    """Seal explicit pin and complete eligible cohort using event metadata only."""
    request = comparison_request(spec, event_ids)
    registration, normalized = _prepare(repository, spec, request, snapshot=snapshot,
                                        scope=scope, as_of_month=as_of_month)
    return native.reserve_native_registration(conn, store, registration, normalized)


def run_runup(conn, store, repository, spec, *, snapshot, scope, as_of_month, event_ids,
              max_evidence_bytes=native.MAX_NATIVE_EVIDENCE_BYTES):
    """Re-admit before touching the outcome slot; replay never opens market data.

    One candidate is one attempted variant, including its fixed-day control.
    Only a comparison's supported scientific refusal may occupy the terminal
    slot. Admission, resource and publication failures cannot poison that slot.
    """
    request = comparison_request(spec, event_ids)
    original = native.read_native_registration(conn, store, spec, max_evidence_bytes=max_evidence_bytes)
    document = original.document
    if document["schema_version"] != "native_experiment_registration.v1.0" or "final_read" in document:
        raise fail("HOLDOUT_ACCESS_DENIED", "RUNUP registration grants selection authority only")
    if (document["event_ids"] != request["event_ids"]
            or document["holdouts"]["holdout_as_of_month"] != as_of_month):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "RUNUP cohort or holdout month differs from registration")
    registration, _ = _prepare(repository, spec, request, snapshot=snapshot, scope=scope,
                               as_of_month=as_of_month, expected=original)
    if registration != original:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "RUNUP request differs from registration")
    native.verify_native_registration(conn, store, registration, max_evidence_bytes=max_evidence_bytes)
    recording = {"no_ledger": True, "max_evidence_bytes": max_evidence_bytes}
    saved = outcomes.replay_native_outcome(conn, store, registration, **recording)
    replayed = saved is not None
    if saved is None:
        # Everything above is admission. Only this call attempts the comparison.
        try:
            with threadpool_limits(limits=1):
                report = compare_runup_exits(repository, snapshot, as_of_month=as_of_month,
                                             document=request)
        except (OpsError, DataError) as error:
            if error.code not in outcomes.STAGES:
                raise
            saved = outcomes.publish_native_outcome(conn, store, registration,
                problem=error.problem, attempted=True, **recording)
        else:
            saved = outcomes.publish_native_outcome(conn, store, registration,
                report=canonical_json(report), attempted=True, **recording)
    return {"variant_id": registration.variant_id, "run_id": registration.run_id,
            "replayed": replayed, **saved}
