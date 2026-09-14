"""Fixed subprocess entrypoint. The launch gate opens only after PID persistence.

No arbitrary modules or executables are accepted. Output is a small protocol
message on an inherited pipe; payloads stay in the assigned staging directory.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import traceback
from pathlib import Path


def _write_diagnostics(root: Path) -> None:
    """A7: the traceback goes to a private file, never the result pipe."""
    directory = root / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / "worker.stderr", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, traceback.format_exc().encode())
    finally:
        os.close(fd)


def main():
    envelope = json.loads(sys.stdin.buffer.readline())
    os.sched_setaffinity(0, envelope["cpu_ids"])
    root = Path(envelope["staging"])
    os.environ["INVESTING_PLAN_ROOT"] = str(envelope.get("legacy_root") or root / "legacy")
    fd = int(envelope["result_fd"])
    try:
        result = dispatch(envelope["worker"], envelope["parameters"], root, envelope=envelope)
        result.update(schema_version="worker_result.v1.0", job_id=envelope["job_id"],
                      attempt_id=envelope["attempt_id"], fence=envelope["fence"])
        result["self_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except BaseException:
        try:
            _write_diagnostics(root)
        except OSError:
            pass
        result = {"schema_version": "worker_result.v1.0", "failure": "WORKER_FAILED"}
    data = json.dumps(result, allow_nan=False).encode()
    os.write(fd, data + b"\n")
    os.close(fd)
    return int("failure" in result)


def dispatch(worker, parameters, root, *, envelope=None):
    envelope = envelope or {}
    if worker == "legacy_materialize":
        from engine.v2.ops.materialization_worker import run_materialize
        return run_materialize(parameters, root, envelope)
    if worker.startswith("legacy_"):
        from engine.v2.ops.legacy_actions import run_action
        values = parameters if isinstance(parameters, dict) else vars(parameters)
        output = run_action(worker, values, root, legacy_root=envelope.get("legacy_root"))
        outputs = [{"name": worker, "path": output["path"], "schema": "legacy_action.v1.0"}]
        # A legacy action may publish additional named outputs alongside its
        # primary one (e.g. legacy_finality's finality_coverage.json) — see
        # ``legacy_adapter._action_finality``.
        outputs.extend({"name": extra["name"], "path": extra["path"], "schema": extra["schema"]}
                       for extra in output.get("extra", []))
        return {"outputs": outputs, "completed_ids": [worker], "action": worker,
                "coverage": output}
    if worker == "decision_evidence":
        return _dispatch_decision_evidence(parameters, root)
    if worker in ("snapshot_import", "legacy_rebuild_candidate"):
        return _dispatch_snapshot_import(worker, parameters, root)
    if worker in ("ledger_export", "engineering_gate", "publication", "backup"):
        return _dispatch_effect_receipt(worker, parameters, root)
    if worker == "artifact_check":
        output = root / "receipt.json"
        output.write_text(json.dumps({"checked": parameters["expected_ids"],
                                     "affinity": sorted(os.sched_getaffinity(0)),
                                     "threads": os.environ["OMP_NUM_THREADS"]}))
        return {"outputs": [{"name": "receipt", "path": "receipt.json", "schema": "receipt.v1.0"}],
                "completed_ids": parameters["expected_ids"],
                "no_work": not parameters["expected_ids"],
                "observed": {"affinity": sorted(os.sched_getaffinity(0)),
                             "threads": os.environ["OMP_NUM_THREADS"]}}
    raise ValueError("unsupported worker")


def _dispatch_decision_evidence(parameters, root):
    """Derive the decision plan/evidence pair from staged, materialized inputs.

    ``score.json``/``finality.json``/``replay.json``/``finality_coverage.json``
    are plain files here — ``executor._materialize_inputs`` writes every
    ``input_bindings`` entry's verified bytes into staging before any worker
    runs, legacy or not. This worker never touches the legacy tree or the
    catalog; it recomputes the score/finality artifact identity locally,
    from the SAME bytes it just read, with the store's own
    :func:`artifact_reference` — the coordinator re-derives independently
    from its recorded bindings and the two must agree byte-for-byte
    (P2-5/B1c).
    """
    from engine.v2.foundation import artifact_reference
    from engine.v2.ops.decision_evidence import derive

    score_bytes = (root / "score.json").read_bytes()
    finality_bytes = (root / "finality.json").read_bytes()
    replay = json.loads((root / "replay.json").read_text())
    coverage = json.loads((root / "finality_coverage.json").read_text())
    plan_bytes, evidence_bytes = derive(
        json.loads(score_bytes), artifact_reference(score_bytes, "legacy_action.v1.0"),
        json.loads(finality_bytes), artifact_reference(finality_bytes, "legacy_action.v1.0"),
        replay, coverage, requested_session=parameters["session"],
        deployment=parameters["deployment"], decision_clock=parameters["decision_clock"],
        scope=parameters.get("effect_scope") or "shadow")
    (root / "decision_plan.json").write_bytes(plan_bytes)
    (root / "decision_evidence.json").write_bytes(evidence_bytes)
    return {"outputs": [
        {"name": "decision_plan", "path": "decision_plan.json", "schema": "decision_plan.v1.0"},
        {"name": "decision_evidence", "path": "decision_evidence.json",
         "schema": "decision_evidence.v1.0"}],
        "completed_ids": list(parameters["expected_ids"]),
        "no_work": not parameters["expected_ids"]}


def _dispatch_snapshot_import(worker, parameters, root):
    """P2-7/Task7b: both §7/§10 workers are pure functions of ``parameters``
    and the staging directory — see ``engine.v2.ops.snapshot_import`` for why
    neither ever touches the catalog or a live store path."""
    from engine.v2.ops.snapshot_import import (
        worker_legacy_rebuild_candidate,
        worker_snapshot_import,
    )

    fn = worker_snapshot_import if worker == "snapshot_import" else worker_legacy_rebuild_candidate
    return fn(parameters, root)


def _dispatch_effect_receipt(worker, parameters, root):
    """Trivial pure worker for a coordinator-driven effect stage (P2-5/Task5).

    ``ledger_export``, ``engineering_gate``, ``publication`` and ``backup``
    never touch the catalog or the outbox from inside a subprocess; all of
    that real work happens in the supervisor's coordinator effect
    (``engine.v2.ops.effects_graph``), after this attempt's tiny receipt is
    validated, inside the same fenced finish path every other coordinator
    effect uses. This worker only proves the attempt ran.
    """
    receipt = {"schema_version": "effect_receipt.v1.0", "kind": worker,
               "expected_ids": list(parameters["expected_ids"])}
    (root / "receipt.json").write_text(json.dumps(receipt))
    return {"outputs": [{"name": worker, "path": "receipt.json", "schema": "effect_receipt.v1.0"}],
            "completed_ids": list(parameters["expected_ids"]),
            "no_work": not parameters["expected_ids"]}


if __name__ == "__main__":
    raise SystemExit(main())
