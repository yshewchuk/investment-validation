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

from engine.v2.ops import worker_progress
from engine.v2.ops.errors import OpsError, fail, make_problem


def _write_diagnostics(root: Path) -> None:
    """A7: the traceback goes to a private file, never the result pipe."""
    directory = root / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / "worker.stderr", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, traceback.format_exc().encode())
    finally:
        os.close(fd)


def _write_failure_details(root: Path, details: dict) -> None:
    """A typed failure's ``details`` go to a private staging file, never the
    small result pipe: the supervisor publishes this file as a verified
    artifact and stamps the problem's ``diagnostic_ref`` with it (real
    nightly attempt 9: a ``VALIDATION_FAILED``'s ``missing``/``unplanned``
    keys were only ever visible in private ``worker.stderr``)."""
    directory = root / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / "failure_details.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(details, allow_nan=False).encode())
    finally:
        os.close(fd)


def main():
    envelope = json.loads(sys.stdin.buffer.readline())
    os.sched_setaffinity(0, envelope["cpu_ids"])
    root = Path(envelope["staging"])
    os.environ["INVESTING_PLAN_ROOT"] = str(envelope.get("legacy_root") or root / "legacy")
    fd = int(envelope["result_fd"])
    worker_progress.configure(root)
    try:
        result = dispatch(envelope["worker"], envelope["parameters"], root, envelope=envelope)
        result.update(schema_version="worker_result.v1.0", job_id=envelope["job_id"],
                      attempt_id=envelope["attempt_id"], fence=envelope["fence"])
        result["self_peak_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except BaseException as exc:
        result = _failure_result(root, exc)
    worker_progress.reset()
    data = json.dumps(result, allow_nan=False).encode()
    os.write(fd, data + b"\n")
    os.close(fd)
    return int("failure" in result)


def _failure_result(root: Path, exc: BaseException) -> dict:
    try:
        _write_diagnostics(root)
    except OSError:
        pass
    problem = _classify(exc)
    try:
        _write_failure_details(root, problem.details)
    except OSError:
        pass
    return {"schema_version": "worker_result.v1.0", "failure": problem.code,
           "problem": {"code": problem.code, "category": problem.category,
                      "retryable": problem.retryable, "message": problem.message}}


def _classify(exc: BaseException):
    """A typed ``Problem`` for a caught worker exception.

    Three narrow, reviewed branches whose message shape is known to carry no
    data value; anything else falls to :func:`_generic_problem`, which never
    includes the message at all (see its docstring)."""
    if isinstance(exc, OpsError):
        return exc.problem
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        # A deterministic import failure (e.g. a pinned model artifact's
        # pickle names an engine.* module absent from this code snapshot) is
        # not a transient worker crash: retrying it wastes an attempt and
        # always fails the same way. Name the module so the failure is
        # diagnosable without reading worker.stderr.
        return make_problem(
            "INPUT_CHANGED",
            f"worker import failed: no module named {getattr(exc, 'name', None) or exc}",
            details={"module": getattr(exc, "name", None)})
    if isinstance(exc, ValueError):
        # A bare ValueError out of adapter/dispatch code is a deterministic
        # defect in what the worker tried to produce, not a transient crash
        # -- real shadow nightly attempt 14: legacy_adapter._write_action's
        # json.dumps(allow_nan=False) raised "Out of range float values are
        # not JSON compliant: nan" on a real (legacy-produced) NaN, twice, on
        # two fresh attempts, because the SAME inputs deterministically
        # produce the SAME exception every retry. Falling through to the
        # untyped WORKER_FAILED branch marks that ("internal", True) --
        # retryable -- exactly the mistake that spent two attempts re-running
        # a bug retrying can never fix. VALIDATION_FAILED ("validation",
        # False) matches how every OTHER adapter defect detected inline is
        # already reported and needs no new failure code. This is
        # intentionally narrow -- a type check, not a blanket
        # reclassification -- so a genuinely transient crash that happens to
        # surface as some other exception type still retries.
        return make_problem("VALIDATION_FAILED",
                            f"worker raised {type(exc).__name__}: {exc}"[:300])
    # Fully unreviewed exception type: never carry its message (real shadow
    # nightly attempt 16, a missing code asset surfaced as a bare
    # FileNotFoundError here) -- an arbitrary message cannot be vetted for a
    # credential, a price or a score (§5.2 redaction rule), unlike the three
    # branches above whose message shape is reviewed. Class name and code
    # location are structural, never a data value, so they are always safe.
    return _generic_problem(exc)


def _last_frame(exc: BaseException) -> str | None:
    frames = traceback.extract_tb(exc.__traceback__)
    return f"{frames[-1].filename}:{frames[-1].lineno}" if frames else None


def _generic_problem(exc: BaseException):
    """``exception_type`` and code ``location`` only -- see the redaction
    comment at the call site for why the message is never included here."""
    return make_problem(
        "WORKER_FAILED", "worker raised an unhandled exception",
        details={"exception_type": type(exc).__name__, "location": _last_frame(exc)})


def dispatch(worker, parameters, root, *, envelope=None):
    envelope = envelope or {}
    if worker == "incremental_refresh":
        from engine.v2.ops.incremental_data import run_refresh_worker
        return run_refresh_worker(parameters, root)
    if worker == "legacy_materialize":
        from engine.v2.ops.materialization_worker import run_materialize
        return run_materialize(parameters, root, envelope)
    if worker.startswith("legacy_"):
        from engine.v2.ops.legacy_actions import run_action
        values = parameters if isinstance(parameters, dict) else vars(parameters)
        # Last read-set gap fix (2026-09-15): only present for a
        # ``legacy_finality`` attempt whose launch resolved a
        # ``finality_check`` materialization (``snapshot_stages.prepare_launch``);
        # ``None`` for every other action, unchanged.
        output = run_action(worker, values, root, legacy_root=envelope.get("legacy_root"),
                            cross_check=envelope.get("finality_cross_check"))
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
    if worker == "adhoc_rescore":
        return _dispatch_adhoc_rescore(parameters, root)
    if worker in ("snapshot_import", "legacy_rebuild_candidate"):
        return _dispatch_snapshot_import(worker, parameters, root)
    if worker in ("ledger_export", "engineering_gate", "publication", "backup",
                  "decisions_supersede"):
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
    if worker == "experiment":
        return _dispatch_experiment(parameters, root)
    if worker == "training":
        from engine.v2.ops.training import run_training_worker
        return run_training_worker(parameters, root)
    if worker == "models_promote":
        from engine.v2.ops.training import run_promote_worker
        return run_promote_worker(parameters, root)
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


def _dispatch_adhoc_rescore(parameters, root):
    """P6 UD-2: re-score one already-captured (ScoreRequest,
    NativeScoreInputs) pair under the v2 no-fit guard. Both documents were
    materialized into staging by ``executor._materialize_inputs`` from
    ``parameters["input_bindings"]`` before this runs -- this function never
    fetches or fits anything, purely a bounded local computation, exactly
    like ``_dispatch_decision_evidence`` above.
    """
    from engine.v2.contracts import ScoreRequest
    from engine.v2.foundation import from_document, to_document
    from engine.v2.models.no_fit import no_fit_guard
    from engine.v2.ops.cli import _load_native_score_inputs
    from engine.v2.scoring.application import score_one

    request_doc = json.loads((root / "request.json").read_text())
    native_doc = json.loads((root / "native_inputs.json").read_text())
    request = from_document(ScoreRequest, request_doc)
    inputs = _load_native_score_inputs(native_doc)
    with no_fit_guard():
        record = score_one(request, inputs)
    (root / "record.json").write_text(
        json.dumps(to_document(record), sort_keys=True, separators=(",", ":")))
    return {"outputs": [{"name": "record", "path": "record.json",
                        "schema": "adhoc_rescore_record.v1.0"}],
            "completed_ids": list(parameters["expected_ids"])}


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

    ``ledger_export``, ``engineering_gate``, ``publication``, ``backup`` and
    P6-3's ``decisions_supersede`` never touch the catalog or the outbox from
    inside a subprocess; all of that real work happens in the supervisor's
    coordinator effect (``engine.v2.ops.effects_graph``, or
    ``engine.v2.ops.decision_commit.commit_supersede`` for the last), whose
    returned closure the supervisor runs inside the fenced ``commit_attempt``
    transaction -- exactly like every other coordinator effect. This worker
    only proves the attempt ran.

    The output is named ``<kind>_receipt``, never the bare kind name: the
    coordinator effect for ``ledger_export``/``engineering_gate`` publishes
    its OWN artifact under the bare kind name (the name downstream
    ``job_<id>#<name>`` bindings expect, e.g. ``#ledger_export``), and both
    this worker's output and the coordinator's ``extra_refs`` land in the
    same ``attempt_outputs`` row set keyed ``(attempt_id, name)`` — a shared
    name collides there. See ``supervisor.Service._commit_success``.
    """
    receipt = {"schema_version": "effect_receipt.v1.0", "kind": worker,
               "expected_ids": list(parameters["expected_ids"])}
    (root / "receipt.json").write_text(json.dumps(receipt))
    name = worker + "_receipt"
    return {"outputs": [{"name": name, "path": "receipt.json", "schema": "effect_receipt.v1.0"}],
            "completed_ids": list(parameters["expected_ids"]),
            "no_work": not parameters["expected_ids"]}


def _runner_headline(root):
    """The headline metrics a legacy runner wrote into its own results JSON.

    ``engine.evaluate`` writes ``results/metrics_<hash>.json`` (its
    ``headline`` block is exactly what ``experiments.lib.record_evaluation``
    puts in the ledger); a runner that writes no such file yields ``{}`` and
    the coordinator's ran row records ``metrics_source: unavailable``.
    """
    results = Path(root) / "results"
    if not results.is_dir():
        return {}
    for path in sorted(results.glob("metrics_*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        headline = document.get("headline") if isinstance(document, dict) else None
        if not isinstance(headline, dict):
            continue
        return {"mean": headline.get("mean", ""),
                "sharpe_trade": headline.get("sharpe_trade", "")}
    return {}


def _experiment_failure(receipt):
    """The typed ``OpsError`` for a failed experiment attempt.

    The runner's own typed problem (code + details, e.g. a nonzero
    returncode's stderr tail) is carried through the receipt so it survives
    ``run_experiment``'s status capture and reaches ``worker.main``'s private
    ``failure_details.json`` -- never the public failure message.
    """
    evidence = receipt.get("evidence") or {}
    details = {"status": receipt["status"], "error_code": evidence.get("error_code")}
    details.update(evidence.get("failure_details") or {})
    return fail(evidence.get("failure_code") or "VALIDATION_FAILED",
                "experiment run did not succeed", details=details)


def _dispatch_experiment(parameters, root):
    """P6 slice 10: run one experiment under admission. Pure function of
    ``parameters`` and staging, like ``_dispatch_adhoc_rescore`` — the runner
    subprocess writes only inside ``root``, never the shared legacy tree, so
    this carries no ``store_domains`` lease.

    ``no_ledger=False`` (P6 slice 11) selects ``mode="primary"`` for the
    coordinator's durable registration and ledger append. The runner
    subprocess itself is ALWAYS invoked through
    :func:`run_legacy_script`, which hardcodes ``--no-ledger``: a killed and
    retried attempt must never be able to double-append a CSV, and the one
    real ledger row is appended by the coordinator effect instead.

    A registered legacy runner that exits nonzero is a typed failure, never
    a success with whatever REPORT.md it happened to write first; its stderr
    tail travels in the problem's details.
    """
    from engine.v2.ops.experiments import (
        experiment_spec_from_document,
        run_experiment,
        synthetic_fixture_runner,
    )
    from engine.v2.ops.legacy_adapter import run_legacy_script

    mode = "smoke" if parameters.get("no_ledger", True) else "primary"
    document = json.loads((root / "spec.json").read_text())
    spec = experiment_spec_from_document(document)
    runner_id = parameters["runner"]
    if runner_id == "synthetic":
        runner, synthetic = synthetic_fixture_runner, True
    else:
        def runner(*, run_dir, no_ledger):
            completed = run_legacy_script(root, runner_id)
            if completed.returncode != 0:
                raise fail("VALIDATION_FAILED", "legacy experiment runner failed",
                           details={"returncode": completed.returncode,
                                    "stderr_tail": (completed.stderr or "")[-2000:]})
            return {"returncode": completed.returncode, "headline": _runner_headline(root)}
        synthetic = False
    receipt = run_experiment(spec, root, root, runner=runner, mode=mode, synthetic=synthetic)
    (root / "experiment_receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "succeeded":
        raise _experiment_failure(receipt)
    expected = parameters["expected_ids"]
    return {"outputs": [{"name": "experiment_receipt", "path": "experiment_receipt.json",
                         "schema": "experiment_receipt.v1.0"}],
            "completed_ids": list(expected), "no_work": False}


if __name__ == "__main__":
    raise SystemExit(main())
