"""Fixed subprocess entrypoint. The launch gate opens only after PID persistence.

No arbitrary modules or executables are accepted. Output is a small protocol
message on an inherited pipe; payloads stay in the assigned staging directory.
"""
from __future__ import annotations

import hashlib
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
    if worker in ("incremental_refresh", "computed_moves_refresh",
                  "forward_calendar_refresh"):
        return _dispatch_refresh(worker, parameters, root)
    if worker == "legacy_materialize":
        from engine.v2.ops.materialization_worker import run_materialize
        return run_materialize(parameters, root, envelope)
    if worker.startswith("legacy_"):
        return _dispatch_legacy_action(worker, parameters, root, envelope)
    if worker == "decision_evidence":
        return _dispatch_decision_evidence(parameters, root)
    if worker == "adhoc_rescore":
        return _dispatch_adhoc_rescore(parameters, root)
    if worker == "native_score_batch":
        return _dispatch_native_score_batch(parameters, root)
    if worker == "native_parity":
        return _dispatch_native_parity(parameters, root)
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
    if worker in ("training", "models_promote", "models_rollback"):
        return _dispatch_model_worker(worker, parameters, root)
    raise ValueError("unsupported worker")


def _dispatch_legacy_action(worker, parameters, root, envelope):
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


def _dispatch_model_worker(worker, parameters, root):
    from engine.v2.ops.training import (
        run_promote_worker,
        run_rollback_worker,
        run_training_worker,
    )

    runners = {"training": run_training_worker, "models_promote": run_promote_worker,
               "models_rollback": run_rollback_worker}
    return runners[worker](parameters, root)


def _dispatch_native_score_batch(parameters, root):
    from engine.v2.ops.native_score_batch import run_native_score_batch_worker
    return run_native_score_batch_worker(parameters, root)


def _dispatch_native_parity(parameters, root):
    from engine.v2.ops.native_parity_report import run_native_parity_worker
    return run_native_parity_worker(parameters, root)


def _dispatch_refresh(worker, parameters, root):
    if worker == "incremental_refresh":
        from engine.v2.ops.incremental_data import run_refresh_worker
        return run_refresh_worker(parameters, root)
    from engine.v2.ops import calendar_moves_jobs
    if worker == "computed_moves_refresh":
        return calendar_moves_jobs.run_computed_moves_worker(parameters, root)
    return calendar_moves_jobs.run_forward_calendar_worker(parameters, root)


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
    from engine.v2.ops.native.input_decoding import _load_native_score_inputs
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
    ``failure_details.json`` -- never the public failure message. A
    ``status="refused"`` receipt adds the pinned private ``refusal_receipt``
    on the same channel, so the typed ``HOLDOUT_ACCESS_DENIED`` reaches the
    supervisor without becoming a public attempt output. Diagnostics carry
    shallow copies with the evidence-bearing ``holdout_exclusions`` omitted;
    the untouched originals live only in the private on-disk receipt.
    """
    evidence = receipt.get("evidence") or {}
    details = {"status": receipt["status"], "error_code": evidence.get("error_code")}
    failure_details = dict(evidence.get("failure_details") or {})
    failure_details.pop("holdout_exclusions", None)
    details.update(failure_details)
    if "refusal_receipt" in evidence:
        refusal_receipt = dict(evidence["refusal_receipt"])
        refusal_receipt.pop("holdout_exclusions", None)
        details["refusal_receipt"] = refusal_receipt
    return fail(evidence.get("failure_code") or "VALIDATION_FAILED",
                "experiment run did not succeed", details=details)


def _declared_experiment_sources(runner_id: str) -> tuple[str, ...]:
    """Return the declared runtime source paths for one registered runner."""
    from engine.v2.ops.experiments import RUNNER_INVENTORY

    entry = RUNNER_INVENTORY.get(runner_id)
    sources = entry.get("declared_runtime_sources") if isinstance(entry, dict) else None
    if (not isinstance(sources, (list, tuple)) or not sources
            or not all(isinstance(relative, str) and relative for relative in sources)):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "registered runner has no valid inventory source record",
                   details={"runner": runner_id})
    return tuple(sources)


_HOLDOUT_REFUSAL_SIGNAL_SCHEMA = "holdout_refusal_signal.v1"
_HOLDOUT_REFUSAL_PINS = ("snapshot_id", "holdout_as_of_month",
                         "random_membership_version", "rolling_membership_version")


def _holdout_refusal_signal(run_dir):
    """The validated private refusal pins from the child loader's sidecar.

    ``legacy_adapter.run_legacy_script`` exports
    ``INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL=<run_dir>/holdout_refusal_signal.json``
    and ``experiment_trades.load_trades`` atomically writes that signal when
    the real child loader refuses holdout data. Only the exact
    schema/failure-code document whose four membership pins are all nonblank
    strings is accepted; unknown keys, stderr, and event IDs never travel in
    the typed problem. Missing, unreadable, malformed, wrong-schema or
    wrong-code files, and invalid pins, all yield ``None`` so the generic
    runner failure — with no sidecar values — is unchanged.

    The signal is confirmed against its sibling ``.sha256`` sidecar: the
    producer writes that file with the SHA-256 hex digest of the exact signal
    bytes only after the signal's parent directory was successfully fsynced,
    so a signal left visible by a failed rollback that never reached that
    fsync is rejected. The digest must equal
    ``hashlib.sha256(signal_bytes).hexdigest()``; a missing, malformed or
    mismatched digest yields ``None`` like every other invalid shape.
    """
    signal_path = Path(run_dir) / "holdout_refusal_signal.json"
    try:
        signal_bytes = signal_path.read_bytes()
        digest = (Path(run_dir) / "holdout_refusal_signal.json.sha256").read_text(
            encoding="ascii").strip()
    except (OSError, ValueError):
        return None
    if digest != hashlib.sha256(signal_bytes).hexdigest():
        return None
    try:
        document = json.loads(signal_bytes)
    except (OSError, ValueError):
        return None
    if (not isinstance(document, dict)
            or document.get("schema_version") != _HOLDOUT_REFUSAL_SIGNAL_SCHEMA
            or document.get("failure_code") != "HOLDOUT_ACCESS_DENIED"):
        return None
    pins = {name: document.get(name) for name in _HOLDOUT_REFUSAL_PINS}
    if not all(isinstance(value, str) and value.strip() for value in pins.values()):
        return None
    return pins


def _registered_experiment_runner(root, runner_id, primary_arm_id):
    """The runner closure for a registered legacy runner (P6 slice 10).

    The one audited selector tuple is ``RUNNER_INVENTORY[runner_id]
    ["fixed_arm_args"][primary_arm_id]``; anything else is a non-retryable
    ``INVALID_EXPERIMENT_SPEC`` naming the runner and arm, raised before any
    resolved plan is written or the runner is invoked. The closure preserves
    the nonzero-return refusal (typed ``VALIDATION_FAILED`` with the stderr
    tail) and the ledger headline. A nonzero exit whose staged run directory
    carries a valid holdout refusal sidecar is instead the typed
    ``HOLDOUT_ACCESS_DENIED`` naming only the four validated pins, which
    ``run_experiment`` captures as the private refusal receipt.
    """
    from engine.v2.ops.experiments import RUNNER_INVENTORY
    from engine.v2.ops.legacy_adapter import run_legacy_script

    fixed_arm_args = RUNNER_INVENTORY.get(runner_id, {}).get("fixed_arm_args", {})
    selected_args = fixed_arm_args.get(primary_arm_id)
    if selected_args is None:
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "registered runner has no audited selector for the primary arm",
                   details={"runner": runner_id, "primary_arm_id": primary_arm_id})

    sources = _declared_experiment_sources(runner_id)

    def runner(*, run_dir, no_ledger):
        completed = run_legacy_script(run_dir, runner_id, args=selected_args,
                                      declared_runtime_sources=sources)
        if completed.returncode != 0:
            pins = _holdout_refusal_signal(run_dir)
            if pins is not None:
                raise fail("HOLDOUT_ACCESS_DENIED",
                           "registered experiment loader refused holdout data",
                           details=pins)
            raise fail("VALIDATION_FAILED", "legacy experiment runner failed",
                       details={"returncode": completed.returncode,
                                "stderr_tail": (completed.stderr or "")[-2000:]})
        return {"returncode": completed.returncode, "headline": _runner_headline(run_dir)}
    return runner


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

    ``resolve_experiment_plan`` runs before any run artifact is written and
    its canonical ``json_bytes`` are persisted as ``resolved_experiment_plan``
    beside the receipt -- the same plan value ``run_experiment`` resolves
    deterministically for the callable, byte-for-byte. Both runner shapes
    here (the synthetic fixture, the legacy-script wrapper) declare no
    ``execution_plan`` channel, so a non-empty economic stance is refused,
    as the resolver's ``INVALID_EXPERIMENT_SPEC``, before the plan is
    persisted or the script invoked; fixed single-variant runs carry empty
    economics.
    """
    from engine.v2.ops.experiments import (
        expected_variant_identity,
        experiment_spec_from_document,
        experiments_ledger_path,
        resolve_experiment_plan,
        run_experiment,
        synthetic_fixture_runner,
    )

    mode = "smoke" if parameters.get("no_ledger", True) else "primary"
    document = json.loads((root / "spec.json").read_text())
    spec = experiment_spec_from_document(document)
    plan = resolve_experiment_plan(spec)
    if plan.economic_params:
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "the experiment worker's runner declares no execution_plan input",
                   details={"economic_keys": sorted(plan.economic_params)})
    # The variant identity is bound to the run here: the registered legacy
    # runner's own spec.yaml identity for a primary run (obtained from the
    # preregistration root the plan recorded), the resolved spec hash for a
    # smoke run or a synthetic primary fallback.
    # Primary runs receive declared dependencies as staged bindings before dispatch.
    checkout_root = parameters.get("preregistration_root")
    variant_id = expected_variant_identity(Path(checkout_root) if checkout_root else root,
                                           spec, mode)
    runner_id = parameters["runner"]
    if runner_id == "synthetic":
        runner, synthetic = synthetic_fixture_runner, True
    else:
        runner = _registered_experiment_runner(root, runner_id, spec.primary_arm_id)
        synthetic = False
    (root / "resolved_experiment_plan.json").write_bytes(plan.json_bytes())
    refusal_ledger_path = (experiments_ledger_path(checkout_root)
                           if mode == "primary" and checkout_root else None)
    receipt = run_experiment(spec, root, root, runner=runner, mode=mode, synthetic=synthetic,
                             resolved_plan=plan, variant_id=variant_id,
                              refusal_ledger_path=refusal_ledger_path,
                              defer_refusal_ledger=True)
    (root / "experiment_receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "succeeded":
        raise _experiment_failure(receipt)
    expected = parameters["expected_ids"]
    return {"outputs": [{"name": "experiment_receipt", "path": "experiment_receipt.json",
                         "schema": "experiment_receipt.v1.0"},
                        {"name": "resolved_experiment_plan",
                         "path": "resolved_experiment_plan.json",
                         "schema": "experiment_execution_plan.v1.0"},
                        {"name": "experiment_variant_report", "path": "REPORT.md",
                         "schema": "experiment_variant_report.v1.0"}],
            "completed_ids": list(expected), "no_work": False}


if __name__ == "__main__":
    raise SystemExit(main())
