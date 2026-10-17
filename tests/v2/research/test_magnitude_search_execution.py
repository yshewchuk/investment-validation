"""Bounded real native fits, causal ordering, sealed zero-retry and crash controls."""
# ruff: noqa: F811
import json
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.foundation import to_document
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_execution as execution
from experiments import magnitude_search_trials as trials
from tests.v2.research.magnitude_trial_support import trial_cases, trial_forecasts, trial_slots
from tests.v2.research.test_magnitude_search_trials import (
    base_source,  # noqa: F401
    case,  # noqa: F401
)

LIMITS = {"memory_bytes": 2 * 1024 ** 3, "cpu_seconds": 30, "wall_seconds": 60}


@contextmanager
def problem(code, message):
    with pytest.raises(OpsError) as caught:
        yield
    assert caught.value.problem == fail(code, message).problem


def inject_child(monkeypatch, body):
    """Inject only a synthetic estimator failure into an otherwise real fixed child."""
    original = subprocess.Popen
    children = []
    script = ("import sys; sys.argv = sys.argv[1:]; from experiments import magnitude_fitted as fitted; "
              + "exec(" + repr(body) + "); fitted.fit_magnitude_predictions = injected; "
              "from experiments.magnitude_search_worker import main; main()")
    def launch(args, **kwargs):
        child = original([args[0], "-c", script, *args[2:]], **kwargs)
        children.append((child, Path(args[3]).parent))
        return child
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    return children


def test_real_ridge_accounting_forecasts_before_labels_and_zero_work_replay(case, monkeypatch):
    args = trial_cases(case)[0]
    original = trials.load_magnitude_targets
    def labels(*a, **kw):
        slots = dict(trial_slots(case))
        assert sum(slot.startswith("fit-started:") for slot in slots) == 3
        assert sum(slot.startswith("fit-completed:") for slot in slots) == 3
        assert sum(slot.startswith("forecasts:") for slot in slots) == 1
        assert sum(slot.startswith("execution:") for slot in slots) == 1
        return original(*a, **kw)
    monkeypatch.setattr(trials, "load_magnitude_targets", labels)
    result = case.recorder.execute(*args, limits=LIMITS)
    assert result["evaluation_authority"] == "none"
    assert result["fit_attempt_count"] == result["completed_fit_count"] == 3
    assert [fit["horizon_sessions"] for fit in result["fits"]] == [5, 3, 1]
    assert result["terminal"]["metrics"] == {"count": 12, "rmse": pytest.approx(39 ** 0.5), "mae": 5.5}
    assert {row["predicted_absolute_move_pct"] for row in result["terminal"]["predictions"]} == {2.0}
    before = trial_slots(case)
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("replay launched fit"))
    monkeypatch.setattr(case.recorder, "_inputs", lambda *a: pytest.fail("replay opened native inputs"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("replay opened labels"))
    with no_fit_guard(), legacy_guard():
        assert case.recorder.execute(*args, limits=LIMITS) == result
    assert trial_slots(case) == before
    with problem("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs"):
        case.recorder.execute(*args, limits={**LIMITS, "cpu_seconds": 31})


@pytest.mark.parametrize("number", [1, 2, 3])
def test_candidate_fit_failure_retains_exact_prefix_and_replays(case, monkeypatch, number):
    inject_child(monkeypatch, "count = 0\ndef injected(recipe, train, labels, test):\n"
        "    global count\n    count += 1\n"
        f"    if count == {number}: raise ValueError('private estimator diagnostic')\n"
        "    return [2.0] * len(test)\n")
    args = trial_cases(case)[0]
    result = case.recorder.execute(*args, limits=LIMITS)
    assert result["evaluation_authority"] == "none"
    assert result["fit_attempt_count"] == number
    assert result["completed_fit_count"] == number - 1
    assert result["status"] == result["terminal"]["status"] == "failed"
    assert result["terminal"]["problem"] == to_document(
        fail("EXPERIMENT_VARIANT_FAILED", "search trial did not complete").problem)
    assert "private" not in str(result)
    before = trial_slots(case)
    replay = case.recorder.execute(*args, limits=LIMITS)
    assert replay["evaluation_authority"] == "none"
    assert replay == result
    assert trial_slots(case) == before


def test_actual_fit_entry_observes_committed_admission_before_child_ack(case, monkeypatch, tmp_path):
    catalog = next(row[2] for row in case.source.conn.execute("PRAGMA database_list") if row[1] == "main")
    observed = tmp_path / "fit-entry.jsonl"
    run_id = case.recorder.registration.run_id
    inject_child(monkeypatch, f"""
def injected(recipe, train, labels, test):
    import json
    import sqlite3
    from pathlib import Path
    with sqlite3.connect(Path({catalog!r}).as_uri() + '?mode=ro', uri=True) as conn:
        slots = [row[0] for row in conn.execute(
            'SELECT slot FROM magnitude_search_records WHERE run_id=?', ({run_id!r},))]
    counts = [sum(slot.startswith(prefix) for slot in slots)
              for prefix in ('fit-started:', 'fit-completed:')]
    with open({str(observed)!r}, 'a') as output:
        output.write(json.dumps(counts) + '\\n')
    return [2.0] * len(test)
""")
    append = case.recorder._append
    admissions = 0

    def delay_admission(slot, body, **kwargs):
        nonlocal admissions
        if slot.startswith("fit-started:"):
            admissions += 1
            # Without the worker's ACK wait, the child can enter and capture the
            # uncommitted count before this deliberately delayed write proceeds.
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if observed.exists() and len(observed.read_text().splitlines()) >= admissions:
                    break
                time.sleep(0.01)
        return append(slot, body, **kwargs)

    monkeypatch.setattr(case.recorder, "_append", delay_admission)
    result = case.recorder.execute(*trial_cases(case)[0], limits=LIMITS)
    assert result["completed_fit_count"] == 3
    assert [json.loads(line) for line in observed.read_text().splitlines()] == [[1, 0], [2, 1], [3, 2]]


@pytest.mark.parametrize("kind", ["memory", "allocation", "io", "runtime", "ops", "setup", "wall", "cpu"])
def test_resource_failures_leave_unfinished_claims_and_no_retry(case, monkeypatch, kind):
    bodies = {"memory": "raise MemoryError", "allocation": "bytearray(4 * 1024 ** 3)", "io": "raise OSError('private path')",
              "runtime": "raise RuntimeError('thread creation failed')",
              "ops": "from engine.v2.ops.errors import fail; raise fail('RESOURCE_UNAVAILABLE', 'private child failure')",
              "setup": "raise AssertionError('setup failure entered estimator')",
              "wall": "import time; time.sleep(10); return [2.0] * len(test)",
              "cpu": "\n    while True: pass"}
    prelude = ("import resource\ndef unavailable(*args):\n    raise ValueError('hard limit forbids request')\n"
               "resource.setrlimit = unavailable\n") if kind == "setup" else ""
    children = inject_child(monkeypatch, prelude + "def injected(recipe, train, labels, test):\n    " + bodies[kind] + "\n")
    limits = {**LIMITS, **({"wall_seconds": 2} if kind == "wall" else
                         {"cpu_seconds": 2, "wall_seconds": 10} if kind == "cpu" else {})}
    code, text = ("RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable") if kind in ("io", "runtime", "ops", "setup") else (
        "RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits")
    with problem(code, text):
        case.recorder.execute(*trial_cases(case)[0], limits=limits)
    child, scratch = children[0]
    assert child.poll() is not None
    assert not scratch.exists()
    if kind == "cpu":
        assert child.returncode == -signal.SIGXCPU
    if kind == "wall":
        assert child.returncode == -signal.SIGKILL
    slots = [slot for slot, _ in trial_slots(case)]
    if kind in ("ops", "setup"):
        assert sum(slot.startswith("fit-started:") for slot in slots) == (0 if kind == "setup" else 1)
    assert not any(slot.startswith("fit-completed:") for slot in slots)
    assert not any(slot.startswith(("terminal:", "execution:", "forecasts:")) for slot in slots)
    with problem("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"):
        case.recorder.freeze()
    with problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        case.recorder.execute(*trial_cases(case)[0], limits=limits)


def test_oversized_native_input_preserves_unfinished_claim_and_blocks_freeze(case, monkeypatch):
    monkeypatch.setattr(execution, "MAX_BYTES", 1)
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("oversized native input launched"))
    args = trial_cases(case)[0]
    with problem("RESOURCE_LIMIT_EXCEEDED", "search execution input exceeds its byte limit"):
        case.recorder.execute(*args, limits=LIMITS)
    slots = [slot for slot, _ in trial_slots(case)]
    for prefix in ("start:", "execution-start:", "inputs:"):
        assert sum(slot.startswith(prefix) for slot in slots) == 1
    assert not any(slot.startswith(("fit-started:", "execution:", "forecasts:", "terminal:")) for slot in slots)
    with problem("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"):
        case.recorder.freeze()
    with problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        case.recorder.execute(*args, limits=LIMITS)


@pytest.mark.parametrize("guard, error", [(no_fit_guard, RuntimeFitForbidden), (legacy_guard, LegacyRuntimeFitForbidden)])
def test_parent_no_fit_guard_precedes_claim_and_launch(case, monkeypatch, guard, error):
    before = trial_slots(case)
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("fit launched"))
    with guard(), pytest.raises(error, match="SearchTrials.execute"):
        case.recorder.execute(*trial_cases(case)[0], limits=LIMITS)
    assert trial_slots(case) == before


@pytest.mark.parametrize("limits", [{}, {**LIMITS, "cpu_seconds": True}, {**LIMITS, "memory_bytes": 0},
                                    {**LIMITS, "wall_seconds": 1.2}, {**LIMITS, "extra": 1}])
def test_invalid_resource_contract_precedes_claim(case, limits):
    before = trial_slots(case)
    with problem("INVALID_EXPERIMENT_SPEC", "search execution requires explicit positive integer resource limits"):
        case.recorder.execute(*trial_cases(case)[0], limits=limits)
    assert trial_slots(case) == before


def test_validation_admission_failure_keeps_native_result_unfinished(case, monkeypatch):
    args = trial_cases(case)[0]
    def refuse(*a, **kw):
        raise fail("EXPERIMENT_VARIANT_FAILED", "shared validation refusal")
    monkeypatch.setattr(trials, "load_magnitude_targets", refuse)
    with problem("EXPERIMENT_VARIANT_FAILED", "shared validation refusal"):
        case.recorder.execute(*args, limits=LIMITS)
    slots = [slot for slot, _ in trial_slots(case)]
    assert sum(slot.startswith("execution:") for slot in slots) == 1
    assert sum(slot.startswith("forecasts:") for slot in slots) == 1
    assert not any(slot.startswith("terminal:") for slot in slots)
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("retried unfinished trial"))
    with problem("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"):
        case.recorder.execute(*args, limits=LIMITS)
    with problem("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"):
        case.recorder.freeze()


def test_launch_error_is_infrastructure_and_cannot_be_submitted_as_candidate_failure(case, monkeypatch):
    def unavailable(*a, **kw):
        raise OSError("private launch")
    monkeypatch.setattr(execution.subprocess, "Popen", unavailable)
    args = trial_cases(case)[0]
    with problem("RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable"):
        case.recorder.execute(*args, limits=LIMITS)
    with problem("EXPERIMENT_IDENTITY_CONFLICT", "native execution must finish before recording"):
        case.recorder.record(*args, failure=fail("EXPERIMENT_VARIANT_FAILED", "candidate failure").problem)
    assert not any(slot.startswith("fit-started:") for slot, _ in trial_slots(case))


def test_submitted_terminal_cannot_gain_native_execution_authority(case, monkeypatch):
    args = trial_cases(case)[0]
    case.recorder.start(*args)
    case.recorder.record(*args, forecasts=trial_forecasts(case, args))
    before = trial_slots(case)
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("already submitted trial launched"))
    with problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        case.recorder.execute(*args, limits=LIMITS)
    assert trial_slots(case) == before


def test_source_seal_explicitly_binds_fixed_child(case):
    seal = case.recorder._read("seal")
    assert "experiments/magnitude_search_worker.py" in seal["recorder_sources"]
    assert "experiments/magnitude_search_execution.py" in seal["recorder_sources"]
    assert "experiments/magnitude_fitted.py" in seal["recorder_sources"]


def test_real_hgb_completes_each_horizon_and_scores_forecasts(base_source, tmp_path):
    import shutil

    from engine.v2.contracts import ArtifactRef
    from engine.v2.data.repository import Repository
    from engine.v2.foundation import ArtifactStore, SystemClock, from_document
    from engine.v2.ops.catalog import connect
    from experiments.magnitude_search_plan import seal_search_plan

    conn = connect(tmp_path / "hgb.sqlite")
    base_source.conn.backup(conn)
    shutil.copytree(base_source.store.root, tmp_path / "hgb-store")
    store = ArtifactStore(tmp_path / "hgb-store")
    request = {"schema_version": "magnitude_search_request.v1", "inner_years": {"2024": ["2023"], "2025": ["2023"]},
        "space": {"history_hgb": {"learning_rate": [0.05, 0.2], "max_iter": [2], "max_leaf_nodes": [3],
            "min_samples_leaf": [1], "l2_regularization": [0]}}, "max_candidates": 2, "max_fits": 18}
    sealed = seal_search_plan(conn, store, base_source.registration, request)
    recorder = trials.SearchTrials(conn, store, Repository(conn, store), base_source.registration,
                                  request, from_document(ArtifactRef, sealed["plan_ref"]))
    try:
        recorder.seal(clock=SystemClock())
        candidate = sealed["plan"]["candidates"][0]
        result = recorder.execute("2024", "2023", candidate["candidate_id"], limits=LIMITS)
        assert candidate["recipe"]["model"] == "history_hgb"
        assert result["fit_attempt_count"] == result["completed_fit_count"] == 3
        assert {row["predicted_absolute_move_pct"] for row in result["terminal"]["predictions"]} == {2.0}
        assert result["terminal"]["metrics"] == {"count": 12, "rmse": pytest.approx(39 ** 0.5), "mae": 5.5}
    finally:
        conn.close()
