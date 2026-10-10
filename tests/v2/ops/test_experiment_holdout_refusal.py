"""End-to-end tests for the private holdout-refusal contract (issue #489).

The real ``experiment_trades.load_trades`` refusal travels through the real
worker dispatcher (``engine.v2.ops.worker._dispatch_experiment``), which
defers the ledger row to the fenced coordinator effect: the runner's partial
output is cleaned, one private receipt is written, and no shared-ledger row
exists yet -- replaying the same denial neither rewrites the receipt nor
touches the ledger. The dispatcher surfaces the typed ``OpsError`` and leaves
the refusal evidence in ``experiment_receipt.json``. Coordinator fencing and
row idempotency are covered by ``test_v2_ops_experiment_job.py``.
"""
from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from engine.v2.ops import effects_graph, experiments, legacy_adapter, worker
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import ExperimentSpec
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.research import experiment_trades
from tests.v2.ops.test_v2_ops_experiment_job import _claimed_primary_effect
from tests.v2.research.test_experiment_trades import _RANDOM, _holdout_snapshot

_HEADER = "id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted\n"


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="EXP-489-refusal", hypothesis="holdout refusal end to end",
        primary_arm_id="arm", arms=("arm",), seed=7, folds=("fold-1",),
        economic_params={}, price_source="synthetic", input_files=(),
        runner="synthetic")


def _spec_document(spec: ExperimentSpec) -> dict:
    """The ``root/spec.json`` fields ``_dispatch_experiment`` parses back."""
    return {"experiment_id": spec.experiment_id, "hypothesis": spec.hypothesis,
            "primary_arm_id": spec.primary_arm_id, "arms": list(spec.arms),
            "seed": spec.seed, "folds": list(spec.folds),
            "economic_params": dict(spec.economic_params),
            "price_source": spec.price_source,
            "input_files": list(spec.input_files), "runner": spec.runner}


def _refusing_runner(repository, snapshot, *, as_of_month="2025-01"):
    """Partial output first, then the real loader over the real snapshot."""
    def runner(*, run_dir, no_ledger):
        run_dir = Path(run_dir)
        (run_dir / "REPORT.md").write_text("# partial report\n")
        (run_dir / "ARMS.md").write_text("# partial arms\n")
        (run_dir / "results").mkdir(exist_ok=True)
        (run_dir / "results" / "metrics_probe.json").write_text('{"probe": true}\n')
        experiment_trades.load_trades(
            repository, snapshot, "STR-THRU", as_of_month=as_of_month,
            event_ids=[_RANDOM[0]])
    return runner


def _dispatch(monkeypatch, spec, root, repository, snapshot, *, mode, checkout,
              as_of_month="2025-01"):
    """Drive production: write ``spec.json`` and run the real worker dispatch.

    The spec document's runner stays ``synthetic`` so the variant identity falls
    back to the resolved spec hash, while ``parameters["runner"]`` selects the
    registered shape whose runner is the real-loader refusal above: only the
    registered-runner factory is replaced, by this closure, while the worker
    dispatcher and ``run_experiment`` stay real. The closure performs the real
    loader read, and the refusal path publishes the private receipt before the
    typed failure is raised; the worker defers the shared-ledger row to the
    fenced coordinator effect.
    """
    root.mkdir(parents=True, exist_ok=True)
    (checkout / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(json.dumps(_spec_document(spec)))
    monkeypatch.setattr(
        worker, "_registered_experiment_runner",
        lambda _root, _runner_id, _arm: _refusing_runner(
            repository, snapshot, as_of_month=as_of_month))
    parameters = {"runner": "registered-test-runner",
                  "expected_ids": ["experiment:" + spec.experiment_id],
                  "no_ledger": mode == "smoke",
                  "preregistration_root": str(checkout)}
    return worker._dispatch_experiment(parameters, root)


def _refused_rows(ledger):
    if not Path(ledger).is_file():
        return []
    with open(ledger, newline="") as fh:
        return [row for row in csv.DictReader(fh) if row["stage"] == "refused"]


def _assert_cleaned(run_dir):
    assert not (run_dir / "REPORT.md").exists()
    assert not (run_dir / "ARMS.md").exists()
    assert not list((run_dir / "results").glob("metrics_*.json"))


def _assert_sync_precedes_append(events):
    """Reject a refused-row append observed before the receipt dir sync."""
    synced = False
    for event in events:
        if event == "sync":
            synced = True
        elif event == "append":
            assert synced, (
                "the refused ledger row was appended before the receipt "
                "directory sync")


def test_real_loader_refusal_replay_and_private_receipt(tmp_path, monkeypatch):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM])
    root = tmp_path / "run"
    checkout = tmp_path / "checkout"
    spec = _spec()
    real_open = experiments.os.open
    real_fsync = experiments.os.fsync
    dir_fds = []
    fsynced_dir_fds = []

    def spy_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if isinstance(path, (str, os.PathLike)) and Path(path) == root:
            dir_fds.append(fd)
        return fd

    def spy_fsync(fd, *args, **kwargs):
        if fd in dir_fds:
            fsynced_dir_fds.append(fd)
        return real_fsync(fd, *args, **kwargs)

    monkeypatch.setattr(experiments.os, "open", spy_open)
    monkeypatch.setattr(experiments.os, "fsync", spy_fsync)
    monkeypatch.setenv("INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL",
                       str(root / "holdout_refusal_signal.json"))
    try:
        with pytest.raises(OpsError) as first_error:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        receipt = root / "holdout_refusal_receipt.json"
        first_bytes = receipt.read_bytes()
        assert dir_fds, "the refusal never opened the receipt directory"
        assert fsynced_dir_fds, "the receipt directory fd was not fsynced"
        result = worker._failure_result(root, first_error.value)
        assert result["failure"] == "HOLDOUT_ACCESS_DENIED"
        assert "refusal_receipt" not in json.dumps(result)
        failure_details = json.loads(
            (root / "diagnostics" / "failure_details.json").read_text())
        assert "holdout_exclusions" not in failure_details
        assert "holdout_exclusions" not in failure_details["refusal_receipt"]
        private_receipt = json.loads(first_bytes)
        assert "holdout_exclusions" in private_receipt
        sanitized_private = dict(private_receipt)
        sanitized_private.pop("holdout_exclusions")
        assert failure_details["refusal_receipt"] == sanitized_private
        signal = json.loads((root / "holdout_refusal_signal.json").read_text())
        assert set(signal) == {"schema_version", "failure_code",
                               *experiments.REFUSAL_PIN_FIELDS}
        assert signal["schema_version"] == "holdout_refusal_signal.v1"
        assert signal["failure_code"] == "HOLDOUT_ACCESS_DENIED"
        signal_pins = {name: signal[name] for name in experiments.REFUSAL_PIN_FIELDS}
        assert signal_pins == {name: private_receipt[name]
                               for name in experiments.REFUSAL_PIN_FIELDS}
        signal_text = json.dumps(signal)
        assert "holdout_exclusions" not in signal_text
        assert _RANDOM[0] not in signal_text
        assert worker._holdout_refusal_signal(root) == signal_pins
        fsynced_dir_fds.clear()
        with pytest.raises(OpsError) as second_error:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        assert fsynced_dir_fds, \
            "the matching receipt replay must sync its directory"
        assert receipt.read_bytes() == first_bytes
        with pytest.raises(OpsError) as conflict_error:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout, as_of_month="2025-05")
        assert conflict_error.value.code == "IDEMPOTENCY_CONFLICT"
        assert receipt.read_bytes() == first_bytes
    finally:
        conn.close()

    for error in (first_error.value, second_error.value):
        assert error.code == "HOLDOUT_ACCESS_DENIED"
        assert error.problem.details["refusal_receipt"]
    document = json.loads(first_bytes)
    assert document["schema_version"] == experiments.REFUSAL_RECEIPT_SCHEMA
    assert document["status"] == "refused"
    assert document["failure_code"] == "HOLDOUT_ACCESS_DENIED"
    assert document["variant_id"] == spec.spec_hash
    assert document["snapshot_id"] == snapshot.snapshot_id
    assert document["holdout_as_of_month"] == "2025-01"
    assert document["random_membership_version"] == "canonical-event-sha256.v1"
    assert document["rolling_membership_version"] == "calendar-months.v1"
    stored = json.loads((root / "experiment_receipt.json").read_text())
    assert stored["status"] == "refused"
    assert stored["evidence"]["failure_code"] == "HOLDOUT_ACCESS_DENIED"
    _assert_cleaned(root)
    assert _refused_rows(checkout / "experiments" / "LEDGER.csv") == []


def test_signal_directory_sync_failure_removes_signal_before_worker_accepts(
        tmp_path, monkeypatch):
    """A failed publication cannot leave a signal the worker accepts."""
    import stat

    signal = tmp_path / "holdout_refusal_signal.json"
    monkeypatch.setenv("INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL", str(signal))
    directory = tmp_path.stat()
    real_fsync = experiment_trades.os.fsync
    failed = False

    def fail_signal_directory_sync(fd):
        nonlocal failed
        info = os.fstat(fd)
        if (stat.S_ISDIR(info.st_mode) and info.st_dev == directory.st_dev
                and info.st_ino == directory.st_ino and not failed):
            failed = True
            raise OSError("simulated signal directory sync failure")
        return real_fsync(fd)

    monkeypatch.setattr(experiment_trades.os, "fsync", fail_signal_directory_sync)
    pins = {"snapshot_id": "snapshot-489", "holdout_as_of_month": "2025-01",
            "random_membership_version": "canonical-event-sha256.v1",
            "rolling_membership_version": "calendar-months.v1"}
    with pytest.raises(OSError, match="simulated signal directory sync failure"):
        experiment_trades._emit_holdout_refusal_signal(pins)
    assert failed
    assert not signal.exists()
    assert worker._holdout_refusal_signal(tmp_path) is None


def test_signal_directory_open_failure_removes_signal_before_worker_accepts(
        tmp_path, monkeypatch):
    """A failed parent-directory open after replacement cannot leave a signal."""
    signal = tmp_path / "holdout_refusal_signal.json"
    monkeypatch.setenv("INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL", str(signal))
    real_open = experiment_trades.os.open
    failed = False

    def fail_signal_directory_open(path, *args, **kwargs):
        nonlocal failed
        if Path(path) == tmp_path and not failed:
            failed = True
            raise OSError("simulated signal directory open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(experiment_trades.os, "open", fail_signal_directory_open)
    pins = {"snapshot_id": "snapshot-489", "holdout_as_of_month": "2025-01",
            "random_membership_version": "canonical-event-sha256.v1",
            "rolling_membership_version": "calendar-months.v1"}
    with pytest.raises(OSError, match="simulated signal directory open failure"):
        experiment_trades._emit_holdout_refusal_signal(pins)
    assert failed
    assert not signal.exists()
    assert worker._holdout_refusal_signal(tmp_path) is None


def test_real_loader_receipt_sync_failure_replays_before_ledger_append(
        tmp_path, monkeypatch):
    """The real loader retries a visible receipt only after syncing its directory.

    The refused row's production path is exercised, not a direct helper call:
    a real claimed primary attempt runs the refusal callback through
    ``commit_attempt`` so the coordinator's own fenced append writes the row,
    and a spy on ``_append_refusal_row`` records its place in the observed
    order. A planted-defect replay that skips the receipt directory sync must
    be rejected by the same ordering assertion.
    """
    import stat

    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM])
    root = tmp_path / "run"
    checkout = tmp_path / "checkout"
    spec = _spec()
    ledger = checkout / "experiments" / "LEDGER.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(_HEADER + f"{spec.experiment_id},planned,2025-01-01,planned,,,False\n")
    root.mkdir()
    real_open = experiments.os.open
    real_fsync = experiments.os.fsync
    real_replace = experiments.os.replace
    real_fsync_directory = experiments.fsync_directory
    real_append = experiments._append_refusal_row
    run_dir_fds = []
    receipt_replaced = False
    receipt_sync_failed = False
    skip_receipt_sync = False
    events = []

    def spy_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if isinstance(path, (str, os.PathLike)) and Path(path) == root:
            run_dir_fds.append(fd)
        return fd

    def spy_replace(src, dst, *args, **kwargs):
        nonlocal receipt_replaced
        result = real_replace(src, dst, *args, **kwargs)
        if Path(dst).name == "holdout_refusal_receipt.json":
            receipt_replaced = True
        return result

    def spy_fsync(fd):
        nonlocal receipt_sync_failed
        if fd in run_dir_fds and receipt_replaced and not receipt_sync_failed:
            receipt_sync_failed = True
            raise OSError("simulated receipt directory sync failure")
        return real_fsync(fd)

    def spy_fsync_directory(path):
        if skip_receipt_sync:
            return
        result = real_fsync_directory(path)
        if Path(path) == root:
            events.append("sync")
        return result

    def spy_append(*args, **kwargs):
        events.append("append")
        return real_append(*args, **kwargs)

    monkeypatch.setattr(experiments.os, "open", spy_open)
    monkeypatch.setattr(experiments.os, "replace", spy_replace)
    monkeypatch.setattr(experiments.os, "fsync", spy_fsync)
    monkeypatch.setattr(experiments, "fsync_directory", spy_fsync_directory)
    monkeypatch.setattr(experiments, "_append_refusal_row", spy_append)
    monkeypatch.setenv("INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL",
                       str(root / "holdout_refusal_signal.json"))
    try:
        with pytest.raises(OSError, match="simulated receipt directory sync failure"):
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        receipt = root / "holdout_refusal_receipt.json"
        first_bytes = receipt.read_bytes()
        assert receipt_sync_failed
        assert _refused_rows(ledger) == []

        with pytest.raises(OpsError) as denied:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        assert denied.value.code == "HOLDOUT_ACCESS_DENIED"
        assert events == ["sync"], (
            "matching receipt replay must sync its directory before returning refusal")
        assert receipt.read_bytes() == first_bytes

        refusal_receipt = dict(denied.value.problem.details["refusal_receipt"])
        conn_, clock_, claim, _, _ = _claimed_primary_effect(
            tmp_path, key="refusal-order", document=_spec_document(spec),
            checkout=checkout)
        try:
            failure_effects = effects_graph._experiment_refusal_failure_effect(
                claim, denied.value.problem, code_source=checkout,
                refusal_receipt=refusal_receipt)
            assert failure_effects is not None
            commit_attempt(conn_, claim.attempt_id, claim.fence,
                           Outcome(False, "verified_dead", 1, denied.value.problem),
                           clock=clock_, failure_effects=failure_effects)
        finally:
            conn_.close()
        assert events == ["sync", "append"], (
            "the coordinator append must follow the receipt directory sync")
        _assert_sync_precedes_append(events)
        assert len(_refused_rows(ledger)) == 1

        events.clear()
        skip_receipt_sync = True
        with pytest.raises(OpsError) as defect_denied:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        assert defect_denied.value.code == "HOLDOUT_ACCESS_DENIED"
        assert events == [], "the planted-defect replay recorded a receipt sync"
        defect_checkout = tmp_path / "defect" / "checkout"
        defect_ledger = defect_checkout / "experiments" / "LEDGER.csv"
        defect_ledger.parent.mkdir(parents=True)
        defect_ledger.write_text(
            _HEADER + f"{spec.experiment_id},planned,2025-01-01,planned,,,False\n")
        defect_receipt = dict(
            defect_denied.value.problem.details["refusal_receipt"])
        conn_def, clock_def, defect_claim, _, _ = _claimed_primary_effect(
            tmp_path / "defect", key="refusal-order-defect",
            document=_spec_document(spec), checkout=defect_checkout)
        try:
            defect_effects = effects_graph._experiment_refusal_failure_effect(
                defect_claim, defect_denied.value.problem,
                code_source=defect_checkout, refusal_receipt=defect_receipt)
            assert defect_effects is not None
            commit_attempt(conn_def, defect_claim.attempt_id, defect_claim.fence,
                           Outcome(False, "verified_dead", 1,
                                   defect_denied.value.problem),
                           clock=clock_def, failure_effects=defect_effects)
        finally:
            conn_def.close()
        assert events == ["append"], (
            "the planted defect must yield an append with no preceding sync")
        with pytest.raises(AssertionError):
            _assert_sync_precedes_append(events)
        assert len(_refused_rows(defect_ledger)) == 1
        assert len(_refused_rows(ledger)) == 1
        _assert_cleaned(root)
    finally:
        conn.close()


def test_registered_runner_consumer_validates_signal_and_falls_back(
        tmp_path, monkeypatch):
    """The registered-runner reader, isolated at its subprocess boundary.

    ``worker._registered_experiment_runner``'s production closure, its
    ``_holdout_refusal_signal`` validation and the typed conversion are under
    test; only ``legacy_adapter.run_legacy_script`` -- the subprocess edge the
    real child loader crosses -- is replaced by a narrow stub returning a
    failed process result. The stub seeds only the private transport document
    (the sidecar the real refusal above writes) under the given ``run_dir``;
    no repository, snapshot or database is touched here.
    """
    from types import SimpleNamespace

    runner_id, primary_arm_id = next(
        (registered, arm)
        for registered, entry in experiments.RUNNER_INVENTORY.items()
        for arm in entry.get("fixed_arm_args", {}))
    pins = {"snapshot_id": "snap-registered", "holdout_as_of_month": "2025-01",
            "random_membership_version": "canonical-event-sha256.v1",
            "rolling_membership_version": "calendar-months.v1"}
    transport = {"schema_version": "holdout_refusal_signal.v1",
                 "failure_code": "HOLDOUT_ACCESS_DENIED", **pins,
                 "holdout_exclusions": [{"event_id": _RANDOM[0],
                                         "memberships": ["random"]}]}
    run_dir = tmp_path / "staged"
    run_dir.mkdir()

    def stub_adapter(staging_root, called_runner_id, *, args,
                     declared_runtime_sources):
        if transport is not None:
            (Path(staging_root) / "holdout_refusal_signal.json").write_text(
                json.dumps(transport))
        return SimpleNamespace(returncode=1, stderr="private loader traceback")

    monkeypatch.setattr(legacy_adapter, "run_legacy_script", stub_adapter)
    runner = worker._registered_experiment_runner(tmp_path, runner_id, primary_arm_id)
    with pytest.raises(OpsError) as denied:
        runner(run_dir=run_dir, no_ledger=True)
    assert denied.value.code == "HOLDOUT_ACCESS_DENIED"
    assert denied.value.problem.details == pins
    leaked = json.dumps(denied.value.problem.details)
    assert "private loader traceback" not in leaked
    assert "holdout_exclusions" not in leaked
    assert _RANDOM[0] not in leaked

    transport = None
    (run_dir / "holdout_refusal_signal.json").write_text(json.dumps(
        {"schema_version": "holdout_refusal_signal.v2", **pins}))
    with pytest.raises(OpsError) as generic:
        runner(run_dir=run_dir, no_ledger=True)
    assert generic.value.code == "VALIDATION_FAILED"
    assert generic.value.problem.details == {
        "returncode": 1, "stderr_tail": "private loader traceback"}


def test_run_legacy_script_sets_private_signal_path_and_clears_stale_file(
        tmp_path, monkeypatch):
    """Adapter configuration/cleanup coverage for ``run_legacy_script``.

    Only the subprocess edge is stubbed here: the adapter must point the
    private ``INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL`` at the resolved run
    directory and unlink any stale sidecar from a previous attempt BEFORE
    launching, so a child never reads a recycled refusal. This is adapter
    configuration/cleanup coverage only; the real producer is already
    exercised by ``test_real_loader_refusal_replay_and_private_receipt`` and
    the registered-runner consumer is covered separately by
    ``test_registered_runner_consumer_validates_signal_and_falls_back``.
    """
    import subprocess

    runner_id, primary_arm_id = next(
        (registered, arm)
        for registered, entry in experiments.RUNNER_INVENTORY.items()
        for arm in entry.get("fixed_arm_args", {}))
    entry = experiments.RUNNER_INVENTORY[runner_id]
    run_dir = tmp_path.resolve()
    signal = run_dir / "holdout_refusal_signal.json"
    signal.write_bytes(b'{"schema_version": "stale-previous-attempt"}\n')

    captured = {}

    def fake_run(command, *, cwd, env, check, capture_output, text, timeout):
        captured.update(cwd=cwd, env=env, stale_present_at_launch=signal.exists())
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(run_dir))
    result = legacy_adapter.run_legacy_script(
        tmp_path, runner_id,
        args=entry["fixed_arm_args"][primary_arm_id],
        declared_runtime_sources=entry["declared_runtime_sources"])

    assert result.returncode == 0
    assert captured["cwd"] == run_dir
    assert captured["env"]["INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL"] == str(signal)
    assert not captured["stale_present_at_launch"], \
        "the stale signal file survived until the subprocess launch"
    assert not signal.exists()


def test_real_subprocess_refusal_signal_round_trip(tmp_path, monkeypatch):
    """One genuine process boundary for the refusal signal, with no stub.

    Real adapter + real producer helper + real Python child + real reader:
    ``legacy_adapter._FIXED_ARM_RUNNER`` is staged at its registered relative
    path under a private ``tmp_path`` run directory (test staging only, never
    the checkout wrapper) with a synthetic child that imports the real
    ``experiment_trades._emit_holdout_refusal_signal``, publishes four
    nonblank synthetic pins and exits nonzero without printing the pins or
    any event IDs. The real ``run_legacy_script`` points the signal env var
    at the staged run directory and launches the child; the real
    ``worker._holdout_refusal_signal`` reads the sidecar back. The real
    loader call-site stays covered by
    ``test_real_loader_refusal_replay_and_private_receipt``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("PYTHONPATH", str(repo_root))
    pins = {"snapshot_id": "snap-subprocess", "holdout_as_of_month": "2025-01",
            "random_membership_version": "canonical-event-sha256.v1",
            "rolling_membership_version": "calendar-months.v1"}
    run_dir = tmp_path.resolve() / "run"
    script = run_dir / legacy_adapter._FIXED_ARM_RUNNER
    script.parent.mkdir(parents=True)
    script.write_text(
        "import sys\n"
        "from engine.v2.research.experiment_trades import "
        "_emit_holdout_refusal_signal\n"
        f"_emit_holdout_refusal_signal({json.dumps(pins)})\n"
        "sys.exit(1)\n")

    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(run_dir))
    result = legacy_adapter.run_legacy_script(
        run_dir, legacy_adapter._FIXED_ARM_RUNNER, args=(),
        declared_runtime_sources=())

    assert result.returncode != 0
    assert worker._holdout_refusal_signal(run_dir) == pins
    signal = json.loads((run_dir / "holdout_refusal_signal.json").read_text())
    assert set(signal) == {"schema_version", "failure_code",
                           *experiments.REFUSAL_PIN_FIELDS}
    assert signal == {"schema_version": "holdout_refusal_signal.v1",
                      "failure_code": "HOLDOUT_ACCESS_DENIED", **pins}
    child_output = result.stdout + result.stderr
    for value in [*pins.values(), _RANDOM[0]]:
        assert value not in child_output


def test_smoke_refusal_never_touches_the_supplied_ledger(tmp_path, monkeypatch):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM])
    root = tmp_path / "run"
    checkout = tmp_path / "checkout"
    ledger = checkout / "experiments" / "LEDGER.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(_HEADER.encode() + b"recognizable,sentinel\n")
    seed = ledger.read_bytes()
    spec = _spec()
    try:
        with pytest.raises(OpsError) as error:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="smoke", checkout=checkout)
    finally:
        conn.close()

    assert error.value.code == "HOLDOUT_ACCESS_DENIED"
    assert error.value.problem.details["refusal_receipt"]
    assert ledger.read_bytes() == seed
    document = json.loads((root / "holdout_refusal_receipt.json").read_text())
    assert document["variant_id"] == spec.spec_hash
    assert document["snapshot_id"] == snapshot.snapshot_id
    _assert_cleaned(root)


def test_failure_ordering_and_replay_recovery(tmp_path, monkeypatch):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM])
    root = tmp_path / "run"
    checkout = tmp_path / "checkout"
    ledger = checkout / "experiments" / "LEDGER.csv"
    spec = _spec()
    real_replace = experiments.os.replace

    def guarded_replace(src, dst, *args, **kwargs):
        if Path(dst).name == "holdout_refusal_receipt.json":
            raise OSError("receipt write blocked")
        return real_replace(src, dst, *args, **kwargs)

    try:
        monkeypatch.setattr(experiments.os, "replace", guarded_replace)
        with pytest.raises(OSError, match="receipt write blocked"):
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        assert not (root / "holdout_refusal_receipt.json").exists()
        assert not ledger.exists()
        assert _refused_rows(ledger) == []

        monkeypatch.setattr(experiments.os, "replace", real_replace)
        with pytest.raises(OpsError) as error:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        assert error.value.code == "HOLDOUT_ACCESS_DENIED"
        assert (root / "holdout_refusal_receipt.json").is_file()
        assert _refused_rows(ledger) == []
    finally:
        conn.close()


def test_refusal_signal_replace_failure_propagates_before_receipt(
        tmp_path, monkeypatch):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM])
    root = tmp_path / "run"
    root.mkdir(parents=True, exist_ok=True)
    signal = root / "holdout_refusal_signal.json"
    real_replace = experiment_trades.os.replace

    def guarded_replace(src, dst, *args, **kwargs):
        if Path(dst) == signal:
            raise OSError("signal write blocked")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setenv("INVESTMENT_PLAN_HOLDOUT_REFUSAL_SIGNAL", str(signal))
    try:
        monkeypatch.setattr(experiment_trades.os, "replace", guarded_replace)
        with pytest.raises(OSError, match="signal write blocked"):
            experiment_trades.load_trades(
                repository, snapshot, "STR-THRU", as_of_month="2025-01",
                event_ids=[_RANDOM[0]])
        assert not signal.exists()
        assert not (root / "holdout_refusal_receipt.json").exists()
        assert _refused_rows(
            tmp_path / "checkout" / "experiments" / "LEDGER.csv") == []
    finally:
        conn.close()


def test_concurrent_refusal_row_append_is_exactly_once(tmp_path):
    ledger = tmp_path / "LEDGER.csv"
    ledger.write_text(_HEADER)
    spec = _spec()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                experiments._append_refusal_row,
                spec.experiment_id, spec.spec_hash, ledger)
            for _ in range(8)]
        for future in futures:
            future.result()

    rows = _refused_rows(ledger)
    assert len(rows) == 1
    assert rows[0]["id"] == spec.experiment_id
    assert rows[0]["spec_hash"] == spec.spec_hash
    with pytest.raises(OpsError) as conflict:
        experiments._append_refusal_row(
            spec.experiment_id, "different-resolved-variant", ledger)
    assert conflict.value.code == "IDEMPOTENCY_CONFLICT"
    rows = _refused_rows(ledger)
    assert len(rows) == 1
    assert rows[0]["spec_hash"] == spec.spec_hash
