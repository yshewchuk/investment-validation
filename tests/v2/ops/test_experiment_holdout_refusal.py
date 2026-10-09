"""End-to-end tests for the private holdout-refusal contract (issue #489).

The real ``experiment_trades.load_trades`` refusal travels through the real
worker dispatcher (``engine.v2.ops.worker._dispatch_experiment``): the runner's
partial output is cleaned, one private receipt is written, and exactly one
``stage="refused"`` ledger row is appended -- replaying the same denial neither
rewrites the receipt nor adds a second row. The dispatcher surfaces the typed
``OpsError`` and leaves the refusal evidence in ``experiment_receipt.json``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from engine.v2.ops import experiments, worker
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import ExperimentSpec
from engine.v2.research import experiment_trades
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


def _refusing_runner(repository, snapshot):
    """Partial output first, then the real loader over the real snapshot."""
    def runner(*, run_dir, no_ledger):
        run_dir = Path(run_dir)
        (run_dir / "REPORT.md").write_text("# partial report\n")
        (run_dir / "ARMS.md").write_text("# partial arms\n")
        (run_dir / "results").mkdir(exist_ok=True)
        (run_dir / "results" / "metrics_probe.json").write_text('{"probe": true}\n')
        experiment_trades.load_trades(
            repository, snapshot, "STR-THRU", as_of_month="2025-01",
            event_ids=[_RANDOM[0]])
    return runner


def _dispatch(monkeypatch, spec, root, repository, snapshot, *, mode, checkout):
    """Drive production: write ``spec.json`` and run the real worker dispatch.

    The spec document's runner stays ``synthetic`` so the variant identity falls
    back to the resolved spec hash, while ``parameters["runner"]`` selects the
    registered shape whose runner is the real-loader refusal above: only the
    registered-runner factory is replaced, by this closure, while the worker
    dispatcher and ``run_experiment`` stay real. The closure performs the real
    loader read, and the refusal path appends the private refusal row before
    the typed failure is raised.
    """
    root.mkdir(parents=True, exist_ok=True)
    (checkout / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(json.dumps(_spec_document(spec)))
    monkeypatch.setattr(
        worker, "_registered_experiment_runner",
        lambda _root, _runner_id, _arm: _refusing_runner(repository, snapshot))
    parameters = {"runner": "registered-test-runner",
                  "expected_ids": ["experiment:" + spec.experiment_id],
                  "no_ledger": mode == "smoke",
                  "preregistration_root": str(checkout)}
    return worker._dispatch_experiment(parameters, root)


def _refused_rows(ledger):
    with open(ledger, newline="") as fh:
        return [row for row in csv.DictReader(fh) if row["stage"] == "refused"]


def _assert_cleaned(run_dir):
    assert not (run_dir / "REPORT.md").exists()
    assert not (run_dir / "ARMS.md").exists()
    assert not list((run_dir / "results").glob("metrics_*.json"))


def test_real_loader_refusal_replay_and_private_receipt(tmp_path, monkeypatch):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM])
    root = tmp_path / "run"
    checkout = tmp_path / "checkout"
    spec = _spec()
    try:
        with pytest.raises(OpsError) as first_error:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        receipt = root / "holdout_refusal_receipt.json"
        first_bytes = receipt.read_bytes()
        result = worker._failure_result(root, first_error.value)
        assert result["failure"] == "HOLDOUT_ACCESS_DENIED"
        assert "refusal_receipt" not in json.dumps(result)
        failure_details = json.loads(
            (root / "diagnostics" / "failure_details.json").read_text())
        assert failure_details["refusal_receipt"] == json.loads(first_bytes)
        with pytest.raises(OpsError) as second_error:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
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
    rows = _refused_rows(checkout / "experiments" / "LEDGER.csv")
    assert len(rows) == 1
    assert rows[0]["spec_hash"] == spec.spec_hash
    assert rows[0]["oos_mean_mid"] == "" and rows[0]["sharpe_trade"] == ""
    assert rows[0]["promoted"] == "False"


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
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(_HEADER)
    seed = ledger.read_bytes()
    spec = _spec()
    real_write_text = Path.write_text

    def guarded(self, *args, **kwargs):
        if self.name == "holdout_refusal_receipt.json":
            raise OSError("receipt write blocked")
        return real_write_text(self, *args, **kwargs)

    try:
        monkeypatch.setattr(Path, "write_text", guarded)
        with pytest.raises(OSError, match="receipt write blocked"):
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        assert ledger.read_bytes() == seed
        assert not (root / "holdout_refusal_receipt.json").exists()

        monkeypatch.setattr(Path, "write_text", real_write_text)
        from experiments import lib

        real_append = lib.ledger_append

        def broken_append(rows, path=None):
            raise OSError("ledger append blocked")

        monkeypatch.setattr(lib, "ledger_append", broken_append)
        with pytest.raises(OSError, match="ledger append blocked"):
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        assert (root / "holdout_refusal_receipt.json").is_file()
        assert ledger.read_bytes() == seed

        monkeypatch.setattr(lib, "ledger_append", real_append)
        with pytest.raises(OpsError) as error:
            _dispatch(monkeypatch, spec, root, repository, snapshot,
                      mode="primary", checkout=checkout)
        assert error.value.code == "HOLDOUT_ACCESS_DENIED"
        rows = _refused_rows(ledger)
        assert len(rows) == 1
        assert rows[0]["spec_hash"] == spec.spec_hash
    finally:
        conn.close()
