"""Supervised legacy experiment integration for smoke and isolated runs."""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from engine.v2.foundation import content_hash
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import (
    environment_identity,
    file_hash,
    source_closure,
    worker_source_manifest,
)


def experiment_plan(spec_path: Path | str, *, smoke=True):
    """Create an immutable infrastructure plan for the registered smoke worker."""
    path = Path(spec_path)
    if not smoke:
        raise fail("INVALID_REQUEST", "production experiment activation is disabled")
    if not path.is_file() or path.is_symlink():
        raise fail("INPUT_CHANGED", "experiment specification is missing")
    document = json.loads(path.read_text())
    experiment_id = document.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise fail("INVALID_REQUEST", "experiment specification has no experiment_id")
    return {
        "schema_version": "operations_plan.v1.0",
        "kind": "artifact_check",
        "mode": "smoke",
        "effects": ["private_artifacts"],
        "parameters": {"expected_ids": ["experiment:" + experiment_id]},
        "input_refs": [],
        "blocked_prerequisites": [],
        "spec_hash": content_hash(document),
        "implementation_ref": content_hash(worker_source_manifest(
            Path(__file__).resolve().parents[3])),
        "environment_ref": content_hash(environment_identity()),
        "resource_class": "delivery",
        "status": "not_applicable",
        "reason": "registered experiment execution is deferred; this plan only checks plumbing",
    }


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    hypothesis: str
    primary_arm_id: str
    arms: tuple[str, ...]
    seed: int
    folds: tuple[str, ...]
    economic_params: dict
    price_source: str
    input_files: tuple[str, ...] = ()
    runner: str = "synthetic"

    @property
    def spec_hash(self) -> str:
        return content_hash({"experiment_id": self.experiment_id,
                             "hypothesis": self.hypothesis,
                             "primary_arm_id": self.primary_arm_id,
                             "arms": self.arms, "seed": self.seed,
                             "folds": self.folds, "economic_params": self.economic_params,
                             "price_source": self.price_source, "runner": self.runner})


@dataclass
class ExperimentReceipt:
    schema_version: str = "experiment_shadow.v1.0"
    experiment_id: str = ""
    spec_hash: str = ""
    input_hash: str = ""
    mode: str = "smoke"
    status: str = "running"
    evidence: dict = field(default_factory=dict)
    ledger_receipt: dict | None = None
    backup_receipt: dict | None = None

    def as_dict(self):
        return self.__dict__.copy()


def register_hypothesis(conn, spec: ExperimentSpec, input_hash: str, *, mode="smoke",
                        run_id=None):
    """Reserve one economic hypothesis; identical retries return its run ID."""
    from uuid import uuid4

    run_id = run_id or "exp_run_" + uuid4().hex
    payload_hash = content_hash({"spec": spec.spec_hash, "input": input_hash})
    with transaction(conn):
        old = conn.execute("SELECT run_id FROM experiment_runs WHERE spec_hash=? AND input_hash=? AND mode=?",
                           (spec.spec_hash, input_hash, mode)).fetchone()
        if old:
            return old[0]
        conn.execute("INSERT INTO experiment_runs VALUES (?,?,?,?,?,?)",
                     (run_id, spec.spec_hash, input_hash, mode, "{}", None))
        conn.execute("INSERT INTO hypotheses VALUES (?,?,?,?,?)",
                     (spec.spec_hash, input_hash, payload_hash, "{}", run_id))
    return run_id


def record_backup_pending(conn, run_id, error_code):
    with transaction(conn):
        row = conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                           (run_id,)).fetchone()
        if row is None:
            raise fail("INPUT_CHANGED", "experiment run is not registered")
        evidence = json.loads(row[0])
        evidence["backup"] = {"status": "backup_pending", "error_code": error_code}
        conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                     (json.dumps(evidence, sort_keys=True), run_id))


def retry_backup(conn, run_id, deliver):
    row = conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                       (run_id,)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "experiment run is not registered")
    result = deliver()
    with transaction(conn):
        evidence = json.loads(row[0])
        evidence["backup"] = result
        conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                     (json.dumps(evidence, sort_keys=True), run_id))
    return result


def input_manifest(root: Path | str, files: tuple[str, ...]) -> dict:
    base = Path(root).resolve()
    result = {}
    for relative in files:
        path = (base / relative).resolve()
        if not path.is_file() or path.is_symlink() or not str(path).startswith(str(base) + "/"):
            raise fail("INPUT_CHANGED", "experiment input is missing or indirect",
                       details={"path": relative})
        result[relative] = file_hash(path)
    return result


def capability_manifest(spec: ExperimentSpec, root: Path | str) -> dict:
    """Describe economic content and resolved dependencies before execution."""
    base = Path(root).resolve()
    runner = Path(spec.runner)
    source = source_closure(base, [str(runner)]) if runner.suffix == ".py" else {}
    inputs = input_manifest(base, spec.input_files)
    return {"schema_version": "experiment_capabilities.v1.0",
            "spec_hash": spec.spec_hash, "economic_params": spec.economic_params,
            "seed": spec.seed, "folds": list(spec.folds), "price_source": spec.price_source,
            "source_closure": source, "input_set": inputs}


#: Reviewed capability inventory, one entry per enabled legacy runner (§11.1).
#: Mechanical evidence (hashes, closure, flag detection) is re-derived on every
#: call; the reviewed statements here are the audit record, not runtime config.
RUNNER_INVENTORY = {
    "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py": {
        "spec_source": "experiments/EXP-182_d_1_gated_execution_parity_registered/spec.yaml",
        "declared_runtime_sources": (
            "experiments/EXP-181_d_1_gated_execution_parity/run.py",
        ),
        "ledger_write_behavior": ("appends experiments/LEDGER.csv rows via main(record=...) "
                                  "unless --no-ledger is passed; the adapter always passes it"),
        "registry_effects": "none; champion promotion is a separately authorized job",
        "report_path": "REPORT.md",
        "resumable_units": "none_declared",
        "backup_behavior": "none internal; the coordinator performs the single final private backup",
        "removal_phase": "phase-5 model extraction",
    },
}


def runner_manifest(root: Path | str, script: str) -> dict:
    """One reviewed capability manifest per enabled runner (§11.1)."""
    relative = str(PurePosixPath(script))
    entry = RUNNER_INVENTORY.get(relative)
    if entry is None:
        raise fail("INVALID_REQUEST", "experiment runner is not audited")
    base = Path(root).resolve()
    script_path = base / relative
    spec_path = base / entry["spec_source"]
    for path in (script_path, spec_path):
        if not path.is_file() or path.is_symlink():
            raise fail("INPUT_CHANGED", "registered runner is missing or indirect")
    closure = source_closure(base, [relative, *entry["declared_runtime_sources"]])
    sources = (script_path, *(base / rel for rel in entry["declared_runtime_sources"]))
    if not any("--no-ledger" in path.read_text() for path in sources):
        raise fail("INVALID_REQUEST", "registered runner lacks the no-ledger control")
    return {"schema_version": "runner_capability_manifest.v1.0",
            "runner": relative,
            "spec_source": entry["spec_source"],
            "spec_hash": file_hash(spec_path),
            "source_closure": closure,
            "no_ledger_support": True,
            "ledger_write_behavior": entry["ledger_write_behavior"],
            "registry_effects": entry["registry_effects"],
            "report_path": entry["report_path"],
            "resumable_units": entry["resumable_units"],
            "backup_behavior": entry["backup_behavior"],
            "removal_phase": entry["removal_phase"]}


def _call_runner(runner: Callable, run_dir: Path, *, no_ledger: bool):
    signature = inspect.signature(runner)
    if "no_ledger" not in signature.parameters:
        raise fail("INVALID_REQUEST", "experiment runner lacks required no-ledger control")
    kwargs = {"run_dir": run_dir, "no_ledger": no_ledger}
    accepted = {name for name in signature.parameters}
    return runner(**{key: value for key, value in kwargs.items() if key in accepted})


def _report_evidence(run_dir: Path) -> dict:
    report = run_dir / "REPORT.md"
    if not report.is_file():
        raise fail("VALIDATION_FAILED", "experiment did not generate REPORT.md")
    text = report.read_text()
    if "by engine.report v" not in text:
        raise fail("VALIDATION_FAILED", "experiment did not generate REPORT.md")
    return {"report": str(report), "report_hash": file_hash(report),
            "report_bytes": report.stat().st_size}


def run_experiment(spec: ExperimentSpec, root: Path | str, run_dir: Path | str,
                   *, runner: Callable, mode="smoke", backup: Callable | None = None,
                   synthetic=False) -> dict:
    """Run one isolated hypothesis; retries reuse its exact spec/input identity."""
    if mode not in ("smoke", "primary"):
        raise fail("INVALID_REQUEST", "unknown experiment mode")
    if mode == "smoke" and backup is not None:
        raise fail("INVALID_REQUEST", "smoke runs cannot request backup")
    base = Path(root).resolve()
    destination = Path(run_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    capabilities = capability_manifest(spec, base)
    input_hash = content_hash(capabilities["input_set"])
    receipt = ExperimentReceipt(experiment_id=spec.experiment_id,
                                spec_hash=spec.spec_hash, input_hash=input_hash,
                                mode=mode, evidence={"capabilities": capabilities})
    (destination / "CAPABILITIES.json").write_text(json.dumps(capabilities, indent=2,
                                                                sort_keys=True))
    try:
        result = _call_runner(runner, destination,
                              no_ledger=(mode == "smoke" or synthetic))
        report = _report_evidence(destination)
        receipt.evidence.update(report)
        receipt.evidence["runner_result"] = result if isinstance(result, dict) else str(result)
        receipt.evidence["synthetic"] = bool(synthetic)
        receipt.status = "succeeded"
    except Exception as exc:
        receipt.status = "failed"
        receipt.evidence["error_code"] = type(exc).__name__
        receipt.evidence["error"] = str(exc)[:240]
        return receipt.as_dict()
    if mode == "primary" and backup is not None and not synthetic:
        try:
            receipt.backup_receipt = backup(receipt.as_dict())
        except Exception as exc:
            receipt.backup_receipt = {"status": "backup_pending",
                                      "error_code": type(exc).__name__}
    elif mode == "smoke":
        receipt.ledger_receipt = {"status": "not_requested", "mode": "smoke"}
    return receipt.as_dict()


def synthetic_fixture_runner(*, run_dir: Path, no_ledger: bool) -> dict:
    """Tiny deterministic runner used to exercise report/evidence plumbing."""
    if not no_ledger:
        raise RuntimeError("synthetic runner requires smoke mode")
    (run_dir / "REPORT.md").write_text("# Synthetic infrastructure report\n\n"
                                         "*Generated by engine.report v1.0.*\n"
                                         "No trading metrics.\n")
    return {"fixture": True, "metrics": {}, "ledger": "disabled"}
