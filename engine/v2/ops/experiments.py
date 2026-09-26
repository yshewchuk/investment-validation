"""Supervised legacy experiment integration for smoke and isolated runs."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from engine.v2.foundation import content_hash
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import (
    environment_identity,
    file_hash,
    source_closure,
    worker_source_manifest,
)
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named


def default_checkout_root() -> Path:
    """The code checkout whose ``experiments/`` tree owns pre-registration.

    One expression, used by the plan-time check and by the CLI's submit-time
    re-check when a plan carries no recorded root. The coordinator's own
    append resolves the same checkout from ``Service.store_root`` (which
    defaults to ``code_source``), never from the operations store root.
    """
    return Path(__file__).resolve().parents[3]


def experiment_plan(spec_path: Path | str, *, smoke=True, root: Path | str | None = None):
    """Create an immutable plan for a supervised smoke or primary experiment run."""
    profile = profile_named(DEFAULT_POLICY, "experiment_heavy")
    threads = profile.thread_count or profile.cpu_count
    path = Path(spec_path)
    if not path.is_file() or path.is_symlink():
        raise fail("INPUT_CHANGED", "experiment specification is missing")
    document = json.loads(path.read_text())
    experiment_id = document.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise fail("INVALID_REQUEST", "experiment specification has no experiment_id")
    runner = document.get("runner")
    if not isinstance(runner, str) or not runner:
        raise fail("INVALID_REQUEST", "experiment specification has no runner")
    plan = {
        "schema_version": "operations_plan.v1.0",
        "kind": "experiment",
        "mode": "smoke" if smoke else "primary",
        "effects": ["staged"],
        "parameters": {"expected_ids": ["experiment:" + experiment_id],
                       "input_bindings": None,
                       "runner": runner, "no_ledger": smoke},
        "input_refs": [],
        "blocked_prerequisites": [],
        "spec_hash": content_hash(document),
        "implementation_ref": content_hash(worker_source_manifest(
            Path(__file__).resolve().parents[3])),
        "environment_ref": content_hash(environment_identity(threads)),
        "resource_class": "experiment_heavy",
        "spec_document": document,
    }
    if not smoke:
        # A primary plan records the one checkout root its pre-registration
        # check read. ``ops submit`` re-checks against this exact path, so a
        # plan and its submission can never bind two different ledgers.
        checkout_root = (Path(root) if root is not None else default_checkout_root()).resolve()
        require_preregistration(checkout_root, experiment_spec_from_document(document))
        plan["preregistration_root"] = str(checkout_root)
        plan["parameters"]["preregistration_root"] = str(checkout_root)
    return plan


def experiment_spec_from_document(document: dict) -> ExperimentSpec:
    """The typed ``ExperimentSpec`` a parsed spec document names.

    ``experiment_plan`` hashes the raw document; this is the one place the
    typed form -- needed by both the worker dispatch and the coordinator's
    durable attempt record -- is built, so a missing required field is refused
    once, as an ``OpsError``, never a bare ``KeyError``.
    """
    for name in ("hypothesis", "primary_arm_id", "economic_params", "price_source"):
        if name not in document or document[name] is None:
            raise fail("INVALID_REQUEST", "experiment specification is missing a required field",
                       details={"field": name})
    return ExperimentSpec(
        experiment_id=document.get("experiment_id", ""),
        hypothesis=document["hypothesis"],
        primary_arm_id=document["primary_arm_id"],
        arms=tuple(document.get("arms", ())),
        seed=document.get("seed", 0),
        folds=tuple(document.get("folds", ())),
        economic_params=document["economic_params"],
        price_source=document["price_source"],
        input_files=tuple(document.get("input_files", ())),
        runner=document.get("runner", "synthetic"),
    )


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


def experiments_ledger_path(repo_root: Path | str) -> Path:
    """The one resolver for the pre-registration ledger.

    Always under a *checkout* root -- the tree that owns the PLANNED row and
    the registered runner's ``spec.yaml`` -- never the operations store root
    (``ArtifactStore.root``, by default ``data/operations``). The plan-time
    check and the coordinator's durable "ran" append both resolve through
    this one function, so the row they read and the row they write can never
    be two different files.
    """
    return Path(repo_root) / "experiments" / "LEDGER.csv"


def planned_rows(repo_root: Path | str, experiment_id: str) -> list[dict]:
    """Every ``stage="planned"`` ledger row for ``experiment_id``.

    Plain CSV read -- no legacy import, no adapter entry
    (``checks/legacy_adapters.json`` is at its 75/75 ceiling).
    """
    ledger = experiments_ledger_path(repo_root)
    if not ledger.is_file():
        return []
    import csv
    with open(ledger, newline="") as fh:
        return [row for row in csv.DictReader(fh)
                if row.get("id") == experiment_id and row.get("stage") == "planned"]


def planned_row_exists(root: Path | str, experiment_id: str) -> bool:
    """True iff the checkout ledger has a PLANNED row for ``experiment_id``."""
    return bool(planned_rows(root, experiment_id))


def legacy_spec_hash(spec: dict) -> str:
    """The planned row's ``spec_hash``, by the same algorithm.

    ``experiments.lib.spec_hash`` delegates to the (legacy)
    ``engine.evaluate.spec_hash``; importing that module from ``engine/v2``
    would need an adapter entry at the 75/75 ceiling, so the pure algorithm
    is copied here byte-for-byte: everything except ``id`` and
    ``preregistered_at``, canonical ``json.dumps``, sha256.
    ``tests/test_experiments.py`` pins the two implementations to the same
    answer, and this is what makes planned and ran rows join.
    """
    document = {key: value for key, value in spec.items()
                if key not in ("id", "preregistered_at")}
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def registered_spec_hash(checkout_root: Path | str, spec: ExperimentSpec) -> str | None:
    """The legacy pre-registration hash of the registered runner's spec.yaml.

    ``None`` for a runner with no audited spec source (the synthetic
    fixture): there is no legacy spec identity to bind. Resolved under the
    *checkout*, so the row's ``spec_hash`` is never empty just because the
    operations store root has no ``experiments/`` tree.
    """
    entry = RUNNER_INVENTORY.get(spec.runner)
    if entry is None:
        return None
    spec_path = Path(checkout_root) / entry["spec_source"]
    if not spec_path.is_file() or spec_path.is_symlink():
        raise fail("INPUT_CHANGED",
                   "registered runner's pre-registered specification is missing or indirect")
    from experiments.lib import load_spec
    return legacy_spec_hash(load_spec(spec_path))


def require_preregistration(root: Path | str, spec: ExperimentSpec) -> None:
    """Refuse activation (mode="primary") with no PLANNED ledger row for
    spec.experiment_id, and refuse a spec that no longer hashes to the
    registered row's ``spec_hash``.

    The first half closes the gap in engine.evaluate's own guard (it
    silently no-ops when there are zero planned rows -- see docs memory
    prereg-guard-noops-without-planned-row / EXP-173..177), which only ever
    fires INSIDE the runner subprocess, after real work has already
    happened. The second half binds the spec: the same
    :func:`legacy_spec_hash` the planned row was written with is recomputed
    here at plan time AND again at submit, so a spec edited after
    registration is refused before any job exists. A synthetic runner has
    no legacy spec to compare and checks only the row's existence.
    """
    rows = planned_rows(root, spec.experiment_id)
    if not rows:
        raise fail("INVALID_REQUEST",
                   "experiment has no PLANNED ledger row; scaffold it with "
                   "experiments/new_experiment.py before activating a real run",
                   details={"experiment_id": spec.experiment_id})
    expected = registered_spec_hash(root, spec)
    if expected is not None and expected not in {row.get("spec_hash") for row in rows}:
        raise fail("SPEC_CHANGED",
                   "experiment specification changed after pre-registration; scaffold a "
                   "new experiment for the changed hypothesis",
                   details={"experiment_id": spec.experiment_id})


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


def register_hypothesis_in_transaction(conn, spec: ExperimentSpec, input_hash: str, *,
                                       mode="smoke", run_id=None):
    """The body of :func:`register_hypothesis`, without its own transaction.

    The experiment coordinator effect runs inside the fenced
    ``commit_attempt`` transaction (``catalog.transaction`` refuses a nested
    one outright), so the registration has to be available in that shape.
    """
    from uuid import uuid4

    run_id = run_id or "exp_run_" + uuid4().hex
    payload_hash = content_hash({"spec": spec.spec_hash, "input": input_hash})
    old = conn.execute("SELECT run_id FROM experiment_runs WHERE spec_hash=? AND input_hash=? AND mode=?",
                       (spec.spec_hash, input_hash, mode)).fetchone()
    if old:
        return old[0], False
    if mode == "primary":
        existing = conn.execute(
            "SELECT run_id, input_hash FROM hypotheses WHERE spec_hash=?",
            (spec.spec_hash,)).fetchone()
        if existing:
            if existing["input_hash"] != input_hash:
                raise fail("IDEMPOTENCY_CONFLICT",
                           "hypothesis is already registered with a different input",
                           details={"spec_hash": spec.spec_hash})
            return existing["run_id"], False
    conn.execute("INSERT INTO experiment_runs VALUES (?,?,?,?,?,?)",
                 (run_id, spec.spec_hash, input_hash, mode, "{}", None))
    if mode == "primary":
        conn.execute("INSERT INTO hypotheses VALUES (?,?,?,?,?)",
                     (spec.spec_hash, input_hash, payload_hash, "{}", run_id))
    return run_id, True


def register_hypothesis(conn, spec: ExperimentSpec, input_hash: str, *, mode="smoke",
                        run_id=None):
    """Reserve one economic run; identical retries return its run ID.

    Smoke mode has no promotion authority: it holds a namespace in
    ``experiment_runs`` only and never creates a ``hypotheses`` row, even on
    request. Primary mode reserves exactly one hypothesis per ``spec_hash``;
    a retry with the same input returns it, and the same ``spec_hash`` asked
    again with a different input is refused rather than raising the raw
    ``IntegrityError`` a second unconditional insert would produce.

    Returns ``(run_id, created)``. ``created`` is deliberately NOT what
    decides the ledger append: a retry after a partially completed effect
    must still find the one run and append the missing row.
    """
    with transaction(conn):
        return register_hypothesis_in_transaction(conn, spec, input_hash,
                                                  mode=mode, run_id=run_id)


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
        if isinstance(exc, OpsError):
            receipt.evidence["failure_code"] = exc.code
            # The typed problem's details (e.g. a runner's nonzero returncode
            # and stderr tail) survive the status capture so the worker can
            # re-raise them into the private diagnostics path.
            receipt.evidence["failure_details"] = dict(exc.problem.details)
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
