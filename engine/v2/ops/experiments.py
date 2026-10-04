"""Supervised legacy experiment integration for smoke and isolated runs."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
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

#: The closed top-level field set of a spec document: a key this resolver
#: cannot interpret is refused, never quietly ignored into a runner call.
SPEC_FIELDS = frozenset({"experiment_id", "hypothesis", "primary_arm_id", "arms",
                         "seed", "folds", "economic_params", "price_source",
                         "input_files", "runner"})

#: The only economic declaration with a defined execution meaning today;
#: any other key is an unused declaration, refused before the runner.
SUPPORTED_ECONOMIC_KEYS = frozenset({"fill"})


def default_checkout_root() -> Path:
    """The code checkout whose ``experiments/`` tree owns pre-registration.

    One expression, used by the plan-time check and by the CLI's submit-time
    re-check when a plan carries no recorded root. The coordinator's own
    append resolves the same checkout from ``Service.store_root`` (which
    defaults to ``code_source``), never from the operations store root.
    """
    return Path(__file__).resolve().parents[3]


def _require_mapping(document) -> None:
    """Refuse a non-mapping spec document as the resolver's typed code, before
    any ``.get`` or key enumeration touches it. The plan entry point and direct
    ``experiment_spec_from_document`` callers both pass through here."""
    if not isinstance(document, Mapping):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "experiment specification document is not a mapping",
                   details={"type": type(document).__name__})


def experiment_plan(spec_path: Path | str, *, smoke=True, root: Path | str | None = None):
    """Create an immutable plan for a supervised smoke or primary experiment run."""
    profile = profile_named(DEFAULT_POLICY, "experiment_heavy")
    threads = profile.thread_count or profile.cpu_count
    path = Path(spec_path)
    if not path.is_file() or path.is_symlink():
        raise fail("INPUT_CHANGED", "experiment specification is missing")
    document = json.loads(path.read_text())
    _require_mapping(document)
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
    _require_mapping(document)
    unknown = sorted(set(document) - SPEC_FIELDS)
    if unknown:
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "experiment specification declares fields this resolver does not know",
                   details={"fields": unknown})
    for name in ("hypothesis", "primary_arm_id", "economic_params", "price_source"):
        if name not in document or document[name] is None:
            raise fail("INVALID_REQUEST", "experiment specification is missing a required field",
                       details={"field": name})
    for name in ("arms", "folds"):
        if name in document and not isinstance(document[name], list):
            raise fail("INVALID_EXPERIMENT_SPEC", "arms and folds must be JSON arrays",
                       details={"field": name, "type": type(document[name]).__name__})
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


def _freeze(value):
    """Immutable snapshot over FRESH dicts: a proxy over the parsed document's
    own mapping would still let that document mutate the "immutable" plan."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _require_immutable_economics(spec: ExperimentSpec,
                                 plan: ResolvedExperimentPlan) -> None:
    """Accept a supplied plan's economics only in the shapes ``_freeze``
    itself produces: immutable JSON scalars, tuples of immutable values, and
    read-only mapping proxies with immutable keys and values. Built-in dicts,
    lists, and any other mutable container are refused as the same typed
    refusal -- canonical bytes can match while the nested values are still
    editable behind the adopted plan's back."""
    def immutable(value) -> bool:
        if isinstance(value, MappingProxyType):
            return all(immutable(key) and immutable(item)
                       for key, item in value.items())
        if isinstance(value, tuple):
            return all(immutable(item) for item in value)
        return isinstance(value, (str, int, float, bool, type(None)))
    if not immutable(plan.economic_params):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "supplied resolved plan carries mutable nested economics",
                   details={"experiment_id": spec.experiment_id})


def _thaw(value):
    """Plain JSON structures rebuilt at serialization time, never stored."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ResolvedExperimentPlan:
    """The one execution plan an ``ExperimentSpec`` resolves to.

    Deeply immutable (a frozen dataclass over frozen mappings), and its
    ``json_bytes`` are canonical: sorted keys, compact separators, one stable
    UTF-8 encoding. A runner that declares ``execution_plan`` receives this
    plan -- never the raw document it was parsed from.
    """
    schema_version: str
    experiment_id: str
    arms: tuple[str, ...]
    seed: int
    folds: tuple[str, ...]
    economic_params: Mapping
    price_source: str
    runner: str
    #: Set by :func:`resolve_experiment_plan` on the exact object it returns.
    #: A non-init field whose ``dataclasses.replace`` default is ``False``, so a
    #: rebuilt or hand-built plan never carries the resolver's provenance and
    #: :func:`_adopt_resolved_plan` refuses adoption before any filesystem
    #: effect. Excluded from ``repr``/``compare`` to keep canonical identity.
    _provenance: bool = field(default=False, init=False, repr=False, compare=False)

    def as_document(self) -> dict:
        return {"schema_version": self.schema_version, "experiment_id": self.experiment_id,
                "arms": list(self.arms), "seed": self.seed, "folds": list(self.folds),
                "economic_params": _thaw(self.economic_params),
                "price_source": self.price_source, "runner": self.runner}

    def json_bytes(self) -> bytes:
        return json.dumps(self.as_document(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")


def _validate_plan_fields(spec: ExperimentSpec) -> None:
    """Refuse values that cannot become the plan's immutable, correctly
    typed fields, before any plan is built or any field is frozen. A
    dataclass annotation checks nothing at runtime, so a JSON object
    supplied as ``seed`` used to travel into ``ResolvedExperimentPlan`` by
    reference and mutating its source rewrote the plan's canonical bytes
    after the fact. ``experiment_id``/``price_source``/``runner`` must be
    non-empty strings; ``arms``/``folds`` must be tuples of strings -- the
    fields' declared type, so a mutable list or a bare string is refused no
    matter what it contains (the parser converts valid document arrays to
    tuples); ``seed`` must be an ``int``, never a ``bool``; and
    every nested economic value must be JSON-representable -- finite
    numbers, strings, booleans, null, lists and string-keyed mappings --
    anything else refused rather than frozen into a plan and handed to a
    runner. Assumes the caller's mapping check already established
    ``economic_params`` is a mapping."""
    def json_value(value) -> bool:
        if value is None or isinstance(value, (bool, int, str)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(json_value(item) for item in value)
        if isinstance(value, Mapping):
            return all(isinstance(key, str) and json_value(item)
                       for key, item in value.items())
        return False

    for name in ("experiment_id", "price_source", "runner"):
        value = getattr(spec, name)
        if not isinstance(value, str) or not value:
            raise fail("INVALID_EXPERIMENT_SPEC",
                       "experiment plan field must be a non-empty string",
                       details={"field": name, "type": type(value).__name__})
    for name in ("arms", "folds"):
        value = getattr(spec, name)
        if not isinstance(value, tuple) or not all(
                isinstance(item, str) for item in value):
            raise fail("INVALID_EXPERIMENT_SPEC",
                       "experiment plan field must be a tuple of strings",
                       details={"field": name, "type": type(value).__name__})
    if isinstance(spec.seed, bool) or not isinstance(spec.seed, int):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "experiment seed must be an integer",
                   details={"type": type(spec.seed).__name__})
    if not all(isinstance(key, str) and json_value(item)
               for key, item in spec.economic_params.items()):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "experiment economic parameters must be string-keyed "
                   "JSON-representable values",
                   details={"type": type(spec.economic_params).__name__})


def resolve_experiment_plan(spec: ExperimentSpec) -> ResolvedExperimentPlan:
    """Refuse malformed plan fields and economically unused declarations,
    then freeze the one plan."""
    if not isinstance(spec.economic_params, Mapping):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "economic_params must be a mapping",
                   details={"type": type(spec.economic_params).__name__})
    # Order matters: the field check types the economic keys before the sort.
    _validate_plan_fields(spec)
    unused = sorted(set(spec.economic_params) - SUPPORTED_ECONOMIC_KEYS)
    if unused:
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "experiment declares economically unused parameters",
                   details={"keys": unused})
    plan = ResolvedExperimentPlan(
        schema_version="experiment_execution_plan.v1.0", experiment_id=spec.experiment_id,
        arms=_freeze(list(spec.arms)), seed=spec.seed, folds=_freeze(list(spec.folds)),
        economic_params=_freeze(spec.economic_params), price_source=spec.price_source,
        runner=spec.runner)
    object.__setattr__(plan, "_provenance", True)
    return plan


def _adopt_resolved_plan(spec: ExperimentSpec,
                         resolved_plan: ResolvedExperimentPlan) -> ResolvedExperimentPlan:
    """Seal plan adoption to the exact object ``resolve_experiment_plan`` stamped.

    Only a resolver-produced plan carries the private provenance marker; a
    ``dataclasses.replace`` rebuild or a hand-built plan loses it, so a
    byte-equal impostor hiding a mutable ``economic_params`` dict, a
    ``MappingProxyType`` over a caller-held mapping, or mutable ``arms``/other
    fields is refused here -- the resolver's typed ``INVALID_EXPERIMENT_SPEC``
    -- before ``run_experiment`` touches the filesystem. The immutable-economics
    shape check and canonical-byte equality are kept as defense in depth, and
    the accepted object is returned unchanged so its identity reaches the
    runner.
    """
    if not resolved_plan._provenance:
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "supplied resolved plan was not produced by this resolver",
                   details={"experiment_id": spec.experiment_id})
    _require_immutable_economics(spec, resolved_plan)
    if resolved_plan.json_bytes() != resolve_experiment_plan(spec).json_bytes():
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "supplied resolved plan is not the plan this spec resolves to",
                   details={"experiment_id": spec.experiment_id})
    return resolved_plan


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
    "experiments/EXP-184_str_thru_gate_promotion_confirmatory_val_registered/run.py": {
        "spec_source": "experiments/EXP-184_str_thru_gate_promotion_confirmatory_val_registered/spec.yaml",
        "declared_runtime_sources": (
            "experiments/EXP-147_str_thru_gate_promotion_confirmatory_val/run.py",
        ),
        "ledger_write_behavior": ("appends experiments/LEDGER.csv rows via main()'s "
                                  "--no-ledger gate unless disabled; the adapter always "
                                  "passes --no-ledger"),
        "registry_effects": ("promotion candidate: gate_midfill_str_thru_forecast_analog "
                             "vs champion gate_midfill_str_thru, per promotion_target"),
        "report_path": "REPORT.md",
        "resumable_units": "none_declared",
        "backup_behavior": "none internal; the coordinator performs the single final private backup",
        "removal_phase": "phase-5 model extraction",
    },
    "experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered/run.py": {
        "spec_source": "experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered/spec.yaml",
        "declared_runtime_sources": (
            "experiments/EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline/run.py",
        ),
        "ledger_write_behavior": (
            "appends experiments/LEDGER.csv rows via main()'s --no-ledger gate unless "
            "disabled -- both the candidate's own ungated/arms/primary rows and the "
            "champion grid cell's PLANNED row -- the adapter always passes --no-ledger"
        ),
        "registry_effects": (
            "challenger evaluation only: native_nan vs champion gate_midfill_str_runup "
            "(stored threshold 0.0725137593996064) re-scored on the identical v2 trades, "
            "per promotion_target; no registry write here"
        ),
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


def _execution_plan_compatibility(runner: Callable,
                                  execution_plan: ResolvedExperimentPlan) -> None:
    """Refuse a legacy callable that cannot carry the plan's economic stance.

    A callable without an ``execution_plan`` parameter has no channel for
    ``economic_params``, so silently dropping a declared economic stance is
    refused as the resolver's typed ``INVALID_EXPERIMENT_SPEC``. Called by
    :func:`run_experiment` as a preflight -- before the run directory or any
    evidence exists -- and again by :func:`_call_runner` as defense in depth.
    """
    if not execution_plan.economic_params:
        return
    if "execution_plan" in {name for name in inspect.signature(runner).parameters}:
        return
    raise fail("INVALID_EXPERIMENT_SPEC",
               "experiment runner does not declare execution_plan but the "
               "resolved plan carries economic parameters",
               details={"economic_keys": sorted(execution_plan.economic_params)})


def _call_runner(runner: Callable, run_dir: Path, *, no_ledger: bool,
                 execution_plan: ResolvedExperimentPlan):
    """Invoke one runner with the resolved plan as its ``execution_plan``.

    A callable that declares ``execution_plan`` receives exactly that plan --
    never the raw document. A callable that does not is the legacy shape and
    stays compatible only while the resolved plan carries no
    ``economic_params``: the refusal is the shared preflight above, which
    ``run_experiment`` already ran before any artifact existed and which is
    re-checked here as defense in depth.
    """
    signature = inspect.signature(runner)
    if "no_ledger" not in signature.parameters:
        raise fail("INVALID_REQUEST", "experiment runner lacks required no-ledger control")
    _execution_plan_compatibility(runner, execution_plan)
    kwargs = {"run_dir": run_dir, "no_ledger": no_ledger}
    accepted = {name for name in signature.parameters}
    if "execution_plan" in accepted:
        kwargs["execution_plan"] = execution_plan
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
                   synthetic=False,
                   resolved_plan: ResolvedExperimentPlan | None = None) -> dict:
    """Run one isolated hypothesis; retries reuse its exact spec/input identity."""
    if mode not in ("smoke", "primary"):
        raise fail("INVALID_REQUEST", "unknown experiment mode")
    if mode == "smoke" and backup is not None:
        raise fail("INVALID_REQUEST", "smoke runs cannot request backup")
    # Resolved once, here: an unused economic declaration is a typed refusal
    # before any directory is created, any evidence persisted, or the runner
    # is invoked, and the very plan is the one handed to the callable below.
    # A supplied plan is adopted only when it is the exact object this resolver
    # produced -- its private provenance marker, which a ``dataclasses.replace``
    # rebuild loses -- so a byte-equal plan hiding a mutable dict, a proxy over
    # a caller-held mapping, or mutable ``arms``/other fields is refused before
    # any filesystem effect; its canonical bytes must still match, and the
    # accepted object then travels to the runner unchanged, so a caller's plan
    # identity survives the call unchanged.
    if resolved_plan is None:
        plan = resolve_experiment_plan(spec)
    else:
        plan = _adopt_resolved_plan(spec, resolved_plan)
    # The legacy-callable compatibility check is the same kind of preflight:
    # a runner with no ``execution_plan`` channel is refused, typed, before
    # the run directory, ``CAPABILITIES.json``, or any receipt can exist.
    _execution_plan_compatibility(runner, plan)
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
        result = _call_runner(runner, destination, no_ledger=(mode == "smoke" or synthetic),
                              execution_plan=plan)
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
