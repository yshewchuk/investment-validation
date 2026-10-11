"""Admitted smoke pricing from staged registration; no scientific recording."""
import errno
import json
import os
import sqlite3
import stat
from dataclasses import asdict
from math import isfinite
from pathlib import Path

from engine.v2.contracts import FAILURE_CODES
from engine.v2.contracts.data import SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore, canonical_json, from_document
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.executor import THREAD_VARIABLES
from engine.v2.ops.experiments import experiment_spec_from_document, resolve_experiment_plan
from engine.v2.ops.fingerprints import environment_identity, source_closure
from experiments.native_registration import NativeRegistration, verify_native_registration
from experiments.v2_gate_variant import price_variant

RUNNER = "experiments/native_gate_worker.py"
BINDINGS = {"spec.json", "native_registration.json"}
MAX_STAGED_BYTES = 16 * 1024 * 1024


def _read_staged(path):
    """Bound allocation and validate the descriptor actually supplying bytes."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STAGED_BYTES:
                raise fail("INVALID_EXPERIMENT_SPEC", "native staging requires bounded regular files")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(MAX_STAGED_BYTES + 1)
            if len(data) > MAX_STAGED_BYTES:
                raise fail("INVALID_EXPERIMENT_SPEC", "native staged document exceeds its bound")
            return data
        finally:
            os.close(fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise fail("INVALID_EXPERIMENT_SPEC", "native staging must not follow symlinks") from None
        raise


def _admit(parameters, root, conn, store):
    """Validate the staged request against existing immutable native admission."""
    if (parameters.get("no_ledger") is not True
            or set(parameters.get("input_bindings") or {}) != BINDINGS):
        raise fail("INVALID_EXPERIMENT_SPEC", "native pricing requires bound smoke inputs")
    try:
        spec = experiment_spec_from_document(json.loads(_read_staged(root / "spec.json")))
    except OpsError as error:
        if error.code != "INVALID_REQUEST":
            raise
        raise fail("INVALID_EXPERIMENT_SPEC", "native staged specification is malformed") from None
    plan = resolve_experiment_plan(spec)
    if plan.runner != RUNNER:
        raise fail("INVALID_EXPERIMENT_SPEC", "native pricing runner is not allowlisted")
    if not isinstance(plan.economic_params.get("strategy"), str):
        raise fail("INVALID_EXPERIMENT_SPEC", "native worker requires a registered strategy")
    registration = NativeRegistration(parameters["run_id"],
                                      _read_staged(root / "native_registration.json"))
    verify_native_registration(conn, store, registration)
    binding = registration.document
    if (registration.variant_id != parameters["variant_id"]
            or list(parameters["expected_ids"]) != [registration.variant_id]
            or binding["spec_hash"] != spec.spec_hash
            or canonical_json(binding["execution_plan"]) != canonical_json(plan.as_document())
            or binding["source_closure"] != source_closure(Path.cwd(), [RUNNER])
            or binding["environment"] != environment_identity(1)
            or any(os.environ.get(name) != "1" for name in THREAD_VARIABLES)):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native staged execution differs from registration")
    return spec, registration


def run(parameters, root):
    """Read only the registered snapshot and return a complete staged receipt."""
    if parameters.get("purpose", "sweep") != "sweep":
        raise fail("HOLDOUT_ACCESS_DENIED", "native pricing permits sweep reads only")
    if parameters.get("no_ledger") is not True:
        raise fail("INVALID_EXPERIMENT_SPEC", "native pricing is smoke-only")
    try:
        conn = sqlite3.connect(Path(parameters["catalog_path"]).resolve().as_uri() + "?mode=ro",
                               uri=True, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            store = ArtifactStore(parameters["objects_root"])
            spec, registration = _admit(parameters, root, conn, store)
            binding = registration.document
            result = price_variant(
                Repository(conn, store), from_document(SnapshotRef, binding["snapshot"]), spec,
                strategy=spec.economic_params["strategy"], as_of_month=binding["holdouts"]["holdout_as_of_month"],
                event_ids=binding["event_ids"])
            inputs = asdict(result.stage_inputs)
            if (sorted(inputs["event_ids"]) != binding["event_ids"]
                    or any(inputs[name] != value for name, value in binding["holdouts"].items())):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "consumed population differs from registration")
            cashflows = result.replay.trades[
                ["event_id", "fill_alpha", "entry_cost", "exit_value", "pnl"]].to_dict("records")
            if any(not isfinite(row[name]) for row in cashflows
                   for name in ("fill_alpha", "entry_cost", "exit_value", "pnl")):
                raise fail("EXPERIMENT_VARIANT_FAILED", "native pricing produced nonfinite cash flows")
            receipt = {"schema_version": "native_gate_consumption.v1.0",
                       "variant_id": registration.variant_id, "run_id": registration.run_id,
                       "recording_mode": "no-ledger", "execution_plan": result.execution_plan.as_document(),
                       "stage_inputs": inputs, "mark_based_cashflows": cashflows}
        finally:
            conn.close()
    except DataError as error:
        code = error.code if error.code in FAILURE_CODES else "EXPERIMENT_VARIANT_FAILED"
        raise fail(code, "native pinned pricing refused", details={"data_failure_code": error.code}) from None
    except (ArtifactError, ValueError, TypeError, KeyError):
        raise fail("INVALID_EXPERIMENT_SPEC", "native staged input is malformed") from None
    except (OSError, sqlite3.Error):
        raise fail("INPUT_CHANGED", "native staged input is unavailable") from None
    (root / "native_gate_consumption.json").write_text(canonical_json(receipt))
    return {"outputs": [{"name": "native_gate_consumption", "path": "native_gate_consumption.json",
                         "schema": "native_gate_consumption.v1.0"}],
            "completed_ids": [registration.variant_id], "no_work": False}
