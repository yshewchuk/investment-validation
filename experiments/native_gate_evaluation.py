"""Read-only fitted gate child; only the supervisor may commit staged evidence."""
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from engine.v2.contracts import FAILURE_CODES
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore, from_document
from engine.v2.ops.errors import fail
from engine.v2.ops.stages import NativeGateParameters
from experiments.gate_report import RUNNER, gate_result
from experiments.native_gate_worker import (
    admit_native_gate_inputs,
    native_gate_data_failure,
    write_native_gate_receipt,
)


def run(parameters, root):
    """Return one complete unrecorded result; refusals never publish success."""
    if not isinstance(parameters, Mapping) or not isinstance(parameters.get("purpose", "sweep"), str):
        raise fail("INVALID_EXPERIMENT_SPEC", "native gate arguments must be typed")
    if parameters.get("purpose", "sweep") != "sweep":
        raise fail("HOLDOUT_ACCESS_DENIED", "native gate permits sweep research only")
    if parameters.get("no_ledger") is not True:
        raise fail("INVALID_EXPERIMENT_SPEC", "native gate child has no recording authority")
    evaluating = False
    try:
        from_document(NativeGateParameters, dict(parameters))
        conn = sqlite3.connect(Path(parameters["catalog_path"]).resolve().as_uri() + "?mode=ro",
                               uri=True, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            store = ArtifactStore(parameters["objects_root"])
            spec, registration = admit_native_gate_inputs(parameters, root, conn, store, runner=RUNNER)
            evaluating = True
            result = gate_result(Repository(conn, store, pin_snapshots=True), registration, spec)
        finally:
            conn.close()
        evaluating = False
        try:
            write_native_gate_receipt(root / "native_gate_result.json", result)
        except (ValueError, TypeError, RecursionError) as error:
            raise fail("EXPERIMENT_VARIANT_FAILED", "native gate result serialization failed",
                       details={"data_failure_code": type(error).__name__}) from None
    except DataError as error:
        raise native_gate_data_failure(error) from None
    except ArtifactError as error:
        raise fail(error.code if error.code in FAILURE_CODES else "INTEGRITY_FAILED",
                   "native gate artifact cannot be verified", details={"data_failure_code": error.code}) from None
    except (OSError, sqlite3.Error) as error:
        raise fail("INPUT_CHANGED", "native gate I/O is unavailable",
                   details={"data_failure_code": type(error).__name__}) from None
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        if evaluating:
            raise fail("EXPERIMENT_VARIANT_FAILED", "native gate inputs failed",
                       details={"data_failure_code": getattr(error, "code", type(error).__name__)}) from None
        raise fail("INVALID_EXPERIMENT_SPEC", "native gate staged input is malformed") from None
    return {"outputs": [{"name": "native_gate_result", "path": "native_gate_result.json",
                         "schema": "native_gate_result.v1.0"}],
            "completed_ids": [registration.variant_id], "no_work": False}
