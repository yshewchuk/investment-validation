"""Read explicit pinned D14 inputs; no mutable paths, latest receipt or legacy reader."""
from __future__ import annotations

import argparse
import sqlite3
from io import BytesIO
from pathlib import Path

import pandas as pd

from engine.v2.contracts import ArtifactRef
from engine.v2.data import errors
from engine.v2.data.computed_moves import native_trading_calendar
from engine.v2.data.reference_catalog import reference_inputs_for_receipt
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore
from engine.v2.models.training.d14_dataset import build_d14_dataset

MAX_CALENDAR_BYTES = 16 * 1024 * 1024


def _calendar(store, reference):
    if reference.byte_size > MAX_CALENDAR_BYTES:
        raise errors.fail("RESULT_LIMIT_EXCEEDED", "D14 calendar exceeds its byte bound")
    digest = reference.content_hash.removeprefix("sha256:")
    ref = ArtifactRef(artifact_id=reference.object_id, content_hash=reference.content_hash,
                      byte_size=reference.byte_size, schema_ref="parquet_fragment.v1",
                      storage_key=f"objects/{digest[:2]}/{digest}")
    try:
        data = store.read_verified(ref)
    except (ArtifactError, OSError) as exc:
        raise errors.fail("OBJECT_CORRUPT", "D14 calendar object failed verification") from exc
    try:
        observed = pd.read_csv(BytesIO(data), skiprows=3, header=None, usecols=[0], names=["date"])
        observed["date"] = pd.to_datetime(observed["date"], errors="raise")
        if observed.empty or observed["date"].isna().any():
            raise ValueError("missing calendar date")
    except (ValueError, pd.errors.ParserError) as exc:
        raise errors.fail("CONTRACT_MISMATCH", "D14 calendar has invalid dates") from exc
    return native_trading_calendar({"index": observed}, extend_days=400)


def build_dataset(role, *, catalog, artifact_root, snapshot_id, receipt_id):
    """Load one pinned calendar and snapshot, then attach source resume identity."""
    try:
        conn = sqlite3.connect(Path(catalog).resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise errors.fail("CONTRACT_MISMATCH", "D14 catalog is unavailable") from exc
    conn.row_factory = sqlite3.Row
    try:
        store = ArtifactStore(artifact_root)
        repository = Repository(conn, store, pin_snapshots=True)
        snapshot = repository.resolve(snapshot_id)
        receipt = conn.execute(
            "SELECT status, result_snapshot_id FROM data_import_receipts WHERE receipt_id = ?",
            (receipt_id,)).fetchone()
        if receipt is None or tuple(receipt) != ("committed", snapshot.snapshot_id):
            raise errors.fail("IDENTITY_CONFLICT", "D14 import receipt does not bind the snapshot")
        calendars = [ref for ref in reference_inputs_for_receipt(conn, receipt_id=receipt_id)
                     if ref.kind == "calendar"]
        if len(calendars) != 1:
            raise errors.fail("CONTRACT_MISMATCH", "D14 receipt must pin exactly one calendar")
        calendar_ref = calendars[0]
        calendar = _calendar(store, calendar_ref)
        frame = build_d14_dataset(repository, snapshot, calendar=calendar, role=role)
        frame.attrs["training_inputs"] = {
            "schema_version": "d14.snapshot_dataset.v1", "snapshot_id": snapshot.snapshot_id,
            "receipt_id": receipt_id, "calendar_hash": calendar_ref.content_hash,
            "table_versions": {table: snapshot.table_versions[table].dataset_version_id
                               for table in ("earnings_events", "feature_panel", "daily_market")},
        }
        return frame
    except sqlite3.Error as exc:
        raise errors.fail("CONTRACT_MISMATCH", "D14 catalog is invalid") from exc
    finally:
        conn.close()


def main(argv=None):
    """Explicit native monthly training; existing supervised routes stay unchanged."""
    from engine import paths
    from engine.v2.models.training import RecipeKey, current_recipes, run_training_job
    from engine.v2.models.training.estimators import forbid_fitting

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("implied_t1", "runup_move"), required=True)
    for field in ("catalog", "artifact-root", "snapshot-id", "receipt-id", "out"):
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    data_roots = (paths.DATA.resolve(), Path(args.artifact_root).resolve())
    if any(out == root or root in out.parents for root in data_roots):
        parser.error("--out must be outside data/ and the input artifact store")
    forbid_fitting("tools.phase5_d14_inputs.main")
    frame = build_dataset(args.role, catalog=args.catalog, artifact_root=args.artifact_root,
                          snapshot_id=args.snapshot_id, receipt_id=args.receipt_id)
    recipe = current_recipes()[RecipeKey(args.role, "*", "tier4_monthly")]
    result = run_training_job(recipe, frame, out, plan_only=args.plan_only,
                              input_identity=frame.attrs["training_inputs"])
    print(f"{recipe.recipe_id}: rows={len(frame)} folds={len(result.outcomes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
