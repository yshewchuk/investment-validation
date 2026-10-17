"""Verified snapshot/receipt-bound inputs for monthly native D14 datasets."""
from __future__ import annotations

import sqlite3
from io import BytesIO
from pathlib import Path

import pandas as pd

from engine.v2.contracts import ArtifactRef
from engine.v2.data import computed_moves, errors, reference_catalog, repository
from engine.v2.foundation import ArtifactError, ArtifactStore

from .d14_dataset import calculate_frame, validate_role

MAX_CALENDAR_BYTES = 16 * 1024 * 1024


def _calendar(store, reference):
    if reference.byte_size > MAX_CALENDAR_BYTES:
        raise errors.fail("RESULT_LIMIT_EXCEEDED", "D14 calendar exceeds its byte bound")
    # The original reference importer publishes every file, including CSV,
    # under parquet_fragment.v1; preserve that registered artifact identity.
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
    return computed_moves.native_trading_calendar({"index": observed}, extend_days=400)


def build_d14_dataset(role, *, catalog, artifact_root, snapshot_id, receipt_id):
    """Load pinned inputs; set frame.attrs["training_inputs"] for job resume binding."""
    validate_role(role)
    try:
        conn = sqlite3.connect(Path(catalog).resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise errors.fail("CONTRACT_MISMATCH", "D14 catalog is unavailable") from exc
    conn.row_factory = sqlite3.Row
    try:
        store = ArtifactStore(artifact_root)
        reader = repository.Repository(conn, store, pin_snapshots=True)
        snapshot = reader.resolve(snapshot_id)
        receipt = conn.execute(
            "SELECT status, result_snapshot_id FROM data_import_receipts WHERE receipt_id = ?",
            (receipt_id,)).fetchone()
        if receipt is None or tuple(receipt) != ("committed", snapshot.snapshot_id):
            raise errors.fail("IDENTITY_CONFLICT", "D14 import receipt does not bind the snapshot")
        calendars = [ref for ref in reference_catalog.reference_inputs_for_receipt(conn, receipt_id=receipt_id)
                     if ref.kind == "calendar"]
        if len(calendars) != 1:
            raise errors.fail("CONTRACT_MISMATCH", "D14 receipt must pin exactly one calendar")
        calendar_ref = calendars[0]
        calendar = _calendar(store, calendar_ref)
        frame = calculate_frame(reader, snapshot, calendar=calendar, role=role)
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
