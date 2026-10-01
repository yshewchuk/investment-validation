"""Pinned original-panel COPY identity: object/catalog consistency, path-free refusals."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from engine.v2.data import errors, objects
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore, CONTENT_HASH_PREFIX

PANEL_TABLE = "feature_panel"
COPY_PREFIX = "panel_copy_"


@dataclass(frozen=True)
class PinnedPanelContext:
    """The genuine app catalog/store trust anchor for one pinned panel."""
    catalog_path: Path
    snapshot_id: str
    artifact_root: Path


def _refuse(reason: str) -> ValueError:
    return ValueError(f"{COPY_PREFIX}refused: {reason}")


def verify_panel_copy(context: PinnedPanelContext,
                      expected_sha256: str | None = None) -> dict[str, str]:
    try:
        conn = sqlite3.connect(Path(context.catalog_path).resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except (sqlite3.Error, OSError) as exc:
        raise _refuse("pinned catalog is not readable") from exc
    try:
        try:
            store = ArtifactStore(context.artifact_root)
            repository = Repository(conn, store)
            snapshot = repository.resolve(context.snapshot_id)
            records = repository.fragment_records(snapshot, PANEL_TABLE)
        except (errors.DataError, ArtifactError, sqlite3.Error, OSError) as exc:
            raise _refuse("pinned snapshot has no resolvable original panel") from exc
        if len(records) != 1:
            raise _refuse("original panel is not exactly one fragment")
        version = snapshot.table_versions[PANEL_TABLE]
        record = records[0]
        if record.table_contract_ref != version.table_contract_ref:
            raise _refuse("panel fragment contract is not the snapshot's table contract")
        object_id, content_hash = record.object_ref.object_id, record.object_ref.content_hash
        if not object_id or not content_hash:
            raise _refuse("panel fragment identity fields are incomplete")
        digest = content_hash.removeprefix(CONTENT_HASH_PREFIX)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise _refuse("recorded panel hash is malformed")
        try:
            objects.verify_object_path(store, record.object_ref)
        except (errors.DataError, ArtifactError, OSError) as exc:
            raise _refuse("original panel bytes are not verifiable") from exc
        if expected_sha256 is not None:
            expected = expected_sha256.removeprefix(CONTENT_HASH_PREFIX)
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise _refuse("expected panel hash is malformed")
            if digest != expected:
                raise _refuse("original panel hash does not match the expected value")
        return {"panel_copy_mode": "COPY", "panel_copy_snapshot_id": snapshot.snapshot_id,
                "panel_copy_dataset_version_id": version.dataset_version_id,
                "panel_copy_object_id": object_id, "panel_copy_sha256": digest}
    finally:
        conn.close()
