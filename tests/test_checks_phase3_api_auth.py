"""Unit tests for checks/rearchitecture_phase3_api_auth.py (L08 producer)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase3_api_auth import build  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from tests.test_v2_serving_projections import (  # noqa: E402
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _preview_input,
    _row,
    _score_doc,
)

TOKEN = "test-l08-token"


def _write_bundle(root: Path, release_id: str) -> None:
    release_dir = root / "releases" / release_id
    (release_dir / "data").mkdir(parents=True)
    (release_dir / "index.html").write_text(f"<html>{release_id}</html>")
    (release_dir / "data" / "board.json").write_text(json.dumps({"release": release_id}))


def _write_health(root: Path) -> Path:
    health = root / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "t0", "withheld_release": None}))
    return health


def _committed_release(tmp_path):
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2", [_event_row("e0", "T0", datetime(2024, 1, 1))])
    repo = Repository(conn, store)
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    serving_store = ArtifactStore(serving_root / "objects")
    serving_conn = projections.connect(str(serving_root / "serving.sqlite"))
    rows = [_row(ticker="T0", event_date="2024-01-01", strike=100.0)]
    release = projections.build_candidate(
        _preview_input(), _score_doc(rows=rows), _bundle(*[_compact(r) for r in rows]),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-01", resolved_as_of="2024-01-01")
    serving_conn.close()
    return serving_root, release


def test_auth_traversal_negative_control_differs_on_a_real_hardened_pair(tmp_path):
    op_root = tmp_path / "op"
    _write_bundle(op_root, "r1")
    (op_root / "CURRENT").write_text("r1\n")
    health = _write_health(op_root)
    serving_root, release = _committed_release(tmp_path)
    scratch = tmp_path / "scratch"
    receipt = build(op_root, health, release_id="r1", token=TOKEN,
                    serving_db=serving_root / "serving.sqlite", store_root=serving_root / "objects",
                    serving_root=serving_root, scratch_dir=scratch, api_release_id=release.release_id)
    assert receipt.verdict == "differ"
    assert not receipt.findings
    assert receipt.population.compared > 0


def test_auth_traversal_negative_control_catches_a_real_symlink_escape(tmp_path, monkeypatch):
    """A real path-traversal bug: operations.py's symlink guard disabled.
    Patch _safe_file to skip the is_symlink() checks and confirm the
    producer catches the leak of a real planted secret file."""
    import engine.v2.serving.operations as operations_module

    op_root = tmp_path / "op"
    _write_bundle(op_root, "r1")
    (op_root / "CURRENT").write_text("r1\n")
    health = _write_health(op_root)
    serving_root, release = _committed_release(tmp_path)
    scratch = tmp_path / "scratch"

    def _unsafe_file(root, relative):
        from engine.v2.foundation import safe_relative_path
        parts = safe_relative_path(relative)
        current = root
        for part in parts:
            current = current / part
        if not current.is_file() and not current.is_symlink():
            raise FileNotFoundError(relative)
        return current

    monkeypatch.setattr(operations_module, "_safe_file", _unsafe_file)
    receipt = build(op_root, health, release_id="r1", token=TOKEN,
                    serving_db=serving_root / "serving.sqlite", store_root=serving_root / "objects",
                    serving_root=serving_root, scratch_dir=scratch, api_release_id=release.release_id)
    assert receipt.verdict == "agree"
    assert any(f.field_path == "symlink_escape" for f in receipt.findings)
