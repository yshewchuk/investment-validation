"""Shared serving-release builder for serving tests."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import PreviewRelease  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from tests.test_v2_serving_projections import (  # noqa: E402
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _preview_input,
    _row,
    _score_doc,
    _serving,
)


def build_releases(tmp_path):
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(tmp_path / "phase2", [
        _event_row("e1", "AAA", datetime(2024, 1, 5)),
        _event_row("e2", "BBB", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row_a, row_b = _row(ticker="AAA"), _row(ticker="BBB")
    release_a = projections.build_candidate(
        _preview_input(), _score_doc(rows=[row_a]), _bundle(_compact(row_a)),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-04", resolved_as_of="2024-01-04")
    release_b = projections.build_candidate(
        _preview_input(), _score_doc(rows=[row_b]), _bundle(_compact(row_b)),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-05", resolved_as_of="2024-01-05")
    assert isinstance(release_a, PreviewRelease) and isinstance(release_b, PreviewRelease)
    return repo, snap, serving_conn, serving_store, release_a, release_b
