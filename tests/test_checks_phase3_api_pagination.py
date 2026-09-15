"""Unit tests for checks/rearchitecture_phase3_api_pagination.py (L07 producer)."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase3_api_pagination import build  # noqa: E402
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

TOKEN = "test-l07-token"


def _committed_release(tmp_path):
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2", [_event_row(f"e{i}", f"T{i}", datetime(2024, 1, i + 1)) for i in range(10)])
    repo = Repository(conn, store)
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    serving_store = ArtifactStore(serving_root / "objects")
    serving_conn = projections.connect(str(serving_root / "serving.sqlite"))
    rows = [_row(ticker=f"T{i}", event_date=f"2024-01-{i + 1:02d}", strike=100.0 + i) for i in range(10)]
    release = projections.build_candidate(
        _preview_input(), _score_doc(rows=rows), _bundle(*[_compact(r) for r in rows]),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-10", resolved_as_of="2024-01-10")
    serving_conn.close()
    return serving_root, release


def test_pagination_and_cursor_negative_control_on_a_real_committed_release(tmp_path):
    serving_root, release = _committed_release(tmp_path)
    pagination, cursor_negative = build(
        serving_root / "serving.sqlite", serving_root / "objects", serving_root,
        token=TOKEN, release_id=release.release_id)
    assert pagination.verdict == "agree"
    assert pagination.population.compared == 10
    assert cursor_negative.verdict == "differ"
    assert not cursor_negative.findings


def test_cursor_negative_control_catches_a_real_signature_bypass(tmp_path, monkeypatch):
    """A real integrity bug: the API accepts ANY cursor signature. Patch
    hmac.compare_digest to always return True and confirm the producer
    reports the bypass instead of silently passing."""
    import hmac as hmac_module

    import engine.v2.serving.api as api_module

    serving_root, release = _committed_release(tmp_path)
    monkeypatch.setattr(api_module, "hmac", type("F", (), {
        "compare_digest": staticmethod(lambda a, b: True),
        "new": staticmethod(hmac_module.new),
    }))
    _, cursor_negative = build(
        serving_root / "serving.sqlite", serving_root / "objects", serving_root,
        token=TOKEN, release_id=release.release_id)
    assert cursor_negative.verdict == "agree"
    assert cursor_negative.findings
