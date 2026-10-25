"""Regression tests for native stage failure health rendering in the operations shell."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from engine.v2.serving.operations import (
    _read_calibration_health,
    _read_health,
    shell_document,
)


def _native_stage_failure() -> dict:
    return {
        "occurrence": "2026-10-24",
        "stage_id": "native_score_batch",
        "error_code": "MODEL_NOT_READY",
        "status": "FAILED",
    }


def _health_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "operations-health.json"
    artifact.write_text(json.dumps({
        "schema_version": "operations_health.v1.2",
        "generated_at": "2026-10-25T12:00:00Z",
        "native_stage_failures": [_native_stage_failure()],
    }))
    return artifact


@pytest.mark.parametrize("reader", [_read_health, _read_calibration_health])
def test_health_readers_accept_and_preserve_native_stage_failures(tmp_path, reader):
    """Both readers must accept v1.2 and keep the additive failure entry."""
    artifact = _health_artifact(tmp_path)
    raw = reader(artifact)

    document = json.loads(raw)
    assert document["schema_version"] == "operations_health.v1.2"
    assert document["generated_at"] == "2026-10-25T12:00:00Z"
    assert document["native_stage_failures"] == [_native_stage_failure()]

    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert raw == canonical


def test_shell_document_renders_native_stage_failures_into_health_reason():
    """The embedded JS must accept v1.2 and name each failure in its reason path."""
    document = shell_document().decode()
    compact = re.sub(r"\s+", " ", document)

    assert "operations_health.v1.2" in compact
    assert "const nsf=health.native_stage_failures;" in compact

    # The failure reason path reads occurrence, stage_id, error_code and status.
    assert "native stage failure: occurrence " in compact
    assert "occurrence '+occ+', stage '+sid+', error '+ec+', status '+st" in compact

    # Dynamic output goes through textContent (never innerHTML).
    assert "state.textContent=" in compact
    assert "stamp.textContent=" in compact
