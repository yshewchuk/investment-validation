"""Unit tests for checks/rearchitecture_phase3_preview.py (L01 producer)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase3_preview import build  # noqa: E402


def _write_bundle(root: Path, release_id: str) -> None:
    release_dir = root / "releases" / release_id
    (release_dir / "data").mkdir(parents=True)
    (release_dir / "index.html").write_text(f"<html>marker-{release_id}</html>")
    (release_dir / "data" / "board.json").write_text(json.dumps({"release": release_id}))


def _write_health(root: Path) -> Path:
    health = root / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "t0", "withheld_release": None}))
    return health


def test_open_parity_agrees_on_a_real_bundle(tmp_path):
    _write_bundle(tmp_path, "r1")
    (tmp_path / "CURRENT").write_text("r1\n")
    health = _write_health(tmp_path)
    open_receipt, auth_receipt = build(tmp_path, health, token="secret")
    assert open_receipt.verdict == "agree"
    assert open_receipt.population.compared > 0
    assert auth_receipt.verdict == "differ"
    assert auth_receipt.population.compared > 0


def test_open_parity_refuses_an_empty_release_as_agreement(tmp_path):
    """CURRENT names a release id with no materialized directory at all
    (a collapsed population) -- must never report verdict agree just
    because zero file comparisons means zero findings."""
    (tmp_path / "releases" / "r1").mkdir(parents=True)
    (tmp_path / "CURRENT").write_text("r1\n")
    health = _write_health(tmp_path)
    open_receipt, _ = build(tmp_path, health, token="secret")
    assert open_receipt.verdict != "agree"
    assert any(f.kind == "missing_field" for f in open_receipt.findings)


def test_open_parity_catches_served_bytes_that_diverge_from_disk(tmp_path, monkeypatch):
    """A real "the server lied" bug: operations.py serves DIFFERENT bytes
    than what is actually on disk (patched at the handler level, so the
    producer's own on-disk baseline read is untouched). The receipt must
    catch this as a real mismatch."""
    import engine.v2.serving.operations as operations_module

    _write_bundle(tmp_path, "r1")
    (tmp_path / "CURRENT").write_text("r1\n")
    health = _write_health(tmp_path)

    real_release_route = operations_module.OperationsHandler._release_route

    def _tampered_release_route(self, config, path):
        if path.endswith("board.json"):
            return self._send(200, b'{"release": "TAMPERED"}', "application/json")
        return real_release_route(self, config, path)

    monkeypatch.setattr(operations_module.OperationsHandler, "_release_route", _tampered_release_route)
    open_receipt, _ = build(tmp_path, health, token="secret")
    assert open_receipt.verdict != "agree"
    assert any(f.field_path == "data/board.json" for f in open_receipt.findings)


def test_auth_negative_control_catches_a_real_bypass(tmp_path, monkeypatch):
    """A real auth bug: patch OperationsHandler._authorized to always
    return True (every request looks authenticated, including the
    no-token/wrong-token probes). The producer must catch this as a real
    leak (verdict AGREE -- the "bad" outcome for a negative control) rather
    than silently reporting the safe DIFFER."""
    import engine.v2.serving.operations as operations_module
    from checks.rearchitecture_phase3_preview import build

    _write_bundle(tmp_path, "r1")
    (tmp_path / "CURRENT").write_text("r1\n")
    health = _write_health(tmp_path)
    monkeypatch.setattr(operations_module.OperationsHandler, "_authorized", lambda self: True)
    _, auth_receipt = build(tmp_path, health, token="secret")
    assert auth_receipt.verdict == "agree"
    assert auth_receipt.findings
