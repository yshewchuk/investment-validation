"""Unit tests for checks/rearchitecture_phase3_current_switch.py (L02 producer)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase3_current_switch import build  # noqa: E402


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


def test_pinned_urls_survive_a_current_switch_between_two_real_releases(tmp_path):
    _write_bundle(tmp_path, "r1")
    _write_bundle(tmp_path, "r2")
    health = _write_health(tmp_path)
    receipt = build(tmp_path, health, r1="r1", r2="r2", token="secret")
    assert receipt.verdict == "agree"
    assert not receipt.findings
    assert receipt.population.compared > 0


def test_catches_a_real_pin_leak_where_the_pinned_url_moves_with_current(tmp_path, monkeypatch):
    """A real regression: operations.py's release route resolves 'current'
    on EVERY request instead of trusting the caller's own release id
    segment. Patch _release_route to always substitute the live CURRENT
    pointer, simulating that bug, and confirm the producer catches it."""
    import engine.v2.serving.operations as operations_module

    _write_bundle(tmp_path, "r1")
    _write_bundle(tmp_path, "r2")
    health = _write_health(tmp_path)

    real_release_route = operations_module.OperationsHandler._release_route

    def _always_follow_current(self, config, path):
        rel = path.removeprefix("/release/").split("/", 1)
        if len(rel) == 2 and rel[0] in ("r1", "r2"):
            current = operations_module._resolve_current_id(config)
            path = "/release/" + current + "/" + rel[1]
        return real_release_route(self, config, path)

    monkeypatch.setattr(operations_module.OperationsHandler, "_release_route", _always_follow_current)
    receipt = build(tmp_path, health, r1="r1", r2="r2", token="secret")
    assert receipt.verdict != "agree"
    assert any(f.field_path == "pinned_data_survives_switch" for f in receipt.findings)
