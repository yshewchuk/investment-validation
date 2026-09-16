"""Unit tests for checks/rearchitecture_phase3_startup.py (L09 producer)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase3_startup import build  # noqa: E402


def test_no_scoring_or_provider_touched_on_a_real_empty_serving_db(tmp_path):
    serving_root = tmp_path / "serving"
    receipt = build(serving_root / "serving.sqlite", serving_root / "objects", serving_root,
                    token="test-l09-token", release_id=None)
    assert receipt.verdict == "differ"
    assert not receipt.findings
    assert receipt.population.compared == 3


def test_catches_a_real_rigged_constructor_invocation(monkeypatch):
    """A real regression: the read path actually imports and constructs a
    scorer during create_app/startup. Patch _GUARD_SCRIPT's injected rig to
    fire immediately at import time (simulating create_app itself touching
    the rigged module) and confirm the producer reports it, rather than the
    subprocess crashing silently and the producer reporting a clean pass."""
    import checks.rearchitecture_phase3_startup as mod

    tampered_script = mod._GUARD_SCRIPT.replace(
        "from engine.v2.serving.api import create_app",
        "import engine.score as _rigged_probe\n"
        "_rigged_probe.anything()\n"
        "from engine.v2.serving.api import create_app")
    monkeypatch.setattr(mod, "_GUARD_SCRIPT", tampered_script)
    receipt = mod.build(Path("/tmp/nope/serving.sqlite"), Path("/tmp/nope/objects"), Path("/tmp/nope"),
                        token="test-l09-token", release_id=None)
    assert receipt.verdict == "agree"
    assert receipt.findings
