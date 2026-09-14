"""D19: ``compare_bundles`` -- pure filesystem-level tests, no catalog, no
legacy import, no subprocess. Exercises the "identical", "one planted
non-metadata change" and "metadata-only change" scenarios the brief asks for
directly against two bundle directories.
"""
from __future__ import annotations

import json
from pathlib import Path

from checks.rearchitecture_phase2_render_parity import compare_bundles


def _write_bundle(root: Path, *, board_note="hello", generated_at="2020-01-01T00:00:00Z"):
    data = root / "data"
    data.mkdir(parents=True)
    (data / "board.json").write_text(json.dumps({"rows": [{"ticker": "ZZ", "note": board_note}]}))
    (data / "meta.json").write_text(json.dumps({
        "generated_at": generated_at, "freshness": {}, "quota": {}, "cron": {},
        "horizon_days": 35}))
    (data / "health.json").write_text(json.dumps({"generated_at": generated_at, "ok": True}))
    (data / "flags.js").write_text("window.FLAGS = " + json.dumps({"flags": []}) + ";\n")
    return root


def test_identical_bundles_agree(tmp_path):
    a = _write_bundle(tmp_path / "a")
    b = _write_bundle(tmp_path / "b")
    receipt = compare_bundles(a, b)
    assert receipt.verdict == "agree", receipt.summary()
    assert receipt.findings == ()
    assert receipt.population.expected == receipt.population.compared == 4


def test_one_planted_non_metadata_change_is_caught_and_named(tmp_path):
    a = _write_bundle(tmp_path / "a")
    b = _write_bundle(tmp_path / "b", board_note="changed")
    receipt = compare_bundles(a, b)
    assert receipt.verdict == "differ"
    named = [f.field_path for f in receipt.findings]
    assert any(path.startswith("data/board.json:rows[0].note") for path in named)
    # nothing else moved
    assert len(receipt.findings) == 1


def test_metadata_only_change_agrees(tmp_path):
    a = _write_bundle(tmp_path / "a", generated_at="2020-01-01T00:00:00Z")
    b = _write_bundle(tmp_path / "b", generated_at="2031-06-15T00:00:00Z")
    receipt = compare_bundles(a, b)
    assert receipt.verdict == "agree", receipt.summary()


def test_missing_file_on_one_side_is_a_named_disagreement(tmp_path):
    a = _write_bundle(tmp_path / "a")
    b = _write_bundle(tmp_path / "b")
    (b / "data" / "flags.js").unlink()
    receipt = compare_bundles(a, b)
    assert receipt.verdict == "differ"
    assert any(f.field_path.startswith("data/flags.js:present") for f in receipt.findings)
