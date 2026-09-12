"""The phase-0 gate's own negative controls.

The 2026-09-12 review planted a tier-1 receipt that replayed 1 of 18 pairs from
another commit, and the gate accepted it. Every way a receipt can stop
describing the current state is planted here and must fail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import rearchitecture_phase0_gate as g  # noqa: E402
from checks.replay_identity import evaluate_receipt  # noqa: E402
from engine.v2.diagnosis import content_hash  # noqa: E402

EXPECT = {
    "corpus_hash": "sha256:corpus",
    "declared_pairs": 3,
    "code": "sha256:code",
    "dependencies": "sha256:deps",
    "drift": [],
    "baseline_version": "2026-09-12c",
    "snapshot": "snap",
}


def receipt(verdict: str = "agree", *, seeded: bool = False, controls: dict | None = None,
            **bindings) -> dict:
    bind = {"corpus_hash": "sha256:corpus", "code_hash": "sha256:code",
            "dependencies_hash": "sha256:deps", "baseline": "2026-09-12c",
            "store_snapshot_at_replay": "snap", "deps_unverified": False,
            "limit": None, "declared": 3, "replayed": 3, "skipped": 0,
            "seeded": seeded, "engine_commit": "abc"}
    bind.update(bindings)
    payload = {"verdict": verdict, "bindings": bind, "findings": []}
    if controls is not None:
        payload["controls"] = controls
    return {"payload": payload}


# --------------------------------------------------------------------------
# evaluate_receipt
# --------------------------------------------------------------------------


def test_a_receipt_bound_to_the_current_state_passes():
    assert evaluate_receipt(receipt(), **EXPECT) == []


def test_the_review_probe_fails():
    """One of eighteen pairs replayed, from a different commit."""
    probe = receipt(replayed=1, skipped=1, engine_commit="0000000", code_hash="sha256:other")
    problems = evaluate_receipt(probe, **EXPECT)
    assert any("replayed 1 of 3" in p for p in problems)
    assert any("code changed" in p for p in problems)


@pytest.mark.parametrize("bindings,expected", [
    ({"limit": 1}, "partial run"),
    ({"replayed": 2}, "replayed 2 of 3"),
    ({"skipped": 1}, "skipped"),
    ({"declared": 2}, "declares 2 pairs"),
    ({"corpus_hash": "sha256:old"}, "stale corpus"),
    ({"code_hash": "sha256:old"}, "code changed"),
    ({"dependencies_hash": "sha256:old"}, "different frozen dependencies"),
    ({"baseline": "2026-09-12b"}, "binds baseline"),
    ({"store_snapshot_at_replay": "old"}, "snapshot changed"),
    ({"deps_unverified": True}, "--skip-deps-verify"),
])
def test_every_way_a_receipt_goes_stale_is_a_named_problem(bindings, expected):
    problems = evaluate_receipt(receipt(**bindings), **EXPECT)
    assert any(expected in p for p in problems), problems


def test_a_disagreeing_verdict_fails():
    assert any("verdict" in p for p in evaluate_receipt(receipt("differ"), **EXPECT))


def test_dependencies_that_drifted_on_disk_since_the_replay_fail():
    drifted = dict(EXPECT, drift=[{"path": "data/models/x.joblib", "issue": "sha256 differs"}])
    assert any("differ on disk" in p for p in evaluate_receipt(receipt(), **drifted))


def test_a_baseline_without_a_dependency_identity_fails():
    assert any("no frozen dependency identity" in p
               for p in evaluate_receipt(receipt(), **dict(EXPECT, dependencies=None)))


# --------------------------------------------------------------------------
# receipt rows
# --------------------------------------------------------------------------


def _write_receipt(tmp_path: Path, name: str, doc: dict) -> None:
    (tmp_path / "receipts").mkdir(exist_ok=True)
    (tmp_path / "receipts" / name).write_text(json.dumps(doc))


def test_the_compatibility_row_reads_the_current_versions_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "CORPUS", tmp_path)
    _write_receipt(tmp_path, "v1.json", receipt())
    assert g._receipt_row(EXPECT, "v1", seeded=False)["ok"]


def test_a_missing_receipt_fails_rather_than_skipping(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "CORPUS", tmp_path)
    row = g._receipt_row(EXPECT, "v1", seeded=False)
    assert not row["ok"] and not row.get("skipped")


def test_a_compatibility_receipt_does_not_satisfy_the_seeded_row(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "CORPUS", tmp_path)
    _write_receipt(tmp_path, "v1.seeded.json", receipt())
    assert not g._receipt_row(EXPECT, "v1", seeded=True)["ok"]


def test_a_seeded_receipt_with_a_failed_control_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "CORPUS", tmp_path)
    controls = {"forecast_suppressed": {"problems": ["record: no finding"]}}
    _write_receipt(tmp_path, "v1.seeded.json", receipt(seeded=True, controls=controls))
    row = g._receipt_row(EXPECT, "v1", seeded=True)
    assert not row["ok"] and "control forecast_suppressed" in row["detail"]


def test_a_seeded_receipt_whose_controls_behaved_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "CORPUS", tmp_path)
    controls = {"forecast_suppressed": {"problems": []}}
    _write_receipt(tmp_path, "v1.seeded.json", receipt(seeded=True, controls=controls))
    assert g._receipt_row(EXPECT, "v1", seeded=True)["ok"]


# --------------------------------------------------------------------------
# the baseline package
# --------------------------------------------------------------------------


def _package(root: Path, *, identical: bool = True, receipt_hash: str | None = None) -> None:
    files = {"definitions/a.json": "{}\n", "requirements.txt": "numpy==2.5.1\n"}
    package = root / "baseline" / "v1"
    for name, text in files.items():
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text(text)
    parts = {name: content_hash(text) for name, text in sorted(files.items())}
    package_hash = content_hash(parts)
    (package / "MANIFEST.json").write_text(json.dumps(
        {"parts": parts, "package_hash": package_hash}))
    (root / "baseline" / "CURRENT").write_text(json.dumps({"version": "v1"}))
    (root / "requirements.txt").write_text(files["requirements.txt"])
    (root / "baseline" / "receipts").mkdir()
    (root / "baseline" / "receipts" / "v1.json").write_text(json.dumps({"payload": {
        "package_hash": receipt_hash or package_hash, "byte_identical": identical,
        "differing": [] if identical else ["definitions/a.json"]}}))


def test_an_intact_reproducible_package_passes(tmp_path):
    _package(tmp_path)
    row = g._baseline_package(tmp_path)
    assert row["ok"], row["detail"]


def test_a_tampered_part_fails(tmp_path):
    _package(tmp_path)
    (tmp_path / "baseline" / "v1" / "definitions" / "a.json").write_text('{"x": 1}\n')
    assert "hash mismatch" in g._baseline_package(tmp_path)["detail"]


def test_a_package_without_a_reexport_receipt_fails(tmp_path):
    _package(tmp_path)
    (tmp_path / "baseline" / "receipts" / "v1.json").unlink()
    assert "re-export receipt" in g._baseline_package(tmp_path)["detail"]


def test_a_reexport_that_was_not_byte_identical_fails(tmp_path):
    _package(tmp_path, identical=False)
    assert "NOT byte-identical" in g._baseline_package(tmp_path)["detail"]


def test_a_reexport_receipt_for_another_package_fails(tmp_path):
    _package(tmp_path, receipt_hash="sha256:other")
    assert "different package_hash" in g._baseline_package(tmp_path)["detail"]


def test_lock_drift_fails(tmp_path):
    _package(tmp_path)
    (tmp_path / "requirements.txt").write_text("numpy==9.9.9\n")
    assert "differs" in g._baseline_package(tmp_path)["detail"]


def test_no_current_pointer_fails_rather_than_guessing_the_latest(tmp_path):
    _package(tmp_path)
    (tmp_path / "baseline" / "CURRENT").unlink()
    assert not g._baseline_package(tmp_path)["ok"]


# --------------------------------------------------------------------------
# changed since the previous run — information, not a verdict
# --------------------------------------------------------------------------


def test_a_fingerprint_ignores_timing_and_follows_the_answer():
    a = {"ok": False, "seconds": 1.0, "detail": "stale"}
    assert g.fingerprint(a) == g.fingerprint({**a, "seconds": 9.0})
    assert g.fingerprint(a) != g.fingerprint({**a, "detail": "code changed"})


def test_a_duration_inside_the_detail_is_not_a_change():
    """pytest's "224 passed in 4.17s" must not flag the row as changed every run."""
    one = {"ok": True, "detail": "224 passed in 4.17s"}
    assert g.fingerprint(one) == g.fingerprint({**one, "detail": "224 passed in 3.98s"})
    assert g.fingerprint(one) != g.fingerprint({**one, "detail": "223 passed in 4.17s"})


def test_changed_since_names_only_the_rows_whose_answer_moved():
    previous = {"fingerprints": {"tier0_corpus": "sha256:a", "code_budgets": "sha256:b"}}
    now = {"tier0_corpus": "sha256:a", "code_budgets": "sha256:c", "new_row": "sha256:d"}
    assert g.changed_since(previous, now) == ["code_budgets", "new_row"]
