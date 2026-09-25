"""P6-6 controlled-failure drill: a named crash point must leave recoverable
state, and the same production code must finish on retry.

The published fixture is real production code end to end
(``checks.rearchitecture_phase6_restore_drill``'s ``_build_fixture`` plus its
new ``_build_published_fixture`` layer: one committed decision, one backed-up
score fixture, one exported generation, one backup, one staged and published
``R0``). No provider call, no real panel, no decision in this file -- the
negative side is a ``fault=`` hook that already exists in
``publication.publish_local`` / ``backup.run_backup``.

The CLI is the interface under test, exactly as an operator runs it: the
receipt it prints is parsed and the exit code must match the verdict.
"""
from __future__ import annotations

import json
import sqlite3

from checks.rearchitecture_phase6_restore_drill import _build_published_fixture
from engine.v2.ops.publication import current
from tools import v2_controlled_failure_drill

SCENARIOS = ("after-good-release", "before-any-release")
CAPABILITIES = {
    "after-good-release": ["nightly-publish", "nightly-backup"],
    "before-any-release": ["nightly-backup"],
}


def _run_cli(tmp_path, scenario, *, artifact_root=None):
    scratch = tmp_path / ("scratch-" + scenario)
    argv = ["--scenario", scenario, "--scratch-root", str(scratch)]
    if artifact_root is not None:
        argv += ["--artifact-root", str(artifact_root)]
    code = v2_controlled_failure_drill.main(argv)
    return scratch, code


def _receipt(capsys):
    return json.loads(capsys.readouterr().out)


def test_published_fixture_layers_a_live_r0_over_the_real_backup_fixture(tmp_path):
    (backup_dir, original_export_dir, expected_score_hash, catalog_path, store_root,
     publication_target, conn_factory) = _build_published_fixture(tmp_path)

    assert backup_dir.is_dir()
    assert original_export_dir.is_dir()
    assert catalog_path.is_file()
    assert (publication_target / "releases" / "R0" / "request.json").is_file()
    assert current(publication_target) == "R0"
    assert store_root.is_dir()
    assert expected_score_hash.startswith("sha256:")

    conn = conn_factory()
    try:
        row = conn.execute(
            "SELECT published_at FROM releases WHERE release_id='R0'").fetchone()
        assert row is not None and row["published_at"] is not None
    finally:
        conn.close()


def test_after_good_release_crash_recovers_and_never_tears_current(tmp_path, capsys):
    scratch, code = _run_cli(tmp_path, "after-good-release")
    receipt = _receipt(capsys)

    assert code == 0
    assert receipt["verdict"] == "PASS"
    assert receipt["schema_version"] == "controlled_failure_drill_receipt.v1.0"
    assert receipt["scenario"] == "after-good-release"
    assert receipt["fault_point"] == "pointer_before_ack"
    assert receipt["injected_exception_raised"] is True
    assert receipt["capabilities_covered"] == ["nightly-publish", "nightly-backup"]

    crash = receipt["state_after_crash"]
    assert crash["current"] == "R1"
    assert crash["release_bytes_verified"] is True
    assert crash["published_at"] is None

    retry = receipt["retry"]
    assert retry["delivered"] is True
    assert retry["published_at"] is not None
    assert retry["current"] == "R1"

    target = scratch / "publication"
    assert current(target) == "R1"
    assert (target / "CURRENT").read_text() == "R1\n"
    assert (target / "releases" / "R1" / "index.html").read_bytes() == \
        b"<html>controlled-failure R1</html>"
    assert (target / "releases" / "R0" / "request.json").is_file()


def test_before_any_release_leaves_the_publication_target_untouched(tmp_path, capsys):
    scratch, code = _run_cli(tmp_path, "before-any-release")
    receipt = _receipt(capsys)

    assert code == 0
    assert receipt["verdict"] == "PASS"
    assert receipt["scenario"] == "before-any-release"
    assert receipt["fault_point"] == "after_manifest_before_ack"
    assert receipt["injected_exception_raised"] is True
    assert receipt["capabilities_covered"] == ["nightly-backup"]

    crash = receipt["state_after_crash"]
    assert crash["manifest_present"] is True
    assert crash["manifest_verified"] is True
    assert crash["outbox_state"] == "running"
    assert crash["outbox_state_after_recovery"] == "pending"
    assert crash["current"] is None

    retry = receipt["retry"]
    assert retry["manifest_matches"] is True
    assert retry["outbox_state"] == "delivered"
    assert retry["current"] is None

    assert not (scratch / "publication").exists()


def test_cli_exit_code_matches_verdict_for_both_scenarios(tmp_path, capsys):
    for scenario in SCENARIOS:
        _, code = _run_cli(tmp_path, scenario)
        receipt = _receipt(capsys)
        assert receipt["verdict"] == "PASS"
        assert code == {"PASS": 0, "FAIL": 1}[receipt["verdict"]]


def test_evidence_receipt_copies_to_artifact_root_in_scratch_mode(
        tmp_path, capsys, monkeypatch):
    fake_root = tmp_path / "fake-root"
    monkeypatch.setattr(v2_controlled_failure_drill, "ROOT", fake_root)
    for scenario in SCENARIOS:
        evidence = tmp_path / ("evidence-" + scenario)
        _, code = _run_cli(tmp_path, scenario, artifact_root=evidence)
        receipt = _receipt(capsys)

        assert code == 0
        name = f"controlled_failure_{scenario}_receipt.json"
        copy = evidence / name
        assert copy.is_file()
        assert json.loads(copy.read_text()) == receipt
        assert receipt["capabilities_covered"] == CAPABILITIES[scenario]

        assert not (fake_root / "reports" / "phase6_evidence"
                    / "controlled_failure" / name).exists()


def test_evidence_receipt_written_under_root_against_real_candidate(
        tmp_path, capsys, monkeypatch):
    fake_root = tmp_path / "fake-root"
    monkeypatch.setattr(v2_controlled_failure_drill, "ROOT", fake_root)
    backup, _, _, catalog, store_root, target, _ = _build_published_fixture(tmp_path)
    scratch = tmp_path / "candidate-evidence-scratch"

    code = v2_controlled_failure_drill.main([
        "--scenario", "after-good-release", "--scratch-root", str(scratch),
        "--against-real-candidate", str(catalog), "--store-root", str(store_root),
        "--target", str(target), "--backup", str(backup)])
    receipt = _receipt(capsys)

    assert code == 0
    name = "controlled_failure_after-good-release_receipt.json"
    reported = (fake_root / "reports" / "phase6_evidence" / "controlled_failure" / name)
    assert reported.is_file()
    assert json.loads(reported.read_text()) == receipt


def test_existing_scratch_root_is_refused_with_exit_2(tmp_path, capsys):
    existing = tmp_path / "already-there"
    existing.mkdir()

    code = v2_controlled_failure_drill.main(
        ["--scenario", "before-any-release", "--scratch-root", str(existing)])

    captured = capsys.readouterr()
    assert code == 2
    assert "refused" in captured.err
    assert captured.out == ""
    assert sorted(existing.iterdir()) == []


def test_partial_real_candidate_flags_are_refused_with_exit_2(tmp_path, capsys):
    scratch = tmp_path / "half-specified"

    code = v2_controlled_failure_drill.main(
        ["--scenario", "after-good-release", "--scratch-root", str(scratch),
         "--store-root", str(tmp_path / "objects")])

    captured = capsys.readouterr()
    assert code == 2
    assert "refused" in captured.err
    assert not scratch.exists()


def test_against_real_candidate_rehearses_on_copies_and_leaves_the_candidate_alone(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(v2_controlled_failure_drill, "ROOT", tmp_path / "fake-root")
    backup, _, _, catalog, store_root, target, _ = _build_published_fixture(tmp_path)
    assert current(target) == "R0"
    scratch = tmp_path / "candidate-scratch"

    code = v2_controlled_failure_drill.main([
        "--scenario", "after-good-release", "--scratch-root", str(scratch),
        "--against-real-candidate", str(catalog), "--store-root", str(store_root),
        "--target", str(target), "--backup", str(backup)])
    receipt = _receipt(capsys)

    assert code == 0
    assert receipt["verdict"] == "PASS"
    assert current(scratch / "target") == "R1"
    assert (scratch / "copy" / "ops.sqlite").is_file()

    # The real deployment is untouched: CURRENT still R0, no R1 in the catalog.
    assert current(target) == "R0"
    probe = sqlite3.connect("file:" + catalog.resolve().as_posix() + "?mode=ro", uri=True)
    try:
        release_ids = {row[0] for row in probe.execute("SELECT release_id FROM releases")}
    finally:
        probe.close()
    assert release_ids == {"R0"}
