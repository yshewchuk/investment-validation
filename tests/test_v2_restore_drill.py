"""P6-5 restore drill: real backup -> restore -> replay -> reconcile.

The synthetic fixture comes from the acceptance check itself
(``checks.rearchitecture_phase6_restore_drill._build_fixture``): one committed
decision, one ``ScoreRequest``/``NativeScoreInputs`` pair backed up as
artifacts, and one exported generation, all built by real production code. No
provider calls, no real panel.
"""
from __future__ import annotations

import json

import pytest

from checks.rearchitecture_phase6_restore_drill import _build_fixture
from tools import v2_restore_drill
from tools.v2_restore_drill import RestoreDrillError, run_drill


def _run(tmp_path, fixture, restore_root, **kwargs):
    backup_dir, original_export_dir, expected_hash = fixture
    return run_drill(
        backup_dir=backup_dir, restore_root=restore_root,
        score_request_name="request.json", native_inputs_name="native_inputs.json",
        expected_score_hash=kwargs.pop("expected_score_hash", expected_hash),
        original_export_dir=original_export_dir, generation="g1", **kwargs)


def test_real_backup_replays_the_same_score_and_reconciles(tmp_path):
    fixture = _build_fixture(tmp_path)

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1)

    assert receipt["verdict"] == "PASS"
    assert receipt["score_replay"]["match"] is True
    assert receipt["ledger_reconciliation"]["match"] is True


def test_wrong_expected_hash_fails_the_drill(tmp_path):
    fixture = _build_fixture(tmp_path)

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1,
                   expected_score_hash="sha256:" + "0" * 64)

    assert receipt["verdict"] == "FAIL"
    assert receipt["score_replay"]["match"] is False


def test_wrong_expected_decisions_count_fails_the_drill(tmp_path):
    fixture = _build_fixture(tmp_path)

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=99)

    assert receipt["verdict"] == "FAIL"
    assert receipt["ledger_reconciliation"]["match"] is False


def test_restoring_into_an_existing_directory_raises(tmp_path):
    fixture = _build_fixture(tmp_path)
    restore_root = tmp_path / "restored"
    restore_root.mkdir()

    with pytest.raises(RestoreDrillError):
        _run(tmp_path, fixture, restore_root, expected_decisions_count=1)


def test_writes_an_evidence_receipt_when_artifact_root_given(tmp_path):
    fixture = _build_fixture(tmp_path)

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1,
                   artifact_root=tmp_path / "evidence")

    path = tmp_path / "evidence" / "restore_drill_receipt.json"
    assert path.is_file()
    assert json.loads(path.read_text()) == receipt


def test_cli_exit_code_matches_verdict(tmp_path, capsys):
    backup_dir, original_export_dir, expected_hash = _build_fixture(tmp_path)

    code = v2_restore_drill.main([
        "--backup", str(backup_dir), "--restore-root", str(tmp_path / "restored-cli"),
        "--score-request", "request.json", "--native-inputs", "native_inputs.json",
        "--expected-score-hash", expected_hash, "--original-export", str(original_export_dir),
        "--generation", "g1", "--expected-decisions-count", "1"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"