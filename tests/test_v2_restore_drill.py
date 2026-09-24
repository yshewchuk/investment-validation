"""P6-5 restore drill: real backup -> restore -> replay -> reconcile.

The synthetic fixture comes from the acceptance check itself
(``checks.rearchitecture_phase6_restore_drill._build_fixture``): one committed
decision, one ``ScoreRequest``/``NativeScoreInputs`` pair backed up as
artifacts, and one exported generation, all built by real production code. No
provider calls, no real panel.

Review C6 regression coverage: the drill must never report PASS without
actually comparing baseline export bytes -- a missing/symlinked baseline is
refused; empty, missing, extra, type-mismatched, symlinked or byte-differing
export files fail; and a legitimate zero-decision export passes only on
attested affirmative expected-zero evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from checks.rearchitecture_phase6_restore_drill import _build_fixture
from engine.v2.foundation import ArtifactStore, SystemClock
from engine.v2.ledger.export import export_generation
from engine.v2.ops.backup import prepare_backup, run_backup
from engine.v2.ops.bootstrap import open_catalog
from tools import v2_restore_drill
from tools.v2_restore_drill import RestoreDrillError, run_drill


def _run(tmp_path, fixture, restore_root, **kwargs):
    backup_dir, original_export_dir, expected_hash = fixture
    return run_drill(
        backup_dir=kwargs.pop("backup_dir", backup_dir), restore_root=restore_root,
        score_request_name="request.json", native_inputs_name="native_inputs.json",
        expected_score_hash=kwargs.pop("expected_score_hash", expected_hash),
        original_export_dir=kwargs.pop("original_export_dir", original_export_dir),
        generation="g1", **kwargs)


def _first_baseline_file(original_export_dir: Path) -> Path:
    files = sorted(p for p in original_export_dir.rglob("*") if p.is_file() and not p.is_symlink())
    assert files
    return files[0]


def _zero_decision_backup(tmp_path, fixture):
    """Real production code, one variation: the SAME two score artifacts over
    a catalog with ZERO decisions, plus that empty catalog's genuine (empty,
    CURRENT-pointed) g1 export. Returns (backup_dir, zero_export_dir)."""
    backup_dir = fixture[0]
    (tmp_path / "zero").mkdir()
    clock = SystemClock()
    conn = open_catalog(tmp_path / "zero" / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path / "zero" / "objects")
    artifacts = {}
    for name, schema in (("request.json", "score_request.v1.0"),
                         ("native_inputs.json", "native_score_inputs.v1.0")):
        artifacts[name] = store.publish_bytes(
            (backup_dir / "artifacts" / name).read_bytes(), schema_ref=schema)
    prepare_backup(conn, "zero-1", artifacts, clock=clock)
    zero_backup = tmp_path / "zero" / "backup"
    run_backup(conn, key="zero-1", owner="operator", target=zero_backup, clock=clock, store=store)
    zero_export = export_generation(conn, tmp_path / "zero" / "live-export", generation="g1")
    conn.close()
    assert list(zero_export.rglob("*")) == []
    assert (zero_export.parent / "CURRENT").read_text() == "g1\n"
    return zero_backup, zero_export


def _tamper_restored_export(monkeypatch, mutate):
    """Alter the freshly written RESTORED export after the real
    export_generation ran -- so the original baseline stays untouched and the
    divergence is entirely on the restored side."""
    real = v2_restore_drill.export_generation

    def wrapper(conn, root, **kwargs):
        destination = real(conn, root, **kwargs)
        mutate(destination)
        return destination

    monkeypatch.setattr(v2_restore_drill, "export_generation", wrapper)


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


def test_missing_original_export_dir_is_refused(tmp_path):
    fixture = _build_fixture(tmp_path)

    with pytest.raises(RestoreDrillError, match="original export directory"):
        _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1,
             original_export_dir=tmp_path / "never-exported")


def test_cli_refuses_a_missing_original_export_with_exit_2(tmp_path, capsys):
    backup_dir, _, expected_hash = _build_fixture(tmp_path)

    code = v2_restore_drill.main([
        "--backup", str(backup_dir), "--restore-root", str(tmp_path / "restored-cli"),
        "--score-request", "request.json", "--native-inputs", "native_inputs.json",
        "--expected-score-hash", expected_hash, "--original-export", str(tmp_path / "gone"),
        "--generation", "g1", "--expected-decisions-count", "1"])

    assert code == 2
    assert "refused" in capsys.readouterr().err


def test_symlinked_original_export_dir_is_refused(tmp_path):
    fixture = _build_fixture(tmp_path)
    _, original_export_dir, _ = fixture
    link = tmp_path / "symlinked-baseline"
    os.symlink(original_export_dir, link)

    with pytest.raises(RestoreDrillError, match="symlink"):
        _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1,
             original_export_dir=link)


def test_empty_original_export_dir_fails_even_when_the_count_would_match(tmp_path):
    fixture = _build_fixture(tmp_path)
    empty = tmp_path / "arbitrary-empty" / "g1"
    empty.mkdir(parents=True)

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1,
                   original_export_dir=empty)
    ledger = receipt["ledger_reconciliation"]
    assert receipt["verdict"] == "FAIL"
    assert ledger["match"] is False
    assert ledger["baseline_problem"]
    assert ledger["files_compared"] == 0
    assert ledger["files_extra_in_restored"]

    no_claim = _run(tmp_path, fixture, tmp_path / "restored-none", original_export_dir=empty)
    assert no_claim["verdict"] == "FAIL"
    assert no_claim["ledger_reconciliation"]["baseline_problem"]


def test_baseline_symlink_is_never_read_as_trusted_evidence(tmp_path):
    fixture = _build_fixture(tmp_path)
    _, original_export_dir, _ = fixture
    victim = _first_baseline_file(original_export_dir)
    relative = victim.relative_to(original_export_dir).as_posix()
    twin = tmp_path / "same-bytes-elsewhere.jsonl"
    twin.write_bytes(victim.read_bytes())
    victim.unlink()
    os.symlink(twin, victim)

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1)
    ledger = receipt["ledger_reconciliation"]
    assert receipt["verdict"] == "FAIL"
    assert any("original/" + relative in entry for entry in ledger["entries_untrusted"])
    assert relative in ledger["files_type_mismatched"]


def test_baseline_file_missing_from_the_restored_export_fails(tmp_path):
    fixture = _build_fixture(tmp_path)
    _, original_export_dir, _ = fixture
    (original_export_dir / "predictions" / "2099-01-01.jsonl").write_bytes(b"{}\n")

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1)
    ledger = receipt["ledger_reconciliation"]
    assert receipt["verdict"] == "FAIL"
    assert ledger["files_missing_in_restored"] == ["predictions/2099-01-01.jsonl"]


def test_extra_restored_export_file_fails_although_the_count_matches(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path)
    _tamper_restored_export(
        monkeypatch, lambda export_dir: (export_dir / "predictions" / "ghost.jsonl")
        .write_bytes(b"{}\n"))

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1)
    ledger = receipt["ledger_reconciliation"]
    assert receipt["verdict"] == "FAIL"
    assert ledger["decisions_count"] == 1
    assert ledger["files_extra_in_restored"] == ["predictions/ghost.jsonl"]


def test_byte_mismatch_fails_and_never_touches_the_baseline(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path)
    _, original_export_dir, _ = fixture
    victim = _first_baseline_file(original_export_dir)
    relative = victim.relative_to(original_export_dir).as_posix()
    baseline_bytes = victim.read_bytes()
    _tamper_restored_export(
        monkeypatch, lambda export_dir: (export_dir / relative).write_bytes(
            (export_dir / relative).read_bytes() + b"tampered\n"))

    receipt = _run(tmp_path, fixture, tmp_path / "restored", expected_decisions_count=1)
    ledger = receipt["ledger_reconciliation"]
    assert receipt["verdict"] == "FAIL"
    assert ledger["files_mismatched"] == [relative]
    assert victim.read_bytes() == baseline_bytes


def test_attested_zero_decision_export_passes_only_with_affirmative_expected_zero(tmp_path):
    fixture = _build_fixture(tmp_path)
    zero_backup, zero_export = _zero_decision_backup(tmp_path, fixture)

    passed = _run(tmp_path, fixture, tmp_path / "restored-zero", backup_dir=zero_backup,
                  original_export_dir=zero_export, expected_decisions_count=0)
    ledger = passed["ledger_reconciliation"]
    assert passed["verdict"] == "PASS"
    assert ledger["match"] is True
    assert ledger["files_compared"] == 0
    assert ledger["decisions_count"] == 0

    unclaimed = _run(tmp_path, fixture, tmp_path / "restored-zero-none", backup_dir=zero_backup,
                     original_export_dir=zero_export)
    assert unclaimed["verdict"] == "FAIL"
    assert unclaimed["ledger_reconciliation"]["baseline_problem"]


def test_unattested_empty_directory_is_no_proof_even_with_expected_zero(tmp_path):
    fixture = _build_fixture(tmp_path)
    zero_backup, _ = _zero_decision_backup(tmp_path, fixture)
    arbitrary = tmp_path / "handed-in" / "g1"
    arbitrary.mkdir(parents=True)

    receipt = _run(tmp_path, fixture, tmp_path / "restored-unattested", backup_dir=zero_backup,
                   original_export_dir=arbitrary, expected_decisions_count=0)
    ledger = receipt["ledger_reconciliation"]
    assert receipt["verdict"] == "FAIL"
    assert "CURRENT" in ledger["baseline_problem"]


def test_empty_sibling_of_attested_zero_export_root_is_no_proof(tmp_path):
    """A genuine export root (CURRENT names g1) does not attest an EMPTY
    SIBLING of its g1 directory: the baseline must be the directory the
    pointer names, so the sibling fails while the genuine zero-decision
    export still passes with expected count 0."""
    fixture = _build_fixture(tmp_path)
    zero_backup, zero_export = _zero_decision_backup(tmp_path, fixture)
    sibling = zero_export.parent / "unrelated-empty"
    sibling.mkdir()

    receipt = _run(tmp_path, fixture, tmp_path / "restored-sibling", backup_dir=zero_backup,
                   original_export_dir=sibling, expected_decisions_count=0)
    ledger = receipt["ledger_reconciliation"]
    assert receipt["verdict"] == "FAIL"
    assert ledger["match"] is False
    assert "CURRENT" in ledger["baseline_problem"]

    genuine = _run(tmp_path, fixture, tmp_path / "restored-named", backup_dir=zero_backup,
                   original_export_dir=zero_export, expected_decisions_count=0)
    assert genuine["verdict"] == "PASS"
    assert genuine["ledger_reconciliation"]["baseline_problem"] is None
