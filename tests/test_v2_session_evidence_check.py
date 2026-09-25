"""P6-6 session-evidence completeness check, against fixture declarations
and a fixture evidence directory (no real evidence tree is required).

The fixture toml carries the three row shapes that matter -- one row needing
evidence, one ``missing``, one ``dormant-historical`` -- plus a
``[[user_decision]]`` entry, which ``build_document`` keeps under its own
``user_decisions`` key and the check must never treat as a row.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tools import v2_session_evidence_check as sec

FIXTURE_TOML = textwrap.dedent('''
    [[row]]
    id = "needs-evidence"
    area = "board"
    capability = "c"
    new = []
    producer = "p"
    identity = "i"
    tests = []
    disposition = "native"
    owner = "P6-4"

    [[row]]
    id = "declared-missing"
    area = "board"
    capability = "c"
    new = ["MISSING"]
    producer = "p"
    identity = "i"
    tests = []
    disposition = "missing"
    owner = "P6-2"

    [[row]]
    id = "dormant-old"
    area = "research"
    capability = "c"
    new = []
    producer = "p"
    identity = "i"
    tests = []
    disposition = "dormant-historical"
    owner = "8A"

    [[user_decision]]
    id = "UD-9"
    rows = ["needs-evidence"]
    question = "accept?"
''')

NO_DISPOSITION_ROW = textwrap.dedent('''
    [[row]]
    id = "undeclared"
    area = "board"
    capability = "c"
    new = []
    producer = "p"
    identity = "i"
    tests = []
    owner = "P6-4"
''')


@pytest.fixture
def fixtures(tmp_path):
    declarations = tmp_path / "capabilities.toml"
    declarations.write_text(FIXTURE_TOML)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    return declarations, evidence


def _write_receipt(evidence: Path, name: str, covered) -> Path:
    path = evidence / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"capabilities_covered": list(covered)}))
    return path


def test_rows_are_covered_and_the_user_decision_is_ignored(tmp_path, fixtures):
    declarations, evidence = fixtures
    _write_receipt(evidence, "session/receipt.json", ["needs-evidence"])

    result = sec.check(declarations=declarations, evidence_dir=evidence)

    assert result == {
        "rows_total": 3,
        "rows_exempt": 2,
        "rows_covered": 1,
        "rows_uncovered": [],
        "unreadable_evidence_files": [],
    }


def test_uncovered_row_exits_one_naming_exactly_that_id(tmp_path, fixtures, capsys):
    declarations, evidence = fixtures
    _write_receipt(evidence, "receipt.json", ["declared-missing"])  # exempt row; not enough

    code = sec.main(["--json", "--declarations", str(declarations),
                     "--evidence-dir", str(evidence)])

    assert code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["rows_uncovered"] == ["needs-evidence"]


def test_unreadable_evidence_file_is_reported_not_fatal(tmp_path, fixtures, capsys):
    declarations, evidence = fixtures
    bad = evidence / "broken.json"
    bad.write_text("{ this is not json")
    _write_receipt(evidence, "good.json", ["needs-evidence"])

    code = sec.main(["--json", "--declarations", str(declarations),
                     "--evidence-dir", str(evidence)])

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rows_uncovered"] == []
    assert result["unreadable_evidence_files"] == [str(bad)]


def test_row_without_a_disposition_key_is_not_exempt(tmp_path, fixtures):
    declarations, evidence = fixtures
    declarations.write_text(FIXTURE_TOML + NO_DISPOSITION_ROW)
    _write_receipt(evidence, "receipt.json", ["needs-evidence"])

    result = sec.check(declarations=declarations, evidence_dir=evidence)

    assert result["rows_total"] == 4
    assert result["rows_exempt"] == 2
    assert result["rows_uncovered"] == ["undeclared"]


def test_evidence_scan_is_recursive_and_unions_every_receipt(tmp_path, fixtures):
    declarations, evidence = fixtures
    _write_receipt(evidence, "resource_measurement/run.json", ["needs-evidence", "other"])
    _write_receipt(evidence, "top.json", [])

    result = sec.check(declarations=declarations, evidence_dir=evidence)

    assert result["rows_uncovered"] == []
    assert result["rows_covered"] == 1


def test_json_flag_prints_only_the_json_document(tmp_path, fixtures, capsys):
    declarations, evidence = fixtures
    _write_receipt(evidence, "receipt.json", ["needs-evidence"])

    code = sec.main(["--json", "--declarations", str(declarations),
                     "--evidence-dir", str(evidence)])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["rows_uncovered"] == []


def test_human_output_names_the_uncovered_rows(tmp_path, fixtures, capsys):
    declarations, evidence = fixtures

    code = sec.main(["--declarations", str(declarations), "--evidence-dir", str(evidence)])

    assert code == 1
    out = capsys.readouterr().out
    assert "3 rows, 2 exempt, 0 covered, 1 uncovered" in out
    assert "UNCOVERED needs-evidence" in out
