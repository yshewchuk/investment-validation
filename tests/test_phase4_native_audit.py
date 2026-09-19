from copy import deepcopy
from pathlib import Path

from checks.phase4_native_audit import audit

ROOT = Path(__file__).resolve().parents[1]


def _safe_root(tmp_path: Path) -> Path:
    application = tmp_path / "engine/v2/scoring/application.py"
    application.parent.mkdir(parents=True)
    application.write_text(
        "def score_one(request: ScoreRequest, inputs: NativeScoreInputs):\n"
        "    return execute_native_stages(request, inputs)\n"
    )
    return tmp_path


def _complete_evidence() -> dict:
    dimensions = [
        "keys", "contracts", "verdicts", "flags", "null_masks",
        "forecasts", "simulation", "financial_diagnostics",
    ]
    stages = [
        "resolve_context", "features", "forecast", "geometry", "pricing",
        "analogs", "simulation", "gate", "chooser", "serialization",
    ]
    return {
        "completion_controls": {"full_saved_release_compared": True},
        "saved_release_comparison": {
            "complete": True,
            "population": {"expected": 121, "compared": 121},
            "source_release": {
                "release_id": "release-1",
                "manifest_hash": "sha256:source",
            },
            "native_execution_receipt": "sha256:native",
            "legacy_execution_receipt": "sha256:legacy",
            "comparison_receipt": "sha256:comparison",
            "comparison_dimensions": dimensions,
        },
        "native_parity": {
            "synthetic": False,
            "input_provenance": {
                "kind": "saved_release",
                "release_id": "release-1",
                "manifest_hash": "sha256:source",
            },
            "same_input_hashes": True,
            "population": {"expected": 121, "compared": 121},
            "stages": stages,
            "comparison_dimensions": dimensions,
            "planted_defect": {
                "detected": True,
                "receipt": "sha256:planted-defect",
            },
        },
    }


def test_rejects_canonical_precomputed_mapping_passthrough():
    bad_root = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "phase4-bad"
    result = audit(bad_root, _complete_evidence())

    assert result["status"] == "FAIL"
    assert result["finding_ids"] == ["P4N-001"]
    assert "mapping_record_sinks" in " ".join(result["findings"][0]["facts"])


def test_rejects_hash_only_saved_release_claim(tmp_path):
    evidence = _complete_evidence()
    evidence.pop("saved_release_comparison")
    evidence["artifact_hashes_verified"] = True

    result = audit(_safe_root(tmp_path), evidence)

    assert result["finding_ids"] == ["P4N-002"]
    assert result["ok"] is False


def test_rejects_saved_release_without_numeric_dimensions(tmp_path):
    evidence = _complete_evidence()
    evidence["saved_release_comparison"]["comparison_dimensions"] = [
        "keys", "contracts", "verdicts", "flags", "null_masks",
    ]

    result = audit(_safe_root(tmp_path), evidence)

    assert result["finding_ids"] == ["P4N-002"]
    assert "financial_diagnostics" in " ".join(result["findings"][0]["facts"])


def test_rejects_synthetic_parity_even_with_green_controls(tmp_path):
    evidence = _complete_evidence()
    evidence["native_parity"]["synthetic"] = True
    evidence["native_parity"]["input_provenance"]["kind"] = "synthetic_fixture"

    result = audit(_safe_root(tmp_path), evidence)

    assert result["finding_ids"] == ["P4N-003"]
    facts = " ".join(result["findings"][0]["facts"])
    assert "synthetic=True" in facts
    assert "synthetic_fixture" in facts


def test_accepts_typed_native_execution_with_complete_real_receipts(tmp_path):
    result = audit(_safe_root(tmp_path), deepcopy(_complete_evidence()))

    assert result == {
        "schema_version": "phase4_native_audit.v1.0",
        "status": "PASS",
        "ok": True,
        "finding_ids": [],
        "findings": [],
    }


def _with_excluded(excluded) -> dict:
    evidence = _complete_evidence()
    for section in ("saved_release_comparison", "native_parity"):
        evidence[section]["population"]["excluded"] = deepcopy(excluded)
    return evidence


def test_accepts_only_approved_excluded_kinds(tmp_path):
    from checks.phase4_real import PHASE4_EXCLUDED_RECORD_KINDS

    approved = {kind: 3 for kind in PHASE4_EXCLUDED_RECORD_KINDS}
    result = audit(_safe_root(tmp_path), _with_excluded(approved))

    assert result["ok"] is True


def test_flags_an_excluded_kind_not_approved_by_phase4_real(tmp_path):
    result = audit(_safe_root(tmp_path),
                   _with_excluded({"research_replay": 2, "dyn_sv_choice": 5}))

    assert result["finding_ids"] == ["P4N-002", "P4N-003"]
    for finding in result["findings"]:
        facts = " ".join(finding["facts"])
        assert "population_excluded_unapproved_kinds=['dyn_sv_choice']" in facts
        assert "research_replay" not in facts


def test_flags_malformed_excluded_population(tmp_path):
    root = _safe_root(tmp_path)
    not_mapping = audit(root, _with_excluded(["research_replay"]))
    bad_count = audit(root, _with_excluded({"research_replay": -1}))
    bool_count = audit(root, _with_excluded({"research_replay": True}))

    assert not_mapping["finding_ids"] == ["P4N-002", "P4N-003"]
    assert "population_excluded_type=list" in " ".join(not_mapping["findings"][0]["facts"])
    for result in (bad_count, bool_count):
        assert result["finding_ids"] == ["P4N-002", "P4N-003"]
        assert "population_excluded_bad_counts=['research_replay']" in " ".join(
            result["findings"][1]["facts"])
