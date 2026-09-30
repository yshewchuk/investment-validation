"""Public failure controls plus an opt-in retained, licensed capture regression."""
import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.foundation import content_hash
from engine.v2.serving.native_parity_projection import native_parity_summary
from tools import captured_native_comparison as export
from tests.test_phase4_targeted_replay import _clean_record, _native_record, _pair_doc, _write_corpus


@pytest.fixture
def selected(tmp_path, monkeypatch):
    """Synthetic shape/control only; financial acceptance uses the private test."""
    record = _clean_record(ticker="TEST", strategy="STR-THRU", event_date="2026-09-17",
                           session="BMO", as_of="2026-09-16")
    pair = _pair_doc("one", record)
    pair["payload"]["input_trace"] = {
        "request": {"requested_decision_at": "2026-09-12"},
        "request_hash": "request-1",
        "metadata": {"frozen_inference": {"release_resource_id": "release-1"}},
    }
    pair["payload_hash"] = content_hash(pair["payload"])
    declared = {"one": {key: pair[key] for key in ("payload_hash", "request_hash", "covers")}}
    declared["one"]["record_kind"] = "score_result"
    root = _write_corpus(tmp_path / "corpus", [pair], declared=declared)
    verified = {"frozen_replay": object(), "trace_hash": "trace-1",
                "same_input_receipt": "receipt-1",
                "inputs": SimpleNamespace(context={"chain_as_of": "2026-09-12"})}
    monkeypatch.setattr(export.phase4_real, "_verified_trace_bundle", lambda *args: verified)
    native = _native_record(record)
    monkeypatch.setattr(export.phase4_real, "_replayed_member", lambda *args: (native, (1,) * 12, ()))
    return root, pair, native


def test_export_preserves_dates_values_and_existing_reader(selected, tmp_path):
    root, _, _ = selected
    report = export.build_captured_comparison(root, "one")
    captured = report["captured_comparison"]
    assert captured["clocks"]["requested_decision_at"] == "2026-09-12"
    assert captured["clocks"]["decision_as_of"] == "2026-09-16"
    assert captured["clocks"]["corpus_as_of"] is None
    assert captured["runtime_stage_count"] == 12
    assert captured["legacy"] == captured["native"]
    assert report["mismatches"] == []
    assert not captured["full_population_verified"]
    assert not captured["cutover_qualified"] and not captured["current_board"]
    assert str(root) not in json.dumps(report)
    assert export.build_captured_comparison(root, "one") == report
    output = tmp_path / "report.json"
    assert export.main(["--corpus", str(root), "--fixture-id", "one", "--output", str(output)]) == 0
    status, summary = native_parity_summary(output)
    assert status == 200 and summary["status"] == "available"


@pytest.mark.parametrize("fixture_id", ["missing", "../one", "", "."])
def test_refuses_invalid_selection(selected, fixture_id):
    with pytest.raises(export.ComparisonRefused):
        export.build_captured_comparison(selected[0], fixture_id)


@pytest.mark.parametrize("target", ["payload", "manifest", "request", "kind", "strategy"])
def test_tampered_capture_never_publishes(selected, tmp_path, target):
    root, pair, _ = selected
    if target == "payload":
        pair["payload"]["record"]["spot"] += 1
    elif target == "manifest":
        pair["covers"] = ["changed"]
    elif target == "request":
        pair["request_hash"] = "wrong"
    else:
        if target == "kind":
            pair["payload"]["record_kind"] = "dyn_sv_choice"
        else:
            pair["payload"]["record"]["strategy"] = "RUNUP"
        pair["payload_hash"] = content_hash(pair["payload"])
        index = json.loads((root / "INDEX.json").read_text())
        index["pairs"]["one"]["payload_hash"] = pair["payload_hash"]
        index["pairs"]["one"]["record_kind"] = pair["payload"]["record_kind"]
        index["corpus_hash"] = content_hash({"one": pair["payload_hash"]})
        (root / "INDEX.json").write_text(json.dumps(index))
    (root / "pairs" / "one.json").write_text(json.dumps(pair))
    output = tmp_path / "report.json"
    output.write_text("previous")
    assert export.main(["--corpus", str(root), "--fixture-id", "one", "--output", str(output)]) == 1
    assert output.read_text() == "previous"
    assert not list(tmp_path.glob("report.json.tmp*"))


@pytest.mark.parametrize("defect", ["resource", "runtime", "identity", "missing_identity", "no_frozen"])
def test_verification_and_identity_refusal_preserves_output(selected, tmp_path, monkeypatch, defect):
    root, _, native = selected
    def refuse(*args):
        raise ValueError("private /source/path must never escape")
    if defect == "resource":
        monkeypatch.setattr(export.phase4_real, "_verified_trace_bundle", refuse)
    elif defect == "runtime":
        monkeypatch.setattr(export.phase4_real, "_replayed_member", refuse)
    elif defect == "no_frozen":
        monkeypatch.setattr(export.phase4_real, "_verified_trace_bundle", lambda *args: {"frozen_replay": None})
    else:
        resolved = dict(native.resolved_request, as_of=None if defect == "missing_identity" else "wrong")
        native = replace(native, resolved_request=resolved)
        monkeypatch.setattr(export.phase4_real, "_replayed_member", lambda *args: (native, (), ()))
    output = tmp_path / "report.json"
    output.write_text("previous")
    assert export.main(["--corpus", str(root), "--fixture-id", "one", "--output", str(output)]) == 1
    assert output.read_text() == "previous"
    with pytest.raises(export.ComparisonRefused) as error:
        export.build_captured_comparison(root, "one")
    assert "/source/path" not in str(error.value)


def test_source_output_and_symlink_are_refused(selected, tmp_path):
    root = selected[0]
    original = (root / "INDEX.json").read_bytes()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    for output in (root / "INDEX.json", alias / "new.json"):
        assert export.main(["--corpus", str(root), "--fixture-id", "one", "--output", str(output)]) == 1
    assert (root / "INDEX.json").read_bytes() == original
    assert not (root / "new.json").exists()


@pytest.mark.parametrize("field", ["request_hash", "metadata"])
def test_missing_provenance_is_not_published(selected, monkeypatch, field):
    root, pair, _ = selected
    index, _ = export._selected_pair(root, "one")
    del pair["payload"]["input_trace"][field]
    # Isolate the post-verification projection boundary; digest tampering is
    # separately tested above using the real selected-payload loader.
    monkeypatch.setattr(export, "_selected_pair", lambda *args: (index, pair))
    with pytest.raises(export.ComparisonRefused):
        export.build_captured_comparison(root, "one")


@pytest.mark.parametrize("stage", ["write", "replace"])
def test_publication_failure_removes_temp_and_preserves_output(selected, tmp_path, monkeypatch, stage):
    output = tmp_path / "report.json"
    output.write_text("previous")
    write = Path.write_text
    def refuse(path, *args, **kwargs):
        if stage == "write":
            write(path, "partial")
        raise OSError("private /source/path")
    monkeypatch.setattr(Path, "write_text" if stage == "write" else "replace", refuse)
    assert export.main(["--corpus", str(selected[0]), "--fixture-id", "one", "--output", str(output)]) == 1
    assert output.read_text() == "previous"
    assert not list(tmp_path.glob("report.json.tmp*"))


def test_cli_reports_safe_typed_refusal(selected, tmp_path, capsys):
    assert export.main(["--corpus", str(selected[0]), "--fixture-id", "missing",
                        "--output", str(tmp_path / "report.json")]) == 1
    assert "fixture is not manifest-declared" in capsys.readouterr().err


@pytest.mark.needs_corpus
def test_retained_real_replay_and_planted_native_defect(monkeypatch):
    """Use a completed Phase 4 paired capture; never commit its licensed values.

    CAPTURED_COMPARISON_CORPUS and CAPTURED_COMPARISON_FIXTURE explicitly select
    retained data. A configured missing path fails instead of silently skipping.
    """
    configured = os.environ.get("CAPTURED_COMPARISON_CORPUS")
    if not configured:
        pytest.skip("explicit private captured comparison corpus is not configured")
    root = Path(configured)
    fixture_id = os.environ["CAPTURED_COMPARISON_FIXTURE"]
    observed = {}
    replay = export._replay_regular_member
    verify = export.phase4_real._verified_trace_bundle
    def capture_verify(*args):
        observed["verified"] = verify(*args)
        return observed["verified"]
    def capture_replay(*args):
        observed["member"] = replay(*args)
        return observed["member"]
    monkeypatch.setattr(export.phase4_real, "_verified_trace_bundle", capture_verify)
    monkeypatch.setattr(export, "_replay_regular_member", capture_replay)
    report = export.build_captured_comparison(root, fixture_id)
    captured = report["captured_comparison"]
    index, pair = export._selected_pair(root, fixture_id)
    legacy = pair["payload"]["record"]
    trace = pair["payload"]["input_trace"]
    assert report["mismatches"] == [] and all(captured["checks"].values())
    assert captured["runtime_stage_count"] == 12
    assert captured["provenance"]["payload_hash"] == content_hash(pair["payload"])
    assert captured["provenance"]["trace_hash"] == trace["trace_hash"]
    assert captured["provenance"]["same_input_receipt"] == observed["verified"]["same_input_receipt"]
    assert captured["clocks"]["decision_as_of"] == legacy["as_of"]
    assert captured["clocks"]["corpus_as_of"] == index["as_of"]
    assert captured["clocks"]["quote_as_of"] == trace["native_inputs"]["context"]["chain_as_of"]
    for dimension in ("forecasts", "simulation", "verdicts", "analogs"):
        assert captured["legacy"][dimension] == {key: legacy.get(key) for key in captured["legacy"][dimension]}
    financial = observed["member"]["native"].financial_diagnostics
    assert all(value == financial[key] for key, value in captured["native"]["financial_diagnostics"].items())
    assert str(root) not in json.dumps(report)
    baseline = copy.deepcopy(captured["legacy"])
    native = observed["member"]["native"]
    altered = replace(native, forecasts=dict(native.forecasts, driver_prediction=native.forecasts["driver_prediction"] + 1))
    monkeypatch.setattr(export.phase4_real, "_verified_trace_bundle", lambda *args: observed["verified"])
    monkeypatch.setattr(export.phase4_real, "_replayed_member", lambda *args: (altered, (1,) * 12, ()))
    monkeypatch.setattr(export, "_replay_regular_member", replay)
    changed = export._comparison(index, pair, root, fixture_id)
    assert changed["captured_comparison"]["legacy"] == baseline
    assert any("driver_prediction" in str(row["finding_fields"]) for row in changed["mismatches"])
