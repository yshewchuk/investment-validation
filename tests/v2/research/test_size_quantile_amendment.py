"""EXP-190 Amendment 1 re-read (experiments.size_quantile_amendment) on synthetic out-dirs built by hand. Every expected
share and verdict here is a literal written in this file; none is computed by the code under test."""
import json

import pytest
import yaml

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import size_quantile_amendment as amendment
from experiments import size_quantile_prereg as prereg

ORIG = "a" * 64
PERIODS = ["2023H2", "2024", "2025"]
YEAR = {"2023H2": 2023, "2024": 2024, "2025": 2025}
ARMS = ("Q", "Q_raw")
FLOORS = ["2023H2:scored_share_below_floor", "2024:scored_share_below_floor", "2025:scored_share_below_floor"]
RUN = {"schema_version": "size_quantile_run_inputs.v2", "mode": "full", "spec_hash": ORIG,
       "years": ["2023", "2024", "2025"]}
GAP = [("scored", "forecast_missing", 30)]  # per period drops: (arm_reason, b0_reason, n)
MIXED = [("scored", "forecast_missing", 14), ("calibration_burn_in", "forecast_missing", 6)]


def _drops(triples):
    return [{"year": YEAR[p], "arm_reason": a, "b0_reason": b, "n": n} for p in PERIODS for a, b, n in triples]


def _shares(paired):
    rows = [{"period": p, "admitted": 100, "paired": paired, "share": paired / 100, "below_floor": paired < 80}
            for p in PERIODS]
    rows.append({"period": "pooled", "admitted": 300, "paired": 3 * paired, "share": 3 * paired / 300,
                 "below_floor": 3 * paired / 300 < 0.80})
    return rows


def _arm(triples, paired, *, quarters_below=9, reasons=None):
    """A stored arm: drops per period as (arm_reason, b0_reason, n) triples, paired events per period out of 100."""
    if reasons is None:
        reasons = [f"{p}:scored_share_below_floor" for p in PERIODS if paired < 80]
    return {"status": "not_testable" if reasons else "testable", "not_testable_reasons": reasons,
            "share": _shares(paired), "drops": _drops(triples),
            "accuracy": {"periods_below": {p: True for p in PERIODS}, "quarters_below": quarters_below,
                         "quarter_interval": [-0.02, -0.01], "ticker_interval": [-0.02, -0.01], "passed": None},
            "calibration": {"quarters_no_worse": {"p25_p75": 7, "p10_p90": 6}, "passed": None},
            "conditional": {"periods": [{"period": p, "W_spearman": 0.02, "L_spearman": 0.0, "W_tercile_diff": 0.3,
                                         "L_tercile_diff": 0.0} for p in PERIODS],
                            "W_interval": [0.01, 0.03], "diff_interval": [0.01, 0.03], "passed": None}}


def _arms(triples=GAP, paired=70, **kw):
    return {arm: _arm(triples, paired, **kw) for arm in ARMS}


def _report(arms, run_hash=None):
    run_hash = content_hash(RUN) if run_hash is None else run_hash
    return {"schema_version": "size_quantile_stage_a.v4", "inputs": {"run_inputs_hash": run_hash, "spec_hash": ORIG},
            "status": "not_passed",
            "reasons": [f"{arm}:{r}" for arm in ARMS for r in arms[arm]["not_testable_reasons"]],
            "arm_verdicts": {arm: "not_testable" for arm in ARMS},
            "unlabelled_in_period": {"2023": 0, "2024": 0, "2025": 0}, "arms": arms}


def _write(path, body):
    digest = content_hash(body)
    path.write_text(json.dumps({**body, "content_hash": digest}, sort_keys=True), encoding="utf-8")
    return digest


def _case(tmp, *, arms=None, report=None, run=None, run_pin=None, report_pin=None, amendments=True):
    """A synthetic out-dir and spec. Pins default to the hashes of the bodies written, so a case edits one thing."""
    run = RUN if run is None else run
    arms = _arms() if arms is None else arms
    report = _report(arms) if report is None else report
    out = tmp / "out"
    out.mkdir()
    run_digest = _write(out / "run_inputs.json", run)
    report_digest = _write(out / "stage_a_report.json", report)
    entry = {"id": "A1", "label": "POST-HOC (test)", "scored_share_floor": {"min_scored_share": 0.80},
             "applies_to": {"spec_hash": ORIG, "run_inputs_hash": run_pin or run_digest,
                            "report_content_hash": report_pin or report_digest}}
    spec = {"id": "EXP-190", "stage_a": {"min_scored_share": 0.80}, "amendments": [entry] if amendments else []}
    path = tmp / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return out, path


def _refused(out, spec, reason):
    with pytest.raises(OpsError) as exc:
        amendment.run(out, spec_path=spec)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == reason
    assert not list(out.glob("stage_a_report_amendment1_*"))


def test_baseline_gap_only_passes_and_keeps_the_original_share(tmp_path):
    out, spec = _case(tmp_path)
    before = {n: (out / n).read_bytes() for n in ("run_inputs.json", "stage_a_report.json")}
    doc = amendment.run(out, spec_path=spec)
    for arm in ARMS:
        rows = doc["arms"][arm]["share"]
        assert [r["period"] for r in rows] == ["2023H2", "2024", "2025", "pooled"]
        assert [r["share"] for r in rows] == [1.0, 1.0, 1.0, 1.0]
        assert [r["original_share"] for r in rows] == [0.7, 0.7, 0.7, 0.7]
        assert [r["original_below_floor"] for r in rows] == [True, True, True, True]
        assert [r["b0_gap"] for r in rows] == [30, 30, 30, 90]
        assert rows[0]["b0_gap_by_reason"] == {"forecast_missing": 30}
        assert doc["arms"][arm]["verdict"] == "pass"
        assert doc["arms"][arm]["not_testable_reasons"] == []
    assert doc["status"] == "passed"
    assert doc["arm_verdicts"] == {"Q": "pass", "Q_raw": "pass"}
    assert doc["registered_result"]["status"] == "not_passed"
    assert {n: (out / n).read_bytes() for n in before} == before


def test_arm_abstention_stays_not_testable_with_the_three_floor_reasons(tmp_path):
    out, spec = _case(tmp_path, arms=_arms([("calibration_burn_in", "scored", 30)], 70))
    doc = amendment.run(out, spec_path=spec)
    for arm in ARMS:
        res = doc["arms"][arm]
        assert [r["share"] for r in res["share"]] == [0.7, 0.7, 0.7, 0.7]
        assert res["verdict"] == "not_testable"
        assert res["not_testable_reasons"] == FLOORS
        assert res["accuracy_passed"] is None
        assert res["conditional_passed"] is None
        assert res["calibration_passed"] is None
    assert doc["status"] == "not_passed"


def test_boundary_at_0_80_is_not_below_and_just_under_is_below(tmp_path):
    out, spec = _case(tmp_path, arms=_arms(MIXED + [("calibration_burn_in", "scored", 16)], 64))
    doc = amendment.run(out, spec_path=spec)
    for arm in ARMS:
        row = doc["arms"][arm]["share"][0]
        assert (row["b0_gap"], row["with_forecast"], row["share"], row["below_floor"]) == (20, 80, 0.8, False)
    assert doc["status"] == "passed"

    (tmp_path / "under").mkdir()
    out, spec = _case(tmp_path / "under", arms=_arms(MIXED + [("calibration_burn_in", "scored", 17)], 63))
    doc = amendment.run(out, spec_path=spec)
    for arm in ARMS:
        row = doc["arms"][arm]["share"][0]
        assert (row["b0_gap"], row["with_forecast"], row["share"], row["below_floor"]) == (20, 80, 0.7875, True)
        assert doc["arms"][arm]["not_testable_reasons"] == FLOORS
    assert doc["status"] == "not_passed"


def test_a_non_floor_reason_stays_and_keeps_the_arm_not_testable(tmp_path):
    reasons = ["2024:no_usable_cell", *FLOORS]
    out, spec = _case(tmp_path, arms=_arms(GAP, 70, reasons=reasons))
    doc = amendment.run(out, spec_path=spec)
    for arm in ARMS:
        res = doc["arms"][arm]
        assert res["not_testable_reasons"] == ["2024:no_usable_cell"]
        assert res["verdict"] == "not_testable"
        assert res["accuracy_passed"] is None and res["calibration_passed"] is None
    assert doc["status"] == "not_passed"


def test_one_failing_rule_with_no_gap_over_the_floor_is_not_passed(tmp_path):
    out, spec = _case(tmp_path, arms=_arms(GAP, 70, quarters_below=7, reasons=FLOORS))
    doc = amendment.run(out, spec_path=spec)
    for arm in ARMS:
        assert doc["arms"][arm]["share"][3]["share"] == 1.0
        assert doc["arms"][arm]["not_testable_reasons"] == []
        assert doc["arms"][arm]["accuracy_passed"] is False
        assert doc["arms"][arm]["verdict"] == "fail"
    assert doc["status"] == "not_passed"
    assert doc["arm_verdicts"] == {"Q": "fail", "Q_raw": "fail"}


def test_missing_amendment_refuses(tmp_path):
    out, spec = _case(tmp_path, amendments=False)
    _refused(out, spec, "amendment_missing")


def test_missing_amendment_is_checked_before_a_missing_report(tmp_path):
    out, spec = _case(tmp_path, amendments=False)
    (out / "stage_a_report.json").unlink()
    _refused(out, spec, "amendment_missing")


def test_report_edited_without_rehash_is_unbound(tmp_path):
    out, spec = _case(tmp_path)
    path = out / "stage_a_report.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["status"] = "passed"
    path.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    _refused(out, spec, "amended_report_unbound")


def test_report_edited_and_rehashed_is_unbound_by_the_pin(tmp_path):
    edited = {**_report(_arms()), "status": "passed"}
    out, spec = _case(tmp_path, report=edited, report_pin=content_hash(_report(_arms())))
    _refused(out, spec, "amended_report_unbound")


def test_run_inputs_rehashed_differently_is_unbound(tmp_path):
    out, spec = _case(tmp_path, run={**RUN, "years": ["2023", "2024", "2025", "2026"]}, run_pin=content_hash(RUN))
    _refused(out, spec, "amended_run_inputs_unbound")


def test_report_with_a_wrong_run_inputs_binding_is_unbound(tmp_path):
    out, spec = _case(tmp_path, report=_report(_arms(), run_hash="sha256:" + "0" * 64))
    _refused(out, spec, "amended_report_unbound")


def test_missing_report_file_is_unreadable(tmp_path):
    out, spec = _case(tmp_path)
    (out / "stage_a_report.json").unlink()
    _refused(out, spec, "amended_input_unreadable")


def test_label_missing_drop_refuses_as_unlabelled(tmp_path):
    out, spec = _case(tmp_path, arms=_arms(GAP + [("label_missing", "label_missing", 1)], 69))
    _refused(out, spec, "amended_share_unlabelled")


def test_drops_that_do_not_sum_refuse_as_inconsistent(tmp_path):
    out, spec = _case(tmp_path, arms=_arms([("scored", "forecast_missing", 29)], 70))
    _refused(out, spec, "amended_shares_inconsistent")


def test_replay_returns_the_same_document_and_tampering_refuses(tmp_path):
    out, spec = _case(tmp_path)
    first = amendment.run(out, spec_path=spec)
    assert amendment.run(out, spec_path=spec) == first
    (path,) = out.glob("stage_a_report_amendment1_*.json")
    before = path.read_bytes()
    path.write_bytes(before + b" ")
    with pytest.raises(OpsError) as exc:
        amendment.run(out, spec_path=spec)
    assert exc.value.problem.details["reason"] == "amended_report_changed"
    assert path.read_bytes() == before + b" "


def test_main_prints_one_json_object_and_returns_zero(tmp_path, monkeypatch, capsys):
    out, spec = _case(tmp_path)
    monkeypatch.setattr(amendment, "SPEC_PATH", spec)
    assert amendment.main(["--out-dir", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"status", "reasons", "arm_verdicts", "content_hash", "shares"}
    assert printed["status"] == "passed"
    assert [r["share"] for r in printed["shares"]["Q"]] == [1.0, 1.0, 1.0, 1.0]


def test_main_refusal_returns_one_with_the_reason_on_stderr(tmp_path, monkeypatch, capsys):
    out, spec = _case(tmp_path, amendments=False)
    monkeypatch.setattr(amendment, "SPEC_PATH", spec)
    assert amendment.main(["--out-dir", str(out)]) == 1
    assert "refused: EXPERIMENT_IDENTITY_CONFLICT: amendment_missing" in capsys.readouterr().err


def test_committed_spec_has_exactly_one_a1_amendment_pinned_to_the_bound_run():
    spec, _ = prereg.load_spec()
    assert len(spec["amendments"]) == 1
    entry = spec["amendments"][0]
    assert entry["id"] == "A1"
    assert entry["scored_share_floor"]["min_scored_share"] == 0.80
    assert spec["stage_a"]["min_scored_share"] == 0.80
    assert entry["applies_to"]["spec_hash"] == "5347a4f5c2599793a811c9adef8fc9cb8c836636473c56cd359cd6c27b18fd9f"
