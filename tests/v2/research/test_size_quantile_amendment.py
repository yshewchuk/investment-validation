"""EXP-190 Amendment 1 re-read (experiments.size_quantile_amendment) on synthetic out-dirs built by hand. Every expected
share and verdict here is a literal written in this file; none is computed by the code under test."""
import copy
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
    name = f"stage_a_report_amendment1_{prereg.load_spec(spec)[1][:12]}.json"
    assert doc["post_hoc"] is True
    assert doc["label"] == "POST-HOC (test)"
    assert doc["preregistration_checked"] is False
    assert doc["min_scored_share"] == 0.8
    assert doc["inputs"] == {
        "report_content_hash": json.loads(before["stage_a_report.json"])["content_hash"],
        "run_inputs_hash": json.loads(before["run_inputs.json"])["content_hash"],
        "original_spec_hash": ORIG,
        "amendment_spec_hash": prereg.load_spec(spec)[1],
    }
    assert doc["registered_result"]["arm_verdicts"] == {"Q": "not_testable", "Q_raw": "not_testable"}
    assert sorted(p.name for p in out.iterdir()) == sorted(["run_inputs.json", "stage_a_report.json", name])


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
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert exc.value.problem.details["reason"] == "amended_report_changed"
    assert path.read_bytes() == before + b" "


def test_main_prints_one_json_object_and_returns_zero(tmp_path, monkeypatch, capsys):
    out, spec = _case(tmp_path)
    monkeypatch.setattr(amendment, "SPEC_PATH", spec)
    assert amendment.main(["--out-dir", str(out), "--no-ledger"]) == 0
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
    assert entry["applies_to"]["run_inputs_hash"] == "sha256:6c5850f44cb23e6f6723ade37e7185a2f591bc17a15a73d93115c41d5ee64ec4"
    assert entry["applies_to"]["report_content_hash"] == "sha256:30dabc2ad6bbb68c7193319073808600821ce75b089e93a7699dade1d123734d"


def _period(arm, p):
    return next(r for r in arm["conditional"]["periods"] if r["period"] == p)


RULE_FAILS = [
    ("calibration_passed", lambda a: a["calibration"]["quarters_no_worse"].update(p10_p90=4)),
    ("conditional_passed", lambda a: _period(a, "2024").update(W_spearman=-0.01)),
    ("conditional_passed", lambda a: _period(a, "2024").update(L_tercile_diff=0.5)),
    ("conditional_passed", lambda a: a["conditional"].update(W_interval=[-0.01, 0.03])),
    ("conditional_passed", lambda a: a["conditional"].update(diff_interval=[0.0, 0.03])),
    ("accuracy_passed", lambda a: a["accuracy"]["periods_below"].update({"2025": False})),
    ("accuracy_passed", lambda a: a["accuracy"].update(quarter_interval=[-0.02, 0.0])),
    ("accuracy_passed", lambda a: a["accuracy"].update(ticker_interval=[-0.02, 0.01])),
]


@pytest.mark.parametrize("flag, mutate", RULE_FAILS)
def test_each_rule_fails_on_its_own_on_the_baseline_gap(tmp_path, flag, mutate):
    arms = _arms(GAP, 70, reasons=FLOORS)
    for arm in ARMS:
        mutate(arms[arm])
    out, spec = _case(tmp_path, arms=arms)
    doc = amendment.run(out, spec_path=spec)
    others = {"accuracy_passed", "conditional_passed", "calibration_passed"} - {flag}
    for arm in ARMS:
        res = doc["arms"][arm]
        assert res[flag] is False
        assert all(res[k] is True for k in others)
        assert res["verdict"] == "fail"
    assert doc["arm_verdicts"] == {"Q": "fail", "Q_raw": "fail"}
    assert doc["status"] == "not_passed"


def test_calibration_at_five_quarters_still_passes(tmp_path):
    arms = _arms(GAP, 70, reasons=FLOORS)
    for arm in ARMS:
        arms[arm]["calibration"]["quarters_no_worse"]["p10_p90"] = 5
    out, spec = _case(tmp_path, arms=arms)
    doc = amendment.run(out, spec_path=spec)
    for arm in ARMS:
        assert doc["arms"][arm]["calibration_passed"] is True
        assert doc["arms"][arm]["verdict"] == "pass"
    assert doc["status"] == "passed"


def test_an_abstaining_drop_in_one_period_fails_only_that_arm(tmp_path):
    q_raw = _arm(GAP, 70, reasons=FLOORS)
    q_raw["drops"] = [d for d in q_raw["drops"] if d["year"] != 2024] + [
        {"year": 2024, "arm_reason": "calibration_burn_in", "b0_reason": "scored", "n": 30}]
    out, spec = _case(tmp_path, arms={"Q": _arm(GAP, 70, reasons=FLOORS), "Q_raw": q_raw})
    doc = amendment.run(out, spec_path=spec)
    assert doc["arms"]["Q"]["verdict"] == "pass"
    res = doc["arms"]["Q_raw"]
    assert res["verdict"] == "not_testable"
    assert res["not_testable_reasons"] == ["2024:scored_share_below_floor"]
    assert [r["share"] for r in res["share"]] == [1.0, 0.7, 1.0, 0.875]
    assert doc["status"] == "passed"
    assert doc["arm_verdicts"] == {"Q": "pass", "Q_raw": "not_testable"}


def test_two_b0_reasons_in_one_period_are_both_reported(tmp_path):
    arms = {}
    for arm in ARMS:
        a = _arm(GAP, 70, reasons=FLOORS)
        a["drops"] = [d for d in a["drops"] if d["year"] != 2024] + [
            {"year": 2024, "arm_reason": "scored", "b0_reason": "forecast_missing", "n": 20},
            {"year": 2024, "arm_reason": "scored", "b0_reason": "refused_draw", "n": 10}]
        arms[arm] = a
    out, spec = _case(tmp_path, arms=arms)
    doc = amendment.run(out, spec_path=spec)
    for arm in ARMS:
        rows = doc["arms"][arm]["share"]
        assert rows[1]["b0_gap_by_reason"] == {"forecast_missing": 20, "refused_draw": 10}
        assert rows[1]["b0_gap"] == 30 and rows[1]["share"] == 1.0
        assert rows[0]["b0_gap_by_reason"] == {"forecast_missing": 30}


def test_unlabelled_count_alone_refuses(tmp_path):
    report = _report(_arms())
    report["unlabelled_in_period"]["2024"] = 1
    out, spec = _case(tmp_path, report=report)
    _refused(out, spec, "amended_share_unlabelled")


def test_report_arm_without_drops_is_malformed(tmp_path):
    arms = _arms()
    del arms["Q"]["drops"]
    out, spec = _case(tmp_path, arms=arms)
    _refused(out, spec, "amended_report_malformed")


def test_smoke_run_inputs_are_unbound(tmp_path):
    out, spec = _case(tmp_path, run={**RUN, "mode": "smoke"})
    _refused(out, spec, "amended_run_inputs_unbound")


def test_run_inputs_of_another_spec_are_unbound(tmp_path):
    run = {**RUN, "spec_hash": "b" * 64}
    out, spec = _case(tmp_path, run=run, report=_report(_arms(), run_hash=content_hash(run)))
    _refused(out, spec, "amended_run_inputs_unbound")


def test_report_of_another_spec_is_unbound(tmp_path):
    report = {**_report(_arms()), "inputs": {"run_inputs_hash": content_hash(RUN), "spec_hash": "b" * 64}}
    out, spec = _case(tmp_path, report=report)
    _refused(out, spec, "amended_report_unbound")


def _edit_spec(spec, edit):
    doc = yaml.safe_load(spec.read_text(encoding="utf-8"))
    edit(doc)
    spec.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize("edit", [
    lambda d: d.update(amendments=d["amendments"] + copy.deepcopy(d["amendments"])),
    lambda d: d["amendments"][0]["scored_share_floor"].update(min_scored_share=0.75),
    lambda d: d["stage_a"].update(min_scored_share=0.75),
], ids=["two_a1_entries", "entry_floor_0_75", "stage_a_floor_0_75"])
def test_spec_admission_refuses_unless_one_a1_entry_at_the_floor(tmp_path, edit):
    out, spec = _case(tmp_path)
    _edit_spec(spec, edit)
    _refused(out, spec, "amendment_missing")


def test_memory_error_is_a_resource_limit_and_writes_nothing(tmp_path, monkeypatch):
    out, spec = _case(tmp_path)

    def boom(*args, **kwargs):
        raise MemoryError

    monkeypatch.setattr(amendment, "load_spec", boom)
    with pytest.raises(OpsError) as exc:
        amendment.run(out, spec_path=spec)
    assert exc.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not list(out.glob("stage_a_report_amendment1_*"))
