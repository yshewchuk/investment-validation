"""Registration/recording boundaries over committed synthetic daily chains."""
import json
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest
from threadpoolctl import threadpool_info

from engine.v2.contracts import ArtifactRef, FinalHoldoutDeclaration
from engine.v2.data.catalog import move_head
from engine.v2.foundation import (
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments import registered_runup as study
from experiments import runup_exit_report as comparison
from experiments import v2_runup_study as cli
from tests.data_scan_support import catalog_and_store
from tests.v2.research.magnitude_trial_support import TrackingRepository
from tests.v2.research.test_runup_exit_report import DATES, EVENT_ID, chain_rows, commit, event_rows


def specification(**changes):
    return experiment_spec_from_document({
        "experiment_id": "synthetic-runup-study", "hypothesis": "Synthetic exit comparison",
        "primary_arm_id": "candidate", "arms": ["candidate"], "seed": 0, "folds": [],
        "price_source": "option_chains", "runner": study.RUNNER,
        "economic_params": {"fill": 0.5, "exit": {"kind": "target_stop", "trading_days": 14,
            "threshold_units": "entry_debit_fraction", "target_pnl": 0.25, "stop_pnl": -0.5}},
        **changes})


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(outcomes, "datetime", SimpleNamespace(
        now=lambda tz: datetime(2025, 1, 15, tzinfo=tz)))
    conn, clock, store = catalog_and_store(tmp_path)
    repository = TrackingRepository(conn, store)
    snapshot = commit((conn, clock, store))
    yield conn, clock, store, repository, snapshot
    conn.close()


def invoke(source, action, spec=None, **changes):
    conn, _, store, repository, snapshot = source
    return action(conn, store, repository, spec or specification(),
                  **{ "snapshot": snapshot, "scope": "shadow", "as_of_month": "2025-01",
                      "event_ids": [EVENT_ID], **changes})


def evidence(source):
    return [json.loads(row[0]) for row in source[0].execute(
        "SELECT evidence_json FROM experiment_runs")]


def forbid(*args, **kwargs):
    raise AssertionError("forbidden access")


def assert_refusal(error, code, message, *, category="validation", retryable=False):
    assert error.code == error.problem.code == code
    assert error.problem.message == message
    assert error.problem.category == category
    assert error.problem.retryable is retryable


def direct_comparison(source):
    return comparison.compare_runup_exits(source[3], source[4], as_of_month="2025-01",
        document={"event_ids": [EVENT_ID], "fill_alpha": 0.5,
                  "threshold_units": "entry_debit_fraction",
                  "arms": [{"arm_id": "candidate", "target_pnl": 0.25, "stop_pnl": -0.5}]})


def expected_success(registration, report):
    report_ref = to_document(artifact_reference(
        canonical_json(report).encode(), "native_experiment_report.v1.0"))
    outcome = {"schema_version": "native_experiment_outcome.v1.0",
        "run_id": registration.run_id, "variant_id": registration.variant_id,
        "attempted_variants": 1, "failure_code": None, "failure_details": {},
        "report_ref": report_ref, "ledger_destination": None,
        "ledger_row": {"id": "synthetic-runup-study", "spec_hash": registration.variant_id,
                       "date": "2025-01-15", "stage": "ran", "oos_mean_mid": "",
                       "sharpe_trade": "", "promoted": "False"}}
    receipt = {"schema_version": "native_experiment_completion.v1.0",
        "outcome_ref": to_document(artifact_reference(
            canonical_json(outcome).encode(), "native_experiment_outcome.v1.0")),
        "recording_mode": "smoke", "report_ref": report_ref,
        "ledger_destination": None, "ledger_row_hash": None}
    return {"variant_id": registration.variant_id, "run_id": registration.run_id,
        "replayed": False, "outcome": outcome, "receipt": receipt,
        "receipt_ref": to_document(artifact_reference(
            canonical_json(receipt).encode(), "native_experiment_completion.v1.0"))}


def assert_saved_artifacts(source, result, report):
    for document, ref in ((report, result["outcome"]["report_ref"]),
            (result["outcome"], result["receipt"]["outcome_ref"]),
            (result["receipt"], result["receipt_ref"])):
        assert source[2].read_verified(from_document(ArtifactRef, ref)) == canonical_json(document).encode()
    saved = evidence(source)[0]
    assert saved["native_smoke"] == result["receipt"]["outcome_ref"]
    assert saved["native_smoke_receipt"] == result["receipt_ref"]


def test_register_is_metadata_only_idempotent_and_not_attempted(source, monkeypatch):
    monkeypatch.setattr(study, "compare_runup_exits", forbid)
    monkeypatch.setattr(outcomes, "replay_native_outcome", forbid)
    first = invoke(source, study.register_runup)
    assert invoke(source, study.register_runup) == first
    assert {name for name, _ in source[3].scans} == {"earnings_events"}
    assert set(evidence(source)[0]) == {"native_registration", "variant_id", "planned_variants"}
    assert evidence(source)[0]["planned_variants"] == 1
    assert first.document["snapshot"]["snapshot_id"] == source[4].snapshot_id


POLICY_REFUSAL = "RUNUP requires its explicit single target/stop policy"


@pytest.mark.parametrize("action", [study.register_runup, study.run_runup])
@pytest.mark.parametrize("changes,message", [
    ({"runner": "experiments/v2_gate.py"}, POLICY_REFUSAL),
    ({"seed": 1}, POLICY_REFUSAL), ({"folds": ["2024"]}, POLICY_REFUSAL),
    ({"price_source": "computed_moves"},
     "invalid fixed-day or target/stop exit economics or mark source"),
    ({"input_files": ["unused"]}, POLICY_REFUSAL),
    ({"arms": ["candidate", "extra"]},
     "experiment must declare exactly one arm and name it as its primary arm"),
    ({"primary_arm_id": "fixed_day", "arms": ["fixed_day"]}, "invalid RUNUP comparison request"),
    ({"economic_params": {"fill": 0.5}}, POLICY_REFUSAL),
    ({"economic_params": {"fill": 0.5, "exit": {"kind": "fixed_day", "trading_days": 14}}},
     POLICY_REFUSAL),
    ({"economic_params": {"fill": 0.5, "exit": {"kind": "target_stop", "trading_days": 14,
        "target_pnl": 2, "stop_pnl": -2}}}, POLICY_REFUSAL),
    ({"economic_params": {"fill": 0.5, "exit": {"kind": "target_stop", "trading_days": 13,
        "target_pnl": 2, "stop_pnl": -2, "threshold_units": "quote_price_times_quantity"}}},
     POLICY_REFUSAL),
])
def test_invalid_contract_refuses_before_any_access(source, monkeypatch, action, changes, message):
    monkeypatch.setattr(study, "_prepare", forbid)
    monkeypatch.setattr(native, "read_native_registration", forbid)
    monkeypatch.setattr(source[3], "resolve", forbid)
    monkeypatch.setattr(source[3], "scan", forbid)
    monkeypatch.setattr(source[2], "read_verified", forbid)
    monkeypatch.setattr(source[2], "publish_bytes", forbid)
    with pytest.raises(OpsError) as caught:
        invoke(source, action, specification(**changes))
    assert_refusal(caught.value, "INVALID_EXPERIMENT_SPEC", message)
    assert evidence(source) == []
    assert source[3].scans == []


@pytest.mark.parametrize("action", [study.register_runup, study.run_runup])
@pytest.mark.parametrize("ids", [None, [], [EVENT_ID, EVENT_ID], [""], (EVENT_ID,)])
def test_explicit_unique_cohort_required_before_access(source, monkeypatch, action, ids):
    monkeypatch.setattr(study, "_prepare", forbid)
    monkeypatch.setattr(native, "read_native_registration", forbid)
    monkeypatch.setattr(source[3], "resolve", forbid)
    monkeypatch.setattr(source[3], "scan", forbid)
    with pytest.raises(OpsError) as caught:
        invoke(source, action, event_ids=ids)
    assert_refusal(caught.value, "INVALID_EXPERIMENT_SPEC", "invalid RUNUP comparison request")
    assert evidence(source) == []
    assert source[3].scans == []


def test_success_preserves_json_report_and_terminal_replay_reads_no_market(source, monkeypatch):
    report_expected = direct_comparison(source)
    registration = invoke(source, study.register_runup)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    result = invoke(source, study.run_runup)
    assert result == expected_success(registration, report_expected)
    assert_saved_artifacts(source, result, report_expected)
    assert result["variant_id"] == registration.variant_id
    assert result["replayed"] is False
    assert result["outcome"]["attempted_variants"] == 1
    assert result["outcome"]["failure_code"] is None
    assert result["receipt"]["recording_mode"] == "smoke"
    assert result["receipt"]["ledger_destination"] is None
    report = json.loads(source[2].read_verified(from_document(ArtifactRef, result["receipt"]["report_ref"])))
    assert report == json.loads(canonical_json(report_expected))
    assert report["comparison_id"] == content_hash(report["binding"])
    assert report["report_hash"] == content_hash({k: v for k, v in report.items() if k != "report_hash"})
    assert report["mode"] == "exploratory_no_ledger"
    assert report["binding"]["purpose"] == "selection"
    assert report["counts"] == dict(requested_count=1, admitted_count=1,
                                     selected_count=1, compared_count=1, arm_count=2)
    assert {(row["trade_id"], row["arm_id"]) for row in report["rows"]} == {
        (EVENT_ID, "fixed_day"), (EVENT_ID, "candidate")}
    assert next(row for row in report["rows"] if row["arm_id"] == "candidate")["pnl"] == 2.0
    assert not list(source[2].root.glob("native_reports/**/REPORT.md"))
    before = evidence(source)
    source[3].scans.clear()
    monkeypatch.setattr(study, "compare_runup_exits", forbid)
    replay = invoke(source, study.run_runup)
    assert {**replay, "replayed": False} == result
    assert {name for name, _ in source[3].scans} == {"earnings_events"}
    assert evidence(source) == before


def test_single_thread_context_is_active_during_real_comparison(source, monkeypatch):
    invoke(source, study.register_runup)
    real_limits, compare = study.threadpool_limits, study.compare_runup_exits
    transitions = []

    @contextmanager
    def observe_limits(*, limits):
        assert limits == 1
        with real_limits(limits=limits):
            transitions.append("entered")
            yield
        transitions.append("exited")

    def observe_comparison(*args, **kwargs):
        assert transitions == ["entered"]
        pools = threadpool_info()
        assert pools and all(pool["num_threads"] == 1 for pool in pools)
        report = compare(*args, **kwargs)
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        transitions.append("compared")
        return report

    monkeypatch.setattr(study, "threadpool_limits", observe_limits)
    monkeypatch.setattr(study, "compare_runup_exits", observe_comparison)
    result = invoke(source, study.run_runup)
    assert result["outcome"]["failure_code"] is None
    assert transitions == ["entered", "compared", "exited"]


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("changes", [
    {"event_ids": ["unknown-event"]}, {"as_of_month": "2024-01"}, {"scope": "other"},
])
def test_wrong_request_cannot_poison_or_read_terminal_slot(source, monkeypatch, changes, terminal):
    invoke(source, study.register_runup)
    result = invoke(source, study.run_runup) if terminal else None
    before = evidence(source)
    source[3].scans.clear()
    with monkeypatch.context() as patch:
        patch.setattr(outcomes, "replay_native_outcome", forbid)
        patch.setattr(outcomes, "publish_native_outcome", forbid)
        with pytest.raises(OpsError) as caught:
            invoke(source, study.run_runup, **changes)
        assert_refusal(caught.value, "EXPERIMENT_IDENTITY_CONFLICT",
            "native execution inputs changed before population admission" if "scope" in changes else
            "RUNUP cohort or holdout month differs from registration")
    assert evidence(source) == before
    assert all(name == "earnings_events" for name, _ in source[3].scans)
    if terminal:
        monkeypatch.setattr(study, "compare_runup_exits", forbid)
        assert {**invoke(source, study.run_runup), "replayed": False} == result


def test_missing_registration_never_attempts(source, monkeypatch):
    monkeypatch.setattr(study, "compare_runup_exits", forbid)
    with pytest.raises(OpsError) as caught:
        invoke(source, study.run_runup)
    assert_refusal(caught.value, "INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
    assert evidence(source) == []
    assert source[3].scans == []


def test_implicit_head_is_never_admitted(source, monkeypatch):
    with pytest.raises(OpsError) as caught:
        invoke(source, study.register_runup, snapshot=None)
    assert_refusal(caught.value, "INVALID_EXPERIMENT_SPEC", "RUNUP requires an explicit snapshot reference")
    assert evidence(source) == []
    invoke(source, study.register_runup)
    monkeypatch.setattr(outcomes, "replay_native_outcome", forbid)
    with pytest.raises(OpsError) as caught:
        invoke(source, study.run_runup, snapshot=None)
    assert_refusal(caught.value, "INVALID_EXPERIMENT_SPEC", "RUNUP requires an explicit snapshot reference")


def test_final_read_declaration_cannot_enter_selection_slot(source, monkeypatch):
    declaration = FinalHoldoutDeclaration(
        evaluation_recipe_ref=artifact_reference(b"{}", "final_holdout_recipe.v1.0"),
        completeness_bar_ref=artifact_reference(b"{}", "final_holdout_completeness.v1.0"),
        frozen_model_refs=())
    registration, spec = native.prepare_native_registration(source[3], specification(),
        code_root=study.ROOT, snapshot=source[4], scope="shadow", as_of_month="2025-01",
        event_ids=[EVENT_ID], final_read=declaration)
    native.reserve_native_registration(source[0], source[2], registration, spec)
    before = evidence(source)
    source[3].scans.clear()
    monkeypatch.setattr(outcomes, "replay_native_outcome", forbid)
    with pytest.raises(OpsError) as caught:
        invoke(source, study.run_runup)
    assert_refusal(caught.value, "HOLDOUT_ACCESS_DENIED", "RUNUP registration grants selection authority only")
    assert evidence(source) == before
    assert source[3].scans == []


@pytest.mark.parametrize("case", ["budget", "tampered", "spec"])
def test_registration_integrity_and_budget_refuse_before_population_or_slot(source, monkeypatch, case):
    invoke(source, study.register_runup)
    original_spec = specification()
    options = {}
    if case == "budget":
        options["max_evidence_bytes"] = 1
    elif case == "tampered":
        value = evidence(source)[0]
        value["variant_id"] = "sha256:" + "0" * 64
        source[0].execute("UPDATE experiment_runs SET evidence_json=?", (canonical_json(value),))
    else:
        original_spec = specification(hypothesis="Changed synthetic hypothesis")
    before = evidence(source)
    source[3].scans.clear()
    monkeypatch.setattr(outcomes, "replay_native_outcome", forbid)
    with pytest.raises(OpsError) as caught:
        invoke(source, study.run_runup, original_spec, **options)
    assert_refusal(caught.value, "EXPERIMENT_IDENTITY_CONFLICT", {
        "budget": "native evidence requires bounded JSON object text",
        "tampered": "native registration differs from its immutable binding",
        "spec": "requested specification differs from registration"}[case])
    assert evidence(source) == before
    assert source[3].scans == []


def test_explicit_old_pin_survives_scope_head_change_and_wrong_pin_refuses(source, monkeypatch):
    invoke(source, study.register_runup)
    result = invoke(source, study.run_runup)
    rows = [{**row, "bid": row["bid"] + 0.25} for row in chain_rows()]
    newer = commit(source[:3], rows, scope="new-head")
    generation = source[0].execute("SELECT generation FROM data_snapshot_heads WHERE scope='shadow'").fetchone()[0]
    move_head(source[0], scope="shadow", to_snapshot_id=newer.snapshot_id,
              expected_snapshot_id=source[4].snapshot_id, expected_generation=generation,
              receipt_ref="synthetic-head-move", clock=source[1])
    assert newer != source[4]
    before = evidence(source)
    with monkeypatch.context() as patch:
        patch.setattr(outcomes, "replay_native_outcome", forbid)
        with pytest.raises(OpsError) as caught:
            invoke(source, study.run_runup, snapshot=newer)
        assert_refusal(caught.value, "EXPERIMENT_IDENTITY_CONFLICT",
                       "native execution inputs changed before population admission")
    assert evidence(source) == before
    monkeypatch.setattr(study, "compare_runup_exits", forbid)
    assert {**invoke(source, study.run_runup), "replayed": False} == result


def test_source_drift_refuses_before_slot_without_overwriting_registration(source, monkeypatch):
    invoke(source, study.register_runup)
    before = evidence(source)
    original = native.source_closure
    monkeypatch.setattr(native, "source_closure", lambda *a, **k: {**original(*a, **k), "drift.py": "changed"})
    monkeypatch.setattr(outcomes, "replay_native_outcome", forbid)
    with pytest.raises(OpsError) as caught:
        invoke(source, study.run_runup)
    assert_refusal(caught.value, "EXPERIMENT_IDENTITY_CONFLICT",
                   "native execution inputs changed before population admission")
    assert evidence(source) == before


def test_environment_drift_cannot_reuse_existing_terminal(source, monkeypatch):
    invoke(source, study.register_runup)
    result = invoke(source, study.run_runup)
    before = evidence(source)
    with monkeypatch.context() as patch:
        original = native.environment_identity
        patch.setattr(native, "environment_identity", lambda *a: {**original(*a), "drift": True})
        patch.setattr(outcomes, "replay_native_outcome", forbid)
        with pytest.raises(OpsError) as caught:
            invoke(source, study.run_runup)
        assert_refusal(caught.value, "EXPERIMENT_IDENTITY_CONFLICT",
                       "native execution inputs changed before population admission")
    assert evidence(source) == before
    monkeypatch.setattr(study, "compare_runup_exits", forbid)
    assert {**invoke(source, study.run_runup), "replayed": False} == result


def test_late_missing_leg_records_whole_candidate_failure_and_replays(source, monkeypatch):
    rows = [row for row in chain_rows() if not (row["obs_date"] == DATES[8]
            and row["strike"] == 100 and row["right"] == "P")]
    snapshot = commit(source[:3], rows, scope="incomplete")
    invoke(source, study.register_runup, snapshot=snapshot)
    result = invoke(source, study.run_runup, snapshot=snapshot)
    outcome = result["outcome"]
    assert outcome["attempted_variants"] == 1
    assert outcome["failure_code"] == "EXPERIMENT_VARIANT_FAILED"
    assert outcome["failure_details"]["stage"] == "common_horizon_coverage"
    assert outcome["failure_details"]["compared_count"] == 0
    assert outcome["report_ref"] is None
    monkeypatch.setattr(study, "compare_runup_exits", forbid)
    source[3].scans.clear()
    replay = invoke(source, study.run_runup, snapshot=snapshot)
    assert {**replay, "replayed": False} == result
    assert all(name == "earnings_events" for name, _ in source[3].scans)


def test_resource_failure_leaves_outcome_slot_free(source, monkeypatch):
    invoke(source, study.register_runup)
    before = evidence(source)
    def interrupted(*args, **kwargs):
        raise fail("RESOURCE_UNAVAILABLE", "synthetic interruption")
    monkeypatch.setattr(study, "compare_runup_exits", interrupted)
    with pytest.raises(OpsError) as caught:
        invoke(source, study.run_runup)
    assert_refusal(caught.value, "RESOURCE_UNAVAILABLE", "synthetic interruption",
                   category="resource", retryable=True)
    assert evidence(source) == before


def test_interrupted_completion_reconciles_intent_without_rewalking(source, monkeypatch):
    report = direct_comparison(source)
    registration = invoke(source, study.register_runup)
    expected = expected_success(registration, report)
    publish = source[2].publish_bytes
    def interrupted(payload, *, schema_ref, **kwargs):
        if schema_ref == "native_experiment_completion.v1.0":
            raise OSError("synthetic interruption")
        return publish(payload, schema_ref=schema_ref, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(source[2], "publish_bytes", interrupted)
        with pytest.raises(OpsError) as caught:
            invoke(source, study.run_runup)
        assert_refusal(caught.value, "RESOURCE_UNAVAILABLE",
                       "native publication was interrupted; replay its intent",
                       category="resource", retryable=True)
    assert evidence(source)[0]["native_smoke"] == expected["receipt"]["outcome_ref"]
    assert "native_smoke_receipt" not in evidence(source)[0]
    monkeypatch.setattr(study, "compare_runup_exits", forbid)
    result = invoke(source, study.run_runup)
    assert result == {**expected, "replayed": True}
    assert_saved_artifacts(source, result, report)


def test_multiple_events_have_exact_complete_pairs(source):
    second = next(f"SECOND_{n}" for n in range(100)
                  if not ExperimentHoldouts.random_membership(f"SECOND_{n}"))
    rows = chain_rows() + [{**row, "ticker": "SECOND", "bid": row["bid"] * 2,
                          "ask": row["ask"] * 2} for row in chain_rows()]
    events = event_rows() + [{**event_rows()[0], "event_id": second, "ticker": "SECOND"}]
    snapshot = commit(source[:3], rows, events=events, scope="two-events")
    binding = dict(snapshot=snapshot, event_ids=[second, EVENT_ID])
    invoke(source, study.register_runup, **binding)
    result = invoke(source, study.run_runup, **binding)
    report = json.loads(source[2].read_verified(from_document(ArtifactRef, result["receipt"]["report_ref"])))
    assert report["counts"] == dict(requested_count=2, admitted_count=2,
                                     selected_count=2, compared_count=2, arm_count=2)
    assert len(report["rows"]) == 4
    assert {(row["trade_id"], row["arm_id"]) for row in report["rows"]} == {
        (event, arm) for event in [EVENT_ID, second] for arm in ["candidate", "fixed_day"]}
    assert {row["entry_cost"] for row in report["rows"]} == {6.0, 12.0}


def test_cli_register_run_and_replay_emit_one_document(source, tmp_path, capsys):
    spec = specification()
    path = tmp_path / "spec.json"
    path.write_text(canonical_json({**vars(spec), "arms": list(spec.arms), "folds": []}))
    catalog = source[0].execute("PRAGMA database_list").fetchone()[2]
    args = ["--spec", str(path), "--catalog", catalog, "--store-root", str(source[2].root),
            "--snapshot-id", source[4].snapshot_id, "--scope", "shadow", "--as-of-month", "2025-01",
            "--event-id", EVENT_ID]
    expected_registration, _ = native.prepare_native_registration(source[3], spec,
        code_root=study.ROOT, snapshot=source[4], scope="shadow", as_of_month="2025-01",
        event_ids=[EVENT_ID])
    report_expected = direct_comparison(source)
    capsys.readouterr()  # The independent comparison's diagnostics are not CLI output.
    assert expected_registration.run_id == "native_" + content_hash({
        "experiment_id": "synthetic-runup-study", "arm": "candidate"}).removeprefix("sha256:")
    assert cli.main(["register", *args]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "registered": expected_registration.variant_id, "run_id": expected_registration.run_id}
    assert cli.main(["run", *args]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first == expected_success(expected_registration, report_expected)
    assert cli.main(["run", *args]) == 0
    assert {**json.loads(capsys.readouterr().out), "replayed": False} == first


@pytest.mark.parametrize("text", ['{"experiment_id":1,"experiment_id":2}', '[]', '{', '{}',
    canonical_json({**vars(specification()), "input_files": None}),
    canonical_json({**vars(specification()), "input_files": 1})])
def test_cli_bad_json_refuses_before_catalog(tmp_path, capsys, text):
    path = tmp_path / "spec.json"
    path.write_text(text)
    args = ["register", "--spec", str(path), "--catalog", str(tmp_path / "missing"),
            "--store-root", str(tmp_path / "store"), "--snapshot-id", "missing", "--scope", "shadow",
            "--as-of-month", "2025-01", "--event-id", EVENT_ID]
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert not (tmp_path / "missing").exists()
    assert not (tmp_path / "store").exists()
