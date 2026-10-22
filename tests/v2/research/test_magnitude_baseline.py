"""Synthetic fold-local magnitude arithmetic and the real native CLI lifecycle."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_report as magnitude
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments import v2_prediction as cli
from tests.v2.research.magnitude_trial_support import (
    PlanRepository,
    calendar_sessions,
    catalog_rows,
    commit_history_inputs,
    forbid_access,
    history_event,
    outcome_move,
    poison_outcomes,
    prediction_spec,
)
from tests.v2.research.test_native_outcomes import (
    read_ledger_rows,
    read_outcome_document,
    set_outcome_clock,
)
from tests.v2.research.test_native_registration import artifact_object_bytes

MONTH, START, CUTOFF = "2026-01", "2023-12-01", "2025-02-01"
ROOT = Path(__file__).resolve().parents[3]


def _spec(**changes):
    return prediction_spec(**{"experiment_id": "synthetic-magnitude", "primary_arm_id": "mean",
        "arms": ("mean",), "folds": ("2024", "2025"), **changes})


def _dataset():
    dates = ("2023-12-04", "2023-12-05", "2023-12-06", "2023-12-07",
             "2023-12-08", "2023-12-11", "2023-12-12", "2024-01-03",
             "2024-06-12", "2024-09-04", "2025-01-07")
    events = [history_event(day) for day in dates]
    events[7]["session"] = "BMO"
    values = [-2., 0., 4., 12., None, None, None, -1., 7., -10., 4.]
    moves = [outcome_move(event, value) for event, value in zip(events, values)]
    moves[3]["available_as_of_date"] = "2023-12-22"  # Exactly the first fit boundary.
    moves[4].update(skipped=True, available_as_of_date=None)
    moves[5]["available_as_of_date"] = None
    moves[6]["available_as_of_date"] = CUTOFF  # Unparseable but ineligible for both fits.
    excluded = [history_event("2024-06-13", protected=True), history_event("2025-12-03"),
                history_event("2023-10-02")]
    return events + excluded, moves + [outcome_move(event, None) for event in excluded]


def _source(tmp_path, *, register=True, spec=None, events=None, moves=None, mixed_days=()):
    if events is None:
        events, moves = _dataset()
    conn, original, snapshot = commit_history_inputs(tmp_path, events, moves, mixed_days=mixed_days,
        days=calendar_sessions("2023-11-01", "2025-12-05").days)
    source = SimpleNamespace(conn=conn, store=original._store, snapshot=snapshot,
        repository=PlanRepository(conn, original._store), spec=spec or _spec(),
        events=events, root=tmp_path, spec_path=tmp_path / "spec.json")
    source.spec_path.write_text(json.dumps(asdict(source.spec)))
    if register:
        source.registration = native.register_native(conn, source.store, source.repository, source.spec,
            code_root=ROOT, as_of_month=MONTH, magnitude_training_start=START,
            magnitude_labels_before=CUTOFF)
        source.repository.scans.clear()
    return source


@pytest.fixture
def source(tmp_path):
    value = _source(tmp_path)
    yield value
    value.conn.close()


def baseline_evidence(source):
    row = source.conn.execute("SELECT evidence_json FROM experiment_runs").fetchone()
    return json.loads(row[0]) if row else {}


def _args(source, action, *extra):
    argv = [action, "--spec", str(source.spec_path), "--catalog", str(source.root / "catalog.sqlite"),
        "--store-root", str(source.store.root), "--as-of-month", MONTH,
        "--target", "absolute_move", "--magnitude-training-start", START,
        "--magnitude-labels-before", CUTOFF, *map(str, extra)]
    if action == "run":
        argv += ["--exp-number", "101", "--experiments-root", str(source.root / "experiments")]
    return argv


def _invoke(source, capsys, action, *extra, status=0):
    assert cli.main(_args(source, action, *extra)) == status
    captured = capsys.readouterr()
    assert not captured.err and "Traceback" not in captured.out
    return json.loads(captured.out)


def result_from_report(path):
    return json.loads(Path(path).read_text().split("```json\n", 1)[1].split("\n```", 1)[0])


def _metadata_only(monkeypatch):
    original = Repository.scan

    def scan(repository, query, *, table_name):
        assert table_name in ("earnings_events", "daily_market"), "replay opened outcomes"
        yield from original(repository, query, table_name=table_name)

    monkeypatch.setattr(Repository, "scan", scan)
    monkeypatch.setattr(magnitude, "prediction_result", forbid_access)
    monkeypatch.setattr(magnitude, "render_prediction_report", forbid_access)


def test_fold_means_event_pooled_losses_and_all_horizons_are_real_and_read_only(source, monkeypatch):
    poison_outcomes(monkeypatch, source.repository, source.snapshot,
            [event["event_date"].date().isoformat() for event in source.events[-3:]])
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    binding = source.registration.binding_json
    result = magnitude.prediction_result(source.repository, source.registration)
    assert result == magnitude.prediction_result(source.repository, source.registration)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["target"] == source.registration.document["magnitude_plan"]["target_recipe"]
    assert result["target"]["target"] == "absolute_earnings_close_to_close_return"
    assert result["features"] == ["constant_intercept"]
    assert result["attempted_variants"] == 1
    assert result["variant_id"] == source.registration.variant_id
    assert result["run_id"] == source.registration.run_id
    assert result["provenance"] == source.registration.document
    assert result["registered_population_count"] == 11
    assert result["population_labels"] == ["post-release selection"]
    assert result["holdout_results"] == {"random": "excluded; not evaluated",
                                         "rolling": "excluded; not evaluated"}
    folds = result["folds"]
    assert [(fold["year"], fold["train_count"], fold["test_count"]) for fold in folds] == [
        ("2024", 3, 3), ("2025", 7, 1)]
    assert [fold["mean_absolute_move_pct"] for fold in folds] == pytest.approx([2, 36 / 7])
    assert [row["absolute_move_pct"] for row in result["predictions"]] == [1, 7, 10, 4]
    assert [row["predicted_absolute_move_pct"] for row in result["predictions"]] == pytest.approx(
        [2, 2, 2, 36 / 7])
    expected_rmse = math.sqrt((1 + 25 + 64 + (8 / 7) ** 2) / 4)
    expected_mae = (1 + 5 + 8 + 8 / 7) / 4
    assert result["metrics"] == pytest.approx({"count": 4, "rmse": expected_rmse, "mae": expected_mae})
    assert result["metrics"]["rmse"] != pytest.approx(sum(f["metrics"]["rmse"] for f in folds) / 2)
    # Handwritten fold oracles: errors (1, 5, 8), then the single error 8/7.
    assert folds[0]["metrics"] == pytest.approx({"count": 3, "rmse": math.sqrt(30), "mae": 14 / 3})
    assert folds[1]["metrics"] == pytest.approx({"count": 1, "rmse": 8 / 7, "mae": 8 / 7})
    sealed = {event["event_id"]: event for event in result["provenance"]["magnitude_plan"]["events"]}
    assert len({row["event_id"] for row in result["predictions"]}) == 4
    for row in result["predictions"]:
        assert row["forecasts"] == sealed[row["event_id"]]["forecasts"]
        assert [clock["horizon_sessions"] for clock in row["forecasts"]] == [5, 3, 1]
        assert row["population_use"] == "post-release selection"
    assert result["forecast_count"] == 12
    assert result["per_horizon"] == [{"horizon_sessions": horizon, **result["metrics"]}
                                     for horizon in (5, 3, 1)]
    assert result["result_id"] == content_hash({k: v for k, v in result.items() if k != "result_id"})
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert source.registration.binding_json == binding
    assert all(query.snapshot_id == source.snapshot.snapshot_id for _, query in source.repository.scans)


def test_provenance_binds_qualified_omitted_and_evaluated_objects_and_cutoffs(source):
    result = magnitude.prediction_result(source.repository, source.registration)
    records = {record.time_min: record for record in source.repository.fragment_records(
        source.snapshot, "computed_moves")}
    expected = {}
    for event in source.events:
        record = records[event["event_date"].date().isoformat()]
        expected[event["event_id"]] = {"event_id": event["event_id"], "fragment_id": record.fragment_id,
            "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash}
    for fold, sealed in zip(result["folds"], source.registration.document["magnitude_plan"]["folds"]):
        training, evaluation = fold["training_inputs"], fold["evaluation_inputs"]
        assert training["labels_before"] == sealed["training_labels_before"]
        assert training["candidate_event_ids"] == sealed["training_candidate_event_ids"]
        assert len(training["qualified_event_ids"]) == fold["train_count"]
        assert len(training["read_inputs"]) == len(sealed["training_candidate_event_ids"])
        assert training["read_inputs"] == [expected[key] for key in sealed["training_candidate_event_ids"]]
        assert evaluation["labels_before"] == CUTOFF
        assert [item["event_id"] for item in evaluation["consumed_inputs"]] == sealed["evaluation_event_ids"]
        assert evaluation["consumed_inputs"] == [expected[key] for key in sealed["evaluation_event_ids"]]
        for inputs in (training, evaluation):
            assert inputs["inputs_id"] == content_hash({k: v for k, v in inputs.items()
                                                       if k not in ("inputs_id", "holdout_exclusions")})
            assert inputs["snapshot_id"] == source.snapshot.snapshot_id
        assert {item["event_id"]: item["reason"] for item in training["omitted_labels"]} == {
            **({source.events[3]["event_id"]: "not_available_before_fit"} if fold["year"] == "2024" else {}),
            source.events[4]["event_id"]: "skipped", source.events[5]["event_id"]: "unavailable",
            source.events[6]["event_id"]: "not_available_before_fit"}


def test_mean_is_frozen_before_evaluation_loader_can_mutate_training_frame(source, monkeypatch):
    qualify, evaluate = magnitude.qualify_magnitude_training_targets, magnitude.load_magnitude_targets
    training, calls = [], []

    def qualified(*args, **kwargs):
        calls.append("qualify")
        frame = qualify(*args, **kwargs)
        training.append(frame)
        return frame

    def evaluated(*args, **kwargs):
        calls.append("evaluate")
        training[-1].loc[:, "absolute_move_pct"] = 1e9
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(magnitude, "qualify_magnitude_training_targets", qualified)
    monkeypatch.setattr(magnitude, "load_magnitude_targets", evaluated)
    result = magnitude.prediction_result(source.repository, source.registration)
    assert calls == ["qualify", "evaluate", "qualify", "evaluate"]
    assert [fold["mean_absolute_move_pct"] for fold in result["folds"]] == pytest.approx([2, 36 / 7])


@pytest.mark.parametrize("value", [0., -8., 80.])
def test_changed_future_evaluation_cannot_change_earlier_fold_prediction(tmp_path, value):
    events, moves = _dataset()
    moves[8]["realized_move_pct"] = value
    source = _source(tmp_path, events=events, moves=moves)
    try:
        result = magnitude.prediction_result(source.repository, source.registration)
        assert result["folds"][0]["mean_absolute_move_pct"] == 2
        assert result["folds"][1]["mean_absolute_move_pct"] == pytest.approx((29 + abs(value)) / 7)
    finally:
        source.conn.close()


@pytest.mark.parametrize("scale", [0., 1e308])
def test_zero_and_extreme_finite_labels_keep_mean_and_scaled_losses_finite(tmp_path, scale):
    events, moves = _dataset()
    for index, multiple in ((0, -1), (1, 0), (2, 1), (7, 0), (8, 1), (9, -1)):
        moves[index]["realized_move_pct"] = multiple * scale
    source = _source(tmp_path, events=events, moves=moves, spec=_spec(folds=("2024",)))
    try:
        result = magnitude.prediction_result(source.repository, source.registration)
        assert result["folds"][0]["mean_absolute_move_pct"] == pytest.approx((2 / 3) * scale)
        assert result["metrics"] == pytest.approx({"count": 3, "rmse": (math.sqrt(2) / 3) * scale,
                                                  "mae": (4 / 9) * scale})
        assert all(math.isfinite(result["metrics"][name]) for name in ("rmse", "mae"))
        json.dumps(result, allow_nan=False)
    finally:
        source.conn.close()


@pytest.mark.parametrize("changes", [{"seed": 1}, {"seed": False}, {"economic_params": {"fill": .5}},
    {"runner": "other.py"}, {"price_source": "option_chains"}, {"input_files": ("other.csv",)},
    {"folds": ()}, {"folds": ("2025", "2024")}, {"folds": ("2024", "2024")},
    {"arms": ("mean", "candidate")}])
def test_fixed_baseline_spec_refuses_unsupported_declarations(changes):
    with pytest.raises(OpsError) as caught:
        magnitude.validate_prediction_spec(_spec(**changes))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


@pytest.mark.parametrize("kind", ["skipped", "unavailable", "equal-cutoff", "nonfinite", "empty-training"])
def test_bad_later_evaluation_or_empty_training_never_returns_partial_result(tmp_path, kind):
    events, moves = _dataset()
    if kind == "empty-training":
        for row in moves[:7]:
            row.update(skipped=True, available_as_of_date=None, realized_move_pct=None)
    else:
        moves[10].update({"skipped": {"skipped": True}, "unavailable": {"available_as_of_date": None},
            "equal-cutoff": {"available_as_of_date": CUTOFF},
            "nonfinite": {"realized_move_pct": float("nan")}}[kind])
    source = _source(tmp_path, events=events, moves=moves)
    try:
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        with pytest.raises((DataError, OpsError)) as caught:
            magnitude.prediction_result(source.repository, source.registration)
        assert caught.value.code == ("POPULATION_COLLAPSED" if kind == "empty-training"
                                     else "EXPERIMENT_VARIANT_FAILED")
        assert not caught.value.problem.retryable
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
        assert not list(tmp_path.rglob("REPORT.md"))
    finally:
        source.conn.close()


def test_later_mixed_object_blocks_entire_run_before_first_training_label(tmp_path, monkeypatch):
    events, moves = _dataset()
    protected = history_event("2025-01-08", protected=True)
    source = _source(tmp_path, events=events + [protected], moves=moves + [outcome_move(protected, None)],
                     mixed_days=("2025-01-07", "2025-01-08"))
    try:
        poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
        with pytest.raises(DataError) as caught:
            magnitude.prediction_result(source.repository, source.registration)
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED" and not caught.value.problem.retryable
        assert all(name != "computed_moves" for name, _ in source.repository.scans)
    finally:
        source.conn.close()


def test_rendered_report_has_reviewable_clock_provenance_losses_and_limits(source):
    result = magnitude.prediction_result(source.repository, source.registration)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    for no_ledger in (False, True):
        report = magnitude.render_prediction_report(source.registration, result,
                                                     spec=source.spec, no_ledger=no_ledger)
        assert report == magnitude.render_prediction_report(source.registration, result,
                                                            spec=source.spec, no_ledger=no_ledger)
        for text in (source.spec.hypothesis, source.spec.spec_hash, source.registration.variant_id,
                     source.snapshot.snapshot_id, "RMSE", "MAE", "percentage_points", CUTOFF,
                     "quantile", "unverified", "Random holdout", "Rolling holdout"):
            assert text in report
        assert ("no-ledger smoke; not recorded" if no_ledger else "recorded evaluation") in report
        assert json.loads(report.split("```json\n", 1)[1].split("\n```", 1)[0]) == result
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("field", ["variant_id", "run_id", "provenance"])
def test_renderer_refuses_cross_registration_results(source, field):
    result = magnitude.prediction_result(source.repository, source.registration)
    result[field] = "another-registration"
    result["result_id"] = content_hash({key: value for key, value in result.items() if key != "result_id"})
    reference = {"variant_id": source.registration.variant_id, "run_id": source.registration.run_id,
                 "provenance": source.registration.document}
    assert {key: result[key] != value for key, value in reference.items()} == {
        key: key == field for key in reference}
    assert source.spec.spec_hash == source.registration.document["spec_hash"]
    with pytest.raises(OpsError) as caught:
        magnitude.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False


def test_renderer_refuses_other_hypothesis_with_otherwise_valid_result(source):
    result = magnitude.prediction_result(source.repository, source.registration)
    original = canonical_json(result)
    spec = replace(source.spec, hypothesis="A different hypothesis")
    assert spec.spec_hash != source.registration.document["spec_hash"]
    assert result["variant_id"] == source.registration.variant_id
    assert result["run_id"] == source.registration.run_id
    assert result["provenance"] == source.registration.document
    assert result["result_id"] == content_hash({key: value for key, value in result.items() if key != "result_id"})
    with pytest.raises(OpsError) as caught:
        magnitude.render_prediction_report(source.registration, result, spec=spec, no_ledger=True)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert canonical_json(result) == original


def test_renderer_digest_guard_independently_rejects_metrics_only_tampering(source, monkeypatch):
    result = magnitude.prediction_result(source.repository, source.registration)
    original_id = result["result_id"]
    result["metrics"]["rmse"] += 1
    assert result["variant_id"] == source.registration.variant_id
    assert result["run_id"] == source.registration.run_id
    assert result["provenance"] == source.registration.document
    payload = {key: value for key, value in result.items() if key != "result_id"}
    assert result["result_id"] == original_id != content_hash(payload)

    def assert_refused():
        with pytest.raises(OpsError) as caught:
            magnitude.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert caught.value.problem.retryable is False

    assert_refused()
    calls = []

    def neutralize_only_result_digest(document):
        assert document == payload
        calls.append(document)
        return original_id

    # Isolated mutant control: making only this digest check pass defeats the refusal oracle.
    with monkeypatch.context() as patch:
        patch.setattr(magnitude, "content_hash", neutralize_only_result_digest)
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            assert_refused()
    assert calls == [payload]


def test_cli_metadata_registration_then_durable_smoke_never_touches_ledger(tmp_path, monkeypatch, capsys):
    source = _source(tmp_path, register=False)
    try:
        with monkeypatch.context() as patch:
            _metadata_only(patch)
            registration = _invoke(source, capsys, "register")
        evidence = baseline_evidence(source)
        assert registration["registered"] == evidence["variant_id"]
        assert evidence["planned_variants"] == 1
        assert not {"native_smoke", "native_outcome", "attempted_variants"} & evidence.keys()
        binding = read_outcome_document(source, evidence["native_registration"])
        assert binding["schema_version"] == "native_experiment_registration.v1.3"
        assert binding["source_closure"] == native.source_closure(ROOT, [source.spec.runner])
        assert {"experiments/v2_prediction.py", "experiments/magnitude_report.py"} <= set(binding["source_closure"])
        poison = tmp_path / "do-not-touch" / "ledger.csv"

        def protect(operation):
            def guarded(path, *args, **kwargs):
                assert "do-not-touch" not in path.parts, "smoke inspected a ledger path"
                return operation(path, *args, **kwargs)
            return guarded

        with monkeypatch.context() as patch:
            for method in ("resolve", "open", "stat", "lstat", "mkdir", "touch"):
                patch.setattr(Path, method, protect(getattr(Path, method)))
            patch.setattr(outcomes, "ledger_append", forbid_access)
            output = _invoke(source, capsys, "run", "--no-ledger", "--ledger", poison)
        result = result_from_report(output["report"])
        assert output["recording_mode"] == "smoke"
        assert result["target"]["target"] == "absolute_earnings_close_to_close_return"
        assert result["metrics"]["count"] == 4
        assert result["metrics"]["rmse"] == pytest.approx(math.sqrt((90 + (8 / 7) ** 2) / 4))
        evidence = baseline_evidence(source)
        outcome, completion = (read_outcome_document(source, evidence[key]) for key in ("native_smoke", "native_smoke_receipt"))
        assert "native_outcome" not in evidence
        assert outcome["failure_code"] is None and outcome["attempted_variants"] == 1
        assert outcome["ledger_destination"] is None
        assert completion["schema_version"] == "native_experiment_completion.v1.0"
        assert completion["outcome_ref"] == evidence["native_smoke"]
        assert completion["report_ref"] == outcome["report_ref"]
        assert completion["recording_mode"] == "smoke"
        assert completion["ledger_row_hash"] is None
        assert not poison.parent.exists() and not list(tmp_path.rglob("*.csv"))
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        with monkeypatch.context() as patch:
            _metadata_only(patch)
            replayed = _invoke(source, capsys, "run", "--no-ledger")
        assert replayed["record_export"]["status"] == "unchanged"
        assert output["record_export"]["status"] == "written"
        assert {key: value for key, value in replayed.items() if key != "record_export"} == {
            key: value for key, value in output.items() if key != "record_export"}
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()


def test_cli_recorded_lifecycle_replays_original_date_with_exactly_one_ledger_row(tmp_path, monkeypatch, capsys):
    source = _source(tmp_path, register=False)
    try:
        _invoke(source, capsys, "register")
        smoke = _invoke(source, capsys, "run", "--no-ledger")
        ledger = tmp_path / "recorded" / "ledger.csv"
        set_outcome_clock(monkeypatch, "2026-01-02")
        first = _invoke(source, capsys, "run", "--ledger", ledger)
        assert first["recording_mode"] == "recorded" and first["report"] != smoke["report"]
        assert result_from_report(first["report"]) == result_from_report(smoke["report"])
        evidence = baseline_evidence(source)
        outcome = read_outcome_document(source, evidence["native_outcome"])
        completion = read_outcome_document(source, evidence["native_outcome_receipt"])
        assert outcome["failure_code"] is None and outcome["attempted_variants"] == 1
        assert completion["recording_mode"] == "recorded"
        assert completion["ledger_row_hash"] == content_hash(outcome["ledger_row"])
        assert read_ledger_rows(ledger) == [outcome["ledger_row"]]
        assert read_ledger_rows(ledger)[0]["date"] == "2026-01-02"
        assert read_ledger_rows(ledger)[0]["stage"] == "ran" and read_ledger_rows(ledger)[0]["promoted"] == "False"
        before = catalog_rows(source.conn), artifact_object_bytes(source.store), ledger.read_bytes()
        set_outcome_clock(monkeypatch, "2026-02-10")
        _metadata_only(monkeypatch)
        replayed = _invoke(source, capsys, "run", "--ledger", ledger)
        assert {key: value for key, value in replayed.items() if key != "record_export"} == {
            key: value for key, value in first.items() if key != "record_export"}
        assert first["record_export"]["status"] == "written"
        assert replayed["record_export"]["status"] == "unchanged"
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store), ledger.read_bytes()) == before
    finally:
        source.conn.close()


def test_cli_interrupted_append_resumes_saved_result_without_fitting(tmp_path, monkeypatch, capsys):
    source = _source(tmp_path, register=False)
    try:
        _invoke(source, capsys, "register")
        ledger = tmp_path / "ledger.csv"
        set_outcome_clock(monkeypatch, "2026-01-02")

        def interrupted(*args, **kwargs):
            raise OSError("synthetic interrupted append")

        with monkeypatch.context() as patch:
            patch.setattr(outcomes, "ledger_append", interrupted)
            failed = _invoke(source, capsys, "run", "--ledger", ledger, status=2)
        assert failed["refused"] == "RESOURCE_UNAVAILABLE"
        evidence = baseline_evidence(source)
        assert "native_outcome" in evidence and "native_outcome_receipt" not in evidence
        outcome = read_outcome_document(source, evidence["native_outcome"])
        assert not ledger.exists() and not list(tmp_path.rglob("REPORT.md"))
        set_outcome_clock(monkeypatch, "2026-02-10")
        _metadata_only(monkeypatch)
        completed = _invoke(source, capsys, "run", "--ledger", ledger)
        assert completed["recording_mode"] == "recorded"
        assert read_ledger_rows(ledger) == [outcome["ledger_row"]]
        assert read_outcome_document(source, baseline_evidence(source)["native_outcome"]) == outcome
    finally:
        source.conn.close()


@pytest.mark.parametrize("extra", [
    ["--target", "absolute_move"],
    ["--target", "absolute_move", "--magnitude-training-start", START],
    ["--target", "absolute_move", "--magnitude-labels-before", CUTOFF],
    ["--magnitude-training-start", START],
    ["--magnitude-labels-before", CUTOFF],
    ["--target", "positive_move", "--magnitude-training-start", START, "--magnitude-labels-before", CUTOFF],
])
@pytest.mark.parametrize("action", ["register", "run"])
def test_cli_invalid_mode_combinations_refuse_before_store_or_spec_access(monkeypatch, capsys, action, extra):
    monkeypatch.setattr(cli.foundation, "ArtifactStore", forbid_access)
    monkeypatch.setattr(cli, "connect", forbid_access)
    monkeypatch.setattr(Path, "read_text", forbid_access)
    with pytest.raises(SystemExit) as caught:
        cli.main([action, "--spec", "unused", "--catalog", "unused", "--store-root", "unused",
                  "--as-of-month", MONTH, "--no-ledger", *extra])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert captured.err.splitlines()[-1].endswith(
        ": error: absolute_move requires both magnitude clocks; positive_move accepts neither")
    assert not captured.out


@pytest.mark.parametrize("kind", ["v1.0", "v1.1", "v1.2", "null-envelope", "list-envelope",
    "missing-execution", "null-execution", "wrong-metric", "changed-cutoff", "missing-plan", "broken-clock"])
def test_registration_version_and_closed_contract_refuse_before_any_label_read(source, monkeypatch, kind):
    document = source.registration.document
    if kind.startswith("v1."):
        document["schema_version"] = "native_experiment_registration." + kind
        document.pop("magnitude_execution")
    elif kind.endswith("envelope"):
        document = None if kind == "null-envelope" else []
    elif kind == "missing-execution":
        document.pop("magnitude_execution")
    elif kind == "null-execution":
        document["magnitude_execution"] = None
    elif kind == "wrong-metric":
        document["magnitude_execution"]["primary_metric"] = "fold_mean_rmse"
    elif kind == "changed-cutoff":
        document["magnitude_execution"]["evaluation_labels_before"] = "2025-01-07"
    elif kind == "missing-plan":
        document.pop("magnitude_plan")
    else:
        clock = document["magnitude_plan"]["events"][0]["forecasts"][0]
        clock["horizon_sessions"] = 4
        clock["cutoff_id"] = content_hash({k: v for k, v in clock.items() if k != "cutoff_id"})
        plan = document["magnitude_plan"]
        plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    invalid = native.NativeRegistration(source.registration.run_id, canonical_json(document).encode())
    monkeypatch.setattr(source.repository, "scan", forbid_access)
    with pytest.raises(OpsError) as caught:
        magnitude.prediction_result(source.repository, invalid)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not caught.value.problem.retryable


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("kind", ["empty-training", "bad-later-evaluation"])
def test_cli_failed_attempt_is_durable_without_partial_report_or_refit(tmp_path, monkeypatch, capsys, smoke, kind):
    events, moves = _dataset()
    if kind == "empty-training":
        for row in moves[:7]:
            row.update(skipped=True, available_as_of_date=None, realized_move_pct=None)
    else:
        moves[10]["available_as_of_date"] = CUTOFF
    source = _source(tmp_path, register=False, events=events, moves=moves)
    try:
        _invoke(source, capsys, "register")
        ledger = tmp_path / "recorded" / "ledger.csv"
        args = ["--ledger", ledger, *(["--no-ledger"] if smoke else [])]
        monkeypatch.setattr(magnitude, "render_prediction_report", forbid_access)
        response = _invoke(source, capsys, "run", *args, status=2)
        assert response["refused"] == "EXPERIMENT_VARIANT_FAILED"
        assert read_outcome_document(source, response["refusal_ref"])["failure_code"] == "EXPERIMENT_VARIANT_FAILED"
        key = "native_smoke" if smoke else "native_outcome"
        evidence = baseline_evidence(source)
        outcome = read_outcome_document(source, evidence[key])
        assert outcome["failure_code"] == "EXPERIMENT_VARIANT_FAILED"
        assert outcome["attempted_variants"] == 1 and outcome["report_ref"] is None
        assert outcome["ledger_row"]["stage"] == "failed"
        assert read_outcome_document(source, evidence[key + "_receipt"])["report_ref"] is None
        assert not list(tmp_path.rglob("REPORT.md"))
        if smoke:
            assert not ledger.parent.exists()
        else:
            assert read_ledger_rows(ledger) == [outcome["ledger_row"]]
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        _metadata_only(monkeypatch)
        replay = _invoke(source, capsys, "run", *args, status=2)
        assert replay["refused"] == "EXPERIMENT_VARIANT_FAILED"
        assert replay["receipt_ref"] == evidence[key + "_receipt"]
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()
