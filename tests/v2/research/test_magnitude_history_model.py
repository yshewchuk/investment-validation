"""Synthetic native stored-history forecasts, sealed admission and CLI replay."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.data import catalog, objects
from engine.v2.data.errors import DataError
from engine.v2.data.manifests import table_contract_hash
from engine.v2.data.repository import Repository
from engine.v2.foundation import SystemClock, canonical_json, content_hash, to_document
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_registration import validate_magnitude_history_execution
from engine.v2.research import magnitude_history_store as stored
from engine.v2.research.magnitude_history_inputs import admit_stored
from engine.v2.research.magnitude_history_table import FEATURES, METADATA, TABLE
from experiments import magnitude_history_report as model
from experiments import magnitude_history_store as history
from experiments import magnitude_inputs as inputs
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments import v2_prediction as cli
from tests.v2.research import magnitude_trial_support as support
from tests.v2.research.magnitude_trial_support import (
    CUTOFF,
    HISTORY,
    MONTH,
    ROOT,
    START,
    build_history,
    catalog_rows,
    forbid_history_rebuild,
    history_binding,
    history_event,
    magnitude_dataset,
    magnitude_source,
    outcome_move,
    poison_outcomes,
    register_history,
)
from tests.v2.research.test_magnitude_baseline import baseline_evidence, result_from_report
from tests.v2.research.test_native_outcomes import (
    read_ledger_rows,
    read_outcome_document,
    set_outcome_clock,
)
from tests.v2.research.test_native_registration import artifact_object_bytes


@pytest.fixture
def source(tmp_path):
    value = magnitude_source(tmp_path)
    yield value
    value.conn.close()


def _fallback_source(tmp_path):
    events = [history_event("2023-01-03", ticker="ZERO"), history_event("2023-06-02", ticker="TRAIN"),
              history_event("2024-06-12", ticker="ZERO"), history_event("2024-06-17", ticker="NEW")]
    return magnitude_source(tmp_path, events=events,
        moves=[outcome_move(event, value) for event, value in zip(events, (0., 8., 10., 6.))], folds=("2024",))


def history_model_args(source, action, *extra):
    argv = [action, "--spec", str(source.spec_path), "--catalog", str(source.root / "catalog.sqlite"),
        "--store-root", str(source.store.root), "--as-of-month", MONTH, "--target", "absolute_move",
        "--magnitude-training-start", START, "--magnitude-labels-before", CUTOFF,
        "--magnitude-model", "history_mean", "--magnitude-feature-snapshot-id",
        source.feature_snapshot.snapshot_id, "--magnitude-history-start", HISTORY, *map(str, extra)]
    if action == "run":
        argv += ["--exp-number", "101", "--experiments-root", str(source.root / "experiments")]
    return argv


def _invoke(source, capsys, action, *extra, status=0):
    assert cli.main(history_model_args(source, action, *extra)) == status
    captured = capsys.readouterr()
    err_lines = [line for line in captured.err.splitlines()
                 if not line.startswith("WARNING: record_export")]
    assert all(line.startswith("  [") for line in err_lines) and "Traceback" not in captured.out
    return json.loads(captured.out)


def forbid_history_model_payloads(monkeypatch):
    original = Repository.scan

    def scan(repository, query, *, table_name):
        assert table_name in ("earnings_events", "daily_market"), "admission/replay opened payload"
        yield from original(repository, query, table_name=table_name)

    monkeypatch.setattr(Repository, "scan", scan)
    monkeypatch.setattr(model, "prediction_result", support.forbid_access)
    monkeypatch.setattr(model, "render_prediction_report", support.forbid_access)
    forbid_history_rebuild(monkeypatch)


def _expected(source):
    values = {("SYNTH_A", "2024-06-12"): (10., (4., 4., 4.)),
        ("SYNTH_B", "2024-06-12"): (20., (22 / 3, 22 / 3, 22 / 3)),
        ("SYNTH_A", "2024-06-17"): (8., (4., 4., 11 / 2)),
        ("SYNTH_B", "2024-06-17"): (4., (22 / 3, 22 / 3, 21 / 2)),
        ("SYNTH_A", "2025-01-07"): (12., (6., 6., 6.)),
        ("SYNTH_B", "2025-01-07"): (16., (46 / 5, 46 / 5, 46 / 5))}
    return {(event["event_id"], horizon): (actual, predicted)
        for event in source.events
        if (key := (event["ticker"], event["event_date"].date().isoformat())) in values
        for actual, forecasts in (values[key],)
        for horizon, predicted in zip((5, 3, 1), forecasts)}


def expected_history_metrics(pairs):
    # Independent elementary arithmetic, never the production metric implementation.
    errors = [abs(actual - predicted) for actual, predicted in pairs]
    return dict(count=len(errors), rmse=math.sqrt(sum(error * error for error in errors) / len(errors)),
                mae=sum(errors) / len(errors))


def _assert_fold_metrics(source, result):
    expected = _expected(source)
    event_years = {event["event_id"]: event["event_date"].year for event in source.events}
    assert [fold["year"] for fold in result["folds"]] == ["2024", "2025"]
    for fold, year, count in zip(result["folds"], (2024, 2025), (12, 6)):
        pairs = [pair for (event_id, _), pair in expected.items() if event_years[event_id] == year]
        assert len(pairs) == count and fold["test_count"] == count // 3
        assert fold["metrics"] == pytest.approx(expected_history_metrics(pairs)), "fold metrics must use only that fold's pairs"


def history_event_ids(source, dates):
    return [event["event_id"] for event in sorted(source.events,
        key=lambda event: (event["event_date"], event["event_id"]))
        if event["event_date"].date().isoformat() in dates]


def test_real_stored_rows_produce_issuer_and_horizon_specific_read_only_forecasts(source, monkeypatch):
    forbid_history_rebuild(monkeypatch)
    # Pre-window history is needed by the stored model, but must never be decoded again.
    poison_outcomes(monkeypatch, source.repository, source.snapshot, ("2022-06-02", "2023-04-03", "2025-12-03"))
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    result = model.prediction_result(source.repository, source.registration)
    assert result == model.prediction_result(source.repository, source.registration)
    assert result["schema_version"] == "native_magnitude_history_result.v1"
    assert result["model"] == "stored_ticker_arithmetic_mean"
    assert result["features"] == ["n_prior", "mean_prior_abs_move"]
    assert result["weighting"] == "equal_event_and_horizon"
    assert result["population_labels"] == ["post-release selection"]
    assert result["registered_population_count"] == 10
    assert result["quantile_calibration"] == "not_implemented"
    assert result["event_count"] == 6 and result["forecast_count"] == 18
    assert result["attempted_variants"] == 1
    assert result["variant_id"] == source.registration.variant_id
    assert result["run_id"] == source.registration.run_id
    assert result["provenance"] == source.registration.document
    assert result["holdout_results"] == {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"}
    expected = _expected(source)
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in result["predictions"]}
    assert len(indexed) == len(result["predictions"]) == 18 and indexed.keys() == expected.keys()
    sealed = {event["event_id"]: event for event in source.registration.document["magnitude_plan"]["events"]}
    for key, (actual, predicted) in expected.items():
        row = indexed[key]
        assert row["absolute_move_pct"] == actual
        assert row["predicted_absolute_move_pct"] == pytest.approx(predicted)
        assert row["source"] == "stored_history"
        day = sealed[key[0]]["event_date"]
        assert row["n_prior"] == (5 if day == "2025-01-07" else
                                 4 if day == "2024-06-17" and key[1] == 1 else 3)
        assert row["fold"] == sealed[key[0]]["event_date"][:4]
        assert row["forecast"] == next(clock for clock in sealed[key[0]]["forecasts"]
                                      if clock["horizon_sessions"] == key[1])
    assert result["metrics"] == pytest.approx(expected_history_metrics(expected.values()))
    for horizon in result["per_horizon"]:
        assert {key: value for key, value in horizon.items() if key != "horizon_sessions"} == pytest.approx(
            expected_history_metrics(pair for (_, count), pair in expected.items() if count == horizon["horizon_sessions"]))
    assert {row["horizon_sessions"] for row in result["per_horizon"]} == {5, 3, 1}
    _assert_fold_metrics(source, result)
    for fold, count, dates, mean in zip(result["folds"], (4, 8), (
        ("2023-01-03", "2023-06-02"),
        ("2023-01-03", "2023-06-02", "2024-06-12", "2024-06-17")), (7.5, 9.)):
        assert fold["train_count"] == count
        assert fold["train_event_ids"] == history_event_ids(source, dates)
        assert fold["mean_absolute_move_pct"] == mean
    assert result["metrics"]["rmse"] != pytest.approx(sum(fold["metrics"]["rmse"] for fold in result["folds"]) / 2)
    assert result["result_id"] == content_hash({key: value for key, value in result.items() if key != "result_id"})
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_fold_oracle_rejects_cumulative_later_fold_diagnostics(source):
    result = model.prediction_result(source.repository, source.registration)
    _assert_fold_metrics(source, result)
    # Exact counterfactual to magnitude_metrics(predictions) instead of magnitude_metrics(scored):
    # the second fold would score all 18 forecasts rather than its own six.
    cumulative = json.loads(json.dumps(result))
    cumulative["folds"][1]["metrics"] = expected_history_metrics(_expected(source).values())
    with pytest.raises(AssertionError, match="fold metrics must use only that fold's pairs"):
        _assert_fold_metrics(source, cumulative)


def test_direct_reporter_preflights_later_label_before_feature_or_label_payloads(tmp_path, monkeypatch):
    events, moves = magnitude_dataset()
    # This unselected sibling shares the last admitted label's physical object.
    # Neither row is historical input to any forecast, so the stored build is valid.
    moves.append(outcome_move(history_event("2025-01-08", ticker="SYNTH_A"), None))
    source = magnitude_source(tmp_path, events=events, moves=moves, mixed_days=("2025-01-07", "2025-01-08"))
    try:
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
        for name in ("read_stored", "qualify_magnitude_training_targets", "load_magnitude_targets"):
            monkeypatch.setattr(model, name, support.forbid_access)
        with support.refuses(DataError, "HOLDOUT_ACCESS_DENIED"):
            model.prediction_result(source.repository, source.registration)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
        # Without run-wide label preflight, the forbidden feature read is reached.
        monkeypatch.setattr(model, "preflight_magnitude_targets", lambda *args, **kwargs: {})
        with pytest.raises(AssertionError, match="forbidden access"):
            model.prediction_result(source.repository, source.registration)
    finally:
        source.conn.close()


def test_genuine_zero_history_does_not_take_the_empty_history_fallback(tmp_path, monkeypatch):
    source = _fallback_source(tmp_path)
    try:
        forbid_history_rebuild(monkeypatch)
        result = model.prediction_result(source.repository, source.registration)
        assert result["event_count"] == 2 and result["forecast_count"] == 6
        for row in result["predictions"]:
            if row["event_id"] == source.events[2]["event_id"]:
                assert row["source"] == "stored_history" and row["predicted_absolute_move_pct"] == 0
            else:
                assert row["event_id"] == source.events[3]["event_id"]
                assert row["source"] == "fold_mean" and row["predicted_absolute_move_pct"] == 4
        assert result["metrics"] == pytest.approx({"count": 6, "rmse": math.sqrt(52), "mae": 6})
    finally:
        source.conn.close()


def test_history_values_counts_and_fold_fallback_are_frozen_before_evaluation(tmp_path, monkeypatch):
    source = _fallback_source(tmp_path)
    read, qualify, evaluate = model.read_stored, model.qualify_magnitude_training_targets, model.load_magnitude_targets
    stored_rows, training, calls = [], [], []

    def read_features(*args, **kwargs):
        calls.append("stored")
        result = read(*args, **kwargs)
        stored_rows.extend(result["rows"])
        return result

    def qualified(*args, **kwargs):
        calls.append("qualify")
        frame = qualify(*args, **kwargs)
        training.append(frame)
        return frame

    def evaluated(*args, **kwargs):
        calls.append("evaluate")
        assert stored_rows and training
        training[-1].loc[:, "absolute_move_pct"] = 1e9
        for row in stored_rows:
            row.update(n_prior=999, mean_prior_abs_move=1e9)
        return evaluate(*args, **kwargs)

    try:
        monkeypatch.setattr(model, "read_stored", read_features)
        monkeypatch.setattr(model, "qualify_magnitude_training_targets", qualified)
        monkeypatch.setattr(model, "load_magnitude_targets", evaluated)
        result = model.prediction_result(source.repository, source.registration)
        assert calls == ["stored", "qualify", "evaluate"]
        assert result["folds"][0]["mean_absolute_move_pct"] == 4
        for row in result["predictions"]:
            if row["event_id"] == source.events[2]["event_id"]:
                assert (row["source"], row["n_prior"], row["predicted_absolute_move_pct"]) == ("stored_history", 1, 0)
            else:
                assert (row["source"], row["n_prior"], row["predicted_absolute_move_pct"]) == ("fold_mean", 0, 4)
    finally:
        source.conn.close()


def test_result_binds_exact_stored_feature_objects_rows_columns_and_input_digest(source):
    read = stored.read_stored(source.repository, source.feature_snapshot, source.build["context"])
    result = model.prediction_result(source.repository, source.registration)
    assert result["stored_inputs"] == {"inputs_id": read["inputs_id"],
        "feature_snapshot": to_document(source.feature_snapshot), "columns": list((*METADATA, *FEATURES)),
        "row_keys": [[row[key] for key in ("event_id", "context_id", "horizon_sessions")] for row in read["rows"]],
        "objects": [{"fragment_id": record.fragment_id, "object_id": record.object_ref.object_id,
                     "content_hash": record.object_ref.content_hash}
                    for record in source.repository.fragment_records(source.feature_snapshot, TABLE)]}
    assert len(result["stored_inputs"]["row_keys"]) == 30
    assert len(result["stored_inputs"]["objects"]) == 10
    for fold in result["folds"]:
        assert fold["training_inputs"]["labels_before"] == fold["training_labels_before"]
        assert fold["evaluation_inputs"]["labels_before"] == CUTOFF
        assert {row["event_id"] for row in fold["evaluation_inputs"]["consumed_inputs"]} == set(fold["test_event_ids"])


def _qualification_source(tmp_path):
    # New issuers in each evaluation year force the fallback in both folds.
    days = ("2023-01-03", "2023-06-02", "2023-06-05", "2023-06-06",
            "2023-06-07", "2023-06-08", "2024-06-12", "2025-01-07")
    events = [history_event(day, ticker=ticker) for day, ticker in zip(days,
        ("TRAIN_A", "TRAIN_B", "LATE", "SKIP", "UNKNOWN", "FUTURE", "NEW_A", "NEW_B"))]
    moves = [outcome_move(event, value) for event, value in zip(events, (-2., 8., 12., None, None, None, -10., -4.))]
    moves[2]["available_as_of_date"] = "2024-01-01"
    moves[3].update(skipped=True, available_as_of_date=None)
    moves[4]["available_as_of_date"] = None
    moves[5]["available_as_of_date"] = CUTOFF
    return magnitude_source(tmp_path, events=events, moves=moves)


def _expected_label_evidence(source):
    # Construct membership from fixture identities and physical catalog records,
    # never the model result, returned attrs, target loader or sealed fold fields.
    keys = [event["event_id"] for event in source.events]
    records = {(record.partition_key, record.time_min): record for record in
        source.repository.fragment_records(source.snapshot, "computed_moves")}
    physical = {}
    for event in source.events:
        record = records[event["ticker"], event["event_date"].date().isoformat()]
        physical[event["event_id"]] = {"event_id": event["event_id"], "fragment_id": record.fragment_id,
            "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash}
    recipe = {"schema_version": "earnings_magnitude.v1", "units": "percentage_points",
        "target": "absolute_earnings_close_to_close_return", "price_basis": "split_neutral_non_dividend_adjusted",
        "estimand": "conditional_mean", "primary_loss": "pooled_rmse",
        "secondary_diagnostics": ["mae", "quantile_calibration"]}
    common = {"snapshot_id": source.snapshot.snapshot_id, "holdout_as_of_month": "2026-01",
        "random_membership_version": "canonical-event-sha256.v1", "rolling_membership_version": "calendar-months.v1",
        "population_use": "post-release selection", "purpose": "selection",
        "computed_moves_version": source.snapshot.table_versions["computed_moves"].dataset_version_id}
    preflight = {"schema_version": "research_magnitude_preflight.v1", **common,
        "event_ids": keys, "planned_inputs": [physical[key] for key in keys]}
    preflight["preflight_id"] = content_hash(preflight)
    folds = []
    for year, before, candidates, qualified, test in (
        ("2024", "2024-01-01", keys[:6], keys[:2], keys[6:7]),
        ("2025", "2024-12-30", keys[:7], [*keys[:3], keys[6]], keys[7:])):
        omitted = ([{"event_id": keys[2], "reason": "not_available_before_fit"}] if year == "2024" else [])
        omitted += [{"event_id": key, "reason": reason} for key, reason in zip(keys[3:6],
            ("skipped", "unavailable", "not_available_before_fit"))]
        training = {"schema_version": "research_magnitude_training_targets.v1", **common,
            "labels_before": before, "target_recipe": recipe, "target_recipe_id": content_hash(recipe),
            "candidate_event_ids": candidates, "qualified_event_ids": qualified,
            "read_inputs": [physical[key] for key in candidates], "omitted_labels": omitted}
        evaluation = {"schema_version": "research_magnitude_targets.v1", **common,
            "labels_before": "2025-02-01", "target_recipe": recipe, "target_recipe_id": content_hash(recipe),
            "consumed_inputs": [physical[key] for key in test], "excluded_future_events": []}
        for evidence in (training, evaluation):
            evidence.update(inputs_id=content_hash(evidence), holdout_exclusions=[])
        folds.append({"year": year, "training_labels_before": before,
            "training_candidate_event_ids": candidates, "evaluation_event_ids": test,
            "train_count": len(qualified), "test_count": 1, "train_event_ids": qualified,
            "test_event_ids": test, "training_inputs": training, "evaluation_inputs": evaluation})
    return recipe, preflight, folds


def _assert_label_evidence(source, result):
    recipe, preflight, folds = _expected_label_evidence(source)
    assert result["target"] == recipe
    assert result["preflight"] == preflight
    assert len(result["folds"]) == 2
    for actual, expected in zip(result["folds"], folds):
        assert {key: actual[key] for key in expected} == expected


def _assert_fallbacks(source, result):
    assert [fold["mean_absolute_move_pct"] for fold in result["folds"]] == [5., 8.]
    expected = {(source.events[index]["event_id"], horizon): (year, mean, actual)
        for index, year, mean, actual in ((6, "2024", 5., 10.), (7, "2025", 8., 4.))
        for horizon in (5, 3, 1)}
    assert len(result["predictions"]) == len(expected) == 6
    assert {(row["event_id"], row["horizon_sessions"]) for row in result["predictions"]} == expected.keys()
    for row in result["predictions"]:
        assert (row["fold"], row["predicted_absolute_move_pct"], row["absolute_move_pct"]) == expected[
            row["event_id"], row["horizon_sessions"]]
        assert row["source"] == "fold_mean" and row["n_prior"] == 0
    assert result["metrics"] == pytest.approx({"count": 6, "rmse": math.sqrt(20.5), "mae": 4.5})


def test_two_fold_empty_histories_use_distinct_absolute_qualified_means_and_complete_evidence(tmp_path, monkeypatch):
    source = _qualification_source(tmp_path)
    try:
        result = model.prediction_result(source.repository, source.registration)
        _assert_label_evidence(source, result)
        _assert_fallbacks(source, result)
        # Execute both counterfactuals through the reporter: neither constant 4
        # nor signed training means (3, 2) may pass the independent prediction oracle.
        qualify = model.qualify_magnitude_training_targets

        def signed_training(*args, **kwargs):
            frame = qualify(*args, **kwargs)
            frame["absolute_move_pct"] = frame.realized_move_pct
            return frame

        for name, replacement in (("mean", lambda _: 4.),
            ("qualify_magnitude_training_targets", signed_training)):
            with monkeypatch.context() as patch:
                patch.setattr(model, name, replacement)
                changed = model.prediction_result(source.repository, source.registration)
            with pytest.raises(AssertionError):
                _assert_fallbacks(source, changed)
        for field in ("target", "preflight"):
            changed = json.loads(json.dumps(result))
            changed[field] = {}
            with pytest.raises(AssertionError):
                _assert_label_evidence(source, changed)
        # A self-consistent digest cannot conceal a lost physical content hash.
        for field, objects in (("training_inputs", "read_inputs"), ("evaluation_inputs", "consumed_inputs")):
            changed = json.loads(json.dumps(result))
            evidence = changed["folds"][0][field]
            evidence[objects][0].pop("content_hash")
            evidence["inputs_id"] = content_hash({key: value for key, value in evidence.items()
                                                 if key not in ("inputs_id", "holdout_exclusions")})
            with pytest.raises(AssertionError):
                _assert_label_evidence(source, changed)
        for field, key in (("training_inputs", "read_inputs"), ("training_inputs", "inputs_id"),
            ("training_inputs", "qualified_event_ids"), ("training_inputs", "omitted_labels"),
            ("evaluation_inputs", "consumed_inputs"), ("evaluation_inputs", "inputs_id")):
            changed = json.loads(json.dumps(result))
            changed["folds"][0][field].pop(key)
            with pytest.raises(AssertionError):
                _assert_label_evidence(source, changed)
    finally:
        source.conn.close()


def test_history_binding_is_exact_metadata_only_and_registers_one_attempt(tmp_path, monkeypatch):
    source = magnitude_source(tmp_path, register=False)
    try:
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        with monkeypatch.context() as patch:
            forbid_history_model_payloads(patch)
            poison_outcomes(patch, source.repository, source.snapshot, (), all_outcomes=True)
            registration = register_history(source)
            assert register_history(source) == registration
            assert native.require_native_registration(source.conn, source.store, source.repository,
                source.spec, **history_binding(source)) == registration
            admitted, preflight = inputs.admit_magnitude_inputs(source.conn, source.store,
                source.repository, source.spec, **history_binding(source))
        assert admitted == registration and preflight
        binding = registration.document
        assert binding["schema_version"] == "native_experiment_registration.v1.4"
        assert binding["magnitude_history_request"] == {
            "feature_snapshot_id": source.feature_snapshot.snapshot_id, "history_start": HISTORY}
        assert binding["magnitude_history"] == {"schema_version": "magnitude_history_binding.v1",
            "feature_snapshot": to_document(source.feature_snapshot), "context": source.build["context"]}
        assert binding["magnitude_history"]["context"]["temporal_plan"] == binding["magnitude_plan"]
        assert binding["magnitude_history"]["context"]["source_snapshot"] == binding["snapshot"]
        assert binding["magnitude_execution"] == {
            "schema_version": "magnitude_history_execution.v1", "evaluation_labels_before": CUTOFF,
            "training_labels": "omit_skipped_unavailable_or_late_before_numeric_parse",
            "evaluation_labels": "complete_finite_strictly_before_cutoff",
            "model": "stored_ticker_arithmetic_mean", "features": ["n_prior", "mean_prior_abs_move"],
            "empty_history": "fold_local_arithmetic_mean", "primary_metric": "event_horizon_pooled_rmse",
            "secondary_metrics": ["mae"], "quantile_calibration": "not_implemented",
            "weighting": "equal_event_and_horizon"}
        assert {"experiments/v2_prediction.py", "experiments/magnitude_history_report.py"} <= binding["source_closure"].keys()
        assert registration.binding_json == canonical_json(binding).encode()
        evidence = baseline_evidence(source)
        assert evidence["planned_variants"] == 1 and evidence["variant_id"] == registration.variant_id
        assert not {"native_smoke", "native_outcome", "attempted_variants"} & evidence.keys()
        after = catalog_rows(source.conn), artifact_object_bytes(source.store)
        assert {table for table in before[0] if before[0][table] != after[0][table]} == {"hypotheses", "experiment_runs"}
        assert [after[1][path] for path in after[1].keys() - before[1].keys()] == [registration.binding_json]
    finally:
        source.conn.close()


@pytest.mark.parametrize("changes", [
    {"magnitude_feature_snapshot_id": "other-snapshot"}, {"magnitude_history_start": "2022-02-01"},
    {"magnitude_feature_snapshot_id": None, "magnitude_history_start": None}, {"as_of_month": "2026-02"},
])
def test_request_drift_refuses_before_metadata_or_feature_payload(source, monkeypatch, changes):
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    monkeypatch.setattr(stored, "read_stored", support.forbid_access)
    for operation in (native.require_native_registration, inputs.admit_magnitude_inputs):
        with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
            operation(source.conn, source.store, source.repository, source.spec,
                      **{**history_binding(source), **changes})
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("kind", ["absent", "wrong-scope", "wrong-snapshot"])
def test_absent_or_colliding_original_receipt_refuses_without_payloads(tmp_path, monkeypatch, kind):
    source = magnitude_source(tmp_path, register=False, build=False)
    try:
        context = history.context(source.repository, source.snapshot, as_of_month=MONTH,
            training_start=START, history_start=HISTORY, evaluation_years=list(source.spec.folds))
        if kind != "absent":
            scope, receipt = stored._names(context)
            parent = source.repository.resolve_full(source.snapshot.snapshot_id)
            catalog.commit_snapshot(source.conn, scope=scope if kind == "wrong-snapshot" else "wrong-scope",
                request_hash=context["context_id"], receipt_id=receipt, attempt_id=receipt, fence=1,
                expected_head_snapshot_id=None, expected_head_generation=0, contracts=parent.contracts,
                objects=parent.objects, records=parent.records, manifests=tuple(parent.table_manifests.values()),
                snapshot=parent.snapshot, fence_check=lambda conn: None, clock=SystemClock(),
                store=source.store, audit_partitions=False)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
        monkeypatch.setattr(source.repository, "scan", support.forbid_access)
        for operation in (admit_stored, stored.read_stored):
            with support.refuses(DataError, "IDENTITY_CONFLICT"):
                operation(source.repository, source.snapshot, context)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()


@pytest.mark.parametrize("kind,code", [("old-contract", "CONTRACT_MISMATCH"), ("mixed-events", "HOLDOUT_ACCESS_DENIED")])
def test_real_native_feature_contract_and_fragment_isolation_precede_payloads(tmp_path, monkeypatch, kind, code):
    with monkeypatch.context() as producer:
        if kind == "old-contract":
            old = replace(stored.CONTRACT, contract_id=TABLE + ".v0", semantic_version="0.0.0")
            producer.setattr(stored, "CONTRACT", replace(old, definition_hash=table_contract_hash(old)))
        else:
            original = stored._event_rows

            def mixed(*args):
                rows = original(*args)
                if rows[0]["event_date"] == "2025-01-07" and rows[0]["event_id"].startswith("SYNTH_B:"):
                    rows[-1]["event_id"] += ":foreign-event"
                return rows

            producer.setattr(stored, "_event_rows", mixed)
        source = magnitude_source(tmp_path, register=False)
    try:
        forbidden = {r.object_ref.object_id for r in source.repository.fragment_records(source.feature_snapshot, TABLE)}
        original_verify = objects.verify_object_path

        def verify(store, ref):
            assert ref.object_id not in forbidden, "admission opened a feature object"
            return original_verify(store, ref)

        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        monkeypatch.setattr(objects, "verify_object_path", verify)
        poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
        for operation in (admit_stored, stored.read_stored):
            with support.refuses(DataError, code):
                operation(source.repository, source.feature_snapshot, source.build["context"])
        with support.refuses(OpsError if kind == "old-contract" else DataError,
                      "EXPERIMENT_IDENTITY_CONFLICT" if kind == "old-contract" else code):
            register_history(source)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()


@pytest.mark.parametrize("kind", ["v1.0", "v1.1", "v1.2", "v1.3", "missing-history", "missing-request", "null-history",
    "wrong-feature-id", "wrong-source", "bad-context-hash", "unsupported-policy", "boolean-horizon",
    "extra-request", "extra-binding", "extra-context", "wrong-history-schema", "wrong-feature-recipe",
    "wrong-producer-environment", "wrong-producer-source", "missing-producer-source", "wrong-history-start",
    "outer-source-list", "producer-source-list", "null-producer", "list-context", "list-execution"])
def test_closed_schema_or_cross_context_binding_refuses_before_payload(source, monkeypatch, kind):
    document = source.registration.document
    if kind.startswith("v1."):
        document["schema_version"] = "native_experiment_registration." + kind
    elif kind.startswith("missing-"):
        document.pop("magnitude_history" if kind == "missing-history" else "magnitude_history_request")
    elif kind == "null-history":
        document["magnitude_history"] = None
    elif kind == "wrong-feature-id":
        document["magnitude_history_request"]["feature_snapshot_id"] = "other-snapshot"
    elif kind == "wrong-source":
        context = document["magnitude_history"]["context"]
        context["source_snapshot"]["snapshot_id"] = "other-source"
        context["context_id"] = content_hash({key: value for key, value in context.items() if key != "context_id"})
    elif kind == "bad-context-hash":
        document["magnitude_history"]["context"]["context_id"] = "sha256:" + "0" * 64
    elif kind == "unsupported-policy":
        document["magnitude_execution"]["model"] = "choose_best_history_feature"
    elif kind == "outer-source-list":
        document["source_closure"] = []
    elif kind == "list-context":
        document["magnitude_history"]["context"] = []
    elif kind == "list-execution":
        document["magnitude_execution"] = []
    elif kind == "boolean-horizon":
        clock = document["magnitude_plan"]["events"][0]["forecasts"][-1]
        assert clock["horizon_sessions"] == 1
        clock["horizon_sessions"] = True
        clock["cutoff_id"] = content_hash({key: value for key, value in clock.items() if key != "cutoff_id"})
        plan = document["magnitude_plan"]
        plan["plan_id"] = content_hash({key: value for key, value in plan.items() if key != "plan_id"})
        context = document["magnitude_history"]["context"]
        context["temporal_plan"] = json.loads(json.dumps(plan))
        context["context_id"] = content_hash({key: value for key, value in context.items() if key != "context_id"})
    else:
        binding = document["magnitude_history"]
        context = binding["context"]
        if kind == "extra-request":
            document["magnitude_history_request"]["model"] = "another-model"
        elif kind == "extra-binding":
            binding["unchecked"] = True
        elif kind == "extra-context":
            context["unchecked"] = True
        elif kind == "wrong-history-schema":
            binding["schema_version"] = "magnitude_history_binding.v2"
        elif kind == "wrong-feature-recipe":
            context["feature_recipe"]["features"][1]["on_missing"] = "zero"
        elif kind == "wrong-producer-environment":
            context["producer"]["environment"]["numerical_threads"] = 2
        elif kind == "wrong-producer-source":
            context["producer"]["source_closure"]["engine/v2/features/panel_math.py"] = "sha256:" + "f" * 64
        elif kind == "missing-producer-source":
            context["producer"]["source_closure"].pop("experiments/magnitude_history_store.py")
        elif kind == "producer-source-list":
            context["producer"]["source_closure"] = []
        elif kind == "null-producer":
            context["producer"] = None
        else:
            context["history_start"] = "2022-02-01"
        context["context_id"] = content_hash({key: value for key, value in context.items() if key != "context_id"})
    invalid = native.NativeRegistration(source.registration.run_id, canonical_json(document).encode())
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        model.prediction_result(source.repository, invalid)


def test_changed_execution_source_refuses_before_metadata_or_feature_payload(source, monkeypatch):
    original = native.source_closure

    def changed(*args, **kwargs):
        return {**original(*args, **kwargs), "experiments/magnitude_history_report.py": "sha256:" + "f" * 64}

    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(native, "source_closure", changed)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        inputs.admit_magnitude_inputs(source.conn, source.store, source.repository, source.spec, **history_binding(source))
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_changed_producer_environment_refuses_without_feature_or_label_payload(source, monkeypatch):
    environment = {**source.build["context"]["producer"]["environment"], "numerical_threads": 2}
    monkeypatch.setattr(history, "environment_identity", lambda _: environment)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
    monkeypatch.setattr(model, "read_stored", support.forbid_access)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        model.prediction_result(source.repository, source.registration)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        inputs.admit_magnitude_inputs(source.conn, source.store, source.repository, source.spec, **history_binding(source))
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_direct_reporter_rederives_self_rehashed_calendar_context_before_payloads(source, monkeypatch):
    document = source.registration.document
    context = document["magnitude_history"]["context"]
    plan = context["temporal_plan"]
    for event in plan["events"]:
        for clock in event["forecasts"]:
            clock["calendar_id"] = "sha256:" + "0" * 64
            clock["cutoff_id"] = content_hash({key: value for key, value in clock.items() if key != "cutoff_id"})
    plan["plan_id"] = content_hash({key: value for key, value in plan.items() if key != "plan_id"})
    context["context_id"] = content_hash({key: value for key, value in context.items() if key != "context_id"})
    feature = stored.materialize(source.repository, source.snapshot, context)
    # Lower-level storage deliberately permits a self-consistent supplied context.
    assert len(stored.read_stored(source.repository, feature, context)["rows"]) == 30
    document["magnitude_plan"] = plan
    document["magnitude_history"]["feature_snapshot"] = to_document(feature)
    document["magnitude_history_request"]["feature_snapshot_id"] = feature.snapshot_id
    validate_magnitude_history_execution(document)  # Integrity alone is insufficient admission.
    invalid = native.NativeRegistration(source.registration.run_id, canonical_json(document).encode())
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
    monkeypatch.setattr(model, "read_stored", support.forbid_access)
    monkeypatch.setattr(stored, "_scan", support.forbid_access)
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        model.prediction_result(source.repository, invalid)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_fake_runner_closure_cannot_publish_invalid_first_registration(tmp_path, monkeypatch):
    source = magnitude_source(tmp_path, register=False)
    try:
        code = tmp_path / "fake-code"
        (code / "experiments").mkdir(parents=True)
        (code / "experiments" / "v2_prediction.py").write_text("pass\n")
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        forbid_history_model_payloads(monkeypatch)
        with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
            native.register_native(source.conn, source.store, source.repository, source.spec,
                                   **{**history_binding(source), "code_root": code})
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
        assert not baseline_evidence(source)
    finally:
        source.conn.close()


def test_history_registration_cannot_prepare_final_holdout_authority(source, monkeypatch):
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", support.forbid_access)
    with support.refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        native.prepare_final_holdout_request(source.conn, source.store, source.registration)
    assert source.conn.execute("SELECT COUNT(*) FROM final_holdout_authorizations").fetchone()[0] == 0
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("field", ["variant_id", "run_id", "provenance", "result_id", "schema_version"])
def test_renderer_refuses_identity_and_digest_drift(source, field):
    result = model.prediction_result(source.repository, source.registration)
    # Positive control: this same real result renders before its identity changes.
    assert model.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)
    result[field] = "native_magnitude_result.v1" if field == "schema_version" else "another-result"
    if field != "result_id":
        result["result_id"] = content_hash({key: value for key, value in result.items() if key != "result_id"})
    if field == "schema_version":
        # Otherwise-valid identity and checksum isolate the incompatible-schema guard.
        assert result["variant_id"] == source.registration.variant_id
        assert result["run_id"] == source.registration.run_id
        assert result["provenance"] == source.registration.document
        assert result["result_id"] == content_hash({key: value for key, value in result.items() if key != "result_id"})
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        model.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)


def test_report_is_deterministic_and_binds_stored_snapshot_clocks_and_limits(source):
    result = model.prediction_result(source.repository, source.registration)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    for no_ledger in (False, True):
        report = model.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=no_ledger)
        assert report == model.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=no_ledger)
        for text in (source.spec.hypothesis, source.spec.spec_hash, source.snapshot.snapshot_id,
                     source.feature_snapshot.snapshot_id, "RMSE", "MAE", "percentage_points", CUTOFF,
                     "quantile", "unverified", "Random and rolling holdouts excluded"):
            assert text in report
        assert ("no-ledger smoke; not recorded" if no_ledger else "recorded evaluation") in report
        assert json.loads(report.split("```json\n", 1)[1].split("\n```", 1)[0]) == result
    with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        model.render_prediction_report(source.registration, result,
            spec=replace(source.spec, hypothesis="Another hypothesis"), no_ledger=True)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("kind", ["late-evaluation", "nonfinite-evaluation", "empty-training"])
def test_bad_later_label_or_empty_training_refuses_without_partial_result(tmp_path, monkeypatch, kind):
    events, moves = magnitude_dataset()
    if kind == "empty-training":
        for row in moves:
            if row["event_date"].startswith("2023"):
                row.update(skipped=True, available_as_of_date=None, realized_move_pct=None)
    else:
        row = next(row for row in moves if row["event_date"] == "2025-01-07" and row["ticker"] == "SYNTH_B")
        row.update({"available_as_of_date": CUTOFF} if kind == "late-evaluation"
                   else {"realized_move_pct": float("nan")})
    source = magnitude_source(tmp_path, events=events, moves=moves)
    try:
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        forbid_history_rebuild(monkeypatch)
        with support.refuses(DataError, "POPULATION_COLLAPSED" if kind == "empty-training" else "EXPERIMENT_VARIANT_FAILED"):
            model.prediction_result(source.repository, source.registration)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
        assert not list(tmp_path.rglob("REPORT.md")) and not list(tmp_path.rglob("*.csv"))
    finally:
        source.conn.close()


def test_native_build_cli_register_smoke_record_and_replay_without_rebuilding(tmp_path, monkeypatch, capsys):
    source = magnitude_source(tmp_path, register=False)
    try:
        with monkeypatch.context() as patch:
            forbid_history_model_payloads(patch)
            registered = _invoke(source, capsys, "register")
        assert registered["registered"] == baseline_evidence(source)["variant_id"]
        forbid_history_rebuild(monkeypatch)
        poison_outcomes(monkeypatch, source.repository, source.snapshot, ("2022-06-02", "2023-04-03", "2025-12-03"))
        poison = tmp_path / "do-not-touch" / "ledger.csv"

        def protect(operation):
            def guarded(path, *args, **kwargs):
                assert "do-not-touch" not in path.parts, "smoke inspected a ledger path"
                return operation(path, *args, **kwargs)
            return guarded

        with monkeypatch.context() as patch:
            for name in ("resolve", "open", "stat", "lstat", "mkdir", "touch"):
                patch.setattr(Path, name, protect(getattr(Path, name)))
            patch.setattr(outcomes, "ledger_append", support.forbid_access)
            smoke = _invoke(source, capsys, "run", "--no-ledger", "--ledger", poison)
        result = result_from_report(smoke["report"])
        assert result["metrics"] == pytest.approx(expected_history_metrics(_expected(source).values()))
        assert result["event_count"] == 6 and result["forecast_count"] == 18
        assert smoke["recording_mode"] == "smoke"
        evidence = baseline_evidence(source)
        smoke_outcome = read_outcome_document(source, evidence["native_smoke"])
        assert smoke_outcome["failure_code"] is None and smoke_outcome["ledger_destination"] is None
        assert read_outcome_document(source, evidence["native_smoke_receipt"])["ledger_row_hash"] is None
        assert "native_outcome" not in evidence and not poison.parent.exists()
        assert not list(tmp_path.rglob("*.csv"))
        ledger = tmp_path / "recorded" / "ledger.csv"
        set_outcome_clock(monkeypatch, "2026-01-02")
        recorded = _invoke(source, capsys, "run", "--ledger", ledger)
        assert recorded["recording_mode"] == "recorded" and recorded["report"] != smoke["report"]
        assert result_from_report(recorded["report"]) == result
        evidence = baseline_evidence(source)
        outcome = read_outcome_document(source, evidence["native_outcome"])
        completion = read_outcome_document(source, evidence["native_outcome_receipt"])
        assert outcome["attempted_variants"] == 1 and outcome["failure_code"] is None
        assert completion["ledger_row_hash"] == content_hash(outcome["ledger_row"])
        assert read_ledger_rows(ledger) == [outcome["ledger_row"]]
        assert read_ledger_rows(ledger)[0]["date"] == "2026-01-02"
        assert read_ledger_rows(ledger)[0]["stage"] == "ran" and read_ledger_rows(ledger)[0]["promoted"] == "False"
        before = catalog_rows(source.conn), artifact_object_bytes(source.store), ledger.read_bytes()
        set_outcome_clock(monkeypatch, "2026-02-10")
        forbid_history_model_payloads(monkeypatch)
        smoke_replay = _invoke(source, capsys, "run", "--no-ledger")
        assert smoke_replay["record_export"]["status"] == "unchanged"
        assert smoke["record_export"]["status"] == "written"
        assert {key: value for key, value in smoke_replay.items() if key != "record_export"} == {
            key: value for key, value in smoke.items() if key != "record_export"}
        recorded_replay = _invoke(source, capsys, "run", "--ledger", ledger)
        assert {key: value for key, value in recorded_replay.items() if key != "record_export"} == {
            key: value for key, value in recorded.items() if key != "record_export"}
        assert recorded["record_export"]["status"] == "written"
        assert recorded_replay["record_export"]["status"] == "unchanged"
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store), ledger.read_bytes()) == before
    finally:
        source.conn.close()


def test_omitted_and_explicit_pooled_model_keep_identical_v13_registration(tmp_path, capsys):
    source = magnitude_source(tmp_path, register=False)
    try:
        arguments = history_model_args(source, "register")
        arguments = arguments[:arguments.index("--magnitude-model")]
        assert cli.main(arguments) == 0
        original = json.loads(capsys.readouterr().out)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        assert cli.main([*arguments, "--magnitude-model", "pooled_mean"]) == 0
        assert json.loads(capsys.readouterr().out) == original
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
        binding = read_outcome_document(source, baseline_evidence(source)["native_registration"])
        assert binding["schema_version"] == "native_experiment_registration.v1.3"
        assert "magnitude_history_request" not in binding and "magnitude_history" not in binding
    finally:
        source.conn.close()


def test_cli_history_drift_is_r6_before_attempt_and_never_uses_smoke_or_ledger_slot(source, monkeypatch, capsys):
    _, changed_feature = build_history(source, evaluation_years=list(source.spec.folds), history_start="2022-02-01")
    assert changed_feature != source.feature_snapshot
    before_catalog, before_objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
    forbid_history_model_payloads(monkeypatch)
    monkeypatch.setattr(outcomes, "ledger_append", support.forbid_access)
    ledger = source.root / "must-not-exist" / "ledger.csv"
    response = _invoke(source, capsys, "run", "--no-ledger", "--ledger", ledger,
                       "--magnitude-history-start", "2022-02-01", "--magnitude-feature-snapshot-id",
                       changed_feature.snapshot_id, status=2)
    assert response["refused"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert read_outcome_document(source, response["refusal_ref"])["failure_code"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(source.conn) == before_catalog
    assert not {"native_smoke", "native_outcome", "attempted_variants"} & baseline_evidence(source).keys()
    assert not ledger.parent.exists() and not list(source.root.rglob("REPORT.md"))
    assert {path: artifact_object_bytes(source.store)[path] for path in before_objects} == before_objects


@pytest.mark.parametrize("extra", [
    ["--magnitude-model", "history_mean"], ["--magnitude-model", "pooled_mean"],
    ["--target", "absolute_move", "--magnitude-model", "history_mean",
     "--magnitude-training-start", START, "--magnitude-labels-before", CUTOFF],
    ["--target", "absolute_move", "--magnitude-model", "pooled_mean",
     "--magnitude-training-start", START, "--magnitude-labels-before", CUTOFF,
     "--magnitude-history-start", HISTORY, "--magnitude-feature-snapshot-id", "unused"],
    ["--magnitude-history-start", HISTORY], ["--magnitude-feature-snapshot-id", "unused"],
])
@pytest.mark.parametrize("action", ["register", "run"])
def test_invalid_model_options_refuse_before_store_or_spec(monkeypatch, capsys, action, extra):
    monkeypatch.setattr(cli.foundation, "ArtifactStore", support.forbid_access)
    monkeypatch.setattr(cli, "connect", support.forbid_access)
    monkeypatch.setattr(Path, "read_text", support.forbid_access)
    with pytest.raises(SystemExit) as caught:
        cli.main([action, "--spec", "unused", "--catalog", "unused", "--store-root", "unused",
                  "--as-of-month", MONTH, "--no-ledger", *extra])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert ": error:" in captured.err and not captured.out


@pytest.mark.parametrize("action", ["register", "run"])
@pytest.mark.parametrize("kind", ["history_mean", "history_ridge", "history_hgb"])
@pytest.mark.parametrize("option,value", [
    ("--magnitude-feature-snapshot-id", "synthetic-feature-pin"),
    ("--magnitude-history-start", HISTORY),
])
def test_history_cli_half_pair_refuses_at_exact_parser_guard(monkeypatch, capsys, action, kind, option, value):
    monkeypatch.setattr(cli.foundation, "ArtifactStore", support.forbid_access)
    monkeypatch.setattr(cli, "connect", support.forbid_access)
    monkeypatch.setattr(Path, "read_text", support.forbid_access)
    with pytest.raises(SystemExit) as caught:
        cli.main([action, "--spec", "unused", "--catalog", "unused", "--store-root", "unused",
            "--as-of-month", MONTH, "--no-ledger", "--target", "absolute_move",
            "--magnitude-training-start", START, "--magnitude-labels-before", CUTOFF,
            "--magnitude-model", kind, option, value])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err.split(": error: ")[-1] == (
        "history_mean/history_ridge/history_hgb require absolute_move and both history options; other models accept neither\n")


@pytest.fixture
def fresh_history_source(tmp_path):
    source = magnitude_source(tmp_path, register=False, build=False)
    yield source
    source.conn.close()


def _fresh_history_arguments(**changes):
    return {"code_root": ROOT, "as_of_month": MONTH, "magnitude_training_start": START,
        "magnitude_labels_before": CUTOFF, "magnitude_feature_snapshot_id": "synthetic-feature-pin",
        "magnitude_history_start": HISTORY, **changes}


def _assert_history_problem(error, code, message):
    assert error.code == code
    assert error.problem.retryable is False
    assert error.problem.message == message


def _forbid_native_history_effects(monkeypatch, source):
    monkeypatch.setattr(source.store, "publish_bytes", support.forbid_access)
    monkeypatch.setattr(native, "transaction", support.forbid_access)
    monkeypatch.setattr(native, "register_hypothesis_in_transaction", support.forbid_access)
    for name in ("publish_native_outcome", "export_native_report", "ledger_append"):
        monkeypatch.setattr(outcomes, name, support.forbid_access)


def _assert_history_request_before_access(source, monkeypatch, changes, message, *, spec=None):
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(native, "source_closure", support.forbid_access)
    for name in ("resolve_pinned", "resolve", "scan"):
        monkeypatch.setattr(source.repository, name, support.forbid_access)
    _forbid_native_history_effects(monkeypatch, source)
    for operation in (native.register_native, native.require_native_registration):
        with pytest.raises(OpsError) as caught:
            operation(source.conn, source.store, source.repository, spec or source.spec,
                      **_fresh_history_arguments(**changes))
        _assert_history_problem(caught.value, "INVALID_EXPERIMENT_SPEC", message)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("changes", [
    {"magnitude_feature_snapshot_id": None}, {"magnitude_feature_snapshot_id": ""},
    {"magnitude_feature_snapshot_id": " padded "}, {"magnitude_feature_snapshot_id": 123},
    {"magnitude_feature_snapshot_id": []}, {"magnitude_history_start": None},
    {"magnitude_history_start": ""}, {"magnitude_history_start": "20220101"},
    {"magnitude_history_start": "2022-1-01"}, {"magnitude_history_start": "2022-02-30"},
    {"magnitude_history_start": "2022-01-01T00:00:00Z"}, {"magnitude_history_start": 20220101},
])
def test_history_library_literal_guards_precede_all_access(fresh_history_source, monkeypatch, changes):
    _assert_history_request_before_access(fresh_history_source, monkeypatch, changes,
        "history requires an exact feature snapshot and canonical start")


@pytest.mark.parametrize("kind", ["incompatible-runner", "missing-cutoff", "missing-execution"])
def test_history_library_execution_guards_precede_all_access(fresh_history_source, monkeypatch, kind):
    source = fresh_history_source
    spec = replace(source.spec, runner="experiments/magnitude_report.py") if kind == "incompatible-runner" else source.spec
    changes = {} if kind == "incompatible-runner" else {"magnitude_labels_before": None}
    if kind == "missing-execution":
        changes["magnitude_training_start"] = None
    _assert_history_request_before_access(source, monkeypatch, changes,
        "stored history requires magnitude execution and its native runner", spec=spec)


def test_history_fresh_missing_pin_translates_real_resolution_before_effects(fresh_history_source, monkeypatch):
    source = fresh_history_source
    missing = "synthetic-feature-pin"
    with pytest.raises(DataError) as caught:
        source.repository.resolve(missing)
    _assert_history_problem(caught.value, "SNAPSHOT_NOT_FOUND", "unknown snapshot id")
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    resolve, resolved = source.repository.resolve, []

    def track_resolve(snapshot_id):
        resolved.append(snapshot_id)
        return resolve(snapshot_id)

    monkeypatch.setattr(source.repository, "resolve", track_resolve)
    forbid_history_model_payloads(monkeypatch)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
    _forbid_native_history_effects(monkeypatch, source)
    monkeypatch.setattr(stored, "read_stored", support.forbid_access)
    monkeypatch.setattr(model, "read_stored", support.forbid_access)
    with pytest.raises(OpsError) as caught:
        native.register_native(source.conn, source.store, source.repository, source.spec,
                               **_fresh_history_arguments())
    _assert_history_problem(caught.value, "EXPERIMENT_IDENTITY_CONFLICT",
                            "stored history identity could not be admitted")
    assert resolved[-1] == missing and resolved.count(missing) == 1
    assert {name for name, _ in source.repository.scans} == {"earnings_events", "daily_market"}
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert not list(source.root.rglob("REPORT.md")) and not list(source.root.rglob("*.csv"))


def test_history_reader_identity_translation_is_reached_after_real_admission(source, monkeypatch):
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    forbid_history_rebuild(monkeypatch)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
    feature_objects = {r.object_ref.object_id
                       for r in source.repository.fragment_records(source.feature_snapshot, TABLE)}
    verify, calls = objects.verify_object_path, []

    def metadata_only(store, ref):
        assert ref.object_id not in feature_objects, "identity refusal opened feature bytes"
        return verify(store, ref)

    def changed_reader(repository, feature, context):
        calls.append((feature, context))
        # Substitute another real committed snapshot only at the reader boundary.
        # The registrar and the reporter's metadata admission still see the valid pin.
        return stored.read_stored(repository, source.snapshot, context)

    monkeypatch.setattr(objects, "verify_object_path", metadata_only)
    _forbid_native_history_effects(monkeypatch, source)
    for name in ("qualify_magnitude_training_targets", "load_magnitude_targets"):
        monkeypatch.setattr(model, name, support.forbid_access)
    with pytest.raises(DataError) as caught:
        stored.read_stored(source.repository, source.snapshot, source.build["context"])
    _assert_history_problem(caught.value, "IDENTITY_CONFLICT",
                            "feature snapshot is not the committed context output")
    monkeypatch.setattr(model, "read_stored", changed_reader)
    with pytest.raises(OpsError) as caught:
        model.prediction_result(source.repository, source.registration)
    _assert_history_problem(caught.value, "EXPERIMENT_IDENTITY_CONFLICT",
                            "stored feature identity changed after admission")
    assert calls == [(source.feature_snapshot, source.build["context"])]
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert not list(source.root.rglob("REPORT.md")) and not list(source.root.rglob("*.csv"))
