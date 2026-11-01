"""Real native fitted-history admission, independent numerics and replay evidence."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.v2.data.errors import DataError
from engine.v2.foundation import canonical_json, to_document
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.training import magnitude as training
from engine.v2.ops.errors import OpsError
from experiments import magnitude_history_report as model
from experiments import magnitude_inputs as inputs
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments import v2_prediction as cli
from tests.v2.research.magnitude_trial_support import (
    CUTOFF,
    HISTORY,
    MONTH,
    START,
    catalog_rows,
    forbid_access,
    forbid_history_rebuild,
    history_binding,
    history_event,
    magnitude_dataset,
    magnitude_source,
    outcome_move,
    poison_outcomes,
    refuses,
)
from tests.v2.research.test_magnitude_baseline import baseline_evidence, result_from_report
from tests.v2.research.test_magnitude_history_model import (
    expected_history_metrics,
    forbid_history_model_payloads,
    history_event_ids,
    history_model_args,
)
from tests.v2.research.test_native_outcomes import (
    read_ledger_rows,
    read_outcome_document,
    set_outcome_clock,
)
from tests.v2.research.test_native_registration import artifact_object_bytes

FEATURES = ["n_prior", "mean_prior_move", "mean_prior_abs_move", "ema2_prior_move",
    "ema2_prior_abs_move", "ema4_prior_move", "ema4_prior_abs_move", "ema8_prior_move",
    "ema8_prior_abs_move", "ema12_prior_move", "ema12_prior_abs_move"]
HORIZONS = (5, 3, 1)
DATES = ("2023-01-03", "2023-06-02", "2024-06-12", "2024-06-17", "2025-01-07")
# These are independently authored features for the fixture's signed histories.
# No production feature function, returned result, or fitting helper supplies them.
HISTORY_ROWS = {
    "SYNTH_A": [
        [1, -2, 2, None, None, None, None, None, None, None, None],
        [2, -3, 3, -10 / 3, 10 / 3, None, None, None, None, None, None],
        [3, 0, 4, 26 / 9, 46 / 9, None, None, None, None, None, None],
        [4, 5 / 2, 11 / 2, 206 / 27, 226 / 27, 554 / 125, 806 / 125, None, None, None, None],
        [5, 18 / 5, 6, 638 / 81, 658 / 81, 3662 / 625, 4418 / 625, None, None, None, None]],
    "SYNTH_B": [
        [1, 2, 2, None, None, None, None, None, None, None, None],
        [2, 5, 5, 6, 6, None, None, None, None, None, None],
        [3, -2 / 3, 22 / 3, -6, 10, None, None, None, None, None, None],
        [4, 9 / 2, 21 / 2, 34 / 3, 50 / 3, 838 / 125, 1558 / 125, None, None, None, None],
        [5, 22 / 5, 46 / 5, 58 / 9, 74 / 9, 3514 / 625, 5674 / 625, None, None, None, None]],
}
TARGETS = {"SYNTH_A": (4., 6., 10., 8., 12.), "SYNTH_B": (8., 12., 20., 4., 16.)}


def _sha(document):
    return "sha256:" + hashlib.sha256(canonical_json(document).encode()).hexdigest()


def _register(source, kind="history_ridge"):
    source.registration = native.register_native(source.conn, source.store, source.repository,
        source.spec, **history_binding(source, magnitude_fit_model=kind))
    return source.registration


@pytest.fixture
def source(tmp_path):
    value = magnitude_source(tmp_path, register=False)
    _register(value)
    yield value
    value.conn.close()


def _fixture_rows(source, dates, horizon):
    events = {event["event_id"]: event for event in source.events}
    ids = history_event_ids(source, dates)
    rows, labels = [], []
    for key in ids:
        event = events[key]
        index = DATES.index(event["event_date"].date().isoformat())
        count = 2 if index == 3 and horizon != 1 else index
        rows.append(HISTORY_ROWS[event["ticker"]][count])
        labels.append(TARGETS[event["ticker"]][index])
    return ids, rows, labels


def _oracle_fits(source):
    for year, train_dates, test_dates in (("2024", DATES[:2], DATES[2:4]),
                                         ("2025", DATES[:4], DATES[4:])):
        if year not in source.spec.folds:
            continue
        for horizon in HORIZONS:
            train_ids, train, labels = _fixture_rows(source, train_dates, horizon)
            test_ids, test, actual = _fixture_rows(source, test_dates, horizon)
            yield year, horizon, train_ids, train, labels, test_ids, test, actual


def _ridge_oracle(train, labels, test):
    """Median, all flags, population scaling and alpha-one normal equations."""
    train, test, y = (np.asarray(value, dtype=float) for value in (train, test, labels))
    medians = np.asarray([np.median(column[~np.isnan(column)]) if np.isfinite(column).any()
                          else 0. for column in train.T])
    def expanded(matrix):
        return np.column_stack([np.where(np.isnan(matrix), medians, matrix), np.isnan(matrix)])
    x, future = expanded(train), expanded(test)
    center, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.
    x, future = (x - center) / scale, (future - center) / scale
    beta = np.linalg.solve(x.T @ x + np.eye(22), x.T @ (y - y.mean()))
    return np.maximum(future @ beta + y.mean(), 0.).tolist()


def _fit_spy(monkeypatch):
    calls, original = [], training.fit_magnitude_predictions
    def fit(recipe, train, labels, test):
        assert recipe == source_recipe(recipe["model"])
        calls.append((recipe["model"], np.array(train, dtype=float), np.array(labels, dtype=float),
                      np.array(test, dtype=float)))
        before = canonical_json(recipe)
        result = original(recipe, train, labels, test)
        assert canonical_json(recipe) == before
        return result
    def source_recipe(kind):
        return native.magnitude_fitted_execution_document(CUTOFF, kind)["fit_recipe"]
    monkeypatch.setattr(training, "fit_magnitude_predictions", fit)
    return calls


def test_real_native_ridge_six_fits_match_independent_features_normal_equations_and_metrics(source, monkeypatch):
    forbid_history_rebuild(monkeypatch)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, ("2022-06-02", "2023-04-03", "2025-12-03"))
    calls = _fit_spy(monkeypatch)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    stored = model.read_stored(source.repository, source.feature_snapshot, source.build["context"])
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in stored["rows"]}
    result = model.prediction_result(source.repository, source.registration)
    assert set(result) == {"schema_version", "target", "model", "features", "attempted_variants",
        "variant_id", "run_id", "provenance", "preflight", "stored_inputs", "registered_population_count",
        "population_labels", "event_count", "forecast_count", "weighting", "metrics", "per_horizon",
        "holdout_results", "quantile_calibration", "folds", "predictions", "fit_recipe", "result_id"}
    assert result["schema_version"] == "native_magnitude_fitted_result.v1"
    assert result["features"] == FEATURES and result["model"] == "history_ridge"
    assert result["event_count"] == 6 and result["forecast_count"] == 18
    assert [fold["year"] for fold in result["folds"]] == ["2024", "2025"]
    assert [(fold["train_count"], fold["test_count"]) for fold in result["folds"]] == [(4, 4), (8, 2)]
    assert result["provenance"] == source.registration.document
    assert result["variant_id"] == source.registration.variant_id and result["run_id"] == source.registration.run_id
    assert result["weighting"] == "equal_event_and_horizon" and result["attempted_variants"] == 1
    assert result["quantile_calibration"] == "not_implemented"
    assert result["holdout_results"] == {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"}
    assert len(calls) == 6
    expected, expected_fits = {}, {year: [] for year in source.spec.folds}
    for call, (year, horizon, train_ids, train, labels, test_ids, test, actual) in zip(calls, _oracle_fits(source)):
        assert call[0] == "history_ridge"
        for got, want in zip(call[1:], (train, labels, test)):
            np.testing.assert_allclose(got, np.asarray(want, dtype=float), equal_nan=True, rtol=1e-14)
        for key, observed, forecast in zip(test_ids, actual, _ridge_oracle(train, labels, test)):
            expected[key, horizon] = (year, observed, forecast)
        # Membership and schema come from the fixture, not the fitted report.
        # Hash exact physical floats: the independent rational oracle above checks
        # their values without assuming a particular last-bit EMA operation order.
        def matrix(ids):
            return {"columns": FEATURES, "row_keys": [
                [key, source.build["context"]["context_id"], horizon] for key in ids],
                "values": [[indexed[key, horizon][name] for name in FEATURES] for key in ids]}
        expected_fits[year].append({"horizon_sessions": horizon, "training_event_ids": train_ids,
            "evaluation_event_ids": test_ids, "training_matrix_id": _sha({**matrix(train_ids),
                "absolute_move_pct": labels}), "prediction_matrix_id": _sha(matrix(test_ids))})
    rows = {(row["event_id"], row["horizon_sessions"]): row for row in result["predictions"]}
    assert len(rows) == len(result["predictions"]) == len(expected)
    assert rows.keys() == expected.keys()
    for key, (year, actual, forecast) in expected.items():
        assert rows[key]["fold"] == year and rows[key]["absolute_move_pct"] == actual
        assert rows[key]["predicted_absolute_move_pct"] == pytest.approx(forecast, abs=1e-11)
        assert rows[key]["source"] == "history_ridge"
        assert rows[key]["forecast"]["horizon_sessions"] == key[1]
    assert result["metrics"] == pytest.approx(expected_history_metrics((a, p) for _, a, p in expected.values()))
    for fold in result["folds"]:
        assert fold["metrics"] == pytest.approx(expected_history_metrics((a, p) for year, a, p in expected.values()
                                                         if year == fold["year"]))
        assert "mean_absolute_move_pct" not in fold, "fitted candidates must never silently fall back"
        assert fold["fits"] == expected_fits[fold["year"]]
        assert fold["train_event_ids"] == expected_fits[fold["year"]][0]["training_event_ids"]
        assert fold["test_event_ids"] == expected_fits[fold["year"]][0]["evaluation_event_ids"]
        assert all(len(fit[field]) == 71 for fit in fold["fits"]
                   for field in ("training_matrix_id", "prediction_matrix_id"))
    for horizon in result["per_horizon"]:
        assert {k: v for k, v in horizon.items() if k != "horizon_sessions"} == pytest.approx(
            expected_history_metrics((a, p) for (_, h), (_, a, p) in expected.items() if h == horizon["horizon_sessions"]))
    assert result["result_id"] == _sha({k: v for k, v in result.items() if k != "result_id"})
    assert result["stored_inputs"] == {"inputs_id": stored["inputs_id"],
        "feature_snapshot": to_document(source.feature_snapshot), "columns": [
            "event_id", "context_id", "horizon_sessions", "event_date", "cutoff_id",
            "source_session_ceiling", "history_status", "history_json", *FEATURES],
        "row_keys": [[row[key] for key in ("event_id", "context_id", "horizon_sessions")]
                     for row in stored["rows"]],
        "objects": [{"fragment_id": record.fragment_id, "object_id": record.object_ref.object_id,
            "content_hash": record.object_ref.content_hash} for record in source.repository.fragment_records(
                source.feature_snapshot, "research_magnitude_history")]}
    assert result["fit_recipe"] == source.registration.document["magnitude_execution"]["fit_recipe"]
    assert len(result["result_id"]) == 71 and json.loads(json.dumps(result, allow_nan=False)) == result
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_sufficient_history_consumes_all_ema_columns_for_both_issuers(tmp_path, monkeypatch):
    events, moves = magnitude_dataset()
    days = ("2022-06-03", "2022-06-06", "2022-06-07", "2022-06-08", "2022-06-09",
            "2022-06-10", "2022-06-13", "2022-06-14", "2022-06-15", "2022-06-16", "2022-06-17")
    for ticker, scale in (("SYNTH_A", 1), ("SYNTH_B", 3)):
        for index, day in enumerate(days):
            event = history_event(day, ticker=ticker)
            events.append(event)
            moves.append(outcome_move(event, scale * (-1 if index % 2 else 1) * (index + 2)))
    authored = {(row["ticker"], row["event_date"]): row["realized_move_pct"] for row in moves}
    identities = {row["event_id"]: (row["ticker"], row["event_date"].date().isoformat()) for row in events}
    def row_for(event_id, horizon):
        issuer, day = identities[event_id]
        before = "2024-06-12" if day == "2024-06-17" and horizon != 1 else day
        signed = [Fraction(value) for (ticker, date), value in sorted(authored.items())
                  if ticker == issuer and date < before]
        absolute = list(map(abs, signed))
        values = [len(signed), float(sum(signed) / len(signed)), float(sum(absolute) / len(signed))]
        for span in (2, 4, 8, 12):
            alpha = Fraction(2, span + 1)
            for sequence in (signed, absolute):
                value = sequence[0]
                for observed in sequence[1:]:
                    value += alpha * (observed - value)
                values.append(float(value))
        return values
    source = magnitude_source(tmp_path, register=False, events=events, moves=moves, folds=("2024",))
    try:
        _register(source)
        forbid_history_rebuild(monkeypatch)
        calls = _fit_spy(monkeypatch)
        result = model.prediction_result(source.repository, source.registration)
        train_ids = history_event_ids(source, DATES[:2])
        test_ids = history_event_ids(source, DATES[2:4])
        expected = {}
        for call, horizon in zip(calls, HORIZONS):
            train = [row_for(key, horizon) for key in train_ids]
            test = [row_for(key, horizon) for key in test_ids]
            labels = [abs(authored[identities[key]]) for key in train_ids]
            assert np.isfinite(train).all() and all(row[0] >= 12 for row in train)
            np.testing.assert_allclose(call[1], train, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(call[3], test, rtol=1e-13, atol=1e-13)
            for key, prediction in zip(test_ids, _ridge_oracle(train, labels, test)):
                expected[key, horizon] = prediction
        assert len(calls) == 3
        for row in result["predictions"]:
            assert row["predicted_absolute_move_pct"] == pytest.approx(
                expected[row["event_id"], row["horizon_sessions"]], abs=1e-10)
    finally:
        source.conn.close()


def test_forecasts_are_detached_before_any_evaluation_label_opens(tmp_path, monkeypatch):
    source = magnitude_source(tmp_path, register=False, folds=("2024",))
    try:
        _register(source)
        baseline = model.prediction_result(source.repository, source.registration)
        read, qualify, evaluate, fit = (model.read_stored, model.qualify_magnitude_training_targets,
            model.load_magnitude_targets, training.fit_magnitude_predictions)
        stored_rows, frames, outputs, order = [], [], [], []
        def read_features(*args, **kwargs):
            value = read(*args, **kwargs)
            stored_rows.extend(value["rows"])
            return value
        def training_labels(*args, **kwargs):
            order.append("train")
            value = qualify(*args, **kwargs)
            frames.append(value)
            return value
        def fit_model(*args):
            order.append("fit")
            value = fit(*args)
            outputs.append(value)
            return value
        def evaluation_labels(*args, **kwargs):
            assert order == ["train", "fit", "fit", "fit"]
            order.append("evaluation")
            frames[0].loc[:, "absolute_move_pct"] = 1e9
            for row in stored_rows:
                row.update({name: 1e9 for name in FEATURES})
            for output in outputs:
                output[:] = [1e9] * len(output)
            return evaluate(*args, **kwargs)
        monkeypatch.setattr(model, "read_stored", read_features)
        monkeypatch.setattr(model, "qualify_magnitude_training_targets", training_labels)
        monkeypatch.setattr(model, "load_magnitude_targets", evaluation_labels)
        monkeypatch.setattr(training, "fit_magnitude_predictions", fit_model)
        actual = model.prediction_result(source.repository, source.registration)
        assert actual["predictions"] == baseline["predictions"]
        assert actual["folds"] == baseline["folds"]
    finally:
        source.conn.close()


def test_evaluation_feature_extremes_cannot_enter_training_matrices_or_preprocessing(source, monkeypatch):
    baseline = model.prediction_result(source.repository, source.registration)
    original = model.read_stored
    final_ids = history_event_ids(source, DATES[4:])
    def extreme(*args, **kwargs):
        value = original(*args, **kwargs)
        for row in value["rows"]:
            if row["event_id"] in final_ids:
                row.update({name: 1e6 * (index + 1) for index, name in enumerate(FEATURES)})
        return value
    monkeypatch.setattr(model, "read_stored", extreme)
    calls = _fit_spy(monkeypatch)
    changed = model.prediction_result(source.repository, source.registration)
    assert changed["predictions"][:12] == baseline["predictions"][:12]
    for call, (year, horizon, _, train, labels, test_ids, _, _) in zip(calls, _oracle_fits(source)):
        np.testing.assert_allclose(call[1], np.asarray(train, dtype=float), equal_nan=True)
        np.testing.assert_array_equal(call[2], labels)
        expected = _ridge_oracle(train, labels, call[3])
        actual = {row["event_id"]: row["predicted_absolute_move_pct"] for row in changed["predictions"]
                  if row["fold"] == year and row["horizon_sessions"] == horizon}
        assert [actual[key] for key in test_ids] == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["history_ridge", "history_hgb"])
def test_empty_histories_keep_rows_and_late_nan_labels_are_omitted_before_fit(tmp_path, monkeypatch, kind):
    dates = ("2023-01-03", "2023-06-02", "2023-06-05", "2023-06-06", "2023-06-07", "2024-06-12")
    events = [history_event(day, ticker=f"UNIQUE_{index}") for index, day in enumerate(dates)]
    moves = [outcome_move(event, value) for event, value in zip(events, (-2., 8., math.nan, math.nan, math.nan, 10.))]
    moves[2]["available_as_of_date"] = "2024-01-01"
    moves[3].update(skipped=True, available_as_of_date=None)
    moves[4]["available_as_of_date"] = None
    source = magnitude_source(tmp_path, register=False, events=events, moves=moves, folds=("2024",))
    try:
        _register(source, kind)
        calls = _fit_spy(monkeypatch)
        result = model.prediction_result(source.repository, source.registration)
        assert len(calls) == 3
        for candidate, train, labels, test in calls:
            assert candidate == kind and train.shape == (2, 11) and test.shape == (1, 11)
            np.testing.assert_array_equal(labels, [2., 8.])
            assert not train[:, 0].any() and np.isnan(train[:, 1:]).all()
            assert test[0, 0] == 0 and np.isnan(test[:, 1:]).all()
        fold = result["folds"][0]
        assert fold["train_event_ids"] == [event["event_id"] for event in events[:2]]
        assert fold["training_inputs"]["omitted_labels"] == [
            {"event_id": event["event_id"], "reason": reason} for event, reason in zip(events[2:5],
                ("not_available_before_fit", "skipped", "unavailable"))]
        assert len(result["predictions"]) == 3
        assert all(row["predicted_absolute_move_pct"] == pytest.approx(5.) for row in result["predictions"])
        assert all(row["source"] == kind for row in result["predictions"])
    finally:
        source.conn.close()


def test_protected_later_physical_object_refuses_before_features_labels_and_fits(tmp_path, monkeypatch):
    events, moves = magnitude_dataset()
    moves.append(outcome_move(history_event("2025-01-08", ticker="SYNTH_A"), None))
    source = magnitude_source(tmp_path, register=False, events=events, moves=moves,
                     mixed_days=("2025-01-07", "2025-01-08"))
    try:
        _register(source)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
        for name in ("read_stored", "qualify_magnitude_training_targets", "load_magnitude_targets"):
            monkeypatch.setattr(model, name, forbid_access)
        monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
        with refuses(DataError, "HOLDOUT_ACCESS_DENIED"):
            model.prediction_result(source.repository, source.registration)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
        monkeypatch.setattr(model, "preflight_magnitude_targets", lambda *args, **kwargs: {})
        with pytest.raises(AssertionError, match="forbidden access"):
            model.prediction_result(source.repository, source.registration)
    finally:
        source.conn.close()


def _invoke(source, kind, capsys, action, *extra):
    arguments = history_model_args(source, action, *extra)
    arguments[arguments.index("--magnitude-model") + 1] = kind
    assert cli.main(arguments) == 0
    captured = capsys.readouterr()
    err_lines = captured.err.splitlines()
    assert all(line.startswith("  [") for line in err_lines) and "Traceback" not in captured.out
    return json.loads(captured.out)


@pytest.mark.parametrize("kind", ["history_ridge", "history_hgb"])
def test_cli_register_smoke_record_and_replay_never_refits_or_duplicates_ledger(tmp_path, monkeypatch, capsys, kind):
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    source = magnitude_source(tmp_path, register=False)
    try:
        with monkeypatch.context() as patch:
            forbid_history_model_payloads(patch)
            patch.setattr(training, "fit_magnitude_predictions", forbid_access)
            patch.setattr(training, "fit_magnitude_predictions", forbid_access)
            poison_outcomes(patch, source.repository, source.snapshot, (), all_outcomes=True)
            registered = _invoke(source, kind, capsys, "register")
            assert _invoke(source, kind, capsys, "register") == registered
        binding = read_outcome_document(source, baseline_evidence(source)["native_registration"])
        assert binding["schema_version"] == "native_experiment_registration.v1.5"
        assert binding["magnitude_execution"]["features"] == FEATURES
        assert binding["magnitude_execution"]["fit_recipe"]["model"] == kind
        assert registered["registered"] == _sha(binding)
        assert {"experiments/magnitude_fitted.py", "engine/v2/models/training/magnitude.py"} <= binding["source_closure"].keys()
        assert not {"native_smoke", "native_outcome", "attempted_variants"} & baseline_evidence(source).keys()
        forbid_history_rebuild(monkeypatch)
        poison_outcomes(monkeypatch, source.repository, source.snapshot, ("2022-06-02", "2023-04-03", "2025-12-03"))
        with monkeypatch.context() as patch:
            patch.setattr(outcomes, "ledger_append", forbid_access)
            smoke = _invoke(source, kind, capsys, "run", "--no-ledger")
        result = result_from_report(smoke["report"])
        assert result["schema_version"] == "native_magnitude_fitted_result.v1"
        assert result["model"] == kind and result["forecast_count"] == 18
        expected = {}
        for _, horizon, _, train, labels, ids, test, actual in _oracle_fits(source):
            # Both tiny HGB folds are below the fixed 20-samples-per-leaf limit.
            forecasts = (_ridge_oracle(train, labels, test) if kind == "history_ridge"
                         else [sum(labels) / len(labels)] * len(ids))
            expected.update({(key, horizon): (observed, forecast)
                             for key, observed, forecast in zip(ids, actual, forecasts)})
        assert result["metrics"] == pytest.approx(expected_history_metrics(expected.values()))
        for row in result["predictions"]:
            assert row["predicted_absolute_move_pct"] == pytest.approx(
                expected[row["event_id"], row["horizon_sessions"]][1])
        assert smoke["recording_mode"] == "smoke" and not list(tmp_path.rglob("*.csv"))
        assert read_outcome_document(source, baseline_evidence(source)["native_smoke"])["ledger_destination"] is None
        ledger = tmp_path / "ledger.csv"
        set_outcome_clock(monkeypatch, "2026-01-02")
        recorded = _invoke(source, kind, capsys, "run", "--ledger", ledger)
        assert result_from_report(recorded["report"]) == result
        evidence = baseline_evidence(source)
        outcome = read_outcome_document(source, evidence["native_outcome"])
        assert read_ledger_rows(ledger) == [outcome["ledger_row"]]
        assert outcome["attempted_variants"] == 1 and outcome["failure_code"] is None
        assert read_ledger_rows(ledger)[0]["date"] == "2026-01-02" and read_ledger_rows(ledger)[0]["stage"] == "ran"
        assert read_outcome_document(source, evidence["native_outcome_receipt"])["ledger_row_hash"] == _sha(outcome["ledger_row"])
        before = catalog_rows(source.conn), artifact_object_bytes(source.store), ledger.read_bytes()
        forbid_history_model_payloads(monkeypatch)
        monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
        monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
        set_outcome_clock(monkeypatch, "2026-02-10")
        smoke_replay = _invoke(source, kind, capsys, "run", "--no-ledger")
        assert smoke_replay["record_export"]["status"] == "unchanged"
        assert smoke["record_export"]["status"] == "written"
        assert {key: value for key, value in smoke_replay.items() if key != "record_export"} == {
            key: value for key, value in smoke.items() if key != "record_export"}
        recorded_replay = _invoke(source, kind, capsys, "run", "--ledger", ledger)
        assert {key: value for key, value in recorded_replay.items() if key != "record_export"} == {
            key: value for key, value in recorded.items() if key != "record_export"}
        assert recorded["record_export"]["status"] == "written"
        assert recorded_replay["record_export"]["status"] == "unchanged"
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store), ledger.read_bytes()) == before
    finally:
        source.conn.close()


@pytest.mark.parametrize("changes", [
    {"magnitude_fit_model": "history_hgb"}, {"magnitude_fit_model": None},
    {"magnitude_feature_snapshot_id": "other-feature"}, {"magnitude_history_start": "2022-02-01"},
])
def test_request_drift_refuses_before_scans_fitting_and_effects(source, monkeypatch, changes):
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", forbid_access)
    monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
    for operation in (native.require_native_registration, inputs.admit_magnitude_inputs):
        with refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
            operation(source.conn, source.store, source.repository, source.spec,
                **{**history_binding(source, magnitude_fit_model="history_ridge"), **changes})
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("field", ["source", "alpha", "boolean-alpha", "boolean-threads",
    "extra", "preprocessing", "features", "schema"])
def test_changed_recipe_source_and_closed_schema_refuse_before_payloads(source, monkeypatch, field):
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", forbid_access)
    monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
    if field == "source":
        original = native.source_closure
        monkeypatch.setattr(native, "source_closure", lambda *a, **k: {
            **original(*a, **k), "engine/v2/models/training/magnitude.py": "sha256:" + "f" * 64})
        with refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
            inputs.admit_magnitude_inputs(source.conn, source.store, source.repository, source.spec,
                **history_binding(source, magnitude_fit_model="history_ridge"))
    else:
        document = source.registration.document
        execution = document["magnitude_execution"]
        if field == "alpha":
            execution["fit_recipe"]["estimator"]["params"]["alpha"] = 2.
        elif field == "boolean-alpha":
            execution["fit_recipe"]["estimator"]["params"]["alpha"] = True
        elif field == "boolean-threads":
            execution["fit_recipe"]["numerical_threads"] = True
        elif field == "extra":
            execution["fit_recipe"]["unregistered_override"] = 1
        elif field == "preprocessing":
            execution["fit_recipe"]["preprocessing"]["imputer"]["strategy"] = "mean"
        elif field == "features":
            execution["features"].reverse()
            execution["fit_recipe"]["features"].reverse()
        else:
            document["schema_version"] = "native_experiment_registration.v1.4"
        changed = native.NativeRegistration(source.registration.run_id, canonical_json(document).encode())
        with refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
            model.prediction_result(source.repository, changed)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("kind", ["history_ridge", "history_hgb"])
@pytest.mark.parametrize("action", ["register", "run"])
@pytest.mark.parametrize("missing", ["--target", "--magnitude-training-start", "--magnitude-labels-before",
    "--magnitude-feature-snapshot-id", "--magnitude-history-start"])
def test_cli_missing_required_fitted_inputs_refuses_before_spec_or_store(monkeypatch, capsys, kind, action, missing):
    monkeypatch.setattr(cli.foundation, "ArtifactStore", forbid_access)
    monkeypatch.setattr(cli, "connect", forbid_access)
    monkeypatch.setattr(Path, "read_text", forbid_access)
    options = {"--target": "absolute_move", "--magnitude-training-start": START,
        "--magnitude-labels-before": CUTOFF, "--magnitude-model": kind,
        "--magnitude-feature-snapshot-id": "unused", "--magnitude-history-start": HISTORY}
    options.pop(missing)
    with pytest.raises(SystemExit) as caught:
        cli.main([action, "--spec", "unused", "--catalog", "unused", "--store-root", "unused",
            "--as-of-month", MONTH, "--no-ledger", *[part for pair in options.items() for part in pair]])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert ": error:" in captured.err and not captured.out


@pytest.mark.parametrize("field", ["variant_id", "run_id", "provenance", "result_id", "schema_version"])
def test_renderer_rejects_rehashed_fitted_identity_or_schema_drift(source, field):
    result = model.prediction_result(source.repository, source.registration)
    report = model.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)
    assert json.loads(report.split("```json\n", 1)[1].split("\n```", 1)[0]) == result
    for text in ("history_ridge", "RMSE", "MAE", CUTOFF, source.feature_snapshot.snapshot_id):
        assert text in report
    result[field] = "native_magnitude_history_result.v1" if field == "schema_version" else "changed"
    if field != "result_id":
        result["result_id"] = _sha({key: value for key, value in result.items() if key != "result_id"})
    with refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        model.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)


@pytest.mark.parametrize("field", ["fit_recipe", "provenance"])
def test_renderer_rejects_boolean_recipe_even_with_valid_result_hash(source, field):
    result = model.prediction_result(source.repository, source.registration)
    # Python equality treats True and 1.0 as equal, but they are different sealed policies.
    result[field] = json.loads(json.dumps(result[field]))
    recipe = (result[field] if field == "fit_recipe" else result[field]["magnitude_execution"]["fit_recipe"])
    recipe["estimator"]["params"]["alpha"] = True
    result["result_id"] = _sha({key: value for key, value in result.items() if key != "result_id"})
    unchanged = "provenance" if field == "fit_recipe" else "fit_recipe"
    expected = (source.registration.document if unchanged == "provenance"
                else source.registration.document["magnitude_execution"]["fit_recipe"])
    assert canonical_json(result[unchanged]) == canonical_json(expected)
    with refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
        model.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)


@pytest.mark.parametrize("field,replacement", [
    ("model", "history_hgb"), ("features", list(reversed(FEATURES))),
    ("weighting", "equal_fold"), ("target", {"units": "decimal_return"}),
], ids=["model", "features", "weighting", "target"])
def test_renderer_rejects_each_rehashed_policy_field_without_provenance_drift(source, field, replacement):
    result = model.prediction_result(source.repository, source.registration)
    original = json.loads(json.dumps(result))
    assert result[field] != replacement
    result[field] = replacement
    result["result_id"] = _sha({key: value for key, value in result.items() if key != "result_id"})
    assert result["result_id"] != original["result_id"]
    assert canonical_json(result["provenance"]) == canonical_json(source.registration.document)
    assert {key: value for key, value in result.items() if key not in (field, "result_id")} == {
        key: value for key, value in original.items() if key not in (field, "result_id")}
    with pytest.raises(OpsError) as caught:
        model.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "history report differs from its registered result"


@pytest.mark.parametrize("api", ["prediction", "renderer"])
@pytest.mark.parametrize("defect", ["missing", "null", "list", "string", "unknown-schema",
    "null-schema", "list-schema", "empty-v1.4", "empty-v1.5"])
def test_malformed_registration_is_typed_before_report_reads_or_fitting(source, monkeypatch, api, defect):
    result = model.prediction_result(source.repository, source.registration) if api == "renderer" else None
    document = source.registration.document
    if defect == "missing":
        document.pop("schema_version")
    elif defect in ("null", "list", "string"):
        document = {"null": None, "list": [], "string": "invalid"}[defect]
    elif defect.startswith("empty-"):
        document = {"schema_version": "native_experiment_registration." + defect.removeprefix("empty-")}
    else:
        document["schema_version"] = {"unknown-schema": "unknown", "null-schema": None, "list-schema": []}[defect]
    malformed = native.NativeRegistration(source.registration.run_id, canonical_json(document).encode())
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", forbid_access)
    monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
    for name in ("history_binding", "read_stored", "preflight_magnitude_targets",
                 "qualify_magnitude_training_targets", "load_magnitude_targets"):
        monkeypatch.setattr(model, name, forbid_access)
    with pytest.raises(OpsError) as caught:
        if api == "prediction":
            model.prediction_result(source.repository, malformed)
        else:
            model.render_prediction_report(malformed, result, spec=source.spec, no_ledger=True)
    message = ("fitted magnitude execution requires a sealed v1.5 plan" if defect == "empty-v1.5"
               else "history execution requires a sealed v1.4 plan")
    assert to_document(caught.value.problem) == {
        "code": "EXPERIMENT_IDENTITY_CONFLICT", "category": "validation", "retryable": False,
        "message": message, "stage": None, "trace_id": None, "dependency_refs": [],
        "retry_after_seconds": None, "diagnostic_ref": None, "details": {}, "schema_version": "problem.v1.0"}
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_catalog_backed_fitted_registration_has_no_final_holdout_authority(source, monkeypatch):
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(source.repository, "scan", forbid_access)
    monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
    with refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
        native.prepare_final_holdout_request(source.conn, source.store, source.registration)
    assert source.conn.execute("SELECT COUNT(*) FROM final_holdout_authorizations").fetchone()[0] == 0
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("failure", ["exception", "wrong-shape", "nonfinite", "negative"])
def test_failed_fit_never_falls_back_or_returns_partial_success(source, monkeypatch, failure):
    calls = []
    original = training.fit_magnitude_predictions
    def broken(kind, train, labels, test):
        calls.append(kind)
        if len(calls) == 1:
            return original(kind, train, labels, test)
        if failure == "exception":
            raise ValueError("synthetic estimator failure")
        return [] if failure == "wrong-shape" else [math.nan if failure == "nonfinite" else -1.] * len(test)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(training, "fit_magnitude_predictions", broken)
    with refuses(OpsError, "EXPERIMENT_VARIANT_FAILED"):
        model.prediction_result(source.repository, source.registration)
    assert len(calls) == 2
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert not list(source.root.rglob("REPORT.md")) and not list(source.root.rglob("*.csv"))


@pytest.mark.parametrize("guard", [RuntimeFitForbidden, LegacyRuntimeFitForbidden])
def test_no_fit_guard_exception_propagates_unchanged(source, monkeypatch, guard):
    error = guard("no-fit guard active")
    def forbidden(kind, train, labels, test):
        raise error
    monkeypatch.setattr(training, "fit_magnitude_predictions", forbidden)
    with pytest.raises(guard) as caught:
        model.prediction_result(source.repository, source.registration)
    assert caught.value is error


@pytest.mark.parametrize("kind", ["empty-training", "late-evaluation", "nonfinite-evaluation"])
def test_invalid_targets_refuse_whole_fitted_assessment(tmp_path, monkeypatch, kind):
    events, moves = magnitude_dataset()
    if kind == "empty-training":
        for row in moves:
            if row["event_date"].startswith("2023"):
                row.update(skipped=True, available_as_of_date=None, realized_move_pct=None)
    else:
        row = next(row for row in moves if row["event_date"] == "2025-01-07" and row["ticker"] == "SYNTH_B")
        row.update({"available_as_of_date": CUTOFF} if kind == "late-evaluation"
                   else {"realized_move_pct": math.nan})
    source = magnitude_source(tmp_path, register=False, events=events, moves=moves)
    try:
        _register(source)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        calls = _fit_spy(monkeypatch)
        with refuses(DataError, "POPULATION_COLLAPSED" if kind == "empty-training" else "EXPERIMENT_VARIANT_FAILED"):
            model.prediction_result(source.repository, source.registration)
        assert len(calls) == (0 if kind == "empty-training" else 6)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
        assert not list(tmp_path.rglob("REPORT.md")) and not list(tmp_path.rglob("*.csv"))
    finally:
        source.conn.close()


def test_omitted_fit_model_preserves_exact_v14_registration_and_admission(tmp_path, monkeypatch):
    source = magnitude_source(tmp_path)
    try:
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        forbid_history_model_payloads(monkeypatch)
        monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
        for operation in (native.register_native, native.require_native_registration):
            unchanged = operation(source.conn, source.store, source.repository, source.spec,
                **history_binding(source, magnitude_fit_model=None))
            assert unchanged == source.registration
            assert unchanged.document["schema_version"] == "native_experiment_registration.v1.4"
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()


@pytest.mark.parametrize("changes", [
    {"magnitude_fit_model": "history_search"}, {"magnitude_fit_model": True},
    {"magnitude_training_start": None}, {"magnitude_labels_before": None},
    {"magnitude_feature_snapshot_id": None}, {"magnitude_history_start": None},
])
def test_library_invalid_fitted_request_refuses_before_repository_admission(tmp_path, monkeypatch, changes):
    source = magnitude_source(tmp_path, register=False, build=False)
    try:
        arguments = {"code_root": Path(__file__).resolve().parents[3], "as_of_month": MONTH,
            "magnitude_fit_model": "history_ridge", "magnitude_training_start": START,
            "magnitude_labels_before": CUTOFF, "magnitude_feature_snapshot_id": "unused",
            "magnitude_history_start": HISTORY, **changes}
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        monkeypatch.setattr(native, "source_closure", forbid_access)
        for name in ("resolve_pinned", "resolve", "scan"):
            monkeypatch.setattr(source.repository, name, forbid_access)
        monkeypatch.setattr(training, "fit_magnitude_predictions", forbid_access)
        for operation in (native.register_native, native.require_native_registration):
            with refuses(OpsError, "INVALID_EXPERIMENT_SPEC"):
                operation(source.conn, source.store, source.repository, source.spec, **arguments)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()
