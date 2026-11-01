"""SIZE v1.4 candidate fits use only admitted fold training rows and count abstentions."""
import numpy as np
import pandas as pd
import pytest

from experiments import size_comparator as comparator
from experiments.magnitude_search_evidence import fit_evidence as shared_fit_evidence
from experiments.magnitude_search_evidence import normalize_forecasts


def _row(event_id, *, n_prior=4, missing=False):
    return {
        "event_id": event_id, "horizon_sessions": 5, "context_id": "ctx-" + event_id,
        "n_prior": n_prior, "current_quote_status": "absent", "has_implied_quote": 0,
        "mean_prior_abs_move": 1.0, "abs_dist_ema": 1.0, "ema12r_abs": 1.0,
        "mean_prior_move": 1.0, "signed_streak": 1.0, "dist_high": 1.0,
        "dist_ema": 1.0, "spy_vol20": 1.0, "spy_dd252": 1.0,
        "mean_prior_or_implied": 1.0, "implied_move": None if missing else 5.0,
        "rvol30": 30.0, "mcap_log": 1.0,
    }


def test_inner_fold_fit_uses_only_pre_cutoff_training_rows_and_counts_abstentions(monkeypatch):
    training_ids = [f"train-{index}" for index in range(500)]
    evaluation_ids = ["low-prior", "missing-input", "complete"]
    features = [_row(event_id) for event_id in training_ids]
    features.extend([_row("low-prior", n_prior=3), _row("missing-input", missing=True), _row("complete")])
    train = pd.DataFrame({"event_id": training_ids, "absolute_move_pct": [10.0] * 500,
                          "label_end": ["2024-12-31"] * 500})
    fold = {"evaluation_event_ids": evaluation_ids,
            "training_labels_before": "2025-01-01"}
    seen = {}

    class Predictor:
        def predict(self, matrix):
            return np.full(len(matrix), 12.0)

    def fit_spy(recipe, train_features, train_labels):
        seen["labels"] = np.asarray(train_labels).copy()
        seen["matrix"] = np.asarray(train_features).copy()
        return Predictor()

    monkeypatch.setattr(comparator, "fit_recipe_estimator", fit_spy)
    assert all(label_end < fold["training_labels_before"] for label_end in train["label_end"])
    plan = {"horizons": [5]}
    execution = {"schema_version": "size_candidate_execution.v1", "features": comparator.COLUMNS,
                 "model": "size_v1_4", "fit_recipe": comparator.recipe()}
    inputs = {"training": train.to_dict("records"), "features": features}
    expected = shared_fit_evidence(inputs, fold, plan, execution)
    captured = []
    result = comparator.forecasts(
        {(row["event_id"], 5): row for row in features}, train, fold,
        plan, execution, progress=lambda *pair: captured.append(pair))

    assert captured == [("started", expected[0]), ("completed", expected[0])]
    assert seen["labels"].tolist() == [10.0] * 500
    assert seen["matrix"].shape == (500, len(comparator.FEATURES))
    assert result["fit_details"][0]["training_event_ids"] == training_ids
    assert result["fit_details"][0]["training_abstention_counts"] == {}
    assert result["fit_details"][0]["abstention_counts"] == {
        "insufficient_history": 1, "unavailable_input": 1}
    forecasts = {row["event_id"]: row for row in result["forecasts"]}
    assert forecasts["low-prior"]["predicted_absolute_move_pct"] is None
    assert forecasts["low-prior"]["abstention"] == "insufficient_history"
    assert forecasts["missing-input"]["predicted_absolute_move_pct"] is None
    assert forecasts["missing-input"]["abstention"] == "unavailable_input"
    assert forecasts["complete"]["predicted_absolute_move_pct"] == 12.0


def test_normalize_forecasts_allows_incomplete_nonempty_subset_but_strict_path_requires_complete():
    fold = {"evaluation_event_ids": ["a", "b"]}
    rows = [
        {"event_id": "a", "horizon_sessions": 5, "predicted_absolute_move_pct": -2.5},
        {"event_id": "b", "horizon_sessions": 5, "predicted_absolute_move_pct": None},
    ]

    assert normalize_forecasts(rows, fold, [5], allow_incomplete=True) == [
        {"event_id": "a", "horizon_sessions": 5, "predicted_absolute_move_pct": -2.5}]

    with pytest.raises(ValueError):
        normalize_forecasts(rows, fold, [5])

    with pytest.raises(ValueError):
        normalize_forecasts(rows[:1], fold, [5], allow_incomplete=True)


def test_insufficient_training_abstentions_are_counted():
    training_ids = [f"train-{index}" for index in range(499)]
    features = [_row(event_id) for event_id in training_ids + ["held-out"]]
    train = pd.DataFrame({"event_id": training_ids, "absolute_move_pct": [10.0] * 499})
    fold = {"evaluation_event_ids": ["held-out"],
            "training_labels_before": "2025-01-01"}
    plan = {"horizons": [5]}
    execution = {"schema_version": "size_candidate_execution.v1", "features": comparator.COLUMNS,
                 "model": "size_v1_4", "fit_recipe": comparator.recipe()}
    result = comparator.forecasts(
        {(row["event_id"], 5): row for row in features}, train, fold,
        plan, execution, progress=lambda *pair: None)

    assert result["fit_details"][0]["abstention_counts"] == {"insufficient_training": 1}
    forecasts = {row["event_id"]: row for row in result["forecasts"]}
    assert forecasts["held-out"]["predicted_absolute_move_pct"] is None
    assert forecasts["held-out"]["abstention"] == "insufficient_training"
