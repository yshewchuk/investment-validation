"""Tiny real numeric fits through immutable CAS results into causal OOF controls."""
from copy import deepcopy

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.research.magnitude_distribution import recipe as output_recipe
from engine.v2.research.magnitude_distribution_math import mean
from experiments import distribution_fitted
from experiments.distribution_candidate_fit import fit_candidate
from experiments.distribution_candidate_recipe import trial_identity
from experiments.distribution_oof import control_for_prediction, residual_pool
from tests.v2.research.test_distribution_fitted import case as fitted_case


@pytest.mark.parametrize("horizon", [5, 3, 1])
def test_actual_fit_receipts_feed_strict_prefix_controls(tmp_path, monkeypatch, horizon):
    case = fitted_case.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(distribution_fitted, "fit_candidate", fit_candidate)
    base = [row["event_id"] for row in case.inputs["training"]]
    labels = {row["event_id"]: row["absolute_move_pct"] for row in case.inputs["training"]}
    labels.update(E1=6.0, E2=8.0)
    folds = [{"year": year, "training_labels_before": year + "-01-01",
              "evaluation_labels_before": str(int(year) + 1) + "-01-01",
              "training_candidate_event_ids": base + before, "evaluation_event_ids": [event]}
             for year, before, event in [("2021", [], "E1"), ("2022", ["E1"], "E2"),
                                         ("2023", ["E1", "E2"], "Z")]]
    plan = {"plan_id": content_hash("fixed plan"), "recipe": case.recipe,
            "context_id": content_hash("context"), "budget": {"budget_id": content_hash("budget")}}

    def context(role, outer, inner):
        assert outer == "2023"
        fold = next(fold for fold in folds if fold["year"] == (inner if role == "inner" else outer))
        train = fold["training_candidate_event_ids"]
        inputs = {"training": [{"event_id": event, "absolute_move_pct": labels[event]} for event in train],
            "features": [{"event_id": event, "context_id": plan["context_id"], "horizon_sessions": h,
                          **dict.fromkeys(case.columns, 0.0)}
                         for event in train + fold["evaluation_event_ids"] for h in (5, 3, 1)]}
        fold_id = content_hash({"fold": fold, "role": role, "outer_year": outer})
        inputs_id = content_hash(inputs)
        admission = {"inputs_id": inputs_id, "fold_id": fold_id, "trials": [trial_identity(case.recipe,
            context_id=plan["context_id"], fold_id=fold_id, inputs_id=inputs_id,
            budget_id=plan["budget"]["budget_id"], role=role, horizon=h) for h in (5, 3, 1)]}
        return deepcopy({"plan": plan, "native_plan": {"folds": [{"outer": folds[-1], "inner": folds[:2]}]},
                         "fold": fold, "inputs": inputs, "admission": admission})

    case.trials.read_fit_context = context
    first = case.fits.fit("inner", "2023", horizon, "2021")
    second = case.fits.fit("inner", "2023", horizon, "2022")
    outer = case.fits.fit("outer", "2023", horizon)
    with no_fit_guard():
        assert residual_pool(case.fits, "inner", "2023", horizon, "2021")["point_status"] == "no_earlier_inner"
        earlier = residual_pool(case.fits, "inner", "2023", horizon, "2022")
        pool = residual_pool(case.fits, "outer", "2023", horizon)
        assert [entry["result_id"] for entry in pool["sources"]] == [first["receipt"]["result_id"], second["receipt"]["result_id"]]
        assert earlier["keys"] == [["E1", horizon]]
        assert pool["keys"] == [["E1", horizon], ["E2", horizon]]
        document = {**output_recipe(), "probabilities": outer["rows"][0]["probabilities"], "tail": outer["tail"]}
        controls = control_for_prediction(pool, document, outer["rows"][0]["point_prediction"])
        assert controls["histogram"]["anchored_mean"] == pytest.approx(mean(document))
        assert controls["calibration_identity"] == "uncalibrated"
        labels["E2"] = 80.0
        assert residual_pool(case.fits, "inner", "2023", horizon, "2022") == earlier
