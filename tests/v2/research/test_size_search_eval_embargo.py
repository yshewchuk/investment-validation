from copy import deepcopy

import pytest

from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_plan as planner
from experiments import size_search_binding as binding


def _event(event_id, ticker):
    return {"event_id": event_id, "ticker": ticker, "label_end": "2020-01-01"}


def _fold(evaluation_ids):
    return {
        "year": "2024",
        "training_candidate_event_ids": ["training", "sibling"],
        "training_labels_before": "2021-01-01",
        "evaluation_event_ids": evaluation_ids,
        "evaluation_labels_before": "2025-01-01",
    }


def _binding():
    return {
        "size_panel": {"context": {
            "target_events": [
                _event("ambiguous", "SYNTH"), _event("sibling", "SYNTH"),
                _event("training", "OTHER"), _event("evaluation", "OTHER"),
            ],
            "eligibility_plan": {"random_windows": [
                {"ticker": "SYNTH", "start": "0001-01-01", "end": "9999-12-31"},
            ]},
        }},
        "magnitude_execution": {"evaluation_labels_before": "2025-01-01"},
    }


def test_fold_binding_embargoes_tickers_from_training_and_evaluation(monkeypatch):
    nested = [{"outer": _fold(["ambiguous", "sibling", "evaluation"]),
               "inner": [_fold(["sibling", "evaluation"])]}]
    monkeypatch.setattr(planner, "_nested", lambda *_args: deepcopy(nested))

    folds = binding._folds(_binding(), {})

    for fold in [folds[0]["outer"], *folds[0]["inner"]]:
        assert fold["training_candidate_event_ids"] == ["training"]
        assert fold["evaluation_event_ids"] == ["evaluation"]
        assert fold["evaluation_exclusions"] == [
            {"event_id": eid, "reasons": ["embargo_ticker"]}
            for eid in (["ambiguous", "sibling"] if fold is folds[0]["outer"] else ["sibling"])
        ]
        event_by_id = {event["event_id"]: event for event in _binding()["size_panel"]["context"]["target_events"]}
        assert all(event_by_id[eid]["ticker"] != "SYNTH" for eid in
                   fold["training_candidate_event_ids"] + fold["evaluation_event_ids"])


def test_fold_binding_refuses_when_embargo_removes_every_evaluation(monkeypatch):
    nested = [{"outer": _fold(["ambiguous", "sibling"]), "inner": []}]
    monkeypatch.setattr(planner, "_nested", lambda *_args: deepcopy(nested))

    with pytest.raises(OpsError) as caught:
        binding._folds(_binding(), {})

    assert caught.value.code == "POPULATION_COLLAPSED"
    assert caught.value.problem.message == "ambiguous-random ticker embargo removed every fold evaluation event"
