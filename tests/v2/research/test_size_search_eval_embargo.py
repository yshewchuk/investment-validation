from copy import deepcopy

import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_evidence as evidence
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

    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "ambiguous-random ticker embargo removed every fold evaluation event"


@pytest.mark.parametrize(("cutoff", "before", "after"), [
    ("2024-12-24", "2024-12-23", "2024-12-25"),
    ("2024-12-31", "2024-12-30", "2025-01-01"),
])
def test_fold_binding_excludes_evaluation_at_cutoff_and_keeps_earlier_event(monkeypatch, cutoff, before, after):
    fold = _fold(["at_cutoff", "before_cutoff", "after_cutoff", "also_embargoed_at_cutoff"])
    fold["evaluation_labels_before"] = cutoff
    nested = [{"outer": fold, "inner": []}]
    monkeypatch.setattr(planner, "_nested", lambda *_args: deepcopy(nested))
    source = _binding()
    source["size_panel"]["context"]["target_events"].extend([
        {"event_id": "at_cutoff", "ticker": "OTHER", "label_end": cutoff},
        {"event_id": "before_cutoff", "ticker": "OTHER", "label_end": before},
        {"event_id": "after_cutoff", "ticker": "OTHER", "label_end": after},
        {"event_id": "also_embargoed_at_cutoff", "ticker": "SYNTH", "label_end": cutoff},
    ])

    result = binding._folds(source, {})[0]["outer"]

    assert result["evaluation_event_ids"] == ["before_cutoff"]
    assert result["evaluation_exclusions"] == [
        {"event_id": "at_cutoff", "reasons": ["label_end_at_or_after_cutoff"]},
        {"event_id": "after_cutoff", "reasons": ["label_end_at_or_after_cutoff"]},
        {"event_id": "also_embargoed_at_cutoff", "reasons": [
            "embargo_ticker", "label_end_at_or_after_cutoff"
        ]},
    ]


def test_fold_binding_refuses_when_cutoff_removes_every_evaluation(monkeypatch):
    fold = _fold(["at_cutoff"])
    fold["evaluation_labels_before"] = "2024-12-24"
    nested = [{"outer": fold, "inner": []}]
    monkeypatch.setattr(planner, "_nested", lambda *_args: deepcopy(nested))
    source = _binding()
    source["size_panel"]["context"]["target_events"].append(
        {"event_id": "at_cutoff", "ticker": "OTHER", "label_end": "2024-12-24"}
    )

    with pytest.raises(OpsError) as caught:
        binding._folds(source, {})

    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "SIZE fold has no eligible evaluation events"


@pytest.mark.parametrize("schema_version", [
    "magnitude_search_request.v2", "magnitude_search_request.v3"])
def test_repeated_inner_year_uses_shared_plan_v3_or_legacy_owner_cutoff(monkeypatch, schema_version):
    source = _binding()
    source["size_panel"]["context"]["target_events"].extend([
        {"event_id": "early_inner", "ticker": "OTHER", "label_end": "2023-01-01"},
        {"event_id": "between_cutoffs", "ticker": "OTHER", "label_end": "2024-06-01"},
    ])

    def owner(cutoff):
        outer = _fold(["evaluation"])
        outer["training_labels_before"] = cutoff
        inner = {**_fold(["early_inner", "between_cutoffs"]),
                 "year": "2023", "evaluation_labels_before": cutoff}
        return {"outer": outer, "inner": [inner]}

    nested = [owner("2024-01-01"), owner("2025-01-01")]
    monkeypatch.setattr(planner, "_nested", lambda *_args: deepcopy(nested))

    folds = binding._folds(source, {"schema_version": schema_version})

    assert [group["inner"][0]["evaluation_labels_before"] for group in folds] == [
        "2024-01-01", "2025-01-01"]
    if schema_version == "magnitude_search_request.v3":
        assert folds[0]["inner"][0]["evaluation_event_ids"] == ["early_inner"]
        assert folds[0]["inner"][0]["evaluation_exclusions"] == [
            {"event_id": "between_cutoffs", "reasons": ["label_end_at_or_after_cutoff"]}
        ]
        assert folds[1]["inner"][0]["evaluation_event_ids"] == ["early_inner", "between_cutoffs"]
        assert "evaluation_exclusions" not in folds[1]["inner"][0]
    else:
        for group in folds:
            inner = group["inner"][0]
            assert inner["evaluation_event_ids"] == ["early_inner"]
            assert inner["evaluation_exclusions"] == [
                {"event_id": "between_cutoffs", "reasons": ["label_end_at_or_after_cutoff"]}
            ]
        assert folds[0]["inner"][0]["evaluation_event_ids"] == folds[1]["inner"][0]["evaluation_event_ids"]
        binding._shared_inner(folds)


def test_fold_input_admission_explicit_plan_v2_uses_owner_specific_cutoff(monkeypatch):
    source = _binding()
    source["schema_version"] = "native_experiment_registration.v1.7"
    source["size_request"] = {"schema_version": "magnitude_search_request.v2"}
    source["snapshot"] = {}
    source["holdouts"] = {"holdout_as_of_month": "2025-01"}
    source["size_panel"]["context"]["target_events"].extend([
        {"event_id": "early_inner", "ticker": "OTHER", "label_end": "2023-01-01"},
        {"event_id": "between_cutoffs", "ticker": "OTHER", "label_end": "2024-06-01"},
    ])

    def owner(cutoff):
        outer = _fold(["evaluation"])
        outer["training_labels_before"] = cutoff
        inner = {**_fold(["early_inner", "between_cutoffs"]),
                 "year": "2023", "evaluation_labels_before": cutoff}
        return {"outer": outer, "inner": [inner]}

    nested = [owner("2024-01-01"), owner("2025-01-01")]
    monkeypatch.setattr(planner, "_nested", lambda *_args, **_kwargs: deepcopy(nested))

    legacy = binding._folds(source, source["size_request"], planner.PLAN_V2)
    default = binding._folds(source, source["size_request"])
    fold = legacy[1]["inner"][0]
    assert fold["evaluation_event_ids"] == ["early_inner", "between_cutoffs"]
    assert default[1]["inner"][0]["evaluation_event_ids"] == ["early_inner"]

    monkeypatch.setattr(binding, "verify_installation", lambda *_args: None)
    monkeypatch.setattr(binding, "from_document", lambda *_args: object())
    monkeypatch.setattr(binding, "_panel_columns", lambda *_args: [])
    monkeypatch.setattr(binding, "_stored_panel", lambda *_args: {
        "context": source["size_panel"]["context"], "rows": []})
    monkeypatch.setattr(binding, "preflight_magnitude_targets", lambda *_args, **_kwargs: {})
    train = pd.DataFrame([{"event_id": "training", "absolute_move_pct": 1.0}])
    train.attrs.update({"inputs_id": "train"})
    monkeypatch.setattr(binding, "qualify_magnitude_training_targets", lambda *_args, **_kwargs: train)

    inputs = evidence.admit_fold_inputs(source, source, fold,
        history_failure=RuntimeError("history changed"), plan={"schema_version": planner.PLAN_V2})

    assert inputs["stored_inputs_id"] == content_hash(source["size_panel"])
    with pytest.raises(OpsError) as caught:
        evidence.admit_fold_inputs(source, source, fold, history_failure=RuntimeError("history changed"))
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
