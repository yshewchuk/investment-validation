"""Pure causal folds, diagnostic poison, resource and concrete-request checks."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_folds import annual_fold
from experiments.magnitude_candidate_fit import fit_candidate
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_search_plan import (
    PLAN_V2,
    PLAN_V3,
    canonical_inner,
    cohort_key,
    trial_key,
)
from experiments.magnitude_search_request_versions import compile_request
from experiments.size_fit_diagnostics import evidence
from experiments.size_search import _contain
from experiments.size_search_binding import _folds, _shared_inner


def temporal():
    events = [{"event_id": name, "event_date": day,
               "forecasts": [{"source_session_ceiling": ceiling}]} for name, day, ceiling in (
        ("early", "2022-06-01", "2022-05-20"), ("late", "2022-12-30", "2022-12-20"),
        ("inner", "2023-06-01", "2023-05-20"), ("outer", "2024-06-01", "2024-05-20"))]
    targets = [{**event, "label_end": "2023-01-10" if event["event_id"] == "late" else
                event["event_date"][:8] + "02"} for event in events]
    binding = {"magnitude_plan": {"events": events, "evaluation_years": ["2024"],
                                  "folds": [annual_fold("2024", events)]},
        "magnitude_execution": {"evaluation_labels_before": "2025-01-01"},
        "size_panel": {"context": {"target_events": targets}}}
    return binding, {"inner_years": {"2024": ["2023"]}}


def test_actual_label_end_purges_training_even_when_event_precedes_fit():
    binding, request = temporal()
    original = deepcopy(binding)
    folds = _folds(binding, request)
    assert folds[0]["inner"][0]["training_candidate_event_ids"] == ["early"]
    assert folds[0]["outer"]["training_candidate_event_ids"] == ["early", "late", "inner"]
    assert binding == original


def test_actual_inner_label_end_must_precede_outer_fit():
    binding, request = temporal()
    binding["size_panel"]["context"]["target_events"][2]["label_end"] = "2024-01-01"
    with pytest.raises(OpsError, match="differs"):
        _folds(binding, request)


def test_missing_inner_year_is_not_silently_skipped():
    binding, request = temporal()
    request["inner_years"]["2024"] = ["2021"]
    with pytest.raises(OpsError):
        _folds(binding, request)


def diagnostic():
    recipe = candidate_recipe("elasticnet", {"alpha": 0.1, "l1_ratio": 0.25},
                              columns=["a", "b"], feature_dictionary_hash="sha256:" + "a" * 64)
    result = fit_candidate(recipe, [[1, None], [1, None]], [2, 2], [[1, None]], columns=["a", "b"])
    detail = {"horizon_sessions": 1, **{k: v for k, v in result.items() if k != "predictions"}}
    execution = {"schema_version": "size_candidate_execution.v1", "fit_recipe": recipe}
    return {"event": "succeeded", "diagnostics": [detail]}, execution


def test_constant_training_zero_iterations_remains_valid_converged_evidence():
    result, execution = diagnostic()
    assert result["diagnostics"][0]["n_iter"] == 0
    assert evidence(result, execution, [1]) == {"diagnostics": result["diagnostics"]}


@pytest.mark.parametrize("change", ["hash", "order", "selected", "converged", "nonfinite", "horizon", "extra"])
def test_fit_diagnostic_poison_refuses(change):
    result, execution = diagnostic()
    d = result["diagnostics"][0]
    if change == "hash":
        d["recipe_id"] = content_hash({})
    elif change == "order":
        d["coefficients"].reverse()
    elif change == "selected":
        d["selected_columns"] = ["not-selected"]
    elif change == "converged":
        d["converged"] = False
    elif change == "nonfinite":
        d["dual_gap"] = float("inf")
    elif change == "horizon":
        d["horizon_sessions"] = 3
    else:
        d["unbound"] = 1
    with pytest.raises(OpsError, match="diagnostics"):
        evidence(result, execution, [1])


def test_concrete_request_fixture_has_exact_schedule_and_is_not_a_data_reference():
    root = Path(__file__).resolve().parents[3]
    request = json.loads((root / "experiments/SUPER_SIZE/request.v2.example.json").read_text())
    schedule = compile_request(request)
    assert schedule["counts"]["candidates"] == 33
    assert schedule["counts"]["search_fits"] == 603
    assert schedule["counts"]["total_logical_fits"] == 612
    assert schedule["counts"]["total_component_fits"] == 621
    assert request["feature_snapshot_id"] == "snap_" + "1" * 32


def test_inherited_child_memory_conflict_refuses_before_process_limit_changes(monkeypatch):
    from tests.v2.research.test_magnitude_search_request import request
    value = request()
    value["limits"]["child"]["memory_bytes"] = 100
    value["limits"]["parent"]["memory_bytes"] = 99
    monkeypatch.setattr("resource.setrlimit", lambda *a: pytest.fail("changed process limits"))
    with pytest.raises(OpsError, match="inherited"):
        _contain(value)


def test_cli_has_no_ledger_mode_and_refuses_before_file_access(monkeypatch):
    from experiments import size_search
    monkeypatch.setattr(size_search, "_read_document", lambda *a: pytest.fail("read metadata"))
    with pytest.raises(SystemExit) as error:
        size_search.main(["run", "--spec", "spec", "--request", "request", "--catalog", "catalog",
                          "--store-root", "store", "--no-ledger", "--ledger", "forbidden"])
    assert error.value.code == 2


def test_post_import_source_drift_refuses_before_preparation_reads(monkeypatch):
    from experiments import size_search_binding as binding
    monkeypatch.setattr(binding, "_source_files", lambda: {"changed": "runner-or-registry"})
    monkeypatch.setattr(binding, "prepare_native_registration", lambda *a, **k: pytest.fail("source read"))
    with pytest.raises(OpsError, match="binding differs"):
        binding.prepare(None, None, {}, {}, scope="unused", training_start="2020-01-01",
                        labels_before="2025-01-01")


def test_stable_imported_source_identity_is_not_caller_selected():
    from experiments import size_search_binding as binding
    assert binding._sources() == binding._IMPORTED_SOURCES


def multi_year_temporal():
    years = ("2020", "2021", "2022", "2023", "2024", "2025")
    events = [{"event_id": "e" + year, "event_date": year + "-06-01",
               "forecasts": [{"source_session_ceiling": year + "-05-20"}]} for year in years]
    targets = [{**event, "ticker": "T" + event["event_id"],
                "label_end": event["event_date"][:8] + "10"} for event in events]
    binding = {"magnitude_plan": {"events": events, "evaluation_years": ["2024", "2025"],
                                  "folds": [annual_fold(year, events) for year in ("2024", "2025")]},
        "magnitude_execution": {"evaluation_labels_before": "2026-01-01"},
        "size_panel": {"context": {"target_events": targets}}}
    request = {"inner_years": {"2024": ["2021", "2022", "2023"],
                               "2025": ["2022", "2023", "2024"]}}
    return binding, request


def test_nested_multi_year_schedule_keeps_owner_specific_inner_folds():
    binding, request = multi_year_temporal()
    folds = _folds(binding, request)
    assert [group["outer"]["year"] for group in folds] == ["2024", "2025"]
    assert [[fold["year"] for fold in group["inner"]] for group in folds] == [
        ["2021", "2022", "2023"], ["2022", "2023", "2024"]]
    _shared_inner(folds)


def synthetic_plan(schema, candidates=("c1", "c2")):
    inner = {"training_candidate_event_ids": ["train"], "evaluation_event_ids": ["event"]}
    folds = [{"outer": {"year": "2024"}, "inner": [{**inner, "year": year}
              for year in ("2021", "2022", "2023")]},
             {"outer": {"year": "2025"}, "inner": [{**inner, "year": year}
              for year in ("2022", "2023", "2024")]}]
    return {"schema_version": schema, "plan_id": content_hash({"synthetic": schema}),
            "folds": folds, "candidates": [{"candidate_id": value} for value in candidates]}


def test_v3_shared_owner_and_exact_trial_cohort_identities():
    plan = synthetic_plan(PLAN_V3)
    owner, fold = canonical_inner(plan, "2025", "2023")
    assert owner == "2024" and fold is plan["folds"][0]["inner"][2]
    assert canonical_inner(plan, "2024", "2023")[0] == owner
    assert canonical_inner(plan, "2025", "2024")[0] == "2025"
    assert trial_key(plan, "2025", "2023", "c1") == content_hash([plan["plan_id"], "2023", "c1"])
    assert trial_key(plan, "2024", "2023", "c1") == trial_key(plan, "2025", "2023", "c1")
    assert cohort_key(plan, "2025", "2023") == content_hash([plan["plan_id"], "2023"])
    assert cohort_key(plan, "2024", "2023") == cohort_key(plan, "2025", "2023")
    assert trial_key(plan, "2025", "2023", "c1") != trial_key(plan, "2025", "2023", "c2")
    assert trial_key(plan, "2025", "2023", "c1") != trial_key(plan, "2025", "2024", "c1")
    assert cohort_key(plan, "2025", "2023") != cohort_key(plan, "2025", "2024")


def test_v2_trial_and_cohort_identities_remain_outer_scoped():
    plan = synthetic_plan(PLAN_V2)
    owner, fold = canonical_inner(plan, "2025", "2023")
    assert owner == "2025" and fold is plan["folds"][1]["inner"][1]
    assert trial_key(plan, "2024", "2023", "c1") == content_hash(
        [plan["plan_id"], "2024", "2023", "c1"])
    assert cohort_key(plan, "2024", "2023") == content_hash([plan["plan_id"], "2024", "2023"])
    assert trial_key(plan, "2024", "2023", "c1") != trial_key(plan, "2025", "2023", "c1")
    assert cohort_key(plan, "2024", "2023") != cohort_key(plan, "2025", "2023")


def test_shared_inner_refuses_modified_shared_membership():
    plan = synthetic_plan(PLAN_V3)
    _shared_inner(plan["folds"])
    plan["folds"][1]["inner"][1]["training_candidate_event_ids"] = ["unshared"]
    with pytest.raises(OpsError, match="differ"):
        _shared_inner(plan["folds"])


@pytest.mark.parametrize("plan_schema", ["magnitude_search_plan.v4", 0, []])
def test_prepare_plan_refuses_unsupported_schema_before_registration_reads(monkeypatch, plan_schema):
    from experiments import size_search_binding as binding
    monkeypatch.setattr(binding, "verify_native_registration", lambda *a, **k: pytest.fail("registration read"))
    with pytest.raises(OpsError) as error:
        binding.prepare_plan(None, None, None, {}, plan_schema=plan_schema)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
