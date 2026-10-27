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
from experiments.magnitude_search_request_versions import compile_request
from experiments.size_fit_diagnostics import evidence
from experiments.size_search import _contain
from experiments.size_search_binding import _folds


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
