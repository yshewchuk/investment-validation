"""Mixed candidate terminal states retain diagnostics without inventing convergence."""
from types import SimpleNamespace

import pytest

from experiments import size_search_report as report


def test_mixed_failed_and_successful_trials_report_diagnostics(monkeypatch):
    plan = {"plan_id": "plan", "horizons": [5, 3, 1], "folds": [{"outer": {"year": "2024", "evaluation_event_ids": ["e"]},
        "inner": [{"year": "2023"}]}], "candidates": [{"candidate_id": "ok"}, {"candidate_id": "failed"}]}
    result = {"predictions": [{"event_id": "e", "absolute_move_pct": 2}], "folds": []}
    registration = SimpleNamespace(run_id="run", document={"size_panel": {"context": {
        "target_events": [{"event_id": "e"}], "eligibility_plan": {"frozen": True}}}})
    monkeypatch.setattr(report, "_admit_driver", lambda *a: plan)
    monkeypatch.setattr(report, "collect_outer_results", lambda *a, **k: result)
    monkeypatch.setattr(report, "resource_report", lambda p: {})
    monkeypatch.setattr(report, "paired_report", lambda *a, **k: {"paired": True})
    def read(conn, store, run, slot):
        if slot.startswith("execution:"):
            succeeded = slot.endswith(report.content_hash(["plan", "2024", "2023", "ok"]))
            if succeeded:
                return {"status": "succeeded", "fit_attempt_count": 3, "completed_fit_count": 3,
                    "fits": [{"horizon_sessions": 1}, {"horizon_sessions": 5}],
                    "diagnostics": [{"converged": True}]}
            return {"status": "failed", "fit_attempt_count": 3, "completed_fit_count": 2,
                "fits": [{"horizon_sessions": 1}],
                "diagnostics": [{"converged": False}]}
        if slot.startswith("fit-completed:"):
            success_key = report.content_hash(["plan", "2024", "2023", "ok"])
            if slot.startswith("fit-completed:" + success_key + ":"):
                horizon = slot.rsplit(":", 1)[-1]
                return {"horizon_sessions": int(horizon), "fit_seconds": 1.25 if horizon == "1" else 2.75}
            if slot.endswith(":1"):
                return {"horizon_sessions": 1, "fit_seconds": 0.5}
            return None
        if slot.startswith("terminal:"):
            succeeded = slot.endswith(report.content_hash(["plan", "2024", "2023", "ok"]))
            terminal = {"status": "succeeded" if succeeded else "failed",
                "score_seconds": 0.7 if succeeded else 0.3,
                "rss_delta_kib": 2048 if succeeded else 512}
            if succeeded:
                terminal["inner_metrics"] = [{"horizon_sessions": 1, "mean_absolute_move_pct": 0.4}]
            return terminal
        if slot.startswith("outer-forecasts:"):
            return []
        return {"result": {"forecasts": [], "fit_details": [
            {"horizon_sessions": 1, "component_fits": 2, "logical_fits": 1}]}}
    monkeypatch.setattr(report, "read_record", read)
    observed = report.collect(None, None, None, registration, {}, None)
    succeeded, failed = observed["inner_fits"]
    assert [f["status"] for f in observed["inner_fits"]] == ["succeeded", "failed"]
    assert [f["fit_attempt_count"] for f in observed["inner_fits"]] == [3, 3]
    assert [f["completed_fit_count"] for f in observed["inner_fits"]] == [3, 2]
    assert succeeded["diagnostics"] == [{"converged": True}]
    assert failed["diagnostics"] == []
    assert succeeded["fit_seconds"] == [
        {"horizon_sessions": 5, "seconds": 2.75},
        {"horizon_sessions": 3, "seconds": 2.75},
        {"horizon_sessions": 1, "seconds": 1.25},
    ]
    assert failed["fit_seconds"] == [{"horizon_sessions": 1, "seconds": 0.5}]
    assert succeeded["score_seconds"] == 0.7
    assert failed["score_seconds"] == 0.3
    assert succeeded["rss_delta_kib"] == 2048
    assert failed["rss_delta_kib"] == 512
    assert succeeded["inner_metrics"] == [{"horizon_sessions": 1, "mean_absolute_move_pct": 0.4}]
    assert failed["inner_metrics"] is None
    assert observed["comparison_fits"][0]["component_fits"] == 2


def test_candidate_evidence_distinguishes_missing_metrics_from_empty(monkeypatch):
    plan = {"plan_id": "plan", "horizons": [1], "folds": [{"outer": {"year": "2024"},
        "inner": [{"year": "2023"}]}], "candidates": [
            {"candidate_id": "legacy"}, {"candidate_id": "empty"}]}
    legacy_key = report.content_hash(["plan", "2024", "2023", "legacy"])
    empty_key = report.content_hash(["plan", "2024", "2023", "empty"])

    def read(conn, store, run_id, slot):
        if slot.startswith("execution:"):
            return {"status": "succeeded", "fits": []}
        if slot.startswith("fit-completed:"):
            return None
        if slot == "terminal:" + legacy_key:
            return {"status": "succeeded"}
        if slot == "terminal:" + empty_key:
            return {"status": "succeeded", "inner_metrics": []}
        raise AssertionError(f"unexpected journal slot: {slot}")

    monkeypatch.setattr(report, "read_record", read)
    rows = report._candidate_evidence(None, None, "run", plan)

    assert [row["inner_metrics"] for row in rows] == [None, []]


def test_collect_uses_verified_callback_inputs_per_outer_fold(monkeypatch):
    folds = [{"outer": {"year": year, "training_candidate_event_ids": ["t" + year],
                        "evaluation_event_ids": ["e" + year]}, "inner": []}
             for year in ("2024", "2025")]
    plan = {"plan_id": "plan", "horizons": [1], "folds": folds, "candidates": []}
    result = {"predictions": [], "folds": [
        {"outer_year": group["outer"]["year"], "evaluation_inputs": {"omitted_labels": []}}
        for group in folds]}
    registration = SimpleNamespace(run_id="run", document={
        "size_panel": {"context": {
            "target_events": [{"event_id": "e2024"}, {"event_id": "e2025"}],
            "eligibility_plan": {"frozen": True}}},
        "magnitude_execution": {"evaluation_labels_before": "2025-01-01"}})
    monkeypatch.setattr(report, "_admit_driver", lambda *a: plan)
    monkeypatch.setattr(report, "resource_report", lambda p: {})
    invoked = []

    def fake_collect(*args, input_callback=None):
        assert input_callback is not None
        for group in folds:
            outer = group["outer"]
            fold = {**outer, "evaluation_labels_before": "2025-01-01"}
            inputs = {"training": [{"event_id": "t" + outer["year"], "absolute_move_pct": 4.0}],
                      "features": [
                          {"event_id": "t" + outer["year"], "horizon_sessions": 1, "implied_move": 2.0},
                          {"event_id": "e" + outer["year"], "horizon_sessions": 1, "implied_move": 3.0},
                          {"event_id": "e" + outer["year"], "horizon_sessions": 3, "implied_move": 9.0}]}
            input_callback(fold, inputs)
            invoked.append(outer["year"])
        return result

    monkeypatch.setattr(report, "collect_outer_results", fake_collect)
    captured = {}

    def paired(*args, **kwargs):
        captured.update(kwargs)
        return {"paired": True}

    monkeypatch.setattr(report, "paired_report", paired)

    def read(conn, store, run_id, slot):
        if slot.startswith("outer-forecasts:"):
            return []
        return {"result": {"forecasts": [], "fit_details": []}}

    monkeypatch.setattr(report, "read_record", read)
    report.collect(None, None, None, registration, {}, None)

    assert invoked == ["2024", "2025"]
    assert captured["implied_scales"] == {
        "2024": {"scale": 2.0, "fit_n": 1}, "2025": {"scale": 2.0, "fit_n": 1}}
    assert captured["implied_moves"] == {("e2024", 1): 3.0, ("e2025", 1): 3.0}
    assert not hasattr(report, "fold_inputs")
    assert not hasattr(report, "read_stored_panel")
    assert not hasattr(report, "_panel_columns")
    assert not hasattr(report, "qualified_recipe")
    assert not hasattr(report, "SnapshotRef")
    assert not hasattr(report, "from_document")
    assert not hasattr(report, "_implied_move_scales")
    assert not hasattr(report, "_implied_moves")


@pytest.mark.parametrize("value", [-1, True, "2.0", float("nan"), float("inf")])
def test_fit_implied_scale_rejects_invalid_non_null_implied_move(value):
    features = [{"event_id": "e", "horizon_sessions": 1, "implied_move": value}]
    with pytest.raises(ValueError, match="invalid implied move"):
        report.fit_implied_scale(["e"], {"e": 4.0}, features, [1])


def test_fit_implied_scale_null_training_feature_abstains_without_fit_count():
    features = [
        {"event_id": "e1", "horizon_sessions": 1, "implied_move": None},
        {"event_id": "e2", "horizon_sessions": 1, "implied_move": 2.0},
    ]
    result = report.fit_implied_scale(["e1", "e2"], {"e1": 100.0, "e2": 4.0}, features, [1])
    assert result == {"scale": 2.0, "fit_n": 1}


def test_fit_implied_scale_null_only_training_feature_abstains():
    features = [{"event_id": "e", "horizon_sessions": 1, "implied_move": None}]
    result = report.fit_implied_scale(["e"], {"e": 4.0}, features, [1])
    assert result == {"scale": None, "fit_n": 0}


def test_fit_implied_scale_still_skips_unavailable_or_invalid_labels():
    features = [
        {"event_id": "missing", "horizon_sessions": 1, "implied_move": 3.0},
        {"event_id": "invalid", "horizon_sessions": 1, "implied_move": 3.0},
        {"event_id": "valid", "horizon_sessions": 1, "implied_move": 2.0},
    ]
    labels = {"invalid": float("nan"), "valid": 4.0}
    result = report.fit_implied_scale(["missing", "invalid", "valid"], labels, features, [1])
    assert result == {"scale": 2.0, "fit_n": 1}


@pytest.mark.parametrize("horizon", [True, 1.0])
def test_fit_implied_scale_refuses_malformed_feature_horizon(horizon):
    features = [{"event_id": "e", "horizon_sessions": horizon, "implied_move": 2.0}]
    with pytest.raises(ValueError, match="invalid feature horizon"):
        report.fit_implied_scale(["e"], {"e": 4.0}, features, [1])


@pytest.mark.parametrize("horizons", [(), [], (True,), (1.0,), (2,), (1, 1)])
def test_fit_implied_scale_refuses_invalid_horizon_declarations(horizons):
    features = [{"event_id": "e", "horizon_sessions": 1, "implied_move": 2.0}]
    with pytest.raises(ValueError, match="nonempty unique subset of integer leads"):
        report.fit_implied_scale(["e"], {"e": 4.0}, features, horizons)
