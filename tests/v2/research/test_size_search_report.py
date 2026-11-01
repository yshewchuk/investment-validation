"""Mixed candidate terminal states retain diagnostics without inventing convergence."""
from types import SimpleNamespace

from experiments import size_search_report as report


def test_mixed_failed_and_successful_trials_report_diagnostics(monkeypatch):
    plan = {"plan_id": "plan", "horizons": [5, 3, 1], "folds": [{"outer": {"year": "2024", "evaluation_event_ids": ["e"]},
        "inner": [{"year": "2023"}]}], "candidates": [{"candidate_id": "ok"}, {"candidate_id": "failed"}]}
    result = {"predictions": [{"event_id": "e", "absolute_move_pct": 2}], "folds": []}
    registration = SimpleNamespace(run_id="run", document={"size_panel": {"context": {
        "target_events": [{"event_id": "e"}], "eligibility_plan": {"frozen": True}}}})
    monkeypatch.setattr(report, "_admit_driver", lambda *a: plan)
    monkeypatch.setattr(report, "collect_outer_results", lambda *a: result)
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
        if slot.startswith("outer-validation:"):
            return {"targets": [{"event_id": "e", "absolute_move_pct": 2}]}
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


def test_collect_preserves_labels_for_labeled_challenger_abstention(monkeypatch):
    from types import SimpleNamespace

    plan = {"plan_id": "plan", "horizons": [1], "folds": [
        {"outer": {"year": "2024", "evaluation_event_ids": ["e1", "e2", "e3"]}}]}
    events = [
        {"event_id": "e1", "event_date": "2024-01-15", "ticker": "AAA"},
        {"event_id": "e2", "event_date": "2024-04-15", "ticker": "BBB"},
        {"event_id": "e3", "event_date": "2024-07-15", "ticker": "CCC"},
    ]
    registration = SimpleNamespace(run_id="run", document={"size_panel": {"context": {
        "target_events": events, "eligibility_plan": {"frozen": True}}}})
    outer_result = {
        "predictions": [{"event_id": "e1", "horizon_sessions": 1,
                         "absolute_move_pct": 2.0, "predicted_absolute_move_pct": 1.5}],
        "folds": [{"outer_year": "2024", "evaluation_inputs": {"omitted_labels": [
            {"event_id": "e3", "reason": "not_computed"}]}}],
    }
    challenger = [
        {"event_id": "e1", "horizon_sessions": 1, "predicted_absolute_move_pct": 1.5},
        {"event_id": "e3", "horizon_sessions": 1, "predicted_absolute_move_pct": 2.5},
    ]
    baseline = [{"event_id": event, "horizon_sessions": 1, "history_mean": 1.0,
                 "predicted_absolute_move_pct": 1.25, "abstention": None, "quote_status": "available"}
                for event in ("e1", "e2", "e3")]
    validation = {"targets": [
        {"event_id": "e1", "absolute_move_pct": 2.0},
        {"event_id": "e2", "absolute_move_pct": 3.0},
    ]}

    monkeypatch.setattr(report, "_admit_driver", lambda *args: plan)
    monkeypatch.setattr(report, "collect_outer_results", lambda *args: outer_result)
    monkeypatch.setattr(report, "_candidate_evidence", lambda *args: [])
    monkeypatch.setattr(report, "resource_report", lambda *args: {})

    def read_record(conn, store, run_id, slot):
        if slot.startswith("outer-forecasts:"):
            return challenger
        if slot.startswith("outer-validation:"):
            return validation
        if slot.startswith("comparison-terminal:"):
            return {"result": {"forecasts": baseline, "fit_details": []}}
        raise AssertionError(f"unexpected report slot: {slot}")

    monkeypatch.setattr(report, "read_record", read_record)
    observed = report.collect(None, None, None, registration, {}, None)["paired"]
    assert observed["label_unavailable_events"] == ["e3"]
    assert observed["paired_events"] == 1
    assert observed["challenger_abstentions"] == 1
    assert any(row["arm"] == "challenger" and row["forecast_status"] == "abstained"
               and row["label_status"] == "available" and row["count"] == 1
               for row in observed["coverage"])
