"""Mixed candidate terminal states retain diagnostics without inventing convergence."""
from types import SimpleNamespace

from experiments import size_search_report as report


def test_mixed_failed_and_successful_trials_report_diagnostics(monkeypatch):
    plan = {"plan_id": "plan", "folds": [{"outer": {"year": "2024", "evaluation_event_ids": ["e"]},
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
            return {"status": "succeeded" if succeeded else "failed", "fit_attempt_count": 3,
                "completed_fit_count": 3 if succeeded else 2,
                **({"diagnostics": [{"converged": True}]} if succeeded else {})}
        if slot.startswith("outer-forecasts:"):
            return []
        return {"result": {"forecasts": [], "fit_details": [
            {"horizon_sessions": 1, "component_fits": 2, "logical_fits": 1}]}}
    monkeypatch.setattr(report, "read_record", read)
    observed = report.collect(None, None, None, registration, {}, None)
    assert [f["status"] for f in observed["inner_fits"]] == ["succeeded", "failed"]
    assert observed["inner_fits"][0]["diagnostics"] == [{"converged": True}]
    assert observed["inner_fits"][1]["diagnostics"] == []
    assert observed["inner_fits"][1]["completed_fit_count"] == 2
    assert observed["comparison_fits"][0]["component_fits"] == 2
