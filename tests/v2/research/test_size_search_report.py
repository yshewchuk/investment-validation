"""Mixed candidate terminal states retain diagnostics without inventing convergence."""
from types import SimpleNamespace

from experiments import size_search_report as report


def test_mixed_failed_and_successful_trials_report_diagnostics(monkeypatch):
    plan = {"plan_id": "plan", "horizons": [1, 5], "folds": [{"outer": {"year": "2024", "evaluation_event_ids": ["e"]},
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
    assert succeeded["fit_seconds"] == [{"horizon_sessions": 1, "seconds": 1.25},
                                        {"horizon_sessions": 5, "seconds": 2.75}]
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
