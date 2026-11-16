"""Report-only IV empirical distribution bands over synthetic prediction arrays."""
import json

import numpy as np
import pytest

from experiments import iv_distribution_report
from experiments.iv_distribution_report import (
    SCHEMA,
    TrialArrays,
    band_metrics,
    collect_distribution,
    outer_arrays,
    read_trial_arrays,
)
from experiments.magnitude_search_plan import trial_key
from experiments.size_distribution_report import collect_distribution as size_collect

POOL = np.arange(1, 101, dtype=float)
POOL_IDS = [f"e{index}" for index in range(POOL.size)]


def arrays(pred, y, event_ids=None, horizon=1):
    pred = np.asarray(pred, dtype=float)
    if event_ids is None:
        event_ids = [f"e{index}" for index in range(pred.size)]
    return {horizon: (pred, np.asarray(y, dtype=float), np.asarray(event_ids, dtype=object))}


def pool():
    return arrays(POOL, POOL + 3.0, POOL_IDS)


def plan_with_years():
    return {"plan_id": "plan", "schema_version": "magnitude_search_plan.v1",
        "folds": [{"outer": {"year": "2024", "training_candidate_event_ids": POOL_IDS}, "inner": [
            {"year": "2021", "training_candidate_event_ids": POOL_IDS},
            {"year": "2022", "training_candidate_event_ids": POOL_IDS},
            {"year": "2023", "training_candidate_event_ids": POOL_IDS}]}],
        "candidates": [{"candidate_id": "c"}], "horizons": [1]}


def test_additive_band_is_exact():
    observed = band_metrics([pool()], arrays([50.0], [53.0]), [1])
    horizon = observed["per_horizon"][0]
    assert observed["status"] == "ok"
    assert horizon["status"] == "ok"
    assert horizon["n_pool"] == 100 and horizon["n_scored"] == 1
    for name in ("band_p10", "band_p50", "band_p90"):
        assert horizon[name] == pytest.approx(53.0)
    assert horizon["pinball_p50"] == pytest.approx(0.0)
    assert horizon["coverage_p10"] == pytest.approx(1.0)
    assert horizon["coverage_p90"] == pytest.approx(1.0)
    assert horizon["coverage_p10_p90"] == pytest.approx(1.0)


def test_signed_targets_near_zero_stay_finite():
    pred = np.linspace(-10.0, 10.0, 100)
    observed = band_metrics([arrays(pred, pred - 5.0)],
                            arrays([-5.0, 0.0, -10.0], [-10.0, -5.0, -15.0]), [1])
    horizon = observed["per_horizon"][0]
    assert observed["status"] == "ok"
    assert all(np.isfinite(horizon[key]) for key in (
        "pinball_p10", "pinball_p50", "pinball_p90", "pinball_mean",
        "coverage_p10", "coverage_p90", "coverage_p10_p90",
        "band_p10", "band_p50", "band_p90"))


def test_statuses_and_pooled_require_every_horizon():
    assert band_metrics([], arrays([1.0], [2.0]), [1]) == {
        "status": "no_pool",
        "per_horizon": [{"horizon_sessions": 1, "status": "no_pool"}], "pooled": None}
    small = arrays(np.arange(1.0, 51.0), np.arange(1.0, 51.0) + 2.0)
    observed = band_metrics([small], arrays([10.0], [12.0]), [1])
    assert observed["status"] == "pool_too_small"
    assert observed["per_horizon"] == [{"horizon_sessions": 1, "status": "pool_too_small"}]
    assert observed["pooled"] is None
    observed = band_metrics([pool()], arrays([], []), [1])
    assert observed["status"] == "no_scored_rows"
    assert observed["pooled"] is None
    pool_two = [{**arrays(POOL, POOL + 3.0, POOL_IDS, horizon=1),
                 **arrays(np.arange(1.0, 51.0), np.arange(1.0, 51.0) + 3.0, horizon=3)}]
    scored = {**arrays([50.0], [53.0], horizon=1), **arrays([10.0], [13.0], horizon=3)}
    observed = band_metrics(pool_two, scored, [1, 3])
    assert [row["status"] for row in observed["per_horizon"]] == ["ok", "pool_too_small"]
    assert observed["status"] == "pool_too_small"
    assert observed["pooled"] is None


def test_non_finite_input_refuses():
    with pytest.raises(ValueError):
        band_metrics([arrays([1.0, np.nan, 3.0], [2.0, 4.0, 6.0])], arrays([1.0], [2.0]), [1])
    with pytest.raises(ValueError):
        band_metrics([pool()], arrays([np.inf], [2.0]), [1])


def test_collect_target_plan_and_arms():
    plan = plan_with_years()
    plan["target"] = {"column": "crush_pct_iv30"}
    keys = {year: trial_key(plan, "2024", year, "c") for year in ("2021", "2022", "2023")}
    trials = {keys["2021"]: pool(), keys["2023"]: pool()}
    result = collect_distribution(plan, trials, {"2024": arrays([10.0], [20.0])}, {"2024": "c"})
    assert result["schema_version"] == SCHEMA and result["status"] == "ok"
    assert "ratio_floor" not in result
    rows = {row["inner_year"]: row for row in result["inner"]}
    assert rows["2021"]["status"] == "no_pool"
    assert rows["2022"]["status"] == "trial_failed"
    arms = {(row["outer_year"], row["arm"]): row for row in result["outer"]}
    assert arms[("2024", "selected")]["status"] == "ok"
    for arm in ("incumbent", "training_mean"):
        assert arms[("2024", arm)] == {"outer_year": "2024", "arm": arm, "status": "no_pool"}


def test_capped_trial_arrays_skip(monkeypatch):
    monkeypatch.setattr(TrialArrays, "MAX_BYTES", 70)
    trial_arrays = TrialArrays()
    trial_arrays.add("first", arrays([1.0, 2.0], [3.0, 4.0]))
    trial_arrays.add("second", arrays([1.0, 2.0], [3.0, 4.0]))
    assert trial_arrays.capped is True
    assert collect_distribution(plan_with_years(), trial_arrays, {}, {}) == {
        "schema_version": SCHEMA, "status": "skipped_resource_cap"}


def test_read_trial_arrays_only_succeeded_unique_keys(monkeypatch):
    plan = {"plan_id": "plan", "schema_version": "magnitude_search_plan.v3",
        "folds": [{"outer": {"year": "2024"}, "inner": [{"year": "2021"}]},
                  {"outer": {"year": "2025"}, "inner": [{"year": "2021"}]}],
        "candidates": [{"candidate_id": "c"}, {"candidate_id": "d"}], "horizons": [1, 3]}
    keys = {name: trial_key(plan, "2024", "2021", name) for name in ("c", "d")}
    assert keys["c"] == trial_key(plan, "2025", "2021", "c")
    calls = []

    def fake_read(conn, store, run_id, slot):
        calls.append(slot)
        key = slot[len("terminal:"):]
        if key == keys["c"]:
            return {"status": "succeeded", "predictions": [
                {"event_id": 1, "horizon_sessions": 1,
                 "predicted_crush_pct_iv30": 2.5, "crush_pct_iv30": 3.5},
                {"event_id": 2, "horizon_sessions": 1,
                 "predicted_crush_pct_iv30": 4.5, "crush_pct_iv30": 5.5},
                {"event_id": 3, "horizon_sessions": 3,
                 "predicted_crush_pct_iv30": 6.5, "crush_pct_iv30": 7.5}]}
        if key == keys["d"]:
            return {"status": "failed"}
        return None

    monkeypatch.setattr(iv_distribution_report, "read_record", fake_read)
    observed = read_trial_arrays(None, None, "run", plan)
    assert set(observed) == {keys["c"]}
    assert len(calls) == len(set(keys.values())) == 2
    pred, y, event_ids = observed[keys["c"]][1]
    assert pred.tolist() == [2.5, 4.5]
    assert y.tolist() == [3.5, 5.5]
    assert event_ids.tolist() == ["1", "2"]
    pred3, y3, event_ids3 = observed[keys["c"]][3]
    assert pred3.tolist() == [6.5] and y3.tolist() == [7.5] and event_ids3.tolist() == ["3"]


def test_band_metrics_deterministic():
    first = band_metrics([pool()], arrays([10.0, 50.0], [13.0, 53.0]), [1])
    second = band_metrics([pool()], arrays([10.0, 50.0], [13.0, 53.0]), [1])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_outer_arrays_include_every_plan_horizon():
    plan = {"schema_version": "magnitude_search_plan.v3", "plan_id": "plan",
        "folds": [{"outer": {"year": "2024"}, "inner": []},
                  {"outer": {"year": "2025"}, "inner": []}],
        "candidates": [], "horizons": [1, 3]}
    predictions = [
        {"outer_year": "2024", "horizon_sessions": 1, "event_id": "a",
         "predicted_crush_pct_iv30": 1.0, "crush_pct_iv30": 2.0},
        {"outer_year": "2024", "horizon_sessions": 3, "event_id": "b",
         "predicted_crush_pct_iv30": 3.0, "crush_pct_iv30": 4.0},
        {"outer_year": "2025", "horizon_sessions": 1, "event_id": "c",
         "predicted_crush_pct_iv30": 5.0, "crush_pct_iv30": 6.0}]
    observed = outer_arrays(plan, predictions)
    assert set(observed) == {"2024", "2025"}
    for year in observed:
        assert set(observed[year]) == {1, 3}
    assert observed["2024"][1][0].tolist() == [1.0]
    assert observed["2024"][1][1].tolist() == [2.0]
    assert observed["2024"][1][2].tolist() == ["a"]
    assert observed["2025"][3][0].size == 0 and observed["2025"][3][1].size == 0
    assert observed["2025"][3][2].size == 0


def test_size_distribution_target_gate_is_unchanged():
    assert size_collect({"target": {}}, {}, {}, {}) == {
        "schema_version": "size_distribution.v1", "status": "unsupported_target"}
