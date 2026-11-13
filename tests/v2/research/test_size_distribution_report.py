"""Report-only empirical distribution bands over synthetic prediction arrays."""
import json

import numpy as np
import pytest

from experiments.magnitude_search_plan import trial_key
from experiments.size_distribution_report import (
    OBJECT_ELEMENT_BYTES,
    RATIO_FLOOR,
    SCHEMA,
    TrialArrays,
    band_metrics,
    collect_distribution,
)

POOL = np.arange(1, 101, dtype=float)
POOL_IDS = [f"e{index}" for index in range(POOL.size)]


def arrays(pred, y, event_ids=None, horizon=1):
    pred = np.asarray(pred, dtype=float)
    if event_ids is None:
        event_ids = [f"e{index}" for index in range(pred.size)]
    return {horizon: (pred, np.asarray(y, dtype=float), np.asarray(event_ids, dtype=object))}


def pooled(ratio=2.0):
    return arrays(POOL, ratio * POOL, POOL_IDS)


def plan_with_years():
    return {"plan_id": "plan", "schema_version": "magnitude_search_plan.v1",
        "folds": [{"outer": {"year": "2024", "training_candidate_event_ids": POOL_IDS}, "inner": [
            {"year": "2021", "training_candidate_event_ids": POOL_IDS},
            {"year": "2022", "training_candidate_event_ids": POOL_IDS},
            {"year": "2023", "training_candidate_event_ids": POOL_IDS}]}],
        "candidates": [{"candidate_id": "c"}], "horizons": [1]}


def test_constant_ratio_band_matches_hand_arithmetic():
    observed = band_metrics([pooled()], arrays([10.0, 10.0, 10.0], [20.0, 10.0, 25.0]), [1])
    horizon = observed["per_horizon"][0]
    assert observed["status"] == "ok"
    assert horizon["status"] == "ok"
    assert horizon["n_pool"] == 100 and horizon["n_scored"] == 3
    assert horizon["pinball_p50"] == pytest.approx(2.5)
    assert horizon["pinball_p90"] == pytest.approx(11 / 6)
    assert horizon["coverage_p90"] == pytest.approx(2 / 3)
    assert horizon["coverage_p10_p90"] == pytest.approx(1 / 3)
    assert observed["pooled"]["n_scored"] == 3 and observed["pooled"]["n_pool"] == 100
    assert observed["pooled"]["pinball_p50"] == pytest.approx(horizon["pinball_p50"])


def test_two_horizon_pooled_metrics_are_scored_row_weighted():
    pool = [{**arrays(POOL, 2.0 * POOL, POOL_IDS, horizon=1),
             **arrays(POOL, 3.0 * POOL, POOL_IDS, horizon=3)}]
    scored = {**arrays([10.0, 10.0], [15.0, 25.0], horizon=1),
              **arrays([10.0, 10.0, 10.0], [10.0, 30.0, 50.0], horizon=3)}
    observed = band_metrics(pool, scored, [1, 3])
    assert observed["status"] == "ok"
    assert [row["n_scored"] for row in observed["per_horizon"]] == [2, 3]
    assert [row["n_pool"] for row in observed["per_horizon"]] == [100, 100]
    pooled = observed["pooled"]
    assert pooled["n_scored"] == 5 and pooled["n_pool"] == 200
    assert pooled["pinball_p50"] == pytest.approx((2.5 * 2 + (20 / 3) * 3) / 5)
    assert pooled["pinball_p90"] == pytest.approx((2.5 * 2 + (20 / 3) * 3) / 5)
    assert pooled["pinball_mean"] == pytest.approx((2.5 * 2 + (20 / 3) * 3) / 5)
    assert pooled["coverage_p90"] == pytest.approx((0.5 * 2 + (2 / 3) * 3) / 5)
    assert pooled["coverage_p10_p90"] == pytest.approx((0 * 2 + (1 / 3) * 3) / 5)


def test_pool_rows_outside_destination_training_events_are_excluded():
    ratios = np.full(POOL.size, 2.0)
    ratios[90:] = 4.0
    y = POOL * ratios
    pool = [arrays(np.append(POOL, 1000.0), np.append(y, 1000.0 * 3.5),
                   POOL_IDS + ["outside"])]
    scored = arrays([90.0], [270.0])
    full = band_metrics(pool, scored, [1])
    allowed = band_metrics(pool, scored, [1], allowed=set(POOL_IDS))
    only_allowed = band_metrics([arrays(POOL, y, POOL_IDS)], scored, [1])
    assert full["per_horizon"][0]["n_pool"] == 101
    assert allowed["per_horizon"][0]["n_pool"] == 100
    assert allowed["per_horizon"][0] == only_allowed["per_horizon"][0]
    assert full["per_horizon"][0]["pinball_p50"] != allowed["per_horizon"][0]["pinball_p50"]


def test_empty_pool_is_no_pool_and_never_uses_in_sample_residuals():
    observed = band_metrics([], arrays([10.0], [20.0]), [1])
    assert observed == {"status": "no_pool",
        "per_horizon": [{"horizon_sessions": 1, "status": "no_pool"}], "pooled": None}


def test_small_pool_bucket_is_pool_too_small_without_merging():
    observed = band_metrics([arrays(np.arange(1, 100), 2 * np.arange(1, 100))],
                            arrays([10.0], [20.0]), [1])
    assert observed["status"] == "pool_too_small"
    assert observed["per_horizon"] == [{"horizon_sessions": 1, "status": "pool_too_small"}]
    assert observed["pooled"] is None


def test_collect_inner_pool_is_strictly_prior_years_only():
    plan = plan_with_years()
    keys = {year: trial_key(plan, "2024", year, "c") for year in ("2021", "2022", "2023")}
    scored, outer, selected = arrays([10.0], [20.0]), {"2024": arrays([10.0], [20.0])}, {"2024": "c"}

    def row_for(third, first=pooled()):
        trials = {keys["2021"]: first, keys["2022"]: scored, keys["2023"]: third}
        result = collect_distribution(plan, trials, outer, selected)
        return result, next(row for row in result["inner"] if row["inner_year"] == "2022")

    result, base = row_for(pooled())
    assert base["pool_years"] == ["2021"]
    assert base["status"] == "ok" and base["pooled"] is not None
    assert row_for(pooled(3.0))[1] == base
    assert row_for(pooled(3.0), pooled(3.0))[1] != base
    json.dumps(result, allow_nan=False)


def test_middle_inner_pool_uses_only_allowed_first_year_rows():
    plan = plan_with_years()
    keys = {year: trial_key(plan, "2024", year, "c") for year in ("2021", "2022", "2023")}
    pool = np.arange(1, 201, dtype=float)
    ids = [f"e{index}" for index in range(pool.size)]
    allowed = {f"e{index}" for index in range(100)}
    plan["folds"][0]["inner"][1]["training_candidate_event_ids"] = sorted(allowed) + ["later-only"]
    trials = {keys["2021"]: arrays(pool, 2.0 * pool, ids),
              keys["2022"]: arrays([10.0], [20.0]),
              keys["2023"]: arrays([10.0], [20.0])}
    result = collect_distribution(plan, trials, {"2024": arrays([10.0], [20.0])}, {"2024": "c"})
    row = next(row for row in result["inner"] if row["inner_year"] == "2022")
    assert row["pool_years"] == ["2021"]
    assert row["status"] == "ok"
    assert row["pooled"]["n_pool"] == len(allowed)


def test_failed_trial_is_reported_and_omitted_from_later_pools():
    plan = plan_with_years()
    keys = {year: trial_key(plan, "2024", year, "c") for year in ("2021", "2022", "2023")}
    trials = {keys["2021"]: pooled(), keys["2023"]: pooled()}
    result = collect_distribution(plan, trials, {"2024": arrays([10.0], [20.0])}, {"2024": "c"})
    rows = {row["inner_year"]: row for row in result["inner"]}
    assert rows["2021"]["status"] == "no_pool"
    assert rows["2022"]["status"] == "trial_failed"
    assert rows["2022"]["pool_years"] == ["2021"]
    assert rows["2022"]["pooled"] is None and rows["2022"]["per_horizon"] == []
    assert rows["2023"]["pool_years"] == ["2021"]
    arms = {(row["outer_year"], row["arm"]): row for row in result["outer"]}
    assert arms[("2024", "selected")]["status"] == "ok"
    for arm in ("incumbent", "history_mean", "implied_move"):
        assert arms[("2024", arm)] == {"outer_year": "2024", "arm": arm, "status": "no_pool"}


def test_non_finite_values_refuse_and_target_plan_is_unsupported():
    with pytest.raises(ValueError):
        band_metrics([arrays([1.0, np.nan, 3.0], [2.0, 4.0, 6.0])], arrays([1.0], [2.0]), [1])
    with pytest.raises(ValueError):
        band_metrics([pooled()], arrays([np.inf], [2.0]), [1])
    assert collect_distribution({"target": {"column": "x"}}, {}, {}, {}) == {
        "schema_version": SCHEMA, "status": "unsupported_target"}


def test_floored_zero_and_negative_predictions_stay_finite():
    pred = POOL.copy()
    pred[0], pred[1] = 0.0, -5.0
    observed = band_metrics([arrays(pred, 2 * pred)], arrays([0.0, -10.0, 10.0], [0.0, -20.0, 20.0]), [1])
    horizon = observed["per_horizon"][0]
    assert observed["status"] == "ok"
    assert 0.0 < RATIO_FLOOR < 1.0
    assert all(np.isfinite(horizon[key]) for key in (
        "pinball_p50", "pinball_p90", "pinball_mean", "coverage_p90", "coverage_p10_p90"))


def test_pool_years_drop_donors_with_no_allowed_rows():
    plan = plan_with_years()
    plan["folds"][0]["inner"][1]["training_candidate_event_ids"] = ["not-a-pool-event"]
    keys = {year: trial_key(plan, "2024", year, "c") for year in ("2021", "2022", "2023")}
    trials = {key: pooled() for key in keys.values()}
    result = collect_distribution(plan, trials, {"2024": arrays([10.0], [20.0])}, {"2024": "c"})
    rows = {row["inner_year"]: row for row in result["inner"]}
    assert rows["2022"]["pool_years"] == []
    assert rows["2022"]["status"] == "no_pool"
    assert rows["2022"]["pooled"] is None
    assert rows["2023"]["pool_years"] == ["2021", "2022"]


def test_every_donor_filtered_reports_no_pool():
    plan = plan_with_years()
    for fold in plan["folds"][0]["inner"]:
        fold["training_candidate_event_ids"] = ["not-a-pool-event"]
    keys = {year: trial_key(plan, "2024", year, "c") for year in ("2021", "2022", "2023")}
    trials = {key: pooled() for key in keys.values()}
    result = collect_distribution(plan, trials, {"2024": arrays([10.0], [20.0])}, {"2024": "c"})
    for row in result["inner"]:
        assert row["pool_years"] == []
        assert row["status"] == "no_pool"


def test_non_finite_unrequested_horizon_refuses():
    pool = {**arrays(POOL, 2.0 * POOL, POOL_IDS, horizon=1),
            **arrays([np.nan], [1.0], ["unused"], horizon=3)}
    with pytest.raises(ValueError, match="non-finite distribution input"):
        band_metrics([pool], arrays([10.0], [20.0]), [1])


def test_trial_arrays_add_refuses_non_finite_before_and_after_cap(monkeypatch):
    trial_arrays = TrialArrays()
    with pytest.raises(ValueError, match="non-finite distribution input"):
        trial_arrays.add("nan", arrays([np.nan], [1.0]))
    monkeypatch.setattr(TrialArrays, "MAX_BYTES", 70)
    trial_arrays.add("first", arrays([1.0, 2.0], [3.0, 4.0]))
    trial_arrays.add("second", arrays([1.0, 2.0], [3.0, 4.0]))
    assert trial_arrays.capped is True
    with pytest.raises(ValueError, match="non-finite distribution input"):
        trial_arrays.add("nan", arrays([1.0], [np.nan]))


def test_trial_arrays_add_refuses_malformed_before_and_after_cap(monkeypatch):
    trial_arrays = TrialArrays()
    with pytest.raises(ValueError, match="malformed distribution input"):
        trial_arrays.add("bad", {1: (np.array([1.0]), np.array([2.0]))})
    monkeypatch.setattr(TrialArrays, "MAX_BYTES", 70)
    trial_arrays.add("first", arrays([1.0, 2.0], [3.0, 4.0]))
    trial_arrays.add("second", arrays([1.0, 2.0], [3.0, 4.0]))
    assert trial_arrays.capped is True
    with pytest.raises(ValueError, match="malformed distribution input"):
        trial_arrays.add("bad", {1: (np.array([1.0]), np.array([2.0]))})


def test_non_finite_outer_rows_refuses_even_when_trial_arrays_capped(monkeypatch):
    monkeypatch.setattr(TrialArrays, "MAX_BYTES", 70)
    trial_arrays = TrialArrays()
    trial_arrays.add("first", arrays([1.0, 2.0], [3.0, 4.0]))
    trial_arrays.add("second", arrays([1.0, 2.0], [3.0, 4.0]))
    assert trial_arrays.capped is True
    with pytest.raises(ValueError, match="non-finite distribution input"):
        collect_distribution(plan_with_years(), trial_arrays,
                             {"2024": arrays([np.nan], [1.0])}, {"2024": "c"})


def test_trial_arrays_size_includes_object_event_id_strings(monkeypatch):
    monkeypatch.setattr(TrialArrays, "MAX_BYTES", 100)
    trial_arrays = TrialArrays()
    trial_arrays.add("first", arrays([1.0, 2.0], [3.0, 4.0]))
    assert trial_arrays.capped is True
    assert trial_arrays == {}

    stored = TrialArrays()
    stored.MAX_BYTES = 1000
    stored.add("small", arrays([1.0], [2.0]))
    assert stored.capped is False
    assert stored.nbytes == 8 + 8 + OBJECT_ELEMENT_BYTES


def test_trial_arrays_add_same_key_is_idempotent():
    trial_arrays = TrialArrays()
    trial_arrays.add("shared", arrays([1.0], [2.0]))
    single = trial_arrays.nbytes
    trial_arrays.add("shared", arrays([1.0], [2.0]))
    assert trial_arrays.nbytes == single == 8 + 8 + OBJECT_ELEMENT_BYTES
    assert len(trial_arrays) == 1


def test_trial_arrays_cap_marks_empty_and_collect_skips(monkeypatch):
    monkeypatch.setattr(TrialArrays, "MAX_BYTES", 70)
    trial_arrays = TrialArrays()
    trial_arrays.add("first", arrays([1.0, 2.0], [3.0, 4.0]))
    trial_arrays.add("second", arrays([1.0, 2.0], [3.0, 4.0]))
    assert trial_arrays.capped is True
    assert trial_arrays == {}
    assert collect_distribution(plan_with_years(), trial_arrays, {}, {}) == {
        "schema_version": SCHEMA, "status": "skipped_resource_cap"}
