"""Report-only empirical distribution bands over synthetic prediction arrays."""
import json

import numpy as np
import pytest

from experiments.magnitude_search_plan import trial_key
from experiments.size_distribution_report import (
    RATIO_FLOOR,
    SCHEMA,
    band_metrics,
    collect_distribution,
)

POOL = np.arange(1, 101, dtype=float)


def arrays(pred, y):
    return {1: (np.asarray(pred, dtype=float), np.asarray(y, dtype=float))}


def pooled(ratio=2.0):
    return arrays(POOL, ratio * POOL)


def plan_with_years():
    return {"plan_id": "plan", "schema_version": "magnitude_search_plan.v1",
        "folds": [{"outer": {"year": "2024"}, "inner": [
            {"year": "2021"}, {"year": "2022"}, {"year": "2023"}]}],
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
