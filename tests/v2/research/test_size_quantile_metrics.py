"""Tests for experiments/size_quantile_metrics.py (EXP-190 Q5).

Synthetic arrays and fixed seeds only.
"""

import numpy as np
import pytest

from experiments.size_quantile_metrics import (
    Rules, cell_scores, coverage, crossing_rate, crps_quantile, level_index, level_width,
    pinball,
)

LEVELS = np.array(Rules().levels)
MONTH = "2025-03-01"


def _iqr(v):
    p25, p75 = np.quantile(v, [0.25, 0.75])
    return p75 - p25


def _hist(values, y, label):
    values = np.asarray(values, float)
    return values, np.asarray(y, float), np.full(values.size, np.datetime64(label, "ns"))


def _cat(*parts):
    return tuple(np.concatenate(cols) for cols in zip(*parts))


def _months(*labels):
    return np.array(labels, dtype="datetime64[ns]")


def test_pinball_hand_cases_and_shape():
    assert pinball([10.0], [[8.0]], [0.9])[0, 0] == pytest.approx(1.8)
    assert pinball([8.0], [[10.0]], [0.9])[0, 0] == pytest.approx(0.2)
    out = pinball([1.0, -1.0], np.zeros((2, 7)), LEVELS)
    assert out.shape == (2, 7)
    np.testing.assert_allclose(out, np.vstack([LEVELS, 1 - LEVELS]))


def test_coverage_inclusive_bands_le_flags_and_level_index():
    q = np.tile([1.0, 2.0, 2.5, 2.7, 2.9, 3.0, 4.0], (6, 1))
    cov = coverage(np.array([2.0, 3.0, 1.0, 4.0, 0.5, 4.5]), q, LEVELS)
    assert cov["p25_p75"].tolist() == [True, True, False, False, False, False]
    assert cov["p10_p90"].tolist() == [True, True, True, True, False, False]
    assert cov["le_p25"].tolist() == [True, False, True, False, True, False]
    assert cov["le_p90"].tolist() == [True, True, True, True, True, False]
    assert level_index(LEVELS, 0.25) == 1
    with pytest.raises(ValueError, match="missing"):
        level_index(LEVELS, 0.33)


def test_crps_closed_forms():
    assert crps_quantile(0.5, [0, 1], [0, 1]) == pytest.approx(1 / 12)
    assert crps_quantile(0.0, [0, 1], [0, 1]) == pytest.approx(1 / 3)
    assert crps_quantile(2.0, [0, 1], [0, 1]) == pytest.approx(4 / 3)
    assert crps_quantile(5.0, [0, 1], [3, 3]) == pytest.approx(2.0)
    assert crps_quantile(0.0, [0, 0.5, 0.5, 1], [0, 0, 1, 1]) == pytest.approx(0.25)


def test_crps_matches_fine_grid_and_rejects_bad_input():
    u, x, y = np.array([0.0, 0.2, 0.7, 1.0]), np.array([-1.0, 0.5, 0.5, 3.0]), 0.3
    n = 10**6
    tau = (np.arange(n) + 0.5) / n
    q = np.interp(tau, u, x)
    grid = np.mean(2 * ((y < q) - tau) * (q - y))
    assert crps_quantile(y, u, x) == pytest.approx(grid, abs=1e-5)
    with pytest.raises(ValueError, match="from 0 to 1"):
        crps_quantile(0.5, [0.1, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="from 0 to 1"):
        crps_quantile(0.5, [0.0, 0.9], [0.0, 1.0])
    with pytest.raises(ValueError, match="nondecreasing"):
        crps_quantile(0.5, [0.0, 1.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        crps_quantile(float("nan"), [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        crps_quantile(0.5, [0.0, 1.0], [0.0, float("nan")])


def test_crps_quantile_refusals():
    with pytest.raises(ValueError, match="equal length"):
        crps_quantile(0.5, [0.0, 0.5, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="equal length"):
        crps_quantile(0.5, [0.0], [0.0])
    with pytest.raises(ValueError, match="equal length"):
        crps_quantile(0.5, [[0.0, 1.0]], [[0.0, 1.0]])
    with pytest.raises(ValueError, match="nondecreasing"):
        crps_quantile(0.5, [0.0, 0.6, 0.5, 1.0], [0.0, 0.5, 0.7, 1.0])
    with pytest.raises(ValueError, match="finite"):
        crps_quantile(0.5, [0.0, np.inf], [0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        crps_quantile(0.5, [0.0, 1.0], [0.0, np.inf])


def test_calibrated_exponential_coverage_and_pinball():
    rng = np.random.default_rng(190)
    n = 20000
    y = rng.exponential(1.0, n)
    q = np.tile(-np.log1p(-LEVELS), (n, 1))
    cov = coverage(y, q, LEVELS)
    for name, nominal in (("p25_p75", 0.5), ("p10_p90", 0.8)):
        assert abs(cov[name].mean() - nominal) < 0.015
    for a in LEVELS:
        assert abs(cov[f"le_p{round(a * 100)}"].mean() - a) < 0.015
    assert pinball(y, q, LEVELS).mean() < pinball(y, 1.2 * q, LEVELS).mean()


def test_crossing_rate_ties_mixed_and_empty():
    raw = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [1.0, 2.0, 3.0],
                    [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
    assert crossing_rate(raw) == pytest.approx(0.4)
    assert crossing_rate(np.array([[1.0, 1.0, 2.0], [5.0, 5.0, 5.0]])) == 0.0
    assert np.isnan(crossing_rate(np.empty((0, 3))))
    # Single row: only the last pair (3.0 > 2.5) is out of order.
    assert crossing_rate(np.array([[1.0, 2.0, 3.0, 2.5]])) == pytest.approx(1.0)


def test_level_width_burn_in_and_later_pairs_do_not_count():
    rng = np.random.default_rng(1)
    months = _months(MONTH, MONTH)
    early = _hist(rng.normal(size=499), rng.normal(size=499), "2024-06-01")
    later = _hist(rng.normal(size=300), rng.normal(size=300), "2025-03-15")
    at_start = _hist(rng.normal(size=5), rng.normal(size=5), MONTH)
    L, reason = level_width(months, [0.0, 1.0], *_cat(early, later, at_start))
    assert np.isnan(L).all()
    assert list(reason) == ["level_width_burn_in"] * 2
    # 500 earlier pairs: changing the y of later pairs must not move L.
    early = _hist(rng.normal(size=500), rng.normal(size=500), "2024-06-01")
    later_q, later_y = rng.normal(size=200), rng.normal(size=200)
    later2 = _hist(later_q, later_y * 100 + 7, "2025-03-15")
    L1, _ = level_width(months, [0.0, 0.5], *_cat(early, _hist(later_q, later_y, "2025-03-15")))
    L2, _ = level_width(months, [0.0, 0.5], *_cat(early, later2))
    assert np.isfinite(L1).all()
    np.testing.assert_array_equal(L1, L2)


def test_level_width_each_month_uses_its_own_history():
    rng = np.random.default_rng(3)
    a = _hist(np.full(500, 3.0), rng.normal(0, 1, 500), "2024-12-01")
    b = _hist(np.full(500, 3.0), rng.normal(4, 2, 500), "2025-01-15")
    hq, hy, hl = _cat(a, b)
    L, reason = level_width(_months("2025-01-01", "2025-02-01"), [3.0, 3.0], hq, hy, hl)
    assert list(reason) == ["", ""]
    assert L[0] == pytest.approx(_iqr(hy[:500]))
    assert L[1] == pytest.approx(_iqr(hy))
    assert not np.isclose(L[0], L[1])


def test_level_width_identical_history_is_one_bin():
    y = np.random.default_rng(4).normal(size=600)
    hq, hy, hl = _hist(np.full(600, 3.0), y, "2024-06-01")
    L, reason = level_width(_months(MONTH, MONTH), [3.0, 7.0], hq, hy, hl)
    assert list(reason) == ["", ""]
    np.testing.assert_allclose(L, _iqr(y))


def test_level_width_outside_edges_takes_end_bins():
    rng = np.random.default_rng(5)
    hq = np.linspace(0.0, 10.0, 1000)
    hy = rng.normal(size=1000)
    hl = np.full(1000, np.datetime64("2024-06-01", "ns"))
    L, reason = level_width(_months(*[MONTH] * 4), [-5.0, 0.0, 10.0, 50.0], hq, hy, hl)
    assert list(reason) == [""] * 4
    assert L[0] == L[1]
    assert L[2] == L[3]
    inner = np.unique(np.quantile(hq, np.linspace(0, 1, 11)))[1:-1]
    assert L[0] == pytest.approx(_iqr(hy[hq <= inner[0]]))
    assert L[2] == pytest.approx(_iqr(hy[hq > inner[-1]]))


def test_level_width_edge_value_takes_lower_bin():
    hq = np.arange(501, dtype=float)
    hy = hq ** 2
    hl = np.full(501, np.datetime64("2024-06-01", "ns"))
    inner = np.unique(np.quantile(hq, np.linspace(0, 1, 6)))[1:-1]
    cur = inner[1]
    assert np.searchsorted(inner, cur) == 1
    L, reason = level_width(_months(MONTH), [cur], hq, hy, hl)
    assert reason[0] == ""
    lower = (hq > inner[0]) & (hq <= inner[1])
    upper = (hq > inner[1]) & (hq <= inner[2])
    assert L[0] == pytest.approx(_iqr(hy[lower]), rel=1e-12)
    assert not np.isclose(_iqr(hy[lower]), _iqr(hy[upper]))


def test_level_width_empty_bin():
    # 1000 pairs, 10 bins. Interior edges are 0, 3.0, 109.2, 209.1, so no
    # history point lies in (0, 3.0] and a current q50 of 1.5 lands in an empty bin.
    hq = np.concatenate([[-1.0], np.zeros(699), 10.0 + np.arange(300)])
    hy = np.random.default_rng(6).normal(size=1000)
    hl = np.full(1000, np.datetime64("2024-01-01", "ns"))
    inner = np.unique(np.quantile(hq, np.linspace(0, 1, 11)))[1:-1]
    assert np.searchsorted(inner, 1.5) == 1
    assert not np.any((hq > inner[0]) & (hq <= inner[1]))
    L, reason = level_width(_months("2025-01-01", "2025-01-01"), [1.5, 50.0], hq, hy, hl)
    assert reason[0] == "level_width_empty_bin"
    assert np.isnan(L[0])
    assert reason[1] == "" and np.isfinite(L[1])


def test_level_width_rejects_mismatched_lengths():
    hq, hy, hl = _hist(np.full(600, 3.0), np.zeros(600), "2024-06-01")
    # Current group: month_start has 2 rows but q50 has 1.
    with pytest.raises(ValueError, match="equal length"):
        level_width(_months(MONTH, MONTH), [3.0], hq, hy, hl)
    # History group: history_y has one fewer row than history_q50.
    with pytest.raises(ValueError, match="equal length"):
        level_width(_months(MONTH), [3.0], hq, hy[:-1], hl)


def test_cell_scores_known_answer_single_cell():
    # Hand-computed, not taken from the code under test.
    # q50 ranks = [1,2,3,4,5]; e ranks = [2,1,3,5,4]; score ranks = [1,3,2,5,4].
    # Regress each rank on q50 rank (slope 0.8, mean 3):
    #   e residuals     = [0.6,-1.2,0,1.2,-0.6]
    #   score residuals = [-0.4,0.8,-1.0,1.2,-0.6]
    # Products e_res*score_res = [-0.24,-0.96,0,1.44,0.36], summing to 0.6;
    # sum of squares is 3.6 for each residual vector, so correlation = 0.6 / 3.6 = 1/6.
    # Contributions = residual product / (3.6 / 5) = [-1/3,-4/3,0,2,1/2].
    # Score residual order (i2,i4,i0,i1,i3) gives tercile labels [1,1,0,2,0].
    # Tercile diff uses raw errors: mean raw e of top tercile (i3: 50)
    #              - mean raw e of bottom tercile (i2,i4: (30+40)/2 = 35) = 15.
    rules = Rules(min_cell=5)
    ids = np.arange(5)
    q50 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    e = np.array([20.0, 10.0, 30.0, 50.0, 40.0])  # same ranks as [2,1,3,5,4]
    score = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    per_event, per_cell = cell_scores(ids, ["A"] * 5, e, q50, score, rules)
    assert per_cell.correlation.iloc[0] == pytest.approx(1 / 6)
    assert per_cell.tercile_diff.iloc[0] == pytest.approx(15.0)
    np.testing.assert_allclose(per_event.contribution.to_numpy(),
                               [-1 / 3, -4 / 3, 0.0, 2.0, 0.5], atol=1e-12)
    assert per_event.tercile.to_numpy().astype(int).tolist() == [1, 1, 0, 2, 0]


def test_level_width_nonfinite_history_is_dropped():
    rng = np.random.default_rng(15)
    months = _months(MONTH)
    fin = _hist(rng.normal(size=499), rng.normal(size=499), "2024-06-01")
    # 10 earlier pairs: 5 with NaN history q50, 5 with NaN y. Dropped, so 499 finite is burn-in.
    bad = _hist(np.r_[np.full(5, np.nan), rng.normal(size=5)],
                np.r_[rng.normal(size=5), np.full(5, np.nan)], "2024-06-01")
    L, reason = level_width(months, [0.0], *_cat(fin, bad))
    assert list(reason) == ["level_width_burn_in"]
    # 600 finite earlier pairs: extra non-finite earlier pairs must not change L.
    base = _hist(rng.normal(size=600), rng.normal(size=600), "2024-06-01")
    extra = _hist(np.r_[np.full(3, np.nan), rng.normal(size=4)],
                  np.r_[rng.normal(size=3), np.inf, np.inf, np.nan, np.nan], "2024-06-01")
    L1, _ = level_width(months, [0.0], *base)
    L2, _ = level_width(months, [0.0], *_cat(base, extra))
    assert np.isfinite(L1).all()
    np.testing.assert_array_equal(L1, L2)


def test_level_width_rejects_nat_and_nonfinite_q50():
    hq, hy, hl = _hist(np.full(600, 3.0), np.zeros(600), "2024-06-01")
    with pytest.raises(ValueError, match="NaT"):
        level_width(_months(MONTH, "NaT"), [3.0, 3.0], hq, hy, hl)
    with pytest.raises(ValueError, match="finite"):
        level_width(_months(MONTH), [np.nan], hq, hy, hl)


def test_cell_scores_skip_reasons_ignore_score():
    rng = np.random.default_rng(7)
    ids = np.array([f"e{i:03d}" for i in range(120)])
    cell = np.array(["small"] * 29 + ["flat"] * 30 + ["mono"] * 30 + ["ok"] * 31)
    q50 = rng.normal(size=120)
    e = rng.normal(size=120)
    e[29:59] = 2.0
    e[59:89] = q50[59:89]
    for score in (rng.normal(size=120), rng.normal(size=120)):
        per_event, per_cell = cell_scores(ids, cell, e, q50, score)
        skips = dict(zip(per_cell.cell, per_cell.skip))
        assert skips == {"small": "cell_small", "flat": "constant_e",
                         "mono": "zero_e_residual", "ok": ""}
        assert not per_event.used[:89].any()
        assert per_event.used[89:].all()


def test_cell_scores_score_equal_to_q50_is_zero_and_equal_to_e_is_positive():
    rng = np.random.default_rng(8)
    n = 50
    ids, cell = np.arange(n), ["A"] * n
    q50, e = rng.normal(size=n), rng.normal(size=n)
    per_event, per_cell = cell_scores(ids, cell, e, q50, q50.copy())
    assert per_cell.correlation.iloc[0] == 0.0
    assert per_cell.tercile_diff.iloc[0] == 0.0
    assert per_event.used.all()
    np.testing.assert_array_equal(per_event.contribution.to_numpy(), np.zeros(n))
    _, per_cell = cell_scores(ids, cell, e, q50, e.copy())
    # score equals e, so its residual equals e's residual and the correlation is 1.
    assert per_cell.correlation.iloc[0] == pytest.approx(1.0)
    assert per_cell.tercile_diff.iloc[0] > 0


def test_cell_scores_mean_contribution_is_weighted_correlation():
    rng = np.random.default_rng(9)
    n = 110
    cell = np.array(["A"] * 40 + ["B"] * 70)
    e, q50 = rng.normal(size=n), rng.normal(size=n)
    score = e + rng.normal(size=n)
    per_event, per_cell = cell_scores(np.arange(n), cell, e, q50, score)
    assert (per_cell.skip == "").all()
    expected = np.average(per_cell.correlation.to_numpy(), weights=per_cell.n.to_numpy())
    assert per_event.contribution.mean() == pytest.approx(expected, rel=1e-12)


def test_cell_scores_monotone_in_q50_scores_zero_but_v_shape_does_not():
    rng = np.random.default_rng(10)
    n = 200
    ids, cell = np.arange(n), ["A"] * n
    q50 = rng.uniform(0, 10, n)
    e = np.abs(q50 - 5) + 0.1 * rng.normal(size=n)
    mono = 3.0 * q50 + 1.0
    v_shape = np.abs(q50 - 5) + 0.1 * rng.normal(size=n)
    assert cell_scores(ids, cell, e, q50, mono)[1].correlation.iloc[0] == 0.0
    assert cell_scores(ids, cell, e, q50, v_shape)[1].correlation.iloc[0] > 0.5


def test_cell_scores_string_ids_and_tercile_sizes():
    rng = np.random.default_rng(11)
    n = 31
    ids = np.array([f"ev-{i:02d}" for i in range(n)])
    e, q50, score = rng.normal(size=(3, n))
    per_event, per_cell = cell_scores(ids, ["A"] * n, e, q50, score)
    assert per_cell.skip.iloc[0] == ""
    assert per_event.used.all()
    assert np.bincount(per_event.tercile.to_numpy().astype(int)).tolist() == [11, 10, 10]


def test_cell_scores_tie_break_by_event_id():
    # score = q50 gives zero score residual, so tercile labels fall back to event id.
    rng = np.random.default_rng(13)
    n = 31
    ids = np.array([f"ev-{i:02d}" for i in range(n)])[rng.permutation(n)]
    e, q50 = rng.normal(size=(2, n))
    per_event, _ = cell_scores(ids, ["A"] * n, e, q50, q50.copy())
    num = np.array([int(s[3:]) for s in ids])
    expected = np.where(num <= 10, 0, np.where(num <= 20, 1, 2))
    np.testing.assert_array_equal(per_event.tercile.to_numpy().astype(int), expected)


def test_cell_scores_constant_q50_still_scores():
    rng = np.random.default_rng(14)
    n = 40
    e = rng.normal(size=n)
    _, per_cell = cell_scores(np.arange(n), ["A"] * n, e, np.ones(n), e.copy())
    assert per_cell.skip.iloc[0] == ""
    assert per_cell.correlation.iloc[0] == pytest.approx(1.0)


def test_cell_scores_validation():
    rng = np.random.default_rng(12)
    n = 40
    ids, cell = np.arange(n), ["A"] * n
    e, q50, score = rng.normal(size=(3, n))
    with pytest.raises(ValueError, match="equal length"):
        cell_scores(ids[:-1], cell, e, q50, score)
    bad = e.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cell_scores(ids, cell, bad, q50, score)
    with pytest.raises(ValueError, match="finite"):
        cell_scores(ids, cell, e, q50, np.where(np.arange(n) == 0, np.inf, score))
