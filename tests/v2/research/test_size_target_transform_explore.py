import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from experiments.size_target_transform_explore import (
    block_bootstrap_ci, distribution_metrics, history_stats, inverse,
    ridge_loo_residuals, score_metrics, transform, validate_panel_population,
)


def test_inverse_transforms_round_trip():
    y, mu, sigma = np.array([.4, 2., 12.]), np.array([0., 1., 5.]), np.array([.5, 1.5, 3.])
    expected = {"raw": y, "residual": np.array([.4, 1., 7.]), "z": np.array([.8, 2/3, 7/3]), "log_ratio": np.log((y+1e-6)/(mu+1e-6))}
    for kind in ("raw", "residual", "z", "log_ratio"):
        np.testing.assert_allclose(transform(y, mu, sigma, kind), expected[kind])
        np.testing.assert_allclose(inverse(transform(y, mu, sigma, kind), mu, sigma, kind), y)


def test_history_is_strictly_prior_and_sigma_floor_uses_training():
    dates = pd.date_range("2024-01-01", periods=5, freq="90D"); ids = [f"AAA_{d:%Y-%m-%d}" for d in dates]
    frame = pd.DataFrame({"ticker": "AAA", "date": dates, "move": [2., -4., 6., -8., 200.], "abs_move": [2., 4., 6., 8., 200.], "event_id": ids})
    mu, sigma, floor, features = history_stats(frame, set(ids[:4]))
    assert mu.tolist() == [0., 2., 3., 4., 5.]
    assert np.isclose(floor, .25) and sigma[0] == floor and sigma[1] == floor
    assert np.isclose(sigma[4], np.std([2., 4., 6., 8.], ddof=0))
    assert features.loc[4, "n_prior"] == 4 and features.loc[4, "mean_prior_move"] == -1.


def test_loo_ridge_residual_matches_direct_refits():
    x = np.array([[0., 1.], [1., 0.], [2., 1.], [3., 3.], [4., 2.]])
    y = np.array([1., 2., 2.5, 4., 3.]); fitted = Ridge(alpha=1).fit(x, y).predict(x)
    actual = ridge_loo_residuals(y, fitted, x)
    expected = np.array([y[i]-Ridge(alpha=1).fit(np.delete(x, i, 0), np.delete(y, i)).predict(x[i:i+1])[0] for i in range(len(y))])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_bootstrap_resamples_unequal_quarter_ticker_blocks():
    ci = block_bootstrap_ci(np.zeros(4), np.array([0., 1., 3., 6.]), np.zeros(4), np.array([1, 1, 1, 2]), np.array(["A", "A", "B", "A"]))
    assert ci[0][0] < ci[0][1] and ci[1][0] < ci[1][1]


def test_scores_bootstrap_and_distribution_match_oracles():
    y, pred, base = np.array([1., 4.]), np.array([2., 2.]), np.array([1., 1.])
    mae, rmse = score_metrics(y, pred); assert np.isclose(mae, 1.5) and np.isclose(rmse, np.sqrt(2.5))
    bmae, brmse = score_metrics(y, base); ci = block_bootstrap_ci(y, pred, base, np.array([1, 1]), np.array(["A", "A"]))
    np.testing.assert_allclose(ci, [[mae-bmae, mae-bmae], [rmse-brmse, rmse-brmse]])
    p, residual, scale, actual = np.array([.1, 5.]), np.array([-1., 0., 2.]), np.array([1., .5]), np.array([9., 9.])
    got = distribution_metrics(p, residual, scale, actual)
    raw = p[:, None]+np.random.default_rng(72).choice(residual, (2, 200), replace=True)*scale[:, None]; draws = np.maximum(raw, 0.)
    assert raw.min() < 0. and draws.min() == 0.
    for c in (.5, .8, .9):
        lo, hi = np.quantile(draws, [(1-c)/2, (1+c)/2], axis=1)
        assert np.isclose(got[str(int(c*100))]["coverage"], np.mean((actual>=lo)&(actual<=hi)))
        assert np.isclose(got[str(int(c*100))]["mean_width"], np.mean(hi-lo))
    assert got["50"]["coverage"] < 1.
    crps = [np.mean(abs(row-target))-.5*np.mean(abs(row[:, None]-row[None, :])) for row, target in zip(draws, actual)]
    assert np.isclose(got["crps"], np.mean(crps))


def test_missing_training_panel_event_is_tolerated_below_two_percent():
    retained = {f"e{i}" for i in range(100)}; test = {"e99"}; panel = retained-{"e0"}
    assert validate_panel_population(retained, test, panel, False) == {"e0"}
    assert validate_panel_population(retained, test, panel, False, {"e1"}) == {"e0", "e1"}
    with pytest.raises(ValueError, match="panel population mismatch"): validate_panel_population(retained, test, panel-test, False)
    with pytest.raises(ValueError, match="panel population mismatch"): validate_panel_population(retained, test, panel, False, {"e1", "e2"})
    with pytest.raises(ValueError, match="panel population mismatch"): validate_panel_population(retained, test, retained-{"e0", "e1", "e2"}, False)
    with pytest.raises(ValueError, match="panel population mismatch"): validate_panel_population(retained, test, panel, True)
