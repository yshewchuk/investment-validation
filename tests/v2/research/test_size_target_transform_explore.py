import numpy as np
import pandas as pd

from experiments.size_target_transform_explore import history_stats, inverse, transform


def test_inverse_transforms_round_trip():
    y = np.array([0.4, 2.0, 12.0])
    mu = np.array([0.2, 1.0, 5.0])
    sigma = np.array([0.5, 1.5, 3.0])
    for kind in ("raw", "residual", "z", "log_ratio"):
        np.testing.assert_allclose(inverse(transform(y, mu, sigma, kind), mu, sigma, kind), y)


def test_history_is_strictly_prior_and_sigma_floor_uses_training():
    dates = pd.date_range("2024-01-01", periods=5, freq="90D")
    ids = [f"AAA_{day:%Y-%m-%d}" for day in dates]
    frame = pd.DataFrame({"ticker": "AAA", "date": dates, "abs_move": [2., 4., 6., 8., 200.], "event_id": ids})
    mu, sigma, floor = history_stats(frame, set(ids[:4]))
    assert mu.tolist() == [0., 2., 3., 4., 5.]
    assert np.isclose(floor, 0.25)
    assert sigma[0] == floor and sigma[1] == floor
    assert np.isclose(sigma[4], np.std([2., 4., 6., 8.], ddof=0))
