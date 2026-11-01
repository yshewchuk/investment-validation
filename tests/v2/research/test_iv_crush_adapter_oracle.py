"""Independent nonconstant signed oracle through the actual IV worker adapter."""
import pytest

from engine.v2.models.regression_candidate import regression_candidate_recipe
from experiments import magnitude_search_worker as worker


@pytest.mark.parametrize("alpha", [0.1, 1.0])
def test_signed_adapter_preserves_ridge_values_event_order_and_horizons(alpha):
    horizons = [5, 3, 1]
    indexed = {}
    for horizon in horizons:
        offset = 10. * horizon
        for event, x in (("train-a", 0.), ("train-z", 2.),
                         ("eval-a", -1. - horizon), ("eval-z", 3. + horizon)):
            indexed[event, horizon] = {"event_id": event, "context_id": "synthetic",
                "horizon_sessions": horizon, "x": offset + x}
    training = [{"event_id": "train-z", "crush_pct_iv30": 20.},
                {"event_id": "train-a", "crush_pct_iv30": -140.}]
    recipe = regression_candidate_recipe("ridge", {"alpha": alpha}, target_domain="signed",
        columns=["x"], feature_dictionary_hash="sha256:" + "a" * 64)
    request = {"execution": {"target": {"column": "crush_pct_iv30", "domain": "signed"},
        "features": ["x"], "fit_recipe": recipe}, "inner": {"evaluation_event_ids": ["eval-z", "eval-a"]},
        "plan": {"horizons": horizons}}
    progress, diagnostics = [], []
    predictions = worker.regression_forecasts(indexed, training, request,
        lambda state, fit: progress.append((state, fit)), diagnostics)
    # Training standardizes to [+1, -1], y mean=-60, X'y=160, X'X=2.
    # Ridge slope=160/(2+alpha); evaluation centers differ across horizons.
    expected = {(event, horizon): -60. + centered * 160. / (2. + alpha)
        for horizon in horizons for event, centered in (("eval-z", 2. + horizon), ("eval-a", -2. - horizon))}
    assert predictions == pytest.approx(expected)
    assert len(predictions) == 6 and len(set(predictions.values())) == 6
    assert min(predictions.values()) < -140. < 0. < max(predictions.values())
    assert [(state, fit["horizon_sessions"]) for state, fit in progress] == [
        (state, horizon) for horizon in horizons for state in ("started", "completed")]
    assert all(fit["training_event_ids"] == ["train-z", "train-a"]
        and fit["evaluation_event_ids"] == ["eval-z", "eval-a"] for _, fit in progress)
    assert [row["horizon_sessions"] for row in diagnostics] == horizons
