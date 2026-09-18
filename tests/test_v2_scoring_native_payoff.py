"""Native payoff-calibration/model layer: parity, causality and behavior.

Legacy reference: ``engine/score.py::Scorer._score_model`` (:2067-2218),
``engine/payoff.py`` (``fit_payoff``, ``PayoffMap``, ``simulate_returns``)
and ``engine/models/registry.py`` (``bucket_residuals``,
``ModelArtifact.residual_pool``/``residual_draws``).

Coordinator condition (2026-09-18): a parity test must import legacy
``engine.payoff``/``engine.models.registry`` directly and prove the native
reimplementation (fit, decile bucketing, draw order, same seed) agrees on
the same rows and seed. ``engine/v2`` code itself never imports legacy --
only this test file does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.registry import ModelArtifact, bucket_residuals
from engine.payoff import PayoffMap, fit_payoff, simulate_returns
from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application, native_payoff
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs

# ---------------------------------------------------------------------------
# Parity against legacy's own functions
# ---------------------------------------------------------------------------


def _synthetic_trades(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.0, 8.0, size=n)
    spot = np.full(n, 100.0)
    noise = rng.normal(0.0, 0.01, size=n)
    exit_value = (0.02 + 0.004 * driver + noise) * spot
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "strategy": "STR-THRU",
        "fill_alpha": 0.5,
        "abs_move": driver,
        "spot_entry": spot,
        "exit_value": exit_value,
        "exit_date": dates,
    })


def _rows_from_trades(trades: pd.DataFrame) -> list[dict]:
    return [
        {
            "driver": float(row.abs_move),
            "spot_entry": float(row.spot_entry),
            "exit_value": float(row.exit_value),
            "exit_date": str(row.exit_date.date()),
        }
        for row in trades.itertuples()
    ]


def test_fit_payoff_line_matches_legacy_small_sample_and_cutoff():
    trades = _synthetic_trades(300, seed=7)
    rows = _rows_from_trades(trades)
    cutoff = trades["exit_date"].iloc[250]

    legacy = fit_payoff(trades, "STR-THRU", alpha=0.5, before=cutoff)
    native = native_payoff.fit_payoff_line(rows, before=str(cutoff.date()))

    assert native is not None
    assert native["n"] == legacy.n == 250
    assert native["intercept"] == pytest.approx(legacy.intercept, rel=1e-12)
    assert native["slope"] == pytest.approx(legacy.slope, rel=1e-12)
    assert native["resid_sd"] == pytest.approx(legacy.resid_sd, rel=1e-12)
    assert native["r"] == pytest.approx(legacy.r, rel=1e-12)
    np.testing.assert_allclose(native["residuals"], legacy.residuals)


def test_fit_payoff_line_matches_legacy_residual_capping_above_5000():
    trades = _synthetic_trades(6000, seed=11)
    rows = _rows_from_trades(trades)

    legacy = fit_payoff(trades, "STR-THRU", alpha=0.5)
    native = native_payoff.fit_payoff_line(rows)

    assert legacy.n == 6000 > native_payoff.MAX_RESIDUALS
    assert native is not None
    assert native["residuals"].size == native_payoff.MAX_RESIDUALS == legacy.residuals.size
    np.testing.assert_array_equal(native["residuals"], legacy.residuals)


def test_bucket_residual_pool_matches_legacy_registry_decile_bucketing():
    rng = np.random.default_rng(3)
    n = 3000
    predictions = rng.normal(5.0, 2.0, size=n)
    residuals = rng.normal(0.0, 1.0, size=n)

    legacy_buckets = bucket_residuals(predictions, residuals, deciles=10, min_pool=250)
    native_buckets = native_payoff.bucket_residual_pool(
        predictions, residuals, deciles=10, min_pool=250,
    )

    assert legacy_buckets is not None and native_buckets is not None
    np.testing.assert_allclose(native_buckets["edges"], legacy_buckets["edges"])
    assert len(native_buckets["pools"]) == len(legacy_buckets["pools"])
    for native_pool, legacy_pool in zip(native_buckets["pools"], legacy_buckets["pools"]):
        np.testing.assert_array_equal(native_pool, legacy_pool)

    artifact = ModelArtifact(
        model=None, role="size", features=(), residuals=residuals,
        target="abs_move", residual_buckets=legacy_buckets,
    )
    flat = np.asarray(residuals, dtype=float)
    for point in (1.0, 5.0, 9.0):
        legacy_pool, _ = artifact.residual_pool(prediction=point)
        native_pool, _ = native_payoff.residual_pool_for(native_buckets, point, flat)
        np.testing.assert_array_equal(native_pool, legacy_pool)


def test_simulate_model_returns_matches_legacy_draw_order_and_seed():
    """Proves the fit -> draw -> return pipeline is bit-identical, including
    draw ORDER (model residual pool first, payoff residuals second, one
    shared rng -- engine/score.py:2164-2208)."""
    driver_pool = np.array([-0.5, 0.0, 0.5, 1.0, -1.0])
    payoff_residuals = np.array([-0.02, 0.0, 0.01, 0.03, -0.01, 0.02])
    intercept, slope = 0.02, 0.004
    point, spot, cost, draws = 7.0, 100.0, 4.0, 500

    payoff = PayoffMap(
        strategy="STR-THRU", driver="abs_move", alpha=0.5,
        intercept=intercept, slope=slope,
        resid_sd=float(payoff_residuals.std()), n=250, r=0.9,
        residuals=payoff_residuals,
    )

    seed = 12345
    legacy_rng = np.random.default_rng(seed)
    legacy_draws = point + legacy_rng.choice(driver_pool, size=draws, replace=True)
    legacy_noise = payoff.residual_draws(draws, legacy_rng)
    legacy_returns = simulate_returns(legacy_draws, payoff, spot, cost, legacy_noise)

    native_rng = np.random.default_rng(seed)
    native_returns = native_payoff.simulate_model_returns(
        point, driver_pool, payoff_residuals, intercept, slope, spot, cost,
        draws, native_rng,
    )

    np.testing.assert_array_equal(native_returns, legacy_returns)

    # Reordering the two draws (payoff first, model second) must NOT match --
    # otherwise this test could not tell a real order bug from an accident.
    reordered_rng = np.random.default_rng(seed)
    reordered_noise = payoff.residual_draws(draws, reordered_rng)
    reordered_draws = point + reordered_rng.choice(driver_pool, size=draws, replace=True)
    reordered_returns = simulate_returns(reordered_draws, payoff, spot, cost, reordered_noise)
    assert not np.array_equal(native_returns, reordered_returns)


# ---------------------------------------------------------------------------
# Causality: the payoff fit only ever uses rows before the decision cutoff
# ---------------------------------------------------------------------------


def test_causal_cutoff_excludes_a_post_cutoff_row():
    good_rows = [
        {"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0, "exit_date": "2026-09-01"},
        {"driver": 10.0, "spot_entry": 100.0, "exit_value": 6.0, "exit_date": "2026-09-01"},
    ]
    future_row = {"driver": 5.0, "spot_entry": 100.0, "exit_value": 999.0,
                  "exit_date": "2026-09-20"}
    cutoff = "2026-09-16"

    baseline = native_payoff.fit_payoff_line(good_rows, before=cutoff, min_trades=2)
    with_future_row = native_payoff.fit_payoff_line(
        good_rows + [future_row], before=cutoff, min_trades=2,
    )

    assert baseline is not None and with_future_row is not None
    assert with_future_row["n"] == baseline["n"] == 2
    assert with_future_row["intercept"] == baseline["intercept"]
    assert with_future_row["slope"] == baseline["slope"]
    np.testing.assert_array_equal(with_future_row["residuals"], baseline["residuals"])

    # Sanity: without the cutoff the same row DOES change the fit -- proving
    # the equality above is the causal filter doing real work, not a no-op
    # (e.g. a `before` argument that was silently ignored).
    without_cutoff = native_payoff.fit_payoff_line(
        good_rows + [future_row], before=None, min_trades=2,
    )
    assert without_cutoff["n"] == 3
    assert without_cutoff["slope"] != baseline["slope"]


# ---------------------------------------------------------------------------
# Native stage behavior, end to end through build_native_score_inputs
# ---------------------------------------------------------------------------


def _request(strategy: str = "STR-THRU") -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-model", calendar_revision="cal-1", strategy_version=strategy,
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5},
    )


_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
    ("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
}
_GOOD_PAYOFF_ROWS = [
    {"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0, "exit_date": "2026-09-01"},
    {"driver": 10.0, "spot_entry": 100.0, "exit_value": 6.0, "exit_date": "2026-09-01"},
]
_ZERO_MODEL_RESIDUAL_ROWS = [{"prediction": 7.0, "residual": 0.0}]


def _bundle(**overrides) -> SourceBundle:
    base: dict = dict(
        source_ref="native-model-test",
        context={
            "ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
            "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0,
        },
        raw_quotes=_QUOTES,
        feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "m1"}},
        forecast_recipes={"driver_prediction": {"intercept": 7.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:m1"},
        residual_recipe={
            "terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
            "capital_at_risk": 1.0,
        },
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
    )
    base.update(overrides)
    return SourceBundle(**base)


def test_model_number_computed_from_answer_free_inputs():
    # Same exact-fit-through-two-points + single-valued residual pools as
    # checks/phase4_real.py's control: exp_pnl_model/win_model are
    # closed-form (see that file for the by-hand derivation), so this proves
    # a REAL computation, not a copied placeholder.
    bundle = _bundle(
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 16},
        payoff_source_rows=_GOOD_PAYOFF_ROWS,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request(), inputs)

    assert record.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
    assert record.resolved_request["win_model"] == pytest.approx(1.0)
    assert "NO_PAYOFF_MAP" not in record.reason_codes
    assert record.validation_status == "scored"


def test_no_payoff_map_for_a_strategy_without_a_driver():
    # TWIN-P is absent from legacy's PAYOFF_DRIVER (engine/payoff.py:79-82);
    # driver_for() would raise for it. A declared payoff_recipe for it must
    # still refuse with NO_PAYOFF_MAP, exactly as legacy's _score_model does
    # before ever touching a model or a trade.
    bundle = _bundle(
        strategy="TWIN-P",
        forecast_recipes={"forecast_abs_move": {"intercept": 7.0, "coefficients": {}}},
        model_artifact_refs={"forecast_abs_move": "sha256:m1"},
        payoff_recipe={"min_trades": 2, "seed": 42},
        payoff_source_rows=_GOOD_PAYOFF_ROWS,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request("TWIN-P"), inputs)

    assert "NO_PAYOFF_MAP" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None


def test_missing_payoff_rows_flags_no_payoff_map_without_a_number():
    bundle = _bundle(
        payoff_recipe={"min_trades": 200, "seed": 42},  # only 2 rows below
        payoff_source_rows=_GOOD_PAYOFF_ROWS,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request(), inputs)

    assert "NO_PAYOFF_MAP" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_missing_model_residual_rows_flags_and_withholds_the_number():
    bundle = _bundle(
        payoff_recipe={"min_trades": 2, "seed": 42},
        payoff_source_rows=_GOOD_PAYOFF_ROWS,
        model_residual_rows=[],
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request(), inputs)

    assert "MISSING_MODEL_RESIDUALS" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_end_to_end_causal_cutoff_excludes_a_post_cutoff_payoff_row():
    future_row = {"driver": 5.0, "spot_entry": 100.0, "exit_value": 999.0,
                  "exit_date": "2026-09-20"}
    bundle = _bundle(
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 16,
                       "before": "2026-09-16"},
        payoff_source_rows=_GOOD_PAYOFF_ROWS + [future_row],
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request(), inputs)

    # Same closed-form answer as the two-row-only case: the future row was
    # excluded, not silently included.
    assert record.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
    assert record.resolved_request["win_model"] == pytest.approx(1.0)


def test_poisoned_model_block_answers_are_not_preserved():
    """The model stage must recompute, never copy a pre-supplied answer."""
    bundle = _bundle(
        payoff_recipe={"min_trades": 2, "seed": 42},
        payoff_source_rows=_GOOD_PAYOFF_ROWS,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    from dataclasses import replace

    poisoned = replace(inputs, model={"exp_pnl_model": 991.5, "win_model": 0.0})
    record = application.score_one(_request(), poisoned)

    assert record.resolved_request.get("exp_pnl_model") != 991.5
    assert record.resolved_request.get("win_model") != 0.0
    assert "UNOWNED_MODEL_OUTPUT" in record.reason_codes
    assert record.validation_status == "refused"
