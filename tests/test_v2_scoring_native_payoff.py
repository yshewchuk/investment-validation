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
from engine.payoff import (
    PayoffMap,
    RunupPayoffSurface,
    fit_payoff,
    fit_runup_payoff,
    scale_runup_move as legacy_scale_runup_move,
    simulate_returns,
    simulate_runup_returns,
)
from engine.payoff import runup_payoff_design as legacy_runup_payoff_design
from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application, native_payoff, stages
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


# ---------------------------------------------------------------------------
# STR-RUNUP -- the two-driver payoff surface (R4-17)
# ---------------------------------------------------------------------------


def _synthetic_runup_trades(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    implied = rng.uniform(0.0, 8.0, size=n)
    spot_entry = np.full(n, 100.0)
    signed_move = rng.normal(0.0, 3.0, size=n)
    spot_exit = spot_entry * np.exp(signed_move / 100.0)
    strike = np.full(n, 100.0)
    moneyness = 100.0 * np.log(spot_exit / strike)
    design = legacy_runup_payoff_design(implied, moneyness)
    coeffs = np.array([0.02, 0.004, 0.001, 0.0005, 0.0007, 0.0003])
    noise = rng.normal(0.0, 0.002, size=n)
    target = design @ coeffs + noise
    exit_value = target * spot_entry
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "strategy": "STR-RUNUP",
        "fill_alpha": 0.5,
        "im_t1": implied,
        "spot_entry": spot_entry,
        "spot_exit": spot_exit,
        "strike": strike,
        "exit_value": exit_value,
        "exit_date": dates,
    })


def _runup_rows_from_trades(trades: pd.DataFrame) -> list[dict]:
    return [
        {
            "driver": float(row.im_t1),
            "spot_entry": float(row.spot_entry),
            "spot_exit": float(row.spot_exit),
            "strike": float(row.strike),
            "exit_value": float(row.exit_value),
            "exit_date": str(row.exit_date.date()),
        }
        for row in trades.itertuples()
    ]


def test_fit_runup_payoff_surface_matches_legacy_small_sample_and_cutoff():
    trades = _synthetic_runup_trades(300, seed=7)
    rows = _runup_rows_from_trades(trades)
    cutoff = trades["exit_date"].iloc[250]

    legacy = fit_runup_payoff(trades, alpha=0.5, before=cutoff)
    native = native_payoff.fit_runup_payoff_surface(rows, before=str(cutoff.date()))

    assert native is not None
    assert native["n"] == legacy.n == 250
    np.testing.assert_allclose(native["coefficients"], legacy.coefficients, rtol=1e-10)
    assert native["resid_sd"] == pytest.approx(legacy.resid_sd, rel=1e-10)
    assert native["r"] == pytest.approx(legacy.r, rel=1e-10)
    np.testing.assert_allclose(native["residuals"], legacy.residuals)


def test_fit_runup_payoff_surface_matches_legacy_residual_capping_above_5000():
    trades = _synthetic_runup_trades(6000, seed=11)
    rows = _runup_rows_from_trades(trades)

    legacy = fit_runup_payoff(trades, alpha=0.5)
    native = native_payoff.fit_runup_payoff_surface(rows)

    assert legacy.n == 6000 > native_payoff.MAX_RESIDUALS
    assert native is not None
    assert native["residuals"].size == native_payoff.MAX_RESIDUALS == legacy.residuals.size
    np.testing.assert_array_equal(native["residuals"], legacy.residuals)


def test_runup_exit_value_per_spot_matches_legacy_surface():
    coefficients = (0.02, 0.004, 0.001, 0.0005, 0.0007, 0.0003)
    payoff = RunupPayoffSurface(
        alpha=0.5, coefficients=coefficients, resid_sd=0.01, n=250, r=0.9,
    )
    implied = np.array([1.0, 3.0, 5.0, 7.0])
    signed_move = np.array([-2.0, 0.0, 1.5, 4.0])
    spot, strike = 123.0, 118.0

    legacy = payoff.value_per_spot(implied, signed_move, spot=spot, strike=strike)
    native = native_payoff.runup_exit_value_per_spot(
        implied, signed_move, coefficients, spot=spot, strike=strike,
    )
    np.testing.assert_allclose(native, legacy, rtol=1e-12)

    # Broadcasting a scalar move across several implied-move draws (the shape
    # simulate_runup_model_returns actually calls it with) must also agree.
    legacy_scalar = payoff.value_per_spot(implied, 0.0, spot=spot, strike=strike)
    native_scalar = native_payoff.runup_exit_value_per_spot(
        implied, 0.0, coefficients, spot=spot, strike=strike,
    )
    np.testing.assert_allclose(native_scalar, legacy_scalar, rtol=1e-12)


def test_simulate_runup_model_returns_matches_legacy_draw_order_and_seed():
    """Proves the fit -> draw -> return pipeline is bit-identical, including
    draw ORDER (implied pool, then move pool, then the sign draw, then the
    payoff surface's own residuals -- engine/score.py:2385-2421)."""
    implied_pool = np.array([-0.5, 0.0, 0.5, 1.0, -1.0])
    move_pool = np.array([-0.2, 0.1, 0.0, 0.3, -0.1])
    payoff_residuals = np.array([-0.02, 0.0, 0.01, 0.03, -0.01, 0.02])
    coefficients = (0.02, 0.004, 0.001, 0.0005, 0.0007, 0.0003)
    point_implied, point_move_d14 = 7.0, 3.0
    spot, strike, cost, draws, days = 100.0, 100.0, 4.0, 500, 7.0

    payoff = RunupPayoffSurface(
        alpha=0.5, coefficients=coefficients,
        resid_sd=float(payoff_residuals.std()), n=250, r=0.9,
        residuals=payoff_residuals,
    )

    seed = 12345
    legacy_rng = np.random.default_rng(seed)
    implied_draws = point_implied + legacy_rng.choice(implied_pool, size=draws, replace=True)
    implied_draws = np.maximum(implied_draws, 0.0)
    move_draws_d14 = point_move_d14 + legacy_rng.choice(move_pool, size=draws, replace=True)
    move_draws = legacy_scale_runup_move(np.maximum(move_draws_d14, 0.0), days)
    signed_moves = legacy_rng.choice((-1.0, 1.0), size=draws) * move_draws
    legacy_noise = payoff.residual_draws(draws, legacy_rng)
    legacy_returns = simulate_runup_returns(
        implied_draws, signed_moves, payoff, spot=spot, strike=strike, cost=cost,
        payoff_noise=legacy_noise,
    )

    native_rng = np.random.default_rng(seed)
    native_returns = native_payoff.simulate_runup_model_returns(
        point_implied, point_move_d14, implied_pool, move_pool, coefficients,
        payoff_residuals, spot, strike, cost, days, draws, native_rng,
    )

    np.testing.assert_array_equal(native_returns, legacy_returns)

    # Reordering the four draws must NOT match -- otherwise this test could
    # not tell a real order bug from an accident.
    reordered_rng = np.random.default_rng(seed)
    reordered_noise = payoff.residual_draws(draws, reordered_rng)
    reordered_implied = point_implied + reordered_rng.choice(implied_pool, size=draws, replace=True)
    reordered_implied = np.maximum(reordered_implied, 0.0)
    reordered_move_d14 = point_move_d14 + reordered_rng.choice(move_pool, size=draws, replace=True)
    reordered_move = legacy_scale_runup_move(np.maximum(reordered_move_d14, 0.0), days)
    reordered_signed = reordered_rng.choice((-1.0, 1.0), size=draws) * reordered_move
    reordered_returns = simulate_runup_returns(
        reordered_implied, reordered_signed, payoff, spot=spot, strike=strike,
        cost=cost, payoff_noise=reordered_noise,
    )
    assert not np.array_equal(native_returns, reordered_returns)


# ---------------------------------------------------------------------------
# Causality: the runup payoff fit only ever uses rows before the decision
# cutoff
# ---------------------------------------------------------------------------


def test_runup_causal_cutoff_excludes_a_post_cutoff_row():
    good_rows = [
        {"driver": 0.0, "spot_entry": 100.0, "spot_exit": 100.0, "strike": 100.0,
         "exit_value": 2.0, "exit_date": "2026-09-01"},
        {"driver": 10.0, "spot_entry": 100.0, "spot_exit": 100.0, "strike": 100.0,
         "exit_value": 6.0, "exit_date": "2026-09-01"},
    ]
    future_row = {"driver": 5.0, "spot_entry": 100.0, "spot_exit": 100.0,
                  "strike": 100.0, "exit_value": 999.0, "exit_date": "2026-09-20"}
    cutoff = "2026-09-16"

    baseline = native_payoff.fit_runup_payoff_surface(good_rows, before=cutoff, min_trades=2)
    with_future_row = native_payoff.fit_runup_payoff_surface(
        good_rows + [future_row], before=cutoff, min_trades=2,
    )

    assert baseline is not None and with_future_row is not None
    assert with_future_row["n"] == baseline["n"] == 2
    np.testing.assert_array_equal(with_future_row["coefficients"], baseline["coefficients"])
    np.testing.assert_array_equal(with_future_row["residuals"], baseline["residuals"])

    without_cutoff = native_payoff.fit_runup_payoff_surface(
        good_rows + [future_row], before=None, min_trades=2,
    )
    assert without_cutoff["n"] == 3
    assert without_cutoff["coefficients"] != baseline["coefficients"]


# ---------------------------------------------------------------------------
# Native stage behavior, end to end through build_native_score_inputs
# ---------------------------------------------------------------------------


def _runup_request() -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-runup-model", calendar_revision="cal-1",
        strategy_version="STR-RUNUP", deployment_id="dep-1",
        decision_clock_id="entry-close", requested_decision_at="2026-09-16",
        snapshot_id="snap-1", mode="replay", fill_model={"alpha": 0.5},
    )


_GOOD_RUNUP_PAYOFF_ROWS = [
    {"driver": 0.0, "spot_entry": 100.0, "spot_exit": 100.0, "strike": 100.0,
     "exit_value": 2.0, "exit_date": "2026-09-01"},
    {"driver": 10.0, "spot_entry": 100.0, "spot_exit": 100.0, "strike": 100.0,
     "exit_value": 6.0, "exit_date": "2026-09-01"},
]


def _runup_bundle(**overrides) -> SourceBundle:
    base: dict = dict(
        source_ref="native-runup-model-test",
        context={
            "ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
            "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0,
            "strike": 100.0, "days_before_print": 7.0,
        },
        raw_quotes=_QUOTES,
        feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "m1"}},
        forecast_recipes={
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
            "runup_move_prediction": {"intercept": 0.0, "coefficients": {}},
        },
        model_artifact_refs={
            "driver_prediction": "sha256:m1", "runup_move_prediction": "sha256:m2",
        },
        residual_recipe={
            "terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
            "capital_at_risk": 1.0,
        },
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        strategy="STR-RUNUP",
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 16},
        payoff_source_rows=_GOOD_RUNUP_PAYOFF_ROWS,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
        runup_move_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    base.update(overrides)
    return SourceBundle(**base)


def test_runup_model_number_computed_from_answer_free_inputs():
    # Both drivers' forecasts (7.0, 0.0), spot == strike and a zero-residual
    # move pool degenerate the two-driver surface onto the same fitted line
    # as STR-THRU's own closed-form control (0.02 + 0.004*driver): exit
    # value / spot = 0.02 + 0.004*7 = 0.048, so exp_pnl_model =
    # (0.048*100 - 4.0) / 4.0 == 0.2, win_model == 1.0.
    bundle = _runup_bundle()
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_runup_request(), inputs)

    assert record.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
    assert record.resolved_request["win_model"] == pytest.approx(1.0)
    assert "NO_PAYOFF_MAP" not in record.reason_codes
    assert record.validation_status == "scored"


def test_runup_no_payoff_map_without_source_rows():
    bundle = _runup_bundle(
        payoff_recipe={"min_trades": 200, "seed": 42},  # only 2 rows below
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_runup_request(), inputs)

    assert "NO_PAYOFF_MAP" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_runup_missing_implied_residual_rows_flags_and_withholds_the_number():
    bundle = _runup_bundle(model_residual_rows=[])
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_runup_request(), inputs)

    assert "MISSING_MODEL_RESIDUALS" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_runup_missing_move_residual_rows_flags_and_withholds_the_number():
    bundle = _runup_bundle(runup_move_residual_rows=[])
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_runup_request(), inputs)

    assert "MISSING_MODEL_RESIDUALS" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_runup_missing_days_before_print_flags_and_withholds_the_number():
    bundle = _runup_bundle(context={
        "ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
        "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0,
        "strike": 100.0,
    })
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_runup_request(), inputs)

    assert "MISSING_MODEL_INPUT:days_before_print" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_runup_end_to_end_causal_cutoff_excludes_a_post_cutoff_payoff_row():
    future_row = {"driver": 5.0, "spot_entry": 100.0, "spot_exit": 100.0,
                  "strike": 100.0, "exit_value": 999.0, "exit_date": "2026-09-20"}
    bundle = _runup_bundle(
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 16,
                       "before": "2026-09-16"},
        payoff_source_rows=_GOOD_RUNUP_PAYOFF_ROWS + [future_row],
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_runup_request(), inputs)

    # Same closed-form answer as the two-row-only case: the future row was
    # excluded, not silently included.
    assert record.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
    assert record.resolved_request["win_model"] == pytest.approx(1.0)


def test_runup_poisoned_model_block_answers_are_not_preserved():
    """The model stage must recompute, never copy a pre-supplied answer."""
    bundle = _runup_bundle()
    inputs = build_native_score_inputs(bundle)
    from dataclasses import replace

    poisoned = replace(inputs, model={"exp_pnl_model": 991.5, "win_model": 0.0})
    record = application.score_one(_runup_request(), poisoned)

    assert record.resolved_request.get("exp_pnl_model") != 991.5
    assert record.resolved_request.get("win_model") != 0.0
    assert "UNOWNED_MODEL_OUTPUT" in record.reason_codes
    assert record.validation_status == "refused"


# ---------------------------------------------------------------------------
# P5-4: the frozen payoff-calibration artifact path
#
# guides/rearchitecture_phase5_models.md P5-4 acceptance: artifact-path
# exp_pnl_model/win_model identical to the inline path on the same rows and
# seed; under the v2 guard, the inline path raises and the artifact path
# succeeds; a missing artifact gives MODEL_NOT_READY; a tampered hash is
# refused; causality (a post-cutoff row cannot enter a fold's artifact).
# ---------------------------------------------------------------------------

from engine.v2.models.no_fit import RuntimeFitForbidden, no_fit_guard  # noqa: E402
from engine.v2.models.payoff_artifact import (  # noqa: E402
    PayoffArtifactError,
    PayoffArtifactLoader,
    PayoffArtifactRef,
    serialize_payoff_artifact,
)
from engine.v2.models.training.payoff import (  # noqa: E402
    build_payoff_line_artifact,
    build_payoff_surface_artifact,
)


def test_artifact_path_matches_inline_path_same_rows_and_seed():
    artifact = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        min_trades=2,
    )
    bundle = _bundle(
        payoff_artifact_recipe={"seed": 42, "draw_count": 16},
        payoff_artifact=artifact,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request(), inputs)

    # Same closed-form answer as the inline-fit control
    # (test_model_number_computed_from_answer_free_inputs): the artifact
    # carries the identical line/residuals, so the simulation -- same driver,
    # pool, seed, draw count -- must land on the exact same numbers.
    assert record.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
    assert record.resolved_request["win_model"] == pytest.approx(1.0)
    assert "NO_PAYOFF_MAP" not in record.reason_codes
    assert "MODEL_NOT_READY" not in record.reason_codes
    assert record.validation_status == "scored"


def test_artifact_path_matches_inline_path_arbitrary_sample():
    """A less degenerate sample: the artifact and inline paths must still
    agree bit for bit, not merely on a closed-form special case."""
    trades = _synthetic_trades(300, seed=11)
    rows = _rows_from_trades(trades)
    cutoff = str(trades["exit_date"].iloc[250].date())

    inline_bundle = _bundle(
        payoff_recipe={"before": cutoff, "seed": 7, "draw_count": 500},
        payoff_source_rows=rows,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inline_record = application.score_one(
        _request(), build_native_score_inputs(inline_bundle),
    )

    artifact = build_payoff_line_artifact(
        rows, strategy="STR-THRU", driver="abs_move", alpha=0.5, before=cutoff,
    )
    artifact_bundle = _bundle(
        payoff_artifact_recipe={"before": cutoff, "seed": 7, "draw_count": 500},
        payoff_artifact=artifact,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    artifact_record = application.score_one(
        _request(), build_native_score_inputs(artifact_bundle),
    )

    assert artifact_record.resolved_request["exp_pnl_model"] == pytest.approx(
        inline_record.resolved_request["exp_pnl_model"], rel=1e-12,
    )
    assert artifact_record.resolved_request["win_model"] == pytest.approx(
        inline_record.resolved_request["win_model"], rel=1e-12,
    )


def test_runup_artifact_path_matches_inline_path_same_rows_and_seed():
    artifact = build_payoff_surface_artifact(
        _GOOD_RUNUP_PAYOFF_ROWS, alpha=0.5, min_trades=2,
    )
    bundle = _runup_bundle(
        payoff_recipe={}, payoff_source_rows=(),
        payoff_artifact_recipe={"seed": 42, "draw_count": 16},
        payoff_artifact=artifact,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_runup_request(), inputs)

    assert record.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
    assert record.resolved_request["win_model"] == pytest.approx(1.0)
    assert "NO_PAYOFF_MAP" not in record.reason_codes
    assert "MODEL_NOT_READY" not in record.reason_codes
    assert record.validation_status == "scored"


def test_artifact_path_causal_cutoff_excludes_a_post_cutoff_row():
    """A row dated on/after the fit cutoff cannot enter the artifact --
    proven both on the artifact's own fields and on the score it produces."""
    future_row = {"driver": 5.0, "spot_entry": 100.0, "exit_value": 999.0,
                  "exit_date": "2026-09-20"}
    with_future = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS + [future_row], strategy="STR-THRU", driver="abs_move",
        alpha=0.5, before="2026-09-16", min_trades=2,
    )
    without_future = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        before="2026-09-16", min_trades=2,
    )
    assert with_future.n == without_future.n == 2
    assert with_future.intercept == pytest.approx(without_future.intercept)
    assert with_future.slope == pytest.approx(without_future.slope)
    assert with_future.content_hash == without_future.content_hash
    assert with_future.window_end == "2026-09-01"

    bundle = _bundle(
        payoff_artifact_recipe={
            "before": "2026-09-16", "seed": 42, "draw_count": 16,
        },
        payoff_artifact=with_future,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    record = application.score_one(_request(), build_native_score_inputs(bundle))
    assert record.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
    assert record.resolved_request["win_model"] == pytest.approx(1.0)


def test_missing_artifact_gives_model_not_ready():
    bundle = _bundle(
        payoff_artifact_recipe={"seed": 42},
        payoff_artifact=None,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request(), inputs)

    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_incompatible_artifact_kind_gives_model_not_ready():
    """A surface artifact handed to a single-driver strategy is incompatible,
    not merely unlucky -- MODEL_NOT_READY, not a crash or a silent number."""
    surface = build_payoff_surface_artifact(
        _GOOD_RUNUP_PAYOFF_ROWS, alpha=0.5, min_trades=2,
    )
    bundle = _bundle(
        payoff_artifact_recipe={"seed": 42},
        payoff_artifact=surface,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request(), inputs)

    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_incompatible_artifact_strategy_gives_model_not_ready():
    """An artifact fitted for a different strategy must not be silently reused."""
    wrong_strategy = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS, strategy="STR-RUNUP", driver="abs_move", alpha=0.5,
        min_trades=2,
    )
    bundle = _bundle(
        payoff_artifact_recipe={"seed": 42},
        payoff_artifact=wrong_strategy,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    inputs = build_native_score_inputs(bundle)
    record = application.score_one(_request(), inputs)

    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None


def test_tampered_artifact_file_is_refused_by_the_loader(tmp_path):
    artifact = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        min_trades=2,
    )
    path = tmp_path / "line.json"
    path.write_bytes(serialize_payoff_artifact(artifact))
    ref = PayoffArtifactRef(path="line.json", content_hash=artifact.content_hash)

    loader = PayoffArtifactLoader(tmp_path)
    loaded = loader.load(ref)
    # resid_sd is NaN on this degenerate 2-row fit (nan != nan), so compare
    # the fields that matter for the round trip individually rather than by
    # dataclass equality.
    assert loaded.content_hash == artifact.content_hash
    assert loaded.intercept == pytest.approx(artifact.intercept)
    assert loaded.slope == pytest.approx(artifact.slope)
    assert loaded.residuals == artifact.residuals

    path.write_bytes(b'{"schema_version": "payoff_line_artifact.v1.0", "tampered": true}')
    tampered_loader = PayoffArtifactLoader(tmp_path)
    with pytest.raises(PayoffArtifactError):
        tampered_loader.load(ref)


def test_under_v2_guard_inline_path_raises_and_artifact_path_succeeds():
    """The exact P5-4 negative control: wrap both paths in the SAME
    no_fit_guard() block. The inline fit must raise; the artifact path,
    which never calls fit_payoff_line, must still produce the score."""
    artifact = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        min_trades=2,
    )
    inline_bundle = _bundle(
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 16},
        payoff_source_rows=_GOOD_PAYOFF_ROWS,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    artifact_bundle = _bundle(
        payoff_artifact_recipe={"seed": 42, "draw_count": 16},
        payoff_artifact=artifact,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )

    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            native_payoff.fit_payoff_line(_GOOD_PAYOFF_ROWS, min_trades=2)

        artifact_record = application.score_one(
            _request(), build_native_score_inputs(artifact_bundle),
        )
        assert artifact_record.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
        assert artifact_record.validation_status == "scored"

        # score_one itself must also raise for the compatibility path under
        # the guard -- it reaches fit_payoff_line via _model_fit_and_pool.
        with pytest.raises(RuntimeFitForbidden):
            application.score_one(_request(), build_native_score_inputs(inline_bundle))


# ---------------------------------------------------------------------------
# P5-4 coordinator condition (2026-09-18): choosing the right artifact is
# still the release's job, but the stage itself must check the FULL causal
# key (strategy, alpha at 4dp, cutoff) against what the inline fit would
# have used for THIS request -- not just the artifact's kind and strategy.
# A wrong-fold artifact must never produce a plausible-looking number.
# ---------------------------------------------------------------------------


def test_artifact_wrong_alpha_gives_model_not_ready():
    """An artifact fitted at a different fill alpha than the request's own
    resolved alpha must be refused, not silently scored -- the alpha is part
    of the identity the inline fit would have keyed on."""
    wrong_alpha = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS, strategy="STR-THRU", driver="abs_move", alpha=0.7,
        min_trades=2,
    )
    bundle = _bundle(
        # _request() resolves fill alpha 0.5; the artifact was fit at 0.7.
        payoff_artifact_recipe={"seed": 42, "draw_count": 16},
        payoff_artifact=wrong_alpha,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    record = application.score_one(_request(), build_native_score_inputs(bundle))

    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_artifact_wrong_cutoff_gives_model_not_ready():
    """An artifact fitted with a different causal cutoff than this request's
    own recipe declares must be refused -- the cutoff is part of the fold
    identity, not a detail the stage can ignore once kind/strategy match."""
    wrong_cutoff = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        before="2026-09-01", min_trades=2,
    )
    bundle = _bundle(
        # The recipe's own cutoff (what the inline fit would have used) is
        # a DIFFERENT date than the one the artifact was actually fit under.
        payoff_artifact_recipe={
            "before": "2026-09-10", "seed": 42, "draw_count": 16,
        },
        payoff_artifact=wrong_cutoff,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    record = application.score_one(_request(), build_native_score_inputs(bundle))

    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_artifact_leaked_future_fold_cutoff_gives_model_not_ready():
    """The specific failure mode the coordinator called out: a release binds
    an artifact whose cutoff is LATER than the request's own causal cutoff --
    a future fold leaking backward. Kind and strategy match, and the wrong
    number would otherwise look entirely plausible. It must still be
    MODEL_NOT_READY, not a silently-wrong score."""
    future_fold = build_payoff_line_artifact(
        _GOOD_PAYOFF_ROWS, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        before="2026-09-20", min_trades=2,
    )
    bundle = _bundle(
        # This request's own causal cutoff is 2026-09-16 -- strictly earlier
        # than the artifact's 2026-09-20 fit cutoff.
        payoff_artifact_recipe={
            "before": "2026-09-16", "seed": 42, "draw_count": 16,
        },
        payoff_artifact=future_fold,
        model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS,
    )
    record = application.score_one(_request(), build_native_score_inputs(bundle))

    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_runup_artifact_wrong_alpha_gives_model_not_ready():
    """The two-driver surface path applies the same full-key check as the
    single-driver line path -- a wrong alpha must not slip through."""
    wrong_alpha = build_payoff_surface_artifact(
        _GOOD_RUNUP_PAYOFF_ROWS, alpha=0.9, min_trades=2,
    )
    bundle = _runup_bundle(
        payoff_recipe={}, payoff_source_rows=(),
        payoff_artifact_recipe={"seed": 42, "draw_count": 16},
        payoff_artifact=wrong_alpha,
    )
    record = application.score_one(_runup_request(), build_native_score_inputs(bundle))

    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


def test_runup_artifact_leaked_future_fold_cutoff_gives_model_not_ready():
    """Same future-fold leak, on the two-driver surface path."""
    future_fold = build_payoff_surface_artifact(
        _GOOD_RUNUP_PAYOFF_ROWS, alpha=0.5, before="2026-09-20", min_trades=2,
    )
    bundle = _runup_bundle(
        payoff_recipe={}, payoff_source_rows=(),
        payoff_artifact_recipe={
            "before": "2026-09-16", "seed": 42, "draw_count": 16,
        },
        payoff_artifact=future_fold,
    )
    record = application.score_one(_runup_request(), build_native_score_inputs(bundle))

    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None
    assert record.validation_status == "refused"


# ---------------------------------------------------------------------------
# A malformed causal cutoff fails closed (found by mutation triage,
# native_payoff._parse_day#6): it used to parse to None and read as "no
# cutoff", fitting on every row including post-cutoff ones. Legacy
# fit_payoff / fit_runup_payoff raise on the same values.
# ---------------------------------------------------------------------------

_MALFORMED_CUTOFFS = ["not-a-date", "", "NaT", pd.NaT]


@pytest.mark.parametrize("before", _MALFORMED_CUTOFFS, ids=repr)
def test_fit_payoff_line_refuses_a_malformed_cutoff(before):
    rows = _rows_from_trades(_synthetic_trades(300, seed=7))
    assert native_payoff.fit_payoff_line(rows) is not None  # no cutoff: fits
    assert native_payoff.fit_payoff_line(rows, before=before) is None
    trades = _synthetic_trades(300, seed=7)
    with pytest.raises(Exception):
        fit_payoff(trades, "STR-THRU", alpha=0.5, before=before)


@pytest.mark.parametrize("before", _MALFORMED_CUTOFFS, ids=repr)
def test_fit_runup_payoff_surface_refuses_a_malformed_cutoff(before):
    trades = _synthetic_runup_trades(300, seed=7)
    rows = _runup_rows_from_trades(trades)
    assert native_payoff.fit_runup_payoff_surface(rows) is not None
    assert native_payoff.fit_runup_payoff_surface(rows, before=before) is None
    with pytest.raises(Exception):
        fit_runup_payoff(trades, alpha=0.5, before=before)


def test_malformed_cutoff_withholds_the_model_number_end_to_end():
    good = dict(payoff_source_rows=_GOOD_PAYOFF_ROWS,
                model_residual_rows=_ZERO_MODEL_RESIDUAL_ROWS)
    dated = application.score_one(_request(), build_native_score_inputs(_bundle(
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 16,
                       "before": "2026-09-16"}, **good)))
    assert dated.resolved_request["exp_pnl_model"] == pytest.approx(0.2)
    record = application.score_one(_request(), build_native_score_inputs(_bundle(
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 16,
                       "before": "not-a-date"}, **good)))
    assert "NO_PAYOFF_MAP" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None


# ---------------------------------------------------------------------------
# mutation-pilot triage: legacy behaviour the parity tests above never hit
# (dirty rows, degenerate fits, tiny samples, bucket boundaries, cost <= 0).
# Every expected value comes from the legacy function on the same input.
# ---------------------------------------------------------------------------

import warnings  # noqa: E402
from contextlib import contextmanager  # noqa: E402


@contextmanager
def _quiet():
    """Degenerate fits warn (RankWarning, ddof); legacy warns identically."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield


def _assert_line_matches_legacy(native, legacy):
    assert native is not None
    assert native["n"] == legacy.n
    assert native["intercept"] == pytest.approx(legacy.intercept, rel=1e-12, abs=1e-15)
    assert native["slope"] == pytest.approx(legacy.slope, rel=1e-12, abs=1e-15)
    if legacy.r is None:
        assert native["r"] is None
    else:
        assert native["r"] == pytest.approx(legacy.r, rel=1e-12)
    if np.isnan(legacy.resid_sd):
        assert np.isnan(native["resid_sd"])
    else:
        assert native["resid_sd"] == pytest.approx(legacy.resid_sd, rel=1e-9, abs=1e-15)
    np.testing.assert_allclose(native["residuals"], legacy.residuals, atol=1e-15)


# One bad field per row (and small positive spots, which stay): legacy's
# ``ok`` mask drops exactly the rows with a non-finite value or spot <= 0.
_DIRTY_LINE = [
    {"driver": float("nan"), "spot_entry": 100.0, "exit_value": 3.0},
    {"driver": 5.0, "spot_entry": float("nan"), "exit_value": 3.0},
    {"driver": 5.0, "spot_entry": 100.0, "exit_value": float("nan")},
    {"driver": 5.0, "spot_entry": 100.0, "exit_value": float("inf")},
    {"driver": 5.0, "spot_entry": 0.0, "exit_value": 3.0},
    {"driver": 5.0, "spot_entry": -1.0, "exit_value": 3.0},
    {"driver": 5.0, "spot_entry": 0.5, "exit_value": 0.02},
    {"driver": 1.0, "spot_entry": 0.9, "exit_value": 0.01},
]


def test_fit_payoff_line_drops_exactly_legacy_dirty_rows_and_skips_junk_first():
    trades = _synthetic_trades(250, seed=5)
    dirty = pd.DataFrame([dict(row, strategy="STR-THRU", fill_alpha=0.5,
                               exit_date=pd.Timestamp("2019-12-01"), abs_move=row["driver"])
                          for row in _DIRTY_LINE]).drop(columns="driver")
    legacy = fit_payoff(pd.concat([dirty, trades], ignore_index=True), "STR-THRU", alpha=0.5)
    # Rows legacy's frame cannot even hold come FIRST: skipping them must not
    # end the scan.
    junk = ["not a mapping", {"spot_entry": 100.0, "exit_value": 3.0},
            {"driver": "abc", "spot_entry": 100.0, "exit_value": 3.0}]
    rows = junk + [dict(row, exit_date="2019-12-01") for row in _DIRTY_LINE]
    native = native_payoff.fit_payoff_line(rows + _rows_from_trades(trades))
    assert legacy.n == 252
    _assert_line_matches_legacy(native, legacy)


def _line_trades(driver, exit_value) -> pd.DataFrame:
    return pd.DataFrame({
        "strategy": "STR-THRU", "fill_alpha": 0.5, "abs_move": driver,
        "spot_entry": 100.0, "exit_value": exit_value,
        "exit_date": pd.date_range("2020-01-01", periods=len(driver), freq="D"),
    })


@pytest.mark.parametrize("case", ["constant driver", "constant exit", "narrow driver"])
def test_fit_payoff_line_correlation_is_none_exactly_when_legacys_is(case):
    rng = np.random.default_rng(2)
    driver = rng.uniform(0.0, 8.0, 250)
    exit_value = 2.0 + 0.4 * driver + rng.normal(0.0, 1.0, 250)
    if case == "constant driver":
        driver = np.full(250, 4.0)
    elif case == "constant exit":
        exit_value = np.full(250, 25.0)  # y = 0.25 exactly: std is exactly 0
    else:  # 0 < std < 1: still a correlation
        driver = rng.uniform(0.0, 1.0, 250)
    trades = _line_trades(driver, exit_value)
    with _quiet():
        legacy = fit_payoff(trades, "STR-THRU", alpha=0.5)
        native = native_payoff.fit_payoff_line(_rows_from_trades(trades))
    assert (legacy.r is None) == (case != "narrow driver")
    _assert_line_matches_legacy(native, legacy)


@pytest.mark.parametrize("n", [2, 3])
def test_fit_payoff_line_residual_sd_at_two_and_three_trades(n):
    trades = _synthetic_trades(n, seed=9)
    with _quiet():
        legacy = fit_payoff(trades, "STR-THRU", alpha=0.5, min_trades=2)
        native = native_payoff.fit_payoff_line(_rows_from_trades(trades), min_trades=2)
    assert np.isnan(legacy.resid_sd) == (n == 2)
    _assert_line_matches_legacy(native, legacy)


def test_fit_residual_cap_and_seed_are_honoured():
    """stages passes both from the payoff recipe; the default is legacy's.
    The cap samples the unsorted residuals, so a capped fit is a sorted
    subset of the uncapped one and different seeds keep different subsets."""
    fits = {
        "line": (native_payoff.fit_payoff_line,
                 _rows_from_trades(_synthetic_trades(300, seed=7))),
        "surface": (native_payoff.fit_runup_payoff_surface,
                    _runup_rows_from_trades(_synthetic_runup_trades(300, seed=7))),
    }
    for fit, rows in fits.values():
        uncapped = fit(rows, max_residuals=10**6)["residuals"]
        assert uncapped.size == 300
        capped = {seed: fit(rows, max_residuals=10, residual_seed=seed)["residuals"]
                  for seed in (1, 2)}
        for kept in capped.values():
            assert kept.size == 10
            assert np.all(np.diff(kept) >= 0)
            assert np.isin(kept, uncapped).all()
        assert not np.array_equal(capped[1], capped[2])


def test_fit_payoff_line_reads_full_timestamp_dates_and_cutoffs():
    trades = _synthetic_trades(300, seed=7)
    rows = _rows_from_trades(trades)
    stamped = [dict(row, exit_date=row["exit_date"] + "T15:30:00") for row in rows]
    cutoff = str(trades["exit_date"].iloc[250].date())
    day = native_payoff.fit_payoff_line(rows, before=cutoff)
    for native in (native_payoff.fit_payoff_line(stamped, before=cutoff),
                   native_payoff.fit_payoff_line(rows, before=cutoff + "T00:00:00"),
                   native_payoff.fit_payoff_line(rows, before=cutoff + " 00:00:00")):
        assert native["n"] == day["n"] == 250
        assert native["slope"] == day["slope"]


# -- STR-RUNUP surface ------------------------------------------------------


def _assert_surface_matches_legacy(native, legacy):
    assert native is not None
    assert native["n"] == legacy.n
    np.testing.assert_allclose(native["coefficients"], legacy.coefficients,
                               rtol=1e-9, atol=1e-12)
    if legacy.r is None:
        assert native["r"] is None
    else:
        assert native["r"] == pytest.approx(legacy.r, rel=1e-9)
    if np.isnan(legacy.resid_sd):
        assert np.isnan(native["resid_sd"])
    else:
        assert native["resid_sd"] == pytest.approx(legacy.resid_sd, rel=1e-6, abs=1e-15)
    np.testing.assert_allclose(native["residuals"], legacy.residuals, atol=1e-12)


_GOOD_RUNUP = {"driver": 4.0, "spot_entry": 100.0, "spot_exit": 101.0, "strike": 100.0,
               "exit_value": 3.0}
_DIRTY_RUNUP = [
    dict(_GOOD_RUNUP, **{field: bad})
    for field in ("driver", "spot_entry", "spot_exit", "strike", "exit_value")
    for bad in (float("nan"), float("inf"))
] + [
    dict(_GOOD_RUNUP, **{field: bad})
    for field in ("spot_entry", "spot_exit", "strike") for bad in (0.0, -1.0)
] + [  # small positive prices stay
    dict(_GOOD_RUNUP, spot_entry=0.5, exit_value=0.02),
    dict(_GOOD_RUNUP, spot_exit=0.6, strike=0.55),
    dict(_GOOD_RUNUP, spot_exit=0.5, strike=0.52),
]


def test_fit_runup_surface_drops_exactly_legacy_dirty_rows_and_skips_junk_first():
    trades = _synthetic_runup_trades(250, seed=5)
    dirty = pd.DataFrame([dict(row, strategy="STR-RUNUP", fill_alpha=0.5, im_t1=row["driver"],
                               exit_date=pd.Timestamp("2019-12-01"))
                          for row in _DIRTY_RUNUP]).drop(columns="driver")
    legacy = fit_runup_payoff(pd.concat([dirty, trades], ignore_index=True), alpha=0.5)
    junk = [None, dict(_GOOD_RUNUP, strike="abc"),
            {k: v for k, v in _GOOD_RUNUP.items() if k != "spot_exit"}]
    rows = junk + [dict(row, exit_date="2019-12-01") for row in _DIRTY_RUNUP]
    native = native_payoff.fit_runup_payoff_surface(rows + _runup_rows_from_trades(trades))
    assert legacy.n == 253
    _assert_surface_matches_legacy(native, legacy)


def test_fit_runup_surface_correlation_is_none_for_a_constant_target_like_legacy():
    trades = _synthetic_runup_trades(250, seed=4)
    trades["exit_value"] = 25.0  # target 0.25 exactly: std is exactly 0
    with _quiet():
        legacy = fit_runup_payoff(trades, alpha=0.5)
        native = native_payoff.fit_runup_payoff_surface(_runup_rows_from_trades(trades))
    assert legacy.r is None
    _assert_surface_matches_legacy(native, legacy)


@pytest.mark.parametrize("n", [2, 3])
def test_fit_runup_surface_residual_sd_at_two_and_three_trades(n):
    trades = _synthetic_runup_trades(n, seed=9)
    with _quiet():
        legacy = fit_runup_payoff(trades, alpha=0.5, min_trades=2)
        native = native_payoff.fit_runup_payoff_surface(
            _runup_rows_from_trades(trades), min_trades=2)
    assert np.isnan(legacy.resid_sd) == (n == 2)
    assert np.isnan(native["resid_sd"]) == (n == 2)
    assert native["n"] == legacy.n


# -- residual buckets and pool selection -----------------------------------


def _assert_buckets_match_legacy(native, legacy):
    if legacy is None:
        assert native is None
        return
    np.testing.assert_array_equal(native["edges"], legacy["edges"])
    assert native["min_pool"] == legacy["min_pool"]
    assert len(native["pools"]) == len(legacy["pools"])
    for mine, theirs in zip(native["pools"], legacy["pools"]):
        np.testing.assert_array_equal(mine, theirs)


@pytest.mark.parametrize("case", ["exactly 2500", "2499", "100 rows", "nan rows",
                                  "two values", "three values"])
def test_bucket_residual_pool_matches_legacy_at_its_boundaries(case):
    rng = np.random.default_rng(8)
    n = {"2499": 2499, "100 rows": 100}.get(case, 2500)
    pred = rng.normal(5.0, 2.0, n)
    res = rng.normal(0.0, 1.0, n)
    if case == "nan rows":  # 2500 finite pairs plus pairs missing one side
        pred = np.concatenate([pred, [np.nan, 1.0, np.inf]])
        res = np.concatenate([res, [0.5, np.nan, 0.5]])
    elif case == "two values":  # quantile edges {0, 0.5, 1}: two buckets
        pred = np.repeat([0.0, 1.0], 1250)
    elif case == "three values":  # edges fall ON values: right-closed ties
        pred = np.repeat([0.0, 1.0, 2.0], [600, 1300, 600])
    native = native_payoff.bucket_residual_pool(pred, res)
    legacy = bucket_residuals(pred, res)
    assert (legacy is None) == (case in ("2499", "100 rows"))
    _assert_buckets_match_legacy(native, legacy)


def _artifact(flat, buckets):
    return ModelArtifact(model=None, role="size", features=(), residuals=flat,
                         target="abs_move", residual_buckets=buckets)


def test_residual_pool_for_matches_legacy_selection_at_its_boundaries():
    flat = np.arange(10.0)
    thin = np.arange(4.0)
    exact = np.arange(5.0) + 100.0
    buckets = {"edges": np.array([-np.inf, 1.0, 2.0, np.inf]),
               "pools": [thin, exact, np.arange(6.0) + 200.0], "min_pool": 5}
    legacy = _artifact(flat, buckets)
    # NaN prediction -> flat; on an edge -> the bucket above (right-closed);
    # a pool of exactly min_pool is used; a thinner one falls back.
    for point in (float("nan"), None, 0.0, 1.0, 1.5, 2.0, 9.0):
        np.testing.assert_array_equal(
            native_payoff.residual_pool_for(buckets, point, flat)[0],
            legacy.residual_pool(point)[0])
    assert native_payoff.residual_pool_for(buckets, 1.0, flat)[0] is exact
    assert native_payoff.residual_pool_for(buckets, 0.5, flat)[0] is flat
    assert native_payoff.residual_pool_for(buckets, float("nan"), flat)[0] is flat

    # A mapping without min_pool never falls back, even for an empty pool.
    empty = np.array([])
    bare = {"edges": buckets["edges"], "pools": [thin, empty, exact]}
    legacy_bare = _artifact(flat, bare)
    for point in (0.5, 1.5, 2.5):
        np.testing.assert_array_equal(
            native_payoff.residual_pool_for(bare, point, flat)[0],
            legacy_bare.residual_pool(point)[0])
    assert native_payoff.residual_pool_for(bare, 1.5, flat)[0] is empty


def test_driver_residual_pool_matches_legacy_artifact_with_its_own_split():
    """deciles=5, min_pool=100 over 600 rows buckets; legacy's defaults would
    not (600 < 10 * 250), so a dropped argument shows."""
    rng = np.random.default_rng(12)
    pred = rng.normal(5.0, 2.0, 600)
    res = rng.normal(0.0, 1.0, 600)
    legacy = _artifact(res, bucket_residuals(pred, res, deciles=5, min_pool=100))
    junk = ["not a mapping", {"prediction": "x", "residual": 1.0},
            {"prediction": float("nan"), "residual": 9.0},
            {"prediction": 5.0, "residual": float("inf")}]
    rows = junk + [{"prediction": p, "residual": r} for p, r in zip(pred, res)]
    for point in (1.0, 5.0, 9.0):
        native = native_payoff.driver_residual_pool(rows, point, deciles=5, min_pool=100)
        expected, label = legacy.residual_pool(point)
        assert label.startswith("bucket")
        np.testing.assert_array_equal(native, expected)
    # No prediction: the flat pool, which holds only complete finite pairs.
    np.testing.assert_array_equal(
        native_payoff.driver_residual_pool(rows, None, deciles=5, min_pool=100), res)


def test_driver_residual_pool_drops_rows_with_a_missing_prediction_before_bucketing():
    rng = np.random.default_rng(13)
    pred = rng.normal(5.0, 2.0, 2500)
    res = rng.normal(0.0, 1.0, 2500)
    rows = [{"prediction": p, "residual": r} for p, r in zip(pred, res)]
    rows += [{"prediction": float("nan"), "residual": 50.0 + i} for i in range(300)]
    legacy = _artifact(res, bucket_residuals(pred, res))
    for point in (1.0, 5.0, 9.0):
        np.testing.assert_array_equal(native_payoff.driver_residual_pool(rows, point),
                                      legacy.residual_pool(point)[0])


# -- exit value and the zero-cost guard ------------------------------------


def test_payoff_exit_value_floors_at_zero_like_legacy():
    payoff = PayoffMap(strategy="STR-THRU", driver="abs_move", alpha=0.5, intercept=-0.05,
                       slope=0.01, resid_sd=0.01, n=250, r=0.9, residuals=np.zeros(3))
    drivers = [0.0, 2.0, 5.0, 8.0]
    native = native_payoff.payoff_exit_value(drivers, 100.0, -0.05, 0.01)
    np.testing.assert_array_equal(native, payoff.exit_value(drivers, 100.0))
    assert native[0] == 0.0 and native[-1] == pytest.approx(3.0)


_COEFFICIENTS = (0.02, 0.004, 0.001, 0.0005, 0.0007, 0.0003)


@pytest.mark.parametrize("cost", [0.0, -1.0, 0.5])
def test_simulated_returns_are_nan_for_non_positive_cost_like_legacy(cost):
    thru = native_payoff.simulate_model_returns(
        7.0, [-0.5, 0.5], [0.0, 0.01], 0.02, 0.004, 100.0, cost, 50,
        np.random.default_rng(1))
    payoff = PayoffMap(strategy="STR-THRU", driver="abs_move", alpha=0.5, intercept=0.02,
                       slope=0.004, resid_sd=0.01, n=250, r=0.9,
                       residuals=np.array([0.0, 0.01]))
    legacy_rng = np.random.default_rng(1)
    draws = 7.0 + legacy_rng.choice([-0.5, 0.5], size=50, replace=True)
    legacy = simulate_returns(draws, payoff, 100.0, cost, payoff.residual_draws(50, legacy_rng))
    assert thru.shape == legacy.shape == (50,)
    np.testing.assert_array_equal(thru, legacy)
    assert np.isnan(thru).all() == (cost <= 0)

    runup = native_payoff.simulate_runup_model_returns(
        7.0, 3.0, [-0.5, 0.5], [-0.2, 0.2], _COEFFICIENTS, [0.0, 0.01],
        100.0, 100.0, cost, 7.0, 50, np.random.default_rng(1))
    assert runup.shape == (50,)
    assert np.isnan(runup).all() == (cost <= 0)
    if cost > 0:
        assert np.isfinite(runup).all()


def test_runup_draws_floor_implied_and_move_at_zero_like_legacy():
    """Small points and pools straddling zero: legacy clips negative draws to
    exactly 0, not to any positive value."""
    implied_pool = np.array([-0.4, -0.1, 0.2, 0.5])
    move_pool = np.array([-0.6, -0.2, 0.1, 0.4])
    payoff_residuals = np.array([-0.01, 0.0, 0.01])
    point_implied, point_move, spot, strike, cost, draws, days = (
        0.3, 0.3, 100.0, 100.0, 1.0, 400, 7.0)
    payoff = RunupPayoffSurface(alpha=0.5, coefficients=_COEFFICIENTS, resid_sd=0.01, n=250,
                                r=0.9, residuals=payoff_residuals)
    rng = np.random.default_rng(21)
    implied = np.maximum(point_implied + rng.choice(implied_pool, size=draws, replace=True),
                         0.0)
    move_d14 = point_move + rng.choice(move_pool, size=draws, replace=True)
    move = legacy_scale_runup_move(np.maximum(move_d14, 0.0), days)
    signed = rng.choice((-1.0, 1.0), size=draws) * move
    legacy = simulate_runup_returns(implied, signed, payoff, spot=spot, strike=strike,
                                    cost=cost, payoff_noise=payoff.residual_draws(draws, rng))
    native = native_payoff.simulate_runup_model_returns(
        point_implied, point_move, implied_pool, move_pool, _COEFFICIENTS, payoff_residuals,
        spot, strike, cost, days, draws, np.random.default_rng(21))
    np.testing.assert_array_equal(native, legacy)


def test_line_payoff_document_rounds_intercept_and_slope_to_8dp():
    fit = {"intercept": 0.123456789012, "slope": -0.0000000156253}
    doc = stages._line_payoff_document(fit)
    assert doc == {
        "intercept": round(0.123456789012, 8),
        "slope": round(-0.0000000156253, 8),
    }


def test_surface_payoff_document_rounds_coefficients_to_8dp():
    raw = [
        0.1234567891, -0.0000000123456, 1.999999995,
        0.3333333335, -1.23456789012, 0.00000001005,
    ]
    fit = {"coefficients": raw}
    doc = stages._surface_payoff_document(fit)
    assert doc["kind"] == "runup_payoff_surface"
    assert doc["coefficients"] == {
        name: round(value, 8)
        for name, value in zip(native_payoff.RUNUP_TERMS, raw)
    }
