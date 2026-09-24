"""Phase 4 native execution: unpriced residual-band parity (R4 residual-only).

The legacy scorer takes STR-THRU's ``driver_p10``/``driver_p90``
(engine/score.py:2976-2982) and STR-RUNUP's ``driver_p10``/``driver_p90`` and
``runup_move_p10``/``runup_move_p90`` (engine/score.py:3142-3163) from the
captured driver residual pools BEFORE the entry-cost/payoff guard, so a row
with no premium, strike or payoff surface still carries its bands while every
P&L field stays withheld (fixture 019 DYN-SV LEN member 2: an observed-empty
quote lookup). This pins the native half of that behaviour -- a legitimate
residual-only model block banded before the pricing guards -- against the
legacy draw order (implied pool first, runup-move pool second), the request's
own derived seed and legacy's MODEL_DRAWS, non-negative clipping and the
runup horizon scaling. ``engine/v2`` never imports legacy; the oracle below
restates ``engine/score.py``'s own quantile call, as
test_v2_scoring_native_payoff.py does for the priced path.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application, native_payoff, stages
from engine.v2.scoring.identity import bootstrap_seed, score_request_key
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from engine.v2.scoring.stages import (
    RESIDUAL_ONLY_FIELD,
    STAGE_NAMES,
    NativeScoreInputs,
    StageReceipt,
    assemble_native_values,
)

_MODEL_DRAWS = native_payoff.MODEL_DRAWS
_RECEIPTS = tuple(
    StageReceipt(stage, "declared-input", "declared-output")
    for stage in STAGE_NAMES if stage != "diagnostics"
)

# Flat pools (< deciles * MIN_POOL rows, so ``driver_residual_pool`` returns
# the whole residual column in row order -- no bucketing to shadow the check).
# Continuous, with enough tail mass that a slice lands in the non-negative
# clip (implied: residuals <= -1, i.e. draw <= 0; move: residuals <= -3), while
# the p10/p90 quantiles of a 4000-draw sample still shift with the seed.
def _residual_group(count: int, low: float, high: float) -> np.ndarray:
    return np.random.default_rng(count).uniform(low, high, count)


IMPLIED_RESIDUALS = sorted(float(x) for x in np.concatenate([
    _residual_group(10, -3.5, -1.0), _residual_group(30, -0.9, 5.0)]))
MOVE_RESIDUALS = sorted(float(x) for x in np.concatenate([
    _residual_group(8, -8.0, -3.2), _residual_group(32, -2.5, 6.0)]))
POINTIMPLIED, POINTMOVE = 1.0, 3.0
_DAYS = 7.0
IMPLIED_ROWS = [{"prediction": POINTIMPLIED, "residual": r} for r in IMPLIED_RESIDUALS]
MOVE_ROWS = [{"prediction": POINTMOVE, "residual": r} for r in MOVE_RESIDUALS]


def _identity_context(**overrides) -> dict:
    context = {
        "ticker": "AAA", "strategy": "STR-RUNUP",
        "requested_as_of": "2026-09-16", "requested_event_date": "2026-09-16",
        "requested_strike": 100.0, "requested_expiry": "2026-09-18",
        "fill_alpha": 0.5, "variant": None, "decision_offset": None,
        "quote_max_age_sessions": None, "chain_as_of": None,
        "snapshot": "snap-1",
        "event_date": "2026-09-16", "entry_date": "2026-09-16",
        "exit_date": "2026-09-18", "expiry": "2026-09-18", "session": "AMC",
        "spot": 100.0, "strike": 100.0, "days_before_print": _DAYS,
    }
    context.update(overrides)
    return context


def _derived_seed(context: dict) -> int:
    """The request's own Monte Carlo seed -- the identity legacy seeds on."""
    return bootstrap_seed(context["snapshot"], score_request_key(context))


def _runup_forecast_models() -> dict:
    return {
        "driver_prediction": {"intercept": POINTIMPLIED, "coefficients": {}},
        "runup_move_prediction": {"intercept": POINTMOVE, "coefficients": {}},
    }


def _runup_oracle(seed: int) -> dict:
    """Restatement of engine/score.py:3143-3163: implied pool drawn FIRST,
    runup-move pool SECOND from one shared rng, both non-negative clipped, the
    move band scaled by days/14, then the 10th/90th percentiles."""
    rng = np.random.default_rng(seed)
    implied_draws = POINTIMPLIED + rng.choice(
        np.asarray(IMPLIED_RESIDUALS, float), size=_MODEL_DRAWS, replace=True,
    )
    implied_draws = np.maximum(implied_draws, 0.0)
    move_draws = POINTMOVE + rng.choice(
        np.asarray(MOVE_RESIDUALS, float), size=_MODEL_DRAWS, replace=True,
    )
    move_draws = native_payoff.scale_runup_move(np.maximum(move_draws, 0.0), _DAYS)
    return {
        "driver_p10": float(np.quantile(implied_draws, 0.10)),
        "driver_p90": float(np.quantile(implied_draws, 0.90)),
        "runup_move_p10": float(np.quantile(move_draws, 0.10)),
        "runup_move_p90": float(np.quantile(move_draws, 0.90)),
    }


def _residual_only_inputs(strategy="STR-RUNUP", *, implied_rows=IMPLIED_ROWS,
                          move_rows=MOVE_ROWS, context=None, models=None):
    context = _identity_context() if context is None else dict(context)
    # No captured ``expiry`` + an observed-empty quote domain: the row reaches
    # pricing and refuses NO_CHAIN exactly as the fixture-019 member does, yet
    # the model stage still bands it (the bands are chain-independent).
    context.pop("expiry", None)
    context.pop("post_event_expiry", None)
    context["quotes"] = {}
    context["strategy"] = strategy
    if strategy == "STR-THRU":
        forecast_models = models or {
            "driver_prediction": {"intercept": POINTIMPLIED, "coefficients": {}},
        }
    else:
        forecast_models = models or _runup_forecast_models()
    model_block = {RESIDUAL_ONLY_FIELD: True,
                   "model_residual_rows": list(implied_rows),
                   "runup_move_residual_rows": list(move_rows)}
    return NativeScoreInputs(
        context=context,
        features={"model_inputs": {}},
        forecast={"driver_name": "implied_t1", "models": forecast_models},
        geometry=None, pricing=None,
        analogs={"recipe": None}, simulation={"mode": "not_applicable"},
        gate={"mode": "not_applicable"}, chooser={}, diagnostics={},
        source_ref="residual-only-test", stage_receipts=_RECEIPTS,
        model=model_block,
    )


def _band_fields(values):
    return {key: values.get(key) for key in (
        "driver_p10", "driver_p90", "runup_move_p10", "runup_move_p90")}


# ---------------------------------------------------------------------------
# unpriced residual-only STR-RUNUP bands, computed before the pricing guards
# ---------------------------------------------------------------------------


def test_unpriced_runup_residual_only_bands_match_legacy_draw_order():
    values = assemble_native_values(
        _residual_only_inputs(), strategy="STR-RUNUP",
    )
    assert _band_fields(values) == pytest.approx(
        _runup_oracle(_derived_seed(_identity_context())), rel=1e-12)


def test_residual_only_bands_are_chain_independent_and_withhold_pnl():
    values = assemble_native_values(
        _residual_only_inputs(), strategy="STR-RUNUP",
    )
    # No premium/strike/payoff -> the driver bands still exist, every P&L field
    # stays absent, and the row is refused only for the missing chain.
    assert values["driver_p10"] is not None and values["driver_p90"] is not None
    assert values["runup_move_p10"] is not None
    assert "NO_CHAIN" in values["flags"]
    for field in ("exp_pnl_model", "win_model", "win_model_raw", "payoff",
                  "model_p10", "model_p90"):
        assert field not in values or values.get(field) is None, field


def test_runup_residual_only_bands_reflect_clipping_and_horizon_scaling():
    values = assemble_native_values(
        _residual_only_inputs(), strategy="STR-RUNUP",
    )
    # Non-negative clipping: negative implied/move residuals floor the low tail
    # at zero (both pools contain residuals that push a draw below 0).
    assert values["driver_p10"] == 0.0
    assert values["runup_move_p10"] == 0.0
    # Horizon scaling: a longer horizon scales the (already clipped) move band
    # but never the implied-move band, which is not horizon-dependent.
    longer = assemble_native_values(
        _residual_only_inputs(
            context=_identity_context(days_before_print=14.0)),
        strategy="STR-RUNUP",
    )
    assert longer["driver_p10"] == pytest.approx(values["driver_p10"])
    assert longer["driver_p90"] == pytest.approx(values["driver_p90"])
    assert longer["runup_move_p90"] == pytest.approx(2.0 * values["runup_move_p90"])


# ---------------------------------------------------------------------------
# deterministic seed / draw behaviour: the ORIGINAL request seed is used
# ---------------------------------------------------------------------------


def test_residual_only_bands_are_deterministic_and_seed_sensitive():
    first = assemble_native_values(_residual_only_inputs(), strategy="STR-RUNUP")
    second = assemble_native_values(_residual_only_inputs(), strategy="STR-RUNUP")
    assert _band_fields(first) == _band_fields(second)  # same seed -> same draws

    # A different snapshot is a different request seed: the bands must move,
    # proving the band uses the request's own derived seed rather than a
    # constant or an rng that ignores identity.
    other = assemble_native_values(
        _residual_only_inputs(context=_identity_context(snapshot="snap-2")),
        strategy="STR-RUNUP",
    )
    assert _band_fields(other) != _band_fields(first)
    assert _band_fields(other) == pytest.approx(
        _runup_oracle(_derived_seed(_identity_context(snapshot="snap-2"))),
        rel=1e-12)


# ---------------------------------------------------------------------------
# priced-path numeric stability: same seed/pools -> identical bands, priced
# row also carries its P&L; the unpriced residual-only row carries only bands
# ---------------------------------------------------------------------------

_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
    ("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
}
_RUNUP_PAYOFF_ROWS = [
    {"driver": 0.0, "spot_entry": 100.0, "spot_exit": 100.0, "strike": 100.0,
     "exit_value": 2.0, "exit_date": "2026-09-01"},
    {"driver": 10.0, "spot_entry": 100.0, "spot_exit": 100.0, "strike": 100.0,
     "exit_value": 6.0, "exit_date": "2026-09-01"},
]


def _runup_request() -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-runup-residual", calendar_revision="cal-1",
        strategy_version="STR-RUNUP", deployment_id="dep-1",
        decision_clock_id="entry-close", requested_decision_at="2026-09-16",
        snapshot_id="snap-1", mode="replay", fill_model={"alpha": 0.5},
    )


def _priced_runup_bundle(**overrides) -> SourceBundle:
    base = dict(
        source_ref="residual-only-priced", strategy="STR-RUNUP",
        context=_identity_context(), raw_quotes=_QUOTES,
        feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "m1"}},
        forecast_recipes=_runup_forecast_models(),
        model_artifact_refs={"driver_prediction": "sha256:m1",
                             "runup_move_prediction": "sha256:m2"},
        residual_recipe={"terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
                         "capital_at_risk": 1.0},
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}},
                     "threshold": 0.0},
        # No ``seed`` / ``draw_count``: the priced path falls back to the same
        # derived seed and MODEL_DRAWS the residual-only path uses.
        payoff_recipe={"min_trades": 2},
        payoff_source_rows=_RUNUP_PAYOFF_ROWS,
        model_residual_rows=IMPLIED_ROWS,
        runup_move_residual_rows=MOVE_ROWS,
    )
    base.update(overrides)
    return SourceBundle(**base)


def test_residual_only_bands_equal_the_priced_path_bands():
    # Priced runup row: bands + a real P&L, computed from the same pools.
    priced = application.score_one(
        _runup_request(), build_native_score_inputs(_priced_runup_bundle()),
    )
    assert priced.resolved_request["exp_pnl_model"] is not None
    # The SAME request with the payoff stripped and an observed-empty chain:
    # the bands are byte-identical, only the P&L disappears.
    residual = assemble_native_values(
        replace(
            build_native_score_inputs(_priced_runup_bundle(
                payoff_recipe={}, payoff_source_rows=())),
            context={**_identity_context(), "quotes": {}},
            model={RESIDUAL_ONLY_FIELD: True,
                   "model_residual_rows": list(IMPLIED_ROWS),
                   "runup_move_residual_rows": list(MOVE_ROWS)},
        ),
        strategy="STR-RUNUP",
    )
    assert residual.get("exp_pnl_model") is None
    for field in ("driver_p10", "driver_p90", "runup_move_p10", "runup_move_p90"):
        assert residual[field] == pytest.approx(
            priced.resolved_request[field], rel=1e-12)
    # And both agree with the legacy-formula oracle on the derived seed.
    assert _band_fields(residual) == pytest.approx(
        _runup_oracle(_derived_seed(_identity_context())), rel=1e-12)


# ---------------------------------------------------------------------------
# single-driver STR-THRU residual-only band (unclipped, as legacy)
# ---------------------------------------------------------------------------


def _thru_oracle(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = POINTIMPLIED + rng.choice(
        np.asarray(IMPLIED_RESIDUALS, float), size=_MODEL_DRAWS, replace=True,
    )
    return float(np.quantile(draws, 0.10)), float(np.quantile(draws, 0.90))


def test_unpriced_thru_residual_only_driver_band_without_runup():
    values = assemble_native_values(
        _residual_only_inputs(strategy="STR-THRU"), strategy="STR-THRU",
    )
    lo, hi = _thru_oracle(_derived_seed(_identity_context(strategy="STR-THRU")))
    assert values["driver_p10"] == pytest.approx(lo, rel=1e-12)
    assert values["driver_p90"] == pytest.approx(hi, rel=1e-12)
    # STR-THRU has no second driver; runup fields stay absent, P&L withheld.
    assert "runup_move_p10" not in values and "runup_move_p90" not in values
    assert values.get("exp_pnl_model") is None
    # Legacy does NOT floor the STR-THRU driver band, so a negative point's
    # low tail can sit below zero (unlike the clipped runup bands).
    assert lo < 0.0


# ---------------------------------------------------------------------------
# no legitimate residual inputs -> bands stay absent, never synthesized
# ---------------------------------------------------------------------------


def test_no_residual_rows_withholds_bands_and_reports_missing_residuals():
    values = assemble_native_values(
        _residual_only_inputs(implied_rows=[], move_rows=[]), strategy="STR-RUNUP",
    )
    assert all(key not in values for key in (
        "driver_p10", "driver_p90", "runup_move_p10", "runup_move_p90"))
    assert "MISSING_MODEL_RESIDUALS" in values["flags"]
    assert values.get("exp_pnl_model") is None


def test_empty_model_block_is_not_applicable_not_a_residual_request():
    inputs = replace(_residual_only_inputs(), model={})
    values = assemble_native_values(inputs, strategy="STR-RUNUP")
    assert all(key not in values for key in (
        "driver_p10", "driver_p90", "runup_move_p10", "runup_move_p90"))
    # A not-applicable model layer reports no missing-residual refusal.
    assert "MISSING_MODEL_RESIDUALS" not in values["flags"]


# ---------------------------------------------------------------------------
# the source-input builder now carries a legitimate residual-only block
# ---------------------------------------------------------------------------


def test_source_builder_emits_residual_only_block_without_payoff_recipe():
    inputs = build_native_score_inputs(_priced_runup_bundle(
        payoff_recipe={}, payoff_source_rows=()))
    assert inputs.model[RESIDUAL_ONLY_FIELD] is True
    assert "payoff_recipe" not in inputs.model
    assert inputs.model["model_residual_rows"] == [dict(r) for r in IMPLIED_ROWS]
    assert inputs.model["runup_move_residual_rows"] == [dict(r) for r in MOVE_ROWS]


def test_source_builder_still_refuses_payoff_rows_without_a_recipe():
    with pytest.raises(ValueError, match="payoff_source_rows supplied without"):
        build_native_score_inputs(_priced_runup_bundle(payoff_recipe={}))


def test_source_builder_keeps_no_model_block_for_a_bundle_without_residuals():
    inputs = build_native_score_inputs(_priced_runup_bundle(
        payoff_recipe={}, payoff_source_rows=(), model_residual_rows=(),
        runup_move_residual_rows=()))
    assert inputs.model == {}
