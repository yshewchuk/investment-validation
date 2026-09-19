"""R4-10: the legacy planned-exit simulation, expressed through SourceBundle.

``residual_recipe={"mode": "planned_exit", ...}`` declares the simulation
(pre_iv30, dte_exit, event_date, draws), and the paired pool arrives one of two
ways: the frozen ``PairedResidualPoolArtifact`` under its causal-key check
(P5-4), or ``paired_residual_rows``, the compatibility path that carries a
recorded population in its recorded order. Both are executed natively by the
simulation stage. Parity is against legacy ``engine.pnl_sim.expected_pnl``
called exactly as ``engine.score.Scorer._expectation`` calls it: key = the
strategy, event_date = a normalized ``pd.Timestamp``, dte_exit = expiry -
exit_date. Equality is bit for bit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import pnl_sim
from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import generate, price
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.models.residual_artifact import make_paired_residual_pool_artifact
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs

EVENT = "2026-09-16"
CONTEXT = {"ticker": "AAA", "event_date": EVENT, "entry_date": EVENT,
           "exit_date": "2026-09-09", "expiry": "2026-09-18", "spot": 100.0}
QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
    ("P", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
}
SIM_FIELDS = ("exp_pnl_sim", "win_sim", "sim_p10", "sim_p90", "pool_n")
KEY = {"move_model_id": "size_v1_4", "crush_model_id": "iv_crush_v1_gbm",
       "cutoff": EVENT}
LINEAGE = Lineage(data=(DataDependency(table="tier3.panel", end_exclusive=EVENT),))


def _history(*, ties: bool) -> pd.DataFrame:
    """A shuffled paired pool; with ``ties`` several events share each date,
    so legacy's own sort decides the order the index-based draws read."""
    rng = np.random.default_rng(29)
    days = pd.date_range("2025-03-01", periods=500 if not ties else 170, freq="D")
    rows = []
    for index in range(510):
        day = days[index % len(days)] if ties else days[min(index, len(days) - 1)]
        if not ties and index >= len(days):
            break
        rows.append({"event_date": day, "ticker": f"T{index % 7}",
                     "pred_abs_move": 3.0 + (index * 7919 % 400) / 50.0,
                     "err_move": float(rng.normal(0.0, 3.0)),
                     "err_crush": float(rng.normal(-5.0, 12.0))})
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=3).reset_index(drop=True)


def _recorded_rows(pool: pnl_sim.ResidualPool) -> list[dict]:
    """The population a Phase 4 capture records: legacy's own sorted order,
    causal prefix only (``ResidualPool.documented_population``)."""
    return pool.documented_population(pool.before(pd.Timestamp(EVENT)))


def _request(alpha: float = 0.5) -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-planned", calendar_revision="cal-1", strategy_version="STR-THRU",
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at=EVENT, snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": alpha},
    )


def _bundle(context=None, **overrides) -> SourceBundle:
    base: dict = dict(
        source_ref="planned-exit-bundle",
        context=dict(CONTEXT if context is None else context),
        raw_quotes=QUOTES, feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "m1"}},
        forecast_recipes={
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
            "forecast_abs_move": {"intercept": 7.0, "coefficients": {}},
            "pred_iv_crush": {"intercept": -20.0, "coefficients": {}},
        },
        model_artifact_refs={"driver_prediction": "sha256:m1",
                             "forecast_abs_move": "sha256:m2", "pred_iv_crush": "sha256:m3"},
        residual_recipe={"mode": "planned_exit", "pre_iv30": 40.0},
        analog_recipe={},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {"exp_pnl_sim": 1.0}},
                     "threshold": 0.0},
    )
    base.update(overrides)
    return SourceBundle(**base)


def _native(bundle: SourceBundle, alpha: float = 0.5) -> tuple:
    record = application.score_one(_request(alpha), build_native_score_inputs(bundle))
    return tuple(record.resolved_request.get(field) for field in SIM_FIELDS), record


def _legacy(pool, *, alpha=0.5, dte_exit=9.0, event_date=EVENT, pre_iv30=40.0) -> tuple:
    """Legacy expected_pnl, called the way Scorer._expectation calls it."""
    geometry = generate("STR-THRU", {**CONTEXT, "forecast_abs_move": 7.0})
    priced = price(geometry, {(leg.right, leg.strike, leg.expiry): QUOTES[
        (leg.right, leg.strike, leg.expiry)] for leg in geometry.legs}, alpha)
    result = pnl_sim.expected_pnl(
        exit_legs=[{"strike": float(leg.strike), "qty": float(leg.quantity),
                    "side": "sell" if str(leg.side).lower() == "buy" else "buy"}
                   for leg in priced.legs],
        spot=100.0, entry_cost=priced.entry_cost, pre_iv30=pre_iv30,
        pred_abs_move=7.0, pred_iv_crush=-20.0, dte_exit=dte_exit,
        event_date=pd.Timestamp(event_date).normalize(), pool=pool, key="STR-THRU",
    )
    return tuple(result[field] for field in SIM_FIELDS)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_rows_path_equals_legacy_scorer_expectation_bit_for_bit_with_tied_dates(alpha):
    pool = pnl_sim.ResidualPool(_history(ties=True))
    native, record = _native(_bundle(paired_residual_rows=_recorded_rows(pool)), alpha)
    assert native[0] is not None, record.reason_codes
    assert native == _legacy(pool, alpha=alpha)


@pytest.mark.parametrize("spelling", [EVENT, f"{EVENT}T00:00:00", pd.Timestamp(EVENT)])
def test_event_date_spelling_does_not_move_the_draws(spelling):
    pool = pnl_sim.ResidualPool(_history(ties=True))
    native, _ = _native(_bundle(context={**CONTEXT, "event_date": spelling},
                                paired_residual_rows=_recorded_rows(pool)))
    assert native == _legacy(pool)


def test_recipe_declared_horizon_and_cutoff_are_the_ones_simulated():
    pool = pnl_sim.ResidualPool(_history(ties=True))
    rows = pool.documented_population(len(pool))
    recipe = {"mode": "planned_exit", "pre_iv30": 35.0, "dte_exit": 0,
              "event_date": "2025-07-20"}
    native, _ = _native(_bundle(residual_recipe=recipe, paired_residual_rows=rows))
    assert native == _legacy(pool, dte_exit=0.0, event_date="2025-07-20", pre_iv30=35.0)


def test_artifact_path_through_the_bundle_equals_legacy_bit_for_bit():
    history = _history(ties=False)
    artifact = make_paired_residual_pool_artifact(
        move_model_id=KEY["move_model_id"], crush_model_id=KEY["crush_model_id"],
        cutoff=KEY["cutoff"], lineage=LINEAGE,
        rows=[(str(row.event_date.date()), row.ticker, row.pred_abs_move,
               row.err_move, row.err_crush) for row in history.itertuples()],
    )
    native, record = _native(_bundle(paired_residual_recipe=dict(KEY),
                                     paired_residual_artifact=artifact))
    assert native[0] is not None, record.reason_codes
    assert native == _legacy(pnl_sim.ResidualPool(history))


def test_planted_defects_are_caught_by_the_comparison():
    pool = pnl_sim.ResidualPool(_history(ties=True))
    rows = [dict(row) for row in _recorded_rows(pool)]
    expected = _legacy(pool)
    perturbed = [dict(row) for row in rows]
    for row in perturbed:
        row["err_move"] += 0.5
    assert _native(_bundle(paired_residual_rows=perturbed))[0] != expected
    # Swapping two rows that share a date changes which pair each index draws:
    # the compatibility path must carry the recorded order, not a re-sort.
    swapped = list(rows)
    first = next(i for i in range(len(rows) - 1)
                 if rows[i]["event_date"] == rows[i + 1]["event_date"])
    swapped[first], swapped[first + 1] = swapped[first + 1], swapped[first]
    assert _native(_bundle(paired_residual_rows=swapped))[0] != expected
    assert _native(_bundle(paired_residual_rows=rows,
                           residual_recipe={"mode": "planned_exit", "pre_iv30": 41.0}))[0] \
        != expected


def test_declared_but_unfed_pool_refuses_without_numbers():
    native, record = _native(_bundle())
    assert native[0] is None
    assert "MISSING_SIMULATION_INPUT:residuals" in record.reason_codes


def test_malformed_planned_exit_bundles_are_refused_at_build():
    pool = pnl_sim.ResidualPool(_history(ties=False))
    rows = _recorded_rows(pool)
    artifact = make_paired_residual_pool_artifact(
        move_model_id="m", crush_model_id="c", cutoff=EVENT, lineage=LINEAGE,
        rows=[(row["event_date"], "T", row["pred_abs_move"], row["err_move"],
               row["err_crush"]) for row in rows])
    with pytest.raises(ValueError, match="compatibility path"):
        build_native_score_inputs(_bundle(
            paired_residual_rows=rows, paired_residual_artifact=artifact,
            paired_residual_recipe={"move_model_id": "m", "crush_model_id": "c",
                                    "cutoff": EVENT}))
    with pytest.raises(ValueError, match="unsupported recipe fields"):
        build_native_score_inputs(_bundle(
            residual_recipe={"mode": "planned_exit", "terminal_spots": (90.0,)},
            paired_residual_rows=rows))
    with pytest.raises(ValueError, match="calculated answer"):
        build_native_score_inputs(_bundle(
            paired_residual_rows=[{**rows[0], "exp_pnl_sim": 0.1}]))
    with pytest.raises(ValueError, match="unsupported fields"):
        build_native_score_inputs(_bundle(
            paired_residual_rows=[{**rows[0], "abs_move": 5.0}]))
    with pytest.raises(ValueError, match="terminal"):
        build_native_score_inputs(_bundle(
            residual_recipe={"terminal_spots": (95.0, 105.0)}, paired_residual_rows=rows))


def test_rows_path_scores_under_the_no_fit_guard():
    pool = pnl_sim.ResidualPool(_history(ties=True))
    bundle = _bundle(paired_residual_rows=_recorded_rows(pool))
    with no_fit_guard():
        native, _ = _native(bundle)
    assert native == _legacy(pool)
