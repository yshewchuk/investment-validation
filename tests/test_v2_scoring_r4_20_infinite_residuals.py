"""R4-20 gap 1: +/-inf paired residual rows, exactly as legacy simulates them.

Legacy ``engine.pnl_sim.ResidualPool`` drops a row only where a value is
MISSING (``history.dropna(subset=[event_date, pred_abs_move, err_move,
err_crush])``). A +/-inf error or prediction stays in the pool: it counts in
``pool_n``, it moves the decile edges, and a draw that lands on it runs
through ``expected_pnl``'s numpy arithmetic, which turns it into a finite
number (``err_move=-inf`` clips the move to 0; ``err_crush=-inf`` floors the
vol at ``MIN_VOL``) or into NaN (``err_move=+inf`` reaches ``inf * 0`` in the
Black-Scholes put; ``err_crush=+inf`` gives ``inf / inf``). Native used to
drop every non-finite row, so it simulated a different pool. Each case below
is compared, NaN-for-NaN, with legacy called the way
``Scorer._expectation`` calls it, on the rows path (a capture's recorded
population), the frozen-artifact path and the artifact builder.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from engine import pnl_sim
from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import generate, price
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.residual_artifact import make_paired_residual_pool_artifact
from engine.v2.models.training.residuals import build_paired_residual_pool_artifact
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
KEY = {"move_model_id": "size_v1_4", "crush_model_id": "iv_crush_v1_gbm", "cutoff": EVENT}
LINEAGE = Lineage(data=(DataDependency(table="tier3.panel", end_exclusive=EVENT),))
INF = float("inf")

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _history(column: str | None, value: float | None, *, every: int = 9) -> pd.DataFrame:
    """500 distinct-date events; every ``every``-th row gets ``column=value``
    (predictions near the query's 7.0, so the special rows share its decile
    and are drawn), plus NaN rows legacy drops."""
    rng = np.random.default_rng(41)
    days = pd.date_range("2025-03-01", periods=520, freq="D")
    rows = []
    for index, day in enumerate(days):
        row = {"event_date": day, "ticker": f"T{index % 7}",
               "pred_abs_move": 3.0 + (index * 7919 % 400) / 50.0,
               "err_move": float(rng.normal(0.0, 3.0)),
               "err_crush": float(rng.normal(-5.0, 12.0))}
        if column is not None and index % every == 0:
            row["pred_abs_move"] = 7.0 if column != "pred_abs_move" else value
            if column != "pred_abs_move":
                row[column] = value
        if index % 50 == 7:  # missing values: dropped by both sides
            row["err_crush"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sample(frac=1.0, random_state=5).reset_index(drop=True)


CASES = {
    "err_move=+inf": ("err_move", INF),
    "err_move=-inf": ("err_move", -INF),
    "err_crush=+inf": ("err_crush", INF),
    "err_crush=-inf": ("err_crush", -INF),
    "pred_abs_move=+inf": ("pred_abs_move", INF),
    "pred_abs_move=-inf": ("pred_abs_move", -INF),
}


def _request() -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-inf", calendar_revision="cal-1", strategy_version="STR-THRU",
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at=EVENT, snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5},
    )


def _bundle(**overrides) -> SourceBundle:
    base: dict = dict(
        source_ref="r4-20-inf", context=dict(CONTEXT), raw_quotes=QUOTES,
        feature_vector={}, feature_missing_mask={},
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
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
    )
    base.update(overrides)
    return SourceBundle(**base)


def _native(bundle: SourceBundle) -> tuple:
    record = application.score_one(_request(), build_native_score_inputs(bundle))
    return tuple(record.resolved_request.get(field) for field in SIM_FIELDS)


def _legacy(pool: pnl_sim.ResidualPool) -> tuple:
    geometry = generate("STR-THRU", {**CONTEXT, "forecast_abs_move": 7.0})
    priced = price(geometry, {(leg.right, leg.strike, leg.expiry): QUOTES[
        (leg.right, leg.strike, leg.expiry)] for leg in geometry.legs}, 0.5)
    result = pnl_sim.expected_pnl(
        exit_legs=[{"strike": float(leg.strike), "qty": float(leg.quantity),
                    "side": "sell" if str(leg.side).lower() == "buy" else "buy"}
                   for leg in priced.legs],
        spot=100.0, entry_cost=priced.entry_cost, pre_iv30=40.0,
        pred_abs_move=7.0, pred_iv_crush=-20.0, dte_exit=9.0,
        event_date=pd.Timestamp(EVENT).normalize(), pool=pool, key="STR-THRU",
    )
    return tuple(result[field] for field in SIM_FIELDS)


def _same(left: tuple, right: tuple) -> bool:
    """Bit-for-bit, with NaN equal to NaN (legacy's own answer can be NaN)."""
    def one(a, b):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return type(a) is type(b) and a == b
    return len(left) == len(right) and all(one(a, b) for a, b in zip(left, right))


def _artifact(history: pd.DataFrame):
    clean = history.dropna(subset=["event_date", "pred_abs_move", "err_move", "err_crush"])
    return make_paired_residual_pool_artifact(
        move_model_id=KEY["move_model_id"], crush_model_id=KEY["crush_model_id"],
        cutoff=KEY["cutoff"], lineage=LINEAGE,
        rows=[(str(row.event_date.date()), row.ticker, row.pred_abs_move,
               row.err_move, row.err_crush) for row in clean.itertuples()],
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_rows_path_keeps_infinite_rows_and_equals_legacy(case):
    pool = pnl_sim.ResidualPool(_history(*CASES[case]))
    rows = pool.documented_population(pool.before(pd.Timestamp(EVENT)))
    assert any(not math.isfinite(row[name]) for row in rows
               for name in ("pred_abs_move", "err_move", "err_crush"))
    native, expected = _native(_bundle(paired_residual_rows=rows)), _legacy(pool)
    assert _same(native, expected), (native, expected)


@pytest.mark.parametrize("case", sorted(CASES))
def test_artifact_path_keeps_infinite_rows_and_equals_legacy(case):
    history = _history(*CASES[case])
    native = _native(_bundle(paired_residual_recipe=dict(KEY),
                             paired_residual_artifact=_artifact(history)))
    expected = _legacy(pnl_sim.ResidualPool(history))
    assert _same(native, expected), (native, expected)


def test_the_cases_cover_both_finite_and_nan_legacy_answers():
    """The fixture is informative: some cases stay finite, some go NaN."""
    answers = {case: _legacy(pnl_sim.ResidualPool(_history(*CASES[case])))[0]
               for case in CASES}
    assert math.isnan(answers["err_move=+inf"]) and math.isnan(answers["err_crush=+inf"])
    assert math.isfinite(answers["err_move=-inf"]) and math.isfinite(answers["err_crush=-inf"])


@pytest.mark.parametrize("case", ["err_move=-inf", "err_crush=-inf", "pred_abs_move=+inf"])
def test_planted_defect_dropping_infinite_rows_is_caught(case):
    """Negative control: the old finite-only filter simulates another pool,
    and the comparison above would have failed on it."""
    history = _history(*CASES[case])
    finite = history[np.isfinite(history[["pred_abs_move", "err_move", "err_crush"]]
                                 .to_numpy(dtype=float)).all(axis=1)]
    pool = pnl_sim.ResidualPool(finite)
    rows = pool.documented_population(pool.before(pd.Timestamp(EVENT)))
    assert not _same(_native(_bundle(paired_residual_rows=rows)),
                     _legacy(pnl_sim.ResidualPool(history)))


def test_builder_keeps_infinite_errors_exactly_as_the_legacy_merge_does():
    """``build_paired_residual_pool_artifact`` over the legacy
    ``Scorer._residual_pool`` inputs: an infinite realized value survives
    (legacy ``dropna``), inf - inf is NaN and drops, NaN drops."""
    forecasts = [
        {"ticker": "A", "event_date": "2025-01-02", "pred_abs_move": 5.0, "pred_iv_crush_30": -10.0},
        {"ticker": "B", "event_date": "2025-01-03", "pred_abs_move": INF, "pred_iv_crush_30": -10.0},
        {"ticker": "C", "event_date": "2025-01-06", "pred_abs_move": INF, "pred_iv_crush_30": -10.0},
        {"ticker": "D", "event_date": "2025-01-07", "pred_abs_move": 4.0, "pred_iv_crush_30": None},
        {"ticker": "E", "event_date": "2025-01-08", "pred_abs_move": 4.0, "pred_iv_crush_30": -9.0},
    ]
    outcomes = [
        {"ticker": "A", "event_date": "2025-01-02", "abs_move": INF},
        {"ticker": "B", "event_date": "2025-01-03", "abs_move": 3.0},
        {"ticker": "C", "event_date": "2025-01-06", "abs_move": INF},
        {"ticker": "D", "event_date": "2025-01-07", "abs_move": 3.0},
        {"ticker": "E", "event_date": "2025-01-08", "abs_move": 3.0},
    ]
    crush = [
        {"ticker": "A", "event_date": "2025-01-02", "crush_pct_iv30": -20.0},
        {"ticker": "B", "event_date": "2025-01-03", "crush_pct_iv30": -INF},
        {"ticker": "C", "event_date": "2025-01-06", "crush_pct_iv30": -20.0},
        {"ticker": "D", "event_date": "2025-01-07", "crush_pct_iv30": -20.0},
        {"ticker": "E", "event_date": "2025-01-08", "crush_pct_iv30": float("nan")},
    ]
    artifact = build_paired_residual_pool_artifact(
        forecasts, outcomes, crush, move_model_id="m", crush_model_id="c",
        cutoff=None, lineage=LINEAGE)
    # engine/score.py Scorer._residual_pool, verbatim over the same frames.
    frame = (pd.DataFrame(forecasts)
             .merge(pd.DataFrame(outcomes), on=["ticker", "event_date"], how="inner")
             .merge(pd.DataFrame(crush), on=["ticker", "event_date"], how="inner"))
    frame["pred_iv_crush_30"] = frame["pred_iv_crush_30"].astype(float)
    frame["err_move"] = frame["abs_move"] - frame["pred_abs_move"]
    frame["err_crush"] = frame["crush_pct_iv30"] - frame["pred_iv_crush_30"]
    legacy = pnl_sim.ResidualPool(frame.dropna(subset=["err_move", "err_crush"]))
    expected = [(row["event_date"][:10], row["pred_abs_move"], row["err_move"], row["err_crush"])
                for row in legacy.documented_population(len(legacy))]
    assert [(r[0], r[2], r[3], r[4]) for r in artifact.rows] == expected
    assert [r[1] for r in artifact.rows] == ["A", "B"]
