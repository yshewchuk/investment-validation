"""P5-4: scoring reads frozen residual pools -- key-checked, never rebuilt.

Covers the two residual populations the native stages used to re-derive per
request: the paired (err_move, err_crush) pool of the planned-exit
simulation (legacy ``engine.pnl_sim.ResidualPool``) and the driver pools of
the model stage. Parity is judged three ways: the artifact path against the
unchanged rows path on the same rows (bit for bit), against legacy
``pnl_sim.expected_pnl`` over ``ResidualPool`` (bit for bit on a pool with a
total date order), and refusal (MODEL_NOT_READY, no number) whenever the
artifact is missing or its causal key/hash disagrees with the request.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from engine import pnl_sim
from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import generate, price
from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.training.residuals import (
    build_driver_residual_pool_artifact,
    build_paired_residual_pool_artifact,
)
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from engine.v2.scoring.stages import STAGE_NAMES, NativeScoreInputs, StageReceipt

LINEAGE = Lineage(data=(DataDependency(table="tier3.panel", end_exclusive="2026-09-01"),))
KEY = {"move_model_id": "size_v1_4", "crush_model_id": "iv_crush_v1_gbm",
       "cutoff": "2026-09-01"}
TICKERS = ("AAA", "BBB", "CCC")


def _universe(days: int = 400):
    rng = np.random.default_rng(17)
    forecasts, outcomes, crush = [], [], []
    for index, day in enumerate(pd.date_range("2025-06-01", periods=days, freq="D")):
        ticker, stamp = TICKERS[index % 3], str(day.date())
        forecasts.append({"ticker": ticker, "event_date": stamp,
                          "pred_abs_move": 4.0 + index % 20 / 10.0,
                          "pred_iv_crush_30": -20.0})
        outcomes.append({"ticker": ticker, "event_date": stamp,
                         "abs_move": float(rng.uniform(0, 12))})
        crush.append({"ticker": ticker, "event_date": stamp,
                      "crush_pct_iv30": float(rng.uniform(-45, 0))})
    return forecasts, outcomes, crush


def _paired(crush_filter=None):
    forecasts, outcomes, crush = _universe()
    if crush_filter is not None:
        crush = [row for row in crush if crush_filter(row)]
    return build_paired_residual_pool_artifact(
        forecasts, outcomes, crush, move_model_id=KEY["move_model_id"],
        crush_model_id=KEY["crush_model_id"], cutoff=KEY["cutoff"], lineage=LINEAGE)


def _rows(artifact) -> list[dict]:
    return [{"event_date": row[0], "pred_abs_move": row[2], "err_move": row[3],
             "err_crush": row[4]} for row in artifact.rows]


def _request(alpha: float = 0.5, strategy: str = "STR-THRU") -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-frozen", calendar_revision="cal-1", strategy_version=strategy,
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": alpha},
    )


# ---------------------------------------------------------------------------
# paired pool, direct native inputs (same fixture shape as
# test_v2_scoring_stage_ownership.py's planned-exit parity test)
# ---------------------------------------------------------------------------


def _planned(simulation: dict, context_extra: dict | None = None) -> NativeScoreInputs:
    context = {"ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
               "exit_date": "2026-09-09", "expiry": "2026-09-18", "spot": 100.0,
               **(context_extra or {})}
    geometry = generate("STR-THRU", {**context, "forecast_abs_move": 7.0})
    quotes = {(leg.right, leg.strike, leg.expiry): {"bid": 1.95, "ask": 2.05}
              for leg in geometry.legs}
    return NativeScoreInputs(
        context=context, features={"model_inputs": {}},
        forecast={"driver_name": "abs_move", "models": {
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
            "forecast_abs_move": {"intercept": 7.0, "coefficients": {}},
            "pred_iv_crush": {"intercept": -20.0, "coefficients": {}},
        }},
        geometry=geometry, pricing=price(geometry, quotes, 0.5), analogs={},
        simulation={"pre_iv30": 40.0, **simulation},
        gate={"model": {"intercept": 0.0, "coefficients": {"exp_pnl_sim": 1.0}},
              "threshold": 0.0},
        chooser={}, diagnostics={}, source_ref="frozen-residual-fixture",
        stage_receipts=tuple(StageReceipt(stage, "declared", "declared")
                             for stage in STAGE_NAMES if stage != "diagnostics"),
    )


def _artifact_sim(artifact, key=KEY) -> dict:
    return {"mode": "planned_exit", "paired_residual_artifact": artifact,
            "paired_residual_key": dict(key)}


_SIM_FIELDS = ("exp_pnl_sim", "win_sim", "sim_p10", "sim_p90", "pool_n")


def _sim(record) -> tuple:
    return tuple(record.resolved_request.get(field) for field in _SIM_FIELDS)


def _simulated(record) -> bool:
    """The frozen pool was read and simulated. (The fixture row itself is
    refused NO_SCORE for reasons unrelated to residuals; what matters here is
    that the simulation ran and nothing refused MODEL_NOT_READY.)"""
    return (record.resolved_request.get("exp_pnl_sim") is not None
            and "MODEL_NOT_READY" not in record.reason_codes)


def test_artifact_path_equals_rows_path_and_legacy_expected_pnl_bit_for_bit():
    artifact = _paired()
    frozen = application.score_one(_request(), _planned(_artifact_sim(artifact)))
    rows = application.score_one(
        _request(), _planned({"mode": "planned_exit", "residuals": _rows(artifact)}))

    assert _simulated(frozen)
    assert _sim(frozen) == _sim(rows)
    inputs = _planned(_artifact_sim(artifact))
    priced = price(inputs.geometry, {
        (leg.right, leg.strike, leg.expiry): {"bid": leg.bid, "ask": leg.ask}
        for leg in inputs.pricing.legs}, 0.5)
    history = pd.DataFrame(_rows(artifact))
    history["event_date"] = pd.to_datetime(history["event_date"])
    legacy = pnl_sim.expected_pnl(
        exit_legs=[{"strike": leg.strike, "qty": leg.quantity,
                    "side": "sell" if leg.side == "buy" else "buy"} for leg in priced.legs],
        spot=100.0, entry_cost=priced.entry_cost, pre_iv30=40.0, pred_abs_move=7.0,
        pred_iv_crush=-20.0, dte_exit=9.0, event_date=pd.Timestamp("2026-09-16").normalize(),
        pool=pnl_sim.ResidualPool(history), key="STR-THRU",
    )
    assert _sim(frozen) == tuple(legacy[field] for field in _SIM_FIELDS)


@pytest.mark.parametrize("change", [
    {"move_model_id": "size_v1_3"},
    {"crush_model_id": "iv_crush_v0"},
    {"cutoff": "2026-09-15"},   # a later-fold pool than the request asked for
    {"cutoff": None},
    {"content_hash": "sha256:" + "0" * 64},
])
def test_any_key_or_pin_disagreement_is_model_not_ready(change):
    record = application.score_one(
        _request(), _planned(_artifact_sim(_paired(), {**KEY, **change})))
    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_sim") is None
    assert record.validation_status == "refused"


def test_missing_artifact_missing_key_part_or_two_pools_is_model_not_ready():
    artifact = _paired()
    partial = {key: value for key, value in KEY.items() if key != "cutoff"}
    for simulation in (
        _artifact_sim(None),
        _artifact_sim(artifact, partial),
        {**_artifact_sim(artifact), "residuals": _rows(artifact)},
    ):
        record = application.score_one(_request(), _planned(simulation))
        assert "MODEL_NOT_READY" in record.reason_codes
        assert record.resolved_request.get("exp_pnl_sim") is None


def test_request_context_cannot_change_the_frozen_pool():
    """Two requests whose contexts differ in everything a bounded scorer
    context would (ticker, the loaded universe) read the same frozen pool
    and produce the same simulation; the artifact is unchanged afterwards and
    what scoring caches from it is read-only."""
    artifact = _paired()
    before = artifact.payload()
    first = application.score_one(_request(), _planned(
        _artifact_sim(artifact), {"ticker": "AAA", "loaded_tickers": ["AAA"]}))
    second = application.score_one(_request(), _planned(
        _artifact_sim(artifact), {"ticker": "ZZZ", "loaded_tickers": list(TICKERS) + ["ZZZ"]}))

    assert _sim(first) == _sim(second)
    assert artifact.payload() == before
    from engine.v2.scoring.native_residuals import paired_arrays_from_artifact

    arrays, flag = paired_arrays_from_artifact(_artifact_sim(artifact))
    assert flag is None
    with pytest.raises(ValueError):
        arrays[2][0] = 0.0


def test_a_context_scoped_pool_is_refused_under_the_release_pin():
    universe = _paired()
    scoped = _paired(lambda row: row["ticker"] == "AAA")
    pinned = {**KEY, "content_hash": universe.content_hash}
    ok = application.score_one(_request(), _planned(_artifact_sim(universe, pinned)))
    refused = application.score_one(_request(), _planned(_artifact_sim(scoped, pinned)))
    assert _simulated(ok)
    assert "MODEL_NOT_READY" in refused.reason_codes


def test_frozen_path_scores_under_the_no_fit_guard_and_a_rebuild_cannot():
    artifact = _paired()
    with no_fit_guard():
        record = application.score_one(_request(), _planned(_artifact_sim(artifact)))
        with pytest.raises(RuntimeFitForbidden):
            _paired()
    assert _simulated(record)


def test_simulation_receipt_binds_the_artifact_identity_not_its_rows():
    universe, scoped = _paired(), _paired(lambda row: row["ticker"] != "CCC")
    receipts = []
    for artifact in (universe, scoped, universe):
        record = application.score_one(_request(), _planned(_artifact_sim(artifact)))
        receipts.append({item["stage"]: item["input_hash"]
                         for item in record.resolved_request["native_stage_receipts"]})
    assert receipts[0]["simulation"] == receipts[2]["simulation"]
    assert receipts[0]["simulation"] != receipts[1]["simulation"]


# ---------------------------------------------------------------------------
# paired pool through SourceBundle
# ---------------------------------------------------------------------------


_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
    ("P", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
}


def _bundle(**overrides) -> SourceBundle:
    base: dict = dict(
        source_ref="frozen-residual-bundle",
        context={"ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-09", "expiry": "2026-09-18", "spot": 100.0,
                 "pre_iv30": 40.0},
        raw_quotes=_QUOTES, feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "m1"}},
        forecast_recipes={
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
            "forecast_abs_move": {"intercept": 7.0, "coefficients": {}},
            "pred_iv_crush": {"intercept": -20.0, "coefficients": {}},
        },
        model_artifact_refs={"driver_prediction": "sha256:m1",
                             "forecast_abs_move": "sha256:m2", "pred_iv_crush": "sha256:m3"},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {"exp_pnl_sim": 1.0}},
                     "threshold": 0.0},
    )
    base.update(overrides)
    return SourceBundle(**base)


def test_bundle_declared_pool_matches_the_direct_rows_path():
    artifact = _paired()
    inputs = build_native_score_inputs(_bundle(
        paired_residual_recipe=dict(KEY), paired_residual_artifact=artifact))
    frozen = application.score_one(_request(), inputs)
    rows = application.score_one(_request(), replace(
        inputs, simulation={"mode": "planned_exit", "residuals": _rows(artifact)}))
    assert _simulated(frozen)
    assert _sim(frozen) == _sim(rows)


def test_bundle_declared_but_unresolved_pool_is_model_not_ready():
    inputs = build_native_score_inputs(_bundle(paired_residual_recipe=dict(KEY)))
    record = application.score_one(_request(), inputs)
    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_sim") is None


def test_bundle_refuses_to_combine_a_terminal_simulation_with_a_frozen_pool():
    with pytest.raises(ValueError, match="terminal"):
        build_native_score_inputs(_bundle(
            residual_recipe={"terminal_spots": (95.0, 105.0)},
            paired_residual_recipe=dict(KEY), paired_residual_artifact=_paired()))


def test_undeclared_bundle_simulation_block_is_unchanged():
    recipe = {"terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5), "capital_at_risk": 1.0}
    inputs = build_native_score_inputs(_bundle(residual_recipe=recipe))
    assert inputs.simulation == recipe


# ---------------------------------------------------------------------------
# driver pools through SourceBundle (model stage)
# ---------------------------------------------------------------------------


_PAYOFF_ROWS = [
    {"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0, "exit_date": "2026-09-01"},
    {"driver": 10.0, "spot_entry": 100.0, "exit_value": 6.0, "exit_date": "2026-09-01"},
]


def _driver_rows(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"prediction": float(p), "residual": float(r)}
            for p, r in zip(rng.uniform(0, 12, n), rng.normal(0, 1.5, n))]


def _driver_artifact(rows, role="size", model_id="m1", fold="2026-09-01"):
    return build_driver_residual_pool_artifact(
        rows, role=role, model_id=model_id, fold=fold, lineage=LINEAGE)


_DRIVER_KEY = {"role": "size", "model_id": "m1", "fold": "2026-09-01"}


def _model_bundle(**overrides) -> SourceBundle:
    base = dict(
        context={"ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0},
        forecast_recipes={"driver_prediction": {"intercept": 7.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:m1"},
        residual_recipe={"terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
                         "capital_at_risk": 1.0},
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 64},
        payoff_source_rows=_PAYOFF_ROWS,
    )
    base.update(overrides)
    return _bundle(**base)


@pytest.mark.parametrize("n", [50, 3000])
def test_driver_artifact_path_equals_rows_path(n):
    rows = _driver_rows(n, seed=n)
    by_rows = application.score_one(_request(), build_native_score_inputs(
        _model_bundle(model_residual_rows=rows)))
    by_artifact = application.score_one(_request(), build_native_score_inputs(_model_bundle(
        model_residual_artifact_recipe={"driver": _DRIVER_KEY},
        model_residual_artifacts={"driver": _driver_artifact(rows)})))
    assert by_rows.resolved_request["exp_pnl_model"] is not None
    for field in ("exp_pnl_model", "win_model"):
        assert by_artifact.resolved_request[field] == by_rows.resolved_request[field]


@pytest.mark.parametrize("key, artifacts", [
    ({"driver": {**_DRIVER_KEY, "fold": "2026-10-01"}}, "good"),
    ({"driver": {**_DRIVER_KEY, "model_id": "m2"}}, "good"),
    ({"driver": {"role": "size", "model_id": "m1"}}, "good"),
    ({"driver": _DRIVER_KEY}, "none"),
    ({"driver": {**_DRIVER_KEY, "content_hash": "sha256:" + "1" * 64}}, "good"),
])
def test_driver_artifact_disagreement_is_model_not_ready(key, artifacts):
    artifact = _driver_artifact(_driver_rows(50, seed=1)) if artifacts == "good" else None
    record = application.score_one(_request(), build_native_score_inputs(_model_bundle(
        model_residual_artifact_recipe=key, model_residual_artifacts={"driver": artifact})))
    assert "MODEL_NOT_READY" in record.reason_codes
    assert record.resolved_request.get("exp_pnl_model") is None


def test_runup_reads_both_frozen_driver_pools():
    implied = _driver_rows(80, seed=3)
    move = _driver_rows(80, seed=4)
    runup_rows = [{**row, "spot_exit": 100.0, "strike": 100.0} for row in _PAYOFF_ROWS]
    common = dict(
        strategy="STR-RUNUP",
        context={"ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0,
                 "strike": 100.0, "days_before_print": 7.0},
        forecast_recipes={"driver_prediction": {"intercept": 7.0, "coefficients": {}},
                          "runup_move_prediction": {"intercept": 1.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:m1",
                             "runup_move_prediction": "sha256:m2"},
        payoff_source_rows=runup_rows,
    )
    by_rows = application.score_one(_request(strategy="STR-RUNUP"), build_native_score_inputs(
        _model_bundle(**common, model_residual_rows=implied, runup_move_residual_rows=move)))
    move_key = {"role": "runup_move", "model_id": "m2", "fold": "2026-09-01"}
    by_artifact = application.score_one(
        _request(strategy="STR-RUNUP"), build_native_score_inputs(_model_bundle(
            **common,
            model_residual_artifact_recipe={"driver": _DRIVER_KEY, "runup_move": move_key},
            model_residual_artifacts={
                "driver": _driver_artifact(implied),
                "runup_move": _driver_artifact(move, role="runup_move", model_id="m2")})))
    assert by_rows.resolved_request["exp_pnl_model"] is not None
    assert by_artifact.resolved_request["exp_pnl_model"] == by_rows.resolved_request["exp_pnl_model"]

    missing_move = application.score_one(
        _request(strategy="STR-RUNUP"), build_native_score_inputs(_model_bundle(
            **common, model_residual_artifact_recipe={"driver": _DRIVER_KEY},
            model_residual_artifacts={"driver": _driver_artifact(implied)})))
    assert "MODEL_NOT_READY" in missing_move.reason_codes


def test_bundle_refuses_rows_and_artifacts_together_and_unknown_slots():
    rows = _driver_rows(50, seed=1)
    with pytest.raises(ValueError, match="cannot combine"):
        build_native_score_inputs(_model_bundle(
            model_residual_rows=rows,
            model_residual_artifact_recipe={"driver": _DRIVER_KEY},
            model_residual_artifacts={"driver": _driver_artifact(rows)}))
    with pytest.raises(ValueError, match="unknown model residual artifact slots"):
        build_native_score_inputs(_model_bundle(
            model_residual_artifact_recipe={"other": _DRIVER_KEY}))
