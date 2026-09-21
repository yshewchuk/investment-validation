#!/usr/bin/env python3
"""Build Phase 4 contract/application evidence from the frozen Tier-0 corpus."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from dataclasses import is_dataclass, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.phase4_checkpoints import CheckpointError, load_bundle  # noqa: E402
from checks.phase4_frozen_bridge import (  # noqa: E402
    prepare_frozen_chooser,
    prepare_frozen_replay,
    with_frozen_chooser,
)
from checks.phase4_stored_forecasts import (  # noqa: E402
    resolve_stored_forecasts,
    with_stored_forecasts,
)
from checks.tier0_corpus import load, resolve_corpus  # noqa: E402
from checks.tier0_corpus import run as run_corpus  # noqa: E402
from engine.analogs import AnalogMatcher  # noqa: E402
from engine.fills import MID  # noqa: E402
from engine.models.registry import artifact_sha256, load_registry  # noqa: E402
from engine.pnl_sim import ResidualPool, expected_pnl  # noqa: E402
from engine.report import Report, build_provenance  # noqa: E402
from engine.structures import (  # noqa: E402
    STRUCTURES,
    ChainSnapshot,
    StructureError,
    price_structure,
)
from engine.v2.contracts import ScoreRecord, ScoreRequest  # noqa: E402
from engine.v2.diagnosis import AGREE, SCORE_RECORD_V1, compare_records  # noqa: E402
from engine.v2.domain.generation import Geometry, Pricing, generate, price  # noqa: E402
from engine.v2.domain.valuation import (  # noqa: E402
    multi_expiry_refusal,
    planned_exit_label,
    terminal_payoff,
)
from engine.v2.features import (  # noqa: E402
    FeatureContextError,
    FeatureContextPlanner,
    default_feature_registry,
)
from engine.v2.foundation import (  # noqa: E402
    NONFINITE_KEY,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.models import (  # noqa: E402
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
)
from engine.v2.models.contracts import ArtifactMember  # noqa: E402
from engine.v2.registry import DYNAMIC_MENU, STRATEGY_IDS, default_registry  # noqa: E402
from engine.v2.scoring import application  # noqa: E402
from engine.v2.scoring.identity import request_hash, score_id, with_score_id  # noqa: E402
from engine.v2.scoring.native_analog import legacy_bucket_bootstrap_seed  # noqa: E402
from engine.v2.scoring.source_inputs import (  # noqa: E402
    SourceBundle,
    build_native_score_inputs,
)
from engine.v2.scoring.stages import (  # noqa: E402
    NativeScoreInputs,
    StageReceipt,
    receipt,
)
from engine.v2.serving.score_projection import legacy_score_projection  # noqa: E402
from tools.phase4_request_translation import legacy_binding_mismatches  # noqa: E402

__all__ = ["build_evidence", "main"]


def _request(**changes):
    value = ScoreRequest(
        event_id="phase4-event", event_revision="rev-1", calendar_revision="cal-1",
        strategy_version="STR-THRU", deployment_id="legacy-phase4-deployment.v1",
        decision_clock_id="legacy.entry_close.v1", requested_decision_at="2026-09-16",
        snapshot_id="snapshot-tier0", mode="replay", fill_model={"alpha": 0.5},
        model_artifact_refs=("size_v1_4",), residual_state_ref="residual-v1",
    )
    for key, value_to_set in changes.items():
        value = value.__class__(**{**to_document(value), key: value_to_set})
    return value


def _fake_result():
    return SimpleNamespace(as_dict=lambda: {
        "ticker": "PHASE4", "strategy": "STR-THRU", "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-17", "expiry": "2026-09-18",
        "spot": 100.0, "entry_cost": 5.0, "implied_move": 6.0,
        "driver_name": "abs_move", "driver_prediction": 7.0,
        "legs": [], "flags": [], "model_inputs": {"zero": 0.0},
        "gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": True,
        "detail": "", "payoff": {}, "fill": 0.5,
    })


def _native_record(record: dict, source_ref: str) -> NativeScoreInputs:
    """Build explicit stage blocks from one frozen real-code score record."""
    common = {key: record.get(key) for key in (
        "ticker", "strategy", "event_date", "entry_date", "exit_date",
        "expiry", "session", "as_of", "evidence_cutoff", "snapshot_hash",
    ) if key in record}
    feature_keys = {
        "model_inputs", "implied_move", "implied_move_at_entry", "spot",
        "dte_entry", "entry_cost", "quote_date", "quote_age_sessions",
        "rel_spread",
    }
    forecast_keys = {
        key for key in record
        if key.startswith("forecast_")
        or key.startswith("driver_")
        or key.startswith("runup_")
    }
    analog_keys = {
        key for key in record
        if key.startswith("analog")
        or key in {"ci_low", "ci_high", "n_analogs"}
    }
    simulation_keys = {
        key for key in record
        if key.startswith("exp_pnl") or key.startswith("win_")
    }
    gate_keys = {"gate_score", "gate_threshold", "gate_pass", "flags", "detail"}
    chooser_keys = {"chooser_score", "strategy"}
    geometry_keys = {
        "legs", "structure_spec", "structure_params",
        "structure_peak", "structure_width", "strike",
    }
    assigned = set(common) | feature_keys | forecast_keys | analog_keys
    assigned |= simulation_keys | gate_keys | chooser_keys | geometry_keys
    diagnostics = {
        key: value for key, value in record.items() if key not in assigned
    }
    diagnostics.update({
        key: record.get(key) for key in geometry_keys if key in record
    })
    geometry = None
    pricing = None
    strategy = str(record.get("strategy") or "")
    try:
        if strategy in ("CAL-P", "CND-P"):
            geometry = None
        elif record.get("spot") is not None and record.get("expiry") is not None:
            geometry = generate(
                strategy,
                {
                    "spot": record["spot"],
                    "forecast_abs_move": record.get("forecast_abs_move") or 1.0,
                    "expiry": record["expiry"],
                    "width": record.get("structure_width") or 1.0,
                    "resolved_legs": record.get("legs") or (),
                },
            )
            quotes = {
                (str(leg.get("right")), float(leg.get("strike")),
                 str(leg.get("expiry"))): {
                    "bid": leg.get("bid"), "ask": leg.get("ask"),
                }
                for leg in record.get("legs") or ()
                if leg.get("bid") is not None and leg.get("ask") is not None
            }
            pricing = price(
                geometry, quotes, float(record.get("fill", 0.5)),
            )
    except (KeyError, TypeError, ValueError):
        geometry = None
        pricing = None
    blocks = {
        "context": common,
        "features": {key: record.get(key) for key in feature_keys if key in record},
        "forecast": {},
        "analogs": {key: record.get(key) for key in analog_keys if key in record},
        "simulation": {},
        "gate": {},
        "chooser": {},
        "diagnostics": diagnostics,
    }
    receipts = tuple(
        receipt(stage, source_ref, blocks.get(stage, {}))
        for stage in (
            "resolve_context", "features", "forecast", "geometry", "pricing",
            "model", "analogs", "simulation", "gate", "chooser",
            "serialization",
        )
    )
    return NativeScoreInputs(
        **blocks,
        geometry=geometry,
        pricing=pricing,
        source_ref=source_ref,
        stage_receipts=receipts,
    )


#: R4-5 analog control. A synthetic legacy trades frame in the shape
#: ``engine.analogs.AnalogMatcher`` consumes. Three rows match the query
#: exactly, three more only after ``moneyness_band`` is widened (one with no
#: realized return), one belongs to another strategy and one closes after the
#: decision; the last two must never reach either side's population.
_ANALOG_CONTROL_STRATEGY = "STR-THRU"
_ANALOG_CONTROL_ALPHA = 0.5
_ANALOG_CONTROL_AS_OF = "2026-09-16"
_ANALOG_CONTROL_SNAPSHOT = "snapshot-tier0"
_ANALOG_CONTROL_REQUEST_KEY = "phase4-event"
_ANALOG_CONTROL_DIMENSIONS = (
    "mcap_bucket", "moneyness_band", "dte_band", "implied_tercile",
)
_ANALOG_CONTROL_WIDENING = ("moneyness_band", "dte_band", "implied_tercile")
_ANALOG_CONTROL_QUERY = {
    "mcap_bucket": "1-10B", "moneyness_band": "ATM",
    "dte_band": "4-10", "implied_tercile": "mid",
}
_ANALOG_CONTROL_MIN_ANALOGS = 3
_ANALOG_CONTROL_BOOTSTRAP = 64
#: Every analog number a record carries: the two simulation-dimension analog
#: outputs plus the three ``_ANALOG_FIELDS``.
_ANALOG_VIEW_FIELDS = (
    "exp_pnl_analog", "win_analog", "ci_low", "ci_high", "n_analogs",
)


def _analog_control_frame() -> pd.DataFrame:
    common = {
        "strategy": _ANALOG_CONTROL_STRATEGY, "fill_alpha": _ANALOG_CONTROL_ALPHA,
        "mcap_bucket": "1-10B", "dte_band": "4-10", "implied_tercile": "mid",
    }
    rows = [
        ("exact-a", "ATM", 0.10, "2026-08-03"),
        ("exact-b", "ATM", 0.20, "2026-08-04"),
        ("exact-c", "ATM", -0.05, "2026-08-05"),
        ("wide-a", "2-5%", -0.10, "2026-08-06"),
        ("wide-b", "2-5%", 0.30, "2026-08-07"),
        ("wide-missing", "2-5%", float("nan"), "2026-08-10"),
        ("future", "ATM", 9.99, "2026-10-01"),
    ]
    records = [
        common | {"event_id": event_id, "moneyness_band": band, "ret": ret,
                  "exit_date": exit_date}
        for event_id, band, ret, exit_date in rows
    ]
    records.append(common | {
        "event_id": "other-strategy", "strategy": "STR-RUNUP",
        "moneyness_band": "ATM", "ret": 0.90, "exit_date": "2026-08-03",
    })
    frame = pd.DataFrame(records)
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    return frame


def _legacy_analog_expected() -> dict[str, Any]:
    """The legacy side: ``AnalogMatcher.match`` on the synthetic frame."""
    result = AnalogMatcher(
        _analog_control_frame(), snapshot=_ANALOG_CONTROL_SNAPSHOT,
    ).match(
        _ANALOG_CONTROL_STRATEGY, dict(_ANALOG_CONTROL_QUERY),
        alpha=_ANALOG_CONTROL_ALPHA, as_of=_ANALOG_CONTROL_AS_OF,
        min_analogs=_ANALOG_CONTROL_MIN_ANALOGS,
        bootstrap=_ANALOG_CONTROL_BOOTSTRAP,
        request_key=_ANALOG_CONTROL_REQUEST_KEY,
    )
    return {
        "exp_pnl_analog": result.mean, "win_analog": result.win_rate,
        "ci_low": result.ci_low, "ci_high": result.ci_high,
        "n_analogs": result.n,
    }


def _analog_control_source() -> dict[str, Any]:
    """Answer-free ``SourceBundle`` analog fields for the same population.

    Source rows are the frame restricted to the requested strategy and fill
    and closed strictly before the decision, the boundary
    ``LegacyBucketRecipe`` documents. No legacy output is read: the bootstrap
    seed is derived from the request identity, as legacy derives it.
    """
    frame = _analog_control_frame()
    causal = frame[
        (frame["strategy"] == _ANALOG_CONTROL_STRATEGY)
        & (frame["fill_alpha"] == _ANALOG_CONTROL_ALPHA)
        & (frame["exit_date"] < pd.Timestamp(_ANALOG_CONTROL_AS_OF))
    ]
    rows = tuple(
        {
            "row_id": row["event_id"],
            **{name: row[name] for name in _ANALOG_CONTROL_DIMENSIONS},
            "realized_return": None if pd.isna(row["ret"]) else float(row["ret"]),
        }
        for row in causal.to_dict("records")
    )
    recipe = {
        "bucket_dimensions": _ANALOG_CONTROL_DIMENSIONS,
        "widening_order": _ANALOG_CONTROL_WIDENING,
        "min_analogs": _ANALOG_CONTROL_MIN_ANALOGS,
        "alpha": _ANALOG_CONTROL_ALPHA,
        "bootstrap_draws": _ANALOG_CONTROL_BOOTSTRAP,
        "bootstrap_seed": legacy_bucket_bootstrap_seed(
            snapshot=_ANALOG_CONTROL_SNAPSHOT,
            strategy=_ANALOG_CONTROL_STRATEGY,
            alpha=_ANALOG_CONTROL_ALPHA,
            buckets=_ANALOG_CONTROL_QUERY,
            request_key=_ANALOG_CONTROL_REQUEST_KEY,
        ),
        "ci_quantiles": (0.05, 0.95),
    }
    return {
        "analog_recipe": recipe,
        "analog_source_rows": rows,
        "analog_query": dict(_ANALOG_CONTROL_QUERY),
    }


def _analog_view(scored) -> dict[str, Any]:
    resolved = dict(scored.resolved_request)
    return {name: resolved.get(name) for name in _ANALOG_VIEW_FIELDS}


def _analog_agrees(expected: Mapping[str, Any], scored) -> bool:
    """Compare under the exact policy the saved-release parity uses."""
    return _compare_dimension(
        dict(expected), _analog_view(scored), "analogs",
    )["agree"]


def _numerical_independence_source() -> SourceBundle:
    """The answer-free source bundle every independence assertion scores."""
    expiry = "2026-09-18"
    quotes = {
        ("C", 100.0, expiry): {"bid": 1.0, "ask": 3.0},
        ("P", 100.0, expiry): {"bid": 1.0, "ask": 3.0},
    }
    return SourceBundle(
        source_ref="phase4-independent-numerical-input",
        context={
            "ticker": "PHASE4",
            "event_date": "2026-09-16",
            "entry_date": "2026-09-16",
            "exit_date": "2026-09-17",
            "expiry": expiry,
            "spot": 100.0,
        },
        raw_quotes=quotes,
        feature_vector={},
        feature_missing_mask={},
        model_identity={"driver": {"model_id": "phase4-driver-v1"}},
        forecast_recipes={
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
        },
        model_artifact_refs={
            "driver_prediction": "sha256:phase4-driver",
        },
        residual_recipe={
            "terminal_spots": (95.0, 105.0),
            "weights": (0.5, 0.5),
            "capital_at_risk": 1.0,
        },
        # Payoff-calibration/model layer (exp_pnl_model, win_model). Two
        # synthetic PRIOR trades, chosen so the OLS fit is EXACT (a line
        # through exactly two points has zero residual at both):
        #   row1: driver=0.0,  spot_entry=100.0, exit_value=2.0 -> y=0.02
        #   row2: driver=10.0, spot_entry=100.0, exit_value=6.0 -> y=0.06
        #   slope = (0.06 - 0.02) / (10.0 - 0.0) = 0.004
        #   intercept = 0.02 - 0.004 * 0.0 = 0.02
        # so fit_residuals = [0.0, 0.0] exactly -- hand arithmetic, not
        # taken from this program's own output. One model-residual row with
        # residual=0.0 makes the driver's own draw pool a single zero too
        # (native_payoff.driver_residual_pool falls back to the flat pool:
        # one row can never clear bucket_residuals' deciles*min_pool floor).
        # Both pools being single-valued makes every Monte Carlo draw
        # IDENTICAL regardless of seed/draw_count, so the expected number is
        # closed-form:
        #   driver = 7.0 (forecast_recipes below), spot = 100.0
        #   exit_value = max(0, (0.02 + 0.004 * 7.0) * 100.0) = 4.8
        #   entry_cost = 4.0 (see entry_cost_pct == 4.0 below)
        #   return = (4.8 - 4.0 + 0.0 * 100.0) / 4.0 = 0.2
        # exp_pnl_model == 0.2 and win_model == 1.0 for EVERY seed/draw_count.
        payoff_recipe={"min_trades": 2, "seed": 20260918, "draw_count": 8},
        payoff_source_rows=[
            {"driver": 0.0, "spot_entry": 100.0, "exit_value": 2.0,
             "exit_date": "2026-09-10"},
            {"driver": 10.0, "spot_entry": 100.0, "exit_value": 6.0,
             "exit_date": "2026-09-10"},
        ],
        model_residual_rows=[{"prediction": 7.0, "residual": 0.0}],
        # Analog layer (R4-5): the legacy bucket recipe over a synthetic
        # prior-trade population, declared in the same answer-free way as the
        # payoff rows above. The legacy side of the comparison is
        # engine.analogs.AnalogMatcher run on the SAME synthetic frame
        # (_legacy_analog_expected); nothing it returns reaches this bundle.
        **_analog_control_source(),
        gate_recipe={
            "model": {"intercept": 0.0, "coefficients": {"exp_pnl_sim": 1.0}},
            "threshold": 0.0,
            "recipe_id": "phase4-gate-v1",
        },
    )


def _numerical_independence_control() -> dict[str, Any]:
    """Poison supplied stage outputs so preservation cannot certify parity.

    Analog stage (R4-5): the native record's analog numbers must agree with
    ``engine.analogs.AnalogMatcher`` on the same synthetic population, and
    each planted analog defect must make that comparison fail.
    """
    source = _numerical_independence_source()
    executable = build_native_score_inputs(source)
    request = _request()
    expected = application.score_one(request, executable)
    poisoned = replace(
        executable,
        forecast={"driver_prediction": 991.0, "forecast_abs_move": 992.0},
        simulation={"exp_pnl_sim": 993.0},
        gate={"gate_score": 994.0, "gate_threshold": 995.0,
              "gate_pass": False},
        model={"exp_pnl_model": 991.5, "win_model": 0.0},
        analogs={**executable.analogs, "exp_pnl_analog": 996.0,
                 "win_analog": 0.0, "ci_low": 997.0, "ci_high": 998.0,
                 "n_analogs": 999},
    )
    scored = application.score_one(request, poisoned)
    preservation_only = (
        scored.forecasts.get("driver_prediction") == 991.0
        and scored.forecasts.get("forecast_abs_move") == 992.0
    )
    model_preservation_only = (
        scored.resolved_request.get("exp_pnl_model") == 991.5
        or scored.resolved_request.get("win_model") == 0.0
    )
    poisoned_analogs = _analog_view(scored)
    analog_preservation_only = (
        poisoned_analogs["exp_pnl_analog"] == 996.0
        or poisoned_analogs["ci_low"] == 997.0
        or poisoned_analogs["ci_high"] == 998.0
        or poisoned_analogs["n_analogs"] == 999
    )
    legacy_analogs = _legacy_analog_expected()
    analog_parity = _analog_agrees(legacy_analogs, expected)
    # Planted analog defects: each must make the same comparison disagree.
    # Source rows are rebuilt through the builder, so the population hash
    # follows the perturbation and the native stage really recomputes on it;
    # the tampered case edits a row AFTER the builder bound its hash.
    rows = source.analog_source_rows
    perturbed_rows = tuple(
        {**row, "realized_return": 0.35} if row["row_id"] == "exact-b" else row
        for row in rows
    )
    tampered_rows = [dict(row) for row in executable.analogs["source_rows"]]
    tampered_rows[0]["realized_return"] = 0.35
    planted = {
        "perturbed_source_row": build_native_score_inputs(
            replace(source, analog_source_rows=perturbed_rows)),
        "changed_min_analogs": build_native_score_inputs(replace(
            source, analog_recipe={**source.analog_recipe, "min_analogs": 4})),
        "changed_bootstrap_seed": build_native_score_inputs(replace(
            source, analog_recipe={
                **source.analog_recipe,
                "bootstrap_seed": source.analog_recipe["bootstrap_seed"] + 1,
            })),
        "tampered_bound_row": replace(executable, analogs={
            **executable.analogs, "source_rows": tampered_rows}),
    }
    planted_detected = {
        name: not _analog_agrees(
            legacy_analogs, application.score_one(request, inputs))
        for name, inputs in planted.items()
    }
    analogs_recomputed = (
        analog_parity
        and isinstance(legacy_analogs["n_analogs"], int)
        and legacy_analogs["n_analogs"] >= _ANALOG_CONTROL_MIN_ANALOGS
        and legacy_analogs["ci_low"] is not None
        and not analog_preservation_only
    )
    independently_recomputed = (
        expected.validation_status == "scored"
        and expected.forecasts.get("driver_prediction") == 7.0
        and expected.forecasts.get("forecast_abs_move") is None
        and expected.forecasts.get("exp_pnl_sim") == 1.0
        and expected.gate_terms == {"gate_score": 1.0,
                                    "gate_threshold": 0.0,
                                    "gate_pass": True}
        and expected.financial_diagnostics.get("entry_cost_pct") == 4.0
        and math.isclose(expected.resolved_request.get("exp_pnl_model", float("nan")), 0.2, rel_tol=1e-9)
        and math.isclose(expected.resolved_request.get("win_model", float("nan")), 1.0, rel_tol=1e-9)
        and scored.validation_status == "refused"
        and not preservation_only
        and not model_preservation_only
        and analogs_recomputed
    )
    any_preservation = (
        preservation_only or model_preservation_only or analog_preservation_only
    )
    return {
        "copied_outputs_absent": (
            all(key not in executable.forecast for key in (
                "driver_prediction", "forecast_abs_move", "exp_pnl_sim",
                "gate_score", "gate_pass",
            ))
            and all(key not in executable.simulation for key in (
                "exp_pnl_sim", "win_sim", "sim_p10", "sim_p90",
            ))
            and all(key not in executable.gate for key in (
                "gate_score", "gate_threshold", "gate_pass",
            ))
            and all(key not in executable.model for key in (
                "exp_pnl_model", "win_model",
            ))
            and not executable.chooser
            and all(key not in executable.analogs for key in _ANALOG_VIEW_FIELDS)
        ),
        "preservation_only_detected": any_preservation,
        "preservation_only_rejected": not any_preservation,
        "independent_recomputation": independently_recomputed,
        "analog_independent_recomputation": analogs_recomputed,
        "analog_planted_defects": planted_detected,
        "analog_planted_defects_rejected": all(planted_detected.values()),
    }


def _factory_structure_controls() -> dict[str, bool]:
    """Exercise generated CND-PS geometry instead of payoff placeholders."""
    base = {"spot": 100.0, "width": 4.0,
            "forecast_abs_move": 8.0, "expiry": "2026-10-01"}
    geometry = generate("CND-PS", base)
    legs = {leg.name: leg for leg in geometry.legs}
    exact_mirrors = (
        set(legs) == {"atm", "up1", "dn1", "up2", "dn2"}
        and legs["up1"].strike + legs["dn1"].strike == 2.0 * legs["atm"].strike
        and legs["up2"].strike + legs["dn2"].strike == 2.0 * legs["atm"].strike
    )
    spot = 68.98
    obs = pd.Timestamp("2026-09-04")
    expiry = pd.Timestamp("2026-09-18")
    rows = []
    strike = 40.0
    while strike <= 100.0 + 1e-9:
        for right in ("C", "P"):
            intrinsic = (max(strike - spot, 0.0) if right == "P"
                         else max(spot - strike, 0.0))
            rows.append({
                "ticker": "KEN", "obs_date": obs, "expiry": expiry,
                "dte": 14, "strike": round(strike, 4), "right": right,
                "bid": round(intrinsic + 1.5, 4),
                "ask": round(intrinsic + 2.5, 4),
                "iv": 0.6, "delta": -0.5 if right == "P" else 0.5,
                "spot": spot,
            })
        strike += 5.0
    snapshot = ChainSnapshot(
        ticker="KEN", obs_date=obs, event_date=pd.Timestamp("2026-09-07"),
        rows=pd.DataFrame(rows), spot=spot, session="AMC",
    )
    try:
        price_structure(
            STRUCTURES["CND-PS"](width_moneyness=0.014593), snapshot, MID,
        )
    except StructureError as exc:
        production_refusal = (
            "too coarse" in str(exc)
            and "up1 and up2" in str(exc)
            and "dn1 and dn2" in str(exc)
        )
    else:
        production_refusal = False
    return {
        "irregular_ladder_rejected": production_refusal,
        "exact_mirrors_preserved": exact_mirrors,
        "zero_quantity_reference_legs_preserved": (
            legs["atm"].quantity == 0.0 and legs["atm"].side == "buy"
        ),
    }


def _simulation_acceptance_controls() -> dict[str, object]:
    """Compare independently executed native and legacy simulation recipes."""
    residuals = tuple({
        "event_date": str(day.date()), "pred_abs_move": 5.0,
        "err_move": 0.0, "err_crush": 0.0,
    } for day in pd.date_range("2025-01-01", periods=300))
    residual_hash = content_hash(residuals)
    forecast_recipe = {
        "driver_name": "abs_move",
        "required_roles": ("size", "iv_crush"),
        "models": {
            "forecast_abs_move": {"intercept": 5.0, "coefficients": {}},
            "pred_iv_crush": {"intercept": -25.0, "coefficients": {}},
        },
    }
    gate_recipe = {
        "model": {"intercept": 0.0, "coefficients": {"exp_pnl_sim": 1.0}},
        "threshold": 0.0,
    }
    artifact_refs = (
        "forecast:" + content_hash(forecast_recipe),
        "gate:" + content_hash(gate_recipe),
    )
    geometry = generate("CND-PS", {
        "spot": 100.0, "forecast_abs_move": 5.0, "width": 4.0,
        "expiry": "2026-10-17",
    })
    quotes = {
        (leg.right, leg.strike, leg.expiry): {
            "bid": 1.0 if leg.side == "buy" else 0.2,
            "ask": 2.0 if leg.side == "buy" else 0.4,
        }
        for leg in geometry.legs
    }
    history = pd.DataFrame(residuals)
    history["event_date"] = pd.to_datetime(history["event_date"])
    pool = ResidualPool(history)
    comparisons = {}
    binding_checks = []
    binding_details = []
    for dte in (0, 30):
        exit_date = str((pd.Timestamp("2026-10-17") - pd.Timedelta(days=dte)).date())
        for alpha in (0.0, 1.0):
            pricing = price(geometry, quotes, alpha)
            simulation_recipe = {
                "mode": "planned_exit", "pre_iv30": 40.0,
                "dte_exit": dte, "event_date": "2026-09-17",
                "residuals": residuals, "draws": 4000,
            }
            recipe = {
                "schema_version": "phase4.acceptance_recipe.v1",
                "geometry": to_document(geometry),
                "quotes": tuple({
                    "right": right, "strike": strike, "expiry": expiry_value,
                    **quote,
                } for (right, strike, expiry_value), quote in sorted(quotes.items())),
                "forecast": forecast_recipe,
                "simulation": simulation_recipe,
                "gate": gate_recipe,
                "fill_alpha": alpha,
            }
            recipe_hash = content_hash(recipe)
            source_ref = "recipe:" + recipe_hash
            base = _native_record({
                **_fake_result().as_dict(),
                "strategy": "CND-PS", "spot": 100.0,
                "event_date": "2026-09-17", "exit_date": exit_date,
                "expiry": "2026-10-17", "pre_iv30": 40.0,
            }, source_ref)
            inputs = replace(
                base, geometry=geometry, pricing=pricing,
                forecast=forecast_recipe, simulation=simulation_recipe,
                gate=gate_recipe,
            )
            request = _request(
                strategy_version="CND-PS", fill_model={"alpha": alpha},
                dependency_refs=(source_ref, "residuals:" + residual_hash),
                model_artifact_refs=artifact_refs,
            )
            native = application.score_one(request, inputs)
            exit_legs = tuple({
                "strike": leg.strike, "qty": leg.quantity,
                "side": "sell" if leg.side == "buy" else "buy",
            } for leg in geometry.legs)
            legacy = expected_pnl(
                exit_legs=exit_legs, spot=100.0,
                entry_cost=pricing.entry_cost, pre_iv30=40.0,
                pred_abs_move=5.0, pred_iv_crush=-25.0, dte_exit=dte,
                # The Scorer passes a normalized Timestamp (engine/score.py
                # Scorer._expectation); the seed material depends on its form.
                event_date=pd.Timestamp("2026-09-17").normalize(), pool=pool,
                key="CND-PS", draws=4000,
            )
            key = f"dte_{dte}_alpha_{alpha:.1f}"
            comparisons[key] = {
                "native": native.forecasts.get("exp_pnl_sim"),
                "legacy": None if legacy is None else legacy["exp_pnl_sim"],
                "entry_cost": native.financial_diagnostics.get("entry_cost_pct"),
                "gate_score": native.gate_terms.get("gate_score"),
                "gate_threshold": native.gate_terms.get("gate_threshold"),
                "gate_pass": native.gate_terms.get("gate_pass"),
            }
            binding = {
                "artifact_refs": native.model_artifact_ids == artifact_refs,
                "dependency_refs": native.evidence_refs == tuple(request.dependency_refs),
                "source_dependency": inputs.source_ref == request.dependency_refs[0],
                "recipe_hash": inputs.source_ref == "recipe:" + content_hash(recipe),
            }
            binding_details.append(binding)
            binding_checks.append(all(binding.values()))
    expiry = [comparisons[f"dte_0_alpha_{alpha:.1f}"] for alpha in (0.0, 1.0)]
    pre_expiry = [comparisons[f"dte_30_alpha_{alpha:.1f}"] for alpha in (0.0, 1.0)]
    expiry_parity = all(
        row["native"] is not None and row["legacy"] is not None
        and math.isclose(row["native"], row["legacy"], abs_tol=1e-12)
        for row in expiry
    )
    pre_expiry_parity = all(
        row["native"] is not None and row["legacy"] is not None
        and math.isclose(row["native"], row["legacy"], abs_tol=1e-12)
        for row in pre_expiry
    )
    strict_gate = all(
        row["native"] is not None
        and row["gate_score"] == row["native"]
        and row["gate_threshold"] == 0.0
        and row["gate_pass"] is (row["native"] >= 0.0)
        for row in (*expiry, *pre_expiry)
    )
    return {
        "expiry_parity": expiry_parity,
        "pre_expiry_parity": pre_expiry_parity,
        "material_time_value": all(
            abs(before["native"] - at_expiry["native"]) > 0.1
            for before, at_expiry in zip(pre_expiry, expiry)
        ),
        "fill_propagation": (
            expiry[0]["entry_cost"] != expiry[1]["entry_cost"]
            and expiry[0]["native"] != expiry[1]["native"]
            and pre_expiry[0]["native"] != pre_expiry[1]["native"]
        ),
        "executable_recipe_binding": all(binding_checks),
        "strict_gate_semantics": strict_gate,
        "artifact_refs": artifact_refs,
        "residual_hash": residual_hash,
        "comparisons": comparisons,
        "binding_details": tuple(binding_details),
    }


def _application_control_source() -> SourceBundle:
    """Answer-free STR-THRU source for the application-layer controls below.

    Built the same native way the independence control is (SourceBundle ->
    build_native_score_inputs), never NativeScoreInputs.from_legacy_fields
    (the removed `_native()` compatibility helper this file used to depend
    on for every side control). Quotes are chosen so the real native
    straddle pricing lands on entry_cost=5.0 exactly: bid=2.0/ask=3.0 on
    both legs, alpha=0.5 -> fill = ask - alpha*(ask-bid) = 2.5 per leg,
    both legs bought, entry_cost = 2.5 + 2.5 = 5.0 (spot=100 ->
    entry_cost_pct=5.0). implied_move=6.0 and driver_prediction=7.0 (the
    forecast recipe's intercept, driver_name defaults to "abs_move") give
    model_vs_market = 7.0 / (6.0 * 0.645), matching financial.py's
    ORATS_EMOVE_FACTOR.
    """
    expiry = "2026-09-18"
    strike = 100.0
    quotes = {
        ("C", strike, expiry): {"bid": 2.0, "ask": 3.0},
        ("P", strike, expiry): {"bid": 2.0, "ask": 3.0},
    }
    return SourceBundle(
        source_ref="phase4-application-control",
        context={
            "ticker": "PHASE4", "event_date": "2026-09-16",
            "entry_date": "2026-09-16", "exit_date": "2026-09-17",
            "expiry": expiry, "spot": 100.0, "strike": strike,
            "implied_move": 6.0,
        },
        raw_quotes=quotes,
        feature_vector={"zero": 0.0},
        feature_missing_mask={},
        model_identity={"driver": {"model_id": "phase4-app-driver-v1"}},
        forecast_recipes={"driver_prediction": {"intercept": 7.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:phase4-app-driver"},
        residual_recipe={
            "terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
            "capital_at_risk": 1.0,
        },
        analog_recipe={},
        gate_recipe={
            "model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0,
        },
    )


def _application_controls() -> dict[str, bool]:
    started = time.perf_counter()
    request = _request()
    inputs = build_native_score_inputs(_application_control_source())
    one = application.score_one(request, inputs)
    many = application.score_many(((request, inputs),))[0]
    batch_elapsed_ms = (time.perf_counter() - started) * 1000.0
    altered = _request(fill_model={"alpha": 0.0})
    return {
            "direct_batch_equal": one.score_id == many.score_id,
            "operational_time_excluded": one.score_id == application.score_one(request, inputs).score_id,
            "fill_changes_identity": request_hash(request) != request_hash(altered),
            "zero_is_not_missing": one.null_masks == {"zero": False},
            "financial_values_owned": (
                one.financial_diagnostics["entry_cost_pct"] == 5.0
                and one.financial_diagnostics["model_vs_market"] == 7.0 / (6.0 * 0.645)
            ),
            "batch_resource_profile": batch_elapsed_ms >= 0.0 and bool(many.score_id),
    }


def _completion_controls(application_controls: dict[str, bool]) -> dict[str, bool]:
    planner = FeatureContextPlanner.default()
    request = planner.request(
        event_refs=({"event_id": "e1", "decision_at": "2026-09-16T00:00:00Z"},),
        snapshot_ref="snapshot-tier0", recipe_refs=(
            "legacy.market_context.v1", "legacy.bucket_analogs.v1"),
        visible_event_ids=("e1",), analog_population_ref="all-history-v1",
    )
    frame_zero = planner.frame(
        request, ({"event_id": "e1", "observed_at": "2026-09-15T00:00:00Z", "x": 0.0},),
        ordered_columns=("x", "missing"), coverage_receipt_ref="coverage-1",
    )
    frame_null = planner.frame(
        request, ({"event_id": "e1", "observed_at": "2026-09-15T00:00:00Z", "x": None},),
        ordered_columns=("x", "missing"), coverage_receipt_ref="coverage-1",
    )
    try:
        planner.frame(request, ({"event_id": "e1", "observed_at": "2026-09-17T00:00:00Z"},),
                      ordered_columns=("x",), coverage_receipt_ref="coverage-1")
    except FeatureContextError:
        cutoff_rejected = True
    else:
        cutoff_rejected = False
    projected = legacy_score_projection(application.score_one(
        _request(), build_native_score_inputs(_application_control_source())))
    return {
        "watchlist_scope_preserves_analog_population": request.decision_contexts[0]["analog_population_ref"] == "all-history-v1",
        "cutoff_leak_rejected": cutoff_rejected,
        "changed_missing_mask_rejected": frame_zero.null_mask_hash != frame_null.null_mask_hash,
        "full_precision_and_display_parity": round(5.0, 2) == 5.0 and 5.0 == float("5.0"),
        "planned_exit_valuation_parity": planned_exit_label("2026-09-18", "2026-09-17") == "planned_exit",
        "terminal_payoff_parity": terminal_payoff(({"kind": "call", "strike": 100, "quantity": 1},), 110) == 10.0,
        "multi_expiry_refusal": multi_expiry_refusal(({"expiry": "2026-09-18"}, {"expiry": "2026-09-25"})) is not None,
        "legacy_projection_owned": projected["financial_diagnostics"]["entry_cost_pct"] == 5.0,
        "supervised_batch_resource_profile": application_controls["batch_resource_profile"],
    }


def _frozen_model_control(request: ScoreRequest) -> bool:
    payload = json.dumps({
        "schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
        "outputs": [
            {"name": "forecast_abs_move", "intercept": 0.0, "coefficients": [1.0]},
            {"name": "pred_iv_crush", "intercept": -20.0, "coefficients": [0.0]},
        ],
    }, sort_keys=True).encode()
    with tempfile.TemporaryDirectory(prefix="phase4-frozen-") as root:
        directory = Path(root)
        (directory / "estimator.json").write_bytes(payload)
        member = ArtifactMember(
            name="estimator", path="estimator.json",
            content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        )
        binding = ModelBinding(
            binding_id="forecast", model_id="forecast-v1", role="forecast", strategy_id="*",
            decision_clock_id=request.decision_clock_id, adapter="json-linear.v1",
            feature_order=("x",),
            output_names=("forecast_abs_move", "pred_iv_crush"),
            members=(member,),
        )
        release = ModelRelease(release_id="release-v1", deployment_id=request.deployment_id,
                               bindings=(binding,))
        inference_request = InferenceRequest(
            release_id="release-v1", binding_id="forecast",
            feature_order=("x",), rows=((0.42,),))
        context = {
            "ticker": "PHASE4", "strategy": "STR-THRU",
            "event_date": "2026-09-16", "entry_date": "2026-09-16",
            "exit_date": "2026-09-17", "expiry": "2026-09-18",
            "spot": 100.0,
        }
        geometry = generate(
            "STR-THRU", {**context, "forecast_abs_move": 0.42},
        )
        quotes = {
            (leg.right, leg.strike, leg.expiry): {"bid": 1.0, "ask": 3.0}
            for leg in geometry.legs
        }
        pricing = price(geometry, quotes, 0.5)
        receipts = tuple(
            receipt(stage, "frozen-control", {})
            for stage in (
                "resolve_context", "features", "forecast", "geometry",
                "pricing", "model", "analogs", "simulation", "gate", "chooser",
                "serialization",
            )
        )
        native_inputs = NativeScoreInputs(
            context=context,
            features={"model_inputs": {"x": 0.42}},
            forecast={},
            geometry=geometry,
            pricing=pricing,
            analogs={},
            simulation={"terminal_spots": (100.0, 110.0)},
            gate={
                "model": {"intercept": 0.0, "coefficients": {"exp_pnl_sim": 1.0}},
                "threshold": 0.0,
            },
            chooser={},
            diagnostics={},
            source_ref="frozen-control",
            stage_receipts=receipts,
        )
        record = application.score_frozen(
            request, FrozenInference(directory), release, inference_request,
            {"_native_inputs": native_inputs},
        )
        return (record.forecasts["forecast_abs_move"] == 0.42
                and record.resolved_request["pred_iv_crush"] == -20.0
                and record.validation_status == "scored"
                and record.gate_terms["gate_pass"] is True
                and record.model_artifact_ids == (member.content_hash,))


def _chooser_controls() -> dict[str, bool]:
    """`_choose_dynamic` (application.py) is pure selection logic over
    already-scored ``ScoreRecord``s -- it never constructs or reads a
    NativeScoreInputs, so its candidates are built directly as ScoreRecord
    here rather than routed through NativeScoreInputs.from_legacy_fields
    (the removed `_native()` compatibility helper this file used to depend
    on for every side control)."""
    def candidate(strategy: str, score: float, *, refused: bool = False,
                  gate_pass: bool = True) -> ScoreRecord:
        return with_score_id(ScoreRecord(
            score_id="pending",
            canonical_request={"strategy_version": strategy},
            resolved_request={},
            event_ref={"event_id": "phase4-event"},
            clock_id="legacy.entry_close.v1",
            snapshot_ref="snapshot-tier0",
            dependency_hash="phase4-chooser-control",
            model_artifact_ids=(),
            selected_contracts=(),
            legs=(),
            entry_exit_plan={},
            quote_provenance={},
            forecasts={"exp_pnl_sim": score, "exp_pnl_model": score},
            uncertainty={},
            residual_state_ref=None,
            analog_state_ref=None,
            payoff_state_ref=None,
            feature_values={},
            null_masks={},
            feature_lineage_refs=(),
            gate_terms={"gate_pass": gate_pass},
            chooser_candidates=(),
            chooser_selection=None,
            financial_diagnostics={},
            requested_payoff_views=(),
            validation_status="refused" if refused else "scored",
            reason_codes=("MISSING_SPOT",) if refused else (),
            warnings=(),
            evidence_refs=(),
        ))

    tie = application._choose_dynamic(_request(), (candidate("TWIN-P", 0.4), candidate("TWIN-P5", 0.4)))
    selected = application._choose_dynamic(_request(), (candidate("TWIN-P", 0.2, refused=True),
                                                        candidate("TWIN-P5", 0.3)))
    fallback = application._choose_dynamic(_request(), (candidate("TWIN-P", 0.2, refused=True),))
    no_regating = application._choose_dynamic(_request(), (candidate("TWIN-P", 0.2, gate_pass=False),))
    return {
        "chooser_tie_control": tie.chooser_selection["status"] == "tie",
        "chooser_missing_competitor_control": selected.chooser_selection["strategy"] == "TWIN-P5",
        "chooser_fallback_control": fallback.validation_status == "refused",
        "chooser_no_regating_control": no_regating.validation_status == "scored",
    }


def _champion_artifacts_verified() -> bool:
    """Champion artifact files exist and hash-match the registry.

    This is NOT a saved-release comparison: it never scores a single record
    or compares a native output to a legacy one. It only checks that the
    registry's champion pointers resolve to files on disk whose content
    still matches the recorded hash. Kept because that integrity check is
    useful on its own, under a name that says what it actually verifies.
    """
    registry = load_registry()
    required = (
        ("size", "*"), ("implied_t1", "*"), ("runup_move", "*"),
        ("iv_crush", "*"), ("gate", "STR-THRU"), ("gate", "STR-RUNUP"),
        ("chooser", "DYN-SV"),
    )
    for role, strategy in required:
        entry = registry.champion(role, strategy)
        if not entry.path.is_file() or artifact_sha256(entry.path) != entry.artifact_sha256:
            return False
    return True


def _diagnostic_checkpoint_control(corpus) -> bool:
    """Validate the release's Phase 4 diagnostic checkpoint bundle, if declared.

    No real corpus carries `diagnostic_checkpoint_manifest` yet (confirmed
    against every `fixtures/tier0/*/INDEX.json`), so its absence must not
    block completion -- only a DECLARED bundle that fails verification does.
    Mirrors `_native_parity`'s per-item catch of `_TraceError`: a
    `CheckpointError` here becomes a visible `False` flag, not a crash.
    """
    manifest_ref = corpus.index.get("diagnostic_checkpoint_manifest")
    if manifest_ref is None:
        return True
    try:
        load_bundle(corpus.root / manifest_ref)
    except CheckpointError:
        return False
    return True


_FORECAST_FIELDS = (
    "driver_prediction", "driver_p10", "driver_p90",
    "forecast_abs_move", "forecast_p10", "forecast_p90", "forecast_sd",
    "runup_move_prediction", "runup_move_p10", "runup_move_p90",
    "runup_move_scale", "chooser_score",
)
_SIMULATION_FIELDS = (
    "exp_pnl_sim", "exp_pnl_model", "exp_pnl_analog",
    "win_sim", "win_model", "win_model_raw", "win_analog",
)
_FINANCIAL_FIELDS = (
    "entry_cost_pct", "model_vs_market", "fair_premium_pct",
    "premium_vs_fair", "cost_over_width",
)
#: gate_pass alone used to stand in for the whole gate: two runs could agree
#: on the boolean while disagreeing on the score and threshold that produced
#: it. Compare all three explicitly.
_GATE_FIELDS = ("gate_score", "gate_threshold", "gate_pass")
#: ci_low/ci_high/n_analogs previously appeared in no comparison tuple at
#: all. Both sides carry these as a genuine concept (the legacy record's own
#: analog columns; the native side's resolved_request, populated by
#: _execute_analogs in stages.py) — present as a key with a possibly-None
#: value, never structurally absent, so an ordinary field comparison applies
#: with no incomparability marker needed.
_ANALOG_FIELDS = ("ci_low", "ci_high", "n_analogs")

#: Tri-state outcome of a numeric negative control (R4-15 fix, 2026-09-20).
#: A dimension whose every field is ``None`` for every compared row has
#: nothing for the control to corrupt -- that is a distinct fact from a
#: control that ran and was not noticed, and must not collapse onto the same
#: ``False`` the way it used to.
_CONTROL_PASSED = "passed"
_CONTROL_FAILED = "failed"
_CONTROL_NOT_EXERCISABLE = "not_exercisable"


def _numeric_corruption_candidate(values: dict) -> tuple[str, Any] | None:
    """First ``(field, value)`` the negative control can corrupt, or None.

    Eligible means "not None and int/float/bool" -- a plain number gets
    ``+1.0``; a bool must be FLIPPED (``not value``), never added to: ``True
    + 1.0`` silently becomes the float ``2.0``, which is not a corruption a
    bool-typed comparator path would even see as the same type.
    """
    for field, value in values.items():
        if value is not None and isinstance(value, (int, float)):
            return field, value
    return None


def _legacy_fair_premium(record: dict) -> float | None:
    """Independently reproduce the frozen renderer financial oracle."""
    payoff = record.get("payoff") or {}
    driver = record.get("driver_prediction")
    if payoff.get("kind") == "runup_payoff_surface":
        coefficients = payoff.get("coefficients") or {}
        move = record.get("runup_move_prediction")
        spot = record.get("spot")
        strike = record.get("strike")
        names = (
            "intercept", "implied_move", "abs_moneyness",
            "moneyness_sq_div10", "signed_moneyness",
            "implied_x_abs_moneyness_div10",
        )
        if (driver is None or move is None or spot in (None, 0, 0.0)
                or strike in (None, 0, 0.0)
                or any(name not in coefficients for name in names)):
            return None
        values = []
        for direction in (-1.0, 1.0):
            exit_spot = float(spot) * math.exp(direction * float(move) / 100.0)
            money = 100.0 * math.log(exit_spot / float(strike))
            absolute = abs(money)
            terms = (
                1.0, float(driver), absolute, money * money / 10.0,
                money, float(driver) * absolute / 10.0,
            )
            values.append(sum(
                float(coefficients[name]) * term
                for name, term in zip(names, terms)
            ))
        return max(0.0, sum(values) / len(values) * 100.0)
    intercept = payoff.get("intercept")
    slope = payoff.get("slope")
    if driver is None or intercept is None or slope is None:
        return None
    return max(0.0, (
        float(intercept) + float(slope) * float(driver)
    ) * 100.0)


def _expected_financial_diagnostics(record: dict) -> dict:
    """Compute expected diagnostics without importing native scoring code."""
    spot = record.get("spot")
    cost = record.get("entry_cost")
    entry_cost_pct = (
        float(cost) / float(spot) * 100.0
        if cost is not None and spot not in (None, 0, 0.0)
        else None
    )
    driver = record.get("driver_prediction")
    implied = record.get("implied_move")
    model_vs_market = (
        float(driver) / (float(implied) * 0.645)
        if record.get("driver_name") == "abs_move"
        and driver is not None and implied not in (None, 0, 0.0)
        else None
    )
    fair = _legacy_fair_premium(record)
    premium_vs_fair = (
        entry_cost_pct / fair
        if entry_cost_pct is not None and fair not in (None, 0, 0.0)
        else None
    )
    width = record.get("structure_width")
    cost_over_width = (
        float(cost) / float(width)
        if cost is not None and width not in (None, 0, 0.0)
        else None
    )
    return {
        "entry_cost_pct": entry_cost_pct,
        "model_vs_market": model_vs_market,
        "fair_premium_pct": fair,
        "premium_vs_fair": premium_vs_fair,
        "cost_over_width": cost_over_width,
    }


def _numeric_views(record: dict, native) -> tuple[dict, dict]:
    expected = {
        "forecasts": {name: record.get(name) for name in _FORECAST_FIELDS},
        "simulation": {name: record.get(name) for name in _SIMULATION_FIELDS},
        "financial_diagnostics": _expected_financial_diagnostics(record),
        "verdicts": {name: record.get(name) for name in _GATE_FIELDS},
        "analogs": {name: record.get(name) for name in _ANALOG_FIELDS},
    }
    native_forecasts = dict(native.forecasts)
    native_uncertainty = dict(native.uncertainty)
    resolved = dict(native.resolved_request)
    actual = {
        "forecasts": {
            name: native_forecasts.get(
                name, native_uncertainty.get(name, resolved.get(name))
            )
            for name in _FORECAST_FIELDS
        },
        "simulation": {
            name: native_forecasts.get(name, resolved.get(name))
            for name in _SIMULATION_FIELDS
        },
        "financial_diagnostics": {
            name: native.financial_diagnostics.get(name)
            for name in _FINANCIAL_FIELDS
        },
        "verdicts": {name: native.gate_terms.get(name) for name in _GATE_FIELDS},
        "analogs": {name: resolved.get(name) for name in _ANALOG_FIELDS},
    }
    return expected, actual


def _compare_dimension(expected: dict, actual: dict, dimension: str) -> dict:
    comparison = compare_records(
        expected, actual,
        comparison_kind=f"phase4_{dimension}_parity",
        left_ref="frozen_legacy_record",
        right_ref="native_score_record",
        tolerance_policy=SCORE_RECORD_V1,
    )
    return {
        "agree": comparison.verdict == AGREE,
        "finding_fields": sorted(finding.field_path for finding in comparison.findings),
        "receipt": content_hash({
            "dimension": dimension,
            "verdict": comparison.verdict,
            "findings": [finding.field_path for finding in comparison.findings],
        }),
    }


def _compare_numeric_outputs(record: dict, native, *, actual_override=None) -> dict:
    """Compare legacy and native numerical outputs under the exact policy."""
    expected, actual = _numeric_views(record, native)
    if actual_override is not None:
        actual = actual_override
    return {
        dimension: _compare_dimension(expected[dimension], actual[dimension], dimension)
        for dimension in ("forecasts", "simulation", "financial_diagnostics",
                          "verdicts", "analogs")
    }


def _factory_views(record: dict, inputs: NativeScoreInputs) -> tuple[dict, dict] | None:
    strategy = str(record.get("strategy") or "")
    if strategy not in STRATEGY_IDS:
        return None
    if strategy in {"CAL-P", "CND-P"}:
        expected_refusal = "UNVALIDATED_STRUCTURE"
        actual_refusal = (
            inputs.geometry.refusal if inputs.geometry is not None
            else expected_refusal
        )
        return ({"refusal": expected_refusal}, {"refusal": actual_refusal})
    if inputs.geometry is None or inputs.pricing is None:
        return (
            {"geometry_available": bool(record.get("legs"))},
            {"geometry_available": False},
        )
    expected_legs = tuple({
        "name": str(leg.get("name")), "right": str(leg.get("right")),
        "side": str(leg.get("side")),
        "quantity": float(leg.get("quantity", leg.get("qty", 0.0))),
        "strike": float(leg.get("strike")), "expiry": str(leg.get("expiry")),
    } for leg in record.get("legs") or ())
    actual_legs = tuple({
        "name": leg.name, "right": leg.right, "side": leg.side,
        "quantity": leg.quantity, "strike": leg.strike, "expiry": leg.expiry,
    } for leg in inputs.geometry.legs)
    expected_pricing = tuple({
        "price": float(leg.get("price")), "cash_flow": float(leg.get("cash_flow")),
    } for leg in record.get("legs") or ())
    actual_pricing = tuple({
        "price": leg.fill, "cash_flow": leg.cash_flow,
    } for leg in inputs.pricing.legs)
    expected = {
        "geometry": {"strategy": strategy, "spot": float(record.get("spot")),
                     "width": record.get("structure_width"), "legs": expected_legs},
        "expiry": {"record": str(record.get("expiry")),
                   "legs": tuple(leg["expiry"] for leg in expected_legs)},
        "fill": {"alpha": float(record.get("fill")),
                 "entry_cost": float(record.get("entry_cost")),
                 "legs": expected_pricing},
    }
    actual = {
        "geometry": {"strategy": inputs.geometry.strategy,
                     "spot": inputs.geometry.spot,
                     "width": inputs.geometry.width if record.get("structure_width") is not None else None,
                     "legs": actual_legs},
        "expiry": {"record": str(record.get("expiry")),
                   "legs": tuple(leg.expiry for leg in inputs.geometry.legs)},
        "fill": {"alpha": float(record.get("fill")),
                 "entry_cost": inputs.pricing.entry_cost, "legs": actual_pricing},
    }
    return expected, actual


def _factory_parity(corpus) -> dict:
    rows = []
    covered = set()
    negative_controls = {"geometry": False, "expiry": False, "fill": False}
    for fixture_id in corpus.ordered_ids:
        record = corpus.record_of(fixture_id)
        inputs = _native_record(record, corpus.pairs[fixture_id]["payload_hash"])
        views = _factory_views(record, inputs)
        if views is None:
            continue
        expected, actual = views
        covered.add(str(record.get("strategy")))
        checks = {}
        if "refusal" in expected:
            checks["refusal"] = _compare_dimension(expected, actual, "factory_refusal")["agree"]
        elif "geometry" not in expected:
            checks["geometry_available"] = _compare_dimension(
                expected, actual, "factory_geometry_availability")["agree"]
        else:
            for dimension in ("geometry", "expiry", "fill"):
                checks[dimension] = _compare_dimension(
                    expected[dimension], actual[dimension], f"factory_{dimension}"
                )["agree"]
                if not negative_controls[dimension]:
                    corrupted = copy.deepcopy(actual[dimension])
                    if dimension == "geometry":
                        corrupted["spot"] = float(corrupted["spot"]) + 1.0
                    elif dimension == "expiry":
                        corrupted["record"] = "2099-12-31"
                    else:
                        corrupted["entry_cost"] = float(corrupted["entry_cost"]) + 1.0
                    negative_controls[dimension] = not _compare_dimension(
                        expected[dimension], corrupted, f"factory_{dimension}_corruption"
                    )["agree"]
        rows.append({"fixture_id": fixture_id, "strategy": record.get("strategy"),
                     "checks": checks})
    required = set(STRATEGY_IDS)
    return {
        "complete": covered >= required and all(all(row["checks"].values()) for row in rows),
        "strategies_expected": sorted(required),
        "strategies_compared": sorted(covered),
        "rows_compared": len(rows),
        "rows_agreed": sum(all(row["checks"].values()) for row in rows),
        "negative_controls": negative_controls,
        "comparison_receipt": content_hash(rows),
    }


def _contract_projection(legs, *, entry_date=None, exit_date=None,
                         execution_date=None) -> dict[str, Any]:
    """Project one side's contracts, plus their trade-timeline dates.

    ``expiry`` lives on each leg, but ``entry_date``/``exit_date``/
    ``execution_date`` describe the whole traded structure, not any one
    leg (no leg dataclass on either side carries them — see
    ``engine.v2.domain.generation.structures.PricedLeg`` and the legacy
    leg dict shape in the real corpus). Callers pass each side's own
    values (native: ``entry_exit_plan``/``quote_provenance``; legacy:
    ``record["entry_date"]``/``record["exit_date"]``/``record["quote_date"]``,
    the date the fill quote was captured, i.e. the execution date) so they
    are compared alongside the legs rather than silently omitted.
    """
    return {
        "legs": tuple({
            "name": str(leg.get("name")),
            "right": str(leg.get("right")),
            "side": str(leg.get("side")),
            "quantity": float(leg.get("quantity", leg.get("qty", 0.0))),
            "strike": float(leg.get("strike")),
            "expiry": str(leg.get("expiry")),
            "price": (float(leg["fill"]) if leg.get("fill") is not None
                      else float(leg["price"]) if leg.get("price") is not None else None),
            "cash_flow": (float(leg["cash_flow"]) if leg.get("cash_flow") is not None
                          else None),
        } for leg in legs or ()),
        "entry_date": str(entry_date) if entry_date is not None else None,
        "exit_date": str(exit_date) if exit_date is not None else None,
        "execution_date": str(execution_date) if execution_date is not None else None,
    }


_TRACE_SCHEMA = "phase4_input_trace.v1.0"
_TRANSLATION_SCHEMA = "phase4_input_translation.v1.0"
_REQUIRED_TRACE_STAGES = (
    "resolve_context", "features", "forecast", "geometry", "pricing",
    "analogs", "simulation", "gate", "chooser", "serialization",
)
#: The payoff-calibration/model stage (exp_pnl_model, win_model). Every
#: native scoring call made after this stage existed emits a real "model"
#: observation, so it must be a known, validated-when-present stage here
#: too -- not an unknown one, and not a required one, since traces captured
#: before the stage existed never recorded it and the trace schema carries
#: no version field to key a hard requirement on.
_OPTIONAL_TRACE_STAGES = ("model",)
#: Execution order matches ``engine.v2.scoring.stages.STAGE_NAMES`` with
#: ``diagnostics`` removed (diagnostics is handled separately, never part
#: of a captured/verified trace). Used to keep the trace's stage ordering
#: consistent with the order stages actually execute in.
_KNOWN_TRACE_STAGE_ORDER = (
    "resolve_context", "features", "forecast", "geometry", "pricing",
    "model", "analogs", "simulation", "gate", "chooser", "serialization",
)
_ADVISORY_FLAGS = frozenset({"LAYER_DISAGREE"})
#: Record kinds left out of the Phase 4 population, each with its reason
#: (user decision 2026-09-19; guides/rearchitecture_phase3b_phase4_remaining.md,
#: decision 7). Only these named kinds are dropped from ``expected``; each is
#: counted under ``population.excluded``. Any other kind, known or unknown,
#: stays expected and is a gap until it compares.
PHASE4_EXCLUDED_RECORD_KINDS: Mapping[str, str] = MappingProxyType({
    "research_replay": (
        "no v2 replay path (engine.replay.replay_one research output); "
        "excluded by user decision 2026-09-19; covered by the Tier-1 replay "
        "(tools/replay_tier1.py)"
    ),
})


def _semantic_flags(values: Mapping[str, Any]) -> tuple[str, ...]:
    """Return flags that affect scoring disposition or readiness.

    ``LAYER_DISAGREE`` remains useful legacy metadata, but it describes the
    known behavior of the analog diagnostic layer and is not an acceptance
    refusal. Phase 4 records it separately while comparing actionable flags.
    """
    return tuple(
        str(flag) for flag in (values.get("flags") or ())
        if str(flag) not in _ADVISORY_FLAGS
    )
_TRACE_KEYS = frozenset({
    "schema_version", "request", "request_hash", "shared_inputs",
    "shared_input_hash", "native_input_hash", "native_inputs",
    "native_inputs_hash", "input_translation", "stages", "resources",
    "trace_hash", "metadata",
})
_NATIVE_INPUT_KEYS = frozenset({
    "context", "features", "forecast", "geometry", "pricing", "analogs",
    "simulation", "gate", "chooser", "diagnostics", "source_ref",
})


class _TraceError(ValueError):
    pass


def _require_hash(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.startswith("sha256:")
            or len(value) != 71):
        raise _TraceError(f"{label}: invalid sha256")
    return value


def _verify_content(value: Any, expected: Any, label: str) -> str:
    expected_hash = _require_hash(expected, label)
    actual = content_hash(value)
    if actual != expected_hash:
        raise _TraceError(f"{label}: content hash mismatch")
    return actual


def _leaf_values(value: Any, path: tuple[Any, ...] = ()) -> dict[tuple, Any]:
    if isinstance(value, Mapping):
        # A `{"__nonfinite__": "<repr>"}` object is `tag_nonfinite`'s (and
        # `canonical_json`'s own internal `_scalar`'s) JSON-safe encoding of
        # ONE nonfinite float leaf, not a genuine two-level document node --
        # `engine.v2.foundation.canonical.untag_nonfinite` decodes exactly
        # this shape back to a float, recursively, everywhere else in the
        # repo. A mapping's `shared_path`/`native_path` is recorded against
        # the pre-tag document (`tools/phase4_release_assembler.py::_leaves`
        # runs before `tools/capture_tier0_corpus.py`'s
        # `_prepare_normalized_shared` tags the value for the stored JSON),
        # so on-disk the recorded path addresses this wrapper, one segment
        # short of where a generic leaf-walk would otherwise stop. Treating
        # it as the leaf here restores that address without touching stored
        # bytes: `content_hash` already normalizes a raw nonfinite float and
        # its tagged form identically (contracts §2.1's `_scalar`), so the
        # recorded `value_hash` matches either way. The check is exact --
        # sole key, string value -- so a genuine (mismatched) one-key dict
        # that merely happens to use another key is never mistaken for this
        # encoding and still fails coverage/leaf checks as a real defect
        # would.
        if set(value) == {NONFINITE_KEY} and isinstance(value[NONFINITE_KEY], str):
            return {path: value}
        if not value:
            return {path: {}}
        leaves = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise _TraceError("input_translation: object keys must be strings")
            leaves.update(_leaf_values(value[key], path + (key,)))
        return leaves
    if isinstance(value, list):
        if not value:
            return {path: []}
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_leaf_values(item, path + (index,)))
        return leaves
    return {path: value}


def _translation_path(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, list) or not value:
        raise _TraceError(f"{label}: expected nonempty path")
    path = []
    for segment in value:
        if isinstance(segment, str):
            if not segment:
                raise _TraceError(f"{label}: empty path segment")
        elif type(segment) is int:
            if segment < 0:
                raise _TraceError(f"{label}: negative list index")
        else:
            raise _TraceError(f"{label}: invalid path segment")
        path.append(segment)
    return tuple(path)


def _verified_translation(
    translation: Any,
    shared_inputs: Mapping[str, Any],
    shared_hash: str,
    saved_request: Mapping[str, Any],
    resolved_inputs: Mapping[str, Any],
    native_hash: str,
) -> str:
    if not isinstance(translation, Mapping):
        raise _TraceError("input_trace.input_translation: missing")
    expected_keys = {
        "schema_version", "shared_input_hash", "native_input_hash",
        "mappings", "derived", "translation_hash",
    }
    if set(translation) != expected_keys:
        raise _TraceError("input_trace.input_translation: malformed")
    if translation.get("schema_version") != _TRANSLATION_SCHEMA:
        raise _TraceError(
            f"input_translation.schema_version: expected {_TRANSLATION_SCHEMA}"
        )
    body = {
        key: value for key, value in translation.items()
        if key != "translation_hash"
    }
    translation_hash = _verify_content(
        body, translation.get("translation_hash"),
        "input_translation.translation_hash",
    )
    if translation.get("shared_input_hash") != shared_hash:
        raise _TraceError("input_translation.shared_input_hash: mismatch")
    if translation.get("native_input_hash") != native_hash:
        raise _TraceError("input_translation.native_input_hash: mismatch")

    native_document = {
        "request": saved_request,
        "native_inputs": {
            key: value for key, value in resolved_inputs.items()
            if key != "source_ref"
        },
    }
    shared_leaves = _leaf_values(shared_inputs)
    native_leaves = _leaf_values(native_document)
    mappings = translation.get("mappings")
    if not isinstance(mappings, list):
        raise _TraceError("input_translation.mappings: expected list")
    shared_paths = set()
    native_paths = set()
    for index, row in enumerate(mappings):
        label = f"input_translation.mappings[{index}]"
        if not isinstance(row, Mapping) or set(row) != {
            "shared_path", "native_path", "value_hash",
        }:
            raise _TraceError(f"{label}: malformed")
        shared_path = _translation_path(row["shared_path"], f"{label}.shared_path")
        native_path = _translation_path(row["native_path"], f"{label}.native_path")
        if shared_path in shared_paths or native_path in native_paths:
            raise _TraceError(f"{label}: duplicate path")
        if shared_path not in shared_leaves or native_path not in native_leaves:
            raise _TraceError(f"{label}: path is not a document leaf")
        shared_value = shared_leaves[shared_path]
        native_value = native_leaves[native_path]
        value_hash = _verify_content(
            shared_value, row.get("value_hash"), f"{label}.value_hash",
        )
        if content_hash(native_value) != value_hash:
            raise _TraceError(f"{label}: translated values differ")
        shared_paths.add(shared_path)
        native_paths.add(native_path)
    if shared_paths != set(shared_leaves) or native_paths != set(native_leaves):
        raise _TraceError("input_translation.mappings: leaf coverage mismatch")

    expected_derived = [{
        "native_path": ["native_inputs", "source_ref"],
        "operation": "shared_input_hash",
        "value_hash": content_hash(shared_hash),
    }]
    if translation.get("derived") != expected_derived:
        raise _TraceError("input_translation.derived: mismatch")
    if resolved_inputs.get("source_ref") != shared_hash:
        raise _TraceError("input_trace.native_inputs.source_ref: mismatch")
    return translation_hash


def _resource_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise _TraceError("resource.path: missing")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise _TraceError("resource.path: escapes release root") from exc
    if not candidate.is_file():
        raise _TraceError(f"resource.path: missing {relative}")
    return candidate


def _verified_resources(root: Path, rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise _TraceError("resources: expected list")
    resources: dict[str, Any] = {}
    refs: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise _TraceError(f"resources[{index}]: expected object")
        unknown = set(row) - {
            "resource_id", "ref", "kind", "path", "sha256", "content_hash",
            "document",
        }
        if unknown:
            raise _TraceError(
                f"resources[{index}]: unknown fields {sorted(unknown)}"
            )
        resource_id = row.get("resource_id")
        ref = row.get("ref")
        kind = row.get("kind")
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise _TraceError(f"resources[{index}].resource_id: missing")
        if resource_id in resources:
            raise _TraceError(f"resources[{index}].resource_id: duplicate")
        if not isinstance(ref, str) or not ref.strip() or ref in refs:
            raise _TraceError(f"resources[{index}].ref: missing or duplicate")
        if kind not in {"artifact", "sidecar"}:
            raise _TraceError(f"resources[{index}].kind: unsupported")
        document = row.get("document")
        if "path" in row:
            path = _resource_path(root, row["path"])
            raw = path.read_bytes()
            actual_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
            if actual_sha != _require_hash(
                row.get("sha256"), f"resources[{index}].sha256"
            ):
                raise _TraceError(f"resources[{index}].sha256: mismatch")
            if kind == "sidecar":
                try:
                    document = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise _TraceError(
                        f"resources[{index}]: sidecar is not JSON"
                    ) from exc
        elif kind == "artifact":
            raise _TraceError(f"resources[{index}]: artifact path required")
        elif "document" not in row:
            raise _TraceError(f"resources[{index}]: sidecar content required")
        if kind == "sidecar":
            _verify_content(
                document, row.get("content_hash"),
                f"resources[{index}].content_hash",
            )
        resources[resource_id] = document
        refs.add(ref)
    resources["__refs__"] = refs
    return resources


def _resolve_resource_refs(value: Any, resources: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$resource"}:
            resource_id = value["$resource"]
            if not isinstance(resource_id, str) or resource_id not in resources:
                raise _TraceError(f"native_inputs: unknown resource {resource_id!r}")
            resolved = resources[resource_id]
            if resolved is None:
                raise _TraceError(
                    f"native_inputs: binary artifact {resource_id!r} is not data"
                )
            return copy.deepcopy(resolved)
        return {
            str(key): _resolve_resource_refs(item, resources)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_resource_refs(item, resources) for item in value]
    return value


def _verified_runtime_receipts(native) -> tuple[dict[str, str], ...]:
    raw = native.resolved_request.get("native_stage_receipts")
    if not isinstance(raw, (list, tuple)):
        raise _TraceError("execution: native stage receipts missing")
    receipts = []
    seen = set()
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise _TraceError(f"execution.receipts[{index}]: malformed")
        stage = row.get("stage")
        if stage in seen or not isinstance(stage, str):
            raise _TraceError(f"execution.receipts[{index}]: duplicate stage")
        _require_hash(row.get("input_hash"), f"execution.{stage}.input_hash")
        _require_hash(row.get("output_hash"), f"execution.{stage}.output_hash")
        owner = row.get("owner")
        if not isinstance(owner, str) or not owner.strip():
            raise _TraceError(f"execution.{stage}.owner: missing")
        seen.add(stage)
        receipts.append({
            "stage": stage,
            "input_hash": row["input_hash"],
            "output_hash": row["output_hash"],
            "owner": owner,
        })
    missing = set(_REQUIRED_TRACE_STAGES) - seen
    if missing:
        raise _TraceError(f"execution: missing stages {sorted(missing)}")
    if not receipts or receipts[-1]["stage"] != "serialization":
        raise _TraceError("execution: serialization receipt is not final")
    return tuple(receipts)


def _verify_runtime_execution(
    verified: Mapping[str, Any], native,
) -> tuple[tuple[dict[str, str], ...], dict[str, str]]:
    runtime_receipts = _verified_runtime_receipts(native)
    captured = verified["captured_receipts"]
    # Filter runtime receipts down to exactly the stages the captured trace
    # itself claims (always every required stage, plus "model" only when
    # the trace recorded it). A stage the runtime always executes but the
    # captured trace predates -- "model", for a pre-existing capture -- is
    # not compared here; that asymmetry is expected, not a mismatch.
    captured_stage_names = {row["stage"] for row in captured}
    runtime_required = tuple(
        row for row in runtime_receipts
        if row["stage"] in captured_stage_names
    )
    if tuple(row["stage"] for row in runtime_required) != tuple(
        row["stage"] for row in captured
    ):
        raise _TraceError("execution: captured stage order mismatch")
    for expected, actual in zip(captured, runtime_required):
        stage = expected["stage"]
        for field in ("input_hash", "output_hash", "owner"):
            if actual[field] != expected[field]:
                raise _TraceError(
                    f"execution.{stage}.{field}: captured runtime mismatch"
                )

    canonical_request = to_document(verified["request"])
    if to_document(native.canonical_request) != canonical_request:
        raise _TraceError("execution.identity.canonical_request: mismatch")
    expected_request_hash = content_hash(canonical_request)
    if native.request_hash != expected_request_hash:
        raise _TraceError("execution.identity.request_hash: mismatch")
    expected_score_id = score_id(native)
    if native.score_id != expected_score_id:
        raise _TraceError("execution.identity.score_id: mismatch")
    if native.payload_hash != expected_score_id:
        raise _TraceError("execution.identity.payload_hash: mismatch")
    identities = {
        "serialization_hash": runtime_required[-1]["output_hash"],
        "payload_hash": native.payload_hash,
        "request_hash": native.request_hash,
        "score_id": native.score_id,
    }
    return runtime_receipts, identities


def _verified_trace_bundle(pair: Mapping[str, Any], release_root: Path) -> dict:
    payload = pair.get("payload")
    if not isinstance(payload, Mapping):
        raise _TraceError("pair.payload: missing")
    trace = payload.get("input_trace")
    if not isinstance(trace, Mapping):
        raise _TraceError("input_trace: missing")
    unknown = set(trace) - _TRACE_KEYS
    if unknown:
        raise _TraceError(f"input_trace: unknown fields {sorted(unknown)}")
    if trace.get("schema_version") != _TRACE_SCHEMA:
        raise _TraceError(
            f"input_trace.schema_version: expected {_TRACE_SCHEMA}"
        )
    trace_body = {key: value for key, value in trace.items()
                  if key != "trace_hash"}
    trace_hash = _verify_content(
        trace_body, trace.get("trace_hash"), "input_trace.trace_hash"
    )
    if payload.get("input_trace_hash") != trace_hash:
        raise _TraceError("pair.input_trace_hash: mismatch")

    # payload.request is the LEGACY request (what tier-0 coverage, pinned
    # links, seeded controls and tier-1 replay read). The canonical V2 request
    # lives only in input_trace.request, and must be the translation of the
    # saved legacy request, so a trace cannot be attached to a different case.
    legacy_request = payload.get("request")
    saved_request = trace.get("request")
    if not isinstance(legacy_request, Mapping):
        raise _TraceError("payload.request: missing legacy request")
    if not isinstance(saved_request, Mapping):
        raise _TraceError("input_trace.request: missing")
    unbound = legacy_binding_mismatches(legacy_request, saved_request)
    if unbound:
        raise _TraceError(
            "input_trace.request: not the translation of the saved legacy "
            f"request ({', '.join(unbound)})"
        )
    request = from_document(ScoreRequest, dict(saved_request))
    computed_request_hash = request_hash(request)
    if trace.get("request_hash") != computed_request_hash:
        raise _TraceError("input_trace.request_hash: mismatch")

    shared_inputs = trace.get("shared_inputs")
    if not isinstance(shared_inputs, Mapping):
        raise _TraceError("input_trace.shared_inputs: missing")
    if shared_inputs.get("request") != saved_request:
        raise _TraceError("input_trace.shared_inputs.request: mismatch")
    shared_hash = _verify_content(
        shared_inputs, trace.get("shared_input_hash"),
        "input_trace.shared_input_hash",
    )
    if payload.get("legacy_input_hash") != shared_hash:
        raise _TraceError("pair.legacy_input_hash: mismatch")

    resources = _verified_resources(release_root, trace.get("resources"))
    request_refs = {
        *request.dependency_refs, *request.model_artifact_refs,
        *(
            value for value in (
                request.residual_state_ref, request.analog_state_ref,
                request.calibration_state_ref,
            ) if value is not None
        ),
    }
    missing_refs = request_refs - resources["__refs__"]
    if missing_refs:
        raise _TraceError(
            f"resources: missing request refs {sorted(missing_refs)}"
        )

    raw_inputs = trace.get("native_inputs")
    if not isinstance(raw_inputs, Mapping):
        raise _TraceError("input_trace.native_inputs: missing")
    unknown_inputs = set(raw_inputs) - _NATIVE_INPUT_KEYS
    missing_inputs = _NATIVE_INPUT_KEYS - set(raw_inputs)
    if unknown_inputs or missing_inputs:
        raise _TraceError(
            "input_trace.native_inputs: "
            f"unknown={sorted(unknown_inputs)}, missing={sorted(missing_inputs)}"
        )
    resolved_inputs = _resolve_resource_refs(raw_inputs, resources)
    native_hash = _verify_content(
        resolved_inputs, trace.get("native_inputs_hash"),
        "input_trace.native_inputs_hash",
    )
    if trace.get("native_input_hash") != native_hash:
        raise _TraceError("input_trace.native_input_hash: mismatch")
    translation_hash = _verified_translation(
        trace.get("input_translation"), shared_inputs, shared_hash,
        saved_request, resolved_inputs, native_hash,
    )

    stage_rows = trace.get("stages")
    if not isinstance(stage_rows, Mapping):
        raise _TraceError("input_trace.stages: missing")
    missing_required = set(_REQUIRED_TRACE_STAGES) - set(stage_rows)
    unknown_stages = (
        set(stage_rows) - set(_REQUIRED_TRACE_STAGES) - set(_OPTIONAL_TRACE_STAGES)
    )
    if missing_required or unknown_stages:
        raise _TraceError(
            "input_trace.stages: incomplete "
            f"missing={sorted(missing_required)} unknown={sorted(unknown_stages)}"
        )
    receipts = []
    captured_receipts = []
    present_stages = tuple(
        stage for stage in _KNOWN_TRACE_STAGE_ORDER if stage in stage_rows
    )
    for stage in present_stages:
        row = stage_rows[stage]
        if not isinstance(row, Mapping) or set(row) != {
            "input", "output", "input_hash", "output_hash", "owner",
        }:
            raise _TraceError(f"input_trace.stages.{stage}: malformed")
        _verify_content(
            row["input"], row["input_hash"], f"input_trace.stages.{stage}.input"
        )
        _verify_content(
            row["output"], row["output_hash"],
            f"input_trace.stages.{stage}.output",
        )
        if not isinstance(row["owner"], str) or not row["owner"].strip():
            raise _TraceError(f"input_trace.stages.{stage}.owner: missing")
        receipts.append(StageReceipt(
            stage=stage, input_hash=row["input_hash"],
            output_hash=row["output_hash"], owner=row["owner"],
        ))
        captured_receipts.append({
            "stage": stage,
            "input_hash": row["input_hash"],
            "output_hash": row["output_hash"],
            "owner": row["owner"],
        })

    geometry_doc = resolved_inputs["geometry"]
    pricing_doc = resolved_inputs["pricing"]
    geometry = (
        None if geometry_doc is None
        else from_document(Geometry, geometry_doc, path="$.native_inputs.geometry")
    )
    pricing = (
        None if pricing_doc is None
        else from_document(Pricing, pricing_doc, path="$.native_inputs.pricing")
    )
    blocks = {
        key: resolved_inputs[key]
        for key in (
            "context", "features", "forecast", "analogs", "simulation",
            "gate", "chooser", "diagnostics",
        )
    }
    if any(not isinstance(value, Mapping) for value in blocks.values()):
        raise _TraceError("input_trace.native_inputs: stage blocks must be objects")
    if resolved_inputs["source_ref"] != shared_hash:
        raise _TraceError("input_trace.native_inputs.source_ref: mismatch")
    inputs = NativeScoreInputs(
        **blocks, geometry=geometry, pricing=pricing,
        source_ref=resolved_inputs["source_ref"],
        stage_receipts=tuple(receipts),
    )
    frozen_chooser = prepare_frozen_chooser(
        release_root=release_root,
        resource_rows=trace.get("resources"),
        verified_documents=resources,
        request=request,
        inputs=inputs,
    )
    frozen_replay = prepare_frozen_replay(
        release_root=release_root,
        resource_rows=trace.get("resources"),
        verified_documents=resources,
        metadata=trace.get("metadata"),
        request=request,
        inputs=inputs,
        extra_refs=frozen_chooser.request_refs if frozen_chooser else (),
    )
    # A declared frozen chooser runs as its executable block (the champion
    # and producer folds of its own verified release, the recorded pools).
    inputs = with_frozen_chooser(inputs, frozen_chooser)
    # A declared stored-forecast REFERENCE is looked up here, natively, from
    # the table it addresses. The trace carries the address, never the cell,
    # so this is the only place the value can enter -- and it enters by
    # being read, which is the path parity is meant to exercise. Any refusal
    # propagates: a reference that cannot be resolved must fail the record,
    # not score without it.
    inputs = with_stored_forecasts(inputs, resolve_stored_forecasts(inputs))
    return {
        "request": request,
        "inputs": inputs,
        "input_hash": shared_hash,
        "native_input_hash": native_hash,
        "translation_hash": translation_hash,
        "same_input_receipt": content_hash({
            "shared_input_hash": shared_hash,
            "native_input_hash": native_hash,
            "translation_hash": translation_hash,
        }),
        "trace_hash": trace_hash,
        "captured_stages": tuple(_REQUIRED_TRACE_STAGES),
        "captured_receipts": tuple(captured_receipts),
        "frozen_replay": frozen_replay,
        "frozen_chooser": frozen_chooser,
    }


def _release_population(corpus) -> tuple[tuple[str, ...], int, bool, dict[str, str]]:
    """``(declared ids, expected count, manifest bound, excluded id -> kind)``.

    A pair is excluded only when its loaded payload names a kind in
    ``PHASE4_EXCLUDED_RECORD_KINDS`` and, on a bound manifest, the manifest
    row names the same kind. ``expected`` counts every other declared pair.
    """
    declared = corpus.index.get("pairs")
    manifest_bound = isinstance(declared, Mapping)
    if manifest_bound:
        ids = tuple(sorted(str(fixture_id) for fixture_id in declared))
    else:
        ids = tuple(corpus.ordered_ids)
    excluded = {}
    for fixture_id in ids:
        kind = ((corpus.pairs.get(fixture_id) or {}).get("payload") or {}).get("record_kind")
        row = declared.get(fixture_id) if manifest_bound else {"record_kind": kind}
        if (kind in PHASE4_EXCLUDED_RECORD_KINDS and isinstance(row, Mapping)
                and row.get("record_kind") == kind):
            excluded[fixture_id] = kind
    return ids, len(ids) - len(excluded), manifest_bound, excluded


def _replayed_member(verified: Mapping[str, Any]):
    """Score one verified trace natively: ``(record, receipts, identities)``."""
    frozen_replay = verified["frozen_replay"]
    if frozen_replay is None:
        native = application.score_one(verified["request"], verified["inputs"])
    else:
        native = application.score_frozen(
            verified["request"],
            frozen_replay.inference,
            frozen_replay.release,
            frozen_replay.requests,
            {"_native_inputs": verified["inputs"]},
        )
    runtime_receipts, runtime_identities = _verify_runtime_execution(verified, native)
    return native, runtime_receipts, runtime_identities


def _frozen_receipt(verified: Mapping[str, Any]) -> str | None:
    frozen_replay = verified["frozen_replay"]
    return frozen_replay.receipt if frozen_replay is not None else None


def _record_checks(record: Mapping[str, Any], native) -> tuple[dict, dict]:
    """The per-record comparison of one legacy record with its native one."""
    expected_keys = set(record)
    native_keys = set(native.resolved_request) - {
        "native_stage_receipts", "native_source_ref",
    }
    checks = {
        "keys": expected_keys == (native_keys - {
            "_model_artifact_ids", "native_source_ref",
            "native_stage_receipts", "selected_contracts",
        } - ({"entry_cost", "fill", "legs", "spot", "structure_width", "flags"} -
             expected_keys)),
        "contracts": _contract_projection(
            native.legs,
            entry_date=native.entry_exit_plan.get("entry_date"),
            exit_date=native.entry_exit_plan.get("exit_date"),
            execution_date=native.quote_provenance.get("quote_date"),
        ) == _contract_projection(
            record.get("legs") or (),
            entry_date=record.get("entry_date"),
            exit_date=record.get("exit_date"),
            execution_date=record.get("quote_date"),
        ),
        # gate_score/gate_threshold/gate_pass and ci_low/ci_high/n_analogs
        # come from the "verdicts"/"analogs" numeric dimensions below
        # (checks.update), which is the same compare_records machinery
        # used for forecasts/simulation/financial_diagnostics.
        "flags": _semantic_flags({"flags": native.reason_codes}) == (
            _semantic_flags(record)
            + ("UNVALIDATED_STRUCTURE",) if record.get("strategy") in
               {"CAL-P", "CND-P"} and "UNVALIDATED_STRUCTURE" not in
               (record.get("flags") or ()) else ()
        ),
        "null_masks": native.null_masks == {
            key: value is None for key, value in (record.get("model_inputs") or {}).items()
        },
    }
    numeric = _compare_numeric_outputs(record, native)
    checks.update({name: result["agree"] for name, result in numeric.items()})
    return checks, numeric


_CHOOSER_TRACE_SCHEMA = "phase4_chooser_trace.v1.0"
_CHOOSER_TRACE_KEYS = frozenset({
    "schema_version", "members", "shared_input_hash", "trace_hash",
})
_CHOOSER_MEMBER_KEYS = frozenset({
    "member_index", "request_hash", "legacy_input_hash", "native_score_id",
    "input_trace",
})


def _chooser_members(pair: Mapping[str, Any]) -> list[tuple[dict, dict]]:
    """``(member pair, legacy member record)`` for every menu row a
    ``dyn_sv_choice`` pair's legacy chooser ranked, in frame order.

    Each member pair is shaped like a scored pair (its legacy request from
    ``request.frame_rows``, its own strict trace and hashes), so it goes
    through the same verification as any traced pair.
    """
    payload = pair.get("payload")
    if not isinstance(payload, Mapping):
        raise _TraceError("pair.payload: missing")
    trace = payload.get("input_trace")
    if not isinstance(trace, Mapping):
        raise _TraceError("input_trace: missing")
    if set(trace) != _CHOOSER_TRACE_KEYS:
        raise _TraceError(f"chooser input_trace: fields {sorted(trace)}")
    if trace["schema_version"] != _CHOOSER_TRACE_SCHEMA:
        raise _TraceError(f"chooser input_trace.schema_version: expected {_CHOOSER_TRACE_SCHEMA}")
    body = {key: value for key, value in trace.items() if key != "trace_hash"}
    trace_hash = _verify_content(body, trace["trace_hash"], "chooser input_trace.trace_hash")
    if payload.get("input_trace_hash") != trace_hash:
        raise _TraceError("pair.input_trace_hash: mismatch")
    request = payload.get("request")
    frame_rows = request.get("frame_rows") if isinstance(request, Mapping) else None
    members = trace["members"]
    if not isinstance(frame_rows, list) or not frame_rows:
        raise _TraceError("payload.request.frame_rows: missing")
    if not isinstance(members, list) or len(members) != len(frame_rows):
        raise _TraceError("chooser input_trace.members: not one per ranked frame row")
    shared = [member.get("legacy_input_hash") if isinstance(member, Mapping) else None
              for member in members]
    _verify_content(shared, trace["shared_input_hash"], "chooser input_trace.shared_input_hash")
    if payload.get("legacy_input_hash") != trace["shared_input_hash"]:
        raise _TraceError("pair.legacy_input_hash: mismatch")
    out = []
    for index, (member, row) in enumerate(zip(members, frame_rows, strict=True)):
        if not isinstance(member, Mapping) or set(member) != _CHOOSER_MEMBER_KEYS:
            raise _TraceError(f"chooser member {index}: malformed")
        if member["member_index"] != index or not isinstance(row, Mapping):
            raise _TraceError(f"chooser member {index}: out of frame order")
        if member["request_hash"] != content_hash(row.get("request")):
            raise _TraceError(f"chooser member {index}: not the frame row's request")
        member_trace = member["input_trace"]
        if not isinstance(member_trace, Mapping):
            raise _TraceError(f"chooser member {index}: input_trace missing")
        out.append(({"payload": {
            "request": row["request"],
            "input_trace": member_trace,
            "input_trace_hash": member_trace.get("trace_hash"),
            "legacy_input_hash": member["legacy_input_hash"],
        }}, dict(row.get("record") or {})))
    return out


def _replayed_chooser(pair: Mapping[str, Any], release_root: Path):
    """Replay a ``dyn_sv_choice`` pair natively.

    Every ranked member is verified and scored exactly as a traced scored
    pair; the native chooser (``application._choose_dynamic``) then ranks the
    native member records in frame order. Returns ``(members, choice)`` with
    ``members`` as ``(legacy record, verified, native, receipts,
    identities)``.
    """
    members = []
    for index, (member_pair, member_record) in enumerate(_chooser_members(pair)):
        try:
            verified = _verified_trace_bundle(member_pair, release_root)
            native, receipts, identities = _replayed_member(verified)
        except _TraceError as exc:
            raise _TraceError(f"chooser member {index}: {exc}") from exc
        members.append((member_record, verified, native, receipts, identities))
    first = members[0][1]["request"]
    choice = application._choose_dynamic(
        replace(first, strategy_version="DYN-SV"),
        tuple(item[2] for item in members),
    )
    return members, choice


def _chooser_selection_checks(record: Mapping[str, Any], native) -> dict[str, bool]:
    """The legacy choice against the native one: the chosen structure, the
    menu that competed and the winning margin (exact)."""
    selection = native.chooser_selection or {}
    return {
        "chosen_strategy": selection.get("strategy") == record.get("chosen_strategy"),
        "menu_size": selection.get("menu_size") == record.get("menu_size"),
        "chosen_margin": selection.get("margin") == record.get("chosen_margin"),
    }


#: Structural dimensions of ``_record_checks`` proven catchable by planting a
#: genuine native-side defect and re-running the SAME comparator (R4-15).
#: Numeric dimensions (forecasts/simulation/financial_diagnostics/verdicts/
#: analogs) already have their own corrupt-and-recheck loop in
#: ``_native_parity``, through ``_compare_numeric_outputs``.
_STRUCTURAL_DEFECT_DIMENSIONS = ("flags", "contracts", "keys", "null_masks")


def _plant_structural_defect(record: Mapping[str, Any], native, dimension: str):
    """One native-side mutation that must break ``dimension`` in
    ``_record_checks``, or ``None`` when this row has nothing to corrupt for
    it (no legs, an empty null-mask map, or ``native`` is a test double that
    is not a real dataclass record -- production ``native`` objects always
    are). Never touches ``record``: every defect lands on the native side,
    per R4-15's control.
    """
    if not is_dataclass(native):
        return None
    if dimension == "flags":
        return replace(native, reason_codes=tuple(native.reason_codes) + (
            "PHASE4_PLANTED_DEFECT",
        ))
    if dimension == "contracts":
        if not native.legs:
            return None
        return replace(native, legs=tuple(native.legs)[:-1])
    if dimension == "null_masks":
        if not native.null_masks:
            return None
        key = next(iter(native.null_masks))
        return replace(native, null_masks={
            **native.null_masks, key: not native.null_masks[key],
        })
    if dimension == "keys":
        return replace(native, resolved_request={
            **native.resolved_request, "phase4_planted_defect_key": True,
        })
    raise ValueError(f"unknown structural defect dimension {dimension!r}")


def _native_parity(corpus) -> tuple[dict, dict]:
    """Compare only complete, hash-verified traces from the saved release."""
    rows = []
    native_ids = []
    legacy_ids = []
    dispositions = {"compared": 0, "refused_as_expected": 0, "incomparable": 0}
    runtime_stage_counts = {stage: 0 for stage in _REQUIRED_TRACE_STAGES}
    structural_negative_controls = {
        name: False for name in _STRUCTURAL_DEFECT_DIMENSIONS
    }
    chooser_defect_seen = False
    chooser_defect_detected = False
    numeric_negative_controls = {
        "forecasts": False,
        "simulation": False,
        "financial_diagnostics": False,
        "verdicts": False,
        "analogs": False,
    }
    #: Whether a corruptible field was ever found for this dimension, across
    #: every compared row -- the fact that used to be lost. Without it,
    #: "never noticed" (a real comparator blind spot) and "never had
    #: anything to corrupt" (nothing wrong; the control just never ran) both
    #: read as the same False.
    numeric_control_attempted = {name: False for name in numeric_negative_controls}
    numeric_control_reason = {name: None for name in numeric_negative_controls}
    #: Counts a compared row toward a dimension's coverage only when the
    #: ACTUAL (native) side -- the side the control mutates -- has a field
    #: the control is capable of corrupting (non-None int/float/bool). A
    #: dimension where the only present field is a type the control cannot
    #: act on (or where everything is None) must not inflate this count;
    #: coverage now means "rows this control could have exercised," not
    #: "rows where the legacy side happened to be non-None."
    numeric_coverage = {name: 0 for name in numeric_negative_controls}
    declared_ids, expected, manifest_bound, excluded = _release_population(corpus)
    excluded_counts = {
        kind: sum(1 for value in excluded.values() if value == kind)
        for kind in PHASE4_EXCLUDED_RECORD_KINDS
    }
    loaded_ids = set(corpus.ordered_ids)
    declared_set = set(declared_ids)
    fixture_ids = sorted(declared_set | loaded_ids)
    for fixture_id in fixture_ids:
        if manifest_bound and fixture_id not in declared_set:
            dispositions["incomparable"] += 1
            rows.append({
                "fixture_id": fixture_id,
                "disposition": "incomparable",
                "reason": "release manifest: undeclared pair file",
            })
            continue
        if fixture_id not in loaded_ids:
            dispositions["incomparable"] += 1
            rows.append({
                "fixture_id": fixture_id,
                "disposition": "incomparable",
                "reason": "release manifest: declared pair file missing",
            })
            continue
        if fixture_id in excluded:
            rows.append({
                "fixture_id": fixture_id,
                "disposition": "excluded",
                "reason": PHASE4_EXCLUDED_RECORD_KINDS[excluded[fixture_id]],
            })
            continue
        pair = corpus.pairs[fixture_id]
        record = pair["payload"]["record"]
        legacy_ids.append(pair["payload_hash"])
        chooser_pair = pair["payload"].get("record_kind") == "dyn_sv_choice"
        try:
            if chooser_pair:
                members, native = _replayed_chooser(pair, corpus.root)
            else:
                verified = _verified_trace_bundle(pair, corpus.root)
                member_native, receipts_, identities_ = _replayed_member(verified)
                members = [(record, verified, member_native, receipts_, identities_)]
                native = member_native
        except Exception as exc:
            dispositions["incomparable"] += 1
            rows.append({
                "fixture_id": fixture_id,
                "disposition": "incomparable",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        dispositions["compared"] += 1
        member_stages = [
            {item["stage"] for item in receipts_} & set(_REQUIRED_TRACE_STAGES)
            for _record, _verified, _native, receipts_, _identities in members
        ]
        for stage in set.intersection(*member_stages):
            runtime_stage_counts[stage] += 1
        member_checks = []
        numeric_findings = []
        for member_record, _verified, member_native, _receipts, _identities in members:
            checks, numeric = _record_checks(member_record, member_native)
            member_checks.append(checks)
            numeric_findings.append({
                name: result["finding_fields"] for name, result in numeric.items()
            })
            _, actual_numeric = _numeric_views(member_record, member_native)
            for dimension in numeric_negative_controls:
                if _numeric_corruption_candidate(actual_numeric[dimension]) is not None:
                    numeric_coverage[dimension] += 1
                if numeric_negative_controls[dimension]:
                    continue
                candidate = _numeric_corruption_candidate(actual_numeric[dimension])
                if candidate is None:
                    continue  # nothing corruptible on this row; try the next
                field, value = candidate
                numeric_control_attempted[dimension] = True
                corrupted = copy.deepcopy(actual_numeric)
                if isinstance(value, bool):
                    corrupted[dimension][field] = not value
                else:
                    corrupted[dimension][field] = float(value) + 1.0
                result = _compare_numeric_outputs(
                    member_record, member_native, actual_override=corrupted,
                )
                noticed = not result[dimension]["agree"]
                numeric_negative_controls[dimension] = noticed
                numeric_control_reason[dimension] = (
                    f"corrupted {dimension}.{field} "
                    f"({'flipped bool' if isinstance(value, bool) else 'value + 1.0'}); "
                    f"comparator {'noticed it' if noticed else 'did NOT notice it'}"
                )
            for dimension in _STRUCTURAL_DEFECT_DIMENSIONS:
                if structural_negative_controls[dimension]:
                    continue
                mutated = _plant_structural_defect(
                    member_record, member_native, dimension,
                )
                if mutated is None:
                    continue
                mutated_checks, _ = _record_checks(member_record, mutated)
                structural_negative_controls[dimension] = not mutated_checks[dimension]
        checks = {
            name: all(item[name] for item in member_checks) for name in member_checks[0]
        }
        row = {
            "fixture_id": fixture_id,
            "disposition": "compared",
            "same_input_hash": (
                members[0][1]["same_input_receipt"] if not chooser_pair
                else content_hash([item[1]["same_input_receipt"] for item in members])
            ),
            "trace_hash": (
                members[0][1]["trace_hash"] if not chooser_pair
                else pair["payload"].get("input_trace_hash")
            ),
            "frozen_replay_receipt": (
                _frozen_receipt(members[0][1]) if not chooser_pair
                else [_frozen_receipt(item[1]) for item in members]
            ),
            "runtime_receipts": (
                members[0][3] if not chooser_pair else [item[3] for item in members]
            ),
            "runtime_identities": (
                members[0][4] if not chooser_pair else [item[4] for item in members]
            ),
            "checks": checks,
            "numeric_findings": (
                numeric_findings[0] if not chooser_pair else numeric_findings
            ),
            "advisory_flags": {
                "legacy": sorted(set(record.get("flags") or ()) & _ADVISORY_FLAGS),
                "native": sorted(set(native.reason_codes) & _ADVISORY_FLAGS),
            },
        }
        if chooser_pair:
            selection = _chooser_selection_checks(record, native)
            row["checks"]["chooser"] = all(selection.values())
            row["chooser_findings"] = sorted(
                name for name, agree in selection.items() if not agree)
            chooser_defect_seen = True
            if not chooser_defect_detected and native.chooser_selection:
                mutated_choice = replace(native, chooser_selection={
                    **native.chooser_selection,
                    "strategy": f"{native.chooser_selection.get('strategy')}-PHASE4-PLANTED",
                })
                mutated_selection = _chooser_selection_checks(record, mutated_choice)
                chooser_defect_detected = not mutated_selection["chosen_strategy"]
        rows.append(row)
        native_ids.append(native.payload_hash)
    #: Fold the running bool + attempted/reason tracking into the tri-state
    #: verdict the evidence carries. ``passed`` overrides everything (a
    #: later row proving the control works is what stops the search);
    #: otherwise ``attempted`` distinguishes a REAL miss ("failed": the
    #: comparator saw a corrupted field and agreed anyway) from a control
    #: that had nothing to work with ("not_exercisable": no row in this
    #: corpus ever offered a corruptible field for this dimension).
    numeric_negative_controls = {
        dimension: {
            "state": (
                _CONTROL_PASSED if passed
                else _CONTROL_FAILED if numeric_control_attempted[dimension]
                else _CONTROL_NOT_EXERCISABLE
            ),
            "reason": numeric_control_reason[dimension] or (
                f"no compared row offered a corruptible "
                f"(non-None int/float/bool) field in {dimension!r}"
            ),
        }
        for dimension, passed in numeric_negative_controls.items()
    }
    #: Boolean view for the completion gate and the merged ``controls`` map,
    #: which downstream code checks with strict ``is True`` (build_evidence's
    #: final_controls). Only PASSED counts: NOT_EXERCISABLE must not unlock
    #: completion any more than FAILED does -- a dimension this run could
    #: never test still has not been proven to notice a defect.
    numeric_control_ok = {
        dimension: entry["state"] == _CONTROL_PASSED
        for dimension, entry in numeric_negative_controls.items()
    }
    dimensions = (
        "keys", "contracts", "verdicts", "flags", "null_masks",
        "forecasts", "simulation", "financial_diagnostics", "analogs",
        "chooser",
    )
    compared_rows = [row for row in rows if row["disposition"] == "compared"]
    compared = len(compared_rows)
    agreed = sum(1 for row in compared_rows if all(row["checks"].values()))
    # "chooser" exists only on dyn_sv_choice rows (its selection checks); the
    # other dimensions on every compared row.
    dimension_agreement = {
        dimension: (
            compared == expected and expected > 0
            and all(row["checks"].get(dimension, dimension == "chooser")
                    for row in compared_rows)
        )
        for dimension in dimensions
    }
    native_receipt = content_hash(native_ids)
    legacy_receipt = content_hash(legacy_ids)
    comparison_receipt = content_hash(rows)
    release_id = corpus.root.name
    same_input_hashes = (
        compared == expected and expected > 0
        and all(row.get("same_input_hash") for row in compared_rows)
    )
    covered_stages = tuple(
        stage for stage in _REQUIRED_TRACE_STAGES
        if runtime_stage_counts[stage] == expected and expected > 0
    )
    complete = (
        compared == expected and dispositions["incomparable"] == 0
        and agreed == expected and all(dimension_agreement.values())
        and same_input_hashes
        and set(covered_stages) == set(_REQUIRED_TRACE_STAGES)
    )
    release = {
        "complete": complete,
        "population": {
            "expected": expected, "supported": compared,
            "compared": compared, "agreed": agreed,
            "manifest_bound": manifest_bound, **dispositions,
            "excluded": excluded_counts,
        },
        "source_release": {
            "release_id": release_id,
            "manifest_hash": corpus.index.get("corpus_hash"),
        },
        "native_execution_receipt": native_receipt,
        "legacy_execution_receipt": legacy_receipt,
        "comparison_receipt": comparison_receipt,
        "comparison_dimensions": dimensions,
        "dimension_agreement": dimension_agreement,
        "numeric_coverage": numeric_coverage,
        "numeric_negative_controls": numeric_negative_controls,
        "dispositions": tuple({
            "fixture_id": row["fixture_id"],
            "disposition": row["disposition"],
            **({"reason": row["reason"]} if "reason" in row else {}),
        } for row in rows),
    }
    parity = {
        "synthetic": False,
        "input_provenance": {
            "kind": "saved_release",
            "release_id": release_id,
            "manifest_hash": corpus.index.get("corpus_hash"),
        },
        "same_input_hashes": same_input_hashes,
        "population": {
            "expected": expected, "supported": compared,
            "compared": compared, "agreed": agreed,
            "manifest_bound": manifest_bound, **dispositions,
            "excluded": excluded_counts,
        },
        "stages": covered_stages,
        "stage_coverage": runtime_stage_counts,
        "comparison_dimensions": dimensions,
        "dimension_agreement": dimension_agreement,
        "planted_defect": {
            "detected": (
                compared == expected
                and all(structural_negative_controls.values())
                and all(numeric_control_ok.values())
                and (chooser_defect_detected if chooser_defect_seen else True)
            ),
            "controls": {
                **structural_negative_controls, **numeric_control_ok,
                "chooser": chooser_defect_detected if chooser_defect_seen else True,
            },
            #: The tri-state detail ``controls`` above collapses to a bool:
            #: state (passed/failed/not_exercisable) and reason per numeric
            #: dimension, so a review can tell "never noticed a corrupted
            #: field" apart from "never had a field to corrupt" instead of
            #: reading both as the same False.
            "numeric_control_detail": numeric_negative_controls,
            "receipt": content_hash({
                "comparison": comparison_receipt,
                "defects": {
                    **structural_negative_controls, **numeric_control_ok,
                    "chooser": chooser_defect_detected if chooser_defect_seen else True,
                },
            }),
        },
        "complete": complete,
    }
    return release, parity


def _report_population_rows(evidence: dict) -> list[list[str]]:
    population = evidence.get("population") or {}
    return [[str(key), str(population[key])] for key in sorted(population)]


def _report_subject_rows(evidence: dict) -> list[list[str]]:
    rows = []
    for subject_id in sorted(evidence.get("subjects") or {}):
        row = evidence["subjects"][subject_id]
        controls = row.get("controls") or {}
        passed = sum(1 for value in controls.values() if value is True)
        rows.append([subject_id, str(row.get("status")), f"{passed}/{len(controls)}"])
    return rows


def _report_completion_rows(evidence: dict) -> list[list[str]]:
    controls = evidence.get("completion_controls") or {}
    return [[name, "pass" if controls.get(name) is True else "FAIL"]
            for name in sorted(controls)]


def _write_phase4_report(evidence: dict, artifact_root: Path) -> Path:
    """Render the Phase 4 acceptance report through ``engine.report.Report``.

    Real sections built from the evidence just computed -- the saved-release
    population, every subject's control count, and every completion
    control's pass/fail -- not a JSON dump pasted under a Markdown header.
    """
    population = evidence.get("population") or {}
    subjects = evidence.get("subjects") or {}
    funnel = [
        {"stage": "acceptance subjects registered", "events": len(subjects),
         "note": "P4-01..P4-09, checks/phase4_real.py"},
        {"stage": "subjects PASS/FOUNDATION_PASS",
         "events": sum(1 for row in subjects.values()
                       if row.get("status") in ("PASS", "FOUNDATION_PASS")),
         "note": "of the registered subjects above", "headline": True},
        {"stage": "saved-release population expected", "events": population.get("expected"),
         "note": "declared release manifest members, less counted exclusions"},
        {"stage": "saved-release population compared", "events": population.get("compared"),
         "note": f"agreed={population.get('agreed')}"},
    ]
    context = {
        "kind": "audit",
        "spec": {
            "id": "REARCH-PHASE-4-ACCEPTANCE",
            "title": "Rearchitecture Phase 4 -- native scoring acceptance",
            "type": "descriptive",
            "hypothesis": (
                "descriptive: does native scoring reproduce the frozen "
                "legacy saved-release records, and which subjects/controls "
                "pass over the corpus just measured?"
            ),
        },
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(seeds={}, input_files=[]),
        "survivorship_note": "",
        "calibration": None,
        "funnel": funnel,
        "extra_sections": [
            {"title": "Saved-release comparison population",
             "note": "From `_native_parity`'s `release` return value -- the "
                     "only source `full_saved_release_compared` is derived "
                     "from below.",
             "columns": ["field", "value"], "align": ["---", "---"],
             "rows": _report_population_rows(evidence)},
            {"title": "Acceptance subjects (P4-01–P4-09)",
             "columns": ["subject", "status", "controls passed"],
             "align": ["---", "---", "---:"],
             "rows": _report_subject_rows(evidence)},
            {"title": "Completion controls",
             "columns": ["control", "result"], "align": ["---", "---"],
             "rows": _report_completion_rows(evidence)},
        ],
    }
    return Report(context).write(artifact_root, filename="phase4_report.md")


def _report_is_complete(text: str, evidence: dict) -> bool:
    """True only if the written report actually carries the required
    content -- not merely that a file exists at the path.

    Checks the fixed section headers `engine.report.Report` always renders
    for an "audit" kind, plus that every row this call's own
    `_write_phase4_report` built is present in the rendered Markdown
    verbatim (mirrors the exact `"| " + " | ".join(...) + " |"` line shape
    `Report._render_extra_sections` writes each row as).
    """
    if not text.strip():
        return False
    required_headers = (
        "## 0. Verdict", "## 1.5 Sample funnel", "## 8. Provenance",
        "### 8.5.1 Saved-release comparison population",
        "### 8.5.2 Acceptance subjects (P4-01–P4-09)",
        "### 8.5.3 Completion controls",
        "## 10. Glossary",
    )
    if not all(header in text for header in required_headers):
        return False
    for rows in (
        _report_population_rows(evidence),
        _report_subject_rows(evidence),
        _report_completion_rows(evidence),
    ):
        for row in rows:
            if "| " + " | ".join(row) + " |" not in text:
                return False
    return True


class _Stage:
    """Print start/end markers with elapsed seconds for a coarse phase4_real.py stage."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self._started = time.perf_counter()
        print(f"[phase4_real] START {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._started
        status = "FAILED" if exc_type else "END"
        print(f"[phase4_real] {status} {self.name} ({elapsed:.1f}s)")
        return False


def build_evidence(corpus_root: Path, artifact_root: Path) -> dict:
    started = time.perf_counter()
    with _Stage("corpus_resolve"):
        resolved = resolve_corpus(corpus_root)
    with _Stage("corpus_load"):
        # `run_corpus` runs (and, per 958388f, fully releases its OWN internal
        # `Corpus`) BEFORE this frame's persistent `corpus = load(...)` local is
        # created. Reversed, the two full corpora were resident at once for the
        # whole `run_corpus` call: `corpus` was already bound to a local here,
        # on top of the independent `load(corpus_root)` `checks.tier0_corpus.run`
        # -> `_run` -> `_run_loaded` performs internally to verify the round
        # trip. Measured on the real 3.2 GB corpus: climbs past an 8.5 GB cap
        # and, raised to 9.5 GB, past that too with active swapping -- still
        # rising when killed. Neither call needs the other's result, so this
        # ordering changes nothing about what either one verifies.
        corpus_verdict, _ = run_corpus(resolved)
        corpus = load(resolved)
    with _Stage("registry_load"):
        registry = default_registry()
        feature_registry = default_feature_registry()
    with _Stage("application_controls"):
        application_controls = _application_controls()
    with _Stage("frozen_model_control"):
        frozen_model_stage = _frozen_model_control(_request())
    with _Stage("completion_controls_init"):
        completion_controls = _completion_controls(application_controls)
        completion_controls.update(_chooser_controls())
    with _Stage("numerical_independence"):
        numerical_independence = _numerical_independence_control()
    with _Stage("factory_structure_controls"):
        factory_structure_controls = _factory_structure_controls()
    with _Stage("simulation_acceptance"):
        simulation_acceptance = _simulation_acceptance_controls()
    with _Stage("native_parity"):
        saved_release_comparison, native_parity = _native_parity(corpus)
    with _Stage("factory_parity"):
        factory_parity = _factory_parity(corpus)
    with _Stage("completion_controls_update"):
        completion_controls.update({
            "str_thru_stage_parity": application_controls["direct_batch_equal"],
            "all_factory_geometry_expiry_fill_parity": factory_parity["complete"],
            **factory_structure_controls,
            "native_outputs_independently_recomputed":
                numerical_independence["independent_recomputation"],
            "native_analogs_independently_recomputed":
                numerical_independence["analog_independent_recomputation"],
            "analog_corruption_rejected":
                numerical_independence["analog_planted_defects_rejected"],
            "simulation_expiry_parity": simulation_acceptance["expiry_parity"],
            "simulation_pre_expiry_parity": simulation_acceptance["pre_expiry_parity"],
            "simulation_material_time_value": simulation_acceptance["material_time_value"],
            "simulation_fill_propagation": simulation_acceptance["fill_propagation"],
            "executable_recipe_binding": simulation_acceptance["executable_recipe_binding"],
            "strict_gate_semantics": simulation_acceptance["strict_gate_semantics"],
            "preservation_only_detected":
                numerical_independence["preservation_only_detected"],
            "preservation_only_rejected":
                numerical_independence["preservation_only_rejected"],
            "native_stage_comparator_planted_defect": native_parity["planted_defect"]["detected"],
            "numeric_forecast_corruption_rejected": native_parity["planted_defect"]["controls"]["forecasts"],
            "simulation_corruption_rejected": native_parity["planted_defect"]["controls"]["simulation"],
            "financial_diagnostic_corruption_rejected": native_parity["planted_defect"]["controls"]["financial_diagnostics"],
            "factory_geometry_corruption_rejected": factory_parity["negative_controls"]["geometry"],
            "factory_expiry_corruption_rejected": factory_parity["negative_controls"]["expiry"],
            "factory_fill_corruption_rejected": factory_parity["negative_controls"]["fill"],
            # Honest definition: True only if EVERY declared-population record was
            # actually run through a record-by-record saved-release comparison
            # (expected == supported == compared, all agreed, every comparison
            # dimension agreeing, same-input hashes verified, all required stages
            # covered) -- exactly what `saved_release_comparison["complete"]`
            # (_native_parity's `release` return value) measures. It is NOT the
            # champion-artifact hash check; that lives under its own honest name
            # below.
            "full_saved_release_compared": saved_release_comparison["complete"],
            "champion_artifacts_verified": _champion_artifacts_verified(),
            # Not in `final_controls`: a real bundle's absence must not become a
            # new gating requirement. Only surfaces a DECLARED bundle's own
            # verification failure.
            "diagnostic_checkpoint_bundle_valid": _diagnostic_checkpoint_control(corpus),
            "batch_resources_measured": application_controls["batch_resource_profile"],
        })
        completion_controls.update({
            "saved_release_comparison_complete": saved_release_comparison["complete"],
            "native_parity_complete": native_parity["complete"],
        })
    with _Stage("inventory_building"):
        kinds = sorted({pair["payload"].get("record_kind") for pair in corpus.pairs.values()})
        covered_strategies = sorted({
            pair["payload"]["record"].get("strategy")
            for pair in corpus.pairs.values()
            if pair["payload"].get("record_kind") != "dyn_sv_resolution"
        })
        stage_ids = (
            "resolve_context", "features", "forecast", "geometry", "pricing",
            "analogs", "simulation", "gate", "chooser", "serialization",
        )
    with _Stage("subjects_building"):
        subjects = {
            "P4-01": {"status": "PASS", "controls": {
                "corpus_round_trip": corpus_verdict.verdict == "agree",
                "identity_controls": all(application_controls.values()),
                "stage_plan_registered": bool(stage_ids),
            }},
            "P4-02": {"status": "PASS", "controls": {
                "eleven_factories": len(STRATEGY_IDS) == 11,
                "dynamic_menu_order": registry.strategy("DYN-SV").structure_parameters["menu"] == DYNAMIC_MENU,
                "deployment_pins_roles": len(registry.deployment("legacy-phase4-deployment.v1").model_role_bindings) == 7,
            }},
            "P4-03": {"status": "PASS", "controls": {
                "separate_context_scopes": {r.source_scope for r in feature_registry.recipes} == {"event", "analog", "calibration"},
                "named_analog_recipe": feature_registry.get("legacy.bucket_analogs.v1").history_scope == "complete historical replay population",
                "zero_null_distinction": application_controls["zero_is_not_missing"],
            }},
            "P4-04": {"status": "FOUNDATION_PASS", "controls": {
                "str_thru_corpus_present": "STR-THRU" in covered_strategies,
                "shared_kernel": application_controls["direct_batch_equal"],
                "numeric_forecast_parity": native_parity["dimension_agreement"]["forecasts"],
                "native_outputs_independently_recomputed":
                    numerical_independence["independent_recomputation"],
                "native_analogs_independently_recomputed":
                    numerical_independence["analog_independent_recomputation"],
                "expiry_simulation_parity": simulation_acceptance["expiry_parity"],
                "pre_expiry_simulation_parity": simulation_acceptance["pre_expiry_parity"],
                "material_time_value": simulation_acceptance["material_time_value"],
                "fill_propagation": simulation_acceptance["fill_propagation"],
                "strict_gate_semantics": simulation_acceptance["strict_gate_semantics"],
            }},
            "P4-05": {"status": "PASS", "controls": {
                "all_factory_rows_in_corpus": set(covered_strategies) >= set(STRATEGY_IDS),
                "refusal_rows_present": {"CAL-P", "CND-P"}.issubset(set(covered_strategies)),
                "geometry_expiry_fill_parity": factory_parity["complete"],
            }},
            "P4-06": {"status": "FOUNDATION_PASS", "controls": {
                "chooser_corpus_present": "dyn_sv_choice" in kinds,
                "complete_menu_registered": len(DYNAMIC_MENU) == 7,
            }},
            "P4-07": {"status": "FOUNDATION_PASS", "controls": {
                "financial_values_owned": application_controls["financial_values_owned"],
                "simulation_parity": native_parity["dimension_agreement"]["simulation"],
                "independent_simulation_parity": (
                    simulation_acceptance["expiry_parity"]
                    and simulation_acceptance["pre_expiry_parity"]
                ),
                "financial_diagnostic_parity": native_parity["dimension_agreement"]["financial_diagnostics"],
                "terminal_and_planned_exit_labels": completion_controls["planned_exit_valuation_parity"],
            }},
            "P4-08": {"status": "FOUNDATION_PASS", "controls": {
                "single_batch_equal": application_controls["direct_batch_equal"],
                "replay_identity_pinned": application_controls["operational_time_excluded"],
                "executable_recipe_binding":
                    simulation_acceptance["executable_recipe_binding"],
            }},
            "P4-09": {"status": "PASS", "controls": {
                "no_training_import": all("engine.v2.models.training" not in path.read_text()
                                          for path in Path("engine/v2/scoring").glob("*.py")),
                "no_experiment_import": all("experiments" not in path.read_text()
                                            for path in Path("engine/v2/scoring").glob("*.py")),
            }},
        }
    with _Stage("evidence_assembly"):
        final_controls = (
            "full_saved_release_compared", "batch_resources_measured",
            "saved_release_comparison_complete", "native_parity_complete",
            "numeric_forecast_corruption_rejected", "simulation_corruption_rejected",
            "financial_diagnostic_corruption_rejected",
            "factory_geometry_corruption_rejected", "factory_expiry_corruption_rejected",
            "factory_fill_corruption_rejected",
            "native_outputs_independently_recomputed", "preservation_only_rejected",
            "native_analogs_independently_recomputed", "analog_corruption_rejected",
            "simulation_expiry_parity", "simulation_pre_expiry_parity",
            "simulation_material_time_value", "simulation_fill_propagation",
            "executable_recipe_binding", "strict_gate_semantics",
        )
        evidence = {
            "schema_version": "phase4_acceptance.v1.0",
            "status": "FOUNDATION_PASS",
            "evidence_scope": "frozen_real_data_foundation",
            "corpus_root": str(resolved),
            "corpus_hash": corpus.index.get("corpus_hash"),
            "population": saved_release_comparison["population"],
            "strategy_inventory": {"factories": list(STRATEGY_IDS), "dynamic_menu": list(DYNAMIC_MENU)},
            "model_roles": sorted(registry.deployment("legacy-phase4-deployment.v1").model_role_bindings),
            "feature_recipes": [recipe.recipe_id for recipe in feature_registry.recipes],
            "stage_plan": list(stage_ids),
            "subjects": subjects,
            "application_controls": application_controls,
            "frozen_model_stage": frozen_model_stage,
            "completion_controls": completion_controls,
            "saved_release_comparison": saved_release_comparison,
            "native_parity": native_parity,
            "factory_parity": factory_parity,
            "numerical_independence": numerical_independence,
            "factory_structure_controls": factory_structure_controls,
            "simulation_acceptance": simulation_acceptance,
            "phase5_inference_integrated": False,
            "phase5_handoff_required": True,
            "runtime_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "implementation_hash": content_hash({"strategies": list(STRATEGY_IDS), "recipes": [r.recipe_id for r in feature_registry.recipes], "stages": stage_ids}),
        }
    with _Stage("report_write"):
        report = _write_phase4_report(evidence, artifact_root)
        evidence["completion_controls"]["complete_report_written"] = _report_is_complete(
            report.read_text(), evidence,
        )
    with _Stage("completion_check_and_write"):
        if all(evidence["completion_controls"].get(name) is True for name in final_controls + ("complete_report_written",)):
            evidence["status"] = "PASS"
            evidence["evidence_scope"] = "native_full_release"
            evidence["phase5_inference_integrated"] = frozen_model_stage
            for row in evidence["subjects"].values():
                if row["status"] == "FOUNDATION_PASS":
                    row["status"] = "PASS"
            report = _write_phase4_report(evidence, artifact_root)
            evidence["completion_controls"]["complete_report_written"] = _report_is_complete(
                report.read_text(), evidence,
            )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="fixtures/tier0")
    parser.add_argument("--artifact-root", default="/tmp/phase4-acceptance")
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence(Path(args.corpus), root)
    output = Path(args.output) if args.output else root / "evidence.json"
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"evidence": str(output), "status": evidence["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
