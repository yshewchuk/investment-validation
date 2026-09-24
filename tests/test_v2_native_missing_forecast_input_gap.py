"""Regression test for the native-only MISSING_FORECAST_INPUT gap on the
DISABLED strategies CAL-P and CND-P.

The strict-corpus replay compared legacy vs native for the two calendar/condor
rows (``010_CAL-P-...`` / ``011_CND-P-...``) and every dimension matched except
flags: legacy stamped ``[]``, native stamped ``["MISSING_FORECAST_INPUT"]``.

Root cause is the same class of defect already fixed for ``NO_PAYOFF_MAP``
(``tests/test_v2_native_no_payoff_map_gap.py``): legacy's ``Scorer.score``
returns at its disabled-strategy check (engine/score.py:1779-1798), flagging
only ``UNVALIDATED_STRUCTURE`` and running a request-only trace bundle,
BEFORE ``_crush_forecast``/forecast sizing is ever reached. Native's
``_execute_forecast`` (engine/v2/scoring/stages.py) stamped
``MISSING_FORECAST_INPUT`` from "the forecast stage declared nothing" alone,
without the reachability guard the model stage has -- describing a refusal of a
stage legacy never ran.

Fixed by excluding ``engine.v2.domain.generation.DISABLED`` members from that
stamp -- the same registry ``generate`` uses to refuse their geometry and
``_execute_model`` uses to withhold ``NO_PAYOFF_MAP``. It is keyed off the
registry, not a ticker/fixture branch, and is deliberately NOT "geometry
refused for any reason": a forecast-sized strategy whose geometry refuses for an
unrelated capture gap still reaches legacy's forecast and must still refuse. The
genuine required-forecast absence (a non-disabled strategy, or a
``MISSING_FORECAST_OUTPUT`` for a declared role) is untouched -- see the
positive controls below.
"""
from __future__ import annotations

from types import SimpleNamespace

from engine.v2.scoring.stages import (
    STAGE_NAMES,
    NativeScoreInputs,
    StageReceipt,
    _execute_forecast,
    assemble_native_values,
)


def _exec_inputs(strategy):
    return SimpleNamespace(
        forecast={},
        context={"strategy": strategy},
        features={},
        simulation={},
        geometry=None,
    )


def _assemble_inputs(strategy, forecast):
    receipts = tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics"
    )
    return NativeScoreInputs(
        context={"strategy": strategy, "spot": 100.0, "expiry": "2026-10-16"},
        features={},
        forecast=forecast,
        geometry=None,
        pricing=None,
        analogs={},
        simulation={},
        gate={},
        chooser={},
        diagnostics={},
        source_ref="missing-forecast-input-gap",
        stage_receipts=receipts,
    )


def test_missing_forecast_input_does_not_fire_for_a_disabled_strategy():
    """CAL-P/CND-P never reach legacy's forecast stage; native must not
    describe a stage legacy never ran."""
    for strategy in ("CAL-P", "CND-P"):
        flags: list = []
        _execute_forecast(_exec_inputs(strategy), {}, flags, strategy)
        assert "MISSING_FORECAST_INPUT" not in flags


def test_missing_forecast_input_still_fires_for_a_forecast_sized_strategy():
    """Positive control: a strategy that genuinely consumes a forecast and
    declares none must still refuse with MISSING_FORECAST_INPUT."""
    for strategy in ("TWIN-P", "STR-THRU", "STR-RUNUP", "RAMP7"):
        flags: list = []
        _execute_forecast(_exec_inputs(strategy), {}, flags, strategy)
        assert "MISSING_FORECAST_INPUT" in flags


def test_disabled_strategy_declaring_a_required_role_still_reports_the_gap():
    """The fix suppresses only the stage-wide absence stamp, not a genuinely
    required-but-unfilled forecast role (no blanket suppression)."""
    block = {"required_roles": ("size",)}
    inputs = SimpleNamespace(
        forecast=block, context={"strategy": "CAL-P"}, features={},
        simulation={}, geometry=None,
    )
    flags: list = []
    _execute_forecast(inputs, {}, flags, "CAL-P")
    assert "MISSING_FORECAST_INPUT" not in flags
    assert "MISSING_FORECAST_OUTPUT:size" in flags


def test_disabled_row_refuses_only_with_validated_structure_end_to_end():
    """The stage-reachable refusal these rows actually carry: native reaches
    UNVALIDATED_STRUCTURE via geometry and does not also leak
    MISSING_FORECAST_INPUT."""
    for strategy in ("CAL-P", "CND-P"):
        flags = set(assemble_native_values(
            _assemble_inputs(strategy, {}), strategy=strategy,
        )["flags"])
        assert "UNVALIDATED_STRUCTURE" in flags
        assert "MISSING_FORECAST_INPUT" not in flags


def test_supported_row_still_carries_missing_forecast_input_end_to_end():
    """Positive control through the full assembly: a non-disabled strategy with
    an undeclared forecast keeps MISSING_FORECAST_INPUT (parity with the
    existing forecast-role refusals)."""
    flags = set(assemble_native_values(
        _assemble_inputs("STR-THRU", {"driver_name": "abs_move"}),
        strategy="STR-THRU",
    )["flags"])
    assert "MISSING_FORECAST_INPUT" in flags
    assert "MISSING_FORECAST_OUTPUT:driver" in flags
