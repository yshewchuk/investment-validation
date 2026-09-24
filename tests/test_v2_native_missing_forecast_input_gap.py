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

It also covers the sibling reachability defect in Phase 4 strict-parity row
``016_RAMP7-HAIN-2026-09-14``: a FORECAST_SIZED strategy whose served forecast
cannot size a legal width must decline at the sizing stage (NO_FORECAST) and
withhold the geometry/pricing/model/gate/chooser stages legacy never reached,
without ever blanket-suppressing a genuinely missing expiry for a row that DOES
size -- see the ``test_forecast_sized_*`` / ``test_decline_marker_*`` controls.
"""
from __future__ import annotations

from types import SimpleNamespace

from engine.v2.scoring.stages import (
    STAGE_NAMES,
    NativeScoreInputs,
    StageReceipt,
    _execute_forecast,
    _forecast_sizing_declines,
    assemble_native_values,
)

# RAMP7 sizes its outer strike at 3x the forecast (``forecast_sizing``'s
# ``SHORT_VOL_OUTER``), so at spot 100 a forecast of 6.0 -> width_moneyness
# 0.02 sizes cleanly, while 60.0 -> 0.20 clears ``WIDTH_MAX`` (0.15) and
# 0.5 -> 0.00167 falls under ``WIDTH_MIN`` (0.005): both are declines.
_RAMP7_VALID_FORECAST = 6.0
_RAMP7_TOO_WIDE_FORECAST = 60.0


def _size_forecast(intercept):
    """A forecast block that serves ``forecast_abs_move`` as a constant."""
    return {"forecast_abs_move": {"intercept": intercept, "coefficients": {}}}


def _size_exec_inputs(strategy, intercept, *, geometry=None):
    return SimpleNamespace(
        forecast=_size_forecast(intercept),
        context={"strategy": strategy, "spot": 100.0},
        features={},
        simulation={},
        geometry=geometry,
    )


def _size_assemble_inputs(strategy, context, intercept):
    receipts = tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics"
    )
    return NativeScoreInputs(
        context={"strategy": strategy, **context},
        features={},
        forecast=_size_forecast(intercept),
        geometry=None,
        pricing=None,
        analogs={},
        simulation={},
        gate={},
        chooser={},
        diagnostics={},
        source_ref="forecast-sizing-gap",
        stage_receipts=receipts,
    )


def _receipt_stages(values):
    return {row["stage"] for row in values["native_stage_receipts"]}


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


# -- Phase 4 strict-parity gap 016 (RAMP7/HAIN): NO_FORECAST pre-geometry ----
# Legacy ``Scorer._size_from_forecast`` declines to size when the served
# forecast cannot turn into a legal width and RETURNS before ``_price_entry``
# (engine/score.py), so geometry, pricing and every later stage never run and
# the row carries only ``PROJECTED_CALENDAR, NO_FORECAST``. Native used to fall
# through into those stages and leak their refusals -- MISSING_EXPIRY (empty
# quotes, no captured expiry), MISSING_FEATURES (the gate's missing inputs) and
# NO_PAYOFF_MAP (RAMP7 has no payoff driver). The fix withholds every stage
# legacy never reached and stamps NO_FORECAST instead; it is keyed off the
# FORECAST_SIZED registry and whether the shape is still open, never a
# ticker/fixture branch, and it never suppresses a genuinely missing expiry for
# a row whose forecast DOES size.


def test_forecast_sized_decline_stamps_no_forecast_and_withholds_later_stages():
    """The HAIN shape: RAMP7 with an in-range spot but a forecast that clears
    ``WIDTH_MAX``, empty quotes and no captured expiry. Native refuses at the
    sizing stage and describes none of the stages legacy never reached."""
    values = assemble_native_values(
        _size_assemble_inputs("RAMP7", {"spot": 0.64}, _RAMP7_TOO_WIDE_FORECAST),
        strategy="RAMP7",
    )
    flags = set(values["flags"])
    assert "NO_FORECAST" in flags
    assert "MISSING_EXPIRY" not in flags
    assert "MISSING_FEATURES" not in flags
    assert "NO_PAYOFF_MAP" not in flags
    # The withheld stages still emit receipts (the execution graph stays whole),
    # but the row is never priced: no legs, entry cost or resolved expiry.
    assert _receipt_stages(values) == set(STAGE_NAMES)
    assert not values.get("legs")
    assert values.get("entry_cost") is None


def test_forecast_sized_decline_is_not_blanket_suppression_of_missing_expiry():
    """Negative control: a forecast-sized row whose forecast DOES size still
    reaches ``_price_entry`` and still refuses a genuinely missing expiry -- the
    withhold is scoped to rows legacy stopped sizing, never to MISSING_EXPIRY."""
    values = assemble_native_values(
        _size_assemble_inputs("CND-PS", {"spot": 100.0}, _RAMP7_VALID_FORECAST),
        strategy="CND-PS",
    )
    flags = set(values["flags"])
    assert "MISSING_EXPIRY" in flags
    assert "NO_FORECAST" not in flags


def test_supplied_width_is_never_spuriously_no_forecast():
    """A caller-pinned width is legacy's ``size=False`` path: the forecast is
    recorded but never declines, even when it could not have sized on its own."""
    values = assemble_native_values(
        _size_assemble_inputs(
            "RAMP7",
            {"spot": 100.0, "expiry": "2026-10-16", "width": 2.0},
            _RAMP7_TOO_WIDE_FORECAST,
        ),
        strategy="RAMP7",
    )
    assert "NO_FORECAST" not in set(values["flags"])


def test_requested_structure_params_is_never_spuriously_no_forecast():
    """The Astra blocker's exact source representation: a RAMP7 replay whose
    context carries ``requested_structure_params`` (engine/score.py:1937,
    scoring/identity.py:105) and NO separate ``width``/geometry carrier. Legacy
    computes ``size=not request.structure_params`` as False and keeps going, so
    an out-of-band forecast must still proceed to pricing/model -- not decline
    and withhold. This is the captured-contract path that the earlier fix keyed
    only off ``width``/``geometry`` and therefore mishandled."""
    values = assemble_native_values(
        _size_assemble_inputs(
            "RAMP7",
            {"spot": 100.0, "expiry": "2026-10-16",
             "requested_structure_params": {"width_moneyness": 0.02}},
            _RAMP7_TOO_WIDE_FORECAST,
        ),
        strategy="RAMP7",
    )
    flags = set(values["flags"])
    assert "NO_FORECAST" not in flags
    # Reached the model stage (RAMP7 has no payoff driver) instead of being
    # withheld at sizing, proving the pinned shape let the row proceed.
    assert "NO_PAYOFF_MAP" in flags


def test_sizing_decline_guard_honours_requested_structure_params():
    """Unit: the guard treats a non-empty ``requested_structure_params`` as a
    pinned shape -- so an otherwise-declining (out-of-band) forecast does NOT
    decline, while the identical context with no such carrier still does."""
    pinned_values = {
        "strategy": "RAMP7", "spot": 100.0,
        "requested_structure_params": {"width_moneyness": 0.02},
    }
    inputs = _size_exec_inputs("RAMP7", _RAMP7_TOO_WIDE_FORECAST)
    output = {"forecast_abs_move": _RAMP7_TOO_WIDE_FORECAST}
    assert _forecast_sizing_declines(
        inputs, pinned_values, "RAMP7", output, set(),
    ) is False
    # Unpinned, same out-of-band forecast: still declines (retains the case).
    assert _forecast_sizing_declines(
        inputs, {"strategy": "RAMP7", "spot": 100.0}, "RAMP7", output, set(),
    ) is True


def test_pinned_shape_suppresses_only_the_size_decline():
    """A pinned shape suppresses the SIZE decline only. A genuinely undetermined
    (NaN-serving) fold still declines for an OPEN shape, so pinning must not
    blanket-suppress a missing-feature/missing-serving failure: with a pinned
    shape the guard returns False (legacy records the NaN and prices the
    caller's contract), while the same undetermined fold on an open shape
    declines exactly as before."""
    inputs = _size_exec_inputs("RAMP7", _RAMP7_VALID_FORECAST)
    output = {}
    pinned = {
        "strategy": "RAMP7", "spot": 100.0,
        "requested_structure_params": {"width_moneyness": 0.02},
    }
    assert _forecast_sizing_declines(
        inputs, pinned, "RAMP7", output, {"forecast_abs_move"},
    ) is False
    assert _forecast_sizing_declines(
        inputs, {"strategy": "RAMP7", "spot": 100.0}, "RAMP7", output,
        {"forecast_abs_move"},
    ) is True


def test_decline_marker_respects_open_shape():
    """The control signal ``assemble_native_values`` reads is set only for a
    forecast-sized strategy still being sized: not for a valid width, not for a
    pinned width, and not once a geometry has been supplied."""
    declined: dict = {}
    _execute_forecast(
        _size_exec_inputs("RAMP7", _RAMP7_TOO_WIDE_FORECAST), declined, [], "RAMP7",
    )
    assert declined.get("forecast_sizing_declined") is True

    sized: dict = {}
    _execute_forecast(
        _size_exec_inputs("RAMP7", _RAMP7_VALID_FORECAST), sized, [], "RAMP7",
    )
    assert "forecast_sizing_declined" not in sized

    pinned: dict = {"width": 2.0}
    _execute_forecast(
        _size_exec_inputs("RAMP7", _RAMP7_TOO_WIDE_FORECAST), pinned, [], "RAMP7",
    )
    assert "forecast_sizing_declined" not in pinned

    captured: dict = {}
    _execute_forecast(
        _size_exec_inputs("RAMP7", _RAMP7_TOO_WIDE_FORECAST,
                          geometry=object()),
        captured, [], "RAMP7",
    )
    assert "forecast_sizing_declined" not in captured
