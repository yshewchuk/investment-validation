"""Phase 4 ordered-flag parity: a pricing-boundary refusal precedes the late
stage (payoff/model/gate) refusals it accompanies.

The strict-replay comparator's ``checks.flags`` compares ORDERED flag tuples,
while its ``flag_differences`` reports the set-symmetric difference. Fixtures
009 (RAMP7/LUXE) and 015 (RAMP7/ISPR) therefore failed ``checks.flags`` with an
EMPTY ``flag_differences``: native carried the right flag NAMES but in the wrong
ORDER. Legacy stamps a pricing-boundary refusal (COARSE_LADDER / NO_CHAIN) in
``Scorer._price_entry`` (engine/score.py:2348-2402) and only AFTERWARDS reaches
the model / analog / gate / chooser layers, so the refusal sits at the FRONT of
the flag tuple. Native's ``assemble_native_values`` used to append
``geometry.refusal`` / ``pricing.refusal`` only after ``_append_late_stages``,
burying the refusal behind NO_PAYOFF_MAP / the gate refusal.

These tests pin the corrected semantic order directly: the refusal precedes the
late-stage refusals, and an ordinary priced row (no refusal to hoist) is
unchanged.
"""
from __future__ import annotations

from engine.v2.scoring.stages import (
    STAGE_NAMES,
    NativeScoreInputs,
    StageReceipt,
    assemble_native_values,
)

_RECEIPTS = tuple(
    StageReceipt(stage, "declared-input", "declared-output")
    for stage in STAGE_NAMES if stage != "diagnostics"
)
_EXPIRY = "2026-09-18"
# A grid fine enough (2-unit spacing at spot 100) that CND-PS sizes and prices
# cleanly off a pinned width, versus the coarse 5-unit / missing-strike ladders
# that refuse -- the only difference the ordered flags should track.
_PRICED_GRID = (92.0, 94.0, 96.0, 98.0, 100.0, 102.0, 104.0, 106.0, 108.0)


def _row(strategy, strikes, width):
    return NativeScoreInputs(
        context={
            "ticker": "AAA", "strategy": strategy,
            "event_date": "2026-09-16", "entry_date": "2026-09-16",
            "exit_date": _EXPIRY, "expiry": _EXPIRY,
            "strike": strikes[0], "spot": 100.0, "width": width,
            "quotes": {("P", s, _EXPIRY): {"bid": 1.9, "ask": 2.1} for s in strikes},
        },
        features={"model_inputs": {}},
        # Serve the ``size`` forecast role every short-vol shape requires, so the
        # only flags on the row are the pricing-boundary refusal and the
        # late-stage refusals -- nothing else competes for the front of the tuple.
        forecast={"forecast_abs_move": {"intercept": 2.0, "coefficients": {}}},
        geometry=None, pricing=None,
        analogs={"recipe": None}, simulation={"mode": "not_applicable"},
        # A gate champion naming a feature the row never carries: the gate stage
        # refuses with MISSING_GATE_INPUT:<feature>, a late-stage flag that must
        # land AFTER the pricing-boundary refusal, never before it.
        gate={"model": {"intercept": 0.5,
                        "coefficients": {"nonexistent_feature": 1.0}},
              "threshold": 0.0},
        chooser={}, diagnostics={},
        source_ref="flag-order-pricing-refusal", stage_receipts=_RECEIPTS,
    )


def test_coarse_ladder_refusal_precedes_late_stage_refusals():
    flags = assemble_native_values(
        _row("CND-PS", (95.0, 100.0, 105.0), 1.0), strategy="CND-PS",
    )["flags"]
    assert list(flags) == ["COARSE_LADDER", "NO_PAYOFF_MAP",
                           "MISSING_GATE_INPUT:nonexistent_feature"]


def test_no_chain_refusal_precedes_late_stage_refusals():
    flags = assemble_native_values(
        _row("BFLY-P", (100.0, 105.0), 5.0), strategy="BFLY-P",
    )["flags"]
    assert list(flags) == ["NO_CHAIN", "NO_PAYOFF_MAP",
                           "MISSING_GATE_INPUT:nonexistent_feature"]


def test_ordinary_priced_row_keeps_only_the_late_stage_refusals():
    # Same strategy, shape and gate, but a legal (non-coarse) ladder that prices:
    # with no pricing-boundary refusal there is nothing to hoist, so the ordered
    # flags are exactly the late-stage refusals -- unchanged by the fix.
    flags = assemble_native_values(
        _row("CND-PS", _PRICED_GRID, 2.0), strategy="CND-PS",
    )["flags"]
    assert list(flags) == ["NO_PAYOFF_MAP",
                           "MISSING_GATE_INPUT:nonexistent_feature"]
