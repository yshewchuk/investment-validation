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

Phase 4 replay 002 (STR-RUNUP member 2) exposed a sibling ordering gap on the
FORECAST side: legacy stamps the champion driver/runup model's MISSING_FEATURES
when the MODEL layer runs -- AFTER ``_price_entry`` and the calendar flag -- so
its flag order is PROJECTED_CALENDAR, NO_CHAIN, MISSING_FEATURES. Native's
``_initial_values`` executes those champion forecast executors BEFORE the
context and pricing checks, so stamping their refusal inline produced the
wrong order (MISSING_FEATURES first). ``assemble_native_values`` now defers a
reporting-role (driver/implied_t1/runup_move) MISSING_FEATURES to the model
boundary, while genuine pre-pricing refusals -- a size fold, a forecast-sizing
NO_FORECAST, a BAD_QUOTE early return -- keep their legacy position and never
publish an unreachable champion refusal.
"""
from __future__ import annotations

from engine.v2.scoring.stages import (
    STAGE_NAMES,
    TIER4_FOLD_ADAPTER,
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


# -- Champion forecast-refusal ordering (Phase 4 replay 002, member 2) --------
# A reporting-role (driver/implied_t1/runup_move) MISSING_FEATURES is served at
# legacy's MODEL layer -- AFTER ``_price_entry`` and the calendar flag -- so its
# ordered position is PROJECTED_CALENDAR, [pricing refusal], MISSING_FEATURES.
# ``assemble_native_values`` defers exactly that refusal to the model boundary
# (``_append_late_stages``), while a genuine pre-pricing failure (a size fold, a
# forecast-sizing NO_FORECAST, a BAD_QUOTE early return) keeps its legacy spot
# and never publishes a champion refusal legacy's model layer never reached.


class _MissingFeatures(ValueError):
    """A champion ``predict`` refusal whose reason codes carry MISSING_FEATURES
    -- mirrors ``engine/v2/scoring/frozen_executor.py``'s ``FrozenStageRefusal``:
    a ``ValueError`` (so ``_execute_forecast_executor`` catches it) with a
    ``reason_codes`` tuple (so ``_missing_feature_refusal`` recognizes it)."""

    def __init__(self):
        super().__init__("MISSING_FEATURES")
        self.reason_codes = ("MISSING_FEATURES",)


class _ChampionExecutor:
    """A reporting-role forecast executor: refuses (a missing/non-finite feature)
    or serves a finite value, identified by ``role`` and a non-Tier-4 ``adapter``
    so ``_account_forecast_refusal`` treats it as a champion, not a silent fold."""

    adapter = "joblib-estimator.v1"

    def __init__(self, role, value=None):
        self.role = role
        self.feature_order = ("im", "iv30")
        self._value = value

    def predict(self, facts):
        if self._value is None:
            raise _MissingFeatures()
        return float(self._value)


class _Tier4SizeFold:
    """A Tier-4 serving fold bound to the ``size`` role: a missing feature is a
    silent NaN (``TIER4_FOLD_ADAPTER``), which the sizing stage declines as
    NO_FORECAST before ``_price_entry`` -- not a champion MISSING_FEATURES."""

    adapter = TIER4_FOLD_ADAPTER

    def __init__(self, role="size"):
        self.role = role
        self.feature_order = ("im", "iv30")

    def predict(self, facts):
        raise _MissingFeatures()


def _runup_inputs(executors, *, projected, priced_quotes=None):
    context = {
        "ticker": "AAA", "strategy": "STR-RUNUP",
        "event_date": "2026-09-16", "entry_date": "2026-09-16",
        "exit_date": _EXPIRY, "strike": 100.0, "spot": 100.0,
        "days_before_print": 7.0,
    }
    if projected:
        context["calendar_observed_through"] = "2026-09-16"
    if priced_quotes is None:
        # No captured expiry + an observed-empty quote domain: legacy ran the
        # chain lookup, found nothing, and stamped NO_CHAIN at the pricing
        # boundary (engine/score.py:2347-2351).
        context["quotes"] = {}
    else:
        context["expiry"] = _EXPIRY
        context["quotes"] = priced_quotes
    return NativeScoreInputs(
        context=context, features={"model_inputs": {}},
        forecast={"executors": dict(executors)},
        geometry=None, pricing=None,
        analogs={"recipe": None}, simulation={"mode": "not_applicable"},
        gate={"mode": "not_applicable"}, chooser={}, diagnostics={},
        source_ref="flag-order-champion", stage_receipts=_RECEIPTS,
    )


def _refusing_runup_champion():
    return {
        "driver_prediction": _ChampionExecutor("implied_t1"),
        "runup_move_prediction": _ChampionExecutor("runup_move"),
    }


def _finite_runup_champion():
    return {
        "driver_prediction": _ChampionExecutor("implied_t1", 6.0),
        "runup_move_prediction": _ChampionExecutor("runup_move", 3.0),
    }


# Small quotes price a clean (non-BAD_QUOTE) straddle; the wide 30/34 spread
# costs 64 on a 100 spot (>BAD_QUOTE_COST_PCT=30%) to force legacy's exit.
_NARROW_QUOTES = {
    ("C", 100.0, _EXPIRY): {"bid": 1.9, "ask": 2.1},
    ("P", 100.0, _EXPIRY): {"bid": 1.9, "ask": 2.1},
}
_EXPENSIVE_QUOTES = {
    ("C", 100.0, _EXPIRY): {"bid": 30.0, "ask": 34.0},
    ("P", 100.0, _EXPIRY): {"bid": 30.0, "ask": 34.0},
}


def _receipt_stages(values):
    return {row["stage"] for row in values["native_stage_receipts"]}


def _sizing_decline_inputs(executors):
    return NativeScoreInputs(
        context={
            "ticker": "AAA", "strategy": "RAMP7", "spot": 100.0,
            "event_date": "2026-09-16", "entry_date": "2026-09-16",
            "exit_date": _EXPIRY, "calendar_observed_through": "2026-09-16",
        },
        features={"model_inputs": {}},
        forecast={"executors": dict(executors)},
        geometry=None, pricing=None,
        analogs={"recipe": None}, simulation={"mode": "not_applicable"},
        gate={"mode": "not_applicable"}, chooser={}, diagnostics={},
        source_ref="flag-order-sizing-decline", stage_receipts=_RECEIPTS,
    )


def test_projected_empty_chain_champion_refusal_exact_legacy_order():
    # 002 member 2's signature: the calendar flag, then the pricing-boundary
    # NO_CHAIN, then -- only at the model boundary -- the champion MISSING_FEATURES.
    flags = assemble_native_values(
        _runup_inputs(_refusing_runup_champion(), projected=True),
        strategy="STR-RUNUP",
    )["flags"]
    assert list(flags) == ["PROJECTED_CALENDAR", "NO_CHAIN", "MISSING_FEATURES"]


def test_projected_priced_champion_refusal_calendar_precedes_missing():
    # A priced row has no pricing-boundary refusal to hoist, so the ordered flags
    # are just the calendar flag then the (deferred) champion refusal.
    flags = assemble_native_values(
        _runup_inputs(_refusing_runup_champion(), projected=True,
                      priced_quotes=_NARROW_QUOTES),
        strategy="STR-RUNUP",
    )["flags"]
    assert list(flags) == ["PROJECTED_CALENDAR", "MISSING_FEATURES"]


def test_nonprojected_empty_chain_champion_refusal_no_chain_precedes_missing():
    # No projected calendar on the row: the pricing refusal (NO_CHAIN) still
    # precedes the champion MISSING_FEATURES deferred to the model boundary.
    flags = assemble_native_values(
        _runup_inputs(_refusing_runup_champion(), projected=False),
        strategy="STR-RUNUP",
    )["flags"]
    assert list(flags) == ["NO_CHAIN", "MISSING_FEATURES"]


def test_finite_runup_champion_is_unchanged_no_missing_features():
    # Positive control: a finite champion publishes no MISSING_FEATURES at all,
    # so the deferral machinery leaves a projected empty-chain row untouched.
    flags = assemble_native_values(
        _runup_inputs(_finite_runup_champion(), projected=True),
        strategy="STR-RUNUP",
    )["flags"]
    assert list(flags) == ["PROJECTED_CALENDAR", "NO_CHAIN"]
    assert "MISSING_FEATURES" not in flags


def test_tier4_sizing_decline_keeps_calendar_then_no_forecast_withholds_later():
    # A Tier-4 size fold's missing feature is a silent NaN, so the sizing stage
    # declines NO_FORECAST after the calendar flag and BEFORE _price_entry --
    # geometry/pricing/model are withheld and the champion refusal never appears.
    values = assemble_native_values(
        _sizing_decline_inputs({"forecast_abs_move": _Tier4SizeFold()}),
        strategy="RAMP7",
    )
    flags = list(values["flags"])
    assert flags == ["PROJECTED_CALENDAR", "NO_FORECAST"]
    for withheld in ("NO_CHAIN", "COARSE_LADDER", "MISSING_EXPIRY",
                     "MISSING_FEATURES", "NO_PAYOFF_MAP", "MISSING_GATE_INPUT"):
        assert withheld not in flags
    # The withheld stages still emit receipts (the graph stays complete), but the
    # row is never priced.
    assert _receipt_stages(values) == set(STAGE_NAMES)
    assert values.get("entry_cost") is None


def test_duplicate_missing_feature_refusal_is_deduplicated():
    # Both implied_t1 and runup_move refuse with MISSING_FEATURES; the deferral
    # carries two entries, but _add_flag collapses them to one ordered flag.
    flags = assemble_native_values(
        _runup_inputs(_refusing_runup_champion(), projected=True),
        strategy="STR-RUNUP",
    )["flags"]
    assert flags.count("MISSING_FEATURES") == 1


def test_bad_quote_early_return_does_not_publish_champion_refusal():
    # Legacy's BAD_QUOTE exit leaves before _score_model, so the champion model
    # layer never runs: its deferred MISSING_FEATURES must never be published.
    flags = assemble_native_values(
        _runup_inputs(_refusing_runup_champion(), projected=True,
                      priced_quotes=_EXPENSIVE_QUOTES),
        strategy="STR-RUNUP",
    )["flags"]
    assert "BAD_QUOTE" in flags
    assert "MISSING_FEATURES" not in flags
    assert list(flags).index("PROJECTED_CALENDAR") < list(flags).index("BAD_QUOTE")
