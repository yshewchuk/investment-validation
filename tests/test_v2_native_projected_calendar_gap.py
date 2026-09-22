"""Regression test for the PROJECTED_CALENDAR gap: ``_check_projected_calendar``
(engine/v2/scoring/stages.py:335-349) is correct, but read
``values.get("calendar_observed_through")``, a key nothing in the repo ever
populated. Fixed by carrying ``engine.calendar.TradingCalendar
.observed_through`` -- a real, source-owned calendar fact (the last date
backed by real observed price history, constant for every row scored
against one calendar instance) -- from ``Scorer._score``'s
``capture_source_bundle`` call (engine/score.py, right where the legacy
PROJECTED_CALENDAR flag itself is raised) into native's context. This is
never legacy's own PROJECTED_CALENDAR verdict (a per-row calculated
answer, which would be circular per checks/phase4_frozen_bridge.py's
answer-field classification) -- only the calendar's own horizon.
"""
from __future__ import annotations

import pandas as pd

from engine.score import Phase4TraceCollector
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    STAGE_NAMES,
    StageReceipt,
    _check_projected_calendar,
    assemble_native_values,
)


def test_check_projected_calendar_is_still_dead_without_the_key():
    """Documents the pre-fix defect directly: the check itself was always
    correct, but with no ``calendar_observed_through`` in ``values`` it can
    never fire."""
    flags: list = []
    _check_projected_calendar({"exit_date": "2026-09-20"}, flags)
    assert flags == []


def test_legacy_trace_collector_captures_calendar_observed_through_as_source_fact():
    """The exact mechanism ``engine/score.py``'s ``Scorer._score`` now uses:
    ``capture_source_bundle(context={..., "calendar_observed_through": ...})``
    with a real ``TradingCalendar.observed_through`` (a ``pd.Timestamp``).
    Proves it round-trips through the collector's JSON-safe documenting the
    same way ``entry_date``/``exit_date`` already do, with no special-casing
    needed, and is not rejected as a scoring answer.
    """
    trace = Phase4TraceCollector(retain_full_trace=False)
    observed_through = pd.Timestamp("2026-09-18")
    trace.capture_source_bundle(
        context={
            "ticker": "AAA",
            "strategy": "STR-THRU",
            "entry_date": pd.Timestamp("2026-09-16"),
            "exit_date": pd.Timestamp("2026-09-20"),
            "calendar_observed_through": observed_through,
        },
        quote_status="not_reached",
    )
    frozen = trace._checkpoint_groups["source_inputs"]
    assert frozen["context"]["calendar_observed_through"] == "2026-09-18"


def test_projected_calendar_fires_end_to_end_from_captured_context():
    """Full plumbing, capture -> native: a captured
    ``calendar_observed_through`` strictly before the resolved exit date
    makes ``assemble_native_values`` emit PROJECTED_CALENDAR, using the same
    JSON-safe string form the real capture path produces (not a raw
    ``pd.Timestamp``)."""
    trace = Phase4TraceCollector(retain_full_trace=False)
    trace.capture_source_bundle(
        context={
            "ticker": "AAA",
            "strategy": "STR-THRU",
            "entry_date": pd.Timestamp("2026-09-16"),
            "exit_date": pd.Timestamp("2026-09-20"),
            "calendar_observed_through": pd.Timestamp("2026-09-18"),
        },
        quote_status="not_reached",
    )
    captured_context = dict(trace._checkpoint_groups["source_inputs"]["context"])

    quotes = {
        ("C", 100.0, "2026-09-20"): {"bid": 1.9, "ask": 2.1},
        ("P", 100.0, "2026-09-20"): {"bid": 1.9, "ask": 2.1},
    }
    context = {
        **captured_context,
        "event_date": "2026-09-16", "expiry": "2026-09-20",
        "strike": 100.0, "spot": 100.0, "quotes": quotes,
    }
    forecast = {
        "driver_name": "abs_move",
        "models": {"driver_prediction": {"intercept": 0.0, "coefficients": {}}},
    }
    receipts = tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics"
    )
    inputs = NativeScoreInputs(
        context=context,
        features={"model_inputs": {}},
        forecast=forecast,
        geometry=None,
        pricing=None,
        analogs={"recipe": None},
        simulation={"mode": "not_applicable"},
        gate={"mode": "not_applicable"},
        chooser={},
        diagnostics={},
        source_ref="gap3-fixture",
        stage_receipts=receipts,
    )
    values = assemble_native_values(inputs)

    assert "PROJECTED_CALENDAR" in values["flags"]


def test_projected_calendar_does_not_fire_when_exit_on_or_before_observed_through():
    trace = Phase4TraceCollector(retain_full_trace=False)
    trace.capture_source_bundle(
        context={
            "ticker": "AAA",
            "strategy": "STR-THRU",
            "entry_date": pd.Timestamp("2026-09-16"),
            "exit_date": pd.Timestamp("2026-09-18"),
            "calendar_observed_through": pd.Timestamp("2026-09-18"),
        },
        quote_status="not_reached",
    )
    captured_context = dict(trace._checkpoint_groups["source_inputs"]["context"])

    flags: list = []
    _check_projected_calendar(captured_context, flags)

    assert "PROJECTED_CALENDAR" not in flags
