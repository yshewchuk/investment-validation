"""R4-9: native derivation of the five legacy-only actionable flags.

Legacy reference sites (engine/score.py): STALE_QUOTE:1687, PROJECTED_CALENDAR:1337,
OUT_OF_DOMAIN:3022, WIDE_MARKET:1789, EXTRAPOLATED:1807. Each has a both-ways
test: it fires when the legacy trigger condition holds and does not fire when
it does not, from synthetic source-owned inputs (no corpus needed).
"""
from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    STAGE_NAMES,
    StageReceipt,
    assemble_native_values,
)

_RECEIPTS = tuple(
    StageReceipt(stage, "declared-input", "declared-output")
    for stage in STAGE_NAMES if stage != "diagnostics"
)
_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
    ("P", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
}


def _request() -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-1", calendar_revision="cal-1", strategy_version="STR-THRU",
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5}, dependency_refs=("analog-pop",),
        model_artifact_refs=("model-a",), residual_state_ref="res-a",
    )


def _inputs(*, context_overrides=None, model_inputs=None, gate=None,
           quotes=None, diagnostics=None, simulation=None) -> NativeScoreInputs:
    context = {
        "ticker": "AAA",
        "strategy": "STR-THRU",
        "event_date": "2026-09-16",
        "entry_date": "2026-09-16",
        "exit_date": "2026-09-18",
        "expiry": "2026-09-18",
        "strike": 100.0,
        "spot": 100.0,
        "quotes": dict(quotes if quotes is not None else _QUOTES),
    }
    context.update(context_overrides or {})
    forecast = {
        "driver_name": "abs_move",
        "models": {"driver_prediction": {"intercept": 0.0, "coefficients": {}}},
    }
    return NativeScoreInputs(
        context=context,
        features={"model_inputs": dict(model_inputs or {})},
        forecast=forecast,
        geometry=None,
        pricing=None,
        analogs={"recipe": None},
        simulation=simulation if simulation is not None else {},
        gate=gate if gate is not None else {"mode": "not_applicable"},
        chooser={},
        diagnostics=diagnostics if diagnostics is not None else {},
        source_ref="actionable-flags-fixture",
        stage_receipts=_RECEIPTS,
    )


def test_projected_calendar_fires_past_observed_through():
    values = assemble_native_values(_inputs(context_overrides={
        "exit_date": "2026-09-20", "calendar_observed_through": "2026-09-18",
    }))
    assert "PROJECTED_CALENDAR" in values["flags"]


def test_projected_calendar_does_not_fire_on_or_before_observed_through():
    values = assemble_native_values(_inputs(context_overrides={
        "exit_date": "2026-09-18", "calendar_observed_through": "2026-09-18",
    }))
    assert "PROJECTED_CALENDAR" not in values["flags"]


def test_stale_quote_fires_when_quote_predates_entry():
    values = assemble_native_values(_inputs(context_overrides={
        "entry_date": "2026-09-16", "quote_date": "2026-09-12",
    }))
    assert "STALE_QUOTE" in values["flags"]


def test_stale_quote_does_not_fire_when_quote_matches_entry():
    values = assemble_native_values(_inputs(context_overrides={
        "entry_date": "2026-09-16", "quote_date": "2026-09-16",
    }))
    assert "STALE_QUOTE" not in values["flags"]


def test_out_of_domain_fires_below_mcap_floor():
    import math
    gate = {"model": {"intercept": 0.5, "coefficients": {}}, "threshold": 0.0}
    values = assemble_native_values(_inputs(
        model_inputs={"mcap_log": math.log(1e8)}, gate=gate,
    ))
    assert "OUT_OF_DOMAIN" in values["flags"]
    assert values.get("gate_score") is None


def test_out_of_domain_does_not_fire_above_mcap_floor():
    import math
    gate = {"model": {"intercept": 0.5, "coefficients": {}}, "threshold": 0.0}
    values = assemble_native_values(_inputs(
        model_inputs={"mcap_log": math.log(2e9)}, gate=gate,
    ))
    assert "OUT_OF_DOMAIN" not in values["flags"]
    assert values.get("gate_score") == 0.5


def test_wide_market_fires_on_a_wide_spread():
    wide_quotes = {
        ("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 5.0},
        ("P", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
    }
    values = assemble_native_values(_inputs(quotes=wide_quotes))
    assert "WIDE_MARKET" in values["flags"]


def test_wide_market_does_not_fire_on_a_tight_spread():
    values = assemble_native_values(_inputs())
    assert "WIDE_MARKET" not in values["flags"]


def test_extrapolated_fires_far_from_atm():
    far_quotes = {
        ("C", 110.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
        ("P", 110.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
    }
    values = assemble_native_values(_inputs(
        context_overrides={"strike": 110.0}, quotes=far_quotes,
    ))
    assert "EXTRAPOLATED" in values["flags"]


def test_extrapolated_does_not_fire_at_the_money():
    values = assemble_native_values(_inputs())
    assert "EXTRAPOLATED" not in values["flags"]


# -- Advisory-vs-refusal taxonomy: both directions, at the full-row level --
# (application.score_one, not just assemble_native_values) -- proving the
# row's validation_status/readiness, not merely the flags list, respects
# the taxonomy in engine/v2/scoring/stages.py's ADVISORY_FLAGS.

def test_row_carrying_only_an_advisory_flag_still_scores():
    inputs = _inputs(
        diagnostics={"flags": ("CHOOSER_MISSING_FEATURES",)},
        simulation={"mode": "not_applicable"},
    )
    record = application.score_one(_request(), inputs)

    assert record.reason_codes == ("CHOOSER_MISSING_FEATURES",)
    assert record.validation_status == "scored"
    assert record.readiness == "ready"


def test_row_carrying_a_refusal_code_is_refused():
    inputs = _inputs(diagnostics={"flags": ("NO_CHAIN",)})
    record = application.score_one(_request(), inputs)

    assert "NO_CHAIN" in record.reason_codes
    assert record.validation_status == "refused"
    assert record.readiness == "refused"
