"""R4-9: native derivation of the five legacy-only actionable flags.

Legacy reference sites (engine/score.py): STALE_QUOTE:1687, PROJECTED_CALENDAR:1337,
OUT_OF_DOMAIN:3022, WIDE_MARKET:1789, EXTRAPOLATED:1807. Each has a both-ways
test: it fires when the legacy trigger condition holds and does not fire when
it does not, from synthetic source-owned inputs (no corpus needed).
"""
import math

from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application
from engine.v2.scoring.native_analog import source_population_hash
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    STAGE_NAMES,
    StageReceipt,
    assemble_native_values,
)


def _request(strategy="STR-THRU") -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-flags",
        calendar_revision="cal-1",
        strategy_version=strategy,
        deployment_id="dep-1",
        decision_clock_id="entry-close",
        requested_decision_at="2026-09-16",
        snapshot_id="snap-1",
        mode="replay",
        fill_model={"alpha": 0.5},
    )

_RECEIPTS = tuple(
    StageReceipt(stage, "declared-input", "declared-output")
    for stage in STAGE_NAMES if stage != "diagnostics"
)
_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
    ("P", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
}


def _analog_block() -> dict:
    # A minimal real analog recipe so a row can carry an actual
    # exp_pnl_analog number (engine/score.py:847-848 legacy parity, applied
    # in application._has_score_number) instead of the default
    # not-applicable block, for tests that need "scored" to mean something.
    rows = [
        {"row_id": "a", "features": {"move": 1.0}, "realized_pnl": 3.0},
        {"row_id": "b", "features": {"move": -1.0}, "realized_pnl": -1.0},
    ]
    return {
        "recipe": {
            "feature_names": ("move",),
            "neighbors": 2,
            "population_hash": source_population_hash(rows),
        },
        "source_rows": rows,
        "query_features": {"move": 1.0},
    }


def _inputs(*, context_overrides=None, model_inputs=None, source_features=None,
           gate=None, quotes=None, diagnostics=None, simulation=None,
           analogs=None) -> NativeScoreInputs:
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
    features = {"model_inputs": dict(model_inputs or {})}
    if source_features is not None:
        features["source_features"] = dict(source_features)
    return NativeScoreInputs(
        context=context,
        features=features,
        forecast=forecast,
        geometry=None,
        pricing=None,
        analogs=analogs if analogs is not None else {"recipe": None},
        simulation=simulation if simulation is not None else {"mode": "not_applicable"},
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


# -- Gate domain guard reads the base frame, not a shadowing model vector ----
# engine/score.py:4113 evaluates the market-cap floor on the ONE base feature
# frame (``_gate_in_domain(request, features)``), never on a model's own
# feature vector. Native's ``_facts`` merges the flattened forecast/chooser
# ``model_inputs`` OVER ``source_features`` (right for the models, wrong here),
# and a role vector keeps a non-finite DAILY_STATE ``mcap_log`` verbatim as NaN
# (capture_tier0_corpus.py:577-587). Both directions below fail if the guard
# reads ``_facts`` instead of the base frame legacy gates on.


def test_out_of_domain_fires_when_a_nan_model_vector_shadows_the_base_frame():
    # The row-008 class of divergence: base frame mcap_log is a real, finite
    # value below the floor; the model_inputs vector carries NaN (a role's
    # non-finite market cap, kept as-is at capture). Legacy still refuses;
    # native must stamp OUT_OF_DOMAIN off the base frame, not read the NaN as
    # "no market cap" and let the small name through.
    import math
    gate = {"model": {"intercept": 0.5, "coefficients": {}}, "threshold": 0.0}
    values = assemble_native_values(_inputs(
        model_inputs={"mcap_log": float("nan")},
        source_features={"mcap_log": math.log(1e8)},
        gate=gate,
    ))
    assert "OUT_OF_DOMAIN" in values["flags"]
    assert values.get("gate_score") is None


def test_out_of_domain_does_not_fire_when_the_base_frame_is_in_domain():
    # Mirror of the case above: the base frame is comfortably above the floor
    # while a shadowing model vector sits below it. An unrelated STR-THRU row
    # must NOT be stamped OUT_OF_DOMAIN off the model vector; the guard reads
    # the base frame and lets the gate run.
    import math
    gate = {"model": {"intercept": 0.5, "coefficients": {}}, "threshold": 0.0}
    values = assemble_native_values(_inputs(
        model_inputs={"mcap_log": math.log(1e8)},
        source_features={"mcap_log": math.log(2e9)},
        gate=gate,
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


def test_geometry_refusal_codes_and_details_reach_score_record_without_clobbering_detail():
    cases = (
        ("CND-PS", (95.0, 100.0, 105.0), 1.0, "COARSE_LADDER"),
        ("BFLY-P", (100.0, 105.0), 5.0, "NO_CHAIN"),
    )
    for strategy, strikes, width, expected_code in cases:
        expiry = "2026-09-18"
        quotes = {
            ("P", strike, expiry): {"bid": 1.0, "ask": 2.0}
            for strike in strikes
        }
        native = _inputs(
            context_overrides={
                "strategy": strategy, "expiry": expiry, "width": width,
            },
            quotes=quotes,
            diagnostics={"detail": "prior diagnostic"},
        )
        record = application.score_one(_request(strategy), native)
        assert expected_code in record.reason_codes
        if expected_code == "COARSE_LADDER":
            assert "NO_CHAIN" not in record.reason_codes
            assert record.warnings[1] == "dn1+dn2+up1+up2"
        else:
            assert record.warnings[1] == "NO_LISTED_STRIKE:dn1"
        assert record.warnings[0] == "prior diagnostic"
        assert len(record.warnings) == 2


# -- R4-9 follow-up: annotation-only flags must not refuse the row ----------
# Legacy ScoreResult.scored (engine/score.py:847) depends only on whether the
# numbers are present, never on flags; WIDE_MARKET/EXTRAPOLATED/STALE_QUOTE/
# PROJECTED_CALENDAR coexist with a scored row there. OUT_OF_DOMAIN is a
# real refusal in both legacy (declines to gate) and the Phase 4 §7.1
# refusal codes (tools/capture_tier0_corpus.py:102), and must still refuse.

_WIDE_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 5.0},
    ("P", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
}
_GATE_MODEL = {"model": {"intercept": 0.5, "coefficients": {}}, "threshold": 0.0}


def test_annotation_only_flag_still_scores():
    # Needs a real number (2026-09-18 fix: a flagless-but-numberless row is
    # NO_SCORE, not "scored") so this isolates what it claims to test: that
    # WIDE_MARKET alone does not force a refusal.
    record = application.score_one(_request(), _inputs(
        quotes=_WIDE_QUOTES, analogs=_analog_block(),
    ))
    assert record.validation_status == "scored"
    assert record.readiness == "ready"
    assert "WIDE_MARKET" in record.reason_codes


def test_annotation_flag_alongside_a_refusal_flag_still_refuses():
    record = application.score_one(_request(), _inputs(
        quotes=_WIDE_QUOTES, gate=_GATE_MODEL,
        model_inputs={"mcap_log": math.log(1e8)},
    ))
    assert record.validation_status == "refused"
    assert record.readiness == "refused"
    assert "WIDE_MARKET" in record.reason_codes
    assert "OUT_OF_DOMAIN" in record.reason_codes


def test_out_of_domain_alone_refuses():
    record = application.score_one(_request(), _inputs(
        gate=_GATE_MODEL, model_inputs={"mcap_log": math.log(1e8)},
    ))
    assert record.validation_status == "refused"
    assert record.readiness == "refused"
    assert record.reason_codes == ("OUT_OF_DOMAIN",)


# -- Advisory-vs-refusal taxonomy: both directions, at the full-row level --
# (application.score_one, not just assemble_native_values) -- proving the
# row's validation_status/readiness, not merely the flags list, respects
# the taxonomy in engine/v2/scoring/stages.py's ADVISORY_FLAGS.

def test_row_carrying_only_an_advisory_flag_still_scores():
    # Needs a real number for the same reason as test_annotation_only_flag_
    # still_scores above.
    inputs = _inputs(
        diagnostics={"flags": ("CHOOSER_MISSING_FEATURES",)},
        simulation={"mode": "not_applicable"},
        analogs=_analog_block(),
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


# -- BAD_QUOTE (tier-0 vetting gap 004, 2026-09-19) --------------------------
# Legacy engine/score.py `_price_entry` flags a priced row whose entry cost is
# more than BAD_QUOTE_COST_PCT of spot, and `Scorer.score` then leaves before
# the model, analog, gate and chooser layers. Native follows the same rule.

_BAD_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 15.9, "ask": 16.1},
    ("P", 100.0, "2026-09-18"): {"bid": 15.9, "ask": 16.1},
}
_NEAR_QUOTES = {
    ("C", 100.0, "2026-09-18"): {"bid": 14.4, "ask": 14.6},
    ("P", 100.0, "2026-09-18"): {"bid": 14.4, "ask": 14.6},
}


def test_bad_quote_threshold_is_legacys():
    from engine.fills import BAD_QUOTE_COST_PCT as legacy
    from engine.v2.scoring.stages import BAD_QUOTE_COST_PCT

    assert BAD_QUOTE_COST_PCT == legacy


def test_bad_quote_follows_the_legacy_predicate_exactly():
    """The same arithmetic legacy runs (`entry_cost / spot * 100.0 >
    BAD_QUOTE_COST_PCT`, only once priced, only with a nonzero spot), at and
    around the threshold, including 30/100*100 == 30.000000000000004."""
    from engine.fills import BAD_QUOTE_COST_PCT as threshold
    from engine.v2.domain.generation import Pricing
    from engine.v2.scoring.stages import _check_bad_quote

    for spot, cost, refusal in (
        (100.0, 29.99, None), (100.0, 30.0, None), (100.0, 30.01, None),
        (100.0, 45.0, None), (0.0, 45.0, None), (100.0, 45.0, "NO_CHAIN"),
        (7.3, 2.19, None), (7.3, 2.2, None),
    ):
        flags: list[str] = []
        _check_bad_quote(Pricing("STR-THRU", spot, cost, (), refusal), flags)
        legacy = bool(refusal is None and spot and cost / spot * 100.0 > threshold)
        assert ("BAD_QUOTE" in flags) is legacy, (spot, cost, refusal)


def test_bad_quote_fires_and_withholds_every_layer_legacy_skips():
    values = assemble_native_values(_inputs(
        quotes=_BAD_QUOTES, gate=_GATE_MODEL, analogs=_analog_block(),
        model_inputs={"mcap_log": math.log(2e9)},
    ))

    assert "BAD_QUOTE" in values["flags"]
    assert values["entry_cost"] is not None
    for field in ("gate_score", "exp_pnl_analog", "exp_pnl_model",
                  "driver_prediction", "driver_name"):
        assert values.get(field) is None, field
    receipts = [row["stage"] for row in values["native_stage_receipts"]]
    assert receipts[-1] == "serialization" and "gate" in receipts

    record = application.score_one(_request(), _inputs(
        quotes=_BAD_QUOTES, gate=_GATE_MODEL, analogs=_analog_block(),
        model_inputs={"mcap_log": math.log(2e9)},
    ))
    assert "BAD_QUOTE" in record.reason_codes
    assert record.validation_status == "refused"


def test_bad_quote_does_not_fire_below_the_ceiling():
    values = assemble_native_values(_inputs(
        quotes=_NEAR_QUOTES, gate=_GATE_MODEL,
        model_inputs={"mcap_log": math.log(2e9)},
    ))

    assert "BAD_QUOTE" not in values["flags"]
    assert values.get("gate_score") == 0.5
