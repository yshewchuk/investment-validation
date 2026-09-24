"""Phase 4 parity gap (observational replay 13, 2026-09-24): unpriced rows.

Fixtures 002 (members 2,4,5,6,8,9,10), 009, 015 and 019 (member 2) failed the
``contracts`` comparison with exactly one differing field: legacy's
``record["quote_date"]`` -- projected into the contract timeline as
``execution_date`` by ``checks/phase4_real.py::_contract_projection`` -- is a
real date, while ``native.quote_provenance`` had no ``quote_date`` at all.

Root cause: ``application._pricing_dependent_fields`` popped ``quote_date``
whenever ``entry_cost`` was None -- conflating "pricing refused" with
"pricing never ran". Legacy stamps ``result.quote_date`` on the first line of
``_price_entry`` (engine/score.py:2287), BEFORE the chain lookup, so a row
that reached pricing and refused (NO_CHAIN, COARSE_LADDER, an empty quote
domain) still retains its observed quote date; only a row whose pricing stage
was withheld (a FORECAST_SIZED decline returns before ``_price_entry``,
engine/score.py:1921-1922) carries no date at all. Native mirrors that
run/never-ran fact in ``values``: ``stages._publish_pricing`` writes the
``entry_cost`` KEY whenever pricing ran (value ``None`` on a refusal), and
the sizing-withhold path merges no owned output into ``values`` at all.
The projection preserves a genuinely observed date on a pricing-refused row
and never synthesizes one; truly absent dates and priced rows are unchanged.
"""
from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application
from engine.v2.scoring.stages import (
    STAGE_NAMES,
    NativeScoreInputs,
    StageReceipt,
)

_RECEIPTS = tuple(
    StageReceipt(stage, "declared-input", "declared-output")
    for stage in STAGE_NAMES if stage != "diagnostics"
)
# A RAMP7 forecast whose width_moneyness clears WIDTH_MAX (see
# engine/forecast_sizing.py:54 and tests/test_v2_native_missing_forecast_input_gap.py):
# legacy's ``Scorer._size_from_forecast`` declines it with NO_FORECAST and
# returns before ``_price_entry`` ever runs.
_RAMP7_TOO_WIDE_FORECAST = 60.0


def _request(strategy: str) -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-quote-date", calendar_revision="cal-1",
        strategy_version=strategy, deployment_id="dep-1",
        decision_clock_id="entry-close", requested_decision_at="2026-09-16",
        snapshot_id="snap-1", mode="replay", fill_model={"alpha": 0.5},
    )


def _inputs(*, strategy="STR-THRU", context_overrides=None,
            forecast=None) -> NativeScoreInputs:
    context = {
        "ticker": "AAA", "strategy": strategy, "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-18",
        "expiry": "2026-09-18", "strike": 100.0, "spot": 100.0,
        "quotes": {},
    }
    context.update(context_overrides or {})
    if forecast is None:
        forecast = {
            "driver_name": "abs_move",
            "models": {"driver_prediction": {"intercept": 0.0,
                                             "coefficients": {}}},
        }
    return NativeScoreInputs(
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
        source_ref="unpriced-quote-date-fixture",
        stage_receipts=_RECEIPTS,
    )


def test_pricing_refused_row_keeps_its_observed_quote_date():
    # An empty quote domain refuses pricing (MISSING_PRICING_INPUT) and
    # leaves entry_cost None, but the quote date the lookup ran on is a
    # genuinely observed fact the legacy record retains -- native must
    # preserve it in BOTH contract projections, exactly as
    # _contract_projection reads execution_date off quote_provenance.
    record = application.score_one(
        _request("STR-THRU"),
        _inputs(context_overrides={"quote_date": "2026-09-15"}),
    )

    assert record.resolved_request["entry_cost"] is None
    assert "MISSING_PRICING_INPUT" in record.reason_codes
    assert record.entry_exit_plan.get("quote_date") == "2026-09-15"
    assert record.quote_provenance.get("quote_date") == "2026-09-15"


def test_pricing_refused_row_without_an_observed_date_stays_undated():
    # The truly-absent case, unchanged: no observed quote date means the
    # key must be ABSENT from both dicts (never present-as-None), and it
    # must not be synthesized from entry_date or expiry either.
    record = application.score_one(_request("STR-THRU"), _inputs())

    assert record.resolved_request["entry_cost"] is None
    assert "MISSING_PRICING_INPUT" in record.reason_codes
    assert "quote_date" not in record.entry_exit_plan
    assert "quote_date" not in record.quote_provenance


def test_sizing_declined_row_never_leaks_its_forward_looking_date():
    # The pricing stage withheld: legacy returned before ``_price_entry``
    # and its record.quote_date is None, even though the captured bundle
    # context carries the quote date the lookup WOULD have asked for
    # (engine/score.py:1854-1866, quote_status="not_reached"). Preservation
    # is keyed off pricing having RUN, never off the date merely sitting
    # in values -- so the projected contract stays undated like legacy's.
    record = application.score_one(
        _request("RAMP7"),
        _inputs(
            strategy="RAMP7",
            context_overrides={"spot": 0.64, "quote_date": "2026-09-15"},
            forecast={
                "forecast_abs_move": {"intercept": _RAMP7_TOO_WIDE_FORECAST,
                                      "coefficients": {}},
            },
        ),
    )

    assert "NO_FORECAST" in record.reason_codes
    assert "entry_cost" not in record.resolved_request
    assert "quote_date" not in record.entry_exit_plan
    assert "quote_date" not in record.quote_provenance


def test_priced_row_projection_is_unchanged():
    # Positive control on the untouched branch: a row that priced keeps
    # its quote date in both dicts, as before.
    record = application.score_one(
        _request("STR-THRU"),
        _inputs(
            context_overrides={
                "quote_date": "2026-09-15",
                "quotes": {
                    ("C", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
                    ("P", 100.0, "2026-09-18"): {"bid": 1.9, "ask": 2.1},
                },
            },
        ),
    )

    assert record.resolved_request["entry_cost"] is not None
    assert record.entry_exit_plan.get("quote_date") == "2026-09-15"
    assert record.quote_provenance.get("quote_date") == "2026-09-15"
