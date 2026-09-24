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


def test_quote_date_present_iff_genuinely_observed():
    # Legacy stamps ``result.quote_date`` on the first line of
    # ``_price_entry`` (engine/score.py:2287), before any chain lookup, so
    # an unpriced row that REACHED pricing keeps its observed quote date
    # (Phase 4 parity gap: the projection used to pop it whenever
    # ``entry_cost`` was None). It must stay ABSENT -- never
    # present-as-None -- exactly where nothing was observed: no date in
    # the row's own values, or a pricing stage that never ran (see the
    # sizing-withhold control in tests/test_v2_scoring_unpriced_quote_date.py).
    fields = {**_fields(), "quote_date": "2026-09-16"}
    # Compatibility input with no quotes: _resolve_pricing returns the
    # fields' own entry_cost (5.0) with refusal=None, so pricing ran.
    priced = _score(fields)

    assert priced.resolved_request["entry_cost"] == 5.0
    assert priced.entry_exit_plan.get("quote_date") == "2026-09-16"
    assert priced.quote_provenance.get("quote_date") == "2026-09-16"

    # spot=None is the missing essential _resolve_geometry refuses on
    # (MISSING_SPOT) before pricing: entry_cost ends up None -- but the
    # pricing stage RAN and the date was genuinely observed, so it is
    # preserved in both dicts, exactly as the legacy record retains it.
    refused = _score({**fields, "spot": None})

    assert refused.resolved_request["entry_cost"] is None
    assert refused.entry_exit_plan.get("quote_date") == "2026-09-16"
    assert refused.quote_provenance.get("quote_date") == "2026-09-16"

    # Same refused row with NO observed date: the key must not appear in
    # either dict at all (the entry_cost convention's present-as-None
    # ban), and no date may be synthesized from entry_date.
    undated = _score({**fields, "spot": None, "quote_date": None})

    assert undated.resolved_request["entry_cost"] is None
    assert "quote_date" not in undated.entry_exit_plan
    assert "quote_date" not in undated.quote_provenance
