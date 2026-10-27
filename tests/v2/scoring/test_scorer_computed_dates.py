"""Scorer-computed date features reach the gate and stay out of the source
bundle: ``days_to_print`` (legacy ``engine/features.py:1166-1168``) and
``dte_entry`` (legacy ``engine/score.py:2451``: ``dte_entry = int(priced.legs[0].dte)``).
"""
from datetime import date, datetime
import hashlib
import json
import types
from dataclasses import replace

import pandas as pd
import pytest

from engine.v2.models import FrozenInference, ModelBinding, ModelRelease
from engine.v2.models.contracts import ArtifactMember
from engine.v2.scoring import financial, stages
from engine.v2.scoring.frozen_executor import FrozenStageExecutor
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal, _observed_date, assemble_nightly_source_bundle,
    quote_domain_map)
from engine.v2.scoring.source_inputs import _quote_block, build_native_score_inputs
from engine.v2.scoring.stages import (
    NativeScoreInputs, STAGE_NAMES, StageReceipt, _check_pricing_boundary,
    _facts, _priced_dte_entry, assemble_native_values)

_RECEIPTS = tuple(
    StageReceipt(s, "declared-input", "declared-output")
    for s in STAGE_NAMES if s != "diagnostics")
_QUOTES = {r: {"bid": 1.9, "ask": 2.1}
           for r in (("C", 100.0, "2026-11-20"), ("P", 100.0, "2026-11-20"))}
#: Context-override sentinel: a value of ``_DROP`` removes the key entirely.
_DROP = object()


def _inputs(*, gate=None, quotes=None, quote_bid_ask=None,
            forecast_coefficients=None,
            diagnostics=None, expiry=None, event_date=None,
            drop_decision=False, context_overrides=None,
            quote_date=None) -> NativeScoreInputs:
    resolved_expiry = "2026-11-20" if expiry is None else expiry
    if quotes is None:
        if quote_bid_ask is None:
            quotes = {(right, strike, resolved_expiry): dict(value)
                      for (right, strike, _), value in _QUOTES.items()}
        else:
            bid, ask = quote_bid_ask
            quotes = {(right, strike, resolved_expiry): {"bid": bid, "ask": ask}
                      for right, strike, _ in _QUOTES}
    context = {"ticker": "AAA", "strategy": "STR-THRU",
               "event_date": "2026-10-27" if event_date is None else event_date,
               "entry_date": "2026-10-08",
               "quote_date": "2026-10-08" if quote_date is None else quote_date,
               "exit_date": "2026-10-27",
               "expiry": resolved_expiry, "strike": 100.0, "spot": 100.0,
               "chain_spot": 100.0,
               "as_of": "2026-10-07",
               "quotes": quotes}
    if drop_decision:
        del context["as_of"]
    for key, value in (context_overrides or {}).items():
        if value is _DROP:
            context.pop(key, None)
        else:
            context[key] = value
    coefficients = ({} if forecast_coefficients is None
                    else forecast_coefficients)
    return NativeScoreInputs(
        context=context,
        features={"model_inputs": {}},
        forecast={"driver_name": "abs_move",
                  "models": {"driver_prediction": {
                      "intercept": 0.0, "coefficients": coefficients}}},
        geometry=None, pricing=None, analogs={"recipe": None},
        simulation={"mode": "not_applicable"},
        gate=gate if gate is not None else {"mode": "not_applicable"},
        chooser={}, diagnostics=diagnostics if diagnostics is not None else {},
        source_ref="scorer-computed-dates-fixture",
        stage_receipts=_RECEIPTS,
    )


class _FrozenGate:
    """Adapts ``FrozenStageExecutor`` to the ``executors["gate_score"]`` contract."""

    def __init__(self, executor: FrozenStageExecutor, binding: ModelBinding) -> None:
        self._executor = executor
        self.feature_order = binding.feature_order

    def predict(self, features):
        return self._executor.predict(features)


def _frozen_gate(tmp_path, feature_order: tuple[str, ...]) -> dict:
    payload = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": list(feature_order),
        "outputs": [{"name": "gate_score", "intercept": 0.1,
                     "coefficients": [0.0] * len(feature_order)}],
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    (tmp_path / "estimator.json").write_bytes(raw)
    binding = ModelBinding(
        binding_id="gate-binding", model_id="gate-v1", role="gate",
        strategy_id="STR-THRU", decision_clock_id="entry-close",
        adapter="json-linear.v1", feature_order=feature_order,
        output_names=("gate_score",),
        members=(ArtifactMember(
            name="estimator", path="estimator.json",
            content_hash="sha256:" + hashlib.sha256(raw).hexdigest()),),
    )
    release = ModelRelease(release_id="release-gate-v1",
                           deployment_id="deployment-gate-v1", bindings=(binding,))
    executor = FrozenStageExecutor(
        inference=FrozenInference(tmp_path), release=release,
        binding_id=binding.binding_id)
    return {"executors": {"gate_score": _FrozenGate(executor, binding)},
            "threshold": 0.0}


def _kwargs(**overrides):
    base = dict(
        source_ref="test:scorer-computed-dates:v1",
        strategy="STR-THRU", as_of="2026-01-16",
        calendar_row={"ticker": "TEST", "event_date": "2026-01-15",
                      "entry_date": "2026-01-15", "exit_date": "2026-01-16",
                      "expiry": "2026-01-16", "spot": 100.0,
                      "calendar_observed_through": "2026-01-09"},
        panel_row={"date": "2026-01-15", "signal": 1.5},
        panel_anchor="2026-01-10",
        tier4_row={"pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01"},
        quote_rows=[{"right": r, "strike": 100.0, "expiry": "2026-01-16",
                     "bid": b, "ask": a, "observed_at": "2026-01-15"}
                    for r, b, a in (("C", 1.0, 1.2), ("P", 1.1, 1.3))],
        quote_status=None,
        feature_names=("signal", "days_to_print", "dte_entry", "entry_cost_pct"),
        forecast_recipes={"driver_prediction": {"intercept": 0.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:answer-named"},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0},
    )
    base.update(overrides)
    return base


def test_gate_receives_scorer_computed_dates(tmp_path):
    gate = _frozen_gate(tmp_path, ("days_to_print", "dte_entry"))
    values = assemble_native_values(_inputs(gate=gate))
    assert "MISSING_FEATURES" not in values["flags"]
    assert not [f for f in values["flags"] if f.startswith("MISSING_QUOTE")]
    assert values.get("entry_cost") is not None  # pricing succeeded
    assert values.get("gate_score") is not None
    # decision date 2026-10-07 differs from entry_date 2026-10-08: legacy uses the decision date
    assert values["days_to_print"] == 20.0
    assert values["dte_entry"] == 43  # priced first leg expiry 2026-11-20 minus quote date 2026-10-08


def test_forecast_consumer_receives_days_to_print():
    values = assemble_native_values(_inputs(forecast_coefficients={"days_to_print": 1.0}))
    assert not [f for f in values["flags"] if f.startswith("MISSING_FORECAST_INPUT")]
    assert values["days_to_print"] == 20.0
    assert values["driver_prediction"] == 20.0


def test_days_to_print_never_falls_back_to_entry_date():
    values = assemble_native_values(
        _inputs(forecast_coefficients={"days_to_print": 1.0}, drop_decision=True))
    assert "days_to_print" not in values
    assert "MISSING_FORECAST_INPUT:days_to_print" in values["flags"]


def test_forecast_receipt_input_records_days_to_print():
    seen: list = []
    assemble_native_values(
        _inputs(forecast_coefficients={"days_to_print": 1.0}),
        observer=seen.append)
    forecast = [o for o in seen if o.receipt.stage == "forecast"]
    assert len(forecast) == 1
    assert forecast[0].input_document["values"]["days_to_print"] == 20.0

    dropped: list = []
    assemble_native_values(
        _inputs(forecast_coefficients={"days_to_print": 1.0}, drop_decision=True),
        observer=dropped.append)
    dropped_forecast = [o for o in dropped if o.receipt.stage == "forecast"]
    assert len(dropped_forecast) == 1
    assert "days_to_print" not in dropped_forecast[0].input_document["values"]
    # the staged date is part of the declared forecast input, so the receipt moves
    assert forecast[0].receipt.input_hash != dropped_forecast[0].receipt.input_hash


def test_frozen_role_row_receives_computed_days_to_print(tmp_path):
    gate = _frozen_gate(tmp_path, ("days_to_print", "signal"))
    executor = gate["executors"]["gate_score"]._executor
    result = executor.execute({
        "role_model_inputs": {"gate": {"signal": 2.0}},
        "_native_stage_facts": {"days_to_print": 20.0},
    })
    # the captured role row omits days_to_print; the staged fact supplies it
    assert result.request.rows == ((20.0, 2.0),)

    facts = _facts(_inputs(), {})
    assert facts["days_to_print"] == 20.0
    assert facts["_native_stage_facts"]["days_to_print"] == 20.0
    dropped = _facts(_inputs(drop_decision=True), {})
    assert "days_to_print" not in dropped
    assert "days_to_print" not in dropped["_native_stage_facts"]


def test_computed_days_to_print_overrides_captured_role_row(tmp_path):
    gate = _frozen_gate(tmp_path, ("days_to_print", "signal"))
    executor = gate["executors"]["gate_score"]._executor
    result = executor.execute({
        "role_model_inputs": {"gate": {"days_to_print": 99.0, "signal": 2.0}},
        "_native_stage_facts": {"days_to_print": 20.0},
    })
    # the staged fact overrides the captured role-row value
    assert result.request.rows == ((20.0, 2.0),)


def test_facts_overlay_carries_priced_dte_entry_and_entry_cost_pct(tmp_path):
    gate = _frozen_gate(tmp_path, ("dte_entry", "entry_cost_pct"))
    inputs = _inputs(gate=gate, context_overrides={"chain_spot": 80.0})
    values = assemble_native_values(inputs)
    facts = _facts(inputs, values)
    assert values["entry_cost_pct"] is not None
    assert values["entry_cost_pct"] == pytest.approx(
        values["entry_cost"] / 80.0 * 100.0)
    assert values["entry_cost_pct"] != pytest.approx(
        values["entry_cost"] / 100.0 * 100.0)
    assert facts["dte_entry"] == 43
    assert facts["_native_stage_facts"]["dte_entry"] == 43
    assert facts["_native_stage_facts"]["entry_cost_pct"] == values["entry_cost_pct"]


def test_gate_role_row_missing_or_stale_dte_entry_uses_computed(tmp_path):
    gate = _frozen_gate(tmp_path, ("dte_entry", "entry_cost_pct"))
    executor = gate["executors"]["gate_score"]._executor
    inputs = _inputs(gate=gate)
    native = _facts(inputs, assemble_native_values(inputs))["_native_stage_facts"]
    # (a) the captured role row omits the scorer-computed columns entirely.
    missing = executor.execute({
        "role_model_inputs": {"gate": {}},
        "_native_stage_facts": native,
    })
    # (b) a stale captured row is overridden by the staged facts.
    stale = executor.execute({
        "role_model_inputs": {"gate": {"dte_entry": 99, "entry_cost_pct": 99.0}},
        "_native_stage_facts": native,
    })
    assert missing.request.rows[0][0] == 43
    assert missing.request.rows == stale.request.rows


def test_dte_entry_not_written_when_pricing_refuses():
    values = assemble_native_values(_inputs(quotes={}))
    assert "dte_entry" not in values
    assert values.get("entry_cost") is None


def test_overlay_omits_dte_entry_when_pricing_refuses():
    inputs = _inputs(quotes={})
    values = assemble_native_values(inputs)
    assert "dte_entry" not in _facts(inputs, values)["_native_stage_facts"]


def test_dates_not_projected_into_source_mask():
    bundle = assemble_nightly_source_bundle(**_kwargs())
    for name in ("days_to_print", "dte_entry", "entry_cost_pct"):
        assert name not in bundle.feature_vector
        assert name not in bundle.feature_missing_mask
    assert "signal" in bundle.feature_vector


#: The scorer-computed feature names a source row must never contribute.
_COMPUTED_NAMES = ("days_to_print", "dte_entry", "entry_cost_pct")


@pytest.mark.parametrize("source", ["tier4", "panel", "both"])
def test_computed_names_not_projected_even_when_a_source_row_carries_them(source):
    # A guard that skips computed names only when they are absent from BOTH
    # rows would still satisfy test_dates_not_projected_into_source_mask,
    # whose rows carry none of them. Here the named row really does carry
    # each computed name (with the fold_start a non-null Tier-4 value needs),
    # so the guard must skip on the name alone, not on its absence.
    computed = {"days_to_print": 7.0, "dte_entry": 9, "entry_cost_pct": 3.0}
    fold_starts = {f"{name}_fold_start": "2026-01-01" for name in _COMPUTED_NAMES}
    tier4_row = {"pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01"}
    panel_row = {"date": "2026-01-15", "signal": 1.5}
    if source in ("tier4", "both"):
        tier4_row.update(computed)
        tier4_row.update(fold_starts)
    if source in ("panel", "both"):
        panel_row.update(computed)
    bundle = assemble_nightly_source_bundle(**_kwargs(
        tier4_row=tier4_row, panel_row=panel_row))
    for name in _COMPUTED_NAMES:
        assert name not in bundle.feature_vector
        assert name not in bundle.feature_missing_mask
    assert "signal" in bundle.feature_vector


def test_bundle_through_scoring_gives_forecast_days_to_print():
    bundle = assemble_nightly_source_bundle(**_kwargs(
        as_of="2026-01-14",
        calendar_row={"ticker": "TEST", "event_date": "2026-01-30",
                      "entry_date": "2026-01-14", "exit_date": "2026-01-30",
                      "expiry": "2026-02-20", "spot": 100.0,
                      "calendar_observed_through": "2026-01-09"},
        panel_row={"date": "2026-01-30", "signal": 1.5},
        quote_rows=[{"right": r, "strike": 100.0, "expiry": "2026-02-20",
                     "bid": b, "ask": a, "observed_at": "2026-01-14"}
                    for r, b, a in (("C", 1.0, 1.2), ("P", 1.1, 1.3))],
        forecast_recipes={"driver_prediction": {
            "intercept": 0.0, "coefficients": {"days_to_print": 1.0}}},
    ))
    values = assemble_native_values(build_native_score_inputs(bundle))
    assert values["days_to_print"] == 16.0  # 2026-01-30 minus 2026-01-14
    assert "MISSING_FORECAST_INPUT:days_to_print" not in values["flags"]
    assert bundle.context["as_of"] == "2026-01-14"

    # Negative control: dropping the validated as_of from the context leaves
    # the forecast consumer without its days_to_print input.
    stripped = replace(bundle, context={
        k: v for k, v in bundle.context.items() if k != "as_of"})
    negative = assemble_native_values(build_native_score_inputs(stripped))
    assert "days_to_print" not in negative


def test_helpers_match_legacy_arithmetic():
    a, b = "2026-10-27", "2026-10-08"
    assert financial.days_to_print(a, b) == 19.0
    assert financial.days_to_print(a, b) == float(
        (pd.Timestamp(a) - pd.Timestamp(b)).days)
    assert financial.dte_entry("2026-11-20", b) == 43
    assert financial.days_to_print(None, b) is None
    assert financial.days_to_print("not-a-date", b) is None
    assert financial.dte_entry(None, b) is None
    assert financial.dte_entry("not-a-date", b) is None


def test_diagnostics_cannot_overwrite_computed_dates():
    values = assemble_native_values(
        _inputs(diagnostics={"days_to_print": 99.0, "dte_entry": 99}))
    assert values["days_to_print"] == 20.0
    assert values["dte_entry"] == 43

    refused = assemble_native_values(
        _inputs(quotes={},
                diagnostics={"days_to_print": 99.0, "dte_entry": 99}))
    assert "dte_entry" not in refused


def test_dte_entry_follows_the_priced_expiry(tmp_path):
    gate = _frozen_gate(tmp_path, ("dte_entry", "entry_cost_pct"))
    values = assemble_native_values(_inputs(gate=gate, expiry="2026-12-18"))
    assert values["dte_entry"] == 71
    assert values.get("entry_cost") is not None


def test_priced_leg_expiry_wins_over_context_expiry():
    priced = {"legs": ({"expiry": "2026-12-18"},), "entry_cost": 1.0,
              "quote_date": "2026-10-08", "expiry": "2026-11-20"}
    assert _priced_dte_entry(priced) == 71
    assert _priced_dte_entry({**priced, "entry_cost": None}) is None
    assert _priced_dte_entry({**priced, "legs": ()}) is None
    # The priced leg's own expiry is the reference date: moving the quote
    # date one day later shortens the count by one.
    assert _priced_dte_entry({**priced, "quote_date": "2026-10-09"}) == 70
    # Quote date equal to the leg expiry is zero days, not absent.
    assert _priced_dte_entry({**priced, "quote_date": "2026-12-18"}) == 0


def test_days_to_print_second_value():
    values = assemble_native_values(
        _inputs(event_date="2026-11-05",
                forecast_coefficients={"days_to_print": 1.0}))
    assert values["days_to_print"] == 29.0


def test_helper_edge_cases():
    assert financial.dte_entry(float("nan"), "2026-10-08") is None
    assert financial.dte_entry(True, "2026-10-08") is None
    assert financial.dte_entry("2026-12-18", "2026-10-08") == 71
    assert financial.days_to_print("2026-11-05", "2026-10-07") == 29.0


def _days_surfaces(inputs):
    """The three surfaces the staged decision date must reach.

    Returns ``(values, receipt_values, overlay)``: the computed
    ``values["days_to_print"]``, the forecast receipt's recorded input values,
    and the frozen-role overlay ``_facts(...)["_native_stage_facts"]`` the
    frozen executor's role row consumes.
    """
    seen: list = []
    values = assemble_native_values(inputs, observer=seen.append)
    forecast = [o for o in seen if o.receipt.stage == "forecast"]
    assert len(forecast) == 1
    facts = _facts(inputs, values)
    return values, forecast[0].input_document["values"], \
        facts["_native_stage_facts"]


def _assert_days_on_all_surfaces(tmp_path, inputs, expected):
    values, receipt_values, overlay = _days_surfaces(inputs)
    assert values["days_to_print"] == expected
    assert receipt_values["days_to_print"] == expected
    assert overlay["days_to_print"] == expected
    gate = _frozen_gate(tmp_path, ("days_to_print", "signal"))
    executor = gate["executors"]["gate_score"]._executor
    result = executor.execute({
        "role_model_inputs": {"gate": {"signal": 2.0}},
        "_native_stage_facts": overlay,
    })
    assert result.request.rows == ((expected, 2.0),)


def _assert_days_absent(inputs):
    values, receipt_values, overlay = _days_surfaces(inputs)
    assert "days_to_print" not in values
    assert "days_to_print" not in receipt_values
    facts = _facts(inputs, values)
    assert "days_to_print" not in facts
    assert "days_to_print" not in facts["_native_stage_facts"]
    assert "days_to_print" not in overlay


def test_only_requested_as_of_is_used(tmp_path):
    # No as_of at all: the identity requested_as_of is the decision date.
    _assert_days_on_all_surfaces(
        tmp_path,
        _inputs(context_overrides={"as_of": _DROP,
                                   "requested_as_of": "2026-10-07"}),
        20.0)


def test_as_of_wins_over_conflicting_requested_as_of(tmp_path):
    # as_of 2026-10-07 -> 20.0; the stale requested_as_of 2026-10-01 -> 26.0.
    _assert_days_on_all_surfaces(
        tmp_path,
        _inputs(context_overrides={"requested_as_of": "2026-10-01"}),
        20.0)


def test_neither_decision_date_leaves_it_absent():
    _assert_days_absent(_inputs(context_overrides={"as_of": _DROP}))


#: Supplied-but-invalid as_of values: only a missing/None as_of may fall back.
_INVALID_AS_OF = (False, 0, 0.0, "", "junk", 20261007)


@pytest.mark.parametrize("invalid_as_of", _INVALID_AS_OF)
def test_invalid_as_of_never_falls_back_to_requested_as_of(invalid_as_of):
    # A supplied (even falsy/invalid) as_of is parsed strictly; the identity
    # requested_as_of must never replace it, so days_to_print is absent from
    # every surface even though the forecast consumer wants it.
    inputs = _inputs(forecast_coefficients={"days_to_print": 1.0},
                     context_overrides={"as_of": invalid_as_of,
                                        "requested_as_of": "2026-10-07"})
    _assert_days_absent(inputs)


def test_as_of_none_or_absent_falls_back_to_requested_as_of(tmp_path):
    # A present-but-None as_of is the only supplied-as_of fallback case...
    _assert_days_on_all_surfaces(
        tmp_path,
        _inputs(context_overrides={"as_of": None,
                                   "requested_as_of": "2026-10-07"}),
        20.0)
    # ...and an absent as_of key falls back too.
    _assert_days_on_all_surfaces(
        tmp_path,
        _inputs(context_overrides={"as_of": _DROP,
                                   "requested_as_of": "2026-10-07"}),
        20.0)


def test_missing_event_date_leaves_days_absent():
    _assert_days_absent(_inputs(context_overrides={"event_date": _DROP}))


def test_entry_cost_pct_overlay():
    inputs = _inputs()
    values = assemble_native_values(inputs)
    overlay = _facts(inputs, values)["_native_stage_facts"]
    assert values["entry_cost_pct"] is not None
    assert overlay["entry_cost_pct"] == values["entry_cost_pct"]

    refused = _inputs(quotes={})
    refused_values = assemble_native_values(refused)
    refused_overlay = _facts(refused, refused_values)["_native_stage_facts"]
    assert "entry_cost_pct" not in refused_overlay


def test_missing_quote_date_keeps_captured_dte_entry():
    values = assemble_native_values(_inputs(
        context_overrides={"dte_entry": 55, "quote_date": _DROP}))
    # No quote_date means no computed dte_entry, so the value already in the
    # context survives: it must not be overwritten with None.
    assert values["dte_entry"] == 55


def _pricing_stand_ins(*, pricing_refusal=None, geometry_refusal=None,
                       pricing_legs=None, geometry_legs=None):
    leg = types.SimpleNamespace(bid=1.9, ask=2.1, strike=100.0,
                                expiry="2026-12-18", quote_date=None)
    pricing = types.SimpleNamespace(
        refusal=pricing_refusal,
        legs=(leg,) if pricing_legs is None else pricing_legs,
        spot=100.0, entry_cost=1.0)
    geometry = types.SimpleNamespace(
        refusal=geometry_refusal,
        legs=(leg,) if geometry_legs is None else geometry_legs,
        spot=100.0)
    return geometry, pricing


def test_geometry_or_pricing_refusal_blocks_dte_write():
    boundary_values = {"entry_cost": 1.0, "spot": 100.0,
                       "quote_date": "2026-10-08"}
    geometry, pricing = _pricing_stand_ins(pricing_refusal="X")
    _check_pricing_boundary(boundary_values, geometry, pricing, [])
    assert "dte_entry" not in boundary_values

    boundary_values = {"entry_cost": 1.0, "spot": 100.0,
                       "quote_date": "2026-10-08"}
    geometry, pricing = _pricing_stand_ins(geometry_refusal="Y")
    _check_pricing_boundary(boundary_values, geometry, pricing, [])
    assert "dte_entry" not in boundary_values

    boundary_values = {"entry_cost": 1.0, "spot": 100.0,
                       "quote_date": "2026-10-08"}
    geometry, pricing = _pricing_stand_ins(pricing_legs=())
    _check_pricing_boundary(boundary_values, geometry, pricing, [])
    assert "dte_entry" not in boundary_values

    boundary_values = {"entry_cost": 1.0, "spot": 100.0,
                       "quote_date": "2026-10-08"}
    geometry, pricing = _pricing_stand_ins()
    _check_pricing_boundary(boundary_values, geometry, pricing, [])
    assert boundary_values["dte_entry"] == 71


#: Distinct scenarios so a production constant (e.g. a hard-coded
#: ``dte_entry`` or ``days_to_print``) cannot satisfy every assertion: the
#: literals differ across A, B and C. Each ``inputs`` block builds via ``_inputs``.
_SCENARIOS = {
    "A": {
        "inputs": {},
        "bid_ask": (1.9, 2.1),
        "days_to_print": 20.0,   # event 2026-10-27 - decision as_of 2026-10-07
        "dte_entry": 43,         # priced expiry 2026-11-20 - quote_date 2026-10-08
        "entry_cost_pct": 4.0,   # 2 legs x mid 2.0 / spot 100 * 100
    },
    "B": {
        "inputs": {"event_date": "2026-11-05", "expiry": "2026-12-18",
                   "quote_bid_ask": (2.9, 3.1)},
        "bid_ask": (2.9, 3.1),
        "days_to_print": 29.0,   # event 2026-11-05 - decision as_of 2026-10-07
        "dte_entry": 71,         # priced expiry 2026-12-18 - quote_date 2026-10-08
        "entry_cost_pct": 6.0,   # 2 legs x mid 3.0 / spot 100 * 100
    },
    "C": {
        "inputs": {"quote_date": "2026-10-09"},
        "bid_ask": (1.9, 2.1),
        "days_to_print": 20.0,   # event 2026-10-27 - decision as_of 2026-10-07
        "dte_entry": 42,         # priced expiry 2026-11-20 - quote_date 2026-10-09
        "entry_cost_pct": 4.0,   # 2 legs x mid 2.0 / spot 100 * 100
    },
}


def _row(scenario: str) -> tuple:
    spec = _SCENARIOS[scenario]
    return (spec["days_to_print"], spec["dte_entry"], spec["entry_cost_pct"])


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_computed_dates_matrix(scenario, tmp_path):
    spec = _SCENARIOS[scenario]
    days, dte, pct = spec["days_to_print"], spec["dte_entry"], spec["entry_cost_pct"]
    # Independent arithmetic: the pct is 2 legs x mid x 100 / spot (spot = 100).
    bid, ask = spec["bid_ask"]
    assert pct == pytest.approx(2 * ((bid + ask) / 2.0) * 100.0 / 100.0)

    # Surface 1 (top-level values) + Surface 2 (forecast-stage receipt input).
    seen: list = []
    inputs = _inputs(**spec["inputs"])
    values = assemble_native_values(inputs, observer=seen.append)
    assert values["days_to_print"] == days
    assert values["dte_entry"] == dte
    assert values["entry_cost_pct"] == pytest.approx(pct)

    forecast = [o for o in seen if o.receipt.stage == "forecast"]
    assert len(forecast) == 1
    receipt_values = forecast[0].input_document["values"]
    assert receipt_values["days_to_print"] == days
    # dte_entry/entry_cost_pct are computed after forecasting: absent there.
    assert "dte_entry" not in receipt_values
    assert "entry_cost_pct" not in receipt_values

    # Surface 3 (overlay: facts + _native_stage_facts).
    facts = _facts(inputs, values)
    overlay = facts["_native_stage_facts"]
    for name, expected in (("days_to_print", days), ("dte_entry", dte),
                           ("entry_cost_pct", pct)):
        assert facts[name] == pytest.approx(expected)
        assert overlay[name] == pytest.approx(expected)

    # Surface 4 (frozen consumer row): both a missing row and a stale row must
    # resolve to the scenario literals, not to a captured/constant value.
    gate = _frozen_gate(
        tmp_path, ("days_to_print", "dte_entry", "entry_cost_pct"))
    executor = gate["executors"]["gate_score"]._executor
    missing = executor.execute({
        "role_model_inputs": {"gate": {}},
        "_native_stage_facts": overlay,
    })
    assert missing.request.rows[0] == pytest.approx(_row(scenario))
    stale = executor.execute({
        "role_model_inputs": {"gate": {
            "days_to_print": 99.0, "dte_entry": 99, "entry_cost_pct": 99.0}},
        "_native_stage_facts": overlay,
    })
    assert stale.request.rows[0] == pytest.approx(_row(scenario))

    # Surface 5 (diagnostics cannot overwrite the computed values).
    diagnosed = assemble_native_values(_inputs(
        diagnostics={"days_to_print": 99.0, "dte_entry": 99,
                     "entry_cost_pct": 99.0},
        **spec["inputs"]))
    assert diagnosed["days_to_print"] == days
    assert diagnosed["dte_entry"] == dte
    assert diagnosed["entry_cost_pct"] == pytest.approx(pct)


#: Non-exact or wrong-typed date values the strict parser must reject.
_BAD_DATES = [
    20261007, 20261007.0, True, float("nan"), pd.NaT,
    "2026-10-07junk", "20261007", " 2026-10-07", "2026-10-07 ",
    "2026-13-45", "", "2026-10-07T",
]


@pytest.mark.parametrize("bad", _BAD_DATES)
@pytest.mark.parametrize("helper", ["days_to_print", "dte_entry"])
@pytest.mark.parametrize("position", ["first", "second"])
def test_date_helpers_reject_bad_types_and_suffixes(helper, position, bad):
    good = "2026-10-08"
    args = (bad, good) if position == "first" else (good, bad)
    assert getattr(financial, helper)(*args) is None


#: Exact ISO forms (objects or complete strings) that collapse to the same day.
_GOOD_FORMS = [
    "2026-10-27",
    date(2026, 10, 27),
    datetime(2026, 10, 27, 15, 30),
    pd.Timestamp("2026-10-27"),
    "2026-10-27T00:00:00",
    "2026-10-27 00:00:00",
]


@pytest.mark.parametrize("form", _GOOD_FORMS)
def test_date_helpers_accept_exact_forms(form):
    # 2026-10-27 minus 2026-10-08 is 19 days, computed independently.
    assert financial.days_to_print(form, "2026-10-08") == 19.0
    assert financial.dte_entry(form, "2026-10-08") == 19


def test_scoring_ignores_malformed_decision_and_event_dates():
    # as_of is an int and the identity requested_as_of is dropped: no decision date.
    values = assemble_native_values(_inputs(
        context_overrides={"as_of": 20261007, "requested_as_of": _DROP}))
    assert "days_to_print" not in values

    values = assemble_native_values(_inputs(
        context_overrides={"as_of": "2026-10-07junk"}))
    assert "days_to_print" not in values

    values = assemble_native_values(_inputs(
        context_overrides={"event_date": "2026-10-27junk"}))
    assert "days_to_print" not in values

    values = assemble_native_values(_inputs(
        context_overrides={"quote_date": "2026-10-08junk"}))
    assert values.get("entry_cost") is not None  # the run priced
    assert "dte_entry" not in values  # its malformed quote_date blocks the write


def test_answer_denylist_claim_matches_code():
    from engine.v2.scoring.source_inputs import (
        _reject_answers, AnswerFieldsInSource)
    with pytest.raises(AnswerFieldsInSource):
        _reject_answers("probe", {"entry_cost_pct": 1.0})
    _reject_answers("probe", {"dte_entry": 5})
    _reject_answers("probe", {"days_to_print": 3.0})


def _quote_row(strike, observed_at, bid=1.0, ask=1.2):
    return {"right": "C", "strike": strike, "expiry": "2026-11-20",
            "bid": bid, "ask": ask, "observed_at": observed_at}


def test_quote_domain_map_carries_each_rows_observed_date():
    quotes = quote_domain_map([
        _quote_row(100.0, "2026-10-08T15:00:00"),
        _quote_row(105.0, "2026-10-09T15:00:00"),
    ])
    assert quotes["C:100.0:2026-11-20"]["observed_date"] == "2026-10-08"
    assert quotes["C:105.0:2026-11-20"]["observed_date"] == "2026-10-09"


def test_quote_domain_map_duplicate_contract_keeps_earliest_date_or_refuses():
    same = quote_domain_map([
        _quote_row(100.0, "2026-10-09T15:00:00"),
        _quote_row(100.0, "2026-10-08T15:00:00"),
    ])
    assert same["C:100.0:2026-11-20"]["observed_date"] == "2026-10-08"
    with pytest.raises(NightlySourceBundleRefusal) as caught:
        quote_domain_map([
            _quote_row(100.0, "2026-10-08T15:00:00"),
            _quote_row(100.0, "2026-10-09T15:00:00", bid=1.1, ask=1.3),
        ])
    assert "INVALID_QUOTE_DOMAIN" in str(caught.value) or getattr(
        caught.value, "code", "") == "INVALID_QUOTE_DOMAIN"


@pytest.mark.parametrize(("legs", "expected"), [
    ([{"expiry": "2026-11-20", "quote_date": "2026-10-09"},
      {"expiry": "2026-11-20", "quote_date": "2026-10-09"}], 42),
    ([{"expiry": "2026-11-20", "quote_date": "2026-10-08"},
      {"expiry": "2026-11-20", "quote_date": "2026-10-08"}], 43),
    ([{"expiry": "2026-11-20", "quote_date": "2026-10-09"},
      {"expiry": "2026-11-20"}], 42),
    ([{"expiry": "2026-11-20", "quote_date": "2026-10-08"},
      {"expiry": "2026-11-20", "quote_date": "2026-10-09"}], 43),
    ([{"expiry": "2026-11-20", "quote_date": "2026-10-09"},
      {"expiry": "2026-11-20", "quote_date": "2026-10-08"}], 42),
    ([{"expiry": "2026-11-20"}, {"expiry": "2026-11-20"}], 43),
    ([{"expiry": "2026-11-20"},
      {"expiry": "2026-11-20", "quote_date": "2026-10-09"}], 43),
])
def test_priced_dte_entry_uses_the_legs_own_quote_date(legs, expected):
    values = {"entry_cost": 2.0, "legs": legs, "quote_date": "2026-10-08"}
    assert stages._priced_dte_entry(values) == expected


def _dated_quotes(call_date, put_date):
    return {("C", 100.0, "2026-11-20"): {"bid": 1.9, "ask": 2.1, "observed_date": call_date},
            ("P", 100.0, "2026-11-20"): {"bid": 1.9, "ask": 2.1, "observed_date": put_date}}


def test_dte_entry_uses_the_priced_legs_date_not_the_earliest_domain_date(tmp_path):
    # context quote_date is the domain's earliest observation (2026-10-08); both
    # legs were priced from rows observed on 2026-10-09.
    gate = _frozen_gate(tmp_path, ("dte_entry", "entry_cost_pct"))
    inputs = _inputs(gate=gate, quotes=_dated_quotes("2026-10-09", "2026-10-09"))
    values = assemble_native_values(inputs)
    assert values["dte_entry"] == 42
    facts = _facts(inputs, values)
    assert facts["dte_entry"] == 42
    assert facts["_native_stage_facts"]["dte_entry"] == 42
    executor = gate["executors"]["gate_score"]._executor
    row = executor.execute({
        "role_model_inputs": {"gate": {"dte_entry": 99, "entry_cost_pct": 99.0}},
        "_native_stage_facts": facts["_native_stage_facts"],
    })
    assert row.request.rows[0][0] == 42


def test_dte_entry_unchanged_when_legs_use_the_earliest_date():
    values = assemble_native_values(
        _inputs(quotes=_dated_quotes("2026-10-08", "2026-10-08")))
    assert values["dte_entry"] == 43


@pytest.mark.parametrize(("call_date", "put_date", "expected"), [
    ("2026-10-08", "2026-10-09", 43),
    ("2026-10-09", "2026-10-08", 42),
])
def test_legs_priced_from_different_dates_use_the_first_legs_date(
        call_date, put_date, expected):
    # leg 0 is the call; legacy takes dte_entry from legs[0] (engine/score.py:2451)
    values = assemble_native_values(
        _inputs(quotes=_dated_quotes(call_date, put_date)))
    assert values["legs"][0]["right"] == "C"
    assert not any(str(flag).startswith("INVALID_QUOTE:") for flag in values["flags"])
    assert values["entry_cost"] is not None
    assert values["dte_entry"] == expected


@pytest.mark.parametrize("observed_date", [
    "junk", "2026-02-30", "2026-10-09T15:00:00", "", 20261009,
])
def test_junk_observed_date_leaves_the_leg_undated_and_falls_back(observed_date):
    # A junk observed_date must not parse into the leg's quote_date, so the
    # priced leg stays undated and dte_entry falls back to the context
    # quote_date 2026-10-08 (expiry 2026-11-20 minus it is 43).
    values = assemble_native_values(
        _inputs(quotes=_dated_quotes(observed_date, observed_date)))
    assert values["legs"][0].get("quote_date") is None
    assert values["dte_entry"] == 43
    assert not any(str(flag).startswith("INVALID_QUOTE")
                   for flag in values["flags"])


_CONTRACT = ("C", 100.0, "2026-11-20")


@pytest.mark.parametrize("observed_date", [
    "junk", "2026-02-30", "2026-1-5", "2026-10-09T15:00:00", 20261009, True,
])
def test_quote_block_rejects_bad_observed_date(observed_date):
    with pytest.raises(ValueError, match="observed_date"):
        _quote_block({_CONTRACT: {"bid": 1.9, "ask": 2.1,
                                  "observed_date": observed_date}})


def test_quote_block_accepts_absent_observed_date():
    quotes = _quote_block({_CONTRACT: {"bid": 1.9, "ask": 2.1}})
    assert "observed_date" not in quotes[_CONTRACT]


def test_quote_block_keeps_a_valid_observed_date():
    quotes = _quote_block({_CONTRACT: {"bid": 1.9, "ask": 2.1,
                                       "observed_date": "2026-10-09"}})
    assert quotes[_CONTRACT]["observed_date"] == "2026-10-09"
    assert quotes[_CONTRACT]["bid"] == 1.9
    assert quotes[_CONTRACT]["ask"] == 2.1


@pytest.mark.parametrize("value", [None, "", "   ", "junk", float("nan"), pd.NaT, 0, 20261009, True, 1.5])
def test_observed_date_omits_non_dates(value):
    assert _observed_date(value) is None


@pytest.mark.parametrize(("value", "expected"), [
    ("2026-10-09", "2026-10-09"),
    ("2026-10-09T15:00:00", "2026-10-09"),
    (pd.Timestamp("2026-10-09 15:00"), "2026-10-09"),
    (datetime(2026, 10, 9, 15, 0), "2026-10-09"),
    (date(2026, 10, 9), "2026-10-09"),
])
def test_observed_date_parses_real_dates(value, expected):
    assert _observed_date(value) == expected


def test_quote_domain_map_omits_observed_date_for_empty_or_nat_rows():
    for bad in ("", pd.NaT, float("nan"), 0):
        quotes = quote_domain_map([_quote_row(100.0, bad)])
        assert "observed_date" not in quotes["C:100.0:2026-11-20"]
