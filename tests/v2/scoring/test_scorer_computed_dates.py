"""Scorer-computed date features reach the gate and stay out of the source
bundle: ``days_to_print`` (legacy ``engine/features.py:1166-1168``) and
``dte_entry`` (legacy ``engine/score.py:2451``: ``dte_entry = int(priced.legs[0].dte)``).
"""
import hashlib
import json
import types
from dataclasses import replace

import pandas as pd

from engine.v2.models import FrozenInference, ModelBinding, ModelRelease
from engine.v2.models.contracts import ArtifactMember
from engine.v2.scoring import financial
from engine.v2.scoring.frozen_executor import FrozenStageExecutor
from engine.v2.scoring.nightly_source_bundle import assemble_nightly_source_bundle
from engine.v2.scoring.source_inputs import build_native_score_inputs
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


def _inputs(*, gate=None, quotes=None, forecast_coefficients=None,
            diagnostics=None, expiry=None, event_date=None,
            drop_decision=False, context_overrides=None) -> NativeScoreInputs:
    resolved_expiry = "2026-11-20" if expiry is None else expiry
    if quotes is None:
        quotes = {(right, strike, resolved_expiry): dict(value)
                  for (right, strike, _), value in _QUOTES.items()}
    context = {"ticker": "AAA", "strategy": "STR-THRU",
               "event_date": "2026-10-27" if event_date is None else event_date,
               "entry_date": "2026-10-08",
               "quote_date": "2026-10-08", "exit_date": "2026-10-27",
               "expiry": resolved_expiry, "strike": 100.0, "spot": 100.0,
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
    inputs = _inputs(gate=gate)
    values = assemble_native_values(inputs)
    facts = _facts(inputs, values)
    assert values["entry_cost_pct"] is not None
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
    # No quote_date means no computed dte_entry, so the captured primitive
    # survives: it must not be overwritten with None.
    assert values["dte_entry"] == 55


def _pricing_stand_ins(*, pricing_refusal=None, geometry_refusal=None,
                       pricing_legs=None, geometry_legs=None):
    leg = types.SimpleNamespace(bid=1.9, ask=2.1, strike=100.0,
                                expiry="2026-12-18")
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