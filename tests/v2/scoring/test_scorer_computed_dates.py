"""Scorer-computed date features reach the gate and stay out of the source
bundle: ``days_to_print`` (legacy ``engine/features.py:1166-1168``) and
``dte_entry`` (legacy ``engine/score.py:2451``: ``dte_entry = int(priced.legs[0].dte)``).
"""
import hashlib
import json
from dataclasses import replace

import pandas as pd

from engine.v2.models import FrozenInference, ModelBinding, ModelRelease
from engine.v2.models.contracts import ArtifactMember
from engine.v2.scoring import financial
from engine.v2.scoring.frozen_executor import FrozenStageExecutor
from engine.v2.scoring.nightly_source_bundle import assemble_nightly_source_bundle
from engine.v2.scoring.source_inputs import build_native_score_inputs
from engine.v2.scoring.stages import (
    NativeScoreInputs, STAGE_NAMES, StageReceipt, _facts,
    assemble_native_values)

_RECEIPTS = tuple(
    StageReceipt(s, "declared-input", "declared-output")
    for s in STAGE_NAMES if s != "diagnostics")
_QUOTES = {r: {"bid": 1.9, "ask": 2.1}
           for r in (("C", 100.0, "2026-11-20"), ("P", 100.0, "2026-11-20"))}


def _inputs(*, gate=None, quotes=None, forecast_coefficients=None,
            drop_decision=False) -> NativeScoreInputs:
    context = {"ticker": "AAA", "strategy": "STR-THRU",
               "event_date": "2026-10-27", "entry_date": "2026-10-08",
               "quote_date": "2026-10-08", "exit_date": "2026-10-27",
               "expiry": "2026-11-20", "strike": 100.0, "spot": 100.0,
               "as_of": "2026-10-07",
               "quotes": dict(_QUOTES) if quotes is None else quotes}
    if drop_decision:
        del context["as_of"]
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
        chooser={}, diagnostics={},
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


def test_dte_entry_not_written_when_pricing_refuses():
    values = assemble_native_values(_inputs(quotes={}))
    assert "dte_entry" not in values
    assert values.get("entry_cost") is None


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