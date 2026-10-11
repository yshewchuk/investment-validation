"""Scorer-computed date features reach the gate and stay out of the source
bundle: ``days_to_print`` (legacy ``engine/features.py:1166-1168``) and
``dte_entry`` (legacy ``engine/score.py:2451``: ``dte_entry = int(priced.legs[0].dte)``).
"""
import hashlib
import json

import pandas as pd

from engine.v2.models import FrozenInference, ModelBinding, ModelRelease
from engine.v2.models.contracts import ArtifactMember
from engine.v2.scoring import financial
from engine.v2.scoring.frozen_executor import FrozenStageExecutor
from engine.v2.scoring.nightly_source_bundle import assemble_nightly_source_bundle
from engine.v2.scoring.stages import (
    NativeScoreInputs, STAGE_NAMES, StageReceipt, assemble_native_values)

_RECEIPTS = tuple(
    StageReceipt(s, "declared-input", "declared-output")
    for s in STAGE_NAMES if s != "diagnostics")
_QUOTES = {r: {"bid": 1.9, "ask": 2.1}
           for r in (("C", 100.0, "2026-09-18"), ("P", 100.0, "2026-09-18"))}


def _inputs(*, gate=None) -> NativeScoreInputs:
    return NativeScoreInputs(
        context={"ticker": "AAA", "strategy": "STR-THRU",
                 "event_date": "2026-10-27", "entry_date": "2026-10-08",
                 "quote_date": "2026-10-08", "exit_date": "2026-09-18",
                 "expiry": "2026-11-20", "strike": 100.0, "spot": 100.0,
                 "quotes": dict(_QUOTES)},
        features={"model_inputs": {}},
        forecast={"driver_name": "abs_move",
                  "models": {"driver_prediction": {"intercept": 0.0, "coefficients": {}}}},
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
    assert values.get("gate_score") is not None
    assert values["days_to_print"] == 19.0
    assert values["dte_entry"] == 43


def test_dates_not_projected_into_source_mask():
    bundle = assemble_nightly_source_bundle(**_kwargs())
    for name in ("days_to_print", "dte_entry", "entry_cost_pct"):
        assert name not in bundle.feature_vector
        assert name not in bundle.feature_missing_mask
    assert "signal" in bundle.feature_vector


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