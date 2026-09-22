"""Regression coverage for ``checks.phase4_real._frozen_model_control``.

``_frozen_model_control`` feeds the ``frozen_model_stage`` value in the
Phase 4 evidence. It has returned False on every run since 2026-09-17:
commit aa97f53 ("Require strategy-specific forecast outputs") set
``_STRATEGY_FORECAST_ROLES["STR-THRU"] = ("driver",)``
(``engine/v2/scoring/stages.py``), but the control's synthetic
``ModelRelease``/``ModelBinding`` only ever bound a ``forecast`` role
(``forecast_abs_move``, ``pred_iv_crush``), never a ``driver`` role
(``driver_prediction``). Scoring therefore refused every run with
``MISSING_FORECAST_OUTPUT:driver`` and the control silently returned False
-- and because ``frozen_model_stage`` is not one of ``final_controls``,
nothing else noticed.

The fix binds a second ``driver``-role artifact (``pred_driver_thru`` ->
``driver_prediction``), matching what a real STR-THRU release binds today,
plus a handful of synthetic payoff-model rows so the record clears the
separate NO_SCORE guard (``application.py``'s
``_record_payload``: "neither layer produced a number") that would
otherwise still refuse it once the driver refusal is gone.

Covers:
- the control returns True with the driver binding present (the positive:
  frozen inference genuinely works end to end for STR-THRU today).
- the control returns False with the driver binding removed (the negative:
  the control can still catch the exact defect that broke it, so it is not
  a check that can never fail).
- the specific refusal reason on the negative path is
  ``MISSING_FORECAST_OUTPUT:driver`` -- pinning the mechanism, not just the
  boolean, so a future change that breaks the control for an unrelated
  reason is distinguishable from this one.
"""
from __future__ import annotations

from checks.phase4_real import _frozen_model_control, _request
from engine.v2.models import FrozenInference, InferenceRequest, ModelRelease
from engine.v2.scoring import application


def test_frozen_model_control_passes_with_the_driver_binding():
    assert _frozen_model_control(_request()) is True


def test_frozen_model_control_fails_without_the_driver_binding():
    """The negative control: removing the ``driver``-role binding must make
    the control fail again, the way it did on every run from 2026-09-17
    (aa97f53) until the driver binding was added. A control that returns
    True regardless of what it is fed is worthless."""
    assert _frozen_model_control(_request(), bind_driver=False) is False


def test_frozen_model_control_negative_reason_is_the_documented_one():
    """Pin the actual refusal mechanism (not just the boolean) for the
    no-driver-binding case, so a future unrelated regression that also
    returns False here is not mistaken for this one."""
    import hashlib
    import json
    import tempfile
    from pathlib import Path

    from engine.v2.domain.generation import generate, price
    from engine.v2.models import ArtifactMember, ModelBinding
    from engine.v2.scoring.stages import NativeScoreInputs, receipt

    request = _request()
    with tempfile.TemporaryDirectory(prefix="phase4-frozen-test-") as root:
        directory = Path(root)
        payload = json.dumps({
            "schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
            "outputs": [
                {"name": "forecast_abs_move", "intercept": 0.0, "coefficients": [1.0]},
                {"name": "pred_iv_crush", "intercept": -20.0, "coefficients": [0.0]},
            ],
        }, sort_keys=True).encode()
        (directory / "estimator.json").write_bytes(payload)
        member = ArtifactMember(
            name="estimator", path="estimator.json",
            content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        )
        binding = ModelBinding(
            binding_id="forecast", model_id="forecast-v1", role="forecast", strategy_id="*",
            decision_clock_id=request.decision_clock_id, adapter="json-linear.v1",
            feature_order=("x",), output_names=("forecast_abs_move", "pred_iv_crush"),
            members=(member,),
        )
        release = ModelRelease(release_id="release-v1", deployment_id=request.deployment_id,
                               bindings=(binding,))
        inference_request = InferenceRequest(
            release_id="release-v1", binding_id="forecast",
            feature_order=("x",), rows=((0.42,),))
        context = {
            "ticker": "PHASE4", "strategy": "STR-THRU",
            "event_date": "2026-09-16", "entry_date": "2026-09-16",
            "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0,
        }
        geometry = generate("STR-THRU", {**context, "forecast_abs_move": 0.42})
        quotes = {
            (leg.right, leg.strike, leg.expiry): {"bid": 1.0, "ask": 3.0}
            for leg in geometry.legs
        }
        pricing = price(geometry, quotes, 0.5)
        receipts = tuple(
            receipt(stage, "frozen-control-test", {})
            for stage in (
                "resolve_context", "features", "forecast", "geometry",
                "pricing", "model", "analogs", "simulation", "gate", "chooser",
                "serialization",
            )
        )
        native_inputs = NativeScoreInputs(
            context=context,
            features={"model_inputs": {"x": 0.42}},
            forecast={},
            geometry=geometry,
            pricing=pricing,
            analogs={},
            simulation={"terminal_spots": (100.0, 110.0)},
            gate={
                "model": {"intercept": 0.0, "coefficients": {"exp_pnl_sim": 1.0}},
                "threshold": 0.0,
            },
            chooser={},
            diagnostics={},
            source_ref="frozen-control-test",
            stage_receipts=receipts,
        )
        record = application.score_frozen(
            request, FrozenInference(directory), release, inference_request,
            {"_native_inputs": native_inputs},
        )
    assert record.validation_status == "refused"
    assert "MISSING_FORECAST_OUTPUT:driver" in record.reason_codes
