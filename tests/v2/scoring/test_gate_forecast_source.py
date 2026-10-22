"""Slice A of the forward forecast assembly: ``GateForecastSource`` (issue #1104).

The gate block is BUILT, never executed: synthetic bindings, a synthetic
release and a path-only ``FrozenInference`` are enough, because every case
here is a build-time contract (members built, or ``ValueError`` at bundle
build). The champion ``model_release``/``frozen_inference`` are deliberately
absent from every bundle, so a resolved forecast executor can only have come
from the source itself.
"""
from __future__ import annotations

import pytest

from engine.v2.models.contracts import ArtifactMember, ModelBinding, ModelRelease
from engine.v2.models.loader import FrozenInference
from engine.v2.scoring.gate_forecast_source import GateForecastSource
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs

EVENT = "2026-09-16"
CONTEXT = {"ticker": "AAA", "event_date": EVENT, "entry_date": EVENT,
           "exit_date": EVENT, "expiry": "2026-09-18", "spot": 100.0}
QUOTES = {("C", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
          ("P", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05}}
ROW = {"mean_prior_abs_move": 6.2, "iv30": 55.0}
PRED = (1.0, 2.0, 3.0)
RES = (0.25, -0.5, 1.25)


def _binding(binding_id: str, role: str, output: str = "pred_abs_move") -> ModelBinding:
    return ModelBinding(
        binding_id=binding_id, model_id=f"{role}_synthetic", role=role,
        strategy_id="*", decision_clock_id="legacy.entry_close.v1",
        adapter="joblib-estimator.v1", feature_order=("iv30",),
        output_names=(output,),
        members=(ArtifactMember(name="estimator", path=f"{role}.joblib",
                                 content_hash="sha256:synthetic"),))


def _release() -> ModelRelease:
    return ModelRelease(release_id="rel-slice-a", deployment_id="dep-1", bindings=(
        _binding("b-size-fold", "size"),
        _binding("b-gate", "gate", "gate_score"),
    ))


def _source(tmp_path, **overrides) -> GateForecastSource:
    values: dict = dict(
        binding_id="b-size-fold", release=_release(),
        inference=FrozenInference(tmp_path),
        pool_pred=list(PRED), pool_res=list(RES), interval_floor=0.0,
    )
    values.update(overrides)
    return GateForecastSource(**values)


def _bundle(tmp_path, source, *, gate_recipe=None, pool=None) -> SourceBundle:
    """A minimal STR-THRU bundle. ``source=None`` is the pre-slice-A bundle;
    the champion release/inference are never supplied (None below)."""
    return SourceBundle(
        source_ref="slice-a", strategy="STR-THRU", context=dict(CONTEXT),
        raw_quotes=QUOTES, feature_vector=dict(ROW),
        feature_missing_mask={name: False for name in ROW},
        model_identity={"driver": {"model_id": "synthetic"}},
        forecast_recipes={"driver_prediction": {"intercept": 6.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:driver"},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}},
                     "threshold": 0.5} if gate_recipe is None else gate_recipe,
        gate_forecast_pool={} if pool is None else pool,
        gate_forecast_source=source,
        frozen_inference=None, model_release=None,
    )


def test_source_builds_forecast_executor_and_pool_without_the_champion_release(tmp_path):
    gate = build_native_score_inputs(_bundle(tmp_path, _source(tmp_path))).gate
    executor = gate["forecast_executor"]
    assert executor.binding_id == "b-size-fold"
    # Resolved (role/adapter come from the resolved binding only when the
    # source's own release and inference are used -- the bundle's are None) ...
    assert executor.role == "size"
    assert executor.adapter == "joblib-estimator.v1"
    assert executor.feature_order == ("iv30",)
    assert "rel-slice-a" in str(executor)
    assert gate["forecast_recipe"] == str(executor)
    # ... and the source's pool is the gate's pool, coerced to tuples.
    assert gate["forecast_pool"] == {"predictions": PRED, "residuals": RES,
                                     "interval_floor": 0.0}
    assert "forecast" not in gate


def test_source_with_gate_recipe_forecast_is_refused(tmp_path):
    recipe = {"binding_id": "b-gate", "threshold": 0.5,
              "forecast": {"binding_id": "b-size-fold"}}
    with pytest.raises(ValueError, match="cannot combine with gate_recipe.forecast"):
        build_native_score_inputs(_bundle(tmp_path, _source(tmp_path), gate_recipe=recipe))


def test_source_with_a_declared_gate_forecast_pool_is_refused(tmp_path):
    pool = {"predictions": (1.0,), "residuals": (0.1,), "interval_floor": 0.0}
    with pytest.raises(ValueError, match="cannot combine with gate_forecast_pool"):
        build_native_score_inputs(_bundle(tmp_path, _source(tmp_path), pool=pool))


@pytest.mark.parametrize("case,overrides,message", [
    ("blank_binding_id", {"binding_id": ""}, "binding_id must be non-empty"),
    ("binding_absent_from_release", {"binding_id": "b-absent"}, "exactly one"),
    ("binding_of_role_gate", {"binding_id": "b-gate"}, "exactly one"),
], ids=["blank", "missing", "wrong-role"])
def test_source_binding_must_be_exactly_one_size_binding(tmp_path, case, overrides, message):
    with pytest.raises(ValueError, match=message):
        build_native_score_inputs(_bundle(tmp_path, _source(tmp_path, **overrides)))


@pytest.mark.parametrize("case,overrides,message", [
    ("ragged", {"pool_pred": [1.0, 2.0], "pool_res": [0.5]}, "differ in length"),
    ("non_finite", {"pool_res": [0.5, float("nan"), 1.5]}, "must be finite"),
], ids=["ragged", "nan"])
def test_source_pool_must_be_ragged_free_and_finite(tmp_path, case, overrides, message):
    with pytest.raises(ValueError, match=message):
        build_native_score_inputs(_bundle(tmp_path, _source(tmp_path, **overrides)))


def test_absent_source_leaves_the_gate_block_unchanged(tmp_path):
    gate = build_native_score_inputs(_bundle(tmp_path, None)).gate
    assert "forecast_executor" not in gate
    assert "forecast_recipe" not in gate
    assert "forecast_pool" not in gate
