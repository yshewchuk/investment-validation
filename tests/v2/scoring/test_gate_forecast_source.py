"""Slice A of the forward forecast assembly: ``GateForecastSource`` (issue #1104).

The gate block is BUILT, never executed: synthetic bindings, a synthetic
release and a path-only ``FrozenInference`` are enough, because every case
here is a build-time contract (members built, or ``ValueError`` at bundle
build). The champion ``model_release``/``frozen_inference`` are deliberately
absent from every bundle, so a resolved forecast executor can only have come
from the source itself.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from engine.v2.models.contracts import ArtifactMember, ModelBinding, ModelRelease
from engine.v2.models.loader import FrozenInference
from engine.v2.scoring.gate_forecast_source import GateForecastSource
from engine.v2.scoring.native_gate_features import gate_forecast_columns
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
        artifact_hash="a" * 64,
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
], ids=["blank", "missing"])
def test_source_binding_must_be_exactly_one_size_binding(tmp_path, case, overrides, message):
    with pytest.raises(ValueError, match=message):
        build_native_score_inputs(_bundle(tmp_path, _source(tmp_path, **overrides)))


def test_binding_of_a_non_size_role_is_refused(tmp_path):
    """The duplicated ``role == "size"`` check was removed: exactly-one stays
    here; a non-size role reaches the executor build, which refuses it by
    role (``_binding_output``)."""
    with pytest.raises(ValueError, match="of role 'gate'"):
        build_native_score_inputs(_bundle(tmp_path, _source(tmp_path, binding_id="b-gate")))


@pytest.mark.parametrize("case,overrides,message", [
    ("ragged", {"pool_pred": [1.0, 2.0], "pool_res": [0.5]}, "differ in length"),
    ("non_finite", {"pool_res": [0.5, float("nan"), 1.5]}, "must be finite"),
], ids=["ragged", "nan"])
def test_source_pool_must_be_ragged_free_and_finite(tmp_path, case, overrides, message):
    with pytest.raises(ValueError, match=message):
        build_native_score_inputs(_bundle(tmp_path, _source(tmp_path, **overrides)))


def test_source_builds_a_thin_pool_and_carries_the_executor_inference(tmp_path):
    """The Builds assertions: interval_floor rides into the pool, the
    executor's inference is the source's OWN object (identity), while a
    DIFFERENT champion ``frozen_inference`` sits on the bundle so a champion
    fallback would be caught."""
    source = _source(tmp_path, pool_pred=(1.0,), pool_res=(0.1,),
                     interval_floor=0.25)
    champion = FrozenInference(tmp_path)
    bundle = dataclasses.replace(_bundle(tmp_path, source),
                                 frozen_inference=champion)
    gate = build_native_score_inputs(bundle).gate
    executor = gate["forecast_executor"]
    assert gate["forecast_pool"]["interval_floor"] == 0.25
    assert executor._executor.inference is source.inference
    assert source.inference is not champion
    # Thin pool (one or two points) still builds without error: only the
    # band columns become unavailable at stage time (ARCHITECTURE.md).
    thin = _source(tmp_path, pool_pred=(1.0, 2.0), pool_res=(0.1, 0.2),
                   interval_floor=None)
    thin_gate = build_native_score_inputs(_bundle(tmp_path, thin)).gate
    thin_pool = thin_gate["forecast_pool"]
    assert thin_pool == {"predictions": (1.0, 2.0), "residuals": (0.1, 0.2),
                         "interval_floor": None}
    band = gate_forecast_columns(1.0, thin_pool, None)
    assert math.isnan(band["pred_abs_move_p10"])
    assert math.isnan(band["pred_abs_move_p90"])
    assert math.isnan(band["pred_abs_move_sd"])


def test_source_not_a_gate_forecast_source_instance(tmp_path):
    with pytest.raises(ValueError, match="must be a GateForecastSource"):
        build_native_score_inputs(
            _bundle(tmp_path, {"binding_id": "b-size-fold"}))


def test_release_not_a_model_release(tmp_path):
    source = _source(tmp_path, release="rel-not-a-release")
    with pytest.raises(ValueError, match="release must be a ModelRelease"):
        build_native_score_inputs(_bundle(tmp_path, source))


def test_none_inference_is_refused(tmp_path):
    source = _source(tmp_path, inference=None)
    with pytest.raises(ValueError, match="inference must be a FrozenInference"):
        build_native_score_inputs(_bundle(tmp_path, source))


def test_duplicate_binding_id_in_release_is_refused(tmp_path):
    release = ModelRelease(release_id="rel-dup", deployment_id="dep-1", bindings=(
        _binding("b-size-fold", "size"),
        _binding("b-size-fold", "size"),
    ))
    source = _source(tmp_path, release=release)
    with pytest.raises(ValueError, match="exactly one"):
        build_native_score_inputs(_bundle(tmp_path, source))


@pytest.mark.parametrize("bad,overrides", [
    (float("nan"), {"pool_pred": [1.0, float("nan"), 3.0]}),
    (float("nan"), {"pool_res": [0.1, float("nan"), 0.3]}),
    (float("inf"), {"pool_res": [0.1, float("inf"), 0.3]}),
], ids=["nan-pred", "nan-res", "inf-res"])
def test_non_finite_pool_values_are_refused(tmp_path, bad, overrides):
    with pytest.raises(ValueError, match="must be finite"):
        build_native_score_inputs(_bundle(tmp_path, _source(tmp_path, **overrides)))


@pytest.mark.parametrize("field", ["gate_forecast_pool", "forecast_pool"])
def test_non_finite_legacy_pool_values_are_refused(tmp_path, field):
    pool = {"predictions": [1.0, float("nan"), 3.0],
            "residuals": [0.1, 0.2, 0.3], "interval_floor": 0.0}
    if field == "gate_forecast_pool":
        bundle = _bundle(tmp_path, None, pool=pool)
    else:
        bundle = dataclasses.replace(_bundle(tmp_path, None), forecast_pool=pool)
    with pytest.raises(ValueError, match="must be finite"):
        build_native_score_inputs(bundle)


def test_non_finite_interval_floor_is_refused(tmp_path):
    source = _source(tmp_path, interval_floor=float("inf"))
    with pytest.raises(ValueError, match="interval_floor must be finite"):
        build_native_score_inputs(_bundle(tmp_path, source))


def test_absent_source_still_builds_a_gate_naming_a_forecast_column(tmp_path):
    """The ordinary path: a gate naming a forecast column with no source still
    builds the gate block (executor declared-but-unresolved here, since the
    bundle carries no champion release); the stage then refuses
    MISSING_GATE_INPUT:forecast at execution time (ARCHITECTURE.md)."""
    recipe = {"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.5,
              "forecast": {"binding_id": "b-size-fold"}}
    gate = build_native_score_inputs(
        _bundle(tmp_path, None, gate_recipe=recipe)).gate
    assert gate["forecast_executor"].binding_id == "b-size-fold"
    assert "forecast_pool" not in gate


def test_absent_source_leaves_the_gate_block_unchanged(tmp_path):
    gate = build_native_score_inputs(_bundle(tmp_path, None)).gate
    assert "forecast_executor" not in gate
    assert "forecast_recipe" not in gate
    assert "forecast_pool" not in gate
