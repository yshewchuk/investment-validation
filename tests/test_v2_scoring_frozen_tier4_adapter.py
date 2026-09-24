"""The frozen-path stage executors must carry the verified binding's adapter.

``stages._execute_forecast_executor`` already treats a ``MISSING_FEATURES``
refusal from a Tier-4 fold (adapter ``tier4-serving-fold.v1``) as a SILENT
undetermined forecast -- legacy ``tier4.ServingModel.predict`` answers NaN and
sizing then declines ``NO_FORECAST`` -- while the same refusal from a champion
model stays an actionable ``MISSING_FEATURES``. It decides between the two from
``getattr(executor, "adapter", None)``.

The Phase 4 frozen bridge builds its forecast executors in
``application._frozen_stage_executors`` out of two wrappers --
``_CanonicalFrozenExecutor`` for a mapped artifact output (a size fold's
``pred_abs_move`` -> ``forecast_abs_move``) and ``_FrozenOmissionRefusal`` for a
still-unmapped canonical output of an omitted binding. Neither wrapper used to
forward the binding's ``adapter``, so a Tier-4 fold served through the frozen
path read as ``getattr(..., "adapter") is None`` and its omission was stamped
champion ``MISSING_FEATURES`` -- exactly the fixture-016 (RAMP7/HAIN) parity gap
where legacy declined ``NO_FORECAST`` but native-only refused ``MISSING_FEATURES``.
The wrappers now preserve ``binding.adapter`` (read generically, failing closed
to ``None``). These tests pin the wiring at both the construction site and the
stage that consumes it, and guard that an ordinary champion ``MISSING_FEATURES``
refusal is untouched.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.v2.models import ArtifactMember, ModelBinding
from engine.v2.scoring import application
from engine.v2.scoring.stages import (
    TIER4_FOLD_ADAPTER,
    _execute_forecast_executor,
    _forecast_sizing_declines,
)

CHAMPION_ADAPTER = "joblib-estimator.v1"
NONFINITE_NAN = {"__nonfinite__": "nan"}


def _binding(binding_id, role, adapter, output_names,
             feature_order=("or_implied", "iv30")):
    return ModelBinding(
        binding_id=binding_id, model_id=f"{role}_synthetic", role=role,
        strategy_id="*", decision_clock_id="legacy.decision_offset.0",
        adapter=adapter, feature_order=tuple(feature_order),
        output_names=tuple(output_names),
        members=(ArtifactMember(name="estimator", path=f"{binding_id}.joblib",
                                content_hash="sha256:deadbeef"),),
    )


def _release(*bindings):
    return SimpleNamespace(release_id="rel-t4", bindings=tuple(bindings))


def _captured_row(role, order=("or_implied", "iv30")):
    vector = {name: 1.0 for name in order}
    vector["or_implied"] = NONFINITE_NAN
    return {"role_model_inputs": {role: vector}}


def _executors(binding):
    # The bridge omitted this binding (empty served set) for a non-finite
    # captured row, so the mapped fold keeps its canonical executor and any
    # still-unmapped canonical output registers the refusal-only fallback.
    executors, owned = application._frozen_stage_executors(
        (binding,), None, _release(binding), 14.0,
        served_ids=frozenset(), features=_captured_row(str(binding.role)),
    )
    return executors, owned


def _run(executor, field, facts):
    flags, output, invalid, undetermined = [], {}, set(), set()
    _execute_forecast_executor(executor, field, facts, output, invalid,
                               flags, undetermined)
    return flags, output, invalid, undetermined


def test_mapped_tier4_fold_executor_preserves_binding_adapter():
    binding = _binding("b-size", "size", TIER4_FOLD_ADAPTER, ("pred_abs_move",))
    executors, owned = _executors(binding)
    executor = executors["forecast_abs_move"]
    assert isinstance(executor, application._CanonicalFrozenExecutor)
    assert executor.adapter == TIER4_FOLD_ADAPTER
    assert executor.role == "size"
    assert "forecast_abs_move" in owned


def test_unmapped_tier4_omission_refusal_preserves_binding_adapter():
    # Two artifact outputs where only pred_iv_crush_30 maps to a canonical
    # output: pred_iv_crush stays unmapped and gets the refusal-only fallback.
    binding = _binding("b-crush", "iv_crush", TIER4_FOLD_ADAPTER,
                       ("pred_iv_crush_30", "pred_iv_crush_extra"))
    executors, _ = _executors(binding)
    refusal = executors["pred_iv_crush"]
    assert isinstance(refusal, application._FrozenOmissionRefusal)
    assert refusal.adapter == TIER4_FOLD_ADAPTER


def test_mapped_tier4_fold_refusal_undetermines_not_missing_features():
    binding = _binding("b-size", "size", TIER4_FOLD_ADAPTER, ("pred_abs_move",))
    executor = _executors(binding)[0]["forecast_abs_move"]
    facts = {"or_implied": NONFINITE_NAN, "iv30": 1.0}
    flags, output, invalid, undetermined = _run(
        executor, "forecast_abs_move", facts)
    assert undetermined == {"forecast_abs_move"}
    assert invalid == {"forecast_abs_move"}
    assert "MISSING_FEATURES" not in flags
    assert output.get("forecast_abs_move") is None
    # The undetermined fold is exactly what makes sizing decline NO_FORECAST.
    declined = _forecast_sizing_declines(
        SimpleNamespace(geometry=None), {}, "TWIN-P", {}, undetermined)
    assert declined is True


def test_unmapped_tier4_refusal_undetermines_not_missing_features():
    binding = _binding("b-crush", "iv_crush", TIER4_FOLD_ADAPTER,
                       ("pred_iv_crush_30", "pred_iv_crush_extra"))
    refusal = _executors(binding)[0]["pred_iv_crush"]
    flags, output, invalid, undetermined = _run(
        refusal, "pred_iv_crush", {"or_implied": NONFINITE_NAN, "iv30": 1.0})
    assert undetermined == {"pred_iv_crush"}
    assert "MISSING_FEATURES" not in flags
    assert output.get("pred_iv_crush") is None


@pytest.mark.parametrize("adapter", [CHAMPION_ADAPTER])
def test_champion_mapped_refusal_still_flags_missing_features(adapter):
    binding = _binding("b-size", "size", adapter, ("pred_abs_move",))
    executor = _executors(binding)[0]["forecast_abs_move"]
    assert executor.adapter == adapter
    flags, output, invalid, undetermined = _run(
        executor, "forecast_abs_move", {"or_implied": NONFINITE_NAN, "iv30": 1.0})
    assert "MISSING_FEATURES" in flags
    assert undetermined == set()


def test_champion_unmapped_refusal_still_flags_missing_features():
    binding = _binding("b-crush", "iv_crush", CHAMPION_ADAPTER,
                       ("pred_iv_crush_30", "pred_iv_crush_extra"))
    refusal = _executors(binding)[0]["pred_iv_crush"]
    assert refusal.adapter == CHAMPION_ADAPTER
    flags, output, invalid, undetermined = _run(
        refusal, "pred_iv_crush", {"or_implied": NONFINITE_NAN, "iv30": 1.0})
    assert "MISSING_FEATURES" in flags
    assert undetermined == set()


def test_binding_without_adapter_fails_closed_to_champion():
    """A synthetic binding (as ``_frozen_binding`` builds for an unresolved
    release) carries no ``adapter``: the generic read fails closed to ``None``,
    so a Tier-4-looking omission is NOT silently suppressed -- MISSING_FEATURES
    is preserved."""
    binding = SimpleNamespace(
        binding_id="b-size", role="size",
        feature_order=("or_implied", "iv30"),
        output_names=("pred_abs_move",),
    )
    executors, _ = application._frozen_stage_executors(
        (binding,), None, _release(binding), 14.0,
        served_ids=frozenset(), features=_captured_row("size"),
    )
    executor = executors["forecast_abs_move"]
    assert executor.adapter is None
    flags, output, invalid, undetermined = _run(
        executor, "forecast_abs_move", {"or_implied": NONFINITE_NAN, "iv30": 1.0})
    assert "MISSING_FEATURES" in flags
    assert undetermined == set()
