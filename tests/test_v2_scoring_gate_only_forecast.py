"""Phase 4: a gate-only size forecast must not leak into the score output.

Legacy ``engine/score.py::_forecast_for_gate`` serves a size fold purely to
fill the gate's derived ``pred_abs_move`` feature columns; it never assigns
``result.forecast_abs_move``. The strict-targeted replay on corpus
c2ff6fd showed the native frozen path leaking exactly that: members 002/1
and 017/0 declare ``forecast.required_roles = ["driver"]`` only, yet the
gate-referenced size binding was collected into ``forecast_abs_move`` (and
therefore into ``structures.generate``'s width and ``cost_over_width``).
``application.score_frozen`` now withholds a size binding from the top-level
collected outputs, ``required_roles`` and the canonical forecast-executor
registration exactly when the source-owned gate names it and the top-level
forecast does not itself declare a size output -- while preserving an
explicitly declared (gate-shared) size forecast, the gate's own executor,
the verified release bindings, and the recipe-stripping/adapter protections.

The two frozen value paths are covered separately: the eager one (results
collected straight off ``inference.infer`` into ``frozen_outputs``) and the
full-release executor one (canonical executors re-serving through the
verified release at stage time).
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from checks.phase4_frozen_bridge import with_frozen_gate_forecast
from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import generate
from engine.v2.models import (
    ArtifactMember,
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
)
from engine.v2.scoring import application
from engine.v2.scoring.stages import STAGE_NAMES, NativeScoreInputs, receipt

X = 5.0                       # the one finite fold feature every artifact reads
SIZE_PREDICTION = X + 1.0     # linear: pred_abs_move = 1.0 + 1.0 * x -> 6.0
DRIVER_PREDICTION = X         # linear: pred_driver = 0.0 + 1.0 * x -> 5.0
GATE_PREDICTION = X + SIZE_PREDICTION  # gate_score = x + pred_abs_move -> 11.0
GATE_THRESHOLD = 10.0
NAN_TAG = {"__nonfinite__": "nan"}

CONTEXT = {"ticker": "AAA", "strategy": "STR-THRU", "event_date": "2026-09-16",
           "entry_date": "2026-09-16", "exit_date": "2026-09-16",
           "expiry": "2026-09-18", "spot": 100.0,
           # An explicit ATM straddle ladder: the row prices, so every late
           # stage (the gate above all) is legacy-reachable in the test.
           "quotes": {("C", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05},
                      ("P", 100.0, "2026-09-18"): {"bid": 1.95, "ask": 2.05}}}


def _linear(feature_order, name, intercept, coefficients):
    return {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": list(feature_order),
        "outputs": [{"name": name, "intercept": intercept,
                     "coefficients": list(coefficients)}],
    }


def _member(root, filename, document) -> ArtifactMember:
    path = root / filename
    path.write_text(json.dumps(document, sort_keys=True))
    return ArtifactMember(name="estimator", path=filename, content_hash="sha256:"
                          + hashlib.sha256(path.read_bytes()).hexdigest())


def _binding(binding_id, role, member, feature_order, output_names):
    return ModelBinding(
        binding_id=binding_id, model_id=f"{role}-linear", role=role,
        strategy_id="*", decision_clock_id="entry-close",
        adapter="json-linear.v1", feature_order=tuple(feature_order),
        output_names=tuple(output_names), members=(member,),
    )


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate_only_forecast")
    size = _binding("b-size", "size", _member(
        root, "size.json", _linear(("x",), "pred_abs_move", 1.0, (1.0,))),
        ("x",), ("pred_abs_move",))
    driver = _binding("b-driver", "driver", _member(
        root, "driver.json", _linear(("x",), "pred_driver", 0.0, (1.0,))),
        ("x",), ("pred_driver",))
    gate = _binding("b-gate", "gate", _member(
        root, "gate.json", _linear(("x", "pred_abs_move"), "gate_score",
                                   0.0, (1.0, 1.0))),
        ("x", "pred_abs_move"), ("gate_score",))
    release = ModelRelease(release_id="rel-gate-only", deployment_id="dep-1",
                           bindings=(driver, size, gate))
    return root, release


def _request(strategy="STR-THRU") -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-gate-only", calendar_revision="cal-1",
        strategy_version=strategy, deployment_id="dep-1",
        decision_clock_id="entry-close", requested_decision_at="2026-09-16",
        snapshot_id="snap-1", mode="replay", fill_model={"alpha": 0.5},
    )


def _inference_request(binding, rows):
    return InferenceRequest(release_id="rel-gate-only",
                            binding_id=binding.binding_id,
                            feature_order=binding.feature_order,
                            rows=tuple(rows))


def _receipts():
    return tuple(receipt(stage, {"source": "gate-only"}, {"execution": "pending"})
                 for stage in STAGE_NAMES if stage != "diagnostics")


def _base_inputs(*, context=None, features=None, forecast=None, gate=None,
                 **overrides) -> NativeScoreInputs:
    base = dict(
        context=dict(CONTEXT) if context is None else dict(context),
        features={"model_inputs": {"x": X}} if features is None else dict(features),
        forecast={"required_roles": ["driver"]} if forecast is None else dict(forecast),
        geometry=None, pricing=None, analogs={}, simulation={},
        gate=({"binding_id": "b-gate", "threshold": GATE_THRESHOLD,
               "forecast": {"binding_id": "b-size", "output": "pred_abs_move"}}
              if gate is None else dict(gate)),
        chooser={}, diagnostics={}, source_ref="gate-only-test",
        stage_receipts=_receipts(),
    )
    base.update(overrides)
    return NativeScoreInputs(**base)


def _binding_by_id(release, binding_id):
    return next(b for b in release.bindings if b.binding_id == binding_id)


def _frozen_gate(frozen, inputs):
    root, release = frozen
    return with_frozen_gate_forecast(
        inputs, release_root=root, release=release)


def _score_frozen(frozen, *, declared_size, binding_ids=("b-driver", "b-size", "b-gate"),
                  features=None):
    """One full-release-executor frozen run: the gate names b-size as its
    derived-forecast producer; ``declared_size`` toggles the top-level
    ``required_roles`` declaration -- the only difference between the pair."""
    root, release = frozen
    forecast = {"required_roles": ["driver", "size"] if declared_size else ["driver"]}
    base = _frozen_gate(frozen, _base_inputs(forecast=forecast, features=features))
    rows_by_binding = {"b-driver": ((X,),), "b-size": ((X,),),
                       "b-gate": ((X, SIZE_PREDICTION),)}
    requests = tuple(
        _inference_request(_binding_by_id(release, binding_id),
                           rows_by_binding[binding_id])
        for binding_id in binding_ids)
    record = application.score_frozen(
        _request(), FrozenInference(root), release, requests,
        {"_native_inputs": base})
    return record, base


# == gate-only ownership detection ==============================================


def test_gate_forecast_producer_read_from_both_declared_shapes(frozen):
    root, release = frozen
    json_shape = {"forecast": {"binding_id": "b-size", "output": "pred_abs_move"}}
    assert application._gate_forecast_producer_ids(json_shape) == frozenset({"b-size"})
    executor_shape = _frozen_gate(
        frozen, _base_inputs()).gate["forecast_executor"]
    assert application._gate_forecast_producer_ids(
        {"forecast_executor": executor_shape}) == frozenset({"b-size"})
    assert application._gate_forecast_producer_ids({"threshold": 0.5}) == frozenset()


@pytest.mark.parametrize("declared", [
    {"required_roles": ["driver", "size"]},
    {"required_roles": "size"},
    {"executors": {"forecast_abs_move": object()}},
    {"models": {"forecast_abs_move": {"intercept": 1.0, "coefficients": {}}}},
])
def test_declared_top_level_size_is_never_gate_only(frozen, declared):
    root, release = frozen
    gate = {"forecast": {"binding_id": "b-size", "output": "pred_abs_move"}}
    assert application._gate_only_forecast_ids(
        gate, declared, release.bindings) == frozenset()


def test_gate_reference_without_declaration_is_gate_only(frozen):
    root, release = frozen
    gate = {"forecast": {"binding_id": "b-size", "output": "pred_abs_move"}}
    ids = application._gate_only_forecast_ids(gate, {"required_roles": ["driver"]},
                                              release.bindings)
    assert ids == frozenset({"b-size"})
    # The driver binding is not a size producer and is never withheld.
    assert not application._is_gate_only_size(
        _binding_by_id(release, "b-driver"), ids)
    assert application._is_gate_only_size(_binding_by_id(release, "b-size"), ids)


# == eager collection (frozen_outputs) ==========================================


def _eager_result(binding, value, hashes=("sha256:member",)):
    return SimpleNamespace(
        binding_id=binding.binding_id, status="READY", model_id=binding.model_id,
        release_id="rel-gate-only", artifact_hashes=hashes, reason_codes=(),
        output_names=binding.output_names, predictions=((value,),))


def _collected(frozen, *, declared_size, size_value=SIZE_PREDICTION):
    root, release = frozen
    size = _binding_by_id(release, "b-size")
    driver = _binding_by_id(release, "b-driver")
    gate = _binding_by_id(release, "b-gate")
    results = (_eager_result(driver, DRIVER_PREDICTION),
               _eager_result(size, size_value),
               _eager_result(gate, GATE_PREDICTION))
    gate_only = application._gate_only_forecast_ids(
        {"forecast": {"binding_id": "b-size"}},
        {"required_roles": ["driver", "size"] if declared_size else ["driver"]},
        release.bindings)
    return application._collect_frozen_results(
        results, (driver, size, gate),
        tuple(SimpleNamespace(binding_id=b.binding_id)
              for b in (driver, size, gate)), None, gate_only)


def test_collected_gate_only_size_keeps_provenance_without_value_or_role(frozen):
    outputs, _state, _flags, hashes, roles, gate_result = _collected(frozen,
                                                                     declared_size=False)
    assert outputs == {"driver_prediction": DRIVER_PREDICTION}
    assert roles == ["driver"]
    # Provenance and the gate pairing survive -- only the value and role leave.
    assert list(hashes) == ["sha256:member"] * 3
    assert gate_result[1].binding_id == "b-gate"


def test_collected_gate_only_size_never_stamps_its_nonfinite_value(frozen):
    outputs, _s, _f, _h, roles, _g = _collected(frozen, declared_size=False,
                                                size_value=float("nan"))
    assert "forecast_abs_move" not in outputs  # not even a NaN to stamp
    assert "size" not in roles


def test_collected_declared_size_keeps_the_shared_producer(frozen):
    outputs, _s, _f, _h, roles, _g = _collected(frozen, declared_size=True)
    assert outputs["forecast_abs_move"] == SIZE_PREDICTION
    assert roles == ["driver", "size"]


# == executor registration and recipe stripping =================================


def test_gate_only_size_registers_no_forecast_executor(frozen):
    root, release = frozen
    base = _base_inputs(forecast={
        "required_roles": ["driver"],
        "models": {"forecast_abs_move": {"intercept": 9.9, "coefficients": {}},
                   "pred_iv_crush": {"intercept": 0.0, "coefficients": {}}},
    })
    forecast = application._frozen_forecast_inputs(
        base, release.bindings, {"driver_prediction": DRIVER_PREDICTION},
        (), ["driver"], (), FrozenInference(root), release, None, "STR-THRU",
        release.bindings, frozenset({"b-size"}))
    assert "forecast_abs_move" not in forecast["executors"]
    assert "driver_prediction" in forecast["executors"]
    # The recipe-stripping protection keeps keying on the FULL verified
    # release: even a withheld binding owns its canonical output, so the
    # caller-declared local ``forecast_abs_move`` recipe stays stripped --
    # and an unowned recipe is untouched.
    assert set(forecast["models"]) == {"pred_iv_crush"}


def _score_frozen_with(frozen, record_inputs):
    inputs, _interval = record_inputs
    return application.score_one(_request(), inputs)


def _eager_native(frozen, *, declared_size, features=None):
    """The eager inference path: ``_frozen_native_inputs`` with ``inference``
    None -- values reach the stage only through ``frozen_outputs``."""
    root, release = frozen
    forecast = {"required_roles": ["driver", "size"] if declared_size else ["driver"]}
    base = _frozen_gate(frozen, _base_inputs(forecast=forecast, features=features,
                                             gate=_MODEL_GATE))
    size = _binding_by_id(release, "b-size")
    driver = _binding_by_id(release, "b-driver")
    results = (_eager_result(driver, DRIVER_PREDICTION, ("sha256:driver",)),
               _eager_result(size, SIZE_PREDICTION, ("sha256:size",)))
    requests = tuple(SimpleNamespace(binding_id=b.binding_id)
                     for b in (driver, size))
    return application._frozen_native_inputs(
        {"_native_inputs": base}, results, (driver, size), requests,
        _request(), release, None)


_MODEL_GATE = {
    "model": {"intercept": 0.0, "coefficients": {"x": 1.0, "pred_abs_move": 1.0}},
    "threshold": GATE_THRESHOLD,
    "forecast": {"binding_id": "b-size", "output": "pred_abs_move"},
}


# == end to end: eager inference path ===========================================


def test_eager_gate_only_size_scores_gate_without_top_level_forecast(frozen):
    inputs, _interval = _eager_native(frozen, declared_size=False)
    assert inputs.forecast["required_roles"] == ("driver",)
    assert "executors" not in inputs.forecast
    assert inputs.forecast["frozen_outputs"] == {"driver_prediction": DRIVER_PREDICTION}
    record = _score_frozen_with(frozen, (inputs, _interval))
    assert record.forecasts["forecast_abs_move"] is None
    assert record.gate_terms["gate_score"] == GATE_PREDICTION
    assert record.gate_terms["gate_pass"] is True
    assert record.financial_diagnostics.get("cost_over_width") is None
    assert not any(code.startswith(("MISSING_FORECAST", "NONFINITE_FORECAST",
                                    "UNKNOWN_FROZEN"))
                   for code in record.reason_codes)


def test_eager_declared_size_keeps_the_shared_producer_forecast(frozen):
    inputs, _interval = _eager_native(frozen, declared_size=True)
    record = _score_frozen_with(frozen, (inputs, _interval))
    assert inputs.forecast["required_roles"] == ("driver", "size")
    assert record.forecasts["forecast_abs_move"] == SIZE_PREDICTION
    assert record.gate_terms["gate_score"] == GATE_PREDICTION
    # Control for the null above: with the same quotes and fold, a DECLARED
    # size forecast flows into the generated width and its cost ratio.
    assert record.financial_diagnostics.get("cost_over_width") is not None


# == end to end: full release executor path =====================================


def test_frozen_gate_only_size_leaks_no_forecast_or_width(frozen):
    record, _base = _score_frozen(frozen, declared_size=False)
    assert record.forecasts["driver_prediction"] == DRIVER_PREDICTION
    assert record.forecasts["forecast_abs_move"] is None
    for name in ("forecast_p10", "forecast_p90", "forecast_sd"):
        assert record.uncertainty[name] is None
    # The withheld fold still feeds the gate: x + pred_abs_move = 11.0.
    assert record.gate_terms["gate_score"] == GATE_PREDICTION
    assert record.financial_diagnostics.get("cost_over_width") is None
    assert not any(code.startswith(("MISSING_FORECAST_OUTPUT:size",
                                    "NONFINITE_FORECAST_OUTPUT",
                                    "UNKNOWN_FROZEN", "INVALID_GATE"))
                   for code in record.reason_codes)


def test_frozen_declared_size_keeps_the_shared_producer_forecast(frozen):
    record, _base = _score_frozen(frozen, declared_size=True)
    assert record.forecasts["forecast_abs_move"] == SIZE_PREDICTION
    assert record.gate_terms["gate_score"] == GATE_PREDICTION
    assert record.financial_diagnostics.get("cost_over_width") is not None


def test_frozen_nonfinite_gate_only_fold_stays_undetermined(frozen):
    """The captured fold row reads non-finite (the bridge would omit the
    request; here it is simply not among the requested bindings). The gate
    declines MISSING_FEATURES exactly as legacy's fold NaN does -- and the
    top level neither publishes ``forecast_abs_move`` nor stamps
    ``MISSING_FORECAST_OUTPUT:size`` for a role it never declared."""
    root, release = frozen
    features = {"model_inputs": {"x": X},
                "role_model_inputs": {"size": {"x": NAN_TAG}}}
    record, _base = _score_frozen(
        frozen, declared_size=False, features=features,
        binding_ids=("b-driver", "b-gate"))
    assert record.forecasts["forecast_abs_move"] is None
    assert record.gate_terms["gate_score"] is None
    assert "MISSING_FEATURES" in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:size" not in record.reason_codes
    assert "NONFINITE_FORECAST_OUTPUT:forecast_abs_move" not in record.reason_codes
    assert record.forecasts["driver_prediction"] == DRIVER_PREDICTION


# == an actually sized strategy keeps forecast/band/sizing/simulation ============


def _sized_pool():
    rng = np.random.default_rng(7)
    pred = rng.uniform(1.0, 14.0, 400)
    return pred, rng.normal(0.0, 2.0, 400)


def test_frozen_sized_strategy_retains_forecast_band_sizing_simulation(frozen):
    root, release = frozen
    size = _binding_by_id(release, "b-size")
    predictions, residuals = _sized_pool()
    context = dict(CONTEXT, strategy="TWIN-P", exit_date="2026-09-18")
    legs = generate("TWIN-P", {"spot": 100.0, "expiry": "2026-09-18",
                               "forecast_abs_move": SIZE_PREDICTION}).legs
    # Strike-sorted distinct mids: a symmetric quote ladder would net the
    # put-ladder entry cost to zero (INVALID_SIMULATION_CAPITAL).
    mids = {leg.strike: 0.2 + 0.18 * (leg.strike - 84.0) for leg in legs}
    context["quotes"] = {
        (leg.right, leg.strike, leg.expiry): {
            "bid": mids[leg.strike] - 0.05, "ask": mids[leg.strike] + 0.05}
        for leg in legs}
    base = _base_inputs(
        context=context, forecast={
            "required_roles": ["size"],
            "forecast_pool": {"predictions": tuple(predictions),
                              "residuals": tuple(residuals),
                              "interval_floor": 0.0}},
        gate={"mode": "not_applicable"},
        simulation={"terminal_spots": (90.0, 100.0, 112.0)},
    )
    request = _request("TWIN-P")
    record = application.score_frozen(
        request, FrozenInference(root), release,
        (_inference_request(size, ((X,),)),), {"_native_inputs": base})
    assert record.forecasts["forecast_abs_move"] == SIZE_PREDICTION
    for name in ("forecast_p10", "forecast_p90", "forecast_sd"):
        assert record.uncertainty[name] is not None
    # Sizing: the served forecast still drives the generated width/entry cost.
    assert record.legs and record.selected_contracts
    assert record.financial_diagnostics.get("cost_over_width") is not None
    # Simulation: the planned terminal-spot layer ran off the priced legs.
    assert record.forecasts["exp_pnl_sim"] is not None, record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:size" not in record.reason_codes
