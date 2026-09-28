"""Issue #93: ``score_frozen``'s release bindings are scoped to the request.

A verified ``ModelRelease`` may carry the same canonical role for several
strategies (a STR-RUNUP ``implied_t1`` alongside STR-THRU's own ``driver``,
or one gate per strategy). Before the fix, ``score_frozen`` handed every
release binding to ``_frozen_native_inputs``, so the last binding in release
order silently decided the request's ``driver_prediction``/gate score for
every strategy. ``application._frozen_scoped_bindings`` now keeps only the
request's own ``(strategy_id, decision_clock_id)`` bindings plus shared
``strategy_id="*"`` ones, and refuses a same-scope collision with
``application.FrozenBindingConflict``.

Fixtures follow ``tests/test_v2_scoring_gate_only_forecast.py`` (hash-checked
JSON-linear artifacts under ``tmp_path``, ``FrozenInference``); the scored
regression case builds its native inputs through
``tests/test_v2_scoring_frozen_batch.py``'s ``_bundle``/
``build_native_score_inputs`` pattern.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.models import (
    ArtifactMember,
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
)
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import build_native_score_inputs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v2_scoring_native_payoff import (  # noqa: E402
    _GOOD_PAYOFF_ROWS,
    _bundle,
)

X = 5.0                    # the one finite feature every artifact reads
CLOCK = "entry-close"
STRATEGY = "STR-THRU"
OTHER_STRATEGY = "STR-RUNUP"


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


def _binding(root, binding_id, role, strategy_id, *, output, intercept):
    member = _member(root, f"{binding_id}.json",
                     _linear(("x",), output, intercept, (0.0,)))
    return ModelBinding(
        binding_id=binding_id, model_id=f"{binding_id}-linear", role=role,
        strategy_id=strategy_id, decision_clock_id=CLOCK,
        adapter="json-linear.v1", feature_order=("x",),
        output_names=(output,), members=(member,),
    )


def _release(bindings) -> ModelRelease:
    return ModelRelease(release_id="rel-scope", deployment_id="dep-1",
                        bindings=tuple(bindings))


def _request(strategy=STRATEGY) -> ScoreRequest:
    return ScoreRequest(
        event_id="evt-scope", calendar_revision="cal-1",
        strategy_version=strategy, deployment_id="dep-1",
        decision_clock_id=CLOCK, requested_decision_at="2026-09-16",
        snapshot_id="snap-1", mode="replay", fill_model={"alpha": 0.5},
    )


def _inference_request(release, binding) -> InferenceRequest:
    return InferenceRequest(release_id=release.release_id,
                            binding_id=binding.binding_id,
                            feature_order=binding.feature_order, rows=((X,),))


def _native_inputs(**overrides):
    return build_native_score_inputs(_bundle(
        feature_vector={"x": X}, feature_missing_mask={"x": False},
        **overrides))


def _score(root, release, bindings, *, inputs=None, requests=None):
    if requests is None:
        requests = tuple(_inference_request(release, binding)
                         for binding in bindings)
    return application.score_frozen(
        _request(), FrozenInference(root), release, requests,
        {"_native_inputs": inputs if inputs is not None else _native_inputs()})


# == forecast target scoping ====================================================


def test_forecast_target_scoped_to_request_strategy_both_orders(tmp_path):
    """The STR-RUNUP ``implied_t1`` binding never answers a STR-THRU
    request's ``driver_prediction``, whichever release order it holds."""
    driver = _binding(tmp_path, "b-driver", "driver", STRATEGY,
                      output="driver_prediction", intercept=6.0)
    other = _binding(tmp_path, "b-other", "implied_t1", OTHER_STRATEGY,
                     output="driver_prediction", intercept=100.0)
    for bindings in ((driver, other), (other, driver)):
        record = _score(tmp_path, _release(bindings), bindings)
        assert record.forecasts["driver_prediction"] == 6.0


def test_gate_scoped_to_request_strategy(tmp_path):
    """The STR-RUNUP gate binding never claims a STR-THRU request's gate
    slot, whichever release order it holds."""
    gate = _binding(tmp_path, "b-gate-thru", "gate", STRATEGY,
                    output="gate_score", intercept=1.0)
    other = _binding(tmp_path, "b-gate-runup", "gate", OTHER_STRATEGY,
                     output="gate_score", intercept=-42.0)
    for bindings in ((gate, other), (other, gate)):
        record = _score(tmp_path, _release(bindings), bindings)
        assert record.gate_terms["gate_score"] == 1.0
        assert record.gate_terms["gate_pass"] is True


def test_local_recipe_not_stripped_for_other_strategy_binding(tmp_path):
    """A release binding only STR-RUNUP owns no STR-THRU target, so the
    caller-declared local ``driver_prediction`` recipe still serves."""
    other = _binding(tmp_path, "b-runup", "implied_t1", OTHER_STRATEGY,
                     output="driver_prediction", intercept=100.0)
    record = _score(tmp_path, _release((other,)), (other,), requests=())
    assert record.forecasts["driver_prediction"] == 7.0
    assert "MISSING_FORECAST_OUTPUT:driver" not in record.reason_codes


def test_wildcard_strategy_binding_is_not_scoped_out(tmp_path):
    """A shared ``strategy_id="*"`` binding owns its target for every
    strategy: the local ``forecast_abs_move`` recipe is stripped and the
    wildcard's own value is published."""
    driver = _binding(tmp_path, "b-driver", "driver", STRATEGY,
                      output="driver_prediction", intercept=6.0)
    shared = _binding(tmp_path, "b-forecast", "forecast", "*",
                      output="forecast_abs_move", intercept=3.0)
    inputs = _native_inputs(
        forecast_recipes={
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
            "forecast_abs_move": {"intercept": 88.0, "coefficients": {}},
        },
        model_artifact_refs={"driver_prediction": "sha256:m1",
                             "forecast_abs_move": "sha256:m2"},
    )
    record = _score(tmp_path, _release((driver, shared)), (driver, shared),
                    inputs=inputs)
    assert record.forecasts["forecast_abs_move"] == 3.0


# == scoped collisions ==========================================================


def test_conflict_same_strategy_collision(tmp_path):
    """Two same-scope bindings mapping to ``driver_prediction`` refuse
    instead of letting binding order decide."""
    driver = _binding(tmp_path, "b-driver", "driver", STRATEGY,
                      output="driver_prediction", intercept=6.0)
    implied = _binding(tmp_path, "b-implied", "implied_t1", STRATEGY,
                       output="driver_prediction", intercept=100.0)
    release = _release((driver, implied))
    with pytest.raises(application.FrozenBindingConflict) as excinfo:
        _score(tmp_path, release, (driver, implied), requests=())
    assert excinfo.value.target == "driver_prediction"
    assert set(excinfo.value.binding_ids) == {"b-driver", "b-implied"}


def test_conflict_detected_before_any_inference(tmp_path):
    """FrozenBindingConflict is raised before inference.infer ever runs."""
    driver = _binding(tmp_path, "b-driver", "driver", STRATEGY,
                      output="driver_prediction", intercept=6.0)
    implied = _binding(tmp_path, "b-implied", "implied_t1", STRATEGY,
                       output="driver_prediction", intercept=100.0)
    release = _release((driver, implied))

    class _NeverCalledInference:
        def infer(self, release, item):
            raise AssertionError(
                "inference.infer must not run before the binding-conflict check"
            )

    with pytest.raises(application.FrozenBindingConflict):
        application.score_frozen(
            _request(), _NeverCalledInference(), release, (),
            {"_native_inputs": _native_inputs()})


def test_conflict_duplicate_gate_same_scope(tmp_path):
    """A strategy's own gate plus the shared ``"*"`` gate claim the same
    scoped slot: a release-authoring defect, refused by name."""
    own = _binding(tmp_path, "b-gate", "gate", STRATEGY,
                   output="gate_score", intercept=1.0)
    shared = _binding(tmp_path, "b-gate-shared", "gate", "*",
                      output="gate_score", intercept=-42.0)
    release = _release((own, shared))
    with pytest.raises(application.FrozenBindingConflict) as excinfo:
        _score(tmp_path, release, (own, shared), requests=())
    assert excinfo.value.target == "gate"
    assert set(excinfo.value.binding_ids) == {"b-gate", "b-gate-shared"}


# == single-strategy release regression =========================================


def test_single_strategy_release_unchanged(tmp_path):
    """A release carrying only STR-THRU's own driver + gate scores exactly
    as before: scored, both frozen values published, no refusal."""
    driver = _binding(tmp_path, "b-driver", "driver", STRATEGY,
                      output="driver_prediction", intercept=6.0)
    gate = _binding(tmp_path, "b-gate", "gate", STRATEGY,
                    output="gate_score", intercept=1.0)
    release = _release((driver, gate))
    inputs = build_native_score_inputs(_bundle(
        feature_vector={"x": X}, feature_missing_mask={"x": False},
        payoff_recipe={"min_trades": 2, "seed": 42, "draw_count": 16},
        payoff_source_rows=_GOOD_PAYOFF_ROWS,
        model_residual_rows=[{"prediction": 6.0, "residual": 0.0}],
    ))
    record = _score(tmp_path, release, (driver, gate), inputs=inputs)
    assert record.validation_status == "scored"
    assert record.forecasts["driver_prediction"] == 6.0
    assert record.gate_terms["gate_score"] == 1.0
    assert record.gate_terms["gate_pass"] is True
