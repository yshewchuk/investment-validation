"""P6-2: the production frozen-inputs builder and its Phase 4 bridge callers.

``engine/v2/scoring/frozen_inputs.py`` now owns the feature-vector-to-
``InferenceRequest`` logic the Phase 4 bridge used to implement privately.
These tests pin the two contract halves of that extraction:

* output/refusal parity -- for every row shape the bridge tests exercise
  (finite rows, tagged non-finite omissions, gate derived-column deferral,
  malformed cells, missing features, role-vector rules), the production
  entrypoint and the bridge's compatibility entrypoint agree exactly, the
  bridge converting ``FrozenInputsError`` into ``FrozenBridgeError`` with an
  identical message; and
* one implementation -- the bridge delegates at call time (patching the
  production functions changes what the bridge runs), and the production
  package imports nothing from ``checks/`` or ``tests/`` (the same rule
  ``checks/import_layers.py`` enforces as ``verification-imported``, asserted
  directly here for the scoring package).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.phase4_frozen_bridge import (  # noqa: E402
    FrozenBridgeError,
    _feature_rows,
    binding_feature_row,
)
from engine.v2.models import ModelBinding  # noqa: E402
from engine.v2.models.contracts import ArtifactMember  # noqa: E402
from engine.v2.scoring import frozen_inputs  # noqa: E402
from engine.v2.scoring.frozen_inputs import (  # noqa: E402
    FrozenInputsError,
    build_inference_requests,
)
from engine.v2.scoring.frozen_inputs import binding_feature_row as prod_row  # noqa: E402
from engine.v2.scoring.native_gate_features import (  # noqa: E402
    GATE_ANALOG_COLUMNS,
    GATE_FORECAST_COLUMNS,
)

CLOCK = "legacy.decision_offset.0"


def _binding(role, order):
    return ModelBinding(
        binding_id=f"b-{role}", model_id=f"m-{role}", role=role, strategy_id="STR-THRU",
        decision_clock_id=CLOCK, adapter="joblib-estimator.v1", feature_order=order,
        output_names=("out",),
        members=(ArtifactMember(name="estimator", path="a", content_hash="sha256:" + "0" * 64),),
    )


def _inputs(features):
    return SimpleNamespace(features=features)


# ---------------------------------------------------------------------------
# output/refusal parity: binding_feature_row
# ---------------------------------------------------------------------------

_ROW_CASES = [
    ("complete finite row", _binding("driver", ("n_prior", "x")),
     {"x": 9.0, "n_prior": 5.0}),
    ("tagged non-finite omits", _binding("driver", ("x",)),
     {"x": {"__nonfinite__": "nan"}}),
    ("gate derived deferral", _binding(
        "gate", ("n_prior", "x", GATE_FORECAST_COLUMNS[0], GATE_ANALOG_COLUMNS[0])),
     {"n_prior": 5.0, "x": 9.0}),
]

_REFUSAL_CASES = [
    ("missing feature", _binding("driver", ("missing_base", "im")),
     {"im": float("nan")}, "missing feature missing_base"),
    ("malformed cell", _binding("driver", ("x", "y")),
     {"x": float("nan"), "y": "not-a-number"}, "nonnumeric feature y"),
    ("gate deferral hides nothing", _binding(
        "gate", ("n_prior", "x", GATE_FORECAST_COLUMNS[0])),
     {"n_prior": 5.0, "x": "not-a-number"}, "nonnumeric feature x"),
]


@pytest.mark.parametrize(("label", "binding", "vector"), _ROW_CASES, ids=[c[0] for c in _ROW_CASES])
def test_production_and_bridge_agree_on_inclusion(label, binding, vector):
    production = prod_row(binding, vector)
    delegated = binding_feature_row(binding, vector)
    assert delegated == production
    if production is not None:
        assert isinstance(production, tuple)
        assert all(isinstance(value, float) for value in production)


@pytest.mark.parametrize(
    ("label", "binding", "vector", "message"), _REFUSAL_CASES,
    ids=[c[0] for c in _REFUSAL_CASES],
)
def test_production_and_bridge_agree_on_refusal(label, binding, vector, message):
    with pytest.raises(FrozenInputsError, match=message) as production:
        prod_row(binding, vector)
    with pytest.raises(FrozenBridgeError, match=message) as delegated:
        binding_feature_row(binding, vector)
    assert str(delegated.value) == str(production.value)


# ---------------------------------------------------------------------------
# output/refusal parity: request construction
# ---------------------------------------------------------------------------

def test_per_role_vectors_build_requests_in_binding_order():
    inputs = _inputs({"model_inputs": {"x": 2.0}, "role_model_inputs": {
        "driver": {"x": 2.0}, "gate": {"n_prior": 5.0, "x": 9.0},
    }})
    bindings = (_binding("driver", ("x",)), _binding("gate:midfill", ("n_prior", "x")))
    production = build_inference_requests(inputs, bindings, "release-7")
    delegated = _feature_rows(inputs, bindings, "release-7")
    assert delegated == production
    assert [(item.binding_id, item.release_id, item.feature_order, item.rows)
            for item in production] == [
        ("b-driver", "release-7", ("x",), ((2.0,),)),
        ("b-gate:midfill", "release-7", ("n_prior", "x"), ((5.0, 9.0),)),
    ]


def test_nonfinite_role_vector_omits_only_its_own_request():
    inputs = _inputs({"model_inputs": {}, "role_model_inputs": {
        "driver": {"x": {"__nonfinite__": "nan"}}, "implied_t1": {"x": 3.0},
    }})
    bindings = (_binding("driver", ("x",)), _binding("implied_t1", ("x",)))
    production = build_inference_requests(inputs, bindings, "release")
    assert production == _feature_rows(inputs, bindings, "release")
    assert [item.binding_id for item in production] == ["b-implied_t1"]


def test_merged_only_traces_feed_forecast_roles_and_refuse_private_ones():
    inputs = _inputs({"model_inputs": {"x": 2.0, "n_prior": 5.0}})
    (request,) = build_inference_requests(
        inputs, (_binding("driver", ("x",)),), "release")
    assert request == _feature_rows(inputs, (_binding("driver", ("x",)),), "release")[0]
    assert request.rows == ((2.0,),)
    for role in ("gate", "chooser"):
        binding = _binding(role, ("n_prior", "x"))
        with pytest.raises(FrozenInputsError, match="needs its own captured row") as production:
            build_inference_requests(inputs, (binding,), "release")
        with pytest.raises(FrozenBridgeError, match="needs its own captured row") as delegated:
            _feature_rows(inputs, (binding,), "release")
        assert str(delegated.value) == str(production.value)


def test_request_construction_refusals_match_across_both_entrypoints():
    cases = [
        (_inputs({}), "native inputs require features.model_inputs"),
        (_inputs({"model_inputs": {"x": 2.0}, "role_model_inputs": []}),
         "features.role_model_inputs: expected object"),
        (_inputs({"model_inputs": {"x": 2.0}, "role_model_inputs": {"driver": {"x": 2.0}}}),
         "no captured row for role gate"),
    ]
    for inputs, message in cases:
        with pytest.raises(FrozenInputsError, match=message) as production:
            build_inference_requests(inputs, (_binding("gate", ("x",)),), "release")
        with pytest.raises(FrozenBridgeError, match=message) as delegated:
            _feature_rows(inputs, (_binding("gate", ("x",)),), "release")
        assert str(delegated.value) == str(production.value)


# ---------------------------------------------------------------------------
# one implementation: the bridge delegates at CALL time
# ---------------------------------------------------------------------------

def test_bridge_binding_feature_row_delegates_to_production(monkeypatch):
    calls = []

    def fake(binding, vector):
        calls.append((binding.binding_id, dict(vector)))
        return (1.0, 2.0)

    monkeypatch.setattr(frozen_inputs, "binding_feature_row", fake)
    assert binding_feature_row(_binding("driver", ("a", "b")), {"a": 1.0}) == (1.0, 2.0)
    assert calls == [("b-driver", {"a": 1.0})]


def test_bridge_feature_rows_delegates_to_production(monkeypatch):
    calls = []

    def fake(inputs, bindings, release_id):
        calls.append((tuple(b.binding_id for b in bindings), release_id))
        return ()

    monkeypatch.setattr(frozen_inputs, "build_inference_requests", fake)
    inputs = _inputs({"model_inputs": {"x": 1.0}})
    assert _feature_rows(inputs, (_binding("driver", ("x",)),), "release-3") == ()
    assert calls == [(("b-driver",), "release-3")]


def test_replay_request_build_never_re_implements_the_row_predicate(monkeypatch):
    """The deferral/omission decision runs through the production predicate,
    not a copied-out loop: replacing only ``binding_feature_row`` in the
    production module changes what ``build_inference_requests`` emits."""
    monkeypatch.setattr(frozen_inputs, "binding_feature_row", lambda binding, vector: None)
    inputs = _inputs({"model_inputs": {"x": 2.0}})
    assert build_inference_requests(inputs, (_binding("driver", ("x",)),), "release") == ()
    assert _feature_rows(inputs, (_binding("driver", ("x",)),), "release") == ()


# ---------------------------------------------------------------------------
# no production import from checks (import_layers' verification-imported rule,
# asserted directly so the extraction can never reintroduce the dependency)
# ---------------------------------------------------------------------------

def test_scoring_package_imports_nothing_from_checks_or_tests():
    for path in sorted((ROOT / "engine" / "v2" / "scoring").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [] if node.level else [(node.module or "").split(".")[0]]
            else:
                continue
            assert not {"checks", "tests"} & set(roots), (
                f"{path.name} imports from {sorted(set(roots))}; production "
                "packages may not depend on verification or test code")
