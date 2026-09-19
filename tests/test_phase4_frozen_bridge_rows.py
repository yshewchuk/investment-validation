"""The frozen bridge feeds each binding the row the capture fed it.

``checks/phase4_frozen_bridge._feature_rows`` used to build every binding's
row from the merged forecast-family ``model_inputs``. The capture
(``tools/capture_tier0_corpus._frozen_runtime``) feeds each binding its own
role's vector, and the gate's vector lives only in the ``gate_inputs``
checkpoint. So the replay either refused the gate (a gate-only column) or fed
it the driver's value of a same-named column, in the driver's set rather than
the gate's. The capture now records the per-role rows in the source-bound
features block (``role_model_inputs``) and the bridge reads them.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from checks import phase4_real
from checks.phase4_frozen_bridge import FrozenBridgeError, _feature_rows
from engine.v2.foundation import content_hash
from engine.v2.models import ModelBinding
from engine.v2.models.contracts import ArtifactMember
from engine.v2.scoring import application
from tests.test_phase4_capture_strict import _artifact, _full_strict_candidate
from tools.capture_tier0_corpus import _hydrate_trace, attach_strict_probe

CLOCK = "legacy.decision_offset.0"
DRIVER = {"x": 2.0}
# The gate's own vector: a gate-only column, a same-named column at ANOTHER
# value than the driver's, and an order that is not the driver's.
GATE = {"n_prior": 5.0, "x": 9.0}


def _traced(tmp_path, monkeypatch, gate_vector=GATE):
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="case-0", ticker="AAA", driver_vector=DRIVER,
        gate_vector=gate_vector, path=path, digest=digest,
    )
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    for binding in checkpoint["value"]["model_bindings"]:
        binding["decision_clock"] = CLOCK  # the clock legacy requests carry
    checkpoint["content_hash"] = content_hash(checkpoint["value"])
    attached, gaps = attach_strict_probe([candidate], "snapshot-1", tmp_path)
    assert attached == ("case-0",) and gaps == {}
    trace = _hydrate_trace(candidate["input_trace"])
    pair = {"payload": {
        "request": candidate["request"], "input_trace": trace,
        "input_trace_hash": trace["trace_hash"],
        "legacy_input_hash": candidate["legacy_input_hash"],
    }}
    return pair


def _rows_by_role(plan):
    roles = {binding.binding_id: binding.role for binding in plan.release.bindings}
    return {roles[item.binding_id]: (item.feature_order, item.rows)
            for item in plan.requests}


def test_replay_feeds_the_gate_its_own_vector_in_its_own_order(tmp_path, monkeypatch):
    pair = _traced(tmp_path, monkeypatch)
    features = pair["payload"]["input_trace"]["native_inputs"]["features"]
    assert features["role_model_inputs"] == {"driver": DRIVER, "gate": GATE}

    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    rows = _rows_by_role(verified["frozen_replay"])

    assert rows["gate"] == (("n_prior", "x"), ((5.0, 9.0),))
    assert rows["driver"] == (("x",), ((2.0,),))


def test_replayed_execution_reproduces_the_captured_receipts(tmp_path, monkeypatch):
    pair = _traced(tmp_path, monkeypatch)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    plan = verified["frozen_replay"]

    native = application.score_frozen(
        verified["request"], plan.inference, plan.release, plan.requests,
        {"_native_inputs": verified["inputs"]},
    )
    phase4_real._verify_runtime_execution(verified, native)  # raises on drift
    # The per-role rows are inputs, never record fields.
    assert "role_model_inputs" not in native.resolved_request


def test_a_tampered_gate_row_breaks_the_trace_hashes(tmp_path, monkeypatch):
    pair = _traced(tmp_path, monkeypatch)
    tampered = copy.deepcopy(pair)
    rows = tampered["payload"]["input_trace"]["native_inputs"]["features"]
    rows["role_model_inputs"]["gate"]["x"] = 2.0
    with pytest.raises(phase4_real._TraceError):
        phase4_real._verified_trace_bundle(tampered, tmp_path)


def _binding(role, order):
    return ModelBinding(
        binding_id=f"b-{role}", model_id=f"m-{role}", role=role, strategy_id="STR-THRU",
        decision_clock_id=CLOCK, adapter="joblib-estimator.v1", feature_order=order,
        output_names=("out",),
        members=(ArtifactMember(name="estimator", path="a", content_hash="sha256:" + "0" * 64),),
    )


def _inputs(features):
    # _feature_rows reads only the features block.
    return SimpleNamespace(features=features)


def test_without_per_role_rows_a_gate_binding_is_refused_not_fed_the_merge():
    inputs = _inputs({"model_inputs": {"x": 2.0, "n_prior": 5.0}})
    with pytest.raises(FrozenBridgeError, match="role gate needs its own captured row"):
        _feature_rows(inputs, (_binding("gate", ("n_prior", "x")),), "release")
    # A forecast-family role may still read the merge (older traces).
    (request,) = _feature_rows(inputs, (_binding("driver", ("x",)),), "release")
    assert request.rows == ((2.0,),)


def test_per_role_rows_refuse_a_binding_whose_role_was_not_captured():
    inputs = _inputs({"model_inputs": {"x": 2.0}, "role_model_inputs": {"driver": {"x": 2.0}}})
    with pytest.raises(FrozenBridgeError, match="no captured row for role gate"):
        _feature_rows(inputs, (_binding("gate", ("x",)),), "release")
