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
from dataclasses import replace
from types import SimpleNamespace

import pytest

from checks import phase4_real
from checks.phase4_frozen_bridge import (
    FrozenBridgeError, _feature_rows, binding_feature_row,
)
from engine.v2.foundation import content_hash
from engine.v2.scoring.native_gate_features import (
    GATE_ANALOG_COLUMNS, GATE_FORECAST_COLUMNS,
)
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
    assert native.feature_values == DRIVER
    assert native.null_masks == {"x": False}


def test_replay_model_inputs_do_not_depend_on_binding_order(tmp_path, monkeypatch):
    pair = _traced(tmp_path, monkeypatch)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    plan = verified["frozen_replay"]
    native = application.score_frozen(
        verified["request"], plan.inference, plan.release,
        tuple(reversed(plan.requests)), {"_native_inputs": verified["inputs"]},
    )
    assert native.feature_values == DRIVER
    assert native.null_masks == {"x": False}


def test_missing_driver_request_is_reported_when_native_stage_refuses_it(
    tmp_path, monkeypatch,
):
    pair = _traced(tmp_path, monkeypatch)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    inputs = verified["inputs"]
    features = dict(inputs.features)
    role_rows = dict(features["role_model_inputs"])
    role_rows["driver"] = {"x": None}
    features["role_model_inputs"] = role_rows
    missing = replace(inputs, features=features)
    native = application.score_frozen(
        verified["request"], verified["frozen_replay"].inference,
        verified["frozen_replay"].release, (),
        {"_native_inputs": missing},
    )
    assert native.feature_values == {"x": None}
    assert native.null_masks == {"x": True}


def test_empty_role_row_does_not_claim_shared_same_named_feature(
    tmp_path, monkeypatch,
):
    pair = _traced(tmp_path, monkeypatch)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    inputs = verified["inputs"]
    features = dict(inputs.features)
    features["role_model_inputs"] = {"driver": {}, "gate": GATE}
    native_inputs = replace(inputs, features=features)
    native = application.score_frozen(
        verified["request"], verified["frozen_replay"].inference,
        verified["frozen_replay"].release, (),
        {"_native_inputs": native_inputs},
    )
    assert native.feature_values == {"x": None}
    assert native.null_masks == {"x": True}


def test_tagged_nonfinite_driver_input_is_missing_not_invalid(
    tmp_path, monkeypatch,
):
    pair = _traced(tmp_path, monkeypatch)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    inputs = verified["inputs"]
    features = dict(inputs.features)
    features["role_model_inputs"] = {
        "driver": {"x": {"__nonfinite__": "nan"}}, "gate": GATE,
    }
    native_inputs = replace(inputs, features=features)
    native = application.score_frozen(
        verified["request"], verified["frozen_replay"].inference,
        verified["frozen_replay"].release, (),
        {"_native_inputs": native_inputs},
    )
    assert native.feature_values == {"x": {"__nonfinite__": "nan"}}
    assert native.null_masks == {"x": True}
    assert "MISSING_FEATURES" in native.reason_codes
    assert "INVALID_FEATURE" not in native.reason_codes


def test_gate_only_release_does_not_publish_gate_features_as_model_inputs(
    tmp_path, monkeypatch,
):
    pair = _traced(tmp_path, monkeypatch)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    plan = verified["frozen_replay"]
    gate_bindings = tuple(binding for binding in plan.release.bindings
                          if binding.role == "gate")
    gate_requests = tuple(request for request in plan.requests
                          if request.binding_id in {b.binding_id for b in gate_bindings})
    release = replace(plan.release, bindings=gate_bindings)
    native = application.score_frozen(
        verified["request"], plan.inference, release, gate_requests,
        {"_native_inputs": verified["inputs"]},
    )
    assert native.feature_values == {}
    assert native.null_masks == {}


def test_model_inputs_are_empty_when_no_model_binding_exists(
    tmp_path, monkeypatch,
):
    pair = _traced(tmp_path, monkeypatch)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    plan = verified["frozen_replay"]
    release = replace(plan.release, bindings=())
    native = application.score_frozen(
        verified["request"], plan.inference, release, (),
        {"_native_inputs": verified["inputs"]},
    )
    assert native.feature_values == {}
    assert native.null_masks == {}


def test_unservable_model_refusal_does_not_claim_model_inputs(tmp_path, monkeypatch):
    pair = _traced(tmp_path, monkeypatch)
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)

    class Unavailable:
        def infer(self, release, request):
            return SimpleNamespace(
                status="MODEL_NOT_READY", release_id=release.release_id,
                binding_id=request.binding_id, model_id=None,
                artifact_hashes=(), output_names=(), predictions=(),
                reason_codes=("MODEL_NOT_READY",), detail="artifact unavailable",
            )

    plan = verified["frozen_replay"]
    native = application.score_frozen(
        verified["request"], Unavailable(), plan.release, plan.requests,
        {"_native_inputs": verified["inputs"]},
    )
    assert native.feature_values == {}
    assert native.null_masks == {}
    assert native.readiness == "refused"


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


# ---------------------------------------------------------------------------
# A genuinely non-finite captured feature (contracts §2.1's
# ``{"__nonfinite__": repr(value)}`` tag -- a real sourced NaN, e.g. no
# ORATS quote for that ticker/date, not a dropped column) used to reach
# ``float(vector[name])`` untouched and raise "nonnumeric feature" (a
# TypeError on ``float(dict)``), which killed the WHOLE bridge construction
# and excluded the record from the Phase 4 population before any comparison
# was even attempted. Fixed: the tag decodes to a real NaN, and a binding
# whose row comes back non-finite is silently OMITTED (no request, no
# raise) rather than raised -- mirroring legacy's own behavior (neither
# ``Scorer._score_model`` nor ``ServingModel.predict`` ever calls a model
# on an incomplete row) and ``FrozenStageExecutor._row``'s identical
# "a non-finite value is a missing feature, not an invalid one".
# ---------------------------------------------------------------------------

def test_a_tagged_nonfinite_feature_omits_its_binding_not_raises():
    inputs = _inputs({"model_inputs": {"x": {"__nonfinite__": "nan"}, "n_prior": 5.0}})
    requests = _feature_rows(inputs, (_binding("driver", ("x",)),), "release")
    assert requests == ()  # omitted, not raised


def test_a_tagged_nonfinite_feature_omits_only_its_own_binding():
    inputs = _inputs({"model_inputs": {"x": {"__nonfinite__": "nan"}, "n_prior": 5.0}})
    bindings = (_binding("driver", ("x",)), _binding("implied_t1", ("n_prior",)))
    requests = _feature_rows(inputs, bindings, "release")
    assert [item.binding_id for item in requests] == ["b-implied_t1"]
    assert requests[0].rows == ((5.0,),)


def test_a_tagged_nonfinite_feature_decodes_to_nan_not_zero():
    """The regression this test guards against: a decode that produced 0.0
    instead of NaN would NOT be omitted (0.0 is finite) and would silently
    feed the model a fabricated value. Confirmed via the previous two tests'
    omission; this one pins the exact decoded value so a future change
    cannot swap NaN for 0.0 and still pass "is omitted"."""
    from engine.v2.foundation import untag_nonfinite

    decoded = untag_nonfinite({"__nonfinite__": "nan"})
    assert decoded != decoded  # NaN, not 0.0 (0.0 == 0.0)


def test_a_genuinely_malformed_feature_still_raises_nonnumeric():
    """Not every non-numeric value is a nonfinite tag: a real data-shape
    defect (a string, a list) must still raise loudly."""
    inputs = _inputs({"model_inputs": {"x": "not-a-number", "n_prior": 5.0}})
    with pytest.raises(FrozenBridgeError, match="nonnumeric feature x"):
        _feature_rows(inputs, (_binding("driver", ("x",)),), "release")


def test_a_missing_feature_name_still_raises_not_omitted():
    """A structurally absent column (the key itself never captured) is a
    different, more severe defect than a present-but-non-finite value, and
    stays a hard refusal."""
    inputs = _inputs({"model_inputs": {"n_prior": 5.0}})
    with pytest.raises(FrozenBridgeError, match="missing feature x"):
        _feature_rows(inputs, (_binding("driver", ("x",)),), "release")


# ---------------------------------------------------------------------------
# ``binding_feature_row`` classification order (Astra blocker-1 follow-up).
# The gate-deferral exception (a GATE row whose ONLY absent names are the
# derived forecast/analog columns) must not let a present non-finite cell
# mask an ILLEGAL omission, and a non-finite cell must not mask a later
# malformed one. Absent names are classified FIRST; every present cell is
# then validated (accumulating non-finite, never early-returning); only at the
# end do we yield ``None`` for a legitimate deferral or a non-finite row.
# ---------------------------------------------------------------------------

def test_unknown_missing_name_raises_even_with_a_present_nonfinite_cell():
    """Reproducer (driver role): ``feature_order=("missing_base","im")``,
    ``vector={"im": nan}``. The present NaN used to return ``None`` before the
    absent ``missing_base`` was ever checked, silently omitting a binding that
    legacy would hard-refuse. Missing names are classified first now, so the
    illegal omission raises regardless of the captured NaN."""
    binding = _binding("driver", ("missing_base", "im"))
    with pytest.raises(FrozenBridgeError, match="missing feature missing_base"):
        binding_feature_row(binding, {"im": float("nan")})


def test_unknown_missing_name_raises_for_gate_role_too():
    """Same reproducer on the GATE role: ``missing_base`` is not one of the
    derived forecast/analog columns, so the gate deferral does not apply and
    the illegal omission raises even though a present cell is non-finite."""
    binding = _binding("gate", ("missing_base", "im"))
    with pytest.raises(FrozenBridgeError, match="missing feature missing_base"):
        binding_feature_row(binding, {"im": float("nan")})


def test_nonfinite_cell_does_not_mask_a_later_malformed_cell():
    """A NaN earlier in ``feature_order`` used to return ``None`` before the
    loop reached a later malformed string. Every present cell is now validated
    (non-finite accumulated, not early-returned), so the genuinely non-numeric
    value is still rejected loudly."""
    binding = _binding("driver", ("x", "y"))
    with pytest.raises(FrozenBridgeError, match="nonnumeric feature y"):
        binding_feature_row(binding, {"x": float("nan"), "y": "not-a-number"})


def test_gate_derived_omission_defers_when_present_cells_are_finite():
    """The legal gate deferral still works: a GATE row whose ONLY absent names
    are derived forecast/analog columns, with every present cell finite, is a
    pre-extension base frame -- omit eager inference (``None``), never raise."""
    derived = [GATE_FORECAST_COLUMNS[0], GATE_ANALOG_COLUMNS[0]]
    binding = _binding("gate", ("n_prior", "x", *derived))
    assert binding_feature_row(binding, {"n_prior": 5.0, "x": 9.0}) is None


def test_gate_derived_omission_still_rejects_a_malformed_present_cell():
    """Deferral is not a license to skip validation: a legal gate deferral plus
    a captured malformed string must still raise (validate before yielding)."""
    derived = [GATE_FORECAST_COLUMNS[0], GATE_ANALOG_COLUMNS[0]]
    binding = _binding("gate", ("n_prior", "x", *derived))
    with pytest.raises(FrozenBridgeError, match="nonnumeric feature x"):
        binding_feature_row(binding, {"n_prior": 5.0, "x": "not-a-number"})


def test_complete_finite_row_still_returns_its_values():
    """Unchanged happy path: a fully present, finite row is a plain tuple in
    ``feature_order`` -- not ``None``, not a raise."""
    binding = _binding("driver", ("n_prior", "x"))
    assert binding_feature_row(binding, {"x": 9.0, "n_prior": 5.0}) == (5.0, 9.0)
