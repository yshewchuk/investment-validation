import json
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import Pricing, generate, price
from engine.v2.foundation import DocumentError, to_document
from engine.v2.models import no_fit
from engine.v2.models.payoff_artifact import (
    PayoffLineArtifact,
    PayoffSurfaceArtifact,
    make_payoff_line_artifact,
    make_payoff_surface_artifact,
    payoff_artifact_key,
)
from engine.v2.ops.cli import _load_native_score_inputs, rescore_command
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, StageReceipt


def _receipts():
    return tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics"
    )


def _line_artifact():
    return make_payoff_line_artifact(
        {"n": 42, "resid_sd": 0.25, "r": 0.8, "intercept": 1.02, "slope": 0.35,
         "residuals": [0.1, -0.2, 0.3, -0.15]},
        strategy="STR-THRU", driver="move", alpha=0.5, cutoff="2026-09-15",
        window=("2026-06-15", "2026-09-15"),
    )


def _surface_artifact():
    return make_payoff_surface_artifact(
        {"n": 30, "resid_sd": 0.3, "r": 0.7, "coefficients": [0.9, 0.4, 0.2],
         "residuals": [0.05, -0.1, 0.2]},
        alpha=0.5, cutoff="2026-09-15",
        window=("2026-06-15", "2026-09-15"),
    )


def _write_fixture(tmp_path, model=None):
    request = ScoreRequest(
        event_id="evt-rescore", calendar_revision="cal-1", strategy_version="STR-THRU",
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1",
        mode="replay", fill_model={"alpha": 0.5},
    )
    context = {"ticker": "AAA", "strategy": "STR-THRU", "event_date": "2026-09-16",
              "entry_date": "2026-09-16", "exit_date": "2026-09-17", "expiry": "2026-09-18",
              "spot": 100.0, "strike": 95.0}
    features = {"strike": 100.0, "model_inputs": {"strike": 100.0}}
    forecast = {"models": {"driver_prediction": {"intercept": 7.0, "coefficients": {}}}}
    stale_geometry = generate("STR-THRU", {**context, "strike": 95.0})
    priced_legs = []
    for strike, bid, ask in ((95.0, 0.5, 1.5), (100.0, 1.0, 2.0), (105.0, 2.0, 4.0), (110.0, 3.0, 6.0)):
        geometry = generate("STR-THRU", {**context, "strike": strike})
        quotes = {(leg.right, leg.strike, leg.expiry): {"bid": bid, "ask": ask} for leg in geometry.legs}
        priced_legs.extend(price(geometry, quotes, 0.5).legs)
    pricing = Pricing("STR-THRU", 100.0, 0.0, tuple(priced_legs))
    extra = {} if model is None else {"model": model}
    inputs = NativeScoreInputs(
        context=context, features=features, forecast=forecast, geometry=stale_geometry,
        pricing=pricing, analogs={}, simulation={}, gate={}, chooser={}, diagnostics={},
        source_ref="typed-native-fixture", stage_receipts=_receipts(), **extra,
    )
    request_path = tmp_path / "request.json"
    inputs_path = tmp_path / "native_inputs.json"
    request_path.write_text(json.dumps(to_document(request)))
    inputs_path.write_text(json.dumps(to_document(inputs)))
    return request_path, inputs_path


def _serialized_native_doc(tmp_path, model):
    """A JSON document of ``to_document(NativeScoreInputs)`` carrying ``model``."""
    _, inputs_path = _write_fixture(tmp_path, model=model)
    return json.loads(inputs_path.read_text())


def test_rescore_command_prints_a_real_score_record(tmp_path):
    request_path, inputs_path = _write_fixture(tmp_path)
    args = SimpleNamespace(request=request_path, native_inputs=inputs_path)
    record = rescore_command(args)
    assert record.score_id
    document = to_document(record)
    assert document["canonical_request"]["event_id"] == "evt-rescore"
    # round-trips through json exactly like every other ops command's output
    json.dumps(document)


def test_rescore_command_runs_score_one_under_the_no_fit_guard(tmp_path, monkeypatch):
    import engine.v2.scoring.application as application_module

    request_path, inputs_path = _write_fixture(tmp_path)
    seen = {}

    def fake_score_one(request, inputs, **kwargs):
        seen["forbidden_during_call"] = no_fit.fitting_forbidden()
        return "FAKE_RECORD"

    monkeypatch.setattr(application_module, "score_one", fake_score_one)
    args = SimpleNamespace(request=request_path, native_inputs=inputs_path)
    result = rescore_command(args)
    assert result == "FAKE_RECORD"
    assert seen["forbidden_during_call"] is True
    assert no_fit.fitting_forbidden() is False  # released after the call


def test_no_fit_guard_blocks_a_real_fit_call():
    from engine.v2.models.no_fit import RuntimeFitForbidden, no_fit_guard
    from engine.v2.scoring.native_payoff import fit_payoff_line

    with no_fit_guard():
        with pytest.raises(RuntimeFitForbidden):
            fit_payoff_line([], before=None)


def test_roundtrip_restores_typed_line_artifact_and_leaves_the_document_intact(tmp_path):
    artifact = _line_artifact()
    doc = _serialized_native_doc(
        tmp_path, {"payoff_artifact": artifact, "exp_pnl_model": 1.23})
    assert type(doc["model"]["payoff_artifact"]) is dict  # to_document flattened it
    loaded = _load_native_score_inputs(doc)
    restored = loaded.model["payoff_artifact"]
    assert type(restored) is PayoffLineArtifact
    assert restored.content_hash == artifact.content_hash
    assert restored.key == artifact.key
    assert restored.key == payoff_artifact_key("STR-THRU", 0.5, "2026-09-15")
    assert restored == artifact
    # every other model field passes through, and the caller's document is
    # untouched: the typed artifact lives only in the fresh model block
    assert loaded.model["exp_pnl_model"] == 1.23
    assert loaded.model is not doc["model"]
    assert type(doc["model"]) is dict
    assert type(doc["model"]["payoff_artifact"]) is dict


def test_roundtrip_restores_typed_surface_artifact(tmp_path):
    artifact = _surface_artifact()
    doc = _serialized_native_doc(tmp_path, {"payoff_artifact": artifact})
    restored = _load_native_score_inputs(doc).model["payoff_artifact"]
    assert type(restored) is PayoffSurfaceArtifact
    assert restored.content_hash == artifact.content_hash
    assert restored.key == payoff_artifact_key("STR-RUNUP", 0.5, "2026-09-15")


def test_tampered_payload_declaring_the_old_hash_is_refused(tmp_path):
    artifact = _line_artifact()
    doc = _serialized_native_doc(tmp_path, {"payoff_artifact": artifact})
    doc["model"]["payoff_artifact"]["slope"] += 0.5
    with pytest.raises(DocumentError) as excinfo:
        _load_native_score_inputs(doc)
    assert excinfo.value.code == "CONTENT_HASH_MISMATCH"
    assert excinfo.value.path == "$.model.payoff_artifact.content_hash"
    assert artifact.content_hash not in str(excinfo.value)


def test_missing_declared_hash_is_refused(tmp_path):
    doc = _serialized_native_doc(tmp_path, {"payoff_artifact": _line_artifact()})
    del doc["model"]["payoff_artifact"]["content_hash"]
    with pytest.raises(DocumentError) as excinfo:
        _load_native_score_inputs(doc)
    assert excinfo.value.code == "MISSING_FIELD"
    assert excinfo.value.path == "$.model.payoff_artifact.content_hash"


def test_unknown_artifact_schema_is_refused_without_echoing_the_value(tmp_path):
    doc = _serialized_native_doc(tmp_path, {"payoff_artifact": _line_artifact()})
    canary = "payoff_line_artifact.v9.9-TAMPERED"
    doc["model"]["payoff_artifact"]["schema_version"] = canary
    with pytest.raises(DocumentError) as excinfo:
        _load_native_score_inputs(doc)
    assert excinfo.value.code == "BAD_SCHEMA_VERSION"
    assert excinfo.value.path == "$.model.payoff_artifact.schema_version"
    assert canary not in str(excinfo.value)


def test_missing_artifact_field_is_refused_with_a_child_path(tmp_path):
    doc = _serialized_native_doc(tmp_path, {"payoff_artifact": _line_artifact()})
    del doc["model"]["payoff_artifact"]["slope"]
    with pytest.raises(DocumentError) as excinfo:
        _load_native_score_inputs(doc)
    assert excinfo.value.code == "MISSING_FIELD"
    assert excinfo.value.path == "$.model.payoff_artifact.slope"


def test_nonfinite_n_tag_raises_document_error_not_raw_overflow(tmp_path):
    doc = _serialized_native_doc(tmp_path, {"payoff_artifact": _line_artifact()})
    doc["model"]["payoff_artifact"]["n"] = {"__nonfinite__": "inf"}
    try:
        _load_native_score_inputs(doc)
    except OverflowError as exc:
        pytest.fail(f"raw OverflowError escaped the payoff artifact guard: {exc!r}")
    except DocumentError as exc:
        assert exc.code == "BAD_TYPE"
        assert exc.path.startswith("$.model.payoff_artifact")
    else:
        pytest.fail("expected DocumentError")


def test_non_object_artifact_is_refused_without_echoing_the_value(tmp_path):
    doc = _serialized_native_doc(tmp_path, {"payoff_artifact": _line_artifact()})
    doc["model"]["payoff_artifact"] = "not-an-artifact-document"
    with pytest.raises(DocumentError) as excinfo:
        _load_native_score_inputs(doc)
    assert excinfo.value.code == "BAD_TYPE"
    assert excinfo.value.path == "$.model.payoff_artifact"
    assert "not-an-artifact-document" not in str(excinfo.value)


def test_artifact_free_model_blocks_still_load(tmp_path):
    _, inputs_path = _write_fixture(tmp_path)
    loaded = _load_native_score_inputs(json.loads(inputs_path.read_text()))
    assert loaded.model == {}
    doc = _serialized_native_doc(
        tmp_path, {"payoff_artifact": None, "exp_pnl_model": 1.23})
    loaded = _load_native_score_inputs(doc)
    assert loaded.model["payoff_artifact"] is None
    assert loaded.model["exp_pnl_model"] == 1.23