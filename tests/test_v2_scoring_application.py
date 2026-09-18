from types import SimpleNamespace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.scoring import application
from engine.v2.scoring.identity import score_id
from engine.v2.scoring.stages import NativeScoreInputs


def request():
    return ScoreRequest(
        event_id="evt-1", calendar_revision="cal-1", strategy_version="STR-THRU",
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5}, dependency_refs=("analog-pop",),
        model_artifact_refs=("model-a",), residual_state_ref="res-a",
    )


def result():
    return SimpleNamespace(as_dict=lambda: {
        "ticker": "AAA", "strategy": "STR-THRU", "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-17", "expiry": "2026-09-18",
        "spot": 100.0, "entry_cost": 5.0, "implied_move": 6.0,
        "driver_name": "abs_move", "driver_prediction": 7.0, "legs": [], "flags": [], "model_inputs": {"x": 0.0},
        "gate_score": 0.7, "gate_threshold": 0.6, "gate_pass": True,
        "detail": "", "payoff": {}, "fill": 0.5,
    })


def dynamic_result(strategy):
    value = {
        "TWIN-P": 0.2, "TWIN-P5": 0.4, "CND-PS": 0.1,
        "BFLY-P": 0.15, "BFLY-P5": 0.12, "RAMP7": 0.05, "CTR5": 0.08,
    }[strategy]
    return SimpleNamespace(as_dict=lambda: {
        "ticker": "AAA", "strategy": strategy, "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-17", "expiry": "2026-09-18",
        "spot": 100.0, "entry_cost": 5.0, "implied_move": 6.0,
        "driver_name": "abs_move", "driver_prediction": 7.0,
        "legs": [], "flags": [], "model_inputs": {}, "exp_pnl_sim": value,
        "chooser_score": value, "gate_score": 0.7, "gate_threshold": 0.6,
        "gate_pass": True, "detail": "", "payoff": {}, "fill": 0.5,
    })


def test_single_and_batch_share_score_id(monkeypatch):
    fields = result().as_dict()
    native = NativeScoreInputs.from_legacy_fields(fields)
    single = application.score_one(request(), native)
    batch = application.score_many(((request(), native),))[0]
    assert single.score_id == batch.score_id
    assert single.financial_diagnostics["entry_cost_pct"] == 5.0
    assert single.financial_diagnostics["model_vs_market"] == 7.0 / (6.0 * 0.645)
    assert single.null_masks == {"x": False}


def test_native_inputs_require_stage_receipts():
    fields = result().as_dict()
    native = NativeScoreInputs.from_legacy_fields(fields)
    assert application.score_one(request(), native).evidence_refs == ("analog-pop",)


def test_operational_time_does_not_change_score_id(monkeypatch):
    fields = result().as_dict()
    native = NativeScoreInputs.from_legacy_fields(fields)
    first = application.score_one(request(), native)
    second = application.score_one(request(), native)
    assert first.score_id == second.score_id


def test_score_record_is_deeply_immutable_and_hash_stays_bound():
    fields = result().as_dict()
    fields["model_inputs"] = {"x": {"nested": 1.0}}
    native = NativeScoreInputs.from_legacy_fields(fields)
    record = application.score_one(request(), native)
    original_hash = record.payload_hash

    with pytest.raises(TypeError, match="immutable"):
        record.forecasts["driver_prediction"] = 99.0
    with pytest.raises(TypeError, match="immutable"):
        record.resolved_request["model_inputs"]["x"]["nested"] = 99.0

    fields["model_inputs"]["x"]["nested"] = 99.0
    assert record.feature_values["x"]["nested"] == 1.0
    assert record.payload_hash == original_hash
    assert score_id(record) == original_hash


def test_frozen_inference_path_does_not_call_legacy_backend():
    class Frozen:
        def infer(self, release, inference_request):
            return SimpleNamespace(status="READY", predictions=((0.42,),),
                                    artifact_hashes=("sha256:model",), reason_codes=(), detail=None)

    # Native inputs are built explicitly here (the legitimate non-acceptance
    # use of NativeScoreInputs.from_legacy_fields, mirroring
    # checks/phase4_real.py's _native() helper) rather than reconstructed
    # implicitly inside score_frozen.
    raw = {"ticker": "AAA", "event_date": "2026-09-16", "spot": 100.0,
           "entry_cost": 5.0, "implied_move": 6.0, "driver_name": "abs_move",
           "legs": [], "flags": [], "model_inputs": {}, "payoff": {}, "fill": 0.5}
    record = application.score_frozen(
        request(), Frozen(), object(), object(),
        {"_native_inputs": NativeScoreInputs.from_legacy_fields(raw)},
    )
    assert record.forecasts["driver_prediction"] == 0.42
    assert record.validation_status == "refused"
    assert record.readiness == "refused"


def test_frozen_path_refuses_legacy_answer_fields_without_native_inputs():
    """The acceptance/frozen path must not fall back to reconstructing
    NativeScoreInputs from caller-supplied legacy answer fields; it must
    refuse, naming what it tried to source from an answer."""
    class Frozen:
        def infer(self, release, inference_request):
            return SimpleNamespace(status="READY", predictions=((0.42,),),
                                    artifact_hashes=("sha256:model",), reason_codes=(), detail=None)

    with pytest.raises(TypeError) as excinfo:
        application.score_frozen(
            request(), Frozen(), object(), object(),
            {"driver_prediction": 7.0, "gate_score": 0.7, "gate_threshold": 0.6,
             "gate_pass": True},
        )
    message = str(excinfo.value)
    assert "_native_inputs" in message
    assert "driver_prediction" in message
    assert "gate_threshold" in message


def test_frozen_path_refuses_when_native_inputs_missing_entirely():
    class Frozen:
        def infer(self, release, inference_request):
            return SimpleNamespace(status="READY", predictions=((0.42,),),
                                    artifact_hashes=("sha256:model",), reason_codes=(), detail=None)

    with pytest.raises(TypeError, match="_native_inputs"):
        application.score_frozen(request(), Frozen(), object(), object(), {})


def test_direct_dynamic_request_resolves_complete_menu_without_regating(monkeypatch):
    base = request()
    fields = {"menu": {
        strategy: dynamic_result(strategy).as_dict()
        for strategy in ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")
    }}
    fields["menu"] = {strategy: NativeScoreInputs.from_legacy_fields(value)
                      for strategy, value in fields["menu"].items()}
    selected = application.score_event(
        base.__class__(**{**base.__dict__, "strategy_version": "DYN-SV"}),
        (("DYN-SV", fields),),
    )[0]
    assert selected.chooser_selection["strategy"] == "TWIN-P5"
    assert selected.chooser_selection["menu_size"] == 7
    assert selected.chooser_selection["status"] == "selected"
    assert selected.canonical_request["strategy_version"] == "DYN-SV"
