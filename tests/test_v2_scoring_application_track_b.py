from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ScoreBatch, ScoreRequest
from engine.v2.domain.generation import Pricing, generate, price
from engine.v2.registry import DYNAMIC_MENU
from engine.v2.scoring import application
from engine.v2.scoring.identity import request_hash
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, StageReceipt


def _request(strategy="STR-THRU"):
    return ScoreRequest(
        event_id="evt-1", calendar_revision="cal-1", strategy_version=strategy,
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5}, dependency_refs=("analog-pop",),
        model_artifact_refs=("model-a",), residual_state_ref="res-a",
    )


def _native(strategy="STR-THRU", *, chooser_score=0.2, exp_pnl=0.1,
            flags=(), forged=False):
    context = {
        "ticker": "AAA", "strategy": strategy, "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-17",
        "expiry": "2026-09-18", "spot": 100.0,
    }
    forecast = {"driver_name": "abs_move", "driver_prediction": 7.0,
                "forecast_abs_move": 7.0}
    geometry = generate(strategy, {**context, **forecast})
    quotes = {(leg.right, leg.strike, leg.expiry): {"bid": 1.0, "ask": 2.0}
              for leg in geometry.legs}
    priced = price(geometry, quotes, 0.5)
    if forged:
        context.update({"entry_cost": 999.0, "legs": ({"name": "forged"},),
                        "model_artifact_ids": ("forged-model",)})
        priced = Pricing(priced.strategy, priced.spot, 999.0, priced.legs)
    receipts = tuple(StageReceipt(stage, "declared-input", "declared-output")
                     for stage in STAGE_NAMES if stage != "diagnostics")
    return NativeScoreInputs(
        context=context,
        features={"model_inputs": {"x": 0.0}, "implied_move": 6.0},
        forecast=forecast, geometry=geometry, pricing=priced,
        analogs={}, simulation={"exp_pnl_sim": exp_pnl},
        gate={"flags": flags, "gate_pass": not flags},
        chooser={"chooser_score": chooser_score}, diagnostics={"flags": ()},
        source_ref="fixture", stage_receipts=receipts,
    )


def _override_native(strategy="STR-THRU"):
    fields = {
        "ticker": "AAA", "strategy": strategy, "event_date": "2026-09-16",
        "entry_date": "2026-09-16", "exit_date": "2026-09-17",
        "expiry": "2026-09-18", "spot": 100.0, "entry_cost": 3.0,
        "driver_name": "abs_move", "driver_prediction": 7.0,
        "forecast_abs_move": 7.0, "model_inputs": {}, "flags": (),
    }
    priced_legs = []
    for strike in (100.0, 105.0):
        geometry = generate(strategy, {**fields, "strike": strike})
        quotes = {
            (leg.right, leg.strike, leg.expiry): {"bid": 1.0, "ask": 2.0}
            for leg in geometry.legs
        }
        priced_legs.extend(price(geometry, quotes, 0.5).legs)
    inputs = NativeScoreInputs.from_legacy_fields(fields)
    return replace(
        inputs,
        pricing=Pricing(strategy, 100.0, 0.0, tuple(priced_legs)),
    )


@pytest.mark.parametrize("override_field", ["contract_override", "geometry_override"])
def test_request_overrides_regenerate_and_reprice_selected_contracts(override_field):
    request = replace(_request(), **{override_field: {"strike": 105.0}})
    record = application.score_one(request, _override_native())

    assert {leg["strike"] for leg in record.selected_contracts} == {105.0}
    assert {leg["strike"] for leg in record.legs} == {105.0}
    assert record.financial_diagnostics["entry_cost_pct"] == pytest.approx(3.0)
    assert record.validation_status == "scored"


def test_invalid_geometry_override_preserves_native_refusal():
    request = replace(
        _request("TWIN-P"),
        geometry_override={"width": 0.0},
    )
    record = application.score_one(request, _override_native("TWIN-P"))

    assert record.validation_status == "refused"
    assert "ZERO_WIDTH" in record.reason_codes
    assert record.selected_contracts == ()


def test_batch_uses_full_request_identity_and_rejects_ambiguous_legacy_keys():
    first = replace(_request(), geometry_override={"strike": 100.0})
    second = replace(_request(), geometry_override={"strike": 105.0})
    batch = ScoreBatch(batch_id="b1", requests=(first, second), population_ref="p1")
    inputs = _override_native()

    records = application.score_batch(batch, {
        request_hash(first): inputs,
        request_hash(second): inputs,
    })

    assert [{leg["strike"] for leg in record.selected_contracts}
            for record in records] == [{100.0}, {105.0}]
    with pytest.raises(KeyError, match="ambiguous event/strategy"):
        application.score_batch(
            batch, {(first.event_id, first.strategy_version): inputs},
        )
    with pytest.raises(KeyError, match="ambiguous legacy event-only"):
        application.score_batch(batch, {first.event_id: inputs})


def test_dynamic_requires_simulation_before_ranking_and_preserves_flags():
    no_simulation = application.score_one(
        _request("TWIN-P"),
        replace(
            _native("TWIN-P", chooser_score=0.9, exp_pnl=None),
            source_ref="compatibility-input",
        ),
    )
    eligible = application.score_one(
        _request("TWIN-P5"),
        replace(
            _native(
                "TWIN-P5", chooser_score=0.2, exp_pnl=0.1,
                flags=("ADVISORY",),
            ),
            source_ref="compatibility-input",
        ),
    )

    chosen = application._choose_dynamic(
        _request("DYN-SV"), (no_simulation, eligible),
    )

    assert chosen.chooser_selection["strategy"] == "TWIN-P5"
    assert chosen.chooser_selection["ranking_key"] == "chooser_score"
    assert chosen.chooser_selection["menu_size"] == 1
    assert chosen.reason_codes == ("ADVISORY",)


def test_score_one_executes_pricing_and_owns_refusal_lineage_and_receipts():
    record = application.score_one(_request(), _native(forged=True))

    assert record.financial_diagnostics["entry_cost_pct"] == pytest.approx(3.0)
    assert {leg["name"] for leg in record.legs} == {"call", "put"}
    assert record.model_artifact_ids == ("model-a",)
    receipts = record.resolved_request["native_stage_receipts"]
    assert {item["stage"] for item in receipts} == set(STAGE_NAMES)
    assert all(item["output_hash"] != "declared-output" for item in receipts)


def test_dynamic_uses_one_ranking_rule_and_does_not_veto_flagged_candidate():
    scored = application.score_one(
        _request("TWIN-P"),
        replace(_native("TWIN-P", chooser_score=0.2,
                        exp_pnl=0.1, flags=("ADVISORY",)),
                source_ref="compatibility-input"),
    )
    unscored = application.score_one(
        _request("TWIN-P5"),
        replace(_native("TWIN-P5", chooser_score=None, exp_pnl=0.9),
                source_ref="compatibility-input"),
    )
    chosen = application._choose_dynamic(_request("DYN-SV"), (scored, unscored))

    assert chosen.chooser_selection["strategy"] == "TWIN-P"
    assert chosen.chooser_selection["ranking_key"] == "chooser_score"
    assert chosen.reason_codes == ("ADVISORY",)


def test_batch_keys_inputs_by_event_and_strategy():
    first = _request("STR-THRU")
    second = replace(first, strategy_version="STR-RUNUP")
    batch = ScoreBatch(batch_id="b1", requests=(first, second), population_ref="p1")

    records = application.score_batch(batch, {
        (first.event_id, first.strategy_version): _native("STR-THRU"),
        (second.event_id, second.strategy_version): _native("STR-RUNUP"),
    })

    assert [record.canonical_request["strategy_version"] for record in records] == [
        "STR-THRU", "STR-RUNUP",
    ]
    with pytest.raises(KeyError, match="ambiguous legacy event-only"):
        application.score_batch(batch, {first.event_id: _native("STR-THRU")})


def test_score_event_keeps_strategies_after_dynamic_menu():
    dynamic = {member: _native(member, chooser_score=index / 10.0)
               for index, member in enumerate(DYNAMIC_MENU)}
    records = application.score_event(
        _request("DYN-SV"),
        (("DYN-SV", {"menu": dynamic}), ("STR-THRU", _native("STR-THRU"))),
    )

    assert tuple(record.canonical_request["strategy_version"] for record in records) == (
        "DYN-SV", "STR-THRU",
    )


def test_frozen_inference_preserves_refusals_roles_and_verified_artifacts():
    binding = SimpleNamespace(binding_id="b1", role="size", output_names=("prediction",))
    release = SimpleNamespace(bindings=(binding,))
    inference_request = SimpleNamespace(binding_id="b1")

    class Frozen:
        def infer(self, release, inference_request):
            return SimpleNamespace(
                status="READY", binding_id="b1", output_names=("prediction",),
                predictions=((0.42,),), artifact_hashes=("sha256:verified",),
                reason_codes=(), detail=None,
            )

    record = application.score_frozen(
        _request(), Frozen(), release, inference_request,
        {"spot": 100.0, "implied_move": 6.0, "driver_name": "abs_move",
         "driver_prediction": 0.77, "forecast_abs_move": 9.0,
         "flags": ("UNVALIDATED_STRUCTURE",), "model_inputs": {}},
    )

    assert record.forecasts["forecast_abs_move"] == pytest.approx(0.42)
    assert record.forecasts["driver_prediction"] is None
    assert "UNVALIDATED_STRUCTURE" in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:driver" in record.reason_codes
    assert "MISSING_SIMULATION_INPUT" in record.reason_codes
    assert "MISSING_GATE_INPUT" in record.reason_codes
    assert "MISSING_EXPIRY" in record.reason_codes
    assert record.validation_status == "refused"
    assert record.model_artifact_ids == ("sha256:verified",)
