import json
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import Pricing, generate, price
from engine.v2.foundation import to_document
from engine.v2.models import no_fit
from engine.v2.ops.cli import rescore_command
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, StageReceipt


def _receipts():
    return tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics"
    )


def _write_fixture(tmp_path):
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
    inputs = NativeScoreInputs(
        context=context, features=features, forecast=forecast, geometry=stale_geometry,
        pricing=pricing, analogs={}, simulation={}, gate={}, chooser={}, diagnostics={},
        source_ref="typed-native-fixture", stage_receipts=_receipts(),
    )
    request_path = tmp_path / "request.json"
    inputs_path = tmp_path / "native_inputs.json"
    request_path.write_text(json.dumps(to_document(request)))
    inputs_path.write_text(json.dumps(to_document(inputs)))
    return request_path, inputs_path


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