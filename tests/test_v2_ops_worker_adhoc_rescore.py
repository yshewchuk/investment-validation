import json

from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import Pricing, generate, price
from engine.v2.foundation import to_document
from engine.v2.models import no_fit
from engine.v2.ops import worker
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, StageReceipt


def _receipts():
    return tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics"
    )


def _write_staged_pair(tmp_path):
    request = ScoreRequest(
        event_id="evt-whatif", calendar_revision="cal-1", strategy_version="STR-THRU",
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
    (tmp_path / "request.json").write_text(json.dumps(to_document(request)))
    (tmp_path / "native_inputs.json").write_text(json.dumps(to_document(inputs)))


def test_adhoc_rescore_worker_writes_a_real_record(tmp_path):
    _write_staged_pair(tmp_path)
    result = worker.dispatch("adhoc_rescore", {"expected_ids": ["adhoc_rescore"]}, tmp_path)
    assert result["completed_ids"] == ["adhoc_rescore"]
    assert result["outputs"] == [{"name": "record", "path": "record.json",
                                  "schema": "adhoc_rescore_record.v1.0"}]
    document = json.loads((tmp_path / "record.json").read_text())
    assert document["canonical_request"]["event_id"] == "evt-whatif"
    assert no_fit.fitting_forbidden() is False  # released after the worker call