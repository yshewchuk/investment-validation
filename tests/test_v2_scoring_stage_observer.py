from engine.v2.domain.generation import generate, price
from engine.v2.foundation import content_hash
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    STAGE_NAMES,
    StageObservation,
    StageReceipt,
    assemble_native_values,
)


def _inputs() -> NativeScoreInputs:
    context = {
        "ticker": "AAA",
        "event_date": "2026-09-16",
        "entry_date": "2026-09-16",
        "exit_date": "2026-09-18",
        "expiry": "2026-09-18",
        "spot": 100.0,
    }
    forecast = {
        "models": {
            "driver_prediction": {"intercept": 7.0, "coefficients": {}},
        },
    }
    geometry = generate("STR-THRU", context)
    quotes = {
        (leg.right, leg.strike, leg.expiry): {"bid": 1.0, "ask": 3.0}
        for leg in geometry.legs
    }
    declared = tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics"
    )
    return NativeScoreInputs(
        context=context,
        features={"model_inputs": {}},
        forecast=forecast,
        geometry=geometry,
        pricing=price(geometry, quotes, 0.5),
        analogs={},
        simulation={"terminal_spots": (95.0, 105.0)},
        gate={
            "model": {"intercept": 1.0, "coefficients": {}},
            "threshold": 0.0,
        },
        chooser={},
        diagnostics={},
        source_ref="stage-observer-fixture",
        stage_receipts=declared,
    )


def test_observer_emits_stable_documents_and_published_receipts():
    observations: list[StageObservation] = []

    values = assemble_native_values(
        _inputs(), strategy="STR-THRU", observer=observations.append,
    )

    assert tuple(item.receipt.stage for item in observations) == STAGE_NAMES
    published = values["native_stage_receipts"]
    for item, public_receipt in zip(observations, published, strict=True):
        assert content_hash(item.input_document) == item.receipt.input_hash
        assert content_hash(item.output_document) == item.receipt.output_hash
        assert public_receipt == {
            "stage": item.receipt.stage,
            "input_hash": item.receipt.input_hash,
            "output_hash": item.receipt.output_hash,
            "owner": item.receipt.owner,
        }


def test_observer_is_opt_in_and_cannot_mutate_scoring_state():
    expected = assemble_native_values(_inputs(), strategy="STR-THRU")

    def mutate_snapshot(item: StageObservation) -> None:
        if isinstance(item.output_document, dict):
            item.output_document["observer_only"] = True

    actual = assemble_native_values(
        _inputs(), strategy="STR-THRU", observer=mutate_snapshot,
    )

    assert actual == expected
    assert "observer_only" not in actual
