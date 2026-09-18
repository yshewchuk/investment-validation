from __future__ import annotations

import copy

import pandas as pd
import pytest

from checks import phase4_real
from engine.fills import MID
from engine.score import ScoreRequest
from engine.v2.contracts import ScoreRequest as V2ScoreRequest
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.stages import NativeScoreInputs, receipt
from tools.capture_tier0_corpus import (
    StrictTraceCaptureError,
    canonical_v2_request,
    main,
    native_inputs_from_capture,
    package_strict_trace,
    parse_strategies,
    request_to_dict,
)


def _request() -> V2ScoreRequest:
    return V2ScoreRequest(
        event_id="event-1",
        event_revision="event-revision-1",
        calendar_revision="calendar-1",
        strategy_version="STR-THRU",
        deployment_id="deployment-1",
        decision_clock_id="entry-close",
        requested_decision_at="2026-09-16",
        snapshot_id="snapshot-1",
        mode="replay",
        fill_model={"alpha": 0.5},
    )


def _native(request: V2ScoreRequest):
    expiry = "2026-09-18"
    blocks = {
        "context": {
            "ticker": "ABC",
            "strategy": "STR-THRU",
            "event_date": "2026-09-17",
            "entry_date": "2026-09-16",
            "exit_date": "2026-09-18",
            "expiry": expiry,
            "spot": 100.0,
            "quotes": {
                f"C:100.0:{expiry}": {"bid": 1.0, "ask": 2.0},
                f"P:100.0:{expiry}": {"bid": 1.0, "ask": 2.0},
            },
        },
        "features": {"model_inputs": {"x": 2.0}},
        "forecast": {
            "models": {
                "driver_prediction": {
                    "intercept": 1.0,
                    "coefficients": {"x": 1.0},
                },
            },
        },
        "geometry": None,
        "pricing": None,
        "analogs": {},
        "simulation": {
            "terminal_spots": [95.0, 105.0],
            "capital_at_risk": 3.0,
        },
        "gate": {
            "model": {
                "intercept": 0.0,
                "coefficients": {"x": 1.0},
            },
            "threshold": 0.0,
        },
        "chooser": {},
        "diagnostics": {},
    }
    shared = {"request": to_document(request), "native_inputs": copy.deepcopy(blocks)}
    source_ref = content_hash(shared)
    receipts = tuple(
        receipt(stage, {"source_ref": source_ref}, {"execution": "native-runtime"})
        for stage in (
            "resolve_context", "features", "forecast", "geometry", "pricing",
            "analogs", "simulation", "gate", "chooser", "serialization",
        )
    )
    return NativeScoreInputs(
        **blocks, source_ref=source_ref, stage_receipts=receipts,
    ), shared


def test_strategy_filter_preserves_default_and_accepts_commas():
    assert parse_strategies(None) is None
    assert parse_strategies(["STR-THRU,STR-RUNUP"]) == (
        "STR-THRU", "STR-RUNUP",
    )
    with pytest.raises(StrictTraceCaptureError, match="unknown strategies"):
        parse_strategies(["NOT-A-STRATEGY"])


def test_canonical_request_uses_captured_event_and_legacy_clock():
    legacy = ScoreRequest(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-09-16"),
        event_date=pd.Timestamp("2026-09-17"),
        session="AMC",
        fill=MID,
    )
    request = canonical_v2_request(
        {"event_id": "event-1", "request": request_to_dict(legacy)},
        "snapshot-1",
    )

    assert request.event_id == "event-1"
    assert request.strategy_version == "STR-THRU"
    assert request.requested_decision_at == "2026-09-16"
    assert request.fill_model["alpha"] == 0.5
    assert request.snapshot_id == "snapshot-1"


def test_capture_conversion_refuses_absent_executable_recipes():
    value = {
        "context": {"ticker": "ABC", "strategy": "STR-THRU"},
        "quote_domain": [],
        "features": {},
        "model_bindings": [],
    }
    candidate = {
        "legacy_trace": {
            "checkpoints": {
                "source_inputs": {
                    "value": value,
                    "content_hash": content_hash(value),
                },
            },
        },
    }

    with pytest.raises(StrictTraceCaptureError, match="native_recipes"):
        native_inputs_from_capture(candidate, _request())


def test_native_observer_packages_a_strict_verifiable_trace(tmp_path):
    request = _request()
    inputs, shared = _native(request)

    trace, native = package_strict_trace(request, inputs, shared)
    pair = {
        "payload": {
            "request": to_document(request),
            "record": {},
            "legacy_input_hash": trace["shared_input_hash"],
            "input_trace": trace,
            "input_trace_hash": trace["trace_hash"],
        },
    }
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)

    assert verified["captured_stages"][-1] == "serialization"
    assert to_document(native.canonical_request) == to_document(request)
    assert native.resolved_request["native_stage_receipts"][-1][
        "stage"
    ] == "serialization"


def test_strict_cli_refuses_before_building_scorer(monkeypatch, capsys):
    monkeypatch.setattr(
        "tools.capture_tier0_corpus.score_mod.Scorer",
        lambda: pytest.fail("strict refusal must precede scorer construction"),
    )

    with pytest.raises(SystemExit):
        main(["--strict-phase4-trace", "--strategies", "STR-THRU"])

    assert "source_inputs does not yet emit" in capsys.readouterr().err
