from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import engine.score as score_module
from engine.pnl_sim import DRAWS, ResidualPool
from engine.score import Phase4TraceCollector, ScoreRequest, ScoreResult, Scorer
from engine.v2.foundation import content_hash


def test_collector_is_opt_in_and_default_finish_does_not_mutate_result() -> None:
    scorer = Scorer.__new__(Scorer)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
    )

    finished = scorer._finish_phase4_trace(None, result)

    assert finished is result
    assert not hasattr(result, "_phase4_trace")


def test_collector_records_json_safe_stage_documents_and_refusal_identity() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.begin(ScoreRequest(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
    ))
    collector.record(
        "features",
        {"frame": pd.DataFrame({"x": [1.0], "missing": [float("nan")]})},
        {"as_of": pd.Timestamp("2026-01-02")},
    )
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
    )
    result.flag("NO_CHAIN")
    collector.finish(result)

    document = collector.document()

    assert document["schema_version"] == "phase4_legacy_trace.v1.0"
    assert document["status"] == "refused"
    assert document["request"]["event_date"] == "2026-01-08"
    assert document["stages"]["features"]["input"]["frame"][0] == {
        "x": 1.0,
        "missing": None,
    }
    assert document["stages"]["serialization"]["status"] == "refused"


def test_compact_collector_does_not_retain_broad_stage_values() -> None:
    collector = Phase4TraceCollector(retain_full_trace=False)
    collector.begin(ScoreRequest(ticker="ABC", strategy="STR-THRU"))
    collector.record("features", {"large": ["discard-me"]}, {"value": 1.0})

    document = collector.document()

    assert document["stages"]["features"] == {"status": "completed"}


def test_corpus_capture_requests_only_bounded_checkpoint(monkeypatch) -> None:
    import tools.capture_tier0_corpus as capture

    class StubTrace:
        def __init__(self, *, retain_full_trace: bool, content_hasher) -> None:
            assert retain_full_trace is False
            assert content_hasher is not None

        def diagnostic_checkpoint(self) -> dict:
            return {"bounded": True}

        def document(self) -> dict:
            raise AssertionError("capture requested the broad trace")

    class StubResult:
        def as_dict(self) -> dict:
            return {"strategy": "STR-THRU"}

    class StubScorer:
        snapshot = "snapshot"

        def score(self, request, **kwargs):
            return StubResult()

    monkeypatch.setattr(capture.score_mod, "Phase4TraceCollector", StubTrace)
    request = ScoreRequest(ticker="ABC", strategy="STR-THRU")

    _, record, _, checkpoint = capture._score(StubScorer(), request)

    assert record == {"strategy": "STR-THRU"}
    assert checkpoint == {"bounded": True}


def _checkpoint_value(collector: Phase4TraceCollector, name: str) -> dict:
    row = collector.diagnostic_checkpoint()["checkpoints"][name]
    assert row["content_hash"] == content_hash(row["value"])
    return row["value"]


def test_model_boundary_captures_exact_vector_mask_and_identity() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    request = ScoreRequest(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
    )
    collector.begin(request)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
    )
    result._phase4_checkpoint_collector = collector
    artifact = SimpleNamespace(features=("exact", "missing"))
    entry = SimpleNamespace(id="size-v7")
    scorer = Scorer.__new__(Scorer)
    scorer.model = lambda role, *args, **kwargs: (
        (entry, artifact) if role == "size" else None
    )

    scorer._score_model(
        request,
        result,
        pd.DataFrame({"exact": [1.25], "missing": [float("nan")]}),
    )
    collector.finish(result)

    features = _checkpoint_value(collector, "features")
    assert features == {
        "feature_vector": {
            "abs_move": {"exact": 1.25, "missing": None},
        },
        "missing_mask": {
            "abs_move": {"exact": False, "missing": True},
        },
        "model_identity": {
            "abs_move": {
                "model_id": "size-v7",
                "role": "abs_move",
                "input_as_of": "2026-01-02",
            },
        },
    }
    checkpoint = collector.diagnostic_checkpoint()
    assert checkpoint["disposition"]["status"] == "refused"
    assert checkpoint["disposition"]["flags"] == ["MISSING_FEATURES"]
    assert "simulation" not in checkpoint["checkpoints"]
    assert "gate_inputs" not in checkpoint["checkpoints"]


def test_source_bundle_is_bounded_and_rejects_scoring_answers() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_source_bundle(
        context={"ticker": "ABC", "strategy": "STR-THRU"},
        quote_domain=[{"right": "call", "strike": 100.0, "bid": 1.0, "ask": 2.0}],
        features={"iv30": 0.25},
        model_bindings=({"role": "size", "model_id": "size-v7"},),
    )
    source = _checkpoint_value(collector, "source_inputs")

    assert source["context"] == {"ticker": "ABC", "strategy": "STR-THRU"}
    assert source["quote_domain"][0]["strike"] == 100.0
    assert source["model_bindings"][0]["role"] == "size"

    with pytest.raises(ValueError, match="scoring answers"):
        collector.capture_source_bundle(features={"entry_cost": 4.0})


def test_simulation_checkpoint_keeps_causal_residual_population() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_simulation(
        horizon={"event_date": "2026-01-02", "dte_exit": 5.0},
        capital_denominator=4.25,
        evidence={
            "draw_count": 4,
            "seed": 17,
            "residual_draw": {
                "cutoff": "2026-01-02",
                "cutoff_index": 2,
                "bucket_count": 10,
                "bucket_index": 4,
                "eligible_indices": [0, 1],
                "fallback_used": False,
            },
            "residual_rows": [{"event_date": "2025-12-01", "err_move": 1.0}],
            "residual_population": [
                {"event_date": "2025-11-01", "pred_abs_move": 4.0,
                 "err_move": 0.5, "err_crush": -2.0},
                {"event_date": "2025-12-01", "pred_abs_move": 5.0,
                 "err_move": 1.0, "err_crush": 3.0},
            ],
        },
    )

    simulation = _checkpoint_value(collector, "simulation")
    assert simulation["residual_population"] == [
        {"event_date": "2025-11-01", "pred_abs_move": 4.0,
         "err_move": 0.5, "err_crush": -2.0},
        {"event_date": "2025-12-01", "pred_abs_move": 5.0,
         "err_move": 1.0, "err_crush": 3.0},
    ]


def test_score_captures_boundary_legs_and_cost_before_later_mutation(
        monkeypatch) -> None:
    monkeypatch.setattr(score_module, "assert_decision_causal", lambda *args, **kwargs: None)
    scorer = Scorer.__new__(Scorer)
    scorer.snapshot = "snap-test"
    window = SimpleNamespace(
        entry_date=pd.Timestamp("2026-01-06"),
        exit_date=pd.Timestamp("2026-01-09"),
        decision_date=pd.Timestamp("2026-01-05"),
    )
    scorer.calendar = SimpleNamespace(
        resolve_offsets=lambda *args, **kwargs: window,
        is_projected=lambda value: False,
    )
    scorer._resolve_event = lambda request: (
        pd.Timestamp("2026-01-08"), "AMC",
    )
    scorer._structure = lambda request: SimpleNamespace(
        entry_offset=0,
        exit_offset=1,
        decision_offset=None,
    )
    expected_legs = [{
        "name": "long_put",
        "side": "buy",
        "right": "put",
        "qty": 1.0,
        "strike": 100.0,
        "expiry": "2026-01-16",
    }]

    def price_entry(request, structure, result, chain_index) -> None:
        result.legs = expected_legs
        result.entry_cost = 4.25
        result.spot = 100.0
        result.quote_date = result.entry_date

    scorer._price_entry = price_entry
    scorer._features = lambda request, result: pd.DataFrame({"x": [1.0]})
    scorer._quote_today = lambda ticker, as_of: None

    def score_model(request, result, features) -> None:
        result.exp_pnl_model = 0.1

    def mutate_after_pricing(request, result, features) -> None:
        result.entry_cost = 999.0
        result.legs = [{"name": "contradictory-final-record"}]

    scorer._score_model = score_model
    scorer._score_analogs = lambda request, result, features: None
    scorer._score_gate = mutate_after_pricing
    scorer._compare_layers = lambda result: None
    collector = Phase4TraceCollector(content_hasher=content_hash)

    scorer.score(
        ScoreRequest(
            ticker="ABC",
            strategy="STR-THRU",
            as_of=pd.Timestamp("2026-01-05"),
            event_date=pd.Timestamp("2026-01-08"),
            session="AMC",
        ),
        trace=collector,
    )

    selection = _checkpoint_value(collector, "selection_pricing")
    assert selection == {
        "selected_legs": expected_legs,
        "entry_cost": 4.25,
    }


def _residual_pool() -> ResidualPool:
    index = np.arange(600)
    return ResidualPool(pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=600, freq="D"),
        "pred_abs_move": 2.0 + index / 100.0,
        "err_move": index / 200.0 - 1.5,
        "err_crush": 4.0 - index / 100.0,
    }), buckets=2)


def test_simulation_boundary_is_bounded_and_records_reproducibility() -> None:
    scorer = Scorer.__new__(Scorer)
    scorer._residual_pool = _residual_pool
    scorer._pre_print_iv = lambda request, result, features: 42.0
    scorer._crush_forecast = lambda request, result, features: -18.0
    collector = Phase4TraceCollector(content_hasher=content_hash)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2022-01-01"),
        event_date=pd.Timestamp("2022-01-01"),
        exit_date=pd.Timestamp("2022-01-02"),
        expiry=pd.Timestamp("2022-01-10"),
        spot=100.0,
        entry_cost=4.25,
        forecast_abs_move=3.0,
    )
    result._priced_legs = (
        SimpleNamespace(strike=100.0, qty=1.0, side="buy"),
    )
    result._phase4_trace_requested = True
    result._phase4_checkpoint_collector = collector

    simulated = scorer._expectation(
        ScoreRequest(
            ticker="ABC",
            strategy="STR-THRU",
            event_date=pd.Timestamp("2022-01-01"),
        ),
        result,
    )

    assert simulated is not None
    simulation = _checkpoint_value(collector, "simulation")
    assert simulation["horizon"] == {
        "exit_date": "2022-01-02",
        "expiry": "2022-01-10",
        "dte_exit": 8.0,
    }
    assert simulation["capital_denominator"] == 4.25
    assert simulation["draw_count"] == DRAWS
    assert type(simulation["seed"]) is int
    identity = simulation["residual_population_identity"]
    assert identity["cutoff_index"] == 600
    assert identity["population_hash"].startswith("sha256:")
    assert identity["selected_rows_hash"].startswith("sha256:")
    encoded = json.dumps(collector.diagnostic_checkpoint())
    assert "residual_rows" not in encoded
    assert "selected_indices" not in encoded


def test_model_gate_and_dyn_sv_capture_actual_consumed_vectors() -> None:
    collector = Phase4TraceCollector(content_hasher=content_hash)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
    )
    result._phase4_checkpoint_collector = collector
    gate_entry = SimpleNamespace(id="gate-v3", threshold=0.6)
    gate_artifact = SimpleNamespace(
        features=("gate_x",),
        predict=lambda values: np.asarray([0.75]),
    )
    scorer = Scorer.__new__(Scorer)
    scorer.model = lambda role, *args, **kwargs: (
        (gate_entry, gate_artifact) if role == "gate" else None
    )
    scorer._gate_in_domain = lambda request, features: True
    scorer._gate_feature_frame = (
        lambda request, scored, features, wanted: features
    )

    scorer._score_gate(
        ScoreRequest(ticker="ABC", strategy="STR-THRU"),
        result,
        pd.DataFrame({"gate_x": [2.5]}),
    )

    assert _checkpoint_value(collector, "gate_inputs") == {
        "kind": "model",
        "model_identity": "gate-v3",
        "feature_vector": {"gate_x": 2.5},
        "threshold": 0.6,
    }

    chooser = Phase4TraceCollector(content_hasher=content_hash)
    chooser_result = ScoreResult(
        ticker="ABC",
        strategy="TWIN-P5",
        as_of=pd.Timestamp("2026-01-02"),
    )
    chooser_result._phase4_checkpoint_collector = chooser
    chooser_entry = SimpleNamespace(id="chooser-v2")
    chooser_artifact = SimpleNamespace(
        features=("rank_x",),
        model=SimpleNamespace(predict=lambda values: np.asarray([0.9])),
    )
    scorer.model = lambda role, *args, **kwargs: (
        (chooser_entry, chooser_artifact) if role == "chooser" else None
    )
    scorer._chooser_frame = (
        lambda request, scored, features, wanted: {"rank_x": 7.0}
    )

    scorer._score_chooser(
        ScoreRequest(ticker="ABC", strategy="TWIN-P5"),
        chooser_result,
        pd.DataFrame(),
    )

    assert _checkpoint_value(chooser, "dyn_sv") == {
        "eligibility": {
            "eligible": True,
            "candidate_strategy": "TWIN-P5",
            "missing_features": [],
        },
        "ranking": {
            "model_identity": "chooser-v2",
            "feature_vector": {"rank_x": 7.0},
            "score": 0.9,
        },
    }


def test_refusal_omits_unexecuted_groups_and_default_path_is_invariant() -> None:
    scorer = Scorer.__new__(Scorer)
    scorer.snapshot = "snap-test"
    request = ScoreRequest(
        ticker="ABC",
        strategy="CAL-P",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
        session="AMC",
    )

    default = scorer.score(request)
    collector = Phase4TraceCollector(content_hasher=content_hash)
    traced = scorer.score(request, trace=collector)

    assert default.as_dict() == traced.as_dict()
    assert not hasattr(default, "_phase4_trace")
    checkpoint = collector.diagnostic_checkpoint()
    assert checkpoint["schema_version"] == (
        "phase4_legacy_diagnostic_checkpoint.v1.0"
    )
    assert checkpoint["disposition"]["status"] == "refused"
    assert checkpoint["disposition"]["flags"] == ["UNVALIDATED_STRUCTURE"]
    assert checkpoint["checkpoints"] == {}
