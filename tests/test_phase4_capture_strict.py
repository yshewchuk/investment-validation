from __future__ import annotations

import copy
import hashlib
import json

import joblib
import pandas as pd
import pytest

from checks import phase4_real
from engine.fills import MID
from engine.score import ScoreRequest
from engine.v2.contracts import ScoreRequest as V2ScoreRequest
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.stages import NativeScoreInputs, receipt
from tools.capture_tier0_corpus import (
    STRICT_TRACE_SUPPORTED_STRATEGIES,
    StrictTraceCaptureError,
    _frozen_runtime,
    _role_feature_vectors,
    attach_strict_probe,
    canonical_v2_request,
    main,
    make_pair,
    native_inputs_from_capture,
    package_strict_trace,
    parse_strategies,
    request_to_dict,
    write,
)
from tools.phase4_frozen_resources import package_frozen_resources


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


def test_strict_cli_accepts_every_dyn_sv_menu_strategy_alongside_str_thru():
    # R4-1 originally restricted --strict-phase4-trace to exactly one of
    # STR-THRU/STR-RUNUP. That restriction was lifted: native scoring
    # (engine.v2.scoring.stages / source_inputs) has supported the DYN-SV
    # menu strategies since R4-6, and capture_tier0_corpus.py's own
    # STRICT_TRACE_SUPPORTED_STRATEGIES now matches. A multi-strategy
    # request, including a disabled/unsupported name, must still be caught
    # by the ordinary --strategies validation, not a strict-trace-specific
    # combination check.
    assert parse_strategies(
        ["STR-THRU", "STR-RUNUP", "TWIN-P", "TWIN-P5", "CND-PS",
         "BFLY-P", "BFLY-P5", "RAMP7", "CTR5"],
    ) == (
        "STR-THRU", "STR-RUNUP", "TWIN-P", "TWIN-P5", "CND-PS",
        "BFLY-P", "BFLY-P5", "RAMP7", "CTR5",
    )


def test_strict_cli_requires_known_strategies_before_building_scorer(monkeypatch):
    monkeypatch.setattr(
        "tools.capture_tier0_corpus.score_mod.Scorer",
        lambda: pytest.fail("argument refusal must precede scorer construction"),
    )

    with pytest.raises(SystemExit):
        main(["--strict-phase4-trace", "--strategies", "NOT-A-STRATEGY"])


# --------------------------------------------------------------------------
# R4-1: frozen-inference row assembly must read each binding's OWN per-role
# captured feature vector, not the cross-role merged dict. Reproduces the
# defect measured on the real corpus (`/tmp/phase4-strict-bounded2`,
# 001_STR-THRU-ABBV-2024-02-02_6f33a1ba): the gate binding's `feature_order`
# names features (e.g. `n_prior`) that live only in the `gate_inputs`
# checkpoint's own feature_vector, never in the `features` checkpoint the
# forecast-family roles share.
# --------------------------------------------------------------------------


class _SumEstimator:
    """A stand-in artifact: predicts the sum of one row's features."""

    def predict(self, rows):
        return [sum(row) for row in rows]


def _artifact(source, name: str = "model.joblib") -> tuple:
    path = source / name
    joblib.dump(_SumEstimator(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def _model_binding(path, digest, *, role, feature_order, output_names, model_id):
    return {
        "model_id": model_id,
        "artifact": path.name,
        "artifact_sha256": digest,
        "role": role,
        "feature_order": feature_order,
        "output_names": output_names,
        "strategy": "STR-THRU",
        "decision_clock": "entry-close",
        "adapter": "joblib-estimator.v1",
    }


def _legacy_trace(*, driver_role: str, driver_vector: dict, gate_vector: dict | None):
    features_value = {
        "feature_vector": {driver_role: driver_vector},
        "missing_mask": {driver_role: {k: False for k in driver_vector}},
        "model_identity": {driver_role: {"model_id": "driver-1"}},
    }
    checkpoints = {
        "features": {"value": features_value, "content_hash": content_hash(features_value)},
    }
    if gate_vector is not None:
        gate_value = {
            "kind": "model",
            "model_identity": "gate-1",
            "feature_vector": gate_vector,
            "threshold": 0.0,
        }
        checkpoints["gate_inputs"] = {
            "value": gate_value, "content_hash": content_hash(gate_value),
        }
    return {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": checkpoints,
    }


def test_role_feature_vectors_keep_each_role_separate_and_alias_legacy_names():
    # The gate model's own vector carries `x` at a DIFFERENT value than the
    # driver's own vector. `_merged_model_inputs` would refuse this as a
    # cross-role conflict; `_role_feature_vectors` must not merge at all.
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="abs_move", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0},
    )}

    vectors = _role_feature_vectors(candidate)

    assert vectors == {"driver": {"x": 2.0}, "gate": {"x": 9.0, "n_prior": 5.0}}


def test_frozen_runtime_builds_each_row_from_its_own_bindings_role(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    driver_binding = _model_binding(
        path, digest, role="abs_move", feature_order=["x"],
        output_names=["driver_prediction"], model_id="driver-1",
    )
    # The gate binding's feature_order (50 features in the real corpus row)
    # exceeds and differs from the driver/size vector: `n_prior` has no
    # counterpart there at all.
    gate_binding = _model_binding(
        path, digest, role="gate", feature_order=["x", "n_prior"],
        output_names=["gate_score"], model_id="gate-1",
    )
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="abs_move", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0},
    )}
    package = package_frozen_resources(
        model_bindings=[driver_binding, gate_binding],
        deployment_id="deployment-1",
        release_root=tmp_path / "release",
        source_root=source,
    )
    request = _request()
    inputs, _ = _native(request)

    _, release, inference_requests = _frozen_runtime(
        package, tmp_path / "release", request, inputs, candidate,
    )

    rows_by_role = {
        binding.role: next(
            item.rows for item in inference_requests
            if item.binding_id == binding.binding_id
        )
        for binding in release.bindings
    }
    assert rows_by_role["driver"] == ((2.0,),)
    assert rows_by_role["gate"] == ((9.0, 5.0),)


def test_frozen_runtime_refuses_a_binding_missing_a_feature_by_name(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    gate_binding = _model_binding(
        path, digest, role="gate", feature_order=["x", "n_prior", "missing_feat"],
        output_names=["gate_score"], model_id="gate-1",
    )
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="abs_move", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0},
    )}
    package = package_frozen_resources(
        model_bindings=[gate_binding],
        deployment_id="deployment-1",
        release_root=tmp_path / "release",
        source_root=source,
    )
    request = _request()
    inputs, _ = _native(request)

    with pytest.raises(StrictTraceCaptureError, match=r"gate.*missing_feat") as excinfo:
        _frozen_runtime(package, tmp_path / "release", request, inputs, candidate)
    assert not isinstance(excinfo.value, KeyError)


def _full_strict_candidate(*, fixture_id, ticker, driver_vector, gate_vector,
                           path, digest, strategy="STR-THRU"):
    expiry = "2026-09-18"
    event_date = "2026-09-17"
    legacy_request = ScoreRequest(
        ticker=ticker, strategy=strategy,
        as_of=pd.Timestamp("2026-09-16"), event_date=pd.Timestamp(event_date),
        session="AMC", fill=MID,
    )
    driver_binding = _model_binding(
        path, digest, role="abs_move", feature_order=list(driver_vector),
        output_names=["driver_prediction"], model_id=f"driver-{ticker}",
    )
    gate_binding = _model_binding(
        path, digest, role="gate", feature_order=list(gate_vector),
        output_names=["gate_score"], model_id=f"gate-{ticker}",
    )
    source_inputs_value = {
        "context": {
            "ticker": ticker, "event_date": event_date,
            "entry_date": "2026-09-16", "exit_date": expiry,
            "expiry": expiry, "spot": 100.0,
        },
        "quote_domain": [
            {"right": "C", "strike": 100.0, "expiry": expiry, "bid": 1.0, "ask": 2.0},
            {"right": "P", "strike": 100.0, "expiry": expiry, "bid": 1.0, "ask": 2.0},
        ],
        "features": {},
        "model_bindings": [driver_binding, gate_binding],
        "native_recipes": {"forecast": {"models": {}}},
    }
    legacy_trace = _legacy_trace(
        driver_role="abs_move", driver_vector=driver_vector, gate_vector=gate_vector,
    )
    legacy_trace["checkpoints"]["source_inputs"] = {
        "value": source_inputs_value,
        "content_hash": content_hash(source_inputs_value),
    }
    return {
        "fixture_id": fixture_id,
        "covers": [],
        "kind": "score_result",
        "record": {"strategy": strategy, "ticker": ticker},
        "request": request_to_dict(legacy_request),
        "duration": 0.1,
        "relations": {},
        "legacy_trace": legacy_trace,
        "event_id": f"{ticker}_{event_date}",
    }


def test_attach_strict_probe_traces_every_selected_candidate(tmp_path, monkeypatch):
    # attach_strict_probe resolves captured artifact paths beneath the
    # module's own ROOT; point it at this test's tmp_path instead of the
    # real repo so the artifact stays self-contained.
    monkeypatch.setattr("tools.capture_tier0_corpus.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    chosen = [
        _full_strict_candidate(
            fixture_id=f"case-{i}", ticker=ticker,
            driver_vector={"x": 2.0 + i}, gate_vector={"x": 9.0 + i, "n_prior": 5.0 + i},
            path=path, digest=digest,
        )
        for i, ticker in enumerate(("AAA", "BBB"))
    ]

    attached, gaps = attach_strict_probe(chosen, "snapshot-1", tmp_path / "release")

    assert set(attached) == {"case-0", "case-1"}
    assert gaps == {}
    for candidate in chosen:
        assert "input_trace" in candidate
        assert candidate["legacy_input_hash"] == candidate["input_trace"]["shared_input_hash"]
        assert "serialization" in candidate["input_trace"]["stages"]
    score_ids = {candidate["native_score_id"] for candidate in chosen}
    assert len(score_ids) == 2


# --------------------------------------------------------------------------
# Multi-strategy capture and honest per-case gaps: the CLI used to require
# exactly one of STR-THRU/STR-RUNUP, and attach_strict_probe used to abort
# the WHOLE batch if even one selected row of the target strategy could not
# produce a trace (the real-corpus incident: 2 unsupported rows out of 48
# candidates discarded all 46 that would have traced cleanly). Both
# restrictions are lifted: every score_result row is attempted regardless of
# strategy, a row that cannot trace is recorded with its typed reason, and
# rows that can trace still do, in the same run.
# --------------------------------------------------------------------------


def test_attach_strict_probe_traces_every_dyn_sv_menu_strategy_in_one_pass(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.capture_tier0_corpus.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    strategies = ("STR-THRU", "STR-RUNUP", "TWIN-P", "CTR5")
    assert set(strategies) <= STRICT_TRACE_SUPPORTED_STRATEGIES
    chosen = [
        _full_strict_candidate(
            fixture_id=f"case-{i}", ticker=f"T{i}", strategy=strategy,
            driver_vector={"x": 2.0 + i}, gate_vector={"x": 9.0 + i, "n_prior": 5.0 + i},
            path=path, digest=digest,
        )
        for i, strategy in enumerate(strategies)
    ]

    attached, gaps = attach_strict_probe(chosen, "snapshot-1", tmp_path / "release")

    assert set(attached) == {f"case-{i}" for i in range(len(strategies))}
    assert gaps == {}


def test_attach_strict_probe_records_a_typed_gap_and_still_traces_the_rest(tmp_path, monkeypatch):
    # CAL-P is disabled (research-only): canonical_v2_request must refuse it
    # by name, and that refusal must land as a gap on ONLY that row.
    monkeypatch.setattr("tools.capture_tier0_corpus.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    good = _full_strict_candidate(
        fixture_id="case-good", ticker="AAA", strategy="STR-THRU",
        driver_vector={"x": 2.0}, gate_vector={"x": 9.0, "n_prior": 5.0},
        path=path, digest=digest,
    )
    unsupported = _full_strict_candidate(
        fixture_id="case-bad", ticker="ZZZ", strategy="CAL-P",
        driver_vector={"x": 2.0}, gate_vector={"x": 9.0, "n_prior": 5.0},
        path=path, digest=digest,
    )

    attached, gaps = attach_strict_probe(
        [good, unsupported], "snapshot-1", tmp_path / "release",
    )

    assert attached == ("case-good",)
    assert set(gaps) == {"case-bad"}
    assert "CAL-P" in gaps["case-bad"]
    assert "input_trace" in good
    assert "input_trace" not in unsupported


def test_attach_strict_probe_refuses_only_when_nothing_at_all_traced(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.capture_tier0_corpus.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    unsupported = _full_strict_candidate(
        fixture_id="case-bad", ticker="ZZZ", strategy="CAL-P",
        driver_vector={"x": 2.0}, gate_vector={"x": 9.0, "n_prior": 5.0},
        path=path, digest=digest,
    )

    with pytest.raises(StrictTraceCaptureError, match="no honest strict trace"):
        attach_strict_probe([unsupported], "snapshot-1", tmp_path / "release")


def test_make_pair_records_a_strict_trace_gap_without_fabricating_a_trace():
    pair = make_pair(
        "fixture-0", [], {"strategy": "CAL-P"}, {"strategy": "CAL-P"},
        record_kind="score_result", duration=0.1,
        legacy_trace={"checkpoints": {}},
        strict_trace_gap="strict probe does not support CAL-P",
    )

    assert pair["payload"]["trace_disposition"] == "gap"
    assert pair["payload"]["strict_trace_gap"] == "strict probe does not support CAL-P"
    assert "input_trace" not in pair["payload"]


def test_make_pair_refuses_to_carry_a_trace_and_a_gap_together():
    with pytest.raises(StrictTraceCaptureError, match="cannot carry both"):
        make_pair(
            "fixture-0", [], {}, {}, record_kind="score_result", duration=0.1,
            input_trace={"shared_input_hash": "h", "trace_hash": "t"},
            legacy_input_hash="h",
            strict_trace_gap="reason",
        )


def test_write_persists_a_gap_pair_and_a_traced_pair_in_the_same_corpus(tmp_path, monkeypatch):
    # End-to-end reproduction of the real-corpus incident: a batch with one
    # traceable row and one unsupported row must write BOTH pairs, the first
    # complete and the second an honest gap — never an empty output
    # directory and never a fabricated trace on the second.
    monkeypatch.setattr("tools.capture_tier0_corpus.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    good = _full_strict_candidate(
        fixture_id="case-good", ticker="AAA", strategy="STR-THRU",
        driver_vector={"x": 2.0}, gate_vector={"x": 9.0, "n_prior": 5.0},
        path=path, digest=digest,
    )
    unsupported = _full_strict_candidate(
        fixture_id="case-bad", ticker="ZZZ", strategy="CAL-P",
        driver_vector={"x": 2.0}, gate_vector={"x": 9.0, "n_prior": 5.0},
        path=path, digest=digest,
    )

    doc = write(
        tmp_path / "out", [good, unsupported], {}, pd.Timestamp("2026-09-16"),
        "snapshot-1", strict_trace=True,
    )

    assert doc["pairs"]["case-good"]["trace_disposition"] == "complete"
    assert doc["pairs"]["case-bad"]["trace_disposition"] == "gap"
    good_payload = json.loads(
        (tmp_path / "out" / "pairs" / "case-good.json").read_text())["payload"]
    bad_payload = json.loads(
        (tmp_path / "out" / "pairs" / "case-bad.json").read_text())["payload"]
    assert "input_trace" in good_payload
    assert "input_trace" not in bad_payload
    assert "CAL-P" in bad_payload["strict_trace_gap"]
