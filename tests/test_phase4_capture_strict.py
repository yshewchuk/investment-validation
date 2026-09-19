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
    _merged_model_inputs,
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
            "model", "analogs", "simulation", "gate", "chooser", "serialization",
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


def test_merged_model_inputs_refuses_a_genuinely_absent_feature():
    """Characterizes the STR-THRU `has_implied_quote` gap (not a collector bug).

    `engine.features.add_quote_indicators` guarantees `has_implied_quote` is
    ALWAYS 0.0/1.0 whenever its source column (`or_implied`) exists in the
    frame -- NaN coerces to `values > 0 == False == 0.0`, never to a null
    indicator (see the direct check in test_features_quote_indicator.py). So
    a `None` captured for it, as here, means the whole market block never
    landed in `built` for this row -- the SAME condition that makes legacy's
    own `missing = [f for f in artifact.features if f not in features.columns]`
    check (engine/score.py `_score_model`) flag MISSING_FEATURES and decline
    to score. The strict validator is right to refuse this row rather than
    coerce a fabricated value; `attach_strict_probe` is designed to record
    exactly this as a typed per-pair gap (see
    test_attach_strict_probe_records_a_typed_gap_and_still_traces_the_rest)
    rather than fabricate a value or abort the whole capture.
    """
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="abs_move",
        driver_vector={"has_implied_quote": None, "mcap_log": 10.0},
        gate_vector=None,
    )}

    with pytest.raises(
        StrictTraceCaptureError,
        match=r"feature abs_move\.has_implied_quote is missing or nonnumeric",
    ):
        _merged_model_inputs(candidate)


def test_merged_model_inputs_accepts_the_boolean_quote_indicator_when_present():
    """The companion case: once `or_implied` exists, the indicator is a real
    0.0/1.0 float and the same validator accepts it -- confirming the failure
    above is about genuine absence, not about the value being boolean-shaped.
    """
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="abs_move",
        driver_vector={"has_implied_quote": 0.0, "mcap_log": 10.0},
        gate_vector=None,
    )}

    merged = _merged_model_inputs(candidate)

    assert merged == {"has_implied_quote": 0.0, "mcap_log": 10.0}


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


# ---------------------------------------------------------------------------
# R4-18/R4-19: the frozen bindings and inputs native scoring needs.
#
# Each test runs the REAL legacy method with a collector attached, builds a
# SourceBundle from the recorded ``source_inputs.frozen`` only (through
# ``frozen_source_declarations``), and compares native to legacy.
# ---------------------------------------------------------------------------

import importlib  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import engine.score as score_mod  # noqa: E402
import tests.test_v2_scoring_native_chooser_features as cf  # noqa: E402
import tests.test_v2_scoring_r4_20_frozen_parity as r4  # noqa: E402
from engine.data.features import tier4  # noqa: E402
from engine.models import registry  # noqa: E402
from engine.score import Phase4TraceCollector, Scorer  # noqa: E402
from engine.v2.foundation.canonical import untag_nonfinite  # noqa: E402
from engine.v2.models.admissible_table import legacy_n_admissible_table  # noqa: E402
from engine.v2.scoring import application  # noqa: E402
from engine.v2.scoring.source_inputs import (  # noqa: E402
    SourceBundle,
    build_native_score_inputs,
)
from tools.capture_tier0_corpus import (  # noqa: E402
    CHOOSER_PRIMITIVE_COLUMNS,
    frozen_source_declarations,
)
from tools.phase5_prepare_release import CHOOSER_POOL_ID  # noqa: E402

FOLD = pd.Timestamp("2026-09-01")


def _sha(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _collector() -> Phase4TraceCollector:
    collector = Phase4TraceCollector(retain_full_trace=False, content_hasher=content_hash)
    # Legacy records the context first; `capture_frozen` refreshes the group.
    collector.capture_source_bundle(context={"ticker": "AAA"})
    return collector


def _candidate(collector: Phase4TraceCollector) -> dict:
    return {"legacy_trace": collector.diagnostic_checkpoint()}


def _frozen(candidate: dict) -> dict:
    row = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    assert row["content_hash"] == content_hash(row["value"])
    return row["value"]["frozen"]


def _dump_fold(role, estimator, pool, floor=0.0):
    """One Tier-4 fold cache at the path ``ServingModel.artifact_ref`` names."""
    path = tier4._serving_path(f"{role}_synthetic", FOLD, "abc123")
    joblib.dump({"estimator": estimator, "model_id": f"{role}_synthetic",
                 "fold_start": "2026-09-01", "tier3_snapshot": "abc123",
                 "features": list(cf.FOLD_FEATURES), "pool_pred": pool[0],
                 "pool_res": pool[1], "interval_floor": floor}, path)
    return path


# -- the gate: its binding, output, forecast fold and pool ---------------------------


@pytest.fixture
def gate_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tier4, "SERVING_DIR", root)
    joblib.dump(r4._artifact(r4._hgbr(len(r4.GATE_FEATURES), 2), "gate", r4.GATE_FEATURES),
                root / "gate.joblib")
    pool = r4._pool(3000)
    joblib.dump({"estimator": r4._hgbr(len(r4.FOLD_FEATURES), 3), "model_id": "size_synthetic",
                 "fold_start": "2026-09-01", "tier3_snapshot": "abc123",
                 "features": list(r4.FOLD_FEATURES), "pool_pred": pool[0],
                 "pool_res": pool[1]}, root / "size_fold.joblib")
    shutil.copy(root / "size_fold.joblib",
                tier4._serving_path("size_synthetic", FOLD, "abc123"))
    return root, pool


def _captured_gate(root, pool, row):
    collector = _collector()
    scorer = object.__new__(Scorer)
    artifact = registry.load_artifact(root / "gate.joblib")
    entry = SimpleNamespace(id="gate_synthetic", threshold=0.5, decision_offset=None,
                            path=root / "gate.joblib",
                            artifact_sha256=_sha(root / "gate.joblib"))
    scorer.model = lambda *args, **kwargs: (entry, artifact)
    scorer._gate_in_domain = lambda request, features: True
    scorer._serving = lambda fold, produces="pred_abs_move": r4._served(root, pool)
    result = r4._Result(exp_pnl_analog=0.1, win_analog=0.5, n_analogs=3,
                        _phase4_checkpoint_collector=collector)
    Scorer._score_gate(scorer, SimpleNamespace(strategy="STR-THRU", decision_offset=None),
                       result, pd.DataFrame([row]))
    return result, _candidate(collector)


def _gate_native(declared):
    bundle = SourceBundle(
        source_ref="capture-gate", strategy="STR-THRU", context=dict(r4.STR_THRU_CONTEXT),
        raw_quotes=r4.STR_THRU_QUOTES, feature_vector=declared["feature_vector"],
        feature_missing_mask=declared["feature_missing_mask"],
        model_identity={"driver": {"model_id": "synthetic"}},
        forecast_recipes={"driver_prediction": {"intercept": 6.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:driver"},
        residual_recipe={}, gate_recipe=declared["gate_recipe"],
        gate_forecast_pool=declared.get("gate_forecast_pool", {}),
        frozen_inference=declared["frozen_inference"],
        model_release=declared["model_release"], **r4.ANALOG)
    return r4._score(bundle, "STR-THRU")


def _declared(candidate, tmp_path, **extra):
    return frozen_source_declarations(
        candidate, release_root=tmp_path / "release", deployment_id="dep-1",
        source_root=tmp_path, **extra)


def test_gate_capture_records_binding_output_forecast_fold_and_pool(gate_root):
    root, pool = gate_root
    _result, candidate = _captured_gate(root, pool, r4.ROW)
    frozen = _frozen(candidate)

    gate = frozen["bindings"]["gate"]
    assert (gate["role"], gate["output_names"], gate["adapter"]) == (
        "gate", ["gate_score"], "joblib-estimator.v1")
    assert gate["artifact_sha256"] == _sha(root / "gate.joblib")
    # Only the base-frame columns: the forecast/analog ones are derived.
    assert frozen["inputs"]["gate@gate"] == {"mean_prior_abs_move": 6.2, "iv30": 55.0,
                                             "im": 5.5}
    fold = frozen["bindings"]["fold:size"]
    served_path = tier4._serving_path("size_synthetic", FOLD, "abc123")
    assert (fold["role"], fold["adapter"], fold["output_names"]) == (
        "size", "tier4-serving-fold.v1", ["pred_abs_move"])
    assert (fold["artifact"], fold["artifact_sha256"]) == (
        str(served_path), tier4.store.file_sha256(served_path))
    assert fold["feature_order"] == list(r4.FOLD_FEATURES)
    assert frozen["inputs"]["fold:size@gate"] == {
        name: r4.ROW[name] for name in r4.FOLD_FEATURES}
    recorded = frozen["fold_pools"]["pred_abs_move"]
    assert recorded["predictions"] == pool[0].tolist()
    assert recorded["residuals"] == pool[1].tolist()
    assert recorded["interval_floor"] == 0.0
    assert frozen["declarations"]["gate"] == {
        "binding": "gate", "output": "gate_score", "threshold": 0.5, "site": "gate"}
    assert frozen["declarations"]["gate_forecast"] == {
        "binding": "fold:size", "output": "pred_abs_move", "pool": "pred_abs_move",
        "site": "gate"}


def test_gate_from_the_captured_bundle_equals_legacy(gate_root, tmp_path):
    root, pool = gate_root
    _result, candidate = _captured_gate(root, pool, r4.ROW)
    declared = _declared(candidate, tmp_path)
    assert declared["gate_recipe"]["forecast"]["output"] == "pred_abs_move"
    record, seen = _gate_native(declared)
    legacy = r4._legacy_gate(root, r4.ROW, r4._analog_outputs(record), pool)
    assert legacy.gate_score is not None, legacy.flags
    assert seen["gate"]["gate_score"] == legacy.gate_score
    assert seen["gate"]["gate_pass"] == legacy.gate_pass

    # Planted defect: a pool other than the served one moves the band and
    # therefore the score.
    other = dict(declared, gate_forecast_pool={
        **declared["gate_forecast_pool"],
        "residuals": [2.0 * value for value in declared["gate_forecast_pool"]["residuals"]]})
    _record, planted = _gate_native(other)
    assert planted["gate"]["gate_score"] != legacy.gate_score


def test_gate_capture_declares_the_decline_too(gate_root, tmp_path):
    root, pool = gate_root
    row = {**r4.ROW, "im": float("nan")}
    result, candidate = _captured_gate(root, pool, row)
    assert result.gate_score is None and "MISSING_FEATURES" in result.flags
    declared = _declared(candidate, tmp_path)
    assert declared["feature_missing_mask"]["im"] is True
    record, seen = _gate_native(declared)
    assert seen["gate"].get("gate_score") is None
    assert "MISSING_FEATURES" in record.reason_codes


# -- the chooser: champion, producers, fold pools, keys and primitives -------------


@pytest.fixture
def chooser_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tier4, "SERVING_DIR", root)
    ensemble = importlib.import_module("engine.models.ensemble").MeanEnsemble
    artifact_cls = importlib.import_module("engine.models.registry").ModelArtifact
    joblib.dump(artifact_cls(model=ensemble([cf._hgbr(len(cf.CHOOSER_FEATURES), s)
                                             for s in (7, 8)]),
                             role="chooser", features=cf.CHOOSER_FEATURES,
                             residuals=np.zeros(3), target="synthetic"),
                root / "chooser.joblib")
    for _output, (role, seed, pool, floor) in cf.FOLDS.items():
        _dump_fold(role, cf._hgbr(len(cf.FOLD_FEATURES), seed), pool, floor)
    _dump_fold("size", None, cf.SIZE_POOL)
    return root


REGIME_FILLED = 17.5


def _chooser_scorer(tmp_path, monkeypatch, root):
    scorer = cf.legacy_scorer(tmp_path, monkeypatch, frame=cf.pool_frame())
    # `spy_vol60` is absent from the frame: legacy fills it from its own
    # panel/live sources, and the capture must record that value.
    scorer._regime_extra = lambda request, result, name: (
        REGIME_FILLED if name == "spy_vol60" else float("nan"))
    artifact = registry.load_artifact(root / "chooser.joblib")
    entry = SimpleNamespace(id="dyn_sv_chooser_v1_1", decision_offset=None,
                            path=root / "chooser.joblib",
                            artifact_sha256=_sha(root / "chooser.joblib"))
    scorer.model = lambda *args, **kwargs: (entry, artifact)
    return scorer


def _chooser_case(strategy="TWIN-P"):
    spot = 100.0
    quotes = {("P", float(k), cf.EXPIRY): {"bid": max(spot - k, 0.0) + 0.4,
                                            "ask": max(spot - k, 0.0) + 0.5}
              for k in range(70, 131)}
    case = cf.make_case(9, strategy=strategy, spot=spot, quotes=quotes, entry_date=cf.EVENT)
    case["features"].update({"iv30": 55.0, "signed_streak": 2.0})
    case["features"].pop("spy_vol60")
    return case


def _run_chooser(scorer, case, collector=None):
    result = cf._Result(case)
    if collector is not None:
        result._phase4_checkpoint_collector = collector
    Scorer._score_chooser(scorer, SimpleNamespace(strategy=case["strategy"],
                                                  decision_offset=None),
                          result, cf.legacy_features(case))
    return result


def test_chooser_capture_records_producers_pools_keys_and_17_primitives(
        chooser_root, tmp_path, monkeypatch):
    case = _chooser_case()
    collector = _collector()
    _run_chooser(_chooser_scorer(tmp_path, monkeypatch, chooser_root), case, collector)
    frozen = _frozen(_candidate(collector))
    declarations = frozen["declarations"]

    assert frozen["bindings"]["chooser"]["feature_order"] == list(cf.CHOOSER_FEATURES)
    assert declarations["chooser"] == {"binding": "chooser", "output": "chooser_score"}
    for output, (role, _seed, pool, floor) in cf.FOLDS.items():
        binding = frozen["bindings"][f"fold:{role}"]
        assert (binding["role"], binding["adapter"], binding["output_names"]) == (
            role, "tier4-serving-fold.v1", [output])
        assert declarations[f"chooser_fold:{output}"] == {
            "binding": f"fold:{role}", "output": output, "pool": output, "site": "chooser"}
        assert frozen["fold_pools"][output] == {
            "predictions": pool[0].tolist(), "residuals": pool[1].tolist(),
            "interval_floor": floor}
    assert declarations["chooser_fold:pred_abs_move"] == {"pool": "pred_abs_move"}
    assert frozen["fold_pools"]["pred_abs_move"]["predictions"] == cf.SIZE_POOL[0].tolist()

    primitives = declarations["chooser_primitives"]
    assert tuple(primitives) == CHOOSER_PRIMITIVE_COLUMNS == cf.PRIMITIVES
    assert primitives["spy_vol60"] == REGIME_FILLED
    assert primitives["dte_entry"] == case["features"]["dte_entry"]
    assert declarations["chooser_admissible_table"] == {
        "breakpoints": [list(pair) for pair in Scorer._N_ADMISSIBLE_BY_DEPTH],
        "fallback": score_mod._N_ADMISSIBLE_MEDIAN}
    frame = cf.pool_frame()
    pool_path = tmp_path / score_mod.CHOOSER_ANALOG_POOL
    assert declarations["chooser_analog_pool"] == {
        "path": str(pool_path), "sha256": _sha(pool_path),
        "cutoff": str((pd.to_datetime(frame["exit_date"]).max().normalize()
                       + pd.Timedelta(days=1)).date())}


def _chooser_native(declared, case):
    bundle = SourceBundle(
        source_ref="capture-chooser", strategy=case["strategy"],
        context={"ticker": "AAA", "event_date": cf.EVENT, "entry_date": case["entry_date"],
                 "exit_date": "2026-09-09", "expiry": cf.EXPIRY, "spot": case["spot"]},
        raw_quotes=case["quotes"], feature_vector=dict(declared["feature_vector"]),
        feature_missing_mask=declared["feature_missing_mask"],
        model_identity={"size": {"model_id": "synthetic"}},
        forecast_recipes={"forecast_abs_move": {"intercept": 6.0, "coefficients": {}},
                          "pred_iv_crush": {"intercept": -20.0, "coefficients": {}}},
        model_artifact_refs={"forecast_abs_move": "sha256:a", "pred_iv_crush": "sha256:b"},
        residual_recipe={"mode": "planned_exit", "pre_iv30": 40.0},
        paired_residual_rows=tuple(
            {"event_date": f"2025-{1 + i % 12:02d}-{1 + i % 27:02d}",
             "pred_abs_move": 3.0 + i % 9, "err_move": math.sin(i) * 3.0,
             "err_crush": -5.0 + math.cos(i) * 10.0}
            for i in range(400)),
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        chooser_recipe=declared["chooser_recipe"],
        chooser_fold_pools=declared["chooser_fold_pools"],
        chooser_analog_pool=declared.get("chooser_analog_pool"),
        chooser_admissible_table=declared.get("chooser_admissible_table"),
        frozen_inference=declared["frozen_inference"],
        model_release=declared["model_release"])
    return application.score_one(cf._request(case["strategy"]),
                                 build_native_score_inputs(bundle))


def test_chooser_from_the_captured_bundle_equals_legacy(chooser_root, tmp_path, monkeypatch):
    case = _chooser_case()
    collector = _collector()
    _run_chooser(_chooser_scorer(tmp_path, monkeypatch, chooser_root), case, collector)
    candidate = _candidate(collector)
    cutoff = _frozen(candidate)["declarations"]["chooser_analog_pool"]["cutoff"]
    pool_artifact = cf.build_chooser_analog_pool_artifact(
        cf.pool_frame().to_dict("records"), pool_id=CHOOSER_POOL_ID, cutoff=cutoff,
        lineage=cf.LINEAGE)
    declared = _declared(candidate, tmp_path, chooser_analog_pool=pool_artifact)
    recipe = declared["chooser_recipe"]
    assert set(recipe["producers"]) == set(cf.FOLDS)
    assert recipe["analog_pool"] == {"pool_id": CHOOSER_POOL_ID, "cutoff": cutoff}
    assert recipe["admissible_table"]["content_hash"] == legacy_n_admissible_table().content_hash
    # Only the 17 primitives and the folds' own inputs are declared, so
    # native derives every other chooser column.
    assert set(declared["feature_vector"]) <= set(CHOOSER_PRIMITIVE_COLUMNS) | set(
        cf.FOLD_FEATURES)
    assert declared["feature_vector"]["spy_vol60"] == REGIME_FILLED

    record = _chooser_native(declared, case)
    legacy_case = cf._legacy_case_from_record(case, record)
    legacy = _run_chooser(_chooser_scorer(tmp_path, monkeypatch, chooser_root), legacy_case)
    assert legacy.chooser_score is not None, legacy.flags
    assert record.forecasts["chooser_score"] == legacy.chooser_score

    # Planted defect: a perturbed producer pool is detected.
    pools = dict(declared["chooser_fold_pools"])
    pools["pred_im_t1_d14"] = {**pools["pred_im_t1_d14"], "residuals": [
        3.0 * value for value in pools["pred_im_t1_d14"]["residuals"]]}
    planted = _chooser_native(dict(declared, chooser_fold_pools=pools), case)
    assert planted.forecasts["chooser_score"] != legacy.chooser_score


def test_role_feature_vectors_keep_only_the_17_chooser_primitives():
    collector = Phase4TraceCollector(content_hasher=content_hash)
    collector.capture_features({"x": 1.0}, {"model_id": "m"}, role="abs_move")
    vector = {name: float(index) for index, name in enumerate(cf.CHOOSER_FEATURES)}
    collector.capture_dyn_sv(eligibility={"eligible": True},
                             ranking={"feature_vector": vector, "score": 1.0})
    vectors = _role_feature_vectors(_candidate(collector))
    assert vectors["chooser"] == {name: vector[name] for name in cf.PRIMITIVES}
    assert len(vectors["chooser"]) == 17


def test_converter_refuses_a_table_other_than_the_frozen_v1(chooser_root, tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(Scorer, "_N_ADMISSIBLE_BY_DEPTH", ((6.0, 5.0), (7.0, 12.0)))
    collector = _collector()
    _run_chooser(_chooser_scorer(tmp_path, monkeypatch, chooser_root), _chooser_case(),
                 collector)
    with pytest.raises(StrictTraceCaptureError, match="n_admissible"):
        _declared(_candidate(collector), tmp_path)


# -- R4-18: the STR-THRU recalibration map --------------------------------------------

CUTOFF = pd.Timestamp("2026-09-16")
DRIVER_RESIDUALS = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, -3.0])


def _recal_pairs(n=400, seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, n)
    return pd.DataFrame({
        "strategy": "STR-THRU", "fill_alpha": 0.5, "raw_win": raw,
        "outcome": (rng.uniform(0.0, 1.0, n) < 0.2 + 0.5 * raw).astype(float),
        "exit_date": pd.Timestamp("2025-01-01") + pd.to_timedelta(
            rng.integers(0, 500, n), unit="D"),
    })


def _payoff_trades(n=300, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    driver = rng.uniform(0.0, 12.0, n)
    return pd.DataFrame({
        "strategy": "STR-THRU", "fill_alpha": 0.5, "abs_move": driver,
        "spot_entry": 100.0,
        "exit_value": (0.01 + 0.006 * driver + rng.normal(0.0, 0.01, n)) * 100.0,
        "exit_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n), unit="D"),
    })


def _legacy_model(pairs, collector=None, buckets=None):
    """engine/score.py ``Scorer._score_model`` on a shell scorer: a constant
    driver, real ``fit_payoff`` over synthetic trades, real
    ``Scorer.recalibration`` over synthetic pairs."""
    from sklearn.dummy import DummyRegressor

    scorer = object.__new__(Scorer)
    model = DummyRegressor(strategy="constant", constant=5.0).fit([[0.0]], [5.0])
    artifact = registry.ModelArtifact(model=model, role="size", features=("x",),
                                      residuals=DRIVER_RESIDUALS, target="abs_move")
    artifact.residual_buckets = buckets
    entry = SimpleNamespace(id="size_synthetic", decision_offset=None)
    scorer.model = lambda role, *args, **kwargs: (entry, artifact) if role == "size" else None
    scorer.trades = _payoff_trades()
    scorer.snapshot = "snap-1"
    scorer._payoffs, scorer._recalibrations, scorer._recal_pairs = {}, {}, pairs
    request = ScoreRequest(ticker="AAA", strategy="STR-THRU", as_of=CUTOFF,
                           event_date=CUTOFF, fill=MID)
    result = score_mod.ScoreResult(ticker="AAA", strategy="STR-THRU", as_of=CUTOFF,
                                   event_date=CUTOFF)
    result.entry_cost, result.spot, result.evidence_cutoff = 4.0, 100.0, CUTOFF
    if collector is not None:
        result._phase4_checkpoint_collector = collector
    scorer._score_model(request, result, pd.DataFrame({"x": [1.0]}))
    seed = int.from_bytes(hashlib.sha256(
        f"{scorer.snapshot}|{request.key()}".encode()).digest()[:8], "big")
    return result, seed


def _recal_native(declared, seed):
    rows = [{"driver": float(row.abs_move), "spot_entry": float(row.spot_entry),
             "exit_value": float(row.exit_value), "exit_date": str(row.exit_date.date())}
            for row in _payoff_trades().itertuples()]
    bundle = SourceBundle(
        source_ref="capture-recal", strategy="STR-THRU",
        context={"ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0},
        raw_quotes={("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
                    ("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0}},
        feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "size_synthetic"}},
        forecast_recipes={"driver_prediction": {"intercept": 5.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:m1"},
        residual_recipe={"terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
                         "capital_at_risk": 1.0},
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        payoff_recipe={"before": "2026-09-16", "seed": seed,
                       "draw_count": score_mod.MODEL_DRAWS},
        payoff_source_rows=rows,
        model_residual_rows=[{"prediction": 5.0, "residual": float(value)}
                             for value in DRIVER_RESIDUALS],
        recalibration_declared=declared.get("recalibration_declared", False),
        recalibration_artifact=declared.get("recalibration_artifact"))
    request = V2ScoreRequest(
        event_id="evt-recal", calendar_revision="cal-1", strategy_version="STR-THRU",
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5})
    return application.score_one(request, build_native_score_inputs(bundle))


def test_recalibration_capture_records_the_map_legacy_applied():
    from engine import recalibrate

    pairs = _recal_pairs()
    collector = _collector()
    result, _seed = _legacy_model(pairs, collector)
    assert result.win_model != result.win_model_raw  # legacy really recalibrated
    recorded = _frozen(_candidate(collector))["declarations"]["recalibration"]
    fitted = recalibrate.fit_recalibration("STR-THRU", 0.5, before=CUTOFF, pairs=pairs)
    assert recorded == {
        "strategy": "STR-THRU", "alpha": 0.5, "cutoff": "2026-09-16",
        "min_pairs": recalibrate.MIN_PAIRS, "fitted": True, "n": fitted.n,
        "base_rate": fitted.base_rate, "x_thresholds": fitted.x_thresholds.tolist(),
        "y_thresholds": fitted.y_thresholds.tolist()}

    collector = _collector()
    _legacy_model(pairs.iloc[:10], collector)  # too few pairs: legacy ships raw
    recorded = _frozen(_candidate(collector))["declarations"]["recalibration"]
    assert recorded["fitted"] is False and recorded["x_thresholds"] == []


def test_recalibration_from_the_captured_bundle_equals_legacy(tmp_path):
    collector = _collector()
    result, seed = _legacy_model(_recal_pairs(), collector)
    declared = _declared(_candidate(collector), tmp_path)
    assert declared["recalibration_declared"] is True
    assert declared["recalibration_artifact"].fitted is True

    record = _recal_native(declared, seed)
    assert record.resolved_request["exp_pnl_model"] == result.exp_pnl_model
    assert record.resolved_request["win_model"] == result.win_model
    # Planted defect: the undeclared bundle (every capture before R4-18)
    # ships the raw win, which disagrees with legacy's recalibrated one.
    undeclared = _recal_native({}, seed)
    assert undeclared.resolved_request["win_model"] == result.win_model_raw
    assert undeclared.resolved_request["win_model"] != result.win_model


def test_recorded_frozen_block_round_trips_through_json(gate_root):
    root, pool = gate_root
    _result, candidate = _captured_gate(root, pool, r4.ROW)
    frozen = _frozen(candidate)
    assert untag_nonfinite(json.loads(json.dumps(frozen, allow_nan=False))) == frozen


# -- round 2: per-site inputs, stored crush, the payoff layer --------------------------


def _fold_at_sites(root, pool, sizing_row, gate_row):
    """The size fold recorded by the sizing and the gate call sites, each
    with the frame that site was handed."""
    collector = _collector()
    scorer = object.__new__(Scorer)
    served = r4._served(root, pool)
    request = SimpleNamespace(strategy="STR-THRU", decision_offset=None)
    result = r4._Result(_phase4_checkpoint_collector=collector)
    scorer._phase4_record_fold(
        request, result, served, pd.DataFrame([sizing_row]), site="sizing",
        declarations={"forecast:forecast_abs_move": {
            "binding": "fold:size", "output": "pred_abs_move", "site": "sizing"}})
    scorer._phase4_record_fold(
        request, result, served, pd.DataFrame([gate_row]), site="gate",
        declarations={"gate_forecast": {
            "binding": "fold:size", "output": "pred_abs_move", "pool": "pred_abs_move",
            "site": "gate"}})
    return _candidate(collector)


def test_a_fold_fed_different_rows_at_two_sites_is_recorded_per_site(gate_root, tmp_path):
    root, pool = gate_root
    other = {**r4.ROW, "iv30": 61.0}
    candidate = _fold_at_sites(root, pool, r4.ROW, other)  # never raises
    frozen = _frozen(candidate)
    assert frozen["conflicts"] == []
    assert frozen["inputs"]["fold:size@sizing"]["iv30"] == r4.ROW["iv30"]
    assert frozen["inputs"]["fold:size@gate"]["iv30"] == 61.0
    assert frozen["declarations"]["gate_forecast"]["site"] == "gate"
    # One feature_vector cannot serve both consumers with different rows.
    with pytest.raises(StrictTraceCaptureError,
                       match="iv30 differs between fold:size@forecast|iv30 differs"):
        _declared(candidate, tmp_path)

    agreeing = _fold_at_sites(root, pool, r4.ROW, dict(r4.ROW))
    declared = _declared(agreeing, tmp_path)
    assert declared["feature_vector"] == {name: r4.ROW[name] for name in r4.FOLD_FEATURES}


def test_a_conflicting_second_record_never_raises_and_the_converter_refuses_it(
        gate_root, tmp_path):
    root, pool = gate_root
    collector = _collector()
    collector.capture_frozen(declarations={"gate": {"binding": "gate", "output": "a"}})
    collector.capture_frozen(declarations={"gate": {"binding": "gate", "output": "b"}})
    with pytest.raises(StrictTraceCaptureError, match="declarations.gate"):
        _declared(_candidate(collector), tmp_path)


def _captured_crush(stored_value):
    collector = _collector()
    scorer = object.__new__(Scorer)
    scorer._crush = {("AAA", pd.Timestamp(r4.EVENT)): stored_value}
    scorer._phase4_tier4_sha = "sha256:" + "c" * 64
    request = SimpleNamespace(ticker="AAA", strategy="STR-THRU", decision_offset=None)
    result = r4._Result(_phase4_checkpoint_collector=collector)
    value = Scorer._crush_forecast(scorer, request, result, None)
    return value, _candidate(collector)


def _crush_native(declared, **extra):
    bundle = SourceBundle(
        source_ref="capture-crush", strategy="STR-THRU", context=dict(r4.STR_THRU_CONTEXT),
        raw_quotes=r4.STR_THRU_QUOTES, feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "synthetic"}},
        forecast_recipes={"driver_prediction": {"intercept": 6.0, "coefficients": {}},
                          **extra},
        model_artifact_refs={"driver_prediction": "sha256:driver",
                             **{name: "sha256:crush" for name in extra}},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        stored_forecasts=declared.get("stored_forecasts", {}))
    return r4._score(bundle, "STR-THRU")


def test_stored_crush_from_the_captured_bundle_equals_legacy(tmp_path):
    legacy, candidate = _captured_crush(-17.25)
    assert legacy == -17.25
    declared = _declared(candidate, tmp_path)
    stored = declared["stored_forecasts"]["pred_iv_crush_30"]
    assert stored["value"] == legacy
    assert stored["row"] == {"table": "tier4_forecasts", "table_sha256": "sha256:" + "c" * 64,
                             "ticker": "AAA", "event_date": r4.EVENT}
    _record, seen = _crush_native(declared)
    assert seen["forecast"]["pred_iv_crush_30"] == legacy
    # Legacy prefers the stored row over the served fold: so does native.
    _record, seen = _crush_native(
        declared, pred_iv_crush_30={"intercept": -3.0, "coefficients": {}})
    assert seen["forecast"]["pred_iv_crush_30"] == legacy
    # Planted defect: a value that does not match its row identity is refused.
    tampered = {"stored_forecasts": {"pred_iv_crush_30": {**stored, "value": -1.0}}}
    with pytest.raises(ValueError, match="row_hash"):
        _crush_native(tampered)


def _buckets():
    rng = np.random.default_rng(17)
    predictions = rng.uniform(1.0, 12.0, 200)
    residuals = rng.normal(0.0, 1.0 + 0.2 * predictions, 200)
    return registry.bucket_residuals(predictions, residuals, deciles=4, min_pool=20)


@pytest.mark.parametrize("bucketed", [False, True])
def test_win_model_end_to_end_from_the_captured_bundle(tmp_path, bucketed):
    """The model layer from recorded state only: the payoff fit legacy served,
    the champion residual pool it drew from, its seed and the recalibration
    map. No hand-supplied rows, residuals or seed."""
    collector = _collector()
    result, _seed = _legacy_model(_recal_pairs(), collector,
                                  buckets=_buckets() if bucketed else None)
    declared = _declared(_candidate(collector), tmp_path)
    assert "payoff_source_rows" not in declared and "payoff_recipe" not in declared
    record = _model_native(declared)
    assert record.resolved_request["exp_pnl_model"] == result.exp_pnl_model
    assert record.resolved_request["win_model"] == result.win_model
    assert result.win_model != result.win_model_raw

    # Planted defect: the other residual pool moves the answer.
    other = dict(declared)
    pool = declared["model_residual_artifacts"]["driver"]
    buckets = pool.buckets and {
        "edges": pool.buckets["edges"],
        "pools": [[2.0 * v for v in row] for row in pool.buckets["pools"]]}
    moved = _inline_pool_like(pool, [2.0 * v for v in pool.flat_residuals], buckets=buckets)
    other["model_residual_artifacts"] = {"driver": moved}
    other["model_residual_artifact_recipe"] = {"driver": {
        **declared["model_residual_artifact_recipe"]["driver"],
        "content_hash": moved.content_hash}}
    assert _model_native(other).resolved_request["exp_pnl_model"] != result.exp_pnl_model


def test_release_states_are_pinned_when_they_hold_what_legacy_used(tmp_path):
    from engine.v2.models.lineage import DataDependency, Lineage
    from engine.v2.models.payoff_artifact import make_payoff_line_artifact
    from engine.v2.models.recalibration_artifact import make_recalibration_map_artifact

    collector = _collector()
    result, _seed = _legacy_model(_recal_pairs(), collector)
    candidate = _candidate(collector)
    inline = _declared(candidate, tmp_path)
    line, recal = inline["payoff_artifact"], inline["recalibration_artifact"]
    pool = inline["model_residual_artifacts"]["driver"]
    # The release's own copies: same key and content, other provenance.
    release_line = make_payoff_line_artifact(
        {"n": line.n, "resid_sd": line.resid_sd, "r": line.r, "residuals": line.residuals,
         "intercept": line.intercept, "slope": line.slope},
        strategy=line.strategy, driver=line.driver, alpha=line.alpha, cutoff=line.cutoff,
        window=("2020-01-01", "2026-09-16"))
    release_recal = make_recalibration_map_artifact(
        {"n": recal.n, "base_rate": recal.base_rate, "x_thresholds": recal.x_thresholds,
         "y_thresholds": recal.y_thresholds},
        strategy=recal.strategy, alpha=recal.alpha, cutoff=recal.cutoff,
        min_pairs=recal.min_pairs, window=("2020-01-01", "2026-09-16"))
    release_pool = _inline_pool_like(pool, pool.flat_residuals, lineage=Lineage(
        data=(DataDependency(table="models.registry"),)))
    assert release_line.content_hash != line.content_hash

    pinned = _declared(candidate, tmp_path,
                       release_states=(release_line, release_recal, release_pool))
    assert pinned["payoff_artifact"] is release_line
    assert pinned["recalibration_artifact"] is release_recal
    assert pinned["model_residual_artifacts"]["driver"] is release_pool
    assert pinned["model_residual_artifact_recipe"]["driver"]["content_hash"] == (
        release_pool.content_hash)
    record = _model_native(pinned)
    assert record.resolved_request["win_model"] == result.win_model

    # A release state under the same key but with other content is refused.
    wrong = make_payoff_line_artifact(
        {"n": line.n, "resid_sd": line.resid_sd, "r": line.r, "residuals": line.residuals,
         "intercept": line.intercept + 0.01, "slope": line.slope},
        strategy=line.strategy, driver=line.driver, alpha=line.alpha, cutoff=line.cutoff)
    with pytest.raises(StrictTraceCaptureError, match="not what legacy used"):
        _declared(candidate, tmp_path, release_states=(wrong,))


def _inline_pool_like(pool, flat, lineage=None, buckets=None):
    from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact

    return make_driver_residual_pool_artifact(
        role=pool.role, model_id=pool.model_id, fold=pool.fold, flat_residuals=flat,
        buckets=pool.buckets if buckets is None else buckets, deciles=pool.deciles, min_pool=pool.min_pool,
        lineage=pool.lineage if lineage is None else lineage)


def _model_native(declared):
    bundle = SourceBundle(
        source_ref="capture-model", strategy="STR-THRU",
        context={"ticker": "AAA", "event_date": "2026-09-16", "entry_date": "2026-09-16",
                 "exit_date": "2026-09-17", "expiry": "2026-09-18", "spot": 100.0},
        raw_quotes={("C", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0},
                    ("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 3.0}},
        feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "size_synthetic"}},
        forecast_recipes={"driver_prediction": {"intercept": 5.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:m1"},
        residual_recipe={"terminal_spots": (95.0, 105.0), "weights": (0.5, 0.5),
                         "capital_at_risk": 1.0},
        analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        payoff_artifact_recipe=declared["payoff_artifact_recipe"],
        payoff_artifact=declared["payoff_artifact"],
        model_residual_artifact_recipe=declared["model_residual_artifact_recipe"],
        model_residual_artifacts=declared["model_residual_artifacts"],
        recalibration_declared=declared.get("recalibration_declared", False),
        recalibration_artifact=declared.get("recalibration_artifact"))
    request = V2ScoreRequest(
        event_id="evt-model", calendar_revision="cal-1", strategy_version="STR-THRU",
        deployment_id="dep-1", decision_clock_id="entry-close",
        requested_decision_at="2026-09-16", snapshot_id="snap-1", mode="replay",
        fill_model={"alpha": 0.5})
    return application.score_one(request, build_native_score_inputs(bundle))
