from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import joblib
import pandas as pd
import pytest

from checks import phase4_real
from checks.phase4_frozen_bridge import _feature_rows
from engine.fills import MID
from engine.score import ScoreRequest
from engine.v2.contracts import ScoreRequest as V2ScoreRequest
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.stages import NativeScoreInputs, receipt
from tools.capture_tier0_corpus import (
    STRICT_TRACE_SUPPORTED_STRATEGIES,
    StrictTraceCaptureError,
    _captured_blocks,
    _frozen_runtime,
    _gate_packaged_bindings,
    _hydrate_trace,
    _merged_model_inputs,
    _packaged_gate_recipe,
    _role_feature_vectors,
    _SpilledTrace,
    attach_strict_probe,
    canonical_v2_request,
    frozen_gate_packaging,
    main,
    make_pair,
    native_inputs_from_capture,
    package_strict_trace,
    parse_strategies,
    request_to_dict,
    strict_trace_one,
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
        "model": {},
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


def test_merged_model_inputs_accepts_a_legitimately_nonfinite_feature():
    """The RAMP7 `forecast_sizing.or_implied` gap: a present-but-NaN source
    value (a real ORATS quote gap, not a capture omission) is mirrored as a
    real NaN, not raised as an unexplained defect -- see
    `engine.score._size_feature_capture_value` and
    `tools.capture_tier0_corpus._coerce_feature_value`.
    """
    import math

    candidate = {"legacy_trace": _legacy_trace(
        driver_role="size",
        driver_vector={"or_implied": {"__nonfinite__": "nan"}, "mcap_log": 10.0},
        gate_vector=None,
    )}

    merged = _merged_model_inputs(candidate)

    assert math.isnan(merged["or_implied"])
    assert merged["mcap_log"] == 10.0


def test_merged_model_inputs_accepts_a_missing_daily_state_feature_as_nan():
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="implied_t1",
        driver_vector={"im": None, "mcap_log": 10.0},
        gate_vector=None,
    )}

    merged = _merged_model_inputs(candidate)

    assert merged["mcap_log"] == 10.0
    assert math.isnan(merged["im"])


def test_role_feature_vectors_accepts_a_missing_daily_state_feature_as_nan():
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="implied_t1",
        driver_vector={"im": None, "mcap_log": 10.0},
        gate_vector=None,
    )}

    vectors = _role_feature_vectors(candidate)

    assert vectors["implied_t1"]["mcap_log"] == 10.0
    assert math.isnan(vectors["implied_t1"]["im"])


def test_merged_model_inputs_still_refuses_a_missing_non_daily_state_feature():
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="implied_t1",
        driver_vector={"im": None, "some_other_feature": None},
        gate_vector=None,
    )}

    with pytest.raises(
        StrictTraceCaptureError,
        match=r"feature implied_t1\.some_other_feature is missing or nonnumeric",
    ):
        _merged_model_inputs(candidate)


def test_merged_model_inputs_does_not_conflict_on_two_missing_daily_state_readings():
    features_value = {
        "feature_vector": {
            "implied_t1": {"im": None},
            "runup_move": {"im": None},
        },
        "missing_mask": {
            "implied_t1": {"im": True},
            "runup_move": {"im": True},
        },
        "model_identity": {
            "implied_t1": {"model_id": "implied-1"},
            "runup_move": {"model_id": "runup-1"},
        },
    }
    candidate = {"legacy_trace": {
        "schema_version": "phase4_legacy_diagnostic_checkpoint.v1.0",
        "disposition": {"status": "completed", "flags": []},
        "checkpoints": {
            "features": {
                "value": features_value,
                "content_hash": content_hash(features_value),
            },
        },
    }}

    merged = _merged_model_inputs(candidate)

    assert math.isnan(merged["im"])


def test_native_observer_packages_a_strict_verifiable_trace(tmp_path):
    legacy = request_to_dict(ScoreRequest(
        ticker="ABC", strategy="STR-THRU", as_of=pd.Timestamp("2026-09-16"),
        event_date=pd.Timestamp("2026-09-17"), session="AMC", fill=MID,
    ))
    request = canonical_v2_request({"event_id": "event-1", "request": legacy}, "snapshot-1")
    inputs, shared = _native(request)

    trace, native = package_strict_trace(request, inputs, shared)
    pair = {
        "payload": {
            "request": legacy,
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


def test_native_observer_packages_a_trace_with_a_nonempty_model_block(tmp_path):
    """Regression for the payoff-driver ``model`` block translation gap.

    Capture's ``_model_block_from_frozen`` (STR-THRU/STR-RUNUP and the
    DYN-SV rows built on them) fills ``blocks["model"]`` with real artifact
    dataclasses (``PayoffLineArtifact``, ...) -- required for
    ``NativeScoreInputs`` to execute the model stage. Before this fix,
    ``native_inputs_from_capture`` put that SAME raw block into
    ``shared_inputs`` unchanged, while ``package_strict_trace`` serializes
    the native trace's "model" through ``_model_document`` (tagged kinds).
    The two sides' leaf sets then differed and
    ``tools/phase4_release_assembler.py::_translation``'s identity path
    refused with "translation leaf coverage differs". Every unit test used
    an empty ``model: {}`` (where both sides trivially agree), so none of
    them caught it.
    """
    from engine.v2.models.payoff_artifact import make_payoff_line_artifact
    from tools.capture_tier0_corpus import _model_document

    legacy = request_to_dict(ScoreRequest(
        ticker="ABC", strategy="STR-THRU", as_of=pd.Timestamp("2026-09-16"),
        event_date=pd.Timestamp("2026-09-17"), session="AMC", fill=MID,
    ))
    request = canonical_v2_request({"event_id": "event-1", "request": legacy}, "snapshot-1")
    inputs, shared = _native(request)

    payoff_artifact = make_payoff_line_artifact(
        {"n": 40, "resid_sd": 1.5, "r": 0.6, "residuals": [0.1, -0.2, 0.3],
         "intercept": 0.05, "slope": 0.9},
        strategy="STR-THRU", driver="abs_move", alpha=0.5, cutoff="2026-09-01",
        window=("2020-01-01", "2026-09-16"))
    model_block = {
        "payoff_recipe": {"before": "2026-09-16", "seed": 7, "draw_count": 2000},
        "payoff_artifact": payoff_artifact,
        "model_residual_artifact_recipe": {},
        "model_residual_artifacts": {},
    }
    # Mirrors the fix in `native_inputs_from_capture`: `shared_inputs` carries
    # the DOCUMENTED form of the model block, the native trace carries the
    # raw one that `NativeScoreInputs` executes against.
    shared["native_inputs"]["model"] = _model_document(model_block)
    new_source_ref = content_hash(shared)
    new_receipts = tuple(
        receipt(stage, {"source_ref": new_source_ref}, {"execution": "native-runtime"})
        for stage in (
            "resolve_context", "features", "forecast", "geometry", "pricing",
            "model", "analogs", "simulation", "gate", "chooser", "serialization",
        )
    )
    inputs = replace(
        inputs, model=model_block, source_ref=new_source_ref,
        stage_receipts=new_receipts,
    )

    trace, native = package_strict_trace(request, inputs, shared)
    assert trace["input_translation"]["mappings"]

    pair = {
        "payload": {
            "request": legacy,
            "record": {},
            "legacy_input_hash": trace["shared_input_hash"],
            "input_trace": trace,
            "input_trace_hash": trace["trace_hash"],
        },
    }
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)

    assert verified["captured_stages"][-1] == "serialization"
    values = verified["inputs"].model
    assert "payoff_artifact" in values
    assert values["payoff_artifact"].intercept == payoff_artifact.intercept
    assert to_document(native.canonical_request) == to_document(request)


def test_decoded_driver_residual_pool_bucket_edges_survive_the_json_round_trip():
    """Regression for the production artifact-loader import gap.

    ``checks.phase4_real._decode_model_block`` rebuilds a captured driver
    residual pool via ``_residual_pool_from_document`` ->
    ``engine.v2.models.residual_artifact.residual_artifact_from_document``,
    the SAME production path a release read uses -- which is why that
    reader (and its ``DriverResidualPoolArtifact``/``ResidualArtifactError``
    companions) must be imported directly from
    ``engine.v2.models.residual_artifact`` rather than reconstructed by
    hand. The pool's outermost decile buckets are unbounded
    (``+/-inf`` edges); a strict-JSON round trip has to carry those through
    ``tag_nonfinite`` on write and ``untag_nonfinite`` (inside
    ``residual_artifact_from_document``) on read without a refusal or a
    silent collapse to a finite sentinel.
    """
    import math

    from engine.v2.foundation import tag_nonfinite
    from engine.v2.models.lineage import Lineage
    from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact
    from tools.capture_tier0_corpus import _model_document

    artifact = make_driver_residual_pool_artifact(
        role="size", model_id="size_v1_4", fold="2026-09-01",
        flat_residuals=[0.1, -0.2, 0.3, 1.5, -1.5],
        buckets={"edges": [-math.inf, -1.0, 0.0, 1.0, math.inf],
                 "pools": [[-2.0, -1.5], [-0.5, -0.1], [0.2, 0.4], [1.2, 2.0]]},
        deciles=4, min_pool=2, lineage=Lineage(),
    )
    model_block = {"model_residual_artifacts": {"driver": artifact}}

    doc = _model_document(model_block)
    tagged = tag_nonfinite(doc)
    encoded = json.dumps(tagged, allow_nan=False)
    decoded_doc = json.loads(encoded)
    decoded_block = phase4_real._decode_model_block(decoded_doc)

    decoded_artifact = decoded_block["model_residual_artifacts"]["driver"]
    assert decoded_artifact.bucket_edges == artifact.bucket_edges
    assert decoded_artifact.bucket_edges[0] == -math.inf
    assert decoded_artifact.bucket_edges[-1] == math.inf


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


def test_frozen_runtime_omits_a_nonfinite_binding_exactly_as_feature_rows_does(tmp_path):
    """Regression for the RAMP7-HAIN (016) / DYN-SV-LUXE (002) defect: the
    captured trace's own `role_model_inputs` tags `or_implied` non-finite for
    the driver binding, `_feature_rows` (checks/phase4_frozen_bridge.py,
    replay) correctly OMITS that binding from its requests, but
    `_frozen_runtime` (here, capture) used to build an InferenceRequest for
    EVERY declared binding regardless of finiteness -- so the resolve_context
    receipt this module writes at capture time asserted the driver binding
    ran, while replay re-derives from the identical recorded feature vector
    that it did not, and `execution.resolve_context.input_hash` mismatches
    on a row nothing actually changed about.

    This pins that `_frozen_runtime`'s inference-request binding set and
    `_feature_rows`'s derivation from the SAME per-role vectors always
    agree: the non-finite driver binding is omitted on both sides, and the
    unaffected, fully-finite gate binding is kept on both sides.
    """
    source = tmp_path / "source"
    source.mkdir()
    path, digest = _artifact(source)
    driver_binding = _model_binding(
        path, digest, role="abs_move", feature_order=["x"],
        output_names=["driver_prediction"], model_id="driver-1",
    )
    gate_binding = _model_binding(
        path, digest, role="gate", feature_order=["x", "n_prior"],
        output_names=["gate_score"], model_id="gate-1",
    )
    # The driver role's own captured `x` is a genuine sourced non-finite
    # value (contracts §2.1 tag), same shape as the real fixture's
    # `or_implied` gap; the gate role's captured vector is fully finite.
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="abs_move",
        driver_vector={"x": {"__nonfinite__": "nan"}},
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
    captured_binding_ids = {item.binding_id for item in inference_requests}
    bindings_by_role = {binding.role: binding for binding in release.bindings}

    # What replay derives from the SAME recorded per-role vectors, in the
    # shape a round-tripped JSON trace carries them (a non-finite value
    # serializes as the `{"__nonfinite__": ...}` tag; a finite one round-
    # trips as itself) -- exactly what `checks/phase4_real.py` hands
    # `_feature_rows` from `features.role_model_inputs` at replay.
    replay_inputs = replace(inputs, features={
        **inputs.features,
        "role_model_inputs": {
            "driver": {"x": {"__nonfinite__": "nan"}},
            "gate": {"x": 9.0, "n_prior": 5.0},
        },
    })
    replayed_requests = _feature_rows(
        replay_inputs, release.bindings, release.release_id,
    )
    replayed_binding_ids = {item.binding_id for item in replayed_requests}

    assert captured_binding_ids == replayed_binding_ids
    assert bindings_by_role["driver"].binding_id not in captured_binding_ids
    assert bindings_by_role["gate"].binding_id in captured_binding_ids


def _full_strict_candidate(*, fixture_id, ticker, driver_vector, gate_vector,
                           path, digest, strategy="STR-THRU", absolute=False):
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
    if absolute:  # as the real registry and Tier-4 caches record them
        driver_binding["artifact"] = gate_binding["artifact"] = str(path)
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
    # attach_strict_probe resolves captured artifact paths beneath the data
    # root (engine.paths.ROOT); point it at this test's tmp_path instead of
    # the real repo so the artifact stays self-contained.
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
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
        assert isinstance(candidate["input_trace"], _SpilledTrace)
        trace = _hydrate_trace(candidate["input_trace"])
        assert candidate["legacy_input_hash"] == trace["shared_input_hash"]
        assert "serialization" in trace["stages"]
    score_ids = {candidate["native_score_id"] for candidate in chosen}
    assert len(score_ids) == 2


def test_strict_probe_resolves_artifacts_under_the_data_root_not_the_code_root(
        tmp_path, monkeypatch):
    """A worktree run: the code checkout (the tool's ROOT) is not the data
    root (``engine.paths.ROOT``, INVESTING_PLAN_ROOT) the artifacts live
    under, and the captured paths are absolute under the data root."""
    from tools.phase4_frozen_resources import package_frozen_resources

    code_root, data_root = tmp_path / "worktree", tmp_path / "data_root"
    (data_root / "data" / "models").mkdir(parents=True)
    code_root.mkdir()
    monkeypatch.setattr("tools.capture_tier0_corpus.ROOT", code_root)
    monkeypatch.setattr("engine.paths.ROOT", data_root)
    path, digest = _artifact(data_root / "data" / "models")

    def chosen():
        return [_full_strict_candidate(
            fixture_id="case-0", ticker="AAA", driver_vector={"x": 2.0},
            gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest,
            absolute=True)]

    candidates = chosen()
    attached, gaps = attach_strict_probe(candidates, "snapshot-1", tmp_path / "release")
    assert (list(attached), gaps) == (["case-0"], {})
    # The frozen-resources default is the data root too, not the cwd.
    monkeypatch.chdir(code_root)
    binding = _model_binding(path, digest, role="gate", feature_order=["x"],
                             output_names=["gate_score"], model_id="gate-AAA")
    binding["artifact"] = str(path)
    package_frozen_resources(model_bindings=[binding], deployment_id="dep",
                             release_root=tmp_path / "release-2")

    # Planted defect: resolving against the code checkout is the 09-19 failure.
    monkeypatch.setattr("engine.paths.ROOT", code_root)
    with pytest.raises(StrictTraceCaptureError, match="escapes source root"):
        attach_strict_probe(chosen(), "snapshot-1", tmp_path / "release-3")


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
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
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
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
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
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
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
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
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


def _priced_record(ticker: str, *, structure_params: dict) -> dict:
    return {
        "strategy": "STR-THRU", "ticker": ticker, "session": "AMC",
        "legs": [{"right": "C", "strike": 100.0, "expiry": "2026-09-18", "qty": 1},
                 {"right": "P", "strike": 100.0, "expiry": "2026-09-18", "qty": 1}],
        "entry_cost": 3.0, "structure_params": structure_params,
        "forecast_abs_move": 0.05, "forecast_p10": 0.01, "forecast_p90": 0.09,
        "forecast_sd": 0.02, "forecast_model": "driver-x", "forecast_fold": "2026-09-01",
        "flags": [],
    }


def test_traced_pairs_keep_the_legacy_request_for_tier0_and_tier1(tmp_path, monkeypatch):
    """The 2026-09-19 blocker: the strict probe overwrote ``payload.request``
    with the canonical V2 request on every traced pair. A traced selector pair
    and the pinned re-score made from it must still (a) carry the legacy
    request, which round-trips through ``request_from_dict`` for tier-1
    replay, (b) keep ``geometry:pinned`` and ``geometry:round_listed_strike``
    covered, (c) resolve their ``pinned_from`` link, (d) give the
    ``forecast_suppressed`` seeded control its pinned target, and (e) still
    pass Phase 4's strict trace verifier, which now reads the V2 request from
    ``input_trace.request`` and binds it to the legacy one."""
    from checks import tier0_corpus as t0
    from tools.capture_tier0_corpus import axis_inputs, request_from_dict

    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)

    def candidate(fixture_id, ticker, **request_changes):
        cand = _full_strict_candidate(
            fixture_id=fixture_id, ticker=ticker,
            driver_vector={"x": 2.0, "n_prior": 5.0}, gate_vector={"x": 2.0, "n_prior": 5.0},
            path=path, digest=digest,
        )
        cand["request"] = request_to_dict(
            replace(request_from_dict(cand["request"]), **request_changes))
        # As real legacy bindings record it (decision_offset -> clock), so
        # the frozen replay's clock check matches the canonical request.
        row = cand["legacy_trace"]["checkpoints"]["source_inputs"]
        for binding in row["value"]["model_bindings"]:
            binding["decision_clock"] = "legacy.decision_offset.0"
        row["content_hash"] = content_hash(row["value"])
        return cand

    source = candidate("case-source", "AAA")
    source["record"] = _priced_record("AAA", structure_params={"width_moneyness": 0.05})
    pinned = candidate("case-pinned", "AAA", strike=100.0,
                       structure_params={"width_moneyness": 0.05})
    pinned["record"] = _priced_record("AAA", structure_params={"width_moneyness": 0.05})
    pinned["relations"] = {"pinned_from": content_hash(source["request"])}
    chosen = [source, pinned]
    legacy_requests = {c["fixture_id"]: copy.deepcopy(c["request"]) for c in chosen}
    index: dict[str, list[str]] = {}
    for cand in chosen:
        cand["covers"] = t0.derive_covers(cand["record"], cand["request"], cand["kind"],
                                          axis_inputs(), cand.get("relations"))
        for axis in cand["covers"]:
            index.setdefault(axis, []).append(cand["fixture_id"])
    assert {"geometry:pinned", "geometry:round_listed_strike"} <= set(pinned["covers"])

    write(tmp_path / "out", chosen, index, pd.Timestamp("2026-09-16"), "snapshot-1",
          strict_trace=True)
    corpus = t0.load(tmp_path / "out")

    for fixture_id, legacy in legacy_requests.items():
        payload = corpus.pairs[fixture_id]["payload"]
        assert payload["trace_disposition"] == "complete"
        # (a) the saved request is the legacy one, and replays.
        assert payload["request"] == legacy
        assert request_to_dict(request_from_dict(payload["request"])) == legacy
        # The V2 request lives in the trace, and is the legacy one's translation.
        assert payload["input_trace"]["request"]["strategy_version"] == "STR-THRU"
        assert "strategy" not in payload["input_trace"]["request"]
        # (e) Phase 4 verifies it from its new home.
        verified = phase4_real._verified_trace_bundle(corpus.pairs[fixture_id], corpus.root)
        assert to_document(verified["request"]) == payload["input_trace"]["request"]

    # (b) coverage re-derived from the survivors agrees with the index.
    assert t0.case_coverage(corpus).verdict == t0.AGREE
    derived = t0._derived(corpus)["case-pinned"]
    assert {"geometry:pinned", "geometry:round_listed_strike"} <= set(derived)
    # (c) the pinned link resolves to the source pair.
    pinned_receipt = t0.case_pinned_counterparts(corpus)
    assert pinned_receipt.verdict == t0.AGREE
    assert pinned_receipt.population.compared > 1
    # (d) the forecast_suppressed control finds its pinned target and fires.
    summary = t0.seeded_controls(corpus)
    control = summary["controls"]["forecast_suppressed"]
    assert control["target"] == "case-pinned"
    assert control["problems"] == []

    # Planted defect: the pre-fix layout (payload.request = the V2 request)
    # loses every one of those, and Phase 4 refuses it.
    broken = copy.deepcopy(corpus.pairs["case-pinned"])
    broken["payload"]["request"] = broken["payload"]["input_trace"]["request"]
    assert "geometry:pinned" not in t0.derive_covers(
        broken["payload"]["record"], broken["payload"]["request"], "score_result",
        axis_inputs(), broken["payload"]["relations"])
    with pytest.raises(phase4_real._TraceError, match="not the translation"):
        phase4_real._verified_trace_bundle(broken, corpus.root)
    # ... and a trace re-pointed at another case's legacy request is refused.
    swapped = copy.deepcopy(corpus.pairs["case-pinned"])
    swapped["payload"]["request"]["ticker"] = "ZZZ"
    with pytest.raises(phase4_real._TraceError, match="event_revision"):
        phase4_real._verified_trace_bundle(swapped, corpus.root)


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

import checks.phase4_stored_forecasts as phase4_stored_forecasts  # noqa: E402
import engine.score as score_mod  # noqa: E402
import tests.test_v2_scoring_native_chooser_features as cf  # noqa: E402
import tests.test_v2_scoring_r4_20_frozen_parity as r4  # noqa: E402
from checks.phase4_stored_forecasts import (  # noqa: E402
    StoredForecastError,
    resolve_stored_forecasts,
    with_stored_forecasts,
)
from engine.data.features import tier4  # noqa: E402
from engine.models import registry  # noqa: E402
from engine.score import Phase4TraceCollector, Scorer  # noqa: E402
from engine.v2.foundation.canonical import untag_nonfinite  # noqa: E402
from engine.v2.models.admissible_table import legacy_n_admissible_table  # noqa: E402
from engine.v2.scoring import application  # noqa: E402
from engine.v2.scoring.source_inputs import (  # noqa: E402
    SourceBundle,
    build_native_score_inputs,
    stored_forecast_row_hash,
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


# -- a DECLARED gate the strict recipe path never packaged (fixtures 008/017) ----


def _source_of(candidate: dict) -> dict:
    return candidate["legacy_trace"]["checkpoints"]["source_inputs"]["value"]


def _synth_gate_source(declarations=None, bindings=None, inputs=None):
    """A minimal ``source_inputs`` block: no ``native_recipes.gate``, frozen
    declarations as ``Scorer._score_gate``'s ``capture_frozen`` records them
    on a row it DECLINED before ever feeding ``capture_source_bundle``."""
    if declarations is None:
        declarations = {"gate": {"binding": "gate", "output": "gate_score",
                                 "threshold": 0.5, "site": "gate"}}
    return {
        "native_recipes": {"forecast": {"binding": "fold:size"}},
        "model_bindings": [],
        "frozen": {
            "declarations": declarations,
            "bindings": ({"gate": {"role": "gate", "feature_order": ["im"],
                                   "output_names": ["gate_score"]}}
                         if bindings is None else bindings),
            "inputs": ({"gate@gate": {"im": 5.5}} if inputs is None else inputs),
        },
    }


def test_packaging_none_for_scored_and_ungated_rows():
    # A scored row: ``native_recipes.gate`` exists -- the strict recipe path
    # owns the gate and the derived packaging must not touch it.
    scored = _synth_gate_source()
    scored["native_recipes"]["gate"] = {"threshold": 0.5}
    assert frozen_gate_packaging(scored) is None
    # A genuinely ungated row: nothing declared -- the row keeps its
    # ``not_applicable`` gate block.
    assert frozen_gate_packaging(_synth_gate_source(declarations={})) is None
    assert frozen_gate_packaging({"native_recipes": {"forecast": {}}}) is None


def test_packaging_declared_missing_or_mismatched_artifacts_fail_closed():
    # Declared but never recorded: refused, never repackaged as no-gate.
    with pytest.raises(StrictTraceCaptureError, match="not recorded"):
        frozen_gate_packaging(_synth_gate_source(bindings={}))
    # Malformed declaration (output is not the gate's own).
    with pytest.raises(StrictTraceCaptureError, match="malformed frozen gate"):
        frozen_gate_packaging(_synth_gate_source(
            declarations={"gate": {"binding": "gate", "output": "gate_pass",
                                   "threshold": 0.5, "site": "gate"}}))
    # A binding of another role cannot serve the gate.
    with pytest.raises(StrictTraceCaptureError, match="serves role"):
        frozen_gate_packaging(_synth_gate_source(
            bindings={"gate": {"role": "size", "feature_order": ["im"],
                               "output_names": ["pred_abs_move"]}}))
    # A declared row that names a nonnumeric feature is refused.
    with pytest.raises(StrictTraceCaptureError, match="nonnumeric"):
        frozen_gate_packaging(_synth_gate_source(inputs={"gate@gate": {"im": "x"}}))


def test_packaged_gate_from_a_real_declined_gate_record(gate_root):
    # The legacy scorer's own recording, with the scored-row ``native_recipes``
    # gate entry REMOVED: exactly the gap a row declined before
    # ``capture_source_bundle`` lands in.
    _result, candidate = _captured_gate(gate_root[0], gate_root[1], r4.ROW)
    source = copy.deepcopy(_source_of(candidate))
    recipes = source.get("native_recipes")
    assert isinstance(recipes, dict) and recipes.pop("gate", None) is not None

    packaging = frozen_gate_packaging(source)
    assert packaging is not None and packaging["threshold"] == 0.5
    # The recorded gate row is the base frame only (the eight derived
    # forecast/analog columns are computed natively). The declared binding is
    # ALWAYS retained -- never dropped because the row cannot eagerly feed it;
    # the only names its feature_order has beyond the recorded row are the
    # derived ones eager inference defers on.
    from engine.v2.scoring.native_gate_features import (
        GATE_ANALOG_COLUMNS, GATE_FORECAST_COLUMNS,
    )
    derived = set(GATE_FORECAST_COLUMNS) | set(GATE_ANALOG_COLUMNS)
    assert packaging["gate_binding"] is not None
    assert set(packaging["gate_binding"]["feature_order"]) - set(packaging["gate_row"]) <= derived
    assert packaging["forecast"] is True
    assert packaging["forecast_output"] == "pred_abs_move"
    assert packaging["forecast_binding"] is None or all(
        name in packaging["forecast_row"]
        for name in packaging["forecast_binding"]["feature_order"])

    source_root = gate_root[0]
    extra = _gate_packaged_bindings(source, (), packaging, source_root=source_root)
    roles = [row["role"].split(":", 1)[0] for row in extra]
    assert "size" in roles or packaging["forecast_binding"] is None
    # Roles the checkpoint already carries are never added twice -- but ONLY
    # when the standing binding IS the declared artifact (equal normalized
    # identity, source path excluded), never because a role name collides.
    held = [packaging["gate_binding"]]
    if packaging["forecast_binding"] is not None:
        held.append(packaging["forecast_binding"])
    assert _gate_packaged_bindings(source, held, packaging,
                                   source_root=source_root) == []

    # Planted: a same-role binding naming a DIFFERENT artifact digest cannot
    # stand in for the declared gate -- refuse, never resolve by role alone.
    swapped = dict(packaging["gate_binding"])
    swapped["artifact_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(StrictTraceCaptureError, match="same-role"):
        _gate_packaged_bindings(source, [swapped], packaging,
                                source_root=source_root)

    from types import SimpleNamespace

    sidecar = [{"role": "gate", "binding_id": "b-gate"}]
    if packaging["forecast_binding"] is not None:
        sidecar.append({"role": "size", "binding_id": "b-size"})
    recipe = _packaged_gate_recipe(
        packaging, SimpleNamespace(sidecar_document={"bindings": sidecar}))
    assert recipe["threshold"] == 0.5
    if packaging["forecast_binding"] is not None:
        assert recipe["forecast"] == {"binding_id": "b-size",
                                      "output": "pred_abs_move"}
        assert recipe["forecast_pool"]["predictions"]
    # With no release at all only the threshold survives -- never a
    # reference to a binding that was not packaged.
    assert _packaged_gate_recipe(packaging, None) == {"threshold": 0.5}
    assert _packaged_gate_recipe(None, None) is None


def test_declared_gate_forecast_named_pool_missing_fails_closed(gate_root):
    """A gate forecast is declared with BOTH an ``output`` and a named ``pool``.
    The two usually agree but are distinct facts: the pool is selected by the
    declaration's OWN name, never by the output. Here the recorded fold_pools
    still carry the output-named pool, but the declaration is renamed to a pool
    that was never recorded -- packaging must fail closed on that name, not
    silently fall back to the output-named pool that happens to exist.
    """
    root, pool = gate_root
    _result, candidate = _captured_gate(root, pool, {**r4.ROW, "im": float("nan")})
    source = copy.deepcopy(_source_of(candidate))
    declared = source["frozen"]["declarations"]["gate_forecast"]
    assert declared["output"] == declared["pool"] == "pred_abs_move"
    assert "pred_abs_move" in source["frozen"]["fold_pools"]  # output-named exists
    declared["pool"] = "pred_abs_move_named_only"            # ... this one does not
    with pytest.raises(StrictTraceCaptureError,
                       match=r"declared gate forecast pool .* was not recorded"):
        frozen_gate_packaging(source)


def test_same_role_captured_gate_row_conflict_refuses_not_dedupes(tmp_path):
    """A gate that reached ``capture_gate_inputs`` (scored-path checkpoint) and
    ALSO carries a frozen gate DECLINE row is the same role read twice. If the
    two readings disagree, that is a row no single binding ever had -- it must be
    refused, never resolved to whichever vector was seen first. The scored-path
    gate vector is added first, then the frozen decline's row is merged through
    the SAME conflict check; a differing shared column trips it."""
    path, digest = _artifact(tmp_path)
    candidate = {"legacy_trace": _legacy_trace(
        driver_role="abs_move", driver_vector={"x": 2.0},
        gate_vector={"mean_prior_abs_move": 6.2, "iv30": 55.0},
    )}
    # The frozen decline's own base row: same role "gate", iv30 at ANOTHER value.
    source = _synth_gate_source(
        inputs={"gate@gate": {"mean_prior_abs_move": 6.2, "iv30": 99.0}})
    candidate["legacy_trace"]["checkpoints"]["source_inputs"] = {
        "value": source, "content_hash": content_hash(source),
    }
    with pytest.raises(
        StrictTraceCaptureError,
        match=r"role gate captured twice with different values",
    ):
        _role_feature_vectors(candidate)


# -- the size fold: its binding, forecast declaration and pool --------------------


def _captured_sizing(root, pool, row):
    collector = _collector()
    scorer = object.__new__(Scorer)
    scorer._serving = lambda fold, produces="pred_abs_move": r4._served(root, pool)
    scorer._features = lambda request, result: pd.DataFrame([row])
    result = r4._Result(_phase4_checkpoint_collector=collector)
    request = SimpleNamespace(strategy="TWIN-P", decision_offset=None)
    Scorer._size_from_forecast(scorer, request, result, None, size=False)
    return result, _candidate(collector)


def test_sizing_capture_records_the_fold_and_packages_its_own_pool(gate_root, tmp_path):
    root, pool = gate_root
    result, candidate = _captured_sizing(root, pool, r4.ROW)
    assert result.forecast_model == "size_synthetic"
    frozen = _frozen(candidate)
    assert frozen["declarations"]["forecast:forecast_abs_move"] == {
        "binding": "fold:size", "output": "pred_abs_move", "pool": "pred_abs_move",
        "site": "sizing"}
    declared = _declared(candidate, tmp_path)
    assert declared["forecast_recipes"]["forecast_abs_move"]["output"] == "pred_abs_move"
    packaged = declared["forecast_pool"]
    assert packaged["predictions"] == pool[0].tolist()
    assert packaged["residuals"] == pool[1].tolist()
    assert packaged["interval_floor"] == 0.0

    # The actual strict trace path uses _captured_blocks (not the helper above
    # that produces the frozen release declarations). Verify that this is the
    # source of the executable sizing band and that its serialization is bound
    # to both the declaration and captured pool contents.
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    source = checkpoint["value"]
    source["context"] = {"ticker": "AAA", "event_date": "2026-09-17",
                         "entry_date": "2026-09-16", "exit_date": "2026-09-18",
                         "spot": 100.0}
    source["quote_status"] = "empty"
    checkpoint["content_hash"] = content_hash(source)
    blocks = _captured_blocks(candidate, _request())
    forecast = blocks["forecast"]
    assert forecast["forecast_pool"] == packaged
    provenance = forecast["forecast_pool_provenance"]
    assert provenance["declaration"] == frozen["declarations"]["forecast:forecast_abs_move"]
    assert provenance["pool_hash"] == content_hash(packaged)

    # No declaration means no band. A recorded but thin pool is retained as
    # the source truth; the native interval implementation itself suppresses it.
    without_declaration = copy.deepcopy(candidate)
    thin_source = without_declaration["legacy_trace"]["checkpoints"]["source_inputs"]["value"]
    thin_source["frozen"]["declarations"].pop("forecast:forecast_abs_move")
    thin_source["context"] = source["context"]
    thin_source["quote_status"] = "empty"
    thin_checkpoint = without_declaration["legacy_trace"]["checkpoints"]["source_inputs"]
    thin_checkpoint["content_hash"] = content_hash(thin_source)
    assert "forecast_pool" not in _captured_blocks(without_declaration, _request())["forecast"]

    thin_pool = copy.deepcopy(candidate)
    thin_source = thin_pool["legacy_trace"]["checkpoints"]["source_inputs"]["value"]
    thin_source["context"] = source["context"]
    thin_source["quote_status"] = "empty"
    thin_source["frozen"]["fold_pools"]["pred_abs_move"]["predictions"] = [1.0, 2.0]
    thin_source["frozen"]["fold_pools"]["pred_abs_move"]["residuals"] = [0.1, 0.2]
    thin_checkpoint = thin_pool["legacy_trace"]["checkpoints"]["source_inputs"]
    thin_checkpoint["content_hash"] = content_hash(thin_source)
    thin_block = _captured_blocks(thin_pool, _request())["forecast"]
    assert len(thin_block["forecast_pool"]["residuals"]) == 2
    p10, p90, sd, _ = importlib.import_module(
        "engine.v2.scoring.native_gate_features").forecast_interval(
            [1.0], thin_block["forecast_pool"]["predictions"],
            thin_block["forecast_pool"]["residuals"])
    assert pd.isna(p10[0]) and pd.isna(p90[0]) and pd.isna(sd[0])


@pytest.mark.parametrize("change, message", [
    (lambda entry: entry.update(binding="fold:other"), "malformed sizing forecast declaration"),
    (lambda entry: entry.update(output="wrong"), "malformed sizing forecast declaration"),
    (lambda entry: entry.update(site="gate"), "malformed sizing forecast declaration"),
    (lambda entry: entry.update(pool="absent"), "declared sizing fold pool absent was not recorded"),
])
def test_sizing_capture_rejects_bad_pool_declarations(gate_root, change, message):
    root, pool = gate_root
    _result, candidate = _captured_sizing(root, pool, r4.ROW)
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    source = checkpoint["value"]
    source["context"] = {"ticker": "AAA", "event_date": "2026-09-17",
                         "entry_date": "2026-09-16", "exit_date": "2026-09-18"}
    source["quote_status"] = "empty"
    entry = source["frozen"]["declarations"]["forecast:forecast_abs_move"]
    change(entry)
    checkpoint["content_hash"] = content_hash(source)
    with pytest.raises(StrictTraceCaptureError, match=message):
        _captured_blocks(candidate, _request())


@pytest.mark.parametrize("mutate", [
    lambda pool: pool.update(predictions="123"),
    lambda pool: pool.update(residuals={"0": 0.1}),
    lambda pool: pool.update(predictions=[1.0] * 251, residuals=[0.1] * 250),
    lambda pool: pool.update(predictions=[float("inf")] * 3000),
    lambda pool: pool.update(residuals=["0.1"] * 3000),
])
def test_sizing_capture_rejects_malformed_or_unequal_pool_columns(gate_root, mutate):
    root, pool = gate_root
    _result, candidate = _captured_sizing(root, pool, r4.ROW)
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    source = checkpoint["value"]
    source["context"] = {"ticker": "AAA", "event_date": "2026-09-17",
                         "entry_date": "2026-09-16", "exit_date": "2026-09-18"}
    source["quote_status"] = "empty"
    mutate(source["frozen"]["fold_pools"]["pred_abs_move"])
    checkpoint["content_hash"] = content_hash(source)
    with pytest.raises(StrictTraceCaptureError, match="malformed sizing fold pool"):
        _captured_blocks(candidate, _request())


def test_sizing_band_survives_strict_capture_verified_replay_and_record(gate_root, tmp_path):
    root, pool = gate_root
    _legacy_result, candidate = _captured_sizing(root, pool, r4.ROW)
    legacy_request = ScoreRequest(
        ticker="AAA", strategy="STR-THRU", as_of=pd.Timestamp("2026-09-16"),
        event_date=pd.Timestamp("2026-09-17"), session="AMC", fill=MID,
    )
    candidate.update({"event_id": "AAA_2026-09-17",
                      "request": request_to_dict(legacy_request),
                      "record": {"strategy": "STR-THRU", "ticker": "AAA"}})
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    source = checkpoint["value"]
    source["context"] = {"ticker": "AAA", "event_date": "2026-09-17",
                         "entry_date": "2026-09-16", "exit_date": "2026-09-18",
                         "spot": 100.0}
    source["quote_status"] = "empty"
    source["native_recipes"]["forecast"] = {"models": {
        "forecast_abs_move": {"intercept": 4.0, "coefficients": {}}}}
    source["model_bindings"] = []
    source["frozen"]["bindings"] = {}
    checkpoint["content_hash"] = content_hash(source)

    trace, captured_record = strict_trace_one(candidate, "snapshot-1", tmp_path / "release")
    pool_doc = trace["native_inputs"]["forecast"]["forecast_pool"]
    legacy_band = tier4.interval_for(
        np.array([4.0]), np.asarray(pool_doc["predictions"]),
        np.asarray(pool_doc["residuals"]), floor=pool_doc["interval_floor"])
    expected = tuple(float(values[0]) for values in legacy_band[:3])
    assert tuple(captured_record.uncertainty[key] for key in
                 ("forecast_p10", "forecast_p90", "forecast_sd")) == expected

    pair = {"payload": {"request": candidate["request"], "record": {},
                        "legacy_input_hash": trace["shared_input_hash"],
                        "input_trace": trace, "input_trace_hash": trace["trace_hash"]}}
    verified = phase4_real._verified_trace_bundle(pair, tmp_path / "release")
    replayed_record, _receipts, _identities = phase4_real._replayed_member(verified)
    assert tuple(replayed_record.uncertainty[key] for key in
                 ("forecast_p10", "forecast_p90", "forecast_sd")) == expected

    changed = copy.deepcopy(pair)
    changed["payload"]["input_trace"]["shared_inputs"]["native_inputs"]["forecast"]\
        ["forecast_pool"]["residuals"][0] += 1.0
    with pytest.raises(phase4_real._TraceError, match="trace_hash"):
        phase4_real._verified_trace_bundle(changed, tmp_path / "release")


def test_declined_gate_survives_strict_capture_verified_replay_and_record(
        gate_root, tmp_path, monkeypatch):
    """A real legacy MISSING_FEATURES gate DECLINE (fixtures 008/017) must
    package, verify and replay as the same gate it was, end to end.

    The base row omits the eight derived forecast/analog columns the gate's
    ``feature_order`` names and carries a non-finite ``im`` the gate needs.
    Legacy declares its binding, records the base-frame row it read and then
    declines -- it never feeds the binding to ``capture_source_bundle``, so
    ``native_recipes`` has no gate for the strict recipe path to package. The
    only gate evidence is the frozen declaration, which
    :func:`frozen_gate_packaging` turns into a real release binding. The gate
    row can never EAGERLY feed that binding (the derived columns are computed by
    the native gate stage), so eager inference is omitted -- on capture AND
    replay, the same predicate, over the same recorded ``role_model_inputs`` --
    while the declared artifact still stands behind the gate and the native gate
    stage declines it the way legacy did.
    """
    from engine.v2.scoring.native_gate_features import (
        GATE_ANALOG_COLUMNS, GATE_FORECAST_COLUMNS,
    )
    root, pool = gate_root
    declined_row = {**r4.ROW, "im": float("nan")}
    _legacy_result, candidate = _captured_gate(root, pool, declined_row)
    # The real legacy decline: a MISSING_FEATURES gate, before eager inference.
    assert _legacy_result.gate_score is None and "MISSING_FEATURES" in _legacy_result.flags
    real_frozen = copy.deepcopy(_frozen(candidate))

    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    traced = _full_strict_candidate(
        fixture_id="case-declined-gate", ticker="AAA",
        driver_vector={"x": 2.0}, gate_vector={},  # empty: no scored-path gate
        path=path, digest=digest,
    )
    checkpoint = traced["legacy_trace"]["checkpoints"]["source_inputs"]
    source = checkpoint["value"]
    # Drop the synthetic gate binding AND its empty gate_inputs checkpoint: the
    # only gate this row now knows about is the real frozen declaration grafted
    # in from a genuine legacy decline.
    source["model_bindings"] = [
        b for b in source["model_bindings"] if b["role"] != "gate"]
    assert len(source["model_bindings"]) == 1  # the driver survives
    traced["legacy_trace"]["checkpoints"].pop("gate_inputs")
    source["frozen"] = real_frozen
    # The base frame legacy actually read (a non-finite im is tagged by the
    # collector, untagged by ``frozen_gate_packaging`` back to a real NaN).
    source["features"] = dict(declined_row)
    # Legacy records its clock from a null decision_offset; match the canonical
    # request the frozen replay's clock check reads.
    for binding in source["model_bindings"]:
        binding["decision_clock"] = "legacy.decision_offset.0"
    checkpoint["content_hash"] = content_hash(source)

    trace, captured_native = strict_trace_one(
        traced, "snapshot-1", tmp_path / "release")

    # `strict_trace_one` returns the in-memory trace; every Phase 4 check reads
    # it back only after `write()` + the corpus loader put it through the JSON
    # round trip -- tuples become arrays, non-finite floats become
    # {"__nonfinite__": repr} tags. The packaged gate forecast pool comes from
    # `fold_pool`, which hands back TUPLES, so verifying the bare in-memory
    # trace here would exercise a shape (`checks/phase4_real._leaf_values`
    # walks lists only, while the assembler's `_leaves` walked both) that never
    # reaches the checker in production. Normalize to the stored form -- the
    # exact bytes the loader hands `_verified_trace_bundle` -- first.
    from engine.v2.foundation import tag_nonfinite

    stored_trace = json.loads(json.dumps(tag_nonfinite(trace), allow_nan=False))
    pair = {"payload": {"request": traced["request"], "record": {},
                        "legacy_input_hash": stored_trace["shared_input_hash"],
                        "input_trace": stored_trace,
                        "input_trace_hash": stored_trace["trace_hash"]}}
    verified = phase4_real._verified_trace_bundle(pair, tmp_path / "release")
    replayed_native, runtime_receipts, _identities = phase4_real._replayed_member(
        verified)

    release = verified["frozen_replay"].release
    roles_by_binding = {b.binding_id: b.role for b in release.bindings}

    # (1) The release RETAINS the declared gate as a real binding, backed by the
    # verified artifact legacy actually used (never dropped because the row
    # cannot eagerly feed it).
    gate_bindings = [b for b in release.bindings if b.role == "gate"]
    assert len(gate_bindings) == 1
    gate_binding = gate_bindings[0]
    gate_sha = _sha(root / "gate.joblib")
    assert gate_binding.members[0].content_hash == gate_sha
    assert gate_binding.feature_order == tuple(r4.GATE_FEATURES)
    # The forecast fold that feeds the gate's derived columns is retained too.
    assert "size" in set(roles_by_binding.values())

    # (2) Eager gate inference is OMITTED in replay (the gate row is base-frame
    # only) while the driver/size eager rows run -- and capture's own
    # resolve_context receipt agrees, or _replayed_member above would have
    # raised. This is the shared-predicate guarantee (binding_feature_row).
    replayed_binding_ids = {item.binding_id for item in verified["frozen_replay"].requests}
    assert gate_binding.binding_id not in replayed_binding_ids
    driver_binding_id = next(b for b, role in roles_by_binding.items() if role == "driver")
    assert driver_binding_id in replayed_binding_ids

    # (3) The captured gate row is still the base frame: it lacks every derived
    # forecast/analog column the declared binding's feature_order names.
    gate_row = verified["inputs"].features["role_model_inputs"]["gate"]
    derived = set(GATE_FORECAST_COLUMNS) | set(GATE_ANALOG_COLUMNS)
    assert derived.isdisjoint(gate_row)
    assert "im" in gate_row  # the non-finite column that caused the decline

    # (4) Both passes (capture's own execution and the replay) decline the gate
    # exactly as legacy did: MISSING_FEATURES, never a packaging/eligibility
    # refusal, and never a fabricated gate score.
    for native in (captured_native, replayed_native):
        codes = tuple(native.reason_codes or ())
        assert "MISSING_FEATURES" in codes, codes
        assert "OUT_OF_DOMAIN" not in codes, codes
        assert not any(code.startswith("MISSING_GATE_INPUT") for code in codes), codes
        assert native.resolved_request.get("gate_score") is None

    # (5) The runtime receipts the replay re-derived match the captured ones.
    assert runtime_receipts[-1]["stage"] == "serialization"
    captured_stages = {row["stage"] for row in verified["captured_receipts"]}
    assert "gate" in captured_stages

    # Planted defect: strip the frozen gate declaration and the whole gate story
    # disappears -- the row packages as ungated (not_applicable), the release
    # loses the gate binding, and NOTHING reaches MISSING_FEATURES. The
    # assertions above are load-bearing on the DECLARED gate, not incidental.
    dropped = copy.deepcopy(traced)
    dropped_source = dropped["legacy_trace"]["checkpoints"]["source_inputs"]["value"]
    dropped_source["frozen"] = {"declarations": {}, "bindings": {}, "inputs": {},
                                "fold_pools": {}}
    dropped["legacy_trace"]["checkpoints"]["source_inputs"]["content_hash"] = \
        content_hash(dropped_source)
    dropped_trace, dropped_native = strict_trace_one(
        dropped, "snapshot-1", tmp_path / "drop-release")
    assert dropped_trace["native_inputs"]["gate"] == {"mode": "not_applicable"}
    assert "MISSING_FEATURES" not in (dropped_native.reason_codes or ())


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
    scorer._crush = {("AAA", pd.Timestamp(r4.EVENT)): (
        stored_value, "iv-crush-hgbr-v1", pd.Timestamp("2026-08-01"))}
    scorer._phase4_tier4_sha = "sha256:" + "c" * 64
    request = SimpleNamespace(ticker="AAA", strategy="STR-THRU", decision_offset=None)
    result = r4._Result(_phase4_checkpoint_collector=collector)
    value = Scorer._crush_forecast(scorer, request, result, None)
    return value, _candidate(collector)


def _crush_bundle(declared, **extra):
    return SourceBundle(
        source_ref="capture-crush", strategy="STR-THRU", context=dict(r4.STR_THRU_CONTEXT),
        raw_quotes=r4.STR_THRU_QUOTES, feature_vector={}, feature_missing_mask={},
        model_identity={"driver": {"model_id": "synthetic"}},
        forecast_recipes={"driver_prediction": {"intercept": 6.0, "coefficients": {}},
                          **extra},
        model_artifact_refs={"driver_prediction": "sha256:driver",
                             **{name: "sha256:crush" for name in extra}},
        residual_recipe={}, analog_recipe={},
        gate_recipe={"model": {"intercept": 1.0, "coefficients": {}}, "threshold": 0.0},
        stored_forecast_refs=declared.get("stored_forecast_refs", {}))


def _crush_native(declared, **extra):
    return r4._score(_crush_bundle(declared, **extra), "STR-THRU")


def _crush_native_resolved(declared, reader, **extra):
    inputs = build_native_score_inputs(_crush_bundle(declared, **extra))
    resolved = resolve_stored_forecasts(inputs, reader=reader)
    seen: dict = {}
    record = application.score_one(
        r4._request("STR-THRU"), with_stored_forecasts(inputs, resolved),
        observer=lambda item: seen.setdefault(item.receipt.stage, item.output_document))
    return record, seen


def test_stored_crush_reference_from_the_captured_bundle_resolves_to_legacy(
        tmp_path, monkeypatch):
    legacy, candidate = _captured_crush(-17.25)
    assert legacy == -17.25
    declared = _declared(candidate, tmp_path)
    ref = declared["stored_forecast_refs"]["pred_iv_crush_30"]
    assert set(ref) == {"row", "row_hash"}
    assert ref["row"] == {
        "table": "tier4_forecasts", "table_sha256": "sha256:" + "c" * 64,
        "column": "pred_iv_crush_30", "ticker": "AAA", "event_date": r4.EVENT,
        "model_id": "iv-crush-hgbr-v1", "fold_start": "2026-08-01",
    }
    assert "value" not in json.dumps(ref)

    # The reference is an address, so native must read the table itself. The
    # injected reader stands in for that read; the file stands in for the
    # table vintage the capture named.
    table = tmp_path / "tier4_forecasts.parquet"
    table.write_bytes(b"captured-tier4-vintage")
    monkeypatch.setattr(phase4_stored_forecasts, "STORED_TABLES",
                        {"tier4_forecasts": lambda: table})
    ref["row"]["table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    ref["row_hash"] = stored_forecast_row_hash(ref["row"], legacy)

    def reader(path, ticker, event_date, column):
        return {"value": legacy, "model_id": ref["row"]["model_id"],
                "fold_start": ref["row"]["fold_start"]}

    _record, seen = _crush_native_resolved(declared, reader)
    assert seen["forecast"]["pred_iv_crush_30"] == legacy
    # Legacy prefers the stored row over the served fold: so does native.
    _record, seen = _crush_native_resolved(
        declared, reader, pred_iv_crush_30={"intercept": -3.0, "coefficients": {}})
    assert seen["forecast"]["pred_iv_crush_30"] == legacy
    # Planted defect: a row_hash that disagrees with the resolved cell refuses.
    tampered = {"stored_forecast_refs": {
        "pred_iv_crush_30": {**ref, "row_hash": "sha256:" + "0" * 64}}}
    with pytest.raises(StoredForecastError) as err:
        _crush_native_resolved(tampered, reader)
    assert err.value.reason == "STORED_FORECAST_ROW_HASH_MISMATCH"
    # Planted defect: a value smuggled in beside the reference is refused.
    loophole = {"stored_forecast_refs": {
        "pred_iv_crush_30": {**ref, "value": legacy}}}
    with pytest.raises(ValueError, match="unsupported recipe fields"):
        _crush_native(loophole)


def test_stored_crush_reference_resolves_identically_at_capture_and_replay(
        tmp_path, monkeypatch):
    """Regression for 006/007/012 (BFLY-P5/CTR5/RAMP7): both gates failed at
    ``execution.forecast.{input_hash,output_hash}: captured runtime
    mismatch`` because capture scored the forecast stage with the stored
    crush reference UNRESOLVED while replay resolved it first
    (``checks/phase4_stored_forecasts.py``). Capture must resolve it too, on
    its own execution pass only -- the stored pair still carries only the
    reference, never the value.
    """
    # No frozen driver/gate bindings here on purpose: this test is scoped to
    # the stored-crush forecast path this fix touches, not the separate
    # frozen-release machinery `_full_strict_candidate` exercises elsewhere.
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    ticker, event_date, expiry = "AAA", "2026-09-17", "2026-09-18"
    legacy_request = ScoreRequest(
        ticker=ticker, strategy="STR-THRU",
        as_of=pd.Timestamp("2026-09-16"), event_date=pd.Timestamp(event_date),
        session="AMC", fill=MID,
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
        "model_bindings": [],
        "native_recipes": {"forecast": {"models": {}}},
    }
    legacy_trace = _legacy_trace(driver_role="abs_move", driver_vector={}, gate_vector=None)
    legacy_trace["checkpoints"]["source_inputs"] = {
        "value": source_inputs_value,
        "content_hash": content_hash(source_inputs_value),
    }
    candidate = {
        "fixture_id": "case-crush",
        "covers": [],
        "kind": "score_result",
        "record": {"strategy": "STR-THRU", "ticker": ticker},
        "request": request_to_dict(legacy_request),
        "duration": 0.1,
        "relations": {},
        "legacy_trace": legacy_trace,
        "event_id": f"{ticker}_{event_date}",
    }

    table_path = tmp_path / "tier4_forecasts.parquet"
    value = -12.5
    pd.DataFrame({
        "ticker": ["AAA"],
        "event_date": pd.to_datetime(["2026-09-17"]).astype("datetime64[us]"),
        "pred_iv_crush_30": [value],
        "pred_iv_crush_30_model_id": pd.array(["iv-crush-hgbr-v1"], dtype="string"),
        "pred_iv_crush_30_fold_start": pd.to_datetime(
            ["2026-08-01"]).astype("datetime64[us]"),
    }).to_parquet(table_path)
    monkeypatch.setattr(phase4_stored_forecasts, "STORED_TABLES",
                        {"tier4_forecasts": lambda: table_path})

    ref_row = {
        "table": "tier4_forecasts",
        "table_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "column": "pred_iv_crush_30", "ticker": "AAA", "event_date": "2026-09-17",
        "model_id": "iv-crush-hgbr-v1", "fold_start": "2026-08-01",
    }
    row_hash = stored_forecast_row_hash(ref_row, value)
    checkpoint = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    checkpoint["value"]["frozen"] = {"declarations": {
        "forecast:pred_iv_crush_30": {
            "source": "stored_tier4", "row": ref_row, "value": value,
            "row_hash": row_hash,
        },
    }}
    checkpoint["content_hash"] = content_hash(checkpoint["value"])

    trace, native = strict_trace_one(candidate, "snapshot-1", tmp_path / "release")

    # The pair still carries only the reference: no resolved value anywhere
    # in what gets stored, and the reference itself is unchanged.
    forecast_native_inputs = trace["native_inputs"]["forecast"]
    assert "stored" not in forecast_native_inputs
    assert forecast_native_inputs["stored_refs"] == {
        "pred_iv_crush_30": {"row": ref_row, "row_hash": row_hash},
    }
    assert str(value) not in json.dumps(trace["native_inputs"])

    # Capture's own scoring pass resolved the reference: no unresolved flag.
    assert not any(
        code.startswith("UNRESOLVED_STORED_FORECAST")
        for code in (native.reason_codes or ())
    )
    forecast_stage = trace["stages"]["forecast"]

    # Replay resolves the same reference from the same table and must land
    # on the SAME forecast-stage input_hash/output_hash the capture
    # recorded -- the defect this closes.
    pair = {
        "payload": {
            "request": candidate["request"],
            "record": {},
            "legacy_input_hash": trace["shared_input_hash"],
            "input_trace": trace,
            "input_trace_hash": trace["trace_hash"],
        },
    }
    verified = phase4_real._verified_trace_bundle(pair, tmp_path / "release")
    _replay_native, runtime_receipts, _identities = phase4_real._replayed_member(verified)
    replay_forecast = next(row for row in runtime_receipts if row["stage"] == "forecast")

    assert replay_forecast["input_hash"] == forecast_stage["input_hash"]
    assert replay_forecast["output_hash"] == forecast_stage["output_hash"]


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


# ---------------------------------------------------------------------------
# Tier-0 vetting gaps 001/012/003 (2026-09-19): rows that stop before, at, or
# inside pricing. Legacy now records the resolved context and HOW FAR the
# chain lookup got (``source_inputs.quote_status``); the strict probe accepts
# an explicitly empty quote domain, and native reaches its own refusal.
# ---------------------------------------------------------------------------


def _stopped_candidate(tmp_path, *, quote_status, quote_domain=(), spot=None,
                       fixture_id="case-stopped"):
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id=fixture_id, ticker="AAA",
        driver_vector={"x": 2.0, "n_prior": 5.0},
        gate_vector={"x": 2.0, "n_prior": 5.0}, path=path, digest=digest,
    )
    row = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    value = row["value"]
    value["context"] = {
        "ticker": "AAA", "strategy": "STR-THRU", "event_date": "2026-09-17",
        "session": "AMC", "entry_date": "2026-09-16", "exit_date": "2026-09-18",
        "as_of": "2026-09-16", "quote_date": "2026-09-16",
    }
    if spot is not None:
        value["context"]["spot"] = spot
    value["quote_domain"] = list(quote_domain)
    if quote_status is None:
        value.pop("quote_status", None)
    else:
        value["quote_status"] = quote_status
    row["content_hash"] = content_hash(value)
    return candidate


def _probe_native(candidate, tmp_path, monkeypatch):
    """Run the real probe; return (attached, gaps, native record or None)."""
    import tools.capture_tier0_corpus as capture

    seen = {}
    real = capture.package_strict_trace

    def recording(request, *args, **kwargs):
        trace, native = real(request, *args, **kwargs)
        seen[request.event_id] = native
        return trace, native

    monkeypatch.setattr(capture, "package_strict_trace", recording)
    path, digest = _artifact(tmp_path, "good.joblib")
    good = _full_strict_candidate(
        fixture_id="case-good", ticker="BBB",
        driver_vector={"x": 2.0}, gate_vector={"x": 9.0, "n_prior": 5.0},
        path=path, digest=digest,
    )
    attached, gaps = attach_strict_probe([good, candidate], "snapshot-1",
                                         tmp_path / "release")
    return attached, gaps, seen.get(candidate["event_id"])


@pytest.mark.parametrize("quote_status", ["empty", "not_reached"])
def test_probe_traces_a_row_with_an_explicitly_empty_quote_domain(
        tmp_path, monkeypatch, quote_status):
    """001 (NO_CHAIN) and 012 (NO_FORECAST before pricing): no quotes and no
    spot, recorded on purpose. The probe traces the row and native refuses
    with its own code; parity, not the probe, compares it to legacy's."""
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    candidate = _stopped_candidate(tmp_path, quote_status=quote_status)

    attached, gaps, native = _probe_native(candidate, tmp_path, monkeypatch)

    assert "case-stopped" in attached, gaps
    trace = _hydrate_trace(candidate["input_trace"])
    assert trace["native_inputs"]["context"]["quotes"] == {}
    assert native is not None and native.reason_codes


def test_probe_traces_a_row_whose_pricer_raised_without_a_spot(tmp_path, monkeypatch):
    """003 (COARSE_LADDER): the quotes are recorded, pricing raised, so there
    is no spot; the resolved exit_date is now in the context."""
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    candidate = _stopped_candidate(tmp_path, quote_status="recorded", quote_domain=[
        {"right": "C", "strike": 100.0, "expiry": "2026-09-18", "bid": 1.0, "ask": 2.0},
    ])

    attached, gaps, native = _probe_native(candidate, tmp_path, monkeypatch)

    assert "case-stopped" in attached, gaps
    assert native.reason_codes


_ONE_QUOTE = [{"right": "C", "strike": 100.0, "expiry": "2026-09-18",
               "bid": 1.0, "ask": 2.0}]


@pytest.mark.parametrize(("quote_status", "quote_domain", "spot", "message"), [
    # Planted defect: an empty domain that legacy never said was empty is a
    # capture that never recorded quotes, not an empty lookup.
    (None, [], 100.0, "quote_domain is empty"),
    ("empty", _ONE_QUOTE, None, "is not empty"),
    ("recorded", [], None, "quote_domain is empty"),
    # A priced row must still carry its spot.
    ("priced", _ONE_QUOTE, None, r"missing \['spot'\]"),
    ("bogus", [], None, "unknown quote_status"),
])
def test_probe_still_refuses_unrecorded_or_contradictory_quote_domains(
        tmp_path, quote_status, quote_domain, spot, message):
    candidate = _stopped_candidate(tmp_path, quote_status=quote_status,
                                   quote_domain=quote_domain, spot=spot)
    request = canonical_v2_request(candidate, "snapshot-1")
    with pytest.raises(StrictTraceCaptureError, match=message):
        native_inputs_from_capture(candidate, request)


# ---------------------------------------------------------------------------
# Tier-0 vetting gaps 005 (CAL-P) and 009 (CND-P), 2026-09-19: a disabled
# strategy is refused before any stage. Legacy now records a request-only
# source bundle, the probe emits a minimal trace from it, and Phase 4 compares
# native's own refusal with legacy's instead of marking the row incomparable.
# ---------------------------------------------------------------------------


def _disabled_candidate(strategy: str, fixture_id: str) -> dict:
    scorer = score_mod.Scorer.__new__(score_mod.Scorer)
    scorer.snapshot = "snap-test"
    legacy_request = ScoreRequest(
        ticker="ZZZ", strategy=strategy, as_of=pd.Timestamp("2026-09-16"),
        event_date=pd.Timestamp("2026-09-17"), session="AMC", fill=MID,
    )
    collector = Phase4TraceCollector(retain_full_trace=False,
                                     content_hasher=content_hash)
    result = scorer.score(legacy_request, trace=collector)
    return {
        "fixture_id": fixture_id, "covers": [], "kind": "score_result",
        "record": result.as_dict(), "request": request_to_dict(legacy_request),
        "duration": 0.0, "relations": {},
        "legacy_trace": collector.diagnostic_checkpoint(),
        "event_id": f"ZZZ_2026-09-17_{strategy}",
    }


@pytest.mark.parametrize("strategy", sorted(score_mod.DISABLED_STRATEGIES))
def test_disabled_strategy_row_is_traced_and_compared_not_incomparable(
        tmp_path, monkeypatch, strategy):
    from checks.tier0_corpus import load

    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    good = _full_strict_candidate(
        fixture_id="case-good", ticker="AAA", driver_vector={"x": 2.0},
        gate_vector={"x": 9.0, "n_prior": 5.0}, path=path, digest=digest,
    )
    disabled = _disabled_candidate(strategy, "case-disabled")

    doc = write(tmp_path / "out", [good, disabled], {}, pd.Timestamp("2026-09-16"),
                "snapshot-1", strict_trace=True)

    assert doc["pairs"]["case-disabled"]["trace_disposition"] == "complete"
    corpus = load(tmp_path / "out")
    payload = corpus.pairs["case-disabled"]["payload"]
    assert payload["request"]["strategy"] == strategy
    native_inputs = payload["input_trace"]["native_inputs"]
    assert native_inputs["context"]["quotes"] == {}
    assert native_inputs["features"]["model_inputs"] == {}
    release, _parity = phase4_real._native_parity(corpus)
    row = next(r for r in release["dispositions"]
               if r["fixture_id"] == "case-disabled")
    # Compared, never passed through: parity reports whether the refusals
    # match (the flags check), it is not decided here.
    assert row["disposition"] == "compared", row.get("reason")


def test_request_only_bundle_is_accepted_only_as_the_whole_disabled_trace(tmp_path):
    disabled = _disabled_candidate("CAL-P", "case-disabled")
    request = canonical_v2_request(disabled, "snapshot-1")
    inputs, _shared = native_inputs_from_capture(disabled, request)
    record = application.score_one(request, inputs)
    assert "UNVALIDATED_STRUCTURE" in record.reason_codes

    # Planted defect 1: a request-only bundle next to groups legacy recorded
    # from later stages is not a request-only row.
    extra = copy.deepcopy(disabled)
    features = {"feature_vector": {}, "missing_mask": {}, "model_identity": {}}
    extra["legacy_trace"]["checkpoints"]["features"] = {
        "value": features, "content_hash": content_hash(features),
    }
    with pytest.raises(StrictTraceCaptureError, match="alongside other checkpoint"):
        canonical_v2_request(extra, "snapshot-1")

    # Planted defect 2: without the request-only bundle a disabled row is
    # still unsupported, never traced from nothing.
    bare = copy.deepcopy(disabled)
    bare["legacy_trace"]["checkpoints"] = {}
    with pytest.raises(StrictTraceCaptureError, match="does not support CAL-P"):
        canonical_v2_request(bare, "snapshot-1")

    # Planted defect 3: a request-only bundle claiming an enabled strategy.
    enabled = copy.deepcopy(disabled)
    enabled["request"]["strategy"] = "STR-THRU"
    row = enabled["legacy_trace"]["checkpoints"]["source_inputs"]
    row["value"]["context"]["strategy"] = "STR-THRU"
    row["content_hash"] = content_hash(row["value"])
    with pytest.raises(StrictTraceCaptureError, match="enabled STR-THRU"):
        native_inputs_from_capture(enabled, canonical_v2_request(enabled, "snapshot-1"))
