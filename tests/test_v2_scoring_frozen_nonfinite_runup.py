"""A present frozen binding with a non-finite captured row must refuse MISSING_FEATURES.

``checks/phase4_frozen_bridge.py::binding_feature_row``/``_feature_rows``
intentionally omit an inference request when a captured feature row comes back
non-finite -- never feed NaN to a model, never weaken strict trace validation.
The native replay must still preserve legacy's ``MISSING_FEATURES`` refusal for
that PRESENT binding (``Scorer._score_model``/``_score_runup_model`` flag it on
the same non-finite predicate), and the refusal must stay distinct from a
genuinely ABSENT binding, which legacy reports through a forecast-output
absence code instead.

The driver/size shape already preserved it: ``_ordinary_executor_specs``
resolves a canonical target, so a frozen stage executor is built, re-reads the
recorded ``role_model_inputs`` at stage time, and ``FrozenStageExecutor._row``
raises ``MISSING_FEATURES``. The unmappable shape did not: with no
``days_before_print`` in the row, ``_runup_executor_spec`` resolves no target,
the executor loop registered nothing for a present-but-omitted runup binding,
the stage never re-read its row, and the refusal was lost -- the replay fell
through to ``MISSING_FORECAST_INPUT``/``MISSING_FORECAST_OUTPUT:runup_move``,
codes legacy never stamps on such a row. Fixed in
``engine/v2/scoring/application.py::_frozen_stage_executors``.
"""
from __future__ import annotations

from dataclasses import replace

from checks import phase4_real
from engine.v2.foundation import content_hash
from engine.v2.scoring import application
from tests.test_phase4_capture_strict import _artifact, _full_strict_candidate
from tests.test_v2_scoring_runup_frozen import (
    _Frozen, _analog_block, _inference_requests, _inputs, _release, _request,
)
from tools.capture_tier0_corpus import _hydrate_trace, attach_strict_probe

CLOCK = "legacy.decision_offset.0"
GATE = {"n_prior": 5.0, "x": 9.0}
NONFINITE = {"__nonfinite__": "nan"}


def _round_trip(tmp_path, monkeypatch, *, strategy, driver_vector, as_runup):
    """Capture one strict trace, then replay it through the verified bridge.

    Returns ``(reason_codes, omitted_roles)``: the native record's flags and
    the release binding roles left without an inference request (omitted by
    the bridge for a non-finite captured row).
    """
    monkeypatch.setattr("engine.paths.ROOT", tmp_path)
    path, digest = _artifact(tmp_path)
    candidate = _full_strict_candidate(
        fixture_id="probe-0", ticker="LUXE", strategy=strategy,
        driver_vector=driver_vector, gate_vector=GATE, path=path, digest=digest,
    )
    source = candidate["legacy_trace"]["checkpoints"]["source_inputs"]
    for binding in source["value"]["model_bindings"]:
        binding["decision_clock"] = CLOCK
        binding["strategy"] = strategy
        if as_runup and binding["role"] == "abs_move":
            binding["role"] = "runup_move"
            binding["output_names"] = ["pred_runup_abs_move_d14"]
    source["content_hash"] = content_hash(source["value"])

    features = candidate["legacy_trace"]["checkpoints"]["features"]
    if as_runup:
        value = features["value"]
        for section in ("feature_vector", "missing_mask", "model_identity"):
            block = value[section]
            identity = block.pop("abs_move")
            if section == "model_identity":
                identity = {**identity, "role": "runup_move"}
            block["runup_move"] = identity
    features["content_hash"] = content_hash(features["value"])

    attached, gaps = attach_strict_probe([candidate], "snapshot-1", tmp_path)
    assert attached == ("probe-0",) and gaps == {}
    trace = _hydrate_trace(candidate["input_trace"])
    pair = {"payload": {
        "request": candidate["request"], "input_trace": trace,
        "input_trace_hash": trace["trace_hash"],
        "legacy_input_hash": candidate["legacy_input_hash"],
    }}
    verified = phase4_real._verified_trace_bundle(pair, tmp_path)
    native, _, _ = phase4_real._replayed_member(verified)
    plan = verified["frozen_replay"]
    served = {request.binding_id for request in plan.requests}
    omitted = [binding.role for binding in plan.release.bindings
               if binding.binding_id not in served]
    return tuple(native.reason_codes), omitted


def test_present_runup_binding_with_nonfinite_row_refuses_missing_features(tmp_path, monkeypatch):
    """The regression: no canonical runup target is resolvable (no horizon),
    yet the present binding's non-finite row must still yield
    MISSING_FEATURES -- not MISSING_FORECAST_INPUT, and not the
    MISSING_FORECAST_OUTPUT that legacy never stamps on such a row. The
    same record's genuinely absent implied_t1 role keeps reporting the
    absence code: present-non-finite and absent stay distinct."""
    reasons, omitted = _round_trip(
        tmp_path, monkeypatch, strategy="STR-RUNUP",
        driver_vector={"x": NONFINITE}, as_runup=True,
    )
    assert "runup_move" in omitted  # the bridge omitted the non-finite request
    assert "MISSING_FEATURES" in reasons
    assert "MISSING_FORECAST_INPUT" not in reasons
    assert "MISSING_FORECAST_OUTPUT:runup_move" not in reasons
    assert "MISSING_FORECAST_OUTPUT:implied_t1" in reasons


def test_present_runup_binding_with_finite_row_is_not_refused(tmp_path, monkeypatch):
    """Distinctness guard: a finite runup row is served, not omitted -- the
    refusal above is keyed to the non-finite captured row, never to the mere
    absence of a resolvable canonical runup target."""
    reasons, omitted = _round_trip(
        tmp_path, monkeypatch, strategy="STR-RUNUP",
        driver_vector={"x": 2.0, "days_before_print": 7.0}, as_runup=True,
    )
    assert "runup_move" not in omitted
    assert "MISSING_FEATURES" not in reasons
    assert "MISSING_FORECAST_OUTPUT:runup_move" not in reasons


def test_absent_forecast_role_reports_absence_not_missing_features(tmp_path, monkeypatch):
    """A forecast role with no binding in the release at all is genuinely
    absent: it must report the output-absence code and must NOT borrow the
    present-binding MISSING_FEATURES refusal."""
    reasons, omitted = _round_trip(
        tmp_path, monkeypatch, strategy="STR-RUNUP",
        driver_vector={"x": 2.0}, as_runup=False,
    )
    assert omitted == []  # the served driver row is finite; nothing was omitted
    assert "MISSING_FEATURES" not in reasons
    assert "MISSING_FORECAST_OUTPUT:runup_move" in reasons


# -- the Astra P2: an unrequested binding is not the same as an omitted one ----


def test_subset_requests_never_gain_a_fallback_value():
    """``score_frozen`` legitimately accepts a SUBSET of the release's
    inference requests. A present-but-unrequested binding whose captured row
    is FINITE must not register the refusal fallback: doing so let the
    pre-transformed runup guard (which lives in the result path) be bypassed
    purely by omitting the request, so the row came back scored with the
    frozen value. Astra P2 integrity blocker."""
    frozen = _Frozen()
    frozen.runup_output = "runup_move_prediction"
    inputs = replace(_inputs(), analogs=_analog_block())
    record = application.score_frozen(
        _request(), frozen, _release(frozen.runup_output),
        _inference_requests()[:1], {"_native_inputs": inputs},
    )
    assert record.forecasts.get("runup_move_prediction") is None
    assert record.validation_status == "refused"
    # The subset path refuses through the absence code it always used, never
    # by silently publishing the omitted binding's frozen value.
    assert "MISSING_FORECAST_OUTPUT:runup_move" in record.reason_codes


def test_subset_binding_with_nonfinite_row_still_refuses_missing_features():
    """The fallback is for OMITTED (non-finite-row) bindings only. A subset
    caller that simply never submits a request whose captured row reads
    non-finite still gets MISSING_FEATURES -- the two refusal reasons stay
    distinguishable."""
    frozen = _Frozen()
    inputs = replace(
        _inputs(),
        analogs=_analog_block(),
        features={
            "role_model_inputs": {"runup_move": {"days_before_print": float("nan")}},
            "days_before_print": 14.0,
            "model_inputs": {"days_before_print": 14.0},
        },
    )
    record = application.score_frozen(
        _request(), frozen, _release(frozen.runup_output),
        _inference_requests()[:1], {"_native_inputs": inputs},
    )
    assert "MISSING_FEATURES" in record.reason_codes
    assert record.forecasts.get("runup_move_prediction") is None
    assert "MISSING_FORECAST_OUTPUT:runup_move" not in record.reason_codes
    assert record.validation_status == "refused"


def test_omitted_refusal_survives_runtime_horizon_replacement():
    """Astra P2: the fallback is refusal-only. Here the binding's CAPTURED
    model_inputs row is non-finite (days_before_print NaN), so the bridge
    would omit it and legacy flags MISSING_FEATURES. But the runtime-assembled
    facts (stages._facts) layer computed `values` over the captured cell and
    the context horizon (7.0) replaces the NaN, so a real FrozenStageExecutor
    would READ A FINITE ROW, accept it and publish the unmapped pretransformed
    prediction. The refusal is pinned to the captured row, so replacement
    cannot turn it into a value."""
    f = _Frozen()
    f.runup_output = "runup_move_prediction"
    inputs = replace(
        _inputs(), analogs=_analog_block(), forecast={},
        features={"model_inputs": {"days_before_print": float("nan")}},
    )
    record = application.score_frozen(
        _request(), f, _release(f.runup_output),
        _inference_requests()[:1], {"_native_inputs": inputs},
    )
    # The captured horizon is genuinely non-finite (the context 7.0 the stage
    # assembles later is exactly the replacement this guard must resist).
    assert record.forecasts.get("runup_move_prediction") is None
    assert "MISSING_FEATURES" in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:runup_move" not in record.reason_codes
    assert record.validation_status == "refused"
