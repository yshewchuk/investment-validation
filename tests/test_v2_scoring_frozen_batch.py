"""P6 native frozen batch boundary: ``engine/v2/scoring/frozen_batch.py``.

The batch entrypoint wraps ``application.score_frozen`` for each request of a
declared ``ScoreBatch`` under one pinned snapshot, one resolved release and one
``FrozenInference``, and must reproduce the individual frozen calls
byte-for-byte while preflighting the whole declaration before the first
inference. Fixtures are the P5-2 acceptance ones (``tmp_path``-synthetic
release artifacts, ``test_v2_scoring_native_payoff`` bundles); nothing reads
``data/``.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.contracts import ScoreBatch
from engine.v2.foundation import to_document
from engine.v2.foundation.canonical import canonical_json
from engine.v2.models import (
    MODEL_NOT_READY,
    FrozenInference,
    InferenceRequest,
    RuntimeFitForbidden,
)
from engine.v2.models.no_fit import fitting_forbidden, forbid_fitting
from engine.v2.scoring import application
from engine.v2.scoring.frozen_batch import (
    FrozenBatchPreflightError,
    score_frozen_batch,
)
from engine.v2.scoring.frozen_inputs import build_inference_requests
from engine.v2.scoring.identity import request_hash
from engine.v2.scoring.source_inputs import build_native_score_inputs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v2_models_p5_2_acceptance import (  # noqa: E402
    _INFERENCE_REQUEST,
    CUTOFF,
    _bundle,
    _missing_estimator_release,
    _release,
    _request,
    _tree,
    _write_release,
    both_guards,
)


def _fields(root: Path, hashes: dict) -> dict:
    """The P5-2 READY frozen inputs bundle as ``score_frozen`` wants it."""
    from engine.v2.models.payoff_artifact import PayoffArtifactLoader, PayoffArtifactRef
    from engine.v2.models.recalibration_artifact import (
        RecalibrationArtifactLoader,
        RecalibrationArtifactRef,
    )

    line = PayoffArtifactLoader(root).load(
        PayoffArtifactRef(path="payoff.json", content_hash=hashes["payoff"]))
    recal = RecalibrationArtifactLoader(root).load(
        RecalibrationArtifactRef(path="recal.json", content_hash=hashes["recal"]))
    bundle = _bundle(
        feature_vector={"x": 2.5}, feature_missing_mask={"x": False},
        payoff_artifact_recipe={"before": CUTOFF, "seed": 7, "draw_count": 500},
        payoff_artifact=line, recalibration_artifact=recal,
        model_residual_rows=[{"prediction": 6.0, "residual": -0.5},
                             {"prediction": 6.0, "residual": 0.5}],
    )
    return {"_native_inputs": build_native_score_inputs(bundle)}


def _refusal_fields() -> dict:
    bundle = _bundle(feature_vector={"x": 2.5}, feature_missing_mask={"x": False})
    return {"_native_inputs": build_native_score_inputs(bundle)}


def _nonfinite_only_inputs():
    """A real ``NativeScoreInputs`` whose only feature came back non-finite."""
    bundle = _bundle(feature_vector={"x": {"__nonfinite__": "nan"}},
                     feature_missing_mask={"x": False})
    return build_native_score_inputs(bundle)


def _batch(requests) -> ScoreBatch:
    return ScoreBatch(batch_id="b1", requests=tuple(requests), population_ref="p1")


def _mapping(requests, value) -> dict:
    return {request_hash(request): value for request in requests}


def _kwargs(release, inference, requests, fields) -> dict:
    return dict(
        snapshot_id="snap-1", release=release, inference=inference,
        fields_by_request=_mapping(requests, fields),
        inference_requests_by_request=_mapping(requests, _INFERENCE_REQUEST),
    )


def _document(record) -> str:
    return canonical_json(to_document(record))


def _ready(tmp_path: Path) -> tuple[Path, dict, object]:
    root = tmp_path / "release"
    hashes = _write_release(root)
    return root, hashes, _release(hashes)


class _CountingInference:
    """Wraps a real loader so a preflight failure can be proven to infer nothing."""

    def __init__(self, inner: FrozenInference) -> None:
        self._inner, self.calls = inner, []

    def infer(self, release, request: InferenceRequest):
        self.calls.append(request)
        return self._inner.infer(release, request)


# ---------------------------------------------------------------------------
# batch == individual frozen calls
# ---------------------------------------------------------------------------


def test_batch_equals_individual_frozen_records_byte_for_byte(tmp_path):
    root, hashes, release = _ready(tmp_path)
    before = _tree(root)
    inference = FrozenInference(root)
    fields = _fields(root, hashes)
    plain = _request()
    overridden = replace(plain, geometry_override={"strike": 105.0})

    with both_guards():
        forward = score_frozen_batch(
            _batch([plain, overridden]), **_kwargs(release, inference,
                                                   [plain, overridden], fields))
        singles = tuple(
            application.score_frozen(request, inference, release,
                                     _INFERENCE_REQUEST, fields)
            for request in (plain, overridden))
        reversed_batch = score_frozen_batch(
            _batch([overridden, plain]), **_kwargs(release, inference,
                                                   [overridden, plain], fields))

    assert [forward[0].validation_status, forward[0].readiness] == ["scored", "ready"]
    frozen_source = forward[0].resolved_request["native_source_ref"]
    assert frozen_source.startswith("frozen:r1:")
    assert forward[0].reason_codes == singles[0].reason_codes  # flags/refusals kept
    assert [_document(r) for r in forward] == [_document(r) for r in singles]
    assert [r.score_id for r in forward] == [r.score_id for r in singles]
    # Order follows the declared batch, and a warm re-run is identical.
    assert [_document(r) for r in reversed_batch] == [_document(singles[1]),
                                                      _document(singles[0])]
    assert _tree(root) == before


def test_duplicate_event_with_distinct_overrides_uses_distinct_hashes(tmp_path):
    root, hashes, release = _ready(tmp_path)
    inference = FrozenInference(root)
    fields = _fields(root, hashes)
    first = replace(_request(), geometry_override={"strike": 100.0})
    second = replace(_request(), geometry_override={"strike": 105.0})
    assert first.event_id == second.event_id
    assert request_hash(first) != request_hash(second)

    requests = [first, second]
    kwargs = _kwargs(release, inference, requests, fields)
    with both_guards():
        records = score_frozen_batch(_batch(requests), **kwargs)
        # score_batch's legacy event-only key is not a key here: exact
        # request-hash coverage is the only accepted mapping.
        event_only = {first.event_id: fields}
        with pytest.raises(FrozenBatchPreflightError, match="request-hash"):
            score_frozen_batch(_batch(requests),
                               **{**kwargs, "fields_by_request": event_only})
        with pytest.raises(FrozenBatchPreflightError, match="request-hash"):
            score_frozen_batch(
                _batch(requests),
                **{**kwargs,
                   "fields_by_request": {request_hash(first): fields}})

    assert len({record.score_id for record in records}) == 2
    assert [r.canonical_request["geometry_override"] for r in records] == [
        {"strike": 100.0}, {"strike": 105.0}]
    assert records[0].request_hash != records[1].request_hash
    # each request keeps its own outcome: the unpriceable override refuses on
    # its own, the batch neither propagates nor swallows the refusal.
    assert [r.validation_status for r in records] == ["scored", "refused"]
    assert "NO_CHAIN" in records[1].reason_codes


# ---------------------------------------------------------------------------
# preflight rejects the whole batch before any inference
# ---------------------------------------------------------------------------

_MUTATIONS = {
    "snapshot": lambda kw, keys: kw.update(snapshot_id="snap-2"),
    "deployment": lambda kw, keys: kw.update(
        release=replace(kw["release"], deployment_id="dep-2")),
    "missing-fields-key": lambda kw, keys: kw["fields_by_request"].pop(keys[0]),
    "extra-inference-key": lambda kw, keys: kw["inference_requests_by_request"].update(
        {"not-a-request-hash": _INFERENCE_REQUEST}),
    "inference-names-other-release": lambda kw, keys: kw[
        "inference_requests_by_request"].update(
            {keys[0]: replace(_INFERENCE_REQUEST, release_id="r2")}),
    "binding-strategy": lambda kw, keys: kw.update(
        release=replace(kw["release"], bindings=tuple(
            replace(b, strategy_id="TWIN-P") for b in kw["release"].bindings))),
    "binding-clock": lambda kw, keys: kw.update(
        release=replace(kw["release"], bindings=tuple(
            replace(b, decision_clock_id="open") for b in kw["release"].bindings))),
    "feature-order": lambda kw, keys: kw["inference_requests_by_request"].update(
        {keys[0]: replace(_INFERENCE_REQUEST, feature_order=("y",))}),
    "binding-not-in-release": lambda kw, keys: kw[
        "inference_requests_by_request"].update(
            {keys[0]: replace(_INFERENCE_REQUEST, binding_id="ghost")}),
    "fields-missing-native-inputs": lambda kw, keys: kw[
        "fields_by_request"].update({keys[0]: {}}),
    "none-inference-request": lambda kw, keys: kw[
        "inference_requests_by_request"].update({keys[0]: None}),
    "non-inference-request-value": lambda kw, keys: kw[
        "inference_requests_by_request"].update({keys[0]: "driver"}),
    "non-inference-request-item": lambda kw, keys: kw[
        "inference_requests_by_request"].update({keys[0]: (_INFERENCE_REQUEST, 7)}),
}


@pytest.mark.parametrize("case", sorted(_MUTATIONS))
def test_preflight_mismatch_prevents_all_inference(tmp_path, case):
    root, hashes, release = _ready(tmp_path)
    plain = _request()
    override = replace(plain, geometry_override={"strike": 105.0})
    requests = [plain, override]
    fields = _fields(root, hashes)
    keys = [request_hash(request) for request in requests]
    spy = _CountingInference(FrozenInference(root))

    with both_guards():
        assert spy.calls == []  # control: nothing ran before the batch
        ok = score_frozen_batch(_batch(requests),
                                **_kwargs(release, spy, requests, fields))
        assert spy.calls  # control: a matching batch does infer (folds too)
        spy.calls = []

        kwargs = _kwargs(release, spy, requests, fields)
        _MUTATIONS[case](kwargs, keys)
        with pytest.raises(FrozenBatchPreflightError):
            score_frozen_batch(_batch(requests), **kwargs)
    assert spy.calls == []  # the mismatch refused the batch ahead of inference
    assert len(ok) == len(requests)  # only the complete, matching batch ever ran


# ---------------------------------------------------------------------------
# a legitimately mapped empty inference tuple is a batch, not a batch error
# ---------------------------------------------------------------------------


def test_all_nonfinite_record_maps_to_the_empty_tuple_and_still_batches(tmp_path):
    """``frozen_inputs.build_inference_requests`` legitimately returns ``()``
    when every required feature value came back non-finite, and
    ``score_frozen`` serves that omission as its own no-model refusal. The
    batch preflight must accept the explicitly mapped empty tuple -- and still
    fit nothing and infer nothing for that request."""
    root, hashes, release = _ready(tmp_path)
    before = _tree(root)
    nonfinite = _nonfinite_only_inputs()
    built = build_inference_requests(nonfinite, release.bindings, release.release_id)
    assert built == ()  # the real builder's own all-nonfinite omission

    omitted = _request()
    fields = {request_hash(omitted): {"_native_inputs": nonfinite}}
    mapped = {request_hash(omitted): built}
    spy = _CountingInference(FrozenInference(root))
    guard_state, refusal_attempts = [], []

    def observe(observation):
        guard_state.append(fitting_forbidden())
        try:
            forbid_fitting("frozen-batch-empty-probe")
        except RuntimeFitForbidden:
            refusal_attempts.append(True)

    # no ambient guard: the batch itself must open ``no_fit_guard()``
    [record] = score_frozen_batch(
        _batch([omitted]), snapshot_id="snap-1", release=release, inference=spy,
        fields_by_request=fields, inference_requests_by_request=mapped,
        observer=observe)
    assert spy.calls == []  # the empty mapping never reached ``infer``
    assert guard_state and all(guard_state)  # never a fit-capable code path
    assert len(refusal_attempts) == len(guard_state)
    assert not fitting_forbidden()
    assert (record.validation_status, record.readiness) == ("refused", "refused")
    assert record.forecasts["driver_prediction"] is None
    assert "MISSING_FEATURES" in record.reason_codes
    assert record.resolved_request["native_source_ref"] == "frozen:r1:"

    # byte-for-byte the direct ``score_frozen`` outcome for the same () mapping
    direct = application.score_frozen(omitted, spy, release, built,
                                      fields[request_hash(omitted)])
    assert _document(record) == _document(direct)
    assert record.score_id == direct.score_id

    # and an empty-mapped request rides alongside an inferring one unchanged
    inference = FrozenInference(root)
    finite = replace(omitted, event_id="evt-finite")
    fields[request_hash(finite)] = _fields(root, hashes)
    mapped[request_hash(finite)] = _INFERENCE_REQUEST
    batched = score_frozen_batch(
        _batch([omitted, finite]), snapshot_id="snap-1", release=release,
        inference=inference, fields_by_request=fields,
        inference_requests_by_request=mapped)
    singles = tuple(
        application.score_frozen(request, inference, release,
                                 mapped[request_hash(request)],
                                 fields[request_hash(request)])
        for request in (omitted, finite))
    assert [r.validation_status for r in batched] == ["refused", "scored"]
    assert [_document(r) for r in batched] == [_document(r) for r in singles]
    assert _tree(root) == before  # nothing fitted, fetched or written


def test_later_malformed_native_inputs_payload_refuses_whole_batch(tmp_path):
    """``score_frozen`` type-checks ``_native_inputs`` *after* inferring, so a
    later request's malformed payload must be caught in whole-batch preflight:
    nothing in the batch infers, and the failure is a preflight refusal, not
    the post-inference ``TypeError`` the single call would raise."""
    root, hashes, release = _ready(tmp_path)
    plain = _request()
    later = replace(plain, geometry_override={"strike": 105.0})
    requests = [plain, later]
    keys = [request_hash(request) for request in requests]
    spy = _CountingInference(FrozenInference(root))
    kwargs = _kwargs(release, spy, requests, _fields(root, hashes))
    kwargs["fields_by_request"][keys[1]] = {"_native_inputs": None}

    with both_guards():
        with pytest.raises(FrozenBatchPreflightError) as excinfo:
            score_frozen_batch(_batch(requests), **kwargs)

    assert spy.calls == []  # the first request never ran ahead of the bad payload
    error = excinfo.value
    assert type(error) is FrozenBatchPreflightError
    assert isinstance(error, ValueError) and not isinstance(error, TypeError)
    assert "NativeScoreInputs" in str(error)
    assert keys[1] in str(error)  # the offending request is named


def test_missing_artifact_gives_refusal_not_a_preflight_error(tmp_path):
    """Preflight never opens artifact bytes: the P5-2 MODEL_NOT_READY refusal
    from ``score_frozen`` survives batching unchanged."""
    root, release = _missing_estimator_release(tmp_path)
    before = _tree(root)
    inference = FrozenInference(root)
    fields = _refusal_fields()
    plain = _request()
    override = replace(plain, geometry_override={"strike": 105.0})
    requests = [plain, override]

    with both_guards():
        batched = score_frozen_batch(_batch(requests),
                                     **_kwargs(release, inference, requests, fields))
        singles = tuple(
            application.score_frozen(request, inference, release,
                                     _INFERENCE_REQUEST, fields)
            for request in requests)

    assert all(MODEL_NOT_READY in record.reason_codes for record in batched)
    assert all("ARTIFACT_INVALID" in record.reason_codes for record in batched)
    assert all(record.forecasts["driver_prediction"] is None for record in batched)
    assert [record.validation_status for record in batched] == ["refused", "refused"]
    assert [_document(r) for r in batched] == [_document(r) for r in singles]
    assert inference.cache_size == 0
    assert _tree(root) == before


# ---------------------------------------------------------------------------
# no fitting inside the batch
# ---------------------------------------------------------------------------


def test_batch_execution_cannot_fit(tmp_path):
    root, hashes, release = _ready(tmp_path)
    inference = FrozenInference(root)
    fields = _fields(root, hashes)
    plain = _request()
    requests = [plain, replace(plain, geometry_override={"strike": 100.0})]
    guard_state, refusal_attempts = [], []

    def observe(observation):
        guard_state.append(fitting_forbidden())
        try:
            forbid_fitting("frozen-batch-probe")
        except RuntimeFitForbidden:
            refusal_attempts.append(True)

    # No ambient guard: the batch itself must open ``no_fit_guard()``.
    records = score_frozen_batch(_batch(requests),
                                 **{**_kwargs(release, inference, requests, fields),
                                    "observer": observe})
    assert [r.validation_status for r in records] == ["scored", "scored"]
    assert guard_state and all(guard_state)
    assert len(refusal_attempts) == len(guard_state)
    assert not fitting_forbidden()

    # Control: the same probe outside the batch sees no guard and can fit.
    outside = []
    application.score_frozen(
        plain, inference, release, _INFERENCE_REQUEST, fields,
        observer=lambda observation: outside.append(fitting_forbidden()))
    assert outside and not any(outside)
