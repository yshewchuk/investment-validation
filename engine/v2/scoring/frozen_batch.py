"""Phase 6 native frozen batch boundary over ``application.score_frozen``.

``application.score_batch`` keys inputs by request hash but runs the Phase 4
``score_one`` kernel, bypassing the Phase 5 frozen inference path entirely.
This module is the batch-shaped production entrypoint that does not: one
declared ``ScoreBatch`` is scored by delegating each request, in order, to
``application.score_frozen`` under one explicitly pinned snapshot, one
resolved ``ModelRelease`` and one ``FrozenInference``, so a batch and the
identical sequence of individual ``score_frozen`` calls produce byte-for-byte
the same records — same order, same frozen evidence, same refusals, same
score IDs.

The batch is a unit: preflight validates the whole declaration before the
first inference. Every request must name the pinned snapshot and the release
deployment; the two per-request mappings are keyed *only* by
``application.identity.request_hash`` values with exact coverage (nothing
missing, nothing extra — unlike ``score_batch`` there is no event-only
fallback); every ``InferenceRequest`` must name the release, and the release
binding it selects must be compatible with the request's strategy, decision
clock and feature order. Every request's ``_native_inputs`` payload must
already be a ``NativeScoreInputs`` instance: ``score_frozen`` type-checks it
only *after* running inference, so a malformed payload on a later request
would otherwise let earlier requests infer before the batch dies with a
``TypeError``.

Preflight deliberately never opens artifact bytes. A missing or tampered
member is verified at inference time exactly as an individual call would, so
the batch preserves ``score_frozen``'s MODEL_NOT_READY refusal records (P5-2)
rather than turning them into batch errors. Execution runs inside
``engine.v2.models.no_fit_guard``: every registered v2 fitting path — each
one that opens with ``engine.v2.models.no_fit.forbid_fitting`` — refuses to
run for the block. The guard itself only trips at those registered call
sites; it does not by itself prevent a provider warm or an arbitrary cache
write. The batch path's guarantee against those is structural: every request
runs through ``score_frozen``, which serves only the pinned release's
hash-verified frozen artifacts and fits or fetches nothing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from engine.v2.contracts import ScoreBatch, ScoreRecord, ScoreRequest
from engine.v2.models.no_fit import no_fit_guard

from .application import score_frozen
from .identity import request_hash
from .stages import NativeScoreInputs

if TYPE_CHECKING:
    from engine.v2.models.contracts import InferenceRequest, ModelRelease
    from engine.v2.models.loader import FrozenInference

    from .stages import StageObserver

__all__ = ["FrozenBatchPreflightError", "score_frozen_batch"]

_REQUIRED_FIELD = "_native_inputs"


class FrozenBatchPreflightError(ValueError):
    """The declared batch cannot run as one frozen unit; nothing was inferred."""


def score_frozen_batch(
    batch: ScoreBatch,
    *,
    snapshot_id: str,
    release: ModelRelease,
    inference: FrozenInference,
    fields_by_request: Mapping[str, Mapping[str, Any]],
    inference_requests_by_request: Mapping[
        str, InferenceRequest | tuple[InferenceRequest, ...]],
    observer: StageObserver | None = None,
) -> tuple[ScoreRecord, ...]:
    """Score one declared batch through ``score_frozen`` as a single frozen unit.

    ``fields_by_request`` maps each ``request_hash`` to the ``fields`` mapping
    handed to ``score_frozen`` (``{"_native_inputs": NativeScoreInputs}``);
    ``inference_requests_by_request`` maps the same hash to one
    ``InferenceRequest`` or a tuple of them. Records come back in
    ``batch.requests`` order, identical to individual ``score_frozen`` calls
    with the same pinned ``snapshot_id``, ``release`` and ``inference``.
    Raises ``FrozenBatchPreflightError`` before any inference if the batch
    does not name one coherent frozen unit.
    """
    requests = tuple(batch.requests)
    _preflight(requests, snapshot_id, release, fields_by_request,
               inference_requests_by_request)
    with no_fit_guard():
        return tuple(
            score_frozen(request, inference, release,
                         inference_requests_by_request[request_hash(request)],
                         fields_by_request[request_hash(request)],
                         observer=observer)
            for request in requests
        )


def _preflight(requests: tuple[ScoreRequest, ...], snapshot_id: str,
               release: ModelRelease,
               fields_by_request: Mapping[str, Mapping[str, Any]],
               inference_requests_by_request: Mapping[str, Any]) -> None:
    """Validate the whole batch declaration before the first inference."""
    identities = tuple(request_hash(request) for request in requests)
    _check_coverage(identities, fields_by_request, inference_requests_by_request)
    for request, identity in zip(requests, identities, strict=True):
        _check_pinned_identity(request, identity, snapshot_id, release)
        fields = fields_by_request[identity]
        if not isinstance(fields, Mapping) or _REQUIRED_FIELD not in fields:
            raise FrozenBatchPreflightError(
                f"request {identity} fields must map {_REQUIRED_FIELD!r} to "
                "source-built NativeScoreInputs; artifact content is verified "
                "at inference, not here")
        payload = fields[_REQUIRED_FIELD]
        if not isinstance(payload, NativeScoreInputs):
            raise FrozenBatchPreflightError(
                f"request {identity} maps {_REQUIRED_FIELD!r} to "
                f"{type(payload).__name__}, not a NativeScoreInputs payload; "
                "score_frozen would only fail on it after inferring this and "
                "any earlier requests, so the whole batch is refused here "
                "without opening artifact bytes")
        items = _inference_items(inference_requests_by_request[identity], identity)
        for item in items:
            _check_inference_request(request, identity, release, item)


def _check_coverage(identities: tuple[str, ...], fields_by_request: Mapping[str, Any],
                    inference_requests_by_request: Mapping[str, Any]) -> None:
    wanted = set(identities)
    for label, mapping in (("fields", fields_by_request),
                           ("inference request", inference_requests_by_request)):
        for direction, offenders in (("missing", sorted(wanted - set(mapping))),
                                     ("has unexpected", sorted(set(mapping) - wanted))):
            if offenders:
                raise FrozenBatchPreflightError(
                    f"frozen batch {label} mapping {direction} request-hash "
                    f"key(s) {offenders}; keys must be exactly the "
                    "application.identity.request_hash values of the batch requests")


def _check_pinned_identity(request: ScoreRequest, identity: str, snapshot_id: str,
                           release: ModelRelease) -> None:
    if request.snapshot_id != snapshot_id:
        raise FrozenBatchPreflightError(
            f"request {identity} pins snapshot {request.snapshot_id!r}, not the "
            f"batch snapshot {snapshot_id!r}")
    if request.deployment_id != release.deployment_id:
        raise FrozenBatchPreflightError(
            f"request {identity} targets deployment {request.deployment_id!r}, "
            f"not the release deployment {release.deployment_id!r}")


def _inference_items(value: Any, identity: str) -> tuple[Any, ...]:
    items = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if not items:
        raise FrozenBatchPreflightError(
            f"request {identity} maps to an empty inference request tuple")
    return items


def _check_inference_request(request: ScoreRequest, identity: str,
                             release: ModelRelease, item: Any) -> None:
    if item.release_id != release.release_id:
        raise FrozenBatchPreflightError(
            f"inference request for {identity} names release {item.release_id!r}, "
            f"not the batch release {release.release_id!r}")
    matches = tuple(binding for binding in release.bindings
                    if binding.binding_id == item.binding_id)
    if len(matches) != 1:
        raise FrozenBatchPreflightError(
            f"inference request binding {item.binding_id!r} for request {identity} "
            f"matches {len(matches)} release bindings; exactly one is required")
    binding = matches[0]
    if binding.strategy_id != request.strategy_version:
        raise FrozenBatchPreflightError(
            f"binding {binding.binding_id!r} serves strategy "
            f"{binding.strategy_id!r}, not the request strategy "
            f"{request.strategy_version!r}")
    if binding.decision_clock_id != request.decision_clock_id:
        raise FrozenBatchPreflightError(
            f"binding {binding.binding_id!r} serves clock "
            f"{binding.decision_clock_id!r}, not the request clock "
            f"{request.decision_clock_id!r}")
    if tuple(binding.feature_order) != tuple(item.feature_order):
        raise FrozenBatchPreflightError(
            f"binding {binding.binding_id!r} expects feature order "
            f"{tuple(binding.feature_order)!r}, the inference request supplies "
            f"{tuple(item.feature_order)!r}")
