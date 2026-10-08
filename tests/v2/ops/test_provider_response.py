"""Pure provider-response classification and validation, without orchestration."""
from dataclasses import FrozenInstanceError

import pytest

from engine.v2.ops import provider_response
from engine.v2.ops.errors import OpsError


@pytest.mark.parametrize(("status", "options", "kind"), [
    (200, {"unsupported_keys": ("AAA",)}, "complete"),
    (200, {"returned_keys": ("AAA",), "truncated": True}, "partial"),
    (200, {"final": False, "truncated": True}, "not_final"),
    (200, {"credential_page": True, "final": False}, "credential_invalid"),
    (403, {}, "credential_invalid"),
    (429, {"final": False}, "rate_limited"),
    (503, {"final": False}, "transient"),
    (404, {"final": False}, "unsupported"),
    (302, {}, "transient"),
    (199, {}, "transient"),
    (299, {"returned_keys": ("AAA",)}, "complete"),
])
def test_classification_precedence(status, options, kind):
    outcome = provider_response.classify_response(status, ("AAA",), **options)
    assert outcome.kind == kind


def test_sorted_coverage_and_redacted_metadata_survive_extraction():
    outcome = provider_response.classify_response(
        200, ("DDD", "CCC", "BBB", "AAA"), returned_keys=("DDD", "AAA"),
        empty_keys=("BBB",), unsupported_keys=("CCC",), request_id="request-1",
        receipt_ref="receipt-1", raw_hash="sha256:" + "a" * 64,
        cache_hit=True, quota_remaining=0)
    assert outcome == provider_response.AcquisitionOutcome(
        request_id="request-1", kind="complete",
        requested_keys=("AAA", "BBB", "CCC", "DDD"), returned_keys=("AAA", "DDD"),
        empty_keys=("BBB",), unsupported_keys=("CCC",), receipt_ref="receipt-1",
        raw_hash="sha256:" + "a" * 64, cache_hit=True, quota_remaining=0)
    with pytest.raises(FrozenInstanceError):
        outcome.kind = "partial"


def test_no_requested_keys_is_complete():
    assert provider_response.classify_response(200, ()).kind == "complete"


@pytest.mark.parametrize("field", [
    "requested_keys", "returned_keys", "empty_keys", "unsupported_keys",
])
@pytest.mark.parametrize("values", [("",), ("AAA", "AAA")])
def test_invalid_keys_keep_typed_refusal_even_for_auth_response(field, values):
    options = {"requested_keys": ("AAA",), field: values}
    with pytest.raises(OpsError) as caught:
        provider_response.classify_response(401, **options)
    assert caught.value.code == "INVALID_REQUEST"
    assert caught.value.problem.message == f"{field} must contain unique nonempty keys"
    assert not caught.value.problem.retryable


@pytest.mark.parametrize("field", ["returned_keys", "empty_keys", "unsupported_keys"])
def test_unrequested_keys_keep_typed_refusal(field):
    with pytest.raises(OpsError) as caught:
        provider_response.classify_response(200, ("AAA",), **{field: ("BBB",)})
    assert caught.value.code == "INVALID_REQUEST"
    assert caught.value.problem.message == f"{field} contains an unrequested key"


def test_negative_quota_keeps_typed_refusal():
    with pytest.raises(OpsError) as caught:
        provider_response.classify_response(429, ("AAA",), quota_remaining=-1)
    assert caught.value.code == "INVALID_REQUEST"
    assert caught.value.problem.message == "negative provider quota observation"
