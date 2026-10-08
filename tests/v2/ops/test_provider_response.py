"""Classification stays compatible without pulling orchestration into providers."""
from dataclasses import FrozenInstanceError

import pytest

from engine.v2.ops import incremental_data, provider_response
from engine.v2.ops.errors import OpsError
from engine.v2.ops.providers import orats_daily_market
from tools import mutation_pilot


@pytest.mark.parametrize(("status", "options", "kind", "retry"), [
    (200, {"returned_keys": ("AAA",)}, "complete", "stop"),
    (200, {"empty_keys": ("AAA",)}, "empty", "stop"),
    (200, {"unsupported_keys": ("AAA",)}, "complete", "stop"),
    (200, {}, "partial", "retry_missing"),
    (200, {"returned_keys": ("AAA",), "truncated": True}, "partial", "retry_missing"),
    (200, {"final": False, "truncated": True}, "not_final", "retry"),
    (200, {"credential_page": True, "final": False}, "credential_invalid", "stop"),
    (401, {}, "credential_invalid", "stop"),
    (403, {}, "credential_invalid", "stop"),
    (429, {"final": False}, "rate_limited", "retry"),
    (503, {"final": False}, "transient", "retry"),
    (404, {"final": False}, "unsupported", "stop"),
    (302, {}, "transient", "retry"),
    (199, {}, "transient", "retry"),
    (299, {"returned_keys": ("AAA",)}, "complete", "stop"),
])
def test_classification_and_existing_retry_policy(status, options, kind, retry):
    outcome = provider_response.classify_response(status, ("AAA",), **options)
    assert outcome.kind == kind
    assert incremental_data.retry_action(outcome) == retry


def test_compatibility_names_and_provider_share_one_implementation():
    assert incremental_data.AcquisitionOutcome is provider_response.AcquisitionOutcome
    assert incremental_data.OutcomeKind is provider_response.OutcomeKind
    assert incremental_data.classify_response is provider_response.classify_response
    assert orats_daily_market.classify_response is provider_response.classify_response


def test_sorted_coverage_and_redacted_metadata_survive_extraction():
    outcome = provider_response.classify_response(
        200, ("DDD", "CCC", "BBB", "AAA"), returned_keys=("DDD", "AAA"),
        empty_keys=("BBB",), unsupported_keys=("CCC",), request_id="request-1",
        receipt_ref="receipt-1", raw_hash="sha256:" + "a" * 64,
        cache_hit=True, quota_remaining=0)
    assert outcome == incremental_data.AcquisitionOutcome(
        request_id="request-1", kind="complete",
        requested_keys=("AAA", "BBB", "CCC", "DDD"), returned_keys=("AAA", "DDD"),
        empty_keys=("BBB",), unsupported_keys=("CCC",), receipt_ref="receipt-1",
        raw_hash="sha256:" + "a" * 64, cache_hit=True, quota_remaining=0)
    assert incremental_data.missing_keys(outcome) == ()
    assert incremental_data.coverage_complete(outcome)
    assert incremental_data.retry_action(outcome) == "use_cache"
    with pytest.raises(FrozenInstanceError):
        outcome.kind = "partial"


def test_no_requested_keys_is_complete_and_partial_cache_still_retries():
    assert provider_response.classify_response(200, ()).kind == "complete"
    partial = provider_response.classify_response(
        200, ("BBB", "AAA"), returned_keys=("AAA",), cache_hit=True)
    assert incremental_data.missing_keys(partial) == ("BBB",)
    assert not incremental_data.coverage_complete(partial)
    assert incremental_data.retry_action(partial) == "retry_missing"


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


def test_provider_and_credential_import_closures_exclude_refresh_runtime():
    graph = mutation_pilot.build_import_graph()
    unresolved = mutation_pilot.unresolved_import_files(graph)
    roots = {
        "engine/v2/ops/provider_response.py",
        "engine/v2/ops/providers/__init__.py",
        "engine/v2/ops/providers/orats_daily_market.py",
        "engine/v2/ops/providers/nasdaq_calendar.py",
        "engine/v2/ops/providers/yfinance_edge.py",
        "tests/test_v2_ops_provider_credentials.py",
        "tests/test_v2_ops_providers_nasdaq.py",
        "tests/test_v2_ops_providers_yfinance.py",
    }
    assert roots <= graph.keys()
    closure, tainted = mutation_pilot._closure_from_roots(
        roots, graph, unresolved, taint_exempt=set())
    assert not tainted
    assert "engine/v2/ops/incremental_data.py" not in closure
    assert "engine/v2/ops/legacy_adapter.py" not in closure
    assert not {path for path in closure
                if path.startswith("engine/") and not path.startswith("engine/v2/")
                and path != "engine/__init__.py"}
