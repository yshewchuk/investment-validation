"""Compatibility and import boundaries between providers and orchestration."""
from engine.v2.ops import incremental_data, provider_response
from engine.v2.ops.providers import orats_daily_market
from tools import mutation_pilot


def test_compatibility_names_and_provider_share_one_implementation():
    assert incremental_data.AcquisitionOutcome is provider_response.AcquisitionOutcome
    assert incremental_data.OutcomeKind is provider_response.OutcomeKind
    assert incremental_data.classify_response is provider_response.classify_response
    assert orats_daily_market.classify_response is provider_response.classify_response


def test_leaf_only_private_helpers_are_not_reexported_by_orchestration():
    assert incremental_data._keys is provider_response._keys
    assert not hasattr(incremental_data, "_response_kind")
    assert not hasattr(incremental_data, "_subset")


def test_existing_retry_policy_accepts_leaf_outcomes():
    complete = provider_response.classify_response(
        200, ("AAA",), returned_keys=("AAA",), cache_hit=True)
    assert incremental_data.missing_keys(complete) == ()
    assert incremental_data.coverage_complete(complete)
    assert incremental_data.retry_action(complete) == "use_cache"
    partial = provider_response.classify_response(
        200, ("BBB", "AAA"), returned_keys=("AAA",), cache_hit=True)
    assert incremental_data.missing_keys(partial) == ("BBB",)
    assert not incremental_data.coverage_complete(partial)
    assert incremental_data.retry_action(partial) == "retry_missing"


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
        "tests/v2/ops/test_provider_response.py",
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
