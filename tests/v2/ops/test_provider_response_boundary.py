"""Compatibility and import boundaries between providers and orchestration."""
import ast
from pathlib import Path

from engine.v2.ops import incremental_data, provider_response
from engine.v2.ops.providers import orats_daily_market

ROOT = Path(__file__).resolve().parents[3]


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


def _source_imports(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    modules, targets = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "boundary modules use explicit absolute imports"
            modules.add(node.module)
            targets.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Name):
            assert node.id not in {"__import__", "eval", "exec"}
    assert not {name.split(".")[0] for name in modules} & {
        "importlib", "runpy", "subprocess"}
    return modules, {target for target in targets if target.startswith("engine.")}


def test_provider_and_pure_test_direct_dependency_contracts():
    # Bounded source contracts, not a whole-repository transitive graph rebuild.
    contracts = {
        "engine/v2/ops/provider_response.py": {"engine.v2.ops.errors.fail"},
        "engine/v2/ops/providers/__init__.py": {
            "engine.v2.ops.providers.nasdaq_calendar.nasdaq_calendar_fetcher",
            "engine.v2.ops.providers.orats_daily_market.orats_daily_market_fetcher",
            "engine.v2.ops.providers.yfinance_edge.yfinance_earnings_fetcher",
            "engine.v2.ops.providers.yfinance_edge.yfinance_history_fetcher"},
        "engine/v2/ops/providers/orats_daily_market.py": {
            "engine.v2.foundation.canonical_json", "engine.v2.ops.errors.fail",
            "engine.v2.ops.provider_response.classify_response"},
        "engine/v2/ops/providers/nasdaq_calendar.py": {"engine.v2.ops.errors.fail"},
        "engine/v2/ops/providers/yfinance_edge.py": set(),
        "tests/v2/ops/test_provider_response.py": {
            "engine.v2.ops.provider_response", "engine.v2.ops.errors.OpsError"},
    }
    for path, expected in contracts.items():
        modules, targets = _source_imports(path)
        assert targets == expected, path
        if path == "engine/v2/ops/provider_response.py":
            assert modules == {"__future__", "dataclasses", "typing", "engine.v2.ops.errors"}
