"""Standalone explicit policy validation; no runner, quote reader or fitter."""
from copy import deepcopy

import pytest

from engine.v2.foundation import artifact_reference, to_document
from engine.v2.ops.errors import OpsError
from experiments.distribution_book import validate_book_config
from experiments.distribution_run_request import PRIMARY_ARMS, validate_request


def request():
    ref = to_document(artifact_reference(b"fixture", "payoff_line_artifact.v1"))
    return {"schema_version": "distribution_run_request.v1", "quote_snapshot_id": "quotes", "alphas": [0, .25, .5],
        "bridges": [{"outer_year": year, "alpha": alpha, "ref": ref} for year in ("2023", "2024") for alpha in (0, .25, .5)],
        "benchmarks": [], "required_arms": list(PRIMARY_ARMS), "strict_benchmarks": False,
        "economics": {"currency": "USD", "multiplier": 100, "commission": 0, "exchange_fee": 0,
            "minimum_side_fee": 0, "slippage_per_share_per_leg": 0, "max_relative_spread": 1,
            "capacity_policy": "not_established_no_depth", "adjusted_contracts": "unsupported_native_standard"},
        "book": {"initial_cash": 1000, "contracts_per_trade": 1, "max_trade_debit": 1000,
            "max_ticker_debit": 1000, "max_concurrent": 10, "settlement_delay_days": 0,
            "same_time_order": "settlements_before_entries"}, "minimum_net_ev_usd": 1,
        "report_end": "2025-01-01", "uncertainty": {"seed": 0, "resamples": 4, "block_days": 7,
            "block_origin": "2023-01-01", "confidence": .95},
        "risk": {"minimum_uplift_usd": 0, "maximum_loss_usd": 1000,
            "maximum_settled_drawdown_usd": 1000, "completeness_floor": 1},
        "limits": {"parent": {"memory_bytes": 4 << 30, "cpu_seconds": 60, "wall_seconds": 60}, "job_wall_seconds": 60},
        "max_opportunities": 4, "decision_horizon": 1, "availability_policy": "reconstructed_snapshot_quotes",
        "selector_policy": "fixed_preregistered_net_ev.v1", "primary_alpha": .25}


def _error(action, code="INVALID_EXPERIMENT_SPEC"):
    with pytest.raises(OpsError) as caught:
        action()
    assert caught.value.code == code
    return caught.value


@pytest.mark.parametrize("missing", sorted(request()))
def test_request_requires_every_policy_field(missing):
    value = request()
    del value[missing]
    _error(lambda: validate_request(value))


def test_request_detaches_nested_policy_and_accepts_no_extra_fields():
    value = request()
    saved = validate_request(value)
    value["economics"]["commission"] = 999
    assert saved["economics"]["commission"] == 0
    _error(lambda: validate_request({**request(), "implicit_defaults": True}))


@pytest.mark.parametrize("key,value", [("decision_horizon", True), ("strict_benchmarks", 1),
    ("alphas", [0, .25]), ("minimum_net_ev_usd", 0), ("primary_alpha", .75), ("required_arms", ["histogram"]),
    ("selector_policy", "choose_best_outer"), ("report_end", "bad-date")])
def test_request_rejects_ambiguous_or_undeclared_policy(key, value):
    _error(lambda: validate_request({**request(), key: value}))


@pytest.mark.parametrize("alpha", [False, [], -1, 2, float("nan")])
def test_bridge_entry_requires_declared_plain_numeric_alpha(alpha):
    value = request()
    value["bridges"][0]["alpha"] = alpha
    _error(lambda: validate_request(value))


@pytest.mark.parametrize("changes", [
    {"initial_cash": 0}, {"initial_cash": float("inf")}, {"initial_cash": 10**1000},
    {"contracts_per_trade": True}, {"contracts_per_trade": 1.5}, {"max_concurrent": 0},
    {"max_trade_debit": -1}, {"max_ticker_debit": float("nan")},
    {"settlement_delay_days": -.5}, {"settlement_delay_days": True}, {"same_time_order": "guess"}])
def test_standalone_book_policy_refuses_invalid_limits(changes):
    _error(lambda: validate_book_config(request()["book"] | changes))


def test_standalone_book_policy_is_closed_explicit_and_detached():
    policy = request()["book"]
    original = deepcopy(policy)
    result = validate_book_config(policy)
    assert result == original
    result["initial_cash"] = -1
    assert policy == original
    for key in policy:
        _error(lambda: validate_book_config({name: value for name, value in policy.items() if name != key}))
    _error(lambda: validate_book_config(policy | {"implicit_default": 1}))


@pytest.mark.parametrize("path,value", [
    (("economics", "commission"), -1), (("economics", "exchange_fee"), float("inf")),
    (("economics", "multiplier"), 50), (("economics", "multiplier"), True),
    (("economics", "currency"), "EUR"), (("economics", "capacity_policy"), "assumed_liquid"),
    (("economics", "adjusted_contracts"), "price_any"), (("uncertainty", "seed"), True),
    (("uncertainty", "resamples"), 0), (("uncertainty", "block_days"), -1),
    (("uncertainty", "confidence"), 1), (("uncertainty", "block_origin"), "20230101"),
    (("risk", "minimum_uplift_usd"), -1), (("risk", "completeness_floor"), 1.1),
    (("limits", "job_wall_seconds"), 0), (("limits", "parent", "memory_bytes"), True),
    (("limits", "parent", "cpu_seconds"), 0), (("limits", "parent", "wall_seconds"), 1.5)])
def test_nested_economic_uncertainty_risk_and_resource_contracts(path, value):
    item = request()
    destination = item
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    _error(lambda: validate_request(item))


def test_optional_reference_metadata_stays_detached_and_does_not_open_artifacts(monkeypatch):
    value = request()
    value["benchmarks"] = [{"outer_year": "2023", "ref": deepcopy(value["bridges"][0]["ref"])}]
    before = deepcopy(value)
    def forbidden(*args, **kwargs):
        pytest.fail("policy validation must not open artifacts or invoke a runner")
    monkeypatch.setattr("builtins.open", forbidden)
    result = validate_request(value)
    assert result == before and value == before
    result["benchmarks"][0]["ref"]["byte_size"] = 999
    assert value == before
