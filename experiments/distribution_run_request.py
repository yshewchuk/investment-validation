"""Closed, explicit research economics and resource request; no run defaults."""
from copy import deepcopy
from datetime import date
from math import isfinite

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import from_document
from engine.v2.ops.errors import fail
from experiments.distribution_book import validate_book_config
from experiments.magnitude_search_execution import validate_execution_limits

PRIMARY_ARMS = ("histogram", "same_mean", "matched_point")


def require(condition):
    """Refuse an invalid explicit D request or caller-supplied economic input."""
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution run contract")


def number(value, *, positive=False):
    require(type(value) in (int, float) and isfinite(value) and (value > 0 if positive else value >= 0))
    return value


def validate_request(request):
    """Detach the sealed policy; validation grants no source or execution authority."""
    keys = {"schema_version", "quote_snapshot_id", "alphas", "bridges", "benchmarks", "required_arms",
        "strict_benchmarks", "economics", "book", "minimum_net_ev_usd", "report_end", "uncertainty",
        "risk", "limits", "max_opportunities", "decision_horizon", "availability_policy", "selector_policy", "primary_alpha"}
    try:
        require(type(request) is dict and set(request) == keys
                and request["schema_version"] == "distribution_run_request.v1"
                and request["decision_horizon"] == 1 and type(request["decision_horizon"]) is int
                and request["selector_policy"] == "fixed_preregistered_net_ev.v1"
                and request["availability_policy"] == "reconstructed_snapshot_quotes"
                and type(request["quote_snapshot_id"]) is str and bool(request["quote_snapshot_id"]))
        arms = request["required_arms"]
        require(type(arms) is list and len(arms) == len(set(arms)) and set(PRIMARY_ARMS) <= set(arms)
                and set(arms) <= {*PRIMARY_ARMS, "incumbent", "gate"} and type(request["strict_benchmarks"]) is bool)
        alphas = request["alphas"]
        require(type(alphas) is list and alphas == sorted(set(alphas)) and {0, 0.25, 0.5} <= set(alphas))
        require(all(number(alpha) <= 1 for alpha in alphas))
        require(number(request["primary_alpha"]) in alphas)
        economic = request["economics"]
        require(set(economic) == {"currency", "multiplier", "commission", "exchange_fee", "minimum_side_fee",
            "slippage_per_share_per_leg", "max_relative_spread", "capacity_policy", "adjusted_contracts"}
            and economic["currency"] == "USD" and economic["multiplier"] == 100
            and type(economic["multiplier"]) is int and economic["adjusted_contracts"] == "unsupported_native_standard"
            and economic["capacity_policy"] == "not_established_no_depth")
        for key in ("commission", "exchange_fee", "minimum_side_fee", "slippage_per_share_per_leg", "max_relative_spread"):
            number(economic[key])
        validate_book_config(request["book"])
        number(request["minimum_net_ev_usd"], positive=True)
        require(date.fromisoformat(request["report_end"]).isoformat() == request["report_end"])
        uncertainty = request["uncertainty"]
        require(set(uncertainty) == {"seed", "resamples", "block_days", "block_origin", "confidence"})
        require(all(type(uncertainty[k]) is int and uncertainty[k] > 0 for k in ("resamples", "block_days"))
                and type(uncertainty["seed"]) is int and uncertainty["seed"] >= 0
                and 0 < number(uncertainty["confidence"]) < 1)
        require(date.fromisoformat(uncertainty["block_origin"]).isoformat() == uncertainty["block_origin"])
        require(set(request["risk"]) == {"minimum_uplift_usd", "maximum_loss_usd", "maximum_settled_drawdown_usd", "completeness_floor"})
        for value in request["risk"].values():
            number(value)
        require(request["risk"]["completeness_floor"] <= 1)
        require(set(request["limits"]) == {"parent", "job_wall_seconds"})
        validate_execution_limits(request["limits"]["parent"])
        require(type(request["limits"]["job_wall_seconds"]) is int and request["limits"]["job_wall_seconds"] > 0
                and type(request["max_opportunities"]) is int and request["max_opportunities"] > 0)
        for entries, fields in ((request["bridges"], {"outer_year", "alpha", "ref"}),
                                (request["benchmarks"], {"outer_year", "ref"})):
            require(type(entries) is list)
            for entry in entries:
                require(set(entry) == fields and type(entry["outer_year"]) is str)
                if "alpha" in fields:
                    require(number(entry["alpha"]) in alphas)
                ref = from_document(ArtifactRef, entry["ref"])
                require(0 < ref.byte_size <= 16 * 1024 * 1024)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution run request") from None
    return deepcopy(request)
