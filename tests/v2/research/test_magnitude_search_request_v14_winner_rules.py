import pytest

from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_request_schema import LATEST
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v13 import _request

_RULES = ["overall", "rmse", "mae"]


def _with_rules(version, rules):
    request = _request(version)
    request["selection"]["winner_rules"] = rules
    request["max_fits"] += 2 * 3 * len(request["outer_years"])
    return request


def test_v14_without_winner_rules_matches_v13_counts_and_schedule():
    v14 = compile_request(_request(14))
    assert "winner_rules" not in v14["request"]["selection"]
    v13 = compile_request(_request(13))
    assert v14["counts"] == v13["counts"]
    assert v14["schedule"] == v13["schedule"]


def test_v14_accepts_the_three_rule_order():
    compiled = compile_request(_with_rules(14, list(_RULES)))
    assert compiled["request"]["selection"]["winner_rules"] == _RULES


@pytest.mark.parametrize("rules", [["overall"], ["mae", "rmse", "overall"], [], "overall"])
def test_v14_refuses_any_other_winner_rules(rules):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(_with_rules(14, rules))


def test_v13_refuses_winner_rules():
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(_with_rules(13, list(_RULES)))


def test_latest_bump_keeps_v13_requests_compiling():
    assert LATEST == 14
    assert compile_request(_request(13))["request"]["schema_version"] == (
        "magnitude_search_request.v13")
