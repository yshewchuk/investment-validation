import pytest

from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_request_versions import compile_request
from experiments.size_search_binding import resolve_extra_refits
from tests.v2.research.test_magnitude_search_request_v13 import _request

_RULES = ["overall", "rmse", "mae"]
_A = "a" * 64
_B = "b" * 64


def _with_rules(version=14):
    request = _request(version)
    request["selection"]["winner_rules"] = list(_RULES)
    request["max_fits"] += 2 * 3 * len(request["outer_years"])
    return request


def _with_extras(extras, version=14):
    request = _with_rules(version)
    request["selection"]["extra_refits"] = extras
    request["max_fits"] += len(extras) * len(request["outer_years"]) * 3
    return request


def test_valid_sorted_unique_extras_compile_and_add_fits():
    extras = [_A, _B]
    base = compile_request(_with_rules())
    compiled = compile_request(_with_extras(extras))
    assert compiled["request"]["selection"]["extra_refits"] == extras
    outers = len(compiled["request"]["outer_years"])
    assert compiled["counts"]["search_fits"] == (
        base["counts"]["search_fits"] + 3 * outers * len(extras))


@pytest.mark.parametrize("extras", [
    [_B, _A],
    [_A, _A],
    ["g" * 64],
    ["a" * 63],
    [],
    _A,
    [1],
])
def test_invalid_extra_refits_are_refused(extras):
    request = _with_rules()
    request["selection"]["extra_refits"] = extras
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_extra_refits_without_winner_rules_are_refused():
    request = _request(14)
    request["selection"]["extra_refits"] = [_A]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_v13_refuses_extra_refits():
    request = _request(13)
    request["selection"]["extra_refits"] = [_A]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_extras_refuse_max_fits_at_the_no_extras_value():
    request = _with_extras([_A])
    request["max_fits"] = _with_rules()["max_fits"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_extras_do_not_change_candidate_ids_or_specs():
    with_extras = _with_extras([_A, _B])
    plain = _with_rules()
    # max_fits feeds the candidate-spec hash, so the dry compile must carry the final max_fits.
    plain["max_fits"] = with_extras["max_fits"]
    plain = compile_request(plain)
    extended = compile_request(with_extras)
    assert ([s["candidate_spec_id"] for s in plain["candidate_specs"]]
            == [s["candidate_spec_id"] for s in extended["candidate_specs"]])
    assert (sorted(plain["candidate_specs"], key=lambda s: s["candidate_spec_id"])
            == sorted(extended["candidate_specs"], key=lambda s: s["candidate_spec_id"]))
    assert plain["request_id"] != extended["request_id"]


def test_resolve_extra_refits_returns_none_without_key():
    assert resolve_extra_refits({}, [_A]) is None


def test_resolve_extra_refits_returns_list_when_all_known():
    assert resolve_extra_refits({"extra_refits": [_A]}, [_A, _B]) == [_A]


def test_resolve_extra_refits_refuses_unknown_ids():
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        resolve_extra_refits({"extra_refits": [_A]}, [_B])
