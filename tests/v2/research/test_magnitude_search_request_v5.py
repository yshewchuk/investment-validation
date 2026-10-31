"""v5 fixed size_v1_4 family declaration and refusal-before-expansion contracts."""
import pytest

from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_request as compiler
from experiments import magnitude_search_request_versions as versions
from tests.v2.research.test_magnitude_search_request_v4 import v4_request


def v5_request():
    request = v4_request()
    request["schema_version"] = "magnitude_search_request.v5"
    request["space"]["size_v1_4"] = {}
    inner = sum(len(years) for years in request["inner_years"].values())
    outer = len(request["outer_years"])
    request["max_candidates"] = 102
    request["max_fits"] = 3 * (102 * inner + outer)
    return request


def test_v5_declares_fixed_size_v1_4_family_without_changing_v4():
    result = versions.compile_request(v5_request())
    assert result["request"]["space"]["size_v1_4"] == {}
    assert result["counts"]["candidates"] == 102
    size = [spec for spec in result["candidate_specs"] if spec["family"] == "size_v1_4"]
    assert len(size) == 3
    assert all(spec["parameters"] == {} for spec in size)
    ids = [spec["candidate_spec_id"] for spec in size]
    assert len(ids) == len(set(ids)) == 3
    assert versions.compile_request(v4_request())["counts"]["candidates"] == 99


def _fail_if_called(*args, **kwargs):
    raise AssertionError("specifications must not expand during refusal")


def test_v5_missing_size_v1_4_refuses_before_expansion(monkeypatch):
    request = v5_request()
    del request["space"]["size_v1_4"]
    monkeypatch.setattr(compiler, "_specifications", _fail_if_called)
    with pytest.raises(OpsError) as caught:
        versions.compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
