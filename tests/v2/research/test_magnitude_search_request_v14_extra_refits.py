import json
import types

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import size_search_binding as binding
from experiments.magnitude_search_request_versions import compile_request
from experiments.size_search_binding import resolve_extra_refits
from tests.v2.research.test_magnitude_search_request_v13 import _request

_RULES = ["overall", "rmse", "mae"]
_A = "sha256:" + "a" * 64
_B = "sha256:" + "b" * 64
_PLAN_MAX_FITS = 10 ** 6


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
    ["sha256:" + "g" * 64],
    [_A, "sha256:" + "g" * 64],
    ["sha256:" + "a" * 63],
    ["a" * 64],
    [],
    _A,
    [1],
])
def test_invalid_extra_refits_are_refused(extras):
    request = _with_rules()
    request["max_fits"] = _PLAN_MAX_FITS
    request["selection"]["extra_refits"] = extras
    with pytest.raises(OpsError, match="extra_refits must be a sorted unique list of sha256 candidate ids") as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_extra_refits_without_winner_rules_are_refused():
    request = _request(14)
    request["selection"]["extra_refits"] = [_A]
    with pytest.raises(OpsError, match="extra_refits requires winner_rules") as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v13_refuses_extra_refits():
    request = _request(13)
    request["selection"]["extra_refits"] = [_A]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC") as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_extras_refuse_max_fits_at_the_no_extras_value():
    request = _with_extras([_A])
    request["max_fits"] = _with_rules()["max_fits"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC") as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


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
    spec_ids = [s["candidate_spec_id"] for s in plain["candidate_specs"]]
    assert spec_ids
    assert all(sid.startswith("sha256:") for sid in spec_ids)
    assert len(set(spec_ids)) == len(spec_ids)
    assert len(spec_ids) > 1


def test_max_fits_is_request_content_and_changes_candidate_ids():
    base = compile_request(_with_rules())
    bumped = _with_rules()
    bumped["max_fits"] += 1000
    other = compile_request(bumped)
    base_ids = [s["candidate_spec_id"] for s in base["candidate_specs"]]
    other_ids = [s["candidate_spec_id"] for s in other["candidate_specs"]]
    assert base_ids != other_ids


def test_resolve_extra_refits_returns_none_without_key():
    assert resolve_extra_refits({}, [_A]) is None


def test_resolve_extra_refits_returns_list_when_all_known():
    assert resolve_extra_refits({"extra_refits": [_A]}, [_A, _B]) == [_A]


def test_resolve_extra_refits_refuses_unknown_ids():
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC.*unknown candidate ids") as caught:
        resolve_extra_refits({"extra_refits": [_A]}, [_B])
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_content_hash_id_compiles_as_extra():
    cid = content_hash({"x": 1})
    compiled = compile_request(_with_extras([cid]))
    assert compiled["request"]["selection"]["extra_refits"] == [cid]


def test_resolve_extra_refits_accepts_content_hash_ids():
    x, y = content_hash({"x": 1}), content_hash({"x": 2})
    assert resolve_extra_refits({"extra_refits": [x]}, [x, y]) == [x]


def _prepare(request, monkeypatch):
    compiled = compile_request(request)
    registration = types.SimpleNamespace(
        document={"schema_version": "native_experiment_registration.v1.7",
                  "size_request": compiled["request"],
                  "size_panel_request": {"horizons": [5, 3, 1]},
                  "size_panel": {"p": 1},
                  "size_comparison": {},
                  "source_closure": {},
                  "environment": {}},
        binding_json=json.dumps({"size_panel": {"p": 1}}),
        run_id="r",
        variant_id="v",
    )
    monkeypatch.setattr(binding, "verify_native_registration", lambda *a, **k: None)
    monkeypatch.setattr(binding, "verify_installation", lambda *a, **k: None)
    monkeypatch.setattr(binding, "validate_panel_request", lambda *a, **k: None)
    monkeypatch.setattr(binding, "_folds", lambda *a, **k: [])
    monkeypatch.setattr(binding, "_shared_inner", lambda *a, **k: None)
    monkeypatch.setattr(binding, "size_candidate_recipe", lambda *a, **k: {"stub": True})
    return binding.prepare_plan(None, None, registration, request)


def test_plan_without_extras_has_no_extra_refits_key(monkeypatch):
    request = _with_rules()
    request["max_fits"] = _PLAN_MAX_FITS
    plan = _prepare(request, monkeypatch)
    assert "extra_refits" not in plan["selection"]
    ids = [c["candidate_id"] for c in plan["candidates"]]
    assert ids
    assert all(i.startswith("sha256:") for i in ids)


def test_plan_with_extras_records_them_and_keeps_candidate_ids(monkeypatch):
    base_request = _with_rules()
    base_request["max_fits"] = _PLAN_MAX_FITS
    base_plan = _prepare(base_request, monkeypatch)
    ids = [c["candidate_id"] for c in base_plan["candidates"]]
    extras = sorted(ids)[:2]

    request = _with_extras(extras)
    request["max_fits"] = _PLAN_MAX_FITS
    plan = _prepare(request, monkeypatch)
    assert plan["selection"]["extra_refits"] == extras
    assert [c["candidate_id"] for c in plan["candidates"]] == ids
    assert plan["plan_id"] != base_plan["plan_id"]


def test_plan_refuses_unknown_extra_refit_id(monkeypatch):
    request = _with_extras(["sha256:" + "f" * 64])
    request["max_fits"] = _PLAN_MAX_FITS
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC.*unknown candidate ids") as caught:
        _prepare(request, monkeypatch)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
