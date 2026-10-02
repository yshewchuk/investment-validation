"""#171: ``_validate_finality`` must refuse non-finite / non-numeric shares.

Every ``<`` / ``>`` comparison against NaN is false, so a NaN (or "nan")
``daily_share`` / ``chain_share`` used to pass the floor check silently.
"""
from __future__ import annotations

import pytest

from engine.v2.ops.decision_validation import _validate_finality, validate
from engine.v2.ops.errors import OpsError
from tests.test_v2_ops_decision_validation_order import SESSION, _bound

PLAN = {"session": SESSION}


def _findings(**overrides):
    finality = {"date": SESSION, "is_final": True, "market_wide": True,
                "daily_share": 1.0, "chain_share": 1.0, "covered": 1}
    finality.update(overrides)
    findings = []
    _validate_finality(finality, PLAN, findings)
    return findings


@pytest.mark.parametrize("field", ["daily_share", "chain_share"])
@pytest.mark.parametrize("bad", [float("nan"), "nan", "NaN", float("inf"), float("-inf"),
                                 "inf", None, True, False, "0.9", [], {}])
def test_non_finite_or_non_numeric_share_is_invalid(field, bad):
    assert _findings(**{field: bad}) == [{"field": "finality." + field, "reason": "invalid"}]


@pytest.mark.parametrize("field", ["daily_share", "chain_share"])
@pytest.mark.parametrize("bad", [0.79, 0.0, -0.1, 1.0001, 2])
def test_out_of_range_share_is_outside_floor(field, bad):
    assert _findings(**{field: bad}) == [
        {"field": "finality." + field, "reason": "outside_finality_floor"}]


@pytest.mark.parametrize("good", [0.80, 0.9, 1.0, 1])
def test_valid_boundaries_pass(good):
    assert _findings(daily_share=good, chain_share=good) == []


def test_infinite_covered_does_not_crash():
    assert {"field": "finality", "reason": "insufficient_coverage"} in _findings(
        covered=float("inf"))


@pytest.mark.parametrize("bad", [float("nan"), "nan", float("inf")])
def test_full_route_refuses_bad_share_without_success_receipt(bad):
    score_doc, finality, plan, evidence, bindings, candidates = _bound()
    bad_finality = {**finality, "daily_share": bad}
    with pytest.raises(OpsError) as err:
        validate(candidates, score=score_doc, finality=bad_finality, plan=plan,
                 evidence=evidence, bindings=bindings)
    assert err.value.code == "VALIDATION_FAILED"
    findings = err.value.problem.details["findings"]
    assert {"field": "finality.daily_share", "reason": "invalid"} in findings
