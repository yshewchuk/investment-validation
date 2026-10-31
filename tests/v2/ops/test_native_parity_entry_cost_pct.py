from __future__ import annotations

import pytest

from engine.v2.ops.native_parity_report import (
    _dimension_fields,
    _dimension_view,
    _row_mismatches,
)
from engine.v2.parity.tolerance import SCORE_RECORD_V1


@pytest.mark.parametrize(("cost", "spot", "pct"), [
    (3.0, 250.0, 1.2), (5.0, 80.0, 6.25), (-2.0, 40.0, -5.0), (0.0, 100.0, 0.0)])
def test_legacy_row_derives_entry_cost_pct(cost, spot, pct):
    # Different spots and results: a constant, un-normalised or cost*100 value fails.
    row = {"entry_cost": cost, "spot": spot}
    assert _dimension_view(row, "financial_diagnostics", legacy=True)["entry_cost_pct"] == pytest.approx(pct)


def test_row_carrying_entry_cost_pct_keeps_its_value():
    row = {"entry_cost_pct": 7.5, "entry_cost": 1.0, "spot": 10.0}
    assert _dimension_view(row, "financial_diagnostics", legacy=True)["entry_cost_pct"] == 7.5


@pytest.mark.parametrize("row", [
    {"entry_cost": 2.0, "spot": 0.0},
    {"entry_cost": 2.0, "spot": None},
    {"entry_cost": None, "spot": 100.0},
    {},
])
def test_entry_cost_pct_is_none_when_undefined(row):
    assert _dimension_view(row, "financial_diagnostics", legacy=True)["entry_cost_pct"] is None


def test_explicit_none_entry_cost_pct_is_kept():
    row = {"entry_cost_pct": None, "entry_cost": 2.0, "spot": 100.0}
    assert _dimension_view(row, "financial_diagnostics", legacy=True)["entry_cost_pct"] is None


def test_non_finite_entry_cost_gives_none():
    nan_row = _dimension_view({"entry_cost": float("nan"), "spot": 100.0},
                              "financial_diagnostics", legacy=True)
    assert nan_row["entry_cost_pct"] is None
    inf_row = _dimension_view({"entry_cost": float("inf"), "spot": 100.0},
                              "financial_diagnostics", legacy=True)
    assert inf_row["entry_cost_pct"] is None


def test_other_dimension_unchanged_and_no_entry_cost_pct_key():
    row = {"entry_cost": 2.0, "spot": 100.0}
    view = _dimension_view(row, "forecasts")
    assert set(view) == set(_dimension_fields("forecasts"))
    assert "entry_cost_pct" not in view


def _entry_cost_pct_findings(legacy_row, native_row):
    mismatches = _row_mismatches(
        "k", legacy_row, native_row, ("financial_diagnostics",), SCORE_RECORD_V1)
    return [m for m in mismatches if "entry_cost_pct" in m["finding_fields"]]


_CASES = [(3.0, 250.0, 1.2), (5.0, 80.0, 6.25)]


@pytest.mark.parametrize(("cost", "spot", "pct"), _CASES)
def test_row_mismatches_agree_on_normalised_entry_cost_pct_end_to_end(cost, spot, pct):
    legacy_row = {"entry_cost": cost, "spot": spot}
    assert not _entry_cost_pct_findings(legacy_row, {"entry_cost_pct": pct})


@pytest.mark.parametrize(("cost", "spot", "pct"), _CASES)
def test_row_mismatches_flag_a_wrong_native_entry_cost_pct(cost, spot, pct):
    legacy_row = {"entry_cost": cost, "spot": spot}
    assert _entry_cost_pct_findings(legacy_row, {"entry_cost_pct": cost})
    assert _entry_cost_pct_findings(legacy_row, {"entry_cost_pct": pct * 100})
    # The OTHER case's value: a constant-output implementation fails one of the two cases.
    other = next(p for _, _, p in _CASES if p != pct)
    assert _entry_cost_pct_findings(legacy_row, {"entry_cost_pct": other})


def test_native_view_never_derives_entry_cost_pct():
    row = {"entry_cost": 3.0, "spot": 250.0}
    assert _dimension_view(row, "financial_diagnostics")["entry_cost_pct"] is None
    view = _dimension_view(row, "financial_diagnostics", legacy=False)
    for field in ("entry_cost_pct", "model_vs_market", "fair_premium_pct",
                  "premium_vs_fair", "cost_over_width"):
        assert view[field] is None


def test_row_mismatches_flags_a_native_row_missing_its_own_entry_cost_pct():
    # The native row carries matching entry_cost/spot but no entry_cost_pct output:
    # the comparator must see the missing native value, not derive it.
    legacy_row = {"entry_cost": 3.0, "spot": 250.0}
    native_row = {"entry_cost": 3.0, "spot": 250.0}
    assert _entry_cost_pct_findings(legacy_row, native_row)


@pytest.mark.parametrize("row", [
    {"entry_cost": 3.0, "spot": "0"},
    {"entry_cost": 3.0, "spot": "0.0"},
    {"entry_cost": 3.0, "spot": "bad"},
    {"entry_cost": "bad", "spot": 250.0},
    {"entry_cost": "inf", "spot": 250.0},
    {"entry_cost": 3.0, "spot": "inf"},
    {"entry_cost": 3.0, "spot": float("inf")},
    {"entry_cost": 3.0, "spot": float("-inf")},
    {"entry_cost": 3.0, "spot": float("nan")},
    {"entry_cost": 3.0, "spot": object()},
])
def test_malformed_or_zero_operands_give_none_not_an_exception(row):
    assert _dimension_view(row, "financial_diagnostics", legacy=True)["entry_cost_pct"] is None


def test_numeric_strings_are_converted_before_dividing():
    row = {"entry_cost": "3.0", "spot": "250"}
    assert _dimension_view(row, "financial_diagnostics", legacy=True)["entry_cost_pct"] == pytest.approx(1.2)


def test_a_malformed_legacy_row_is_a_mismatch_not_an_abort():
    legacy_row = {"entry_cost": "bad", "spot": 250.0}
    assert _entry_cost_pct_findings(legacy_row, {"entry_cost_pct": 1.2})


def test_unavailable_on_both_sides_agrees_but_one_sided_is_a_mismatch():
    legacy_row = {"entry_cost": "bad", "spot": 250.0}
    assert not _entry_cost_pct_findings(legacy_row, {"entry_cost_pct": None})
    assert _entry_cost_pct_findings(legacy_row, {"entry_cost_pct": 1.2})


@pytest.mark.parametrize(("cost", "spot"), [
    (10**400, 10**400), (10**400, 250.0), (3.0, 10**400)])
def test_huge_integer_operand_gives_none(cost, spot):
    assert _dimension_view({"entry_cost": cost, "spot": spot},
                           "financial_diagnostics", legacy=True)["entry_cost_pct"] is None


def test_legacy_financial_view_derives_all_five_diagnostics():
    row = {
        "spot": 100.0,
        "entry_cost": 5.0,
        "driver_name": "abs_move",
        "driver_prediction": 0.5,
        "implied_move": 2.0,
        "structure_width": 10.0,
        "payoff": {"intercept": 0.1, "slope": 0.2},
    }

    view = _dimension_view(row, "financial_diagnostics", legacy=True)

    assert view == pytest.approx({
        "entry_cost_pct": 5.0,
        "model_vs_market": 0.3875968992248062,
        "fair_premium_pct": 20.0,
        "premium_vs_fair": 0.25,
        "cost_over_width": 0.5,
    })


_RUNUP_NAMES = (
    "intercept", "implied_move", "abs_moneyness",
    "moneyness_sq_div10", "signed_moneyness",
    "implied_x_abs_moneyness_div10",
)


def test_legacy_runup_fair_premium_has_independent_expected_value():
    coefficients = dict(zip(_RUNUP_NAMES, (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)))
    row = {
        "driver_prediction": 0.5,
        "runup_move_prediction": 10.0,
        "spot": 100.0,
        "strike": 100.0,
        "payoff": {"kind": "runup_payoff_surface", "coefficients": coefficients},
    }

    assert _dimension_view(row, "financial_diagnostics", legacy=True)[
        "fair_premium_pct"
    ] == pytest.approx(75.0)


@pytest.mark.parametrize("row", [
    {"driver_prediction": 0.5, "payoff": {"intercept": "bad", "slope": 0.2}},
    {
        "driver_prediction": 0.5,
        "runup_move_prediction": 1.0,
        "spot": 100.0,
        "strike": 90.0,
        "payoff": {
            "kind": "runup_payoff_surface",
            "coefficients": {name: ("bad" if name == "intercept" else 0.0)
                             for name in _RUNUP_NAMES},
        },
    },
    {
        "driver_prediction": 0.5,
        "runup_move_prediction": 1.0,
        "spot": -100.0,
        "strike": 90.0,
        "payoff": {
            "kind": "runup_payoff_surface",
            "coefficients": {name: 0.0 for name in _RUNUP_NAMES},
        },
    },
    {
        "driver_prediction": 0.5,
        "runup_move_prediction": 1.0,
        "spot": -100.0,
        "strike": -90.0,
        "payoff": {
            "kind": "runup_payoff_surface",
            "coefficients": {name: 0.0 for name in _RUNUP_NAMES},
        },
    },
    {
        "driver_prediction": 0.5,
        "runup_move_prediction": 1.0,
        "spot": 100.0,
        "strike": 90.0,
        "payoff": {"kind": "runup_payoff_surface", "coefficients": [1.0]},
    },
    {
        "driver_prediction": 0.5,
        "runup_move_prediction": 100000.0,
        "spot": 100.0,
        "strike": 90.0,
        "payoff": {
            "kind": "runup_payoff_surface",
            "coefficients": {
                **{name: 0.0 for name in _RUNUP_NAMES},
                "intercept": 1e308,
            },
        },
    },
])
def test_invalid_fair_premium_inputs_are_unavailable_not_report_errors(row):
    view = _dimension_view(row, "financial_diagnostics", legacy=True)

    assert view["fair_premium_pct"] is None


def test_negative_runup_oracle_is_clamped_to_zero():
    coefficients = {name: 0.0 for name in _RUNUP_NAMES}
    coefficients["intercept"] = -0.2
    row = {
        "driver_prediction": 0.5,
        "runup_move_prediction": 1.0,
        "spot": 100.0,
        "strike": 90.0,
        "payoff": {"kind": "runup_payoff_surface", "coefficients": coefficients},
    }

    assert _dimension_view(row, "financial_diagnostics", legacy=True)[
        "fair_premium_pct"
    ] == pytest.approx(0.0)


def test_legacy_row_explicit_fair_premium_pct_is_preserved():
    coefficients = {name: 0.0 for name in _RUNUP_NAMES}
    coefficients["intercept"] = 0.2
    row = {
        "fair_premium_pct": 99.0,
        "driver_prediction": 0.5,
        "runup_move_prediction": 1.0,
        "spot": 100.0,
        "strike": 90.0,
        "payoff": {"kind": "runup_payoff_surface", "coefficients": coefficients},
    }

    assert _dimension_view(row, "financial_diagnostics", legacy=True)[
        "fair_premium_pct"
    ] == pytest.approx(99.0)


def test_malformed_implied_move_and_structure_width_give_none_diagnostics():
    row = {
        "spot": 100.0,
        "entry_cost": 5.0,
        "driver_name": "abs_move",
        "driver_prediction": 0.5,
        "implied_move": "bad",
        "structure_width": "bad",
        "payoff": {"intercept": 0.1, "slope": 0.2},
    }

    view = _dimension_view(row, "financial_diagnostics", legacy=True)

    assert view["model_vs_market"] is None
    assert view["cost_over_width"] is None