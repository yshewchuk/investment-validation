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
    assert _dimension_view(row, "financial_diagnostics", legacy=False)["entry_cost_pct"] is None


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