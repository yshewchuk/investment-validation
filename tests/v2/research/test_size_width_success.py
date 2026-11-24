"""EXP-189 W6a: success table, breakeven alpha, recommendation and T7 rendering on synthetic evaluate-shaped fixtures."""
import pytest

from experiments import size_width_success as sws
from experiments.size_pnl_shortvol import NAMES

PERIODS = sws.PERIODS


def _ev(ret, lost=0):
    """An evaluate-shaped result with account ``ret``; the first ``lost`` structures sit 2.0 below ``ret``."""
    structures = {n: {"total_return": ret - (2.0 if i < lost else 0.0)} for i, n in enumerate(NAMES)}
    return {"account": {"total_return": ret, "trades": 100}, "structures": structures}


def _arm(ret=101.0, lost=0):
    """An arm over every period at account ``ret``."""
    return {p: _ev(ret, lost) for p in PERIODS}


def _be(x):
    """Breakeven alpha ``x`` in every period."""
    return {p: x for p in PERIODS}


PROD = {p: _ev(100.0) for p in PERIODS}
BE = {"production": _be(0.5), "A": _be(0.5)}


def test_breakeven_crossing():
    assert sws.breakeven_alpha(-100.0, 100.0, 5) == pytest.approx(0.5)


def test_breakeven_always_positive():
    assert sws.breakeven_alpha(10.0, 20.0, 5) == 0.0


def test_breakeven_always_negative():
    assert sws.breakeven_alpha(-10.0, -20.0, 5) == 1.0


def test_breakeven_not_available():
    assert sws.breakeven_alpha(-100.0, 100.0, 0) is None
    assert sws.breakeven_alpha(float("nan"), 100.0, 5) is None
    assert sws.breakeven_alpha(0.0, 0.0, 5) is None


def test_pass_case():
    row = sws.success_table(PROD, {"A": _arm()}, BE)["A"]
    assert row["passed"] is True
    assert row["fails"] == []
    assert row["beats"] == {p: True for p in PERIODS}
    assert row["lost_structures"] == 0
    assert row["breakeven"] == {"x": 0.5, "production": 0.5, "margin_ok": True}


def test_tie_is_not_a_beat():
    arm = _arm()
    arm["2024"] = _ev(100.0)
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["beats"]["2024"] is False
    assert row["passed"] is False
    assert row["fails"] == ["not_beat_2024"]


def test_margin_boundary_passes_at_exactly_margin():
    be = {"production": _be(0.5), "A": {**_be(0.5), "2025": 0.5 + sws.MARGIN}}
    assert sws.success_table(PROD, {"A": _arm()}, be)["A"]["passed"] is True


def test_margin_just_over_fails():
    be = {"production": _be(0.5), "A": {**_be(0.5), "2025": 0.5 + sws.MARGIN + 0.001}}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["breakeven"]["margin_ok"] is False
    assert row["passed"] is False
    assert row["fails"] == ["breakeven_margin"]


def test_breakeven_none_fails():
    be = {"production": _be(0.5), "A": {**_be(0.5), "2025": None}}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["passed"] is False
    assert row["fails"] == ["breakeven_not_available"]


def test_production_breakeven_none_fails():
    be = {"production": {**_be(0.5), "2025": None}, "A": _be(0.5)}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["passed"] is False
    assert row["fails"] == ["breakeven_not_available"]


def test_guard_two_lost_ok():
    row = sws.success_table(PROD, {"A": _arm(lost=2)}, BE)["A"]
    assert row["lost_structures"] == 2
    assert row["guard_ok"] is True
    assert row["passed"] is True


def test_guard_three_lost_fails():
    row = sws.success_table(PROD, {"A": _arm(lost=3)}, BE)["A"]
    assert row["lost_structures"] == 3
    assert row["guard_ok"] is False
    assert row["passed"] is False
    assert row["fails"] == ["guard"]


def test_omitted_arm_is_absent():
    table = sws.success_table(PROD, {"A": _arm()}, BE)
    assert set(table) == {"A"}


def test_e_same_k_fails():
    e_summary = {n: {"same_k": n == NAMES[0]} for n in NAMES}
    be = {**BE, "E": _be(0.5)}
    table = sws.success_table(PROD, {"E": _arm(), "A": _arm()}, be, e_summary)
    assert table["E"]["same_k"] is True
    assert table["E"]["passed"] is False
    assert table["E"]["fails"] == ["same_k"]
    assert table["A"]["same_k"] is False
    assert table["A"]["passed"] is True


def test_e_passes_without_same_k():
    e_summary = {n: {"same_k": False} for n in NAMES}
    be = {**BE, "E": _be(0.5)}
    assert sws.success_table(PROD, {"E": _arm()}, be, e_summary)["E"]["passed"] is True


@pytest.mark.parametrize("e_summary", [None, {}], ids=["none", "empty"])
def test_e_summary_absent_fails(e_summary):
    be = {**BE, "E": _be(0.5)}
    row = sws.success_table(PROD, {"E": _arm()}, be, e_summary)["E"]
    assert row["passed"] is False
    assert row["same_k"] is False
    assert row["fails"] == ["e_summary_missing"]


def test_e_summary_partial_fails():
    e_summary = {n: {"same_k": False} for n in NAMES[:6]}
    assert len(e_summary) == 6 and len(NAMES) == 7
    be = {**BE, "E": _be(0.5)}
    row = sws.success_table(PROD, {"E": _arm()}, be, e_summary)["E"]
    assert row["passed"] is False
    assert row["same_k"] is False
    assert row["fails"] == ["e_summary_missing"]


def test_non_e_arm_ignores_e_summary():
    be = {**BE, "A": _be(0.5)}
    row = sws.success_table(PROD, {"A": _arm()}, be, None)["A"]
    assert row["passed"] is True
    assert row["fails"] == []


def test_nan_structure_counts_as_lost():
    arm = _arm()
    arm["2025"]["structures"][NAMES[0]]["total_return"] = float("nan")
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["lost_structures"] == 1
    assert row["guard_ok"] is True
    assert row["passed"] is True


def test_nan_structures_over_guard_fail():
    arm = _arm()
    for n in NAMES[:3]:
        arm["2025"]["structures"][n]["total_return"] = float("nan")
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["lost_structures"] == 3
    assert row["passed"] is False
    assert row["fails"] == ["guard"]


def test_recommend_prefers_a_over_c():
    assert sws.recommend({"C": {"passed": True}, "A": {"passed": True}}) == "A"


def test_recommend_b_when_only_b_and_e_pass():
    table = {"A": {"passed": False}, "B": {"passed": True}, "E": {"passed": True}}
    assert sws.recommend(table) == "B"


def test_recommend_none_when_nothing_passes():
    assert sws.recommend({"A": {"passed": False}}) is None
    assert sws.recommend({}) is None


def test_board_rule_no_change():
    assert sws.board_rule(None, {}, {}, {}) == {"recommendation": "no_change", "rows": []}


def test_board_rule_a_divisors_in_name_order():
    rule = sws.board_rule("A", {n: 1.5 for n in NAMES}, {NAMES[0]: 0.2}, {})
    assert rule["recommendation"] == "A"
    assert [r["structure"] for r in rule["rows"]] == list(NAMES)
    first = rule["rows"][0]
    assert first["divisor"] == "divisor / 1.5"
    assert first["k"] == 1.5
    assert first["flatness_se"] == 0.2
    assert first["declined_share"] is None


def test_board_rule_e_is_expectation_rule():
    rule = sws.board_rule("E", {}, {}, {})
    assert rule["recommendation"] == "E"
    assert rule["rows"] == []
    assert "exp_pnl_sim" in rule["rule"]


def test_render_t7_has_title_arms_and_recommendation():
    table = sws.success_table(PROD, {"A": _arm(), "B": _arm(lost=3)}, BE)
    text = sws.render_t7(table)
    assert "### T7 success table (2025 NOT blind for widths)" in text
    assert "| A |" in text
    assert "| B |" in text
    assert "0.500" in text
    assert "Recommendation: arm A." in text
