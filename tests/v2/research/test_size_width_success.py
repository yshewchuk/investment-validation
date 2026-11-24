"""EXP-189 W6a: success table, breakeven alpha, recommendation and T7 rendering on synthetic evaluate-shaped fixtures."""
import re

import pytest

from experiments import size_width_success as sws
from experiments.size_pnl_shortvol import NAMES
from experiments.size_width_sizing import GRID

PERIODS = sws.PERIODS
LADDER = sws.LADDER


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


def _sweep(means):
    """A breakeven sweep over LADDER with mean return ``means`` (one per ladder alpha, in order)."""
    return {a: {"mean": m} for a, m in zip(LADDER, means)}


PROD = {p: _ev(100.0) for p in PERIODS}
BE = {"production": _be(0.5), "A": _be(0.5)}


# --- breakeven alpha (sweep input) ---

def test_breakeven_crossing():
    assert sws.breakeven_alpha(_sweep([-1.0, -0.5, 0.0, 0.5, 1.0])) == pytest.approx(0.5)


def test_breakeven_asymmetric_crossing_hand_computed():
    # between alpha 0.25 (mean -1) and 0.5 (mean 3): 0.25 + (0 - (-1)) * (0.5 - 0.25) / (3 - (-1)) = 0.3125
    assert sws.breakeven_alpha(_sweep([-2.0, -1.0, 3.0, 4.0, 5.0])) == pytest.approx(0.3125)


def test_breakeven_always_positive():
    assert sws.breakeven_alpha(_sweep([1.0, 2.0, 3.0, 4.0, 5.0])) == 0.0


def test_breakeven_always_negative():
    assert sws.breakeven_alpha(_sweep([-1.0, -2.0, -3.0, -4.0, -5.0])) == 1.0


def test_breakeven_all_zero_is_none():
    assert sws.breakeven_alpha(_sweep([0.0] * 5)) is None


def test_breakeven_missing_ladder_alpha_is_none():
    sweep = _sweep([-1.0, -0.5, 0.0, 0.5, 1.0])
    del sweep[0.5]
    assert sws.breakeven_alpha(sweep) is None


def test_breakeven_string_mean_is_none():
    sweep = _sweep([-1.0, -0.5, 0.0, 0.5, 1.0])
    sweep[0.25]["mean"] = "-0.5"
    assert sws.breakeven_alpha(sweep) is None


def test_breakeven_bool_and_nan_mean_are_none():
    sweep = _sweep([-1.0, -0.5, 0.0, 0.5, 1.0])
    sweep[0.0]["mean"] = True
    assert sws.breakeven_alpha(sweep) is None
    sweep = _sweep([-1.0, -0.5, 0.0, 0.5, 1.0])
    sweep[1.0]["mean"] = float("nan")
    assert sws.breakeven_alpha(sweep) is None


def test_breakeven_not_a_dict_is_none():
    assert sws.breakeven_alpha(None) is None
    assert sws.breakeven_alpha([(0.0, -1.0)]) is None


def _str_sweep(means):
    """A breakeven sweep over LADDER keyed by the .2f strings that ``engine.evaluate.alpha_sweep`` returns."""
    return {f"{a:.2f}": {"mean": m} for a, m in zip(LADDER, means)}


def test_breakeven_string_keys_crossing_matches_float_keys():
    means = [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert sws.breakeven_alpha(_str_sweep(means)) == pytest.approx(0.5)
    assert sws.breakeven_alpha(_str_sweep(means)) == sws.breakeven_alpha(_sweep(means))


def test_breakeven_string_keys_asymmetric_crossing_matches_float_keys():
    means = [-2.0, -1.0, 3.0, 4.0, 5.0]
    assert sws.breakeven_alpha(_str_sweep(means)) == pytest.approx(0.3125)
    assert sws.breakeven_alpha(_str_sweep(means)) == sws.breakeven_alpha(_sweep(means))


def test_breakeven_string_keys_always_positive_and_negative():
    assert sws.breakeven_alpha(_str_sweep([1.0, 2.0, 3.0, 4.0, 5.0])) == 0.0
    assert sws.breakeven_alpha(_str_sweep([-1.0, -2.0, -3.0, -4.0, -5.0])) == 1.0


def test_breakeven_mixed_float_and_string_keys():
    means = [-1.0, -0.5, 0.0, 0.5, 1.0]
    sweep = {}
    for i, (a, m) in enumerate(zip(LADDER, means)):
        sweep[a if i % 2 == 0 else f"{a:.2f}"] = {"mean": m}
    assert set(map(type, sweep)) == {float, str}
    assert sws.breakeven_alpha(sweep) == pytest.approx(0.5)


def test_breakeven_string_keys_missing_one_is_none():
    sweep = _str_sweep([-1.0, -0.5, 0.0, 0.5, 1.0])
    del sweep["0.50"]
    assert sws.breakeven_alpha(sweep) is None


def test_breakeven_float_and_string_both_present_invalid_is_none():
    sweep = _sweep([-1.0, -0.5, 0.0, 0.5, 1.0])
    sweep["0.50"] = {"mean": "0.5"}
    assert sws.breakeven_alpha(sweep) is None
    sweep = _sweep([-1.0, -0.5, 0.0, 0.5, 1.0])
    sweep["0.25"] = {"mean": float("nan")}
    assert sws.breakeven_alpha(sweep) is None


# --- success rule ---

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
    # production breakeven 0.5: x 0.55 is exactly production + 0.05
    be = {"production": _be(0.5), "A": {**_be(0.5), "2025": 0.55}}
    assert sws.success_table(PROD, {"A": _arm()}, be)["A"]["passed"] is True


def test_margin_just_over_fails():
    # production breakeven 0.5: x 0.56 is over production + 0.05
    be = {"production": _be(0.5), "A": {**_be(0.5), "2025": 0.56}}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["breakeven"]["margin_ok"] is False
    assert row["passed"] is False
    assert row["fails"] == ["breakeven_margin"]


def test_breakeven_none_fails():
    be = {"production": _be(0.5), "A": {**_be(0.5), "2025": None}}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["passed"] is False
    assert row["fails"] == ["breakeven_not_available"]


@pytest.mark.parametrize("bad", ["0.5", True, float("nan"), None], ids=["str", "bool", "nan", "none"])
def test_arm_breakeven_not_finite_real_is_not_available(bad):
    be = {"production": _be(0.5), "A": {**_be(0.5), "2025": bad}}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["breakeven"]["x"] is None
    assert row["passed"] is False
    assert row["fails"] == ["breakeven_not_available"]


def test_string_breakeven_not_available():
    be = {"production": _be(0.5), "A": {**_be(0.5), "2025": "0.5"}}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["fails"] == ["breakeven_not_available"]
    assert row["passed"] is False


def test_non_dict_breakeven_entry_not_available():
    be = {"production": _be(0.5), "A": "0.5"}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["fails"] == ["breakeven_not_available"]
    assert row["passed"] is False


def test_production_breakeven_none_fails():
    be = {"production": {**_be(0.5), "2025": None}, "A": _be(0.5)}
    row = sws.success_table(PROD, {"A": _arm()}, be)["A"]
    assert row["passed"] is False
    assert row["fails"] == ["breakeven_not_available"]


def test_arm_structure_equal_to_production_is_not_lost():
    arm = _arm(ret=101.0)
    for n in NAMES:
        arm["2025"]["structures"][n]["total_return"] = 100.0  # production's 2025 structure return
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["lost_structures"] == 0
    assert row["guard_ok"] is True
    assert row["passed"] is True


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


def test_string_structure_return_counts_as_lost():
    arm = _arm()
    arm["2025"]["structures"][NAMES[0]]["total_return"] = "150.0"
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["lost_structures"] == 1
    assert row["passed"] is True


def test_production_structure_invalid_counts_as_lost():
    prod = {p: _ev(100.0) for p in PERIODS}
    prod["2025"]["structures"][NAMES[0]]["total_return"] = None
    row = sws.success_table(prod, {"A": _arm()}, BE)["A"]
    assert row["lost_structures"] == 1


def test_missing_2025_structures_counts_all_lost():
    arm = _arm()
    del arm["2025"]["structures"]
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["lost_structures"] == len(NAMES)
    assert row["fails"] == ["guard"]


# --- admission: missing or invalid periods and accounts ---

def test_arm_missing_account_key_fails_without_raising():
    arm = _arm()
    del arm["2025"]["account"]
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["beats"]["2025"] is False
    assert row["fails"] == ["missing_2025"]
    assert row["passed"] is False


def test_arm_missing_2025_fails_missing_and_guard():
    arm = _arm()
    del arm["2025"]
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["beats"] == {"2025": False, "2023H2": True, "2024": True}
    assert row["lost_structures"] == len(NAMES)
    assert row["fails"] == ["missing_2025", "guard"]
    assert row["passed"] is False


def test_arm_missing_period_fails_missing():
    arm = _arm()
    del arm["2024"]
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["beats"]["2024"] is False
    assert row["fails"] == ["missing_2024"]
    assert row["passed"] is False


def test_arm_string_account_return_fails_missing_period():
    arm = _arm()
    arm["2024"]["account"]["total_return"] = "101.0"
    row = sws.success_table(PROD, {"A": arm}, BE)["A"]
    assert row["beats"]["2024"] is False
    assert row["fails"] == ["missing_2024"]


def test_production_missing_period_fails_missing():
    prod = {p: _ev(100.0) for p in PERIODS}
    del prod["2023H2"]
    row = sws.success_table(prod, {"A": _arm()}, BE)["A"]
    assert row["beats"]["2023H2"] is False
    assert row["fails"] == ["missing_2023H2"]
    assert row["passed"] is False


@pytest.mark.parametrize("prod, arms", [([], {"A": _arm()}), (PROD, [("A", _arm())])],
                         ids=["production_not_dict", "arms_not_dict"])
def test_non_dict_inputs_raise_value_error(prod, arms):
    with pytest.raises(ValueError, match="production and arms must be dicts"):
        sws.success_table(prod, arms, BE)


def test_breakeven_x_and_production_shown_in_their_own_columns():
    be = {"production": _be(0.3), "A": _be(0.4)}
    table = sws.success_table(PROD, {"A": _arm()}, be)
    assert table["A"]["breakeven"]["x"] == 0.4
    assert table["A"]["breakeven"]["production"] == 0.3
    row_line = next(line for line in sws.render_t7(table).splitlines() if line.startswith("| A |"))
    cells = [c.strip() for c in row_line.strip("|").split("|")]
    assert cells[4] == "0.400"  # breakeven x
    assert cells[5] == "0.300"  # breakeven production


# --- invalid board_rule inputs ---

def test_board_rule_unknown_label_raises():
    with pytest.raises(ValueError, match=re.escape("unknown arm 'production'")):
        sws.board_rule("production", {n: 1.0 for n in NAMES}, {}, {})


def test_board_rule_missing_structure_factor_raises():
    factors = {n: 1.0 for n in NAMES}
    del factors[NAMES[0]]
    with pytest.raises(ValueError, match=re.escape(f"missing factor for structure {NAMES[0]!r}")):
        sws.board_rule("A", factors, {}, {})


def test_board_rule_none_factor_raises():
    factors = {n: 1.0 for n in NAMES}
    factors[NAMES[0]] = None
    with pytest.raises(ValueError, match=re.escape(f"factor for structure {NAMES[0]!r} must be a finite real number, got None")):
        sws.board_rule("A", factors, {}, {})


def test_board_rule_non_grid_k_raises():
    assert 1.6 not in GRID
    factors = {n: 1.0 for n in NAMES}
    factors[NAMES[0]] = 1.6
    with pytest.raises(ValueError, match=re.escape(f"factor for structure {NAMES[0]!r} is 1.6, not in size_width_sizing.GRID")):
        sws.board_rule("A", factors, {}, {})


def test_board_rule_bool_k_raises():
    factors = {n: 1.0 for n in NAMES}
    factors[NAMES[0]] = True
    with pytest.raises(ValueError, match=re.escape(f"factor for structure {NAMES[0]!r} must be a finite real number, got True")):
        sws.board_rule("A", factors, {}, {})


def test_board_rule_non_dict_flatness_raises():
    with pytest.raises(ValueError, match="flatness must be a dict"):
        sws.board_rule("A", {n: 1.0 for n in NAMES}, [0.2], {})


def test_board_rule_non_numeric_declined_raises():
    with pytest.raises(ValueError, match=re.escape(f"declined for structure {NAMES[0]!r} must be a finite real number or None, got 'x'")):
        sws.board_rule("A", {n: 1.0 for n in NAMES}, {}, {NAMES[0]: "x"})


# --- invalid render_rule inputs ---

def test_render_rule_non_dict_raises():
    with pytest.raises(ValueError, match="rule must be a dict"):
        sws.render_rule([("recommendation", "A")])


def test_render_rule_unknown_recommendation_raises():
    with pytest.raises(ValueError, match=re.escape("unknown recommendation 'production'")):
        sws.render_rule({"recommendation": "production", "rows": []})


def test_render_rule_a_without_rows_raises():
    with pytest.raises(ValueError, match="needs a list 'rows' of dicts"):
        sws.render_rule({"recommendation": "A"})


def test_render_rule_a_rows_missing_keys_raises():
    rule = {"recommendation": "A", "rows": [{"structure": NAMES[0], "k": 1.0}]}
    with pytest.raises(ValueError, match="needs a list 'rows' of dicts"):
        sws.render_rule(rule)


def test_render_rule_a_rows_not_list_raises():
    with pytest.raises(ValueError, match="needs a list 'rows' of dicts"):
        sws.render_rule({"recommendation": "A", "rows": "rows"})


# --- recommendation order ---

def test_recommend_prefers_a_over_c():
    assert sws.recommend({"C": {"passed": True}, "A": {"passed": True}}) == "A"


def test_recommend_d_before_c():
    assert sws.recommend({"C": {"passed": True}, "D": {"passed": True}}) == "D"


def test_recommend_c_before_b():
    assert sws.recommend({"B": {"passed": True}, "C": {"passed": True}}) == "C"


def test_recommend_e_last():
    assert sws.recommend({"E": {"passed": True}, "B": {"passed": True}}) == "B"
    assert sws.recommend({"E": {"passed": True}}) == "E"


def test_recommend_b_when_only_b_and_e_pass():
    table = {"A": {"passed": False}, "B": {"passed": True}, "E": {"passed": True}}
    assert sws.recommend(table) == "B"


def test_recommend_none_when_nothing_passes():
    assert sws.recommend({"A": {"passed": False}}) is None
    assert sws.recommend({}) is None


# --- board rule ---

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


# --- rendering ---

def test_render_t7_has_title_arms_and_recommendation():
    table = sws.success_table(PROD, {"A": _arm(), "B": _arm(lost=3)}, BE)
    text = sws.render_t7(table)
    assert "### T7 success table (2025 NOT blind for widths)" in text
    assert "| A |" in text
    assert "| B |" in text
    assert "0.500" in text
    assert "Recommendation: arm A." in text


def test_render_t7_exact_no_recommendation():
    table = sws.success_table(PROD, {"A": _arm(ret=100.0)}, BE)
    expected = "\n\n".join([
        "### T7 success table (2025 NOT blind for widths)",
        "\n".join([
            "| arm | beats 2025 | beats 2023H2 | beats 2024 | breakeven x | breakeven production | lost structures | passed | fails |",
            "|---|---|---|---|---|---|---|---|---|",
            "| A | False | False | False | 0.500 | 0.500 | 0 | False | not_beat_2025;not_beat_2023H2;not_beat_2024 |",
        ]),
        "Recommendation: no change recommended.",
    ])
    assert sws.render_t7(table) == expected


def test_render_t7_exact_recommended():
    table = sws.success_table(PROD, {"A": _arm()}, BE)
    expected = "\n\n".join([
        "### T7 success table (2025 NOT blind for widths)",
        "\n".join([
            "| arm | beats 2025 | beats 2023H2 | beats 2024 | breakeven x | breakeven production | lost structures | passed | fails |",
            "|---|---|---|---|---|---|---|---|---|",
            "| A | True | True | True | 0.500 | 0.500 | 0 | True |  |",
        ]),
        "Recommendation: arm A.",
    ])
    assert sws.render_t7(table) == expected


def test_render_rule_a_exact():
    rule = sws.board_rule("A", {n: 1.5 for n in NAMES}, {n: 0.2 for n in NAMES}, {n: 0.1 for n in NAMES})
    rows = "\n".join(f"| {n} | 1.500 | divisor / 1.5 | 0.200 | 0.100 |" for n in NAMES)
    expected = "\n\n".join([
        "### Proposed board width rule (arm A)",
        "\n".join([
            "| structure | k | divisor | flatness_se | declined_share |",
            "|---|---|---|---|---|",
            rows,
        ]),
    ])
    assert sws.render_rule(rule) == expected


def test_render_rule_e_exact():
    assert sws.render_rule(sws.board_rule("E", {}, {}, {})) == \
        "Recommendation: E, pick k maximising exp_pnl_sim per event."


def test_render_rule_no_change_exact():
    assert sws.render_rule(sws.board_rule(None, {}, {}, {})) == "No change recommended."
