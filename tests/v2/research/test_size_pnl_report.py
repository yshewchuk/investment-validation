"""Tests for experiments/size_pnl_report.py (EXP-188 slice S8b).

Small synthetic fixtures only; no real data or store.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.size_pnl_contrast import contrast
from experiments.size_pnl_report import (
    _md_table,
    _per_event,
    build_report,
    t1_roster,
    t2_in_e,
    t2_population,
    t3_forecast,
    t4_shortvol,
    t6_straddle,
    t7_contrasts,
    t8_rungs,
    t9_pool_sensitivity,
    t10_verdicts,
)

QS = ["2024Q1", "2024Q2", "2025Q1", "2025Q2"]


def _summary(sign):
    lo, hi = (0.5, 1.5) if sign > 0 else (-1.5, -0.5)
    return {
        "n_events": 8,
        "mean": float(sign),
        "quarter_signs": {q: sign for q in QS},
        "year_signs": {2024: sign, 2025: sign},
        "quarter_interval": (lo, hi),
        "ticker_interval": (lo, hi),
        "quarter_counts": {"positive": 4, "negative": 0, "zero": 0},
    }


def test_md_table_escapes_formats_and_none():
    out = _md_table(["a", "b", "c"], [["x|y", 0.123456, None], [1, 2.0, np.float64(3.0)]])
    lines = out.split("\n")
    assert lines[0] == "| a | b | c |"
    assert lines[2] == "| x\\|y | 0.1235 | - |"
    assert lines[3] == "| 1 | 2 | 3 |"


def test_md_table_empty_rows():
    assert _md_table(["a"], []) == "(no rows)"


def test_t1_status_column():
    roster = [
        {"id": "m1", "model": "A", "type": "short", "roles": ["x", "y"], "excluded_reason": None},
        {"id": "m2", "model": "B", "type": "long", "roles": ["z"],
         "excluded_reason": "undetermined_under_model"},
    ]
    out = t1_roster(roster)
    assert out.startswith("### T1")
    assert "| m1 | A | short | x+y | included |" in out
    assert "| m2 | B | long | z | excluded: undetermined_under_model |" in out


def test_t2_totals():
    ev = pd.DataFrame({"quarter": ["2024Q1", "2024Q1", "2024Q1", "2024Q2"],
                       "in_e": [True, False, True, False]})
    out = t2_population(ev)
    assert "| 2024Q1 | 3 | 2 | 1 |" in out
    assert "| 2024Q2 | 1 | 0 | 1 |" in out
    assert "| total | 4 | 2 | 2 |" in out


def test_t2_all_in_and_all_excluded_quarters():
    ev = pd.DataFrame({"quarter": ["2024Q1"] * 3 + ["2024Q2"] * 2 + ["2024Q3"] * 2,
                       "in_e": [True, False, True, True, True, False, False]})
    out = t2_population(ev)
    assert "| 2024Q1 | 3 | 2 | 1 |" in out
    assert "| 2024Q2 | 2 | 2 | 0 |" in out
    assert "| 2024Q3 | 2 | 0 | 2 |" in out
    assert "| total | 7 | 4 | 3 |" in out


def test_t2_empty_object_frame():
    out = t2_population(pd.DataFrame([], columns=["quarter", "in_e"]))
    assert "(no rows)" in out
    assert "total" not in out


def test_t2_in_e_unique_counts_per_quarter_and_total():
    ev = pd.DataFrame({"quarter": ["2024Q2", "2024Q1", "2024Q1", "2024Q1"],
                       "event_id": ["E03", "E01", "E01", "E02"]})
    out = t2_in_e(ev, "NOTE_TEXT")
    assert out.startswith("### T2 Population (events in E)")
    assert "| quarter | in_E |" in out
    lines = out.split("\n")
    assert lines.index("| 2024Q1 | 2 |") < lines.index("| 2024Q2 | 1 |") < lines.index("| total | 3 |")
    assert "NOTE_TEXT" in out
    assert "candidates" not in out and "dropped" not in out


def test_t2_in_e_empty_frame_gives_no_rows():
    out = t2_in_e(pd.DataFrame([], columns=["quarter", "event_id"]), "NOTE_TEXT")
    assert out.startswith("### T2 Population (events in E)")
    assert "(no rows)" in out
    assert "total" not in out
    assert "NOTE_TEXT" in out


def test_t2_in_e_refuses_missing_column():
    with pytest.raises(ValueError, match="t2_in_e"):
        t2_in_e(pd.DataFrame({"quarter": ["2024Q1"]}), "NOTE_TEXT")


def test_t3_share_and_not_testable_ratio():
    frame = pd.DataFrame({
        "model_id": ["m1"] * 4 + ["m2"] * 2,
        "pred": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        "actual": [0.5, 2.0, 3.0, 0.1, 1.0, 5.0],
    })
    out = t3_forecast(frame, [("p1", "m1", "m2"), ("p2", "m1", "gone")])
    assert "| m1 | 4 | 1 | 0.5 |" in out
    assert "| m2 | 2 | 2 | 0.5 |" in out
    assert "| p1 | m1 | m2 | 2 |" in out
    assert "| p2 | m1 | gone | not_testable |" in out


def test_t3_strict_above_and_mean_not_median():
    # pred mean 2.5 vs median 2.0; rows 3-4 have actual == pred and must not count as above.
    frame = pd.DataFrame({"model_id": ["m1"] * 4,
                          "pred": [1.0, 2.0, 2.0, 5.0],
                          "actual": [2.0, 3.0, 2.0, 5.0]})
    out = t3_forecast(frame, [])
    assert "| m1 | 4 | 2.5 | 0.5 |" in out
    # Swapping actual and pred: pred becomes [2, 3, 2, 5] (mean 3), actual [1, 2, 2, 5]; none above.
    swapped = frame.rename(columns={"pred": "actual", "actual": "pred"})
    assert "| m1 | 4 | 3 | 0 |" in t3_forecast(swapped, [])


def _scored(arm):
    return pd.DataFrame({
        "model_id": ["m1", "m1", "m1"],
        "structure": ["s", "s", "s"],
        "arm": [arm] * 3,
        "taken": [True, False, True],
        "pnl_dollars": [10.0, np.nan, -4.0],
    })


def test_t4_per_event_counts_non_trade_as_zero():
    out = t4_shortvol(_scored("as_is"), {"m1": 4})
    assert out.startswith("### T4 Short-vol, as is")
    assert "| m1 | 2 | 6.00 | 1.5 | 0.5 |" in out
    assert "skip_note" not in out
    assert "T5" not in out  # Amendment 6: no re-tuned arm and no T5


def test_t4_empty_object_scored_gives_zero_rows():
    empty = pd.DataFrame([], columns=["model_id", "arm", "taken", "pnl_dollars"])
    out = t4_shortvol(empty, {"m1": 4})
    assert "| m1 | 0 | 0.00 | 0 | - |" in out


def test_t4_zero_pnl_is_not_a_hit():
    scored = pd.DataFrame({"model_id": ["m1"] * 4, "structure": ["s"] * 4, "arm": ["as_is"] * 4,
                           "taken": [True] * 4, "pnl_dollars": [0.0, 2.0, 0.0, -1.0]})
    out = t4_shortvol(scored, {"m1": 2})
    # total 1.0 over 2 candidate events; hits = 1 of 4 taken (the 0.0 rows are not hits)
    assert "| m1 | 4 | 1.00 | 0.5 | 0.25 |" in out


@pytest.mark.parametrize("bad", [-4, np.nan, 2.5])
def test_per_event_refuses_bad_counts(bad):
    with pytest.raises(ValueError, match="_per_event"):
        _per_event(1.0, "m1", {"m1": bad})


def test_per_event_zero_count_is_none():
    assert _per_event(5.0, "m1", {"m1": 0}) is None
    assert _per_event(6.0, "m1", {"m1": 4}) == 1.5


def test_t4_refusals():
    with pytest.raises(ValueError, match="candidate-event"):
        t4_shortvol(_scored("as_is"), {})


def test_t4_model_with_no_scored_rows_gets_row():
    scored = _scored("as_is")
    scored = scored[scored["model_id"] == "m1"]
    out = t4_shortvol(scored, {"m1": 4, "B": 4})
    assert "| B | 0 | 0.00 | 0 | - |" in out


def test_t6_undetermined_kept_separate():
    scored = pd.DataFrame({"model_id": ["m1", "m1"], "rule": ["R_L", "R_L"],
                           "pnl_dollars": [3.0, 2.0]})
    decisions = pd.DataFrame({"model_id": ["m1"] * 3, "rule": ["R_L"] * 3,
                              "status": ["taken", "not_taken", "undetermined"]})
    out = t6_straddle(scored, decisions, {"m1": 4})
    assert "| m1 | R_L | 1 | 1 | 1 | 1.25 |" in out


def _t6_decisions():
    return pd.DataFrame({
        "model_id": ["m1"] * 5 + ["m2"] * 2,
        "rule": ["R_L", "R_L", "R_L", "R_S", "R_S", "R_L", "R_S"],
        "status": ["taken", "not_taken", "undetermined", "taken", "taken", "taken", "undetermined"],
    })


T6_N_EVENTS = {"m1": 4, "m2": 2, "m3": 5}  # m3 has no decisions


def test_t6_filters_by_model_and_rule_and_zero_rows():
    scored = pd.DataFrame({
        "model_id": ["m1", "m1", "m1", "m1", "m1", "m2", "m2"],
        "rule": ["R_L", "R_L", "R_S", "R_S", "R_X", "R_L", "R_S"],  # R_X: scored only, no decisions
        "pnl_dollars": [3.0, 2.0, 1.0, -4.0, 7.0, 6.0, 10.0],
    })
    out = t6_straddle(scored, _t6_decisions(), T6_N_EVENTS)
    for line in [
        "| m1 | R_L | 1 | 1 | 1 | 1.25 |",   # (3+2)/4
        "| m1 | R_S | 2 | 0 | 0 | -0.75 |",  # (1-4)/4
        "| m1 | R_X | 0 | 0 | 0 | 1.75 |",   # 7/4
        "| m2 | R_L | 1 | 0 | 0 | 3 |",      # 6/2
        "| m2 | R_S | 0 | 0 | 1 | 5 |",      # 10/2
        "| m3 | R_L | 0 | 0 | 0 | 0 |",      # no decisions, no scored rows
        "| m3 | R_S | 0 | 0 | 0 | 0 |",
    ]:
        assert line in out


def test_t6_empty_scored_gives_zero_totals():
    empty = pd.DataFrame([], columns=["model_id", "rule", "pnl_dollars"])
    out = t6_straddle(empty, _t6_decisions(), T6_N_EVENTS)
    assert "| m1 | R_L | 1 | 1 | 1 | 0 |" in out
    assert "| m2 | R_S | 0 | 0 | 1 | 0 |" in out
    assert "| m3 | R_S | 0 | 0 | 0 | 0 |" in out


def test_t7_real_summary_and_none_row():
    d = np.array([1.0, 2.0, -0.5, 1.5])
    q = np.array(["2024Q1", "2024Q1", "2024Q2", "2024Q2"])
    t = np.array(["A", "B", "A", "B"])
    s = contrast(d, q, t, resamples=50)
    out = t7_contrasts({("H1", "p0", "as_is"): s, ("H1", "p1", "as_is"): None})
    assert "| H1 | p0 | as_is | ok | 4 | 1 | 2/0/0 | 1 | 2024:+ |" in out
    assert "| H1 | p1 | as_is | not_testable |" in out


def test_t7_string_reason_renders_status():
    out = t7_contrasts({("H1", "p9", "as_is"): "too_few_quarters_or_tickers"})
    assert "| H1 | p9 | as_is | not_testable: too_few_quarters_or_tickers | - | - | - | - | - | - | - |" in out


def test_t7_interval_strings_exact():
    s = {"n_events": 4, "mean": 0.5, "quarter_signs": {"2024Q1": 1},
         "year_signs": {2024: 1, 2025: -1}, "quarter_interval": (-0.5, 1.2),
         "ticker_interval": (0.25, 2.0), "quarter_counts": {"positive": 2, "negative": 1, "zero": 1}}
    out = t7_contrasts({("H2", "p3", "as_is"): s})
    assert "| H2 | p3 | as_is | ok | 4 | 0.5 | 2/1/1 | 0.5 | 2024:+ 2025:- | [-0.5, 1.2] | [0.25, 2] |" in out


def test_t8_always_has_rung3_column():
    out = t8_rungs({"m1": {"rung1": 0.5, "rung2": 0.25}})
    assert "| model_id | exit_unquoted_worst_case | rung1 | rung2 |" in out
    assert "| m1 | - | 0.5 | 0.25 |" in out


def test_t8_models_argument_adds_none_row():
    out = t8_rungs({"m1": {"rung1": 0.5}}, models=["m1", "m2"])
    assert "| model_id | exit_unquoted_worst_case | rung1 |" in out
    assert "| m1 | - | 0.5 |" in out
    assert "| m2 | - | - |" in out
    assert "| m2 |" not in t8_rungs({"m1": {"rung1": 0.5}})


def test_t9_difference_and_missing_pool():
    rows = [
        {"model_id": "m1", "pool": "common_2023", "pnl_per_event": 2.0},
        {"model_id": "m1", "pool": "native_full_history", "pnl_per_event": 3.5},
        {"model_id": "m2", "pool": "common_2023", "pnl_per_event": 1.0},
    ]
    out = t9_pool_sensitivity(rows)
    assert "| m1 | 2 | 3.5 | 1.5 |" in out
    assert "| m2 | 1 | - | - |" in out


def test_t10_known_categories_and_note():
    pairs = {"p0": {"as_is": _summary(1)}, "p1": {"as_is": _summary(1)}}
    out = t10_verdicts({"H1": (1, pairs), "H2": (-1, pairs)})
    assert "| H1 | +1 | clear |" in out
    assert "| H2 | -1 | reversed |" in out
    assert "75%" in out and "60%" in out
    assert "judgement calls" in out


def test_input_refusals():
    with pytest.raises(ValueError, match="missing.*pnl_dollars"):
        t4_shortvol(_scored("as_is").drop(columns=["pnl_dollars"]), {"m1": 4})
    bad = _scored("as_is")
    bad["taken"] = ["True", "False", "True"]
    with pytest.raises(ValueError, match="taken"):
        t4_shortvol(bad, {"m1": 4})
    with pytest.raises(ValueError, match="in_e"):
        t2_population(pd.DataFrame({"quarter": ["2024Q1"], "in_e": [np.nan]}))
    nan_taken = _scored("as_is")
    nan_taken.loc[2, "pnl_dollars"] = np.nan  # row 2 is taken
    with pytest.raises(ValueError, match="pnl_dollars"):
        t4_shortvol(nan_taken, {"m1": 4})
    # Row 1 is untaken with NaN pnl (in _scored) and must be ignored.
    assert "| m1 | 2 | 6.00 |" in t4_shortvol(_scored("as_is"), {"m1": 4})
    with pytest.raises(ValueError, match="_per_event"):
        t4_shortvol(_scored("as_is"), {"m1": -4})


def test_cents_and_t3_t6_t9_refusals():
    cents = pd.DataFrame({"model_id": ["m1"] * 2, "structure": ["s"] * 2, "arm": ["as_is"] * 2,
                          "taken": [True, True], "pnl_dollars": [12000.5, 345.17]})
    assert "| m1 | 2 | 12345.67 |" in t4_shortvol(cents, {"m1": 2})
    row = {"model_id": "m1", "pool": "common_2023", "pnl_per_event": 2.0}
    with pytest.raises(ValueError, match="duplicate"):
        t9_pool_sensitivity([row, dict(row)])
    with pytest.raises(ValueError, match="unknown pool"):
        t9_pool_sensitivity([{**row, "pool": "bogus"}])
    scored = pd.DataFrame({"model_id": ["m1"], "rule": ["R_L"], "pnl_dollars": [1.0]})
    decisions = pd.DataFrame({"model_id": ["m1"], "rule": ["R_L"], "status": ["maybe"]})
    with pytest.raises(ValueError, match="unknown decision status"):
        t6_straddle(scored, decisions, {"m1": 4})
    with pytest.raises(ValueError, match="pred"):
        t3_forecast(pd.DataFrame({"model_id": ["m1"], "pred": [np.nan], "actual": [0.5]}), [])


def test_build_report_joins_sections():
    amendment = ("Amendment 6 (post-hoc revision by the user, 2026-11-24): short vol is as-is only; the re-tuned arm, T5 "
                 "and the divisor c were removed.")
    assert build_report(["A", "B"]) == f"# EXP-188 REPORT\n\n{amendment}\n\nA\n\nB\n"
