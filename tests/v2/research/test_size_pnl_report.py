"""Tests for experiments/size_pnl_report.py (EXP-188 slice S8b).

Small synthetic fixtures only; no real data or store.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.size_pnl_contrast import contrast
from experiments.size_pnl_report import (
    _md_table,
    build_report,
    t1_roster,
    t2_population,
    t3_forecast,
    t4_t5_shortvol,
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


def _scored(arm):
    return pd.DataFrame({
        "model_id": ["m1", "m1", "m1"],
        "structure": ["s", "s", "s"],
        "arm": [arm] * 3,
        "taken": [True, False, True],
        "pnl_dollars": [10.0, np.nan, -4.0],
    })


def test_t4_per_event_counts_non_trade_as_zero():
    out = t4_t5_shortvol(_scored("as_is"), {"m1": 4}, {}, arm="as_is")
    assert out.startswith("### T4")
    assert "| m1 | 1 | 2 | 6 | 1.5 | 0.5 |" in out
    assert "skip_note" not in out


def test_t5_retuned_shows_c_and_skip_note():
    tuning = {"m1": {"c": 1.25, "skipped": True, "skip_reason": "fewer_than_300_inner_trades"}}
    out = t4_t5_shortvol(_scored("retuned"), {"m1": 4}, tuning, arm="retuned")
    assert out.startswith("### T5")
    assert "| m1 | 1.25 | 2 | 6 | 1.5 | 0.5 | fewer_than_300_inner_trades |" in out


def test_t4_refusals():
    with pytest.raises(ValueError, match="arm"):
        t4_t5_shortvol(_scored("as_is"), {"m1": 4}, {}, arm="other")
    with pytest.raises(ValueError, match="candidate-event"):
        t4_t5_shortvol(_scored("as_is"), {}, {}, arm="as_is")
    with pytest.raises(ValueError, match="tuning receipt"):
        t4_t5_shortvol(_scored("retuned"), {"m1": 4}, {}, arm="retuned")


def test_t6_undetermined_kept_separate():
    scored = pd.DataFrame({"model_id": ["m1", "m1"], "rule": ["R_L", "R_L"],
                           "pnl_dollars": [3.0, 2.0]})
    decisions = pd.DataFrame({"model_id": ["m1"] * 3, "rule": ["R_L"] * 3,
                              "status": ["taken", "not_taken", "undetermined"]})
    out = t6_straddle(scored, decisions, {"m1": 4})
    assert "| m1 | R_L | 1 | 1 | 1 | 1.25 |" in out


def test_t7_real_summary_and_none_row():
    d = np.array([1.0, 2.0, -0.5, 1.5])
    q = np.array(["2024Q1", "2024Q1", "2024Q2", "2024Q2"])
    t = np.array(["A", "B", "A", "B"])
    s = contrast(d, q, t, resamples=50)
    out = t7_contrasts({("H1", "p0", "as_is"): s, ("H1", "p1", "as_is"): None})
    assert "| H1 | p0 | as_is | ok | 4 | 1 | 2/0/0 | 1 | 2024:+ |" in out
    assert "| H1 | p1 | as_is | not_testable |" in out


def test_t8_always_has_rung3_column():
    out = t8_rungs({"m1": {"rung1": 0.5, "rung2": 0.25}})
    assert "| model_id | exit_unquoted_worst_case | rung1 | rung2 |" in out
    assert "| m1 | - | 0.5 | 0.25 |" in out


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


def test_build_report_joins_sections():
    assert build_report(["A", "B"]) == "# EXP-188 REPORT\n\nA\n\nB\n"
