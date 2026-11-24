"""Tests for experiments/size_pnl_gate_agreement.py (EXP-188 Amendment 6, slice A6-3a). Synthetic fixtures only; tmp_path for files."""
import math
import os

import pandas as pd
import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_pnl_gate_agreement as agreement
from experiments.size_pnl_gate_fold import REGISTERED_THRESHOLD

# Six covered events of M0 (determined 2025 rows): E1..E4 taken by G_PROD, E5 and E6 not taken.
M0 = pd.DataFrame({
    "event_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
    "status": ["taken", "taken", "taken", "taken", "not_taken", "not_taken"],
    "gate_score": [0.06, 0.07, 0.05, 0.08, 0.01, 0.02],
    "entry_cost": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "exit_value_mid": [1.1, 1.3, 0.9, 1.2, 1.0, 1.0],
})
# Production's stored selected set: E1, E2 and E5 of the covered events, and E9, which is outside M0's covered set.
SELECTED = {"E1": 0.1, "E2": 0.2, "E5": 0.5, "E9": 0.9}


def _st(covered=10, n_gprod=4, n_prod=3, both=2, jaccard=0.5):
    return {"covered": covered, "n_gprod": n_gprod, "n_prod": n_prod, "both": both, "jaccard": jaccard}


def _rows(section):
    """The pipe table of a section as a list of {header: cell} dicts; the separator line is skipped."""
    lines = [line for line in section.split("\n") if line.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(header, (c.strip() for c in line.strip("|").split("|")))) for line in lines[2:]]


def _reason(exc):
    return exc.value.problem.details["reason"]


# ---- read_stored -------------------------------------------------------------

def test_read_stored_happy_path(tmp_path):
    path = tmp_path / "transactions.csv"
    path.write_text("event_id,ret,extra\nE1,0.1,x\nE2,-0.05,y\n", encoding="utf-8")
    assert agreement.read_stored(path) == {"selected": {"E1": 0.1, "E2": -0.05}}


def test_read_stored_refuses_a_duplicate_event_id(tmp_path):
    path = tmp_path / "transactions.csv"
    path.write_text("event_id,ret\nE1,0.1\nE1,0.2\n", encoding="utf-8")
    with pytest.raises(OpsError) as caught:
        agreement.read_stored(path)
    assert _reason(caught) == "stored_gate_invalid"


def test_read_stored_refuses_an_empty_event_id(tmp_path):
    path = tmp_path / "transactions.csv"
    path.write_text("event_id,ret\n,0.1\nE2,0.2\n", encoding="utf-8")
    with pytest.raises(OpsError) as caught:
        agreement.read_stored(path)
    assert _reason(caught) == "stored_gate_invalid"


def test_read_stored_refuses_a_symlink(tmp_path):
    target = tmp_path / "transactions.csv"
    target.write_text("event_id,ret\nE1,0.1\n", encoding="utf-8")
    link = tmp_path / "link.csv"
    os.symlink(target, link)
    with pytest.raises(OpsError) as caught:
        agreement.read_stored(link)
    assert _reason(caught) == "stored_gate_unreadable"


def test_read_stored_refuses_a_missing_file(tmp_path):
    with pytest.raises(OpsError) as caught:
        agreement.read_stored(tmp_path / "missing.csv")
    assert _reason(caught) == "stored_gate_unreadable"


def test_read_stored_refuses_an_oversize_file(tmp_path, monkeypatch):
    monkeypatch.setattr(agreement, "MAX_STORED_BYTES", 10)
    path = tmp_path / "transactions.csv"
    path.write_text("event_id,ret\nE1,0.1\n", encoding="utf-8")  # 19 bytes
    with pytest.raises(OpsError) as caught:
        agreement.read_stored(path)
    assert _reason(caught) == "stored_gate_unreadable"


# ---- stats -------------------------------------------------------------------

def test_stats_by_hand_on_the_six_event_fixture():
    st = agreement.stats(M0, SELECTED)
    assert st["covered"] == 6 and st["n_gprod"] == 4 and st["n_prod"] == 3
    assert (st["both"], st["only_prod"], st["only_gprod"], st["outside"]) == (2, 1, 2, 1)
    assert st["rate_gprod"] == pytest.approx(4 / 6)
    assert st["rate_prod"] == pytest.approx(3 / 6)
    assert st["jaccard"] == pytest.approx(2 / 5)  # both 2 over the union {E1..E5}
    assert st["agreement_rate"] == pytest.approx(3 / 6)  # (covered - only_prod - only_gprod) / covered
    assert st["ret_both_prod"] == pytest.approx((0.1 + 0.2) / 2)
    assert st["ret_only_prod"] == pytest.approx(0.5)  # E5 only
    assert st["ret_both_gprod"] == pytest.approx((0.1 + 0.3) / 2)  # exit_value_mid / entry_cost - 1 on E1, E2
    assert st["ret_only_gprod"] == pytest.approx((-0.1 + 0.2) / 2)  # E3, E4


def test_a_selected_event_outside_the_covered_set_is_counted_and_not_used():
    st = agreement.stats(M0, SELECTED)
    without = agreement.stats(M0, {k: v for k, v in SELECTED.items() if k != "E9"})
    assert st["outside"] == 1 and without["outside"] == 0
    for key in st:
        if key != "outside":
            assert st[key] == without[key], key


def test_a_covered_event_absent_from_the_file_is_read_as_not_selected():
    st = agreement.stats(M0, {"E1": 0.1})
    assert (st["n_prod"], st["both"], st["only_prod"]) == (1, 1, 0)
    assert st["only_gprod"] == 3  # E2, E3 and E4 are taken by G_PROD and absent from the file


def test_stats_on_no_covered_events_gives_none_rates():
    empty = M0.iloc[0:0]
    st = agreement.stats(empty, SELECTED)
    assert st["covered"] == 0 and st["jaccard"] is None and st["rate_gprod"] is None


# ---- check -------------------------------------------------------------------

def test_check_passes_at_exactly_the_bar():
    assert agreement.check(_st(jaccard=agreement.MIN_JACCARD), None) == ""


def test_check_below_the_bar_raises_with_the_numbers_in_the_message():
    with pytest.raises(OpsError) as caught:
        agreement.check(_st(covered=10, n_gprod=4, n_prod=3, both=2, jaccard=0.4), None)
    assert _reason(caught) == "gate_agreement_below_bar"
    message = str(caught.value)
    assert "covered 10" in message and "gprod selected 4" in message and "production selected 3" in message
    assert "overlap 2" in message and "jaccard 0.4" in message


def test_check_below_the_bar_with_an_accepted_gap_returns_the_label():
    label = agreement.check(_st(jaccard=0.4), "named gap")
    assert "not production-faithful" in label and "named gap" in label


@pytest.mark.parametrize("gap", ["", "   "])
def test_check_refuses_an_empty_accepted_gap(gap):
    with pytest.raises(OpsError) as caught:
        agreement.check(_st(jaccard=0.9), gap)
    assert _reason(caught) == "accepted_gap_invalid"


def test_check_refuses_when_nothing_is_covered():
    with pytest.raises(OpsError) as caught:
        agreement.check(_st(covered=0, jaccard=None), None)
    assert _reason(caught) == "gate_agreement_below_bar"


# ---- quantile_table ----------------------------------------------------------

def test_quantile_table_on_known_scores_excludes_nan():
    m0 = pd.DataFrame({"gate_score": [0.02, 0.04, 0.06, 0.08, float("nan")]})
    rows = {r["statistic"]: r["value"] for r in _rows(agreement.quantile_table(m0, 0.05))}
    assert rows["finite_scores"] == "4"
    assert float(rows["q0.5"]) == pytest.approx(0.05)  # median of 0.02, 0.04, 0.06, 0.08
    assert float(rows["min"]) == pytest.approx(0.02) and float(rows["max"]) == pytest.approx(0.08)
    assert float(rows["registered threshold"]) == pytest.approx(0.05)
    assert float(rows["share >= threshold"]) == pytest.approx(0.5)  # 0.06 and 0.08 of the four


def test_quantile_table_reports_the_registered_threshold_it_is_given():
    rows = {r["statistic"]: r["value"] for r in _rows(agreement.quantile_table(M0, REGISTERED_THRESHOLD))}
    assert float(rows["registered threshold"]) == pytest.approx(REGISTERED_THRESHOLD, rel=1e-3)


# ---- ladder_table ------------------------------------------------------------

def test_ladder_table_rows_by_hand():
    covered = {"E1", "E2", "E3", "E4"}
    prod = {"E1", "E2"}
    steps = {
        "band": {"E1": 0.10, "E2": 0.02, "E3": 0.07, "E4": float("nan"), "E9": 0.9},  # E4 not finite; E9 not covered
        "analog": {"E1": 0.06, "E2": 0.08, "E3": 0.01, "E4": 0.5, "E5": 0.9},  # E5 not covered
    }
    rows = {r["step"]: r for r in _rows(agreement.ladder_table(steps, covered, prod, 0.05))}
    assert rows["band"] == {"step": "band", "scored": "3", "selected": "2", "rate": "0.6667",
                            "overlap_with_production": "1", "jaccard": "0.3333"}
    assert rows["analog"] == {"step": "analog", "scored": "4", "selected": "3", "rate": "0.75",
                              "overlap_with_production": "2", "jaccard": "0.6667"}


# ---- section -----------------------------------------------------------------

def test_section_with_a_ladder_renders_the_extra_step():
    ladder = {"band": {"E1": 0.10, "E2": 0.02, "E3": 0.07}}
    text = agreement.section(M0, SELECTED, accepted_gap="test gap", ladder=ladder)
    steps = [r["step"] for r in _rows(text.split("#### Decomposition ladder", 1)[1])]
    assert steps == ["decisions (M0 gate_score as run)", "band"]
    assert agreement.NO_LADDER not in text


def test_section_without_a_ladder_renders_the_no_ladder_note():
    text = agreement.section(M0, SELECTED, accepted_gap="test gap")
    steps = [r["step"] for r in _rows(text.split("#### Decomposition ladder", 1)[1])]
    assert steps == ["decisions (M0 gate_score as run)"]
    assert agreement.NO_LADDER in text


# ---- ladder_valid ------------------------------------------------------------

@pytest.mark.parametrize("value", [{}, {"band": {"E1": 0.1}}, {"band": {}}])
def test_ladder_valid_accepts_well_formed_ladders(value):
    assert agreement.ladder_valid(value) is True


@pytest.mark.parametrize("value", [
    None,
    [],
    {"band": ["E1"]},
    {1: {"E1": 0.1}},
    {"band": "E1"},
    {"band": {"E1": "x"}},
    {"band": {"E1": True}},
    {"band": {"E1": float("nan")}},
    {"band": {"": 0.1}},
    {"band": {1: 0.1}},
], ids=["none", "list", "steps_not_dict", "int_label", "scores_not_dict", "string_score", "bool_score", "nan_score",
        "empty_id", "int_id"])
def test_ladder_valid_rejects_bad_shapes(value):
    assert agreement.ladder_valid(value) is False
