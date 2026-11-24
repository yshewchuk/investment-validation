"""Tests for experiments/size_pnl_gprod_report.py (EXP-188 Amendment 3, slice 2b; Amendment 6 agreement section).

Synthetic frames only; sections and _agreement are called directly.
"""

import math

import pandas as pd
import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_pnl_gprod_report as gprod_report
from experiments.size_pnl_gprod_report import _agreement

PREFIX = ("### Sensitivity (Amendment 3): production gate refit once through 2024, not refit per model "
          "(favours M0 forecast scale); not the pre-registered verdict: ")
EVENTS = [f"E{i}" for i in range(1, 9)]
DATES = dict(zip(EVENTS + ["E9"], ["2025-01-15", "2025-02-15", "2025-04-15", "2025-05-15",
                                   "2025-07-15", "2025-08-15", "2025-10-15", "2025-11-15", "2025-12-15"]))
QUARTER = {e: str(pd.Timestamp(DATES[e]).to_period("Q")) for e in EVENTS}
TICK = {e: "AAA" if i % 2 == 0 else "BBB" for i, e in enumerate(EVENTS)}
ROLE_MODEL = {"M0": "m0", "H-sq": "hsq", "H-abs": "habs", "E-raw": "eraw", "E-log": "elog", "N-raw": "nraw", "N-log": "nlog"}
LABEL_OF = {"m0": "M0", "hsq": "H-sq", "habs": "H-abs", "eraw": "E-raw", "elog": "E-log", "nraw": "N-raw", "nlog": "N-log"}
ROSTER = {"pairs": {"H": "testable", "E": "testable", "N": "testable"}}
FRAME = pd.DataFrame({"event_id": EVENTS})
RAW = list(LABEL_OF)
GATE_SCORE, ENTRY_COST, EXIT_VALUE_MID = 0.06, 1.0, 1.1  # every row's gate inputs
# Production selected every covered event: M0's agreement is then 1.0 on the default fixture (the bar passes).
STORED = {"selected": {e: 0.1 for e in EVENTS}}
# Production selected E1, E2, E4 of the covered events and E9, which is outside M0's covered set (counts by hand below).
STORED_HAND = {"selected": {"E1": 0.1, "E2": 0.2, "E4": 0.3, "E9": 0.4}}
# Production selected only E4: no overlap with M0's taken events, so the bar fails.
STORED_BELOW = {"selected": {"E4": 0.3}}
IN_E = set(EVENTS)  # the fixture frame's events E1..E8
IN_E_WITH_E9 = IN_E | {"E9"}  # E9 counted as an event of E


def _cells(over=None):
    """One (model, event, status, pnl, mid, gate_score, entry_cost, exit_value_mid) per raw model and event: taken at 1.0 / 1.0
    unless ``over`` maps the pair to a (status, pnl, mid) state."""
    over = over or {}
    return [(m, e, *over.get((m, e), ("taken", 1.0, 1.0)), GATE_SCORE, ENTRY_COST, EXIT_VALUE_MID) for m in RAW for e in EVENTS]


REASON_OF = {"taken": None, "not_taken": "below_gate_threshold", "undetermined": "thin_pool"}


def _gate(cells, *, with_mid=True):
    gate = pd.DataFrame(cells, columns=["model_id", "event_id", "status", "pnl_dollars", "pnl_dollars_mid",
                                        "gate_score", "entry_cost", "exit_value_mid"])
    gate["reason"] = [REASON_OF[s] for s in gate["status"]]
    gate["event_date"] = pd.to_datetime(gate["event_id"].map(DATES))
    return gate if with_mid else gate.drop(columns="pnl_dollars_mid")


def _det(gate):
    """The determined rows with labelled model ids, as sections() builds them."""
    det = gate[gate["status"] != "undetermined"]
    return det.assign(model_id=det["model_id"].map(LABEL_OF))


def _run(gate, *, outer=None, stored=STORED, quarter=QUARTER, accepted_gap=None, ladder=None):
    return gprod_report.sections(gate, FRAME, label_of=LABEL_OF, role_model=ROLE_MODEL, roster=ROSTER,
                                 outer_models=set(LABEL_OF) if outer is None else outer,
                                 quarter_of=quarter, tick=TICK, stored_gate=stored, accepted_gap=accepted_gap, ladder=ladder)


def _rows(section):
    """The pipe table of a section as a list of {header: cell} dicts; the separator line is skipped."""
    lines = [line for line in section.split("\n") if line.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(header, (c.strip() for c in line.strip("|").split("|")))) for line in lines[2:]]


def _table(text, title):
    """The rows of the pipe table under ``title``, up to the next subsection."""
    body = text.split(title, 1)[1].split("\n#### ", 1)[0]
    return _rows(body)


def _find(rows, hypothesis, pair_id, arm):
    (row,) = [r for r in rows if r["hypothesis"] == hypothesis and r["pair_id"] == pair_id and r["arm"] == arm]
    return row


def _agreement_gate():
    """M0 taken on E1, E2, E3; not taken on E4..E6; undetermined on E7 and E8."""
    over = {("m0", "E4"): ("not_taken", 0.0, 0.0), ("m0", "E5"): ("not_taken", 0.0, 0.0), ("m0", "E6"): ("not_taken", 0.0, 0.0),
            ("m0", "E7"): ("undetermined", 0.0, 0.0), ("m0", "E8"): ("undetermined", 0.0, 0.0)}
    return _gate(_cells(over))


def _counts(text):
    (row,) = _table(text, "#### Counts")
    return row


def test_every_section_is_titled_with_the_amendment_label():
    secs = _run(_gate(_cells()))
    assert [s.split("\n")[0] for s in secs] == [
        PREFIX + "G_PROD per model: coverage and straddle P&L",
        PREFIX + "G_PROD contrasts (covered subset)",
        PREFIX + "G_PROD M0 agreement with production's stored 2025 gate decisions",
    ]


def test_coverage_table_counts_and_pnl_by_hand():
    over = {("m0", "E1"): ("taken", 10.0, 10.0), ("m0", "E2"): ("taken", 20.0, 20.0), ("m0", "E3"): ("taken", 30.0, 30.0),
            ("m0", "E4"): ("not_taken", 0.0, 0.0), ("m0", "E5"): ("not_taken", 0.0, 0.0), ("m0", "E6"): ("not_taken", 0.0, 0.0),
            ("m0", "E7"): ("undetermined", 0.0, 0.0), ("m0", "E8"): ("undetermined", 0.0, 0.0)}
    rows = {r["model_id"]: r for r in _rows(_run(_gate(_cells(over)), accepted_gap="test")[0])}
    m0 = rows["M0"]
    assert (m0["E_events"], m0["covered"], m0["undetermined"], m0["coverage"]) == ("8", "6", "2", "0.75")
    assert m0["trades"] == "3"
    assert m0["total_pnl"] == "60"  # 10 + 20 + 30 on the taken rows; untaken rows count 0
    assert m0["pnl_per_covered_event"] == "10"  # 60 / 6
    assert m0["hit_rate"] == "1"
    for label in ("H-sq", "H-abs", "E-raw", "E-log", "N-raw", "N-log"):
        assert rows[label]["covered"] == "8"


def test_contrast_uses_the_covered_subset_and_hand_mean():
    over = {("m0", e): ("taken", 6.0, 6.0) for e in EVENTS[:6]}
    over.update({("m0", "E7"): ("undetermined", 0.0, 0.0), ("m0", "E8"): ("undetermined", 0.0, 0.0)})
    row = _find(_rows(_run(_gate(_cells(over)))[1]), "M0", "M0-H-sq", "mid")
    assert row["status"] == "ok"
    assert row["n"] == "6"  # E7 and E8 are not in both members' determined events
    assert float(row["mean"]) == 5.0  # M0 pays 6.0 against H-sq's 1.0 on each of E1..E6


def test_mid_arm_reads_the_mid_column_and_is_dropped_without_it():
    over = {("m0", e): ("taken", 1.0, 3.0) for e in EVENTS}
    secs = _run(_gate(_cells(over)))
    rows = _rows(secs[1])
    assert float(_find(rows, "M0", "M0-H-sq", "harsh")["mean"]) == 0.0  # 1.0 - 1.0 on the harsh fills
    assert float(_find(rows, "M0", "M0-H-sq", "mid")["mean"]) == 2.0  # 3.0 - 1.0 on the mid marks
    m0 = {r["model_id"]: r for r in _rows(secs[0])}["M0"]
    assert m0["total_pnl"] == "24"  # headline: 8 events x 3.0 mid
    assert m0["pnl_per_covered_event"] == "3"
    assert m0["total_pnl_harsh"] == "8"  # 8 events x 1.0 harsh
    assert m0["pnl_per_covered_event_harsh"] == "1"

    secs = _run(_gate(_cells(over), with_mid=False))
    assert all(r["arm"] == "primary" for r in _rows(secs[1]))
    m0 = {r["model_id"]: r for r in _rows(secs[0])}["M0"]
    assert m0["total_pnl"] == "8"  # no mid column: the stored primary marks are the headline
    assert m0["total_pnl_harsh"] == "-" and m0["pnl_per_covered_event_harsh"] == "-"


def test_gprod_headline_total_and_hit_rate_use_mid_marks():
    over = {("m0", e): ("not_taken", 0.0, 0.0) for e in EVENTS[1:]}
    over[("m0", "E1")] = ("taken", -10.0, 20.0)  # stored primary -10, mid +20
    m0 = {r["model_id"]: r for r in _rows(_run(_gate(_cells(over)), accepted_gap="test")[0])}["M0"]
    assert m0["total_pnl"] == "20"
    assert m0["hit_rate"] == "1"
    assert m0["total_pnl_harsh"] == "-10"


def test_pair_with_a_model_outside_the_run_is_excluded():
    outer = set(LABEL_OF) - {"nlog"}
    row = _find(_rows(_run(_gate(_cells()), outer=outer)[1]), "H2", "N", "mid")
    assert row["status"] == "not_testable: pair_member_excluded"


def test_single_quarter_is_not_testable():
    quarter = {e: "2025Q1" for e in EVENTS}
    row = _find(_rows(_run(_gate(_cells()), quarter=quarter)[1]), "H2", "H", "mid")
    assert row["status"] == "not_testable: too_few_quarters_or_tickers"


def test_no_gate_rows_gives_no_sections():
    assert _run(_gate(_cells()).iloc[0:0]) == []


def test_events_outside_e_never_count():
    secs = _run(_gate(_cells() + [("m0", "E9", "taken", 100.0, 100.0, GATE_SCORE, ENTRY_COST, EXIT_VALUE_MID)]))
    m0 = {r["model_id"]: r for r in _rows(secs[0])}["M0"]
    assert (m0["E_events"], m0["covered"], m0["total_pnl"]) == ("8", "8", "8")


def test_direct_sections_refuses_a_blank_accepted_gap_on_an_empty_gate_frame():
    with pytest.raises(OpsError) as caught:
        _run(_gate(_cells()).iloc[0:0], accepted_gap="")
    assert caught.value.problem.details["reason"] == "accepted_gap_invalid"


def test_direct_sections_refuses_a_boolean_stored_return():
    with pytest.raises(OpsError) as caught:
        _run(_gate(_cells()), stored={"selected": {"E1": True}})
    assert caught.value.problem.details["reason"] == "stored_gate_invalid"


def test_missing_stored_gate_raises_stored_gate_required():
    with pytest.raises(OpsError) as caught:
        _run(_gate(_cells()), stored=None)
    assert caught.value.problem.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.details["reason"] == "stored_gate_required"


def test_m0_absent_from_the_run_is_not_available_without_a_stored_gate():
    outer = set(LABEL_OF) - {"m0"}
    secs = _run(_gate(_cells()), outer=outer, stored=None)  # no M0 row: the agreement is not available, not an error
    assert "not_available: M0_not_in_run" in secs[2]
    assert not any(r["pair_id"].startswith("M0-") for r in _rows(secs[1]))


def test_agreement_counts_by_hand():
    text = _agreement(_det(_agreement_gate()), ROLE_MODEL, LABEL_OF, set(LABEL_OF), IN_E_WITH_E9, STORED_HAND)
    assert _counts(text) == {"covered": "6", "gprod_selected": "3", "production_selected": "3", "both": "2", "only_production": "1",
                             "only_gprod": "1", "stored_outside_covered": "1", "rate_gprod": "0.5", "rate_production": "0.5",
                             "jaccard": "0.5", "agreement_rate": "0.6667", "bar": "0.5"}


def test_agreement_excludes_m0_rows_dated_outside_2025():
    gate = _agreement_gate()
    gate.loc[(gate["model_id"] == "m0") & (gate["event_id"] == "E6"), "event_date"] = pd.Timestamp("2024-06-15")
    text = _agreement(_det(gate), ROLE_MODEL, LABEL_OF, set(LABEL_OF), IN_E_WITH_E9, STORED_HAND)
    assert _counts(text) == {"covered": "5", "gprod_selected": "3", "production_selected": "3", "both": "2", "only_production": "1",
                             "only_gprod": "1", "stored_outside_covered": "1", "rate_gprod": "0.6", "rate_production": "0.6",
                             "jaccard": "0.5", "agreement_rate": "0.6", "bar": "0.5"}


def test_agreement_below_the_bar_raises_without_an_accepted_gap():
    with pytest.raises(OpsError) as caught:
        _run(_agreement_gate(), stored=STORED_BELOW)
    assert caught.value.problem.details["reason"] == "gate_agreement_below_bar"
    assert "jaccard 0.0" in str(caught.value)


def test_agreement_below_the_bar_passes_with_an_accepted_gap_and_says_so():
    secs = _run(_agreement_gate(), stored=STORED_BELOW, accepted_gap="named gap for the test")
    text = secs[2]
    assert "not production-faithful" in text
    assert "named gap for the test" in text
    assert _counts(text)["jaccard"] == "0"


def test_universe_filtered_row_is_covered_but_not_scored():
    """M0's not_taken universe_filter row on E4 carries gate_score 0.9 (above the registered threshold) but the gate never decided it:
    it stays covered, and neither the as-run ladder step nor the score distribution counts it."""
    gate = _agreement_gate()
    gate.loc[(gate["model_id"] == "m0") & (gate["event_id"] == "E4"), ["reason", "gate_score"]] = ["universe_filter", 0.9]
    text = _agreement(_det(gate), ROLE_MODEL, LABEL_OF, set(LABEL_OF), IN_E, STORED_HAND)
    assert _counts(text)["covered"] == "6"  # E4 is still in M0's covered set
    (step,) = [r for r in _table(text, "#### Decomposition ladder (diagnostic, never a rule)") if r["step"].startswith("decisions")]
    assert step["scored"] == "5"  # E1, E2, E3, E5, E6: E4 is not scored
    assert step["selected"] == "5"  # with E4 counted this would be 6
    (finite,) = [r for r in _table(text, "#### Score distribution") if r["statistic"] == "finite_scores"]
    assert finite["value"] == "5"


def test_missing_gate_score_column_raises_gate_score_missing():
    gate = _agreement_gate().drop(columns="gate_score")
    with pytest.raises(OpsError) as caught:
        _agreement(_det(gate), ROLE_MODEL, LABEL_OF, set(LABEL_OF), IN_E, STORED_HAND)
    assert caught.value.problem.details["reason"] == "gate_score_missing"


@pytest.mark.parametrize("value", [
    {"selected": {"E1": 0.1}},
    {"selected": {}},
], ids=["one_event", "empty_selected"])
def test_stored_gate_valid_accepts_a_well_formed_gate(value):
    assert gprod_report.stored_gate_valid(value) is True


@pytest.mark.parametrize("value", [
    None,
    {},
    {"scored": ["E1"], "selected": {"E1": 0.1}},
    {"selected": ["E1"]},
    {"selected": {"E1": True}},
    {"selected": {"E1": float("nan")}},
    {"selected": {"E1": math.inf}},
    {"selected": {"": 0.1}},
], ids=["none", "empty", "extra_key", "non_dict_selected", "bool_ret", "nan_ret", "inf_ret", "empty_id"])
def test_stored_gate_valid_rejects_malformed_values(value):
    assert gprod_report.stored_gate_valid(value) is False
