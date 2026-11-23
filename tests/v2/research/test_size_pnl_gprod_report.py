"""Tests for experiments/size_pnl_gprod_report.py (EXP-188 Amendment 3, slice 2b).

Synthetic frames only; sections and _agreement are called directly.
"""

import pandas as pd
import pytest

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
STORED = {"scored": ["E1", "E2", "E3", "E4", "E5", "E6", "E9"], "selected": ["E1", "E2", "E4", "E9"]}


def _cells(over=None):
    """One (model, event, status, pnl, mid) per raw model and event: taken at 1.0 / 1.0 unless ``over`` maps the pair to a state."""
    over = over or {}
    return [(m, e, *over.get((m, e), ("taken", 1.0, 1.0))) for m in RAW for e in EVENTS]


def _gate(cells, *, with_mid=True):
    gate = pd.DataFrame(cells, columns=["model_id", "event_id", "status", "pnl_dollars", "pnl_dollars_mid"])
    gate["event_date"] = pd.to_datetime(gate["event_id"].map(DATES))
    return gate if with_mid else gate.drop(columns="pnl_dollars_mid")


def _det(gate):
    """The determined rows with labelled model ids, as sections() builds them."""
    det = gate[gate["status"] != "undetermined"]
    return det.assign(model_id=det["model_id"].map(LABEL_OF))


def _run(gate, *, outer=None, stored=None, quarter=QUARTER):
    return gprod_report.sections(gate, FRAME, label_of=LABEL_OF, role_model=ROLE_MODEL, roster=ROSTER,
                                 outer_models=set(LABEL_OF) if outer is None else outer,
                                 quarter_of=quarter, tick=TICK, stored_gate=stored)


def _rows(section):
    """The pipe table of a section as a list of {header: cell} dicts; the separator line is skipped."""
    lines = [line for line in section.split("\n") if line.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(header, (c.strip() for c in line.strip("|").split("|")))) for line in lines[2:]]


def _find(rows, hypothesis, pair_id, arm):
    (row,) = [r for r in rows if r["hypothesis"] == hypothesis and r["pair_id"] == pair_id and r["arm"] == arm]
    return row


def _agreement_gate():
    """M0 taken on E1, E2, E3; not taken on E4..E6; undetermined on E7 and E8."""
    over = {("m0", "E4"): ("not_taken", 0.0, 0.0), ("m0", "E5"): ("not_taken", 0.0, 0.0), ("m0", "E6"): ("not_taken", 0.0, 0.0),
            ("m0", "E7"): ("undetermined", 0.0, 0.0), ("m0", "E8"): ("undetermined", 0.0, 0.0)}
    return _gate(_cells(over))


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
    rows = {r["model_id"]: r for r in _rows(_run(_gate(_cells(over)))[0])}
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
    row = _find(_rows(_run(_gate(_cells(over)))[1]), "M0", "M0-H-sq", "primary")
    assert row["status"] == "ok"
    assert row["n"] == "6"  # E7 and E8 are not in both members' determined events
    assert float(row["mean"]) == 5.0  # M0 pays 6.0 against H-sq's 1.0 on each of E1..E6


def test_mid_arm_reads_the_mid_column_and_is_dropped_without_it():
    over = {("m0", e): ("taken", 1.0, 3.0) for e in EVENTS}
    secs = _run(_gate(_cells(over)))
    rows = _rows(secs[1])
    assert float(_find(rows, "M0", "M0-H-sq", "primary")["mean"]) == 0.0
    assert float(_find(rows, "M0", "M0-H-sq", "mid")["mean"]) == 2.0  # 3.0 - 1.0 on the mid marks
    m0 = {r["model_id"]: r for r in _rows(secs[0])}["M0"]
    assert m0["total_pnl_mid"] == "24"  # 8 events x 3.0
    assert m0["pnl_per_covered_event_mid"] == "3"

    secs = _run(_gate(_cells(over), with_mid=False))
    assert all(r["arm"] == "primary" for r in _rows(secs[1]))
    m0 = {r["model_id"]: r for r in _rows(secs[0])}["M0"]
    assert m0["total_pnl_mid"] == "-" and m0["pnl_per_covered_event_mid"] == "-"


def test_pair_with_a_model_outside_the_run_is_excluded():
    outer = set(LABEL_OF) - {"nlog"}
    row = _find(_rows(_run(_gate(_cells()), outer=outer)[1]), "H2", "N", "primary")
    assert row["status"] == "not_testable: pair_member_excluded"


def test_single_quarter_is_not_testable():
    quarter = {e: "2025Q1" for e in EVENTS}
    row = _find(_rows(_run(_gate(_cells()), quarter=quarter)[1]), "H2", "H", "primary")
    assert row["status"] == "not_testable: too_few_quarters_or_tickers"


def test_no_gate_rows_gives_no_sections():
    assert _run(_gate(_cells()).iloc[0:0]) == []


def test_events_outside_e_never_count():
    secs = _run(_gate(_cells() + [("m0", "E9", "taken", 100.0, 100.0)]))
    m0 = {r["model_id"]: r for r in _rows(secs[0])}["M0"]
    assert (m0["E_events"], m0["covered"], m0["total_pnl"]) == ("8", "8", "8")


def test_agreement_not_available_without_stored_decisions():
    secs = _run(_gate(_cells()))
    assert "not_available: production's stored walk-forward gate decisions" in secs[2]


def test_agreement_counts_by_hand():
    text = _agreement(_det(_agreement_gate()), ROLE_MODEL, LABEL_OF, set(LABEL_OF), STORED)
    (row,) = _rows(text)
    assert row == {"events": "6", "both_taken": "2", "m0_only": "1", "production_only": "1", "neither": "2",
                   "agreement_rate": "0.6667", "production_scored_outside_m0": "1"}


def test_agreement_excludes_m0_rows_dated_outside_2025():
    gate = _agreement_gate()
    gate.loc[(gate["model_id"] == "m0") & (gate["event_id"] == "E6"), "event_date"] = pd.Timestamp("2024-06-15")
    text = _agreement(_det(gate), ROLE_MODEL, LABEL_OF, set(LABEL_OF), STORED)
    (row,) = _rows(text)
    assert row == {"events": "5", "both_taken": "2", "m0_only": "1", "production_only": "1", "neither": "1",
                   "agreement_rate": "0.6", "production_scored_outside_m0": "2"}


def test_agreement_rate_excludes_events_production_never_scored():
    stored = {"scored": ["E1", "E2", "E3", "E4"], "selected": ["E1", "E2", "E4"]}
    text = _agreement(_det(_agreement_gate()), ROLE_MODEL, LABEL_OF, set(LABEL_OF), stored)
    (row,) = _rows(text)
    assert row == {"events": "4", "both_taken": "2", "m0_only": "1", "production_only": "1", "neither": "0",
                   "agreement_rate": "0.5", "production_scored_outside_m0": "0"}


def test_m0_absent_from_the_run_is_not_available():
    outer = set(LABEL_OF) - {"m0"}
    secs = _run(_gate(_cells()), outer=outer)
    assert "not_available: M0_not_in_run" in secs[2]
    assert not any(r["pair_id"].startswith("M0-") for r in _rows(secs[1]))


def test_stored_gate_valid_accepts_a_well_formed_gate():
    assert gprod_report.stored_gate_valid({"scored": ["E1"], "selected": ["E1"]}) is True
    assert gprod_report.stored_gate_valid(STORED) is True


@pytest.mark.parametrize("value", [
    None,
    {},
    {"scored": ["E1"]},
    {"scored": ["E1"], "selected": ["E2"]},
    {"scored": [1], "selected": []},
    {"scored": ["E1"], "selected": [], "extra": []},
    {"scored": "E1", "selected": []},
], ids=["none", "empty", "missing_selected", "selected_outside_scored", "non_string_id", "extra_key", "string_not_list"])
def test_stored_gate_valid_rejects_malformed_values(value):
    assert gprod_report.stored_gate_valid(value) is False
