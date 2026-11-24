"""EXP-188 Amendment 3 (slice 2b): the G_PROD report sections. Research-only: reached from size_pnl_assemble. Pure functions over published decision rows; no I/O."""

import numpy as np
import pandas as pd

from engine.v2.ops.errors import fail
from experiments import size_pnl_contrast as contrast, size_pnl_gate_agreement, size_pnl_report as report
from experiments.size_pnl_inputs import PAIRS

_INVALID = "INVALID_EXPERIMENT_SPEC"
stored_gate_valid = size_pnl_gate_agreement.valid
LABEL =("Sensitivity (Amendment 3): production gate refit once through 2024, not refit per model "
         "(favours M0 forecast scale); not the pre-registered verdict")
MARKS = (("mid", "pnl_dollars"), ("harsh", "pnl_dollars_harsh"))  # the headline arm (mid) first, then the stored primary as harsh fills
MARKS_PRIMARY = (("primary", "pnl_dollars"),)  # no mid column in the decisions: the stored primary marks are the only arm
COVERAGE_NOTE = ("covered = events with a determined G_PROD decision (taken or not_taken) in E; undetermined (a missing or NaN gate input, "
                 "a thin residual pool) is never counted as no. P&L per covered event counts untaken rows as 0. Contrasts below use the "
                 "covered subset of each pair: the events both members have a determined decision on.")
CONTRAST_NOTE = ("d = median-type minus mean-type for the H2 pairs; M0 minus the named model for the M0 rows. arm mid = marks_mid (the headline basis); "
                 "arm harsh = harsh fills, reported only; arm primary = stored primary marks when no mid column exists.")


def _title(text):
    """A section titled with the Amendment 3 label."""
    return f"### {LABEL}: " + text[len("### "):]


def _per_event(det, col):
    """{label: Series of P&L indexed by event_id}: taken rows' ``col`` summed per event, untaken rows 0; no determined row, no key."""
    pnl = pd.DataFrame({"model_id": det["model_id"].to_numpy(), "event_id": det["event_id"].to_numpy(),
                        "pnl": np.where(det["status"] == "taken", det[col], 0.0)})
    return {label: group.groupby("event_id")["pnl"].sum() for label, group in pnl.groupby("model_id")}


def _cell(pnl, a, b, quarter_of, tick):
    """The contrast a - b over the covered subset (events in both a's and b's P&L), or the reason it is not testable."""
    if a not in pnl or b not in pnl:
        return "no_gate_rows"
    ev = sorted(pnl[a].index.intersection(pnl[b].index))
    d = (pnl[a].reindex(ev) - pnl[b].reindex(ev)).to_numpy(dtype=float)
    quarters = np.array([quarter_of[e] for e in ev])
    tickers = np.array([tick[e] for e in ev])
    if len(set(quarters.tolist())) < 2 or len(set(tickers.tolist())) < 2:
        return "too_few_quarters_or_tickers"
    return contrast.contrast(d, quarters, tickers)


def _total(series_by_label, label):
    return float(series_by_label[label].sum()) if label in series_by_label else 0.0


def _agreement(det, role_model, label_of, outer_models, in_e, stored_gate, accepted_gap=None, ladder=None):
    """M0's agreement with production's stored 2025 gate decisions, as a Markdown section (titled by the caller)."""
    m0_raw = role_model.get("M0")
    m0 = label_of.get(m0_raw)
    if m0 is None or m0_raw not in outer_models:
        return f"{size_pnl_gate_agreement.HEADING}\n\nnot_available: M0_not_in_run"
    if stored_gate is None:
        raise fail(_INVALID, "the run has G_PROD rows: pass --stored-gate (EXP-187's transactions CSV) to run.py --stage report", details={"reason": "stored_gate_required"})
    if "gate_score" not in det.columns:
        raise fail(_INVALID, "the decisions predate the gate_score column: rerun the outer stage", details={"reason": "gate_score_missing"})
    if "reason" not in det.columns:
        raise fail(_INVALID, "the decisions lack the reason column", details={"reason": "gate_score_missing"})
    rows = det[(det["model_id"] == m0) & (det["event_date"].dt.year == 2025) & det["event_id"].isin(in_e) & det["status"].isin(("taken", "not_taken"))]
    m0_frame = rows.drop_duplicates("event_id")
    # month_decider fills gate_score for every event with finite inputs; blank the rows the gate never decided
    # (universe_filter, bad_quote) so the ladder step and score distribution cover only scored events. Covered still includes them.
    m0_frame = m0_frame.assign(gate_score=m0_frame["gate_score"].where((m0_frame["status"] == "taken") | (m0_frame["reason"] == "below_gate_threshold")))
    return size_pnl_gate_agreement.section(m0_frame, stored_gate["selected"], accepted_gap=accepted_gap, ladder=ladder)


def sections(gate_raw, frame, *, label_of, role_model, roster, outer_models, quarter_of, tick, stored_gate=None, accepted_gap=None, ladder=None):
    """The G_PROD sections over the events of E (``frame``): per-model coverage and P&L, the covered-subset contrasts, and M0's
    agreement with production's stored 2025 decisions. [] when there are no gate rows."""
    if accepted_gap is not None and not (isinstance(accepted_gap, str) and accepted_gap.strip()):  # direct callers are admitted here whatever the population
        raise fail(_INVALID, "accepted_gap must be a non-empty string", details={"reason": "accepted_gap_invalid"})
    if ladder is not None and not size_pnl_gate_agreement.ladder_valid(ladder):
        raise fail(_INVALID, "ladder must be a dict of label -> {event_id: finite score}", details={"reason": "ladder_invalid"})
    if stored_gate is not None and not stored_gate_valid(stored_gate):
        raise fail(_INVALID, 'stored_gate must be {"selected": {event_id: finite ret}}', details={"reason": "stored_gate_invalid"})
    if gate_raw.empty:
        return []
    swapped = "pnl_dollars_mid" in gate_raw.columns
    if swapped:  # the headline is the mid marks; the stored primary P&L is reported as harsh fills
        gate_raw = gate_raw.assign(pnl_dollars_harsh=gate_raw["pnl_dollars"], pnl_dollars=gate_raw["pnl_dollars_mid"])
    in_e = set(frame["event_id"])
    g = gate_raw[gate_raw["event_id"].isin(in_e)].copy()
    g["model_id"] = g["model_id"].map(label_of)
    det = g[g["status"] != "undetermined"]
    labels = sorted(label_of[m] for m in outer_models)
    marks = [(arm, col) for arm, col in (MARKS if swapped else MARKS_PRIMARY) if col in gate_raw.columns]
    pnl = {arm: _per_event(det, col) for arm, col in marks}
    headline, has_harsh = marks[0][0], "harsh" in pnl

    headers = ["model_id", "E_events", "covered", "coverage", "undetermined", "trades", "total_pnl", "pnl_per_covered_event",
               "hit_rate", "total_pnl_harsh", "pnl_per_covered_event_harsh"]
    rows = []
    for label in labels:
        mine = det[det["model_id"] == label]
        covered = int(mine["event_id"].nunique())
        und = set(g.loc[(g["model_id"] == label) & (g["status"] == "undetermined"), "event_id"]) - set(mine["event_id"])
        taken = mine[mine["status"] == "taken"]
        total = _total(pnl[headline], label)
        hit = float((taken["pnl_dollars"] > 0).mean()) if len(taken) else None
        row = [label, len(in_e), covered, covered / len(in_e) if in_e else None, len(und), len(taken),
               total, total / covered if covered else None, hit]
        if has_harsh:
            total_harsh = _total(pnl["harsh"], label)
            row += [total_harsh, total_harsh / covered if covered else None]
        else:
            row += [None, None]
        rows.append(row)
    table1 = f"### G_PROD per model: coverage and straddle P&L\n\n{COVERAGE_NOTE}\n\n{report.md_table(headers, rows)}"

    shown = {}
    for pid, (mean_role, median_role) in PAIRS.items():
        roles_ok = all(role_model.get(r) in outer_models for r in (mean_role, median_role))
        for arm, _col in marks:
            if not roles_ok or roster["pairs"].get(pid) == "not_testable":
                shown[("H2", pid, arm)] = "pair_member_excluded"
            else:
                a, b = label_of[role_model[median_role]], label_of[role_model[mean_role]]
                shown[("H2", pid, arm)] = _cell(pnl[arm], a, b, quarter_of, tick)
    m0 = role_model.get("M0")
    if m0 in outer_models:
        m0_label = label_of[m0]
        for label in labels:
            if label == m0_label:
                continue
            for arm, _col in marks:
                shown[("M0", f"M0-{label}", arm)] = _cell(pnl[arm], m0_label, label, quarter_of, tick)
    contrast_text = report.t7_contrasts(shown).replace("### T7 Contrasts", "### G_PROD contrasts (covered subset)", 1)
    contrast_text += "\n\n" + CONTRAST_NOTE

    agreement = _agreement(det, role_model, label_of, outer_models, in_e, stored_gate, accepted_gap, ladder)
    return [_title(table1), _title(contrast_text), _title(agreement)]
