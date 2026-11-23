"""EXP-188 S9c: REPORT.md with the inner window (2023H2-2024) decisions, over synthetic receipts and decisions (tmp_path only)."""
import hashlib

import pandas as pd
import pytest

from experiments import size_pnl_assemble as assemble_mod
from experiments import size_pnl_contrast as contrast
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_tuning as tuning
from experiments.size_pnl_shortvol import Refusal
from tests.v2.research.test_size_pnl_assemble import (  # noqa: F401 (imported so the autouse check_plan no-op applies here)
    EVENT_DATES, ROLES, TICKERS, _cells, _no_plan_admission, _rewrite, _rows, _section_lines, _with_mid, build, plan_frame)

NEW_EVENTS = {"E05": "2025-08-15", "E06": "2025-11-15", "E11": "2023-08-15", "E12": "2024-02-15",
              "E13": "2024-05-15", "E14": "2024-08-15", "E15": "2023-11-15"}
NEW_TICKERS = {"E05": "BBB", "E06": "CCC", "E11": "AAA", "E12": "BBB", "E13": "CCC", "E14": "AAA", "E15": "BBB"}
OUTER_2025 = {e: EVENT_DATES[e] for e in ("E03", "E04")} | {"E05": NEW_EVENTS["E05"], "E06": NEW_EVENTS["E06"]}
INNER_2024 = {e: NEW_EVENTS[e] for e in ("E11", "E12", "E13", "E14")}
INNER_2023 = {e: NEW_EVENTS[e] for e in ("E11", "E15")}
INNER_PREFIX = "### Inner window (2023H2-2024), not the headline: "


@pytest.fixture
def inner_events(monkeypatch):
    for event, day in NEW_EVENTS.items():
        monkeypatch.setitem(EVENT_DATES, event, day)
    for event, ticker in NEW_TICKERS.items():
        monkeypatch.setitem(TICKERS, event, ticker)


def _corrupt_inner(out):
    path = out / "decisions_inner.parquet"
    path.write_bytes(b"CORRUPT" + path.read_bytes())


HASH = "h" * 64  # the inputs the inner receipt and the inner_decisions body share; build writes inner.json without them


def build_inner(tmp_path, inner_dates, *, outer_dates, tuning_override=None, stage="inner_decisions", mutate=None, body_override=None,
                pnl_scale=1.0, mid=False):
    """A full outer run over every event in TICKERS, plus its inner decisions (decisions_inner.parquet and receipt).
    ``pnl_scale`` multiplies the inner rows' pnl_dollars, so the inner window's values differ from the outer's.
    ``mid`` adds the Amendment 1 column (pnl_dollars_mid = pnl_dollars) to the outer and the inner rows."""
    out = build(tmp_path, dates=outer_dates, plan_tickers=TICKERS, rows_fn=_with_mid if mid else None)
    _rewrite(out / "receipts" / "inner.json", lambda body: {**body, "inputs": HASH, "scope": None})
    outer = tuning.load_receipt(out / "receipts" / "outer.json")
    rows = [{**row, "in_sample": row["arm_or_rule"] == "retuned", "pnl_dollars": row["pnl_dollars"] * pnl_scale} for row in _rows(inner_dates)]
    if mid:
        rows = [{**row, "pnl_dollars_mid": row["pnl_dollars"]} for row in rows]
    body = {"stage": stage, "models": list(ROLES),
            "tuning": outer["tuning"] if tuning_override is None else tuning_override,
            "c": {mid: 1.0 for mid in ROLES}, "emit_from": "2023-07-01", "ladder": "x", "scope": None, "inputs": HASH,
            **(body_override or {})}
    decisions.publish(out, [pd.DataFrame(rows)], body, stem="decisions_inner", receipt="inner_decisions")
    receipt = tuning.load_receipt(out / "receipts" / "inner_decisions.json")
    assert receipt["decisions_sha256"] == hashlib.sha256((out / "decisions_inner.parquet").read_bytes()).hexdigest()
    if mutate:
        mutate(out)
    return out


def _t10_verdicts(text):
    """The T10 rows as cells: [hypothesis, sign, category]."""
    return [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in _section_lines(text, "### T10") if line.startswith("| H")]


def test_t10_is_computable_with_inner_2024_and_outer_2025(tmp_path, inner_events, monkeypatch):
    # Oracle by hand: every model pays 10 * (i + 1), the median-type model one dollar more in every event, so each pair's
    # median-minus-mean P&L is +1 per event in both books and both arms. The pooled events are E11 (2023Q3), E12-E14 (2024Q1-Q3),
    # E03-E06 (2025Q1-Q4): one event per quarter, 8 in all, so each quarter, year and ticker group sums to +1 in every pair.
    #   H1 (short-vol, predicted +1): d = +1 -> clear.
    #   H2 (straddle, predicted -1, leg sign +1): d = +1 -> reversed (clear(-1) fails: no pair has sign -1).
    #   H3 (short-vol minus the LONG_RULE straddle): d = +1 - (+1) = 0 -> every sum is 0, no quarter or year has a sign and
    #       no interval excludes zero, so neither clear nor directional holds for either sign -> no_difference.
    # The pooled T10 input is the contrast summary, not a table in REPORT.md, so the spy reads the summaries contrast() returns.
    real, summaries = contrast.contrast, []

    def spy(d, quarters, tickers, **kwargs):
        result = real(d, quarters, tickers, **kwargs)
        summaries.append(result)
        return result

    monkeypatch.setattr(contrast, "contrast", spy)
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert _t10_verdicts(text) == [["H1", "+1", "clear"], ["H2", "-1", "reversed"], ["H3", "+1", "no_difference"]]
    assert "are absent from" not in text
    assert "T10 pools the inner" in text
    pooled = [s for s in summaries if s["n_events"] == len(INNER_2024) + len(OUTER_2025)]  # the first is H1, pair H, as_is
    assert len(pooled) == 18 and pooled[0]["n_events"] == 8
    assert {2024, 2025} <= set(pooled[0]["year_signs"]) and pooled[0]["year_signs"][2024] == pooled[0]["year_signs"][2025] == 1
    assert pooled[0]["mean"] == 1.0


SENS_T10_TITLE = "### Sensitivity (Amendment 1), not the pre-registered verdict: T10"


def test_sensitivity_t10_pools_the_inner_window_too(tmp_path, inner_events):
    # Same hand derivation as test_t10_is_computable_with_inner_2024_and_outer_2025: the mid column equals the primary P&L here,
    # so the sensitivity verdicts are the primary ones. Outer 2025 alone has no full 2024 year, so only the pooled inner rows make them testable.
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, mid=True)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    sens = [_cells(line) for line in _section_lines(text, SENS_T10_TITLE) if line.startswith("| H")]
    assert sens == [["H1", "+1", "clear"], ["H2", "-1", "reversed"], ["H3", "+1", "no_difference"]]
    assert not all(row[2] == "not_testable" for row in sens)


def test_t10_still_refuses_when_a_full_year_is_missing(tmp_path, inner_events):
    out = build_inner(tmp_path, INNER_2023, outer_dates=OUTER_2025)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert [row[2] for row in _t10_verdicts(text)] == ["not_testable"] * 3
    assert "absent from the inner and outer decisions" in text


def test_inner_tables_print_separately_and_the_retuned_rows_are_marked_in_sample(tmp_path, inner_events):
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    for table in ("T4", "T5", "T6", "T7", "T2"):
        assert f"{INNER_PREFIX}{table}" in text
    assert "### T4" in text  # the headline outer heading is unchanged
    assert "IN-SAMPLE" in "\n".join(_section_lines(text, f"{INNER_PREFIX}T5"))
    insample = [line for line in _section_lines(text, "### T10") if "retuned: year_signs" in line]
    assert insample and any("2024*" in line for line in insample)


def _model_rows(text, title, label):
    return [_cells(line) for line in _section_lines(text, title) if line.startswith(f"| {label} |")]


def test_inner_tables_carry_the_inner_windows_own_values(tmp_path, inner_events):
    # Oracle by hand. Outer (2025): H-sq pays 10, 20, 30, 40 per event; H-abs pays one dollar more per event.
    # Inner (pnl_scale=3, INNER_2024): the final pnl_dollars are 3x, so the bump is scaled too. H-sq takes 3*(10+20+30+40)=300
    # (per event 75); H-abs takes 3*(11+21+31+41)=312 (per event 78). The H1 pair H difference is 3 per event inner, 1 outer.
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, pnl_scale=3.0)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    t2 = [_cells(line) for line in _section_lines(text, INNER_PREFIX + "T2") if line.startswith("| 20")]
    assert t2 == [["2023Q3", "1"], ["2024Q1", "1"], ["2024Q2", "1"], ["2024Q3", "1"]]  # no 2025 quarter
    t4 = _model_rows(text, INNER_PREFIX + "T4", "H-sq")
    assert t4 == [["H-sq", "1", "4", "300.00", "75", "1"]]
    assert _model_rows(text, INNER_PREFIX + "T4", "H-abs") == [["H-abs", "1", "4", "312.00", "78", "1"]]
    assert _model_rows(text, INNER_PREFIX + "T5", "H-sq")[0][:6] == t4[0]  # the fixture pays the same under both arms
    assert _model_rows(text, INNER_PREFIX + "T6", "H-sq") == [["H-sq", "R_L", "4", "0", "0", "75"], ["H-sq", "R_L2", "4", "0", "0", "75"]]
    assert _model_rows(text, INNER_PREFIX + "T6", "H-abs")[0] == ["H-abs", "R_L", "4", "0", "0", "78"]
    inner7 = next(_cells(line) for line in _section_lines(text, INNER_PREFIX + "T7") if line.startswith("| H1 | H | as_is |"))
    outer7 = next(_cells(line) for line in _section_lines(text, "### T7") if line.startswith("| H1 | H | as_is |"))
    assert inner7[3:6] == ["ok", "4", "3"]  # n 4, mean 3 * (+1) per event
    assert outer7[3:6] == ["ok", "4", "1"]  # the outer window is unscaled: mean +1
    assert _model_rows(text, "### T4", "H-sq") == [["H-sq", "1", "4", "100.00", "25", "1"]]  # the outer 25 per event
    assert t4 != _model_rows(text, "### T4", "H-sq")


INNER_REFUSALS = [
    ("report_receipt_chain_mismatch", {"tuning_override": {"m-hsq": "0" * 64}}),
    ("report_receipt_chain_mismatch", {"body_override": {"inputs": "g" * 64}}),
    ("report_receipt_chain_mismatch", {"body_override": {"scope": "full"}}),
    ("report_receipt_chain_mismatch", {"body_override": {"models": list(ROLES)[:-1]}}),
    ("report_receipt_stage", {"stage": "inner"}),
    ("report_receipt_chain_mismatch", {"body_override": {"c": {"m-hsq": 9.0, **{m: 1.0 for m in ROLES if m != "m-hsq"}}}}),
    ("report_decisions_mismatch", {"mutate": _corrupt_inner}),
]


@pytest.mark.parametrize("reason, kwargs", INNER_REFUSALS, ids=[reason for reason, _kwargs in INNER_REFUSALS])
def test_inner_receipt_refusals(tmp_path, inner_events, reason, kwargs):
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, **kwargs)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert caught.value.details["reason"] == reason
    assert not (out / "REPORT.md").exists()


def test_without_inner_decisions_the_report_is_unchanged_in_shape(tmp_path):
    out = build(tmp_path)
    summary = assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert summary["sections"] == 12
    assert "### Inner window" not in text
