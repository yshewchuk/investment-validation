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
    EVENT_DATES, ROLES, TICKERS, _no_plan_admission, _rewrite, _rows, _section_lines, build, plan_frame)

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


def build_inner(tmp_path, inner_dates, *, outer_dates, tuning_override=None, stage="inner_decisions", mutate=None, body_override=None):
    """A full outer run over every event in TICKERS, plus its inner decisions (decisions_inner.parquet and receipt)."""
    out = build(tmp_path, dates=outer_dates, plan_tickers=TICKERS)
    _rewrite(out / "receipts" / "inner.json", lambda body: {**body, "inputs": HASH, "scope": None})
    outer = tuning.load_receipt(out / "receipts" / "outer.json")
    rows = [{**row, "in_sample": row["arm_or_rule"] == "retuned"} for row in _rows(inner_dates)]
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


INNER_REFUSALS = [
    ("report_receipt_chain_mismatch", {"tuning_override": {"m-hsq": "0" * 64}}),
    ("report_receipt_chain_mismatch", {"body_override": {"inputs": "g" * 64}}),
    ("report_receipt_chain_mismatch", {"body_override": {"scope": "full"}}),
    ("report_receipt_chain_mismatch", {"body_override": {"models": list(ROLES)[:-1]}}),
    ("report_receipt_stage", {"stage": "inner"}),
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
