"""EXP-188 S9c: REPORT.md with the inner window (2023H2-2024) decisions, over synthetic receipts and decisions (tmp_path only).
Short vol is as-is only (Amendment 6): no re-tuned arm, no T5, no tuning or c binding."""
import hashlib

import pandas as pd
import pytest

from experiments import size_pnl_assemble as assemble_mod
from experiments import size_pnl_contrast as contrast
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_driver
from experiments import size_pnl_receipts as tuning
from experiments.size_pnl_shortvol import Refusal
from tests.v2.research.test_size_pnl_assemble import (  # noqa: F401 (imported so the autouse check_plan no-op applies here)
    EVENT_DATES, ROLES, TICKERS, _cells, _no_plan_admission, _rewrite, _rows, _section_lines, _with_endpoints, _with_mid, build,
    plan_frame)

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


def _with_doubled_mid(rows):
    """The rows with a ``pnl_dollars_mid`` column of twice each primary P&L, so the sensitivity values differ from the primary ones."""
    return [{**row, "pnl_dollars_mid": 2 * row["pnl_dollars"]} for row in rows]


def build_inner(tmp_path, inner_dates, *, outer_dates, body_override=None,
                pnl_scale=1.0, mid=False, outer_rows_fn=None, inner_rows_fn=None, mutate=None):
    """A full outer run over every event in TICKERS, plus its inner decisions (decisions_inner.parquet and receipt).
    ``pnl_scale`` multiplies the inner rows' pnl_dollars, so the inner window's values differ from the outer's.
    ``mid`` adds the Amendment 1 column (pnl_dollars_mid = 2 * pnl_dollars) to the outer and the inner rows.
    ``outer_rows_fn`` transforms the outer decisions rows (passed as ``rows_fn`` to build), e.g. to drop a model the outer excluded.
    ``inner_rows_fn`` transforms the inner decisions rows last, before they are published."""
    if outer_rows_fn is None and not mid:
        rows_fn = None
    else:
        def rows_fn(rows):
            rows = outer_rows_fn(rows) if outer_rows_fn else rows
            return _with_doubled_mid(rows) if mid else rows
    out = build(tmp_path, dates=outer_dates, plan_tickers=TICKERS, rows_fn=rows_fn)
    rows = [{**row, "pnl_dollars": row["pnl_dollars"] * pnl_scale} for row in _rows(inner_dates)]
    if mid:
        rows = [{**row, "pnl_dollars_mid": 2 * row["pnl_dollars"]} for row in rows]
    if inner_rows_fn:
        rows = inner_rows_fn(rows)
    inputs_inner_path = out / "receipts" / "inputs_inner.json"
    tuning.write_receipt(inputs_inner_path,
                         {"stage": "inputs", "plan": size_pnl_driver.inner_plan_binding(plan_frame(TICKERS))})
    inputs_hash = tuning.load_receipt(inputs_inner_path)["content_hash"]  # inner.json and the inner_decisions body bind this receipt
    _rewrite(out / "receipts" / "inner.json", lambda body: {**body, "inputs": inputs_hash, "scope": None})
    body = {"stage": "inner_decisions", "models": list(ROLES), "emit_from": "2023-07-01", "ladder": "x", "scope": None,
            "inputs": inputs_hash, **(body_override or {})}
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
    # median-minus-mean P&L is +1 per event in both books. The pooled events are E11 (2023Q3), E12-E14 (2024Q1-Q3),
    # E03-E06 (2025Q1-Q4): one event per quarter, 8 in all, so each quarter, year and ticker group sums to +1 in every pair.
    #   H1 (short-vol, predicted +1): d = +1 -> clear.
    #   H2 (straddle, predicted -1, leg sign +1): d = +1 -> reversed (clear(-1) fails: no pair has sign -1).
    #   H3 (short-vol minus the G_PROD straddle): d = +1 - (+1) = 0 -> every sum is 0, no quarter or year has a sign and
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
    assert len(pooled) == 9 and pooled[0]["n_events"] == 8  # H1, H2 (G_PROD) and H3 one arm each, over three pairs (the outer G_PROD sections hold 4 events)
    assert {2024, 2025} <= set(pooled[0]["year_signs"]) and pooled[0]["year_signs"][2024] == pooled[0]["year_signs"][2025] == 1
    assert pooled[0]["mean"] == 1.0


HARSH_T10_TITLE = "### Harsh fills, reported only, not a verdict basis: T10 Verdicts"
HARSH_T7_TITLE = "### Harsh fills, reported only, not a verdict basis: T7 Contrasts"


def test_harsh_t10_pools_the_inner_window_too(tmp_path, inner_events):
    # Same hand derivation as test_t10_is_computable_with_inner_2024_and_outer_2025: the mid column is twice the primary P&L,
    # so every mid difference doubles and the signs and categories are the primary ones. Outer 2025 alone has no full 2024 year,
    # so only the pooled inner rows make them testable.
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, mid=True)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    harsh = [_cells(line) for line in _section_lines(text, HARSH_T10_TITLE) if line.startswith("| H")]
    assert harsh == [["H1", "+1", "clear"], ["H2", "-1", "reversed"], ["H3", "+1", "no_difference"]]
    assert not all(row[2] == "not_testable" for row in harsh)
    # The headline T7 is the outer window's mid contrast alone: mean 2 (the harsh T7 shows the primary 1).
    headline7 = next(_cells(line) for line in _section_lines(text, "### T7 Contrasts") if line.startswith("| H1 | H | as_is |"))
    assert headline7[3:6] == ["ok", "4", "2"]
    harsh7 = next(_cells(line) for line in _section_lines(text, HARSH_T7_TITLE) if line.startswith("| H1 | H | as_is |"))
    assert harsh7[3:6] == ["ok", "4", "1"]


def test_t10_still_refuses_when_a_full_year_is_missing(tmp_path, inner_events):
    out = build_inner(tmp_path, INNER_2023, outer_dates=OUTER_2025)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert [row[2] for row in _t10_verdicts(text)] == ["not_testable"] * 3
    assert "absent from the inner and outer decisions" in text


def test_inner_tables_print_separately_without_a_t5_or_in_sample_marks(tmp_path, inner_events):
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    for table in ("T2", "T4", "T6", "T7"):
        assert f"{INNER_PREFIX}{table}" in text
    assert "### T4" in text  # the headline outer heading is unchanged
    assert "### T5" not in text and "IN-SAMPLE" not in text  # Amendment 6: no re-tuned arm, no T5, nothing in-sample
    insample = [line for line in _section_lines(text, "### T10") if "retuned" in line]
    assert insample == []


def _model_rows(text, title, label):
    return [_cells(line) for line in _section_lines(text, title) if line.startswith(f"| {label} |")]


def test_inner_tables_carry_the_inner_windows_own_values(tmp_path, inner_events):
    # Oracle by hand. Outer (2025): H-sq pays 10, 20, 30, 40 per event; H-abs pays one dollar more per event.
    # Inner (pnl_scale=3, INNER_2024): as_is rows pay 3x, so H-sq takes 3*(10+20+30+40)=300 (per event 75);
    # H-abs takes 3*(11+21+31+41)=312 (per event 78). The H1 pair H difference is 3 per event inner, 1 outer.
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, pnl_scale=3.0)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    t2 = [_cells(line) for line in _section_lines(text, INNER_PREFIX + "T2") if line.startswith("| 20")]
    assert t2 == [["2023Q3", "1"], ["2024Q1", "1"], ["2024Q2", "1"], ["2024Q3", "1"]]  # no 2025 quarter
    t4 = _model_rows(text, INNER_PREFIX + "T4", "H-sq")
    assert t4 == [["H-sq", "4", "300.00", "75", "1"]]
    assert _model_rows(text, INNER_PREFIX + "T4", "H-abs") == [["H-abs", "4", "312.00", "78", "1"]]
    assert _model_rows(text, INNER_PREFIX + "T6", "H-sq") == [["H-sq", "G_PROD", "4", "0", "0", "75"]]
    assert _model_rows(text, INNER_PREFIX + "T6", "H-abs")[0] == ["H-abs", "G_PROD", "4", "0", "0", "78"]
    inner7 = next(_cells(line) for line in _section_lines(text, INNER_PREFIX + "T7") if line.startswith("| H1 | H | as_is |"))
    outer7 = next(_cells(line) for line in _section_lines(text, "### T7") if line.startswith("| H1 | H | as_is |"))
    assert inner7[3:6] == ["ok", "4", "3"]  # n 4, mean 3 * (+1) per event
    assert outer7[3:6] == ["ok", "4", "1"]  # the outer window is unscaled: mean +1
    assert _model_rows(text, "### T4", "H-sq") == [["H-sq", "4", "100.00", "25", "1"]]  # the outer 25 per event
    assert t4 != _model_rows(text, "### T4", "H-sq")


INNER_REFUSALS = [
    ("report_receipt_chain_mismatch", {"body_override": {"inputs": "g" * 64}}),
    ("report_receipt_chain_mismatch", {"body_override": {"scope": "full"}}),
    ("report_receipt_chain_mismatch", {"body_override": {"models": list(ROLES)[:-1]}}),
    ("report_receipt_stage", {"body_override": {"stage": "inner"}}),
    # Shape checks run before the chain comparison, so a null models is malformed, not a chain mismatch.
    ("report_receipt_malformed", {"body_override": {"models": None}}),
    ("report_decisions_mismatch", {"mutate": _corrupt_inner}),
]


@pytest.mark.parametrize("reason, kwargs", INNER_REFUSALS, ids=[reason for reason, _kwargs in INNER_REFUSALS])
def test_inner_receipt_refusals(tmp_path, inner_events, reason, kwargs):
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, **kwargs)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert caught.value.details["reason"] == reason
    assert not (out / "REPORT.md").exists()


def test_inner_decisions_for_a_model_the_outer_excluded_are_accepted_and_filtered(tmp_path, inner_events):
    # The inner stage scored N-log, but the outer run excluded it (a missing or partial 2025 refit): the inner receipt covers a
    # superset of the outer's models, and the report keeps only the models the outer ran.
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025,
                      outer_rows_fn=lambda rows: [r for r in rows if r["model_id"] != "m-nlog"])

    def drop_n_log(body):
        return {**body, "models": [m for m in body["models"] if m != "m-nlog"]}

    _rewrite(out / "receipts" / "outer.json", drop_n_log)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert not any("N-log" in line for line in _section_lines(text, INNER_PREFIX + "T4"))
    assert _model_rows(text, INNER_PREFIX + "T4", "H-sq")  # the models the outer ran are still shown
    inner_decisions = pd.read_parquet(out / "decisions_inner.parquet")
    assert "m-nlog" in set(inner_decisions["model_id"])  # the inner decisions still hold its rows


def test_excluded_models_rows_are_still_validated(tmp_path, inner_events):
    # The outer excluded m-nlog, but its inner rows are still checked before the filter: a NaN P&L on a taken row refuses.
    def nan_taken_n_log(rows):
        return [{**row, "pnl_dollars": float("nan")} if row["model_id"] == "m-nlog" and row["status"] == "taken" else row for row in rows]

    def drop_n_log(body):
        return {**body, "models": [m for m in body["models"] if m != "m-nlog"]}

    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, inner_rows_fn=nan_taken_n_log,
                      outer_rows_fn=lambda rows: [r for r in rows if r["model_id"] != "m-nlog"])
    _rewrite(out / "receipts" / "outer.json", drop_n_log)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert caught.value.details["reason"] == "report_nonfinite_pnl"
    assert not (out / "REPORT.md").exists()


def test_a_deleted_inner_receipt_refuses_instead_of_demoting_to_the_legacy_shape(tmp_path, inner_events):
    # decisions_inner.parquet still exists, so a missing inner_decisions.json is a broken chain, not an older run without inner decisions.
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025)
    (out / "receipts" / "inner_decisions.json").unlink()
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert caught.value.details["reason"] == "report_receipt_chain_mismatch"
    assert not (out / "REPORT.md").exists()


def test_a_changed_inner_ticker_refuses_the_report(tmp_path, inner_events):
    # E12 (2024-02-15) is inside the inner window only, so the outer binding still passes; the inner binding must refuse.
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025)
    changed = {**TICKERS, "E12": "ZZZ"}
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(changed), admitted=None)
    assert caught.value.details["reason"] == "report_inputs_mismatch"
    assert not (out / "REPORT.md").exists()


def test_missing_inputs_inner_refuses_when_inner_decisions_are_present(tmp_path, inner_events):
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025)
    (out / "receipts" / "inputs_inner.json").unlink()
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert caught.value.details["reason"] == "report_receipt_malformed"
    assert not (out / "REPORT.md").exists()


def test_a_swapped_inputs_inner_receipt_refuses(tmp_path, inner_events):
    # A different but valid inputs_inner.json (same plan binding, an extra field, so another content_hash) replaces the bound one.
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025)
    path = out / "receipts" / "inputs_inner.json"
    path.unlink()
    tuning.write_receipt(path, {"stage": "inputs", "plan": size_pnl_driver.inner_plan_binding(plan_frame(TICKERS)), "extra": "swapped"})
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert caught.value.details["reason"] == "report_receipt_chain_mismatch"
    assert not (out / "REPORT.md").exists()


def test_without_inner_decisions_the_report_is_unchanged_in_shape(tmp_path):
    out = build(tmp_path)
    summary = assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert summary["sections"] == 19  # same fixture as the happy path: 15 HEAD sections + the three G_PROD sections (main) + the Amendment 6 production-audit section
    assert "### Inner window" not in text


MISMATCHED_VALUATION = [
    ("endpoints_in_outer_only", {"outer_rows_fn": _with_endpoints}),  # inner without a0/a1, outer with them
    ("mid_in_inner_only", {"inner_rows_fn": _with_mid}),  # inner with mid, outer without
]


@pytest.mark.parametrize("kwargs", [kw for _name, kw in MISMATCHED_VALUATION], ids=[name for name, _kw in MISMATCHED_VALUATION])
def test_mismatched_optional_valuation_columns_across_windows_refuse(tmp_path, inner_events, kwargs):
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, **kwargs)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert caught.value.details["reason"] == "report_decisions_columns"
    assert not (out / "REPORT.md").exists()


def _only_a0(rows):
    """The endpoint rows with pnl_dollars_a1 dropped: alpha 0 without its pair."""
    return [{k: v for k, v in row.items() if k != "pnl_dollars_a1"} for row in _with_endpoints(rows)]


def _only_a1(rows):
    """The endpoint rows with pnl_dollars_a0 dropped: alpha 1 without its pair."""
    return [{k: v for k, v in row.items() if k != "pnl_dollars_a0"} for row in _with_endpoints(rows)]


@pytest.mark.parametrize("fixture", [_only_a0, _only_a1], ids=["a0_without_a1", "a1_without_a0"])
def test_a_lone_breakeven_endpoint_refuses_in_every_window(tmp_path, inner_events, fixture):
    # both windows carry the same lone endpoint, so only the single-frame pairing check (not the cross-window check) can refuse
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, outer_rows_fn=fixture, inner_rows_fn=fixture)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert caught.value.details["reason"] == "report_decisions_columns"
    assert not (out / "REPORT.md").exists()


@pytest.mark.parametrize("kwargs", [{}, {"mid": True, "outer_rows_fn": _with_endpoints, "inner_rows_fn": _with_endpoints}],
                         ids=["none_in_either_window", "all_three_in_both_windows"])
def test_matching_optional_valuation_columns_across_windows_do_not_refuse(tmp_path, inner_events, kwargs):
    out = build_inner(tmp_path, INNER_2024, outer_dates=OUTER_2025, **kwargs)
    assemble_mod.assemble(out, plan_frame(TICKERS), admitted=None)
    assert (out / "REPORT.md").exists()
