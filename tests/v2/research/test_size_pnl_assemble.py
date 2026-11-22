"""EXP-188 S9c: REPORT.md assembly over synthetic receipts and decisions (tmp_path only; no store, no data/)."""
import os

import numpy as np
import pandas as pd
import pytest

from experiments import size_pnl_assemble as assemble_mod
from experiments import size_pnl_contrast as contrast
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_tuning as tuning
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_pnl_straddle import RULES

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
EVENT_DATES = {"E01": "2024-02-15", "E02": "2024-05-15", "E03": "2025-02-15", "E04": "2025-05-15"}
TICKERS = {"E01": "AAA", "E02": "BBB", "E03": "CCC", "E04": "AAA"}
ROLES = {"m-hsq": "H-sq", "m-habs": "H-abs", "m-eraw": "E-raw", "m-elog": "E-log", "m-nraw": "N-raw", "m-nlog": "N-log"}
MEDIAN_ROLES = ("H-abs", "E-log", "N-log")  # each pays one dollar more per event than its mean-type partner
PAIRS_OK = {"H": "testable", "E": "testable", "N": "testable"}


def _roster(pairs):
    models = [{"model_id": mid, "roles": [role], "source": "v14", "candidate_id": f"c-{role}", "target_transform": "none", "kind": "v14"}
              for mid, role in ROLES.items()]
    return {"models": models, "not_testable": [], "pairs": pairs}


def _row(mid, book, arm, structure, eid, day, rung, pnl):
    return {"model_id": mid, "book": book, "arm_or_rule": arm, "structure": structure, "event_id": eid,
            "event_date": pd.Timestamp(day), "status": "taken", "reason": "taken", "rung": rung, "pnl_dollars": pnl}


def _rows(dates):
    rows = []
    for mid, role in ROLES.items():
        bump = 1.0 if role in MEDIAN_ROLES else 0.0
        for i, (eid, day) in enumerate(dates.items()):
            pnl = 10.0 * (i + 1) + bump  # the mean-type partner pays 10 * (i + 1)
            for arm in ("as_is", "retuned"):
                rows.append(_row(mid, "short_vol", arm, NAMES[0], eid, day, 2, pnl))
            for rule in RULES:
                rows.append(_row(mid, "straddle", rule, "straddle", eid, day, 1, pnl))
    return rows


def build(tmp_path, *, rows_fn=None, pairs=None, dates=None, scope=None):
    """A complete outer run directory: tuning and inner receipts, inputs_outer, outer.json and decisions.parquet."""
    out = tmp_path / "out"
    rdir = out / "receipts"
    hashes = {}
    for mid in ROLES:
        body = {"stage": "tuning", "model_id": mid, "alpha": 0.5, "grid": [1.0], "c": 1.0, "skipped": True,
                "skip_reason": "few_trades", "trades": 0, "totals": {"1.0": 0.0}}
        hashes[mid] = tuning.write_receipt(rdir / f"tuning_{mid}.json", body)["content_hash"]
    tuning.write_receipt(rdir / "inner.json", {"stage": "inner", "models": hashes, "excluded": {}})
    inputs = tuning.write_receipt(rdir / "inputs_outer.json", {
        "stage": "inputs", "roster": _roster(pairs or PAIRS_OK), "models": {mid: {"status": "ready"} for mid in ROLES}})
    rows = _rows(dates or EVENT_DATES)
    rows = rows_fn(rows) if rows_fn else rows
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    decisions.publish(out, [frame], {
        "stage": "outer", "models": list(ROLES), "scope": scope, "inputs": inputs["content_hash"], "tuning": hashes,
        "c": {mid: 1.0 for mid in ROLES}, "labels": "x", "common_dropped": 0, "scoreability_dropped": 0,
        "ladder_skipped": {}, "ladder": "x"})
    return out


def _rewrite(path, change):
    """Replace a receipt by ``change`` applied to its body (write_receipt never overwrites, so unlink first)."""
    body = {k: v for k, v in tuning.load_receipt(path).items() if k != "content_hash"}
    path.unlink()
    tuning.write_receipt(path, change(body))


def _corrupt(out):
    path = out / "decisions.parquet"
    path.write_bytes(b"CORRUPT" + path.read_bytes())


def _disk(mutate):
    def make(tmp_path):
        out = build(tmp_path)
        mutate(out)
        return out, TICKERS
    return make


CASES = [
    ("tuning_receipt_missing", _disk(lambda out: (out / "receipts" / "outer.json").unlink())),
    ("report_decisions_missing", _disk(lambda out: (out / "decisions.parquet").unlink())),
    ("report_decisions_mismatch", _disk(_corrupt)),
    ("report_receipt_stage", _disk(lambda out: _rewrite(out / "receipts" / "outer.json", lambda b: {**b, "stage": "inner"}))),
    ("report_receipt_chain_mismatch", _disk(lambda out: _rewrite(out / "receipts" / "outer.json", lambda b: {**b, "inputs": "0" * 64}))),
    ("report_ticker_missing", lambda tmp_path: (build(tmp_path), {k: v for k, v in TICKERS.items() if k != "E04"})),
    ("report_nonfinite_pnl", lambda tmp_path: (build(tmp_path, rows_fn=lambda rows: [{**rows[0], "pnl_dollars": float("nan")}, *rows[1:]]), TICKERS)),
    ("report_decisions_columns", lambda tmp_path: (build(tmp_path, rows_fn=lambda rows: [{k: v for k, v in r.items() if k != "rung"} for r in rows]), TICKERS)),
    ("report_decisions_empty", lambda tmp_path: (build(tmp_path, rows_fn=lambda rows: pd.DataFrame(rows).iloc[0:0]), TICKERS)),
]


def _sections(text):
    return [line for line in text.split("\n") if line.startswith("### ")]


def test_happy_path_writes_every_section_in_order(tmp_path):
    out = build(tmp_path)
    summary = assemble_mod.assemble(out, TICKERS)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert summary["report"] == str(out / "REPORT.md") and summary["sections"] == 11 and summary["rows"] > 0
    assert text.startswith("# EXP-188 REPORT")
    heads = ["### T1", "### T2", "### T3", "### T4", "### T5", "### T6", "### T7", "### T8", "### T9", "### T10"]
    positions = [next(i for i, line in enumerate(_sections(text)) if line.startswith(h)) for h in heads]
    assert positions == sorted(positions) and len(set(positions)) == len(heads)
    assert "not_testable: labels_not_in_decisions" in text
    assert "not_testable: native_pool_arm_not_produced" in text
    assert "| H1 | +1 |" in text
    assert "| H1 | H | as_is | ok | 4 | 1 |" in text  # the median-minus-mean P&L is exactly 1 per event
    assert "### T2 Population (events in E)" in text
    assert "not recorded in decisions.parquet" in text
    assert "dropped |" not in text
    assert not list(out.glob("*.tmp"))


def test_contrast_is_called_with_the_shared_statistics(tmp_path, monkeypatch):
    calls = []

    def recorder(d, quarters, tickers, **kwargs):
        calls.append((np.asarray(d, dtype=float).copy(), list(quarters), list(tickers)))
        return {"n_events": len(d), "mean": float(np.mean(d)), "quarter_signs": {"2024Q1": 1}, "year_signs": {2024: 1},
                "quarter_interval": (0.0, 1.0), "ticker_interval": (0.0, 1.0),
                "quarter_counts": {"positive": 1, "negative": 0, "zero": 0}}

    monkeypatch.setattr(contrast, "contrast", recorder)
    assemble_mod.assemble(build(tmp_path), TICKERS)
    assert len(calls) == 18  # three hypotheses, three pairs, two arms each
    d, quarters, tickers = calls[0]  # H1, pair H, as_is
    assert np.allclose(d, 1.0)
    assert quarters == ["2024Q1", "2024Q2", "2025Q1", "2025Q2"]
    assert tickers == ["AAA", "BBB", "CCC", "AAA"]


@pytest.mark.parametrize("reason, make", CASES, ids=[reason for reason, _make in CASES])
def test_refusal_writes_no_report(tmp_path, reason, make):
    out, tickers = make(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, tickers)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == reason
    assert not (out / "REPORT.md").exists()


def test_non_testable_pair_is_reported_and_the_report_is_written(tmp_path):
    out = build(tmp_path, pairs={"H": "not_testable", "E": "testable", "N": "testable"})
    assemble_mod.assemble(out, TICKERS)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | not_testable: pair_member_excluded |" in text
    assert "| H1 | H | retuned | not_testable: pair_member_excluded |" in text
    assert "| H1 | +1 | not_testable |" in text


def test_fewer_than_two_quarters_is_not_testable(tmp_path):
    one_quarter = {"E01": "2024-02-15", "E02": "2024-02-20", "E03": "2024-03-01", "E04": "2024-03-05"}
    out = build(tmp_path, dates=one_quarter)
    assemble_mod.assemble(out, TICKERS)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | not_testable: too_few_quarters_or_tickers |" in text


def test_single_ticker_across_quarters_is_not_testable(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, {event: "AAA" for event in EVENT_DATES})
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | not_testable: too_few_quarters_or_tickers |" in text
    assert "| H1 | H | retuned | not_testable: too_few_quarters_or_tickers |" in text


def _section_lines(text, title):
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("### ")), len(lines))
    return lines[start:end]


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def test_t8_exact_rung_shares_for_the_fixture(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, TICKERS)
    lines = _section_lines((out / "REPORT.md").read_text(encoding="utf-8"), "### T8")
    # Per model: short_vol rung 2 on 2 arms x N events; straddle rung 1 on len(RULES) x N events.
    n = len(EVENT_DATES)
    short, long = 2 * n, len(RULES) * n
    share_rung1, share_rung2 = f"{long / (short + long):.4g}", f"{short / (short + long):.4g}"
    assert (share_rung1, share_rung2) == ("0.5", "0.5")
    header = _cells(next(line for line in lines if line.startswith("| model_id")))
    assert header == ["model_id", "1", "2", "exit_unquoted_worst_case"]
    for label in ROLES.values():
        row = _cells(next(line for line in lines if line.startswith(f"| {label} |")))
        assert row == [label, share_rung1, share_rung2, "-"]


def test_shares_exclude_untaken_and_null_rung_rows():
    frame = pd.DataFrame({
        "model_id": ["m1", "m1", "m1", "m1", "m1", "m2"],
        "taken": [True, True, True, False, True, True],
        "rung": [1.0, 2.0, 2.0, 1.0, np.nan, 1.0],
    })
    assert assemble_mod._shares(frame) == {"m1": {"1": 1 / 3, "2": 2 / 3}, "m2": {"1": 1.0}}


def test_smoke_scope_is_labelled_in_the_run_section(tmp_path):
    scope = "f" * 64
    out = build(tmp_path, scope=scope)
    assemble_mod.assemble(out, TICKERS)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert f"SMOKE scope {scope}, not a full-population result" in text


def test_failed_replace_keeps_the_old_report_and_leaves_no_tmp(tmp_path, monkeypatch):
    out = build(tmp_path)
    (out / "REPORT.md").write_text("OLD", encoding="utf-8")

    def refused(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(os, "replace", refused)
    with pytest.raises(OSError):
        assemble_mod.assemble(out, TICKERS)
    assert (out / "REPORT.md").read_text(encoding="utf-8") == "OLD"
    assert not list(out.glob("*.tmp"))
