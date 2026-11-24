"""EXP-188 S9c: REPORT.md assembly over synthetic receipts and decisions (tmp_path only; no store, no data/)."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.v2.ops.errors import OpsError, fail
from experiments import size_pnl_assemble as assemble_mod
from experiments import size_pnl_contrast as contrast
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_driver
from experiments import size_pnl_ladder as ladder
from experiments import size_pnl_readers as readers
from experiments import size_pnl_receipts as tuning
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_pnl_straddle import GATE_RULE

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
EVENT_DATES = {"E01": "2024-02-15", "E02": "2024-05-15", "E03": "2025-02-15", "E04": "2025-05-15"}
TICKERS = {"E01": "AAA", "E02": "BBB", "E03": "CCC", "E04": "AAA"}
NO_E04 = {k: v for k, v in TICKERS.items() if k != "E04"}
SINGLE_TICKER = {event: "AAA" for event in EVENT_DATES}
IDENTITY = {"plan_id": "plan-1", "registration_id": "reg-1", "seal_id": "seal-1", "panel_id": "panel-1"}
SPEC_HASH, SNAPSHOT = "a" * 64, "snap-1"  # every full-run call must match the fixture's inputs_outer receipt
ROLES = {"m-hsq": "H-sq", "m-habs": "H-abs", "m-eraw": "E-raw", "m-elog": "E-log", "m-nraw": "N-raw", "m-nlog": "N-log"}
MEDIAN_ROLES = ("H-abs", "E-log", "N-log")  # each pays one dollar more per event than its mean-type partner
PAIRS_OK = {"H": "testable", "E": "testable", "N": "testable"}
REAL_CHECK_PLAN = ladder.check_plan  # captured before any fixture patches it


@pytest.fixture(autouse=True)
def _no_plan_admission(monkeypatch):
    """Most tests exercise the report, not plan admission: check_plan is a no-op here, and the dedicated tests re-patch it."""
    monkeypatch.setattr(ladder, "check_plan", lambda admitted, plan: None)


def plan_frame(tickers=TICKERS):
    """The plan frame for ``tickers``: one row per event id, dated by the fixture's event dates."""
    days = pd.to_datetime([EVENT_DATES[e] for e in tickers])
    return pd.DataFrame({"event_id": list(tickers), "ticker": list(tickers.values()), "event_date": days,
                         "entry_date": days - pd.Timedelta(days=1), "exit_date": days + pd.Timedelta(days=1), "session": "AMC"})


def admitted_for(dates=EVENT_DATES):
    """The admitted events for ``dates``, as the real check_plan reads them (an Admitted mapping with an events frame)."""
    return readers.Admitted({"events": pd.DataFrame({"event_id": list(dates), "event_date": pd.to_datetime(list(dates.values()))}),
                             "excluded": {}})


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
            rows.append(_row(mid, "short_vol", "as_is", NAMES[0], eid, day, 2, pnl))  # short vol is as-is only (Amendment 6)
            rows.append(_row(mid, "straddle", GATE_RULE, "straddle", eid, day, 1, pnl))  # the straddle book is G_PROD only (Amendment 6)
    return rows


def build(tmp_path, *, rows_fn=None, pairs=None, dates=None, scope=None, plan_tickers=TICKERS):
    """A complete outer run directory: inner receipt, inputs_outer, outer.json and decisions.parquet.
    inputs_outer records the real plan binding of ``plan_frame(plan_tickers)``."""
    out = tmp_path / "out"
    rdir = out / "receipts"
    tuning.write_receipt(rdir / "inner.json", {"stage": "inner", "models": list(ROLES), "excluded": {}})
    inputs = tuning.write_receipt(rdir / "inputs_outer.json", {
        "stage": "inputs", "roster": _roster(pairs or PAIRS_OK), "models": {mid: {"status": "ready"} for mid in ROLES},
        "identity": IDENTITY, "spec_hash": SPEC_HASH, "snapshot": SNAPSHOT,
        "plan": size_pnl_driver.plan_binding(plan_frame(plan_tickers))})
    rows = _rows(dates or EVENT_DATES)
    rows = rows_fn(rows) if rows_fn else rows
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    decisions.publish(out, [frame], {
        "stage": "outer", "models": list(ROLES), "scope": scope, "inputs": inputs["content_hash"],
        "labels": "x", "common_dropped": 0, "scoreability_dropped": 0,
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
        return out, plan_frame()
    return make


CASES = [
    ("tuning_receipt_missing", _disk(lambda out: (out / "receipts" / "outer.json").unlink())),
    ("report_decisions_missing", _disk(lambda out: (out / "decisions.parquet").unlink())),
    ("report_decisions_mismatch", _disk(_corrupt)),
    ("report_receipt_stage", _disk(lambda out: _rewrite(out / "receipts" / "outer.json", lambda b: {**b, "stage": "inner"}))),
    ("report_receipt_chain_mismatch", _disk(lambda out: _rewrite(out / "receipts" / "outer.json", lambda b: {**b, "inputs": "0" * 64}))),
    # The plan binds the same missing-E04 frame, so the refusal comes from the decisions-vs-ticker check.
    ("report_ticker_missing", lambda tmp_path: (build(tmp_path, plan_tickers=NO_E04), plan_frame(NO_E04))),
    ("report_nonfinite_pnl", lambda tmp_path: (build(tmp_path, rows_fn=lambda rows: [{**rows[0], "pnl_dollars": float("nan")}, *rows[1:]]), plan_frame())),
    ("report_decisions_columns", lambda tmp_path: (build(tmp_path, rows_fn=lambda rows: [{k: v for k, v in r.items() if k != "rung"} for r in rows]), plan_frame())),
    ("report_decisions_empty", lambda tmp_path: (build(tmp_path, rows_fn=lambda rows: pd.DataFrame(rows).iloc[0:0]), plan_frame())),
    ("report_decisions_invalid", lambda tmp_path: (build(tmp_path, rows_fn=lambda rows: [{**rows[0], "book": "bogus"}, *rows[1:]]), plan_frame())),
]


# Each plan that must refuse as report_tickers_invalid (ids and tickers are checked before any file is read).
# check_plan is a no-op in this file, so these reach the report's own ticker check.
BAD_PLANS = {
    "not_a_dataframe": lambda: TICKERS,
    "missing_ticker_column": lambda: plan_frame().drop(columns=["ticker"]),
    "missing_event_id_column": lambda: plan_frame().drop(columns=["event_id"]),
    "blank_event_id": lambda: plan_frame().assign(event_id=[" ", "E02", "E03", "E04"]),
    "duplicate_event_id": lambda: pd.concat([plan_frame(), plan_frame().iloc[:1]]),
    "blank_ticker": lambda: plan_frame().assign(ticker=[" ", "BBB", "CCC", "AAA"]),
    "none_ticker": lambda: plan_frame().assign(ticker=[None, "BBB", "CCC", "AAA"]),
    "numeric_ticker": lambda: plan_frame().assign(ticker=[1, "BBB", "CCC", "AAA"]),
}


def _sections(text):
    return [line for line in text.split("\n") if line.startswith("### ")]


def test_happy_path_writes_every_section_in_order(tmp_path):
    out = build(tmp_path)
    summary = assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert summary["report"] == str(out / "REPORT.md") and summary["sections"] == 16 and summary["rows"] > 0  # 10 + two not_testable sensitivity notes (Amendment 1 and 2) + the Amendment 6 board-match section + the three G_PROD sections
    assert text.startswith("# EXP-188 REPORT")
    heads = ["### T1", "### T2", "### T3", "### T4", "### T6", "### T7", "### T8", "### T9", "### T10"]
    positions = [next(i for i, line in enumerate(_sections(text)) if line.startswith(h)) for h in heads]
    assert positions == sorted(positions) and len(set(positions)) == len(heads)
    assert "not_testable: labels_not_in_decisions" in text
    assert "not_testable: native_pool_arm_not_produced" in text
    assert "| H1 | +1 | clear |" in text and "| H2 | -1 | reversed |" in text and "| H3 | +1 | no_difference |" in text
    assert "| H3 | E | as_is | ok | 4 | 0 | 0/0/4 | 0 | 2024:0 2025:0 | [0, 0] | [0, 0] |" in text  # d = 0 per event
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
    assemble_mod.assemble(build(tmp_path), plan_frame(), admitted=None)
    assert len(calls) == 12  # three pairs: H1, H2 (G_PROD) and H3 once each (9), then the G_PROD sections' H2 contrasts (3)
    d, quarters, tickers = calls[0]  # H1, pair H, as_is
    assert np.allclose(d, 1.0)
    assert quarters == ["2024Q1", "2024Q2", "2025Q1", "2025Q2"]
    assert tickers == ["AAA", "BBB", "CCC", "AAA"]


@pytest.mark.parametrize("reason, make", CASES, ids=[reason for reason, _make in CASES])
def test_refusal_writes_no_report(tmp_path, reason, make):
    out, plan = make(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan, admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == reason
    assert not (out / "REPORT.md").exists()


def test_non_testable_pair_is_reported_and_the_report_is_written(tmp_path):
    out = build(tmp_path, pairs={"H": "not_testable", "E": "testable", "N": "testable"})
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | not_testable: pair_member_excluded |" in text
    assert "| H1 | +1 | not_testable |" in text


def test_fewer_than_two_quarters_is_not_testable(tmp_path):
    one_quarter = {"E01": "2024-02-15", "E02": "2024-02-20", "E03": "2024-03-01", "E04": "2024-03-05"}
    out = build(tmp_path, dates=one_quarter)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | not_testable: too_few_quarters_or_tickers |" in text


def test_single_ticker_across_quarters_is_not_testable(tmp_path):
    out = build(tmp_path, plan_tickers=SINGLE_TICKER)
    assemble_mod.assemble(out, plan_frame(SINGLE_TICKER), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | not_testable: too_few_quarters_or_tickers |" in text


def _section_lines(text, title):
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("### ")), len(lines))
    return lines[start:end]


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def test_t8_exact_rung_shares_for_the_fixture(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    lines = _section_lines((out / "REPORT.md").read_text(encoding="utf-8"), "### T8")
    # Per model: short_vol rung 2 on 1 arm (as_is) x N events; straddle rung 1 on the one G_PROD rule x N events.
    n = len(EVENT_DATES)
    short, long = 1 * n, 1 * n
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


def test_smoke_scope_refuses_as_unsupported_and_writes_no_report(tmp_path):
    out = build(tmp_path, scope="abc")
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_scope_unsupported"
    assert not (out / "REPORT.md").exists()


def test_failed_replace_keeps_the_old_report_and_leaves_no_tmp(tmp_path, monkeypatch):
    out = build(tmp_path)
    (out / "REPORT.md").write_text("OLD", encoding="utf-8")

    def refused(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(os, "replace", refused)
    with pytest.raises(OSError):
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert (out / "REPORT.md").read_text(encoding="utf-8") == "OLD"
    assert not list(out.glob("*.tmp"))


def test_t10_verdicts_are_the_pinned_literals(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    verdicts = [_cells(line) for line in _section_lines(text, "### T10") if line.startswith("| H")]
    assert verdicts == [["H1", "+1", "clear"], ["H2", "-1", "reversed"], ["H3", "+1", "no_difference"]]
    assert "not_testable because the required full year" not in text


def test_t10_section_discloses_amendment_5(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Amendment 5 (2026-11-23): the straddle universe" in "\n".join(_section_lines(text, "### T10"))
    assert "Amendment 6 (2026-11-24, post-hoc revision by the user): the straddle rule is production's STR-THRU gate G_PROD only" in "\n".join(_section_lines(text, "### T10"))


def test_same_plan_frame_as_the_receipt_succeeds(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert (out / "REPORT.md").exists()


def test_changed_ticker_in_the_plan_refuses_a_full_run(tmp_path):
    out = build(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame({**TICKERS, "E03": "ZZZ"}), admitted=None)  # same ids, one ticker changed
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_inputs_mismatch"
    assert not (out / "REPORT.md").exists()


@pytest.mark.parametrize("name", list(BAD_PLANS), ids=list(BAD_PLANS))
def test_malformed_plan_refuses_as_tickers_invalid_before_any_read(tmp_path, name):
    # tmp_path holds no receipts: the refusal must come before any file is read, not from a missing receipt.
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(tmp_path / "missing", BAD_PLANS[name](), admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_tickers_invalid"


def test_plan_without_event_date_refuses_as_tickers_invalid(tmp_path):
    out = build(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame().drop(columns=["event_date"]), admitted=None)  # the binding cannot be computed
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_tickers_invalid"
    assert not (out / "REPORT.md").exists()


def test_invalid_stored_gate_refuses_before_any_read(tmp_path):
    out = tmp_path / "x"  # no receipts: the refusal must come before any file is read
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None, stored_gate={"scored": ["E1"], "selected": ["E2"]})
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_stored_gate_invalid"
    assert not (out / "REPORT.md").exists()


@pytest.mark.parametrize("accepted_gap", ["", "  ", 1, ["named gap"]], ids=["empty", "blank", "int", "list"])
def test_invalid_accepted_gap_refuses_before_any_read(tmp_path, accepted_gap):
    out = tmp_path / "x"  # no receipts and no G_PROD rows: the check runs on every path, before any file is read
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None, accepted_gap=accepted_gap)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_accepted_gap_invalid"
    assert not (out / "REPORT.md").exists()


def test_unknown_decision_status_refuses_as_decisions_invalid(tmp_path):
    out = build(tmp_path, rows_fn=lambda rows: [{**rows[0], "status": "weird"}, *rows[1:]])
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_decisions_invalid"
    assert not (out / "REPORT.md").exists()


def test_2025_only_decisions_are_not_testable_and_t7_still_shows_stats(tmp_path):
    dates = {"E01": "2025-02-15", "E02": "2025-05-15", "E03": "2025-08-15", "E04": "2025-11-15"}
    out = build(tmp_path, dates=dates)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    verdicts = [_cells(line) for line in _section_lines(text, "### T10") if line.startswith("| H")]
    assert verdicts == [["H1", "+1", "not_testable"], ["H2", "-1", "not_testable"], ["H3", "+1", "not_testable"]]
    assert ("not_testable because the required full year(s) [2024] are absent from the outer decisions; "
            "the inner window publishes no decisions") in text
    assert "| H1 | H | as_is | ok | 4 | 1 |" in text


def test_requested_identity_matches_the_receipts(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, plan_frame(), admitted=None, expected=IDENTITY, spec_hash=SPEC_HASH, snapshot=SNAPSHOT)
    assert (out / "REPORT.md").exists()


@pytest.mark.parametrize("change", [{"expected": {**IDENTITY, "plan_id": "other"}}, {"spec_hash": "b" * 64}, {"snapshot": "snap-2"}])
def test_requested_identity_mismatch_refuses(tmp_path, change):
    out = build(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None,
                              **{"expected": IDENTITY, "spec_hash": SPEC_HASH, "snapshot": SNAPSHOT, **change})
    assert caught.value.details["reason"] == "report_inputs_mismatch"
    assert not (out / "REPORT.md").exists()


def test_outer_receipt_without_rows_is_malformed(tmp_path):
    out = build(tmp_path)
    _rewrite(out / "receipts" / "outer.json", lambda body: {k: v for k, v in body.items() if k != "rows"})
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_receipt_malformed"
    assert not (out / "REPORT.md").exists()


def test_non_date_event_date_refuses_as_decisions_invalid(tmp_path):
    out = build(tmp_path, rows_fn=lambda rows: [{**row, "event_date": "not a date"} for row in rows])
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_decisions_invalid"


def test_decisions_one_byte_over_the_cap_refuses(tmp_path, monkeypatch):
    out = build(tmp_path)
    monkeypatch.setattr(decisions, "MAX_DECISIONS_BYTES", (out / "decisions.parquet").stat().st_size - 1)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_decisions_mismatch"


def _no_decode(*_args, **_kwargs):
    raise AssertionError("decisions.parquet was decoded before the metadata checks")


def test_declared_decoded_size_over_the_cap_refuses_before_any_decode(tmp_path, monkeypatch):
    out = build(tmp_path)
    monkeypatch.setattr(assemble_mod, "MAX_DECODED_BYTES", 1)
    monkeypatch.setattr(assemble_mod.pq.ParquetFile, "read", _no_decode)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_decisions_invalid"
    assert not (out / "REPORT.md").exists()


def test_outer_rows_differing_from_parquet_rows_refuses_before_any_decode(tmp_path, monkeypatch):
    out = build(tmp_path)
    _rewrite(out / "receipts" / "outer.json", lambda b: {**b, "rows": b["rows"] + 1})  # decisions_sha256 still matches
    monkeypatch.setattr(assemble_mod.pq.ParquetFile, "read", _no_decode)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_decisions_mismatch"
    assert not (out / "REPORT.md").exists()


def test_hash_loop_refuses_when_fstat_understates_the_size(tmp_path, monkeypatch):
    out = build(tmp_path)
    target, real_fstat = os.stat(out / "decisions.parquet"), os.fstat

    def understated(fd):  # the decisions descriptor reports size 0 and keeps its regular-file mode
        st = real_fstat(fd)
        return SimpleNamespace(st_mode=st.st_mode, st_size=0) if os.path.samestat(st, target) else st

    monkeypatch.setattr(os, "fstat", understated)
    monkeypatch.setattr(decisions, "MAX_DECISIONS_BYTES", 16)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_decisions_mismatch"


def test_decisions_bytes_consumed_are_the_bytes_hashed(tmp_path, monkeypatch):
    out = build(tmp_path)
    other = tmp_path / "other.bin"
    other.write_bytes(b"different bytes, not the receipted decisions")
    real_parquet_file = assemble_mod.pq.ParquetFile
    swapped = []

    def swapping_parquet_file(handle, *args, **kwargs):
        # Runs after the hash check: replace the path on disk, then parse the handle opened before the swap.
        if not swapped:
            swapped.append(True)
            os.replace(other, out / "decisions.parquet")
        return real_parquet_file(handle, *args, **kwargs)

    monkeypatch.setattr(assemble_mod.pq, "ParquetFile", swapping_parquet_file)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert swapped == [True]
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | ok | 4 | 1 |" in text


def _e01_undetermined_for_m_hsq(rows):
    """E01 is undetermined for m-hsq's short-vol rows; every other model takes E01 at 1000 on the same rows."""
    out = []
    for row in rows:
        if row["event_id"] == "E01" and row["book"] == "short_vol":
            if row["model_id"] == "m-hsq":
                out.append({**row, "status": "undetermined", "pnl_dollars": float("nan")})
            else:
                out.append({**row, "pnl_dollars": 1000.0})
        else:
            out.append(row)
    return out


def test_event_undetermined_under_one_model_is_excluded_from_e(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    # The clean comparison is the fixture without E01 at all, so its pnl values for the other events are identical.
    clean = build(clean_dir, rows_fn=lambda rows: [row for row in rows if row["event_id"] != "E01"])
    assemble_mod.assemble(clean, plan_frame(), admitted=None)
    clean_text = (clean / "REPORT.md").read_text(encoding="utf-8")
    dirty_dir = tmp_path / "dirty"
    dirty_dir.mkdir()
    out = build(dirty_dir, rows_fn=_e01_undetermined_for_m_hsq)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | ok | 3 | 1 |" in text  # n is one smaller than the four-event fixture's 4
    assert "| H1 | H | as_is | ok | 4 | 1 |" not in text
    t7, clean_t7 = _section_lines(text, "### T7"), _section_lines(clean_text, "### T7")
    # Short-vol E and H3 (E intersect E_str, where E_str still holds E01) match the clean run: the same events and P&L.
    assert [line for line in t7 if line.startswith(("| H1 |", "| H3 |"))] == [line for line in clean_t7 if line.startswith(("| H1 |", "| H3 |"))]
    # H2 reads E_str, which keeps E01: its short-vol row removed E01 from E only (Amendment 5).
    assert _cells(next(line for line in t7 if line.startswith("| H2 | H | G_PROD |")))[4] == "4"
    assert "undetermined_events_excluded: 1" in text
    assert "undetermined_straddle_events_excluded: 0" in text
    assert ("short-vol undetermined rows exclude events from E: 1; straddle undetermined rows (G_PROD over E_str: H2, H3, T6, "
            "straddle T7) exclude events from E_str: 0") in text
    assert "short-vol non-trades (e.g. width_declined) stay in E at P&L 0" in text


def _e01_index(rows, model_id):
    """Index of E01's as_is short-vol row for ``model_id``."""
    return next(i for i, row in enumerate(rows) if (row["event_id"], row["model_id"], row["book"], row["arm_or_rule"])
                == ("E01", model_id, "short_vol", "as_is"))


def test_taken_nan_pnl_on_an_event_with_an_undetermined_row_refuses(tmp_path):
    def make(rows):
        out = _e01_undetermined_for_m_hsq(rows)
        i = _e01_index(out, "m-eraw")
        out[i] = {**out[i], "pnl_dollars": float("nan")}
        return out

    out = build(tmp_path, rows_fn=make)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_nonfinite_pnl"
    assert not (out / "REPORT.md").exists()


def test_row_for_an_event_missing_from_the_plan_refuses_even_with_an_undetermined_row(tmp_path):
    no_e01 = {k: v for k, v in TICKERS.items() if k != "E01"}
    out = build(tmp_path, rows_fn=_e01_undetermined_for_m_hsq, plan_tickers=no_e01)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(no_e01), admitted=None)
    assert caught.value.details["reason"] == "report_ticker_missing"
    assert not (out / "REPORT.md").exists()


def test_unknown_model_on_an_event_with_an_undetermined_row_refuses(tmp_path):
    def make(rows):
        out = _e01_undetermined_for_m_hsq(rows)
        return [*out, {**out[_e01_index(out, "m-eraw")], "model_id": "m-zzz"}]

    out = build(tmp_path, rows_fn=make)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_receipt_chain_mismatch"
    assert not (out / "REPORT.md").exists()


def test_all_undetermined_decisions_refuse_as_empty(tmp_path):
    out = build(tmp_path, rows_fn=lambda rows: [{**row, "status": "undetermined", "pnl_dollars": float("nan")} for row in rows])
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_decisions_empty"
    assert not (out / "REPORT.md").exists()


def test_load_decisions_keeps_a_short_vol_non_trade_in_e_and_drops_a_burn_in_row(tmp_path):
    """The optional reason column must reach population(): a short-vol width_declined row keeps its event in E,
    while a short-vol thin_history row (burn-in, not a non-trade) drops its event."""
    def make(rows):
        out = []
        for row in rows:
            if row["book"] == "short_vol" and row["model_id"] == "m-hsq" and row["arm_or_rule"] == "as_is":
                if row["event_id"] == "E01":
                    row = {**row, "status": "undetermined", "reason": "width_declined", "pnl_dollars": float("nan")}
                elif row["event_id"] == "E02":
                    row = {**row, "status": "undetermined", "reason": "thin_history", "pnl_dollars": float("nan")}
            out.append(row)
        return out

    out = build(tmp_path, rows_fn=make)
    outer = tuning.load_receipt(out / "receipts" / "outer.json")
    raw = assemble_mod._load_decisions(out, outer)
    assert "reason" in raw.columns
    in_e = set(assemble_mod.synthetic_eval.population(raw)["event_id"].astype(str))
    assert "E01" in in_e
    assert "E02" not in in_e


# ---- out_dir, plan admission (check_plan now runs inside assemble) ----


def test_out_dir_that_is_not_a_path_refuses_as_out_dir_invalid(tmp_path):
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(None, plan_frame(), admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_out_dir_invalid"


@pytest.mark.parametrize("board_ledger", [123, "", "  "], ids=["not_a_path", "empty", "blank"])
def test_board_ledger_that_is_not_a_path_refuses_as_board_ledger_invalid(tmp_path, board_ledger):
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(tmp_path, plan_frame(), admitted=None, board_ledger=board_ledger)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_board_ledger_invalid"


@pytest.mark.parametrize("plan", [None, pd.DataFrame()], ids=["none", "empty_columns"])
def test_none_or_empty_column_plan_refuses_as_tickers_invalid_before_check_plan(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(ladder, "check_plan", REAL_CHECK_PLAN)  # the real admission must not see these frames
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(tmp_path / "missing", plan, admitted=admitted_for())
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_tickers_invalid"


def test_assemble_runs_check_plan_on_the_admitted_events_and_the_plan(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ladder, "check_plan", lambda admitted, plan: seen.append((admitted, plan)))
    sentinel = object()
    plan = plan_frame()
    out = build(tmp_path)
    assemble_mod.assemble(out, plan, admitted=sentinel)
    assert len(seen) == 1 and seen[0][0] is sentinel
    pd.testing.assert_frame_equal(seen[0][1], plan)
    assert (out / "REPORT.md").exists()


@pytest.mark.parametrize("make_exc", [
    lambda: fail(CONFLICT, "the plan has a session outside AMC/BMO", details={"reason": "plan_session_invalid"}),
    lambda: Refusal(CONFLICT, "the plan is malformed", {"reason": "plan_malformed"}),
], ids=["ops_error", "refusal"])
def test_check_plan_typed_refusal_propagates_and_writes_no_report(tmp_path, monkeypatch, make_exc):
    def refuse(admitted, plan):
        raise make_exc()

    monkeypatch.setattr(ladder, "check_plan", refuse)
    out = build(tmp_path)
    with pytest.raises((OpsError, Refusal)) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    reason = caught.value.problem.details["reason"] if isinstance(caught.value, OpsError) else caught.value.details["reason"]
    assert reason in {"plan_session_invalid", "plan_malformed"}
    assert not (out / "REPORT.md").exists()


@pytest.mark.parametrize("exc_type", [KeyError, TypeError, AttributeError], ids=["key", "type", "attribute"])
def test_check_plan_raw_errors_become_tickers_invalid(tmp_path, monkeypatch, exc_type):
    def broken(admitted, plan):
        raise exc_type("event_date")

    monkeypatch.setattr(ladder, "check_plan", broken)
    out = build(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_tickers_invalid"
    assert not (out / "REPORT.md").exists()


def test_real_check_plan_refuses_an_invalid_session_and_writes_no_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "check_plan", REAL_CHECK_PLAN)
    out = build(tmp_path)
    plan = plan_frame().assign(session=["XYZ", "AMC", "AMC", "AMC"])
    with pytest.raises(OpsError) as caught:
        assemble_mod.assemble(out, plan, admitted=admitted_for())
    assert caught.value.problem.details["reason"] == "plan_session_invalid"
    assert not (out / "REPORT.md").exists()


def test_real_check_plan_refuses_a_null_entry_date_and_writes_no_report(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "check_plan", REAL_CHECK_PLAN)
    out = build(tmp_path)
    plan = plan_frame().assign(entry_date=[pd.NaT, *plan_frame()["entry_date"].iloc[1:]])
    with pytest.raises(OpsError) as caught:
        assemble_mod.assemble(out, plan, admitted=admitted_for())
    assert caught.value.problem.details["reason"] == "plan_malformed"
    assert not (out / "REPORT.md").exists()


def test_real_check_plan_admits_the_fixture_plan_and_the_report_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ladder, "check_plan", REAL_CHECK_PLAN)
    out = build(tmp_path)
    assemble_mod.assemble(out, plan_frame(), admitted=admitted_for())
    assert (out / "REPORT.md").exists()


# ---- Amendment 1 sensitivity: pnl_dollars_mid, after the primary verdicts ----

def _with_mid(rows, *, delta=0.0):
    """The rows with a ``pnl_dollars_mid`` column: each primary P&L plus ``delta``."""
    return [{**row, "pnl_dollars_mid": row["pnl_dollars"] + delta} for row in rows]


def test_mid_column_adds_the_sensitivity_sections_after_the_primary_verdicts(tmp_path):
    out = build(tmp_path, rows_fn=_with_mid)
    summary = assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    sens = [line for line in _sections(text) if "not the pre-registered verdict" in line]
    assert summary["sections"] == 19  # the 10 primary sections, 4 sensitivity sections, the Amendment 2 not_testable note, the Amendment 6 board-match section and the three G_PROD sections
    assert [line.rsplit(": ", 1)[1] for line in sens] == ["T4 Short-vol, as is", "T6 Straddle buy rules",
                                                          "T7 Contrasts", "T10 Verdicts"]
    assert all(line.startswith("### Sensitivity (Amendment 1), not the pre-registered verdict: ") for line in sens)
    assert text.index(sens[0]) > text.index("### T10 Verdicts")
    mid_verdicts = [_cells(line) for line in _section_lines(text, sens[-1]) if line.startswith("| H")]
    assert mid_verdicts == [["H1", "+1", "clear"], ["H2", "-1", "reversed"], ["H3", "+1", "no_difference"]]


def test_primary_sections_do_not_depend_on_the_mid_column(tmp_path):
    texts = {}
    for name, rows_fn in (("plain", None), ("mid", _with_mid), ("shifted", lambda rows: _with_mid(rows, delta=500.0))):
        (tmp_path / name).mkdir()
        out = build(tmp_path / name, rows_fn=rows_fn)
        assemble_mod.assemble(out, plan_frame(), admitted=None)
        texts[name] = (out / "REPORT.md").read_text(encoding="utf-8")
    for title in ("### T4", "### T10"):
        assert _section_lines(texts["plain"], title) == _section_lines(texts["mid"], title) == _section_lines(texts["shifted"], title)
    sens = "### Sensitivity (Amendment 1), not the pre-registered verdict: T4 Short-vol, as is"
    assert _section_lines(texts["mid"], sens) != _section_lines(texts["shifted"], sens)  # the sensitivity reads the mid marks
    # Rendered values. H-sq (no median role, so no bump) takes four as-is trades of 10, 20, 30, 40: total 100.00 over
    # 4 events = 25 per event, hit rate 1. The shifted run adds 500 to every taken row, so the sensitivity total is
    # 100 + 4 * 500 = 2100.00 and the per-event value is 2100 / 4 = 525.
    plain_row = next(line for line in _section_lines(texts["plain"], "### T4") if line.startswith("| H-sq |"))
    assert plain_row == "| H-sq | 4 | 100.00 | 25 | 1 |"
    shifted_row = next(line for line in _section_lines(texts["shifted"], sens) if line.startswith("| H-sq |"))
    assert shifted_row == "| H-sq | 4 | 2100.00 | 525 | 1 |"
    # No verdict flip is asserted here. A uniform per-trade shift cancels in a paired contrast only when both models of the
    # pair take the same number of trades per event (the shift then enters both legs equally and subtracts out). This
    # fixture has equal counts (every model takes every row), so the uniform shift leaves the sensitivity verdicts unchanged;
    # the test after this one uses a non-uniform mid column that does change them.


SENS_T7 = "### Sensitivity (Amendment 1), not the pre-registered verdict: T7 Contrasts"
SENS_T10 = "### Sensitivity (Amendment 1), not the pre-registered verdict: T10 Verdicts"


def _negate_median_mid(rows):
    """Non-uniform marks: the median-type model m-habs (H-abs) carries its negated P&L as the mid mark; every other model keeps its P&L."""
    return [{**row, "pnl_dollars_mid": -row["pnl_dollars"] if row["model_id"] == "m-habs" else row["pnl_dollars"]} for row in rows]


def test_non_uniform_mid_changes_the_sensitivity_verdicts_and_not_the_primary(tmp_path):
    texts = {}
    for name, rows_fn in (("plain", None), ("mid", _negate_median_mid)):
        (tmp_path / name).mkdir()
        out = build(tmp_path / name, rows_fn=rows_fn)
        assemble_mod.assemble(out, plan_frame(), admitted=None)
        texts[name] = (out / "REPORT.md").read_text(encoding="utf-8")
    plain, mid = texts["plain"], texts["mid"]
    # Primary T10 is byte-for-byte the plain run's, and still the pinned literals.
    assert _section_lines(mid, "### T10 Verdicts") == _section_lines(plain, "### T10 Verdicts")
    primary = [_cells(line) for line in _section_lines(mid, "### T10 Verdicts") if line.startswith("| H")]
    assert primary == [["H1", "+1", "clear"], ["H2", "-1", "reversed"], ["H3", "+1", "no_difference"]]
    # Pair H's mid contrast per event is -(10k + 1) - 10k = -21, -41, -61, -81 on the short-vol and straddle legs alike.
    # H1 (short-vol, +1): pair H is clearly opposite, which blocks clear and directional -> no_difference.
    # H2 (straddle, -1): pairs E and N are clearly opposite to -1, pair H is clearly opposite to +1 -> no_difference.
    # H3 (short-vol minus G_PROD): the two legs cancel per event for every pair, so d = 0 -> no_difference.
    sens = [_cells(line) for line in _section_lines(mid, SENS_T10) if line.startswith("| H")]
    assert sens == [["H1", "+1", "no_difference"], ["H2", "-1", "no_difference"], ["H3", "+1", "no_difference"]]
    # The sensitivity T7 differs from the primary T7. Pair H, H1, as_is: d = -21, -41, -61, -81, so mean -51, quarters 0/4/0.
    assert _section_lines(mid, SENS_T7) != _section_lines(mid, "### T7 Contrasts")
    sens_h1 = _cells(next(line for line in _section_lines(mid, SENS_T7) if line.startswith("| H1 | H | as_is |")))
    assert sens_h1[:4] == ["H1", "H", "as_is", "ok"] and sens_h1[4] == "4"
    assert float(sens_h1[5]) == -51 and sens_h1[6] == "0/4/0" and float(sens_h1[7]) == 0 and sens_h1[8] == "2024:- 2025:-"
    primary_h1 = _cells(next(line for line in _section_lines(mid, "### T7 Contrasts") if line.startswith("| H1 | H | as_is |")))
    assert primary_h1[:4] == ["H1", "H", "as_is", "ok"] and primary_h1[4] == "4"
    assert float(primary_h1[5]) == 1 and primary_h1[6] == "4/0/0" and primary_h1[8] == "2024:+ 2025:+"


def test_decisions_without_the_mid_column_get_a_not_testable_sensitivity_note(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "### Sensitivity (Amendment 1)\n\nnot_testable: pnl_dollars_mid_not_in_decisions" in text
    assert "not the pre-registered verdict" not in text


def test_non_finite_mid_pnl_on_a_taken_row_refuses(tmp_path):
    def make(rows):
        mid = _with_mid(rows)
        return [{**mid[0], "pnl_dollars_mid": float("nan")}, *mid[1:]]

    out = build(tmp_path, rows_fn=make)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.details["reason"] == "report_nonfinite_pnl"
    assert not (out / "REPORT.md").exists()


def test_non_numeric_mid_pnl_on_a_non_taken_row_refuses_as_decisions_invalid(tmp_path):
    def make(rows):
        # Every mid value is text, so the column is still writable; one value on a non-taken row is not a number.
        out = [{**row, "pnl_dollars_mid": str(row["pnl_dollars"])} for row in rows]
        out[0] = {**out[0], "status": "not_taken", "reason": "width_declined", "pnl_dollars": 0.0, "pnl_dollars_mid": "n/a"}
        return out

    out = build(tmp_path, rows_fn=make)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_decisions_invalid"
    assert not (out / "REPORT.md").exists()


# ---- Amendment 2 account: the taken rows of every event, not only E ----


def _with_account(rows):
    """Every row carries the account columns: entry the day before its event, exit the day after, debit 0.5, exit 0.6,
    and 9,500 secured per contract (so each short-vol trade funds one contract on a $100,000 account)."""
    out = []
    for row in rows:
        day = pd.Timestamp(EVENT_DATES[row["event_id"]])
        out.append({**row, "entry_date": day - pd.Timedelta(days=1), "exit_date": day + pd.Timedelta(days=1),
                    "entry_cost": 0.5, "exit_value": 0.6, "secured_per_contract": 9500.0})
    return out


def test_account_wanted_counts_a_taken_row_of_an_event_removed_from_e(tmp_path):
    # E01 is undetermined for m-hsq, so it leaves short-vol E for every model; m-eraw still takes E01 as_is, and that row must be counted.
    out = build(tmp_path, rows_fn=lambda rows: _with_account(_e01_undetermined_for_m_hsq(rows)))
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | ok | 3 | 1 |" in text  # the T7 population is still three events
    account = _section_lines(text, "### Sensitivity (Amendment 2)")
    wanted = _cells(next(line for line in account if line.startswith("| book | E-raw | as_is |")))[3]
    assert wanted == "4"  # four taken short-vol as_is rows for E-raw: E01 included, not the three events of E


def _straddle_only(rows):
    """A straddle-only run: every short-vol row is a non-trade at P&L 0 with NaN secured cash; the taken straddle rows carry 0.0 secured."""
    out = []
    for row in _with_account(rows):
        if row["book"] == "short_vol":
            row = {**row, "status": "not_taken", "reason": "below_cutoff", "pnl_dollars": 0.0, "secured_per_contract": np.nan}
        else:
            row = {**row, "secured_per_contract": 0.0}
        out.append(row)
    return out


def test_straddle_only_run_is_testable_for_the_account(tmp_path):
    out = build(tmp_path, rows_fn=_straddle_only)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "not_testable: account_columns_not_in_decisions" not in text  # the Amendment 2 table is produced, not the note
    account = _section_lines(text, "### Sensitivity (Amendment 2)")
    assert account[0].startswith("### Sensitivity (Amendment 2), not the pre-registered verdict")
    # Hand-derived from this fixture: debit 0.5 restated to secured 50 (debit x 100); four non-overlapping events, each
    # funded at floor(0.25 * 0.66 * equity / 50) contracts: 330, 340, 352, 363 as equity goes 100000, 103300, 106700,
    # 110220. Each contract nets (0.6 - 0.5) x 100 = 10, so final = 100000 + 10 * 1385 = 113850; drawdown 0 (equity only
    # rises); one position open at a time. The first straddle row in sorted order is E-log (m-elog) with rule G_PROD.
    assert "| straddle | E-log | G_PROD | 4 | 4 | 113850.00 | 0.1385 | 0 | 1 |" in account


def test_empty_taken_frame_with_the_column_reports_zero_trade_accounts(tmp_path):
    # One model, every row not_taken: the taken frame is empty but keeps secured_per_contract, so the account table
    # (zero-trade accounts at $100,000) is produced and the not_testable note is not.
    rows = [_row("m-hsq", "short_vol", "as_is", NAMES[0], "E01", "2024-02-15", 2, 0.0)]
    rows += [_row("m-hsq", "straddle", GATE_RULE, "straddle", "E01", "2024-02-15", 1, 0.0)]
    universe = pd.DataFrame(_with_account([{**row, "status": "not_taken", "reason": "below_cutoff"} for row in rows]))
    taken = universe[universe["status"] == "taken"]
    assert taken.empty and "secured_per_contract" in taken.columns
    sections = assemble_mod._account_sections(taken, {"H-sq": "m-hsq"}, {"m-hsq": "H-sq"}, universe)
    text = "\n".join(sections)
    assert any("| 0 | 0 | 100000.00 |" in line for line in text.splitlines())
    assert "not_testable" not in text


def test_taken_row_with_null_exit_date_refuses_as_decisions_invalid(tmp_path):
    def make(rows):
        out = _with_account(rows)
        return [{**out[0], "exit_date": None}, *out[1:]]  # out[0] is a taken short-vol row

    out = build(tmp_path, rows_fn=make)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_decisions_invalid"
    assert not (out / "REPORT.md").exists()


def test_secured_column_without_exit_date_refuses_as_decisions_columns(tmp_path):
    def make(rows):
        return [{k: v for k, v in row.items() if k != "exit_date"} for row in _with_account(rows)]

    out = build(tmp_path, rows_fn=make)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, plan_frame(), admitted=None)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_decisions_columns"
    assert not (out / "REPORT.md").exists()


# ---- Amendment 3 and 6: G_PROD is the straddle book; an undetermined G_PROD row drops its event from E_str only ----


def _with_gate_undetermined(rows):
    """E01's G_PROD rows are undetermined under every model (a thin pool): E01 leaves E_str and stays in short-vol E."""
    return [{**row, "status": "undetermined", "reason": "thin_pool", "pnl_dollars": 0.0}
            if row["book"] == "straddle" and row["event_id"] == "E01" else row for row in _with_account(rows)]


def test_undetermined_g_prod_row_drops_its_event_from_e_str_but_not_from_e(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = build(tmp_path / "a", rows_fn=_with_account)
    gated = build(tmp_path / "b", rows_fn=_with_gate_undetermined)
    assemble_mod.assemble(plain, plan_frame(), admitted=None)
    assemble_mod.assemble(gated, plan_frame(), admitted=None)
    plain_text = (plain / "REPORT.md").read_text(encoding="utf-8")
    gated_text = (gated / "REPORT.md").read_text(encoding="utf-8")
    assert _cells(next(line for line in _section_lines(gated_text, "### T2") if line.startswith("| total |")))[1] == "4"  # E keeps E01
    assert "undetermined_events_excluded: 0" in gated_text
    assert "undetermined_straddle_events_excluded: 1" in gated_text
    t7_plain, t7_gated = _section_lines(plain_text, "### T7"), _section_lines(gated_text, "### T7")
    assert [line for line in t7_gated if line.startswith("| H1 |")] == [line for line in t7_plain if line.startswith("| H1 |")]  # short-vol is unchanged
    assert _cells(next(line for line in t7_plain if line.startswith("| H2 | H | G_PROD |")))[4] == "4"
    assert _cells(next(line for line in t7_gated if line.startswith("| H2 | H | G_PROD |")))[4] == "3"  # E_str loses E01
    assert any("G_PROD" in line for line in _section_lines(plain_text, "### T6"))  # the G_PROD rows are the straddle book in T6
    assert any("G_PROD" in line for line in _section_lines(gated_text, "### T6"))
    account = "### Sensitivity (Amendment 2)"
    # Hand-derived: G_PROD has three taken events (E02, E03, E04; E01 is undetermined). Debit 0.5 restates to secured 50 per
    # contract; contracts funded floor(0.25 * 0.66 * equity / 50) = 330 at 100000, 340 at 103300, 352 at 106700. Each contract
    # nets (0.6 - 0.5) x 100 = 10, so final = 100000 + 3300 + 3400 + 3520 = 110220.
    assert "| straddle | E-log | G_PROD | 3 | 3 | 110220.00 | 0.1022 | 0 | 1 |" in _section_lines(gated_text, account)
    assert "| straddle | E-log | G_PROD | 4 | 4 | 113850.00 | 0.1385 | 0 | 1 |" in _section_lines(plain_text, account)  # four events, as in the straddle-only test


def test_g_prod_rows_enter_e_str_t6_and_h2(tmp_path):
    out = build(tmp_path)  # every straddle row is a G_PROD row, for every pair model
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    h2 = _cells(next(line for line in _section_lines(text, "### T7") if line.startswith("| H2 | H | G_PROD |")))
    assert h2[3:5] == ["ok", "4"]  # a contrast over E_str on the G_PROD rows, not a missing-row reason
    assert "no_gate_rows" not in text
    assert any("G_PROD" in line for line in _section_lines(text, "### T6"))
    assert "undetermined_straddle_events_excluded: 0" in text


def test_no_g_prod_rows_renders_with_an_empty_straddle_book(tmp_path):
    out = build(tmp_path, rows_fn=lambda rows: [row for row in rows if row["book"] == "short_vol"])  # no gate configured
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    t7 = _section_lines(text, "### T7")
    assert _cells(next(line for line in t7 if line.startswith("| H2 | H | G_PROD |")))[3] == "not_testable: too_few_quarters_or_tickers"
    assert _cells(next(line for line in t7 if line.startswith("| H3 | H | as_is |")))[3] == "not_testable: too_few_quarters_or_tickers"
    assert not any("G_PROD" in line for line in _section_lines(text, "### T6"))  # T6 is empty
    verdicts = [_cells(line) for line in _section_lines(text, "### T10") if line.startswith("| H")]
    assert verdicts == [["H1", "+1", "clear"], ["H2", "-1", "not_testable"], ["H3", "+1", "not_testable"]]
    assert not any(line.startswith("### G_PROD") for line in text.split("\n"))  # no G_PROD section without G_PROD rows


# ---- Amendment 5: short-vol E and straddle E_str are separate populations ----

TWO = {"E01": "2024-02-15", "E02": "2024-05-15"}  # two quarters (2024Q1, 2024Q2) and two tickers (AAA, BBB) in the plan


def _e01_straddle_undetermined(rows):
    """E01's straddle rows are undetermined under every model (a missing forecast); its short-vol rows stay taken."""
    return [{**row, "status": "undetermined", "reason": "missing_forecast", "pnl_dollars": 0.0}
            if row["event_id"] == "E01" and row["book"] == "straddle" else row for row in rows]


def test_straddle_undetermined_row_keeps_its_event_in_e_and_drops_it_from_e_str(tmp_path):
    out = build(tmp_path, rows_fn=_e01_straddle_undetermined, dates=TWO)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "undetermined_events_excluded: 0" in text
    assert "undetermined_straddle_events_excluded: 1" in text
    assert _cells(next(line for line in _section_lines(text, "### T2") if line.startswith("| total |")))[1] == "2"  # E01 stays in short-vol E
    t7 = _section_lines(text, "### T7")
    assert _cells(next(line for line in t7 if line.startswith("| H1 | H | as_is |")))[3:5] == ["ok", "2"]
    # E_str is {E02} alone, so the straddle contrast and the H3 intersection (E intersect E_str = {E02}) are not testable.
    assert _cells(next(line for line in t7 if line.startswith("| H2 | H | G_PROD |")))[3] == "not_testable: too_few_quarters_or_tickers"
    assert _cells(next(line for line in t7 if line.startswith("| H3 | H | as_is |")))[3] == "not_testable: too_few_quarters_or_tickers"


def test_short_vol_undetermined_row_drops_its_event_from_e_but_not_from_e_str(tmp_path):
    out = build(tmp_path, rows_fn=_e01_undetermined_for_m_hsq, dates=TWO)
    assemble_mod.assemble(out, plan_frame(), admitted=None)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "undetermined_events_excluded: 1" in text
    assert "undetermined_straddle_events_excluded: 0" in text
    assert _cells(next(line for line in _section_lines(text, "### T2") if line.startswith("| total |")))[1] == "1"  # E is {E02}
    t7 = _section_lines(text, "### T7")
    assert _cells(next(line for line in t7 if line.startswith("| H2 | H | G_PROD |")))[3:5] == ["ok", "2"]  # E_str keeps E01
    assert _cells(next(line for line in t7 if line.startswith("| H1 | H | as_is |")))[3] == "not_testable: too_few_quarters_or_tickers"
    assert _cells(next(line for line in t7 if line.startswith("| H3 | H | as_is |")))[3] == "not_testable: too_few_quarters_or_tickers"


def test_h3_contrast_pairs_the_intersection_of_the_two_books():
    frame = pd.DataFrame([{"model_id": m, "book": b, "arm_or_rule": v, "event_id": e, "taken": True, "pnl_dollars": p}
                          for m, p in (("m-habs", 11.0), ("m-hsq", 10.0)) for e in ("E01", "E02")
                          for b, v in (("short_vol", "as_is"), ("straddle", "G_PROD"))])
    events = {("m-habs", "short_vol"): ["E01", "E02"], ("m-hsq", "short_vol"): ["E01", "E02"],
              ("m-habs", "straddle"): ["E02"], ("m-hsq", "straddle"): ["E02"]}  # E_str holds E02 only
    quarter_of, tick = {"E01": "2024Q1", "E02": "2024Q2"}, {"E01": "AAA", "E02": "BBB"}
    pair = ("m-habs", "m-hsq")  # (median, mean)
    assert isinstance(assemble_mod._contrast_cell(frame, events, pair, [("short_vol", "as_is", 1)], quarter_of, tick), dict)  # E alone: testable
    h3 = assemble_mod._contrast_cell(frame, events, pair, [("short_vol", "as_is", 1), ("straddle", "G_PROD", -1)], quarter_of, tick)
    assert h3 == "too_few_quarters_or_tickers"  # E intersect E_str = {E02}: one quarter
