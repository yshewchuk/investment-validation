"""EXP-188 S9c: REPORT.md assembly over synthetic receipts and decisions (tmp_path only; no store, no data/)."""
import os
from types import SimpleNamespace

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
IDENTITY = {"plan_id": "plan-1", "registration_id": "reg-1", "seal_id": "seal-1", "panel_id": "panel-1"}
SPEC_HASH, SNAPSHOT, PLAN = "a" * 64, "snap-1", "PLAN"  # PLAN: the receipted plan hash of the fixture's inputs_outer; every full-run call passes it
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
        "stage": "inputs", "roster": _roster(pairs or PAIRS_OK), "models": {mid: {"status": "ready"} for mid in ROLES},
        "identity": IDENTITY, "spec_hash": SPEC_HASH, "snapshot": SNAPSHOT, "plan": PLAN})
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
    summary = assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert summary["report"] == str(out / "REPORT.md") and summary["sections"] == 11 and summary["rows"] > 0
    assert text.startswith("# EXP-188 REPORT")
    heads = ["### T1", "### T2", "### T3", "### T4", "### T5", "### T6", "### T7", "### T8", "### T9", "### T10"]
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
    assemble_mod.assemble(build(tmp_path), TICKERS, plan_binding=PLAN)
    assert len(calls) == 18  # three hypotheses, three pairs, two arms each
    d, quarters, tickers = calls[0]  # H1, pair H, as_is
    assert np.allclose(d, 1.0)
    assert quarters == ["2024Q1", "2024Q2", "2025Q1", "2025Q2"]
    assert tickers == ["AAA", "BBB", "CCC", "AAA"]


@pytest.mark.parametrize("reason, make", CASES, ids=[reason for reason, _make in CASES])
def test_refusal_writes_no_report(tmp_path, reason, make):
    out, tickers = make(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, tickers, plan_binding=PLAN)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == reason
    assert not (out / "REPORT.md").exists()


def test_non_testable_pair_is_reported_and_the_report_is_written(tmp_path):
    out = build(tmp_path, pairs={"H": "not_testable", "E": "testable", "N": "testable"})
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | not_testable: pair_member_excluded |" in text
    assert "| H1 | H | retuned | not_testable: pair_member_excluded |" in text
    assert "| H1 | +1 | not_testable |" in text


def test_fewer_than_two_quarters_is_not_testable(tmp_path):
    one_quarter = {"E01": "2024-02-15", "E02": "2024-02-20", "E03": "2024-03-01", "E04": "2024-03-05"}
    out = build(tmp_path, dates=one_quarter)
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | not_testable: too_few_quarters_or_tickers |" in text


def test_single_ticker_across_quarters_is_not_testable(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, {event: "AAA" for event in EVENT_DATES}, plan_binding=PLAN)
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
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
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
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert f"SMOKE scope {scope}, not a full-population result" in text


def test_failed_replace_keeps_the_old_report_and_leaves_no_tmp(tmp_path, monkeypatch):
    out = build(tmp_path)
    (out / "REPORT.md").write_text("OLD", encoding="utf-8")

    def refused(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr(os, "replace", refused)
    with pytest.raises(OSError):
        assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    assert (out / "REPORT.md").read_text(encoding="utf-8") == "OLD"
    assert not list(out.glob("*.tmp"))


def test_t10_verdicts_are_the_pinned_literals(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    verdicts = [_cells(line) for line in _section_lines(text, "### T10") if line.startswith("| H")]
    assert verdicts == [["H1", "+1", "clear"], ["H2", "-1", "reversed"], ["H3", "+1", "no_difference"]]
    assert "not_testable because the required full year" not in text


def test_plan_binding_matching_the_receipt_succeeds(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    assert (out / "REPORT.md").exists()


def test_plan_binding_mismatch_refuses_a_full_run(tmp_path):
    out = build(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding="q" * 64)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_inputs_mismatch"
    assert not (out / "REPORT.md").exists()


def test_plan_binding_is_skipped_for_a_smoke_scope(tmp_path):
    out = build(tmp_path, scope="f" * 64)
    assemble_mod.assemble(out, TICKERS, plan_binding="q" * 64)
    assert (out / "REPORT.md").exists()


def test_plan_binding_is_a_required_keyword(tmp_path):
    out = build(tmp_path)
    with pytest.raises(TypeError):
        assemble_mod.assemble(out, TICKERS)
    assert not (out / "REPORT.md").exists()


@pytest.mark.parametrize("bad", ["", None, 123])
def test_blank_or_non_string_plan_binding_refuses_before_any_read(tmp_path, bad):
    # tmp_path holds no receipts: the refusal must come before any file is read, not from a missing receipt.
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(tmp_path / "missing", TICKERS, plan_binding=bad)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_inputs_mismatch"


def test_blank_plan_binding_refuses_a_built_run_too(tmp_path):
    out = build(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding="")
    assert caught.value.details["reason"] == "report_inputs_mismatch"
    assert not (out / "REPORT.md").exists()


def test_unknown_decision_status_refuses_as_decisions_invalid(tmp_path):
    out = build(tmp_path, rows_fn=lambda rows: [{**rows[0], "status": "weird"}, *rows[1:]])
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    assert caught.value.details["reason"] == "report_decisions_invalid"
    assert not (out / "REPORT.md").exists()


def test_2025_only_decisions_are_not_testable_and_t7_still_shows_stats(tmp_path):
    dates = {"E01": "2025-02-15", "E02": "2025-05-15", "E03": "2025-08-15", "E04": "2025-11-15"}
    out = build(tmp_path, dates=dates)
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    verdicts = [_cells(line) for line in _section_lines(text, "### T10") if line.startswith("| H")]
    assert verdicts == [["H1", "+1", "not_testable"], ["H2", "-1", "not_testable"], ["H3", "+1", "not_testable"]]
    assert ("not_testable because the required full year(s) [2024] are absent from the outer decisions; "
            "the inner window publishes no decisions") in text
    assert "| H1 | H | as_is | ok | 4 | 1 |" in text


@pytest.mark.parametrize("tickers", [None, ["x"], {"E01": " "}, {" ": "AAA"}, {"E01": None}])
def test_tickers_must_be_a_mapping_of_nonblank_strings(tmp_path, tickers):
    out = build(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, tickers, plan_binding=PLAN)
    assert caught.value.details["reason"] == "report_tickers_invalid"
    assert not (out / "REPORT.md").exists()


def test_requested_identity_matches_the_receipts(tmp_path):
    out = build(tmp_path)
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN, expected=IDENTITY, spec_hash=SPEC_HASH, snapshot=SNAPSHOT)
    assert (out / "REPORT.md").exists()


@pytest.mark.parametrize("change", [{"expected": {**IDENTITY, "plan_id": "other"}}, {"spec_hash": "b" * 64}, {"snapshot": "snap-2"}])
def test_requested_identity_mismatch_refuses(tmp_path, change):
    out = build(tmp_path)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding=PLAN, **{"expected": IDENTITY, "spec_hash": SPEC_HASH, "snapshot": SNAPSHOT, **change})
    assert caught.value.details["reason"] == "report_inputs_mismatch"
    assert not (out / "REPORT.md").exists()


def test_outer_receipt_without_rows_is_malformed(tmp_path):
    out = build(tmp_path)
    _rewrite(out / "receipts" / "outer.json", lambda body: {k: v for k, v in body.items() if k != "rows"})
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    assert caught.value.details["reason"] == "report_receipt_malformed"
    assert not (out / "REPORT.md").exists()


def test_non_date_event_date_refuses_as_decisions_invalid(tmp_path):
    out = build(tmp_path, rows_fn=lambda rows: [{**row, "event_date": "not a date"} for row in rows])
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    assert caught.value.details["reason"] == "report_decisions_invalid"


def test_decisions_one_byte_over_the_cap_refuses(tmp_path, monkeypatch):
    out = build(tmp_path)
    monkeypatch.setattr(decisions, "MAX_DECISIONS_BYTES", (out / "decisions.parquet").stat().st_size - 1)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    assert caught.value.details["reason"] == "report_decisions_mismatch"


def test_hash_loop_refuses_when_fstat_understates_the_size(tmp_path, monkeypatch):
    out = build(tmp_path)
    target, real_fstat = os.stat(out / "decisions.parquet"), os.fstat

    def understated(fd):  # the decisions descriptor reports size 0 and keeps its regular-file mode
        st = real_fstat(fd)
        return SimpleNamespace(st_mode=st.st_mode, st_size=0) if os.path.samestat(st, target) else st

    monkeypatch.setattr(os, "fstat", understated)
    monkeypatch.setattr(decisions, "MAX_DECISIONS_BYTES", 16)
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
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
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
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
    assemble_mod.assemble(clean, TICKERS, plan_binding=PLAN)
    clean_text = (clean / "REPORT.md").read_text(encoding="utf-8")
    dirty_dir = tmp_path / "dirty"
    dirty_dir.mkdir()
    out = build(dirty_dir, rows_fn=_e01_undetermined_for_m_hsq)
    assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| H1 | H | as_is | ok | 3 | 1 |" in text  # n is one smaller than the four-event fixture's 4
    assert "| H1 | H | as_is | ok | 4 | 1 |" not in text
    assert _section_lines(text, "### T7") == _section_lines(clean_text, "### T7")  # same mean and statistics as without E01
    assert "undetermined_events_excluded: 1" in text
    assert "events undetermined under some model are excluded from E: 1" in text


def test_all_undetermined_decisions_refuse_as_empty(tmp_path):
    out = build(tmp_path, rows_fn=lambda rows: [{**row, "status": "undetermined", "pnl_dollars": float("nan")} for row in rows])
    with pytest.raises(Refusal) as caught:
        assemble_mod.assemble(out, TICKERS, plan_binding=PLAN)
    assert caught.value.code == CONFLICT
    assert caught.value.details["reason"] == "report_decisions_empty"
    assert not (out / "REPORT.md").exists()
