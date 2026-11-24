"""EXP-188 A6-4: the board-match reader and section over synthetic board jsonl (tmp_path only; never the real ledger)."""
import json
import os

import pandas as pd
import pytest

from experiments import size_pnl_board_match as bm


def row(row_id, event_id, strategy, strikes, event_date="2024-06-10", as_of="2024-06-01", supersedes=None):
    """A board row; canonical_predictions keys a trade on (ticker, strategy, event_date), so the ticker is the event's."""
    r = {"row_id": row_id, "event_id": event_id, "ticker": event_id, "strategy": strategy, "event_date": event_date,
         "as_of": as_of, "structure": {"legs": [{"strike": k, "side": "buy", "qty": 1} for k in strikes]}}
    if supersedes:
        r["supersedes"] = supersedes
    return r


def write(directory, name, *rows):
    (directory / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def m0_frame(*triples):
    """M0 rows as (event_id, structure, event_date) triples."""
    df = pd.DataFrame(list(triples), columns=["event_id", "structure", "event_date"])
    return df.assign(event_date=pd.to_datetime(df["event_date"]))


# ---- reader ----------------------------------------------------------------

def test_reader_reads_both_files_dedups_the_canonical_trade_and_drops_a_superseded_row(tmp_path):
    write(tmp_path, "a.jsonl", row("a1", "E1", "CND-PS", [94, 97, 103, 106]), row("a2", "E1", "CND-PS", [94, 97, 103, 106], as_of="2024-06-02"))
    write(tmp_path, "b.jsonl", row("b1", "E2", "TWIN-P", [84, 92], as_of="2024-06-05"),
          row("b2", "E2", "TWIN-P", [84, 92, 96], as_of="2024-06-01", supersedes="b1"))  # b1 would win on as_of; it is superseded
    rows, reason = bm.read_board_ledger(tmp_path)
    assert reason is None
    assert sum(r["event_id"] == "E1" for r in rows) == 1  # the same trade, twice
    assert [r["row_id"] for r in rows if r["event_id"] == "E2"] == ["b2"]


def test_reader_skips_a_symlinked_jsonl_and_a_directory_named_jsonl(tmp_path):
    write(tmp_path, "a.jsonl", row("a1", "E1", "CND-PS", [94, 97]))
    os.symlink(tmp_path / "a.jsonl", tmp_path / "link.jsonl")
    (tmp_path / "x.jsonl").mkdir()
    rows, reason = bm.read_board_ledger(tmp_path)
    assert reason is None and [r["row_id"] for r in rows] == ["a1"]


@pytest.mark.parametrize("payload", [b"not json\n", b"[1, 2]\n", b"\xff\n"], ids=["bad_json", "not_an_object", "bad_utf8"])
def test_reader_refuses_a_malformed_line(tmp_path, payload):
    (tmp_path / "a.jsonl").write_bytes(payload)
    assert bm.read_board_ledger(tmp_path) == (None, "board_ledger_malformed")


def test_reader_refuses_a_file_over_the_byte_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "MAX_FILE_BYTES", 10)
    (tmp_path / "a.jsonl").write_bytes(b"\n" * 11)
    assert bm.read_board_ledger(tmp_path) == (None, "board_ledger_too_large")


def test_reader_refuses_when_the_running_total_passes_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "MAX_TOTAL_BYTES", 150)
    (tmp_path / "a.jsonl").write_bytes(b"\n" * 100)
    (tmp_path / "b.jsonl").write_bytes(b"\n" * 100)
    assert bm.read_board_ledger(tmp_path) == (None, "board_ledger_too_large")


def test_reader_refuses_past_the_row_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "MAX_ROWS", 2)
    (tmp_path / "a.jsonl").write_bytes(b"\n\n\n")  # four lines by the estimate
    assert bm.read_board_ledger(tmp_path) == (None, "board_ledger_too_large")


def test_reader_reports_a_missing_path_and_a_file_path_as_unreachable(tmp_path):
    assert bm.read_board_ledger(tmp_path / "missing") == (None, "ledger_unreachable")
    (tmp_path / "f.jsonl").write_text("", encoding="utf-8")
    assert bm.read_board_ledger(tmp_path / "f.jsonl") == (None, "ledger_unreachable")


def test_reader_returns_no_rows_when_a_later_file_is_malformed(tmp_path):
    write(tmp_path, "a.jsonl", row("a1", "E1", "CND-PS", [94, 97]))
    (tmp_path / "b.jsonl").write_bytes(b"not json\n")
    assert bm.read_board_ledger(tmp_path) == (None, "board_ledger_malformed")


# ---- section ---------------------------------------------------------------

def test_section_not_configured_still_ends_with_the_note():
    text = bm.section(None, m0_frame())
    assert text.startswith("### Board-match check (M0)") and "not_available: board_ledger_not_configured" in text
    assert "D0 decision" in text and "row-3 resolver" in text


def test_section_without_common_events_gives_both_date_ranges(tmp_path):
    write(tmp_path, "a.jsonl", row("a1", "E9", "CND-PS", [94, 97], event_date="2024-06-10"),
          row("a2", "E8", "CND-PS", [94, 97], event_date="2024-07-01"))
    text = bm.section(tmp_path, m0_frame(("E1", "CND-PS", "2024-03-01"), ("E2", "CND-PS", "2024-04-02")))
    assert "no_common_events" in text
    assert "board event_date range: 2024-06-10..2024-07-01" in text
    assert "M0 event_date range: 2024-03-01..2024-04-02" in text


def test_section_with_an_empty_m0_frame_reports_none(tmp_path):
    write(tmp_path, "a.jsonl", row("a1", "E1", "CND-PS", [94, 97]))
    text = bm.section(tmp_path, m0_frame())
    assert "no_common_events" in text and "M0 event_date range: none" in text


def test_section_with_common_events_but_no_m0_strikes_says_not_recorded(tmp_path):
    write(tmp_path, "a.jsonl", row("a1", "E1", "CND-PS", [94, 97]))
    text = bm.section(tmp_path, m0_frame(("E1", "CND-PS", "2024-06-10")))
    assert "common events: 1" in text and "strike_match: not_available (m0_strikes_not_recorded)" in text


def test_section_counts_one_match_and_one_mismatch_of_each_cause(tmp_path):
    write(tmp_path, "a.jsonl",
          row("a1", "E1", "CND-PS", [94, 97, 103, 106]),  # match
          row("a2", "E2", "CND-PS", [94, 97, 103, 106]),  # width: same count, span 12 against 20
          row("a3", "E3", "CND-PS", [94, 97, 106]),       # collision: three distinct strikes against four
          row("a4", "E4", "CND-PS", [94, 97, 103, 106]))  # snap: same count and span, different strikes
    m0 = m0_frame(("E1", "CND-PS", "2024-06-10"), ("E2", "CND-PS", "2024-06-10"),
                  ("E3", "CND-PS", "2024-06-10"), ("E4", "CND-PS", "2024-06-10"))
    strikes = {("E1", "CND-PS"): (94.0, 97.0, 103.0, 106.0), ("E2", "CND-PS"): (90.0, 97.0, 103.0, 110.0),
               ("E3", "CND-PS"): (94.0, 97.0, 103.0, 106.0), ("E4", "CND-PS"): (94.0, 98.0, 102.0, 106.0)}
    text = bm.section(tmp_path, m0, strikes)
    assert "common events: 4" in text
    assert "strike_match: 1 of 4 compared events match (match rate 0.25)" in text
    for line in ("| match | 1 |", "| collision | 1 |", "| width | 1 |", "| snap | 1 |"):
        assert line in text
    assert text.rstrip().endswith(bm.NOTE)
