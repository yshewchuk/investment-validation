"""EXP-188 A6-4: the board's recorded short-vol strikes against M0's, read from the board's prediction jsonl. Research-only: imported by size_pnl_assemble; read-only."""
import json
import os
import stat
import sys

import pandas as pd

from engine.ledger import canonical_predictions
from experiments import size_pnl_report as report

MAX_FILE_BYTES = 64 << 20
MAX_TOTAL_BYTES = 512 << 20
MAX_ROWS = 1_000_000
TITLE = "### Board-match check (M0)"
NOTE = ("The board's recorded strikes come from a D0 decision with a live forecast, so a mismatch is expected to include forecast "
        "differences. The in-run substitute is the row-3 resolver, production's own price_structure.")


def read_board_ledger(directory):
    """(rows, None): the board's canonical prediction rows of the .jsonl files in ``directory``; or (None, reason). Never a partial read."""
    if not isinstance(directory, (str, os.PathLike)):
        return None, "ledger_unreachable"
    try:
        with os.scandir(directory) as it:
            entries = sorted((e for e in it if e.name.endswith(".jsonl") and e.is_file(follow_symlinks=False)), key=lambda e: e.name)
    except (OSError, ValueError):  # ValueError: a path with a NUL byte
        return None, "ledger_unreachable"
    rows, total, lines = [], 0, 0
    for entry in entries:
        try:
            fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            return None, "ledger_unreachable"
        with os.fdopen(fd, "rb") as handle:
            st = os.fstat(handle.fileno())
            if not stat.S_ISREG(st.st_mode):
                return None, "ledger_unreachable"
            if st.st_size > MAX_FILE_BYTES or total + st.st_size > MAX_TOTAL_BYTES:
                return None, "board_ledger_too_large"
            data = handle.read(st.st_size + 1)
        if len(data) > st.st_size:
            return None, "board_ledger_too_large"
        total += len(data)
        lines += data.count(b"\n") + 1  # running row estimate, before any parse
        if lines > MAX_ROWS:
            return None, "board_ledger_too_large"
        try:
            for line in data.decode("utf-8").split("\n"):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("a row is not an object")
                    rows.append(row)
        except (UnicodeDecodeError, RecursionError, ValueError):
            return None, "board_ledger_malformed"
    try:  # the engine.ledger.read_predictions rule: a row a later row supersedes is dropped
        superseded = {r["supersedes"] for r in rows if r.get("supersedes")}
        return canonical_predictions([r for r in rows if r.get("row_id") not in superseded]), None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None, "board_ledger_malformed"


def _strikes(strikes):
    """The sorted distinct strikes, rounded to 6 dp."""
    return tuple(sorted({round(float(k), 6) for k in strikes}))


def _cause(board, m0):
    """A changed distinct-strike count is a collision; the same count with another span is a width; else a snap."""
    if len(board) != len(m0):
        return "collision"
    return "width" if round(board[-1] - board[0], 6) != round(m0[-1] - m0[0], 6) else "snap"


def _range(dates):
    """'min..max' of the date strings, or 'none'."""
    return f"{min(dates)}..{max(dates)}" if dates else "none"


def _compare(rows, m0, strikes):
    """(body, outcome) of the comparison of the canonical board ``rows`` with M0's frame ``m0`` and its strikes."""
    board = {}
    for row in rows:
        if row.get("event_id") is not None:
            board.setdefault((str(row["event_id"]), str(row["strategy"])), row)
    m0_keys = {(str(e), str(s)) for e, s in zip(m0["event_id"], m0["structure"])}
    common = sorted(set(board) & m0_keys)
    if not common:
        board_dates = [str(r["event_date"]) for r in rows if r.get("event_date")]
        m0_dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(m0["event_date"]).dropna()]
        return (f"not_available: no_common_events\n\n- board event_date range: {_range(board_dates)}\n"
                f"- M0 event_date range: {_range(m0_dates)}", "no_common_events")
    lines = [f"common events: {len(common)}"]
    have = [k for k in common if strikes is not None and k in strikes]
    if not have:
        lines.append("strike_match: not_available (m0_strikes_not_recorded)")
        return "\n".join(lines), f"common={len(common)} m0_strikes_not_recorded"
    counts = {"match": 0, "collision": 0, "width": 0, "snap": 0}
    for key in have:
        got, want = _strikes(leg["strike"] for leg in board[key]["structure"]["legs"]), _strikes(strikes[key])
        counts["match" if got == want else _cause(got, want)] += 1
    rate = counts["match"] / len(have)
    lines += [f"strike_match: {counts['match']} of {len(have)} compared events match (match rate {rate:.2f})", "",
              report.md_table(["outcome", "events"], [[name, n] for name, n in counts.items()])]
    return "\n".join(lines), f"common={len(common)} compared={len(have)} matches={counts['match']}"


def section(directory, m0, strikes=None):
    """The board-match section (M0): the common events, the strike match and its mismatch causes, or why it is not available."""
    if directory is None:
        body, outcome = "not_available: board_ledger_not_configured", "board_ledger_not_configured"
    else:
        rows, reason = read_board_ledger(directory)
        if reason is not None:
            body, outcome = f"not_available: {reason}", reason
        else:
            try:
                body, outcome = _compare(rows, m0, strikes)
            except (KeyError, TypeError, ValueError, AttributeError):
                body, outcome = "not_available: board_ledger_malformed", "board_ledger_malformed"
    print(f"[board_match] {outcome}", file=sys.stderr)
    return f"{TITLE}\n\n{body}\n\n{NOTE}"
