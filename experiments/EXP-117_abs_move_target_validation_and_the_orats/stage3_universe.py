#!/usr/bin/env python3
"""EXP-117 Stage 3 — can the panel's event universe come from ORATS?

Scoreability bar (registered): a ticker is scoreable once it has >=12 prior
prints (the span-12 EMA features the champion models require). Counts, all from
on-disk data:

  1. tickers reaching >=12 prints on the ORATS calendar that the oquants
     panel (the current universe) does not carry;
  2. the same among the live board's currently-unscoreable tickers;
  3. effect on panel rows; the >=4 admission bar for reference.

Promotion threshold (registered before measurement): adopt the universe change
only if it adds >=15 scoreable tickers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/root/investing-plan")
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data import store  # noqa: E402

HERE = Path(__file__).resolve().parent
MIN_SCOREABLE = 12
MIN_ADMISSION = 4
THRESHOLD = 15

report: dict = {"generated_at": pd.Timestamp.now("UTC").isoformat(),
                "min_scoreable_prints": MIN_SCOREABLE,
                "promotion_threshold_added_tickers": THRESHOLD}


def main() -> None:
    ev = store.read_table("earnings_events",
                          columns=["ticker", "event_date", "session", "src_orats"])
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    today = pd.Timestamp.today().normalize()
    hist = ev[ev["src_orats"] & (ev["event_date"] < today)].copy()
    print(f"ORATS historical events: {len(hist):,} on {hist.ticker.nunique()} tickers", flush=True)

    oq_tickers = {p.name[len("moves_"):-len(".json")]
                  for p in paths.RAW_OQUANTS_MOVES.glob("moves_*.json")}
    print(f"oquants panel tickers: {len(oq_tickers)}", flush=True)

    hist["in_oquants"] = hist["ticker"].isin(oq_tickers)
    counts = hist.groupby("ticker").agg(
        n_prints=("event_date", "size"),
        n_with_session=("session", lambda s: s.notna().sum()),
        in_oquants=("in_oquants", "first"),
    )
    counts["scoreable"] = counts["n_with_session"] >= MIN_SCOREABLE
    counts["admissible"] = counts["n_with_session"] >= MIN_ADMISSION

    n_all_scoreable = int(counts["scoreable"].sum())
    n_oq_scoreable = int(counts[counts["in_oquants"]]["scoreable"].sum())
    new = counts[(~counts["in_oquants"]) & counts["scoreable"]]
    report["orats_universe"] = {
        "tickers_total": int(len(counts)),
        "tickers_scoreable_ge12": n_all_scoreable,
        "tickers_scoreable_in_oquants": n_oq_scoreable,
        "tickers_scoreable_outside_oquants": int(len(new)),
        "tickers_admissible_ge4_outside_oquants": int(
            ((~counts["in_oquants"]) & counts["admissible"]).sum()),
    }

    # new rows the panel would gain (events k>=4 of the new tickers)
    new_tickers = set(new.index)
    gained_rows = hist[hist["ticker"].isin(new_tickers)].groupby("ticker").size()
    gained_rows = (gained_rows - MIN_ADMISSION).clip(lower=0)
    report["panel_rows_added"] = {
        "events_on_new_scoreable_tickers": int(gained_rows.sum()),
        "tickers": int((gained_rows > 0).sum()),
    }

    # the live board's currently-unscoreable tickers
    board_path = ROOT / "dashboard" / "earnings" / "data" / "board.json"
    board = json.loads(board_path.read_text())
    board_tickers = set()
    rows = board.get("rows") or []
    for r in rows:
        tk = r.get("ticker")
        if tk:
            board_tickers.add(tk)
    if not board_tickers:
        for tk, _ in (board.items() if isinstance(board, dict) else []):
            board_tickers.add(tk)
    report["board_tickers"] = len(board_tickers)

    unscoreable = board_tickers - oq_tickers
    report["board_unscoreable_today"] = len(unscoreable)
    cleared = sorted(t for t in unscoreable
                     if t in counts.index and counts.loc[t, "scoreable"])
    report["board_unscoreable_cleared_by_orats"] = {
        "n": len(cleared), "tickers": cleared,
    }
    blocked_no_orats = sorted(t for t in unscoreable if t not in counts.index)
    report["board_unscoreable_no_orats_history"] = {
        "n": len(blocked_no_orats), "tickers": blocked_no_orats,
    }
    thin = sorted(t for t in unscoreable
                  if t in counts.index and not counts.loc[t, "scoreable"])
    report["board_unscoreable_thin_history"] = {
        "n": len(thin),
        "tickers": thin,
        "print_counts": {t: int(counts.loc[t, "n_with_session"]) for t in thin},
    }

    unknown_path = ROOT / "reports" / "orats_unknown_symbols.json"
    if unknown_path.exists():
        unknown = json.loads(unknown_path.read_text())
        syms = unknown if isinstance(unknown, list) else unknown.get("symbols", [])
        report["orats_404_symbols_on_board"] = sorted(set(syms) & board_tickers)

    verdict = len(new) >= THRESHOLD
    report["promotion_threshold_met"] = bool(verdict)
    report["verdict"] = (
        f"universe change adds {len(new)} scoreable tickers "
        f"(threshold {THRESHOLD}): {'ADOPT' if verdict else 'DROP'}"
    )

    # era distribution of the new scoreable tickers (recent listings are
    # expected to dominate; prints before 2017 have no chains anyway)
    new_hist = hist[hist["ticker"].isin(new_tickers)]
    first_print = new_hist.groupby("ticker")["event_date"].min()
    report["new_scoreable_first_print_era"] = {
        "<2010": int((first_print.dt.year < 2010).sum()),
        "2010-2016": int(((first_print.dt.year >= 2010) & (first_print.dt.year < 2017)).sum()),
        "2017+": int((first_print.dt.year >= 2017).sum()),
    }

    (HERE / "results" / "stage3_results.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
