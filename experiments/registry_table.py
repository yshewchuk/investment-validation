#!/usr/bin/env python3
"""Generate the experiment registry table from LEDGER.csv.

    python3 experiments/registry_table.py [--write]

The plan's experiment registry is generated output, not a hand-edited table:
this renders one row per experiment id from the append-only ledger (latest
state per id, planned vs ran, headline numbers, promoted flag). ``--write``
saves ``experiments/EXPERIMENT_REGISTRY.md``; by default the table prints to
stdout for pasting into a plan or report.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import lib  # noqa: E402


def render_table(ledger_path: Path | None = None) -> str:
    frame = lib.ledger_read(ledger_path)
    if frame.empty:
        return "_No experiments in the ledger yet._"
    frame = frame.fillna("")
    frame = frame.astype(str).replace({"": "", "nan": ""})
    lines = [
        "| id | stage | spec_hash | last date | OOS mean @mid | sharpe_trade | promoted |",
        "|---|---|---|---|---|---|---|",
    ]
    # Latest row per id wins; the ledger keeps history, the table shows state.
    for exp_id, rows in frame.groupby("id", sort=True):
        last = rows.iloc[-1]
        mean = last.get("oos_mean_mid", "")
        sharpe = last.get("sharpe_trade", "")
        lines.append(
            f"| {exp_id} | {last['stage']} | {str(last['spec_hash'])[:12]}… | {last['date']} | "
            f"{mean if mean != '' else '—'} | {sharpe if sharpe != '' else '—'} | {last['promoted']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    write = bool(argv and "--write" in argv)
    table = render_table()
    if write:
        out = lib.EXPERIMENTS_DIR / "EXPERIMENT_REGISTRY.md"
        out.write_text(
            "# Experiment registry (generated from LEDGER.csv — do not hand-edit)\n\n"
            + table + "\n")
        print(f"wrote {out}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
