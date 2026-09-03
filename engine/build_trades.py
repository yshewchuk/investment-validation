#!/usr/bin/env python3
"""Rebuild the engine's simulated trade sets and land them in Tier-2 ``trades``.

    python3 -m engine.build_trades                     # all three structures
    python3 -m engine.build_trades --strategy STR-THRU
    python3 -m engine.build_trades --years 2018 2019 --dry-run

This is the analog layer's input and the backtest's substrate, produced by the
same :mod:`engine.replay` code path the scorer calls for a live event.

The legacy S1/S2/S3 rows Phase 0 normalized are **kept**, not replaced. They
are a different thing: worst-fill-only, and specified differently from the three
program structures (S1 is a calendar *straddle*, S3 enters at a calendar T−14).
Phase 0 refused to relabel them as CAL-P/STR-THRU/STR-RUNUP for exactly that
reason, and this preserves the distinction — engine rows carry
``provenance="engine.replay"``, legacy rows carry ``provenance="legacy:..."``,
and the analog layer reads only the former.

A ``--strategy`` run replaces **only the strategies it replayed**. Engine rows
for the others survive it. Anything else would make a partial rebuild a silent
delete of the strategies it did not name, and the table would look complete
afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from engine import replay
from engine.data import manifest, store
from engine.data.schemas import coerce
from engine.structures import STRUCTURES

#: Chains start in 2017 and the first full year of usable pairs is 2018.
FIRST_YEAR = 2017


def event_universe(years=None) -> pd.DataFrame:
    """Every calendar event with a known session, in the chain era.

    Unselected on purpose. The first entry in the HANDOFF's list of traps is
    that any statistic computed over a model-chosen candidate set is
    conditioned on future information about which names moved; the analog layer
    exists to answer "what happened to trades like this one", and it can only do
    that from a population nobody filtered.
    """
    years = list(years) if years is not None else list(range(FIRST_YEAR, 2027))
    events = store.read_table(
        "earnings_events",
        years=years,
        columns=["event_id", "ticker", "event_date", "session"],
    )
    return events[events["session"].notna()].reset_index(drop=True)


def build(strategies, years=None, dry_run: bool = False) -> dict:
    started = time.time()
    events = event_universe(years)
    print(f"event universe: {len(events):,} events with a session", flush=True)

    available = replay.available_chain_keys()
    print(f"chain store: {len(available):,} (ticker, date) chains", flush=True)

    results = []
    for strategy in strategies:
        result = replay.replay(strategy, events)
        results.append(result)

    report = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "years": [min(years), max(years)] if years else [FIRST_YEAR, 2026],
        "events": int(len(events)),
        "results": [r.as_dict() for r in results],
    }

    if dry_run:
        print("\n--dry-run: nothing written", flush=True)
        return report

    engine_rows = replay.to_trades_table(results)
    existing = store.read_table("trades")
    is_engine = existing["provenance"].astype(str).str.startswith("engine.replay")
    # A --strategy run rebuilds SOME strategies. Dropping every engine row and
    # writing back only the rebuilt ones would silently delete the others: one
    # `--strategy CND-P` and the STR-THRU, STR-RUNUP and CAL-P trade sets are
    # gone, with the table looking complete afterwards. Only the strategies
    # actually replayed are replaced.
    rebuilt = {r.strategy for r in results}
    kept = existing[~is_engine | ~existing["strategy"].isin(rebuilt)]
    n_legacy = int((~is_engine.loc[kept.index]).sum())
    n_other_engine = int(len(kept) - n_legacy)
    combined = pd.concat([coerce(kept, "trades"), coerce(engine_rows, "trades")],
                         ignore_index=True)
    store.write_table(combined, "trades")

    stats = store.table_stats("trades")
    print(
        f"\ntrades table: {stats.rows:,} rows "
        f"({n_legacy:,} legacy + {n_other_engine:,} engine rows for strategies "
        f"not rebuilt + {len(engine_rows):,} rebuilt {sorted(rebuilt)})",
        flush=True,
    )

    snapshot = manifest.write_snapshot()
    manifest.write_manifest()
    report["trades_rows"] = int(stats.rows)
    report["legacy_rows"] = n_legacy
    report["kept_engine_rows"] = n_other_engine
    report["rebuilt_strategies"] = sorted(rebuilt)
    report["engine_rows"] = int(len(engine_rows))
    report["snapshot"] = snapshot
    report["elapsed_s"] = round(time.time() - started, 1)
    print(f"snapshot {snapshot}", flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--strategy", action="append", choices=sorted(STRUCTURES),
        help="replay only this strategy (repeatable); default is all three",
    )
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="plan and price, write nothing")
    ap.add_argument("--json", default=None, help="write the run report here")
    args = ap.parse_args(argv)

    strategies = args.strategy or sorted(STRUCTURES)
    report = build(strategies, years=args.years, dry_run=args.dry_run)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
