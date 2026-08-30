#!/usr/bin/env python3
"""Generate the Phase 0 data audit report.

    python3 checks/phase0_audit.py [--report reports/phase0_data_audit.md]

Runs the coverage analysis and the price-sanity battery over the built store,
then renders the audit the plan calls for: coverage heatmaps, call-vs-put
availability, DTE availability, the sanity checks re-run on the current data,
row counts + checksums, and a quota-ledger reconciliation.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import paths  # noqa: E402
from engine.data import coverage, store, validate  # noqa: E402


def quota_reconciliation() -> str:
    """Reconcile the ORATS quota ledger against the logged calls."""
    legacy_log = paths.RAW_ORATS_QUOTA_LOG
    lines = []
    if legacy_log.exists():
        rows = list(csv.DictReader(open(legacy_log, newline="")))
        remaining = [
            int(float(r["quota_remaining"]))
            for r in rows
            if (r.get("quota_remaining") or "").strip() not in ("", "None")
        ]
        by_endpoint: dict[str, int] = {}
        for row in rows:
            by_endpoint[row.get("endpoint", "?")] = by_endpoint.get(row.get("endpoint", "?"), 0) + 1
        lines += [
            f"- Legacy ledger `{legacy_log.relative_to(paths.ROOT)}`: "
            f"**{len(rows):,} logged calls**",
            f"- Last reported remaining: **{remaining[-1]:,}**" if remaining else
            "- No quota headers present in the ledger (ORATS omits them on CDN-cached responses)",
            "- Calls by endpoint: "
            + ", ".join(f"`{k}` {v:,}" for k, v in sorted(by_endpoint.items(), key=lambda kv: -kv[1])),
        ]
        if remaining:
            spent = 20_000 - remaining[-1]
            lines.append(
                f"- Implied spend this cycle: **{spent:,} / 20,000**; "
                f"logged rows {len(rows):,} "
                f"({'consistent' if abs(spent - len(rows)) < 0.25 * max(spent, 1) else 'DIVERGENT — investigate'})"
            )
    else:
        lines.append("- No legacy quota ledger found.")

    new_log = paths.QUOTA_LOG
    if new_log.exists():
        rows = list(csv.DictReader(open(new_log, newline="")))
        lines.append(f"- Tier-1 wrapper ledger: **{len(rows):,} metered calls**")
    else:
        lines.append(
            "- Tier-1 wrapper ledger: **empty** — no call has yet gone through "
            "`engine.data.fetch`. The wrapper's live network path is therefore "
            "UNVERIFIED against the real API; the Sep-1 pull is its first "
            "exercise. Everything else about it is covered by the test suite."
        )
    return "\n".join(lines) + "\n"


def sanity_battery(sample_years: tuple[int, ...] = (2023, 2024, 2025)) -> list:
    """Re-run the price-sanity checks on the built store."""
    checks = []
    print("  spot vs yfinance …", flush=True)
    daily = store.read_table(
        "daily_market", years=list(sample_years), columns=["ticker", "date", "spot"]
    )
    checks.append(validate.spot_vs_yfinance(daily, sample_frac=0.02, max_tickers=150))
    print(f"    {checks[-1]}", flush=True)

    print("  straddle mids vs Polygon real trades …", flush=True)
    frames = []
    for _, chunk in store.iter_table(
        "option_chains",
        years=[2024, 2025],
        columns=["ticker", "obs_date", "expiry", "strike", "right", "mid"],
    ):
        frames.append(chunk)
    chains = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    checks.append(validate.straddle_vs_polygon(chains))
    print(f"    {checks[-1]}", flush=True)
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(paths.REPORTS / "phase0_data_audit.md"))
    ap.add_argument("--min-year", type=int, default=2017)
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args(argv)

    started = time.time()
    print("Phase 0 data audit", flush=True)

    events = coverage.event_chain_coverage(min_year=args.min_year)
    events = coverage.attach_mcap(events)
    print(f"  events analysed: {len(events):,}", flush=True)

    sanity = [] if args.skip_sanity else sanity_battery()

    print("  DTE availability …", flush=True)
    dte = coverage.dte_availability()
    dte_table = (
        dte.pivot_table(index="dte_bucket", columns="year", values="rows", aggfunc="sum")
        if len(dte)
        else pd.DataFrame()
    )

    extra = {
        "DTE availability (rows by bucket)": coverage._table(dte_table, "{:.0f}"),
        "Quota ledger reconciliation": quota_reconciliation(),
    }

    # Emitted through the Phase 4 generator, like every other result in the
    # program: the audit supplies content, the generator owns the frame
    # (provenance, checklist, glossary) so this report is regenerable and
    # reads in the same units as the experiment reports.
    from engine.report import Report, build_provenance  # noqa: E402

    ready = float(events["through_print_ready"].mean()) if len(events) else 0.0
    path = paths.assert_writable(Path(args.report))
    context = {
        "kind": "audit",
        "spec": {"id": "PHASE-0", "title": "Data audit — coverage, sanity, inventory",
                 "type": "descriptive",
                 "hypothesis": "The built store carries the chains the three "
                               "structures need, and the prices in it survive the "
                               "sanity battery."},
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {},
        "checklist": [],
        "provenance": build_provenance(seeds={}, input_files=[]),
        "survivorship_note": "",
        "calibration": None,
        "funnel": [{"stage": "events analysed", "events": int(len(events)),
                    "note": f"{args.min_year}+ with a known session", "headline": True},
                   {"stage": "through-print ready", "events": int(events["through_print_ready"].sum())
                    if len(events) else 0,
                    "note": "a chain on BOTH ends, on BOTH sides"}],
        "extra_sections": coverage.audit_sections(events, sanity, extra),
    }
    # The plan pins this filename, so the generator renders straight to it.
    Report(context).write(path.parent, filename=path.name)
    print(f"\n  report → {path}  ({time.time()-started:.0f}s)", flush=True)

    print(f"  through-print-ready coverage (all slices, {args.min_year}+): {ready:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
