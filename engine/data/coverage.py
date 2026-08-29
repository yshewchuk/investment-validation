"""Coverage analysis and the Phase 0 data audit report.

Answers the question the Sep-1 pull plan depends on: *for which events do we
already have the chains a strategy needs, and where are the holes?*

Three coverage notions, because a strategy needs a chain on a specific date,
not merely somewhere near the event:

``entry``  a chain on the last pre-print close (what every structure opens on)
``exit``   a chain on the first post-print close (what through-print structures close on)
``t14``    a chain around T−14 (what the run-up structure opens on)

Coverage is reported by year × market-cap bucket, because that is how the plan
slices the universe, and split call/put because much of the existing cache was
pulled straddle-centric while CAL-P needs puts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine import paths
from engine.calendar import trading_calendar
from engine.data import store

__all__ = [
    "MCAP_BUCKETS",
    "bucket_mcap",
    "event_chain_coverage",
    "coverage_matrix",
    "side_coverage",
    "dte_availability",
    "render_audit",
]

#: The plan's universe slices. ``1–10B`` is the claimed +5.3% pocket.
MCAP_BUCKETS = (
    ("<1B", 0.0, 1e9),
    ("1-10B", 1e9, 1e10),
    (">10B", 1e10, np.inf),
)


def bucket_mcap(values) -> pd.Series:
    """Label market caps by the plan's slices; null stays ``unknown``."""
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    out = pd.Series("unknown", index=series.index, dtype="object")
    for label, lo, hi in MCAP_BUCKETS:
        out[series.notna() & (series >= lo) & (series < hi)] = label
    return out


def _log(message: str) -> None:
    print(f"  [coverage] {message}", flush=True)


# --------------------------------------------------------------------------
# per-event coverage
# --------------------------------------------------------------------------


def event_chain_coverage(
    events: pd.DataFrame | None = None,
    chains_index: pd.DataFrame | None = None,
    *,
    min_year: int = 2017,
    runup_offset: int = -14,
) -> pd.DataFrame:
    """One row per event with entry/exit/T−14 chain availability, by side.

    ``chains_index`` is the distinct ``(ticker, obs_date, right)`` set of the
    chain store — small enough to hold in memory even though the chain table
    itself is 15M rows.
    """
    if events is None:
        events = store.read_table("earnings_events")
    if chains_index is None:
        _log("indexing chain availability …")
        frames = []
        for _, chunk in store.iter_table(
            "option_chains", columns=["ticker", "obs_date", "right", "dte"]
        ):
            frames.append(chunk.drop_duplicates(["ticker", "obs_date", "right"]))
        chains_index = (
            pd.concat(frames, ignore_index=True).drop_duplicates(["ticker", "obs_date", "right"])
            if frames
            else pd.DataFrame(columns=["ticker", "obs_date", "right", "dte"])
        )
    _log(f"chain index: {len(chains_index):,} (ticker, date, side) combinations")

    events = events[events["year"] >= min_year].copy()
    # Only doubly-sourced events with a known session can have their trading
    # dates resolved, which is what coverage is measured against.
    events = events[events["session"].notna()]
    _log(f"events from {min_year}: {len(events):,} with a known session")

    calendar = trading_calendar()
    entry_dates, exit_dates, runup_dates = [], [], []
    for event_date, session in zip(events["event_date"], events["session"]):
        try:
            window = calendar.resolve_offsets(event_date, session, runup_offset, 1)
            entry_dates.append(window.last_pre_print)
            exit_dates.append(window.first_post_print)
            runup_dates.append(window.entry_date)
        except KeyError:
            entry_dates.append(pd.NaT)
            exit_dates.append(pd.NaT)
            runup_dates.append(pd.NaT)
    events["entry_date"] = entry_dates
    events["exit_date"] = exit_dates
    events["runup_date"] = runup_dates

    have = {
        (str(t), pd.Timestamp(d), str(r))
        for t, d, r in zip(
            chains_index["ticker"], chains_index["obs_date"], chains_index["right"]
        )
    }
    for label, column in (("entry", "entry_date"), ("exit", "exit_date"), ("t14", "runup_date")):
        for side, side_name in (("C", "call"), ("P", "put")):
            events[f"{label}_{side_name}"] = [
                (str(t), pd.Timestamp(d), side) in have if pd.notna(d) else False
                for t, d in zip(events["ticker"], events[column])
            ]
        events[f"{label}_any"] = events[f"{label}_call"] | events[f"{label}_put"]
        events[f"{label}_both"] = events[f"{label}_call"] & events[f"{label}_put"]

    # A through-the-print structure needs BOTH ends, on both sides for CAL-P.
    events["through_print_ready"] = events["entry_both"] & events["exit_both"]
    return events


def attach_mcap(events: pd.DataFrame, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the market-cap bucket for each event from the Tier-3 panel."""
    if panel is None:
        if not paths.PANEL.exists():
            events["mcap_bucket"] = "unknown"
            return events
        panel = pd.read_parquet(paths.PANEL, columns=["ticker", "date", "mcap_usd"])
    sized = panel.rename(columns={"date": "event_date"})
    out = events.merge(sized, on=["ticker", "event_date"], how="left")
    out["mcap_bucket"] = bucket_mcap(out["mcap_usd"])
    return out


# --------------------------------------------------------------------------
# aggregations
# --------------------------------------------------------------------------


def coverage_matrix(events: pd.DataFrame, column: str = "through_print_ready") -> pd.DataFrame:
    """Events × year × mcap bucket, as a coverage rate."""
    if events.empty:
        return pd.DataFrame()
    grid = events.pivot_table(
        index="year", columns="mcap_bucket", values=column, aggfunc="mean"
    )
    counts = events.pivot_table(
        index="year", columns="mcap_bucket", values=column, aggfunc="size"
    )
    return grid.round(4), counts


def side_coverage(events: pd.DataFrame) -> pd.DataFrame:
    """Call vs put availability — the gap CAL-P has to close."""
    rows = []
    for label in ("entry", "exit", "t14"):
        rows.append(
            {
                "point": label,
                "call": float(events[f"{label}_call"].mean()) if len(events) else 0.0,
                "put": float(events[f"{label}_put"].mean()) if len(events) else 0.0,
                "both": float(events[f"{label}_both"].mean()) if len(events) else 0.0,
                "either": float(events[f"{label}_any"].mean()) if len(events) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def dte_availability(sample_years: tuple[int, ...] = (2022, 2023, 2024)) -> pd.DataFrame:
    """Distribution of available DTEs, which bounds the back-leg grid."""
    rows = []
    for year, chunk in store.iter_table("option_chains", years=sample_years, columns=["dte"]):
        dte = pd.to_numeric(chunk["dte"], errors="coerce").dropna()
        for lo, hi in ((0, 2), (3, 7), (8, 14), (15, 21), (22, 30), (31, 45), (46, 10_000)):
            rows.append(
                {
                    "year": year,
                    "dte_bucket": f"{lo}-{hi}" if hi < 10_000 else f"{lo}+",
                    "rows": int(((dte >= lo) & (dte <= hi)).sum()),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    header = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    lines = [header, sep]
    for idx, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float) and np.isfinite(value):
                cells.append(floatfmt.format(value))
            elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                cells.append("—")
            else:
                cells.append(f"{value:,}" if isinstance(value, (int, np.integer)) else str(value))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines) + "\n"


def render_audit(
    events: pd.DataFrame,
    sanity: list | None = None,
    extra: dict[str, str] | None = None,
) -> str:
    from engine.data import manifest

    stats = manifest.collect_stats()
    ready_rate, counts = coverage_matrix(events, "through_print_ready")
    entry_rate, _ = coverage_matrix(events, "entry_both")
    sides = side_coverage(events)

    lines = [
        "# Phase 0 — Data Audit",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"Snapshot: `{manifest.snapshot_hash(stats)}`",
        "",
        "## Store inventory",
        "",
        "| Table | Rows | Years | Content hash |",
        "|---|---:|---|---|",
        *[
            f"| `{n}` | {s['rows']:,} | {s['years']} | `{s['content_hash'][:16]}…` |"
            for n, s in sorted(stats.items())
        ],
        "",
        "## Chain coverage — events × year × market-cap bucket",
        "",
        "`through_print_ready` = a chain exists on BOTH the last pre-print close "
        "and the first post-print close, on BOTH sides. That is what a "
        "through-the-print structure actually needs; a looser definition "
        "(\"some chain near the event\") overstates readiness.",
        "",
        _table(ready_rate),
        "",
        "Event counts behind those rates:",
        "",
        _table(counts, "{:.0f}"),
        "",
        "### Entry-date coverage (both sides)",
        "",
        _table(entry_rate),
        "",
        "## Call vs put coverage",
        "",
        "Much of the cache was pulled straddle-centric. CAL-P is a put "
        "structure, so the put column is the binding constraint on it.",
        "",
        "| Point | Call | Put | Both | Either |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {r['point']} | {r['call']:.3f} | {r['put']:.3f} | {r['both']:.3f} | {r['either']:.3f} |"
            for _, r in sides.iterrows()
        ],
        "",
    ]

    if sanity:
        lines += [
            "## Price-sanity battery",
            "",
            "| Check | Result | Checked | Failed | Detail |",
            "|---|---|---:|---:|---|",
            *[
                f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.n_checked:,} | "
                f"{c.n_failed:,} | {c.detail} |"
                for c in sanity
            ],
            "",
        ]

    for title, body in (extra or {}).items():
        lines += [f"## {title}", "", body, ""]
    return "\n".join(lines)
