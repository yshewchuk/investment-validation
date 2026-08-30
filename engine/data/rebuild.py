"""Rebuild Tier 2 from Tier 1, then Tier 3 from Tier 2.

    python3 -m engine.data.rebuild                    # everything
    python3 -m engine.data.rebuild --table chains     # one table
    python3 -m engine.data.rebuild --sample 20        # a 20-unit slice, for tests

Contracts this orchestrator upholds:

* **No network.** Normalizers read the raw cache only. A Tier-2 rebuild works
  with the network unplugged, which is what makes it a regression test rather
  than another data pull.
* **Idempotent.** Running twice produces byte-identical partitions.
* **Validated.** Every frame passes the ingestion battery before it lands;
  failures quarantine the raw file and exclude the rows, loudly.
* **Observable.** Progress lines at least once a minute — a silent long job is
  indistinguishable from a hung one, especially on a host that sleeps.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from engine import paths
from engine.data import manifest, store, validate
from engine.data.features import panel as panel_mod
from engine.data.normalize import n_chains, n_daily, n_events, n_securities, n_trades

__all__ = ["rebuild", "RebuildResult", "TABLE_ORDER"]

#: ``daily_market`` before ``securities`` (which is derived from it) and before
#: the panel (which joins it).
TABLE_ORDER = ("events", "daily", "securities", "chains", "trades", "panel")


@dataclass
class RebuildResult:
    tables: dict = field(default_factory=dict)
    reports: dict = field(default_factory=dict)
    snapshot: str | None = None
    elapsed_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "tables": self.tables,
            "reports": self.reports,
            "snapshot": self.snapshot,
            "elapsed_s": round(self.elapsed_s, 1),
        }


class _Progress:
    """Heartbeat that guarantees a line at least every ``every`` seconds."""

    def __init__(self, label: str, total: int | None = None, every: float = 30.0):
        self.label = label
        self.total = total
        self.every = every
        self.started = time.time()
        self.last = 0.0

    def tick(self, i: int, extra: str = "", force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        elapsed = now - self.started
        if self.total:
            rate = i / elapsed if elapsed > 0 else 0
            eta = (self.total - i) / rate if rate > 0 else 0
            print(
                f"  [{self.label}] {i}/{self.total} ({100*i/self.total:.1f}%) "
                f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s {extra}",
                flush=True,
            )
        else:
            print(f"  [{self.label}] {i} done, {elapsed:.0f}s {extra}", flush=True)

    def done(self, i: int, extra: str = "") -> None:
        print(
            f"  [{self.label}] complete: {i} unit(s) in {time.time()-self.started:.0f}s {extra}",
            flush=True,
        )


def _banner(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


# --------------------------------------------------------------------------
# per-table builders
# --------------------------------------------------------------------------


def build_events_table(sample: int | None = None) -> dict:
    _banner("earnings_events")
    frame, report = n_events.normalize()
    if sample:
        frame = frame.head(sample)
    store.write_table(frame, "earnings_events")
    stats = store.table_stats("earnings_events")
    print(f"  wrote {stats.rows:,} rows / {len(stats.years)} partitions", flush=True)
    report["sessions"] = n_events.session_coverage(frame)
    return report


def build_daily_table(sample: int | None = None) -> dict:
    _banner("daily_market")
    tickers = n_daily.list_tickers()
    if sample:
        tickers = tickers[:sample]
    progress = _Progress("daily", len(tickers))
    batch = validate.ValidationReport(table="daily_market")
    skipped: list[str] = []

    with store.PartitionedWriter("daily_market") as writer:
        for i, ticker in enumerate(tickers, 1):
            frame, report = n_daily.normalize_ticker(ticker)
            if frame.empty:
                skipped.append(ticker)
            else:
                clean, vreport = validate.validate_daily(
                    frame, source_file=f"orats/summaries/{ticker}.json.gz"
                )
                batch.merge(vreport)
                writer.add(clean)
            progress.tick(i, f"rows={writer.rows_written:,}")
    progress.done(len(tickers), f"rows={writer.rows_written:,}, skipped={len(skipped)}")

    stats = store.table_stats("daily_market")
    print(f"  wrote {stats.rows:,} rows / {len(stats.years)} partitions", flush=True)
    return {
        "tickers": len(tickers),
        "skipped": len(skipped),
        "rows": stats.rows,
        "validation": batch.summary(),
    }


def build_securities_table() -> dict:
    _banner("securities")
    columns = ["ticker", "date", "year", "mcap_usd"]
    frames = []
    for year, chunk in store.iter_table("daily_market", columns=columns):
        frames.append(chunk)
    if not frames:
        print("  no daily_market rows — skipping", flush=True)
        return {"rows": 0}
    daily = pd.concat(frames, ignore_index=True)
    frame, report = n_securities.normalize_from_daily(daily)
    store.write_table(frame, "securities")
    stats = store.table_stats("securities")
    print(f"  wrote {stats.rows:,} rows / {len(stats.years)} partitions", flush=True)
    return report


def build_chains_table(sample: int | None = None) -> dict:
    """Build ``option_chains`` from BOTH pull generations.

    Legacy wrapped files and Tier-1 fetch-store payloads are one stream here.
    Reading only the legacy tree would make the fetch store write-only: the
    Sep-1 pull would land 16,000 calls of chains that no rebuild could see, and
    a restored machine (which has no legacy tree at all) would rebuild empty.
    """
    _banner("option_chains")

    # (label, parse callable) pairs, legacy first so its rows win the dedupe.
    sources: list[tuple[str, object]] = [
        (path.name, ("legacy", path)) for path in n_chains.iter_chain_files()
    ]
    fetch_stats: dict = {}
    fetch_sources = n_chains.iter_fetch_sources(stats=fetch_stats)
    sources += [(s.source_id, ("fetch", s)) for s in fetch_sources]
    if sample:
        sources = sources[:sample]
    print(
        f"  sources: {len(sources):,} "
        f"({len(sources) - len(fetch_sources):,} legacy file(s), "
        f"{len(fetch_sources):,} fetch-store payload(s))",
        flush=True,
    )

    progress = _Progress("chains", len(sources))
    batch = validate.ValidationReport(table="option_chains")
    kinds: dict[str, int] = {}
    unreadable: list[str] = []

    with store.PartitionedWriter("option_chains") as writer:
        for i, (label, (kind, handle)) in enumerate(sources, 1):
            try:
                if kind == "legacy":
                    frame, report = n_chains.normalize_file(handle)
                else:
                    frame, report = n_chains.normalize_fetch_rows(handle)
            except (ValueError, OSError, EOFError) as exc:
                unreadable.append(label)
                validate.quarantine(label, f"unreadable raw payload: {type(exc).__name__}: {exc}")
                progress.tick(i)
                continue
            if not frame.empty:
                clean, vreport = validate.validate_chains(frame, source_file=label)
                batch.merge(vreport)
                kinds[report["chain_kind"]] = kinds.get(report["chain_kind"], 0) + len(clean)
                writer.add(clean)
            progress.tick(i, f"rows={writer.rows_written:,}")
        progress.done(len(sources), f"rows={writer.rows_written:,}")

        # Entry-date and calendar (`_c2_`) pulls overlap on the same trade date,
        # so the same contract arrives from two payloads. Per-source dedupe
        # cannot see that; this pass can.
        print("  deduplicating on the primary key …", flush=True)
        removed = writer.finalize(dedupe=True)
        print(f"  removed {removed:,} duplicate-key row(s)", flush=True)

    stats = store.table_stats("option_chains")
    print(f"  wrote {stats.rows:,} rows / {len(stats.years)} partitions", flush=True)
    return {
        "sources": len(sources),
        "legacy_files": len(sources) - len(fetch_sources),
        "fetch_payloads": len(fetch_sources),
        "fetch_empty": fetch_stats.get("empty", 0),
        "fetch_unrecognized": fetch_stats.get("unrecognized", 0),
        "fetch_unreadable": fetch_stats.get("unreadable", 0),
        "unreadable": len(unreadable),
        "duplicates_removed": removed,
        "rows": stats.rows,
        "rows_by_kind": kinds,
        "validation": batch.summary(),
    }


def build_trades_table() -> dict:
    _banner("trades")
    frame, reports = n_trades.normalize_all()
    store.write_table(frame, "trades")
    stats = store.table_stats("trades")
    print(f"  wrote {stats.rows:,} rows / {len(stats.years)} partitions", flush=True)
    return {"sets": reports, "rows": stats.rows}


def build_panel_table() -> dict:
    _banner("tier 3 — causal panel")
    frame = panel_mod.build_panel()
    out = paths.assert_writable(paths.PANEL)
    out.parent.mkdir(parents=True, exist_ok=True)
    if store.HAVE_PARQUET:
        frame.to_parquet(out, engine="pyarrow", index=False, compression="snappy")
    else:  # pragma: no cover - fallback path
        out = out.with_suffix(".csv.gz")
        frame.to_csv(out, index=False, compression="gzip")
    print(f"  wrote {len(frame):,} rows × {len(frame.columns)} cols → {out}", flush=True)
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "years": f"{int(frame['year'].min())}–{int(frame['year'].max())}",
        "path": str(out),
    }


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def rebuild(tables: tuple[str, ...] = TABLE_ORDER, sample: int | None = None) -> RebuildResult:
    started = time.time()
    paths.ensure_dirs()
    result = RebuildResult()

    builders = {
        "events": lambda: build_events_table(sample),
        "daily": lambda: build_daily_table(sample),
        "securities": build_securities_table,
        "chains": lambda: build_chains_table(sample),
        "trades": build_trades_table,
        "panel": build_panel_table,
    }
    for name in TABLE_ORDER:
        if name not in tables:
            continue
        result.reports[name] = builders[name]()

    _banner("manifest + snapshot")
    stats = manifest.collect_stats()
    result.tables = stats
    result.snapshot = manifest.write_snapshot(stats)
    manifest_path = manifest.write_manifest(stats)
    print(f"  snapshot {result.snapshot}", flush=True)
    print(f"  manifest {manifest_path}", flush=True)

    result.elapsed_s = time.time() - started
    print(f"\nrebuild complete in {result.elapsed_s:.0f}s", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--table",
        action="append",
        choices=TABLE_ORDER,
        help="rebuild only this table (repeatable); default is all",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="limit each table to N source units (tickers/files) — for tests",
    )
    ap.add_argument("--json", default=None, help="write the run report to this path")
    args = ap.parse_args(argv)

    tables = tuple(args.table) if args.table else TABLE_ORDER
    result = rebuild(tables=tables, sample=args.sample)
    if args.json:
        Path(args.json).write_text(json.dumps(result.as_dict(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
