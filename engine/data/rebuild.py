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
* **Coverage-guarded.** When the selected tables include ``daily`` and
  ``sample`` is falsey (``None`` or ``0``), a full ``daily_market`` build
  preflights published-date coverage before any selected table builder,
  refusing with ``DAILY_MARKET_DATE_INCOMPLETE`` instead of replacing the table
  with a gap. Partial-table rebuilds without ``daily`` are unaffected.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from engine import paths
from engine.data import manifest, rebuild_incremental, store, validate
from engine.data.features import panel as panel_mod
from engine.data.features import tier4 as tier4_mod
from engine.data.normalize import n_chains, n_daily, n_events, n_option_daily, n_securities, n_trades
from engine.data.normalize.fetch_store import MARKET_WIDE_META_MAX_BYTES
from engine.data.rebuild_cache import InputCache

__all__ = ["rebuild", "finalize_rebuild", "finalize_current_store", "RebuildResult",
           "DailyMarketCompletenessError", "TABLE_ORDER"]

#: Fraction of the adjacent-date median a published date must reach.
DAILY_COMPLETENESS_FLOOR = 0.75


class DailyMarketCompletenessError(RuntimeError):
    """A full ``daily_market`` build has no source inventory or a gapped date.

    Raised by the preflight before :class:`~engine.data.store.PartitionedWriter`
    is constructed, so the refusal leaves the current curated table untouched
    instead of publishing a partial rebuild or an empty one.
    """

    code = "DAILY_MARKET_DATE_INCOMPLETE"

    def __init__(
        self,
        *,
        date: str | None,
        count: int,
        previous_count: int | None,
        next_count: int | None,
        threshold: float,
        reason: str | None = None,
        details: dict | None = None,
    ):
        self.date = date
        self.count = count
        self.previous_count = previous_count
        self.next_count = next_count
        self.neighbor_counts = (previous_count, next_count)
        self.threshold = threshold
        if reason is None:
            if previous_count is None and next_count is None:
                neighbors = "no adjacent published date"
            elif previous_count is None:
                neighbors = f"its only adjacent published date ({next_count})"
            elif next_count is None:
                neighbors = f"its only adjacent published date ({previous_count})"
            else:
                neighbors = (
                    f"the median of its adjacent published dates "
                    f"({previous_count} and {next_count})"
                )
            self.reason = (
                f"daily_market has {count} rows on {date}, below "
                f"{DAILY_COMPLETENESS_FLOOR:.0%} of {neighbors}; "
                f"threshold {threshold:.1f}"
            )
        else:
            self.reason = reason
        self.details = {
            "date": date,
            "count": count,
            "previous_count": previous_count,
            "next_count": next_count,
            "threshold": threshold,
            "reason": self.reason,
        }
        if details:
            self.details.update(details)
        super().__init__(f"{self.code}: {self.reason}")


#: ``daily_market`` before ``securities`` (which is derived from it) and before
#: the panel (which joins it).
TABLE_ORDER = (
    "events",
    "daily",
    "securities",
    "chains",
    "option_daily",
    "trades",
    "panel",
    "tier4",
)


@dataclass
class RebuildResult:
    tables: dict = field(default_factory=dict)
    reports: dict = field(default_factory=dict)
    snapshot: str | None = None
    elapsed_s: float = 0.0
    #: The wall-clock start of the build, stashed so a caller that rebuilds the
    #: tables with ``finalize=False`` can finalize later (see
    #: :func:`finalize_rebuild`). Not serialised: ``as_dict`` enumerates fields.
    started: float = 0.0

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
    audit = frame.attrs.get("reconciliation_audit")
    frame.attrs.clear()
    if sample:
        frame = frame.head(sample)
    store.write_table(frame, "earnings_events")
    if audit is not None:
        audit_path = paths.assert_writable(paths.EVENT_RECONCILIATION_REPORT)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_parquet(audit_path, engine="pyarrow", index=False, compression="snappy")
        report["reconciliation_audit"]["path"] = str(audit_path)
    stats = store.table_stats("earnings_events")
    print(f"  wrote {stats.rows:,} rows / {len(stats.years)} partitions", flush=True)
    report["sessions"] = n_events.session_coverage(frame)
    return report


def _normalize_daily(ticker: str, cache) -> tuple[pd.DataFrame, dict]:
    if cache is None:
        return n_daily.normalize_ticker(ticker)
    return rebuild_incremental.normalize_ticker_cached(ticker, cache)


def _rss_suffix() -> str:
    """Peak RSS suffix for progress boundaries; empty when unavailable."""
    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except (ImportError, AttributeError, OSError, ValueError):
        return ""
    return f", peak RSS {peak_mb:.0f} MB"


def _market_wide_snapshot_dates() -> list:
    """``tradeDate`` values from successful market-wide cached responses.

    Only the sidecar metadata is read — the cached bodies stay untouched. An
    HTTP-200 snapshot whose body holds no rows leaves no normalized trace, so
    its ``tradeDate`` is added to the count map as a zero-count candidate. That
    is what makes an empty market-wide snapshot visible to the preflight instead
    of silently missing from the dates it checks.
    """
    from engine.data.fetch import iter_cached

    dates: list = []
    for endpoint in ("hist/summaries", "hist/cores"):
        for entry in iter_cached(
            "orats", endpoint, max_meta_bytes=MARKET_WIDE_META_MAX_BYTES
        ):
            params = entry.params
            # Per-ticker responses are addressable by cache key and are appended
            # by _rows_for; only market-wide snapshots belong here.
            if params.get("ticker"):
                continue
            try:
                status = int(entry.meta.get("status", 0))
            except (TypeError, ValueError):
                continue
            if status != 200:
                continue
            day = params.get("tradeDate")
            if day:
                parsed = pd.to_datetime(day, errors="coerce")
                if pd.notna(parsed):
                    dates.append(parsed)
    return dates


def _preflight_daily_completeness(tickers: list[str], cache) -> None:
    """Refuse a full build whose published dates are missing most of the market.

    Counts are taken from exactly the rows the streaming writer will keep: each
    normalized frame goes through :func:`engine.data.validate.validate_daily`
    first, so rows that validation excludes do not inflate a date. The counting
    pass calls ``validate_daily(frame)`` without a source path, so it counts
    only the clean rows and cannot emit a quarantine flag; the write pass adds
    source attribution and can emit the one flag for the same frame.
    Successful market-wide fetch metadata contributes zero-count candidates,
    making an HTTP-200 empty snapshot visible.

    A date is compared with the median of its nearest published neighbours (two,
    so their mean; a first or last date has only one, so that count). A lone
    date with no neighbour at all is skipped. Running this before the writer
    matters: its constructor replaces the existing table.

    An empty ticker inventory is a no-source refusal, not a valid empty rebuild:
    replacing ``daily_market`` with zero rows would erase the table without any
    raw input having been read.
    """
    counts: dict = {}
    total = len(tickers)
    started = time.monotonic()
    if not tickers:
        raise DailyMarketCompletenessError(
            date=None,
            count=0,
            previous_count=None,
            next_count=None,
            threshold=0.0,
            reason=(
                "daily_market has no source tickers: no legacy per-ticker files "
                "or fetch-store daily rows were found; refusing to replace the "
                "table with an empty build"
            ),
            details={"tickers": 0, "dates": 0},
        )
    last = started - 30.0
    print(
        f"  [daily preflight] start 0/{total} tickers, elapsed 0s{_rss_suffix()}",
        file=sys.stderr,
        flush=True,
    )
    for i, ticker in enumerate(tickers, 1):
        frame, _ = _normalize_daily(ticker, cache)
        if not frame.empty:
            clean, _ = validate.validate_daily(frame)
            for day, n in clean.groupby("date").size().items():
                counts[day] = counts.get(day, 0) + int(n)
        now = time.monotonic()
        if now - last >= 30.0:
            last = now
            elapsed = now - started
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(
                f"  [daily preflight] {i}/{total} tickers, {elapsed:.0f}s elapsed, "
                f"{rate:.1f}/s, ETA {eta:.0f}s, dates={len(counts)}",
                file=sys.stderr,
                flush=True,
            )

    for day in _market_wide_snapshot_dates():
        counts.setdefault(day, 0)

    dates = sorted(counts)
    for i, day in enumerate(dates):
        previous = int(counts[dates[i - 1]]) if i > 0 else None
        following = int(counts[dates[i + 1]]) if i + 1 < len(dates) else None
        if previous is None and following is None:
            continue
        neighbours = [n for n in (previous, following) if n is not None]
        # One neighbour is the median by definition; two give their mean.
        threshold = DAILY_COMPLETENESS_FLOOR * (sum(neighbours) / len(neighbours))
        count = int(counts[day])
        if count < threshold:
            date = pd.Timestamp(day).date().isoformat()
            elapsed = time.monotonic() - started
            print(
                f"  [daily preflight] refused after {elapsed:.0f}s: date={date} "
                f"count={count} threshold={threshold:.1f}{_rss_suffix()}",
                file=sys.stderr,
                flush=True,
            )
            raise DailyMarketCompletenessError(
                date=date,
                count=count,
                previous_count=previous,
                next_count=following,
                threshold=threshold,
            )

    elapsed = time.monotonic() - started
    print(
        f"  [daily preflight] done in {elapsed:.0f}s: {total}/{total} tickers, "
        f"{len(dates)} date(s){_rss_suffix()}",
        file=sys.stderr,
        flush=True,
    )


def _daily_tickers(sample: int | None) -> list[str]:
    """The ticker universe for one daily build, sliced when sampling."""
    tickers = n_daily.list_tickers()
    return tickers[:sample] if sample else tickers


def _build_daily_table(tickers: list[str], cache) -> dict:
    """Build ``daily_market`` from ``tickers`` without running the preflight.

    Private on purpose: only :func:`build_daily_table` (which guards first) and
    :func:`rebuild` (which runs the guard once before any selected builder) may
    call it, so no public caller can bypass the completeness refusal.
    """
    _banner("daily_market")

    progress = _Progress("daily", len(tickers))
    batch = validate.ValidationReport(table="daily_market")
    skipped: list[str] = []

    with store.PartitionedWriter("daily_market") as writer:
        for i, ticker in enumerate(tickers, 1):
            frame, report = _normalize_daily(ticker, cache)
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


def build_daily_table(sample: int | None = None, cache=None) -> dict:
    """Build ``daily_market``; ``cache`` (an ``InputCache``) reuses per-file parses.

    A full public call (a falsey ``sample``: ``None`` or ``0``) runs the
    completeness preflight; a sampled call skips it because a partial slice is
    not a full build. There is deliberately no public switch to skip the guard
    on a full build. :func:`rebuild` runs the preflight once up front, before
    any selected table builder, then uses the private implementation helper so
    the scan is not repeated.
    """
    tickers = _daily_tickers(sample)
    if not sample:
        _preflight_daily_completeness(tickers, cache)
    return _build_daily_table(tickers, cache)


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


@dataclass
class _ChainResult:
    """What one chain source contributes: its frame, validation and kind."""

    chain_kind: str
    rows: int
    frame: pd.DataFrame | None
    validation: validate.ValidationReport


def _chain_result(frame: pd.DataFrame, report: dict, label: str, writer) -> _ChainResult | None:
    """Validate and coerce one parsed source; ``None`` when it has no rows."""
    if frame.empty:
        return None
    clean, vreport = validate.validate_chains(frame, source_file=label)
    return _ChainResult(report["chain_kind"], len(clean), writer.prepare(clean), vreport)


def _chain_stream(sample, cache, fetch_stats: dict):
    """``(sources, n_legacy, n_fetch)`` for the chains build."""
    n_fetch = n_chains.count_fetch_sources()
    if cache is not None:
        n_legacy = len(n_chains.iter_chain_files())
        return rebuild_incremental.chain_sources(cache, fetch_stats), n_legacy, n_fetch
    # (label, parse callable) pairs, legacy first so its rows win the dedupe.
    sources: list[tuple[str, object]] = [
        (path.name, ("legacy", path)) for path in n_chains.iter_chain_files()
    ]
    n_legacy = len(sources)

    # Chained lazily: the fetch payloads are parsed one at a time as the writer
    # consumes them, so peak memory is one payload rather than all of them.
    def _fetch_stream():
        for source in n_chains.iter_fetch_sources(stats=fetch_stats):
            yield (source.source_id, ("fetch", source))

    return itertools.chain(iter(sources), _fetch_stream()), n_legacy, n_fetch


def _parse_chain_source(kind: str, handle, cache) -> tuple[pd.DataFrame, dict]:
    if cache is not None:
        handle = handle[0]
    if kind == "legacy":
        return n_chains.normalize_file(handle)
    return n_chains.normalize_fetch_rows(handle)


def _chain_source_result(label, kind, handle, cache, writer):
    """``(result, ok)`` for one source; ``ok`` is False when it was unreadable."""
    if kind == "cached":
        for call in handle.validation.quarantine_calls if handle is not None else ():
            validate.quarantine(call[0], call[1], call[2])
        return handle, True
    try:
        frame, report = _parse_chain_source(kind, handle, cache)
    except (ValueError, OSError, EOFError) as exc:
        validate.quarantine(label, f"unreadable raw payload: {type(exc).__name__}: {exc}")
        if cache is not None:
            cache.note_uncached()
        return None, False
    result = _chain_result(frame, report, label, writer)
    if cache is not None:
        _source, sig, cache_path = handle
        cache.store(cache_path, sig, label, result)
    return result, True


def build_chains_table(sample: int | None = None, cache=None) -> dict:
    """Build ``option_chains`` from BOTH pull generations.

    Legacy wrapped files and Tier-1 fetch-store payloads are one stream here.
    Reading only the legacy tree would make the fetch store write-only: the
    Sep-1 pull would land 16,000 calls of chains that no rebuild could see, and
    a restored machine (which has no legacy tree at all) would rebuild empty.
    """
    _banner("option_chains")

    fetch_stats: dict = {}
    sources, n_legacy, n_fetch = _chain_stream(sample, cache, fetch_stats)
    total = n_legacy + n_fetch
    if sample:
        sources = itertools.islice(sources, sample)
        total = min(total, sample)
    print(
        f"  sources: {total:,} "
        f"({n_legacy:,} legacy file(s), {n_fetch:,} fetch-store payload(s))",
        flush=True,
    )

    progress = _Progress("chains", total)
    batch = validate.ValidationReport(table="option_chains")
    kinds: dict[str, int] = {}
    unreadable: list[str] = []

    with store.PartitionedWriter("option_chains") as writer:
        for i, (label, (kind, handle)) in enumerate(sources, 1):
            result, ok = _chain_source_result(label, kind, handle, cache, writer)
            if not ok:
                unreadable.append(label)
                progress.tick(i)
                continue
            if result is not None:
                batch.merge(result.validation)
                kinds[result.chain_kind] = kinds.get(result.chain_kind, 0) + result.rows
                writer.add_prepared(result.frame)
            progress.tick(i, f"rows={writer.rows_written:,}")
        progress.done(total, f"rows={writer.rows_written:,}")

        # Entry-date and calendar (`_c2_`) pulls overlap on the same trade date,
        # so the same contract arrives from two payloads. Per-source dedupe
        # cannot see that; this pass can.
        print("  deduplicating on the primary key …", flush=True)
        removed = writer.finalize(dedupe=True)
        print(f"  removed {removed:,} duplicate-key row(s)", flush=True)

    stats = store.table_stats("option_chains")
    print(f"  wrote {stats.rows:,} rows / {len(stats.years)} partitions", flush=True)
    return {
        "sources": total,
        "legacy_files": n_legacy,
        "fetch_payloads": n_fetch,
        "fetch_empty": fetch_stats.get("empty", 0),
        "fetch_unrecognized": fetch_stats.get("unrecognized", 0),
        "fetch_unreadable": fetch_stats.get("unreadable", 0),
        "unreadable": len(unreadable),
        "duplicates_removed": removed,
        "rows": stats.rows,
        "rows_by_kind": kinds,
        "validation": batch.summary(),
    }


def build_option_daily_table(sample: int | None = None) -> dict:
    """Build ``option_daily`` from every cached Polygon daily-aggs payload.

    Real traded bars (close/VWAP/volume/trade count) for the contracts the
    simulated trades touch, 2024-08-19 onward — the evidence the fill model is
    calibrated against. Independent of every other table, so it rebuilds in
    isolation with ``--table option_daily``.
    """
    _banner("option_daily")
    fetch_stats: dict = {}
    sources = list(n_option_daily.iter_aggs_sources(stats=fetch_stats))
    if sample:
        sources = sources[:sample]
    print(
        f"  sources: {len(sources):,} aggs payload(s) "
        f"({fetch_stats.get('other', 0)} other polygon payload(s) skipped)",
        flush=True,
    )

    progress = _Progress("option_daily", len(sources))
    failed: list[str] = []
    excluded_total = 0

    with store.PartitionedWriter("option_daily") as writer:
        for i, entry in enumerate(sources, 1):
            try:
                frame, report = n_option_daily.normalize_entry(entry)
            except (ValueError, OSError, EOFError) as exc:
                failed.append(entry.source_id)
                validate.quarantine(
                    entry.source_id, f"unreadable raw payload: {type(exc).__name__}: {exc}"
                )
                progress.tick(i)
                continue
            if not frame.empty:
                excluded_total += int(report.get("excluded", 0))
                if report.get("excluded"):
                    validate.quarantine(
                        entry.source_id,
                        "option_daily row validation",
                        {"excluded": int(report["excluded"]), "rows_in": report.get("rows_in", 0)},
                    )
                writer.add(frame)
            elif report.get("reason") and report["reason"] != "empty results":
                # A payload that parsed to nothing for a reason worth naming —
                # unreadable body, unrecognized envelope, unparseable contract.
                # A genuine zero-trade contract ("empty results") costs nothing
                # extra and is only counted.
                failed.append(entry.source_id)
                validate.quarantine(
                    entry.source_id,
                    f"option_daily payload parsed to nothing: {report['reason']}",
                )
            progress.tick(i, f"rows={writer.rows_written:,}")
        progress.done(len(sources), f"rows={writer.rows_written:,}")

        print("  deduplicating on the primary key …", flush=True)
        removed = writer.finalize(dedupe=True)
        print(f"  removed {removed:,} duplicate-key row(s)", flush=True)

    stats = store.table_stats("option_daily")
    print(f"  wrote {stats.rows:,} rows / {len(stats.years)} partitions", flush=True)
    return {
        "sources": len(sources),
        "scanned": fetch_stats.get("scanned", 0),
        "other_payloads_skipped": fetch_stats.get("other", 0),
        "failed_payloads": len(failed),
        "rows_excluded_by_validation": excluded_total,
        "duplicates_removed": removed,
        "rows": stats.rows,
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


def build_tier4_table(since=None) -> dict:
    """Tier 4 — the feature-model forecasts.

    Runs after the panel because it reads it, and before the manifest because
    the snapshot records its hash. Tier 3 changing at date D makes every
    forecast from D onward stale — not only the rows whose own inputs moved —
    so a partial Tier-3 rebuild must be followed by ``--since D`` here rather
    than by nothing.
    """
    _banner("tier 4 — feature-model forecasts")
    report = tier4_mod.build_table(since=since)
    # One line per producer, not a single total. The table stopped being
    # single-producer when Tier 4 generalized to N of them, and this caller was
    # not updated with it — so a rebuild that had done all 446 fold fits died on
    # `KeyError: 'with_forecast'` while printing its own success, after the
    # parquet was already on disk but before the snapshot recorded its hash.
    print(f"  wrote {report['rows']:,} rows → {report['path']}", flush=True)
    for name, stats in report.get("producers", {}).items():
        n = stats["with_forecast"]
        print(f"    {name}: {n:,} with a forecast "
              f"({n / max(report['rows'], 1):.1%})", flush=True)
    return report


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def _with_cache(table: str, build, incremental: bool):
    """Run ``build(cache)``, with a per-input cache only when ``incremental``."""
    if not incremental:
        return build(None)
    cache = InputCache(table)
    report = build(cache)
    cache.commit()
    print(f"  {cache.summary()}", flush=True)
    return report


def _build_daily_from_rebuild(tickers, incremental: bool, cache) -> dict:
    """Build ``daily_market`` after :func:`rebuild` has run the preflight once."""
    if incremental:
        report = _build_daily_table(tickers, cache)
        cache.commit()
        print(f"  {cache.summary()}", flush=True)
        return report
    return _build_daily_table(tickers, cache)


def rebuild(
    tables: tuple[str, ...] = TABLE_ORDER,
    sample: int | None = None,
    tier4_since=None,
    incremental: bool = False,
    finalize: bool = True,
) -> RebuildResult:
    """Rebuild the named tables.

    ``incremental=True`` (opt-in; the legacy nightly passes it) reuses cached
    per-input parses for ``daily_market`` and ``option_chains`` -- see
    :mod:`engine.data.rebuild_cache`. The outputs are byte-identical to the
    default full rebuild, which neither reads nor writes that cache.

    ``sample`` cannot be combined with ``incremental``: a sampled run sees only
    part of the inputs, so committing its manifest would prune the cache of
    every input the sample skipped.

    ``finalize=False`` returns after the builders, leaving the manifest and
    snapshot unwritten and stashing ``started`` on the result for a later
    :func:`finalize_rebuild`. A caller that must not publish a snapshot for a
    bad build (the nightly, on a gapped Tier-4) uses this and only finalizes
    once the result checks out; the default path is unchanged.
    """
    if incremental and sample:
        raise ValueError("--sample cannot be combined with --incremental")
    started = time.time()
    paths.ensure_dirs()
    result = RebuildResult()

    selected = [name for name in TABLE_ORDER if name in tables]
    daily_cache = None
    daily_tickers = None
    if "daily" in selected:
        daily_tickers = _daily_tickers(sample)
        if not sample:
            # Refuse a gapped full build before any table builder runs: events
            # and the others would otherwise land before the daily check could
            # stop it. A sampled or daily-excluding rebuild is partial by
            # design and does not run this guard.
            if incremental:
                daily_cache = InputCache("daily_market")
            _preflight_daily_completeness(daily_tickers, daily_cache)

    builders = {
        "events": lambda: build_events_table(sample),
        "daily": lambda: _build_daily_from_rebuild(daily_tickers, incremental, daily_cache),
        "securities": build_securities_table,
        "chains": lambda: _with_cache(
            "option_chains", lambda c: build_chains_table(sample, c), incremental),
        "option_daily": lambda: build_option_daily_table(sample),
        "trades": build_trades_table,
        "panel": build_panel_table,
        "tier4": lambda: build_tier4_table(tier4_since),
    }
    for name in TABLE_ORDER:
        if name not in tables:
            continue
        result.reports[name] = builders[name]()

    result.started = started
    if not finalize:
        result.elapsed_s = time.time() - started
        return result
    return finalize_rebuild(result)


def finalize_rebuild(result: RebuildResult) -> RebuildResult:
    """Write the manifest and snapshot for a built result and stamp elapsed.

    Split from :func:`rebuild` so a caller can build the tables first and only
    publish a snapshot after deciding the result is sound -- the nightly
    defers finalizing the incremental Tier-3/4 build until the Tier-4 gap check
    passes, so a night whose full repair is refused leaves no new snapshot behind.
    """
    _banner("manifest + snapshot")
    stats = manifest.collect_stats()
    result.tables = stats
    # Write the manifest BEFORE the snapshot: the snapshot is the file the
    # native import reads, so it must be the last write. A manifest failure
    # then leaves no snapshot for the night to import.
    manifest_path = manifest.write_manifest(stats)
    result.snapshot = manifest.write_snapshot(stats)
    print(f"  snapshot {result.snapshot}", flush=True)
    print(f"  manifest {manifest_path}", flush=True)

    result.elapsed_s = time.time() - result.started
    print(f"\nrebuild complete in {result.elapsed_s:.0f}s", flush=True)
    return result


def finalize_current_store() -> RebuildResult:
    """Write the manifest and SNAPSHOT of the store as found.

    Called by ``engine.dashboard.nightly.mark_tiers_degraded`` on any tiers
    degrade; when refresh ran it also covers the refresh rebuild's deferred
    snapshot. The refresh rebuild was told
    ``finalize=False`` because the tiers step would finalize, and a degraded
    tiers step still scores the store as it is found (the refreshed store when
    refresh ran), so the SNAPSHOT of that store is written here.
    """
    return finalize_rebuild(RebuildResult(started=time.time()))


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
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="reuse cached per-input parses for daily/chains (set "
        "INVESTING_PLAN_REBUILD_FULL=1 to force a full parse that rewrites the cache)",
    )
    ap.add_argument("--json", default=None,
                    help="write the run report to this path (written atomically; "
                    "a failed write keeps the previous report)")
    ap.add_argument(
        "--tier4-since",
        default=None,
        help="rebuild Tier 4 only from this date's fold forward, carrying earlier "
        "rows over; the correct follow-up to a Tier-2 correction at that date",
    )
    args = ap.parse_args(argv)

    if args.incremental and args.sample:
        ap.error("--sample cannot be combined with --incremental")

    tables = tuple(args.table) if args.table else TABLE_ORDER
    result = rebuild(tables=tables, sample=args.sample, tier4_since=args.tier4_since,
                     incremental=args.incremental)
    if args.json:
        manifest.write_text_atomic(Path(args.json),
                                   json.dumps(result.as_dict(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
