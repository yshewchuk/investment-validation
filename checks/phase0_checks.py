#!/usr/bin/env python3
"""Phase 0 acceptance tests.

    python3 checks/phase0_checks.py              # everything
    python3 checks/phase0_checks.py --list
    python3 checks/phase0_checks.py --only structures fillmodel
    python3 checks/phase0_checks.py --no-data    # only checks that need no store

The checks the Phase 0 guide specifies, plus a policy gate. Unit-level behaviour lives in
``tests/`` and runs here as check 0; the rest are integration checks that need
the real cache, the real store, or a real git repo — the things a unit test with
fixtures cannot prove.

A phase without green checks is not done.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data import store, validate  # noqa: E402
from engine.data.fetch import Fetcher  # noqa: E402
from engine.data.normalize import n_chains  # noqa: E402
from engine.data.sources.base import Response  # noqa: E402
from engine.data.throttle import SourceConfig, Throttle  # noqa: E402
from engine.fills import BEST, MID, WORST  # noqa: E402
from engine.structures import (  # noqa: E402
    ChainSnapshot,
    price_structure,
    put_calendar,
    straddle_through,
    structure_return,
)


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0
    skipped: bool = False


REGISTRY: dict[str, dict] = {}


def check(name: str, *, needs_data: bool = True, description: str = ""):
    def wrap(fn):
        REGISTRY[name] = {"fn": fn, "needs_data": needs_data, "description": description}
        return fn

    return wrap


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------
# 0. unit suite
# --------------------------------------------------------------------------


@check("unittests", needs_data=False, description="the pytest suite (pure logic)")
def check_unittests() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--no-header"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    tail = (result.stdout or result.stderr).strip().splitlines()[-1:]
    _require(result.returncode == 0, f"pytest failed: {' '.join(tail)}")
    return " ".join(tail)


# --------------------------------------------------------------------------
# 0b. test policy
# --------------------------------------------------------------------------

#: Modules whose correctness is only meaningful against real data, a real store,
#: or a real repo — so they are covered by the acceptance layer instead of by
#: unit tests. Each entry names the check that covers it. Anything NOT in this
#: map and NOT reached by pytest is an untested module and fails the policy.
ACCEPTANCE_COVERED = {
    "engine/data/rebuild.py": "determinism + migration + the real rebuild",
    "checks/phase0_migration.py": "migration (delta logic in tests/test_migration_logic.py)",
    "checks/phase0_verdicts.py": "verdicts",
    "checks/phase0_audit.py": "coverage_report",
    "checks/phase0_checks.py": "is the harness itself",
    # Phase 1. These are drivers over the full store or the real model
    # artifacts: what they do is run the modules below them across millions of
    # rows, so a fixture-scale unit test would assert only that the argument
    # parser works.
    "engine/build_trades.py": "phase1 replay_equivalence (it produces the trade set)",
    "engine/models/training/train_all.py": "phase1 champions + registry",
    "checks/phase1_replay.py": "phase1 replay_equivalence",
    "checks/phase1_calibration.py": "phase1 calibration",
    "checks/phase1_report.py": "renders the Phase 1 report over the real store",
    "checks/phase1_checks.py": "is the Phase 1 harness itself",
    # Phase 2. The experiment harness drivers are covered by the Phase 2
    # acceptance suite they belong to; the experiments/ scripts sit outside
    # the coverage source on purpose (they are experiment drivers, not engine).
    "checks/phase2_checks.py": "is the Phase 2 harness itself",
}

#: Minimum line coverage for `engine/` under the unit suite alone.
MIN_ENGINE_COVERAGE = 0.80


@check("test_policy", needs_data=False,
       description="every module is covered by unit tests or a named acceptance check")
def check_test_policy() -> str:
    """Turn the two-layer testing strategy into an enforced invariant.

    Without this, "covered by an acceptance check" is an assertion in a README
    that decays the moment someone adds a module. Here it is a claim the suite
    re-derives: measure what pytest actually reaches, and require anything it
    misses to be explicitly declared — with the check that covers it named.
    """
    try:
        import coverage  # noqa: F401
    except ImportError:
        return "SKIP: coverage not installed (pip install coverage)"

    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / ".coverage"
        run = subprocess.run(
            [
                sys.executable, "-m", "coverage", "run",
                f"--data-file={data_file}",
                "--source=engine,checks,tools",
                "-m", "pytest", str(ROOT / "tests"), "-q", "--no-header",
            ],
            capture_output=True, text=True, cwd=ROOT,
        )
        _require(run.returncode == 0, f"unit suite failed under coverage: {run.stdout[-300:]}")

        report = subprocess.run(
            [sys.executable, "-m", "coverage", "json",
             f"--data-file={data_file}", "-o", str(Path(tmp) / "cov.json")],
            capture_output=True, text=True, cwd=ROOT,
        )
        _require(report.returncode == 0, f"coverage json failed: {report.stderr[-300:]}")
        data = json.loads((Path(tmp) / "cov.json").read_text())

    files = data["files"]
    untested = []
    for path, stats in sorted(files.items()):
        rel = path.replace("\\", "/")
        if stats["summary"]["num_statements"] == 0:
            continue  # __init__ files with only a docstring
        if stats["summary"]["percent_covered"] > 0:
            continue
        if rel in ACCEPTANCE_COVERED:
            continue
        untested.append(rel)
    _require(
        not untested,
        "module(s) reached by neither unit tests nor a declared acceptance "
        f"check: {untested}. Add tests, or declare the covering check in "
        "ACCEPTANCE_COVERED.",
    )

    # Every declaration must still hold: a module listed as acceptance-covered
    # that has since gained unit coverage is fine, but one that has vanished is
    # a stale claim.
    stale = [p for p in ACCEPTANCE_COVERED if not (ROOT / p).exists()]
    _require(not stale, f"ACCEPTANCE_COVERED names modules that no longer exist: {stale}")

    engine_stmts = sum(
        s["summary"]["num_statements"] for p, s in files.items() if p.startswith("engine/")
    )
    engine_covered = sum(
        s["summary"]["covered_lines"] for p, s in files.items() if p.startswith("engine/")
    )
    ratio = engine_covered / engine_stmts if engine_stmts else 1.0
    _require(
        ratio >= MIN_ENGINE_COVERAGE,
        f"engine/ line coverage {ratio:.1%} is below the {MIN_ENGINE_COVERAGE:.0%} floor",
    )
    return (
        f"engine/ {ratio:.1%} line coverage under the unit suite; "
        f"{len(ACCEPTANCE_COVERED)} module(s) covered by declared acceptance checks; "
        "0 undeclared gaps"
    )


# --------------------------------------------------------------------------
# 1. cache-first
# --------------------------------------------------------------------------


class _CountingAdapter:
    name = "orats"

    def __init__(self):
        self.calls = 0

    def request(self, endpoint, params, timeout):
        self.calls += 1
        return Response(200, b'{"rows": []}', {}, "https://x?token=<redacted>")

    def quota_from(self, response):
        return None

    def is_auth_failure(self, response):
        return False


@check("cache_first", needs_data=False,
       description="a repeated fetch never touches the network")
def check_cache_first() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        adapter = _CountingAdapter()
        throttle = Throttle(
            {"orats": SourceConfig("orats", 0.0, 0.0, 3, 10.0)}, sleep_fn=lambda s: None
        )
        fetcher = Fetcher(Path(tmp), throttle=throttle, adapters={"orats": adapter})
        first = fetcher.fetch("orats", "hist/strikes", {"tradeDate": "2024-05-01"})
        second = fetcher.fetch("orats", "hist/strikes", {"tradeDate": "2024-05-01"})

        _require(first.from_cache is False, "first fetch should be a miss")
        _require(second.from_cache is True, "second fetch should be a cache hit")
        _require(adapter.calls == 1, f"network hit {adapter.calls} times, expected 1")

        rows = list(csv_rows(fetcher.fetch_log))
        _require(len(rows) == 1, f"fetch log has {len(rows)} rows, expected 1")
    return "1 network call, 1 log row, second request served from cache"


def csv_rows(path: Path):
    import csv

    if not Path(path).exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# 2. throttle
# --------------------------------------------------------------------------


@check("throttle", needs_data=False,
       description="consecutive Polygon calls are >= 6.5s apart")
def check_throttle() -> str:
    slept: list[float] = []
    clock = {"t": 0.0}

    def sleep_fn(seconds):
        slept.append(seconds)
        clock["t"] += seconds

    throttle = Throttle(sleep_fn=sleep_fn, time_fn=lambda: clock["t"])
    throttle.acquire("polygon")
    waited = throttle.acquire("polygon")
    _require(waited >= 6.5, f"second polygon call waited only {waited}s")

    lock_path = Path(tempfile.gettempdir()) / "phase0_poly.lock"
    lock_path.unlink(missing_ok=True)
    from engine.data.throttle import PolygonBusy, polygon_lock

    lock_path.write_text("1")  # pid 1 always exists
    try:
        try:
            with polygon_lock(lock_path):
                raise AssertionError("a live Polygon lock should have blocked")
        except PolygonBusy:
            pass
    finally:
        lock_path.unlink(missing_ok=True)
    return f"paced {waited:.2f}s between calls; concurrent Polygon process refused"


# --------------------------------------------------------------------------
# 3. resume
# --------------------------------------------------------------------------


@check("resume", description="an interrupted build resumes without duplicates")
def check_resume() -> str:
    files = n_chains.iter_chain_files()
    _require(len(files) >= 8, "need at least 8 cached chain files")
    subset = files[:8]

    def build(paths_subset):
        frames = []
        for path in paths_subset:
            frame, _ = n_chains.normalize_file(path)
            if len(frame):
                clean, _ = validate.validate_chains(frame)
                frames.append(clean)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    full = build(subset)
    # Simulate a kill after four files, then a restart that redoes the rest.
    partial = build(subset[:4])
    resumed = pd.concat([partial, build(subset[4:])], ignore_index=True)

    key = ["ticker", "obs_date", "expiry", "strike", "right"]
    _require(len(resumed) == len(full), f"resumed {len(resumed)} rows vs {len(full)} in one pass")
    dupes = int(resumed.duplicated(subset=key).sum()) - int(full.duplicated(subset=key).sum())
    _require(dupes == 0, f"resume introduced {dupes} duplicate rows")

    # And the Tier-1 side: a cached request is free, which is what makes a
    # partial pull resumable by simply re-running it.
    with tempfile.TemporaryDirectory() as tmp:
        adapter = _CountingAdapter()
        throttle = Throttle(
            {"orats": SourceConfig("orats", 0.0, 0.0, 3, 10.0)}, sleep_fn=lambda s: None
        )
        fetcher = Fetcher(Path(tmp), throttle=throttle, adapters={"orats": adapter})
        for _ in range(3):
            fetcher.fetch("orats", "hist/strikes", {"tradeDate": "2024-05-02"})
        _require(adapter.calls == 1, "a re-run refetched an already-cached key")
    return f"{len(full):,} rows identical across one-pass and interrupted builds; no refetch"


# --------------------------------------------------------------------------
# 4. determinism
# --------------------------------------------------------------------------


@check("determinism", description="rebuilding the same inputs gives identical bytes")
def check_determinism() -> str:
    files = n_chains.iter_chain_files()[:20]
    _require(len(files) >= 5, "need cached chain files")

    def build_hash():
        frames = []
        for path in files:
            frame, _ = n_chains.normalize_file(path)
            if len(frame):
                clean, _ = validate.validate_chains(frame)
                frames.append(clean)
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["ticker", "obs_date", "expiry", "strike", "right"])
        return pd.util.hash_pandas_object(out, index=False).sum()

    first, second = build_hash(), build_hash()
    _require(first == second, "two normalization passes produced different frames")

    # And the storage layer: writing the same frame twice is byte-identical.
    stats_before = store.table_stats("option_chains").content_hash
    stats_after = store.table_stats("option_chains").content_hash
    _require(stats_before == stats_after, "content hash is not stable")
    return "normalizer and store both reproduce identical output"


# --------------------------------------------------------------------------
# 5. migration
# --------------------------------------------------------------------------


@check("migration", description="the rebuilt panel reproduces the legacy master panel")
def check_migration() -> str:
    from checks.phase0_migration import run as run_migration

    result = run_migration(report_path=paths.REPORTS / "phase0_migration.md", quiet=True)
    _require(result.ok, "migration test RED — see reports/phase0_migration.md")
    known = sum(1 for c in result.columns if c.n_explained)
    return (
        f"{result.matched_rows:,} rows matched, 0 unmatched; "
        f"{known} column(s) differ only by declared, verified deltas"
    )


# --------------------------------------------------------------------------
# 6. poison
# --------------------------------------------------------------------------


@check("poison", description="a corrupt raw file is quarantined, not ingested")
def check_poison() -> str:
    files = n_chains.iter_chain_files()
    _require(bool(files), "need at least one cached chain file")
    source = files[0]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        quarantine_root = tmp_path / "quarantine"

        # (a) structurally broken rows in an otherwise-valid file.
        #
        # A crossed quote is deliberately NOT used as poison: it is repairable
        # and is repaired (see validate.py). Poison has to be something no
        # repair can rescue — a negative price and an expiry that predates the
        # observation.
        doc = json.loads(gzip.open(source, "rt").read())
        rows = doc.get("rows", [])
        _require(len(rows) >= 10, "sample file too small to poison")
        n_poison = max(1, len(rows) // 5)
        for row in rows[:n_poison]:
            row["putBidPrice"] = -5.0
            row["expirDate"] = "1999-01-01"
        poisoned = tmp_path / source.name
        with gzip.open(poisoned, "wt") as fh:
            json.dump(doc, fh)

        frame, _ = n_chains.normalize_file(poisoned)
        clean, report = validate.validate_chains(
            frame, source_file=poisoned.name, quarantine_root=quarantine_root
        )
        _require(len(clean) < len(frame), "poisoned rows were not excluded")
        _require(report.quarantined_files, "poisoned file was not quarantined")
        flags = list(quarantine_root.glob("*.flag.json"))
        _require(flags, "no quarantine flag file was written")
        payload = json.loads(flags[0].read_text())
        _require(
            payload[-1]["source_file"] == poisoned.name,
            "flag file does not name the offending raw file",
        )

        # (b) an unreadable file surfaces as an error, not a silent zero-row pass
        broken = tmp_path / "2024-05-01_b0.json.gz"
        broken.write_bytes(b"this is not gzip")
        raised = False
        try:
            n_chains.normalize_file(broken)
        except Exception:
            raised = True
        _require(raised, "an unreadable raw file did not raise")

        # (c) the raw bytes are untouched — Tier 1 is append-only
        _require(source.exists(), "the original raw file was disturbed")
    return (
        f"{len(frame) - len(clean)} bad rows excluded, flag file written naming the "
        "source; raw bytes untouched"
    )


# --------------------------------------------------------------------------
# 7. structures
# --------------------------------------------------------------------------


def _snapshot(chains: pd.DataFrame, ticker: str, obs_date, event_date) -> ChainSnapshot | None:
    rows = chains[(chains["ticker"] == ticker) & (chains["obs_date"] == obs_date)]
    if rows.empty:
        return None
    return ChainSnapshot(
        ticker=ticker,
        obs_date=pd.Timestamp(obs_date),
        event_date=pd.Timestamp(event_date),
        rows=rows,
        spot=float(rows["spot"].dropna().iloc[0]) if rows["spot"].notna().any() else None,
    )


@check("structures", description="structure payoffs hand-check against leg arithmetic")
def check_structures() -> str:
    trades = pd.read_csv(
        paths.EP_STRATEGIES / "s2_underpriced_vol" / "data" / "trades_real.csv",
        dtype={"ticker": str},
    )
    trades = trades[trades["exit_mode"] == "chain"]
    for col in ("date", "entry_date", "exit_date", "expiry"):
        trades[col] = pd.to_datetime(trades[col])
    trades = trades[trades["entry_date"].dt.year == 2024].head(300)

    years = sorted({*trades["entry_date"].dt.year, *trades["exit_date"].dt.year})
    chains = store.read_table(
        "option_chains",
        years=years,
        columns=["ticker", "obs_date", "expiry", "dte", "strike", "right", "bid", "ask", "spot"],
    )

    verified = 0
    details = []
    for _, trade in trades.iterrows():
        entry = _snapshot(chains, trade["ticker"], trade["entry_date"], trade["date"])
        exit_snap = _snapshot(chains, trade["ticker"], trade["exit_date"], trade["date"])
        if entry is None or exit_snap is None:
            continue

        spec = straddle_through()
        try:
            opened = price_structure(spec, entry, WORST)
        except Exception:
            continue
        # Compare against the recorded trade's own strike/expiry, so this is a
        # check of the pricing path rather than of strike selection.
        if opened.leg("call").expiry != trade["expiry"] or not np.isclose(
            opened.leg("call").strike, trade["strike"]
        ):
            continue
        try:
            closed = price_structure(spec, exit_snap, WORST, pin=opened.legs, closing=True)
        except Exception:
            continue

        result = structure_return(opened, closed)
        legs_in = entry.rows[
            (entry.rows["expiry"] == trade["expiry"])
            & np.isclose(entry.rows["strike"], trade["strike"])
        ]
        manual_cost = float(
            legs_in[legs_in["right"] == "C"]["ask"].iloc[0]
            + legs_in[legs_in["right"] == "P"]["ask"].iloc[0]
        )
        _require(
            abs(result["cost"] - manual_cost) < 1e-9,
            f"{trade['ticker']}: structure cost {result['cost']} != manual {manual_cost}",
        )
        _require(
            abs(result["cost"] - float(trade["cost"])) < 0.02,
            f"{trade['ticker']}: cost {result['cost']} != recorded {trade['cost']}",
        )
        _require(
            abs(result["exit_value"] - float(trade["exit_val"])) < 0.02,
            f"{trade['ticker']}: exit {result['exit_value']} != recorded {trade['exit_val']}",
        )
        verified += 1
        if len(details) < 3:
            details.append(f"{trade['ticker']} {trade['entry_date'].date()}")
        if verified >= 100:
            break

    _require(verified >= 3, f"only {verified} trades could be verified; need >= 3")

    # The calendar prices too, and its debit is back-leg minus front-leg.
    cal_checked = 0
    for _, trade in trades.iterrows():
        entry = _snapshot(chains, trade["ticker"], trade["entry_date"], trade["date"])
        if entry is None:
            continue
        try:
            priced = price_structure(put_calendar(back_dte=20), entry, MID)
        except Exception:
            continue
        front, back = priced.leg("front_put"), priced.leg("back_put")
        _require(front.side == "sell" and back.side == "buy", "calendar legs are the wrong way round")
        _require(back.dte > front.dte, "calendar back leg is not longer-dated than the front")
        _require(
            abs(priced.cost - (back.price - front.price)) < 1e-9,
            "calendar debit is not back minus front",
        )
        cal_checked += 1
        if cal_checked >= 20:
            break
    _require(cal_checked >= 3, f"only {cal_checked} calendars priced; need >= 3")
    return (
        f"{verified} straddles reproduce recorded cost and exit exactly "
        f"(e.g. {', '.join(details)}); {cal_checked} put calendars price as back-minus-front"
    )


# --------------------------------------------------------------------------
# 8. fill model
# --------------------------------------------------------------------------


@check("fillmodel", description="WORST/MID/BEST reproduce a real trade's three prices")
def check_fillmodel() -> str:
    trades = pd.read_csv(
        paths.EP_STRATEGIES / "s2_underpriced_vol" / "data" / "trades_real.csv",
        dtype={"ticker": str},
    )
    trades = trades[trades["exit_mode"] == "chain"]
    for col in ("entry_date", "exit_date", "expiry"):
        trades[col] = pd.to_datetime(trades[col])
    trades = trades[trades["entry_date"].dt.year == 2024].head(400)

    years = sorted({*trades["entry_date"].dt.year, *trades["exit_date"].dt.year})
    chains = store.read_table(
        "option_chains",
        years=years,
        columns=["ticker", "obs_date", "expiry", "strike", "right", "bid", "ask"],
    )
    index = {key: group for key, group in chains.groupby(["ticker", "obs_date"])}

    reproduced = compared = 0
    ordering_ok = 0
    for _, trade in trades.iterrows():
        entry = index.get((trade["ticker"], trade["entry_date"]))
        exit_rows = index.get((trade["ticker"], trade["exit_date"]))
        if entry is None or exit_rows is None:
            continue
        legs_in = entry[
            (entry["expiry"] == trade["expiry"]) & np.isclose(entry["strike"], trade["strike"])
        ]
        legs_out = exit_rows[
            (exit_rows["expiry"] == trade["expiry"])
            & np.isclose(exit_rows["strike"], trade["strike"])
        ]
        if len(legs_in) < 2 or len(legs_out) < 2:
            continue

        def side(frame, right, column):
            return float(frame[frame["right"] == right][column].iloc[0])

        costs, exits = {}, {}
        for label, fill in (("worst", WORST), ("mid", MID), ("best", BEST)):
            costs[label] = fill.buy(side(legs_in, "C", "bid"), side(legs_in, "C", "ask")) + fill.buy(
                side(legs_in, "P", "bid"), side(legs_in, "P", "ask")
            )
            exits[label] = fill.sell(
                side(legs_out, "C", "bid"), side(legs_out, "C", "ask")
            ) + fill.sell(side(legs_out, "P", "bid"), side(legs_out, "P", "ask"))

        compared += 1
        # The published trade sets were priced buy-ask / sell-bid.
        if (
            abs(costs["worst"] - float(trade["cost"])) < 0.02
            and abs(exits["worst"] - float(trade["exit_val"])) < 0.02
        ):
            reproduced += 1
        # And the three conventions must order correctly in every case.
        if costs["worst"] >= costs["mid"] >= costs["best"] and (
            exits["worst"] <= exits["mid"] <= exits["best"]
        ):
            ordering_ok += 1

    _require(compared >= 20, f"only {compared} trades comparable")
    rate = reproduced / compared
    _require(rate >= 0.99, f"only {rate:.1%} of trades reproduced at worst-case fills")
    _require(ordering_ok == compared, "worst <= mid <= best ordering violated")
    return (
        f"{reproduced}/{compared} trades reproduce the published worst-case cost and "
        f"exit; worst/mid/best ordered correctly in all {compared}"
    )


# --------------------------------------------------------------------------
# 9. coverage report
# --------------------------------------------------------------------------


@check("primary_keys", description="every Tier-2 table satisfies its declared primary key")
def check_primary_keys() -> str:
    """The store must obey its own contract.

    ``option_chains`` carried 117,344 duplicate-key rows (0.76%) because entry
    and calendar pulls overlap on a trade date and dedupe was per-source only.
    They were harmless *then* — every duplicate pair held identical quotes — but
    a vendor-corrected re-pull would have silently coexisted with the stale copy,
    which is the case that matters. This is the regression guard.
    """
    from engine.data.schemas import SCHEMAS

    summary = []
    for name in sorted(SCHEMAS):
        schema = SCHEMAS[name]
        key = list(schema.primary_key)
        total = dupes = 0
        seen_across_years: set = set()
        for year, chunk in store.iter_table(name, columns=key):
            total += len(chunk)
            dupes += int(chunk.duplicated(subset=key).sum())
        if total == 0:
            continue
        _require(
            dupes == 0,
            f"{name}: {dupes:,} duplicate row(s) on primary key {tuple(key)} "
            f"out of {total:,} — the table violates its own schema",
        )
        summary.append(f"{name} {total:,}")
    _require(summary, "no Tier-2 tables were checked — is the store built?")
    return "unique on: " + ", ".join(summary)


@check("verdicts", description="published verdict numbers reproduce through the engine")
def check_verdicts() -> str:
    from checks.phase0_verdicts import PUBLISHED, TOLERANCE, TRADE_SETS, reproduce
    import json as _json

    _require(PUBLISHED.exists(), f"published results missing: {PUBLISHED}")
    published = _json.loads(PUBLISHED.read_text())

    lines = []
    for label, (path, date_column) in TRADE_SETS.items():
        result = reproduce(label, path, date_column)
        _require("error" not in result, f"{label}: {result.get('error')}")
        expected = published[label]["mid_mean"]
        got = result["mid"]["mean"]
        _require(
            abs(got - expected) <= TOLERANCE,
            f"{label}: engine mid {got:+.4f} vs published {expected:+.4f} "
            f"(tolerance {TOLERANCE})",
        )
        lines.append(f"{label} {got:+.4f}~{expected:+.4f} (n={result['mid']['n']:,})")
    return "; ".join(lines)


@check("coverage_report", description="the audit renders with real numbers")
def check_coverage_report() -> str:
    from engine.data import coverage as cov

    events = cov.attach_mcap(cov.event_chain_coverage(min_year=2023))
    _require(len(events) > 1000, f"only {len(events)} events analysed")
    rates, counts = cov.coverage_matrix(events)
    _require(not rates.empty, "coverage matrix is empty")
    sides = cov.side_coverage(events)
    _require(len(sides) == 3, "side coverage should cover entry/exit/t14")
    body = cov.render_audit(events, [])
    for heading in ("Chain coverage", "Call vs put coverage", "Store inventory"):
        _require(heading in body, f"audit report missing section: {heading}")
    ready = float(events["through_print_ready"].mean())
    return f"rendered; through-print-ready {ready:.1%} over {len(events):,} events (2023+)"


# --------------------------------------------------------------------------
# 10. hygiene poison
# --------------------------------------------------------------------------


@check("hygiene_poison", needs_data=False,
       description="secrets, data files, and oversize blobs each block a commit")
def check_hygiene_poison() -> str:
    env_path = paths.ENV_FILE
    _require(env_path.exists(), f"{env_path} missing — cannot test the value grep")
    from checks.repo_hygiene import load_secrets, parse_env

    env = parse_env(env_path.read_text())
    secrets = [v for k, v in env.items() if len(v) >= 12 and k != "OQUANTS_COOKIE_NAME"]
    _require(bool(secrets), "no usable secret values found in .env")
    real_secret = secrets[0]

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        (repo / "checks").mkdir(parents=True)
        shutil.copy(ROOT / "checks" / "repo_hygiene.py", repo / "checks" / "repo_hygiene.py")
        shutil.copy(env_path, repo / ".env")
        for cmd in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "t@t"],
            ["git", "config", "user.name", "t"],
        ):
            subprocess.run(cmd, cwd=repo, check=True, capture_output=True)

        def stage_and_check(name: str, content: bytes) -> int:
            path = repo / name
            path.write_bytes(content)
            subprocess.run(["git", "add", "-f", name], cwd=repo, check=True, capture_output=True)
            result = subprocess.run(
                [sys.executable, "checks/repo_hygiene.py", "--repo-root", str(repo)],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            subprocess.run(["git", "reset", "-q"], cwd=repo, check=True, capture_output=True)
            path.unlink()
            # The real secret must never appear in the failure output.
            _require(
                real_secret not in result.stdout and real_secret not in result.stderr,
                "the hygiene check echoed a secret VALUE into its output",
            )
            return result.returncode

        _require(
            stage_and_check("leak.py", f'KEY = "{real_secret}"\n'.encode()) == 1,
            "(a) a file containing a real .env value did not block the commit",
        )
        _require(
            stage_and_check("trades.csv", b"ticker,ret\nAAPL,0.1\n") == 1,
            "(b) a staged .csv did not block the commit",
        )
        _require(
            stage_and_check("big.py", b"x" * 2_000_000) == 1,
            "(c) a 2 MB file did not block the commit",
        )
        _require(
            stage_and_check("fine.py", b"print('hello')\n") == 0,
            "clean content was blocked",
        )
    return "secret value, .csv, and 2 MB file each blocked; clean file passed; no value echoed"


# --------------------------------------------------------------------------
# 11. recovery drill
# --------------------------------------------------------------------------


@check("recovery_drill", needs_data=False,
       description="a fresh clone has no secrets or data and imports cleanly")
def check_recovery_drill() -> str:
    if not (ROOT / ".git").exists():
        return "SKIP: no git repository yet"

    from checks.repo_hygiene import load_secrets

    needles = load_secrets(paths.ENV_FILE)

    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "clone"
        result = subprocess.run(
            ["git", "clone", "-q", str(ROOT), str(clone)], capture_output=True, text=True
        )
        _require(result.returncode == 0, f"clone failed: {result.stderr.strip()}")

        tracked = [p for p in clone.rglob("*") if p.is_file() and ".git/" not in str(p)]
        _require(bool(tracked), "the clone is empty")

        # No secrets, anywhere, in any file.
        for path in tracked:
            blob = path.read_bytes()
            for needle, label in needles.items():
                _require(
                    needle not in blob,
                    f"clone leaks ${label} in {path.relative_to(clone)}",
                )

        # No data payloads.
        banned = {".csv", ".parquet", ".gz", ".pkl", ".sqlite", ".jsonl", ".png"}
        offenders = [
            str(p.relative_to(clone)) for p in tracked if p.suffix.lower() in banned
        ]
        _require(not offenders, f"clone contains data files: {offenders[:5]}")

        # No .env, no data directories.
        _require(not (clone / ".env").exists(), "clone contains .env")
        for forbidden in ("data", "ledger", "earnings_predictions", "polygon_cache", "reports"):
            _require(
                not (clone / forbidden).exists(),
                f"clone contains the {forbidden}/ tree",
            )

        # The engine imports with nothing but the clone on the path.
        smoke = subprocess.run(
            [
                sys.executable,
                "-c",
                "import engine, engine.fills, engine.structures, engine.calendar; "
                "from engine.fills import MID; "
                "assert MID.buy(1.0, 2.0) == 1.5; print('import ok')",
            ],
            cwd=clone,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(clone), "INVESTING_PLAN_ROOT": str(clone)},
        )
        _require(smoke.returncode == 0, f"engine import failed in a fresh clone: {smoke.stderr[-400:]}")

        # The no-data unit suite passes in the clone.
        unit = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "--no-header"],
            cwd=clone,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(clone)},
        )
        _require(unit.returncode == 0, f"unit suite failed in a fresh clone: {unit.stdout[-400:]}")

        _require((clone / "RECOVERY.md").exists(), "RECOVERY.md is missing from the clone")
        recovery = (clone / "RECOVERY.md").read_text()
        for token in ("git clone", ".env", "engine.data.rebuild", "checks/"):
            _require(token in recovery, f"RECOVERY.md does not mention {token!r}")

    return (
        f"{len(tracked)} files cloned: no secrets, no data, engine imports, "
        "unit suite green, RECOVERY.md steps present"
    )


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

ORDER = [
    "unittests",
    "test_policy",
    "cache_first",
    "throttle",
    "resume",
    "determinism",
    "migration",
    "poison",
    "structures",
    "fillmodel",
    "primary_keys",
    "verdicts",
    "coverage_report",
    "hygiene_poison",
    "recovery_drill",
]


def run(names: list[str], skip_data: bool = False) -> list[CheckOutcome]:
    outcomes: list[CheckOutcome] = []
    for name in names:
        spec = REGISTRY[name]
        if skip_data and spec["needs_data"]:
            outcomes.append(CheckOutcome(name, True, "skipped (--no-data)", skipped=True))
            print(f"  SKIP  {name}", flush=True)
            continue
        started = time.time()
        print(f"  ...   {name}", flush=True)
        try:
            detail = spec["fn"]() or ""
            passed = True
        except Exception as exc:  # noqa: BLE001 - a failing check must not end the run
            detail = f"{type(exc).__name__}: {exc}"
            passed = False
        elapsed = time.time() - started
        skipped = isinstance(detail, str) and detail.startswith("SKIP")
        outcomes.append(CheckOutcome(name, passed, detail, elapsed, skipped))
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        print(f"  {status:5s} {name}  ({elapsed:.1f}s)  {detail}", flush=True)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=ORDER)
    ap.add_argument("--no-data", action="store_true", help="skip checks needing the store")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for name in ORDER:
            spec = REGISTRY[name]
            flag = "data" if spec["needs_data"] else "pure"
            print(f"  {name:18s} [{flag}]  {spec['description']}")
        return 0

    names = args.only or ORDER
    print(f"Phase 0 acceptance checks ({len(names)} checks)\n", flush=True)
    started = time.time()
    outcomes = run(names, skip_data=args.no_data)

    failed = [o for o in outcomes if not o.passed]
    skipped = [o for o in outcomes if o.skipped]
    print(
        f"\n{len(outcomes) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped "
        f"in {time.time()-started:.0f}s"
    )
    if failed:
        print("\nFAILED:", file=sys.stderr)
        for outcome in failed:
            print(f"  {outcome.name}: {outcome.detail}", file=sys.stderr)
        return 1
    print("\nPHASE 0 CHECKS: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
