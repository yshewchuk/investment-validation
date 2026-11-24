#!/usr/bin/env python3
"""Synthetic-only benchmark: ``engine.structures.price_structure`` (legacy) vs
``engine.chain_index.price_structure_indexed`` (twin).

Chains are generated in memory from fixed seeds. This tool never reads data/
or any store. Every twin result, including every timed repeat, is asserted
bit-identical to the legacy result; a mismatch aborts the run.

Usage:
    python3 tools/bench_chain_index.py [--tiny] [--reps N] [--no-profile]
"""
from __future__ import annotations

import argparse
import cProfile
import io
import itertools
import pstats
import statistics
import sys
import time
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.chain_index import price_structure_indexed  # noqa: E402
from engine.fills import FillModel  # noqa: E402
from engine.structures import STRUCTURES, ChainSnapshot, price_structure  # noqa: E402

SPOT = 100.0
OBS = pd.Timestamp("2026-03-09")
EVENT = pd.Timestamp("2026-03-10")
EXPIRIES = ((date(2026, 3, 10), 1), (date(2026, 3, 17), 8), (date(2026, 3, 24), 15))
STRIKE_COUNTS = (40, 80)
EXPIRY_COUNTS = (1, 3)
RIGHTS = ("P", "CP")
ALPHAS = (0.0, 0.5)
PROFILE_CASE = (80, 3, "CP")
DEFAULT_REPS = 20
TINY_STRUCTURES = 3


def _log(msg: str) -> None:
    print(f"[bench] {msg}", file=sys.stderr, flush=True)


def case_seed(n_strikes: int, n_expiries: int, rights: str) -> int:
    return 1000 * n_strikes + 10 * n_expiries + len(rights)


def make_chain(n_strikes: int, n_expiries: int, rights: str, seed: int) -> pd.DataFrame:
    """Synthetic chain shaped like the equivalence battery's ``make_chain``."""
    strikes = SPOT + (np.arange(n_strikes) - n_strikes // 2) * 1.0
    rng = np.random.default_rng(seed)
    call_delta = 1.0 / (1.0 + np.exp((strikes - SPOT) / (0.05 * SPOT)))
    frames = []
    for exp_date, dte in EXPIRIES[:n_expiries]:
        for right in rights:
            n = strikes.size
            bid = rng.uniform(0.05, 20.0, n)
            ask = bid + rng.uniform(0.02, 1.5, n)
            bid = np.where(rng.random(n) < 0.04, 0.0, bid)  # a few zero bids
            frames.append(pd.DataFrame({
                "expiry": np.full(n, np.datetime64(exp_date.isoformat(), "ns")),
                "dte": np.full(n, int(dte), dtype=np.int64),
                "strike": strikes.copy(),
                "right": np.full(n, right, dtype=object),
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "iv": rng.uniform(0.15, 0.8, n),
                "delta": call_delta if right == "C" else call_delta - 1.0,
                "spot": np.full(n, SPOT),
            }))
    return pd.concat(frames, ignore_index=True)


def make_snapshot(rows: pd.DataFrame) -> ChainSnapshot:
    return ChainSnapshot(ticker="T", obs_date=OBS, event_date=EVENT, rows=rows,
                         spot=SPOT, session="AMC")


def outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # every failure is compared, whatever its type
        return ("err", type(exc), str(exc))


def timed(fn):
    start = time.perf_counter()
    result = outcome(fn)
    return time.perf_counter() - start, result


def assert_identical(label, legacy, twin):
    if legacy[0] != twin[0]:
        raise AssertionError(f"{label}: status legacy={legacy[0]} twin={twin[0]}")
    if legacy[0] == "err":
        if legacy[1] is not twin[1] or legacy[2] != twin[2]:
            raise AssertionError(f"{label}: error legacy={legacy[1:]!r} twin={twin[1:]!r}")
        return
    a, b = legacy[1], twin[1]
    if type(a) is not type(b) or a.to_dict() != b.to_dict() or repr(a) != repr(b):
        raise AssertionError(f"{label}: value differs\n  legacy={a!r}\n  twin={b!r}")


def run_case(n_strikes, n_expiries, rights, names, alphas, reps):
    rows = make_chain(n_strikes, n_expiries, rights, case_seed(n_strikes, n_expiries, rights))
    out = []
    for name, alpha in itertools.product(names, alphas):
        structure = STRUCTURES[name]()
        fill = FillModel(alpha)
        label = f"{rights} {n_strikes}x{n_expiries} {name} alpha={alpha}"
        legacy_snap = make_snapshot(rows)
        reference = outcome(partial(price_structure, structure, legacy_snap, fill))
        legacy_times = []
        for _ in range(reps):
            elapsed, res = timed(partial(price_structure, structure, legacy_snap, fill))
            assert_identical(f"{label} legacy", reference, res)
            legacy_times.append(elapsed)
        # Cold: first call on a fresh snapshot (includes the index build).
        twin_snap = make_snapshot(rows.copy())
        cold, res = timed(partial(price_structure_indexed, structure, twin_snap, fill))
        assert_identical(f"{label} twin cold", reference, res)
        # Warm: repeats on the same snapshot the cold call used.
        warm_times = []
        for _ in range(reps):
            elapsed, res = timed(partial(price_structure_indexed, structure, twin_snap, fill))
            assert_identical(f"{label} twin warm", reference, res)
            warm_times.append(elapsed)
        row = {"rights": rights, "strikes": n_strikes, "expiries": n_expiries,
               "structure": name, "alpha": alpha, "ok": reference[0] == "ok"}
        if row["ok"]:
            row["legacy_ms"] = statistics.median(legacy_times) * 1000.0
            row["cold_ms"] = cold * 1000.0
            row["warm_ms"] = statistics.median(warm_times) * 1000.0
            row["speedup"] = row["legacy_ms"] / max(row["warm_ms"], 1e-9)
        out.append(row)
    return out


def _cells(row):
    head = [str(row["rights"]), str(row["strikes"]), str(row["expiries"]),
            row["structure"], f"{row['alpha']:g}"]
    if row["ok"]:
        return head + [f"{row['legacy_ms']:.3f}", f"{row['cold_ms']:.3f}",
                       f"{row['warm_ms']:.3f}", f"{row['speedup']:.2f}x"]
    return head + ["err", "err", "err", "-"]


HEADER = ["rights", "strikes", "expiries", "structure", "alpha",
          "legacy_ms", "twin_cold_ms", "twin_warm_ms", "speedup"]


def print_table(results):
    lines = [HEADER] + [_cells(r) for r in results]
    widths = [max(len(line[i]) for line in lines) for i in range(len(HEADER))]
    for line in lines:
        print("  ".join(cell.ljust(w) for cell, w in zip(line, widths)))
    speeds = [r["speedup"] for r in results if r["ok"]]
    if speeds:
        print(f"summary: ok_rows={len(speeds)} median speedup={statistics.median(speeds):.2f}x "
              f"min speedup={min(speeds):.2f}x")
    else:
        print("summary: ok_rows=0 (no ok rows, no speedup to report)")


def profile_top(call, rows, names):
    snap = make_snapshot(rows)
    prof = cProfile.Profile()
    prof.enable()
    for name, alpha in itertools.product(names, ALPHAS):
        outcome(partial(call, STRUCTURES[name](), snap, FillModel(alpha)))
    prof.disable()
    buf = io.StringIO()
    pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(10)
    return buf.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic-only chain-index benchmark: legacy price_structure vs the twin.")
    parser.add_argument("--tiny", action="store_true",
                        help="smoke run: one 40-strike case, 3 structures, alpha 0, 1 rep, no profile")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS,
                        help=f"timed repeats per side (default {DEFAULT_REPS}; forced to 1 by --tiny)")
    parser.add_argument("--no-profile", action="store_true",
                        help="skip the cProfile top-10 section")
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("--reps must be >= 1")

    if args.tiny:
        cases = [(40, 1, "CP")]
        names = list(STRUCTURES)[:TINY_STRUCTURES]
        alphas, reps, do_profile = (0.0,), 1, False
    else:
        cases = list(itertools.product(STRIKE_COUNTS, EXPIRY_COUNTS, RIGHTS))
        names = list(STRUCTURES)
        alphas, reps, do_profile = ALPHAS, args.reps, not args.no_profile

    t0 = time.perf_counter()
    _log(f"start: {len(cases)} case(s), {len(names)} structure(s), alphas={alphas}, reps={reps}")
    results = []
    for i, (n_strikes, n_expiries, rights) in enumerate(cases, 1):
        results.extend(run_case(n_strikes, n_expiries, rights, names, alphas, reps))
        _log(f"case {i}/{len(cases)} strikes={n_strikes} expiries={n_expiries} "
             f"rights={rights} elapsed={time.perf_counter() - t0:.1f}s")
    print_table(results)
    _log(f"rows done: elapsed={time.perf_counter() - t0:.1f}s")

    if do_profile:
        _log("profile: start")
        rows = make_chain(*PROFILE_CASE, case_seed(*PROFILE_CASE))
        for label, call in (("legacy", price_structure), ("twin", price_structure_indexed)):
            print(f"\n== cProfile top-10 by cumulative time: {label} ==")
            print(profile_top(call, rows, names))
        _log(f"profile: end elapsed={time.perf_counter() - t0:.1f}s")

    _log(f"done: elapsed={time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
