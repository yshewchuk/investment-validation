#!/usr/bin/env python3
"""Benchmark: ``engine.chain_index`` twin against ``engine.structures.price_structure``.

Not a CI test. Reads no data: every chain is synthetic and seeded.

    python3 tools/bench_chain_index.py [--events 300] [--repeats 3] [--seed 0]

Two modes, five short-vol structures, fill MID:

* cold: a fresh snapshot per call, so the twin pays its index build on every call.
* warm: one snapshot per event priced through all five structures, so the twin's
  index is built on the first structure and reused by the rest.

Every call compares outcomes (``to_dict()`` equal, or the same exception type
and message). stderr gets stage progress and a short table; stdout gets one JSON
summary. ``ENGINE_CHAIN_INDEX_VERIFY`` is unset for the whole run, so timings do
not include the verify checksums.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.chain_index import price_structure_indexed  # noqa: E402
from engine.fills import MID  # noqa: E402
from engine.structures import STRUCTURES, ChainSnapshot, price_structure  # noqa: E402

KEYS = ("CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")
VERIFY_ENV = "ENGINE_CHAIN_INDEX_VERIFY"
SPACINGS = (0.5, 1.0, 5.0, 10.0)
SPOT = 100.0
OBS0 = pd.Timestamp("2026-01-05")
T0 = time.perf_counter()


def log(message: str) -> None:
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"[{time.perf_counter() - T0:8.2f}s rss={rss_mb:7.1f}MB] {message}",
          file=sys.stderr, flush=True)


@contextmanager
def stage(name: str):
    started = time.perf_counter()
    log(f"stage {name}: start")
    yield
    log(f"stage {name}: end after {time.perf_counter() - started:.2f}s")


def build_events(count: int, seed: int) -> list[tuple]:
    """Synthetic put-only ladders, 3-6 expiries each, mixed AMC/BMO sessions."""
    rng = np.random.default_rng(seed)
    events = []
    for index in range(count):
        spacing = float(rng.choice(SPACINGS))
        n_expiries = int(rng.integers(3, 7))
        dtes = sorted(int(d) for d in rng.choice(np.arange(1, 61), size=n_expiries, replace=False))
        n_strikes = int(rng.integers(15, 41))
        base = round(SPOT / spacing) * spacing
        strikes = base + (np.arange(n_strikes) - n_strikes // 2) * spacing
        strikes = strikes[strikes > 0]
        obs = OBS0 + pd.Timedelta(days=index)
        event = obs + pd.Timedelta(days=1)
        delta = -1.0 / (1.0 + np.exp(-(strikes - SPOT) / (0.05 * SPOT)))
        frames = []
        for dte in dtes:
            n = strikes.size
            bid = rng.uniform(0.05, 15.0, n)
            ask = bid + rng.uniform(0.02, 1.0, n)
            bid = np.where(rng.random(n) < 0.03, 0.0, bid)
            expiry = np.datetime64((obs + pd.Timedelta(days=dte)).date().isoformat(), "ns")
            frames.append(pd.DataFrame({
                "expiry": np.full(n, expiry),
                "dte": np.full(n, dte, dtype=np.int64),
                "strike": strikes.copy(),
                "right": np.full(n, "P", dtype=object),
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "iv": rng.uniform(0.15, 0.8, n),
                "delta": delta,
                "spot": np.full(n, SPOT),
            }))
        rows = pd.concat(frames, ignore_index=True)
        session = "AMC" if rng.random() < 0.5 else "BMO"
        events.append((index, rows, obs, event, session))
    return events


def make_snapshot(index: int, rows, obs, event, session) -> ChainSnapshot:
    return ChainSnapshot(ticker=f"SYN{index}", obs_date=obs, event_date=event,
                         rows=rows, spot=SPOT, session=session)


def outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # every failure is compared too
        return ("err", type(exc), str(exc))


def same(a, b) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == "err":
        return a[1] is b[1] and a[2] == b[2]
    return a[1].to_dict() == b[1].to_dict()


def timed(fn):
    started = time.perf_counter()
    result = outcome(fn)
    return result, (time.perf_counter() - started) * 1000.0


def run_mode(events, repeats: int, mode: str, structures: dict) -> dict:
    samples = {k: {"legacy": [], "twin": [], "mismatch": 0} for k in KEYS}
    for _ in range(repeats):
        for index, rows, obs, event, session in events:
            shared_legacy = shared_twin = None
            if mode == "warm":
                shared_legacy = make_snapshot(index, rows, obs, event, session)
                shared_twin = make_snapshot(index, rows, obs, event, session)
            for key in KEYS:
                if mode == "cold":
                    legacy_snap = make_snapshot(index, rows, obs, event, session)
                    twin_snap = make_snapshot(index, rows, obs, event, session)
                else:
                    legacy_snap, twin_snap = shared_legacy, shared_twin
                structure = structures[key]
                legacy_out, legacy_ms = timed(lambda: price_structure(structure, legacy_snap, MID))
                twin_out, twin_ms = timed(lambda: price_structure_indexed(structure, twin_snap, MID))
                samples[key]["legacy"].append(legacy_ms)
                samples[key]["twin"].append(twin_ms)
                if not same(legacy_out, twin_out):
                    samples[key]["mismatch"] += 1
    return samples


def stats(ms: list[float]) -> dict:
    arr = np.asarray(ms, dtype=float)
    return {
        "median": round(float(np.median(arr)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "mean": round(float(arr.mean()), 4),
        "calls": int(arr.size),
    }


def summarise(samples: dict) -> dict:
    out = {}
    for key, data in samples.items():
        legacy, twin = stats(data["legacy"]), stats(data["twin"])
        speedup = legacy["median"] / twin["median"] if twin["median"] > 0 else float("nan")
        out[key] = {
            "legacy_ms": legacy,
            "twin_ms": twin,
            "speedup_median": round(speedup, 3),
            "mismatches": data["mismatch"],
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=int, default=300, help="synthetic events (default 300)")
    parser.add_argument("--repeats", type=int, default=3, help="passes over the events (default 3)")
    parser.add_argument("--seed", type=int, default=0, help="synthetic chain seed (default 0)")
    args = parser.parse_args(argv)
    if args.events < 1 or args.repeats < 1:
        parser.error("--events and --repeats must be positive")

    removed = os.environ.pop(VERIFY_ENV, None)
    log(f"{VERIFY_ENV} unset for timing (was {'set' if removed is not None else 'already unset'})")
    structures = {k: STRUCTURES[k]() for k in KEYS}

    with stage("generate"):
        events = build_events(args.events, args.seed)

    modes = {}
    for mode in ("cold", "warm"):
        with stage(f"{mode} mode"):
            samples = run_mode(events, args.repeats, mode, structures)
        modes[mode] = summarise(samples)
        for key in KEYS:
            row = modes[mode][key]
            log(f"{mode} {key}: legacy median {row['legacy_ms']['median']} ms, "
                f"twin median {row['twin_ms']['median']} ms, "
                f"speed-up {row['speedup_median']}x, mismatches {row['mismatches']}")

    mismatches = sum(row["mismatches"] for mode in modes.values() for row in mode.values())
    print(f"{'mode':<6} {'structure':<9} {'legacy med ms':>14} {'twin med ms':>12} {'speed-up':>9}",
          file=sys.stderr)
    for mode in ("cold", "warm"):
        for key in KEYS:
            row = modes[mode][key]
            print(f"{mode:<6} {key:<9} {row['legacy_ms']['median']:>14.3f} "
                  f"{row['twin_ms']['median']:>12.3f} {row['speedup_median']:>8.2f}x",
                  file=sys.stderr)

    summary = {
        "events": args.events,
        "repeats": args.repeats,
        "seed": args.seed,
        "fill": "MID",
        "verify_env_during_timing": "unset",
        "modes": modes,
        "mismatches": mismatches,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "elapsed_s": round(time.perf_counter() - T0, 2),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if mismatches:
        log(f"FAILED: {mismatches} call(s) differed from the legacy path")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
