#!/usr/bin/env python3
"""Run a long job with bounded CPU and memory on a shared box.

This environment has no writable cgroups and no systemd (probed 2026-09-09),
so a hard kernel memory cap is not available. What IS available, applied here:

CPU
  ``taskset`` pins the job to the first ``--cores`` cores (default: half the
  box), so co-running agents keep the rest; BLAS/OMP thread counts are set to
  the same number so numpy cannot oversubscribe the pinned set; ``nice -n 19``
  makes the job yield its cores whenever anything else wants them.

MEMORY
  An external watchdog checks the job's whole process tree every ``--poll-s``
  seconds (default 0.25) and logs a heartbeat line once a minute. Two limits,
  both on real (resident) memory, never on address space:

  * the job's own cap, ``--max-rss-gb``: a cheap VmRSS sum each tick, confirmed
    with proportional RSS (Pss) before acting, so shared pages never trigger a
    false kill. Breach: SIGTERM, then SIGKILL after ``--kill-grace-s`` or at
    once if the box floor is crossed meanwhile.
  * the box floor, ``--min-free-gb`` (default 0.5): MemAvailable for the whole
    machine. Other sessions share this box, so a job under its own cap can
    still starve docker when a neighbour grows. Breach: SIGKILL at once; this
    is an emergency, not a courtesy stop.

  Why the interval matters: the watchdog acts only when it looks, so the worst
  overshoot is growth rate x interval. The 30 s default this replaced let a
  capture climb gigabytes between looks and took docker down (2026-09-18); at
  0.25 s the overshoot is tens of MB. A tick reads a few /proc files.

  Either breach prints the memory breakdown and exits 137. That turns this
  box's failure mode — a silent OOM that kills a neighbor's job with no
  traceback anywhere — into a clean, explained abort of exactly one job.

What an aborted nightly actually costs, since this said "resumable" and there
are no checkpoints to resume from: the FETCHED work survives, because the store
is written as it goes and a re-run skips chain pairs it already holds, the Tier
3/4 tables are rebuilt in place, and `ledger.snapshot` filters on
`existing_row_ids()` so re-recording a night writes nothing twice. The COMPUTED
work does not — scoring, the ladder, rendering and the self-check are held in
memory until the publish, so a kill during them loses all of it, and because
the state file is only written at step 7 the next run re-covers that night via
the backfill. Scoring is the longest phase, so an abort is most likely to land
exactly where the loss is largest.

So: an abort costs the night's compute, not its downloads or its ledger. The
watchdog heartbeat also satisfies the house rule that a long job logs at least
once a minute.

Usage:
    python3 tools/bounded_run.py [--cores N] [--max-rss-gb G] \\
        [--min-free-gb F] [--poll-s S] -- <command...>

    python3 tools/bounded_run.py --max-rss-gb 5.5 -- \\
        python3 -m engine.dashboard.nightly --as-of 2026-09-09
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

POLL_DEFAULT_S = 0.25
MIN_FREE_DEFAULT_GB = 0.5
KILL_GRACE_DEFAULT_S = 45
HEARTBEAT_S = 60.0


def _read_status(pid: int) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                out[key.strip()] = value.strip()
    except (FileNotFoundError, ProcessLookupError):
        pass
    return out


def _rss_mb(pid: int) -> tuple[float, float]:
    """``(vm_mb, pss_mb)`` for one process; Pss from smaps_rollup when readable.

    VmRSS double-counts shared pages across processes, so the tree total uses
    Pss where the kernel provides it and falls back to VmRSS otherwise.
    """
    status = _read_status(pid)
    vm = 0.0
    for key in ("VmRSS",):
        raw = status.get(key, "")
        if raw.endswith("kB"):
            vm = float(raw[:-2]) / 1024.0
    pss = vm
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                pss = float(line.split()[1]) / 1024.0
                break
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        pass
    return vm, pss


def _vmrss_mb(pid: int) -> float:
    """VmRSS alone: the cheap per-tick reading (no smaps walk)."""
    raw = _read_status(pid).get("VmRSS", "")
    return float(raw[:-2]) / 1024.0 if raw.endswith("kB") else 0.0


def _available_mb() -> float:
    """The box's MemAvailable, in MB (inf if unreadable, so it never kills)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except (OSError, IndexError, ValueError):
        pass
    return float("inf")


def _box_top(limit: int = 8) -> list[tuple[float, int]]:
    """The box's largest processes by VmRSS, for the floor-breach report."""
    rows = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            rows.append((_vmrss_mb(int(entry)), int(entry)))
    return sorted(rows, reverse=True)[:limit]


def _descendants(root: int) -> list[int]:
    """``root`` plus every process descended from it, from one /proc scan."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                parts = fh.read().rpartition(b")")[2].split()
            ppid = int(parts[1])
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError,
                OSError, PermissionError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, ()))
    return out


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
            "utf-8", "replace")
        return " ".join(raw.split("\0"))[:100]
    except (FileNotFoundError, ProcessLookupError, OSError):
        return "<gone>"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cores", type=int, default=max(1, os.cpu_count() // 2),
                        help="pin to this many cores (default: half the box)")
    parser.add_argument("--cpu-set", default=None,
                        help="exact taskset CPU list, for example 8 or 8-9; "
                             "overrides --cores placement")
    parser.add_argument("--max-rss-gb", type=float, default=5.5,
                        help="kill the tree when its proportional RSS crosses "
                             "this cap (default: 5.5)")
    parser.add_argument("--warn-pct", type=float, default=85.0,
                        help="log a warning at this share of the cap")
    parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_DEFAULT_GB,
                        help="SIGKILL the job when the whole box's MemAvailable "
                             "drops below this (default: 0.5)")
    parser.add_argument("--poll-s", type=float, default=POLL_DEFAULT_S,
                        help="watchdog interval in seconds (default: 0.25)")
    parser.add_argument("--kill-grace-s", type=int, default=KILL_GRACE_DEFAULT_S,
                        help="seconds between SIGTERM and SIGKILL")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the command to run; prefix it with -- if it "
                             "carries its own flags")
    args = parser.parse_args()
    command = [c for c in args.command if c != "--"] or ["true"]
    if not args.max_rss_gb > 0:
        parser.error("--max-rss-gb must be positive")
    if not args.poll_s > 0:
        parser.error("--poll-s must be positive")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must not be negative")
    floor_mb = args.min_free_gb * 1024.0

    cap_mb = args.max_rss_gb * 1024.0
    cores = args.cpu_set or (f"0-{args.cores - 1}" if args.cores > 1 else "0")
    if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", cores):
        parser.error("--cpu-set must contain comma-separated CPU numbers or ranges")
    selected_cpus: set[int] = set()
    for part in cores.split(","):
        bounds = [int(value) for value in part.split("-")]
        start, end = (bounds[0], bounds[-1])
        if end < start:
            parser.error("--cpu-set ranges must be ascending")
        selected_cpus.update(range(start, end + 1))
    if not selected_cpus or max(selected_cpus) >= os.cpu_count():
        parser.error("--cpu-set contains an unavailable CPU")
    worker_cores = len(selected_cpus)
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(worker_cores)

    print(f"[bounded] cap={args.max_rss_gb:g}G warn at {args.warn_pct:g}% "
          f"box floor={args.min_free_gb:g}G "
          f"cores={cores} of {os.cpu_count()} poll={args.poll_s:g}s "
          f"nice=19 threads={worker_cores}", flush=True)
    print(f"[bounded] command: {' '.join(command)}", flush=True)

    started = time.monotonic()
    proc = subprocess.Popen(
        ["taskset", "-c", cores, "nice", "-n", "19", *command],
        env=env, start_new_session=True,
    )

    warned = False
    last_beat = float("-inf")
    while True:
        code = proc.poll()
        if code is not None:
            elapsed = time.monotonic() - started
            print(f"[bounded] exited {code} after {elapsed / 60.0:.1f} min",
                  flush=True)
            return code
        elapsed = time.monotonic() - started
        available = _available_mb()
        if available < floor_mb:
            tree = set(_descendants(proc.pid))
            print(f"[watchdog] {elapsed / 60.0:6.1f}m box MemAvailable "
                  f"{available / 1024.0:.2f}G < floor {args.min_free_gb:g}G "
                  f"— BOX FLOOR BREACH, SIGKILL", flush=True)
            print("[bounded] largest processes on the box (* = this job):",
                  flush=True)
            for rss, pid in _box_top():
                mark = "*" if pid in tree else " "
                print(f" {mark}{rss / 1024.0:5.2f}G  pid {pid}  {_cmdline(pid)}",
                      flush=True)
            _kill(proc, signal.SIGKILL)
            print("[bounded] killed at the box memory floor", flush=True)
            return 137
        tree = _descendants(proc.pid)
        rss_total = sum(_vmrss_mb(pid) for pid in tree)
        heartbeat = elapsed - last_beat >= HEARTBEAT_S
        if rss_total >= cap_mb * args.warn_pct / 100.0 or heartbeat:
            # VmRSS double-counts pages shared across the tree; confirm with
            # Pss before warning or killing, and for the heartbeat line.
            vm_total = pss_total = 0.0
            for pid in tree:
                vm, pss = _rss_mb(pid)
                vm_total += vm
                pss_total += pss
        else:
            vm_total = pss_total = rss_total
        pct = 100.0 * pss_total / cap_mb
        line = (f"[watchdog] {elapsed / 60.0:6.1f}m rss "
                f"{pss_total / 1024.0:5.2f}G pss ({vm_total / 1024.0:.2f}G vm, "
                f"{len(tree)} procs) = {pct:.0f}% of cap; box free "
                f"{available / 1024.0:.2f}G")
        if pss_total >= cap_mb:
            print(f"{line} — CAP BREACH, killing tree", flush=True)
            print("[bounded] per-process memory at breach:", flush=True)
            rows = sorted(((_rss_mb(p)[1], p) for p in tree), reverse=True)
            for pss, pid in rows[:8]:
                print(f"  {pss / 1024.0:5.2f}G  pid {pid}  {_cmdline(pid)}",
                      flush=True)
            _kill(proc, signal.SIGTERM)
            deadline = time.monotonic() + args.kill_grace_s
            while time.monotonic() < deadline and proc.poll() is None:
                if _available_mb() < floor_mb:
                    print("[bounded] box floor crossed during grace; SIGKILL",
                          flush=True)
                    break
                time.sleep(args.poll_s)
            if proc.poll() is None:
                print("[bounded] SIGTERM not honoured; SIGKILL", flush=True)
                _kill(proc, signal.SIGKILL)
            print("[bounded] killed at the memory cap — the job is resumable; "
                  "raise --max-rss-gb or free memory before re-running",
                  flush=True)
            return 137
        if pct >= args.warn_pct and not warned:
            print(f"{line} — WARNING, approaching cap", flush=True)
            warned = True
        elif pct < args.warn_pct * 0.9:
            warned = False
        if heartbeat:
            print(line, flush=True)
            last_beat = elapsed
        time.sleep(args.poll_s)


def _kill(proc: subprocess.Popen, sig: int) -> None:
    """Signal the job's process group; for SIGKILL, also reap it."""
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    if sig == signal.SIGKILL:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

if __name__ == "__main__":
    sys.exit(main())
