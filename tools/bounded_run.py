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
  An external watchdog polls the job's whole process tree once per
  ``--poll-s`` seconds and logs a heartbeat line with the tree's proportional
  RSS. At ``--warn-pct`` of the cap it logs a warning; at the cap it SIGTERMs
  the tree (SIGKILL after ``--kill-grace-s``), prints the per-process memory
  breakdown, and exits 137. That turns this box's failure mode — a silent OOM
  that kills a neighbor's job with no traceback anywhere — into a clean,
  explained, resumable abort of exactly one job.

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
        [--poll-s S] -- <command...>

    python3 tools/bounded_run.py --max-rss-gb 5.5 -- \\
        python3 -m engine.dashboard.nightly --as-of 2026-09-09
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

POLL_DEFAULT_S = 30
KILL_GRACE_DEFAULT_S = 45


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
    parser.add_argument("--max-rss-gb", type=float, default=5.5,
                        help="kill the tree when its proportional RSS crosses "
                             "this cap (default: 5.5)")
    parser.add_argument("--warn-pct", type=float, default=85.0,
                        help="log a warning at this share of the cap")
    parser.add_argument("--poll-s", type=int, default=POLL_DEFAULT_S,
                        help="watchdog interval in seconds")
    parser.add_argument("--kill-grace-s", type=int, default=KILL_GRACE_DEFAULT_S,
                        help="seconds between SIGTERM and SIGKILL")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the command to run; prefix it with -- if it "
                             "carries its own flags")
    args = parser.parse_args()
    command = [c for c in args.command if c != "--"] or ["true"]
    if not args.max_rss_gb > 0:
        parser.error("--max-rss-gb must be positive")

    cap_mb = args.max_rss_gb * 1024.0
    cores = f"0-{args.cores - 1}" if args.cores > 1 else "0"
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.cores)

    print(f"[bounded] cap={args.max_rss_gb:g}G warn at {args.warn_pct:g}% "
          f"cores={cores} of {os.cpu_count()} poll={args.poll_s}s "
          f"nice=19 threads={args.cores}", flush=True)
    print(f"[bounded] command: {' '.join(command)}", flush=True)

    started = time.monotonic()
    proc = subprocess.Popen(
        ["taskset", "-c", cores, "nice", "-n", "19", *command],
        env=env, start_new_session=True,
    )

    warned = False
    while True:
        code = proc.poll()
        if code is not None:
            elapsed = time.monotonic() - started
            print(f"[bounded] exited {code} after {elapsed / 60.0:.1f} min",
                  flush=True)
            return code
        tree = _descendants(proc.pid)
        vm_total = pss_total = 0.0
        for pid in tree:
            vm, pss = _rss_mb(pid)
            vm_total += vm
            pss_total += pss
        elapsed = time.monotonic() - started
        pct = 100.0 * pss_total / cap_mb
        line = (f"[watchdog] {elapsed / 60.0:6.1f}m rss "
                f"{pss_total / 1024.0:5.2f}G pss ({vm_total / 1024.0:.2f}G vm, "
                f"{len(tree)} procs) = {pct:.0f}% of cap")
        if pss_total >= cap_mb:
            print(f"{line} — CAP BREACH, killing tree", flush=True)
            print("[bounded] per-process memory at breach:", flush=True)
            rows = sorted(((  _rss_mb(p)[1], p) for p in tree), reverse=True)
            for pss, pid in rows[:8]:
                print(f"  {pss / 1024.0:5.2f}G  pid {pid}  {_cmdline(pid)}",
                      flush=True)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            deadline = time.monotonic() + args.kill_grace_s
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(2.0)
            if proc.poll() is None:
                print("[bounded] SIGTERM ignored; SIGKILL", flush=True)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            print("[bounded] killed at the memory cap — the job is resumable; "
                  "raise --max-rss-gb or free memory before re-running",
                  flush=True)
            return 137
        if pct >= args.warn_pct and not warned:
            print(f"{line} — WARNING, approaching cap", flush=True)
            warned = True
        elif pct < args.warn_pct * 0.9:
            warned = False
        if int(elapsed) % max(60, args.poll_s) < args.poll_s:
            print(line, flush=True)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
