#!/usr/bin/env python3
"""Run a long job with bounded CPU and memory on a shared box.

cgroup v2 limits ARE readable here (``/sys/fs/cgroup/memory.max`` and
``memory.swap.max``), and the container is hard-capped by the kernel at
those values (confirmed 2026-09-20; corrects an earlier claim here, probed
2026-09-09, that no writable cgroups or hard kernel memory cap existed).
This tool still supervises from OUTSIDE that cap: it bounds one named
process TREE and reports exactly what it saw, which a bare cgroup limit
does not do on its own — a cgroup OOM kill leaves no per-process breakdown.
Building cgroup-based enforcement is future work, not done here.

Caution, worth carrying into any future work here: ``/proc`` readings on
this box (``MemAvailable``, ``MemTotal``) are the HOST/VM's numbers, not
necessarily the container's cgroup limit, and the two are not guaranteed to
agree. A tier-0 capture was killed twice at the box floor while its own
cgroup had room, because the VM was 7.97 GiB while the container's cgroup
allowed 10.5 GiB; the VM has since been resized to 10.7 GiB so the two now
roughly match, but a future mismatch (either direction) is a live hazard,
not a one-time fluke already closed off.

CPU
  ``taskset`` pins the job to the first ``--cores`` cores (default: half the
  box), so co-running agents keep the rest; BLAS/OMP thread counts are set to
  the same number so numpy cannot oversubscribe the pinned set; ``nice -n 19``
  makes the job yield its cores whenever anything else wants them.

MEMORY
  An external watchdog checks the job's whole process tree every ``--poll-s``
  seconds (default 0.25) and logs a heartbeat line once a minute carrying
  RSS, swap and box-free memory together, so a paging job is visible in the
  log without extra tooling. Three limits act on the tree, all measured from
  ``/proc``, never on address space:

  * the job's own RSS cap, ``--max-rss-gb``: a cheap VmRSS sum each tick,
    confirmed with proportional RSS (Pss) before acting, so shared pages
    never trigger a false kill. Breach: SIGTERM, then SIGKILL after
    ``--kill-grace-s`` or at once if the box floor is crossed meanwhile.
  * the job's own SWAP cap, ``--max-swap-gb`` (default 0.5, ON by default —
    most bounded runs should have a kill switch for thrashing): summed the
    same way over ``VmSwap`` (confirmed with SwapPss). This exists because
    the other two limits both RELAX exactly when a job starts paging: RSS
    counts only resident pages, so a process being swapped out appears to
    SHRINK, and MemAvailable RECOVERS as pages move to swap. Without a swap
    cap, a thrashing job is invisible to this watchdog — comfortable numbers
    on both other limits while the job takes hours instead of minutes.
    ``--max-swap-gb 0`` kills on ANY swap; a NEGATIVE value (for example
    ``-1``) is the only way to disable the swap check entirely. Breach
    behaviour matches the RSS cap (SIGTERM, then SIGKILL after
    ``--kill-grace-s``) and prints the same style of per-process breakdown,
    with each process's swap shown.
  * the box floor, ``--min-free-gb`` (default 0.5): MemAvailable for the whole
    machine. Other sessions share this box, so a job under its own cap can
    still starve docker when a neighbour grows. Breach: SIGKILL at once; this
    is an emergency, not a courtesy stop.

  A ``--max-rss-gb`` set ABOVE this box's MemTotal can only be reached by
  swapping, which is precisely what ``--max-swap-gb`` exists to discourage.
  At startup, such a cap prints a loud warning naming both numbers. It is
  NOT refused — a legitimate caller may want that on a machine whose limits
  differ from this one — just impossible to set by accident and not notice.

  Why the interval matters: the watchdog acts only when it looks, so the worst
  overshoot is growth rate x interval. The 30 s default this replaced let a
  capture climb gigabytes between looks and took docker down (2026-09-18); at
  0.25 s the overshoot is tens of MB. A tick reads a few /proc files.

  Any breach prints the memory breakdown and exits 137. That turns this
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

COORDINATION

  One job's cap says nothing about the next job's, so concurrent runs also
  coordinate through ``BOUNDED_RUN_STATE_DIR`` (default
  ``/tmp/bounded_run_state``, created on demand):

  * ``--heavy`` marks the one big job (the nightly, cap 8 GB): it publishes
    ``heavy-<pid>.json`` (pid, reserve, start, argv0) and holds that file's
    flock for its whole life, so at most one heavy job runs at a time. A
    second heavy job waits (RESOURCE WAIT, every 30 s) for the first to
    finish; a file whose lock is free belonged to a crashed run, and the
    next reader deletes it and ignores it.
  * every non-heavy job holds one of ``BOUNDED_RUN_SLOTS`` test slots
    (default 3, ``slot-<i>.lock`` files), dropping to
    ``BOUNDED_RUN_SLOTS_UNDER_HEAVY`` (default 2) while a heavy reservation
    is live. A job only starts on a slot index below the current count; a
    running job is never preempted. With no free slot it waits (RESOURCE
    WAIT, every 5 s).
  * before starting, a non-heavy job also needs headroom: MemAvailable minus
    the unclaimed part of every live heavy reservation (its ``reserve_gb``
    less the heavy tree's current RSS). It starts only when that covers its
    own ``--max-rss-gb`` plus ``--min-free-gb``; otherwise it waits
    (RESOURCE WAIT, every 5 s), so the heavy job's reservation is not spent
    twice by test runs.

  ``--max-wait-s`` (default 3600) bounds every RESOURCE WAIT: on expiry the
  job exits 75 (EX_TEMPFAIL) without launching. A bounded_run started by
  another bounded_run inherits ``BOUNDED_RUN_NESTED=1`` and skips slots and
  admission entirely, because the outer job already holds a slot and
  reserved the memory; bounded_run sets that variable in every child
  environment it launches.

Usage:
    python3 tools/bounded_run.py [--cores N] [--max-rss-gb G] \\
        [--max-swap-gb G] [--min-free-gb F] [--poll-s S] [--heavy] \\
        [--max-wait-s S] -- <command...>

    python3 tools/bounded_run.py --max-rss-gb 5.5 -- \\
        python3 -m engine.dashboard.nightly --as-of 2026-09-09
"""
from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

POLL_DEFAULT_S = 0.25
MIN_FREE_DEFAULT_GB = 0.5
MAX_SWAP_DEFAULT_GB = 0.5
KILL_GRACE_DEFAULT_S = 45
HEARTBEAT_S = 60.0
MAX_WAIT_DEFAULT_S = 3600.0
STATE_DIR_ENV = "BOUNDED_RUN_STATE_DIR"
STATE_DIR_DEFAULT = "/tmp/bounded_run_state"
SLOTS_ENV = "BOUNDED_RUN_SLOTS"
SLOTS_UNDER_HEAVY_ENV = "BOUNDED_RUN_SLOTS_UNDER_HEAVY"
SLOTS_DEFAULT = 3
SLOTS_UNDER_HEAVY_DEFAULT = 2
NESTED_ENV = "BOUNDED_RUN_NESTED"
SLOT_POLL_S = 5.0
ADMISSION_POLL_S = 5.0
HEAVY_POLL_S = 30.0
EX_TEMPFAIL = 75


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


def _swap_mb(pid: int) -> tuple[float, float]:
    """``(vm_swap_mb, swap_pss_mb)`` for one process; SwapPss from
    smaps_rollup when readable, mirroring ``_rss_mb``.

    VmSwap can double-count swapped-out shared pages across processes, so the
    tree total uses SwapPss where the kernel provides it and falls back to
    VmSwap otherwise.
    """
    status = _read_status(pid)
    raw = status.get("VmSwap", "")
    vm = float(raw[:-2]) / 1024.0 if raw.endswith("kB") else 0.0
    pss = vm
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("SwapPss:"):
                pss = float(line.split()[1]) / 1024.0
                break
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        pass
    return vm, pss


def _vmswap_mb(pid: int) -> float:
    """VmSwap alone: the cheap per-tick reading (no smaps walk)."""
    raw = _read_status(pid).get("VmSwap", "")
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


def _mem_total_mb() -> float:
    """The box's MemTotal, in MB (inf if unreadable, so the startup warning
    about an over-large ``--max-rss-gb`` never false-fires)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
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


# --------------------------------------------------------------------------
# coordination: state dir, heavy reservations, test slots, admission
# --------------------------------------------------------------------------

#: FDs whose locks we hold; closed by ``_cleanup``. Paths are unlinked.
_CLEANUP_FDS: list[int] = []
_CLEANUP_PATHS: list[Path] = []
#: The running child's process group, so a signal to us reaches the job too.
_ACTIVE_PGID: int | None = None


def _cleanup() -> None:
    """Release slot/reservation locks and delete the heavy file. Idempotent."""
    while _CLEANUP_FDS:
        try:
            os.close(_CLEANUP_FDS.pop())
        except OSError:
            pass
    while _CLEANUP_PATHS:
        try:
            _CLEANUP_PATHS.pop().unlink()
        except OSError:
            pass


def _on_signal(signum, _frame) -> None:
    """Clean up, forward the signal to the child, then die by the same one."""
    if _ACTIVE_PGID is not None:
        try:
            os.killpg(_ACTIVE_PGID, signal.SIGTERM)
        except OSError:
            pass
    _cleanup()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_cleanup() -> None:
    atexit.register(_cleanup)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (OSError, ValueError):
            pass


def _state_dir() -> Path:
    path = Path(os.environ.get(STATE_DIR_ENV) or STATE_DIR_DEFAULT)
    path.mkdir(mode=0o755, parents=True, exist_ok=True)
    return path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _try_lock(path: Path) -> int | None:
    """Open/create ``path`` and take LOCK_EX|LOCK_NB; return the held FD."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _read_heavy(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _live_heavies(state: Path) -> list[dict]:
    """Every live heavy reservation, deleting stale files as they are found.

    A file whose flock can be acquired has no live holder (the kernel drops
    the lock when the holder dies), so it is stale: unlink it and ignore it.
    """
    out: list[dict] = []
    for path in sorted(state.glob("heavy-*.json")):
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            entry = _read_heavy(path)
            if entry is not None:
                out.append(entry)
            continue
        try:
            path.unlink()
        except OSError:
            pass
        os.close(fd)
    return out


def _heavy_rss_mb(pid: int) -> float:
    return sum(_vmrss_mb(proc) for proc in _descendants(pid))


def _headroom_mb(heavies: list[dict]) -> float:
    """MemAvailable minus what live heavy reservations have still to claim.

    A reservation's claim is ``reserve_gb`` less its tree's current RSS: it
    already owns the resident part, so only the unclaimed remainder has to be
    held back from new (non-heavy) jobs.
    """
    committed = 0.0
    for entry in heavies:
        try:
            pid = int(entry.get("pid", 0))
            reserve_mb = float(entry.get("reserve_gb", 0.0)) * 1024.0
        except (TypeError, ValueError):
            continue
        committed += max(0.0, reserve_mb - _heavy_rss_mb(pid))
    return _available_mb() - committed


def _slot_count(heavies: list[dict]) -> int:
    if heavies:
        return _env_int(SLOTS_UNDER_HEAVY_ENV, SLOTS_UNDER_HEAVY_DEFAULT)
    return _env_int(SLOTS_ENV, SLOTS_DEFAULT)


def _acquire_slot(state: Path, count: int) -> int | None:
    for index in range(count):
        fd = _try_lock(state / f"slot-{index}.lock")
        if fd is not None:
            print(f"[bounded] holding test slot {index} of {count}", flush=True)
            return fd
    return None


def _reserve_heavy(state: Path, reserve_gb: float, argv0: str) -> None:
    path = state / f"heavy-{os.getpid()}.json"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    _CLEANUP_FDS.append(fd)
    _CLEANUP_PATHS.append(path)
    fcntl.flock(fd, fcntl.LOCK_EX)
    payload = {
        "pid": os.getpid(),
        "reserve_gb": reserve_gb,
        "started_at": time.time(),
        "argv0": argv0,
    }
    os.write(fd, json.dumps(payload).encode())
    print(f"[bounded] heavy reservation held: {path} "
          f"(reserve {reserve_gb:g}G)", flush=True)


def _resource_wait(what: str, poll_s: float) -> None:
    print(f"[bounded] RESOURCE WAIT: {what}; retrying in {poll_s:g}s",
          flush=True)
    time.sleep(poll_s)


def _expired(deadline: float, what: str, max_wait_s: float) -> bool:
    if time.monotonic() < deadline:
        return False
    print(f"[bounded] RESOURCE WAIT timed out after {max_wait_s:g}s waiting "
          f"for {what}; exiting {EX_TEMPFAIL}", flush=True)
    return True


def _coordinate_heavy(state: Path, deadline: float, args,
                      command: list[str]) -> int | None:
    while True:
        live = _live_heavies(state)
        if not live:
            _reserve_heavy(state, args.max_rss_gb,
                           command[0] if command else "")
            return None
        if _expired(deadline, "another heavy reservation", args.max_wait_s):
            return EX_TEMPFAIL
        _resource_wait(f"heavy reservation held by pid {live[0].get('pid')}",
                       HEAVY_POLL_S)


def _coordinate_slot(state: Path, deadline: float, args,
                     cap_mb: float, floor_mb: float) -> int | None:
    required_mb = cap_mb + floor_mb
    while True:
        heavies = _live_heavies(state)
        headroom = _headroom_mb(heavies)
        if headroom < required_mb:
            if _expired(deadline, "headroom under a heavy reservation",
                        args.max_wait_s):
                return EX_TEMPFAIL
            _resource_wait(
                f"heavy reservation leaves headroom "
                f"{headroom / 1024.0:.2f}G < required "
                f"{required_mb / 1024.0:.2f}G", ADMISSION_POLL_S)
            continue
        count = _slot_count(heavies)
        fd = _acquire_slot(state, count)
        if fd is not None:
            _CLEANUP_FDS.append(fd)
            return None
        if _expired(deadline, f"a test slot (of {count})", args.max_wait_s):
            return EX_TEMPFAIL
        _resource_wait(f"test slot ({count} available, all held)", SLOT_POLL_S)


def _coordinate(args, cap_mb: float, floor_mb: float,
                command: list[str]) -> int | None:
    """Take a slot/reservation before launch; None = cleared to start, else."""
    if os.environ.get(NESTED_ENV) == "1":
        print(f"[bounded] nested run ({NESTED_ENV}=1): skipping slots and "
              "admission", flush=True)
        return None
    state = _state_dir()
    deadline = time.monotonic() + args.max_wait_s
    if args.heavy:
        return _coordinate_heavy(state, deadline, args, command)
    return _coordinate_slot(state, deadline, args, cap_mb, floor_mb)


def _terminate_with_grace(proc: subprocess.Popen, kill_grace_s: int,
                          poll_s: float, floor_mb: float) -> None:
    """SIGTERM the tree, wait up to ``kill_grace_s`` honouring the box floor,
    then SIGKILL if it is still alive. Shared by the RSS-cap and swap-cap
    breach paths so their grace-period behaviour cannot drift apart.
    """
    _kill(proc, signal.SIGTERM)
    deadline = time.monotonic() + kill_grace_s
    while time.monotonic() < deadline and proc.poll() is None:
        if _available_mb() < floor_mb:
            print("[bounded] box floor crossed during grace; SIGKILL",
                  flush=True)
            break
        time.sleep(poll_s)
    if proc.poll() is None:
        print("[bounded] SIGTERM not honoured; SIGKILL", flush=True)
        _kill(proc, signal.SIGKILL)


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--max-swap-gb", type=float, default=MAX_SWAP_DEFAULT_GB,
                        help="kill the tree when its swap crosses this cap "
                             "(default: 0.5, on by default). 0 kills on ANY "
                             "swap; a NEGATIVE value disables the swap check "
                             "entirely")
    parser.add_argument("--poll-s", type=float, default=POLL_DEFAULT_S,
                        help="watchdog interval in seconds (default: 0.25)")
    parser.add_argument("--kill-grace-s", type=int, default=KILL_GRACE_DEFAULT_S,
                        help="seconds between SIGTERM and SIGKILL")
    parser.add_argument("--heavy", action="store_true",
                        help="announce a heavy job: one at a time, and reserve "
                             "--max-rss-gb from other jobs while it runs")
    parser.add_argument("--max-wait-s", type=float, default=MAX_WAIT_DEFAULT_S,
                        help="max seconds to wait for a slot, a heavy "
                             "reservation or headroom before exiting 75 "
                             "(default: 3600)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the command to run; prefix it with -- if it "
                             "carries its own flags")
    return parser


def _validate(args, parser) -> None:
    if not args.max_rss_gb > 0:
        parser.error("--max-rss-gb must be positive")
    if not args.poll_s > 0:
        parser.error("--poll-s must be positive")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must not be negative")
    if args.max_wait_s < 0:
        parser.error("--max-wait-s must not be negative")


def _core_placement(args, parser) -> tuple[str, int]:
    cores = args.cpu_set or (f"0-{args.cores - 1}" if args.cores > 1 else "0")
    if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", cores):
        parser.error("--cpu-set must contain comma-separated CPU numbers or ranges")
    selected: set[int] = set()
    for part in cores.split(","):
        bounds = [int(value) for value in part.split("-")]
        start, end = (bounds[0], bounds[-1])
        if end < start:
            parser.error("--cpu-set ranges must be ascending")
        selected.update(range(start, end + 1))
    if not selected or max(selected) >= os.cpu_count():
        parser.error("--cpu-set contains an unavailable CPU")
    return cores, len(selected)


def _child_env(worker_cores: int) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(worker_cores)
    env[NESTED_ENV] = "1"
    return env


def _warn_cap_over_mem_total(max_rss_gb: float, mem_total_mb: float) -> None:
    print(f"[bounded] WARNING: --max-rss-gb {max_rss_gb:g}G exceeds "
          f"this box's MemTotal {mem_total_mb / 1024.0:.2f}G — that cap "
          f"can only be reached by swapping, which --max-swap-gb is here "
          f"to discourage; continuing anyway", flush=True)


def _print_startup(args, cores: str, worker_cores: int,
                   swap_enabled: bool) -> None:
    print(f"[bounded] cap={args.max_rss_gb:g}G warn at {args.warn_pct:g}% "
          f"swap-cap={'disabled' if not swap_enabled else f'{args.max_swap_gb:g}G'} "
          f"box floor={args.min_free_gb:g}G "
          f"cores={cores} of {os.cpu_count()} poll={args.poll_s:g}s "
          f"nice=19 threads={worker_cores}", flush=True)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    command = [c for c in args.command if c != "--"] or ["true"]
    _validate(args, parser)
    floor_mb = args.min_free_gb * 1024.0

    cap_mb = args.max_rss_gb * 1024.0
    swap_enabled = args.max_swap_gb >= 0
    # A NEGATIVE --max-swap-gb disables the check; 0 or above enables it, with
    # 0 meaning "kill on ANY swap". The breach test below uses a strict ">",
    # not ">=": with a 0 cap, ">=" would be true even at zero swap (0 >= 0)
    # and kill every job unconditionally, which is not "any swap", it is
    # "always". ">" only fires once swap actually becomes positive.
    swap_cap_mb = args.max_swap_gb * 1024.0 if swap_enabled else float("inf")

    mem_total_mb = _mem_total_mb()
    if cap_mb > mem_total_mb:
        _warn_cap_over_mem_total(args.max_rss_gb, mem_total_mb)

    cores, worker_cores = _core_placement(args, parser)
    env = _child_env(worker_cores)

    _print_startup(args, cores, worker_cores, swap_enabled)
    print(f"[bounded] command: {' '.join(command)}", flush=True)
    _install_cleanup()
    wait_code = _coordinate(args, cap_mb, floor_mb, command)
    if wait_code is not None:
        return wait_code

    started = time.monotonic()
    proc = subprocess.Popen(
        ["taskset", "-c", cores, "nice", "-n", "19", *command],
        env=env, start_new_session=True,
    )
    global _ACTIVE_PGID
    _ACTIVE_PGID = proc.pid
    try:
        return _watch(proc, args, cap_mb, floor_mb, swap_cap_mb, started)
    finally:
        _ACTIVE_PGID = None
        _cleanup()


def _log_exit(code: int, started: float) -> None:
    elapsed = time.monotonic() - started
    print(f"[bounded] exited {code} after {elapsed / 60.0:.1f} min", flush=True)


def _floor_breach(proc: subprocess.Popen, args, floor_mb: float,
                  elapsed: float, available: float, tree: set[int]) -> None:
    print(f"[watchdog] {elapsed / 60.0:6.1f}m box MemAvailable "
          f"{available / 1024.0:.2f}G < floor {args.min_free_gb:g}G "
          f"— BOX FLOOR BREACH, SIGKILL", flush=True)
    print("[bounded] largest processes on the box (* = this job):", flush=True)
    for rss, pid in _box_top():
        mark = "*" if pid in tree else " "
        print(f" {mark}{rss / 1024.0:5.2f}G  pid {pid}  {_cmdline(pid)}",
              flush=True)
    _kill(proc, signal.SIGKILL)
    print("[bounded] killed at the box memory floor", flush=True)


def _sample_tree(args, cap_mb: float, swap_cap_mb: float, available: float,
                 elapsed: float, heartbeat: bool, tree: list[int],
                 rss_total: float, swap_total: float) -> tuple[float, float, float, str]:
    near_swap_cap = swap_total > swap_cap_mb * args.warn_pct / 100.0
    if rss_total >= cap_mb * args.warn_pct / 100.0 or heartbeat or near_swap_cap:
        # VmRSS/VmSwap double-count pages shared across the tree; confirm
        # with Pss/SwapPss before warning or killing, and for the heartbeat.
        vm_total = pss_total = 0.0
        swap_pss_total = 0.0
        for pid in tree:
            vm, pss = _rss_mb(pid)
            vm_total += vm
            pss_total += pss
            swap_pss_total += _swap_mb(pid)[1]
    else:
        vm_total = pss_total = rss_total
        swap_pss_total = swap_total
    pct = 100.0 * pss_total / cap_mb
    line = (f"[watchdog] {elapsed / 60.0:6.1f}m rss "
            f"{pss_total / 1024.0:5.2f}G pss ({vm_total / 1024.0:.2f}G vm, "
            f"{len(tree)} procs) = {pct:.0f}% of cap; swap "
            f"{swap_pss_total / 1024.0:5.2f}G; box free "
            f"{available / 1024.0:.2f}G")
    return pss_total, swap_pss_total, pct, line


def _print_breach_rows(tree: list[int], kind: str) -> None:
    if kind == "swap":
        rows = sorted(((_swap_mb(p)[1], p) for p in tree), reverse=True)
        for value, pid in rows[:8]:
            print(f"  {value / 1024.0:5.2f}G swap  pid {pid}  {_cmdline(pid)}",
                  flush=True)
        return
    rows = sorted(((_rss_mb(p)[1], p) for p in tree), reverse=True)
    for value, pid in rows[:8]:
        print(f"  {value / 1024.0:5.2f}G  pid {pid}  {_cmdline(pid)}",
              flush=True)


def _watch(proc: subprocess.Popen, args, cap_mb: float, floor_mb: float,
           swap_cap_mb: float, started: float) -> int:
    warned = False
    last_beat = float("-inf")
    while True:
        code = proc.poll()
        if code is not None:
            _log_exit(code, started)
            return code
        elapsed = time.monotonic() - started
        available = _available_mb()
        tree = _descendants(proc.pid)
        if available < floor_mb:
            _floor_breach(proc, args, floor_mb, elapsed, available, set(tree))
            return 137
        rss_total = sum(_vmrss_mb(pid) for pid in tree)
        swap_total = sum(_vmswap_mb(pid) for pid in tree)
        heartbeat = elapsed - last_beat >= HEARTBEAT_S
        pss_total, swap_pss_total, pct, line = _sample_tree(
            args, cap_mb, swap_cap_mb, available, elapsed, heartbeat, tree,
            rss_total, swap_total)
        if swap_pss_total > swap_cap_mb:
            print(f"{line} — SWAP BREACH, killing tree", flush=True)
            print("[bounded] per-process memory at breach:", flush=True)
            _print_breach_rows(tree, "swap")
            _terminate_with_grace(proc, args.kill_grace_s, args.poll_s, floor_mb)
            print("[bounded] killed at the swap cap — the job is resumable; "
                  "raise --max-swap-gb or stop the paging before re-running",
                  flush=True)
            return 137
        if pss_total >= cap_mb:
            print(f"{line} — CAP BREACH, killing tree", flush=True)
            print("[bounded] per-process memory at breach:", flush=True)
            _print_breach_rows(tree, "rss")
            _terminate_with_grace(proc, args.kill_grace_s, args.poll_s, floor_mb)
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
