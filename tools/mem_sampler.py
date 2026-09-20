#!/usr/bin/env python3
"""Sample RSS and the running stack from INSIDE the target process.

Why not py-spy: it cannot attach on this WSL2 host, even as root --
``Failed to copy Py_Version symbol: Permission denied``. The kernel here
refuses the ptrace/``process_vm_readv`` it needs, so every external sampler
is unavailable. This one runs in-process and needs no such permission.

Why the samples survive a kill: a job that is being profiled for memory is
usually about to be killed by ``tools/bounded_run.py`` or the OOM killer, and
SIGKILL runs no ``finally`` and flushes no buffers. Samples are therefore
written line-buffered to a file as they are taken, so the trace that matters
-- the one right before death -- is already on disk.

Run a module or script under the sampler::

    python3 tools/mem_sampler.py --out /tmp/s.log -- checks/tier0_corpus.py --corpus <dir>
    python3 tools/mem_sampler.py --out /tmp/s.log --interval 0.25 -- -m engine.v2.ops --help

Then read it back, aggregated by stack, heaviest first::

    python3 tools/mem_sampler.py --report /tmp/s.log
    python3 tools/mem_sampler.py --report /tmp/s.log --above 2.0

Under bounded_run, put the sampler inside::

    python3 tools/bounded_run.py --max-rss-gb 4 -- python3 tools/mem_sampler.py --out ... -- ...

Cost is a sleeping thread that wakes a few times a second; it does not
change what the target allocates.

Blind spot: the sampler is a Python thread, so it only runs when the main
thread drops the GIL. Time spent inside ONE long C call -- a single huge
``str.join``, ``json.dumps`` or ``hashlib.update`` -- is therefore invisible:
the last sample before it shows the Python line that made the call, and the
next sample lands after it returns. That is usually enough, because the line
that entered the C call is the line that allocated. But it means the RSS
attributed to a stack is the RSS *entering* it, and a stack that appears once
with a large jump to the next sample is a stronger signal than the sample
count suggests. Read ``--report`` for the max per stack, not for the n.
"""
from __future__ import annotations

import argparse
import collections
import runpy
import sys
import threading
import time
import traceback
from pathlib import Path

__all__ = ["rss_gb", "start", "report"]

FRAMES = 8


def rss_gb() -> float:
    """Resident set of this process, in GB, straight from /proc."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def _where(frame) -> str:
    stack = traceback.extract_stack(frame)[-FRAMES:]
    return " < ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                      for f in reversed(stack))


def start(out_path: str, interval: float = 0.5) -> None:
    """Begin sampling the MAIN thread until the process exits."""
    handle = open(out_path, "w", buffering=1)  # line buffered: survives SIGKILL
    main_id = threading.main_thread().ident

    def loop() -> None:
        while True:
            frame = sys._current_frames().get(main_id)
            handle.write(f"{rss_gb():7.3f} {_where(frame) if frame else '<no frame>'}\n")
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True, name="mem-sampler").start()


def report(path: str, above: float = 0.0, top: int = 12) -> int:
    """Aggregate a sample file by stack: where the process spent its memory."""
    peak = 0.0
    per_stack: dict[str, list[float]] = collections.defaultdict(list)
    for line in Path(path).read_text(errors="ignore").splitlines():
        head, _, stack = line.strip().partition(" ")
        try:
            value = float(head)
        except ValueError:
            continue
        peak = max(peak, value)
        if value >= above:
            per_stack[stack].append(value)
    if not per_stack:
        print(f"{path}: no samples at or above {above:.2f}G (peak {peak:.2f}G)")
        return 1
    print(f"{path}: peak {peak:.2f}G, {sum(len(v) for v in per_stack.values())} samples "
          f"at or above {above:.2f}G\n")
    ranked = sorted(per_stack.items(), key=lambda kv: (max(kv[1]), len(kv[1])), reverse=True)
    for stack, values in ranked[:top]:
        print(f"  max {max(values):6.2f}G  n={len(values):<4} {stack}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", help="write samples here (line buffered)")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between samples")
    parser.add_argument("--report", help="summarise an existing sample file and exit")
    parser.add_argument("--above", type=float, default=0.0, help="with --report: ignore samples below this GB")
    parser.add_argument("--top", type=int, default=12, help="with --report: how many stacks to show")
    parser.add_argument("target", nargs=argparse.REMAINDER,
                        help="-- followed by a script path or -m module, then its arguments")
    args = parser.parse_args(argv)

    if args.report:
        return report(args.report, above=args.above, top=args.top)

    target = [a for a in args.target if a != "--"]
    if not target or not args.out:
        parser.error("need --out and a target after --, or --report <file>")

    start(args.out, interval=args.interval)
    if target[0] == "-m":
        if len(target) < 2:
            parser.error("-m needs a module name")
        sys.argv = [target[1], *target[2:]]
        runpy.run_module(target[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = list(target)
        runpy.run_path(target[0], run_name="__main__")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint
    sys.exit(main())
