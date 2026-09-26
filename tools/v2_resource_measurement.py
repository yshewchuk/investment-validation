#!/usr/bin/env python3
"""P6-6 resource measurement: wrap a rehearsal/qualification run in
``tools/bounded_run.py`` and record what the box looked like before, during
and after it.

This recorder never reimplements the watchdog. It launches
``tools/bounded_run.py`` as a child, streams the child's output line by line
while capturing it (so a long real nightly stays watchable), and derives the
two values only the child's own log can supply:

* ``peak_rss_gb`` -- a LOWER BOUND: the maximum ``rss <N.NNG> pss`` reading
  across the ``[watchdog]`` lines ``bounded_run.py`` printed, sampled only
  every ``HEARTBEAT_MINUTES`` minutes, so the true peak between samples is
  unrecorded. ``bounded_run.py`` keeps no peak of its own, so this is parsed
  from the captured output. ``None`` both when no watchdog line ever appeared
  AND when the only line(s) seen were the unconditional ``elapsed=0.0``
  startup reading; ``0.0`` would claim a measurement that was never taken.
* ``kill_reason`` -- for a child that exited 137, which of the watchdog's
  literal breach strings appeared (``BOX FLOOR BREACH``, ``SWAP BREACH``,
  ``CAP BREACH``, in the watchdog's own precedence). ``None`` for any other
  exit, and for a 137 with no breach string at all -- ``killed`` is still
  true there, because the exit code is the fact and the reason is the
  absence.

Contention is recorded, never waited on: the supervisor owns the "one heavy
job at a time" rule (AGENTS.md) and the pre-launch ``free -m`` floor; this
tool only writes down what ``pgrep`` saw at launch, using AGENTS.md's exact
bracket patterns. The available RAM before and after, and the wall clock,
complete the record.

Every run writes
``<evidence-dir>/<workload-label>-<UTC timestamp>.json`` (default evidence
directory ``reports/phase6_evidence/resource_measurement/``), prints the
record, and exits with the CHILD's own exit code -- never swallowing a
non-zero exit or a watchdog kill into 0.

Usage::

    python3 tools/v2_resource_measurement.py --workload-label nightly-shadow \\
        --max-rss-gb 8 --max-swap-gb 6 --cores 8 --cache-state cold \\
        --capabilities-covered nightly-score,nightly-publish \\
        -- python3 -m engine.v2.ops nightly ...
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

__all__ = [
    "SCHEMA_VERSION",
    "BOUNDED_RUN",
    "DEFAULT_EVIDENCE_DIR",
    "CACHE_STATES",
    "HEAVY_JOB_PATTERNS",
    "HEARTBEAT_MINUTES",
    "KILL_REASONS",
    "parse_free_available_gb",
    "collect_contention",
    "watchdog_rss_values",
    "classify_kill",
    "measure",
    "main",
]

SCHEMA_VERSION = "v2_resource_measurement_record.v1.0"
BOUNDED_RUN = ROOT / "tools" / "bounded_run.py"
DEFAULT_EVIDENCE_DIR = Path("reports/phase6_evidence/resource_measurement")
CACHE_STATES = ("cold", "warm", "unknown")

#: AGENTS.md's exact bracket patterns, so pgrep can never match its own
#: command line (or a process that merely carries the pattern text).
HEAVY_JOB_PATTERNS = (
    ("bounded_run.py", "[b]ounded_run.py"),
    ("serve_monitor", "[s]erve_monitor"),
    ("corpus_parity.py run", "[c]orpus_parity.py run"),
)
#: bounded_run.py's three watchdog breach strings, in the watchdog's own order.
KILL_REASONS = ("BOX FLOOR BREACH", "SWAP BREACH", "CAP BREACH")

_WATCHDOG_RSS = re.compile(r"\[watchdog\]\s+([\d.]+)m\s+rss\s+(\d+\.\d+)G\s+pss")
_LABEL = re.compile(r"[A-Za-z0-9._-]+")


def parse_free_available_gb(text: str) -> float | None:
    """The ``available`` column of ``free -m``'s ``Mem:`` row, in GiB."""
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 7 and fields[0].rstrip(":") == "Mem":
            try:
                return round(int(fields[6]) / 1024.0, 3)
            except ValueError:
                return None
    return None


def _available_ram_gb() -> float | None:
    try:
        proc = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_free_available_gb(proc.stdout)


def _pgrep(pattern: str) -> list[int]:
    try:
        proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for token in proc.stdout.split():
        try:
            pids.append(int(token))
        except ValueError:
            continue
    return pids


def collect_contention(pgrep=_pgrep) -> dict:
    """Record-only contention snapshot: each AGENTS.md pattern -> its PIDs."""
    return {"other_heavy_jobs": {name: pgrep(pattern) for name, pattern in HEAVY_JOB_PATTERNS}}


#: bounded_run.py's HEARTBEAT_S (tools/bounded_run.py:109), in minutes. The
#: watchdog's own first-iteration heartbeat fires at elapsed=0.0 regardless of
#: this interval (its ``last_beat`` starts at -inf) -- that reading is a
#: startup artifact, not a real periodic sample, so it is excluded here.
HEARTBEAT_MINUTES = 1.0


def watchdog_rss_values(output: str) -> list[float]:
    """Every ``rss N.NNG pss`` reading from a REAL heartbeat interval --
    excludes bounded_run.py's unconditional elapsed=0.0 first-iteration
    reading, which fires before the child has grown to its true RSS and is
    not a periodic sample."""
    return [float(rss) for elapsed, rss in _WATCHDOG_RSS.findall(output)
            if float(elapsed) >= HEARTBEAT_MINUTES]


def classify_kill(output: str, exit_code: int) -> str | None:
    """Which watchdog breach string killed the child, or None."""
    if exit_code != 137:
        return None
    for reason in KILL_REASONS:
        if reason in output:
            return reason
    return None


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%S%fZ")


def _bounded_argv(bounded_run: Path, max_rss_gb, max_swap_gb, cores, cpu_set,
                  command: list[str]) -> list[str]:
    """The child argv: optional flags appear only when the operator gave them."""
    return [
        "python3", str(bounded_run),
        "--max-rss-gb", str(max_rss_gb),
        *(["--max-swap-gb", str(max_swap_gb)] if max_swap_gb is not None else []),
        *(["--cores", str(cores)] if cores is not None else []),
        *(["--cpu-set", cpu_set] if cpu_set else []),
        "--", *command,
    ]


def _stream(command: list[str]) -> tuple[list[str], int]:
    """Run the child, echoing each captured line as it arrives.

    If forwarding to this process's own stdout raises (a closed pipe, a
    write error), the child is terminated rather than left running: without
    this, a parent-side write failure would fall straight to ``proc.wait()``
    with no signal ever sent, and the child (and whatever heavy-job
    reservation it holds) could keep running indefinitely.
    """
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    proc.wait()
    return lines, proc.returncode


def _write_evidence(record: dict, evidence_dir: Path) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"{record['workload_label']}-{_stamp(datetime.now(timezone.utc))}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def measure(workload_label, max_rss_gb, cache_state, command, *,
            max_swap_gb=None, cores=None, cpu_set=None,
            capabilities_covered=(), evidence_dir=DEFAULT_EVIDENCE_DIR,
            bounded_run=None) -> dict:
    """Run ``command`` under ``bounded_run.py``, write and return the record."""
    started = datetime.now(timezone.utc)
    available_before = _available_ram_gb()
    contention = collect_contention()
    argv = _bounded_argv(Path(bounded_run) if bounded_run is not None else BOUNDED_RUN,
                         max_rss_gb, max_swap_gb, cores, cpu_set, list(command))
    lines, exit_code = _stream(argv)
    ended = datetime.now(timezone.utc)
    output = "".join(lines)
    rss_values = watchdog_rss_values(output)
    record = {
        "schema_version": SCHEMA_VERSION,
        "workload_label": workload_label,
        "command": list(command),
        "started_at": _iso(started),
        "ended_at": _iso(ended),
        "wall_seconds": round((ended - started).total_seconds(), 3),
        "cache_state": cache_state,
        "available_ram_gb_before": available_before,
        "available_ram_gb_after": _available_ram_gb(),
        "contention": contention,
        "cpu_set": cpu_set,
        "cores": cores,
        "max_rss_gb_cap": max_rss_gb,
        "max_swap_gb_cap": max_swap_gb,
        # None (not 0.0) when no full-interval sample exists -- see
        # HEARTBEAT_MINUTES; a returned value is a LOWER BOUND (bounded_run.py
        # only samples every HEARTBEAT_S, so the true peak between samples is
        # unrecorded).
        "peak_rss_gb": max(rss_values) if rss_values else None,
        "exit_code": exit_code,
        "killed": exit_code == 137,
        "kill_reason": classify_kill(output, exit_code),
        "capabilities_covered": list(capabilities_covered),
    }
    _write_evidence(record, Path(evidence_dir))
    return record


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload-label", required=True,
                        help="names the evidence file, so letters/digits/._- only")
    parser.add_argument("--max-rss-gb", required=True, type=float)
    parser.add_argument("--max-swap-gb", type=float, default=None)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--cpu-set", default=None)
    parser.add_argument("--cache-state", required=True, choices=CACHE_STATES)
    parser.add_argument("--capabilities-covered", default="",
                        help="comma-separated tools/phase6_capabilities.toml row ids")
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="the workload, after a -- separator")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("v2-resource-measurement: no workload command after --", file=sys.stderr)
        return 2
    if not _LABEL.fullmatch(args.workload_label):
        print(f"v2-resource-measurement: bad --workload-label {args.workload_label!r}; "
              "letters/digits/._- only (it names the evidence file)", file=sys.stderr)
        return 2
    record = measure(
        workload_label=args.workload_label, max_rss_gb=args.max_rss_gb,
        cache_state=args.cache_state, command=command,
        max_swap_gb=args.max_swap_gb, cores=args.cores, cpu_set=args.cpu_set,
        capabilities_covered=[item.strip() for item in args.capabilities_covered.split(",")
                              if item.strip()],
        evidence_dir=args.evidence_dir)
    print(json.dumps(record, indent=2, sort_keys=True))
    return int(record["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
