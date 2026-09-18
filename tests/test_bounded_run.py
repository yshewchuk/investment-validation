"""tools/bounded_run.py: the watchdog kills on real memory and on the box floor.

Each test runs a tiny child (at most a few hundred MB) under the real watchdog.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "bounded_run.py"


def _run(*flags: str, code: str, timeout: float = 60) -> tuple[int, str, float]:
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, str(RUNNER), "--cores", "1", *flags, "--",
         sys.executable, "-c", code],
        capture_output=True, text=True, timeout=timeout,
    )
    return done.returncode, done.stdout + done.stderr, time.monotonic() - started


def test_exit_code_passes_through_under_the_cap():
    code, out, _ = _run("--max-rss-gb", "2", "--min-free-gb", "0",
                        code="import sys; sys.exit(3)")
    assert code == 3
    assert "exited 3" in out


def test_cap_breach_on_real_memory_kills_within_a_second_or_two():
    grow = ("import time; b = bytearray(300 * 1024 * 1024); "
            "b[::4096] = b'x' * len(b[::4096]); time.sleep(60)")
    code, out, took = _run("--max-rss-gb", "0.1", "--min-free-gb", "0",
                           "--kill-grace-s", "5", code=grow)
    assert code == 137
    assert "CAP BREACH" in out
    assert took < 20  # the child would otherwise sleep for 60 s


def test_reserved_but_untouched_memory_does_not_breach():
    # Address space is not real memory: a large untouched mapping must not
    # trip a cap sized on resident memory.
    reserve = ("import mmap, time; m = mmap.mmap(-1, 2 * 1024**3); "
               "time.sleep(1.5)")
    code, out, _ = _run("--max-rss-gb", "0.5", "--min-free-gb", "0",
                        code=reserve)
    assert code == 0, out
    assert "CAP BREACH" not in out


def test_box_floor_breach_sigkills_at_once():
    # A floor no machine can meet stands in for a box running out of memory.
    code, out, took = _run("--max-rss-gb", "4", "--min-free-gb", "100000",
                           code="import time; time.sleep(60)")
    assert code == 137
    assert "BOX FLOOR BREACH" in out
    assert "largest processes on the box" in out
    assert took < 20


def test_child_ignoring_sigterm_is_sigkilled_after_grace():
    stubborn = ("import signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "b = bytearray(300 * 1024 * 1024); "
                "b[::4096] = b'x' * len(b[::4096]); time.sleep(60)")
    code, out, took = _run("--max-rss-gb", "0.1", "--min-free-gb", "0",
                           "--kill-grace-s", "1", code=stubborn)
    assert code == 137
    assert "SIGKILL" in out
    assert took < 20


def test_fractional_poll_and_negative_values_are_validated():
    code, _, _ = _run("--poll-s", "0", code="pass")
    assert code != 0
    code, _, _ = _run("--min-free-gb", "-1", code="pass")
    assert code != 0
    code, out, _ = _run("--poll-s", "0.1", "--min-free-gb", "0", code="pass")
    assert code == 0
    assert "poll=0.1s" in out
