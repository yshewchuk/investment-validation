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


def _madvise_pageout_child(alloc_mb: int, hold_s: float = 3.0) -> str:
    """Python source for a child that mmaps ``alloc_mb`` MB of PRIVATE
    anonymous memory, dirties every page, and forces it to swap immediately
    via ``madvise(MADV_PAGEOUT)`` -- deterministic, no memory-pressure wait
    (verified 2026-09-20 on this box's 6.18 kernel: a 64 MB region reports
    exactly 65536 kB VmSwap right after the call). If the kernel refuses the
    advice (older kernel, no MADV_PAGEOUT), the child prints a sentinel and
    exits so the test can skip instead of failing on an unrelated box.
    """
    return (
        "import ctypes, sys, time\n"
        "libc = ctypes.CDLL('libc.so.6', use_errno=True)\n"
        "libc.mmap.restype = ctypes.c_void_p\n"
        "libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, "
        "ctypes.c_int, ctypes.c_int, ctypes.c_long]\n"
        f"SIZE = {alloc_mb} * 1024 * 1024\n"
        "PAGE = 4096\n"
        "PROT_READ_WRITE = 0x3\n"
        "MAP_PRIVATE_ANONYMOUS = 0x22\n"
        "MADV_PAGEOUT = 21\n"
        "addr = libc.mmap(None, SIZE, PROT_READ_WRITE, MAP_PRIVATE_ANONYMOUS, -1, 0)\n"
        "if addr in (0, None) or addr == (2 ** 64 - 1):\n"
        "    print('MADVISE_PAGEOUT_UNSUPPORTED'); sys.exit(0)\n"
        "buf = (ctypes.c_char * SIZE).from_address(addr)\n"
        "for off in range(0, SIZE, PAGE):\n"
        "    buf[off] = b'x'\n"
        "ret = libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(SIZE), MADV_PAGEOUT)\n"
        "if ret != 0:\n"
        "    print('MADVISE_PAGEOUT_UNSUPPORTED'); sys.exit(0)\n"
        f"time.sleep({hold_s})\n"
    )


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


def test_swap_breach_kills_the_job_and_names_swap():
    # 40 MB of forced-out swap must breach a 20 MB (0.02G) swap cap, while
    # staying far under the 4G RSS cap (madvise(MADV_PAGEOUT) drops RSS to a
    # few MB), so the kill is attributable to swap alone.
    grow = _madvise_pageout_child(alloc_mb=40, hold_s=5.0)
    code, out, took = _run("--max-rss-gb", "4", "--min-free-gb", "0",
                           "--max-swap-gb", "0.02", "--poll-s", "0.1",
                           "--kill-grace-s", "2", code=grow, timeout=30)
    if any(ln.strip() == "MADVISE_PAGEOUT_UNSUPPORTED" for ln in out.splitlines()):
        import pytest
        pytest.skip("kernel does not support MADV_PAGEOUT")
    assert code == 137
    assert "SWAP BREACH" in out
    assert "CAP BREACH" not in out
    assert "BOX FLOOR BREACH" not in out
    assert took < 20


def test_swap_under_threshold_is_not_killed():
    # A small amount of REAL swap, well under the cap, must not breach --
    # unlike the old version of this test, which allocated no swap at all
    # and would have passed even if swap detection were completely broken.
    grow = _madvise_pageout_child(alloc_mb=10, hold_s=1.5)
    code, out, _ = _run("--max-rss-gb", "4", "--min-free-gb", "0",
                        "--max-swap-gb", "0.5", "--poll-s", "0.1",
                        code=grow, timeout=30)
    if any(ln.strip() == "MADVISE_PAGEOUT_UNSUPPORTED" for ln in out.splitlines()):
        import pytest
        pytest.skip("kernel does not support MADV_PAGEOUT")
    assert code == 0, out
    assert "SWAP BREACH" not in out


def test_heartbeat_line_reports_swap():
    code, out, _ = _run("--min-free-gb", "0", "--poll-s", "0.1",
                        code="import time; time.sleep(1.0)")
    assert code == 0, out
    assert "swap" in out
    import re
    assert re.search(r"swap\s+\d+\.\d+G", out), out


def test_negative_max_swap_gb_disables_the_check():
    grow = _madvise_pageout_child(alloc_mb=40, hold_s=1.0)
    code, out, _ = _run("--max-rss-gb", "4", "--min-free-gb", "0",
                        "--max-swap-gb", "-1", "--poll-s", "0.1",
                        code=grow, timeout=30)
    if any(ln.strip() == "MADVISE_PAGEOUT_UNSUPPORTED" for ln in out.splitlines()):
        import pytest
        pytest.skip("kernel does not support MADV_PAGEOUT")
    assert code == 0, out
    assert "SWAP BREACH" not in out


def test_max_rss_gb_above_memtotal_warns():
    code, out, _ = _run("--max-rss-gb", "999999", "--min-free-gb", "0",
                        code="pass")
    assert code == 0, out
    assert "exceeds this box's MemTotal" in out


def test_max_rss_gb_under_memtotal_does_not_warn():
    code, out, _ = _run("--max-rss-gb", "2", "--min-free-gb", "0",
                        code="pass")
    assert code == 0, out
    assert "exceeds this box's MemTotal" not in out
