"""Shared fixtures.

Tests must not touch the real data store, the real raw cache, or the network.
Anything that writes goes to a ``tmp_path`` root; anything that would fetch uses
a fake adapter.
"""
from __future__ import annotations

import contextlib
import faulthandler
import fcntl
import hashlib
import importlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


#: Tests that need a resource GitHub Actions does not have. CI
#: (.github/workflows/tests.yml) deselects every one of these; the local
#: complement run in tests/README.md ("CI tests") covers the rest. Every mark
#: carries a one-line reason comment where it is applied.
#: tests/test_tests_ci.py checks that the workflow's -m expression names
#: exactly these and that tests/README.md documents each one.
LOCAL_ONLY_MARKERS = {
    "needs_data": "reads the real data/ root (gitignored, absent in CI and worktrees)",
    "needs_corpus": "reads fixtures/tier0 or another untracked fixture tree",
    "heavy_host": "launches real multi-GB workers; run alone on a quiet box",
    "browser": "drives a real Playwright browser or needs node/npm (ui/ build)",
}


def pytest_configure(config):
    for name, why in LOCAL_ONLY_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: local only, {why}")
    # Registered here (not just by the xdist plugin) so the mark is silent
    # even when pytest-xdist is absent -- e.g. `pytest -p no:xdist`.
    config.addinivalue_line(
        "markers",
        "xdist_group(name): pin this test (or, applied at module level via "
        "`pytestmark`, every test in the file) to a single pytest-xdist "
        "worker. Recommended invocation: `-n auto --dist loadgroup`, which "
        "keeps every test not in a named group free to run on any worker "
        "and routes each named group to one worker. See the grouping rule "
        "below for which files use it and why.",
    )
    config.addinivalue_line(
        "markers",
        "test_timeout(seconds): per-phase time budget for this test's setup, "
        "call and teardown, overriding --test-timeout. See the per-test "
        "timeout section below.",
    )


# -- per-test timeout ---------------------------------------------------------
#
# pytest-timeout is not installed on this host (checked 2026-09-19), so this
# is a small SIGALRM equivalent. Each of a test's setup, call and teardown
# phases gets its own budget: `--test-timeout SECONDS`, else env
# PYTEST_TEST_TIMEOUT, else DEFAULT_TEST_TIMEOUT; 0 disables it. A test can
# carry `@pytest.mark.test_timeout(N)`. When the budget runs out the test
# FAILS with its node id and the stack it was stuck in, and the run carries on
# with the next test. Before this, one sleeping test kept its xdist worker
# asleep until the outer timeout killed the run, and no summary or test name
# came out.
#
# 600 s is well above any test that finishes: the slowest known are the
# real-parquet reads (~40 s each) and legacy test_features.py (~33 s serially),
# and contention on this box costs up to 3.4x. The one longer legitimate step,
# the `ui_dist_dir` npm install and build, gets `_UI_SETUP_BUDGET` for its
# setup. Disabled automatically when pytest-timeout is installed, so the two
# never race.
#
# The alarm needs the main thread, which is where pytest and xdist 3.x
# workers run tests. A SIGALRM cannot break into C code that never returns to
# the interpreter; everything that waits in this suite (sleep, subprocess
# wait, flock, sqlite busy wait) does return.

DEFAULT_TEST_TIMEOUT = 600.0


def pytest_addoption(parser):
    parser.addoption(
        "--test-timeout", type=float, default=None,
        help="per-phase budget in seconds for each test's setup/call/teardown "
             f"(default: env PYTEST_TEST_TIMEOUT or {DEFAULT_TEST_TIMEOUT:g}; 0 disables)")


def _base_timeout(config) -> float:
    if config.pluginmanager.hasplugin("timeout"):  # real pytest-timeout wins
        return 0.0
    value = config.getoption("--test-timeout")
    if value is None:
        value = float(os.environ.get("PYTEST_TEST_TIMEOUT", DEFAULT_TEST_TIMEOUT))
    return value


def _phase_timeout(item, when: str) -> float:
    base = _base_timeout(item.config)
    if base <= 0:
        return 0.0
    marker = item.get_closest_marker("test_timeout")
    if marker is not None and marker.args:
        base = float(marker.args[0])
    if when == "setup" and "ui_dist_dir" in getattr(item, "fixturenames", ()):
        base = max(base, _UI_SETUP_BUDGET)
    return base


@contextlib.contextmanager
def _alarm(item, when: str):
    seconds = _phase_timeout(item, when)
    if (seconds <= 0 or not hasattr(signal, "setitimer")
            or threading.current_thread() is not threading.main_thread()):
        yield
        return

    def expired(signum, frame):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        pytest.fail(f"TIMEOUT: {item.nodeid} exceeded its {when} budget of {seconds:g}s "
                    f"(--test-timeout / PYTEST_TEST_TIMEOUT / @pytest.mark.test_timeout). "
                    f"The traceback shows where it was waiting; all threads are dumped "
                    f"in the captured stderr.")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    with _alarm(item, "setup"):
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    with _alarm(item, "call"):
        return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    with _alarm(item, "teardown"):
        return (yield)


# -- xdist grouping rule ------------------------------------------------------
#
# A file gets `pytestmark = pytest.mark.xdist_group("serial")` (or, for a
# single test, the same mark on just that function) when it touches a REAL,
# process-wide or host-wide resource that a concurrent sibling test could
# collide with -- not because it is merely slow. Two xdist workers are two
# independent OS processes; anything scoped to `tmp_path` and this worker's
# own subprocesses is already safe run-anywhere.  Grouped so far, and why:
#
# - tests/test_v2_ops_executor_faults.py (whole file): real subprocess
#   spawn/kill/signal, a real `/sys/fs/cgroup` probe, and real watchdog
#   `observe`/`signal_owned` calls against this HOST's process table --
#   "Small real-process fault controls for the watchdog and admission
#   gates" per its own docstring.
# - tests/test_v2_ops_recovery_ownership.py (whole file): "real short-lived
#   child processes" (its own docstring) with process-group/session
#   signaling and a fixed-deadline poll for a real process to die -- a
#   noisy neighbor process on the same host can push that poll past its
#   deadline.
# - tests/test_v2_ops_recovery_reboot.py (whole file): shares the same real
#   `read_boot_id()`/process-identity machinery as recovery_ownership.py;
#   small enough to group alongside it rather than split hairs per test.
# - tests/test_v2_ops_runtime.py::test_o06_actual_child_affinity_threads_and_outputs
#   only (not the rest of the file): asserts the real CPU affinity and
#   thread count of a real child process it just launched -- exactly the
#   "bounded jobs pin from core 0" collision a concurrent sibling could
#   step on. The file's other tests never touch a real subprocess.
# - tests/test_v2_ops_serving_browser.py (whole file): drives a real
#   Playwright browser.
# - tests/test_v2_ops_coordinator_lease.py (whole file): lost-lease tests settle through
#   the real `prove_ownership_gone` /proc scan; in parallel it quarantined the attempt.
#
# NOT grouped, and why: the ~30 other tests that launch a real worker
# subprocess through `Service`/`tests.ops_support.TEST_POLICY` (in
# test_v2_data_import.py, test_v2_data_rebuild_rollback.py,
# test_v2_ops_store_barrier.py, test_v2_ops_supervised_legacy.py,
# test_v2_ops_nightly_completion.py, and the rest of test_v2_ops_runtime.py
# and test_v2_ops_executor_faults.py already covered above) each use their
# own `tmp_path` catalog and their own short-lived child process with no
# fixed path, port, or process-table assertion against another test's
# process -- static reading found no shared resource, only a possible CPU
# time-slice collision on very short (1-3s) subprocesses, which is a
# performance cost, not a correctness one. Empirically confirmed once
# pytest-xdist was installed: `-n auto --dist loadgroup` over the whole
# tests/test_v2_*.py + tests/test_checks_phase2_gate.py suite passed
# (567 passed, 1 skipped) three consecutive times.
#
# - tests/test_features.py (legacy suite, whole file, grouped 2026-09-13):
#   NOT in the v2 suite above, found during a legacy spot-check. 2+
#   concurrent xdist workers each independently load the real feature panel
#   (`engine.data.features.panel`) on this RAM-constrained shared host,
#   which reliably crashed a worker (`[gwN] node down: Not properly
#   terminated`) and then hung xdist's crashed-worker replacement --
#   `python3 -m pytest -q -n auto --dist loadgroup tests/test_calendar.py
#   tests/test_features.py tests/test_dashboard.py` never returned in 22+
#   minutes (vs ~100s serial) before being killed. Bisected file-by-file:
#   test_calendar.py and test_dashboard.py are each independently
#   parallel-safe; only test_features.py reproduces it, and only at 2+
#   workers (`-n 1` and plain serial both pass in ~33s). Real memory
#   pressure from real data, not a small hidden-shared-state bug, so
#   grouped rather than fixed.


# -- ui/ build fixtures -------------------------------------------------------
#
# tests/test_v2_dashboard_browser.py and tests/test_v2_dashboard_integration.py
# both drive a real Playwright browser against a real `npm --prefix ui run
# build` output. A fresh worktree has no `ui/node_modules` at all (it is
# `.gitignore`d, per ui/README.md's "Allowlist" section), so the bare `npm
# run build` those tests used to call directly fails with `tsc: not found`
# before it ever gets to Vite. `ui_dist_dir` below installs (`npm ci --prefix
# ui`, honoring the committed lockfile byte for byte) whenever `ui/
# node_modules` is missing or its recorded lockfile hash no longer matches
# `ui/package-lock.json`, then builds -- once per pytest session, and safe
# against a second agent building the SAME worktree concurrently via a real
# `flock` on `ui/.npm-ci.lock` (xdist workers are separate OS processes, and
# so is a second agent's own pytest invocation).

UI_ROOT = REPO_ROOT / "ui"
_UI_NPM_CI_LOCK = UI_ROOT / ".npm-ci.lock"
_UI_NPM_CI_MARKER = UI_ROOT / "node_modules" / ".package-lock-hash"
_UI_NPM_CI_TIMEOUT = 600
_UI_NPM_BUILD_TIMEOUT = 180
_UI_NPM_CI_LOCK_WAIT = _UI_NPM_CI_TIMEOUT + 60
#: Worst case for the setup of the first test that uses `ui_dist_dir`: wait
#: out another holder's `npm ci`, run our own, then build.
_UI_SETUP_BUDGET = _UI_NPM_CI_LOCK_WAIT + _UI_NPM_CI_TIMEOUT + _UI_NPM_BUILD_TIMEOUT + 60


def _ui_node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


def _ui_lockfile_hash() -> str:
    return hashlib.sha256((UI_ROOT / "package-lock.json").read_bytes()).hexdigest()


def _ui_node_modules_stale() -> bool:
    if not (UI_ROOT / "node_modules").is_dir():
        return True
    if not _UI_NPM_CI_MARKER.is_file():
        return True
    try:
        return _UI_NPM_CI_MARKER.read_text().strip() != _ui_lockfile_hash()
    except OSError:
        return True


def _ui_ensure_node_modules() -> None:
    """Installs `ui/node_modules` via `npm ci --prefix ui` iff missing or
    stale against `ui/package-lock.json` -- under a real `flock` so
    concurrent callers (xdist workers, or a second agent on this host) wait
    for the install rather than racing it. The wait is bounded by
    `_UI_NPM_CI_LOCK_WAIT` (one full `npm ci` timeout plus a minute): past
    that the holder is stuck, not busy, and the test fails naming the lock
    instead of sleeping on it. Never silently skips an `npm ci` failure: a
    nonzero exit fails the test loudly with the command's own tail output.
    Caller is responsible for skipping first when node/npm itself is not
    installed at all (`_ui_node_available`)."""
    UI_ROOT.mkdir(parents=True, exist_ok=True)
    with open(_UI_NPM_CI_LOCK, "w") as lock_file:
        deadline = time.monotonic() + _UI_NPM_CI_LOCK_WAIT
        while True:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    pytest.fail(
                        f"RESOURCE WAIT: {_UI_NPM_CI_LOCK} stayed locked for "
                        f"{_UI_NPM_CI_LOCK_WAIT}s. Another process (an xdist worker or a "
                        f"second pytest session on this worktree) is holding it through "
                        f"`npm ci`; find it with `fuser {_UI_NPM_CI_LOCK}`.", pytrace=False)
                time.sleep(0.5)
        try:
            if not _ui_node_modules_stale():
                return
            result = subprocess.run(
                ["npm", "ci", "--prefix", str(UI_ROOT)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=_UI_NPM_CI_TIMEOUT)
            if result.returncode != 0:
                pytest.fail(
                    f"npm ci --prefix ui failed (exit {result.returncode}):\n"
                    f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
                    f"--- stderr (tail) ---\n{result.stderr[-4000:]}")
            _UI_NPM_CI_MARKER.write_text(_ui_lockfile_hash())
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


@pytest.fixture(scope="session")
def ui_dist_dir():
    """Builds `ui/dist` once per pytest session (`npm --prefix ui run
    build`), installing/repairing `ui/node_modules` first when needed. Skips
    with a clear reason only when node/npm is not available at all; an `npm
    ci` or `npm run build` failure fails loudly rather than skipping."""
    if not _ui_node_available():
        pytest.skip("node/npm not available in this environment")
    _ui_ensure_node_modules()
    result = subprocess.run(
        ["npm", "--prefix", str(UI_ROOT), "run", "build"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=_UI_NPM_BUILD_TIMEOUT)
    if result.returncode != 0:
        pytest.fail(f"npm --prefix ui run build failed:\n{result.stdout}\n{result.stderr}")
    dist = UI_ROOT / "dist"
    assert dist.is_dir(), "build did not produce ui/dist"
    return dist


@pytest.fixture(scope="module")
def playwright_instance():
    """Module-scoped, not session-scoped: `tests/test_v2_dashboard_preview.py`
    (also `xdist_group("serial")`, so it shares a worker with every module
    using this fixture) opens its own independent `sync_playwright()` context
    directly rather than through this fixture. A session-scoped instance here
    would still be open (and its event loop still current) when that other
    module's own `with sync_playwright() as p:` ran in the same process,
    which playwright refuses ("Sync API inside the asyncio loop") -- module
    scope tears this one down at the end of each module, before the next
    module in the worker's queue gets a turn."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="module")
def browser(playwright_instance):
    b = playwright_instance.chromium.launch(headless=True)
    try:
        yield b
    finally:
        b.close()


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Point ``engine.paths`` at a throwaway tree for the duration of a test."""
    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(tmp_path))
    from engine import paths

    importlib.reload(paths)
    yield tmp_path
    monkeypatch.delenv("INVESTING_PLAN_ROOT", raising=False)
    importlib.reload(paths)


@pytest.fixture
def chain_rows():
    """A small, hand-checkable two-expiry chain for one ticker.

    Spot is 100. Strikes bracket it so ATM selection has a unique answer, and
    every bid/ask is a round number so leg arithmetic can be verified by hand.
    """
    obs = pd.Timestamp("2024-05-01")
    rows = []
    for expiry, dte in ((pd.Timestamp("2024-05-03"), 2), (pd.Timestamp("2024-05-24"), 23)):
        for strike in (95.0, 100.0, 105.0):
            for right, bid, ask in (("C", 2.0, 2.4), ("P", 1.0, 1.4)):
                scale = 1.0 if dte < 10 else 2.0
                rows.append(
                    {
                        "ticker": "TEST",
                        "obs_date": obs,
                        "expiry": expiry,
                        "dte": dte,
                        "strike": strike,
                        "right": right,
                        "bid": round(bid * scale, 4),
                        "ask": round(ask * scale, 4),
                        "mid": round((bid + ask) / 2 * scale, 4),
                        "iv": 0.4,
                        "delta": 0.5 if right == "C" else -0.5,
                        "spot": 100.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def chain_snapshot(chain_rows):
    from engine.structures import ChainSnapshot

    return ChainSnapshot(
        ticker="TEST",
        obs_date=pd.Timestamp("2024-05-01"),
        event_date=pd.Timestamp("2024-05-02"),
        rows=chain_rows,
        spot=100.0,
    )
