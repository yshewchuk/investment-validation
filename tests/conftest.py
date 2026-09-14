"""Shared fixtures.

Tests must not touch the real data store, the real raw cache, or the network.
Anything that writes goes to a ``tmp_path`` root; anything that would fetch uses
a fake adapter.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
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
