"""Expired test budgets remain failures even when diagnostic stderr is broken.

# packages: ops
"""
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest as harness

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("error", [None, OSError("closed stderr"), ValueError("no fileno")])
def test_expired_budget_cannot_be_translated_to_a_domain_refusal(monkeypatch, error):
    handlers, timers, dumps = [], [], []
    previous = object()
    def dump(**kwargs):
        dumps.append(kwargs)
        if error is not None:
            raise error
    item = SimpleNamespace(nodeid="synthetic::validation", fixturenames=[],
        config=SimpleNamespace(pluginmanager=SimpleNamespace(hasplugin=lambda name: False),
                               getoption=lambda name: 0.01),
        get_closest_marker=lambda name: None)
    # Restore before pytest's own call/teardown alarms run, not in teardown.
    with monkeypatch.context() as patch:
        patch.setattr(signal, "signal", lambda sig, handler: handlers.append((sig, handler)) or previous)
        patch.setattr(signal, "setitimer", lambda timer, value: timers.append((timer, value)))
        patch.setattr(harness.faulthandler, "dump_traceback", dump)
        with pytest.raises(pytest.fail.Exception,
                           match=r"TIMEOUT: synthetic::validation exceeded its call budget of 0\.01s "):
            with contextmanager(harness.pytest_runtest_call)(item):
                try:
                    handlers[0][1](None, None)
                except (ValueError, OSError):
                    pytest.fail("timeout was translated to a validation refusal")
    assert dumps == [{"file": sys.__stderr__, "all_threads": True}]
    assert timers == [(signal.ITIMER_REAL, 0.01), (signal.ITIMER_REAL, 0)]
    assert len(handlers) == 2 and handlers[0][0] == signal.SIGALRM
    assert handlers[-1] == (signal.SIGALRM, previous)


def test_alarm_restores_default_handler_when_previous_is_none(monkeypatch):
    handlers = []
    item = SimpleNamespace(nodeid="synthetic::alarm", fixturenames=[],
        config=SimpleNamespace(pluginmanager=SimpleNamespace(hasplugin=lambda name: False),
                               getoption=lambda name: 0.01),
        get_closest_marker=lambda name: None)
    with monkeypatch.context() as patch:
        patch.setattr(signal, "signal", lambda sig, handler: handlers.append((sig, handler)))
        patch.setattr(signal, "setitimer", lambda timer, value: None)
        with contextmanager(harness.pytest_runtest_call)(item):
            pass
    assert len(handlers) == 2 and handlers[0][0] == signal.SIGALRM
    assert callable(handlers[0][1])
    assert handlers[-1] == (signal.SIGALRM, signal.SIG_DFL)


def test_real_signal_under_capsys_reports_timeout_not_fileno_error(tmp_path):
    test = tmp_path / "test_captured_timeout.py"
    test.write_text('''
import time
from contextlib import contextmanager
from types import SimpleNamespace
from tests import conftest as harness

def test_captured(capsys):
    item = SimpleNamespace(nodeid='synthetic::captured', fixturenames=[],
        config=SimpleNamespace(pluginmanager=SimpleNamespace(hasplugin=lambda name: False),
                               getoption=lambda name: 0.02),
        get_closest_marker=lambda name: None)
    with contextmanager(harness.pytest_runtest_call)(item):
        try:
            time.sleep(0.1)
        except (ValueError, OSError):
            raise AssertionError('MASKED_TIMEOUT_AS_VALIDATION')
''')
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "tests.conftest",
                             "--confcutdir=" + str(tmp_path), "-q", str(test)],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Failed: TIMEOUT: synthetic::captured exceeded its call budget of 0.02s " in result.stdout
    assert "AssertionError: MASKED_TIMEOUT_AS_VALIDATION" not in result.stdout
    assert "Current thread" in result.stdout + result.stderr
