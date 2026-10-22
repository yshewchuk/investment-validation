"""Expired test budgets remain failures even when diagnostic stderr is broken.

# packages: ops
"""
import subprocess
import sys
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
    item = SimpleNamespace(nodeid="synthetic::validation")
    # Restore before pytest's own call/teardown alarms run, not in teardown.
    with monkeypatch.context() as patch:
        patch.setattr(harness, "_phase_timeout", lambda *args: 0.01)
        patch.setattr(harness.signal, "signal", lambda sig, handler: handlers.append(handler) or previous)
        patch.setattr(harness.signal, "setitimer", lambda timer, value: timers.append(value))
        patch.setattr(harness.faulthandler, "dump_traceback", dump)
        with pytest.raises(pytest.fail.Exception, match="TIMEOUT: synthetic::validation exceeded its call budget"):
            with harness._alarm(item, "call"):
                try:
                    handlers[0](None, None)
                except (ValueError, OSError):
                    pytest.fail("timeout was translated to a validation refusal")
    assert dumps == [{"file": sys.__stderr__, "all_threads": True}]
    assert timers == [0.01, 0]
    assert handlers[-1] is previous


def test_real_signal_under_capsys_reports_timeout_not_fileno_error(tmp_path):
    test = tmp_path / "test_captured_timeout.py"
    test.write_text('''
import time
from types import SimpleNamespace
from tests.conftest import _alarm

def test_captured(capsys):
    item = SimpleNamespace(nodeid='synthetic::captured', fixturenames=[],
        config=SimpleNamespace(pluginmanager=SimpleNamespace(hasplugin=lambda name: False),
                               getoption=lambda name: 0.02),
        get_closest_marker=lambda name: None)
    with _alarm(item, 'call'):
        try:
            time.sleep(0.1)
        except (ValueError, OSError):
            raise AssertionError('MASKED_TIMEOUT_AS_VALIDATION')
''')
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "tests.conftest",
                             "--confcutdir=" + str(tmp_path), "-q", str(test)],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Failed: TIMEOUT: synthetic::captured exceeded its call budget" in result.stdout
    assert "AssertionError: MASKED_TIMEOUT_AS_VALIDATION" not in result.stdout
    assert "Current thread" in result.stdout + result.stderr
