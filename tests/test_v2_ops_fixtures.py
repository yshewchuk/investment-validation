"""tools/v2_ops_fixtures.py: enqueue_claim works with no tests/ import."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = r"""
import builtins
import sys

class RejectTests:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tests" or fullname.startswith("tests."):
            raise ImportError("tests imports are forbidden")

sys.meta_path.insert(0, RejectTests())
real_import = builtins.__import__

def reject_tests(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tests" or name.startswith("tests."):
        raise ImportError("tests imports are forbidden")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_tests

from datetime import datetime, timezone
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor
from tools.v2_ops_fixtures import enqueue_claim

class Clock:
    def now(self):
        return datetime(2026, 9, 12, tzinfo=timezone.utc)

clock = Clock()
conn = open_catalog(sys.argv[1], clock=clock)
epoch_id = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
assert enqueue_claim(conn, clock, Supervisor(epoch_id, "boot"), key="one") is not None
"""


def test_enqueue_claim_returns_a_claim_with_no_tests_package_import(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(tmp_path / "ops.sqlite")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
