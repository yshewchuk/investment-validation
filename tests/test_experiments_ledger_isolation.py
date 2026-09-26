"""The experiments ledger is structurally unreachable from tests.

``experiments/lib.py`` used to call ``ledger_ensure()`` at import time, so
merely importing the module created ``experiments/LEDGER.csv`` in the checkout
— a test run could write the multiple-testing record. These tests pin both
halves of the fix: importing creates no file, and the default append path is
the per-test tmp tree the autouse conftest fixture installs, never the repo's
``experiments/`` directory.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from engine import paths

REPO = Path(__file__).resolve().parents[1]
ROW = {"id": "EXP-901", "spec_hash": "deadbeef", "date": "2026-01-01",
       "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "",
       "promoted": "False"}
HEADER = "id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted"


def _repo_ledger_state():
    """``(mtime_ns, size)`` of the checkout's ledger, or None if absent."""
    path = REPO / "experiments" / "LEDGER.csv"
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def test_importing_experiments_lib_creates_no_file(tmp_path, monkeypatch):
    """A fresh execution of the module (the test session's own import is
    cached) must leave the filesystem alone: no ledger, no experiments/."""
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    spec = importlib.util.spec_from_file_location(
        "experiments._import_isolation_probe", REPO / "experiments" / "lib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LEDGER_PATH == tmp_path / "experiments" / "LEDGER.csv"
    assert not (tmp_path / "experiments").exists(), "import created a ledger"


def test_ledger_append_with_no_path_writes_under_tmp_path_only(tmp_path):
    from experiments import lib

    assert lib.LEDGER_PATH == tmp_path / "experiments" / "LEDGER.csv"
    before = _repo_ledger_state()
    lib.ledger_append([ROW])
    ledger = tmp_path / "experiments" / "LEDGER.csv"
    assert ledger.is_file()
    assert ledger.read_text().splitlines()[0] == HEADER
    assert lib.ledger_read().iloc[0]["id"] == "EXP-901"
    assert _repo_ledger_state() == before, \
        "the checkout's experiments/LEDGER.csv must not be created or touched"
