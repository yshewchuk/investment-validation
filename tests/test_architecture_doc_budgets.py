"""Negative controls for the ARCHITECTURE.md line budget.

Proved by planting what the check exists to catch, same style as
tests/test_code_budgets.py: asserting only that the real tree is green would
pass identically if the check did nothing.
"""
from __future__ import annotations
# land: always-run

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import architecture_doc_budgets as adb  # noqa: E402


def test_the_real_tree_is_within_budget():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "architecture_doc_budgets.py"), "--all"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr


def test_a_new_doc_over_budget_fails():
    big = ("x\n" * (adb.BUDGET + 1)).encode()
    report = adb.check_files({"engine/v2/newpkg/ARCHITECTURE.md": big})
    assert not report.ok
    assert report.violations[0].path == "engine/v2/newpkg/ARCHITECTURE.md"


def test_a_doc_at_exactly_budget_passes():
    ok = ("x\n" * adb.BUDGET).encode()
    assert adb.check_files({"engine/v2/newpkg/ARCHITECTURE.md": ok}).ok


def test_root_architecture_doc_is_in_scope():
    big = ("x\n" * (adb.BUDGET + 1)).encode()
    assert not adb.check_files({"ARCHITECTURE.md": big}).ok


def test_exempt_doc_is_capped_at_its_pinned_size_not_unlimited():
    path = "engine/v2/ops/ARCHITECTURE.md"
    cap = adb.EXEMPT[path]
    over = ("x\n" * (cap + 1)).encode()
    at_cap = ("x\n" * cap).encode()
    assert not adb.check_files({path: over}).ok
    assert adb.check_files({path: at_cap}).ok


def test_non_architecture_markdown_is_not_checked():
    huge = ("x\n" * 5000).encode()
    report = adb.check_files({"engine/v2/ops/README.md": huge})
    assert report.ok
    assert report.docs == 0