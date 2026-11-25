"""The fair-value experiment modules stay outside the size-search and size-panel closures (EXP-190 F1).

Test 1 always runs. Test 2 compares closure files with CLOSURE_NEUTRAL_BASE=<ref>: an empty diff
means the closure hash is unchanged against that base.
"""
from __future__ import annotations

# packages: engine.v2.ops

import os
import subprocess
from pathlib import Path

import pytest

from engine.v2.ops.fingerprints import source_closure

ROOT = Path(__file__).resolve().parents[3]
CLOSURES = {
    "size-panel": ["experiments/size_panel.py"],
    "size-search": ["experiments/size_search.py", "experiments/magnitude_search_worker.py",
                    "engine/models/registry.json"],
}


@pytest.mark.parametrize("name", list(CLOSURES))
def test_fair_value_modules_are_outside_closure(name):
    found = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").glob("fair_value_*.py"))
    assert found and "experiments/fair_value_inputs.py" in found
    assert not set(found) & set(source_closure(ROOT, CLOSURES[name]))


@pytest.mark.parametrize("name", list(CLOSURES))
def test_closure_unchanged_against_base(name):
    base = os.environ.get("CLOSURE_NEUTRAL_BASE")
    if not base:
        pytest.skip("set CLOSURE_NEUTRAL_BASE=<ref> to compare against a base")
    files = sorted({*source_closure(ROOT, CLOSURES[name]), *CLOSURES[name]})
    changed = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", base, "--", *files],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    assert changed.strip() == "", changed
