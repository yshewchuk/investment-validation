"""The chain-index twin must stay outside the v14 SIZE closure.

Guards the premise of the twin approach: the twin is only safe to add beside the
legacy pricer while nothing in the v14-bound source closure imports it, and
``engine/structures.py`` and ``engine/fills.py`` (which the twin wraps) remain in
that closure, so any change to them is still visible to the fingerprint.
The benchmark under tools/ imports the twin on purpose, and so does
experiments/size_width_pricing.py (the EXP-189 pricing adapter, the chain-index seam);
neither is reachable from the v14/SIZE entries, which the closure tests below still assert.
"""
from __future__ import annotations

# packages: engine.v2.ops

import ast
import os
import subprocess
from pathlib import Path

import pytest

from engine.v2.ops.fingerprints import source_closure

ROOT = Path(__file__).resolve().parents[3]
TWIN = "engine/chain_index.py"
BLOCKED_ENTRIES = (
    ["experiments/size_search.py", "experiments/magnitude_search_worker.py"],
    ["experiments/size_panel.py"],
    ["experiments/v2_magnitude_search.py", "experiments/v2_prediction.py"],
)
V14_ENTRIES = ["experiments/v2_magnitude_search.py", "experiments/v2_prediction.py"]
SCAN_DIRS = ("engine", "experiments", "ops", "tools")
ALLOWED_IMPORTERS = {"engine/chain_index.py", "tools/bench_chain_index.py", "experiments/size_width_pricing.py"}
MAX_SOURCE_BYTES = 8 * 1024 * 1024


@pytest.mark.parametrize("entries", BLOCKED_ENTRIES, ids=["size-search", "size-panel", "v14"])
def test_twin_is_not_in_size_closures(entries):
    assert TWIN not in source_closure(ROOT, entries)


def _imports_twin(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "engine.chain_index" or a.name.startswith("engine.chain_index.")
                   for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "engine.chain_index"
                                    or module.startswith("engine.chain_index.")):
                return True
            if node.level == 0 and module == "engine" and any(a.name == "chain_index" for a in node.names):
                return True
            if node.level > 0 and module == "chain_index":
                return True
            if node.level > 0 and not module and any(a.name == "chain_index" for a in node.names):
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Dynamic imports by string name.
            if node.value in ("engine.chain_index", "engine/chain_index.py"):
                return True
    return False


def test_no_scanned_module_imports_the_twin():
    importers = set()
    for directory in SCAN_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            with path.open("rb") as stream:
                data = stream.read(MAX_SOURCE_BYTES + 1)
            assert len(data) <= MAX_SOURCE_BYTES, path
            text = data.decode("utf-8")
            # Text pre-filter only; the decision is made on the parsed imports.
            if "chain_index" not in text:
                continue
            if _imports_twin(ast.parse(text)):
                importers.add(path.relative_to(ROOT).as_posix())
    assert importers - ALLOWED_IMPORTERS == set(), sorted(importers - ALLOWED_IMPORTERS)


def test_v14_closure_unchanged_against_base():
    base = os.environ.get("CLOSURE_NEUTRAL_BASE")
    if not base:
        pytest.skip("set CLOSURE_NEUTRAL_BASE=<ref> to compare against a base")
    closure = source_closure(ROOT, V14_ENTRIES)
    # Each HEAD closure file must exist at the base and have the same bytes there.
    changed = subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--name-only", base, "--", *closure],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    assert changed.strip() == "", changed


def test_structures_and_fills_are_in_v14_closure():
    closure = source_closure(ROOT, V14_ENTRIES)
    assert "engine/structures.py" in closure
    assert "engine/fills.py" in closure
