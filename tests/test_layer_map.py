"""Regression test: every dotted package in PACKAGES appears exactly once.

`_BY_DOTTED = {pkg.dotted: pkg for pkg in PACKAGES}` means a duplicate
`dotted` entry is a silent dict-collision bug: the later entry in the tuple
wins and shadows every field the earlier entry set, including
`orchestrator`. One such duplicate (`engine.v2.research`) shadowed its
`orchestrator=True` exemption, which broke `checks/code_budgets.py`'s
fan-out budget for `engine/v2/research/_build_run.py`. See issue #251.
"""
from __future__ import annotations
# land: always-run

from collections import Counter

from checks.layer_map import PACKAGES, package_of


def test_no_duplicate_dotted_packages():
    counts = Counter(pkg.dotted for pkg in PACKAGES)
    duplicates = {dotted: n for dotted, n in counts.items() if n > 1}
    assert duplicates == {}, f"duplicate Package(dotted=...) entries: {duplicates}"


def test_engine_v2_research_is_orchestrator():
    pkg = package_of("engine.v2.research")
    assert pkg is not None
    assert pkg.orchestrator is True
