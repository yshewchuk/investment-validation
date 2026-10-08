"""Tier-0 support preserves behavior without pulling in its process runner.

# packages: engine.v2.diagnosis, engine.v2.data, engine.v2.contracts, engine.v2.foundation
"""
from __future__ import annotations

import ast
from pathlib import Path

from checks import rearchitecture_phase2_evidence as evidence
from checks import tier0_corpus as corpus
from checks import tier0_support as support
from tools import mutation_pilot as selector

ROOT = Path(__file__).resolve().parents[3]


def test_compatibility_exports_are_the_same_objects():
    assert corpus.DEFAULT_CORPUS is support.DEFAULT_CORPUS
    assert evidence.DEFAULT_CORPUS is support.DEFAULT_CORPUS
    assert corpus.round_params is support.round_params
    assert corpus.finding_dicts is support.finding_dicts


def test_default_corpus_retains_repository_root_independently_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert support.DEFAULT_CORPUS == ROOT / "fixtures" / "tier0"
    assert support.DEFAULT_CORPUS == corpus.ROOT / "fixtures" / "tier0"


def test_consumers_use_the_lightweight_tier0_boundary(monkeypatch):
    # Check this extraction's direct import contracts, not every transitive
    # repository dependency. The full selector graph is measured separately.
    def refuse_whole_graph(*args, **kwargs):
        raise AssertionError("this bounded check must not rebuild the repository graph")

    monkeypatch.setattr(selector, "build_import_graph", refuse_whole_graph)
    parsed = {}
    consumers = {
        "checks/rearchitecture_phase2_evidence.py": {"DEFAULT_CORPUS"},
        "tests/v2/diagnosis/test_phase0_negative_controls.py": {
            "finding_dicts", "round_params"},
    }
    for path, names in consumers.items():
        tree = ast.parse((ROOT / path).read_text())
        parsed[path] = tree
        support_imports = {alias.name for node in ast.walk(tree)
                           if isinstance(node, ast.ImportFrom)
                           and node.module == "checks.tier0_support"
                           for alias in node.names}
        assert names <= support_imports, path
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""] + [
                    f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            assert not any(module == "checks.tier0_corpus"
                           or module.startswith("checks.tier0_corpus.")
                           for module in modules), path

    leaf = ast.parse((ROOT / "checks/tier0_support.py").read_text())
    parsed["checks/tier0_support.py"] = leaf
    imports = set()
    for node in ast.walk(leaf):
        if isinstance(node, ast.Import):
            imports.update((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.update((node.module, alias.name) for alias in node.names)
    assert imports == {("__future__", "annotations"), ("copy", None),
                       ("pathlib", "Path"), ("engine.v2.diagnosis", "ComparisonReceipt")}

    # Preserve the genuine runner's unresolved-process classification using
    # the selector's real scanner, but parse only that runner rather than
    # constructing and scanning a whole-repository graph for every CI run.
    # Mutation work copies contain these sources but have no Git metadata.
    tracked = {str(path.relative_to(ROOT))
               for source in ("checks", "engine", "tests", "tools")
               for path in (ROOT / source).rglob("*.py")}
    roots = selector._tracked_roots(tracked)
    for path, tree in parsed.items():
        assert not selector._has_unresolved_import_attempt(tree, False), path
        assert not selector._has_unresolved_sys_path_mutation(tree, path), path
        assert not selector._has_unresolved_process_launch(tree), path
        assert not selector._subprocess_targets(tree, tracked, roots)[1], path
    runner = ast.parse((ROOT / "checks/tier0_corpus.py").read_text())
    _, unresolved = selector._subprocess_targets(runner, tracked, roots)
    assert unresolved
