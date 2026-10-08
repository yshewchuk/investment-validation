"""Tier-0 support preserves behavior without pulling in its process runner.

# packages: engine.v2.diagnosis, engine.v2.data, engine.v2.contracts, engine.v2.foundation
"""
from __future__ import annotations

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


def test_pure_consumers_do_not_inherit_corpus_process_taint():
    # Mutation work copies contain these sources but have no Git metadata.
    tracked = sorted(str(path.relative_to(ROOT))
                     for source in ("checks", "engine", "tests", "tools")
                     for path in (ROOT / source).rglob("*.py"))
    graph = selector.build_import_graph(tracked)
    unresolved = selector.unresolved_import_files(tracked)
    runner = "checks/tier0_corpus.py"
    assert runner in unresolved  # The genuine process runner still fails safe.
    for path in (
        "checks/tier0_support.py",
        "checks/rearchitecture_phase2_evidence.py",
        "tests/v2/diagnosis/test_phase0_negative_controls.py",
    ):
        roots = {path} | selector._conftest_ancestors(path, set(tracked))
        closure, tainted = selector._closure_from_roots(
            roots, graph, unresolved, taint_exempt=set())
        assert runner not in closure, path
        assert not tainted, path
