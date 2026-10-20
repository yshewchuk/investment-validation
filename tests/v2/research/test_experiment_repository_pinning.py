"""Experiment CLIs read one fixed snapshot, so every Repository they build pins it (#975): without pinning each partition scan re-decodes the table's whole fragment membership."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def unpinned_repository_calls(source: bytes) -> tuple[int, list[int]]:
    tree = ast.parse(source)
    names: set[str] = {"Repository"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "Repository" and alias.asname:
                    names.add(alias.asname)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname and alias.name.endswith("Repository"):
                    names.add(alias.asname)
    grew = True
    while grew:
        grew = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not (
                isinstance(value, ast.Name) and value.id in names
            ) and not (
                isinstance(value, ast.Attribute) and value.attr == "Repository"
            ):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    grew = True
    seen = 0
    unpinned: list[int] = []
    # A maintained linter cannot express this project-specific rule (ruff/flake8 have no 'constructor keyword must be literal True' check without a custom plugin), so a small stdlib ast guard stays.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            is_repository = node.func.id in names
        elif isinstance(node.func, ast.Attribute):
            is_repository = node.func.attr == "Repository"
        else:
            is_repository = False
        if not is_repository:
            continue
        seen += 1
        pinned = any(
            kw.arg == "pin_snapshots"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        )
        if not pinned:
            unpinned.append(node.lineno)
    return seen, unpinned


def test_experiment_repository_calls_pin_snapshots():
    offenders: list[str] = []
    seen = 0
    for path in sorted((ROOT / "experiments").rglob("*.py")):
        assert not path.is_symlink()
        rel = path.relative_to(ROOT)
        with path.open("rb") as handle:
            data = handle.read(1_000_001)
        assert len(data) <= 1_000_000
        count, unpinned = unpinned_repository_calls(data)
        seen += count
        offenders.extend(f"{rel}:{lineno}" for lineno in unpinned)
    assert offenders == []
    assert seen >= 10


def test_guard_flags_unpinned_aliases():
    source = b"""from engine.v2.data.repository import Repository
from engine.v2.data.repository import Repository as Repo
make_repo = Repository
a = Repository(c, s, pin_snapshots=True)
b = Repo(c, s)
d = make_repo(c, s)
import engine.v2.data.repository as data
e = data.Repository(c, s)
typed: object = Repository
f = typed(c, s)
left = chained = Repository
g = chained(c, s)
"""
    assert unpinned_repository_calls(source) == (6, [5, 6, 8, 10, 12])