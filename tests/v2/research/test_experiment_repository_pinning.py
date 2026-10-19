"""Experiment CLIs read one fixed snapshot, so every Repository they build pins it (#975): without pinning each partition scan re-decodes the table's whole fragment membership."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_experiment_repository_calls_pin_snapshots():
    offenders: list[str] = []
    seen = 0
    for path in sorted((ROOT / "experiments").rglob("*.py")):
        assert not path.is_symlink()
        assert path.stat().st_size <= 1_000_000
        tree = ast.parse(path.read_bytes(), filename=str(path))
        rel = path.relative_to(ROOT)
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
                offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []
    assert seen >= 10