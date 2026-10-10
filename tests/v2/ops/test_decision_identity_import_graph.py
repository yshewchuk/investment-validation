"""Import-graph guard between decision_replay and decision_validation.

Parses both modules with ``ast.parse`` (no import/execution of either
module) and asserts they do not mutually import: validation may import
replay one-way, but replay must never import validation.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPLAY = "engine.v2.ops.decision_replay"
VALIDATION = "engine.v2.ops.decision_validation"
_MODULE_FILES = {
    REPLAY: Path(__file__).resolve().parents[3] / "engine" / "v2" / "ops"
    / "decision_replay.py",
    VALIDATION: Path(__file__).resolve().parents[3] / "engine" / "v2" / "ops"
    / "decision_validation.py",
}


def _absolute_names(node, module_name):
    """Yield fully-qualified module names referenced by an import node,
    resolving relative imports against the package of ``module_name``."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name
    elif isinstance(node, ast.ImportFrom):
        package = module_name.split(".")[:-1]
        base_parts = package[: len(package) - (node.level - 1)] if node.level else []
        base = ".".join(base_parts)
        if node.module:
            resolved = f"{base}.{node.module}" if base else node.module
            yield resolved
            prefix = f"{resolved}."
        else:
            prefix = f"{base}." if base else ""
        for alias in node.names:
            yield f"{prefix}{alias.name}"
        if not node.module and not node.names:
            yield base


def _collect_edges(module_name):
    """Return the set of the two target modules imported by ``module_name``."""
    tree = ast.parse(_MODULE_FILES[module_name].read_text())
    edges = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for referenced in _absolute_names(node, module_name):
                if referenced in (REPLAY, VALIDATION) and referenced != module_name:
                    edges.add(referenced)
    return edges


def test_relative_sibling_import_resolves_to_validation():
    """A relative sibling import resolves to the fully-qualified module name."""
    node = ast.parse("from . import decision_validation").body[0]
    assert list(_absolute_names(node, REPLAY)) == [VALIDATION]


def test_replay_does_not_import_validation():
    assert VALIDATION not in _collect_edges(REPLAY)


def test_modules_do_not_mutually_import():
    replay_targets = _collect_edges(REPLAY)
    validation_targets = _collect_edges(VALIDATION)
    assert not (VALIDATION in replay_targets and REPLAY in validation_targets), (
        "mutual imports between the two modules: "
        f"replay imports {sorted(replay_targets)}, "
        f"validation imports {sorted(validation_targets)}"
    )


def test_modules_do_not_expose_population_key():
    """Neither module may publicly define, import, or export ``population_key``.

    ``foundation.score_population.population_key`` is foundation's public
    helper; ops modules that need it bind it privately as
    ``_population_key``. AST-only: neither source module is imported.
    """
    for module_name in (REPLAY, VALIDATION):
        tree = ast.parse(_MODULE_FILES[module_name].read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                bound = {alias.name.split(".")[0] for alias in node.names if alias.asname is None}
                bound |= {alias.asname for alias in node.names if alias.asname is not None}
                assert "population_key" not in bound, module_name
            elif isinstance(node, ast.ImportFrom):
                assert not any(
                    (alias.asname or alias.name) == "population_key"
                    and alias.asname != "_population_key"
                    for alias in node.names
                ), module_name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assert node.name != "population_key", module_name
            elif isinstance(node, ast.Assign):
                assert not any(
                    isinstance(target, ast.Name) and target.id == "population_key"
                    for target in node.targets
                ), module_name
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                names = ast.literal_eval(node.value)
                assert "population_key" not in names, module_name
            elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "__all__":
                names = ast.literal_eval(node.value)
                assert "population_key" not in names, module_name
