"""The extracted ``legacy_parity_rows`` leaf: import topology and ownership.

Plain pytest functions.  The import-topology helper parses source with
``ast`` (never importing the module under test's internals) so a leaf or a
forbidden edge is caught structurally, including imports nested inside a
function body.
"""
from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_NIGHTLY = "engine.v2.ops.nightly"
_NATIVE_PARITY_REPORT = "engine.v2.ops.native_parity_report"
_ALLOWED_OPS_MODULES = {"engine.v2.ops.decision_validation", "engine.v2.ops.errors"}


def _imported_modules(path: Path) -> set[str]:
    """Every dotted module name imported by ``path``.

    Walks ALL nodes, so an import inside a function body is included.  A
    ``from X import a`` statement resolves to ``X`` and also ``X.a``; a plain
    ``import X.Y`` resolves to ``X.Y``.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            module = node.module or ""
            if module:
                names.add(module)
            for alias in node.names:
                names.add(f"{module}.{alias.name}" if module else alias.name)
    return names


def _imports_module(imports: set[str], module: str) -> bool:
    """Whether ``imports`` contains ``module`` or any dotted child of it."""
    return any(name == module or name.startswith(module + ".") for name in imports)


def test_native_parity_report_and_nightly_do_not_import_each_other():
    # Negative control: this test must fail if the removed import returned --
    # the function-level `from engine.v2.ops.nightly import legacy_parity_rows`
    # in native_parity_report.py.  nightly -> native_parity_report stays a
    # legitimate one-way dependency (its ``native_parity`` stage handler and
    # schema constants), so only the forbidden reverse edge is asserted.
    report_imports = _imported_modules(_ROOT / "engine/v2/ops/native_parity_report.py")
    assert not _imports_module(report_imports, _NIGHTLY)


def test_parity_inputs_is_a_leaf():
    imports = _imported_modules(_ROOT / "engine/v2/ops/native/parity_inputs.py")
    ops_imports = {name for name in imports if name.startswith("engine.v2.ops")}
    # The helper also records the imported symbol (`X.a`); collapse a child
    # onto its parent module before the leaf subset check.
    modules = {name for name in ops_imports
               if not any(other != name and name.startswith(other + ".")
                          for other in ops_imports)}
    assert modules <= _ALLOWED_OPS_MODULES


def test_nightly_no_longer_defines_legacy_parity_rows():
    from engine.v2.ops import nightly
    from engine.v2.ops.native import parity_inputs

    assert not hasattr(nightly, "legacy_parity_rows")
    assert callable(parity_inputs.legacy_parity_rows)
