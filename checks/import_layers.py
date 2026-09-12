#!/usr/bin/env python3
"""Enforce the §4.2 import direction over ``engine/v2/`` — tier 0.

Three rules, each of which alone blocks a commit:

1. **Inside v2, imports point down only.** A package may import a package on a
   strictly lower layer of :mod:`checks.layer_map`, never its own and never a
   higher one. ``engine/v2/dashboard`` may import layer 7 only, and
   ``engine/v2/diagnosis`` is imported by nothing.
2. **v2 reaches legacy only through declared adapters.** Every
   ``engine/v2/** -> engine/*`` dependency is an entry in
   ``checks/legacy_adapters.json``, confined to one adapter module per package.
3. **Legacy never imports v2.** The legacy tree runs the board unchanged and
   must not acquire a dependency on code that is still being proved.

Parsed with stdlib :mod:`ast` over **staged blobs**, following the
``read_staged_blob`` pattern in :mod:`checks.repo_hygiene`. The modules are
never imported to inspect them: importing ``engine.score`` loads a panel, which
is a two-minute pre-commit hook nobody keeps.

Usage::

    python3 checks/import_layers.py            # staged changes (pre-commit)
    python3 checks/import_layers.py --all      # every tracked file
    python3 checks/import_layers.py --paths a.py

Exit code 0 = clean, 1 = blocked.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checks.layer_map import (  # noqa: E402
    CONTAINERS,
    LEGACY_ROOT,
    V2_ROOT,
    PACKAGES,
    Package,
    package_of,
)
from checks.repo_hygiene import (  # noqa: E402
    read_staged_blob,
    read_worktree_blob,
    staged_paths,
    tracked_paths,
)

__all__ = [
    "Edge",
    "Violation",
    "Report",
    "build_graph",
    "check_graph",
    "load_adapters",
    "main",
]

ADAPTERS_PATH = Path(__file__).resolve().parent / "legacy_adapters.json"

#: Third-party and stdlib roots are irrelevant here; only these two matter.
_INTERESTING = (V2_ROOT, LEGACY_ROOT)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """One import, as ``module -> module``."""

    importer: str
    imported: str
    path: str
    lineno: int

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.path}:{self.lineno}: {self.importer} -> {self.imported}"


@dataclass
class Violation:
    rule: str
    edge: Edge
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"  [{self.rule}] {self.edge}\n      {self.detail}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    files: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, rule: str, edge: Edge, detail: str) -> None:
        self.violations.append(Violation(rule, edge, detail))


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def module_name(rel: str) -> str | None:
    """Dotted module for a repo-relative ``.py`` path, or None if not one."""
    if not rel.endswith(".py"):
        return None
    parts = rel[: -len(".py")].replace("\\", "/").split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p)


def package_path(rel: str, importer: str) -> str:
    """The package a file lives in — what a level-1 relative import means.

    ``engine/v2/scoring/__init__.py`` IS ``engine.v2.scoring``, so ``from .
    import x`` there resolves against itself. ``engine/v2/scoring/kernel.py`` is
    a module inside it, so the same statement resolves one level up. Getting
    this backwards makes every relative import resolve into a package that does
    not exist, which reads as "no violation" — a check that passes by being
    wrong.
    """
    if rel.replace("\\", "/").endswith("/__init__.py"):
        return importer
    return importer.rpartition(".")[0]


def _absolute(node: ast.ImportFrom, pkg: str) -> str:
    """Resolve a possibly-relative ``from ... import`` to a dotted module."""
    if not node.level:
        return node.module or ""
    base = pkg.split(".") if pkg else []
    trim = node.level - 1
    root = base[: len(base) - trim] if trim else base
    return ".".join([*root, node.module]) if node.module else ".".join(root)


def imports_in(blob: bytes, rel: str, importer: str) -> list[Edge]:
    """Every ``engine.*`` import in one file, as edges. Syntax errors are skipped."""
    pkg = package_path(rel, importer)
    try:
        tree = ast.parse(blob.decode("utf-8", errors="replace"), filename=rel)
    except SyntaxError:
        return []
    out: list[Edge] = []
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            resolved = _absolute(node, pkg)
            if resolved:
                # `from engine.v2 import contracts` is an edge to the
                # subpackage, not to the container. Record both forms and let
                # the container rule below reject a genuine container import.
                targets = [f"{resolved}.{a.name}" for a in node.names] or [resolved]
                targets.append(resolved)
        for target in targets:
            if any(target == r or target.startswith(r + ".") for r in _INTERESTING):
                out.append(Edge(importer, target, rel, node.lineno))
    return out


def build_graph(files: dict[str, bytes]) -> Report:
    """Parse a ``{repo_relative_path: content}`` mapping into an edge list."""
    report = Report()
    for rel in sorted(files):
        importer = module_name(rel)
        if importer is None or not importer.startswith(("engine.", "engine")):
            continue
        report.files += 1
        report.edges.extend(imports_in(files[rel], rel, importer))
    return report


# --------------------------------------------------------------------------
# the adapter ledger
# --------------------------------------------------------------------------


def load_adapters(path: Path = ADAPTERS_PATH) -> dict:
    """Read ``legacy_adapters.json``, or the empty ledger if it is absent."""
    if not path.exists():
        return {"count": 0, "adapters": []}
    return json.loads(path.read_text())


def _adapter_index(ledger: dict) -> dict[str, set[str]]:
    """``{adapter module: {declared legacy symbols}}``."""
    out: dict[str, set[str]] = {}
    for entry in ledger.get("adapters", []):
        out.setdefault(entry["module"], set()).add(entry["legacy_symbol"])
    return out


def check_ledger(ledger: dict) -> list[str]:
    """Structural problems with the ledger itself, independent of any import."""
    problems: list[str] = []
    adapters = ledger.get("adapters", [])
    if ledger.get("count") != len(adapters):
        problems.append(
            f"count is {ledger.get('count')} but {len(adapters)} adapters are listed"
        )
    required = {"package", "module", "legacy_symbol", "reason", "declared_on"}
    for entry in adapters:
        missing = required - set(entry)
        if missing:
            problems.append(f"adapter {entry!r} is missing {sorted(missing)}")
    per_package: dict[str, set[str]] = {}
    for entry in adapters:
        per_package.setdefault(entry.get("package", "?"), set()).add(
            entry.get("module", "?")
        )
    for pkg, modules in sorted(per_package.items()):
        if len(modules) > 1:
            problems.append(
                f"{pkg} declares {len(modules)} adapter modules "
                f"({sorted(modules)}); §4.2 allows one per package"
            )
    return problems


# --------------------------------------------------------------------------
# the three rules
# --------------------------------------------------------------------------


def _is_v2(module: str) -> bool:
    return module == V2_ROOT or module.startswith(V2_ROOT + ".")


def _allows(importer_pkg: Package, imported_pkg: Package) -> bool:
    if importer_pkg.only_imports is not None:
        return imported_pkg.layer in importer_pkg.only_imports
    return imported_pkg.layer < importer_pkg.layer


def _check_v2_to_v2(edge: Edge, report: Report) -> None:
    if edge.imported in CONTAINERS:
        report.add(
            "container-import", edge,
            f"{edge.imported} is a namespace container with no layer; import "
            "the subpackage that owns the name instead",
        )
        return
    importer_pkg = package_of(edge.importer)
    imported_pkg = package_of(edge.imported)
    if importer_pkg is None or imported_pkg is None:
        report.add(
            "unmapped-package", edge,
            "one end is not in checks/layer_map.py — add it to the map before "
            "importing it",
        )
        return
    if imported_pkg is importer_pkg:
        return  # intra-package imports are free
    if imported_pkg.sink:
        report.add(
            "diagnosis-imported", edge,
            f"{imported_pkg.dotted} is imported by nothing (§4.1) — a "
            "comparator must never become a dependency of what it compares",
        )
        return
    if not _allows(importer_pkg, imported_pkg):
        report.add(
            "upward-import", edge,
            f"layer {importer_pkg.label} may not import layer "
            f"{imported_pkg.label}; imports point down only, and a peer is "
            "not down",
        )


def _check_v2_to_legacy(edge: Edge, report: Report, index: dict[str, set[str]]) -> None:
    declared = index.get(edge.importer, set())
    if any(edge.imported == sym or edge.imported.startswith(sym + ".")
           for sym in declared):
        return
    report.add(
        "undeclared-legacy-import", edge,
        f"{edge.imported} is not declared for {edge.importer} in "
        "checks/legacy_adapters.json — every v2 -> legacy dependency is a "
        "named, dated, reasoned entry confined to one adapter module",
    )


def check_graph(report: Report, ledger: dict) -> Report:
    """Apply the three rules to an already-built graph. Reports every finding."""
    for problem in check_ledger(ledger):
        report.add("adapter-ledger", Edge("", "", str(ADAPTERS_PATH.name), 0), problem)
    index = _adapter_index(ledger)
    for edge in report.edges:
        importer_v2 = _is_v2(edge.importer)
        imported_v2 = _is_v2(edge.imported)
        if importer_v2 and imported_v2:
            _check_v2_to_v2(edge, report)
        elif importer_v2:
            _check_v2_to_legacy(edge, report, index)
        elif imported_v2:
            report.add(
                "legacy-imports-v2", edge,
                "legacy engine/ must not depend on code that is still being "
                "proved (§4.2 rule 3); it runs the board unchanged until it is "
                "deleted whole at phase 8",
            )
    return report


def check_files(files: dict[str, bytes], ledger: dict | None = None) -> Report:
    """The pure core the CLI and the tests both drive."""
    return check_graph(build_graph(files), ledger or load_adapters())


# --------------------------------------------------------------------------
# skeleton completeness
# --------------------------------------------------------------------------


def missing_skeleton(root: Path) -> list[str]:
    """Declared packages with no ``__init__.py`` on disk.

    A layer map naming a package that does not exist is a map nobody has run,
    and an import of it would be an ``unmapped-package`` failure at the far end.
    """
    out: list[str] = []
    for dotted in [*CONTAINERS, *(p.dotted for p in PACKAGES)]:
        init = root / dotted.replace(".", "/") / "__init__.py"
        if not init.exists():
            out.append(str(init.relative_to(root)))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _collect(root: Path, args) -> dict[str, bytes]:
    if args.paths:
        rels = [str(Path(p).resolve().relative_to(root)) for p in args.paths]
        reader = read_worktree_blob
    elif args.all:
        rels = tracked_paths(root)
        reader = read_worktree_blob
    else:
        rels = staged_paths(root)
        reader = read_staged_blob
    return {rel: reader(root, rel) for rel in rels if rel.endswith(".py")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="check every tracked file")
    mode.add_argument("--paths", nargs="+", help="check these paths explicitly")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    report = check_files(_collect(root, args))
    gaps = missing_skeleton(root)
    for gap in gaps:
        report.add("missing-package", Edge("", "", gap, 0),
                   "declared in checks/layer_map.py but absent from the tree")

    ledger = load_adapters()
    if not args.quiet:
        scope = "--paths" if args.paths else ("all tracked" if args.all else "staged")
        print(
            f"import layers: {report.files} module(s) [{scope}], "
            f"{len(report.edges)} engine import(s), "
            f"adapter ledger at {ledger.get('count')}"
        )
    if report.ok:
        if not args.quiet:
            print("LAYERS OK")
        return 0
    print(f"\nLAYERS FAILED — {len(report.violations)} violation(s):", file=sys.stderr)
    for v in report.violations:
        print(str(v), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
