#!/usr/bin/env python3
"""Mechanical code budgets over ``engine/v2/**`` — tier 0, zero exemptions.

`system_rearchitecture.md` §4.3, measured from the legacy tree rather than
chosen, which is what makes them known to be achievable here:

| Budget | Value | Blocking |
|---|---|---|
| Cyclomatic complexity per function | 15 | yes |
| Lines per function | 80 | yes |
| Fan-out, non-orchestrator | 8 | yes |
| Lines per module | 600 | **no — a warning** |

Module length is the one budget §4.3 states as "a soft cap ... enforced as a
warning rather than a failure", so it is reported and counted but does not
block. Every other budget is absolute: §4.6 gives v2 zero tolerance, no
exemption file and no grandfathering, which is only a reasonable rule because
there is nothing inherited to forgive. **Legacy ``engine/`` is exempt
wholesale** — it is frozen apart from bug fixes and deleted at phase 8, so
lowering its complexity is effort spent on a tree that will not exist.

Complexity is a *diagnosis* budget, not an aesthetic one. A comparator can only
name the first differing stage if stages are separable in the code; a function
with sixty branches has no stages to name, which is why one red row could carry
five causes.

Stdlib :mod:`ast` over staged blobs, never an import: importing ``engine.score``
loads a panel. There is deliberately no dependency on a linter or on pytest —
"style linting is a convenience that may be unavailable; the budgets are a gate
that may not be".

Usage::

    python3 checks/code_budgets.py            # staged changes (pre-commit)
    python3 checks/code_budgets.py --all      # every tracked v2 file
    python3 checks/code_budgets.py --json     # machine-readable, for the nightly
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checks.layer_map import V2_ROOT, package_of  # noqa: E402
from checks.import_layers import module_name  # noqa: E402
from checks.repo_hygiene import (  # noqa: E402
    read_staged_blob,
    read_worktree_blob,
    staged_paths,
    tracked_paths,
)

__all__ = ["BUDGETS", "Violation", "Report", "check_files", "complexity", "main"]

BUDGETS = {
    "complexity": 15,
    "function_lines": 80,
    "module_lines": 600,
    "fan_out": 8,
}

#: The one non-blocking budget, per §4.3.
WARNING_ONLY = frozenset({"module_lines"})

_V2_PREFIX = V2_ROOT.replace(".", "/") + "/"


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    metric: str
    path: str
    name: str
    lineno: int
    value: int
    budget: int

    @property
    def blocking(self) -> bool:
        return self.metric not in WARNING_ONLY

    def __str__(self) -> str:  # pragma: no cover - formatting only
        tag = "" if self.blocking else " (warning)"
        return (
            f"  [{self.metric}{tag}] {self.path}:{self.lineno} {self.name} "
            f"= {self.value}, budget {self.budget}"
        )


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    modules: int = 0
    functions: int = 0

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.blocking]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if not v.blocking]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self.violations:
            out[v.metric] = out.get(v.metric, 0) + 1
        return out

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "modules": self.modules,
            "functions": self.functions,
            "budgets": BUDGETS,
            "counts": self.counts(),
            "violations": [asdict(v) for v in self.violations],
        }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

#: Nodes that each add one independent path through a function.
_BRANCH = (
    ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
    ast.With, ast.AsyncWith, ast.Assert, ast.comprehension, ast.match_case,
)


_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def complexity(node: ast.AST) -> int:
    """Cyclomatic complexity of one function: 1 plus its decision points.

    ``BoolOp`` counts ``len(values) - 1`` because ``a and b and c`` is two
    decisions, not one. Nested functions are **pruned**, not merely skipped:
    they are measured as functions in their own right, and walking into them
    anyway would charge a small helper's branches to the small function that
    happens to contain it.
    """
    total = 1
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, _NESTED):
            continue  # measured separately; do not descend
        if isinstance(child, ast.BoolOp):
            total += len(child.values) - 1
        elif isinstance(child, _BRANCH):
            total += 1
        stack.extend(ast.iter_child_nodes(child))
    return total


def _functions(tree: ast.AST) -> list[ast.AST]:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _qualname(tree: ast.AST, node: ast.AST) -> str:
    for parent in ast.walk(tree):
        if isinstance(parent, ast.ClassDef) and node in parent.body:
            return f"{parent.name}.{node.name}"
    return node.name


#: Not a dependency. ``from __future__ import annotations`` is a compiler
#: directive every module in this tree carries, and counting it would spend one
#: of the eight on a line that reaches nothing.
_NOT_A_DEPENDENCY = frozenset({"__future__"})


def fan_out(tree: ast.AST) -> set[str]:
    """Distinct modules this one imports. ``from a.b import c`` counts ``a.b``."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level)
    return out - _NOT_A_DEPENDENCY


# --------------------------------------------------------------------------
# one module
# --------------------------------------------------------------------------


def _check_module(rel: str, blob: bytes, report: Report) -> None:
    text = blob.decode("utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return
    report.modules += 1
    lines = len(text.splitlines())
    if lines > BUDGETS["module_lines"]:
        report.violations.append(Violation(
            "module_lines", rel, rel.rsplit("/", 1)[-1], 1, lines,
            BUDGETS["module_lines"],
        ))

    pkg = package_of(module_name(rel) or "")
    if not (pkg and pkg.orchestrator):
        edges = fan_out(tree)
        if len(edges) > BUDGETS["fan_out"]:
            report.violations.append(Violation(
                "fan_out", rel, rel.rsplit("/", 1)[-1], 1, len(edges),
                BUDGETS["fan_out"],
            ))

    for node in _functions(tree):
        report.functions += 1
        name = _qualname(tree, node)
        score = complexity(node)
        if score > BUDGETS["complexity"]:
            report.violations.append(Violation(
                "complexity", rel, name, node.lineno, score, BUDGETS["complexity"],
            ))
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        if length > BUDGETS["function_lines"]:
            report.violations.append(Violation(
                "function_lines", rel, name, node.lineno, length,
                BUDGETS["function_lines"],
            ))


def in_scope(rel: str) -> bool:
    """``engine/v2/**`` only. Legacy is exempt wholesale (§4.6)."""
    posix = rel.replace("\\", "/")
    return posix.startswith(_V2_PREFIX) and posix.endswith(".py")


def check_files(files: dict[str, bytes]) -> Report:
    """The pure core the CLI and the tests both drive."""
    report = Report()
    for rel in sorted(files):
        if in_scope(rel):
            _check_module(rel, files[rel], report)
    return report


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
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--paths", nargs="+")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    report = check_files(_collect(root, args))

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0 if report.ok else 1
    if not args.quiet:
        print(
            f"code budgets: {report.modules} v2 module(s), "
            f"{report.functions} function(s), zero exemptions"
        )
        for warning in report.warnings:
            print(str(warning))
    if report.ok:
        if not args.quiet:
            print("BUDGETS OK")
        return 0
    print(f"\nBUDGETS FAILED — {len(report.blocking)} violation(s):", file=sys.stderr)
    for v in report.blocking:
        print(str(v), file=sys.stderr)
    print(
        "\nSplit the function by the stage it is doing, rather than raising the "
        "budget: there is no exemption file and adding one is the thing this "
        "check exists to prevent.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
