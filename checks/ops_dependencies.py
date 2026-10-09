"""Ratchet the planned ops package graph from design PR #499, without imports."""
from __future__ import annotations

import ast
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from checks.architecture_doc_budgets import (
    _base_branch,
    _clean_git_env,
    _clean_git_process_env,
    _read_base,
    _resolve_base_ref,
)
from checks.import_layers import _absolute, module_name, package_path
from checks.repo_hygiene import tracked_paths

POLICY = "checks/ops_dependencies.json"
PREFIX = "engine.v2.ops"
SOURCE = "engine/v2/ops/"


def findings(files, policy):
    """Return forbidden and cyclic module edges, including function imports."""
    if any(type(level) is not int for level in policy["levels"].values()):
        raise ValueError("package levels must be integers")
    owners = {}
    for package, names in policy["packages"].items():
        for name in names:
            module = PREFIX + ("." + name if name else "")
            if module in owners or package not in policy["levels"]:
                raise ValueError("duplicate module or missing package level")
            owners[module] = package
    modules = {module_name(path): path for path in files}
    if len(modules) != len(files):
        raise ValueError("duplicate ops source module")
    if modules.keys() != owners.keys():
        raise ValueError(f"ops ownership mismatch: {sorted(modules.keys() ^ owners.keys())}")
    edges = set()
    for module, path in modules.items():
        for node in ast.walk(ast.parse(files[path], filename=path)):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                namespace = _absolute(node, package_path(path, module))
                targets = [f"{namespace}.{alias.name}" if f"{namespace}.{alias.name}" in modules
                           else namespace for alias in node.names]
            for target in targets:
                while target and target not in modules:
                    target = target.rpartition(".")[0]
                if target:
                    edges.add((module, target))
    graph = defaultdict(set)
    for source, target in edges:
        graph[source].add(target)

    def reaches(start, goal):
        pending, seen = [start], set()
        while pending:
            node = pending.pop()
            if node == goal:
                return True
            if node not in seen:
                seen.add(node)
                pending.extend(graph[node] - seen)
        return False

    def short(edge):
        return tuple(name.removeprefix(PREFIX).lstrip(".") for name in edge)

    levels = policy["levels"]
    forbidden = {short((a, b)) for a, b in edges if owners[a] != owners[b]
                 and levels[owners[a]] <= levels[owners[b]]}
    cycles = {short((a, b)) for a, b in edges if reaches(b, a)}
    return {"forbidden": forbidden, "cycles": cycles}


def check_findings(actual, allowed, baseline):
    """Exact exceptions, with no additions even inside an existing cycle."""
    problems = []
    for kind in ("forbidden", "cycles"):
        exceptions = {tuple(edge) for edge in allowed[kind]}
        ceiling = {tuple(edge) for edge in baseline[kind]}
        for label, edges in (("new", actual[kind] - exceptions),
                             ("stale exception", exceptions - actual[kind]),
                             ("expanded exception", exceptions - ceiling)):
            problems.extend(f"ops {kind}: {label}: {a} -> {b}" for a, b in sorted(edges))
    return problems


def check_repository(root: Path, base_ref: str | None = None):
    """Read complete tracked worktree and base policy; bootstrap from base sources."""
    policy = json.loads((root / POLICY).read_bytes())
    with _clean_git_process_env():
        paths = [p for p in tracked_paths(root) if p.startswith(SOURCE) and p.endswith(".py")]
    actual = findings({p: (root / p).read_bytes() for p in paths}, policy)
    ref = base_ref or _resolve_base_ref(root, _base_branch())
    blob = _read_base(root, POLICY, ref)
    if blob is not None:
        baseline = json.loads(blob)["exceptions"]
    else:
        # The first enforcement PR cannot grandfather a violation it introduces.
        result = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "--name-only",
                                 ref, "--", SOURCE], check=True, capture_output=True, text=True,
                                env=_clean_git_env())
        paths = [p for p in result.stdout.splitlines() if p.endswith(".py")]
        baseline = findings({p: _read_base(root, p, ref) for p in paths}, policy)
    return check_findings(actual, policy["exceptions"], baseline)
