#!/usr/bin/env python3
"""Fail closed when production and experiment source paths cross isolation.

Run ``python3 checks/live_experiment_isolation.py``. Import, root, syntax,
encoding, git-listing and source-read failures are loud CI refusals; no partial
scan is clean.
"""
from __future__ import annotations

import argparse
import ast
import os
import selectors
import stat
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = (
    "engine/v2/models/", "engine/v2/ops/retrain_", "tools/phase5_",
    "engine/v2/ops/training.py", "engine/v2/scoring/release_bindings.py",
)
EXPERIMENT_PATHS = ("engine/v2/research/", "experiments/")
EXPERIMENT_BASELINE_PATH = "checks/live_experiment_isolation.baseline"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024
MAX_CATALOG_BYTES = 16 * 1024 * 1024
MAX_CATALOG_STDERR_BYTES = 4 * 1024
GIT_TIMEOUT_SECONDS = 30
PRODUCTION_FORBIDDEN_IMPORTS = (
    "engine.v2.foundation.experiment_holdouts",
    "engine.v2.foundation.final_holdout",
    "engine.v2.ops.final_holdout",
    "engine.v2.research",
)
EXPERIMENT_FORBIDDEN_IMPORTS = (
    "engine.v2.ledger",
    "engine.v2.models.current_release",
    "engine.v2.models.current_pointer",
    "engine.v2.models.resolve_release",
    "engine.v2.models.deployment",
    "engine.v2.models.release_layout",
    "engine.v2.models.release_objects",
    "engine.v2.models.releases",
    "engine.v2.models.serving_fold_loader",
    "engine.v2.models.serving_fold_preflight",
    "engine.v2.models.inventory",
    "engine.v2.scoring.release_bindings",
    "engine.v2.ops.pool_refresh",
    "engine.ledger",
    "engine.paths.LEDGER",
)
EXPERIMENT_FORBIDDEN_TOKENS = (
    "model_release_root", "release_store", "release-store",
    "deployment_pointer", "deployment/deployed", "retrain_output_root",
    "training_output_root", "retrain-output", "training-output-root",
    "live_ledger", "live ledger", "score_ledger",
)
ROOT_CALLERS = {
    "production_release_root": frozenset({
        "engine/v2/models/deployment.py",
        "engine/v2/scoring/release_bindings.py",
        "engine/v2/ops/supervisor.py",
        "engine/v2/ops/state_stage.py",
        "engine/v2/ops/pool_refresh.py",
        "engine/v2/ops/nightly_trigger.py",
    }),
    "production_deployment_root": frozenset({"engine/v2/ops/training.py"}),
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    detail: str


class IsolationCheckError(RuntimeError):
    """The check could not complete its complete source scan."""


def _under(module: str, roots: tuple[str, ...]) -> bool:
    return any(module == root or module.startswith(root + ".") for root in roots)


def _module_path(path: str) -> str:
    return path[:-3].replace("/", ".").removesuffix(".__init__")


def _package(path: str) -> str:
    module = _module_path(path)
    return module if path.endswith("/__init__.py") else module.rpartition(".")[0]


def _imports(node: ast.Import | ast.ImportFrom, path: str) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module or ""
    if node.level:
        parts = _package(path).split(".")
        parts = parts[:len(parts) - node.level + 1]
        base = ".".join([*parts, base] if base else parts)
    if not base:
        return []
    return [base, *(base + "." + alias.name for alias in node.names)]


def _scope(path: str, roots: tuple[str, ...]) -> bool:
    return any(path.startswith(root) for root in roots)


class _Scope:
    def __init__(
        self, parent: _Scope | None, *, is_class: bool = False
    ) -> None:
        self.parent = parent
        self.is_class = is_class
        self.bound: set[str] = set()
        self.symbols: dict[str, set[str]] = {}
        self.dynamic_symbols: dict[str, str] = {}
        self.module_symbols: dict[str, str] = {}

    def bind(self, name: str, symbols: set[str] | None = None) -> None:
        self.bound.add(name)
        if symbols:
            self.symbols.setdefault(name, set()).update(symbols)

    def bind_dynamic(self, name: str, kind: str) -> None:
        self.dynamic_symbols[name] = kind
        self.bound.add(name)

    def bind_module(self, name: str, path: str) -> None:
        self.module_symbols[name] = path
        self.bound.add(name)

    def resolve(self, name: str) -> set[str]:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.bound:
                return set(scope.symbols.get(name, ()))
            scope = scope.parent
        return set()

    def resolve_dynamic(self, name: str) -> str | None:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.bound:
                return scope.dynamic_symbols.get(name)
            scope = scope.parent
        return None

    def resolve_module(self, name: str) -> str | None:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.bound:
                return scope.module_symbols.get(name)
            scope = scope.parent
        return None

    def is_bound(self, name: str) -> bool:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.bound:
                return True
            scope = scope.parent
        return False

    def attribute_scope(self) -> _Scope:
        scope: _Scope | None = self
        while scope is not None:
            if scope.is_class:
                return scope
            scope = scope.parent
        return self


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _qualified_module(node: ast.AST | None, scope: _Scope) -> str:
    if isinstance(node, ast.Name):
        return scope.resolve_module(node.id) or ""
    if isinstance(node, ast.Attribute):
        base = _qualified_module(node.value, scope)
        return f"{base}.{node.attr}" if base else ""
    return ""


def _target_keys(target: ast.AST) -> list[str]:
    key = _dotted(target)
    if key:
        return [key]
    if isinstance(target, (ast.Tuple, ast.List)):
        keys: list[str] = []
        for element in target.elts:
            keys.extend(_target_keys(element))
        return keys
    if isinstance(target, ast.Starred):
        return _target_keys(target.value)
    return []


def _target_scope(scope: _Scope, name: str) -> _Scope:
    return scope.attribute_scope() if "." in name else scope


def _dynamic_kind(value: ast.AST | None, scope: _Scope) -> str | None:
    if value is None:
        return None
    if isinstance(value, ast.Name):
        kind = scope.resolve_dynamic(value.id)
        if kind is not None:
            return kind
        if value.id == "__import__" and not scope.is_bound("__import__"):
            return "__import__"
        return None
    if isinstance(value, ast.Attribute):
        if isinstance(value.value, ast.Name):
            base = value.value.id
            base_kind = scope.resolve_dynamic(base)
            if base_kind == "importlib" and value.attr == "import_module":
                return "import_module"
            if base_kind == "builtins" and value.attr == "__import__":
                return "__import__"
            if base_kind is None and not scope.is_bound(base):
                if base == "importlib" and value.attr == "import_module":
                    return "import_module"
                if base == "builtins" and value.attr == "__import__":
                    return "__import__"
        return None
    return None


def _target_symbol_bindings(
    target: ast.AST, value: ast.AST | None, scope: _Scope
) -> list[tuple[str, set[str]]]:
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
    ):
        pairs: list[tuple[str, set[str]]] = []
        for target_element, value_element in zip(target.elts, value.elts):
            pairs.extend(
                _target_symbol_bindings(target_element, value_element, scope)
            )
        return pairs
    symbols = _resolver_symbols(value, scope)
    return [(name, symbols) for name in _target_keys(target)]


def _bind_target(scope: _Scope, target: ast.AST) -> None:
    for name in _target_keys(target):
        target_scope = _target_scope(scope, name)
        target_scope.bind(name)
        target_scope.dynamic_symbols.pop(name, None)
        target_scope.module_symbols.pop(name, None)


def _call_argument(
    node: ast.Call, position: int, keyword: str
) -> ast.AST | None:
    for item in node.keywords:
        if item.arg == keyword:
            return item.value
    if len(node.args) > position:
        return node.args[position]
    return None


def _literal_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _resolve_relative_import(name: str, package: str) -> str | None:
    level = len(name) - len(name.lstrip("."))
    remainder = name[level:]
    parts = package.split(".")
    if level < 1 or level > len(parts):
        return None
    base = parts[:len(parts) - level + 1]
    return ".".join([*base, remainder] if remainder else base)


def _exposes_forbidden_descendant(module: str, roots: tuple[str, ...]) -> bool:
    return any(root.startswith(module + ".") for root in roots)


def _getattr_symbols(node: ast.Call) -> set[str]:
    function = node.func
    is_getattr = (
        isinstance(function, ast.Name) and function.id == "getattr"
    ) or (
        isinstance(function, ast.Attribute) and _dotted(function) == "builtins.getattr"
    )
    if not is_getattr:
        return set()
    name = _literal_str(_call_argument(node, 1, "name"))
    if name in ROOT_CALLERS:
        return {name}
    return set()


def _resolver_symbols(node: ast.AST | None, scope: _Scope) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return scope.resolve(node.id)
    if isinstance(node, ast.Attribute):
        key = _dotted(node)
        if key:
            symbols = scope.resolve(key)
            if symbols:
                return symbols
        qualified = _qualified_module(node, scope)
        parent, _, final = qualified.rpartition(".")
        if (
            parent in ("engine.v2.models", "engine.v2.models.deployment")
            and final in ROOT_CALLERS
        ):
            return {final}
        return set()
    if isinstance(node, ast.Call):
        return _getattr_symbols(node)
    if isinstance(node, ast.IfExp):
        return _resolver_symbols(node.body, scope) | _resolver_symbols(
            node.orelse, scope)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        symbols: set[str] = set()
        for element in node.elts:
            symbols |= _resolver_symbols(element, scope)
        return symbols
    if isinstance(node, ast.Starred):
        return _resolver_symbols(node.value, scope)
    return set()


def _dynamic_import_spec(
    node: ast.Call,
    scope: _Scope,
    importlib_modules: set[str],
    import_module_names: set[str],
    builtins_modules: set[str],
    dunder_import_names: set[str],
) -> tuple[bool, list[str], bool, bool]:
    function = node.func
    kind: str | None = None
    if isinstance(function, ast.Name):
        kind = scope.resolve_dynamic(function.id)
        if kind is None and not scope.is_bound(function.id):
            if function.id == "__import__":
                kind = "__import__"
            elif function.id in dunder_import_names:
                kind = "__import__"
            elif function.id in import_module_names:
                kind = "import_module"
    elif isinstance(function, ast.Attribute):
        if isinstance(function.value, ast.Name):
            base = function.value.id
            base_kind = scope.resolve_dynamic(base)
            if base_kind == "importlib" and function.attr == "import_module":
                kind = "import_module"
            elif base_kind == "builtins" and function.attr == "__import__":
                kind = "__import__"
            elif base_kind is None and not scope.is_bound(base):
                if base == "importlib" and function.attr == "import_module":
                    kind = "import_module"
                elif base == "builtins" and function.attr == "__import__":
                    kind = "__import__"
                elif base in importlib_modules and function.attr == "import_module":
                    kind = "import_module"
                elif base in builtins_modules and function.attr == "__import__":
                    kind = "__import__"
    if kind is None:
        return False, [], False, False
    name = _literal_str(_call_argument(node, 0, "name"))
    if name is None:
        return True, [], True, False
    if name.startswith("."):
        package = _literal_str(_call_argument(node, 1, "package"))
        if package is None:
            return True, [], True, False
        target = _resolve_relative_import(name, package)
        if target is None:
            return True, [], True, False
    else:
        target = name
    if kind == "__import__":
        fromlist = _call_argument(node, 3, "fromlist")
        if isinstance(fromlist, ast.Constant) and fromlist.value is None:
            return True, [target], False, False
        if fromlist is not None:
            if not isinstance(fromlist, (ast.List, ast.Tuple)):
                return True, [], True, False
            members: list[str] = []
            for element in fromlist.elts:
                member = _literal_str(element)
                if member is None:
                    return True, [], True, False
                members.append(member)
            if members:
                if "*" in members:
                    return True, [target], False, True
                return (
                    True,
                    [target + "." + member for member in members],
                    False,
                    False,
                )
    return True, [target], False, False


def _static_import_targets(
    node: ast.Import | ast.ImportFrom, path: str
) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    targets = _imports(node, path)
    base = targets[0] if targets else ""
    if not base:
        return []
    if any(alias.name == "*" for alias in node.names):
        return [base]
    return [base + "." + alias.name for alias in node.names]


def _scoped_nodes(
    tree: ast.Module,
    path: str,
    importlib_modules: set[str],
    import_module_names: set[str],
    builtins_modules: set[str],
    dunder_import_names: set[str],
) -> tuple[
    list[tuple[_Scope, list[ast.AST], ast.AST | None]],
    list[tuple[_Scope, ast.Call]],
]:
    bindings: list[tuple[_Scope, list[ast.AST], ast.AST | None]] = []
    calls: list[tuple[_Scope, ast.Call]] = []
    module_scope = _Scope(None)
    experiment = _scope(path, EXPERIMENT_PATHS)

    def bind_arguments(
        arguments: ast.arguments, child: _Scope, enclosing: _Scope
    ) -> None:
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            child.bind(argument.arg)
        positional = (*arguments.posonlyargs, *arguments.args)
        for argument, default in zip(
            positional[len(positional) - len(arguments.defaults):],
            arguments.defaults,
        ):
            symbols = _resolver_symbols(default, enclosing)
            if symbols:
                child.bind(argument.arg, symbols)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            if default is not None:
                symbols = _resolver_symbols(default, enclosing)
                if symbols:
                    child.bind(argument.arg, symbols)
        if arguments.vararg is not None:
            child.bind(arguments.vararg.arg)
        if arguments.kwarg is not None:
            child.bind(arguments.kwarg.arg)

    def visit(node: ast.AST, scope: _Scope) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.bind(node.name)
            for item in (*node.decorator_list, *node.args.defaults,
                         *(node.args.kw_defaults or ())):
                if item is not None:
                    visit(item, scope)
            for argument in (*node.args.posonlyargs, *node.args.args,
                             *node.args.kwonlyargs, node.args.vararg,
                             node.args.kwarg):
                if argument is not None and argument.annotation is not None:
                    visit(argument.annotation, scope)
            if node.returns is not None:
                visit(node.returns, scope)
            child = _Scope(scope)
            bind_arguments(node.args, child, scope)
            for statement in node.body:
                visit(statement, child)
            return
        if isinstance(node, ast.Lambda):
            for item in (*node.args.defaults, *(node.args.kw_defaults or ())):
                if item is not None:
                    visit(item, scope)
            child = _Scope(scope)
            bind_arguments(node.args, child, scope)
            visit(node.body, child)
            return
        if isinstance(node, ast.ClassDef):
            scope.bind(node.name)
            for item in (*node.decorator_list, *node.bases,
                         *(keyword.value for keyword in node.keywords)):
                visit(item, scope)
            child = _Scope(scope, is_class=True)
            for statement in node.body:
                visit(statement, child)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                module_path = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
                if alias.name == "importlib":
                    importlib_modules.add(alias.asname or "importlib")
                    scope.bind_dynamic(name, "importlib")
                if alias.name == "builtins":
                    builtins_modules.add(alias.asname or "builtins")
                    scope.bind_dynamic(name, "builtins")
                scope.bind(name)
                if not (
                    experiment and (
                        alias.name == "engine.v2.models"
                        or _under(alias.name, EXPERIMENT_FORBIDDEN_IMPORTS)
                    )
                ):
                    scope.bind_module(name, module_path)
            return
        if isinstance(node, ast.ImportFrom):
            if (node.module or "") == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        name = alias.asname or alias.name
                        import_module_names.add(name)
                        scope.bind_dynamic(name, "import_module")
            if (node.module or "") == "builtins":
                for alias in node.names:
                    if alias.name == "__import__":
                        name = alias.asname or alias.name
                        dunder_import_names.add(name)
                        scope.bind_dynamic(name, "__import__")
            targets = _imports(node, path)
            if targets:
                base = targets[0]
                if (base == "engine.v2.models" or
                        base.rsplit(".", 1)[-1] == "deployment"):
                    for alias in node.names:
                        if alias.name in ROOT_CALLERS:
                            scope.bind(alias.asname or alias.name, {alias.name})
            for alias in node.names:
                if alias.name != "*":
                    name = alias.asname or alias.name
                    scope.bind(name)
                    if targets:
                        candidate = f"{targets[0]}.{alias.name}"
                        if not (
                            experiment and (
                                candidate == "engine.v2.models"
                                or _under(candidate, EXPERIMENT_FORBIDDEN_IMPORTS)
                            )
                        ):
                            scope.bind_module(name, candidate)
            return
        if isinstance(node, ast.Assign):
            kind = _dynamic_kind(node.value, scope)
            module_path = _qualified_module(node.value, scope)
            bindings.append((scope, list(node.targets), node.value))
            for target in node.targets:
                _bind_target(scope, target)
                if kind is not None and isinstance(target, ast.Name):
                    _target_scope(scope, target.id).bind_dynamic(target.id, kind)
                if module_path and isinstance(target, ast.Name):
                    scope.bind_module(target.id, module_path)
                visit(target, scope)
            visit(node.value, scope)
            return
        if isinstance(node, ast.AnnAssign):
            kind = _dynamic_kind(node.value, scope)
            module_path = _qualified_module(node.value, scope)
            bindings.append((scope, [node.target], node.value))
            _bind_target(scope, node.target)
            if kind is not None and isinstance(node.target, ast.Name):
                _target_scope(scope, node.target.id).bind_dynamic(node.target.id, kind)
            if module_path and isinstance(node.target, ast.Name):
                scope.bind_module(node.target.id, module_path)
            visit(node.target, scope)
            if node.value is not None:
                visit(node.value, scope)
            visit(node.annotation, scope)
            return
        if isinstance(node, ast.NamedExpr):
            kind = _dynamic_kind(node.value, scope)
            module_path = _qualified_module(node.value, scope)
            bindings.append((scope, [node.target], node.value))
            _bind_target(scope, node.target)
            if kind is not None and isinstance(node.target, ast.Name):
                _target_scope(scope, node.target.id).bind_dynamic(node.target.id, kind)
            if module_path and isinstance(node.target, ast.Name):
                scope.bind_module(node.target.id, module_path)
            visit(node.target, scope)
            visit(node.value, scope)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            bindings.append((scope, [node.target], node.iter))
            _bind_target(scope, node.target)
            for child in ast.iter_child_nodes(node):
                visit(child, scope)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is None:
                    continue
                bindings.append((scope, [item.optional_vars], item.context_expr))
                _bind_target(scope, item.optional_vars)
            for child in ast.iter_child_nodes(node):
                visit(child, scope)
            return
        if isinstance(node, ast.Call):
            calls.append((scope, node))
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            child = _Scope(scope)
            for index, generator in enumerate(node.generators):
                current = scope if index == 0 else child
                visit(generator.iter, current)
                bindings.append((child, [generator.target], generator.iter))
                _bind_target(child, generator.target)
                for condition in generator.ifs:
                    visit(condition, child)
            if isinstance(node, ast.DictComp):
                visit(node.key, child)
                visit(node.value, child)
            else:
                visit(node.elt, child)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    for statement in tree.body:
        visit(statement, module_scope)
    return bindings, calls


def _transitive_import_chains(trees: dict[str, ast.Module]) -> dict[str, str]:
    """Return a shortest chain per experiment source and forbidden boundary."""
    module_map = {_module_path(path): path for path in trees}
    imports_by_module: dict[str, set[str]] = {}
    for path, tree in trees.items():
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                targets.update(_static_import_targets(node, path))
        imports_by_module[_module_path(path)] = targets

    graph: dict[str, set[str]] = {module: set() for module in module_map}
    for module, targets in imports_by_module.items():
        for target in targets:
            parts = target.split(".")
            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if prefix in module_map:
                    graph[module].add(prefix)

    chains: dict[str, str] = {}
    for path in sorted(trees):
        if not _scope(path, EXPERIMENT_PATHS):
            continue
        start = _module_path(path)
        direct: set[str] = set()
        for target in imports_by_module.get(start, ()):
            parts = target.split(".")
            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if prefix in module_map:
                    direct.add(prefix)
        visited = {start}
        found_roots: set[str] = set()
        queue = deque([(start, [start])])
        while queue:
            current, chain = queue.popleft()
            roots = [
                root for root in EXPERIMENT_FORBIDDEN_IMPORTS
                if current != start and _under(current, (root,))
            ]
            if roots:
                root = max(roots, key=len)
                if current not in direct and root not in found_roots:
                    canonical_chain = chain if current == root else [*chain[:-1], root]
                    key = f"{path} -> {' -> '.join(canonical_chain)}"
                    chains[key] = " -> ".join(canonical_chain)
                    found_roots.add(root)
                continue
            for target in sorted(graph.get(current, ())):
                if target not in visited:
                    visited.add(target)
                    queue.append((target, chain + [target]))
    return chains


def _parse_source_trees(
    files: dict[str, bytes],
) -> tuple[dict[str, ast.Module], list[Finding]]:
    """Parse tracked Python sources; syntax and encoding errors become findings."""
    findings: list[Finding] = []
    trees: dict[str, ast.Module] = {}
    for path, content in sorted(files.items()):
        if not path.endswith(".py"):
            continue
        try:
            source = content.decode("utf-8")
            trees[path] = ast.parse(source, filename=path)
        except (UnicodeDecodeError, SyntaxError) as exc:
            line = getattr(exc, "lineno", 1) or 1
            findings.append(Finding(path, line, "source-invalid", str(exc)))
    return trees, findings


def check_files(
    files: dict[str, bytes],
    *,
    transitive_baseline: frozenset[str] | None = None,
    transitive_baseline_ceiling: frozenset[str] | None = None,
) -> list[Finding]:
    """Return every static violation; syntax and encoding errors are findings."""
    findings: list[Finding] = []
    finding_keys: set[tuple[str, int, str]] = set()
    trees, parse_findings = _parse_source_trees(files)
    findings.extend(parse_findings)

    def add_forbidden_import(path: str, line: int, detail: str) -> None:
        key = (path, line, "forbidden-import")
        if key not in finding_keys:
            findings.append(Finding(path, line, "forbidden-import", detail))
            finding_keys.add(key)

    for path, tree in trees.items():
        production = _scope(path, PRODUCTION_PATHS)
        experiment = _scope(path, EXPERIMENT_PATHS)
        denied = (PRODUCTION_FORBIDDEN_IMPORTS if production else ())
        denied += EXPERIMENT_FORBIDDEN_IMPORTS if experiment else ()
        importlib_modules = {"importlib"}
        import_module_names: set[str] = set()
        builtins_modules: set[str] = set()
        dunder_import_names: set[str] = set()
        bindings, calls = _scoped_nodes(
            tree, path, importlib_modules, import_module_names,
            builtins_modules, dunder_import_names)
        call_scopes = {call: scope for scope, call in calls}
        changed = True
        while changed:
            changed = False
            for scope, targets, value in bindings:
                for target in targets:
                    for name, symbols in _target_symbol_bindings(
                            target, value, scope):
                        if not symbols:
                            continue
                        target_scope = _target_scope(scope, name)
                        current = target_scope.symbols.setdefault(name, set())
                        if not symbols.issubset(current):
                            current.update(symbols)
                            changed = True
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in _static_import_targets(node, path):
                    refused = _under(target, denied) or (
                        experiment and target == "engine.v2.models")
                    if refused:
                        add_forbidden_import(
                            path, node.lineno,
                            f"{target} crosses the D7 isolation boundary",
                        )
                if experiment and isinstance(node, ast.ImportFrom):
                    targets = _imports(node, path)
                    base = targets[0] if targets else ""
                    star_refused = any(
                        alias.name == "*" for alias in node.names
                    ) and any(
                        root.rpartition(".")[0] == base
                        for root in EXPERIMENT_FORBIDDEN_IMPORTS
                    )
                    if star_refused:
                        add_forbidden_import(
                            path, node.lineno,
                            f"{base} star import exposes a forbidden API",
                        )
            if experiment and isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
                if isinstance(node, ast.Attribute) and node.attr == "LEDGER":
                    findings.append(Finding(
                        path, getattr(node, "lineno", 1), "forbidden-live-root",
                        "experiment source accesses isolated live root 'LEDGER'",
                    ))
                else:
                    value = (node.id if isinstance(node, ast.Name) else
                             node.attr if isinstance(node, ast.Attribute) else node.value)
                    if isinstance(value, str):
                        normalized = value.lower().replace("\\", "/")
                        if (value == "DEPLOYED" or
                                any(token in normalized for token in EXPERIMENT_FORBIDDEN_TOKENS)):
                            findings.append(Finding(
                                path, getattr(node, "lineno", 1), "forbidden-live-root",
                                f"experiment source names isolated live root/token {value!r}",
                            ))
            if experiment and isinstance(node, ast.Attribute):
                dotted = _dotted(node)
                if dotted and _under(dotted, EXPERIMENT_FORBIDDEN_IMPORTS):
                    add_forbidden_import(
                        path, node.lineno,
                        f"{dotted} crosses the D7 isolation boundary",
                    )
            if (production or experiment) and isinstance(node, ast.Call):
                scope = call_scopes[node]
                recognized, targets, unresolved, wildcard = _dynamic_import_spec(
                    node, scope, importlib_modules, import_module_names,
                    builtins_modules, dunder_import_names)
                if recognized and unresolved:
                    add_forbidden_import(
                        path, node.lineno,
                        "dynamic import target cannot be resolved statically",
                    )
                elif recognized:
                    for target in targets:
                        refused = _under(target, denied) or (
                            experiment and target == "engine.v2.models")
                        if wildcard and _exposes_forbidden_descendant(
                                target, denied):
                            refused = True
                        if refused:
                            add_forbidden_import(
                                path, node.lineno,
                                f"{target} crosses the D7 isolation boundary",
                            )
        if experiment:
            for scope, call in calls:
                qualified = _qualified_module(call.func, scope)
                if qualified and _under(qualified, EXPERIMENT_FORBIDDEN_IMPORTS):
                    add_forbidden_import(
                        path, call.lineno,
                        f"{qualified} crosses the D7 isolation boundary",
                    )
        if not path.startswith("tests/"):
            for scope, call in calls:
                symbols = _resolver_symbols(call.func, scope)
                if not symbols:
                    function = call.func
                    if isinstance(function, ast.Name) and function.id in ROOT_CALLERS:
                        symbols = {function.id}
                for symbol in symbols:
                    allowed = ROOT_CALLERS.get(symbol)
                    if allowed is not None and path not in allowed:
                        findings.append(Finding(
                            path, call.lineno, "unlisted-root-caller",
                            f"{symbol} caller must be reviewed and added to ROOT_CALLERS",
                        ))
                        break

    chains = _transitive_import_chains(trees)
    baseline = transitive_baseline if transitive_baseline is not None else frozenset()
    baseline_lines: dict[str, int] = {}
    if transitive_baseline is not None and EXPERIMENT_BASELINE_PATH in files:
        try:
            baseline_text = files[EXPERIMENT_BASELINE_PATH].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise IsolationCheckError(
                "invalid UTF-8 transitive import baseline"
            ) from exc
        for line_number, line in enumerate(baseline_text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            baseline_lines.setdefault(stripped, line_number)

    if transitive_baseline_ceiling is not None:
        for entry in sorted(baseline - transitive_baseline_ceiling):
            findings.append(Finding(
                EXPERIMENT_BASELINE_PATH,
                baseline_lines.get(entry, 1),
                "transitive-baseline-growth",
                f"baseline entry exceeds initial ceiling: {entry}",
            ))

    for key in sorted(chains):
        if key not in baseline:
            findings.append(Finding(
                key.split(" -> ", 1)[0],
                1,
                "transitive-forbidden-import",
                f"{key}: {chains[key]}",
            ))

    for key in sorted(baseline):
        if key not in chains:
            findings.append(Finding(
                EXPERIMENT_BASELINE_PATH,
                baseline_lines.get(key, 1),
                "transitive-baseline-stale",
                key,
            ))
    return findings


def _read_source(path: Path) -> bytes:
    """Read one regular tracked source file without following links or exceeding the cap."""
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise IsolationCheckError("safe no-follow source reads are unavailable")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | no_follow)
    except OSError as exc:
        raise IsolationCheckError(f"tracked source read refused: {path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise IsolationCheckError(f"tracked source is not a regular file: {path}")
        chunks = []
        size = 0
        while size <= MAX_SOURCE_BYTES:
            chunk = os.read(fd, min(64 * 1024, MAX_SOURCE_BYTES + 1 - size))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            size += len(chunk)
        raise IsolationCheckError(
            f"tracked source exceeds {MAX_SOURCE_BYTES} bytes: {path}"
        )
    except OSError as exc:
        raise IsolationCheckError(f"tracked source read refused: {path}: {exc}") from exc
    finally:
        os.close(fd)


def _git_catalog(root: Path) -> bytes:
    """Return bounded ``git ls-files`` stdout, draining both pipes before the deadline."""
    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    try:
        selector = selectors.DefaultSelector()
    except OSError as exc:
        raise IsolationCheckError(
            f"complete tracked-source scan refused: git ls-files failed: {exc}"
        ) from exc
    try:
        process = subprocess.Popen(
            ["git", "ls-files", "-z", "--", "*.py", EXPERIMENT_BASELINE_PATH], cwd=root,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as exc:
        selector.close()
        raise IsolationCheckError(
            f"complete tracked-source scan refused: git ls-files failed: {exc}"
        ) from exc
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_size = 0
    stderr_size = 0
    try:
        stdout = process.stdout
        stderr = process.stderr
        if stdout is None or stderr is None:
            raise IsolationCheckError(
                "complete tracked-source scan refused: git ls-files pipes unavailable"
            )
        os.set_blocking(stdout.fileno(), False)
        os.set_blocking(stderr.fileno(), False)
        selector.register(stdout, selectors.EVENT_READ)
        selector.register(stderr, selectors.EVENT_READ)
        open_streams = 2
        while open_streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IsolationCheckError(
                    "complete tracked-source scan refused: git ls-files exceeded "
                    f"{GIT_TIMEOUT_SECONDS}s"
                )
            for key, _ in selector.select(remaining):
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    open_streams -= 1
                    continue
                if stream is stdout:
                    stdout_size += len(chunk)
                    if stdout_size > MAX_CATALOG_BYTES:
                        raise IsolationCheckError(
                            "complete tracked-source scan refused: git ls-files output "
                            f"exceeds {MAX_CATALOG_BYTES} bytes"
                        )
                    stdout_chunks.append(chunk)
                elif stderr_size < MAX_CATALOG_STDERR_BYTES:
                    keep = chunk[:MAX_CATALOG_STDERR_BYTES - stderr_size]
                    stderr_chunks.append(keep)
                    stderr_size += len(keep)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise IsolationCheckError(
                "complete tracked-source scan refused: git ls-files exceeded "
                f"{GIT_TIMEOUT_SECONDS}s"
            )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise IsolationCheckError(
                "complete tracked-source scan refused: git ls-files exceeded "
                f"{GIT_TIMEOUT_SECONDS}s"
            ) from exc
        if returncode != 0:
            diagnostic = b"".join(stderr_chunks).decode("utf-8", "replace").strip()
            raise IsolationCheckError(
                "complete tracked-source scan refused: git ls-files exited "
                f"{returncode}: {diagnostic}"
            )
        return b"".join(stdout_chunks)
    except (OSError, ValueError) as exc:
        raise IsolationCheckError(
            f"complete tracked-source scan refused: git ls-files failed: {exc}"
        ) from exc
    finally:
        try:
            selector.close()
        finally:
            try:
                if process.poll() is None:
                    process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=GIT_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def _load_sources(root: Path = ROOT) -> dict[str, bytes]:
    """Read every tracked Python source; refuse if any part of the scan fails."""
    try:
        catalog = _git_catalog(root)
        paths = []
        for item in catalog.split(b"\0"):
            if not item:
                continue
            try:
                paths.append(item.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise IsolationCheckError(
                    f"complete tracked-source scan refused: invalid UTF-8 path: {exc}"
                ) from exc
        files: dict[str, bytes] = {}
        total_source_bytes = 0
        for path in paths:
            content = _read_source(root / path)
            total_source_bytes += len(content)
            if total_source_bytes > MAX_TOTAL_SOURCE_BYTES:
                raise IsolationCheckError(
                    "complete tracked-source scan refused: aggregate Python source "
                    f"exceeds {MAX_TOTAL_SOURCE_BYTES} bytes"
                )
            files[path] = content
    except (OSError, UnicodeDecodeError, subprocess.SubprocessError) as exc:
        raise IsolationCheckError(f"complete tracked-source scan refused: {exc}") from exc
    return files


def _baseline_entries(files: dict[str, bytes]) -> list[str]:
    """Parse and validate committed baseline lines; refuse malformed baselines."""
    try:
        baseline_text = files[EXPERIMENT_BASELINE_PATH].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IsolationCheckError(
            f"complete tracked-source scan refused: invalid UTF-8 baseline: {exc}"
        ) from exc
    baseline_entries: list[str] = []
    for line in baseline_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        baseline_entries.append(stripped)
    if len(baseline_entries) != len(set(baseline_entries)):
        raise IsolationCheckError(
            "complete tracked-source scan refused: duplicate baseline entries"
        )
    if baseline_entries != sorted(baseline_entries):
        raise IsolationCheckError(
            "complete tracked-source scan refused: unsorted baseline entries"
        )
    return baseline_entries


def _run_bounded_git(
    command: list[str], root: Path, *, output_limit: int, operation: str
) -> bytes:
    """Run Git draining both pipes before the deadline with bounded stdout."""
    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    try:
        selector = selectors.DefaultSelector()
    except OSError as exc:
        raise IsolationCheckError(
            f"complete tracked-source scan refused: {operation} failed: {exc}"
        ) from exc
    process: subprocess.Popen[bytes] | None = None
    try:
        try:
            process = subprocess.Popen(
                command, cwd=root,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise IsolationCheckError(
                f"complete tracked-source scan refused: {operation} failed: {exc}"
            ) from exc
        stdout = process.stdout
        stderr = process.stderr
        if stdout is None or stderr is None:
            raise IsolationCheckError(
                f"complete tracked-source scan refused: {operation} pipes unavailable"
            )
        os.set_blocking(stdout.fileno(), False)
        os.set_blocking(stderr.fileno(), False)
        selector.register(stdout, selectors.EVENT_READ)
        selector.register(stderr, selectors.EVENT_READ)
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stdout_size = 0
        stderr_size = 0
        open_streams = 2
        while open_streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IsolationCheckError(
                    f"complete tracked-source scan refused: {operation} exceeded "
                    f"{GIT_TIMEOUT_SECONDS}s"
                )
            for key, _ in selector.select(remaining):
                stream = key.fileobj
                try:
                    if stream is stdout:
                        read_limit = min(64 * 1024, output_limit - stdout_size + 1)
                    else:
                        read_limit = 64 * 1024
                    chunk = os.read(stream.fileno(), read_limit)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    open_streams -= 1
                    continue
                if stream is stdout:
                    if stdout_size + len(chunk) > output_limit:
                        raise IsolationCheckError(
                            f"complete tracked-source scan refused: {operation} "
                            f"output exceeds {output_limit} bytes"
                        )
                    stdout_size += len(chunk)
                    stdout_chunks.append(chunk)
                elif stderr_size < MAX_CATALOG_STDERR_BYTES:
                    keep = chunk[:MAX_CATALOG_STDERR_BYTES - stderr_size]
                    stderr_chunks.append(keep)
                    stderr_size += len(keep)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise IsolationCheckError(
                f"complete tracked-source scan refused: {operation} exceeded "
                f"{GIT_TIMEOUT_SECONDS}s"
            )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise IsolationCheckError(
                f"complete tracked-source scan refused: {operation} exceeded "
                f"{GIT_TIMEOUT_SECONDS}s"
            ) from exc
        if returncode != 0:
            diagnostic = b"".join(stderr_chunks).decode("utf-8", "replace").strip()
            raise IsolationCheckError(
                f"complete tracked-source scan refused: {operation} exited "
                f"{returncode}: {diagnostic}"
            )
        return b"".join(stdout_chunks)
    except (OSError, ValueError) as exc:
        raise IsolationCheckError(
            f"complete tracked-source scan refused: {operation} failed: {exc}"
        ) from exc
    finally:
        try:
            selector.close()
        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=GIT_TIMEOUT_SECONDS)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()


def _baseline_ceiling(root: Path) -> frozenset[str]:
    """Load the first committed baseline as the maximum allowed debt."""
    history = _run_bounded_git(
        [
            "git", "log", "--reverse", "--diff-filter=A", "--format=%H",
            "--", EXPERIMENT_BASELINE_PATH,
        ],
        root,
        output_limit=MAX_CATALOG_BYTES,
        operation="baseline history",
    )
    try:
        commits = history.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise IsolationCheckError(
            f"complete tracked-source scan refused: invalid baseline history: {exc}"
        ) from exc
    if not commits:
        raise IsolationCheckError(
            "complete tracked-source scan refused: baseline has no committed origin"
        )
    seed = _run_bounded_git(
        ["git", "show", f"{commits[0]}:{EXPERIMENT_BASELINE_PATH}"],
        root,
        output_limit=MAX_SOURCE_BYTES,
        operation="historical baseline",
    )
    entries = _baseline_entries({EXPERIMENT_BASELINE_PATH: seed})
    return frozenset(entries)


def scan_repository(root: Path = ROOT) -> list[Finding]:
    """Read every tracked Python source; refuse if any part of the scan fails."""
    files = _load_sources(root)
    if EXPERIMENT_BASELINE_PATH not in files:
        raise IsolationCheckError(
            f"complete tracked-source scan refused: missing baseline "
            f"{EXPERIMENT_BASELINE_PATH}"
        )
    baseline = frozenset(_baseline_entries(files))
    ceiling = _baseline_ceiling(root)
    return check_files(
        files,
        transitive_baseline=baseline,
        transitive_baseline_ceiling=ceiling,
    )


BASELINE_COMMENT = (
    "# Existing transitive import-chain debt; this baseline can only shrink. "
    "Burn-down issue: #1466."
)


def write_baseline(root: Path = ROOT) -> None:
    """Create a fresh baseline; refuse if it exists or the scan is not clean."""
    target = root / EXPERIMENT_BASELINE_PATH
    if os.path.lexists(target):
        raise IsolationCheckError(
            f"baseline write refused: {EXPERIMENT_BASELINE_PATH} already exists"
        )
    files = _load_sources(root)
    trees, parse_findings = _parse_source_trees(files)
    if parse_findings:
        details = "; ".join(
            f"{item.path}:{item.line}: {item.detail}" for item in parse_findings
        )
        raise IsolationCheckError(
            f"baseline write refused: source reading failed: {details}"
        )
    chains = _transitive_import_chains(trees)
    findings = check_files(files, transitive_baseline=frozenset(chains))
    blocking = [item for item in findings if not item.rule.startswith("transitive-")]
    if blocking:
        details = "; ".join(
            f"{item.path}:{item.line}: [{item.rule}] {item.detail}"
            for item in blocking
        )
        raise IsolationCheckError(
            f"baseline write refused: non-transitive isolation findings: {details}"
        )
    lines = [BASELINE_COMMENT, *sorted(chains)]
    content = ("\n".join(lines) + "\n").encode("utf-8")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow
    try:
        fd = os.open(target, flags, 0o644)
    except FileExistsError as exc:
        raise IsolationCheckError(
            f"baseline write refused: {EXPERIMENT_BASELINE_PATH} already exists"
        ) from exc
    except OSError as exc:
        raise IsolationCheckError(
            f"baseline write refused: cannot create "
            f"{EXPERIMENT_BASELINE_PATH}: {exc}"
        ) from exc
    try:
        handle = os.fdopen(fd, "wb")
    except OSError as exc:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(target)
        except OSError:
            pass
        raise IsolationCheckError(
            f"baseline write refused: cannot write "
            f"{EXPERIMENT_BASELINE_PATH}: {exc}"
        ) from exc
    try:
        with handle:
            handle.write(content)
            handle.flush()
    except OSError as exc:
        try:
            os.unlink(target)
        except OSError:
            pass
        raise IsolationCheckError(
            f"baseline write refused: cannot write "
            f"{EXPERIMENT_BASELINE_PATH}: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="Check live experiment isolation.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="discover and exclusively create the current transitive debt baseline",
    )
    try:
        options = parser.parse_args(arguments)
    except SystemExit as exc:
        return int(exc.code)
    if options.write_baseline:
        try:
            write_baseline()
        except IsolationCheckError as exc:
            print(f"live experiment isolation refused: {exc}", file=sys.stderr)
            return 1
        print(
            f"live experiment isolation: wrote baseline "
            f"{EXPERIMENT_BASELINE_PATH}"
        )
        return 0
    try:
        findings = scan_repository()
    except IsolationCheckError as exc:
        print(f"live experiment isolation refused: {exc}", file=sys.stderr)
        return 1
    if findings:
        for item in findings:
            print(f"{item.path}:{item.line}: [{item.rule}] {item.detail}", file=sys.stderr)
        return 1
    print("live experiment isolation: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())