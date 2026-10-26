#!/usr/bin/env python3
"""Fail closed when production and experiment source paths cross isolation.

Run ``python3 checks/live_experiment_isolation.py``. Import, root, syntax,
encoding, git-listing and source-read failures are loud CI refusals; no partial
scan is clean.
"""
from __future__ import annotations

import ast
import os
import selectors
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = (
    "engine/v2/models/", "engine/v2/ops/retrain_", "tools/phase5_",
    "engine/v2/ops/training.py", "engine/v2/scoring/release_bindings.py",
)
EXPERIMENT_PATHS = ("engine/v2/research/", "experiments/")
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
    "engine.v2.models.inventory",
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
    def __init__(self, parent: _Scope | None) -> None:
        self.parent = parent
        self.bound: set[str] = set()
        self.symbols: dict[str, set[str]] = {}

    def bind(self, name: str, symbols: set[str] | None = None) -> None:
        self.bound.add(name)
        if symbols:
            self.symbols.setdefault(name, set()).update(symbols)

    def resolve(self, name: str) -> set[str]:
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.bound:
                return set(scope.symbols.get(name, ()))
            scope = scope.parent
        return set()


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_target_names(element))
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _dynamic_import_target(
    node: ast.Call,
    importlib_modules: set[str],
    import_module_names: set[str],
) -> str | None:
    function = node.func
    if isinstance(function, ast.Name):
        is_import = function.id == "__import__" or function.id in import_module_names
    elif isinstance(function, ast.Attribute):
        is_import = _dotted(function) == "importlib.import_module" or (
            isinstance(function.value, ast.Name)
            and function.value.id in importlib_modules
            and function.attr == "import_module"
        )
    else:
        is_import = False
    if not is_import or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _scoped_nodes(
    tree: ast.Module,
    path: str,
    importlib_modules: set[str],
    import_module_names: set[str],
) -> tuple[list[tuple[_Scope, ast.Assign]], list[tuple[_Scope, ast.Call]]]:
    assignments: list[tuple[_Scope, ast.Assign]] = []
    calls: list[tuple[_Scope, ast.Call]] = []
    module_scope = _Scope(None)

    def bind_arguments(arguments: ast.arguments, scope: _Scope) -> None:
        for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            scope.bind(argument.arg)
        if arguments.vararg is not None:
            scope.bind(arguments.vararg.arg)
        if arguments.kwarg is not None:
            scope.bind(arguments.kwarg.arg)

    def visit(node: ast.AST, scope: _Scope) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.bind(node.name)
            for item in (*node.decorator_list, *node.args.defaults,
                         *(node.args.kw_defaults or ())):
                if item is not None:
                    visit(item, scope)
            child = _Scope(scope)
            bind_arguments(node.args, child)
            for statement in node.body:
                visit(statement, child)
            return
        if isinstance(node, ast.Lambda):
            for item in (*node.args.defaults, *(node.args.kw_defaults or ())):
                if item is not None:
                    visit(item, scope)
            child = _Scope(scope)
            bind_arguments(node.args, child)
            visit(node.body, child)
            return
        if isinstance(node, ast.ClassDef):
            scope.bind(node.name)
            for item in (*node.decorator_list, *node.bases,
                         *(keyword.value for keyword in node.keywords)):
                visit(item, scope)
            child = _Scope(scope)
            for statement in node.body:
                visit(statement, child)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_modules.add(alias.asname or "importlib")
                scope.bind(alias.asname or alias.name.split(".")[0])
            return
        if isinstance(node, ast.ImportFrom):
            if (node.module or "") == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        import_module_names.add(alias.asname or alias.name)
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
                    scope.bind(alias.asname or alias.name)
            return
        if isinstance(node, ast.Assign):
            assignments.append((scope, node))
            for target in node.targets:
                for name in _target_names(target):
                    scope.bind(name)
                visit(target, scope)
            visit(node.value, scope)
            return
        if isinstance(node, ast.Call):
            calls.append((scope, node))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    for statement in tree.body:
        visit(statement, module_scope)
    return assignments, calls


def check_files(files: dict[str, bytes]) -> list[Finding]:
    """Return every static violation; syntax and encoding errors are findings."""
    findings: list[Finding] = []
    finding_keys: set[tuple[str, int, str]] = set()
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
        assignments, calls = _scoped_nodes(
            tree, path, importlib_modules, import_module_names)
        changed = True
        while changed:
            changed = False
            for scope, assigned in assignments:
                value = assigned.value
                if isinstance(value, ast.Attribute):
                    symbols = {value.attr} if value.attr in ROOT_CALLERS else set()
                elif isinstance(value, ast.Name):
                    symbols = scope.resolve(value.id)
                else:
                    symbols = set()
                for target in assigned.targets:
                    for name in _target_names(target):
                        current = scope.symbols.setdefault(name, set())
                        if not symbols.issubset(current):
                            current.update(symbols)
                            changed = True
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in _imports(node, path):
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
            if experiment and isinstance(node, ast.Call):
                target = _dynamic_import_target(
                    node, importlib_modules, import_module_names)
                if target and _under(target, EXPERIMENT_FORBIDDEN_IMPORTS):
                    add_forbidden_import(
                        path, node.lineno,
                        f"{target} crosses the D7 isolation boundary",
                    )
        if not path.startswith("tests/"):
            for scope, call in calls:
                function = call.func
                if isinstance(function, ast.Name):
                    symbols = scope.resolve(function.id)
                    if not symbols and function.id in ROOT_CALLERS:
                        symbols = {function.id}
                elif isinstance(function, ast.Attribute):
                    symbols = scope.resolve(function.attr)
                    if not symbols and function.attr in ROOT_CALLERS:
                        symbols = {function.attr}
                else:
                    symbols = set()
                for symbol in symbols:
                    allowed = ROOT_CALLERS.get(symbol)
                    if allowed is not None and path not in allowed:
                        findings.append(Finding(
                            path, call.lineno, "unlisted-root-caller",
                            f"{symbol} caller must be reviewed and added to ROOT_CALLERS",
                        ))
                        break
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
            ["git", "ls-files", "-z", "--", "*.py"], cwd=root,
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


def scan_repository(root: Path = ROOT) -> list[Finding]:
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
    return check_files(files)


def main() -> int:
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