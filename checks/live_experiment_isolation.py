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

    for path, tree in trees.items():
        production = _scope(path, PRODUCTION_PATHS)
        experiment = _scope(path, EXPERIMENT_PATHS)
        denied = (PRODUCTION_FORBIDDEN_IMPORTS if production else ())
        denied += EXPERIMENT_FORBIDDEN_IMPORTS if experiment else ()
        root_aliases = {}
        for imported in ast.walk(tree):
            if isinstance(imported, ast.ImportFrom):
                targets = _imports(imported, path)
                if not targets:
                    continue
                base = targets[0]
                if (base == "engine.v2.models" or
                        base.rsplit(".", 1)[-1] == "deployment"):
                    for alias in imported.names:
                        if alias.name in ROOT_CALLERS:
                            root_aliases[alias.asname or alias.name] = alias.name
        changed = True
        while changed:
            changed = False
            for assigned in ast.walk(tree):
                if not isinstance(assigned, ast.Assign):
                    continue
                value = assigned.value
                if isinstance(value, ast.Attribute):
                    symbol = value.attr if value.attr in ROOT_CALLERS else None
                elif isinstance(value, ast.Name):
                    symbol = root_aliases.get(value.id)
                else:
                    symbol = None
                if symbol is None:
                    continue
                for target in assigned.targets:
                    if (isinstance(target, ast.Name) and
                            root_aliases.get(target.id) != symbol):
                        root_aliases[target.id] = symbol
                        changed = True
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for target in _imports(node, path):
                    key = (path, node.lineno, "forbidden-import")
                    if _under(target, denied) and key not in finding_keys:
                        findings.append(Finding(
                            path, node.lineno, "forbidden-import",
                            f"{target} crosses the D7 isolation boundary",
                        ))
                        finding_keys.add(key)
            if experiment and isinstance(node, (ast.Name, ast.Attribute, ast.Constant)):
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
            if isinstance(node, ast.Call) and not path.startswith("tests/"):
                function = node.func
                name = function.id if isinstance(function, ast.Name) else (
                    function.attr if isinstance(function, ast.Attribute) else "")
                name = root_aliases.get(name, name)
                allowed = ROOT_CALLERS.get(name)
                if allowed is not None and path not in allowed:
                    findings.append(Finding(
                        path, node.lineno, "unlisted-root-caller",
                        f"{name} caller must be reviewed and added to ROOT_CALLERS",
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
        files = {path: _read_source(root / path) for path in paths}
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