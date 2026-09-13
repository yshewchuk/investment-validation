"""Declared transitive source closures and explicit change plans."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import platform
import shutil
from pathlib import Path

from engine.v2.foundation import ensure_directory, safe_relative_path
from engine.v2.ops.errors import fail


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_closure(root, entries):
    """Resolve local static imports; callers also declare runtime/configuration edges."""
    pending, found = list(entries), {}
    while pending:
        rel = pending.pop()
        safe_relative_path(rel)
        if rel in found:
            continue
        path = root / rel
        if not path.is_file():
            raise fail("INPUT_CHANGED", "declared implementation file is missing")
        found[rel] = file_hash(path)
        if path.suffix == ".py":
            pending.extend(_imports(root, rel, path.read_text()))
    return dict(sorted(found.items()))


def worker_source_manifest(root):
    """Include package initializers imported before the fixed worker module."""
    return source_closure(root, [
        "engine/__init__.py", "engine/v2/__init__.py",
        "engine/v2/ops/__init__.py", "engine/v2/ops/worker.py",
    ])


def _imports(root, rel, source):
    package = rel.removesuffix(".py").replace("/", ".").split(".")[:-1]
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[:len(package) - node.level + 1] if node.level else []
            base = ".".join(prefix + ([node.module] if node.module else []))
            names.extend([base, *(base + "." + alias.name for alias in node.names)])
    paths = set()
    for name in names:
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            stem = "/".join(parts[:index])
            for candidate in (stem + ".py", stem + "/__init__.py"):
                if (root / candidate).is_file():
                    paths.add(candidate)
    return sorted(paths)


def environment_identity(thread_count=1):
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), machine=platform.machine(),
                libraries=versions, numerical_threads=thread_count)


def snapshot_code(root, destination, manifest):
    """Private code copy; a worker never imports from an editable checkout."""
    ensure_directory(destination)
    for rel, expected in manifest.items():
        safe_relative_path(rel)
        source, target = root / rel, destination / rel
        if file_hash(source) != expected:
            raise fail("INPUT_CHANGED", "source changed during code pinning")
        ensure_directory(target.parent)
        if not target.exists():
            shutil.copyfile(source, target)
            target.chmod(0o444)
        if file_hash(target) != expected:
            raise fail("INTEGRITY_FAILED", "pinned source content differs")


def rerun_plan(stages, previous):
    """Compare a DAG in topological order and invalidate only its changed closure."""
    decisions = {}
    pending = dict(stages)
    while pending:
        ready = [name for name, stage in pending.items()
                 if all(parent in decisions for parent in stage["dependencies"])]
        if not ready:
            raise fail("INVALID_REQUEST", "stage dependency graph is cyclic or incomplete")
        for name in ready:
            stage = pending.pop(name)
            changed = [key for key in ("inputs", "implementation", "parameters", "environment", "schema")
                       if previous.get(name, {}).get(key) != stage.get(key)]
            parents = [p for p in stage["dependencies"] if decisions[p]["action"] == "rerun"]
            decisions[name] = {"action": "rerun" if changed or parents else "reuse",
                               "reasons": changed + ["dependency:" + p for p in parents],
                               "resource_class": stage.get("resource_class")}
    return decisions
