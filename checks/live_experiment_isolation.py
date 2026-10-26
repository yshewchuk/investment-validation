#!/usr/bin/env python3
"""Fail closed when production and experiment source paths cross isolation.

Run ``python3 checks/live_experiment_isolation.py``. Import, root, syntax,
encoding, git-listing and source-read failures are loud CI refusals; no partial
scan is clean.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = ("engine/v2/models/", "engine/v2/ops/retrain_", "tools/phase5_")
EXPERIMENT_PATHS = ("engine/v2/research/", "experiments/")
PRODUCTION_FORBIDDEN_IMPORTS = (
    "engine.v2.foundation.experiment_holdouts",
    "engine.v2.ops.final_holdout",
    "engine.v2.research",
)
EXPERIMENT_FORBIDDEN_IMPORTS = (
    "engine.v2.ledger",
    "engine.v2.models.deployment",
    "engine.v2.models.release_layout",
    "engine.v2.models.release_objects",
    "engine.v2.models.releases",
    "engine.v2.models.inventory",
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
            if (isinstance(imported, ast.ImportFrom) and imported.module and
                    imported.module.rsplit(".", 1)[-1] == "deployment"):
                for alias in imported.names:
                    if alias.name in ROOT_CALLERS:
                        root_aliases[alias.asname or alias.name] = alias.name
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
            if experiment and isinstance(node, (ast.Name, ast.Constant)):
                value = node.id if isinstance(node, ast.Name) else node.value
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


def scan_repository(root: Path = ROOT) -> list[Finding]:
    """Read every tracked Python source; refuse if any part of the scan fails."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.py"], cwd=root,
            capture_output=True, check=True,
        )
        paths = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
        files = {path: (root / path).read_bytes() for path in paths}
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError) as exc:
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