#!/usr/bin/env python3
"""Pinned v2 lint against staged blobs (hook) or the complete working tree."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.repo_hygiene import read_staged_blob, staged_paths


def pinned_version(requirements: str) -> str:
    versions = [line.split("==", 1)[1] for line in requirements.splitlines()
                if line.startswith("ruff==")]
    if len(versions) != 1:
        raise ValueError("the environment lock must pin exactly one ruff version")
    return versions[0]


def sources(root: Path, *, worktree: bool) -> dict[str, bytes]:
    if worktree:
        return {path.relative_to(root).as_posix(): path.read_bytes()
                for path in (root / "engine/v2").rglob("*.py")}
    return {rel: read_staged_blob(root, rel) for rel in staged_paths(root)
            if rel.startswith("engine/v2/") and rel.endswith(".py")}


def check(files: dict[str, bytes], requirements: str, configuration: bytes) -> dict:
    try:
        wanted = pinned_version(requirements)
        installed = importlib.metadata.version("ruff")
    except (ValueError, importlib.metadata.PackageNotFoundError):
        return {"ok": False, "code": "LINTER_UNAVAILABLE", "findings": []}
    if wanted != installed:
        return {"ok": False, "code": "LINTER_VERSION_DRIFT",
                "expected": wanted, "installed": installed, "findings": []}
    with tempfile.TemporaryDirectory(prefix="phase1-lint-") as scratch:
        directory = Path(scratch)
        config = directory / "ruff.toml"
        config.write_bytes(configuration)
        for rel, blob in files.items():
            path = directory / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        if not files:
            return {"ok": True, "version": installed, "files": 0, "findings": []}
        run = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--no-cache", "--config", str(config),
             "--output-format", "json", *(str(directory / rel) for rel in sorted(files))],
            capture_output=True, text=True, check=False)
        try:
            findings = json.loads(run.stdout)
            for finding in findings:
                finding["filename"] = str(Path(finding["filename"]).relative_to(directory))
        except (ValueError, TypeError):
            return {"ok": False, "code": "LINTER_FAILED", "findings": []}
    return {"ok": run.returncode == 0, "version": installed, "files": len(files),
            "findings": findings}


def run(root: Path = ROOT, *, worktree: bool = False) -> dict:
    reader = (lambda rel: (root / rel).read_bytes()) if worktree else (
        lambda rel: read_staged_blob(root, rel))
    try:
        return check(sources(root, worktree=worktree), reader("requirements.txt").decode(),
                     reader("ruff.toml"))
    except (OSError, UnicodeError, subprocess.CalledProcessError):
        return {"ok": False, "code": "LINTER_CONFIGURATION_MISSING", "findings": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.repo_root, worktree=args.worktree)
    if args.json:
        print(json.dumps(result, indent=2))
    elif not result["ok"] or not args.quiet:
        print("v2 lint:", "OK" if result["ok"] else "FAILED", result.get("code", ""))
        for finding in result["findings"]:
            print(f"  {finding['filename']}:{finding['location']['row']} "
                  f"{finding['code']} {finding['message']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
