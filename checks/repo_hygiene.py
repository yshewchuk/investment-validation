#!/usr/bin/env python3
"""Repository hygiene gate — the last line of defence before a push.

Three independent checks, each of which alone blocks a commit:

1. **Secret values.** Reads ``.env`` at runtime and searches every candidate
   blob for the *current* secret values (plus base64 / URL-encoded forms).
   This is the strongest check available: it catches a key pasted into any
   file type, under any name, regardless of what ``.gitignore`` says.
2. **Data payloads.** Blocks data-ish extensions (csv, parquet, gz, jsonl,
   sqlite, pickles, images, ...) and any file larger than ``MAX_BYTES``.
3. **Forbidden paths.** Blocks anything under the data / ledger / results /
   reports / cache trees and the grandfathered research directories — defence
   in depth behind the allowlist ``.gitignore``.

Stdlib only, and it never imports ``engine``: it has to run inside a bare
fresh clone where nothing else is installed.

Usage::

    python3 checks/repo_hygiene.py            # staged changes (pre-commit)
    python3 checks/repo_hygiene.py --all      # every tracked file
    python3 checks/repo_hygiene.py --paths a.py b.py

Exit code 0 = clean, 1 = blocked. There is no ``--no-verify`` habit here: if
the hook is wrong, fix the hook.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

MAX_BYTES = 1_000_000  # 1 MB

BLOCKED_SUFFIXES = frozenset(
    {
        ".csv", ".tsv", ".parquet", ".jsonl", ".ndjson", ".gz", ".bz2", ".xz",
        ".zip", ".tar", ".7z", ".sqlite", ".sqlite3", ".db", ".pkl", ".pickle",
        ".joblib", ".h5", ".hdf5", ".npy", ".npz", ".feather", ".arrow",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".pdf", ".log",
        ".pem", ".key", ".p12", ".pfx", ".crt",
    }
)

# Path prefixes that must never be committed to either remote. Matched against
# the repo-relative POSIX path.
BLOCKED_PREFIXES = (
    "data/",
    "ledger/",
    "reports/",
    "polygon_cache/",
    "earnings_predictions/",
    "bt/",
    "config/thesis/",
    ".venv/",
)

# Path *segments* that must never appear anywhere in a committed path.
BLOCKED_SEGMENTS = ("results", "figures", "cache", "__pycache__", ".pytest_cache")

# Filenames that are never committed regardless of extension.
BLOCKED_NAMES = frozenset({".env"})

# Generic credential shapes, as a backstop for secrets that are not (yet) in
# .env — e.g. a token pasted from somewhere else entirely.
CREDENTIAL_PATTERNS = (
    ("github personal access token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{16,}")),
    ("github fine-grained token", re.compile(rb"github_pat_[A-Za-z0-9_]{20,}")),
    ("aws access key id", re.compile(rb"AKIA[0-9A-Z]{16}")),
    ("openai-style key", re.compile(rb"sk-[A-Za-z0-9]{32,}")),
    ("private key block", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)

# A secret value shorter than this is too generic to search for without
# drowning the check in false positives.
MIN_SECRET_LEN = 8

# .env values that are structurally non-secret (names, flags, booleans).
NON_SECRET_KEYS = frozenset({"OQUANTS_COOKIE_NAME"})


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class Violation:
    path: str
    rule: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"  [{self.rule}] {self.path}\n      {self.detail}"


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    checked: int = 0
    secrets_loaded: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, path: str, rule: str, detail: str) -> None:
        self.violations.append(Violation(path, rule, detail))


# --------------------------------------------------------------------------
# secret loading
# --------------------------------------------------------------------------


def parse_env(text: str) -> dict[str, str]:
    """Parse a shell-style ``.env`` body into a mapping.

    Handles ``export K=V``, ``K=V``, quoted values, comments and blank lines.
    Deliberately tolerant: a line it cannot parse is skipped rather than
    raising, because a hygiene check must never fail open on a formatting
    surprise in an unrelated line.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def secret_needles(env: dict[str, str]) -> dict[bytes, str]:
    """Build the set of byte patterns to search for, keyed to their env name.

    Each secret contributes its literal value plus base64 and percent-encoded
    forms, so a key that has been through an encoder on its way into a file is
    still caught.
    """
    needles: dict[bytes, str] = {}
    for key, val in env.items():
        if key in NON_SECRET_KEYS or len(val) < MIN_SECRET_LEN:
            continue
        raw = val.encode()
        variants = {
            raw: key,
            base64.b64encode(raw): f"{key} (base64)",
            urllib.parse.quote(val, safe="").encode(): f"{key} (url-encoded)",
        }
        for needle, label in variants.items():
            if len(needle) >= MIN_SECRET_LEN:
                needles.setdefault(needle, label)
    return needles


def load_secrets(env_path: Path) -> dict[bytes, str]:
    if not env_path.exists():
        return {}
    return secret_needles(parse_env(env_path.read_text(errors="replace")))


# --------------------------------------------------------------------------
# per-file checks
# --------------------------------------------------------------------------


def check_path_policy(rel: str, report: Report) -> None:
    posix = rel.replace(os.sep, "/")
    name = posix.rsplit("/", 1)[-1]
    suffix = os.path.splitext(name)[1].lower()

    if name in BLOCKED_NAMES:
        report.add(rel, "forbidden-name", f"{name!r} is never committed")
    if suffix in BLOCKED_SUFFIXES:
        report.add(rel, "data-extension", f"{suffix!r} is a data/artifact extension")
    for prefix in BLOCKED_PREFIXES:
        if posix.startswith(prefix):
            report.add(rel, "forbidden-path", f"lives under {prefix!r}")
            break
    parts = posix.split("/")[:-1]
    for seg in BLOCKED_SEGMENTS:
        if seg in parts:
            report.add(rel, "forbidden-path", f"contains a {seg!r} directory")
            break


def check_blob(rel: str, blob: bytes, needles: dict[bytes, str], report: Report,
               max_bytes: int = MAX_BYTES) -> None:
    """Size + secret scan on one blob.

    ``max_bytes`` is a parameter, not a constant, for exactly one caller: the
    private mirror, whose job is to preserve multi-megabyte evidence artifacts
    (transaction logs, the ledger) that this repo must never carry. The secret
    scan below is identical either way — a bigger allowance never means a
    laxer search.
    """
    if len(blob) > max_bytes:
        report.add(
            rel,
            "oversize",
            f"{len(blob):,} bytes exceeds the {max_bytes:,}-byte limit",
        )
    for needle, label in needles.items():
        if needle in blob:
            # The value itself is NEVER printed — only which variable leaked.
            report.add(rel, "secret", f"contains the current value of ${label}")
    for label, pattern in CREDENTIAL_PATTERNS:
        if pattern.search(blob):
            report.add(rel, "credential-shape", f"matches a {label} pattern")


def check_files(
    files: dict[str, bytes],
    needles: dict[bytes, str] | None = None,
) -> Report:
    """Check an in-memory ``{repo_relative_path: content}`` mapping.

    This is the pure core the CLI and the tests both drive.
    """
    report = Report(secrets_loaded=len(needles or {}))
    for rel in sorted(files):
        report.checked += 1
        check_path_policy(rel, report)
        check_blob(rel, files[rel], needles or {}, report)
    return report


# --------------------------------------------------------------------------
# git plumbing
# --------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def staged_paths(root: Path) -> list[str]:
    out = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [p for p in out.split("\0") if p]


def tracked_paths(root: Path) -> list[str]:
    out = _git(root, "ls-files", "-z")
    return [p for p in out.split("\0") if p]


def read_staged_blob(root: Path, rel: str) -> bytes:
    """Read the *staged* content, which is what would actually be committed."""
    proc = subprocess.run(
        ["git", "-C", str(root), "show", f":{rel}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return b""
    return proc.stdout


def read_worktree_blob(root: Path, rel: str) -> bytes:
    path = root / rel
    try:
        return path.read_bytes()
    except OSError:
        return b""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="check every tracked file")
    mode.add_argument("--paths", nargs="+", help="check these paths explicitly")
    ap.add_argument("--env", default=None, help="path to .env (default: <root>/.env)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    env_path = Path(args.env) if args.env else root / ".env"
    needles = load_secrets(env_path)

    if args.paths:
        rels = [str(Path(p).resolve().relative_to(root)) for p in args.paths]
        reader = read_worktree_blob
    elif args.all:
        rels = tracked_paths(root)
        reader = read_worktree_blob
    else:
        rels = staged_paths(root)
        reader = read_staged_blob

    files = {rel: reader(root, rel) for rel in rels}
    report = check_files(files, needles)

    if not args.quiet:
        scope = "--paths" if args.paths else ("all tracked" if args.all else "staged")
        print(
            f"repo hygiene: {report.checked} file(s) [{scope}], "
            f"{report.secrets_loaded} secret pattern(s) loaded from {env_path.name}"
        )
        if not needles:
            print(
                f"  WARNING: no secrets loaded from {env_path} — "
                "the value-grep check is inactive"
            )
    if report.ok:
        if not args.quiet:
            print("HYGIENE OK")
        return 0
    print(f"\nHYGIENE FAILED — {len(report.violations)} violation(s):", file=sys.stderr)
    for v in report.violations:
        print(str(v), file=sys.stderr)
    print(
        "\nCommit blocked. Fix the content or the .gitignore — do not use "
        "--no-verify.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
