#!/usr/bin/env python3
"""Markdown structure lint: runs markdownlint-cli2 on changed lines only.

Rules live in .markdownlint-cli2.jsonc (MD022 headings, MD031 fences, MD058
tables: blank lines around each). The CLI is pinned (MDLINT) and run through
``npx``; only violations on lines this change added or modified are reported, so
existing debt never blocks a commit.

Modes (cwd-independent, ``git -C`` the repo root)::

    python3 checks/markdown_structure.py             # staged blobs, changed lines only
    python3 checks/markdown_structure.py --base-ref origin/main  # REF...HEAD, changed lines only
    python3 checks/markdown_structure.py --all       # every tracked .md, whole file

Exit 0 clean, 1 violations, 2 a git/read/decode/usage error or node/npx missing
or the linter crashing (never read as clean).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MDLINT = "markdownlint-cli2@0.18.1"
CONFIG = ROOT / ".markdownlint-cli2.jsonc"
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_FINDING = re.compile(r"^(?P<path>.+?):(?P<line>\d+)(?::\d+)? (?P<rule>MD\d+)\S* (?P<msg>.*)$")


class LintError(Exception):
    pass


def changed_lines(diff: str) -> set[int]:
    """New-file line numbers touched by a ``git diff -U0`` for one file. A
    pure deletion at new position c marks c and c+1 (the lines it joined)."""
    touched: set[int] = set()
    for row in diff.splitlines():
        m = _HUNK.match(row)
        if not m:
            continue
        start = int(m.group(1))
        count = 1 if m.group(2) is None else int(m.group(2))
        if count == 0:
            touched.update((start, start + 1))
        else:
            touched.update(range(start, start + count))
    return touched


def _git(*args: str) -> bytes:
    # The user's git config (color.ui=always, diff.external) must not change
    # the parsed output; paths are read NUL-separated, so quotepath is moot.
    if args and args[0] == "diff":
        args = (args[0], "--no-color", "--no-ext-diff", *args[1:])
    try:
        proc = subprocess.run(["git", "-c", "color.ui=never", "-C", str(ROOT), *args],
                              capture_output=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise LintError(f"git {' '.join(args)} timed out after 60s") from exc
    if proc.returncode != 0:
        raise LintError(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout


def _paths(*args: str) -> list[str]:
    return [p for p in _git(*args, "-z", "--", "*.md").decode("utf-8").split("\0") if p]


def _decode(path: str, blob: bytes) -> str:
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LintError(f"{path}: not valid UTF-8: {exc}") from exc


def collect(mode: str, base_ref: str | None) -> list[tuple[str, str, set[int] | None]]:
    """(path, text, touched-lines-or-None) for every file in scope."""
    out: list[tuple[str, str, set[int] | None]] = []
    if mode == "all":
        for path in _paths("ls-files"):
            try:
                blob = (ROOT / path).read_bytes()
            except OSError as exc:
                raise LintError(f"{path}: unreadable: {exc}") from exc
            out.append((path, _decode(path, blob), None))
        return out
    if mode == "base":
        span, ref = [f"{base_ref}...HEAD"], "HEAD"
        listing = _paths("diff", "--no-renames", "--name-only", "--diff-filter=AM", *span)
    else:
        span, ref = ["--cached"], ""
        listing = _paths("diff", "--cached", "--no-renames", "--name-only", "--diff-filter=AM")
    for path in listing:
        text = _decode(path, _git("show", f"{ref}:{path}"))
        diff = _git("diff", "--no-renames", "-U0", *span, "--", path).decode("utf-8", "replace")
        out.append((path, text, changed_lines(diff)))
    return out


def run_linter(files: list[tuple[str, str, set[int] | None]]) -> list[tuple[str, int, str, str]]:
    """Lint the given (path, text, touched) files; return (path, line, rule, msg)
    for findings on touched lines (all lines when touched is None)."""
    if not files:
        return []
    npx = shutil.which("npx")
    if npx is None:
        raise LintError("node/npx not found; install Node.js (e.g. `apt install nodejs npm`) "
                        f"so `npx --yes {MDLINT}` can run")
    touched_by = {path: touched for path, _, touched in files}
    with tempfile.TemporaryDirectory() as tmp:
        for path, text, _ in files:
            target = Path(tmp) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        try:
            proc = subprocess.run([npx, "--yes", MDLINT, "--config", str(CONFIG), *touched_by],
                                  cwd=tmp, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LintError(f"{MDLINT} could not run: {exc}") from exc
    rows = (proc.stdout + "\n" + proc.stderr).splitlines()
    matches = [m for m in map(_FINDING.match, rows) if m]
    if proc.returncode not in (0, 1) or (proc.returncode == 1 and not matches):
        raise LintError(f"{MDLINT} crashed (exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()[-300:]}")
    return [(m["path"], int(m["line"]), m["rule"], m["msg"]) for m in matches
            if touched_by.get(m["path"], set()) is None or int(m["line"]) in touched_by.get(m["path"], set())]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="scan every tracked .md in full")
    ap.add_argument("--base-ref", help="check lines changed in REF...HEAD")
    ap.add_argument("--quiet", action="store_true", help="print nothing when clean")
    args = ap.parse_args(argv)
    if args.base_ref is not None and not args.base_ref.strip():
        print("markdown_structure: --base-ref needs a non-empty ref", file=sys.stderr)
        return 2
    mode = "all" if args.all else "base" if args.base_ref else "staged"
    try:
        files = collect(mode, args.base_ref)
        found = run_linter(files)
    except (LintError, UnicodeDecodeError, OSError) as exc:
        print(f"markdown_structure: {exc}", file=sys.stderr)
        return 2
    for path, line, rule, msg in found:
        print(f"{path}:{line}: {rule} {msg}")
    if found:
        print(f"markdown_structure: {len(found)} violation(s); add the missing blank line", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"markdown_structure: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())