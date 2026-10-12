#!/usr/bin/env python3
"""Markdown structure lint for defects that make a page render wrongly.

Rules (no style opinions):

* an ATX heading needs a blank line before it (unless it is line 1) and after;
* a table needs a blank line before it and after it (unless it ends the file),
  so a following paragraph is not absorbed as a table row;
* an unindented fenced code block needs a blank line before and after, and
  every fence must be closed.

Content inside fences is never read as a heading or table. Lists are not
checked at all (no list rule exists). Standard library only.

Modes (cwd-independent, ``git -C`` the repo root)::

    python3 checks/markdown_structure.py             # staged blobs, changed lines only
    python3 checks/markdown_structure.py --base-ref origin/main  # REF...HEAD, changed lines only
    python3 checks/markdown_structure.py --all       # every tracked .md, whole file

Exit 0 clean, 1 violations, 2 a git/read/decode error (never read as clean).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_FENCE = re.compile(r"^( *)(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^ {0,3}#{1,6}(\s|$)")
_DELIM = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class LintError(Exception):
    """A git call, file read or decode failed; the caller exits 2."""


@dataclass(frozen=True)
class Violation:
    line: int  # 1-based line the rule is reported on
    near: int  # 1-based neighbouring line whose change can cause it
    rule: str


def _cells(row: str) -> int:
    return len(row.strip().strip("|").split("|"))


def _is_table_row(row: str, header: str) -> bool:
    """A body row of a table whose header starts with a pipe must start with
    one too; for a pipeless-edge header it must have the header's column count.
    A paragraph that merely contains a pipe is therefore not absorbed as a row."""
    stripped = row.strip()
    if not stripped or "|" not in stripped:
        return False
    if header.strip().startswith("|"):
        return stripped.startswith("|")
    return _cells(stripped) == _cells(header)


def lint_text(text: str) -> list[Violation]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    n = len(lines)
    found: list[Violation] = []

    def blank(i: int) -> bool:
        return i < 0 or i >= n or not lines[i].strip()

    fence = None  # (char, length, indent, opening 1-based line)
    i = 0
    while i < n:
        s = lines[i]
        m = _FENCE.match(s)
        if m and m.group(2)[0] == "`" and "`" in m.group(3):
            m = None  # an inline code span such as ```json```, not a fence
        if fence:
            if (m and m.group(2)[0] == fence[0] and len(m.group(2)) >= fence[1]
                    and not m.group(3).strip()):
                if fence[2] == 0 and not blank(i + 1):
                    found.append(Violation(i + 1, i + 2, "fence-after"))
                fence = None
            i += 1
            continue
        if m:
            fence = (m.group(2)[0], len(m.group(2)), len(m.group(1)), i + 1)
            if not m.group(1) and not blank(i - 1):
                found.append(Violation(i + 1, i, "fence-before"))
            i += 1
            continue
        if _HEADING.match(s):
            if not blank(i - 1):
                found.append(Violation(i + 1, i, "heading-before"))
            if not blank(i + 1):
                found.append(Violation(i + 1, i + 2, "heading-after"))
            i += 1
            continue
        if ("|" in s and not _DELIM.match(s) and i + 1 < n
                and "|" in lines[i + 1] and _DELIM.match(lines[i + 1])):
            if not blank(i - 1):
                found.append(Violation(i + 1, i, "table-before"))
            j = i + 2
            while j < n and _is_table_row(lines[j], s):
                j += 1
            if j < n and lines[j].strip():
                found.append(Violation(j + 1, j, "table-after"))
            i = j
            continue
        i += 1
    if fence:
        found.append(Violation(fence[3], fence[3], "fence-unclosed"))
    return found


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


def scoped(violations: list[Violation], touched: set[int] | None) -> list[Violation]:
    """``touched`` None means the whole file is in scope. An unclosed fence is a
    file-level error: the opening line is usually untouched when the closing
    fence is deleted or text is added after it, so it is always reported for a
    file in the changed set."""
    if touched is None:
        return violations
    return [v for v in violations
            if v.rule == "fence-unclosed" or v.line in touched or v.near in touched]


def _git(*args: str) -> bytes:
    try:
        proc = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, timeout=60)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="scan every tracked .md in full")
    ap.add_argument("--base-ref", help="check lines changed in REF...HEAD")
    ap.add_argument("--quiet", action="store_true", help="print nothing when clean")
    args = ap.parse_args(argv)
    mode = "all" if args.all else "base" if args.base_ref else "staged"
    try:
        files = collect(mode, args.base_ref)
    except (LintError, UnicodeDecodeError, OSError) as exc:
        print(f"markdown_structure: {exc}", file=sys.stderr)
        return 2
    bad = 0
    for path, text, touched in files:
        for v in scoped(lint_text(text), touched):
            print(f"{path}:{v.line}: {v.rule}")
            bad += 1
    if bad:
        print(f"markdown_structure: {bad} violation(s); add the missing blank line / close the fence",
              file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"markdown_structure: {len(files)} file(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
