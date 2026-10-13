#!/usr/bin/env python3
"""Markdown structure lint: blank lines around blocks, closed fences, no absorbed tables.

Only ``table-after`` (a paragraph absorbed into a table) and ``fence-unclosed``
(the rest of the file swallowed) prevent a CommonMark/GFM rendering defect. The
other blank-line rules (around headings and fences, and before a table) are house
style: they keep diffs and other renderers safe, and CommonMark renders those
cases fine.

Rules (no other style opinions):

* an ATX heading needs a blank line before it (unless it is line 1) and after
  (unless it is the last line);
* a table needs a blank line before it (unless it starts the file) and after it
  (unless it ends the file), so a following paragraph is not absorbed as a row;
  a body row starts with a pipe when the header does, else it has the header's
  column count. Known limit: under a pipeless-edge header, a line with the
  header's cell count (``use a | b here`` under ``a | b``) is a row by GFM
  rules, indistinguishable from an intended one, and is not flagged;
* an unindented fenced code block needs a blank line before and after.

Unindented, indented and list-marker fences (``- ```python``) are tracked, and
every fence must be closed: an unclosed fence is reported for any file in the
changed set, because it breaks rendering of the rest of the file. Only
unindented fences get the blank-line rule. Blockquoted fences (``> ```)
are not checked. Content inside a fence is never read as a heading or table.

There is no list rule, and list content is skipped for the heading and table
rules: only column-0 headings and tables are read, so an indented line or one
starting with a list marker (``- ``, ``* ``, ``+ ``, ``1. ``) is never a heading
or a table header. Standard library only.

Modes (cwd-independent, ``git -C`` the repo root)::

    python3 checks/markdown_structure.py             # staged blobs, changed lines only
    python3 checks/markdown_structure.py --base-ref origin/main  # REF...HEAD, changed lines only
    python3 checks/markdown_structure.py --all       # every tracked .md, whole file

A fence may open on a list-marker line (``- ```python``); it counts as indented.
An empty ``--base-ref ""`` is a usage error (exit 2), never staged mode.

Exit 0 clean, 1 violations, 2 a git/read/decode/usage error (never read as clean).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_FENCE = re.compile(r"^( *(?:(?:[-*+]|\d+[.)]) +)?)(`{3,}|~{3,})(.*)$")  # optional list-marker prefix
# A closer has no list-marker prefix and nothing after the fence run. Its indent
# is checked against the opener's in lint_text: at most 3 more than the opener's
# (CommonMark: 0-3 spaces relative to the container), and, for a fence inside a
# list item, not less than the opener's own indent -- a less-indented line leaves
# the list item, so it does not close the fence and the fence is reported unclosed.
_CLOSER = re.compile(r"^( *)(`{3,}|~{3,})[ \t]*$")
_HEADING = re.compile(r"^#{1,6}(\s|$)")
_LIST_ITEM = re.compile(r"^([-*+]|\d+[.)])\s")
_ITEM = re.compile(r"^( *)([-*+]|\d+[.)])( +|$)")  # any-indent list item line
_LEAD = re.compile(r"^[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)?")  # indent + optional list marker
# A line that starts a non-HTML block at the top level (indent 0-3): ATX heading,
# fence, list item, blockquote or thematic break.
_BLOCK_START = re.compile(
    r"^ {0,3}(?:#{1,6}(?:\s|$)|`{3,}|~{3,}|(?:[-*+]|\d{1,9}[.)])(?:\s|$)|>"
    r"|([-*_])(?:[ \t]*\1){2,}[ \t]*$)")

# HTML blocks, CommonMark spec 0.31.2 section 4.6. Start conditions 1-7 and the
# matching end conditions: types 1-5 end at the start-tag/close-tag line itself,
# types 6 and 7 end at a blank line. Type 7 cannot interrupt a paragraph.
_HTML_BLOCK_TAGS = (  # start condition 6, copied from the spec
    "address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup"
    "|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame"
    "|frameset|h1|h2|h3|h4|h5|h6|head|header|hr|html|iframe|legend|li|link|main|menu"
    "|menuitem|nav|noframes|ol|optgroup|option|p|param|search|section|summary|table"
    "|tbody|td|tfoot|th|thead|title|tr|track|ul")
_OPEN_TAG = (r"""<[A-Za-z][A-Za-z0-9-]*(?:[ \t]+[A-Za-z_:][A-Za-z0-9_.:-]*"""
             r"""(?:[ \t]*=[ \t]*(?:[^ \t"'=<>`]+|'[^']*'|"[^"]*"))?)*[ \t]*/?>""")
_CLOSE_TAG = r"</[A-Za-z][A-Za-z0-9-]*[ \t]*>"
_HTML_START = (
    (1, re.compile(r"^ *<(?:script|pre|style|textarea)(?:[ \t>]|$)", re.I)),
    (2, re.compile(r"^ *<!--")),
    (3, re.compile(r"^ *<\?")),
    (4, re.compile(r"^ *<![A-Za-z]")),
    (5, re.compile(r"^ *<!\[CDATA\[")),
    (6, re.compile(r"^ *</?(?:" + _HTML_BLOCK_TAGS + r")(?:[ \t]|/?>|$)", re.I)),
    (7, re.compile(r"^ *(?:" + _OPEN_TAG + "|" + _CLOSE_TAG + r")[ \t]*$")),
)
_THEMATIC = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_HTML_END = {
    1: re.compile(r"</(?:script|pre|style|textarea)>", re.I),
    2: re.compile(r"-->"),
    3: re.compile(r"\?>"),
    4: re.compile(r">"),
    5: re.compile(r"\]\]>"),
}


def _html_start(line: str, in_paragraph: bool, container: int = 0) -> int:
    """The HTML block start condition (1-7) ``line`` matches, else 0. Type 7
    cannot interrupt a paragraph, so it is skipped when ``in_paragraph``. The line
    must sit less than 4 columns beyond ``container`` (the enclosing list item's
    content column, 0 at top level); 4 or more is indented code."""
    if len(line) - len(line.lstrip(" ")) - container >= 4:
        return 0
    for kind, rx in _HTML_START:
        if kind == 7 and in_paragraph:
            continue
        if rx.match(line):
            return kind
    return 0


def _starts_block(line: str) -> bool:
    """Whether ``line`` starts a block other than a paragraph or table row (it can
    interrupt a paragraph). Used for both the table end and the table-header
    check, so they cannot disagree."""
    return bool(_BLOCK_START.match(line)) or _html_start(line, in_paragraph=True) > 0
_DELIM = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class LintError(Exception):
    """A git call, file read or decode failed; the caller exits 2."""


@dataclass(frozen=True)
class Violation:
    line: int  # 1-based line the rule is reported on
    near: int  # 1-based neighbouring line whose change can cause it
    rule: str
    # Table violations only: the whole table block (header .. last row, or the
    # absorbing line) is in scope, because a change to ANY of its lines (e.g. a
    # delimiter row turning text into a table) can create the violation.
    lo: int = 0
    hi: int = 0


_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _cells(row: str) -> int:
    """Cell count of a table row. Unescaped pipes separate cells (also inside code
    spans, per GFM); one leading and one trailing unescaped pipe are edges."""
    body = row.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return len(_UNESCAPED_PIPE.split(body))


def _is_table_row(row: str, header: str) -> bool:
    """Whether ``row`` continues the table under ``header``, as this lint models
    it. GFM ends a table at a blank line or at the start of another block-level
    structure (heading, fence, blockquote, list, HTML block); any other non-blank
    line is absorbed as a row, however it is shaped. The lint is stricter: it
    ends the table at the first line not shaped like a row (a leading pipe when
    the header has one, else the header's column count), so that ``table-after``
    can flag a following line that is, or may be mistaken for, an absorbed
    paragraph. A line after a table that starts another block is flagged as the
    house blank-line rule, not as an absorbed row."""
    stripped = row.strip()
    if not stripped or _starts_block(row) or not _UNESCAPED_PIPE.search(stripped):
        return False
    if header.strip().startswith("|"):
        return stripped.startswith("|")
    return _cells(stripped) == _cells(header)


def _expand(line: str) -> str:
    """Expand tabs in the leading indent (and the gap after a list marker) to the
    next multiple of 4 columns, so every indent below is measured in columns."""
    end = _LEAD.match(line).end()
    out, col = [], 0
    for ch in line[:end]:
        if ch == "\t":
            out.append(" " * (4 - col % 4))
            col += 4 - col % 4
        else:
            out.append(ch)
            col += 1
    return "".join(out) + line[end:]


def lint_text(text: str) -> list[Violation]:
    # CRLF files: every line regex is anchored with $, so drop the \r of each
    # line ending first. Line numbers still match git's, which counts \n only.
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [_expand(x) for x in lines]
    n = len(lines)
    found: list[Violation] = []

    def blank(i: int) -> bool:
        return i < 0 or i >= n or not lines[i].strip()

    fence = None  # (char, length, indent, opening 1-based line, container indent)
    items: list[int] = []  # content columns of the open list items, outermost first
    html = 0  # start condition (1-7) of the HTML block being skipped, else 0
    para = False  # the last line processed is paragraph text (what type 7 cannot interrupt)
    i = 0
    while i < n:
        s = lines[i]
        prev_para, para = para, bool(s.strip()) and not _THEMATIC.match(s)
        m = _FENCE.match(s)
        if m and m.group(2)[0] == "`" and "`" in m.group(3):
            m = None  # an inline code span such as ```json```, not a fence
        if m and m.group(1).strip() and len(m.group(1)) - len(m.group(1).rstrip(" ")) >= 5:
            m = None  # marker + 5 or more spaces: the rest is indented code, not a fence
        if fence:
            c = _CLOSER.match(s)
            if (c and c.group(2)[0] == fence[0] and len(c.group(2)) >= fence[1]
                    and fence[4] <= len(c.group(1)) <= fence[4] + 3):
                if fence[2] == 0 and not blank(i + 1):
                    found.append(Violation(i + 1, i + 2, "fence-after"))
                fence = None
                para = False  # a closed fence is not paragraph text
            i += 1
            continue
        if html:
            if html >= 6 and not s.strip():
                html = 0  # a blank line ends HTML blocks of type 6 and 7
            else:
                # Inside an HTML block nothing is Markdown: no fence, heading or table.
                if html < 6 and _HTML_END[html].search(s):
                    html = 0
                    para = False  # an ended HTML block is not paragraph text
                i += 1
                continue
        else:
            ws = len(s) - len(s.lstrip(" "))
            kind = _html_start(s, prev_para, max((x for x in items if x <= ws), default=0))
            if kind:
                if not s[:1].isspace():
                    items.clear()
                html = 0 if kind < 6 and _HTML_END[kind].search(s) else kind
                if not html:
                    para = False
                i += 1
                continue
        indent = len(m.group(1)) if m else 0
        lead = indent - len(m.group(1).lstrip(" ")) if m else 0
        base = max((x for x in items if x <= lead), default=0)
        if m and lead - base < 4:
            # (else: 4+ columns beyond the enclosing container is indented code,
            # not a fence -- also for a marker line like "    - ```")
            # The container's content column: this line's own list marker, else the
            # innermost open list item whose content starts at or left of the fence,
            # else 0 (top level). A closer may sit 0-3 spaces beyond that column.
            if m.group(1).strip():
                while items and items[-1] > lead:
                    items.pop()
                items.append(indent)
                container = indent
            else:
                container = base
            fence = (m.group(2)[0], len(m.group(2)), indent, i + 1, container)
            if not m.group(1) and not blank(i - 1):
                found.append(Violation(i + 1, i, "fence-before"))
            i += 1
            continue
        item = _ITEM.match(s)
        # A marker indented 4+ columns beyond its container is indented code or a
        # paragraph continuation, not a list item.
        if item and len(item.group(1)) - max(
                (x for x in items if x <= len(item.group(1))), default=0) < 4:
            while items and items[-1] > len(item.group(1)):
                items.pop()
            # Content column: after the marker and its 1-4 spaces; with no text or
            # 5+ spaces after the marker the rest is code, so the column is marker + 1.
            width = len(item.group(1)) + len(item.group(2))
            gap = len(item.group(3))
            items.append(width + gap if 1 <= gap <= 4 else width + 1)
        elif s.strip() and not s[:1].isspace():
            items.clear()
        if _HEADING.match(s):
            if not blank(i - 1):
                found.append(Violation(i + 1, i, "heading-before"))
            if not blank(i + 1):
                found.append(Violation(i + 1, i + 2, "heading-after"))
            para = False  # a heading is not paragraph text
            i += 1
            continue
        if ("|" in s and not _DELIM.match(s) and not s[:1].isspace()
                and not _starts_block(s) and i + 1 < n
                and "|" in lines[i + 1] and _DELIM.match(lines[i + 1])
                and _cells(s) == _cells(lines[i + 1])):
            j = i + 2
            while j < n and _is_table_row(lines[j], s):
                j += 1
            if not blank(i - 1):
                found.append(Violation(i + 1, i, "table-before", i + 1, j))
            if j < n and lines[j].strip():
                found.append(Violation(j + 1, j, "table-after", i + 1, j + 1))
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
            if v.rule == "fence-unclosed" or v.line in touched or v.near in touched
            or any(v.lo <= t <= v.hi for t in touched)]


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
