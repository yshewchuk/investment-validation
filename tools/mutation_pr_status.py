"""Classify a PR's gremlins mutation result as PR-owned (red) or infra (warning).

Inputs: the merged ``summary.json`` (+ ``results.jsonl``) written by
``gremlin_results.py merge``, the PR's ``git diff -U0`` text and, optionally,
main's last merged ``summary.json``. Output: a JSON verdict on stdout and
``state`` / ``description`` lines in ``$GITHUB_OUTPUT``. Only PR-owned problems
make ``state`` ``failure``; infra problems are ``::warning::`` lines. This tool
never exits nonzero for a classified result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DESC_MAX = 140
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_SIMPLE_ESC = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def _unquote(s: str) -> str:
    """Undo git's C-style path quoting (octal bytes and simple escapes)."""
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n in "01234567":
                j, digits = i + 1, ""
                while j < len(s) and len(digits) < 3 and s[j] in "01234567":
                    digits += s[j]
                    j += 1
                out.append(int(digits, 8) & 255)
                i = j
                continue
            if n in _SIMPLE_ESC:
                out.append(_SIMPLE_ESC[n])
                i += 2
                continue
        out += c.encode()
        i += 1
    return out.decode("utf-8", "replace")


def _head_path(line: str) -> str | None:
    """The repo path of a ``+++ b/<path>`` header, or None (``/dev/null``, other forms).

    Git appends a TAB after names containing spaces and double-quotes names with
    non-ASCII or control characters (unless ``core.quotepath=off``)."""
    rest = line[4:].rstrip("\t")
    if len(rest) >= 2 and rest.startswith('"') and rest.endswith('"'):
        rest = _unquote(rest[1:-1])
    return rest[2:] if rest.startswith("b/") else None


def parse_changed_lines(diff_text: str) -> dict[str, set[int]]:
    """Map each file in a ``git diff -U0`` to the head-side line numbers it added or changed."""
    changed: dict[str, set[int]] = {}
    current: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):  # +++ /dev/null: a deleted file has no head lines
            current = _head_path(line)
            if current is not None:
                changed.setdefault(current, set())
            continue
        m = _HUNK.match(line)
        if m and current is not None:
            start = int(m.group(1))
            count = 1 if m.group(2) is None else int(m.group(2))
            changed[current].update(range(start, start + count))
    return changed


def _is_test_failure_rc(rc) -> bool:
    return isinstance(rc, int) and not isinstance(rc, bool) and 0 < rc < 128


def baseline_failing(baseline) -> set[str]:
    """Modules whose run failed with a test-failure exit code in main's merged summary."""
    out: set[str] = set()
    if not isinstance(baseline, dict):
        return out
    modules = baseline.get("modules")
    if not isinstance(modules, dict):
        return out
    for name, mod in modules.items():
        if isinstance(mod, dict) and not mod.get("skipped_reason") \
                and _is_test_failure_rc(mod.get("run_exit_code")):
            out.add(name)
    return out


def classify_module(name: str, mod, rows: list[dict], changed: dict[str, set[int]],
                    failing_on_main: set[str]) -> tuple[str, str]:
    """Return (class, reason) with class in ok / pr_owned / infra."""
    # A survivor on a line this PR changed is PR-owned even if the run later timed
    # out or failed, and even if the summary has no entry for the module (a module
    # reported twice is dropped from ``modules``): the rows are the evidence.
    hits = sorted((r["file"], r["line"]) for r in rows
                  if r.get("module") == name and r.get("status") == "survived"
                  and r.get("line") in changed.get(r.get("file"), ()))
    if hits:
        shown = ", ".join(f"{f}:{n}" for f, n in hits[:5])
        return "pr_owned", f"{len(hits)} survivor(s) on changed lines: {shown}"
    if not isinstance(mod, dict):
        return "infra", "no report: job cancelled, runner or setup error"
    if mod.get("skipped_reason"):
        return "infra", f"skipped ({mod['skipped_reason']})"
    rc = mod.get("run_exit_code")
    if rc == -1:
        return "infra", "timeout or cancelled"
    if isinstance(rc, int) and not isinstance(rc, bool) and (rc < 0 or rc >= 128):
        return "infra", f"killed by signal (exit {rc})"
    counts = mod.get("counts_backend")
    if isinstance(counts, dict) and counts.get("error"):
        return "infra", "gremlin ERROR results (tool errors)"
    reasons = mod.get("failure_reasons")
    if isinstance(reasons, list) and any(str(r).startswith("RAW_") for r in reasons):
        return "infra", "invalid raw report"
    if _is_test_failure_rc(rc):
        if name in failing_on_main:
            return "infra", f"module run fails on main too (exit {rc})"
        return "pr_owned", f"module run failed (exit {rc})"
    return "ok", ""


def _truncate(text: str, keep_tail: str = "") -> str:
    """Fit DESC_MAX; ``keep_tail`` (the infra count) is never cut."""
    text = " ".join(text.split())
    if len(text) + len(keep_tail) <= DESC_MAX:
        return text + keep_tail
    return text[:DESC_MAX - 3 - len(keep_tail)] + "..." + keep_tail


def classify(merged, rows: list[dict], changed: dict[str, set[int]], baseline=None) -> dict:
    """Pure verdict: ``state`` is ``failure`` only when a module is PR-owned."""
    rows = [r for r in rows if _valid_row(r)]
    modules = merged.get("modules") if isinstance(merged, dict) else None
    modules = modules if isinstance(modules, dict) else {}
    expected = merged.get("expected_modules") if isinstance(merged, dict) else None
    expected = [m for m in expected if isinstance(m, str)] if isinstance(expected, list) else []
    names = sorted(set(expected) | {m for m in modules if isinstance(m, str)}
                   | {r["module"] for r in rows if isinstance(r, dict) and isinstance(r.get("module"), str)})
    failing_on_main = baseline_failing(baseline)
    result: dict[str, dict] = {}
    for name in names:
        cls, reason = classify_module(name, modules.get(name), rows, changed, failing_on_main)
        result[name] = {"class": cls, "reason": reason}
    owned = [n for n, v in result.items() if v["class"] == "pr_owned"]
    infra = [n for n, v in result.items() if v["class"] == "infra"]
    tail = ""
    if owned:
        desc = f"gremlins: PR-owned problem in {', '.join(owned)}"
        if infra:
            tail = f"; {len(infra)} infra warning(s)"
    elif infra:
        desc = f"gremlins: no PR-owned problems; {len(infra)} infra warning(s)"
    else:
        desc = f"gremlins: no PR-owned problems in {len(names)} module(s)"
    return {"state": "failure" if owned else "success", "description": _truncate(desc, tail),
            "pr_owned": owned, "infra": infra, "modules": result}


def _load_json(path: str | None):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _valid_summary(s) -> bool:
    """A merged/baseline summary: a mapping whose ``modules`` (if any) is a mapping."""
    return isinstance(s, dict) and (s.get("modules") is None or isinstance(s["modules"], dict))


def _valid_row(r) -> bool:
    return (isinstance(r, dict) and isinstance(r.get("module"), str)
            and isinstance(r.get("file"), str) and isinstance(r.get("line"), int)
            and not isinstance(r.get("line"), bool))


def _load_rows(path: Path) -> list[dict]:
    try:
        lines = [x for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):  # ValueError covers UnicodeDecodeError
        print("::warning::results.jsonl unreadable; survivors cannot be attributed to the PR")
        return []
    rows = []
    for x in lines:
        try:
            r = json.loads(x)
        except ValueError:
            continue
        if _valid_row(r):
            rows.append(r)
    if len(rows) != len(lines):
        print(f"::warning::{len(lines) - len(rows)} malformed results.jsonl row(s) ignored")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--diff-file", default=None)
    p.add_argument("--baseline-summary", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = p.parse_args(argv)

    merged_dir = Path(args.merged_dir)
    merged = _load_json(str(merged_dir / "summary.json"))
    if not _valid_summary(merged):
        verdict = {"state": "success", "pr_owned": [], "infra": ["report"], "modules": {},
                   "description": _truncate("gremlins: merged report unavailable (infra); "
                                            "no PR-owned result")}
        print("::warning::merged mutation report unavailable; not reporting red for infra")
    else:
        changed: dict[str, set[int]] = {}
        try:
            if not args.diff_file:
                raise OSError("no --diff-file given")
            changed = parse_changed_lines(Path(args.diff_file).read_text())
        except (OSError, ValueError):  # ValueError covers UnicodeDecodeError
            changed = {}
            print("::warning::PR diff unavailable; survivors cannot be attributed to the PR")
        baseline = _load_json(args.baseline_summary)
        if not _valid_summary(baseline):
            baseline = None
        if args.baseline_summary and baseline is None:
            print("::warning::main baseline summary unreadable; no failure is excused as pre-existing")
        verdict = classify(merged, _load_rows(merged_dir / "results.jsonl"), changed, baseline)
    for name, v in verdict["modules"].items():
        if v["class"] == "infra":
            print(f"::warning::{name}: {v['reason']} (infra, not PR-owned)")
        elif v["class"] == "pr_owned":
            print(f"::error::{name}: {v['reason']}")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if args.github_output:
        with open(args.github_output, "a") as fh:
            fh.write(f"state={verdict['state']}\ndescription={verdict['description']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())