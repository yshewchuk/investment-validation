"""Classify a PR's gremlins mutation result as PR-owned (red) or infra (warning).

Inputs: the merged ``summary.json`` (+ ``results.jsonl``) written by
``gremlin_results.py merge``, the PR's ``git diff -U0`` text and, optionally,
main's last merged ``summary.json``. Output: a JSON verdict on stdout and
``state`` / ``description`` lines in ``$GITHUB_OUTPUT``. Only PR-owned problems
make ``state`` ``failure``; infra problems are ``::warning::`` lines. When no
module was measured at all, ``state`` is ``neutral`` (the workflow posts every
state, neutral included, as the mutation/pr check run). The exit code is 0 for
every classified or degraded input; it is nonzero only for argument errors, or
when the ``--github-output`` file cannot be written (raised after
classification, so the CI step fails loudly).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DESC_MAX = 140
_ROW_STATUSES = ("killed", "survived", "timeout", "suspicious", "excluded")
# failure_reasons entries that gremlin_results.merge writes without a module prefix.
_MERGE_CODES = ("RUN_PROVENANCE_MISMATCH", "NO_MODULE_REPORTS", "DUPLICATE_MODULES",
                "MISSING_MODULES", "UNEXPECTED_MODULES")
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
    if "run_exit_code" not in mod or not (
            mod["run_exit_code"] is None
            or (isinstance(mod["run_exit_code"], int)
                and not isinstance(mod["run_exit_code"], bool))):
        return "infra", "malformed module entry (run_exit_code)"
    if not isinstance(mod.get("counts_backend"), dict):
        return "infra", "malformed module entry (counts_backend)"
    if not isinstance(mod.get("failure_reasons"), list):
        return "infra", "malformed module entry (failure_reasons)"
    if not isinstance(mod.get("tool_error"), bool):
        return "infra", "malformed module entry (tool_error)"
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
    if mod.get("tool_error") is True:
        return "infra", "module reported a tool error"
    return "ok", ""


def _truncate(text: str, keep_tail: str = "") -> str:
    """Fit DESC_MAX; ``keep_tail`` (the infra count) is never cut."""
    text = " ".join(text.split())
    if len(text) + len(keep_tail) <= DESC_MAX:
        return text + keep_tail
    return text[:DESC_MAX - 3 - len(keep_tail)] + "..." + keep_tail


def _description(owned, infra, names, unattributed, problems, unsafe) -> str:
    """Build the check-run description for :func:`classify`.

    Invariant: whenever ``infra`` is non-empty, the returned string contains
    ``<N> infra warning(s)`` and is <= DESC_MAX chars, the count surviving
    truncation: the tail ``; N infra warning(s)`` is kept by ``_truncate``
    (only the owned and unattributed headlines can be long and both pass it
    as the tail; the other headlines embed the count directly)."""
    tail = ""
    if owned:
        desc = f"gremlins: PR-owned problem in {', '.join(owned)}"
        if infra:
            tail = f"; {len(infra)} infra warning(s)"
    elif infra and len(infra) == len(names):
        # Nothing was measured: still not red (infra), but never worded as a pass.
        desc = f"gremlins: no module measured ({len(infra)} infra warning(s)); not a pass"
    elif unattributed:
        desc = f"gremlins: {problems[0]}"
        if infra:
            tail = f"; {len(infra)} infra warning(s)"
    elif unsafe:
        desc = f"gremlins: report incomplete or tool error ({len(infra)} infra warning(s)); not a pass"
    elif infra:
        desc = f"gremlins: no PR-owned problems; {len(infra)} infra warning(s)"
    else:
        desc = f"gremlins: no PR-owned problems in {len(names)} module(s)"
    return _truncate(desc, tail)


def classify(merged, rows: list[dict], changed: dict[str, set[int]], baseline=None,
             attribution_problems=()) -> dict:
    """Pure verdict: ``failure`` only when a module is PR-owned; ``neutral`` when no
    module was measured (all infra), the merge reported a tool error / incomplete run,
    or survivors could not be attributed to the PR (``attribution_problems``); else
    ``success``."""
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
    # The merge itself can report a tool error (provenance mismatch, missing or duplicate
    # modules, ...) while every module looks clean: that is never a pass.
    unsafe = isinstance(merged, dict) and (merged.get("tool_error") is True
                                           or merged.get("complete") is False)
    reasons = merged.get("failure_reasons") if isinstance(merged, dict) else None
    merge_issues = [r for r in reasons if isinstance(r, str) and r.startswith(_MERGE_CODES)] \
        if isinstance(reasons, list) else []
    problems = [p for p in attribution_problems if isinstance(p, str)]
    unattributed = bool(problems) and bool(names)
    nothing_measured = bool(names) and len(infra) == len(names)
    state = ("failure" if owned
             else "neutral" if (nothing_measured or unsafe or unattributed) else "success")
    return {"state": state,
            "description": _description(owned, infra, names, unattributed, problems, unsafe),
            "pr_owned": owned, "infra": infra, "modules": result, "merge_issues": merge_issues,
            "attribution_problems": problems}


def _load_json(path: str | None):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _valid_summary(s) -> bool:
    """A merged/baseline summary: a mapping whose ``modules`` is a non-empty mapping
    or whose ``expected_modules`` is a non-empty list -- it measured or planned
    something. ``modules`` must be a mapping (absent, null, a list, ... are invalid)
    and ``expected_modules``, when present, must be a list or null (a local merge
    without a plan writes null); ``complete`` and
    ``tool_error`` must be bools and ``failure_reasons`` a list. An invalid summary is
    never success: ``main`` treats an invalid merged summary as an unavailable
    report (neutral) and drops an invalid baseline."""
    if not isinstance(s, dict) or not isinstance(s.get("modules"), dict):
        return False
    if not (isinstance(s.get("complete"), bool) and isinstance(s.get("tool_error"), bool)
            and isinstance(s.get("failure_reasons"), list)):
        return False
    expected = s.get("expected_modules")
    if expected is not None and not isinstance(expected, list):
        return False
    return bool(s["modules"]) or bool(expected)


def _valid_row(r) -> bool:
    return (isinstance(r, dict) and isinstance(r.get("module"), str) and r["module"] != ""
            and isinstance(r.get("file"), str) and r["file"] != ""
            and isinstance(r.get("line"), int) and not isinstance(r.get("line"), bool)
            and r["line"] >= 1
            and isinstance(r.get("status"), str) and r.get("status") in _ROW_STATUSES)


def _load_rows(path: Path) -> tuple[list[dict], bool]:
    """Return (rows, complete); complete is False if the file or any row was unusable."""
    try:
        lines = [x for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):  # ValueError covers UnicodeDecodeError
        print("::warning::results.jsonl unreadable; survivors cannot be attributed to the PR")
        return [], False
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
    return rows, len(rows) == len(lines)


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
        verdict = {"state": "neutral", "pr_owned": [], "infra": ["report"], "modules": {},
                   "merge_issues": [], "attribution_problems": [],
                   "description": _truncate("gremlins: merged report unavailable (infra); "
                                            "no PR-owned result")}
        print("::warning::merged mutation report unavailable; not reporting red for infra")
    else:
        changed: dict[str, set[int]] = {}
        problems: list[str] = []
        # Diff unavailable (not given, unreadable, empty text, or hunks without any file
        # header) => neutral; a parsed diff with zero added lines is valid.
        try:
            if not args.diff_file:
                raise OSError("no --diff-file given")
            diff_text = Path(args.diff_file).read_text()
            if not diff_text.strip():
                raise ValueError("PR diff is empty")
            lines = diff_text.splitlines()
            if any(x.startswith("@@") for x in lines) and not any(
                    x.startswith("+++ ") for x in lines):
                raise ValueError("diff has hunks but no file header")
            changed = parse_changed_lines(diff_text)
        except (OSError, ValueError):  # ValueError covers UnicodeDecodeError
            changed = {}
            problems.append("PR diff unavailable -- survivors not attributed; not a pass")
            print("::warning::PR diff unavailable; survivors cannot be attributed to the PR")
        baseline = _load_json(args.baseline_summary)
        if not _valid_summary(baseline):
            baseline = None
        if args.baseline_summary and baseline is None:
            print("::warning::main baseline summary unreadable; no failure is excused as pre-existing")
        rows, rows_complete = _load_rows(merged_dir / "results.jsonl")
        if not rows_complete:
            problems.append("result rows unreadable or malformed -- survivors not attributed; "
                            "not a pass")
        verdict = classify(merged, rows, changed, baseline, problems)
    for issue in verdict["merge_issues"]:
        print(f"::warning::merge: {issue}")
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