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
_FILE = re.compile(r"^\+\+\+ b/(.+)$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_changed_lines(diff_text: str) -> dict[str, set[int]]:
    """Map each file in a ``git diff -U0`` to the head-side line numbers it added or changed."""
    changed: dict[str, set[int]] = {}
    current: str | None = None
    for line in diff_text.splitlines():
        m = _FILE.match(line)
        if m:
            current = m.group(1)
            changed.setdefault(current, set())
            continue
        if line.startswith("+++ "):  # +++ /dev/null: a deleted file has no head lines
            current = None
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
    for name, mod in (baseline.get("modules") or {}).items():
        if isinstance(mod, dict) and not mod.get("skipped_reason") \
                and _is_test_failure_rc(mod.get("run_exit_code")):
            out.add(name)
    return out


def classify_module(name: str, mod, rows: list[dict], changed: dict[str, set[int]],
                    failing_on_main: set[str]) -> tuple[str, str]:
    """Return (class, reason) with class in ok / pr_owned / infra."""
    if not isinstance(mod, dict):
        return "infra", "no report: job cancelled, runner or setup error"
    if mod.get("skipped_reason"):
        return "infra", f"skipped ({mod['skipped_reason']})"
    rc = mod.get("run_exit_code")
    if rc == -1:
        return "infra", "timeout or cancelled"
    if isinstance(rc, int) and not isinstance(rc, bool) and (rc < 0 or rc >= 128):
        return "infra", f"killed by signal (exit {rc})"
    if (mod.get("counts_backend") or {}).get("error"):
        return "infra", "gremlin ERROR results (tool errors)"
    if any(str(r).startswith("RAW_") for r in mod.get("failure_reasons") or []):
        return "infra", "invalid raw report"
    if _is_test_failure_rc(rc):
        if name in failing_on_main:
            return "infra", f"module run fails on main too (exit {rc})"
        return "pr_owned", f"module run failed (exit {rc})"
    hits = sorted((r["file"], r["line"]) for r in rows
                  if r.get("module") == name and r.get("status") == "survived"
                  and r.get("line") in changed.get(r.get("file"), ()))
    if hits:
        shown = ", ".join(f"{f}:{n}" for f, n in hits[:5])
        return "pr_owned", f"{len(hits)} survivor(s) on changed lines: {shown}"
    return "ok", ""


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= DESC_MAX else text[:DESC_MAX - 3] + "..."


def classify(merged, rows: list[dict], changed: dict[str, set[int]], baseline=None) -> dict:
    """Pure verdict: ``state`` is ``failure`` only when a module is PR-owned."""
    modules = (merged.get("modules") or {}) if isinstance(merged, dict) else {}
    expected = merged.get("expected_modules") if isinstance(merged, dict) else None
    names = sorted(set(expected or []) | set(modules))
    failing_on_main = baseline_failing(baseline)
    result: dict[str, dict] = {}
    for name in names:
        cls, reason = classify_module(name, modules.get(name), rows, changed, failing_on_main)
        result[name] = {"class": cls, "reason": reason}
    owned = [n for n, v in result.items() if v["class"] == "pr_owned"]
    infra = [n for n, v in result.items() if v["class"] == "infra"]
    if owned:
        desc = f"gremlins: PR-owned problem in {', '.join(owned)}"
        if infra:
            desc += f"; {len(infra)} infra warning(s)"
    elif infra:
        desc = f"gremlins: no PR-owned problems; {len(infra)} infra warning(s)"
    else:
        desc = f"gremlins: no PR-owned problems in {len(names)} module(s)"
    return {"state": "failure" if owned else "success", "description": _truncate(desc),
            "pr_owned": owned, "infra": infra, "modules": result}


def _load_json(path: str | None):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _load_rows(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        print("::warning::results.jsonl unreadable; survivors cannot be attributed to the PR")
        return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--diff-file", default=None)
    p.add_argument("--baseline-summary", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = p.parse_args(argv)

    merged_dir = Path(args.merged_dir)
    merged = _load_json(str(merged_dir / "summary.json"))
    if not isinstance(merged, dict):
        verdict = {"state": "success", "pr_owned": [], "infra": ["report"], "modules": {},
                   "description": _truncate("gremlins: merged report unavailable (infra); "
                                            "no PR-owned result")}
        print("::warning::merged mutation report unavailable; not reporting red for infra")
    else:
        changed: dict[str, set[int]] = {}
        try:
            changed = parse_changed_lines(Path(args.diff_file).read_text()) if args.diff_file else {}
        except (OSError, ValueError):  # ValueError covers UnicodeDecodeError
            changed = {}
            print("::warning::PR diff unavailable; survivors cannot be attributed to the PR")
        baseline = _load_json(args.baseline_summary)
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