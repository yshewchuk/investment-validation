"""Classify a PR's merged gremlins mutation report: failure, success or neutral.

Inputs: ``DIR/summary.json`` + ``DIR/results.jsonl`` and ``--changed-files FILE``
(one changed repo path per line). Output: JSON ``{"state", "description",
"reasons"}``, ``::error::``/``::warning::`` annotations, and ``state=``/
``description=`` appended to ``--github-output`` (default env ``$GITHUB_OUTPUT``).
``failure`` only for a PR-owned problem (a surviving mutant in a changed file or
a failing PR test run), ``success`` only for a clean measured run of every
planned module (a module entry whose measured ``run_exit_code`` is missing or
null is not clean), ``neutral`` otherwise; attribution by FILE, never by line. Exit
is 0 for classified/degraded input; nonzero only for argparse errors (exit 2)
or an unwritable output file (the ``OSError`` propagates AFTER the verdict).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DESC_MAX = 140
ROW_STATUSES = ("killed", "survived", "timeout", "suspicious", "excluded")


def _cut(text: str) -> str:
    return text if len(text) <= DESC_MAX else text[:DESC_MAX - 3] + "..."


def _list(label: str, items) -> str:
    items = sorted(items)
    return f"{label} {', '.join(items[:3])}{', ...' if len(items) > 3 else ''}"


def classify(summary, rows, changed) -> dict:
    """Pure verdict ``failure`` > ``success`` > ``neutral``; ``reasons[0]`` sets
    the neutral description."""
    reasons: list[str] = []
    modules = summary.get("modules") if isinstance(summary, dict) else None
    hits = ([r["file"] for r in rows
             if r["status"] == "survived" and r["file"] in changed]
            if rows is not None and changed else [])
    failing = ([str(n) for n, e in modules.items()
                if isinstance(e, dict) and not e.get("skipped_reason")
                and type(e.get("run_exit_code")) is int
                and 1 <= e["run_exit_code"] <= 127]
               if isinstance(modules, dict) else [])
    if hits or failing:
        parts = [p for p in (
            f"{len(hits)} " + _list("surviving mutant(s) in", set(hits)) if hits else None,
            _list("tests fail in", failing) if failing else None) if p]
        desc = "gremlins: PR-owned problem: " + "; ".join(parts)
        return {"state": "failure", "description": _cut(desc), "reasons": parts}
    if not isinstance(summary, dict):
        reasons.append("merged report unavailable")
    elif not isinstance(modules, dict) or not modules:
        reasons.append("no module measured")
    else:
        if summary.get("complete") is not True:
            reasons.append("run incomplete")
        if summary.get("tool_error") is not False:
            reasons.append("tool error")
        if summary.get("failure_reasons") != []:
            reasons.append("failure reasons reported")
        exp = summary.get("expected_modules")
        names = (list(exp) if isinstance(exp, list) and exp
                 and all(isinstance(m, str) for m in exp) else list(modules))
        for name in names:
            e = modules.get(name)
            rc = e.get("run_exit_code") if isinstance(e, dict) else None
            if (not isinstance(e, dict)
                    or e.get("skipped_reason")
                    or not (type(rc) is int and rc == 0)
                    or not isinstance(e.get("counts_backend"), dict)
                    or e["counts_backend"].get("error")):
                reasons.append(f"module {name} not clean")
    if rows is None:
        reasons.append("result rows unreadable")
    if not changed:
        reasons.append("changed files unavailable")
    if not reasons:
        return {"state": "success", "reasons": [],
                "description": f"gremlins: no PR-owned problems in {len(names)} module(s)"}
    desc = f"gremlins: not a clean measured run ({reasons[0]}); not a pass"
    return {"state": "neutral", "description": _cut(desc), "reasons": reasons}


# Any malformed or partial report is neutral, even if some rows are readable (user decision 2026-10-13): no salvaging of partial evidence for this advisory check.
def _load_rows(path: Path) -> list | None:
    """``results.jsonl`` rows, or ``None`` if unreadable/undecodable or any
    non-blank line is not JSON of a valid row (empty file is valid)."""
    try:
        rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return None
    if any(not (isinstance(r, dict) and isinstance(r.get("module"), str)
                and isinstance(r.get("file"), str) and r.get("status") in ROW_STATUSES)
           for r in rows):
        return None
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--changed-files", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = p.parse_args(argv)
    try:
        summary = json.loads((Path(args.merged_dir) / "summary.json").read_text())
    except (OSError, ValueError):
        summary = None
    rows = _load_rows(Path(args.merged_dir) / "results.jsonl")
    changed = None
    if args.changed_files:
        try:
            changed = ({ln.strip() for ln in
                        Path(args.changed_files).read_text().splitlines() if ln.strip()} or None)
        except (OSError, ValueError):
            changed = None
    verdict = classify(summary, rows, changed)
    for reason in verdict["reasons"]:
        print(f"::{'error' if verdict['state'] == 'failure' else 'warning'}::{reason}")
    print(json.dumps(verdict, indent=2, sort_keys=True))
    if args.github_output:
        with open(args.github_output, "a") as fh:
            fh.write(f"state={verdict['state']}\ndescription={verdict['description']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
