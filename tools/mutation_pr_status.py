"""Classify a PR's merged gremlins mutation report: failure, success or neutral.

Inputs: ``DIR/summary.json`` + ``DIR/results.jsonl`` and ``--changed-files FILE``
(NUL-separated changed repo paths, as from ``git diff -z --name-only``). Output: JSON ``{"state", "description",
"reasons"}``, ``::error::``/``::warning::`` annotations, and ``state=``/
``description=`` appended to ``--github-output`` (default env ``$GITHUB_OUTPUT``).
``failure`` only for a PR-owned problem (a surviving mutant in a file the PR
changed), ``success`` only for a clean measured run of every
planned module (a module entry whose measured ``run_exit_code`` is missing or
null is not clean), ``neutral`` otherwise; attribution by FILE, never by line.
A nonzero module exit code is neutral. Exit
is 0 for classified/degraded input; nonzero only for argparse errors (exit 2)
or an unwritable output file (the ``OSError`` propagates AFTER the verdict).
Integrity first: an unavailable summary, a planned module missing (planned
modules come from ``expected_modules``, which must be a non-empty list of
strings), unreadable rows or no changed-file list is neutral before any
attribution; only a complete, well-formed report can be failure or success,
where failure is a surviving mutant in a file the PR changed, in a complete,
clean report.
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


_NEUTRAL_PREFIX = "gremlins: not a clean measured run ("
_NEUTRAL_SUFFIX = "); not a pass"


def _neutral_desc(reason: str) -> str:
    full = f"{_NEUTRAL_PREFIX}{reason}{_NEUTRAL_SUFFIX}"
    if len(full) <= DESC_MAX:
        return full
    room = DESC_MAX - len(_NEUTRAL_PREFIX) - len(_NEUTRAL_SUFFIX) - 3
    if room < 1:
        return full[:DESC_MAX - 3] + "..."
    return f"{_NEUTRAL_PREFIX}{reason[:room]}...{_NEUTRAL_SUFFIX}"


def classify(summary, rows, changed) -> dict:
    """Pure verdict ``neutral`` (integrity) first, then ``failure``, then
    ``success``; ``reasons[0]`` sets the neutral description."""
    reasons: list[str] = []
    modules = summary.get("modules") if isinstance(summary, dict) else None
    if not isinstance(summary, dict):
        reasons.append("merged report unavailable")
    elif not isinstance(modules, dict) or not modules:
        reasons.append("no module measured")
    exp = summary.get("expected_modules") if isinstance(summary, dict) else None
    if isinstance(exp, list) and exp and all(isinstance(m, str) for m in exp):
        names = list(exp)
    else:
        names = []
        if isinstance(summary, dict) and isinstance(modules, dict) and modules:
            reasons.append("planned modules unknown")
    if isinstance(summary, dict) and isinstance(modules, dict):
        for name in names:
            if not isinstance(modules.get(name), dict):
                reasons.append(f"module {name} missing")
    if rows is None:
        reasons.append("result rows unreadable")
    if rows is not None:
        for m in sorted({r["module"] for r in rows if r["module"] not in names}):
            reasons.append(f"row module {m} not planned")
    if not changed:
        reasons.append("changed files unavailable")
    if reasons:
        return {"state": "neutral", "description": _neutral_desc(reasons[0]),
                "reasons": reasons}
    hits = [r["file"] for r in rows
            if r["status"] == "survived" and r["file"] in changed]
    # Survivor-red below needs a complete, clean report: complete=True, no tool error, every planned module clean.
    if summary.get("complete") is not True:
        reasons.append("run incomplete")
    if summary.get("tool_error") is not False:
        reasons.append("tool error")
    if summary.get("failure_reasons") != []:
        reasons.append("failure reasons reported")
    for name in names:
        e = modules.get(name)
        rc = e.get("run_exit_code") if isinstance(e, dict) else None
        fr = e.get("failure_reasons") if isinstance(e, dict) else None
        cb = e.get("counts_backend") if isinstance(e, dict) else None
        cb_err = cb.get("error") if isinstance(cb, dict) else None
        if (not isinstance(e, dict)
                or e.get("skipped_reason")
                or not (type(rc) is int and rc == 0)
                or e.get("complete") is not True
                or e.get("tool_error") is not False
                or not (isinstance(fr, list) and fr == [])
                or not isinstance(cb, dict)
                or not (type(cb_err) is int and cb_err == 0)):
            if (isinstance(e, dict) and not e.get("skipped_reason")
                    and type(rc) is int and rc != 0):
                reasons.append(f"module {name} exited {rc}")
            else:
                reasons.append(f"module {name} not clean")
    if reasons:
        return {"state": "neutral", "description": _neutral_desc(reasons[0]),
                "reasons": reasons}
    if hits:
        parts = [f"{len(hits)} " + _list("surviving mutant(s) in", set(hits))]
        desc = "gremlins: PR-owned problem: " + "; ".join(parts)
        return {"state": "failure", "description": _cut(desc), "reasons": parts}
    return {"state": "success", "reasons": [],
            "description": f"gremlins: no PR-owned problems in {len(names)} module(s)"}


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
            parts = Path(args.changed_files).read_bytes().split(b"\0")
            names = {p.decode("utf-8") for p in parts if p}
            changed = names or None
        except (OSError, UnicodeDecodeError):
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
