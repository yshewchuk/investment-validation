"""Exhaustive ``report_status`` decision table for the gremlins workflow.

Red comes only from a classified failure or the PR-closed placeholder guard; a report job
that did not succeed without a trusted classified verdict is infra (neutral, grey), and a
classified ``success`` never overrides a report job that did not succeed. The decision
script is run under bash for every input combination (up to the neutral guard) and compared
with an independent oracle.

# packages: foundation, ops
"""
from __future__ import annotations

import itertools
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())["jobs"]


def _decide(**env):
    step = next(s for s in JOBS["report_status"]["steps"]
                if str(s.get("name", "")).startswith("Post mutation status"))
    run = step["run"]
    guard = run.index('"$state" = neutral')
    script = "set -euo pipefail\n" + run[:guard].rsplit("\n", 1)[0] + '\necho "${state}|${desc}"\n'
    full = dict(os.environ, MODULES='["a","b"]', REPORT_RESULT="success", SKIPPED_COUNT="0",
                CLASSIFIED_STATE="", CLASSIFIED_DESC="")
    full.update(env)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         check=True, env=full).stdout.strip()
    return tuple(out.split("|", 1))


def expected(modules, skipped, classified, report):
    if modules == "[]":
        return "success"
    if skipped != "0":
        return "failure"
    if classified == "failure":
        return "failure"
    if classified == "neutral":
        return "neutral"
    if classified == "success" and report == "success":
        return "success"
    if report != "success":
        return "neutral"
    return "success"


def test_every_input_combination_posts_the_table_state():
    combos = itertools.product(["[]", '["a","b"]', ""], ["0", "2"],
                               ["", "success", "failure", "neutral"],
                               ["success", "failure", "cancelled", "skipped"])
    seen = set()
    for modules, skipped, classified, report in combos:
        state, desc = _decide(MODULES=modules, SKIPPED_COUNT=skipped, CLASSIFIED_STATE=classified,
                              CLASSIFIED_DESC="classified text", REPORT_RESULT=report)
        want = expected(modules, skipped, classified, report)
        assert state == want, (modules, skipped, classified, report, state)
        seen.add(state)
    assert seen == {"success", "failure", "neutral"}


def test_descriptions_of_the_unclassified_branches():
    for result in ("failure", "cancelled", "skipped"):
        assert _decide(REPORT_RESULT=result) == (
            "neutral", f"gremlins: report job {result} before classification -- infra; not a pass")
        assert _decide(CLASSIFIED_STATE="success", CLASSIFIED_DESC="x", REPORT_RESULT=result) == (
            "neutral", f"gremlins: report job {result} before classification -- infra; not a pass")
    assert _decide() == ("success", "gremlins: report published for 2 module(s)")
    state, desc = _decide(SKIPPED_COUNT="1")
    assert state == "failure" and "skipped (PR closed)" in desc
    assert _decide(MODULES="[]")[0] == "success"
    assert _decide(CLASSIFIED_STATE="failure", CLASSIFIED_DESC="PR-owned problem in a",
                   REPORT_RESULT="failure") == ("failure", "PR-owned problem in a")