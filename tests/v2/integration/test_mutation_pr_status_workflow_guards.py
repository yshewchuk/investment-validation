"""Ordering and baseline-search guards in the gremlins ``report_status`` / classify wiring.

Parses the shipped ``.github/workflows/mutation.yml``; the decision script is run
under bash with fixture env (everything before its ``gh api`` call).

# packages: foundation, ops
"""
from __future__ import annotations

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
    script = "set -euo pipefail\n" + run[:run.index("gh api")] + '\necho "${state}|${desc}"\n'
    full = dict(os.environ, MODULES='["a","b"]', REPORT_RESULT="success", SKIPPED_COUNT="0")
    full.update(env)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         check=True, env=full).stdout.strip()
    return tuple(out.split("|", 1))


def test_pr_closed_placeholder_guard_runs_before_the_classified_branch():
    state, desc = _decide(CLASSIFIED_STATE="success", CLASSIFIED_DESC="x", SKIPPED_COUNT="1")
    assert state == "failure" and "skipped (PR closed)" in desc


def test_empty_plan_is_checked_before_everything():
    state, desc = _decide(MODULES="[]", CLASSIFIED_STATE="failure", CLASSIFIED_DESC="x",
                          SKIPPED_COUNT="2")
    assert state == "success" and "by design" in desc


def test_classified_verdict_is_posted_when_nothing_was_skipped():
    assert _decide(CLASSIFIED_STATE="failure", CLASSIFIED_DESC="PR-owned problem in a") == \
        ("failure", "PR-owned problem in a")
    assert _decide(CLASSIFIED_STATE="success", CLASSIFIED_DESC="no module measured",
                   REPORT_RESULT="failure") == ("success", "no module measured")


def test_baseline_search_walks_recent_push_runs_and_warns_when_none_has_the_artifact():
    step = next(s for s in JOBS["report"]["steps"] if s.get("id") == "classify")
    run = step["run"]
    assert "--limit 10" in run
    assert "for run_id in" in run
    assert "--baseline-summary baseline/summary.json" in run
    assert "::warning::no recent main push run has a mutation-report artifact" in run
    assert "--limit 1 " not in run
