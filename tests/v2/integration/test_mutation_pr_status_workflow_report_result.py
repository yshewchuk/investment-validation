"""A failed report job is never overridden into success by the classified verdict.

Parses the shipped ``.github/workflows/mutation.yml`` and runs the decision script
(up to the neutral guard) under bash with fixture env.

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
    guard = run.index('"$state" = neutral')
    script = "set -euo pipefail\n" + run[:guard].rsplit("\n", 1)[0] + '\necho "${state}|${desc}"\n'
    full = dict(os.environ, MODULES='["a","b"]', REPORT_RESULT="success", SKIPPED_COUNT="0")
    full.update(env)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         check=True, env=full).stdout.strip()
    return tuple(out.split("|", 1))


def test_classified_success_with_a_failed_report_job_is_not_success():
    for result in ("failure", "cancelled", "skipped"):
        state, desc = _decide(CLASSIFIED_STATE="success", CLASSIFIED_DESC="all measured",
                              REPORT_RESULT=result)
        assert state == "failure", result
        assert f"report job {result}" in desc


def test_classified_success_with_a_successful_report_job_is_posted():
    assert _decide(CLASSIFIED_STATE="success", CLASSIFIED_DESC="all measured") == \
        ("success", "all measured")


def test_classified_failure_and_neutral_are_posted_whatever_the_report_result():
    assert _decide(CLASSIFIED_STATE="failure", CLASSIFIED_DESC="PR-owned problem in a",
                   REPORT_RESULT="failure") == ("failure", "PR-owned problem in a")
    assert _decide(CLASSIFIED_STATE="neutral", CLASSIFIED_DESC="not a pass",
                   REPORT_RESULT="failure") == ("neutral", "not a pass")