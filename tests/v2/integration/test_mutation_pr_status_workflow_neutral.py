"""A run in which nothing was measured is posted as a neutral (grey) check run.

Parses the shipped ``.github/workflows/mutation.yml``. ``report_status`` posts no
commit status for the neutral verdict; the separate ``report_neutral`` job (the only
job with ``checks: write``) posts the check run.

# packages: foundation, ops
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())["jobs"]


def _status_run():
    step = next(s for s in JOBS["report_status"]["steps"]
                if str(s.get("name", "")).startswith("Post mutation status"))
    return step["run"]


def test_report_status_posts_no_commit_status_for_neutral():
    run = _status_run()
    guard = run.index('"$state" = neutral')
    post = run.index('gh api "repos/$REPO/statuses')
    assert guard < post  # the neutral exit precedes the status post
    assert "exit 0" in run[guard:post]
    # Simulate the shipped script up to the guard: the neutral verdict reaches it unchanged.
    prefix = "set -euo pipefail\n" + run[:guard].rsplit("\n", 1)[0] + \
        '\necho "${state}|${desc}"\n'
    env = dict(os.environ, MODULES='["a"]', REPORT_RESULT="failure", SKIPPED_COUNT="0",
               CLASSIFIED_STATE="neutral", CLASSIFIED_DESC="nothing measured")
    out = subprocess.run(["bash", "-c", prefix], capture_output=True, text=True,
                         check=True, env=env).stdout.strip()
    assert out == "neutral|nothing measured"


def test_report_neutral_job_is_the_only_checks_writer_and_posts_neutral():
    job = JOBS["report_neutral"]
    assert job["permissions"] == {"checks": "write"}
    assert job["needs"] == ["plan", "report"]
    cond = job["if"]
    assert "needs.report.outputs.state == 'neutral'" in cond
    assert "needs.plan.outputs.modules != '[]'" in cond
    assert "needs.plan.outputs.pr_number != ''" in cond
    assert "skipped_count" in cond  # a PR-closed placeholder keeps the failure mapping
    run = job["steps"][0]["run"]
    assert "repos/$REPO/check-runs" in run
    assert 'name="mutation/pr"' in run
    assert "conclusion=neutral" in run
    assert "status=completed" in run
    others = [n for n, j in JOBS.items()
              if n != "report_neutral" and "checks" in (j.get("permissions") or {})]
    assert others == []
