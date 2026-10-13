"""The gremlins workflow wires tools/mutation_pr_status.py into ``mutation/pr``.

Parses the shipped ``.github/workflows/mutation.yml`` and runs the shipped
``report_status`` decision script (everything before its ``gh api`` call) under
bash with fixture env, so the YAML wiring is checked, not a hand copy.

# packages: foundation, ops
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())["jobs"]


def _classify_step():
    return next(s for s in JOBS["report"]["steps"] if s.get("id") == "classify")


def _status_script():
    step = next(s for s in JOBS["report_status"]["steps"]
                if str(s.get("name", "")).startswith("Post mutation status"))
    return step["run"], step["env"]


def _decide(**env):
    run, _ = _status_script()
    script = "set -euo pipefail\n" + run[:run.index("gh api")] + '\necho "${state}|${desc}"\n'
    full = dict(os.environ, MODULES='["a","b"]', REPORT_RESULT="success", SKIPPED_COUNT="0")
    full.update(env)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         check=True, env=full).stdout.strip()
    return tuple(out.split("|", 1))


def test_report_job_exposes_classification_outputs():
    outputs = JOBS["report"]["outputs"]
    assert outputs["state"] == "${{ steps.classify.outputs.state }}"
    assert outputs["description"] == "${{ steps.classify.outputs.description }}"
    assert JOBS["report"]["permissions"] == {"contents": "read", "actions": "read"}


def test_classify_step_runs_after_the_merge_and_never_gates_the_job():
    step = _classify_step()
    assert step["if"] == "always() && needs.plan.outputs.pr_number != ''"
    assert "tools/mutation_pr_status.py" in step["run"]
    assert "--merged-dir merged" in step["run"]
    assert step["env"]["BASE_SHA"] == "${{ needs.plan.outputs.base_sha }}"
    steps = JOBS["report"]["steps"]
    merge = next(i for i, s in enumerate(steps) if s.get("name") == "Merge module reports")
    assert merge < steps.index(step)


def test_report_status_uses_the_classification_when_present():
    _, env = _status_script()
    assert env["CLASSIFIED_STATE"] == "${{ needs.report.outputs.state }}"
    assert env["CLASSIFIED_DESC"] == "${{ needs.report.outputs.description }}"
    assert _decide(CLASSIFIED_STATE="failure", CLASSIFIED_DESC="PR-owned problem in a") == \
        ("failure", "PR-owned problem in a")
    # A PR-owned verdict is posted as is, whatever the report job result.
    assert _decide(CLASSIFIED_STATE="failure", CLASSIFIED_DESC="PR-owned problem in a",
                   REPORT_RESULT="failure") == ("failure", "PR-owned problem in a")
    assert _decide(CLASSIFIED_STATE="success", CLASSIFIED_DESC="1 infra warning(s)") == \
        ("success", "1 infra warning(s)")


def test_report_status_without_classification_is_infra_when_the_report_job_failed():
    state, desc = _decide(REPORT_RESULT="failure")
    assert state == "neutral" and "before classification -- infra" in desc
    state, desc = _decide()
    assert state == "success" and "2 module(s)" in desc