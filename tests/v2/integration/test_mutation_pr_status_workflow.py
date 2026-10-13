"""Integration tests for the mutation PR status workflow shape.

# packages: foundation, ops
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())["jobs"]

REPORT_ENV_DEFAULTS = {
    "MODULES": '["a","b"]',
    "REPORT_RESULT": "success",
    "SKIPPED_COUNT": "0",
    "CLASSIFIED_STATE": "",
    "CLASSIFIED_DESC": "",
}


def _steps(job: dict) -> list:
    return job["steps"]


def _classify_step() -> dict:
    for step in _steps(JOBS["report"]):
        if step.get("id") == "classify":
            return step
    raise AssertionError("classify step not found")


def _report_status_script() -> str:
    for step in _steps(JOBS["report_status"]):
        if "Post mutation status to the PR head commit" in step.get("name", ""):
            return step["run"]
    raise AssertionError("report status step not found")


def test_classify_step_wiring():
    step = _classify_step()
    assert re.match(r"^always\(\)\s*&&", step["if"])
    env = step["env"]
    assert env["PR"] == "${{ needs.plan.outputs.pr_number }}"
    assert env["BASE_SHA"] == "${{ needs.plan.outputs.base_sha }}"
    assert env["HEAD_SHA"] == "${{ github.event.workflow_run.head_sha }}"
    assert "github.sha" not in env and "github.sha" not in step["run"]
    checkout = next(s for s in _steps(JOBS["report"]) if s.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"]["fetch-depth"] == 0
    names = [s.get("id") or s.get("name") or "" for s in _steps(JOBS["report"])]
    assert names.index("classify") > _steps(JOBS["report"]).index(checkout)
    assert "permissions" not in JOBS["report"]
    assert JOBS["report"]["outputs"]["state"] == "${{ steps.classify.outputs.state }}"
    assert JOBS["report"]["outputs"]["description"] == "${{ steps.classify.outputs.description }}"


def _run_bash(script: str, env_over: dict, functions: str) -> subprocess.CompletedProcess:
    full = functions + "\nset -euo pipefail\n" + script
    env = {**os.environ, **env_over}
    return subprocess.run(["bash", "-c", full], env=env, capture_output=True, text=True, cwd=ROOT)


GIT_PY_FUNCS = """
git() { echo "git $*" >> "$LOG"; [ -n "${GIT_FAIL:-}" ] && return 1; return 0; }
python() { echo "python $*" >> "$LOG"; return 0; }
"""


def test_classify_script_runs_git_and_the_tool(tmp_path):
    log = tmp_path / "log"
    script = _classify_step()["run"]
    proc = _run_bash(script, {"LOG": str(log), "PR": "7", "BASE_SHA": "base1", "HEAD_SHA": "head1"}, GIT_PY_FUNCS)
    assert proc.returncode == 0
    assert log.read_text().splitlines() == [
        "git fetch -q --no-tags origin +refs/pull/7/head:refs/remotes/pr/head",
        "git -c core.quotepath=off diff --name-only --no-renames base1...head1",
        "python tools/mutation_pr_status.py --merged-dir merged --changed-files changed_files.txt",
    ]


def test_classify_script_degrades_when_git_fails(tmp_path):
    log = tmp_path / "log"
    script = _classify_step()["run"]
    proc = _run_bash(
        script,
        {"LOG": str(log), "PR": "7", "BASE_SHA": "base1", "HEAD_SHA": "head1", "GIT_FAIL": "1"},
        GIT_PY_FUNCS,
    )
    assert proc.returncode == 0
    assert "::warning::" in proc.stdout
    assert log.read_text().splitlines() == [
        "git fetch -q --no-tags origin +refs/pull/7/head:refs/remotes/pr/head",
        "python tools/mutation_pr_status.py --merged-dir merged",
    ]


def _decision_script() -> str:
    run = _report_status_script()
    return run[: run.index("gh api")].rsplit("\n", 1)[0]


DECISION_TABLE = [
    # (modules, skipped, classified_state, classified_desc, report_result) -> expected state
    ("[]", "0", "", "", "success", "success"),
    ("[\"a\",\"b\"]", "2", "", "", "success", "failure"),
    ("[\"a\",\"b\"]", "0", "failure", "classified text", "success", "failure"),
    ("[\"a\",\"b\"]", "0", "neutral", "classified text", "success", "neutral"),
    ("[\"a\",\"b\"]", "0", "success", "classified text", "success", "success"),
    ("[\"a\",\"b\"]", "0", "success", "classified text", "failure", "neutral"),
    ("[\"a\",\"b\"]", "0", "", "", "failure", "neutral"),
    ("[\"a\",\"b\"]", "0", "", "", "cancelled", "neutral"),
    ("[\"a\",\"b\"]", "0", "", "", "skipped", "neutral"),
    ("[\"a\",\"b\"]", "0", "", "", "success", "success"),
    ("[\"a\",\"b\"]", "", "success", "classified text", "success", "success"),
]


def test_report_status_decision_table():
    script = _decision_script()
    for modules, skipped, cstate, cdesc, report, expected in DECISION_TABLE:
        env = {
            "MODULES": modules,
            "SKIPPED_COUNT": skipped,
            "CLASSIFIED_STATE": cstate,
            "CLASSIFIED_DESC": cdesc,
            "REPORT_RESULT": report,
        }
        proc = _run_bash(script + '\necho "${state}|${desc}"', env, "")
        assert proc.returncode == 0, proc.stderr
        state, desc = proc.stdout.strip().split("|", 1)
        assert state == expected, (modules, skipped, cstate, report)
        if cstate and not (cstate == "success" and report != "success"):
            assert desc == cdesc
        elif expected == "neutral" and report != "success":
            assert "before classification" in desc


GH_FUNCS = """
gh() { printf 'ARG:%s\\n' "$@"; echo '--'; }
"""


def _assert_gh_args(stdout: str, state: str):
    lines = [ln for ln in stdout.splitlines() if ln.startswith("ARG:") or ln == "--"]
    assert lines.count("--") == 1, stdout
    args = [ln[len("ARG:"):] for ln in lines if ln.startswith("ARG:")]
    assert args == [
        "api",
        "repos/o/r/check-runs",
        "-f",
        "name=mutation/pr",
        "-f",
        "head_sha=sha1",
        "-f",
        "status=completed",
        "-f",
        f"conclusion={state}",
        "-f",
        "details_url=https://x/run/5",
        "-f",
        f"output[title]=Mutation testing: {state}",
        "-f",
        "output[summary]=classified text with spaces",
    ]


def test_check_run_post_keeps_argument_boundaries():
    script = _report_status_script()
    base_env = {
        "SHA": "sha1",
        "RUN_URL": "https://x/run/5",
        "REPO": "o/r",
        "MODULES": '["a"]',
        "SKIPPED_COUNT": "0",
        "REPORT_RESULT": "success",
        "CLASSIFIED_DESC": "classified text with spaces",
    }
    for state in ("failure", "neutral", "success"):
        env = {**base_env, "CLASSIFIED_STATE": state}
        proc = _run_bash(script, env, GH_FUNCS)
        assert proc.returncode == 0, proc.stderr
        _assert_gh_args(proc.stdout, state)


def test_report_status_wiring():
    assert re.match(r"^always\(\)\s*&&", JOBS["report_status"]["if"])
    for step in _steps(JOBS["report_status"]):
        if "Post mutation status to the PR head commit" in step.get("name", ""):
            assert step["env"]["SHA"] == "${{ github.event.workflow_run.head_sha }}"
            assert step["env"]["CLASSIFIED_STATE"] == "${{ needs.report.outputs.state }}"
            assert step["env"]["CLASSIFIED_DESC"] == "${{ needs.report.outputs.description }}"
            break
    else:
        raise AssertionError("report status step not found")
    assert JOBS["report_status"]["permissions"] == {"checks": "write"}
    assert "report_neutral" not in JOBS
    for job in JOBS.values():
        perms = job.get("permissions") or {}
        assert "statuses" not in perms