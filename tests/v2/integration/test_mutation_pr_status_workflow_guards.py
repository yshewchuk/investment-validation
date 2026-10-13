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
    assert _decide(CLASSIFIED_STATE="success", CLASSIFIED_DESC="all measured") == \
        ("success", "all measured")


GH_STUB = r'''
gh() {
  if [ "$1 $2" = "run list" ]; then
    case "$*" in
      *"--branch main"*"--event push"*) printf '%s\n' $FAKE_RUNS ;;
      *) printf '%s\n' 999 ;;   # decoy: a non-main or non-push run, which HAS an artifact
    esac
    return 0
  fi
  if [ "$1 $2" = "run download" ]; then
    id="$3"
    mkdir -p baseline
    case " $NOFILE_IDS " in *" $id "*) return 0 ;; esac   # artifact without summary.json
    case " $ARTIFACT_IDS " in
      *" $id "*) echo "$id" > baseline/summary.json; return 0 ;;
    esac
    return 1
  fi
  return 1
}
'''


def _run_baseline_loop(tmp_path, runs, artifacts, nofile=""):
    step = next(s for s in JOBS["report"]["steps"] if s.get("id") == "classify")
    run = step["run"]
    snippet = run[run.index("have_baseline=0"):run.index("python tools/mutation_pr_status.py")]
    script = ("set -uo pipefail\nargs=(--merged-dir merged)\n" + GH_STUB + snippet
              + 'echo "ARGS=${args[*]}"\n'
              + 'if [ -f baseline/summary.json ]; then echo "CHOSEN=$(cat baseline/summary.json)";'
              + " else echo CHOSEN=none; fi\n")
    env = dict(os.environ, REPO="o/r", FAKE_RUNS=runs, ARTIFACT_IDS=artifacts, NOFILE_IDS=nofile)
    res = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True,
                         env=env, cwd=tmp_path)
    return res.stdout


def test_baseline_loop_picks_the_newest_main_push_run_that_has_the_artifact(tmp_path):
    # newest (30) has no artifact, 20 and 10 do, 999 is the non-main/non-push decoy.
    out = _run_baseline_loop(tmp_path, "30 20 10", "20 10 999")
    assert "CHOSEN=20" in out
    assert "--baseline-summary baseline/summary.json" in out
    assert "::warning::" not in out


def test_baseline_loop_takes_the_newest_run_when_it_has_the_artifact(tmp_path):
    sub = tmp_path / "x"
    sub.mkdir()
    out = _run_baseline_loop(sub, "30 20 10", "30 20 10 999")
    assert "CHOSEN=30" in out


def test_baseline_loop_skips_an_artifact_without_summary_json(tmp_path):
    out = _run_baseline_loop(tmp_path, "30 20", "20 999", nofile="30")
    assert "CHOSEN=20" in out


def test_baseline_loop_warns_when_no_main_push_run_has_the_artifact(tmp_path):
    # Only the decoy (a non-main / non-push run) has an artifact: it must never be picked.
    out = _run_baseline_loop(tmp_path, "30 20 10", "999")
    assert "CHOSEN=none" in out
    assert "--baseline-summary" not in out
    assert ("::warning::no recent main push run has a mutation-report artifact; "
            "no failure is excused as pre-existing") in out
    out = _run_baseline_loop(tmp_path, "", "999")  # no runs at all
    assert "CHOSEN=none" in out and "::warning::no recent main push run" in out
