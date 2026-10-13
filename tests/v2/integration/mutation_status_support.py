"""Shared helpers for the gremlins report_status tests."""
# packages: foundation, ops
from __future__ import annotations

import itertools
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())["jobs"]


def status_step():
    """The ``report_status`` step whose name starts with ``Post mutation status``."""
    return next(s for s in JOBS["report_status"]["steps"]
                if str(s.get("name", "")).startswith("Post mutation status"))


def decide(**env):
    """Run the shipped decision script (everything before its first ``gh api``)."""
    run = status_step()["run"]
    script = ("set -euo pipefail\n"
              + run[:run.index("gh api")].rsplit("\n", 1)[0]
              + '\necho "${state}|${desc}"\n')
    full = dict(os.environ, MODULES='["a","b"]', REPORT_RESULT="success", SKIPPED_COUNT="0",
                CLASSIFIED_STATE="", CLASSIFIED_DESC="")
    full.update(env)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         check=True, env=full).stdout.strip()
    return tuple(out.split("|", 1))


def expected_posted(modules, skipped, classified, report):
    """Independent oracle: the decision table of tests/README.md (skipped '' means 0)."""
    skipped = skipped or "0"
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


def report_status_combos():
    """Every input combination the exhaustive report_status tests iterate over."""
    return itertools.product(["[]", '["a","b"]', ""], ["0", "2"],
                             ["", "success", "failure", "neutral"],
                             ["success", "failure", "cancelled", "skipped"])