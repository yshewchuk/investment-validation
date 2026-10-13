"""Every state -- success, failure and neutral -- is posted as ONE check run.

Parses the shipped ``.github/workflows/mutation.yml``. ``report_status`` holds the
workflow's only ``checks: write`` permission and posts every state as a single
completed ``mutation/pr`` check run; there is no separate ``report_neutral`` job and
no commit status anywhere.

# packages: foundation, ops
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())["jobs"]


def _status_run():
    step = next(s for s in JOBS["report_status"]["steps"]
                if str(s.get("name", "")).startswith("Post mutation"))
    return step["run"]


def test_no_separate_neutral_job():
    assert "report_neutral" not in JOBS


def test_report_status_is_the_only_checks_writer_and_posts_a_check_run():
    assert JOBS["report_status"]["permissions"] == {"checks": "write"}
    # the commit-status permission key is spelled "statuses": no job holds it any more
    assert [n for n, j in JOBS.items() if "statuses" in (j.get("permissions") or {})] == []
    run = _status_run()
    assert "check-runs" in run
    assert "/statuses/" not in run
    checks_writers = [n for n, j in JOBS.items() if "checks" in (j.get("permissions") or {})]
    assert checks_writers == ["report_status"]
