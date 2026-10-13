"""Descriptions of the unclassified ``report_status`` branches of the gremlins workflow.

Red comes only from a classified failure or the PR-closed placeholder guard; a report job
that did not succeed without a trusted classified verdict is infra (neutral, grey), and a
classified ``success`` never overrides a report job that did not succeed. The decision
script is run under bash for each pinned input (up to the final check-run post) and compared
with an independent oracle; the exhaustive input-combination table now lives in the exec
test (``test_mutation_pr_status_workflow_exec.py``).

# packages: foundation, ops
"""
from __future__ import annotations

from pathlib import Path

import yaml

from tests.v2.integration.mutation_status_support import decide

ROOT = Path(__file__).resolve().parents[3]
JOBS = yaml.safe_load((ROOT / ".github" / "workflows" / "mutation.yml").read_text())["jobs"]


def test_descriptions_of_the_unclassified_branches():
    for result in ("failure", "cancelled", "skipped"):
        assert decide(REPORT_RESULT=result) == (
            "neutral", f"gremlins: report job {result} before classification -- infra; not a pass")
        assert decide(CLASSIFIED_STATE="success", CLASSIFIED_DESC="x", REPORT_RESULT=result) == (
            "neutral", f"gremlins: report job {result} before classification -- infra; not a pass")
    assert decide() == ("success", "gremlins: report published for 2 module(s)")
    state, desc = decide(SKIPPED_COUNT="1")
    assert state == "failure" and "skipped (PR closed)" in desc
    assert decide(MODULES="[]")[0] == "success"
    assert decide(CLASSIFIED_STATE="failure", CLASSIFIED_DESC="PR-owned problem in a",
                  REPORT_RESULT="failure") == ("failure", "PR-owned problem in a")