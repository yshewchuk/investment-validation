"""Validate CI workflow configuration against conftest LOCAL_ONLY_MARKERS.

Steps are looked up by name across every job, so these checks hold for the
single-job workflow and for the sharded one (plan -> shard matrix -> `test`
aggregator; tests/shards.toml, tools/ci_shards.py).
"""
from __future__ import annotations

# packages: ops

import re
from pathlib import Path

import pytest
import yaml

from tests.conftest import LOCAL_ONLY_MARKERS

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "tests.yml"
README_PATH = REPO_ROOT / "tests" / "README.md"


def _workflow() -> dict:
    with open(WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


def _job_and_step(workflow: dict, name: str) -> tuple[dict, dict]:
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == name:
                return job, step
    raise AssertionError(f"no step named {name!r} in any job of the workflow")


def _step(workflow: dict, name: str) -> dict:
    return _job_and_step(workflow, name)[1]


def test_workflow_triggers():
    """Assert that the workflow has the correct triggers."""
    triggers = _workflow().get("on", {})
    assert "push" in triggers
    assert "workflow_dispatch" in triggers
    assert triggers["push"].get("branches") == ["main"]


def test_workflow_concurrency():
    """Assert concurrency group and cancel-in-progress are set."""
    concurrency = _workflow().get("concurrency", {})
    assert concurrency.get("group") == (
        "tests-${{ github.event_name == 'pull_request' "
        "&& github.event.pull_request.number || github.ref }}"
    )
    assert concurrency.get("cancel-in-progress") is True


def test_workflow_marker_expression():
    """Assert that the marker names in the -m expression match LOCAL_ONLY_MARKERS."""
    pytest_run = _step(_workflow(), "Run pytest")["run"]

    match = re.search(r'-m\s+"([^"]+)"', pytest_run)
    assert match, f"Could not find -m expression in:\n{pytest_run}"

    markers_in_expr = set(re.findall(r"not\s+(\w+)", match.group(1)))
    markers_in_config = set(LOCAL_ONLY_MARKERS.keys())

    assert markers_in_expr == markers_in_config, (
        f"Markers in workflow ({markers_in_expr}) do not match "
        f"LOCAL_ONLY_MARKERS ({markers_in_config})"
    )


def test_readme_documents_all_markers():
    """Assert that tests/README.md documents each marker name."""
    with open(README_PATH) as f:
        readme_content = f.read()

    for marker_name in LOCAL_ONLY_MARKERS.keys():
        assert marker_name in readme_content, (
            f"Marker '{marker_name}' from LOCAL_ONLY_MARKERS is not mentioned in tests/README.md"
        )


def test_checkout_fetch_depth_is_conditional_on_pull_request():
    job, _ = _job_and_step(_workflow(), "Select PR test files")
    steps = job["steps"]
    assert steps[0]["uses"].startswith("actions/checkout@")
    # A bare, unquoted 0 is FALSY in GitHub Actions expressions, so
    # `condition && 0 || 1` always evaluates to 1 regardless of `condition`
    # (0 is falsy, so `0 || 1` falls through to 1). The 0 must be a quoted
    # string ('0') to survive the || fallback -- matching the working
    # pattern already used in .github/workflows/mutation.yml and
    # mutation-mutmut.yml.
    fetch_depth = steps[0]["with"]["fetch-depth"]
    assert fetch_depth == "${{ github.event_name == 'pull_request' && '0' || 1 }}"
    assert "&& '0'" in fetch_depth
    assert "&& 0 " not in fetch_depth


def test_select_step_only_runs_on_pull_request_and_diffs_against_base():
    step = _step(_workflow(), "Select PR test files")
    assert step["if"] == "github.event_name == 'pull_request'"
    assert "git diff -z --no-renames --name-only" in step["run"]
    assert "tools/mutation_pilot.py select-tests --changed-files" in step["run"]


def test_layout_ratchet_step_runs_only_on_pull_request_with_base_sha_env():
    workflow = _workflow()
    ratchet_job, step = _job_and_step(workflow, "Check test layout ratchet")
    pytest_job, _ = _job_and_step(workflow, "Run pytest")
    if ratchet_job is pytest_job:
        names = [s.get("name") for s in ratchet_job["steps"]]
        assert names.index("Check test layout ratchet") < names.index("Run pytest")
    assert step["if"] == "github.event_name == 'pull_request'"
    assert step["env"]["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert step["run"] == 'python3 checks/test_layout_budget.py --base-ref "$BASE_SHA"'


def test_pytest_step_falls_back_to_tests_dir_and_has_selection_fallback():
    run = _step(_workflow(), "Run pytest")["run"]
    if "tools/ci_shards.py files" in run:
        # Sharded: tools/ci_shards.py owns __ALL__ / empty-selection handling.
        for substring in ("--selection", "No test files", "pytest"):
            assert substring in run, substring
        return
    for substring in ('TARGETS=("tests/")', "SELECTED[0]", "__ALL__", "No test files selected"):
        assert substring in run, substring


def test_upload_step_warns_instead_of_failing_on_no_junit():
    step = _step(_workflow(), "Upload test results")
    assert step["with"]["if-no-files-found"] == "warn"


def test_sharded_workflow_aggregator_named_test_needs_plan_and_every_shard_job():
    jobs = _workflow()["jobs"]
    if "plan" not in jobs:
        pytest.skip("single-job workflow (pre-shard)")
    aggregator = jobs["test"]
    assert set(aggregator["needs"]) == {"plan", "shard"}
    assert aggregator["if"] == "${{ always() }}"
    run = "\n".join(s.get("run", "") for s in aggregator["steps"])
    assert "needs.plan.result" in run and "needs.shard.result" in run
    assert jobs["shard"]["strategy"]["fail-fast"] is False
    assert "fromJSON(needs.plan.outputs.matrix)" in jobs["shard"]["strategy"]["matrix"]