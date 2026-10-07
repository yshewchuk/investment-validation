"""Validate CI workflow configuration against conftest LOCAL_ONLY_MARKERS."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests.conftest import LOCAL_ONLY_MARKERS

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "tests.yml"
README_PATH = REPO_ROOT / "tests" / "README.md"


def test_workflow_triggers():
    """Assert that the workflow has the correct triggers."""
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    triggers = workflow.get("on", {})
    assert "push" in triggers
    assert "workflow_dispatch" in triggers
    assert triggers["push"].get("branches") == ["main"]


def test_workflow_concurrency():
    """Assert concurrency group and cancel-in-progress are set."""
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    concurrency = workflow.get("concurrency", {})
    assert concurrency.get("group") == (
        "tests-${{ github.event_name == 'pull_request' "
        "&& github.event.pull_request.number || github.ref }}"
    )
    assert concurrency.get("cancel-in-progress") is True


def test_workflow_marker_expression():
    """Assert that the marker names in the -m expression match LOCAL_ONLY_MARKERS."""
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    # Find the pytest command line
    steps = workflow.get("jobs", {}).get("test", {}).get("steps", [])
    pytest_step = next(s for s in steps if "pytest" in (s.get("run") or ""))
    pytest_run = pytest_step["run"]

    # Extract the marker expression from -m "..."
    match = re.search(r'-m\s+"([^"]+)"', pytest_run)
    assert match, f"Could not find -m expression in:\n{pytest_run}"

    marker_expr = match.group(1)

    # Parse the marker expression: should be "-m 'not X and not Y and not Z and not W'"
    # Extract all marker names
    markers_in_expr = set(re.findall(r"not\s+(\w+)", marker_expr))
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
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["test"]["steps"]
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
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["test"]["steps"]
    step = next(s for s in steps if s.get("name") == "Select PR test files")
    assert step["if"] == "github.event_name == 'pull_request'"
    assert "git diff -z --no-renames --name-only" in step["run"]
    assert "tools/mutation_pilot.py select-tests --changed-files" in step["run"]


def test_layout_ratchet_step_runs_only_on_pull_request_with_base_sha_env():
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["test"]["steps"]
    names = [s.get("name") for s in steps]
    step = steps[names.index("Check test layout ratchet")]
    assert names.index("Check test layout ratchet") < names.index("Run pytest")
    assert step["if"] == "github.event_name == 'pull_request'"
    assert step["env"]["BASE_SHA"] == "${{ github.event.pull_request.base.sha }}"
    assert step["run"] == 'python3 checks/test_layout_budget.py --base-ref "$BASE_SHA"'


def test_pytest_step_falls_back_to_tests_dir_and_has_selection_fallback():
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["test"]["steps"]
    pytest_step = next((s for s in steps if s.get("name") == "Run pytest"), None)
    assert pytest_step is not None, "no step named 'Run pytest' in the workflow"
    run = pytest_step["run"]
    for substring in ('TARGETS=("tests/")', "SELECTED[0]", "__ALL__", "No test files selected"):
        assert substring in run, substring


def test_upload_step_warns_instead_of_failing_on_no_junit():
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["test"]["steps"]
    step = next(s for s in steps if s.get("name") == "Upload test results")
    assert step["with"]["if-no-files-found"] == "warn"
