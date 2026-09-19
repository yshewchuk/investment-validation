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
    assert concurrency.get("group") == "tests-${{ github.ref }}"
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
