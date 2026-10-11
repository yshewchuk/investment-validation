"""Validate CI workflow configuration against conftest LOCAL_ONLY_MARKERS.

Steps are looked up by name across every job, so these checks hold for the
single-job workflow and for the sharded one (plan -> shard matrix -> `test`
aggregator; tests/shards.toml, tools/ci_shards.py).
"""
from __future__ import annotations

# packages: ops

import os
import re
import subprocess
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


def _stub_bin(tmp_path: Path, python3_script: str) -> dict[str, str]:
    """Env whose `python3` is a stub (the shard tool) and `python` records its argv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").write_text("#!/bin/sh\n" + python3_script + "\n")
    (bin_dir / "python").write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/pytest_calls"\n')
    for stub in ("python3", "python"):
        (bin_dir / stub).chmod(0o755)
    return dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")


def _run_bash(run: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-e", "-c", run], env=env, capture_output=True,
                          text=True, timeout=60)


def test_pytest_step_passes_the_shard_file_list_to_pytest():
    run = _step(_workflow(), "Run pytest")["run"]
    assert "tools/ci_shards.py files" in run and "--selection" in run
    pytest_line = next(ln for ln in run.splitlines() if "-m pytest" in ln)
    assert '"${FILES[@]}"' in pytest_line


def test_failing_shard_list_command_fails_the_step_and_runs_no_pytest(tmp_path):
    run = _step(_workflow(), "Run pytest")["run"]
    env = dict(_stub_bin(tmp_path, "exit 3"), SHARD="ops", SELECTION_FILE="")
    proc = _run_bash(run, env)
    assert proc.returncode != 0
    assert not (tmp_path / "pytest_calls").exists()


def test_empty_shard_list_skips_pytest_and_a_nonempty_one_runs_it(tmp_path):
    run = _step(_workflow(), "Run pytest")["run"]
    env = dict(_stub_bin(tmp_path, "exit 0"), SHARD="ops", SELECTION_FILE="")
    proc = _run_bash(run, env)
    assert proc.returncode == 0 and "No test files" in proc.stdout
    assert not (tmp_path / "pytest_calls").exists()
    other = tmp_path / "other"
    other.mkdir()
    env = dict(_stub_bin(other, "echo tests/a/test_x.py"), SHARD="ops", SELECTION_FILE="")
    assert _run_bash(run, env).returncode == 0
    assert "tests/a/test_x.py" in (other / "pytest_calls").read_text()


def test_failing_matrix_command_fails_the_plan_step(tmp_path):
    run = _step(_workflow(), "Print shard matrix")["run"]
    output = tmp_path / "out"
    env = dict(_stub_bin(tmp_path, "exit 3"), GITHUB_OUTPUT=str(output))
    assert _run_bash(run, env).returncode != 0
    assert not output.exists() or "matrix=" not in output.read_text()


def test_upload_step_warns_instead_of_failing_on_no_junit():
    step = _step(_workflow(), "Upload test results")
    assert step["with"]["if-no-files-found"] == "warn"


def test_shard_job_needs_the_plan_job_and_runs_the_plan_matrix():
    shard = _workflow()["jobs"]["shard"]
    assert shard["needs"] in ("plan", ["plan"])
    assert shard["strategy"]["fail-fast"] is False
    assert "fromJSON(needs.plan.outputs.matrix)" in shard["strategy"]["matrix"]


@pytest.mark.parametrize(("plan", "shard", "ok"), [
    ("success", "success", True),
    ("success", "failure", False),
    ("success", "skipped", False),
    ("success", "cancelled", False),
    ("failure", "success", False),
    ("skipped", "skipped", False),
])
def test_aggregator_named_test_succeeds_only_when_plan_and_every_shard_succeeded(
        plan, shard, ok):
    aggregator = _workflow()["jobs"]["test"]
    assert set(aggregator["needs"]) == {"plan", "shard"}
    assert aggregator["if"] == "${{ always() }}"
    run = "\n".join(s.get("run", "") for s in aggregator["steps"])
    results = {"plan": plan, "shard": shard}
    run = re.sub(r"\$\{\{\s*needs\.(\w+)\.result\s*\}\}", lambda m: results[m.group(1)], run)
    assert "${{" not in run
    assert (_run_bash(run, dict(os.environ)).returncode == 0) is ok