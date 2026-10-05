"""Validate CI workflow configuration against conftest LOCAL_ONLY_MARKERS."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
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


def _workflow_step_run(name):
    """The shipped shell body of one `test`-job step, never a hand copy."""
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)
    steps = workflow["jobs"]["test"]["steps"]
    return next(s for s in steps if s.get("name") == name)["run"]


def _run_selection_then_pytest(tmp_path, selector_rc, pytest_rc):
    """Runs BOTH shipped step scripts back to back in bash, with `git`,
    `python3` (the selector) and `python` (pytest) stubbed on PATH, and
    returns (select_rc, pytest_rc_seen, channel_text, pytest_args)."""
    work = Path(tempfile.mkdtemp(dir=tmp_path))  # one isolated job run per call
    runner_temp = work / "runner-temp"
    runner_temp.mkdir()
    bindir = work / "bin"
    bindir.mkdir()
    channel = runner_temp / "pr-test-selection.txt"
    args_file = work / "pytest-args.txt"

    (bindir / "git").write_text("#!/bin/sh\nexit 0\n")
    # Simulates a selector that dies partway: it prints one narrowed path (the
    # thing that must NOT survive as a silent narrow selection) and exits
    # nonzero, like an unhandled crash in `select-tests`.
    (bindir / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = "select-tests" ]; then\n'
        '  echo "tests/test_only_changed_area.py"\n'
        f"  exit {selector_rc}\n"
        "fi\n"
        "exit 0\n"
    )
    (bindir / "python").write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{args_file}"\n'
        f"exit {pytest_rc}\n"
    )
    for p in (bindir / "git", bindir / "python3", bindir / "python"):
        p.chmod(0o755)

    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
               RUNNER_TEMP=str(runner_temp), BASE_SHA="0" * 40)
    select = subprocess.run(["bash", "-c", _workflow_step_run("Select PR test files")],
                            capture_output=True, text=True, env=env, cwd=work)
    # The `Run pytest` step's SELECTION_FILE is a GitHub expression; for a
    # pull_request run it resolves to this same channel path under runner.temp.
    pytest_env = dict(env, SELECTION_FILE=str(channel))
    run = subprocess.run(["bash", "-c", _workflow_step_run("Run pytest")],
                         capture_output=True, text=True, env=pytest_env, cwd=work)
    args = args_file.read_text().splitlines() if args_file.exists() else []
    text = channel.read_text() if channel.exists() else None
    return select.returncode, run.returncode, text, args


def test_selector_crash_falls_back_to_all_sentinel_and_runs_full_suite(tmp_path):
    """The `test` job must not die in the selection step: a nonzero selector
    writes the exact __ALL__ sentinel to the channel the next step reads (over
    any partial narrowed output) and pytest then runs the full `tests/` dir.
    A pytest failure still propagates -- the fallback covers the selector, not
    the suite."""
    rc, pytest_rc, channel_text, args = _run_selection_then_pytest(tmp_path, 1, 0)
    assert rc == 0, "the selection step must not fail the job on a selector crash"
    assert channel_text == "__ALL__\n"
    assert args[-1] == "tests/"  # full suite, not the narrowed partial path
    assert "tests/test_only_changed_area.py" not in args
    assert pytest_rc == 0

    rc, pytest_rc_fail, _, _ = _run_selection_then_pytest(tmp_path, 1, 1)
    assert rc == 0 and pytest_rc_fail == 1  # never a green run from a red pytest

    rc, pytest_ok, stale_channel, stale_args = _run_selection_then_pytest(tmp_path, 0, 0)
    assert rc == 0 and pytest_ok == 0
    # A selector that exits 0 is untouched by the fallback: its own narrowed
    # output, not __ALL__, still reaches pytest.
    assert stale_channel == "tests/test_only_changed_area.py\n"
    assert stale_args[-1] == "tests/test_only_changed_area.py"
