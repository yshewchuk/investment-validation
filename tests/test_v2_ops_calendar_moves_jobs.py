"""S4C rewrite: the two v2-owned calendar/moves refresh jobs.

Both workers call their own v2 stores through an injected callback -- the
provider fetchers are bound by ``incremental_data._load_*_refresh_callback``
and resolved lazily, exactly like the daily refresh. These tests never
monkeypatch ``run_computed_moves_refresh``/``run_forward_calendar_refresh``
(the old branch's mistake); the callback is injected as data instead.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from engine.v2.ops import incremental_data, worker
from engine.v2.ops.calendar_moves_jobs import (
    COMPUTED_MOVES_REFRESH_ACTION,
    COMPUTED_MOVES_RESULT_PATH,
    COMPUTED_MOVES_RESULT_SCHEMA,
    FORWARD_CALENDAR_REFRESH_ACTION,
    FORWARD_CALENDAR_RESULT_PATH,
    FORWARD_CALENDAR_RESULT_SCHEMA,
    CalendarMovesParameters,
    calendar_moves_parameter_problems,
    computed_moves_job_kind,
    forward_calendar_job_kind,
    run_computed_moves_worker,
    run_forward_calendar_worker,
)
from engine.v2.ops.errors import OpsError
from engine.v2.ops.incremental_data import RefreshCallbackResult
from engine.v2.ops.nightly import (
    COMPUTED_MOVES_REFRESH_ACTION as NIGHTLY_COMPUTED_MOVES,
    FORWARD_CALENDAR_REFRESH_ACTION as NIGHTLY_FORWARD_CALENDAR,
)
from engine.v2.ops.stages import registry

PLAN_HASH = "sha256:" + "a" * 64


def _params(**changes) -> dict:
    fields = dict(expected_ids=("x",), parent_snapshot_id="snap-parent",
                  refresh_plan_hash=PLAN_HASH, provider_calls=0,
                  catalog_path="/catalog", objects_root="/objects", scope="shadow",
                  expected_head_generation=0, expected_head_snapshot_id="snap-parent")
    fields.update(changes)
    return fields


def _result(status="complete", completed=("x",)) -> RefreshCallbackResult:
    return RefreshCallbackResult(
        status=status, completed_ids=completed, coverage_advanced=status == "complete",
        parent_snapshot_id="snap-parent", refresh_plan_hash=PLAN_HASH,
        candidate_snapshot_id="snap-new" if status == "complete" else None)


def test_forward_calendar_parameters_reject_bad_as_of():
    params = CalendarMovesParameters(expected_ids=("x",), as_of="not-a-date")
    assert "as_of must be an ISO date" in calendar_moves_parameter_problems(None, params)


def test_forward_calendar_parameters_reject_empty_ids():
    params = CalendarMovesParameters(expected_ids=(), as_of="2026-09-18")
    assert "expected_ids must contain 1..4096 request ids" in calendar_moves_parameter_problems(
        None, params)


def test_plan_binding_is_validated_only_when_supplied():
    bare = CalendarMovesParameters(expected_ids=("x",))
    assert calendar_moves_parameter_problems(None, bare) == ()
    bound = CalendarMovesParameters(expected_ids=("x",), parent_snapshot_id="snap",
                                    refresh_plan_hash=PLAN_HASH, provider_calls=3)
    assert calendar_moves_parameter_problems(None, bound) == ()
    broken = CalendarMovesParameters(expected_ids=("x",), parent_snapshot_id="snap",
                                     refresh_plan_hash="not-a-hash", provider_calls=3)
    assert "refresh_plan_hash must be a sha256 content hash" in \
        calendar_moves_parameter_problems(None, broken)


def test_both_kinds_are_registered_and_their_contracts_agree():
    names = registry().names()
    assert FORWARD_CALENDAR_REFRESH_ACTION in names
    assert COMPUTED_MOVES_REFRESH_ACTION in names
    # One source of truth: nightly's DAG constants are these kinds' constants.
    assert NIGHTLY_FORWARD_CALENDAR == FORWARD_CALENDAR_REFRESH_ACTION
    assert NIGHTLY_COMPUTED_MOVES == COMPUTED_MOVES_REFRESH_ACTION
    assert forward_calendar_job_kind().checkpoint_contract == FORWARD_CALENDAR_RESULT_SCHEMA
    assert computed_moves_job_kind().checkpoint_contract == COMPUTED_MOVES_RESULT_SCHEMA


def test_calendar_moves_jobs_never_imports_legacy_pulls():
    """Negative control: an ast scan proves this module reaches no legacy pull."""
    from engine.v2.ops import calendar_moves_jobs

    tree = ast.parse(Path(calendar_moves_jobs.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    for module in imported:
        assert not module.startswith("engine.data.pulls.forward_calendar")
        assert not module.startswith("engine.data.pulls.computed_moves")


def test_unknown_worker_name_still_refused(tmp_path):
    """Negative control: the two new branches must not swallow the catch-all."""
    with pytest.raises(ValueError, match="unsupported worker"):
        worker.dispatch("bogus_calendar_job", {}, tmp_path)


def test_loaders_bind_the_fetchers_without_touching_them():
    computed = incremental_data._load_computed_moves_refresh_callback()
    calendar = incremental_data._load_forward_calendar_refresh_callback()
    assert callable(computed.keywords["fetcher"])
    assert callable(calendar.keywords["nasdaq_fetcher"])
    assert callable(calendar.keywords["earnings_fetcher"])


def test_computed_moves_worker_writes_result_and_reports_no_work(tmp_path):
    result = run_computed_moves_worker(
        _params(), tmp_path, refresh_callback=lambda parameters, root: _result(
            status="noop", completed=("x",)))

    [output] = result["outputs"]
    assert output["name"] == COMPUTED_MOVES_REFRESH_ACTION
    assert output["path"] == COMPUTED_MOVES_RESULT_PATH
    written = json.loads((tmp_path / COMPUTED_MOVES_RESULT_PATH).read_text())
    assert written["status"] == "noop"
    assert result["completed_ids"] == ["x"]
    assert result["no_work"] is True


def test_forward_calendar_worker_writes_its_own_result_path(tmp_path):
    result = run_forward_calendar_worker(
        _params(), tmp_path, refresh_callback=lambda parameters, root: _result())

    [output] = result["outputs"]
    assert output["name"] == FORWARD_CALENDAR_REFRESH_ACTION
    assert output["path"] == FORWARD_CALENDAR_RESULT_PATH
    written = json.loads((tmp_path / FORWARD_CALENDAR_RESULT_PATH).read_text())
    assert written["candidate_snapshot_id"] == "snap-new"
    assert result["no_work"] is False


def test_worker_maps_a_failed_status_to_its_own_message(tmp_path):
    def failed(parameters, root):
        return RefreshCallbackResult(
            status="failed", completed_ids=(), coverage_advanced=False,
            parent_snapshot_id="snap-parent", refresh_plan_hash=PLAN_HASH)

    with pytest.raises(OpsError) as exc:
        run_computed_moves_worker(_params(), tmp_path, refresh_callback=failed)
    assert exc.value.code == "WORKER_FAILED"
    assert exc.value.problem.message == "computed moves refresh did not produce complete coverage"

    with pytest.raises(OpsError) as other:
        run_forward_calendar_worker(_params(), tmp_path, refresh_callback=failed)
    assert other.value.problem.message == \
        "forward calendar refresh did not produce complete coverage"


def test_worker_rejects_a_result_bound_to_another_plan(tmp_path):
    def stale(parameters, root):
        return RefreshCallbackResult(
            status="noop", completed_ids=("x",), coverage_advanced=False,
            parent_snapshot_id="some-other-parent", refresh_plan_hash=PLAN_HASH)

    with pytest.raises(OpsError) as exc:
        run_forward_calendar_worker(_params(), tmp_path, refresh_callback=stale)
    assert exc.value.code == "STALE_EXPECTATION"
