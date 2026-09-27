"""S4C Part 3: the two natively-owned calendar/moves refresh job kinds.

``test_provider_failure_code_orders_mixed_kinds`` moved to
``tests/test_v2_ops_unit_receipts.py`` (P6 slice-4c split, Part 0) along with
the primitive it tests. The cross-check against ``nightly.py``'s
``COMPUTED_MOVES_REFRESH_ACTION``/``FORWARD_CALENDAR_REFRESH_ACTION`` imports
is Part 4's own addition (nightly.py does not import from this module until
then); this file keeps only the ``registry()``/``checkpoint_contract``
assertions, which need nothing beyond ``stages.py``.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.foundation import to_document
from engine.v2.ops import calendar_moves_jobs
from engine.v2.ops.calendar_moves_jobs import (
    COMPUTED_MOVES_REFRESH_ACTION,
    COMPUTED_MOVES_RESULT_PATH,
    COMPUTED_MOVES_RESULT_SCHEMA,
    FORWARD_CALENDAR_REFRESH_ACTION,
    FORWARD_CALENDAR_RESULT_PATH,
    FORWARD_CALENDAR_RESULT_SCHEMA,
    CalendarMovesParameters,
    calendar_moves_parameter_problems,
)
from engine.v2.ops.errors import OpsError
from engine.v2.ops.incremental_data import RefreshCallbackResult
from engine.v2.ops.stages import registry


def _params(**overrides):
    base = dict(expected_ids=("AAPL",), parent_snapshot_id="snap-parent",
               refresh_plan_hash="sha256:" + "c" * 64, catalog_path="/ops/catalog.sqlite",
               objects_root="/ops/objects", scope="shadow", expected_head_generation=3,
               expected_head_snapshot_id="snap-parent", as_of="2026-09-27")
    base.update(overrides)
    return CalendarMovesParameters(**base)


def _result(**overrides):
    base = dict(status="complete", completed_ids=("AAPL",), coverage_advanced=True,
               parent_snapshot_id="snap-parent", refresh_plan_hash="sha256:" + "c" * 64,
               candidate_snapshot_id="snap-new")
    base.update(overrides)
    return RefreshCallbackResult(**base)


def _unreachable_callback(parameters, root):
    raise AssertionError("the store callback must never run before parameters validate")


# --------------------------------------------------------------------------
# both kinds are registered with the shared refresh-evidence checkpoint
# --------------------------------------------------------------------------


def test_both_kinds_are_registered_and_their_contracts_agree():
    kinds = registry()
    assert {COMPUTED_MOVES_REFRESH_ACTION, FORWARD_CALENDAR_REFRESH_ACTION} <= set(
        kinds.names())
    computed = kinds.get(COMPUTED_MOVES_REFRESH_ACTION)
    forward = kinds.get(FORWARD_CALENDAR_REFRESH_ACTION)
    assert computed.checkpoint_contract == COMPUTED_MOVES_RESULT_SCHEMA
    assert forward.checkpoint_contract == FORWARD_CALENDAR_RESULT_SCHEMA
    assert computed.worker == COMPUTED_MOVES_REFRESH_ACTION
    assert forward.worker == FORWARD_CALENDAR_REFRESH_ACTION
    assert computed.namespaces == frozenset({"shadow", "smoke"})
    assert forward.namespaces == frozenset({"shadow", "smoke"})


# --------------------------------------------------------------------------
# calendar_moves_parameter_problems: one refusal per bad field, pure function
# --------------------------------------------------------------------------


def test_expected_ids_empty_is_a_problem():
    problems = calendar_moves_parameter_problems(None, _params(expected_ids=()))
    assert any("expected_ids" in problem for problem in problems)


def test_expected_ids_duplicate_is_a_problem():
    problems = calendar_moves_parameter_problems(
        None, _params(expected_ids=("AAPL", "AAPL")))
    assert any("expected_ids" in problem for problem in problems)


def test_as_of_not_iso_is_a_problem():
    problems = calendar_moves_parameter_problems(None, _params(as_of="not-a-date"))
    assert any("as_of" in problem for problem in problems)


def test_as_of_none_is_not_a_problem_here():
    # Required-ness of as_of is each store's own responsibility
    # (run_forward_calendar_refresh/_as_of_day), enforced again before any
    # I/O -- see the module docstring on calendar_moves_parameter_problems.
    problems = calendar_moves_parameter_problems(None, _params(as_of=None))
    assert problems == ()


def test_partial_plan_binding_is_a_problem():
    problems = calendar_moves_parameter_problems(
        None, _params(parent_snapshot_id="snap-parent", refresh_plan_hash="", provider_calls=0))
    assert any("refresh_plan_hash" in problem for problem in problems)


def test_valid_parameters_have_no_problems():
    assert calendar_moves_parameter_problems(None, _params()) == ()


# --------------------------------------------------------------------------
# _decode: strict document decoding, refused before any store callback runs
# --------------------------------------------------------------------------


def test_run_computed_moves_worker_refuses_unknown_field_before_any_io(tmp_path):
    document = to_document(_params())
    document["bogus_field"] = 1
    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_computed_moves_worker(
            document, tmp_path, refresh_callback=_unreachable_callback)
    assert exc_info.value.code == "INVALID_REQUEST"
    assert list(tmp_path.iterdir()) == []


def test_run_computed_moves_worker_refuses_missing_required_field_before_any_io(tmp_path):
    document = to_document(_params())
    del document["expected_ids"]
    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_computed_moves_worker(
            document, tmp_path, refresh_callback=_unreachable_callback)
    assert exc_info.value.code == "INVALID_REQUEST"
    assert list(tmp_path.iterdir()) == []


def test_run_forward_calendar_worker_refuses_bool_where_int_expected_before_any_io(tmp_path):
    document = to_document(_params())
    document["expected_head_generation"] = True
    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_forward_calendar_worker(
            document, tmp_path, refresh_callback=_unreachable_callback)
    assert exc_info.value.code == "INVALID_REQUEST"
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# job-level happy path (injected callback -- no real catalog or provider)
# --------------------------------------------------------------------------


def test_run_computed_moves_worker_happy_path(tmp_path):
    params = _params()
    result = _result()

    def _callback(parameters, root):
        assert parameters == params
        assert root == tmp_path
        return result

    output = calendar_moves_jobs.run_computed_moves_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert output["completed_ids"] == ["AAPL"]
    assert output["no_work"] is False
    assert output["outputs"] == [{"name": COMPUTED_MOVES_REFRESH_ACTION,
                                  "path": COMPUTED_MOVES_RESULT_PATH,
                                  "schema": COMPUTED_MOVES_RESULT_SCHEMA}]
    written = json.loads((tmp_path / COMPUTED_MOVES_RESULT_PATH).read_bytes())
    assert written["status"] == "complete"
    assert written["candidate_snapshot_id"] == "snap-new"


def test_run_forward_calendar_worker_happy_path(tmp_path):
    params = _params()
    result = _result()

    def _callback(parameters, root):
        return result

    output = calendar_moves_jobs.run_forward_calendar_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert output["completed_ids"] == ["AAPL"]
    assert output["outputs"] == [{"name": FORWARD_CALENDAR_REFRESH_ACTION,
                                  "path": FORWARD_CALENDAR_RESULT_PATH,
                                  "schema": FORWARD_CALENDAR_RESULT_SCHEMA}]
    written = json.loads((tmp_path / FORWARD_CALENDAR_RESULT_PATH).read_bytes())
    assert written["status"] == "complete"


def test_run_computed_moves_worker_happy_path_tolerates_reordered_coverage(tmp_path):
    """expected_ids is a caller-supplied, unsorted tuple; the store returns
    completed_ids sorted. Same ticker set, different order -- must still
    validate as complete, not fail a commit that already happened."""
    params = _params(expected_ids=("MSFT", "AAPL"))
    result = _result(completed_ids=("AAPL", "MSFT"))

    def _callback(parameters, root):
        return result

    output = calendar_moves_jobs.run_computed_moves_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert sorted(output["completed_ids"]) == ["AAPL", "MSFT"]


def test_run_forward_calendar_worker_happy_path_tolerates_reordered_coverage(tmp_path):
    params = _params(expected_ids=("MSFT", "AAPL"))
    result = _result(completed_ids=("AAPL", "MSFT"))

    def _callback(parameters, root):
        return result

    output = calendar_moves_jobs.run_forward_calendar_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert sorted(output["completed_ids"]) == ["AAPL", "MSFT"]


def test_run_computed_moves_worker_rejects_a_missing_completed_id(tmp_path):
    """Same coverage comparator as the reordered-coverage tests above, but a
    genuinely corrupt result: completed_ids is missing one of the two
    expected ids. Proves the set-based check still rejects a real mismatch,
    not just order."""
    params = _params(expected_ids=("MSFT", "AAPL"))
    result = _result(completed_ids=("AAPL",))

    def _callback(parameters, root):
        return result

    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_computed_moves_worker(
            to_document(params), tmp_path, refresh_callback=_callback)
    assert exc_info.value.code == "VALIDATION_FAILED"


# --------------------------------------------------------------------------
# a cached re-run comes back noop
# --------------------------------------------------------------------------


def test_run_computed_moves_worker_cached_rerun_is_a_true_noop(tmp_path):
    params = _params()
    result = _result(status="noop", coverage_advanced=False, candidate_snapshot_id=None)

    def _callback(parameters, root):
        return result

    output = calendar_moves_jobs.run_computed_moves_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert output["no_work"] is True
    assert output["completed_ids"] == ["AAPL"]
    written = json.loads((tmp_path / COMPUTED_MOVES_RESULT_PATH).read_bytes())
    assert written["status"] == "noop"
    assert "candidate_snapshot_id" not in written or written["candidate_snapshot_id"] is None


def test_run_forward_calendar_worker_cached_rerun_is_a_true_noop(tmp_path):
    params = _params()
    result = _result(status="noop", coverage_advanced=False, candidate_snapshot_id=None)

    def _callback(parameters, root):
        return result

    output = calendar_moves_jobs.run_forward_calendar_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert output["no_work"] is True


# --------------------------------------------------------------------------
# a mismatched or non-complete callback result still fails typed, not silent
# --------------------------------------------------------------------------


def test_run_computed_moves_worker_refuses_a_result_bound_to_a_different_plan(tmp_path):
    params = _params()
    result = _result(parent_snapshot_id="snap-other")

    def _callback(parameters, root):
        return result

    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_computed_moves_worker(
            to_document(params), tmp_path, refresh_callback=_callback)
    assert exc_info.value.code == "STALE_EXPECTATION"


def test_run_forward_calendar_worker_maps_a_transient_status_to_its_typed_code(tmp_path):
    params = _params()
    result = _result(status="transient", coverage_advanced=False, candidate_snapshot_id=None)

    def _callback(parameters, root):
        return result

    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_forward_calendar_worker(
            to_document(params), tmp_path, refresh_callback=_callback)
    assert exc_info.value.code == "TRANSIENT_SOURCE"
