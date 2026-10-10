"""S4C Part 3/4: the two natively-owned calendar/moves refresh job kinds.

``test_provider_failure_code_orders_mixed_kinds`` moved to
``tests/test_v2_ops_unit_receipts.py`` (P6 slice-4c split, Part 0) along with
the primitive it tests. ``test_nightly_py_imports_the_same_action_constant``
below is Part 4's own addition, carved out per this file's own prior note:
nightly.py now imports COMPUTED_MOVES_REFRESH_ACTION from this module to
wire the stage into its GRAPH, and this cross-check proves that import stays
consistent (a stale, hand-copied constant in nightly.py would fail it).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.foundation import to_document
from engine.v2.ops import calendar_moves_jobs, forward_calendar_store, incremental_data
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
from engine.v2.ops.nightly import COMPUTED_MOVES_REFRESH_ACTION as NIGHTLY_COMPUTED_MOVES
from engine.v2.ops.stores.refresh_contracts import REFRESH_INPUT_DOCUMENT_NAMES
from engine.v2.ops.stages import registry


def _params(**overrides):
    base = dict(expected_ids=("AAPL",), parent_snapshot_id="snap-parent",
               refresh_plan_hash="sha256:" + "c" * 64, catalog_path="/ops/catalog.sqlite",
               objects_root="/ops/objects", scope="shadow", expected_head_generation=3,
               expected_head_snapshot_id="snap-parent", as_of="2026-09-27")
    base.update(overrides)
    return CalendarMovesParameters(**base)


def _job(**overrides):
    base = dict(provider_budget_ref=None)
    base.update(overrides)
    return SimpleNamespace(**base)


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
    for action, schema in ((COMPUTED_MOVES_REFRESH_ACTION, COMPUTED_MOVES_RESULT_SCHEMA),
                           (FORWARD_CALENDAR_REFRESH_ACTION, FORWARD_CALENDAR_RESULT_SCHEMA)):
        assert action in kinds.names()
        kind = kinds.get(action)
        assert kind.checkpoint_contract == schema
        assert kind.worker == action
        assert kind.namespaces == frozenset({"shadow", "smoke"})


def test_nightly_py_imports_the_same_action_constant():
    """S4C Part 4: nightly.py's GRAPH/_NATIVE_ACTION_STAGES wiring imports
    this exact constant, never a second hand-copied string literal."""
    assert NIGHTLY_COMPUTED_MOVES == COMPUTED_MOVES_REFRESH_ACTION


# --------------------------------------------------------------------------
# calendar_moves_parameter_problems: one refusal per bad field, pure function
# --------------------------------------------------------------------------


def test_expected_ids_empty_is_a_problem():
    problems = calendar_moves_parameter_problems(_job(), _params(expected_ids=()))
    assert any("expected_ids" in problem for problem in problems)


def test_expected_ids_duplicate_is_a_problem():
    problems = calendar_moves_parameter_problems(
        _job(), _params(expected_ids=("AAPL", "AAPL")))
    assert any("expected_ids" in problem for problem in problems)


def test_as_of_not_iso_is_a_problem():
    problems = calendar_moves_parameter_problems(_job(), _params(as_of="not-a-date"))
    assert any("as_of" in problem for problem in problems)


def test_as_of_none_is_a_problem():
    """Opus re-gate BLOCK(1) on 609cb2c5: as_of=None used to pass here and
    fail only inside the worker (computed_moves_store._as_of_day always
    refuses None) -- the same admitted-only-to-fail-later pattern already
    fixed for the plan binding. as_of is now required at submission."""
    problems = calendar_moves_parameter_problems(_job(), _params(as_of=None))
    assert any("as_of" in problem for problem in problems)


def test_partial_plan_binding_is_a_problem():
    problems = calendar_moves_parameter_problems(
        _job(), _params(parent_snapshot_id="snap-parent", refresh_plan_hash="", provider_calls=0))
    assert any("refresh_plan_hash" in problem for problem in problems)


def test_plan_binding_is_now_required_even_when_entirely_absent():
    """Round 3 fix (Opus finding 1): previously, a job with NO plan-binding
    field set at all (parent_snapshot_id/refresh_plan_hash/provider_calls all
    blank/zero) passed with no problems -- admitted only to fail inside the
    worker. The binding is now unconditional, matching
    incremental_data.refresh_parameter_problems."""
    problems = calendar_moves_parameter_problems(
        _job(), _params(parent_snapshot_id="", refresh_plan_hash="", provider_calls=0))
    assert any("parent_snapshot_id" in problem for problem in problems)
    assert any("refresh_plan_hash" in problem for problem in problems)


def test_catalog_path_blank_is_a_problem():
    problems = calendar_moves_parameter_problems(_job(), _params(catalog_path=""))
    assert any("catalog_path" in problem for problem in problems)


def test_objects_root_blank_is_a_problem():
    problems = calendar_moves_parameter_problems(_job(), _params(objects_root=""))
    assert any("objects_root" in problem for problem in problems)


def test_scope_blank_is_a_problem():
    problems = calendar_moves_parameter_problems(_job(), _params(scope=""))
    assert any("scope" in problem for problem in problems)


def test_expected_head_generation_negative_is_a_problem():
    problems = calendar_moves_parameter_problems(
        _job(), _params(expected_head_generation=-1))
    assert any("expected_head_generation" in problem for problem in problems)


def test_provider_calls_without_a_provider_budget_is_a_problem():
    problems = calendar_moves_parameter_problems(
        _job(provider_budget_ref=None), _params(provider_calls=3))
    assert any("provider budget" in problem for problem in problems)


def test_provider_budget_without_any_provider_calls_is_a_problem():
    problems = calendar_moves_parameter_problems(
        _job(provider_budget_ref="native-account"), _params(provider_calls=0))
    assert any("planned call" in problem for problem in problems)


def test_result_path_as_an_input_binding_is_a_problem():
    """The worker writes COMPUTED_MOVES_RESULT_PATH itself; binding it as an
    input would let a caller feed the worker's own output back to it."""
    problems = calendar_moves_parameter_problems(
        _job(), _params(input_bindings={COMPUTED_MOVES_RESULT_PATH: "some-ref"}))
    assert any("input binding" in problem for problem in problems)


def test_forward_calendar_result_path_as_an_input_binding_is_a_problem():
    """A forward_calendar_refresh job must refuse a binding of its OWN result
    path. Before this fix the check always compared against
    COMPUTED_MOVES_RESULT_PATH, so a binding literally named
    FORWARD_CALENDAR_RESULT_PATH slipped past it entirely."""
    problems = calendar_moves_parameter_problems(
        _job(kind=FORWARD_CALENDAR_REFRESH_ACTION),
        _params(tickers=("AAPL",),
                input_bindings={FORWARD_CALENDAR_RESULT_PATH: "some-ref"}))
    assert any("input binding" in problem for problem in problems)


def test_valid_parameters_have_no_problems():
    assert calendar_moves_parameter_problems(_job(), _params()) == ()


def test_horizon_days_out_of_range_is_a_problem():
    for horizon_days in (0, 400):
        problems = calendar_moves_parameter_problems(
            _job(), _params(tickers=("AAPL",), horizon_days=horizon_days))
        assert any("horizon_days" in problem for problem in problems), horizon_days


def test_horizon_days_bool_is_a_problem():
    problems = calendar_moves_parameter_problems(
        _job(), _params(tickers=("AAPL",), horizon_days=True))
    assert any("horizon_days" in problem for problem in problems)


def test_tickers_duplicate_is_a_problem():
    problems = calendar_moves_parameter_problems(
        _job(), _params(tickers=("AAPL", "AAPL")))
    assert any("tickers" in problem for problem in problems)


def test_tickers_not_matching_expected_ids_is_a_problem():
    """forward_calendar_store commits whatever tickers set it is passed, so a
    caller could otherwise submit tickers=("MSFT",) with expected_ids=("AAPL",)
    and only discover the mismatch after an unwanted refresh had committed."""
    problems = calendar_moves_parameter_problems(
        _job(), _params(expected_ids=("AAPL",), tickers=("MSFT",)))
    assert any("tickers" in problem and "expected_ids" in problem for problem in problems)


def test_tickers_matching_expected_ids_in_another_order_is_no_problem():
    assert calendar_moves_parameter_problems(
        _job(), _params(expected_ids=("AAPL", "MSFT"), tickers=("MSFT", "AAPL"))) == ()


def test_empty_tickers_never_requires_matching_expected_ids():
    assert calendar_moves_parameter_problems(
        _job(), _params(expected_ids=("AAPL", "MSFT"), tickers=())) == ()


def test_empty_tickers_is_a_problem_for_a_forward_calendar_refresh_job():
    """forward_calendar_store treats an empty tickers tuple as "the whole
    market", so a forward_calendar_refresh job submitted with tickers=() and
    a non-empty expected_ids would commit a whole-market refresh before its
    coverage mismatch was discovered. The previous round's matching check was
    guarded by `elif tickers`, silently skipping exactly this case."""
    problems = calendar_moves_parameter_problems(
        _job(kind=FORWARD_CALENDAR_REFRESH_ACTION),
        _params(expected_ids=("AAPL",), tickers=()))
    assert any("tickers" in problem for problem in problems)


def test_empty_tickers_is_no_problem_without_the_forward_calendar_kind():
    """The empty-tickers rule is kind-specific: computed_moves_refresh never
    reads the field and every real submission leaves it at its empty default,
    so a blanket rule would break every such job. A bare _job() with no kind
    set (like test_empty_tickers_never_requires_matching_expected_ids above)
    must behave the same way."""
    assert calendar_moves_parameter_problems(
        _job(kind=COMPUTED_MOVES_REFRESH_ACTION),
        _params(expected_ids=("AAPL", "MSFT"), tickers=())) == ()
    assert calendar_moves_parameter_problems(
        _job(), _params(expected_ids=("AAPL", "MSFT"), tickers=())) == ()


def test_valid_forward_calendar_fields_have_no_problems():
    assert calendar_moves_parameter_problems(
        _job(), _params(tickers=("AAPL",), horizon_days=30)) == ()


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


def test_run_computed_moves_worker_refuses_a_result_bound_to_a_different_plan(tmp_path):
    params = _params()
    result = _result(parent_snapshot_id="snap-other")

    def _callback(parameters, root):
        return result

    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_computed_moves_worker(
            to_document(params), tmp_path, refresh_callback=_callback)
    assert exc_info.value.code == "STALE_EXPECTATION"


# --------------------------------------------------------------------------
# forward_calendar worker: the same shared worker, its own result artifact
# --------------------------------------------------------------------------


def test_run_forward_calendar_worker_happy_path(tmp_path):
    params = _params(tickers=("AAPL",))
    result = _result()

    def _callback(parameters, root):
        assert parameters == params
        assert root == tmp_path
        return result

    output = calendar_moves_jobs.run_forward_calendar_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert output["completed_ids"] == ["AAPL"]
    assert output["no_work"] is False
    assert output["outputs"] == [{"name": FORWARD_CALENDAR_REFRESH_ACTION,
                                  "path": FORWARD_CALENDAR_RESULT_PATH,
                                  "schema": FORWARD_CALENDAR_RESULT_SCHEMA}]
    written = json.loads((tmp_path / FORWARD_CALENDAR_RESULT_PATH).read_bytes())
    assert written["status"] == "complete"
    assert written["candidate_snapshot_id"] == "snap-new"


def test_run_forward_calendar_worker_happy_path_tolerates_reordered_coverage(tmp_path):
    """expected_ids (and tickers) are caller-supplied, unsorted tuples; the
    store returns completed_ids sorted. Same ticker set, different order --
    must still validate as complete, not fail a commit that already
    happened."""
    params = _params(expected_ids=("MSFT", "AAPL"), tickers=("MSFT", "AAPL"))
    result = _result(completed_ids=("AAPL", "MSFT"))

    def _callback(parameters, root):
        return result

    output = calendar_moves_jobs.run_forward_calendar_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert sorted(output["completed_ids"]) == ["AAPL", "MSFT"]


def test_run_forward_calendar_worker_cached_rerun_is_a_true_noop(tmp_path):
    params = _params(tickers=("AAPL",))
    result = _result(status="noop", coverage_advanced=False, candidate_snapshot_id=None)

    def _callback(parameters, root):
        return result

    output = calendar_moves_jobs.run_forward_calendar_worker(
        to_document(params), tmp_path, refresh_callback=_callback)

    assert output["no_work"] is True
    assert output["completed_ids"] == ["AAPL"]
    written = json.loads((tmp_path / FORWARD_CALENDAR_RESULT_PATH).read_bytes())
    assert written["status"] == "noop"


def test_run_forward_calendar_worker_maps_a_transient_status_to_its_typed_code(tmp_path):
    params = _params(tickers=("AAPL",))
    result = _result(status="transient", coverage_advanced=False, candidate_snapshot_id=None)

    def _callback(parameters, root):
        return result

    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_forward_calendar_worker(
            to_document(params), tmp_path, refresh_callback=_callback)
    assert exc_info.value.code == "TRANSIENT_SOURCE"


def test_run_forward_calendar_worker_refuses_bool_where_int_expected_before_any_io(tmp_path):
    """A bool is an int subtype, but the strict decoder refuses it for
    CalendarMovesParameters.horizon_days, so the refusal happens at decode
    time -- before the injected callback (and thus any I/O) could run."""
    document = to_document(_params(tickers=("AAPL",)))
    document["horizon_days"] = True
    with pytest.raises(OpsError) as exc_info:
        calendar_moves_jobs.run_forward_calendar_worker(
            document, tmp_path, refresh_callback=_unreachable_callback)
    assert exc_info.value.code == "INVALID_REQUEST"
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# the real loader: the head format check runs before any I/O, the staged
# attempt document supplies attempt_id/fence, and every parameter is passed
# straight through to forward_calendar_store.run_forward_calendar_refresh
# --------------------------------------------------------------------------


def _write_attempt_document(root, document):
    name = REFRESH_INPUT_DOCUMENT_NAMES["forward_calendar_refresh"]
    (root / name).write_text(json.dumps(document))


@pytest.mark.parametrize("head", ["", "s" * 129], ids=["empty", "overlong"])
def test_forward_calendar_callback_refuses_a_bad_head_before_reading_root(
        tmp_path, monkeypatch, head):
    callback = incremental_data._load_forward_calendar_refresh_callback()
    params = _params(tickers=("AAPL",), expected_head_snapshot_id=head)

    def _explode(self):
        raise AssertionError("the staged document must not be read before the head check")

    monkeypatch.setattr(Path, "read_bytes", _explode)
    with pytest.raises(OpsError) as exc_info:
        callback(params, tmp_path)
    assert exc_info.value.code == "INVALID_REQUEST"


def test_staged_forward_calendar_attempt_reads_a_real_document(tmp_path):
    _write_attempt_document(tmp_path, {"attempt_id": "att-1", "fence": 3})
    assert incremental_data._staged_forward_calendar_attempt(
        tmp_path, REFRESH_INPUT_DOCUMENT_NAMES["forward_calendar_refresh"]) == ("att-1", 3)


@pytest.mark.parametrize("raw", [
    None,
    "not json",
    "[1, 2]",
    "{}",
    '{"attempt_id": null, "fence": null}',
    '{"attempt_id": "att-1"}',
    '{"attempt_id": "att-1", "fence": null}',
    '{"fence": 3}',
    '{"attempt_id": null, "fence": 3}',
    '{"attempt_id": "", "fence": 1}',
    '{"attempt_id": "att-1", "fence": 0}',
    '{"attempt_id": "att-1", "fence": true}',
    '{"attempt_id": 5, "fence": 1}',
    '{"attempt_id": "att-1", "fence": "3"}',
], ids=["missing", "invalid-json", "non-object", "empty-object", "both-null",
        "fence-absent", "fence-null", "attempt-id-absent", "attempt-id-null",
        "blank-attempt-id", "zero-fence", "bool-fence", "non-string-attempt-id",
        "non-integer-fence"])
def test_staged_forward_calendar_attempt_refuses_a_malformed_document(tmp_path, raw):
    name = REFRESH_INPUT_DOCUMENT_NAMES["forward_calendar_refresh"]
    if raw is not None:
        (tmp_path / name).write_text(raw)
    with pytest.raises(OpsError) as exc_info:
        incremental_data._staged_forward_calendar_attempt(tmp_path, name)
    assert exc_info.value.code == "INVALID_REQUEST"


def test_forward_calendar_callback_passes_parameters_plus_staged_attempt(tmp_path, monkeypatch):
    captured = {}
    sentinel = _result()

    def _fake_runner(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(forward_calendar_store, "run_forward_calendar_refresh", _fake_runner)
    _write_attempt_document(tmp_path, {"attempt_id": "att-9", "fence": 7})
    params = _params(expected_ids=("AAPL", "MSFT"), tickers=("AAPL", "MSFT"),
                     horizon_days=30)
    callback = incremental_data._load_forward_calendar_refresh_callback()

    assert callback(params, tmp_path) is sentinel
    assert set(captured) == {
        "catalog_path", "objects_root", "parent_snapshot_id", "refresh_plan_hash",
        "as_of", "tickers", "horizon_days", "scope", "expected_head_generation",
        "expected_head_snapshot_id", "attempt_id", "fence",
        "nasdaq_fetcher", "earnings_fetcher",
    }
    assert {key: value for key, value in captured.items()
            if not key.endswith("_fetcher")} == {
        "catalog_path": "/ops/catalog.sqlite", "objects_root": "/ops/objects",
        "parent_snapshot_id": "snap-parent", "refresh_plan_hash": "sha256:" + "c" * 64,
        "as_of": "2026-09-27", "tickers": ("AAPL", "MSFT"), "horizon_days": 30,
        "scope": "shadow", "expected_head_generation": 3,
        "expected_head_snapshot_id": "snap-parent", "attempt_id": "att-9", "fence": 7,
    }
    assert captured["nasdaq_fetcher"] is not None
    assert captured["earnings_fetcher"] is not None
