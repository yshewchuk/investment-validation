"""Search parent memory P4: plan-rebuild memo and shared freeze verification."""
import sys
from unittest.mock import Mock

import pytest

from engine.v2.foundation import ArtifactError, artifact_reference, canonical_json, content_hash, to_document
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_plan as plan_mod
from experiments import size_search_binding as binding
from experiments import magnitude_search_trials as trials_module
from experiments import v2_magnitude_search_run as run_mod

PLAN_V3 = plan_mod.PLAN_V3
REQUEST = {"schema_version": "magnitude_search_request.v4"}
SIZE_REQUEST = {"schema_version": "magnitude_search_request.v4"}
FROZEN = {"selections": [{"outer_year": "2024", "selected_candidate_id": "overall"}]}


def _verified(value, run_id="run", plan_ref=None):
    """Register ``value`` as if SearchTrials.freeze() had returned it for ``run_id`` and ``plan_ref``."""
    plan_ref = PLAN_REF if plan_ref is None else plan_ref
    trials_module._FREEZES.append((value, run_id, canonical_json(to_document(plan_ref)), content_hash(value)))
    return value


def _frozen_by_trials(monkeypatch, value, run_id="run", plan_ref=None):
    """Return what SearchTrials.freeze() returns when its private _freeze yields ``value``."""
    plan_ref = PLAN_REF if plan_ref is None else plan_ref
    monkeypatch.setattr(trials_module.SearchTrials, "_freeze", lambda self: value)
    registration = Registration()
    registration.run_id = run_id
    trials = trials_module.SearchTrials(None, Store(), None, registration, {}, plan_ref)
    return trials.freeze()


class Registration:
    run_id = "run"
    variant_id = "variant"

    def __init__(self):
        self.document = {"schema_version": "native_experiment_registration.v1.7",
                         "size_panel_request": {"horizons": [5, 3, 1]},
                         "size_request": dict(SIZE_REQUEST),
                         "source_closure": {}, "environment": {}}


class Store:
    def __init__(self):
        self.reads = 0
        self.fail = False

    def read_verified(self, ref):
        self.reads += 1
        if self.fail:
            raise ArtifactError("INTEGRITY_FAILED", "stored bytes differ")


@pytest.fixture
def calls(monkeypatch):
    counts = {"registration": 0, "installation": 0, "folds": 0, "candidates": 0,
              "fail_installation": False}

    def registration_check(conn, store, registration, **kwargs):
        counts["registration"] += 1

    def installation_check(document):
        counts["installation"] += 1
        if counts["fail_installation"]:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search anchor sources or environment differ")

    def folds(document, request, plan_schema=None):
        counts["folds"] += 1
        return []

    def candidates(registration, schedule):
        counts["candidates"] += 1
        return [], {}

    def compile_fake(request, horizons):
        return {"request": dict(SIZE_REQUEST), "horizons": list(horizons), "candidate_specs": [],
                "counts": {"candidates": 0, "search_fits": 0}}

    monkeypatch.setattr(binding, "verify_native_registration", registration_check)
    monkeypatch.setattr(binding, "verify_installation", installation_check)
    monkeypatch.setattr(binding, "validate_panel_request", lambda request: request)
    monkeypatch.setattr(binding, "compile_request", compile_fake)
    monkeypatch.setattr(binding, "_folds", folds)
    monkeypatch.setattr(binding, "_candidates", candidates)
    return counts


def _setup(calls):
    """Build the reference plan once, then reset counters so only the test's reads are counted."""
    store, registration = Store(), Registration()
    expected = binding.prepare_plan(None, store, registration, REQUEST, plan_schema=PLAN_V3)
    ref = artifact_reference(canonical_json(expected).encode(), PLAN_V3)
    for key in ("registration", "installation", "folds", "candidates"):
        calls[key] = 0
    store.reads = 0
    return store, registration, ref


def test_memo_is_off_by_default_and_rebuilds_each_read(calls):
    store, registration, ref = _setup(calls)
    plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
    plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
    assert calls["folds"] == 2
    assert calls["registration"] == 2 and calls["installation"] == 2


def test_memo_reuses_the_build_but_repeats_identity_checks(calls):
    store, registration, ref = _setup(calls)
    with plan_mod.plan_memo():
        first = plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
        second = plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
    assert first == second
    assert calls["folds"] == 1
    assert calls["registration"] == 2 and calls["installation"] == 2
    assert store.reads == 2


def test_memo_refuses_when_installation_changes_between_reads(calls):
    store, registration, ref = _setup(calls)
    with plan_mod.plan_memo():
        plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
        calls["fail_installation"] = True
        with pytest.raises(OpsError) as caught:
            plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert calls["folds"] == 1


def test_memo_rebuilds_for_a_changed_request(calls):
    _, registration, _ = _setup(calls)
    with plan_mod.plan_memo():
        binding.prepare_plan(None, Store(), registration, REQUEST, plan_schema=PLAN_V3)
        binding.prepare_plan(None, Store(), registration, {**REQUEST, "note": "changed"},
                             plan_schema=PLAN_V3)
    assert calls["folds"] == 2


def test_memo_still_refuses_a_wrong_ref_on_a_hit(calls):
    store, registration, ref = _setup(calls)
    wrong = artifact_reference(b"tampered", PLAN_V3)
    with plan_mod.plan_memo():
        plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
        with pytest.raises(OpsError) as caught:
            plan_mod.read_search_plan(None, store, registration, REQUEST, wrong)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert calls["folds"] == 1


def test_memo_still_verifies_stored_bytes_on_a_hit(calls):
    store, registration, ref = _setup(calls)
    with plan_mod.plan_memo():
        plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
        store.fail = True
        with pytest.raises(OpsError) as caught:
            plan_mod.read_search_plan(None, store, registration, REQUEST, ref)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert store.reads == 2


def test_plan_memo_is_cleared_after_exit_and_nested_use_keeps_the_outer_dict():
    assert plan_mod._PLAN_MEMO is None
    with plan_mod.plan_memo():
        outer_memo = plan_mod._PLAN_MEMO
        assert outer_memo is not None
        with plan_mod.plan_memo():
            assert plan_mod._PLAN_MEMO is outer_memo
        assert plan_mod._PLAN_MEMO is outer_memo
    assert plan_mod._PLAN_MEMO is None


def test_plan_memo_clears_when_the_body_raises():
    with pytest.raises(RuntimeError):
        with plan_mod.plan_memo():
            raise RuntimeError("boom")
    assert plan_mod._PLAN_MEMO is None


def test_memoized_plan_never_stores_a_failed_build():
    attempts = []

    def build():
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("first build fails")
        return {"plan": len(attempts)}

    registration = object()
    with plan_mod.plan_memo():
        with pytest.raises(ValueError):
            plan_mod.memoized_plan(registration, REQUEST, PLAN_V3, build)
        assert plan_mod.memoized_plan(registration, REQUEST, PLAN_V3, build) == {"plan": 2}
        assert plan_mod.memoized_plan(registration, REQUEST, PLAN_V3, build) == {"plan": 2}
    assert len(attempts) == 2


def _plan():
    return {"plan_id": "plan",
            "selection": {"winner_rules": ["overall", "rmse", "mae"]},
            "candidates": [{"candidate_id": value} for value in ("overall", "rmse", "mae")],
            "folds": [{"outer": {"year": "2024"}}]}


def _selection():
    return {"selections": [{"outer_year": "2024", "selected_candidate_id": "overall",
                            "winners": {}, "distinct_winner_ids": ["overall", "rmse"]}],
            "prospective_candidate_id": "overall"}


PLAN_REF = artifact_reference(b"plan", PLAN_V3)
OTHER_REF = artifact_reference(b"other plan", PLAN_V3)


def _execute(monkeypatch, plan_ref=PLAN_REF, shared=None):
    """Run execute_outer_fold once; ``shared`` = (run_id, plan_ref, value) wraps the call in shared_freeze."""
    seen, trials = [], Mock()

    def native(*args):
        seen.append(args)
        return _selection()

    monkeypatch.setattr(outer, "validate_execution_limits", lambda limits: None)
    monkeypatch.setattr(outer, "require_current", lambda *args: None)
    monkeypatch.setattr(outer, "read_search_plan", lambda *args: _plan())
    monkeypatch.setattr(outer, "SearchTrials", lambda *args: trials)
    monkeypatch.setattr(outer, "_native_selection", native)
    monkeypatch.setattr(outer, "_execute_admitted", Mock(return_value={"admitted": True}))

    def call():
        outer.execute_outer_fold(None, None, None, Registration(), {}, plan_ref, "2024",
                                 limits={}, candidate_id="rmse")

    if shared is None:
        call()
    else:
        with outer.shared_freeze(*shared):
            call()
    return seen, trials


def test_execute_outer_fold_no_longer_accepts_frozen():
    with pytest.raises(TypeError):
        outer.execute_outer_fold(None, None, None, Registration(), {}, PLAN_REF, "2024",
                                 limits={}, candidate_id="rmse", frozen=FROZEN)


def test_execute_outer_fold_outside_any_scope_uses_three_arguments(monkeypatch):
    seen, trials = _execute(monkeypatch)
    assert len(seen) == 1 and len(seen[0]) == 3
    assert outer._SHARED is None


def test_execute_outer_fold_inside_matching_scope_passes_the_shared_value(monkeypatch):
    seen, trials = _execute(monkeypatch, shared=("run", PLAN_REF, _verified(FROZEN)))
    assert len(seen) == 1 and len(seen[0]) == 4 and seen[0][3] is FROZEN
    trials.freeze.assert_not_called()


def test_execute_outer_fold_falls_back_to_trials_freeze_for_another_run(monkeypatch):
    seen, _ = _execute(monkeypatch, shared=("other-run", PLAN_REF, _verified(FROZEN, run_id="other-run")))
    assert len(seen) == 1 and len(seen[0]) == 3


def test_execute_outer_fold_falls_back_to_trials_freeze_for_another_plan(monkeypatch):
    seen, _ = _execute(monkeypatch, plan_ref=PLAN_REF, shared=("run", OTHER_REF, _verified(FROZEN, plan_ref=OTHER_REF)))
    assert len(seen) == 1 and len(seen[0]) == 3


def test_shared_freeze_restores_previous_value_on_exit():
    assert outer._SHARED is None
    with outer.shared_freeze("run", PLAN_REF, _verified("outer")):
        with outer.shared_freeze("run", PLAN_REF, _verified("inner")):
            assert outer._SHARED[2] == "inner"
        assert outer._SHARED[2] == "outer"
    assert outer._SHARED is None


def test_shared_freeze_restores_previous_value_when_the_body_raises():
    with outer.shared_freeze("run", PLAN_REF, _verified("outer")):
        with pytest.raises(RuntimeError):
            with outer.shared_freeze("run", PLAN_REF, _verified("inner")):
                raise RuntimeError("boom")
        assert outer._SHARED[2] == "outer"
    assert outer._SHARED is None


def test_shared_freeze_refuses_an_unregistered_equal_value():
    assert outer._SHARED is None
    with pytest.raises(OpsError) as caught:
        with outer.shared_freeze("run", PLAN_REF, dict(FROZEN)):
            pass  # pragma: no cover
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert outer._SHARED is None


def test_shared_freeze_accepts_a_value_returned_by_trials_freeze(monkeypatch):
    value = {"a": 1}
    monkeypatch.setattr(trials_module.SearchTrials, "_freeze", lambda self: value)
    trials = trials_module.SearchTrials(None, Store(), None, Registration(), {}, PLAN_REF)
    frozen = trials.freeze()
    assert frozen is value
    with outer.shared_freeze("run", PLAN_REF, frozen):
        assert outer._SHARED[2] is frozen
    assert outer._SHARED is None


def test_freeze_registry_keeps_only_the_last_four(monkeypatch):
    values = [{"n": number} for number in range(6)]
    pending = iter(values)
    monkeypatch.setattr(trials_module.SearchTrials, "_freeze", lambda self: next(pending))
    trials = trials_module.SearchTrials(None, Store(), None, Registration(), {}, PLAN_REF)
    for _ in values:
        trials.freeze()
    assert len(trials_module._FREEZES) == 4
    plan_key = canonical_json(to_document(PLAN_REF))
    assert all(kept[0] is value for kept, value in zip(trials_module._FREEZES, values[-4:]))
    assert not any(trials_module.freeze_is_verified(value, "run", plan_key) for value in values[:2])
    assert all(trials_module.freeze_is_verified(value, "run", plan_key) for value in values[-4:])


def test_shared_freeze_refuses_a_freeze_registered_for_another_run(monkeypatch):
    frozen = _frozen_by_trials(monkeypatch, {"a": 1}, run_id="run")
    assert outer._SHARED is None
    with pytest.raises(OpsError) as caught:
        with outer.shared_freeze("other-run", PLAN_REF, frozen):
            pass  # pragma: no cover
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert outer._SHARED is None


def test_shared_freeze_refuses_a_freeze_registered_for_another_plan(monkeypatch):
    frozen = _frozen_by_trials(monkeypatch, {"a": 1}, plan_ref=PLAN_REF)
    with pytest.raises(OpsError) as caught:
        with outer.shared_freeze("run", OTHER_REF, frozen):
            pass  # pragma: no cover
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert outer._SHARED is None


def test_shared_freeze_refuses_a_registered_result_mutated_after_freezing(monkeypatch):
    frozen = _frozen_by_trials(monkeypatch, {"a": 1})
    frozen["a"] = 2
    with pytest.raises(OpsError) as caught:
        with outer.shared_freeze("run", PLAN_REF, frozen):
            pass  # pragma: no cover
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert outer._SHARED is None


PLAN = {"folds": [], "candidates": []}


def _reader(stored):
    return lambda slot: stored if slot == "selection" else None


def test_native_selection_refuses_a_shared_value_that_differs():
    trials = Mock()
    with pytest.raises(OpsError) as caught:
        outer._native_selection(trials, PLAN, _reader({"a": 1}), frozen={"a": 2})
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    trials.freeze.assert_not_called()


def test_native_selection_accepts_an_equal_shared_value_without_refreezing():
    trials = Mock()
    assert outer._native_selection(trials, PLAN, _reader({"a": 1}), frozen={"a": 1}) == {"a": 1}
    trials.freeze.assert_not_called()


def test_native_selection_without_frozen_verifies_a_fresh_freeze():
    trials = Mock()
    trials.freeze.return_value = {"a": 1}
    assert outer._native_selection(trials, PLAN, _reader({"a": 1})) == {"a": 1}
    trials.freeze.assert_called_once_with()


def test_native_selection_deferred_path_refuses_a_fresh_freeze_that_differs():
    trials = Mock()
    trials.freeze.return_value = {"a": 2}
    with pytest.raises(OpsError) as caught:
        outer._native_selection(trials, PLAN, _reader({"a": 1}))
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    trials.freeze.assert_called_once_with()


ONE_PLAN = {"plan_id": "plan",
            "folds": [{"outer": {"year": "2024"}, "inner": [{"year": "2023"}]}],
            "candidates": [{"candidate_id": "overall"}]}
ONE_KEY = plan_mod.trial_key(ONE_PLAN, "2024", "2023", "overall")
ONE_ATTEMPT = {"limits": {}, "candidate_id": "overall", "trial_id": ONE_KEY}


def _records(*, missing=(), **overrides):
    records = {"selection": {"a": 1}, "execution:" + ONE_KEY: {"attempt": ONE_ATTEMPT},
               "terminal:" + ONE_KEY: {"status": "completed"},
               "execution-start:" + ONE_KEY: ONE_ATTEMPT}
    records.update(overrides)
    for slot in missing:
        records.pop(slot + ONE_KEY)
    return lambda slot: records.get(slot)


@pytest.fixture
def one_trial(monkeypatch):
    monkeypatch.setattr(outer, "validate_execution_limits", lambda limits: None)


def test_native_selection_shared_branch_refuses_a_missing_terminal_record(one_trial):
    trials = Mock()
    with pytest.raises(OpsError) as caught:
        outer._native_selection(trials, ONE_PLAN, _records(missing=("terminal:",)), frozen={"a": 1})
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    trials.freeze.assert_not_called()


def test_native_selection_shared_branch_refuses_a_missing_execution_record(one_trial):
    trials = Mock()
    with pytest.raises(OpsError) as caught:
        outer._native_selection(trials, ONE_PLAN, _records(missing=("execution:",)), frozen={"a": 1})
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    trials.freeze.assert_not_called()


def test_native_selection_shared_branch_refuses_an_execution_start_that_differs(one_trial):
    trials = Mock()
    read = _records(**{"execution-start:" + ONE_KEY: {**ONE_ATTEMPT, "candidate_id": "rmse"}})
    with pytest.raises(OpsError) as caught:
        outer._native_selection(trials, ONE_PLAN, read, frozen={"a": 1})
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    trials.freeze.assert_not_called()


def test_native_selection_shared_branch_passes_with_complete_records(one_trial):
    trials = Mock()
    assert outer._native_selection(trials, ONE_PLAN, _records(), frozen={"a": 1}) == {"a": 1}
    trials.freeze.assert_not_called()


@pytest.mark.parametrize("per_candidate_plan", [True, False])
def test_run_search_shares_one_freeze_across_the_refit_loop(monkeypatch, per_candidate_plan):
    trials = Mock()
    trials.freeze.return_value = _verified(FROZEN)
    memo_seen, refits = [], []

    def admit(conn, store, registration, request, plan_ref):
        memo_seen.append(plan_mod._PLAN_MEMO is not None)
        return {"folds": []}

    def execute(conn, store, repository, registration, request, plan_ref, year, *,
                limits, candidate_id=None):
        refits.append((year, candidate_id, outer._SHARED))
        return None

    monkeypatch.setattr(run_mod, "validate_workers", lambda request: 1)
    monkeypatch.setattr(run_mod, "validate_execution_limits", lambda limits: None)
    monkeypatch.setattr(run_mod, "_admit_driver", admit)
    monkeypatch.setattr(run_mod, "SearchTrials", lambda *args: trials)
    monkeypatch.setattr(run_mod, "log_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "panel_memo_holder", lambda: None)
    monkeypatch.setattr(run_mod, "read_record", lambda conn, store, run_id, slot: None)
    monkeypatch.setattr(run_mod, "_roster", lambda plan: [])
    monkeypatch.setattr(run_mod, "possible_slots", lambda plan: [])
    monkeypatch.setattr(run_mod, "per_candidate", lambda plan: per_candidate_plan)
    monkeypatch.setattr(run_mod, "refit_units", lambda plan, item: ["overall", "rmse"])
    monkeypatch.setattr(run_mod, "execute_outer_fold", execute)

    result = run_mod.run_search(None, None, None, Registration(), REQUEST, PLAN_REF,
                                limits={}, collect_results=False)

    assert result is None
    assert memo_seen == [True]
    assert plan_mod._PLAN_MEMO is None
    assert outer._SHARED is None
    assert trials.freeze.call_count == 1
    assert [(year, unit) for year, unit, _ in refits] == [("2024", "overall"), ("2024", "rmse")]
    assert all(shared is not None and shared[2] is FROZEN for _, _, shared in refits)


def test_memo_keys_dict_subclass_misses_the_plain_dict_entry(calls):
    _, registration, _ = _setup(calls)

    class Request(dict):
        pass

    with plan_mod.plan_memo():
        binding.prepare_plan(None, Store(), registration, REQUEST, plan_schema=PLAN_V3)
        binding.prepare_plan(None, Store(), registration, Request(REQUEST), plan_schema=PLAN_V3)
    assert calls["folds"] == 2


def test_memoized_plan_key_step_does_not_raise_on_deep_nesting():
    deep = []
    for _ in range(sys.getrecursionlimit() + 100):
        deep = [deep]
    built = []

    def build():
        built.append(1)
        return {"plan": "built"}

    with plan_mod.plan_memo():
        assert plan_mod.memoized_plan(object(), deep, PLAN_V3, build) == {"plan": "built"}
    assert built == [1]


@pytest.mark.parametrize("left, right", [
    ({"max_fits": 1000}, {"max_fits": 1000.0}),
    ({"max_fits": 1}, {"max_fits": True}),
    ({"axes": [1, 2]}, {"axes": (1, 2)}),
])
def test_memo_keys_keep_value_types_apart(calls, left, right):
    _, registration, _ = _setup(calls)
    with plan_mod.plan_memo():
        binding.prepare_plan(None, Store(), registration, {**REQUEST, **left}, plan_schema=PLAN_V3)
        size = len(plan_mod._PLAN_MEMO)
        binding.prepare_plan(None, Store(), registration, {**REQUEST, **right}, plan_schema=PLAN_V3)
        assert len(plan_mod._PLAN_MEMO) == size + 1
    assert calls["folds"] == 2


def test_memo_keys_on_plan_schema(calls):
    _, registration, _ = _setup(calls)
    with plan_mod.plan_memo():
        binding.prepare_plan(None, Store(), registration, REQUEST, plan_schema=PLAN_V3)
        binding.prepare_plan(None, Store(), registration, REQUEST, plan_schema=plan_mod.PLAN_V2)
    assert calls["folds"] == 2


def test_memo_scope_does_not_retain_plans_from_unmemoized_prepare(monkeypatch):
    plan = {"plan_id": "fresh", "folds": []}
    monkeypatch.setattr(plan_mod, "_prepare", lambda conn, store, registration, request, **kwargs: dict(plan))
    store, registration = Store(), Registration()
    ref = artifact_reference(canonical_json(plan).encode(), PLAN_V3)
    tampered = artifact_reference(b"tampered", PLAN_V3)
    with plan_mod.plan_memo():
        for _ in range(3):
            assert plan_mod.read_search_plan(None, store, registration, REQUEST, ref) == plan
        assert len(plan_mod._PLAN_MEMO) == 0
        with pytest.raises(OpsError) as caught:
            plan_mod.read_search_plan(None, store, registration, REQUEST, tampered)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert store.reads == 3
