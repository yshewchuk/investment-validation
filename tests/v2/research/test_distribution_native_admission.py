"""Real native synthetic catalogs/panels, never SIZE fits or private inputs."""
from copy import deepcopy
from pathlib import Path

import pytest

from engine.models.no_fit import no_fit_guard as legacy_no_fit
from engine.v2.data.errors import fail as data_fail
from engine.v2.foundation import SystemClock, canonical_json, content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError, fail
from experiments import distribution_plan, distribution_trials
from experiments.distribution_candidate_recipe import candidate_recipe
from experiments.magnitude_search_inputs import load_fit_inputs
from experiments.native_registration import prepare_native_registration, reserve_native_registration
from tests.v2.research.magnitude_trial_support import poison_outcomes, prediction_spec
from tests.v2.research.test_size_search_integration import case as source_case

pytestmark = pytest.mark.xdist_group("distribution_native_admission")
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    yield from source_case.__wrapped__(tmp_path_factory)


@pytest.fixture
def own(case, request):
    source = case.registration.document
    spec = prediction_spec(experiment_id="synthetic-distribution-" + request.node.name,
        hypothesis="Compare a raw magnitude histogram with its matched point model.",
        primary_arm_id="raw_pair", arms=("raw_pair",), folds=("2024",),
        runner="experiments/distribution_trials.py")
    # Existing library module is a synthetic metadata fixture, not a runnable
    # production entrypoint. Bundle D owns real runner/activation admission.
    registration, spec = prepare_native_registration(case.repository, spec,
        code_root=ROOT, as_of_month=source["holdouts"]["holdout_as_of_month"],
        scope=source["scope"], event_ids=source["event_ids"], snapshot=case.panel_source)
    return reserve_native_registration(case.conn, case.store, registration, spec)


def trials(case, own, feature_set="H", **changes):
    caps = {"max_fits": 12, "max_trees": 49200,
            "child_limits": dict(case.request["limits"]["child"])}
    args = dict(conn=case.conn, store=case.store, repository=case.repository,
        own_registration=own, source_registration=case.registration,
        source_request=case.request, source_plan_ref=case.ref,
        recipe=candidate_recipe(feature_set), budget_caps=caps)
    return distribution_trials.DistributionTrials(**{**args, **changes})


def source_records(case):
    return case.conn.execute("SELECT slot, ref_json FROM magnitude_search_records "
                            "WHERE run_id=? ORDER BY slot", (case.registration.run_id,)).fetchall()


def forbidden(*args, **kwargs):
    pytest.fail("metadata admission must not execute a SIZE or distribution estimator")


@pytest.mark.parametrize("feature_set", ["H", "HM", "HMO"])
@pytest.mark.parametrize("role,inner_year", [("inner", "2023"), ("outer", None)])
def test_native_admission_is_separate_compact_and_all_horizon(case, own, monkeypatch, feature_set, role, inner_year):
    from experiments import magnitude_search_execution, magnitude_search_trials

    monkeypatch.setattr(magnitude_search_trials.SearchTrials, "execute", forbidden)
    monkeypatch.setattr(magnitude_search_execution, "run_fitted_trial", forbidden)
    before = source_records(case)
    adapter = trials(case, own, feature_set)
    with no_fit_guard(), legacy_no_fit():
        plan = adapter.seal(clock=SystemClock())
        admitted = adapter.start(role, "2024", inner_year)
    assert own.run_id != case.registration.run_id
    assert source_records(case) == before
    # Installed source hashes dominate (~40 KiB); no panel/event rows are copied.
    assert len(canonical_json(plan).encode()) < 256 * 1024
    assert "target_events" not in canonical_json(plan)
    assert admitted["status"] == "admitted"
    assert admitted["final_read_authority"] == "none"
    assert {trial["horizon"] for trial in admitted["trials"]} == {1, 3, 5}
    assert all(trial["inputs_id"] == admitted["inputs_id"] for trial in admitted["trials"])
    fold = case.plan["folds"][0]["outer" if role == "outer" else "inner"]
    if role == "inner":
        fold = fold[0]
    recipe = candidate_recipe(feature_set)
    loaded = load_fit_inputs(case.store, admitted["manifest"], admitted["inputs_id"],
        fold["evaluation_event_ids"], [5, 3, 1], recipe["point_recipe"]["features"])
    expected = {"event_id", "context_id", "horizon_sessions", *recipe["point_recipe"]["features"]}
    assert loaded["features"] and all(set(row) == expected for row in loaded["features"])
    events = {row["event_id"] for row in loaded["training"]} | set(fold["evaluation_event_ids"])
    assert {(row["event_id"], row["horizon_sessions"]) for row in loaded["features"]} == {
        (event, horizon) for event in events for horizon in (1, 3, 5)}
    assert {row["event_id"] for row in loaded["training"]}.isdisjoint(fold["evaluation_event_ids"])
    with monkeypatch.context() as patch:
        patch.setattr(distribution_trials, "fold_inputs", forbidden)
        assert adapter.read_admission(role, "2024", inner_year) == admitted
        with pytest.raises(OpsError) as caught:
            adapter.start(role, "2024", inner_year)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_native_training_does_not_open_protected_or_evaluation_outcomes(case, own, monkeypatch):
    adapter = trials(case, own)
    adapter.seal(clock=SystemClock())
    poison_outcomes(monkeypatch, case.repository, case.panel_source,
        ["2023-01-03", "2023-03-20", "2023-04-03", "2023-05-15", "2023-06-02", "2024-06-17"])
    admitted = adapter.start("inner", "2024", "2023")
    assert admitted["status"] == "admitted"


def test_missing_repository_does_not_consume_native_claim(case, own, monkeypatch):
    adapter = trials(case, own, repository=None)
    adapter.seal(clock=SystemClock())
    before = case.conn.total_changes
    with monkeypatch.context() as patch:
        patch.setattr(distribution_trials, "fold_inputs", forbidden)
        with pytest.raises(OpsError) as caught:
            adapter.start("inner", "2024", "2023")
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert case.conn.total_changes == before
    adapter.repository = case.repository
    assert adapter.start("inner", "2024", "2023")["status"] == "admitted"


@pytest.mark.parametrize("seam", ["fold_inputs", "publish_inputs", "append_record"])
@pytest.mark.parametrize("error", [fail("RESOURCE_UNAVAILABLE", "synthetic native admission interruption"),
    data_fail("INPUT_CHANGED", "synthetic private data detail", details={"secret": "not exposed"})])
def test_interrupted_admission_claim_never_restarts(case, own, monkeypatch, seam, error):
    adapter = trials(case, own)
    adapter.seal(clock=SystemClock())
    calls = []
    original = getattr(distribution_trials, seam)

    def unavailable(*args, **kwargs):
        if seam == "append_record" and not args[3].startswith("distribution:admission:"):
            return original(*args, **kwargs)
        if seam == "publish_inputs":
            original(*args, **kwargs)  # A failure may follow complete, unreferenced shard publication.
        calls.append(True)
        raise error

    monkeypatch.setattr(distribution_trials, seam, unavailable)
    with pytest.raises(OpsError) as first:
        adapter.start("inner", "2024", "2023")
    assert first.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert first.value.problem.retryable is False
    assert first.value.problem.message == "distribution admission interrupted after its single-use claim"
    assert first.value.problem.details == {}
    slots = case.conn.execute("SELECT slot FROM magnitude_search_records WHERE run_id=?", (own.run_id,)).fetchall()
    assert {row[0].split(":")[1] for row in slots} == {"seal", "start"}
    monkeypatch.setattr(distribution_trials, "fold_inputs", forbidden)
    for action in (adapter.start, adapter.read_admission):
        with pytest.raises(OpsError) as caught:
            action("inner", "2024", "2023")
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert calls == [True]


@pytest.mark.parametrize("change", ["recipe", "limits", "source_ref", "same_run"])
def test_sealed_identity_changes_refuse_before_claim_or_outcome_read(case, own, monkeypatch, change):
    trials(case, own).seal(clock=SystemClock())
    args = {}
    if change == "recipe":
        args["recipe"] = candidate_recipe("HM")
    elif change == "limits":
        args["budget_caps"] = {"max_fits": 13, "max_trees": 49200,
                              "child_limits": dict(case.request["limits"]["child"])}
    elif change == "source_ref":
        from dataclasses import replace
        args["source_plan_ref"] = replace(case.ref, content_hash="sha256:" + "0" * 64)
    else:
        args["own_registration"] = case.registration
    before = case.conn.total_changes
    monkeypatch.setattr(distribution_trials, "fold_inputs", forbidden)
    with pytest.raises(OpsError) as caught:
        trials(case, own, **args).start("inner", "2024", "2023")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert case.conn.total_changes == before


def test_schedule_budget_uses_two_estimators_not_source_grid(case, own):
    caps = {"max_fits": 12, "max_trees": 49200, "child_limits": case.request["limits"]["child"]}
    plan, native = distribution_plan.prepare_plan(case.conn, case.store, own,
        case.registration, case.request, case.ref, candidate_recipe("H"), caps)
    assert native["candidate_count"] == 33
    assert plan["budget"]["counts"] == {"classifier_fits": 6, "point_fits": 6, "total_fits": 12,
        "classifier_trees_upper": 48600, "point_trees_upper": 600, "total_trees_upper": 49200}
    assert plan["budget"]["budget_id"] == content_hash({
        key: value for key, value in plan["budget"].items() if key != "budget_id"})
    smaller = deepcopy(caps)
    smaller["max_fits"] = 11
    with pytest.raises(OpsError) as caught:
        distribution_plan.prepare_plan(case.conn, case.store, own,
            case.registration, case.request, case.ref, candidate_recipe("H"), smaller)
    assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
