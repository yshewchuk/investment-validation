"""Synthetic v6 source admission, sealed neural execution and read-only replay."""
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from engine.models.no_fit import no_fit_guard as legacy_no_fit_guard
from engine.v2.contracts import ArtifactRef
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock, from_document
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.catalog import connect
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments import eligible_history, size_panel
from experiments import magnitude_search_plan as planner
from experiments import magnitude_search_trials as trials
from experiments import size_search_binding as binding
from experiments.native_registration import reserve_native_registration, verify_native_registration
from experiments.v2_magnitude_search_run import seal_search_run
from tests.v2.models.training.test_magnitude_mlp import independent_oracle
from tests.v2.research.magnitude_trial_support import poison_outcomes, prediction_spec
from tests.v2.research.test_magnitude_panel_qualified import panel_data
from tests.v2.research.test_magnitude_search_request_v6 import PROFILES, v6_request


@pytest.fixture
def case(tmp_path):
    days = ["2022-06-02", "2022-09-06", "2022-11-01", "2023-01-03", "2023-06-02", "2024-06-17"]
    generator = panel_data.__wrapped__(tmp_path, SimpleNamespace(param={
        "days": days, "values": [2., 3., 4., 5., 6., 7.]}))
    data = next(generator)
    ids = [event["event_id"] for event in data.events]
    history_request = {**data.request, "limits": data.panel_request["limits"],
                       "event_ids": ids, "horizons": [1]}
    history = eligible_history.materialize(data.repository, data.panel_source, **history_request)
    panel_request = {**data.panel_request, "event_ids": ids, "horizons": [1],
                     "history_snapshot": history["feature_snapshot"]}
    panel = size_panel.materialize(data.repository, data.panel_source, **panel_request)
    panel_request.update(feature_snapshot_id=panel["feature_snapshot"].snapshot_id,
        history_snapshot=history["feature_snapshot"].snapshot_id, price_history_snapshot=data.prices.snapshot_id)
    request = v6_request(PROFILES[:1])
    request["space"] = {family: {axis: values[:1] for axis, values in axes.items()}
                        for family, axes in request["space"].items()}
    request.update(feature_dictionary_hash=recipe("HMO")["dictionary_hash"],
        feature_snapshot_id=panel["feature_snapshot"].snapshot_id,
        history_snapshot_id=history["feature_snapshot"].snapshot_id,
        outer_years=["2024"], inner_years={"2024": ["2023"]}, max_candidates=13, max_fits=14)
    request["eligibility_policy_ref"].update(plan_id=panel["context"]["eligibility_plan"]["plan_id"])
    # Existing bounded synthetic integration policy, not new estimator/runtime limits.
    request["limits"] = {"child": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60},
        "parent": {"memory_bytes": 4 << 30, "wall_seconds": 600},
        "build": {"memory_bytes": 3 << 30, "wall_seconds": 60}, "job": {"wall_seconds": 600}}
    spec = prediction_spec(experiment_id="synthetic-qualified-size-v6", folds=("2024",),
                           runner="experiments/size_search.py")
    registration, spec = binding.prepare(data.repository, spec, request, panel_request,
        scope="panel-source", training_start="2022-01-01", labels_before="2025-01-01")
    store = data.repository._store
    reserve_native_registration(data.conn, store, registration, spec)
    verify_native_registration(data.conn, store, registration)
    sealed = planner.seal_search_plan(data.conn, store, registration, request)
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    context = (data.conn, store, data.repository, registration, request, ref)
    seal_search_run(*context, clock=SystemClock())
    yield SimpleNamespace(**{**vars(data), "store": store, "request": request, "registration": registration,
                             "plan": sealed["plan"], "ref": ref, "context": context})
    try:
        next(generator)
    except StopIteration:
        pass


def _clone_case(case, path):
    conn = connect(path / "catalog.sqlite")
    case.conn.backup(conn)
    shutil.copytree(case.store.root, path / "store")
    store = ArtifactStore(path / "store")
    return (conn, store, Repository(conn, store, pin_snapshots=True),
            case.registration, case.request, case.ref)


def test_v6_workers_one_and_two_match_candidates_results_and_metrics(case, tmp_path, monkeypatch):
    from experiments import v2_magnitude_search_run as search_run
    from experiments.size_search_report import collect
    assert any(candidate["recipe"].get("model") == "mlp" for candidate in case.plan["candidates"])
    original_executor, executor_calls = search_run.ProcessPoolExecutor, []

    def spy_executor(*args, **kwargs):
        executor_calls.append(kwargs.get("max_workers"))
        return original_executor(*args, **kwargs)

    monkeypatch.setattr(search_run, "ProcessPoolExecutor", spy_executor)

    def execute(name, workers):
        path = tmp_path / name
        path.mkdir()
        conn, store, repository, registration, _, plan_ref = _clone_case(case, path)
        request = {**case.request, "workers": workers}
        context = (conn, store, repository, registration, request, plan_ref)
        try:
            result = search_run.run_search(*context, limits=request["limits"]["child"])
            report = collect(*context)
        finally:
            conn.close()
        return result, report

    serial_result, serial_report = execute("serial", 1)
    parallel_result, parallel_report = execute("parallel", 2)
    assert executor_calls == [2]

    def first_difference(left, right, path="$"):
        if isinstance(left, dict) and isinstance(right, dict):
            keys = list(left) + [key for key in right if key not in left]
            for key in keys:
                if key not in left or key not in right:
                    return f"{path}.{key}", left.get(key), right.get(key)
                difference = first_difference(left[key], right[key], f"{path}.{key}")
                if difference is not None:
                    return difference
            return None
        if isinstance(left, list) and isinstance(right, list):
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                difference = first_difference(left_item, right_item, f"{path}[{index}]")
                if difference is not None:
                    return difference
            if len(left) != len(right):
                return f"{path}.length", len(left), len(right)
            return None
        return None if left == right else (path, left, right)

    first = first_difference(serial_result, parallel_result)
    if first is not None:
        print(f"first result difference: {first[0]}: serial={first[1]!r}; parallel={first[2]!r}")

    # The timing/execution identities differ between serial and worker runs: folds[*].selection_id,
    # folds[*].result_id, and the top-level result_id. They depend on terminal IDs whose records
    # include runtime score_seconds and rss_delta_kib. Remove only these known paths from shallow
    # copies; do not recurse or recompute hashes, and keep all remaining fields exact.
    def shallow_normalize(result):
        normalized = {key: value for key, value in result.items() if key != "result_id"}
        normalized["folds"] = [
            {key: value for key, value in fold.items()
             if key not in ("selection_id", "result_id")}
            for fold in result["folds"]]
        return normalized

    normalized_serial = shallow_normalize(serial_result)
    normalized_parallel = shallow_normalize(parallel_result)
    assert normalized_serial == normalized_parallel
    assert serial_result["folds"][0]["candidate"] == parallel_result["folds"][0]["candidate"]
    assert serial_result["folds"][0]["predictions"] == parallel_result["folds"][0]["predictions"]
    assert serial_result["metrics"] == parallel_result["metrics"]
    assert serial_result["prospective_candidate_id"] == parallel_result["prospective_candidate_id"]
    serial_inner = {row["candidate_id"]: row["inner_metrics"] for row in serial_report["inner_fits"]}
    parallel_inner = {row["candidate_id"]: row["inner_metrics"] for row in parallel_report["inner_fits"]}
    assert serial_inner.keys() == parallel_inner.keys()
    for candidate_id, inner_metrics in serial_inner.items():
        assert inner_metrics == parallel_inner[candidate_id]


def test_v6_native_seal_neural_trial_and_no_fit_no_label_replay(case, monkeypatch):
    plan = case.plan
    assert plan["schema_version"] == "magnitude_search_plan.v3"
    assert plan["candidate_count"] == 13 and plan["planned_fit_count"] == 14
    assert plan["counts"]["total_component_fits"] == 18
    neural = [c for c in plan["candidates"] if c["recipe"].get("model") == "mlp"]
    assert len(neural) == 3
    assert {len(c["recipe"]["features"]) for c in neural} == {11, 28, 43}
    assert len([c for c in plan["candidates"] if c["recipe"]["schema_version"] ==
                "size_history_mean_candidate.v1"]) == 1
    assert planner.read_search_plan(case.conn, case.store, case.registration, case.request, case.ref) == plan
    selected = min(neural, key=lambda c: len(c["recipe"]["features"]))
    recorder = trials.SearchTrials(*case.context)
    receipt = recorder.execute("2024", "2023", selected["candidate_id"], limits=case.request["limits"]["child"])
    assert receipt["status"] == receipt["terminal"]["status"] == "succeeded"
    assert receipt["fit_attempt_count"] == receipt["completed_fit_count"] == 1
    assert receipt["diagnostics"][0]["component_fits"] == 1
    assert receipt["diagnostics"][0]["converged"] is True
    assert receipt["evaluation_authority"] == "none"
    # New recorder proves durable reconstruction, rather than a warm fit/input memo.
    replay = trials.SearchTrials(*case.context)
    before = case.conn.total_changes
    poison_outcomes(monkeypatch, case.repository, case.panel_source, (), all_outcomes=True)
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *args: pytest.fail("replay launched a child"))
    monkeypatch.setattr(trials, "evaluation_targets", lambda *args: pytest.fail("replay opened labels"))
    monkeypatch.setattr(case.store, "publish_bytes", lambda *args, **kwargs: pytest.fail("replay wrote an artifact"))
    with no_fit_guard(), legacy_no_fit_guard():
        assert replay.execute("2024", "2023", selected["candidate_id"],
                              limits=case.request["limits"]["child"]) == receipt
    assert case.conn.total_changes == before


def test_v6_neural_selection_outer_and_report_replay(case, monkeypatch):
    """Real neural/fallback children and journals; explicit cheap numeric rivals."""
    from experiments import magnitude_search_outer as outer
    from experiments import size_comparison_run as comparison
    from experiments.magnitude_search_inputs import load_fit_inputs
    from experiments.size_search_report import collect
    from experiments.v2_magnitude_search_run import run_search
    from tests.v2.research.test_size_candidate_lifecycle import _generic_result

    native = trials.run_fitted_trial
    outer_inputs = []
    jobs = {"mlp": 0, "size_v1_4": 0, "comparator": 0, "numeric_rivals": 0}

    def worker(source, fold, plan, execution, resolved, progress):
        inputs = load_fit_inputs(case.store, source["manifest"], source["inputs_id"],
            fold["evaluation_event_ids"], plan["horizons"], execution["features"])
        assert set(inputs) == {"training", "features"}
        assert {row["event_id"] for row in inputs["training"]}.isdisjoint(fold["evaluation_event_ids"])
        assert all("absolute_move_pct" not in row for row in inputs["features"])
        family = execution.get("model")
        if family in ("mlp", "size_v1_4"):
            jobs[family] += 1
            if family == "mlp" and fold["year"] == "2024":
                outer_inputs.append((inputs, fold, execution))
            return native(source, fold, plan, execution, resolved, progress)
        if execution["schema_version"] == "size_comparison_execution.v1":
            jobs["comparator"] += 1
            return native(source, fold, plan, execution, resolved, progress)
        jobs["numeric_rivals"] += 1
        return _generic_result(inputs, fold, plan, execution, progress)

    for module in (trials, outer, comparison):
        monkeypatch.setattr(module, "run_fitted_trial", worker)
    recorder = trials.SearchTrials(*case.context)
    for candidate in case.plan["candidates"]:
        recorder.execute("2024", "2023", candidate["candidate_id"], limits=case.request["limits"]["child"])
    selection = recorder.freeze()
    chosen = selection["selections"][0]["selected_candidate_id"]
    candidate = next(c for c in case.plan["candidates"] if c["candidate_id"] == chosen)
    assert candidate["recipe"].get("model") == "mlp"
    result = outer.execute_outer_fold(*case.context, "2024", limits=case.request["limits"]["child"])
    assert len(result["predictions"]) == 1
    assert candidate["recipe"]["profile"] == PROFILES[0]
    (inputs, fold, execution), = outer_inputs
    indexed = {row["event_id"]: row for row in inputs["features"] if row["horizon_sessions"] == 1}
    train = inputs["training"]
    matrices = [[[indexed[event][column] for column in execution["features"]] for event in events]
                for events in ([row["event_id"] for row in train], fold["evaluation_event_ids"])]
    # Convert JSON nulls to NaN at the numeric boundary, before missing indicators.
    matrices = [np.asarray(matrix, dtype=float) for matrix in matrices]
    _, expected = independent_oracle(matrices[0], [row["absolute_move_pct"] for row in train], matrices[1])
    assert all(value > 0 for value in expected)
    assert [(row["event_id"], row["horizon_sessions"], row["predicted_absolute_move_pct"])
            for row in result["predictions"]] == [
                (event, 1, value) for event, value in zip(fold["evaluation_event_ids"], expected)]
    report = collect(*case.context)
    assert report["paired"]["challenger_abstentions"] == 0
    assert report["paired"]["broad_coverage_metrics"]["challenger"]["count"] == 1
    assert jobs == {"mlp": 4, "size_v1_4": 1, "comparator": 1, "numeric_rivals": 9}
    before = case.conn.total_changes
    # The unchanged comparator revalidates admitted training targets during replay.
    # Prohibit new outer evaluation reads, fitting and writes, not that existing check.
    monkeypatch.setattr(outer, "load_magnitude_targets", lambda *a, **k: pytest.fail("replay outer labels"))
    monkeypatch.setattr(case.store, "publish_bytes", lambda *a, **k: pytest.fail("replay artifact write"))
    for module in (trials, outer, comparison):
        monkeypatch.setattr(module, "run_fitted_trial", lambda *a, **k: pytest.fail("replay fit"))
    with no_fit_guard(), legacy_no_fit_guard():
        assert run_search(*case.context, limits=case.request["limits"]["child"]) == report["search"]
        assert collect(*case.context) == report
    assert case.conn.total_changes == before
