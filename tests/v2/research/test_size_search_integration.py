"""Synthetic native source→stored panel→v2 journal→bounded child→comparison path."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import SystemClock, content_hash, from_document
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments import eligible_history, magnitude_search_request_versions, size_panel
from experiments import magnitude_search_plan as planner
from experiments import size_search_binding as binding
from experiments.native_registration import reserve_native_registration, verify_native_registration
from experiments.size_search_report import collect
from experiments.v2_magnitude_search_run import run_search, seal_search_run
from tests.v2.research.magnitude_trial_support import poison_outcomes, prediction_spec
from tests.v2.research.test_magnitude_panel_qualified import panel_data
from tests.v2.research.test_magnitude_search_request import request as request_fixture

# Share the bounded native journal fixture instead of rebuilding it on each xdist worker.
pytestmark = pytest.mark.xdist_group("size_search_integration")


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("size-v2")
    generator = panel_data.__wrapped__(root, SimpleNamespace())
    data = next(generator)
    ids = [e["event_id"] for e in data.events if str(e["event_date"].date()) in
           ("2022-06-02", "2023-01-03", "2023-06-02", "2024-06-17")]
    history_request = {**data.request, "limits": data.panel_request["limits"], "event_ids": ids}
    history = eligible_history.materialize(data.repository, data.panel_source, **history_request)
    panel_request = {**data.panel_request, "event_ids": ids, "history_snapshot": history["feature_snapshot"]}
    panel = size_panel.materialize(data.repository, data.panel_source, **panel_request)
    panel_request = {**panel_request, "feature_snapshot_id": panel["feature_snapshot"].snapshot_id,
        "history_snapshot": history["feature_snapshot"].snapshot_id,
        "price_history_snapshot": data.prices.snapshot_id}
    request = request_fixture()
    request.update(feature_dictionary_hash=recipe("HMO")["dictionary_hash"],
        feature_snapshot_id=panel["feature_snapshot"].snapshot_id, history_snapshot_id=history["feature_snapshot"].snapshot_id,
        outer_years=["2024"], inner_years={"2024": ["2023"]}, max_fits=102)
    request["eligibility_policy_ref"].update(plan_id=panel["context"]["eligibility_plan"]["plan_id"])
    request["limits"] = {"child": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60},
        "parent": {"memory_bytes": 3 << 30, "wall_seconds": 600},
        "build": {"memory_bytes": 3 << 30, "wall_seconds": 60}, "job": {"wall_seconds": 600}}
    spec = prediction_spec(experiment_id="synthetic-qualified-size", folds=("2024",), runner="experiments/size_search.py")
    args = dict(scope="panel-source", training_start="2022-01-01", labels_before="2025-01-01")
    registration, spec = binding.prepare(data.repository, spec, request, panel_request, **args)
    store = data.repository._store
    reserve_native_registration(data.conn, store, registration, spec)
    sealed = planner.seal_search_plan(data.conn, store, registration, request)
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    context = (data.conn, store, data.repository, registration, request, ref)
    seal_search_run(*context, clock=SystemClock())
    result = SimpleNamespace(**{**vars(data), "root": root})
    result.__dict__.update(panel=panel, request=request, panel_request=panel_request,
        spec=spec, registration=registration, store=store, plan=sealed["plan"], ref=ref, context=context, args=args)
    yield result
    try:
        next(generator)
    except StopIteration:
        pass


@pytest.fixture(scope="module")
def lead_one_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("size-v2-lead-one")
    generator = panel_data.__wrapped__(root, SimpleNamespace())
    data = next(generator)
    ids = [e["event_id"] for e in data.events if str(e["event_date"].date()) in
           ("2022-06-02", "2023-01-03", "2023-06-02", "2024-06-17")]
    history_request = {**data.request, "limits": data.panel_request["limits"], "event_ids": ids}
    history_request["horizons"] = [1]
    history = eligible_history.materialize(data.repository, data.panel_source, **history_request)
    panel_request = {**data.panel_request, "event_ids": ids, "history_snapshot": history["feature_snapshot"]}
    panel_request["horizons"] = [1]
    panel = size_panel.materialize(data.repository, data.panel_source, **panel_request)
    panel_request = {**panel_request, "feature_snapshot_id": panel["feature_snapshot"].snapshot_id,
        "history_snapshot": history["feature_snapshot"].snapshot_id,
        "price_history_snapshot": data.prices.snapshot_id}
    request = request_fixture()
    request.update(feature_dictionary_hash=recipe("HMO")["dictionary_hash"],
        feature_snapshot_id=panel["feature_snapshot"].snapshot_id, history_snapshot_id=history["feature_snapshot"].snapshot_id,
        outer_years=["2024"], inner_years={"2024": ["2023"]}, max_fits=34)
    request["eligibility_policy_ref"].update(plan_id=panel["context"]["eligibility_plan"]["plan_id"])
    request["limits"] = {"child": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60},
        "parent": {"memory_bytes": 3 << 30, "wall_seconds": 600},
        "build": {"memory_bytes": 3 << 30, "wall_seconds": 60}, "job": {"wall_seconds": 600}}
    spec = prediction_spec(experiment_id="synthetic-qualified-size-lead-one", folds=("2024",),
                           runner="experiments/size_search.py")
    args = dict(scope="panel-source", training_start="2022-01-01", labels_before="2025-01-01")
    registration, spec = binding.prepare(data.repository, spec, request, panel_request, **args)
    store = data.repository._store
    reserve_native_registration(data.conn, store, registration, spec)
    sealed = planner.seal_search_plan(data.conn, store, registration, request)
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    context = (data.conn, store, data.repository, registration, request, ref)
    seal_search_run(*context, clock=SystemClock())
    result = SimpleNamespace(**{**vars(data), "root": root})
    result.__dict__.update(panel=panel, request=request, panel_request=panel_request,
        spec=spec, registration=registration, store=store, plan=sealed["plan"], ref=ref, context=context, args=args)
    yield result
    try:
        next(generator)
    except StopIteration:
        pass


def test_lead_one_seals_single_horizon_plan_and_preserves_v2_request_identity(case, lead_one_case):
    plan = lead_one_case.plan
    assert plan["horizons"] == [1]
    assert plan["planned_fit_count"] == 34
    assert plan["counts"]["search_fits"] == 34
    assert plan["counts"]["baseline_logical_fits"] == 1
    assert plan["counts"]["baseline_component_fits"] == 2
    assert plan["counts"]["total_component_fits"] == 36
    multi = magnitude_search_request_versions.compile_request(case.request, horizons=(5, 3, 1))
    lead = magnitude_search_request_versions.compile_request(case.request, horizons=(1,))
    assert multi["request_id"] == lead["request_id"]
    assert multi["candidate_specs"] == lead["candidate_specs"]
    assert multi["counts"]["search_fits"] == 102
    assert lead["counts"]["search_fits"] == 34


def test_contain_accepts_request_when_given_single_horizon(case, monkeypatch):
    from experiments import size_search
    monkeypatch.setattr(size_search.signal, "signal", lambda *args: None)
    monkeypatch.setattr(size_search.signal, "alarm", lambda seconds: 0)
    assert size_search._contain(case.request, (1,)) == case.request["limits"]["parent"]["memory_bytes"]


def test_native_v17_anchor_and_explicit_v2_fitted_identities(case):
    verify_native_registration(case.conn, case.store, case.registration)
    assert case.registration.document["schema_version"] == "native_experiment_registration.v1.7"
    plan = case.plan
    assert plan["schema_version"] == "magnitude_search_plan.v3"
    assert plan["candidate_count"] == 33 and plan["planned_fit_count"] == 102
    assert plan["counts"]["total_component_fits"] == 108
    assert plan["horizons"] == [5, 3, 1]
    assert case.registration.document["size_comparison"]["component_parameters"]["mlp"]["random_state"] == 20260829
    assert len({c["candidate_id"] for c in plan["candidates"]}) == 33
    assert all(c["candidate_id"] == content_hash({k: v for k, v in c.items() if k != "candidate_id"})
               and c["source_binding_id"] == content_hash(json.loads(case.registration.binding_json)["size_panel"]) for c in plan["candidates"])
    assert {len(c["recipe"]["features"]) for c in plan["candidates"]} == {11, 28, 43}


def test_v3_qualified_request_seals_reconstructs_v2_plan(case):
    v3 = deepcopy(case.request)
    v3["schema_version"] = "magnitude_search_request.v3"
    v3["space"] = {
        "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "elasticnet": {"alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 1.0],
                       "l1_ratio": [0.25, 0.75]},
        "hgb": {"max_leaf_nodes": [7, 15]}}
    v3.update(max_candidates=57, max_fits=174)
    spec = prediction_spec(experiment_id="synthetic-qualified-size-v3", folds=("2024",),
                           runner="experiments/size_search.py")
    registration, spec = binding.prepare(case.repository, spec, v3, case.panel_request, **case.args)
    reserve_native_registration(case.conn, case.store, registration, spec)
    verify_native_registration(case.conn, case.store, registration)
    sealed = planner.seal_search_plan(case.conn, case.store, registration, v3)
    assert sealed["plan"]["schema_version"] == "magnitude_search_plan.v2"
    assert sealed["plan"]["selection"] == {"metric": "inner_pooled_rmse", "ties": "candidate_id_ascending",
        "retries": 0, "outer": "frozen_selected_only", "final_read_authority": "none"}
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    assert planner.read_search_plan(case.conn, case.store, registration, v3, ref) == sealed["plan"]


def test_v4_qualified_request_seals_reconstructs_v3_plan(case):
    v4 = deepcopy(case.request)
    v4["schema_version"] = "magnitude_search_request.v4"
    v4["space"] = {
        "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "elasticnet": {"alpha": [0.001, 0.003, 0.01, 0.03, 0.1, 1.0],
                       "l1_ratio": [0.25, 0.75]},
        "hgb": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 15],
                "min_samples_leaf": [20, 100], "l2_regularization": [0.0, 1.0]}}
    v4.update(max_candidates=99, max_fits=300)
    v4["selection"] = {
        "metric_weights": {"rmse": 0.6, "mae": 0.3, "large_move_underprediction": 0.1},
        "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
        "large_move_threshold_pct": 10.0,
        "tie_band_floor": 0.005,
    }
    spec = prediction_spec(experiment_id="synthetic-qualified-size-v4", folds=("2024",),
                           runner="experiments/size_search.py")
    registration, spec = binding.prepare(case.repository, spec, v4, case.panel_request, **case.args)
    reserve_native_registration(case.conn, case.store, registration, spec)
    verify_native_registration(case.conn, case.store, registration)
    sealed = planner.seal_search_plan(case.conn, case.store, registration, v4)
    assert sealed["plan"]["schema_version"] == "magnitude_search_plan.v3"
    assert sealed["plan"]["selection"] == {
        "metric": "inner_pooled_history_mean_normalized_event_blend",
        "metric_weights": {"rmse": 0.6, "mae": 0.3, "large_move_underprediction": 0.1},
        "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
        "large_move_threshold_pct": 10.0,
        "pooling": "all_inner_year_rows",
        "comparator": "history_mean_report_only",
        "tie_band": "max_sample_se_of_best_pooled_candidate_yearly_blends_or_floor",
        "tie_band_floor": 0.005,
        "ties": "candidate_id_ascending",
        "retries": 0,
        "outer": "frozen_selected_only",
        "final_read_authority": "none",
    }
    assert sealed["plan"]["candidate_count"] == 99
    assert sum(candidate["recipe"]["schema_version"] == "magnitude_ridge_hgb_recipe.v2"
               for candidate in sealed["plan"]["candidates"]) == 48
    ref = from_document(ArtifactRef, sealed["plan_ref"])
    assert planner.read_search_plan(case.conn, case.store, registration, v4, ref) == sealed["plan"]


def test_native_registration_detaches_caller_document(case):
    from experiments.native_registration import NativeRegistration

    source = case.registration.document
    registration = NativeRegistration(case.registration.run_id, case.registration.binding_json, source)
    snapshot = registration.document
    source["size_panel_request"]["event_ids"].append("caller-mutation")
    assert registration.document == snapshot


def test_verified_shard_cache_requires_full_binding(case):
    from experiments.magnitude_search_shards import iter_section
    from experiments.size_registration_artifacts import _position

    compact = json.loads(case.registration.binding_json)
    descriptor = next(iter(compact["size_artifacts"].values()))
    section, entry = descriptor["section"], descriptor["entry"]
    ref = from_document(ArtifactRef, entry["shards"][0]["ref"])
    rows = json.loads(case.store.read_verified(ref))["rows"]
    full_key = (section, ref.schema_ref, ref.storage_key, ref.content_hash, ref.byte_size)
    assert len(list(iter_section(case.store, section, entry, _position,
                                 verified_rows={full_key: rows}))) == len(rows)
    old_cache = {ref.storage_key: rows}
    with pytest.raises(OpsError):
        list(iter_section(case.store, section + ":changed", entry, _position, verified_rows=old_cache))

    changed = deepcopy(entry)
    old_hash = changed["shards"][0]["ref"]["content_hash"]
    zero_hash = "sha256:" + "0" * 64
    changed["shards"][0]["ref"]["content_hash"] = zero_hash if old_hash != zero_hash else "sha256:" + "1" * 64
    with pytest.raises(OpsError):
        list(iter_section(case.store, section, changed, _position, verified_rows=old_cache))


def test_v17_admission_checks_shard_boundaries_paths_and_map_rows(case):
    from engine.v2.ops.native_registration import _size_artifact_verified
    from experiments.magnitude_search_shards import write_section
    from experiments.size_registration_artifacts import _position

    compact = json.loads(case.registration.binding_json)
    map_descriptor = next(descriptor for descriptor in compact["size_artifacts"].values()
                          if descriptor["kind"] == "map")
    assert _size_artifact_verified(case.store, map_descriptor)

    bad_boundary = deepcopy(map_descriptor)
    bad_boundary["entry"]["shards"][0]["first_key"] = [100]
    assert not _size_artifact_verified(case.store, bad_boundary)

    unsafe_path = deepcopy(map_descriptor)
    unsafe_path["entry"]["shards"][0]["ref"]["storage_key"] = "../outside.json"
    assert not _size_artifact_verified(case.store, unsafe_path)

    section = "size_registration.v1.7:synthetic-malformed-map"
    malformed_map = {
        "kind": "map",
        "section": section,
        "entry": write_section(case.store, section,
                                [{"position": 0, "row": ["missing-value"]}], _position),
    }
    assert not _size_artifact_verified(case.store, malformed_map)


def test_v17_storage_outages_stay_retryable_through_library_and_cli(case, monkeypatch, capsys):
    from engine.v2.ops.errors import fail
    from experiments import size_search
    from experiments.native_registration import read_native_registration
    from experiments.size_registration_artifacts import hydrate_size_registration

    class RawUnavailableStore:
        root = case.store.root

        def read_verified(self, ref):
            if ref.schema_ref == "magnitude_search_shard.v1":
                raise OSError("synthetic storage I/O failure")
            return case.store.read_verified(ref)

    with pytest.raises(OSError, match="synthetic storage I/O failure"):
        read_native_registration(case.conn, RawUnavailableStore(), case.spec)

    class TypedUnavailableStore:
        root = case.store.root

        def read_verified(self, ref):
            if ref.schema_ref == "magnitude_search_shard.v1":
                raise fail("RESOURCE_UNAVAILABLE", "synthetic shard I/O failure")
            return case.store.read_verified(ref)

    with pytest.raises(OpsError) as caught:
        hydrate_size_registration(TypedUnavailableStore(), json.loads(case.registration.binding_json))
    assert caught.value.code == "RESOURCE_UNAVAILABLE"
    assert caught.value.problem.retryable is True

    class ConnectionProxy:
        def __getattr__(self, name):
            return getattr(case.conn, name)

        def close(self):
            pass

    monkeypatch.setattr(size_search, "_read_document", lambda path: case.request if path == "request.json" else {})
    monkeypatch.setattr(size_search, "_contain", lambda request, horizons=(5, 3, 1): 2 ** 33)
    monkeypatch.setattr(size_search, "experiment_spec_from_document", lambda document: case.spec)
    monkeypatch.setattr(size_search, "connect", lambda *args, **kwargs: ConnectionProxy())
    monkeypatch.setattr(size_search, "ArtifactStore", lambda root: RawUnavailableStore())
    monkeypatch.setattr(size_search, "Repository", lambda *args, **kwargs: object())
    result = size_search.main([
        "report", "--spec", "spec.json", "--request", "request.json",
        "--catalog", "catalog.sqlite", "--store-root", str(case.store.root),
        "--plan-ref", "plan-ref.json", "--no-ledger",
    ])
    assert result == 2
    assert json.loads(capsys.readouterr().out)["refused"] == "RESOURCE_UNAVAILABLE"


def test_preflight_missing_inner_year_refuses_before_reservation(case):
    changed = deepcopy(case.request)
    changed["inner_years"] = {"2024": ["2020"]}
    before = case.conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0]
    with pytest.raises(Exception):
        binding.prepare(case.repository, case.spec, changed, case.panel_request, **case.args)
    assert case.conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == before


def test_replay_stale_producer_and_request_mismatch_refuse(case, monkeypatch):
    monkeypatch.setattr(binding, "source_closure", lambda *a: {})
    with pytest.raises(OpsError, match="differs"):
        planner.read_search_plan(case.conn, case.store, case.registration, case.request, case.ref)


def test_no_fit_probe_does_not_claim_comparator(case):
    from experiments.size_comparison_run import execute
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    fold = {**case.plan["folds"][0]["outer"], "evaluation_labels_before": "2025-01-01"}
    with no_fit_guard(), pytest.raises(RuntimeFitForbidden):
        execute(case.conn, case.store, case.repository, case.registration, case.plan, fold,
                limits=case.request["limits"]["child"])
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before


def test_real_serial_and_parallel_trials_have_equal_journaled_fit_results(tmp_path, monkeypatch):
    def build(root, workers):
        root.mkdir()
        generator = panel_data.__wrapped__(root, SimpleNamespace())
        data = next(generator)
        ids = [e["event_id"] for e in data.events if str(e["event_date"].date()) in
               ("2022-06-02", "2023-01-03", "2023-06-02", "2024-06-17")]
        history_request = {**data.request, "limits": data.panel_request["limits"],
                           "event_ids": ids, "horizons": [1]}
        history = eligible_history.materialize(data.repository, data.panel_source, **history_request)
        panel_request = {**data.panel_request, "event_ids": ids,
                         "history_snapshot": history["feature_snapshot"], "horizons": [1]}
        panel = size_panel.materialize(data.repository, data.panel_source, **panel_request)
        panel_request.update(feature_snapshot_id=panel["feature_snapshot"].snapshot_id,
            history_snapshot=history["feature_snapshot"].snapshot_id,
            price_history_snapshot=data.prices.snapshot_id)
        request = request_fixture()
        request.update(feature_dictionary_hash=recipe("HMO")["dictionary_hash"],
            feature_snapshot_id=panel["feature_snapshot"].snapshot_id,
            history_snapshot_id=history["feature_snapshot"].snapshot_id,
            outer_years=["2024"], inner_years={"2024": ["2023"]},
            max_candidates=33, max_fits=34)
        request["eligibility_policy_ref"].update(
            plan_id=panel["context"]["eligibility_plan"]["plan_id"])
        request["comparison"].update(max_logical_fits=1, max_component_fits=2)
        request["limits"] = {
            "child": {"memory_bytes": 1 << 30, "cpu_seconds": 30, "wall_seconds": 60},
            "parent": {"memory_bytes": 2 << 30, "wall_seconds": 600},
            "build": {"memory_bytes": 3 << 30, "wall_seconds": 60},
            "job": {"wall_seconds": 600}}
        spec = prediction_spec(experiment_id="synthetic-qualified-size-parity",
                               folds=("2024",), runner="experiments/size_search.py")
        args = dict(scope="panel-source", training_start="2022-01-01",
                    labels_before="2025-01-01")
        registration, spec = binding.prepare(data.repository, spec, request, panel_request, **args)
        store = data.repository._store
        reserve_native_registration(data.conn, store, registration, spec)
        sealed = planner.seal_search_plan(data.conn, store, registration, request)
        ref = from_document(ArtifactRef, sealed["plan_ref"])
        context = (data.conn, store, data.repository, registration, request, ref)
        seal_search_run(*context, clock=SystemClock())
        request["workers"] = workers
        return context, sealed["plan"], generator, data

    serial, serial_plan, serial_generator, serial_data = build(tmp_path / "serial", 1)
    parallel, parallel_plan, parallel_generator, parallel_data = build(tmp_path / "parallel", 2)
    try:
        # The closed production v2 grid has 33 candidates; one fold and horizon keep it bounded.
        assert len(serial_plan["candidates"]) == len(parallel_plan["candidates"]) == 33
        assert serial_plan["folds"] == parallel_plan["folds"]
        assert [candidate["candidate_id"] for candidate in serial_plan["candidates"]] == [
            candidate["candidate_id"] for candidate in parallel_plan["candidates"]]

        from experiments import v2_magnitude_search_run as search_run_module
        original_executor = search_run_module.ProcessPoolExecutor
        executor_calls = []

        def spy_executor(*args, **kwargs):
            executor_calls.append(kwargs.get("max_workers"))
            return original_executor(*args, **kwargs)

        monkeypatch.setattr(search_run_module, "ProcessPoolExecutor", spy_executor)

        run_search(*serial, limits=serial[4]["limits"]["child"])
        serial_executor_calls = list(executor_calls)
        run_search(*parallel, limits=parallel[4]["limits"]["child"])
        parallel_executor_calls = executor_calls[len(serial_executor_calls):]

        assert serial_executor_calls == []
        assert parallel_executor_calls == [2]

        from experiments.magnitude_search_plan import trial_key
        from experiments.magnitude_search_records import read_record

        def without_operational_fields(value):
            if isinstance(value, dict):
                return {key: without_operational_fields(item) for key, item in value.items()
                        if not (key.startswith("worker_") or "elapsed" in key.lower()
                                or "duration" in key.lower() or "timing" in key.lower())}
            if isinstance(value, list):
                return [without_operational_fields(item) for item in value]
            return value

        def journaled(context, plan):
            _, _, _, registration, _, _ = context
            rows = []
            for candidate in plan["candidates"]:
                key = trial_key(plan, "2024", "2023", candidate["candidate_id"])
                execution = read_record(context[0], context[1], registration.run_id,
                                        "execution:" + key)
                terminal = read_record(context[0], context[1], registration.run_id,
                                       "terminal:" + key)
                errors = [abs(row["predicted_absolute_move_pct"] - row["absolute_move_pct"])
                          for row in terminal["predictions"]]
                assert terminal["metrics"] == {
                    "count": len(errors),
                    "mae": pytest.approx(sum(errors) / len(errors)),
                    "rmse": pytest.approx((sum(error * error for error in errors) / len(errors)) ** 0.5),
                }
                rows.append({
                    "candidate_id": candidate["candidate_id"], "trial_id": key,
                    "attempt": execution["attempt"], "fits": execution["fits"],
                    "diagnostics": without_operational_fields(execution["diagnostics"]),
                    "forecasts_id": execution["forecasts_id"],
                    "terminal": {name: terminal[name] for name in (
                        "status", "inputs_id", "validation_id", "evaluation_inputs",
                        "predictions", "metrics", "inner_metrics") if name in terminal},
                })
            selection = read_record(context[0], context[1], registration.run_id, "selection")
            # terminal_id hashes the whole receipt, including runtime telemetry; compare
            # the trial identities and scored selection result it refers to instead.
            selected = {name: selection[name] for name in (
                "plan_id", "selections", "prospective_candidate_id",
                "estimator_execution", "evaluation_authority")}
            selected["attempts"] = [row["trial_id"] for row in selection["attempts"]]
            return rows, selected

        assert journaled(serial, serial_plan) == journaled(parallel, parallel_plan)
    finally:
        serial_data.conn.close()
        parallel_data.conn.close()
        for generator in (serial_generator, parallel_generator):
            try:
                next(generator)
            except StopIteration:
                pass


def test_complete_synthetic_serial_search_child_diagnostics(case, monkeypatch):
    poison_outcomes(monkeypatch, case.repository, case.panel_source, ["2023-03-20", "2023-04-03", "2023-05-15"])
    for producer in (size_panel, eligible_history):
        monkeypatch.setattr(producer, "materialize", lambda *a, **k: pytest.fail("hidden feature rebuild"))
    import experiments.magnitude_search_outer as outer
    import experiments.magnitude_search_trials as trials
    import experiments.size_comparison_run as comparison
    jobs = {"inner_candidates": 0, "outer_challengers": 0, "outer_comparators": 0}
    for module, role in ((trials, "inner_candidates"), (outer, "outer_challengers"), (comparison, "outer_comparators")):
        original = module.run_fitted_trial
        def counted(*args, _original=original, _role=role, **kwargs):
            jobs[_role] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, "run_fitted_trial", counted)
    first = run_search(*case.context, limits=case.request["limits"]["child"])
    report = collect(*case.context)
    assert report["search"] == first
    assert jobs == {"inner_candidates": 33, "outer_challengers": 1, "outer_comparators": 1}
    assert report["preflight"]["child_jobs"] == jobs
    assert report["preflight"]["maximum_child_invocations"] == sum(jobs.values()) == 35
    assert len(report["inner_fits"]) == 33
    assert sum(f["completed_fit_count"] for f in report["inner_fits"]) == 99
    assert sum(f["component_fits"] for f in report["comparison_fits"]) == 0
    assert sum(len(f["diagnostics"]) for f in report["inner_fits"]) == 99
    assert report["paired"]["eligible_events"] == 1
    assert report["paired"]["comparison_status"] == "unavailable_no_complete_matched_events"
    assert report["paired"]["broad_coverage_metrics"]["history_mean"]["count"] == 3
    from experiments.magnitude_search_records import read_record
    diagnostics = []
    for candidate in case.plan["candidates"]:
        key = planner.trial_key(case.plan, "2024", "2023", candidate["candidate_id"])
        receipt = read_record(case.conn, case.store, case.registration.run_id, "execution:" + key)
        assert receipt["fit_attempt_count"] == 3
        if candidate["recipe"]["model"] == "elasticnet":
            diagnostics.extend(receipt["diagnostics"])
    assert len(diagnostics) == 54 and all(d["converged"] for d in diagnostics)


def test_completed_search_exact_and_expired_replay(completed_case, monkeypatch):
    case = completed_case
    poison_outcomes(monkeypatch, case.repository, case.panel_source, ["2023-03-20", "2023-04-03", "2023-05-15"])
    for producer in (size_panel, eligible_history):
        monkeypatch.setattr(producer, "materialize", lambda *a, **k: pytest.fail("hidden feature rebuild"))
    report = collect(*case.context)
    first = report["search"]
    import experiments.magnitude_search_outer as outer
    import experiments.magnitude_search_trials as trials
    import experiments.size_comparison_run as comparison
    from experiments.magnitude_search_records import read_record
    for module in (trials, outer, comparison):
        monkeypatch.setattr(module, "run_fitted_trial", lambda *a, **k: pytest.fail("retry fit"))
    assert run_search(*case.context, limits=case.request["limits"]["child"]) == first
    assert collect(*case.context) == report
    from datetime import timedelta

    from engine.v2.foundation import parse_timestamp
    from experiments import size_job_budget, size_search
    saved = read_record(case.conn, case.store, case.registration.run_id, "size-job-deadline")
    frozen = SimpleNamespace(now=lambda: parse_timestamp(saved["deadline"]) + timedelta(seconds=1))
    monkeypatch.setattr(size_job_budget, "SystemClock", lambda: frozen)
    monkeypatch.setattr(size_search, "run_search", lambda *a, **k: pytest.fail("completed CLI dispatched"))
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    assert size_search._run(case.context) == report
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before



@pytest.mark.parametrize("field", ["size_request", "size_panel_request", "size_panel", "size_comparison"])
def test_native_v17_missing_or_nondictionary_additive_field_refuses(case, field):
    from engine.v2.ops.native_registration import _magnitude_binding
    for bad in (None, [], "untrusted"):
        document = case.registration.document
        document[field] = bad
        assert _magnitude_binding(document) is False
    document = case.registration.document
    del document[field]
    assert _magnitude_binding(document) is False


@pytest.mark.parametrize("field", ["memory_bytes", "cpu_seconds", "wall_seconds"])
def test_public_comparator_limit_drift_refuses_before_inputs(case, monkeypatch, field):
    from experiments import size_comparison_run as comparison
    monkeypatch.setattr(comparison, "fold_inputs", lambda *a: pytest.fail("opened inputs"))
    limits = dict(case.request["limits"]["child"])
    limits[field] += 1
    fold = {**case.plan["folds"][0]["outer"], "evaluation_labels_before": "2025-01-01"}
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    with pytest.raises(OpsError, match="differs"):
        comparison.execute(case.conn, case.store, case.repository, case.registration, case.plan, fold, limits=limits)
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before


@pytest.fixture(scope="module")
def completed_case(case):
    """Explicit setup: standalone poison selections also get a completed journal."""
    run_search(*case.context, limits=case.request["limits"]["child"])
    return case


@pytest.mark.parametrize("slot,mutation", [("start", "missing"), ("start", "changed"),
    ("inputs", "missing"), ("inputs", "changed"), ("terminal", "missing"),
    ("terminal", "result"), ("started:1", "missing"), ("completed:1", "missing"),
    ("started:1", "changed"), ("completed:1", "changed"),
    ("terminal", "missing_result"), ("terminal", "result_none")])
def test_report_replays_comparator_evidence(completed_case, monkeypatch, slot, mutation):
    case = completed_case
    from datetime import timedelta

    from engine.v2.foundation import parse_timestamp
    from experiments import size_job_budget
    from experiments.magnitude_search_records import read_record
    saved = read_record(case.conn, case.store, case.registration.run_id, "size-job-deadline")
    frozen = SimpleNamespace(now=lambda: parse_timestamp(saved["deadline"]) + timedelta(seconds=1))
    monkeypatch.setattr(size_job_budget, "SystemClock", lambda: frozen)

    from experiments import magnitude_search_outer as outer
    from experiments import magnitude_search_trials as trials
    from experiments import size_comparison_run as comparison
    key = content_hash([case.plan["plan_id"], "2024", "comparison"])
    target = "comparison-" + slot + ":" + key
    original = comparison.read_record
    observed = []
    def read(*args):
        value = original(*args)
        if args[-1] == target:
            observed.append(target)
            if mutation == "missing":
                return None
            value = deepcopy(value)
            if mutation == "missing_result":
                del value["result"]
            elif mutation == "result_none":
                value["result"] = None
            elif mutation == "result":
                value["result"]["forecasts"][0]["history_mean"] = 987654.
            else:
                value["planted_mismatch"] = True
        return value
    monkeypatch.setattr(comparison, "read_record", read)
    for module in (comparison, outer, trials):
        monkeypatch.setattr(module, "run_fitted_trial", lambda *a, **k: pytest.fail("report launched fit"))
    for name in ("append_record", "publish_inputs", "resolved_limits"):
        monkeypatch.setattr(comparison, name, lambda *a, **k: pytest.fail("report wrote or claimed budget"))
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    with pytest.raises(OpsError) as caught:
        collect(*case.context)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert observed == [target]
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before


@pytest.mark.parametrize("headroom", [-1, 0, 1])
def test_prepared_registration_byte_boundary_matches_reserve_and_verify(case, monkeypatch, headroom):
    from experiments.native_registration import read_native_registration
    expected = case.registration.binding_json
    budget = len(expected) + headroom
    monkeypatch.setattr(binding, "MAX_NATIVE_EVIDENCE_BYTES", budget)
    before = case.conn.total_changes
    if headroom < 0:
        monkeypatch.setattr(binding, "_folds", lambda *a: pytest.fail("oversize binding reached fold work"))
        for action in (
            lambda: binding.prepare(case.repository, case.spec, case.request, case.panel_request, **case.args),
            lambda: reserve_native_registration(case.conn, case.store, case.registration, case.spec,
                                                 max_evidence_bytes=budget),
        ):
            with pytest.raises(OpsError) as caught:
                action()
            assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
            assert caught.value.problem.message == "native registration exceeds its UTF-8 byte budget"
            assert caught.value.problem.details == {"registration_bytes": len(expected), "max_registration_bytes": budget}
        for action in (
            lambda: verify_native_registration(case.conn, case.store, case.registration, max_evidence_bytes=budget),
            lambda: read_native_registration(case.conn, case.store, case.spec, max_evidence_bytes=budget),
        ):
            with pytest.raises(OpsError) as caught:
                action()
            assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
            assert caught.value.problem.message == "native registration exceeds its UTF-8 byte budget"
            assert caught.value.problem.details == {"registration_bytes": len(expected), "max_registration_bytes": budget}
    else:
        prepared, spec = binding.prepare(case.repository, case.spec, case.request, case.panel_request, **case.args)
        assert prepared.binding_json == expected
        assert prepared.document == case.registration.document
        assert reserve_native_registration(case.conn, case.store, prepared, spec,
                                            max_evidence_bytes=budget) == prepared
        verify_native_registration(case.conn, case.store, prepared, max_evidence_bytes=budget)
        assert read_native_registration(case.conn, case.store, spec, max_evidence_bytes=budget) == prepared
        assert binding.prepare_plan(case.conn, case.store, prepared, case.request) == case.plan
    assert case.conn.total_changes == before


@pytest.mark.parametrize("kind", ["execution", "terminal", "outer-forecasts", "comparison-terminal"])
def test_report_native_absent_record_refuses_before_presentation(completed_case, monkeypatch, kind):
    from experiments import magnitude_search_outer as outer
    from experiments import magnitude_search_records as records
    from experiments import magnitude_search_trials as trials
    from experiments import size_comparison_run as comparison
    from experiments import size_search_report as presentation
    case = completed_case
    group, candidate = case.plan["folds"][0], case.plan["candidates"][0]
    if kind in ("execution", "terminal"):
        key = planner.trial_key(case.plan, group["outer"]["year"], group["inner"][0]["year"],
                                candidate["candidate_id"])
    elif kind == "comparison-terminal":
        key = content_hash([case.plan["plan_id"], group["outer"]["year"], "comparison"])
    else:
        key = content_hash([case.plan["plan_id"], group["outer"]["year"]])
    target = kind + ":" + key
    native_read, seen = records.read_record, []
    assert native_read(case.conn, case.store, case.registration.run_id, target) is not None
    assert native_read(case.conn, case.store, case.registration.run_id, "never-published") is None
    def read(conn, store, run_id, slot):
        if slot == target:
            seen.append(slot)
            slot = "never-published"
        return native_read(conn, store, run_id, slot)
    def forbidden(*a, **k):
        pytest.fail("read-only report performed work or reached unvalidated presentation")
    for module in (outer, trials, comparison, presentation):
        monkeypatch.setattr(module, "read_record", read)
    for module in (outer, trials, comparison):
        for name in ("run_fitted_trial", "publish_inputs"):
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(records, "transaction", forbidden)
    monkeypatch.setattr(case.store, "publish_bytes", forbidden)
    monkeypatch.setattr(presentation, "paired_report", forbidden)
    before = case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0]
    with pytest.raises(OpsError) as caught:
        collect(*case.context)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert seen == [target]
    assert case.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == before


@pytest.mark.parametrize("kind", ["inner", "outer"])
def test_public_fold_inputs_accepts_exact_registered_fold(case, monkeypatch, kind):
    poison_outcomes(monkeypatch, case.repository, case.panel_source, ["2023-03-20", "2023-04-03", "2023-05-15"])
    group = case.plan["folds"][0]
    fold = group["inner"][0] if kind == "inner" else {**group["outer"],
        "evaluation_labels_before": case.registration.document["magnitude_execution"]["evaluation_labels_before"]}
    inputs = binding.fold_inputs(case.repository, case.registration.document, fold)
    training = {row["event_id"] for row in inputs["training"]}
    assert training == set(fold["training_candidate_event_ids"])
    assert training.isdisjoint(fold["evaluation_event_ids"])
    assert {row["event_id"] for row in inputs["features"]} == training | set(fold["evaluation_event_ids"])


@pytest.mark.parametrize("change", ["cutoff", "training", "evaluation", "order", "year", "evaluation_cutoff", "extra", "missing", "none", "evaluation_in_training"])
def test_public_fold_inputs_refuses_caller_drift_before_dependent_reads(case, monkeypatch, change):
    fold = {**deepcopy(case.plan["folds"][0]["outer"]),
        "evaluation_labels_before": case.registration.document["magnitude_execution"]["evaluation_labels_before"]}
    if change in ("cutoff", "evaluation_in_training"):
        fold["training_labels_before"] = "2025-01-01"
    if change in ("training", "evaluation_in_training"):
        fold["training_candidate_event_ids"] += fold["evaluation_event_ids"]
    elif change == "evaluation":
        fold["evaluation_event_ids"] = fold["training_candidate_event_ids"][:1]
    elif change == "order":
        assert len(fold["training_candidate_event_ids"]) > 1
        fold["training_candidate_event_ids"].reverse()
    elif change == "year":
        fold["year"] = "2025"
    elif change == "evaluation_cutoff":
        fold["evaluation_labels_before"] = "2026-01-01"
    elif change == "extra":
        fold["caller_authority"] = True
    elif change == "missing":
        del fold["training_labels_before"]
    elif change == "none":
        fold = None
    def forbidden(*a, **k):
        pytest.fail("unregistered fold reached dependent panel or outcome reads")
    for name in ("read_panel", "preflight_magnitude_targets", "qualify_magnitude_training_targets"):
        monkeypatch.setattr(binding, name, forbidden)
    before = case.conn.total_changes
    with pytest.raises(OpsError) as caught:
        binding.fold_inputs(case.repository, case.registration.document, fold)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
    assert case.conn.total_changes == before


def test_v17_large_registration_is_compact_and_missing_shard_refuses(case):
    from collections import Counter
    from engine.v2.foundation import ArtifactError, canonical_json
    from experiments.native_registration import NativeRegistration, read_native_registration
    from experiments.size_registration_artifacts import pack_size_registration

    logical = case.registration.document
    context = logical["size_panel"]["context"]
    context["confirmed_metadata"] = {
        "schema_version": "magnitude_confirmed_metadata.v1",
        "policy": {"mode": "synthetic"},
        "source": {"fragment_records": [{"fragment_id": "synthetic-fragment",
                                          "content_hash": "sha256:synthetic"}]},
        "tickers": ["SYN"],
        "sequences": [{
            "ticker": "SYN", "status": "complete", "selected_rows": 4200,
            "confirmed_rows": 4200, "first_confirmed_date": "2020-01-01",
            "events": [{"ordinal": index, "warmup_complete": True,
                        "source_note": "m" * 4096} for index in range(4200)],
        }],
        "counts": {"tickers": 1, "events": 4200},
        "index_id": "synthetic-large",
    }
    context["context_id"] = content_hash({key: value for key, value in context.items() if key != "context_id"})
    spec = prediction_spec(experiment_id="synthetic-large-size-registration",
                           folds=("2024",), runner="experiments/size_search.py")
    logical["spec_hash"] = spec.spec_hash
    key = content_hash({"experiment_id": spec.experiment_id, "arm": spec.primary_arm_id})
    run_id = "native_" + key.removeprefix("sha256:")
    compact, logical = pack_size_registration(case.store, logical)
    assert any(descriptor.get("kind") == "map" for descriptor in compact["size_artifacts"].values())
    assert any(descriptor.get("kind") == "list"
               and descriptor.get("entry", {}).get("row_count") == 4200
               for descriptor in compact["size_artifacts"].values())
    from engine.v2.foundation.canonical import iter_canonical_json
    old_bytes = sum(len(chunk.encode("utf-8")) for chunk in iter_canonical_json(logical))
    payload = canonical_json(compact).encode()
    assert old_bytes > 16 * 1024 * 1024
    assert 0 < len(payload) < 1024 * 1024
    assert len(payload) < 16 * 1024 * 1024
    registration = NativeRegistration(run_id, payload, logical)
    reserve_native_registration(case.conn, case.store, registration, spec)

    class CountingStore:
        def __init__(self):
            self.root = case.store.root
            self.reads = Counter()

        def read_verified(self, ref):
            self.reads[ref.storage_key] += 1
            return case.store.read_verified(ref)

    counting_store = CountingStore()
    replayed = read_native_registration(case.conn, counting_store, spec)
    assert replayed.document == logical
    assert replayed.binding_json == payload
    for descriptor in compact["size_artifacts"].values():
        for shard in descriptor["entry"]["shards"]:
            assert counting_store.reads[shard["ref"]["storage_key"]] == 1
    sealed = planner.seal_search_plan(case.conn, case.store, replayed, case.request)
    plan_ref = from_document(ArtifactRef, sealed["plan_ref"])
    assert planner.read_search_plan(case.conn, case.store, replayed, case.request, plan_ref) == sealed["plan"]
    print(f"v1.7 registration bytes: {len(payload)}")

    first = next(iter(compact["size_artifacts"].values()))["entry"]["shards"][0]
    missing_ref = first["ref"]

    class MissingShardStore:
        root = case.store.root

        def read_verified(self, ref):
            from engine.v2.contracts import ArtifactRef
            from engine.v2.foundation import from_document
            if ref == from_document(ArtifactRef, missing_ref):
                raise ArtifactError("INTEGRITY_FAILED", "synthetic missing registration shard")
            return case.store.read_verified(ref)

    with pytest.raises(OpsError) as caught:
        verify_native_registration(case.conn, MissingShardStore(), replayed)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"

    class AlteredShardStore:
        root = case.store.root

        def read_verified(self, ref):
            from engine.v2.contracts import ArtifactRef
            from engine.v2.foundation import from_document
            payload = case.store.read_verified(ref)
            if ref == from_document(ArtifactRef, missing_ref):
                return payload[:-1]
            return payload

    with pytest.raises(OpsError) as caught:
        verify_native_registration(case.conn, AlteredShardStore(), replayed)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_v17_packer_refuses_non_object_document_before_publish(case, monkeypatch):
    from experiments.size_registration_artifacts import pack_size_registration

    published = []
    publish_bytes = case.store.publish_bytes

    def record_publish(payload, *, schema_ref):
        published.append((payload, schema_ref))
        return publish_bytes(payload, schema_ref=schema_ref)

    monkeypatch.setattr(case.store, "publish_bytes", record_publish)
    with pytest.raises(OpsError) as caught:
        pack_size_registration(case.store, [])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not published


def test_v17_artifact_cap_precedes_publish_and_preflight_is_read_only(case, monkeypatch):
    from experiments import size_registration_artifacts as artifacts
    from experiments.magnitude_search_shards import write_section

    published = []
    publish_bytes = case.store.publish_bytes
    def record_publish(payload, *, schema_ref):
        published.append((payload, schema_ref))
        return publish_bytes(payload, schema_ref=schema_ref)
    monkeypatch.setattr(case.store, "publish_bytes", record_publish)

    cap = artifacts.MAX_SIZE_ARTIFACT_BYTES
    monkeypatch.setattr(artifacts, "MAX_SIZE_ARTIFACT_BYTES", 1)
    with pytest.raises(OpsError) as caught:
        binding._preview_payload(case.registration.document)
    assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not published

    monkeypatch.setattr(artifacts, "MAX_SIZE_ARTIFACT_BYTES", cap)
    shard_limit = artifacts.MAX_SHARDS
    monkeypatch.setattr(artifacts, "MAX_SHARDS", 0)
    with pytest.raises(OpsError) as caught:
        binding.prepare(case.repository, case.spec, case.request, case.panel_request,
                        publish_artifacts=True, **case.args)
    assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not published
    monkeypatch.setattr(artifacts, "MAX_SHARDS", shard_limit)
    prepared, _ = binding.prepare(case.repository, case.spec, case.request, case.panel_request,
                                  publish_artifacts=False, **case.args)
    assert prepared.binding_json == b""
    assert not published

    with pytest.raises(OpsError) as caught:
        write_section(case.store, "synthetic:oversized-row",
                      [{"position": 0, "row": "x" * (8 * 1024 * 1024)}],
                      lambda row: (row["position"],))
    assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert not published


def test_v16_inline_size_registration_still_round_trips(case):
    from engine.v2.foundation import canonical_json
    from experiments.native_registration import NativeRegistration, read_native_registration

    logical = case.registration.document
    logical["schema_version"] = "native_experiment_registration.v1.6"
    spec = prediction_spec(experiment_id="synthetic-legacy-v16-registration",
                           folds=("2024",), runner="experiments/size_search.py")
    logical["spec_hash"] = spec.spec_hash
    key = content_hash({"experiment_id": spec.experiment_id, "arm": spec.primary_arm_id})
    run_id = "native_" + key.removeprefix("sha256:")
    payload = canonical_json(logical).encode()
    registration = NativeRegistration(run_id, payload)
    reserve_native_registration(case.conn, case.store, registration, spec)
    replayed = read_native_registration(case.conn, case.store, spec)
    assert replayed.document == logical
    assert replayed.binding_json == payload
    verify_native_registration(case.conn, case.store, replayed)
    sealed = planner.seal_search_plan(case.conn, case.store, replayed, case.request)
    plan_ref = from_document(ArtifactRef, sealed["plan_ref"])
    assert planner.read_search_plan(case.conn, case.store, replayed, case.request, plan_ref) == sealed["plan"]


def test_interrupted_v3_start_refuses_replay_before_native_read(case, monkeypatch):
    """Interrupt a v3 start in a fresh journal namespace so replay refuses before native reads."""
    from engine.v2.foundation import canonical_json
    from experiments.magnitude_search_plan import trial_key
    from experiments.magnitude_search_trials import SearchTrials
    from experiments.native_registration import NativeRegistration

    assert case.plan["schema_version"] == "magnitude_search_plan.v3"
    # The module fixture's journal already owns other claims; a fresh registration
    # run isolates this test's start/claim without deleting or mutating any row.
    fresh_spec = prediction_spec(experiment_id="synthetic-interrupted-v3-start",
                                 folds=("2024",), runner="experiments/size_search.py")
    document = case.registration.document
    document["schema_version"] = "native_experiment_registration.v1.6"
    document["spec_hash"] = fresh_spec.spec_hash
    registration_key = content_hash({"experiment_id": fresh_spec.experiment_id,
                                     "arm": fresh_spec.primary_arm_id})
    registration = NativeRegistration("native_" + registration_key.removeprefix("sha256:"),
                                      canonical_json(document).encode())
    reserve_native_registration(case.conn, case.store, registration, fresh_spec)
    recorder = SearchTrials(case.conn, case.store, case.repository, registration, case.request, case.ref)
    monkeypatch.setattr(recorder, "_admit", lambda: case.plan)
    outer = case.plan["folds"][0]
    inner = outer["inner"][0]
    candidate_id = case.plan["candidates"][0]["candidate_id"]
    args = (outer["outer"]["year"], inner["year"], candidate_id)
    admissions = []

    def interrupt(fold):
        admissions.append(fold["year"])
        raise RuntimeError("interrupted native admission")

    monkeypatch.setattr(recorder, "_inputs", interrupt)
    with pytest.raises(RuntimeError, match="interrupted native admission"):
        recorder.start(*args)
    key = trial_key(case.plan, *args)
    assert recorder._read("start:" + key) is not None
    assert recorder._read("inputs:" + key) is None

    with pytest.raises(OpsError) as error:
        recorder.start(*args)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert admissions == [inner["year"]]
    assert recorder._read("inputs:" + key) is None


def test_report_reuses_shared_execution_receipt_read_and_keeps_outer_rows(monkeypatch):
    from collections import Counter
    import weakref

    from experiments import size_search_report as report
    from experiments.magnitude_search_plan import PLAN_V3

    limits = {"child": {"memory_bytes": 2147483648, "cpu_seconds": 30, "wall_seconds": 60}}
    plan = {"schema_version": PLAN_V3, "plan_id": "plan", "candidates": [{"candidate_id": "c"}],
        "counts": {"candidates": 1, "inner_folds": 3, "outer_folds": 2},
        "request": {"limits": limits},
        "horizons": [5, 3, 1],
        "folds": [
            {"outer": {"year": "2024", "evaluation_event_ids": ["evt"]},
             "inner": [{"year": "2023"}]},
            {"outer": {"year": "2025", "evaluation_event_ids": ["evt"]},
             "inner": [{"year": "2023"}, {"year": "2024"}]},
        ]}
    result = {"predictions": [{"event_id": "evt", "absolute_move_pct": 1.0}],
        "folds": [{"outer_year": year, "evaluation_inputs": {"omitted_labels": []}}
                  for year in ("2024", "2025")]}
    registration = SimpleNamespace(run_id="run", document={"size_panel": {"context": {
        "target_events": [{"event_id": "evt"}], "eligibility_plan": {}}}})
    monkeypatch.setattr(report, "_admit_driver", lambda *args: plan)
    monkeypatch.setattr(report, "collect_outer_results", lambda *args: result)
    monkeypatch.setattr(report, "paired_report", lambda *args, **kwargs: {})
    reads = Counter()
    payload_refs = []
    alive_before_read = []
    terminal_payload_refs = []
    alive_before_terminal_read = []

    class ReceiptPayload:
        def __init__(self):
            self.training_rows = ["event-id"] * 10000

    def read_record(conn, store, run_id, slot):
        reads[slot] += 1
        if slot.startswith("execution:"):
            alive_before_read.append(sum(ref() is not None for ref in payload_refs))
            payload = ReceiptPayload()
            payload_refs.append(weakref.ref(payload))
            return {"status": "succeeded", "fit_attempt_count": 1,
                    "completed_fit_count": 1, "diagnostics": [], "fits": payload,
                    "training_event_ids": payload, "evaluation_event_ids": payload}
        if slot.startswith("outer-forecasts:"):
            return []
        if slot.startswith("comparison-terminal:"):
            return {"result": {"forecasts": [], "fit_details": []}}
        if slot.startswith("terminal:"):
            alive_before_terminal_read.append(sum(ref() is not None for ref in terminal_payload_refs))
            payload = ReceiptPayload()
            terminal_payload_refs.append(weakref.ref(payload))
            return {"status": "succeeded", "score_seconds": 0.25, "rss_delta_kib": 64,
                    "inner_metrics": [], "predictions": payload, "evaluation_receipt": payload}
        if slot.startswith("fit-completed:"):
            return {"fit_seconds": 0.125}
        pytest.fail("unexpected journal slot " + slot)

    monkeypatch.setattr(report, "read_record", read_record)
    output = report.collect(None, None, None, registration, {}, None)
    from experiments.magnitude_search_plan import trial_key
    shared_key = trial_key(plan, "2024", "2023", "c")
    assert reads["execution:" + shared_key] == 1
    assert len(output["inner_fits"]) == 3
    assert all(row["fit_seconds"] == [
        {"horizon_sessions": 5, "seconds": 0.125},
        {"horizon_sessions": 3, "seconds": 0.125},
        {"horizon_sessions": 1, "seconds": 0.125},
    ] for row in output["inner_fits"])
    assert all(row["score_seconds"] == 0.25 and row["rss_delta_kib"] == 64
               and row["inner_metrics"] == [] for row in output["inner_fits"])
    assert [row["outer_year"] for row in output["inner_fits"]] == ["2024", "2025", "2025"]
    assert alive_before_read == [0, 0]
    assert all(ref() is None for ref in payload_refs)
    assert alive_before_terminal_read == [0, 0]
    assert all(ref() is None for ref in terminal_payload_refs)
