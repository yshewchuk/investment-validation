"""Distribution-owned planning/admission call chain; synthetic public seam doubles."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.foundation import artifact_reference, canonical_json, content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import distribution_plan as planner
from experiments import distribution_trials as journal
from experiments.distribution_candidate_recipe import candidate_recipe
from experiments.native_registration import NativeRegistration


def _registration(run, document):
    return NativeRegistration(run, canonical_json(document).encode())


@pytest.fixture
def context(monkeypatch):
    runner = "experiments/distribution_trials.py"
    environment = {"test_environment": 1}
    sources = {runner: content_hash("runner")}
    execution = {"runner": runner, "seed": 0, "price_source": "computed_moves",
                 "economic_params": {}, "folds": ["2024"]}
    common = {"snapshot": {"snapshot_id": "frozen"}, "scope": "selection", "event_ids": ["train", "inner", "outer"],
              "holdouts": {"holdout_as_of_month": "2025-01", "random_membership_version": "r1"},
              "environment": environment, "source_closure": sources}
    own = _registration("own", {**common, "schema_version": "native_experiment_registration.v1.0",
                                "execution_plan": execution})
    source = _registration("source", {**common, "schema_version": "native_experiment_registration.v1.7",
        "execution_plan": {**execution, "runner": "experiments/size_search.py"},
        "magnitude_execution": {"evaluation_labels_before": "2025-01-01"},
        "size_panel": {"context": {"context_id": content_hash("qualified-source-context")}}})
    inner = {"year": "2023", "training_labels_before": "2023-01-01", "evaluation_labels_before": "2024-01-01",
             "training_candidate_event_ids": ["train"], "evaluation_event_ids": ["inner"]}
    outer = {"year": "2024", "training_labels_before": "2024-01-01",
             "training_candidate_event_ids": ["train", "inner"], "evaluation_event_ids": ["outer"]}
    native = {"plan_id": content_hash("source-plan"), "candidate_count": 33,
              "planned_fit_count": 102, "folds": [{"outer": outer, "inner": [inner]}]}
    ref = artifact_reference(canonical_json(native).encode(), "magnitude_search_plan.v2")
    events, records = [], {}
    monkeypatch.setattr(planner, "verify_native_registration", lambda *args, **kw: events.append(("verify", args[2].run_id, kw)))

    def read_plan(*args):
        assert args[2:] == (source, {"source": "request"}, ref)
        events.append(("read_source_plan", "source"))
        return deepcopy(native)

    monkeypatch.setattr(planner, "read_search_plan", read_plan)
    monkeypatch.setattr(planner, "source_closure", lambda *args: deepcopy(sources))
    monkeypatch.setattr(planner, "environment_identity", lambda seed: deepcopy(environment))
    caps = {"max_fits": 12, "max_trees": 49200,
            "child_limits": {"memory_bytes": 1024, "cpu_seconds": 3, "wall_seconds": 5}}
    args = (object(), object(), own, source, {"source": "request"}, ref, candidate_recipe("H"), caps)

    def append(conn, store, run, slot, body, *, replay=True):
        events.append(("append", run, slot, replay))
        key = (run, slot)
        if key in records and (not replay or records[key] != body):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "slot already claimed")
        records[key] = deepcopy(body)
        return deepcopy(body)

    monkeypatch.setattr(journal, "append_record", append)
    monkeypatch.setattr(journal, "read_record", lambda conn, store, run, slot: deepcopy(records.get((run, slot))))
    monkeypatch.setattr(journal, "migrate", lambda *args, **kwargs: events.append(("migrate", kwargs["clock"])))
    monkeypatch.setattr(journal, "require_current", lambda *args: events.append(("current",)))
    columns = args[-2]["point_recipe"]["features"]
    inputs = {"features": [{"event_id": event, "context_id": "panel-context", "horizon_sessions": horizon,
              "unselected": 99, **dict.fromkeys(columns, 1.0)}
              for event in ("train", "inner", "outer") for horizon in (5, 3, 1)],
              "training": [{"event_id": "train", "absolute_move_pct": 1.0}]}

    def fold_inputs(repository, binding, fold):
        events.append(("fold_inputs", deepcopy(fold)))
        assert binding == source.document
        return deepcopy(inputs)

    published = []
    manifest = {"schema_version": "test_manifest", "sections": {"features": "shards"}}
    inputs_id = content_hash(manifest)

    def publish(store, value):
        events.append(("publish_inputs",))
        published.append(deepcopy(value))
        return deepcopy(manifest), inputs_id

    def load(*values):
        events.append(("load_fit_inputs", deepcopy(values[1:])))
        if values[1:3] != (manifest, inputs_id):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "changed manifest")
        return deepcopy(inputs)

    monkeypatch.setattr(journal, "fold_inputs", fold_inputs)
    monkeypatch.setattr(journal, "publish_inputs", publish)
    monkeypatch.setattr(journal, "load_fit_inputs", load)
    def premature_resolve(*args):
        pytest.fail("repository validation must not read snapshot or outcome data")
    repository = SimpleNamespace(resolve=premature_resolve)
    trials = journal.DistributionTrials(args[0], args[1], repository, *args[2:])
    return SimpleNamespace(args=args, trials=trials, events=events, records=records, native=native,
        columns=columns, published=published, inputs=inputs, manifest=manifest, inputs_id=inputs_id)


def _code(code, action):
    with pytest.raises(OpsError) as caught:
        action()
    assert caught.value.code == code
    return caught.value


def test_small_plan_uses_native_schedule_not_size_grid_and_detaches(context):
    plan, native = planner.prepare_plan(*context.args)
    assert context.events == [("verify", "own", {"max_evidence_bytes": planner.MAX_NATIVE_EVIDENCE_BYTES}), ("read_source_plan", "source")]
    assert native == context.native
    assert plan["budget"]["counts"] == {"classifier_fits": 6, "point_fits": 6, "total_fits": 12,
        "classifier_trees_upper": 48600, "point_trees_upper": 600, "total_trees_upper": 49200}
    assert plan["registrations"]["own"]["run_id"] == "own"
    assert plan["registrations"]["source"]["registration_ref"]["schema_ref"] == "native_experiment_registration.v1.7"
    assert "event_ids" not in canonical_json(plan) and "folds" not in plan
    assert plan["plan_id"] == content_hash({key: value for key, value in plan.items() if key != "plan_id"})
    plan["recipe"]["horizons"].clear()
    native["folds"].clear()
    assert planner.prepare_plan(*context.args)[0]["recipe"]["horizons"] == [5, 3, 1]


@pytest.mark.parametrize("field,value", [("schema_version", "native_experiment_registration.v1.3"),
    ("snapshot", {}), ("scope", "other"), ("holdouts", {}), ("event_ids", ["different"]),
    ("environment", {}), ("source_closure", {}), ("final_read", {})])
def test_owner_context_drift_refuses(context, field, value):
    args = list(context.args)
    args[2] = _registration("own", {**args[2].document, field: value})
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: planner.prepare_plan(*args))
    assert not context.published


@pytest.mark.parametrize("field,value", [("runner", "experiments/size_search.py"), ("seed", True),
    ("seed", 1), ("price_source", "other"), ("economic_params", {"cost": 1}), ("folds", ["2023"])])
def test_owner_execution_must_be_distinct_non_size_metadata(context, field, value):
    args = list(context.args)
    document = args[2].document
    document["execution_plan"][field] = value
    args[2] = _registration("own", document)
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: planner.prepare_plan(*args))


def test_same_run_and_source_version_refuse(context, monkeypatch):
    args = list(context.args)
    args[2] = _registration("source", args[2].document)
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: planner.prepare_plan(*args))
    args = list(context.args)
    args[3] = _registration("source", {**args[3].document, "schema_version": "native_experiment_registration.v1.6"})
    monkeypatch.setattr(planner, "read_search_plan", lambda *unused: context.native)
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: planner.prepare_plan(*args))


@pytest.mark.parametrize("caps", [None, {}, {"extra": 1}])
def test_bad_caps_refuse_before_registration_access(context, caps):
    _code("INVALID_EXPERIMENT_SPEC", lambda: planner.prepare_plan(*context.args[:-1], caps))
    assert context.events == []


def test_insufficient_actual_pair_cap_preserves_resource_code(context):
    caps = {**context.args[-1], "max_trees": 49199}
    _code("RESOURCE_LIMIT_EXCEEDED", lambda: planner.prepare_plan(*context.args[:-1], caps))
    assert not context.published


@pytest.mark.parametrize("role,year,inner", [("inner", "2024", "2023"), ("outer", "2024", None)])
def test_claim_precedes_native_admission_then_project_and_bind_actual_manifest(context, role, year, inner):
    c = context
    plan = c.trials.seal(clock="clock")
    receipt = c.trials.start(role, year, inner)
    names = [event[0] for event in c.events]
    starts = [index for index, event in enumerate(c.events) if event[0] == "append" and ":start:" in event[2]]
    assert len(starts) == 1 and starts[0] < names.index("fold_inputs") < names.index("publish_inputs")
    assert c.events[starts[0]][-1] is False
    fold = next(event[1] for event in c.events if event[0] == "fold_inputs")
    assert fold["evaluation_labels_before"] == ("2024-01-01" if role == "inner" else "2025-01-01")
    assert receipt["fold_id"] == content_hash({"fold": fold, "role": role, "outer_year": year})
    assert all(set(row) == {"event_id", "context_id", "horizon_sessions", *c.columns}
               for row in c.published[0]["features"])
    assert "unselected" in c.inputs["features"][0]
    assert {row["horizon_sessions"] for row in c.published[0]["features"]} == {1, 3, 5}
    assert [trial["horizon"] for trial in receipt["trials"]] == [5, 3, 1]
    assert all(trial["inputs_id"] == c.inputs_id and trial["context_id"] == plan["context_id"]
               for trial in receipt["trials"])
    assert receipt["manifest"] == c.manifest and receipt["status"] == "admitted"
    assert receipt["final_read_authority"] == "none"
    assert {run for run, slot in c.records} == {"own"}
    assert all(slot.startswith("distribution:") for run, slot in c.records)
    receipt["manifest"]["sections"].clear()
    assert c.trials.read_admission(role, year, inner)["manifest"] == c.manifest


def test_replay_revalidates_no_outcome_reread_or_write(context, monkeypatch):
    c = context
    c.trials.seal(clock=1)
    receipt = c.trials.start("inner", "2024", "2023")
    monkeypatch.setattr(journal, "fold_inputs", lambda *args: pytest.fail("reopened native outcomes"))
    c.events.clear()
    assert c.trials.read_admission("inner", "2024", "2023") == receipt
    assert [event[0] for event in c.events] == ["current", "verify", "read_source_plan", "load_fit_inputs"]
    assert c.events[-1][1][2:] == (["inner"], [5, 3, 1], c.columns)
    assert set(receipt).isdisjoint({"features", "training", "forecasts", "completed"})


@pytest.mark.parametrize("role,year,inner", [("bad", "2024", None), (True, "2024", None),
    ("inner", 2024, "2023"), ("inner", "2024", None), ("outer", "2024", "2023"),
    ("inner", "2024", "1900"), ("outer", "2025", None)])
def test_unknown_or_malformed_selection_has_no_claim(context, role, year, inner):
    context.trials.seal(clock=1)
    _code("INVALID_EXPERIMENT_SPEC", lambda: context.trials.start(role, year, inner))
    assert len(context.records) == 1 and not context.published


def test_unsealed_and_changed_seal_refuse_before_claim(context):
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: context.trials.start("inner", "2024", "2023"))
    context.trials.seal(clock=1)
    context.records["own", "distribution:seal"]["recipe"]["horizons"] = [1]
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: context.trials.start("inner", "2024", "2023"))
    assert not context.published


@pytest.mark.parametrize("repository", [None, object(), SimpleNamespace(resolve=None), SimpleNamespace(resolve=1)])
def test_invalid_repository_refuses_before_claim_and_can_be_corrected(context, repository):
    context.trials.seal(clock=1)
    valid = context.trials.repository
    context.trials.repository = repository
    error = _code("INVALID_EXPERIMENT_SPEC", lambda: context.trials.start("inner", "2024", "2023"))
    assert error.problem.retryable is False
    assert error.problem.message == "distribution admission requires a repository snapshot resolver"
    assert error.problem.details == {}
    assert set(context.records) == {("own", "distribution:seal")}
    assert not context.published and not any(event[0] == "fold_inputs" for event in context.events)
    context.trials.repository = valid
    assert context.trials.start("inner", "2024", "2023")["status"] == "admitted"


@pytest.mark.parametrize("seam", ["fold_inputs", "publish_inputs", "append_record"])
@pytest.mark.parametrize("error", [
    fail("RESOURCE_UNAVAILABLE", "synthetic private error detail", details={"secret": "not exposed"}),
    data_fail("INPUT_CHANGED", "synthetic private data detail", details={"secret": "not exposed"}),
    data_fail("CONTRACT_MISMATCH", "native data contract refusal"),
    fail("RESOURCE_LIMIT_EXCEEDED", "native cap refusal"),
    fail("INVALID_EXPERIMENT_SPEC", "native specification refusal"),
    OSError("native I/O"), KeyError("native column"),
])
def test_interrupted_claim_never_retries_and_replay_refuses(context, monkeypatch, seam, error):
    context.trials.seal(clock=1)
    calls = []
    original = getattr(journal, seam)

    def explode(*args, **kwargs):
        if seam == "append_record" and not args[3].startswith("distribution:admission:"):
            return original(*args, **kwargs)
        calls.append(1)
        raise error

    monkeypatch.setattr(journal, seam, explode)
    retryable = isinstance(error, (OpsError, DataError)) and error.problem.retryable
    with pytest.raises(OpsError if retryable else type(error)) as caught:
        context.trials.start("inner", "2024", "2023")
    if retryable:
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert caught.value.problem.retryable is False
        assert caught.value.problem.message == "distribution admission interrupted after its single-use claim"
        assert caught.value.problem.details == {}
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
    else:
        assert caught.value is error
    assert {slot.split(":")[1] for run, slot in context.records} == {"seal", "start"}
    assert len(context.published) == (1 if seam == "append_record" else 0)
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: context.trials.start("inner", "2024", "2023"))
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: context.trials.read_admission("inner", "2024", "2023"))
    assert calls == [1]


@pytest.mark.parametrize("seam", ["prepare_plan", "append_record"])
@pytest.mark.parametrize("error", [fail("RESOURCE_UNAVAILABLE", "preclaim resource interruption"),
    data_fail("INPUT_CHANGED", "preclaim data interruption")])
def test_preclaim_transient_stays_retryable_and_can_start_after_recovery(context, monkeypatch, seam, error):
    context.trials.seal(clock=1)

    def explode(*args, **kwargs):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(journal, seam, explode)
        with pytest.raises(type(error)) as caught:
            context.trials.start("inner", "2024", "2023")
        assert caught.value is error
    assert error.problem.retryable is True
    assert set(context.records) == {("own", "distribution:seal")}
    assert not context.published and not any(event[0] == "fold_inputs" for event in context.events)
    admitted = context.trials.start("inner", "2024", "2023")
    assert context.trials.read_admission("inner", "2024", "2023") == admitted


@pytest.mark.parametrize("member", ["plan_id", "inputs_id", "manifest", "trials", "status", "extra"])
def test_replay_refuses_altered_durable_admission(context, member):
    context.trials.seal(clock=1)
    receipt = context.trials.start("inner", "2024", "2023")
    record = context.records["own", "distribution:admission:" + receipt["fold_id"]]
    record[member] = "changed"
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: context.trials.read_admission("inner", "2024", "2023"))


@pytest.mark.parametrize("seam", ["verify_native_registration", "read_search_plan", "source_closure", "environment_identity"])
def test_planning_native_errors_preserve_identity(context, monkeypatch, seam):
    error = fail("RESOURCE_LIMIT_EXCEEDED", "native resource failure")

    def explode(*args, **kwargs):
        raise error

    monkeypatch.setattr(planner, seam, explode)
    assert _code(error.code, lambda: planner.prepare_plan(*context.args)) is error


@pytest.mark.parametrize("error", [fail("RESOURCE_UNAVAILABLE", "native store unavailable"),
    data_fail("INPUT_CHANGED", "native data unavailable")])
def test_replay_loader_resource_failure_is_not_identity_conflict(context, monkeypatch, error):
    context.trials.seal(clock=1)
    context.trials.start("inner", "2024", "2023")

    def explode(*args, **kwargs):
        raise error

    monkeypatch.setattr(journal, "load_fit_inputs", explode)
    with pytest.raises(type(error)) as caught:
        context.trials.read_admission("inner", "2024", "2023")
    assert caught.value is error
    assert error.problem.retryable is True


def test_constructor_detaches_declared_recipe_request_and_caps(context):
    context.args[4].clear()
    context.args[-2]["horizons"].clear()
    context.args[-1]["child_limits"].clear()
    assert context.trials.seal(clock=1)["recipe"]["horizons"] == [5, 3, 1]


def test_public_exports_are_small():
    assert planner.__all__ == ["prepare_plan"]
    assert journal.__all__ == ["DistributionTrials"]


@pytest.mark.parametrize("index", [2, 3])
def test_oversized_registration_refuses_before_decode_or_catalog(context, monkeypatch, index):
    monkeypatch.setattr(planner, "MAX_NATIVE_EVIDENCE_BYTES", 2048)
    args = list(context.args)
    args[index] = NativeRegistration(args[index].run_id, b"not JSON" * 257)
    _code("RESOURCE_LIMIT_EXCEEDED", lambda: planner.prepare_plan(*args))
    assert context.events == []


@pytest.mark.parametrize("error", [MemoryError(), KeyError("native column"), OSError("native I/O")])
def test_native_replay_loader_untyped_errors_propagate(context, monkeypatch, error):
    context.trials.seal(clock=1)
    context.trials.start("inner", "2024", "2023")

    def explode(*args):
        raise error

    monkeypatch.setattr(journal, "load_fit_inputs", explode)
    with pytest.raises(type(error)) as caught:
        context.trials.read_admission("inner", "2024", "2023")
    assert caught.value is error


def test_second_complete_start_is_not_replay_or_new_native_read(context):
    context.trials.seal(clock=1)
    context.trials.start("inner", "2024", "2023")
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: context.trials.start("inner", "2024", "2023"))
    assert sum(event[0] == "fold_inputs" for event in context.events) == 1


@pytest.mark.parametrize("change", ["remove", "alter"])
def test_replay_requires_exact_start_before_loading_cas(context, change):
    context.trials.seal(clock=1)
    receipt = context.trials.start("inner", "2024", "2023")
    key = "own", "distribution:start:" + receipt["fold_id"]
    if change == "remove":
        del context.records[key]
    else:
        context.records[key]["role"] = "outer"
    _code("EXPERIMENT_IDENTITY_CONFLICT", lambda: context.trials.read_admission("inner", "2024", "2023"))
    assert not any(event[0] == "load_fit_inputs" for event in context.events)


def test_alternative_existing_owner_runner_is_metadata_only(context, monkeypatch):
    args = list(context.args)
    document = args[2].document
    runner = "experiments/real_distribution_runner.py"
    document["execution_plan"]["runner"] = runner
    document["source_closure"] = {runner: content_hash("real runner")}
    args[2] = _registration("own", document)
    monkeypatch.setattr(planner, "source_closure", lambda *unused: document["source_closure"].copy())
    assert planner.prepare_plan(*args)[0]["registrations"]["own"]["run_id"] == "own"


def test_projection_progress_is_bounded_native_primitive(context, monkeypatch):
    calls = []

    class Progress:
        def __init__(self, name, total):
            calls.append((name, total))

        def tick(self):
            calls.append("tick")

    monkeypatch.setattr(journal, "Progress", Progress)
    context.trials.seal(clock=1)
    context.trials.start("inner", "2024", "2023")
    assert calls == [("distribution feature projection", 9)] + ["tick"] * 9
