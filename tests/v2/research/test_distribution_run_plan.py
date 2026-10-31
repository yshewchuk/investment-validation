"""Standalone metadata/seal/section contract; no pipeline or runner imports."""
import os
import subprocess
import sys
from collections import UserDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.models.no_fit import no_fit_guard as legacy_no_fit
from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import (
    ArtifactStore,
    SystemClock,
    canonical_json,
    content_hash,
    to_document,
)
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.executor import THREAD_VARIABLES
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.migrations import migrate
from experiments import distribution_run_plan as metadata
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record
from experiments.magnitude_search_shards import iter_section, write_section
from tests.v2.research.test_distribution_run_request import request


def _error(action, code="INVALID_EXPERIMENT_SPEC"):
    with pytest.raises(OpsError) as caught:
        action()
    assert caught.value.code == code
    return caught.value


@pytest.fixture
def plan_case(tmp_path, monkeypatch):
    conn, store, clock = connect(tmp_path / "catalog.sqlite"), ArtifactStore(tmp_path / "store"), SystemClock()
    conn.execute("CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO experiment_runs VALUES ('distribution-own')")
    migrate(conn, OWNER, MIGRATIONS, clock=clock)
    snapshot = SnapshotRef(snapshot_id="quotes", manifest_hash=content_hash("snapshot"), table_versions={},
        calendar_version="calendar", source_priority_version="priority", finality_receipt_refs=(), knowledge_mode_by_table={})
    events = [{"event_id": f"{year}-{ticker}", "ticker": ticker, "event_date": f"{year}-06-03",
        "forecasts": [{"horizon_sessions": horizon, "session": "AMC", "source_session_ceiling": f"{year}-06-02"}
                      for horizon in (5, 3, 1)]} for year in ("2023", "2024") for ticker in ("AAA", "BBB")]
    groups = [{"outer": {"year": year, "evaluation_event_ids": [event["event_id"] for event in events if event["event_date"].startswith(year)],
        "training_labels_before": year + "-01-01"}, "inner": [{"year": str(int(year) - 1)}]} for year in ("2023", "2024")]
    base = {"plan_id": content_hash("B-plan"), "budget": {"child_limits": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 30}}}
    binding = {"snapshot": to_document(snapshot), "holdouts": {"holdout_as_of_month": "2025-01"},
        "magnitude_execution": {"evaluation_labels_before": "2025-01-01"}, "size_panel": {"context": {"target_events": events}}}
    calls, state = [], {"worker_hash": content_hash("worker")}
    def slots():
        return [row[0] for row in conn.execute("SELECT slot FROM magnitude_search_records ORDER BY slot")]
    def seal(*, clock):
        calls.append(("B-seal",))
        return append_record(conn, store, "distribution-own", "distribution:seal", base)
    trials = SimpleNamespace(conn=conn, store=store, own=SimpleNamespace(run_id="distribution-own"),
        source=SimpleNamespace(document=binding), repository=SimpleNamespace(resolve=lambda value: snapshot),
        request={}, source_plan_ref=None, recipe={"horizons": [5, 3, 1]}, caps={}, seal=seal)
    monkeypatch.setattr(metadata, "prepare_plan", lambda *args: (deepcopy(base), {"folds": deepcopy(groups)}))
    monkeypatch.setattr(metadata, "source_closure", lambda root, entries: {entry: state["worker_hash"] for entry in entries})
    monkeypatch.setattr(metadata, "read_bridge", lambda *args: SimpleNamespace(intercept=1, slope=1, residuals=[0]))
    benchmarks = []
    for group in groups:
        fold = group["outer"]
        rows = [{"event_id": event, "incumbent_magnitudes": [1], "gate_selected": True} for event in sorted(fold["evaluation_event_ids"])]
        descriptor = write_section(store, "distribution_benchmark", rows, lambda row: (row["event_id"],))
        document = {"schema_version": "distribution_benchmark.v1", "outer_year": fold["year"],
            "training_labels_before": fold["training_labels_before"], "rows": descriptor,
            "forecast_horizon": 1, "decision_policy": "native_h1_source_session_ceiling"}
        benchmarks.append({"outer_year": fold["year"], "ref": to_document(store.publish_bytes(canonical_json(document).encode(), schema_ref=document["schema_version"]))})
    yield SimpleNamespace(trials=trials, request=request(), conn=conn, store=store, clock=clock, groups=groups,
        base=base, events=events, calls=calls, state=state, slots=slots, benchmarks=benchmarks)
    conn.close()


def make_plan(case, **changes):
    return metadata.DistributionRunPlan(case.trials, {**case.request, **changes})


def test_standalone_plan_import_does_not_require_future_pipeline_or_runner():
    script = """import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name not in ('experiments.distribution_pipeline', 'experiments.distribution_run'), name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments.distribution_run_plan import DistributionRunPlan
assert not hasattr(DistributionRunPlan, 'run')
assert 'experiments.distribution_pipeline' not in sys.modules
assert 'experiments.distribution_run' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **dict.fromkeys(THREAD_VARIABLES, "1")})
    assert result.returncode == 0, result.stderr


def test_real_source_closure_covers_metadata_numeric_worker_and_adapters():
    closure = source_closure(Path(__file__).resolve().parents[3], list(metadata.DistributionRunPlan.source_roots))
    required = {"experiments/distribution_run_plan.py", "experiments/distribution_worker.py",
        "experiments/distribution_plan.py", "experiments/distribution_candidate_fit.py",
        "experiments/distribution_fitted.py", "experiments/distribution_market.py",
        "experiments/distribution_run_request.py"}
    assert required <= closure.keys()
    assert not {"experiments/distribution_pipeline.py", "experiments/distribution_run.py"} & closure.keys()


def test_central_plan_refusal_propagates_without_d_writes(plan_case, monkeypatch):
    error = fail("EXPERIMENT_IDENTITY_CONFLICT", "synthetic native source refusal")
    def refused(*args):
        raise error
    monkeypatch.setattr(metadata, "prepare_plan", refused)
    assert _error(lambda: make_plan(plan_case).preflight(), error.code) is error
    assert plan_case.calls == [] and plan_case.slots() == []


def test_plan_preflight_is_readonly_under_both_fit_guards(plan_case):
    with no_fit_guard(), legacy_no_fit():
        plan = make_plan(plan_case).preflight()
    assert plan["opportunities"] == 4 and plan["final_read_authority"] == "none"
    assert set(plan["source_closure"]) == set(metadata.DistributionRunPlan.source_roots)
    assert plan["available_benchmarks"] == {"incumbent": False, "gate": False}
    assert plan_case.calls == [] and plan_case.slots() == []


def test_preflight_result_policy_is_detached_from_owned_state(plan_case):
    run = make_plan(plan_case)
    returned = run.preflight()
    expected = deepcopy(returned)
    returned["request"]["minimum_net_ev_usd"] = -1
    returned["request"]["bridges"] = None
    returned["request"]["economics"]["commission"] = -1
    assert run.request == plan_case.request
    assert run.preflight() == expected
    assert plan_case.calls == [] and plan_case.slots() == []


@pytest.mark.parametrize("field,value", [("minimum_net_ev_usd", -1), ("bridges", None)])
def test_preflight_revalidates_owned_policy_before_metadata_or_artifact_access(plan_case, monkeypatch, field, value):
    run = make_plan(plan_case)
    run.request[field] = value
    before = plan_case.conn.total_changes
    def forbidden(*args, **kwargs):
        pytest.fail("invalid policy reached metadata or artifact access")
    monkeypatch.setattr(run, "plans", forbidden)
    monkeypatch.setattr(plan_case.store, "read_verified", forbidden)
    monkeypatch.setattr(plan_case.store, "publish_bytes", forbidden)
    error = _error(run.preflight)
    assert error.problem.retryable is False
    assert plan_case.conn.total_changes == before and plan_case.calls == [] and plan_case.slots() == []


def test_preflight_revalidation_preserves_valid_control_and_seal_identity(plan_case):
    baseline = make_plan(plan_case).preflight()
    run = make_plan(plan_case, minimum_net_ev_usd=2)
    expected_request = {**plan_case.request, "minimum_net_ev_usd": 2}
    plan = run.preflight()
    assert plan["request"] == run.request == expected_request
    assert plan["run_plan_id"] != baseline["run_plan_id"]
    assert run.seal(clock=plan_case.clock) == plan
    assert run.admit() == plan


@pytest.mark.parametrize("changes", [{"max_opportunities": 3}, {"bridges": []}, {"strict_benchmarks": True}])
def test_plan_refuses_unfunded_schedule_or_missing_required_inputs_before_seal(plan_case, changes):
    _error(lambda: make_plan(plan_case, **changes).preflight())
    assert plan_case.calls == [] and plan_case.slots() == []


def test_plan_preflight_verifies_complete_optional_benchmark_cohort(plan_case):
    result = make_plan(plan_case, benchmarks=plan_case.benchmarks, strict_benchmarks=True).preflight()
    assert result["available_benchmarks"] == {"incumbent": True, "gate": True}
    _error(lambda: make_plan(plan_case, benchmarks=plan_case.benchmarks[:1], strict_benchmarks=True).preflight())


def test_plan_seal_replays_exact_identity_but_refuses_policy_or_source_drift(plan_case):
    run = make_plan(plan_case)
    plan = run.seal(clock=plan_case.clock)
    assert run.admit() == plan
    _error(lambda: make_plan(plan_case, minimum_net_ev_usd=2).admit())
    plan_case.state["worker_hash"] = content_hash("changed")
    _error(run.admit)
    assert not any("started" in slot for slot in plan_case.slots())


def test_distinct_entrypoint_source_roots_cannot_reuse_metadata_seal(plan_case):
    run = make_plan(plan_case)
    run.seal(clock=plan_case.clock)
    class OtherEntrypoint(metadata.DistributionRunPlan):
        source_roots = ("experiments/distribution_worker.py",)
    _error(OtherEntrypoint(plan_case.trials, plan_case.request).admit)


def test_plan_sections_are_real_cas_verified_and_collect_never_fits(plan_case, monkeypatch):
    run = make_plan(plan_case)
    plan = run.seal(clock=plan_case.clock)
    section = run.publish("fixture", ({"value": number} for number in (3, 1)))
    rows = list(iter_section(plan_case.store, "fixture", section["descriptor"], lambda row: (row["position"],)))
    assert rows == [{"position": 0, "value": 3}, {"position": 1, "value": 1}]
    result = {"run_plan_id": plan["run_plan_id"], "sections": [section]}
    run.record("completed", result, replay=False)
    changes = plan_case.conn.total_changes
    with no_fit_guard(), legacy_no_fit():
        assert run.collect() == result
    assert plan_case.conn.total_changes == changes
    bad = section["descriptor"]["shards"][0]["ref"]["content_hash"]
    original = plan_case.store.read_verified
    failure = fail("RESOURCE_UNAVAILABLE", "synthetic missing report shard")
    def read(ref):
        if ref.content_hash == bad:
            raise failure
        return original(ref)
    monkeypatch.setattr(plan_case.store, "read_verified", read)
    assert _error(run.collect, failure.code) is failure


def test_plan_collection_without_completion_has_no_execution_fallback(plan_case):
    run = make_plan(plan_case)
    run.seal(clock=plan_case.clock)
    _error(run.collect)
    assert plan_case.calls == [("B-seal",)]


@pytest.mark.parametrize("slot", [None, [], {}, 0, ""])
@pytest.mark.parametrize("body", [None, {"value": 1}])
def test_journal_refuses_malformed_slots_before_storage(plan_case, slot, body):
    before = plan_case.conn.total_changes
    error = _error(lambda: make_plan(plan_case).record(slot, body))
    assert error.problem.retryable is False
    assert plan_case.conn.total_changes == before and plan_case.slots() == []


@pytest.mark.parametrize("body", [[], 1, "body"])
def test_journal_refuses_nonmapping_bodies_before_storage(plan_case, body):
    before = plan_case.conn.total_changes
    _error(lambda: make_plan(plan_case).record("fixture", body))
    assert plan_case.conn.total_changes == before and plan_case.slots() == []


@pytest.mark.parametrize("body", [{}, {"sections": []}, {"run_plan_id": "ID"},
    {"run_plan_id": "ID", "sections": None}, {"run_plan_id": "ID", "sections": {}},
    {"run_plan_id": "ID", "sections": [None]}, {"run_plan_id": "ID", "sections": [{}]},
    {"run_plan_id": "ID", "sections": [{"name": "fixture"}]},
    {"run_plan_id": "ID", "sections": [{"name": None, "descriptor": {}}]},
    {"run_plan_id": "ID", "sections": [{"name": "", "descriptor": {}}]},
    {"run_plan_id": "ID", "sections": [{"name": "fixture", "descriptor": None}]},
    {"run_plan_id": "ID", "sections": [{"name": "fixture", "descriptor": {}}]}])
@pytest.mark.parametrize("stored", [False, True])
def test_completion_envelope_is_checked_before_write_and_on_collect(plan_case, body, stored):
    run = make_plan(plan_case)
    plan = run.seal(clock=plan_case.clock)
    body = deepcopy(body)
    if body.get("run_plan_id") == "ID":
        body["run_plan_id"] = plan["run_plan_id"]
    if stored:
        append_record(plan_case.conn, plan_case.store, plan_case.trials.own.run_id,
            "distribution:run-completed", body)
    before = plan_case.conn.total_changes
    error = _error(run.collect if stored else lambda: run.record("completed", body))
    assert error.problem.retryable is False and plan_case.conn.total_changes == before


@pytest.mark.parametrize("defect", ["empty", "extra", "count_type", "count_mismatch", "empty_shards",
    "shard_shape", "shard_count", "first_key", "ref_schema", "ref_path", "ref_size"])
def test_completion_descriptor_refusal_preserves_slot_for_correction(plan_case, monkeypatch, defect):
    run = make_plan(plan_case)
    plan = run.seal(clock=plan_case.clock)
    section = run.publish("fixture", [{"value": 3}])
    malformed = deepcopy(section["descriptor"])
    if defect == "empty":
        malformed = {}
    elif defect == "extra":
        malformed["extra"] = 1
    elif defect == "count_type":
        malformed["row_count"] = True
    elif defect == "count_mismatch":
        malformed["row_count"] = 2
    elif defect == "empty_shards":
        malformed["shards"] = []
    elif defect == "shard_shape":
        malformed["shards"] = [{}]
    elif defect == "shard_count":
        malformed["shards"][0]["rows"] = 0
    elif defect == "first_key":
        malformed["shards"][0]["first_key"] = [None]
    elif defect == "ref_schema":
        malformed["shards"][0]["ref"]["schema_ref"] = "wrong.v1"
    elif defect == "ref_path":
        malformed["shards"][0]["ref"]["storage_key"] = "../outside"
    else:
        malformed["shards"][0]["ref"]["byte_size"] = 0
    # A valid earlier section must not permit committing a malformed later one.
    result = {"run_plan_id": plan["run_plan_id"], "sections": [section,
        {"name": "later", "descriptor": malformed}]}
    before = plan_case.conn.total_changes
    def forbidden(*args, **kwargs):
        pytest.fail("malformed descriptor reached artifact I/O before refusal")
    with monkeypatch.context() as patch:
        patch.setattr(plan_case.store, "read_verified", forbidden)
        patch.setattr(plan_case.store, "publish_bytes", forbidden)
        error = _error(lambda: run.record("completed", result))
        assert error.problem.retryable is False
    assert plan_case.conn.total_changes == before and run.record("completed") is None
    corrected = {"run_plan_id": plan["run_plan_id"], "sections": [section]}
    assert run.record("completed", corrected) == corrected
    assert run.collect() == corrected


def test_completion_descriptor_validation_keeps_empty_sections_collectible(plan_case):
    run = make_plan(plan_case)
    plan = run.seal(clock=plan_case.clock)
    empty = run.publish("empty", [])
    valid = {"run_plan_id": plan["run_plan_id"], "sections": [empty]}
    assert run.record("completed", valid) == valid
    assert run.collect() == valid


@pytest.mark.parametrize("name", [None, [], {}, 0, ""])
def test_section_refuses_malformed_name(plan_case, name):
    _error(lambda: make_plan(plan_case).publish(name, []))


@pytest.mark.parametrize("rows", [None, 1, [None], [1], ["row"], [[("value", 1)]],
    [{"position": 0, "value": 1}], [{"position": -1}], [{"position": "0"}]])
def test_section_refuses_nonmapping_rows_or_reserved_position(plan_case, rows):
    error = _error(lambda: make_plan(plan_case).publish("fixture", rows))
    assert error.problem.retryable is False


def test_section_accepts_mapping_iterators_and_collect_rejects_wrong_plan_id(plan_case):
    run = make_plan(plan_case)
    plan = run.seal(clock=plan_case.clock)
    section = run.publish("fixture", (UserDict(value=value) for value in (3, 1)))
    rows = list(iter_section(plan_case.store, "fixture", section["descriptor"], lambda row: (row["position"],)))
    assert rows == [{"position": 0, "value": 3}, {"position": 1, "value": 1}]
    append_record(plan_case.conn, plan_case.store, plan_case.trials.own.run_id,
        "distribution:run-completed", {"run_plan_id": content_hash("different-plan"), "sections": [section]})
    assert plan["run_plan_id"] != content_hash("different-plan")
    _error(run.collect)


@pytest.mark.parametrize("wrong_id", [content_hash("stale-run-plan"), "mistyped-plan-id"])
def test_completion_id_refusal_preserves_slot_for_correction_without_readmission(plan_case, monkeypatch, wrong_id):
    run = make_plan(plan_case)
    plan = run.seal(clock=plan_case.clock)
    section = run.publish("fixture", [{"value": 1}])
    correct = {"run_plan_id": plan["run_plan_id"], "sections": [section]}
    before = plan_case.conn.total_changes
    def forbidden(*args, **kwargs):
        pytest.fail("completion binding repeated admission or read report shards")
    with monkeypatch.context() as patch, no_fit_guard(), legacy_no_fit():
        patch.setattr(run, "preflight", forbidden)
        patch.setattr(metadata, "iter_section", forbidden)
        error = _error(lambda: run.record("completed", {**correct, "run_plan_id": wrong_id}))
        assert error.problem.retryable is False
        assert plan_case.conn.total_changes == before and run.record("completed") is None
        assert run.record("completed", correct) == correct
        after = plan_case.conn.total_changes
        assert run.record("completed", correct) == correct
        assert plan_case.conn.total_changes == after
        _error(lambda: run.record("completed", correct, replay=False), "EXPERIMENT_IDENTITY_CONFLICT")
    assert run.collect() == correct


def test_completion_without_seal_refuses_then_can_be_sealed_and_completed(plan_case, monkeypatch):
    run = make_plan(plan_case)
    plan = run.preflight()
    complete = {"run_plan_id": plan["run_plan_id"], "sections": []}
    before = plan_case.conn.total_changes
    def forbidden(*args, **kwargs):
        pytest.fail("missing seal caused artifact publication")
    with monkeypatch.context() as patch:
        patch.setattr(plan_case.store, "publish_bytes", forbidden)
        _error(lambda: run.record("completed", complete))
    assert plan_case.conn.total_changes == before and plan_case.slots() == []
    assert run.record("completed") is None
    assert run.seal(clock=plan_case.clock) == plan
    assert run.record("completed", complete) == complete
    assert run.collect() == complete


@pytest.mark.parametrize("limits", [None, [], {}, {"memory_bytes": 1},
    {"memory_bytes": 1, "cpu_seconds": 1, "wall_seconds": 1, "cpu_id": 0}])
def test_containment_refuses_malformed_limit_shape_before_process_access(limits, monkeypatch):
    _refuse_containment_without_process_access(limits, 1, monkeypatch)


@pytest.mark.parametrize("field", ["memory_bytes", "cpu_seconds", "wall_seconds"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), float("inf"), sys.maxsize])
def test_containment_reuses_positive_integer_limit_contract(field, value, monkeypatch):
    limits = {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 30}
    limits[field] = value
    _refuse_containment_without_process_access(limits, 1, monkeypatch)


@pytest.mark.parametrize("seconds", [None, "1", True, 0, -0.5, float("nan"),
    float("inf"), float("-inf")])
def test_containment_refuses_invalid_remaining_duration_before_process_access(seconds, monkeypatch):
    limits = {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 30}
    _refuse_containment_without_process_access(limits, seconds, monkeypatch)


def _refuse_containment_without_process_access(limits, seconds, monkeypatch):
    def forbidden(*args):
        pytest.fail("invalid containment request reached process state or watchdog")
    monkeypatch.setattr(metadata.signal, "getitimer", forbidden)
    monkeypatch.setattr(metadata, "rss_watchdog", forbidden)
    with pytest.raises(OpsError) as caught:
        with metadata.contained(limits, seconds):
            pytest.fail("invalid containment request entered its body")
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False


def test_actual_containment_times_out_and_restores_limits_in_dedicated_process():
    script = """import resource, signal, time
from experiments.distribution_run_plan import contained
from engine.v2.ops.errors import OpsError
old = [resource.getrlimit(kind) for kind in (resource.RLIMIT_AS, resource.RLIMIT_CPU)]
handler = signal.getsignal(signal.SIGALRM)
try:
    with contained({'memory_bytes': 4 << 30, 'cpu_seconds': 30, 'wall_seconds': 30}, .02):
        time.sleep(.2)
    raise AssertionError('deadline did not fire')
except OpsError as error:
    assert error.code == 'RESOURCE_LIMIT_EXCEEDED'
assert [resource.getrlimit(kind) for kind in (resource.RLIMIT_AS, resource.RLIMIT_CPU)] == old
assert signal.getsignal(signal.SIGALRM) == handler
assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
print('restored')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **dict.fromkeys(THREAD_VARIABLES, "1")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "restored"


@pytest.mark.parametrize("scenario", ["virtual", "rss"])
def test_containment_uses_resident_memory_and_restores_process_state(scenario):
    script = """import mmap, resource, signal, time
from experiments import memory_guard
from experiments.distribution_run_plan import contained
old = {kind: resource.getrlimit(kind) for kind in (resource.RLIMIT_AS, resource.RLIMIT_CPU)}
handler = signal.getsignal(signal.SIGALRM)
def previous_rss_handler(*args): pass
signal.signal(signal.SIGUSR1, previous_rss_handler)
scenario = SCENARIO
if scenario == 'rss': memory_guard.rss_bytes = lambda: 4 << 30
try:
    with contained({'memory_bytes': 2 << 30, 'cpu_seconds': 30, 'wall_seconds': 30}, 30):
        assert resource.getrlimit(resource.RLIMIT_AS) == old[resource.RLIMIT_AS]
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        assert soft != resource.RLIM_INFINITY and hard == old[resource.RLIMIT_CPU][1]
        if scenario == 'virtual':
            with mmap.mmap(-1, 4 << 30) as reserved:
                reserved[0] = 1
        else:
            time.sleep(.2)
except MemoryError:
    assert scenario == 'rss'
else:
    assert scenario == 'virtual', 'RSS watchdog did not interrupt'
assert {kind: resource.getrlimit(kind) for kind in old} == old
assert signal.getsignal(signal.SIGALRM) == handler
assert signal.getsignal(signal.SIGUSR1) == previous_rss_handler
assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
print('restored')
""".replace("SCENARIO", repr(scenario))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **dict.fromkeys(THREAD_VARIABLES, "1")})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "restored"
