"""Synthetic real-journal oracles for driver provenance and atomic phase admission."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, to_document
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import source_closure
from experiments import magnitude_search_records as records
from experiments import magnitude_search_trials as trials
from experiments import v2_magnitude_search_run as driver

DRIVER_PATH = Path(driver.__file__)
DRIVER_ENTRY = "experiments/v2_magnitude_search_run.py"
LIMITS = {"memory_bytes": 2 * 1024 ** 3, "cpu_seconds": 30, "wall_seconds": 60}
TRIALS = [(outer, inner, candidate) for outer, inners in
          (("2024", ("2022", "2023")), ("2025", ("2023", "2024")))
          for inner in inners for candidate in ("a-first", "z-last")]
PHASES = [*("inner:" + ":".join(args) for args in TRIALS), "outer:2024", "outer:2025", "selection"]


@pytest.fixture
def driver_case(tmp_path, monkeypatch):
    """Only plan validation is synthetic; hashes, migrations, records and locks are real."""
    path = tmp_path / "catalog.sqlite"
    conn = connect(path)
    conn.execute("CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY) STRICT")
    conn.execute("INSERT INTO experiment_runs VALUES ('run')")
    store = ArtifactStore(tmp_path / "store")
    plan = {"plan_id": "plan", "horizons": [5, 3, 1], "candidates": [
        {"candidate_id": "z-last"}, {"candidate_id": "a-first"}], "folds": [
        {"outer": {"year": outer}, "inner": [{"year": inner} for inner in inners]}
        for outer, inners in (("2024", ("2022", "2023")), ("2025", ("2023", "2024")))]}
    plan_ref = store.publish_bytes(b"synthetic plan", schema_ref="magnitude_search_plan.v1")
    registration = SimpleNamespace(run_id="run", document={})
    request = {"synthetic": True}
    args = (conn, store, None, registration, request, plan_ref)
    source = tmp_path / "source"
    (source / "experiments").mkdir(parents=True)
    (source / "experiments/__init__.py").write_text("")
    (source / DRIVER_ENTRY).write_text("from experiments.driver_dependency import VALUE\n")
    (source / "experiments/driver_dependency.py").write_text("VALUE = 1\n")
    monkeypatch.setattr(driver, "__file__", str(source / DRIVER_ENTRY))
    monkeypatch.setattr(driver, "read_search_plan", lambda *a: plan)
    scientific = {"plan_id": "plan", "plan_ref": to_document(plan_ref), "recorder_sources": {"synthetic": "v1"}}
    monkeypatch.setattr(trials.SearchTrials, "_binding", lambda self: (plan, scientific))
    case = SimpleNamespace(conn=conn, catalog=path, store=store, plan=plan, plan_ref=plan_ref,
        registration=registration, request=request, args=args, source=source,
        recorder=trials.SearchTrials(*args), clock=SystemClock(), scientific=scientific)
    yield case
    conn.close()


def seal(case):
    return driver.seal_search_run(*case.args, clock=case.clock)


def read(case, slot):
    return records.read_record(case.conn, case.store, "run", slot)


def phase_slot(case, phase):
    if phase.startswith("inner:"):
        return "start:" + content_hash([case.plan["plan_id"], *phase.split(":")[1:]])
    if phase.startswith("outer:"):
        return "outer-start:" + content_hash([case.plan["plan_id"], phase.split(":")[1]])
    assert phase == "selection"
    return phase


def snapshot(case):
    return (list(case.conn.iterdump()), case.conn.total_changes,
        {path.relative_to(case.store.root): path.read_bytes()
         for path in (case.store.root / "objects").rglob("*") if path.is_file()})


def no_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("driver admission must precede fitting, collection, and scientific work")
    for method in ("execute", "reconcile", "freeze", "start"):
        monkeypatch.setattr(trials.SearchTrials, method, forbidden)
    monkeypatch.setattr(driver, "execute_outer_fold", forbidden)
    monkeypatch.setattr(driver, "collect_outer_results", forbidden)


def test_seal_binds_exact_plan_reference_and_transitive_driver_sources(driver_case):
    case = driver_case
    result = seal(case)
    binding = read(case, "driver-seal")
    assert binding["plan_id"] == case.plan["plan_id"]
    assert binding["plan_ref"] == to_document(case.plan_ref)
    assert binding["driver_sources"] == source_closure(case.source, [DRIVER_ENTRY])
    assert "experiments/driver_dependency.py" in binding["driver_sources"]
    assert result == {"sealed": "plan", "run_id": "run", "driver_id": content_hash(binding)}
    assert read(case, "seal") == case.scientific
    assert driver._admit_driver(case.conn, case.store, case.registration,
                                case.request, case.plan_ref) == case.plan


def test_existing_empty_scientific_journal_can_be_driver_sealed(driver_case):
    case = driver_case
    scientific = case.recorder.seal(clock=case.clock)
    assert read(case, "driver-seal") is None
    result = seal(case)
    assert result["driver_id"] == content_hash(read(case, "driver-seal"))
    assert read(case, "seal") == scientific
    assert [row[0] for row in case.conn.execute(
        "SELECT slot FROM magnitude_search_records ORDER BY slot")] == ["driver-seal", "seal"]


@pytest.mark.parametrize("phase", [None, *PHASES])
def test_exact_driver_seal_replay_is_read_only_even_after_work(driver_case, monkeypatch, phase):
    case = driver_case
    result = seal(case)
    if phase:
        records.append_record(case.conn, case.store, "run", phase_slot(case, phase), {"synthetic": phase})
    before = snapshot(case)
    no_work(monkeypatch)
    monkeypatch.setattr(case.store, "publish_bytes", lambda *a, **kw: pytest.fail("seal replay published bytes"))
    with no_fit_guard(), legacy_guard():
        assert seal(case) == result
    assert snapshot(case) == before


@pytest.mark.parametrize("phase", PHASES)
def test_missing_driver_binding_cannot_be_added_after_any_admitted_phase(driver_case, monkeypatch, phase):
    case = driver_case
    case.recorder.seal(clock=case.clock)
    records.append_record(case.conn, case.store, "run", phase_slot(case, phase), {"synthetic": phase})
    before = snapshot(case)
    no_work(monkeypatch)
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        seal(case)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT",
        "search driver cannot bind an already started journal").problem
    assert read(case, "driver-seal") is None
    assert snapshot(case) == before


@pytest.mark.parametrize("phase", PHASES)
def test_phase_inserted_during_driver_publication_wins_atomic_absence_guard(driver_case, monkeypatch, phase):
    case = driver_case
    case.recorder.seal(clock=case.clock)
    slot = phase_slot(case, phase)
    original = case.store.publish_bytes
    raced = []

    def publish(payload, **kwargs):
        document = json.loads(payload)
        if document.get("slot") == "driver-seal":
            other = connect(case.catalog, must_exist=True)
            try:
                records.append_record(other, ArtifactStore(case.store.root), "run", slot, {"raced": phase})
                raced.append(slot)
            finally:
                other.close()
        return original(payload, **kwargs)

    monkeypatch.setattr(case.store, "publish_bytes", publish)
    with pytest.raises(OpsError) as caught:
        seal(case)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial phase is already committed").problem
    assert raced == [slot]
    assert read(case, slot) == {"raced": phase}
    assert read(case, "driver-seal") is None


@pytest.mark.parametrize("change", ["driver", "dependency", "plan", "plan_ref", "missing_source"])
@pytest.mark.parametrize("operation", ["seal", "run", "admit"])
def test_changed_driver_binding_refuses_without_writes_or_work(driver_case, monkeypatch, change, operation):
    case = driver_case
    seal(case)
    records.append_record(case.conn, case.store, "run", "selection", {"synthetic": "selected"})
    if change in ("driver", "dependency"):
        name = DRIVER_ENTRY if change == "driver" else "experiments/driver_dependency.py"
        with (case.source / name).open("a") as stream:
            stream.write("# changed after seal\n")
    elif change == "missing_source":
        (case.source / DRIVER_ENTRY).unlink()
    elif change == "plan":
        changed = {**case.plan, "plan_id": "other-plan"}
        monkeypatch.setattr(driver, "read_search_plan", lambda *a: changed)
    else:
        reference = case.store.publish_bytes(b"other plan", schema_ref="magnitude_search_plan.v1")
        case.plan_ref = reference
        case.args = (*case.args[:-1], reference)
    before = snapshot(case)
    no_work(monkeypatch)
    operations = {"seal": lambda: seal(case), "run": lambda: driver.run_search(*case.args, limits=LIMITS),
        "admit": lambda: driver._admit_driver(case.conn, case.store, case.registration, case.request, case.plan_ref)}
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        operations[operation]()
    message = ("search driver sources cannot be verified" if change == "missing_source" else
        "search journal slot is already committed" if operation == "seal" else
        "search execution requires its exact driver seal")
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", message).problem
    assert snapshot(case) == before


@pytest.mark.parametrize("scientific_sealed", [False, True])
def test_run_without_driver_seal_is_read_only_and_cannot_start_work(driver_case, monkeypatch, scientific_sealed):
    case = driver_case
    if scientific_sealed:
        case.recorder.seal(clock=case.clock)
    before = snapshot(case)
    no_work(monkeypatch)
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        driver.run_search(*case.args, limits=LIMITS)
    expected = (fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution requires its exact driver seal") if scientific_sealed
        else fail("INTEGRITY_FAILED", "magnitude_search schema is missing migrations",
            details={"reason": "schema_older", "owner": "magnitude_search", "missing": [1]}))
    assert caught.value.problem == expected.problem
    assert snapshot(case) == before


def test_valid_hash_different_driver_seal_cannot_admit_or_be_replaced(driver_case, monkeypatch):
    case = driver_case
    case.recorder.seal(clock=case.clock)
    _, binding = driver._binding(case.conn, case.store, case.registration, case.request, case.plan_ref)
    changed = copy.deepcopy(binding)
    changed["driver_sources"][DRIVER_ENTRY] = "sha256:" + "0" * 64
    records.append_record(case.conn, case.store, "run", "driver-seal", changed)
    before = snapshot(case)
    no_work(monkeypatch)
    for operation, message in ((lambda: seal(case), "search journal slot is already committed"),
            (lambda: driver.run_search(*case.args, limits=LIMITS), "search execution requires its exact driver seal")):
        with pytest.raises(OpsError) as caught:
            operation()
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", message).problem
        assert snapshot(case) == before


def test_driver_uses_only_public_search_trials_methods():
    tree = ast.parse(DRIVER_PATH.read_text())
    receivers = {node.targets[0].id for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "SearchTrials"}
    assert receivers, "driver should compose SearchTrials through its public API"
    assert not [(node.func.value.id, node.func.attr) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id in receivers | {"SearchTrials"}
        and node.func.attr.startswith("_")]
