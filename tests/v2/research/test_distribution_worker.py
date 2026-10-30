"""Standalone worker slice: real small CAS fits, no later executor dependency."""
# ruff: noqa: F811
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import canonical_json, content_hash, from_document
from engine.v2.ops.executor import THREAD_VARIABLES
from experiments import distribution_fitted as fitted
from experiments import distribution_worker as worker
from experiments.magnitude_search_evidence import fit_evidence
from experiments.magnitude_search_inputs import publish_inputs
from experiments.magnitude_search_shards import LIST_MEMBERS, iter_section
from tests.v2.research.test_distribution_fitted import case  # noqa: F401

LIMITS = {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 30}
ROOT = Path(__file__).resolve().parents[3]


def publish_case(case):
    """Small real manifest shared by independent worker and later executor tests."""
    complete = deepcopy(case.inputs)
    complete["features"].sort(key=lambda row: (row["event_id"], row["horizon_sessions"]))
    for member in LIST_MEMBERS:
        *parents, leaf = member["section"].split(".")
        target = complete
        for name in parents:
            target = target.setdefault(name, {})
        target.setdefault(leaf, [])
    manifest, inputs_id = publish_inputs(case.trials.store, complete)
    original = case.trials.read_fit_context

    def context(*args):
        value = original(*args)
        value["plan"]["budget"] = {"child_limits": LIMITS.copy()}
        value["admission"]["manifest"] = manifest
        for trial in value["admission"]["trials"]:
            trial["inputs_id"] = inputs_id
            trial["trial_id"] = content_hash({key: val for key, val in trial.items() if key != "trial_id"})
        return value

    case.trials.read_fit_context = context
    case.source = {"manifest": manifest, "inputs_id": inputs_id, "store_root": str(case.trials.store.root)}
    return case


@pytest.fixture
def admitted(case):
    return publish_case(case)


def launch(admitted, tmp_path, *, setup="", overrides=None):
    header = tmp_path / "header.json"
    header.write_text(canonical_json({**admitted.source, "recipe": admitted.recipe,
        "events": ["Z", "A"], "horizon": 1, **(overrides or {})}))
    # Derive the tested module's root so the standalone checkpoint also works
    # under an isolated import overlay, without swapping shared checkout files.
    code_root = Path(fitted.__file__).resolve().parents[1]
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join(map(str, (code_root, ROOT))), **dict.fromkeys(THREAD_VARIABLES, "1")}
    poison = """import importlib.abc, sys
class NoExecutor(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'experiments.distribution_execution':
            raise AssertionError('worker slice imported future executor')
sys.meta_path.insert(0, NoExecutor())
from engine.v2.ops import catalog
from engine.v2.research import magnitude_inputs
from engine.v2.data.repository import Repository
def forbidden(*args, **kwargs):
    raise AssertionError('worker accessed catalog or outcomes')
catalog.connect = forbidden
magnitude_inputs.load_magnitude_targets = forbidden
Repository.__init__ = forbidden
"""
    script = poison + setup + "\nfrom experiments.distribution_worker import main\nmain()\n"
    completed = subprocess.run([sys.executable, "-c", script, str(header),
        canonical_json({**LIMITS, "cpu_id": min(os.sched_getaffinity(0))})], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=40)
    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.encode()) < 4096 and len(completed.stdout.splitlines()) == 1
    return json.loads(completed.stdout), completed


@pytest.mark.parametrize("labels,tail,trees", [([0, .25, 0, .25], [], 1), ([0, .25, 20, 22], [0, 2], 3)])
def test_standalone_real_fit_returns_only_verified_cas_descriptor(admitted, tmp_path, labels, tail, trees):
    for row, value in zip(admitted.inputs["training"], labels):
        row["absolute_move_pct"] = value
    publish_case(admitted)
    ref, completed = launch(admitted, tmp_path)
    reference = from_document(ArtifactRef, ref)
    assert reference.schema_ref == "distribution_fit_output.v1"
    result = json.loads(admitted.trials.store.read_verified(reference))
    assert set(result) == {"recipe_id", "forecasts", "tail", "diagnostics", "evidence", "ordered_events_id"}
    assert result["recipe_id"] == admitted.recipe["recipe_id"]
    expected = fit_evidence(admitted.inputs, {"evaluation_event_ids": ["Z", "A"]}, {"horizons": [1]}, {"features": admitted.columns})[0]
    assert result["evidence"] == {name: expected[name] for name in ("horizon_sessions", "training_matrix_id", "prediction_matrix_id")}
    assert result["ordered_events_id"] == content_hash([expected["training_event_ids"], ["Z", "A"]])
    rows = list(iter_section(admitted.trials.store, "distribution_fit_forecasts", result["forecasts"], lambda row: (row["event_id"],)))
    assert [row["event_id"] for row in rows] == ["A", "Z"]
    assert rows[0]["point_prediction"] == pytest.approx(sum(labels) / len(labels))
    expected_masses = [0.] * 81
    for label in labels:
        expected_masses[min(int(label * 4), 80)] += 1 / len(labels)
    assert rows[0]["probabilities"] == pytest.approx(expected_masses)
    samples = list(iter_section(admitted.trials.store, "distribution_fit_tail", result["tail"]["excesses"], lambda row: (row["position"],)))
    assert samples == [{"position": index, "excess": value} for index, value in enumerate(tail)]
    assert result["tail"]["policy"] == ("empirical_excess" if tail else "unresolved")
    assert result["diagnostics"]["classifier"]["trees_per_iteration"] == trees
    assert admitted.records == {("source-run", "seal"): {"untouched": True}}
    assert "probabilities" not in completed.stdout


def test_wrong_manifest_identity_refuses_before_numeric_work(admitted, tmp_path):
    admitted.source["inputs_id"] = content_hash("wrong manifest")
    response, _ = launch(admitted, tmp_path,
        setup="from experiments import distribution_candidate_fit as numeric\nnumeric.fit_candidate = forbidden\n")
    assert response == {"error": "EXPERIMENT_IDENTITY_CONFLICT"}


@pytest.mark.parametrize("mutation", [
    "result['schema_version'] = 'unsupported'",
    "result['recipe_id'] = 'wrong recipe'",
    "result['probabilities'].pop()",
    "result['point_predictions'].pop()",
    "result['probabilities'].append(result['probabilities'][0])",
    "result['point_predictions'].append(1.0)",
    "result['probabilities'][0][0] = 0.; result['probabilities'][0][2] = .25",
    "result['point_predictions'][0] = -1.0",
    "result['point_predictions'][0] = float('nan')",
    "result['point_predictions'][0] = float('inf')",
    "result['tail']['excesses'][-1] = 3.0",
    "result['diagnostics']['classifier']['total_trees'] += 1",
    "result['diagnostics']['classifier']['class_labels'] = [0, 1]",
    "result['diagnostics']['prediction_rows'] += 1",
    "result['diagnostics']['all_null_training_columns'] = []",
])
def test_corrupted_numeric_output_refuses_before_any_cas_publication(admitted, tmp_path, mutation):
    before = {path.relative_to(admitted.trials.store.root): path.read_bytes()
              for path in admitted.trials.store.root.rglob('*') if path.is_file()}
    setup = ("from experiments import distribution_candidate_fit as numeric\n"
             "original_fit = numeric.fit_candidate\n"
             "def corrupted_fit(*args, **kwargs):\n"
             " result = original_fit(*args, **kwargs)\n"
             f" {mutation}\n"
             " return result\n"
             "numeric.fit_candidate = corrupted_fit\n")
    response, _ = launch(admitted, tmp_path, setup=setup)
    assert response == {"error": "EXPERIMENT_IDENTITY_CONFLICT"}
    after = {path.relative_to(admitted.trials.store.root): path.read_bytes()
             for path in admitted.trials.store.root.rglob('*') if path.is_file()}
    assert after == before


def test_one_class_is_candidate_failure_without_success_reference(admitted, tmp_path):
    for row in admitted.inputs["training"]:
        row["absolute_move_pct"] = 1
    publish_case(admitted)
    response, _ = launch(admitted, tmp_path)
    assert response == {"error": "EXPERIMENT_VARIANT_FAILED"}


@pytest.mark.parametrize("exception,code", [("MemoryError", "RESOURCE_LIMIT_EXCEEDED"), ("RuntimeError", "RESOURCE_UNAVAILABLE")])
def test_worker_failures_are_small_sanitized_refusals(admitted, tmp_path, exception, code):
    response, completed = launch(admitted, tmp_path,
        setup="from experiments import distribution_candidate_fit as numeric\n"
              f"def fail(*args, **kwargs): raise {exception}('private-synthetic-sentinel')\n"
              "numeric.fit_candidate = fail\n")
    assert response == {"error": code}
    assert "private-synthetic-sentinel" not in completed.stdout + completed.stderr


def test_reserved_address_space_above_rss_budget_does_not_refuse(admitted, tmp_path):
    response, _ = launch(admitted, tmp_path, setup=
        "import mmap\nreserved = mmap.mmap(-1, 4 << 30)\n")
    assert from_document(ArtifactRef, response).schema_ref == "distribution_fit_output.v1"


def test_shared_rss_watchdog_exhaustion_returns_resource_refusal(admitted, tmp_path):
    response, _ = launch(admitted, tmp_path, setup=
        "import time\nfrom experiments import memory_guard, distribution_candidate_fit as numeric\n"
        "memory_guard.rss_bytes = lambda: 4 << 30\noriginal = numeric.fit_candidate\n"
        "def wait_for_watchdog(*args, **kwargs):\n time.sleep(.2)\n return original(*args, **kwargs)\n"
        "numeric.fit_candidate = wait_for_watchdog\n")
    assert response == {"error": "RESOURCE_LIMIT_EXCEEDED"}


def test_worker_header_budget_is_enforced_before_decode(admitted, tmp_path):
    response, _ = launch(admitted, tmp_path,
        setup="from experiments import magnitude_search_records as records\nrecords.MAX_BYTES = 32\n")
    assert response == {"error": "RESOURCE_LIMIT_EXCEEDED"}


@pytest.mark.parametrize("field,value", [
    ("horizon", True), ("horizon", 9), ("horizon", "1"),
    ("events", ["Z", "Z"]), ("events", ["unknown"]), ("events", []),
    ("events", None), ("events", "Z"), ("events", [""]),
    ("events", [True]), ("events", [{}]),
    ("store_root", ""), ("store_root", None), ("store_root", 1),
])
def test_invalid_worker_axes_refuse_before_evidence_fit_or_publication(admitted, tmp_path, field, value):
    setup = ("from engine.v2.foundation import ArtifactStore\n"
             "from experiments import distribution_candidate_fit as numeric, magnitude_search_evidence as evidence\n"
             "numeric.fit_candidate = evidence.fit_evidence = ArtifactStore.publish_bytes = forbidden\n")
    if (field, value) != ("events", ["unknown"]):
        setup += "ArtifactStore.__init__ = forbidden\n"
    response, _ = launch(admitted, tmp_path, setup=setup, overrides={field: value})
    assert response == {"error": "INVALID_EXPERIMENT_SPEC"}
    assert admitted.records == {("source-run", "seal"): {"untouched": True}}


@pytest.mark.parametrize("field", ["memory_bytes", "cpu_seconds"])
@pytest.mark.parametrize("value", [-1, 0, True, 1.0, "1", None, sys.maxsize, sys.maxsize + 1])
def test_invalid_resource_scalars_refuse_before_applying_any_limit(monkeypatch, capsys, field, value):
    limits = {**LIMITS, "cpu_id": 0, field: value}
    monkeypatch.setattr(sys, "argv", ["worker", "unused-header", json.dumps(limits)])
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid limits reached containment or header access")

    monkeypatch.setattr(worker.resource, "setrlimit", forbidden)
    monkeypatch.setattr(worker.os, "sched_setaffinity", forbidden)
    monkeypatch.setattr(worker, "start_rss_watchdog", forbidden, raising=False)
    monkeypatch.setattr("builtins.open", forbidden)
    worker.main()
    assert json.loads(capsys.readouterr().out) == {"error": "INVALID_EXPERIMENT_SPEC"}
    assert calls == []


@pytest.mark.parametrize("value", ["missing", None, False, True, -1, 2 ** 31 - 1,
                                   2 ** 31, 0.0, 1.5, "0", [], {}])
def test_invalid_affinity_refuses_before_any_os_or_header_access(monkeypatch, capsys, value):
    limits = {**LIMITS, "cpu_id": value}
    if value == "missing":
        del limits["cpu_id"]
    monkeypatch.setattr(sys, "argv", ["worker", "unused-header", json.dumps(limits)])
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid affinity reached containment or header access")

    monkeypatch.setattr(worker.resource, "setrlimit", forbidden)
    monkeypatch.setattr(worker.os, "sched_setaffinity", forbidden)
    monkeypatch.setattr(worker, "start_rss_watchdog", forbidden, raising=False)
    monkeypatch.setattr("builtins.open", forbidden)
    worker.main()
    assert json.loads(capsys.readouterr().out) == {"error": "INVALID_EXPERIMENT_SPEC"}
    assert calls == []


@pytest.mark.parametrize("cpu_id", [0, 2 ** 31 - 2])
def test_valid_affinity_syntax_reaches_containment_before_header(monkeypatch, capsys, cpu_id):
    monkeypatch.setattr(sys, "argv", ["worker", "unused-header", json.dumps({**LIMITS, "cpu_id": cpu_id})])
    calls = []
    monkeypatch.setattr(worker.resource, "setrlimit", lambda *args: calls.append(("limit", *args)))
    monkeypatch.setattr(worker.os, "sched_setaffinity", lambda *args: calls.append(("affinity", *args)))
    monkeypatch.setattr(worker, "start_rss_watchdog", lambda value: calls.append(("rss", value)), raising=False)

    def stop_at_header(*args):
        calls.append(("header", *args))
        raise RuntimeError("synthetic header stop")

    monkeypatch.setattr("builtins.open", stop_at_header)
    worker.main()
    assert json.loads(capsys.readouterr().out) == {"error": "RESOURCE_UNAVAILABLE"}
    assert calls == [("limit", worker.resource.RLIMIT_CPU, (30, 31)), ("affinity", 0, {cpu_id}),
        ("rss", LIMITS["memory_bytes"]),
        ("header", "unused-header", "rb")]


@pytest.mark.parametrize("recipe", [None, [], True, 1, "recipe", {},
                                    {"horizons": [1]}, {"horizons": [1], "point_recipe": None}])
def test_invalid_worker_recipe_refuses_before_cas_or_numeric_work(admitted, tmp_path, recipe):
    setup = ("from engine.v2.foundation import ArtifactStore\n"
             "from experiments import distribution_candidate_fit as numeric, magnitude_search_inputs as inputs\n"
             "ArtifactStore.__init__ = numeric.fit_candidate = inputs.load_fit_inputs = forbidden\n")
    response, _ = launch(admitted, tmp_path, setup=setup, overrides={"recipe": recipe})
    assert response == {"error": "INVALID_EXPERIMENT_SPEC"}


@pytest.mark.parametrize("header", [None, [], 7, {"horizon": 1, "events": ["Z", "A"]}])
def test_nonmapping_header_or_missing_recipe_refuses_before_storage(admitted, tmp_path, header):
    setup = ("from pathlib import Path\nfrom engine.v2.foundation import ArtifactStore\n"
             "ArtifactStore.__init__ = forbidden\n"
             f"Path(sys.argv[1]).write_text({json.dumps(header)!r})\n")
    response, _ = launch(admitted, tmp_path, setup=setup)
    assert response == {"error": "INVALID_EXPERIMENT_SPEC"}


def test_representable_but_unavailable_affinity_remains_resource_failure(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["worker", "unused-header", json.dumps({**LIMITS, "cpu_id": 0})])
    monkeypatch.setattr(worker.resource, "setrlimit", lambda *args: None)
    calls = []

    def unavailable(*args):
        calls.append("affinity")
        raise OSError("private-synthetic-sentinel")

    def forbidden(*args, **kwargs):
        calls.append("header")
        raise AssertionError("unavailable affinity reached header")

    monkeypatch.setattr(worker.os, "sched_setaffinity", unavailable)
    monkeypatch.setattr("builtins.open", forbidden)
    worker.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"error": "RESOURCE_UNAVAILABLE"}
    assert "private-synthetic-sentinel" not in captured.out + captured.err
    assert calls == ["affinity"]
