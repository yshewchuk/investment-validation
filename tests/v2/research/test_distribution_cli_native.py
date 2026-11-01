"""Real operator lifecycle over a tiny synthetic qualified source and pinned CAS."""
import json
import os
import subprocess
import sys
from pathlib import Path

from engine.v2.ops.executor import THREAD_VARIABLES

SCRIPT = r'''
import json
import subprocess
import sys
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace

from engine.v2.foundation import to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.experiments import experiment_spec_from_document
from tests.v2.research.magnitude_trial_support import prediction_spec
from tests.v2.research.test_distribution_pipeline_native import make_run, native_case

root = Path(sys.argv[1])
generator = native_case.__wrapped__(SimpleNamespace(mktemp=lambda _: root))
case = next(generator)
template = make_run(case, experiment_id="synthetic-distribution-cli-inputs")
template.request["limits"]["parent"].update(cpu_seconds=180, wall_seconds=180)
template.request["limits"]["job_wall_seconds"] = 180
spec = prediction_spec(experiment_id="synthetic-distribution-cli", folds=("2024",),
    runner="experiments/distribution_run.py", hypothesis="Compare a raw histogram and matched point model.",
    primary_arm_id="raw_pair", arms=("raw_pair",))
documents = {"spec": to_document(spec), "source-spec": to_document(case.spec),
    "source-request": case.request, "source-plan-ref": to_document(case.ref),
    "recipe": template.trials.recipe, "budget-caps": template.trials.caps, "request": template.request}
arguments = []
for name, document in documents.items():
    path = root / (name + ".json")
    path.write_text(json.dumps(document), encoding="utf-8")
    arguments.extend(("--" + name, str(path)))
catalog = case.conn.execute("PRAGMA database_list").fetchone()[2]
arguments.extend(("--catalog", catalog, "--store-root", str(case.store.root), "--no-ledger"))
guard = """
import json, runpy, sys, threading
from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.data import objects
from experiments import distribution_cli as cli
assert threading.current_thread() is threading.main_thread()
def no_writer(*args, **kwargs):
    raise AssertionError("completed replay opened the writable catalog")
cli.connect = no_writer
targets = set(json.loads(sys.argv.pop(1)))
original, original_bytes = objects.verify_object_path, objects.read_verified_bytes
def verify(store, ref):
    assert ref.object_id not in targets, "replay reopened outer targets"
    return original(store, ref)
def read_bytes(store, ref, *, max_bytes):
    assert ref.object_id not in targets, "replay reopened bounded outer targets"
    return original_bytes(store, ref, max_bytes=max_bytes)
objects.verify_object_path, objects.read_verified_bytes = verify, read_bytes
with no_fit_guard(), legacy_guard():
    runpy.run_module("experiments.distribution_run", run_name="__main__")
"""
def invoke(action, *, replay=False, expected_status=0):
    command = ([sys.executable, "-c", guard, json.dumps(sorted(case.target_objects))] if replay
        else [sys.executable, "-m", "experiments.distribution_run"])
    child = subprocess.run([*command, action, *arguments],
        capture_output=True, text=True, timeout=210)
    assert child.returncode == expected_status, (action, child.stdout, child.stderr)
    result = json.loads(child.stdout)
    assert "distribution " + action in child.stderr
    return result
def rows():
    return {name: case.conn.execute('SELECT * FROM "' + name + '"').fetchall()
        for name, in case.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
def objects_snapshot():
    return sorted((str(path.relative_to(case.store.root)), sha256(path.read_bytes()).hexdigest())
        for path in case.store.root.rglob("*") if path.is_file())
def refuse_external_inputs(actions):
    before, before_objects = rows(), objects_snapshot()
    expected = to_document(fail("INVALID_EXPERIMENT_SPEC", "distribution lifecycle requires no file inputs").problem)
    for name in ("spec", "source-spec"):
        path = root / (name + ".json")
        original = path.read_text()
        document = json.loads(original)
        before_hash = experiment_spec_from_document(document).spec_hash
        document["input_files"] = ["SECRET-unbound-input.csv"]
        assert experiment_spec_from_document(document).spec_hash == before_hash
        path.write_text(json.dumps(document))
        for action in actions:
            assert invoke(action, expected_status=2) == {"refused": expected}
            assert rows() == before and objects_snapshot() == before_objects
        path.write_text(original)
before, before_objects = rows(), objects_snapshot()
prepared = invoke("prepare")
assert prepared["registered"] is False and rows() == before and objects_snapshot() == before_objects
registered = invoke("register")
assert registered == {**prepared, "registered": True}
preflight = invoke("preflight")
sealed = invoke("seal")
assert preflight == sealed and sealed["final_read_authority"] == "none"
refuse_external_inputs(("run",))  # Registered/sealed, but not started: neither spec may add an unbound input.
before, before_objects = rows(), objects_snapshot()
request_file = root / "request.json"
changed = json.loads(request_file.read_text())
changed["limits"]["parent"]["wall_seconds"] -= 1
request_file.write_text(json.dumps(changed))
assert invoke("run", expected_status=2)["refused"]["code"] == "INVALID_EXPERIMENT_SPEC"
assert rows() == before and objects_snapshot() == before_objects
request_file.write_text(json.dumps(template.request))  # Admission refused before an attempt began.
result = invoke("run")
assert result["run_plan_id"] == sealed["run_plan_id"]
assert result["promotion_authority"] == "none" and result["final_read_authority"] == "none"
own_slots = [slot for slot, in case.conn.execute("SELECT slot FROM magnitude_search_records WHERE run_id=?", (prepared["run_id"],))]
assert sum(slot.startswith("distribution:fit-bound:") for slot in own_slots) == 6
assert sum(slot.startswith("distribution:fit-result:") for slot in own_slots) == 6
assert {section["name"] for section in result["sections"] if section["name"].startswith("scores:")} == {
    "scores:2024:1", "scores:2024:3", "scores:2024:5"}
assert [item["alpha"] for item in result["reports"]] == sealed["request"]["alphas"]
before, before_objects = rows(), objects_snapshot()
assert all(invoke(action, replay=True) == result for action in ("collect", "report", "run"))
assert rows() == before and objects_snapshot() == before_objects
refuse_external_inputs(("prepare", "register", "preflight", "seal", "run", "collect", "report"))
for item in result["reports"]:
    assert item["report"]["arms"]["histogram"]["settled_net_profit"] == 100
    assert item["report"]["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 0
print(json.dumps({"status": "completed", "actions": 7, "run_id": prepared["run_id"]}))
try:
    next(generator)
except StopIteration:
    pass
'''


def test_native_operator_prepare_register_seal_run_and_read_only_report(tmp_path):
    child = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / "native")],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=420,
        env={**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout.splitlines()[-1])
    assert result["status"] == "completed" and result["actions"] == 7
