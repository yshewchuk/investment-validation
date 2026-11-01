"""Dedicated-process native synthetic run with six actual bounded fit children."""
import json
import os
import subprocess
import sys
from pathlib import Path

from engine.v2.ops.executor import THREAD_VARIABLES

SCRIPT = r'''
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from engine.models.no_fit import no_fit_guard as legacy_no_fit
from engine.v2.data import objects
from engine.v2.foundation import SystemClock, canonical_json
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.research.event_history import MAX_OUTCOME_OBJECT_BYTES
from experiments import distribution_run as runner
from experiments.distribution_fitted import DistributionFits
from experiments.distribution_run import DistributionRun
from tests.v2.research.test_distribution_pipeline_native import make_run, native_case

root = Path(sys.argv[1])
generator = native_case.__wrapped__(SimpleNamespace(mktemp=lambda _: root))
case = next(generator)
run = make_run(case, run_type=DistributionRun, runner="experiments/distribution_run.py",
               experiment_id="synthetic-distribution-bounded-run")
run.seal(clock=SystemClock())
original, original_bytes, opened = objects.verify_object_path, objects.read_verified_bytes, set()
def verify(store, ref):
    assert ref.object_id not in case.target_objects, "outer targets require a byte-bounded read"
    return original(store, ref)
def read_bytes(store, ref, *, max_bytes):
    if ref.object_id in case.target_objects:
        assert max_bytes == MAX_OUTCOME_OBJECT_BYTES
        assert all(run.record(f"intents:2024:{alpha}") is not None for alpha in run.request["alphas"])
        opened.add(ref.object_id)
    return original_bytes(store, ref, max_bytes=max_bytes)
objects.verify_object_path = verify
objects.read_verified_bytes = read_bytes
evaluate, evaluations = runner.evaluate_outer, []
def complete_outer(run, group, fits, results, plan, books, quotes, *rest):
    assert set(results) == {1, 3, 5}
    assert set(books) == set(quotes) == set(plan["request"]["alphas"]) == {0, .25, .5}
    expected = set(group["outer"]["evaluation_event_ids"])
    assert expected and all({row["event_id"] for row in result["rows"]} == expected for result in results.values())
    evaluations.append(group["outer"]["year"])
    return evaluate(run, group, fits, results, plan, books, quotes, *rest)
runner.evaluate_outer = complete_outer
result = run.run()
assert evaluations == ["2024"]
assert opened == case.target_objects and len(opened) == 1
assert result["final_read_authority"] == "none" and result["promotion_authority"] == "none"
assert result["current_strategy_superiority_claim"] is False
assert run.record("failed") is None
fits = DistributionFits(run.trials)
for role, inner in (("inner", "2023"), ("outer", None)):
    for horizon in (5, 3, 1):
        receipt = fits.read(role, "2024", horizon, inner)["receipt"]
        assert receipt["schema_version"] == "distribution_fitted.v2"
        assert receipt["execution"]["worker"] == "experiments/distribution_worker.py"
        assert receipt["execution"]["limits"] == case.request["limits"]["child"]
        assert run.record("fit:" + receipt["claim"]["trial"]["trial_id"]) == {
            "result_id": receipt["result_id"], "status": "bounded_completed"}
for item in result["reports"]:
    assert item["report"]["arms"]["histogram"]["funded"] == 1
    assert item["report"]["arms"]["histogram"]["settled_net_profit"] == 100
    assert item["report"]["contrasts"]["same_mean"]["net_delta_per_opportunity"] == 0

def records():
    return case.conn.execute("SELECT slot, ref_json FROM magnitude_search_records WHERE run_id=? ORDER BY slot",
        (run.trials.own.run_id,)).fetchall()
before = records()
assert sum(slot.startswith("distribution:fit-bound:") for slot, _ in before) == 6
assert sum(slot.startswith("distribution:fit-result:") for slot, _ in before) == 6
def no_targets(store, ref):
    assert ref.object_id not in case.target_objects, "completed replay reopened evaluation target bytes"
    return original(store, ref)
objects.verify_object_path = no_targets
def no_target_bytes(store, ref, *, max_bytes):
    assert ref.object_id not in case.target_objects, "completed replay reopened evaluation target bytes"
    return original_bytes(store, ref, max_bytes=max_bytes)
objects.read_verified_bytes = no_target_bytes
with no_fit_guard(), legacy_no_fit():
    assert canonical_json(run.collect()) == canonical_json(result)
    assert canonical_json(run.run()) == canonical_json(result)
assert records() == before
objects.verify_object_path = original
objects.read_verified_bytes = original_bytes
print(json.dumps({"status": "completed", "bounded_children": 6, "read_only_replay": True,
                  "reports": len(result["reports"]), "sections": len(result["sections"])}))
try:
    next(generator)
except StopIteration:
    pass
'''


def test_real_native_bounded_run_and_read_only_collection(tmp_path):
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / "native")],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=180,
        env={**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload["status"] == "completed" and payload["bounded_children"] == 6
    assert payload["read_only_replay"] is True and payload["reports"] == 3
    assert payload["sections"] > 0
