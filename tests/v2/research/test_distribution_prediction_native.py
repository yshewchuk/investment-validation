"""Actual prediction-only CLI over fresh public synthetic native SIZE evidence."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from engine.v2.ops.executor import THREAD_VARIABLES


CHILD = r'''
import json
import runpy
import sqlite3
import sys
import threading
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.data import objects
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.research.event_history import MAX_OUTCOME_OBJECT_BYTES
from experiments import distribution_prediction_cli as cli
from experiments import distribution_prediction_evaluation as evaluation
from experiments import distribution_prediction_evidence as evidence
from experiments.distribution_fitted import DistributionFits
from experiments.distribution_prediction_plan import PredictionPlan
from experiments.distribution_trials import DistributionTrials
from experiments.magnitude_search_records import read_record
from experiments.magnitude_search_shards import iter_section, write_section

assert sys.platform == "linux" and threading.current_thread() is threading.main_thread()
policy = json.loads(sys.argv.pop(1))
targets, opened = set(policy["targets"]), set()
evaluation_reads, native_baselines, substitutions = [], [], []
def frozen_record(store):
    conn = sqlite3.connect(Path(policy["catalog"]).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        frozen = read_record(conn, store, policy["run_id"], "distribution:prediction-frozen:2024")
        assert frozen is not None and [entry["horizon"] for entry in frozen["horizons"]] == [1, 3, 5]
        assert frozen["schema_version"] == "distribution_prediction_frozen.v2"
        assert frozen["reference_encoding"] == "canonical_json_fragments.v1" and frozen["reference_chunk_chars"] == 65536
        for entry in frozen["horizons"]:
            for name in ("forecasts", "reference"):
                section = entry[name]
                assert list(iter_section(store, section["name"], section["descriptor"], lambda row: (row["position"],)))
        return frozen
    finally:
        conn.close()
if policy["mode"] == "substitution":
    # This process operates on a disposable catalog/store copy. Publish valid
    # replacement CAS bytes, then substitute only the record reader's result.
    store = ArtifactStore(sys.argv[sys.argv.index("--store-root") + 1])
    replaced = deepcopy(frozen_record(store))
    section = replaced["horizons"][0]["reference"]
    fragments = list(iter_section(store, section["name"], section["descriptor"], lambda row: (row["position"],)))
    # Only this synthetic test decodes fragments to alter the logical reference.
    reference = json.loads("".join(row["canonical_json"] for row in fragments))
    baseline = reference["baseline"]
    old_id = baseline["baseline_id"]
    baseline["training"][0]["absolute_move_pct"] += .125
    baseline["baseline_id"] = content_hash({key: value for key, value in baseline.items() if key != "baseline_id"})
    assert baseline["baseline_id"] != old_id
    text, chunk_chars = canonical_json(reference), replaced["reference_chunk_chars"]
    altered = [{"position": index, "canonical_json": text[start:start + chunk_chars]}
        for index, start in enumerate(range(0, len(text), chunk_chars))]
    section["descriptor"] = write_section(store, section["name"], altered, lambda row: (row["position"],))
    assert section["descriptor"]["row_count"] == len(altered)
    assert list(iter_section(store, section["name"], section["descriptor"], lambda row: (row["position"],))) == altered
    original_record, original_baseline = PredictionPlan.record, evidence.training_baseline
    def substituted(run, slot, body=None):
        if slot == "frozen:2024" and body is None:
            substitutions.append(slot)
            return deepcopy(replaced)
        return original_record(run, slot, body)
    def rebuilt(*args):
        value = original_baseline(*args)
        native_baselines.append(args[-1])
        assert value["training"][0]["absolute_move_pct"] != baseline["training"][0]["absolute_move_pct"]
        return value
    PredictionPlan.record, evidence.training_baseline = substituted, rebuilt
original_path, original_bytes = objects.verify_object_path, objects.read_verified_bytes
def forbidden(*args, **kwargs):
    raise AssertionError("completed prediction replay attempted a write, fit or new target read")
def verify(store, ref):
    assert ref.object_id not in targets, "outer targets require bounded reads after all-horizon freeze"
    return original_path(store, ref)
def read_bytes(store, ref, *, max_bytes):
    if ref.object_id in targets:
        assert policy["mode"] == "fresh", "replay reopened outer target bytes"
        assert max_bytes == MAX_OUTCOME_OBJECT_BYTES
        frozen_record(store)
        opened.add(ref.object_id)
    return original_bytes(store, ref, max_bytes=max_bytes)
objects.verify_object_path, objects.read_verified_bytes = verify, read_bytes
if policy["mode"] != "fresh":
    cli.connect = forbidden
    ArtifactStore._commit = forbidden
    DistributionFits.fit = forbidden
    DistributionTrials.start = forbidden
    evaluation.evaluation_targets = forbidden
    original_connect = sqlite3.connect
    def readonly(target, *args, **kwargs):
        assert str(target).endswith("?mode=ro") and "immutable" not in str(target) and kwargs.get("uri") is True
        return original_connect(target, *args, **kwargs)
    sqlite3.connect = readonly
else:
    original_evaluation = evaluation.evaluation_targets
    def after_freeze(repository, binding, fold):
        frozen_record(repository._store)
        evaluation_reads.append(fold["year"])
        return original_evaluation(repository, binding, fold)
    evaluation.evaluation_targets = after_freeze
with ExitStack() as stack:
    if policy["mode"] != "fresh":
        stack.enter_context(no_fit_guard())
        stack.enter_context(legacy_guard())
    try:
        runpy.run_module("experiments.distribution_prediction_run", run_name="__main__")
    except SystemExit as exit:
        if exit.code == 0:
            assert opened == (targets if policy["mode"] == "fresh" else set())
            assert evaluation_reads == (["2024"] if policy["mode"] == "fresh" else [])
            for name in ("distribution_run", "distribution_run_plan", "distribution_run_request",
                         "distribution_pipeline", "distribution_market", "distribution_book", "distribution_payoff"):
                assert "experiments." + name not in sys.modules, name
            print("[native prediction test] verified " + policy["mode"] + " target boundary", file=sys.stderr)
        elif policy["mode"] == "substitution" and exit.code == 2:
            assert substitutions and set(native_baselines) == {1, 3, 5} and not opened
            print("[native prediction test] verified substitution target boundary", file=sys.stderr)
        raise
'''


SCRIPT = r'''
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import SystemClock, from_document, to_document
from engine.v2.research.magnitude_panel_qualified_table import recipe as feature_recipe
from engine.v2.research.progress import stage
from experiments import eligible_history, size_panel, size_search_binding
from experiments.distribution_candidate_recipe import candidate_recipe
from experiments.magnitude_search_plan import seal_search_plan
from experiments.magnitude_search_records import read_record
from experiments.magnitude_search_shards import iter_section
from experiments.native_registration import reserve_native_registration
from experiments.v2_magnitude_search_run import seal_search_run
from tests.v2.research.magnitude_trial_support import prediction_spec
from tests.v2.research.test_magnitude_panel_qualified import panel_data
from tests.v2.research.test_magnitude_search_request import request as search_request

root, child_script = Path(sys.argv[1]), sys.argv[2]
generator = panel_data.__wrapped__(root, SimpleNamespace(param={
    "days": ["2022-06-02", "2022-09-02", "2023-06-02", "2024-06-17"], "values": [2., 22., 6., 8.]}))
data = next(generator)
ids = [event["event_id"] for event in data.events]
history = eligible_history.materialize(data.repository, data.panel_source,
    **{**data.request, "limits": data.panel_request["limits"], "event_ids": ids})
panel_request = {**data.panel_request, "event_ids": ids, "history_snapshot": history["feature_snapshot"]}
panel = size_panel.materialize(data.repository, data.panel_source, **panel_request)
panel_request.update(feature_snapshot_id=panel["feature_snapshot"].snapshot_id,
    history_snapshot=history["feature_snapshot"].snapshot_id, price_history_snapshot=data.prices.snapshot_id)
source_request = search_request()
source_request.update(feature_dictionary_hash=feature_recipe("HMO")["dictionary_hash"],
    feature_snapshot_id=panel["feature_snapshot"].snapshot_id, history_snapshot_id=history["feature_snapshot"].snapshot_id,
    outer_years=["2024"], inner_years={"2024": ["2023"]}, max_fits=102)
source_request["eligibility_policy_ref"]["plan_id"] = panel["context"]["eligibility_plan"]["plan_id"]
source_request["limits"] = {"child": {"memory_bytes": 2 << 30, "cpu_seconds": 30, "wall_seconds": 60},
    "parent": {"memory_bytes": 3 << 30, "wall_seconds": 600},
    "build": {"memory_bytes": 3 << 30, "wall_seconds": 60}, "job": {"wall_seconds": 600}}
source_spec = prediction_spec(experiment_id="synthetic-prediction-source", folds=("2024",),
    runner="experiments/size_search.py")
source, source_spec = size_search_binding.prepare(data.repository, source_spec, source_request, panel_request,
    scope="panel-source", training_start="2022-01-01", labels_before="2025-01-01")
store = data.repository._store
assert source.document["schema_version"] == "native_experiment_registration.v1.7"
reserve_native_registration(data.conn, store, source, source_spec)
source_plan = seal_search_plan(data.conn, store, source, source_request)
source_ref = from_document(ArtifactRef, source_plan["plan_ref"])
seal_search_run(data.conn, store, data.repository, source, source_request, source_ref, clock=SystemClock())
targets = {record.object_ref.object_id for record in data.repository.fragment_records(data.panel_source, "computed_moves")
    if record.time_min == "2024-06-17"}
assert len(targets) == 1
assert "option_chains" not in data.panel_source.table_versions
spec = prediction_spec(experiment_id="synthetic-prediction-operator", folds=("2024",),
    runner="experiments/distribution_prediction_run.py", hypothesis="Compare frozen magnitude distributions.",
    primary_arm_id="raw_pair", arms=("raw_pair",))
assert spec.seed == 0 and not spec.input_files and not spec.economic_params
policy = {"schema_version": "distribution_prediction_request.v1", "max_events": 1,
    "limits": {"parent": {"memory_bytes": 3 << 30, "cpu_seconds": 600, "wall_seconds": 600},
        "job_wall_seconds": 600},
    "uncertainty": {"seed": 0, "resamples": 4, "block_days": 7, "block_origin": "2023-01-01", "confidence": .95}}
documents = {"spec": to_document(spec), "source-spec": to_document(source_spec), "source-request": source_request,
    "source-plan-ref": to_document(source_ref), "recipe": candidate_recipe("H"),
    "budget-caps": {"max_fits": 12, "max_trees": 49200, "child_limits": source_request["limits"]["child"]},
    "request": policy}
arguments = []
for name, document in documents.items():
    path = root / (name + ".json")
    path.write_text(json.dumps(document), encoding="utf-8")
    arguments.extend(("--" + name, str(path)))
catalog = Path(data.conn.execute("PRAGMA database_list").fetchone()[2])
arguments.extend(("--catalog", str(catalog), "--store-root", str(store.root), "--no-ledger"))
def invoke(action, *, mode=None, expected_status=0, catalog_path=None, store_path=None):
    command = [sys.executable, "-m", "experiments.distribution_prediction_run"]
    actual_arguments = list(arguments)
    if catalog_path is not None:
        actual_arguments[actual_arguments.index("--catalog") + 1] = str(catalog_path)
    if store_path is not None:
        actual_arguments[actual_arguments.index("--store-root") + 1] = str(store_path)
    if mode is not None:
        guard = {"mode": mode, "targets": sorted(targets), "catalog": str(catalog_path or catalog), "run_id": prepared["run_id"]}
        command = [sys.executable, "-c", child_script, json.dumps(guard)]
    with stage("native prediction CLI " + action + " " + (mode or "normal")):
        process = subprocess.run([*command, action, *actual_arguments], capture_output=True, text=True, timeout=660)
    assert process.returncode == expected_status, (action, mode, process.stdout, process.stderr)
    assert "distribution prediction " + action in process.stderr
    if mode is not None:
        assert "verified " + mode + " target boundary" in process.stderr
    return json.loads(process.stdout)
def rows(run_id=None):
    if run_id is not None:
        return [tuple(row) for row in data.conn.execute(
            "SELECT * FROM magnitude_search_records WHERE run_id=? ORDER BY slot", (run_id,))]
    return {name: [tuple(row) for row in data.conn.execute('SELECT * FROM "' + name + '"')]
        for name, in data.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
def objects_snapshot():
    return sorted((str(path.relative_to(store.root)), sha256(path.read_bytes()).hexdigest())
        for path in store.root.rglob("*") if path.is_file())
def database_snapshot():
    # Read-only WAL replay may change SHM reader marks, not durable main/WAL bytes.
    return {path.name: path.read_bytes() for path in catalog.parent.glob(catalog.name + "*")
        if path.is_file() and not path.name.endswith("-shm")}
def record(slot):
    return read_record(data.conn, store, prepared["run_id"], "distribution:prediction-" + slot)
def section_rows(section):
    return [{key: value for key, value in row.items() if key != "position"}
        for row in iter_section(store, section["name"], section["descriptor"], lambda row: (row["position"],))]

before, before_objects, source_records = rows(), objects_snapshot(), rows(source.run_id)
prepared = invoke("prepare")
assert prepared["registered"] is False and rows() == before and objects_snapshot() == before_objects
assert prepared["run_id"] != source.run_id and prepared["fit_budget"]["counts"]["total_fits"] == 12
assert invoke("register") == {**prepared, "registered": True}
preflight, sealed = invoke("preflight"), invoke("seal")
assert preflight == sealed and sealed["final_read_authority"] == "none" and sealed["event_count"] == 1
assert set(sealed["arms"]) == {"histogram", "same_mean", "matched_point", "training_empirical"}
assert data.conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
assert data.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
checkpointed = catalog.read_bytes()
result = invoke("run", mode="fresh")
assert result["schema_version"] == "distribution_prediction_run.v1" and result["run_plan_id"] == sealed["run_plan_id"]
assert result["promotion_authority"] == result["final_read_authority"] == "none"
assert {key: value for key, value in result.items() if key != "summary"} == record("completed")
assert len(result["groups"]) == 1 and set(result["reports"]) == set(result["summary"])
assert rows(source.run_id) == source_records
slots = [row[0] for row in data.conn.execute("SELECT slot FROM magnitude_search_records WHERE run_id=?", (prepared["run_id"],))]
assert sum(slot.startswith("distribution:fit-bound:") for slot in slots) == 6
assert sum(slot.startswith("distribution:fit-result:") for slot in slots) == 6
assert record("failed") is None
frozen, evaluated = record("frozen:2024"), record("evaluated:2024")
assert frozen["schema_version"] == "distribution_prediction_frozen.v2"
assert frozen["reference_encoding"] == "canonical_json_fragments.v1" and frozen["reference_chunk_chars"] == 65536
assert [entry["horizon"] for entry in frozen["horizons"]] == [1, 3, 5]
assert [entry["horizon"] for entry in evaluated["horizons"]] == [1, 3, 5]
scores, diagnostics = [], []
for frozen_horizon, evaluated_horizon in zip(frozen["horizons"], evaluated["horizons"]):
    forecasts = section_rows(frozen_horizon["forecasts"])
    fragments = section_rows(frozen_horizon["reference"])
    assert len(forecasts) == 1 and len(fragments) == frozen_horizon["reference"]["descriptor"]["row_count"]
    assert fragments and all(set(row) == {"canonical_json"} and 0 < len(row["canonical_json"]) <= frozen["reference_chunk_chars"]
        for row in fragments)
    reference = json.loads("".join(row["canonical_json"] for row in fragments))
    assert reference["receipt"]["status"] == "completed" and reference["tail"]["policy"] == "empirical_excess"
    assert {row["absolute_move_pct"] for row in reference["baseline"]["training"]} == {2., 22., 6.}
    scores.extend(section_rows(evaluated_horizon["scores"]))
    diagnostics.extend(section_rows(evaluated_horizon["diagnostics"]))
assert len(scores) == len(diagnostics) == 12
assert {(row["horizon"], row["arm"]) for row in scores} == {
    (horizon, arm) for horizon in (1, 3, 5) for arm in sealed["arms"]}
assert all(row["status"] == "available" and row["observation"] == 8. for row in scores + diagnostics)
assert all(math.isfinite(row["crps"]) and row["crps"] >= 0 for row in scores)
assert all(math.isclose(row["crps"], 26 / 9) for row in scores if row["arm"] == "training_empirical")
assert all(len(row["intervals"]) == 4 and 0 <= row["pit_lower"] <= row["pit_upper"] <= 1 for row in diagnostics)
assert set(result["summary"]) == {"forecast", "diagnostics"}
proper, coverage = result["summary"]["forecast"], result["summary"]["diagnostics"]
assert proper["schema_version"] == "distribution_prediction_score_summary.v1"
assert coverage["schema_version"] == "distribution_prediction_diagnostic_summary.v1"
assert all("rows" not in report and "events" not in report for report in (proper, coverage))
assert proper["primary"] == "h1_same_mean_minus_histogram_crps" and proper["uncertainty"]["resamples"] == 4
assert proper["population"] == coverage["population"] == 1
assert proper["calibration_identity"] == coverage["calibration_identity"] == "uncalibrated"
assert proper["final_read_authority"] == coverage["final_read_authority"] == "none"
assert coverage["pit_policy"] == "brackets_only_unrandomized"
assert proper["horizons"]["1"]["contrasts"]["same_mean"]["role"] == "primary"
for horizon in ("1", "3", "5"):
    assert set(proper["horizons"][horizon]["arms"]) == set(coverage["horizons"][horizon]["arms"]) == set(sealed["arms"])
    for arm in sealed["arms"]:
        summary = proper["horizons"][horizon]["arms"][arm]
        assert summary["population"] == 1 and summary["counts"]["available"] == 1
        assert all(tail["count"] == 1 and "mean_calibration_gap" in tail for tail in summary["tails"])
        intervals = coverage["horizons"][horizon]["arms"][arm]["intervals"]
        assert all(row["count"] == 1 and row["mean_width"] >= 0 and row["observed_coverage"] in (0, 1) for row in intervals)
for name, section in result["reports"].items():
    assert section_rows(section) == [result["summary"][name]]
def no_trading_outputs(value):
    if isinstance(value, dict):
        assert set(value).isdisjoint({"quote_snapshot_id", "bridges", "book", "books", "settled_net_profit", "net_delta_per_opportunity"})
        for child in value.values():
            no_trading_outputs(child)
    elif isinstance(value, list):
        for child in value:
            no_trading_outputs(child)
no_trading_outputs(result)

# The completion is committed in a live WAL while the fixture writer remains open.
assert catalog.read_bytes() == checkpointed
assert catalog.with_name(catalog.name + "-wal").stat().st_size > 0
before, before_objects, durable = rows(), objects_snapshot(), database_snapshot()
for action in ("collect", "report", "run"):
    assert invoke(action, mode="replay") == result
    assert rows() == before and objects_snapshot() == before_objects and database_snapshot() == durable
assert data.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
assert data.conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
data.conn.close()
durable = database_snapshot()
assert set(durable) == {catalog.name} and catalog.read_bytes()[18:20] == b"\x01\x01"
for action in ("collect", "report", "run"):
    assert invoke(action, mode="replay") == result
    assert database_snapshot() == durable and objects_snapshot() == before_objects
with tempfile.TemporaryDirectory(prefix="prediction-native-substitution-") as scratch:
    copied_catalog, copied_store = Path(scratch) / "catalog.sqlite", Path(scratch) / "store"
    shutil.copytree(store.root, copied_store)
    original = sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)
    copied = sqlite3.connect(copied_catalog)
    original.backup(copied)
    original.close()
    copied.close()
    copied_bytes = copied_catalog.read_bytes()
    refused = invoke("collect", mode="substitution", expected_status=2,
        catalog_path=copied_catalog, store_path=copied_store)
    assert refused["refused"]["code"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert copied_catalog.read_bytes() == copied_bytes
    assert database_snapshot() == durable and objects_snapshot() == before_objects
try:
    next(generator)
except StopIteration:
    pass
print(json.dumps({"status": "completed", "actions": 7, "horizons": 3, "arms": 4,
    "bounded_fit_pairs": 6, "replay_modes": ["wal", "delete"], "baseline_substitution": "refused",
    "run_id": prepared["run_id"]}))
'''


def test_native_prediction_operator_and_guarded_replay(tmp_path):
    ast.parse(CHILD)
    ast.parse(SCRIPT)
    # Forward the driver's bounded source/action stages for pytest -s. Each
    # actual CLI still captures stdout/stderr for exact JSON and failure checks.
    child = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path / "native-prediction"), CHILD],
        cwd=Path(__file__).resolve().parents[3], stdout=subprocess.PIPE, text=True, timeout=900,
        env={**os.environ, **dict.fromkeys(THREAD_VARIABLES, "1"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert child.returncode == 0, child.stdout  # Driver stderr is captured directly by pytest.
    result = json.loads(child.stdout.splitlines()[-1])
    assert result["status"] == "completed" and result["actions"] == 7
    assert result["horizons"] == 3 and result["arms"] == 4 and result["bounded_fit_pairs"] == 6
    assert result["replay_modes"] == ["wal", "delete"]
    assert result["baseline_substitution"] == "refused"
