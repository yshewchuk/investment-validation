"""P6 slice 10: smoke-mode experiments run as supervised v2 jobs."""
import csv
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops import cli, effects_graph, experiments, legacy_adapter, stages, supervisor, worker
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import experiment_plan
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.input_bindings import resolve_and_record
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog, run_until, sample

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _spec_document(**changes):
    document = {"experiment_id": "x", "hypothesis": "plumbing", "primary_arm_id": "fixture",
                "arms": ["fixture"], "seed": 7, "folds": ["fold-1"],
                "economic_params": {"fill": "mid"}, "price_source": "synthetic",
                "runner": "synthetic"}
    document.update(changes)
    return document


def test_experiment_kind_is_registered_with_experiment_heavy_profile():
    kind = stages.registry().get("experiment")
    assert kind.resource_classes == frozenset({"experiment_heavy"})
    assert kind.checkpoint_contract == "experiment_receipt.v1.0"
    with pytest.raises(OpsError, match="unsupported job kind"):
        stages.registry().get("bogus_experiment_kind")


def test_experiment_worker_runs_synthetic_runner_and_writes_receipt(tmp_path):
    (tmp_path / "spec.json").write_text(json.dumps(_spec_document(economic_params={})))
    result = worker.dispatch("experiment", {"expected_ids": ["experiment:x"],
                                            "runner": "synthetic", "no_ledger": True}, tmp_path)
    assert result["completed_ids"] == ["experiment:x"]
    assert result["outputs"][0]["name"] == "experiment_receipt"
    assert (tmp_path / "experiment_receipt.json").is_file()
    assert (tmp_path / "REPORT.md").is_file()

    broken = _spec_document()
    del broken["hypothesis"]
    (tmp_path / "spec.json").write_text(json.dumps(broken))
    with pytest.raises(OpsError, match="required field"):
        worker.dispatch("experiment", {"expected_ids": ["experiment:x"],
                                       "runner": "synthetic", "no_ledger": True}, tmp_path)


def test_worker_dispatch_refuses_a_malformed_fixed_arm_before_the_runner(tmp_path,
                                                                         monkeypatch):
    """The component contract's order, on the worker path: a fixed-arm run
    validates its one primary arm before any runner execution. Through the
    real dispatch on a synthetic staging directory -- document parsing and
    ``resolve_experiment_plan`` stay in the path -- a staged spec declaring
    two arms is the resolver's typed ``INVALID_EXPERIMENT_SPEC``, so the
    synthetic runner sentinel is never invoked and no report output exists."""
    (tmp_path / "spec.json").write_text(json.dumps(
        _spec_document(economic_params={}, arms=["fixture", "control"])))
    invoked = []

    def sentinel(*, run_dir, no_ledger):
        invoked.append(run_dir)

    monkeypatch.setattr(experiments, "synthetic_fixture_runner", sentinel)
    with pytest.raises(OpsError) as excinfo:
        worker.dispatch("experiment", {"expected_ids": ["experiment:x"],
                                       "runner": "synthetic", "no_ledger": True}, tmp_path)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not invoked
    assert not (tmp_path / "REPORT.md").exists()
    assert not (tmp_path / "experiment_receipt.json").exists()


def _planned_ledger(path, experiment_id="x", stage="planned"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted\n"
                    f"{experiment_id},deadbeef,2026-01-01,{stage},,,False\n")


def _ledger_rows(path):
    with open(path, newline="") as fh:
        return [{"id": row["id"], "stage": row["stage"]} for row in csv.DictReader(fh)]


def _primary_cli_checkout(tmp_path):
    """Copy the real EXP-182 wrapper and its audited source set to a tiny checkout."""
    import shutil

    from experiments import lib

    manifest = experiments.runner_manifest(REPO, REGISTERED_RUNNER)
    entry = experiments.RUNNER_INVENTORY[REGISTERED_RUNNER]
    declared = entry["declared_runtime_sources"]
    checkout = tmp_path / "checkout"
    relatives = dict.fromkeys(
        [manifest["runner"], manifest["spec_source"], *declared,
         *manifest["source_closure"]])
    for relative in relatives:
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    for relative in entry.get("declared_runtime_inputs", ()):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic runtime input")
    (checkout / declared[0]).write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "HERE = None\n"
        "def main():\n"
        "    assert '--no-ledger' in sys.argv\n"
        "    assert '--clock' in sys.argv and 'd1' in sys.argv\n"
        "    assert Path(HERE, 'spec.yaml').is_file()\n"
        "    trade_path = Path(HERE, 'experiments/EXP-179_execution_clock_d1_parity/results/trades/STR-THRU/d1_matched.parquet')\n"
        "    assert trade_path.is_file()\n"
        "    Path(HERE, 'REPORT.md').write_text(\n"
        "        '# EXP-182 staged report\\n\\n*Generated by engine.report v1.0.*\\n')\n")
    spec_path = checkout / manifest["spec_source"]
    lib.ledger_append([{"id": "EXP-182",
                        "spec_hash": lib.spec_hash(lib.load_spec(spec_path)),
                        "date": "2026-01-01", "stage": "planned",
                        "oos_mean_mid": "", "sharpe_trade": "",
                        "promoted": "False"}],
                      path=checkout / "experiments" / "LEDGER.csv")
    return checkout


def test_primary_cli_stages_registered_runner_and_publishes_root_report(
        tmp_path, monkeypatch, capsys):
    """A primary CLI run materializes the registered source set and publishes
    the wrapper-produced staging-root REPORT.md through the supervised worker."""
    checkout = _primary_cli_checkout(tmp_path)
    monkeypatch.setattr(experiments, "default_checkout_root", lambda: checkout)
    assert "primary" in stages.registry().get("experiment").namespaces
    ops = tmp_path / "ops"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document(
        experiment_id="EXP-182", primary_arm_id="d1", arms=["d1"],
        runner=REGISTERED_RUNNER, economic_params={})))

    assert cli.main(["--root", str(ops), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(ops), "plan", "experiment", "--spec", str(spec_path),
                     "--activate-ledger"]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(ops), "submit", "--plan", plan_ref,
                     "--idempotency-key", "cli-primary-staging"]) == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]

    clock = SystemClock()
    conn = open_catalog(ops / "catalog.sqlite", clock=clock)
    try:
        service = Service(conn, ops, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=checkout)
        try:
            service.start()
            assert run_until(service, conn, job_id, timeout=180) == "succeeded"
        finally:
            service.close()
        row = conn.execute("SELECT attempt_id, state, failure_json FROM attempts WHERE job_id=?",
                           (job_id,)).fetchone()
        assert row["state"] == "succeeded"
        assert row["failure_json"] is None
        assert conn.execute(
            "SELECT COUNT(*) FROM attempt_outputs WHERE attempt_id=? "
            "AND name='experiment_variant_report'", (row["attempt_id"],)
        ).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
        assert _ledger_rows(checkout / "experiments" / "LEDGER.csv") == [
            {"id": "EXP-182", "stage": "planned"},
            {"id": "EXP-182", "stage": "ran"}]
        report_evidence = json.loads(conn.execute(
            "SELECT evidence_json FROM experiment_runs").fetchone()[0])
        assert report_evidence["variant_id"]
        assert report_evidence["variants_tried"] == 1
    finally:
        conn.close()


def test_primary_runner_bindings_reject_parent_symlink_escape(tmp_path, monkeypatch):
    base = tmp_path / "checkout"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (outside / "run.py").write_text("external source")
    (outside / "spec.yaml").write_text("external spec")
    (base / "linked").symlink_to(outside, target_is_directory=True)
    manifest = {"runner": "linked/run.py", "spec_source": "linked/spec.yaml",
                "source_closure": []}
    monkeypatch.setattr(
        cli, "_registered_runner_manifest",
        lambda plan: ((), base, manifest))

    class CapturingStore:
        def __init__(self):
            self.published = []

        def publish_bytes(self, data, *, schema_ref):
            self.published.append((schema_ref, data))
            return data

    store = CapturingStore()
    with pytest.raises(OpsError) as excinfo:
        cli._primary_runner_bindings({}, store)
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert store.published == []


def test_primary_runner_bindings_publish_exp185_simulation_dependency(tmp_path):
    """The EXP-142 module and frozen EXP-144 population are declared and
    published with the primary runner's registered input bindings."""
    import shutil

    runner = "experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered/run.py"
    dependency = "experiments/EXP-142_str_runup_t14_factor_simulation_pnl_gate/simulation.py"
    population = "experiments/EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline/results/oos_scores.parquet"
    entry = experiments.RUNNER_INVENTORY[runner]
    assert dependency in entry["declared_runtime_sources"]
    assert population in entry["declared_runtime_inputs"]
    source_manifest = experiments.runner_manifest(REPO, runner)
    checkout = tmp_path / "checkout"
    for relative in dict.fromkeys(
            [runner, entry["spec_source"], *entry["declared_runtime_sources"],
             *source_manifest["source_closure"]]):
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    input_path = checkout / population
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"synthetic frozen population")

    class CapturingStore:
        def __init__(self):
            self.published = []

        def publish_bytes(self, data, *, schema_ref):
            self.published.append((schema_ref, data))
            return data

    store = CapturingStore()
    plan = {"kind": "experiment",
            "spec_document": _spec_document(experiment_id="EXP-185", runner=runner,
                                             economic_params={}),
            "parameters": {"runner": runner, "no_ledger": False},
            "preregistration_root": str(checkout)}
    bindings = dict(cli._primary_runner_bindings(plan, store))
    assert bindings[dependency] == (checkout / dependency).read_bytes()
    assert bindings[population] == b"synthetic frozen population"
    assert bindings["spec.yaml"] == (checkout / entry["spec_source"]).read_bytes()
    assert ("experiment_runner_input.v1.0", b"synthetic frozen population") in store.published


def test_activation_refuses_unregistered_experiment_ledger_untouched(tmp_path):
    conn, _, _ = catalog(tmp_path)
    try:
        experiments_dir = tmp_path / "experiments"
        experiments_dir.mkdir()
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps(_spec_document()))

        with pytest.raises(OpsError) as excinfo:
            experiment_plan(spec_path, smoke=False, root=tmp_path)
        assert excinfo.value.code == "INVALID_REQUEST"
        assert not (experiments_dir / "LEDGER.csv").exists()
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0

        ledger = experiments_dir / "LEDGER.csv"
        _planned_ledger(ledger, experiment_id="EXP-OTHER")
        before = ledger.read_bytes()
        with pytest.raises(OpsError):
            experiment_plan(spec_path, smoke=False, root=tmp_path)
        assert ledger.read_bytes() == before
    finally:
        conn.close()


def test_activation_succeeds_with_planned_row(tmp_path):
    _planned_ledger(tmp_path / "experiments" / "LEDGER.csv")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))

    plan = experiment_plan(spec_path, smoke=False, root=tmp_path)
    assert plan["mode"] == "primary"
    assert plan["parameters"]["no_ledger"] is False
    assert plan["parameters"]["runner"] == "synthetic"
    assert plan["parameters"]["expected_ids"] == ["experiment:x"]


REGISTERED_RUNNER = "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py"


def _registered_checkout(tmp_path, experiment_id="EXP-182"):
    """A tmp checkout with a registered runner, its legacy spec.yaml and a
    PLANNED ledger row whose spec_hash is the legacy ``experiments.lib`` hash."""
    from experiments import lib

    checkout = tmp_path / "checkout"
    runner = checkout / REGISTERED_RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text("if __name__ == '__main__':\n    pass\n")
    legacy_spec = runner.parent / "spec.yaml"
    legacy_spec.write_text("id: EXP-182\nprimary_spec:\n  x: 1\n")
    lib.ledger_append([{"id": experiment_id,
                        "spec_hash": lib.spec_hash(lib.load_spec(legacy_spec)),
                        "date": "2026-01-01", "stage": "planned",
                        "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}],
                      path=checkout / "experiments" / "LEDGER.csv")
    return checkout, legacy_spec


def test_plan_refuses_a_spec_edited_after_preregistration(tmp_path):
    """Review fix item 4: the registered runner's legacy spec.yaml must still
    hash to the PLANNED row's spec_hash at plan time."""
    checkout, legacy_spec = _registered_checkout(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document(
        experiment_id="EXP-182", runner=REGISTERED_RUNNER)))

    plan = experiment_plan(spec_path, smoke=False, root=checkout)
    assert plan["mode"] == "primary"
    assert plan["preregistration_root"] == str(checkout)

    legacy_spec.write_text("id: EXP-182\nprimary_spec:\n  x: 2\n")
    with pytest.raises(OpsError) as excinfo:
        experiment_plan(spec_path, smoke=False, root=checkout)
    assert excinfo.value.code == "SPEC_CHANGED"


def test_submit_refuses_a_spec_edited_after_preregistration(tmp_path, monkeypatch, capsys):
    """Review fix item 4: the same binding is recomputed at submit, from the
    checkout root the plan recorded."""
    checkout, legacy_spec = _registered_checkout(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document(
        experiment_id="EXP-182", runner=REGISTERED_RUNNER)))
    monkeypatch.setattr(experiments, "default_checkout_root", lambda: checkout)

    ops = tmp_path / "ops"
    assert cli.main(["--root", str(ops), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(ops), "plan", "experiment", "--spec", str(spec_path),
                     "--activate-ledger"]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]

    legacy_spec.write_text("id: EXP-182\nprimary_spec:\n  x: 2\n")
    assert cli.main(["--root", str(ops), "submit", "--plan", plan_ref,
                     "--idempotency-key", "k"]) == 2
    problem = json.loads(capsys.readouterr().out)
    assert problem["code"] == "SPEC_CHANGED"


def test_cli_activate_ledger_and_no_ledger_are_mutually_exclusive(tmp_path, capsys):
    assert cli.main(["--root", str(tmp_path), "init"]) == 0
    capsys.readouterr()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))

    assert cli.main(["--root", str(tmp_path), "plan", "experiment",
                     "--spec", str(spec_path), "--no-ledger", "--activate-ledger"]) == 2
    problem = json.loads(capsys.readouterr().out)
    assert problem["code"] == "INVALID_REQUEST"


def test_cli_experiment_plan_without_either_flag_still_refuses(tmp_path, capsys):
    """Slice 10 callers who omit both flags keep their old refusal; the real
    activation path is the explicit --activate-ledger opt-in only."""
    assert cli.main(["--root", str(tmp_path), "init"]) == 0
    capsys.readouterr()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))

    assert cli.main(["--root", str(tmp_path), "plan", "experiment",
                     "--spec", str(spec_path)]) == 2
    problem = json.loads(capsys.readouterr().out)
    assert problem["code"] == "INVALID_REQUEST"
    assert "production experiment activation is disabled" in problem["message"]


def test_worker_dispatch_primary_mode_never_grants_runner_ledger_writes(tmp_path,
                                                                        monkeypatch):
    from experiments import lib

    runner_id = "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py"
    runner_path = tmp_path / runner_id
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("if __name__ == '__main__':\n    pass\n")
    legacy_spec = runner_path.parent / "spec.yaml"
    legacy_spec.write_text("id: EXP-182\n")
    document = _spec_document(runner=runner_id, economic_params={},
                              primary_arm_id="d1", arms=["d1"])
    (tmp_path / "spec.json").write_text(json.dumps(document))
    registered_identity = experiments.legacy_spec_hash(lib.load_spec(legacy_spec))
    resolved_spec_hash = experiments.experiment_spec_from_document(document).spec_hash

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (tmp_path / "REPORT.md").write_text(
            "# Synthetic infrastructure report\n\n*Generated by engine.report v1.0.*\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = worker.dispatch("experiment",
                             {"expected_ids": ["experiment:x"], "runner": runner_id,
                              "no_ledger": False}, tmp_path)
    assert result["completed_ids"] == ["experiment:x"]
    assert len(commands) == 1
    assert "--no-ledger" in commands[0]
    assert "--clock" in commands[0]
    assert commands[0][commands[0].index("--clock") + 1] == "d1"
    assert "both" not in commands[0]

    report = (tmp_path / "REPORT.md").read_text()
    assert f"Variant ID: {registered_identity}\n" in report
    assert "Variants tried: 1\n" in report
    assert resolved_spec_hash not in report
    variant_report = [output for output in result["outputs"]
                      if output["name"] == "experiment_variant_report"]
    assert variant_report == [{"name": "experiment_variant_report", "path": "REPORT.md",
                              "schema": "experiment_variant_report.v1.0"}]


def test_worker_refuses_a_legacy_runner_that_exits_nonzero_after_the_report(tmp_path):
    """Review fix item 3: a registered runner that writes REPORT.md and then
    exits 1 is a typed failure -- never a success with whatever report it
    happened to leave behind -- and its stderr tail travels in the details."""
    runner_id = "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py"
    runner_path = tmp_path / runner_id
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "Path('REPORT.md').write_text('# half a report\\n\\n"
        "*Generated by engine.report v1.0.*\\n')\n"
        "print('runner exploded', file=sys.stderr)\n"
        "sys.exit(1)\n")
    (runner_path.parent / "spec.yaml").write_text("id: EXP-182\n")
    (tmp_path / "spec.json").write_text(
        json.dumps(_spec_document(runner=runner_id, economic_params={},
                                  primary_arm_id="d1", arms=["d1"])))

    with pytest.raises(OpsError) as excinfo:
        worker.dispatch("experiment",
                        {"expected_ids": ["experiment:x"], "runner": runner_id,
                         "no_ledger": True}, tmp_path)
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["returncode"] == 1
    assert "runner exploded" in excinfo.value.problem.details["stderr_tail"]
    assert not (tmp_path / "experiments" / "LEDGER.csv").exists(), \
        "the worker never appends a ran row"


FIXED_ARM_RUNNER_SOURCE = r"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--clock", choices=("d0", "d1", "both"), default="both")
parser.add_argument("--no-ledger", action="store_true")
parsed = parser.parse_args()
if not parsed.no_ledger:
    raise SystemExit("ledger writes must be disabled")
Path("REPORT.md").write_text(
    "# EXP-182 fixed-arm subprocess regression\n\n"
    "*Generated by engine.report v1.0.*\n\n"
    f"Clocks evaluated: {parsed.clock}\n"
    "Ledger writes: disabled\n")
print("done")
"""


def test_worker_dispatch_sends_the_audited_clock_selector_to_a_real_subprocess(tmp_path):
    """Subprocess-boundary regression: the worker dispatch runs the real
    ``run_legacy_script`` subprocess boundary with a small synthetic argparse
    runner installed at the EXP-182 registered entrypoint path. It verifies
    the adapter forwards the audited ``--clock d1`` selector and
    ``--no-ledger``, that the synthetic runner evaluates only d1, and that
    the worker records one variant identity/count without creating a
    ledger."""
    from experiments import lib

    runner_id = REGISTERED_RUNNER
    runner_path = tmp_path / runner_id
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text(FIXED_ARM_RUNNER_SOURCE)
    legacy_spec = runner_path.parent / "spec.yaml"
    legacy_spec.write_text("id: EXP-182\n")
    document = _spec_document(runner=runner_id, economic_params={},
                              primary_arm_id="d1", arms=["d1"])
    (tmp_path / "spec.json").write_text(json.dumps(document))
    registered_identity = experiments.legacy_spec_hash(lib.load_spec(legacy_spec))

    result = worker.dispatch("experiment",
                             {"expected_ids": ["experiment:x"], "runner": runner_id,
                              "no_ledger": False}, tmp_path)
    assert result["completed_ids"] == ["experiment:x"]
    report = (tmp_path / "REPORT.md").read_text()
    assert [line for line in report.splitlines()
            if line.startswith("Clocks evaluated:")] == ["Clocks evaluated: d1"]
    assert "Clocks evaluated: both" not in report
    assert "Ledger writes: disabled" in report
    assert f"Variant ID: {registered_identity}\n" in report
    assert "Variants tried: 1\n" in report
    receipt = json.loads((tmp_path / "experiment_receipt.json").read_text())
    assert receipt["evidence"]["variants_tried"] == 1
    assert receipt["evidence"]["variant_id"] == registered_identity
    assert not (tmp_path / "experiments" / "LEDGER.csv").exists()


@pytest.mark.parametrize("runner_id", [
    "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py",
    "experiments/EXP-184_str_thru_gate_promotion_confirmatory_val_registered/run.py",
    "experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered/run.py",
])
def test_registered_wrappers_separate_data_root_from_adapter_staging(tmp_path, runner_id):
    """A general data-root override keeps wrapper outputs experiment-relative;
    only the adapter's pinned-source signal selects the staging root."""
    import os
    import subprocess
    import sys

    from engine.v2.ops.experiments import RUNNER_INVENTORY

    wrapper = Path(__file__).resolve().parents[1] / runner_id
    source_relative = RUNNER_INVENTORY[runner_id]["declared_runtime_sources"][0]
    captured_here = tmp_path / "wrapper_here.txt"
    synthetic_source = (
        "import os\n"
        "from pathlib import Path\n"
        "HERE = None\n"
        "def main():\n"
        "    Path(os.environ['WRAPPER_HERE_CAPTURE']).write_text(str(HERE))\n"
    )

    for staged in (False, True):
        root = tmp_path / ("staged" if staged else "data-root")
        root.mkdir()
        source = root / source_relative
        source.parent.mkdir(parents=True)
        source.write_text(synthetic_source)
        env = dict(os.environ, INVESTING_PLAN_ROOT=str(root),
                   WRAPPER_HERE_CAPTURE=str(captured_here))
        if staged:
            env["INVESTING_PLAN_PINNED_SOURCE"] = str(source)
        else:
            env.pop("INVESTING_PLAN_PINNED_SOURCE", None)

        completed = subprocess.run([sys.executable, str(wrapper)], cwd=root, env=env,
                                   capture_output=True, text=True, check=False)

        assert completed.returncode == 0, completed.stderr
        expected_here = root if staged else wrapper.parent
        assert captured_here.read_text() == str(expected_here.resolve())


@pytest.mark.parametrize(("runner_id", "primary_arm_id"), [
    ("experiments/EXP-184_str_thru_gate_promotion_confirmatory_val_registered/run.py",
     "gate_midfill_str_thru_forecast_analog"),
    ("experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered/run.py",
     "native_nan"),
])
def test_worker_dispatch_runs_registered_no_argument_primary_arms(
        tmp_path, monkeypatch, runner_id, primary_arm_id):
    """Each registered primary selector reaches the runner adapter as an
    empty tuple and writes its report in the supplied staging root."""
    from types import SimpleNamespace

    run_dir = tmp_path / primary_arm_id
    run_dir.mkdir()
    calls = []

    def adapter(staging_root, called_runner_id, *, args, declared_runtime_sources):
        calls.append((called_runner_id, args))
        Path(staging_root, "REPORT.md").write_text("# Registered primary report\n")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(legacy_adapter, "run_legacy_script", adapter)
    runner = worker._registered_experiment_runner(tmp_path, runner_id, primary_arm_id)
    result = runner(run_dir=run_dir, no_ledger=True)

    assert calls == [(runner_id, ())]
    assert result["returncode"] == 0
    assert (run_dir / "REPORT.md").read_text() == "# Registered primary report\n"


def test_worker_dispatch_refuses_a_registered_arm_without_an_audited_selector(
        tmp_path, monkeypatch):
    """Fail closed: a registered runner with no audited selector for the
    dispatched primary arm is the non-retryable ``INVALID_EXPERIMENT_SPEC``
    refusal before the resolved plan is written or any runner could be
    invoked -- even one replaced by a sentinel here. EXP-184 declares no arm
    selectors at all, and EXP-182's registered spec.yaml hashes its D-1
    primary declaration, so the selector map carries only the registered
    primary ``d1``: a ``d0`` dispatch would label a D0 result with EXP-182's
    D1 identity and must fail closed too."""
    cases = [
        ("experiments/EXP-184_str_thru_gate_promotion_confirmatory_val_registered/run.py",
         "fixture"),
        (REGISTERED_RUNNER, "d0"),
    ]
    for runner_id, primary_arm_id in cases:
        staging = tmp_path / primary_arm_id
        staging.mkdir()
        (staging / "spec.json").write_text(json.dumps(
            _spec_document(runner=runner_id, economic_params={},
                           primary_arm_id=primary_arm_id, arms=[primary_arm_id])))
        invoked = []

        def sentinel(*args, **kwargs):
            invoked.append((args, kwargs))

        monkeypatch.setattr(legacy_adapter, "run_legacy_script", sentinel)
        with pytest.raises(OpsError) as excinfo:
            worker.dispatch("experiment", {"expected_ids": ["experiment:x"],
                                           "runner": runner_id, "no_ledger": True},
                            staging)
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
        assert excinfo.value.problem.retryable is False
        assert excinfo.value.problem.details["runner"] == runner_id
        assert excinfo.value.problem.details["primary_arm_id"] == primary_arm_id
        assert not invoked
        assert not (staging / "resolved_experiment_plan.json").exists()
        assert not (staging / "REPORT.md").exists()
        assert not (staging / "experiment_receipt.json").exists()


def test_experiment_effect_appends_ledger_row_once_in_the_checkout_only(tmp_path,
                                                                        monkeypatch):
    """Review fix item 1: the ops root and the checkout are DIFFERENT tmp
    dirs, and the ran row lands in the checkout ledger only -- never beside
    the catalog. ``store_root`` is the checkout; the Service's ``code_source``
    stays the real repo so the launch manifest still matches."""
    ops_root, checkout = tmp_path / "ops", tmp_path / "checkout"
    ledger = checkout / "experiments" / "LEDGER.csv"
    _planned_ledger(ledger)
    ops_root.mkdir()
    conn, clock, _ = catalog(ops_root)
    calls = []
    real_effect = effects_graph.experiment_effect

    def spy(conn, store, claim, refs, *, clock, code_source, store_root=None):
        calls.append((conn, store, claim, refs, clock, code_source, store_root))
        return real_effect(conn, store, claim, refs, clock=clock, code_source=code_source,
                           store_root=store_root)

    monkeypatch.setattr(supervisor, "experiment_effect", spy)
    try:
        job = _submit_experiment(conn, ops_root, clock, "primary-1", no_ledger=False,
                                 preregistration_root=checkout)
        service = Service(conn, ops_root, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=checkout)
        try:
            service.start()
            assert run_until(service, conn, job.job_id, timeout=90) == "succeeded"
        finally:
            service.close()
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "planned"},
                                        {"id": "x", "stage": "ran"}]
        identity = experiments.expected_variant_identity(
            checkout,
            experiments.experiment_spec_from_document(_spec_document(economic_params={})),
            "primary")
        evidence = json.loads(conn.execute(
            "SELECT evidence_json FROM experiment_runs").fetchone()[0])
        assert evidence["variant_id"] == identity
        assert evidence["variants_tried"] == 1
        with open(ledger, newline="") as fh:
            ran_rows = [row for row in csv.DictReader(fh) if row["stage"] == "ran"]
        assert len(ran_rows) == 1
        assert ran_rows[0]["spec_hash"] == identity
        assert not (ops_root / "experiments").exists(), \
            "the ledger is never written beside the operations catalog"

        conn_, store, claim, refs, clock_, code_source, store_root = calls[-1]
        assert store_root == checkout
        effect, extra_refs = real_effect(conn_, store, claim, refs, clock=clock_,
                                         code_source=code_source, store_root=store_root)
        assert extra_refs == ()
        for _ in range(2):  # two retries: still exactly one row
            with transaction(conn_):
                effect(conn_)
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "planned"},
                                        {"id": "x", "stage": "ran"}]
    finally:
        conn.close()


_DEFAULT_EVIDENCE = object()


def _claimed_primary_effect(tmp_path, key="primary-fence", *, document=None, checkout=None,
                            receipt_evidence=_DEFAULT_EVIDENCE):
    """A real claimed attempt + the coordinator's commit closure, plus the
    checkout ledger it appends to. Returns ``(conn, clock, claim, effect, checkout)``.
    The synthetic receipt's ``evidence`` defaults to the valid primary variant
    identity the run's bound spec resolves to; an explicit value is placed
    verbatim, so a refusal-shaped receipt can be staged. With no ``document``
    or ``checkout`` the synthetic spec and its PLANNED-only checkout are kept;
    pass a ``_registered_checkout`` pair and its document to exercise the
    registered identity branch. The default evidence is always calculated
    from the exact document and checkout the helper submits."""
    ops_root = tmp_path / "ops"
    if checkout is None:
        checkout = tmp_path / "checkout"
        _planned_ledger(checkout / "experiments" / "LEDGER.csv")
    if document is None:
        document = _spec_document()
    ops_root.mkdir()
    conn, clock, supervisor_ = catalog(ops_root)
    _submit_experiment(conn, ops_root, clock, key, document=document, no_ledger=False,
                       preregistration_root=checkout)
    claim = claim_next(conn, policy=TEST_POLICY, sample=sample(clock),
                       supervisor=supervisor_, clock=clock, registry=stages.registry())
    assert claim is not None
    store = ArtifactStore(ops_root)
    resolve_and_record(conn, store, claim)
    if receipt_evidence is _DEFAULT_EVIDENCE:
        spec = experiments.experiment_spec_from_document(document)
        receipt_evidence = {"variant_id": experiments.expected_variant_identity(
            checkout, spec, "primary"), "variants_tried": 1}
    receipt_ref = store.publish_bytes(
        json.dumps({"input_hash": "input-" + key, "evidence": receipt_evidence}).encode(),
        schema_ref="experiment_receipt.v1.0")
    with transaction(conn):
        register_artifact(conn, receipt_ref, None, clock)
    effect, extra_refs = effects_graph.experiment_effect(
        conn, store, claim, [("experiment_receipt", receipt_ref)], clock=clock,
        code_source=REPO, store_root=checkout)
    assert extra_refs == ()
    return conn, clock, claim, effect, checkout


def test_experiment_effect_lost_fence_appends_no_ran_row(tmp_path):
    """Review fix item 2b: a commit refused before its effects run (a stale
    fence) never appends the ran row."""
    conn, clock, claim, effect, checkout = _claimed_primary_effect(tmp_path)
    try:
        with pytest.raises(OpsError) as excinfo:
            commit_attempt(conn, claim.attempt_id, claim.fence + 1,
                           Outcome(True, "verified_dead", 0), clock=clock,
                           effects=lambda txn: effect(txn))
        assert excinfo.value.code == "LEASE_LOST"
        assert _ledger_rows(checkout / "experiments" / "LEDGER.csv") == [
            {"id": "x", "stage": "planned"}]
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
    finally:
        conn.close()


def test_experiment_effect_retry_after_crash_appends_exactly_one_row(tmp_path, monkeypatch):
    """Review fix item 2a/2c: a crash between register and append rolls the
    whole commit back; the retry appends exactly one row, and two more retries
    do not append another."""
    conn, clock, claim, effect, checkout = _claimed_primary_effect(tmp_path, key="primary-crash")
    ledger = checkout / "experiments" / "LEDGER.csv"
    real_append = effects_graph._append_ledger_row
    crashed = []

    def flaky(txn, checkout_root, spec, receipt, *, run_id, variant_id):
        if not crashed:
            crashed.append(True)
            raise RuntimeError("crash after register, before append")
        return real_append(txn, checkout_root, spec, receipt, run_id=run_id,
                           variant_id=variant_id)

    monkeypatch.setattr(effects_graph, "_append_ledger_row", flaky)
    try:
        with pytest.raises(RuntimeError):
            commit_attempt(conn, claim.attempt_id, claim.fence,
                           Outcome(True, "verified_dead", 0), clock=clock,
                           effects=lambda txn: effect(txn))
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "planned"}]
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0

        monkeypatch.setattr(effects_graph, "_append_ledger_row", real_append)
        commit_attempt(conn, claim.attempt_id, claim.fence,
                       Outcome(True, "verified_dead", 0), clock=clock,
                       effects=lambda txn: effect(txn))
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "planned"},
                                        {"id": "x", "stage": "ran"}]
        with transaction(conn):
            effect(conn)
        with transaction(conn):
            effect(conn)
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "planned"},
                                        {"id": "x", "stage": "ran"}]
    finally:
        conn.close()


def test_primary_effect_replay_refuses_conflicting_stored_variant_evidence(tmp_path):
    """A durable primary run's variant evidence is immutable: after one real
    successful commit created the run row and the ran ledger row, replaying
    the same effect/receipt against stored evidence that a later attempt
    corrupted through SQL is the non-retryable ``INVALID_EXPERIMENT_SPEC``
    refusal -- first for a conflicting stored ID (count 1), then for a
    conflicting stored count (same ID, count 2) -- the conflicting stored
    evidence is never overwritten, and the ran ledger row keeps the original
    identity throughout."""
    conn, clock, claim, effect, checkout = _claimed_primary_effect(tmp_path,
                                                                   key="primary-replay")
    ledger = checkout / "experiments" / "LEDGER.csv"
    run_id = claim.attempt_id

    def stored_evidence():
        row = conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                           (run_id,)).fetchone()
        return json.loads(row[0])

    def ran_rows():
        with open(ledger, newline="") as fh:
            return [row for row in csv.DictReader(fh) if row["stage"] == "ran"]

    try:
        commit_attempt(conn, claim.attempt_id, claim.fence,
                       Outcome(True, "verified_dead", 0), clock=clock,
                       effects=lambda txn: effect(txn))
        assert len(ran_rows()) == 1
        original_id = ran_rows()[0]["spec_hash"]
        assert stored_evidence()["variant_id"] == original_id
        assert stored_evidence()["variants_tried"] == 1

        # First replay: a conflicting stored variant ID (count 1), written
        # through SQL only now that the real commit created the row.
        with transaction(conn):
            conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                         (json.dumps({**stored_evidence(), "variant_id": "foreign-variant"},
                                     sort_keys=True), run_id))
        with pytest.raises(OpsError) as excinfo:
            with transaction(conn):
                effect(conn)
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
        assert excinfo.value.problem.retryable is False
        assert stored_evidence()["variant_id"] == "foreign-variant"
        assert stored_evidence()["variants_tried"] == 1
        assert [row["spec_hash"] for row in ran_rows()] == [original_id]

        # Second replay: the original incoming ID, but a conflicting count.
        with transaction(conn):
            conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                         (json.dumps({**stored_evidence(), "variant_id": original_id,
                                      "variants_tried": 2}, sort_keys=True), run_id))
        with pytest.raises(OpsError) as excinfo:
            with transaction(conn):
                effect(conn)
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
        assert excinfo.value.problem.retryable is False
        assert stored_evidence()["variant_id"] == original_id
        assert stored_evidence()["variants_tried"] == 2
        assert [row["spec_hash"] for row in ran_rows()] == [original_id]
    finally:
        conn.close()


def test_primary_effect_replay_refuses_historical_ran_row_identity_conflict(tmp_path):
    """A legacy primary run whose durable evidence predates the variant
    fields, beside a ran ledger row written under a DIFFERENT historical
    identity, can never adopt the incoming identity: the replay's evidence
    backfill rolls back with the non-retryable ``INVALID_EXPERIMENT_SPEC``
    and the historical ledger bytes stay untouched."""
    conn, clock, claim, effect, checkout = _claimed_primary_effect(
        tmp_path, key="primary-legacy-ran")
    ledger = checkout / "experiments" / "LEDGER.csv"
    run_id = claim.attempt_id

    def stored_evidence():
        row = conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                           (run_id,)).fetchone()
        return json.loads(row[0])

    try:
        commit_attempt(conn, claim.attempt_id, claim.fence,
                       Outcome(True, "verified_dead", 0), clock=clock,
                       effects=lambda txn: effect(txn))
        assert conn.execute(
            "SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
        with open(ledger, newline="") as fh:
            ran_rows = [row for row in csv.DictReader(fh) if row["stage"] == "ran"]
        historical = ran_rows[0]["spec_hash"]
        assert stored_evidence()["variant_id"] == historical
        assert stored_evidence()["variants_tried"] == 1

        # Historical durable state, written through SQL and the file only now
        # that the real commit created it: the evidence loses ONLY its two
        # variant fields and the ran row's spec_hash becomes a foreign one.
        raw = ledger.read_text()
        assert raw.count(historical) == 1
        with transaction(conn):
            conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                         (json.dumps({key: value for key, value in stored_evidence().items()
                                      if key not in ("variant_id", "variants_tried")},
                                     sort_keys=True), run_id))
            ledger.write_text(raw.replace(historical, "historical-spec", 1))
        historical_bytes = ledger.read_bytes()

        with pytest.raises(OpsError) as excinfo:
            with transaction(conn):
                effect(conn)
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
        assert excinfo.value.problem.retryable is False
        assert ledger.read_bytes() == historical_bytes
        after = stored_evidence()
        assert "variant_id" not in after and "variants_tried" not in after
    finally:
        conn.close()


@pytest.mark.parametrize("build_evidence", (
    pytest.param(lambda vid: None, id="none-evidence"),
    pytest.param(lambda vid: [], id="list-evidence"),
    pytest.param(lambda vid: {"variants_tried": 1}, id="missing-variant-id"),
    pytest.param(lambda vid: {"variant_id": "not-the-run-variant", "variants_tried": 1},
                 id="wrong-variant-id"),
    pytest.param(lambda vid: {"variant_id": vid}, id="missing-variants-tried"),
    pytest.param(lambda vid: {"variant_id": vid, "variants_tried": 0},
                 id="zero-variants-tried"),
    pytest.param(lambda vid: {"variant_id": vid, "variants_tried": True},
                 id="boolean-variants-tried"),
))
def test_effect_refuses_receipt_evidence_that_declares_no_valid_variant(tmp_path,
                                                                        build_evidence):
    """The new effect contract: a primary commit whose receipt carries no, a
    foreign, or a miscounted variant identity is refused inside the fenced
    transaction as the non-retryable ``INVALID_EXPERIMENT_SPEC`` -- before the
    run is registered or any ledger row appended -- so only the pre-seeded
    planned row survives and no durable run or hypothesis row exists."""
    expected = experiments.expected_variant_identity(
        tmp_path / "checkout",
        experiments.experiment_spec_from_document(_spec_document()), "primary")
    conn, clock, claim, effect, checkout = _claimed_primary_effect(
        tmp_path, key="primary-evidence", receipt_evidence=build_evidence(expected))
    try:
        with pytest.raises(OpsError) as excinfo:
            commit_attempt(conn, claim.attempt_id, claim.fence,
                           Outcome(True, "verified_dead", 0), clock=clock,
                           effects=lambda txn: effect(txn))
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
        assert excinfo.value.problem.retryable is False
        assert _ledger_rows(checkout / "experiments" / "LEDGER.csv") == [
            {"id": "x", "stage": "planned"}]
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
    finally:
        conn.close()


def test_registered_primary_effect_persists_registered_variant_identity(tmp_path, monkeypatch):
    """The registered branch end to end: a primary commit through a
    registered runner binds the legacy spec.yaml identity its PLANNED row
    preregistered -- in the durable evidence and in the checkout's ran row."""
    from experiments import lib

    checkout, legacy_spec = _registered_checkout(tmp_path)
    document = _spec_document(experiment_id="EXP-182", runner=REGISTERED_RUNNER,
                              economic_params={})
    spec = experiments.experiment_spec_from_document(document)
    registered = experiments.legacy_spec_hash(lib.load_spec(legacy_spec))
    assert registered == experiments.expected_variant_identity(checkout, spec, "primary")

    lookups = []

    def shifting_identity(checkout_root, spec_for_lookup, mode):
        lookups.append(mode)
        return registered if len(lookups) <= 2 else registered + "-changed"

    monkeypatch.setattr(experiments, "expected_variant_identity", shifting_identity)
    conn, clock, claim, effect, checkout = _claimed_primary_effect(
        tmp_path, document=document, checkout=checkout)
    try:
        commit_attempt(conn, claim.attempt_id, claim.fence,
                       Outcome(True, "verified_dead", 0), clock=clock,
                       effects=lambda txn: effect(txn))
        assert lookups == ["primary", "primary"]
        evidence = json.loads(conn.execute(
            "SELECT evidence_json FROM experiment_runs").fetchone()[0])
        assert evidence["variant_id"] == registered
        assert evidence["variants_tried"] == 1
        with open(checkout / "experiments" / "LEDGER.csv", newline="") as fh:
            ran_rows = [row for row in csv.DictReader(fh) if row["stage"] == "ran"]
        assert len(ran_rows) == 1
        assert ran_rows[0]["spec_hash"] == registered
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
    finally:
        conn.close()


def test_experiment_plan_names_a_runner_and_experiment_heavy_profile(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))
    plan = experiment_plan(spec_path, smoke=True)
    assert plan["kind"] == "experiment"
    assert plan["resource_class"] == "experiment_heavy"
    assert plan["effects"] == ["staged"]
    assert plan["parameters"]["runner"] == "synthetic"
    assert plan["parameters"]["no_ledger"] is True

    broken = _spec_document()
    del broken["runner"]
    spec_path.write_text(json.dumps(broken))
    with pytest.raises(OpsError, match="runner"):
        experiment_plan(spec_path, smoke=True)


def _submit_experiment(conn, root, clock, key, *, document=None, no_ledger=True,
                       preregistration_root=None):
    document = document or _spec_document(economic_params={})
    store = ArtifactStore(root)
    spec_ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(),
                                   schema_ref="experiment_spec.v1.0")
    with transaction(conn):
        register_artifact(conn, spec_ref, None, clock)
    profile = profile_named(DEFAULT_POLICY, "experiment_heavy")
    parameters = {"expected_ids": ["experiment:x"],
                  "input_bindings": {"spec.json": spec_ref.artifact_id},
                  "runner": "synthetic", "no_ledger": no_ledger}
    if preregistration_root is not None:
        parameters["preregistration_root"] = str(preregistration_root)
    job = JobSpec(
        kind="experiment",
        implementation_ref=content_hash(worker_source_manifest(REPO)),
        spec_hash=content_hash(document),
        environment_ref=content_hash(
            environment_identity(profile.thread_count or profile.cpu_count)),
        parameters=parameters,
        input_refs=(spec_ref.artifact_id,), output_namespace="shadow",
        resource_class="experiment_heavy", retry_policy_ref="bounded",
        checkpoint_contract_ref="experiment_receipt.v1.0")
    return submit(conn, stages.registry(), POLICY,
                  SubmitRequest(namespace="shadow", idempotency_key=key,
                                principal="operator", job=job), clock=clock)


def test_experiment_effect_registers_a_durable_run(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    try:
        first = _submit_experiment(conn, tmp_path, clock, "experiment-1")
        service = Service(conn, tmp_path, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO)
        try:
            service.start()
            assert run_until(service, conn, first.job_id, timeout=90) == "succeeded"
            attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                      (first.job_id,)).fetchone()[0]
            assert conn.execute("SELECT COUNT(*) FROM experiment_runs WHERE run_id=?",
                                (attempt_id,)).fetchone()[0] == 1
            # Smoke mode holds no promotion authority: no hypotheses row.
            assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0

            second = _submit_experiment(conn, tmp_path, clock, "experiment-2")
            assert run_until(service, conn, second.job_id, timeout=90) == "succeeded"
        finally:
            service.close()
        # Same spec_hash/input_hash: register_hypothesis' early return keeps
        # exactly one durable run row.
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
    finally:
        conn.close()


def _cli_plan_and_submit(root, spec_path, capsys, key="cli-experiment-1"):
    """The operator's real path: ``ops plan experiment`` then ``ops submit``."""
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "experiment",
                     "--spec", str(spec_path), "--no-ledger"]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", key]) == 0
    return json.loads(capsys.readouterr().out)["job_id"]


def test_cli_experiment_plan_and_submit_run_under_the_planned_profile(tmp_path, capsys):
    """End-to-end through the real CLI: the plan's ``environment_ref`` must
    match what ``Service._launch`` fingerprints for the ``experiment_heavy``
    profile the job actually runs under (the bug this pins: the plan used the
    hard-coded 1-thread identity and every CLI submission was refused)."""
    root = tmp_path / "ops"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document(economic_params={})))
    job_id = _cli_plan_and_submit(root, spec_path, capsys)

    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        service = Service(conn, root, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO)
        try:
            service.start()
            assert run_until(service, conn, job_id, timeout=90) == "succeeded"
        finally:
            service.close()
        row = conn.execute("SELECT state, failure_json FROM attempts WHERE job_id=?",
                           (job_id,)).fetchone()
        assert row["state"] == "succeeded"
        assert row["failure_json"] is None
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
    finally:
        conn.close()


def test_cli_experiment_with_stale_thread_count_is_refused_input_changed(
        tmp_path, capsys, monkeypatch):
    """Negative control: force the pre-fix 1-thread environment into the plan
    and the same CLI flow is refused at launch with INPUT_CHANGED."""
    real_profile_named = experiments.profile_named
    monkeypatch.setattr(
        experiments, "profile_named",
        lambda policy, name: replace(real_profile_named(policy, name), thread_count=1))

    root = tmp_path / "ops"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document()))
    job_id = _cli_plan_and_submit(root, spec_path, capsys, key="cli-experiment-stale")

    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        service = Service(conn, root, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO)
        try:
            service.start()
            # Wait through admission (host-dependent) rather than a single
            # tick: the refusal happens at launch, and a queued job has no
            # attempt row yet.
            assert run_until(service, conn, job_id, timeout=90) == "failed"
        finally:
            service.close()
        failure = json.loads(conn.execute("SELECT failure_json FROM attempts WHERE job_id=?",
                                           (job_id,)).fetchone()[0])
        assert failure["code"] == "INPUT_CHANGED"
        assert conn.execute("SELECT state FROM jobs WHERE job_id=?",
                             (job_id,)).fetchone()[0] != "succeeded"
    finally:
        conn.close()


def test_spec_refuses_an_unknown_top_level_key_as_invalid_experiment_spec():
    with pytest.raises(OpsError) as excinfo:
        experiments.experiment_spec_from_document(_spec_document(author="operator"))
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"


def test_experiment_document_rejects_non_mapping_input():
    for malformed in (["fill", "mid"], 7):
        with pytest.raises(OpsError) as excinfo:
            experiments.experiment_spec_from_document(malformed)
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("changes", (
    pytest.param({"arms": "fixture"}, id="string-arms-document"),
    pytest.param({"arms": {"fixture": True}}, id="mapping-arms-document"),
    pytest.param({"folds": "fold-1"}, id="string-folds-document"),
    pytest.param({"folds": {"fold-1": True}}, id="mapping-folds-document"),
))
def test_experiment_spec_parser_rejects_non_array_arm_and_fold_fields(changes):
    """Gate round-3 finding: the parser ran ``tuple(...)`` over the raw
    document value before any validation, so an explicitly present string
    became character IDs and a mapping became its keys, and malformed
    documents reached the plan-aware runners. The document boundary now
    refuses every non-array value, and the refusal is parsing-level: a spec
    rebuilt with ``dataclasses.replace`` is the other test's job."""
    with pytest.raises(OpsError) as excinfo:
        experiments.experiment_spec_from_document(_spec_document(**changes))
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC", changes
    assert excinfo.value.problem.details["field"] == next(iter(changes))

    parsed = experiments.experiment_spec_from_document(_spec_document())
    assert parsed.arms == ("fixture",) and parsed.folds == ("fold-1",)
    optional = _spec_document()
    del optional["arms"], optional["folds"]
    defaulted = experiments.experiment_spec_from_document(optional)
    assert defaulted.arms == () and defaulted.folds == ()

    # A well-formed array whose items are not strings stays the resolver's
    # refusal, exactly as before: the parser only checks the raw shape.
    nested = experiments.experiment_spec_from_document(
        _spec_document(arms=["fixture", ["mutable"]]))
    assert nested.arms == ("fixture", ["mutable"])
    with pytest.raises(OpsError) as excinfo:
        experiments.resolve_experiment_plan(nested)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"


def test_experiment_plan_rejects_non_mapping_document_before_field_access(tmp_path):
    """CLI-facing entry point: a JSON document that is not a mapping -- even
    ``[{}]``, whose sole element is -- must be refused as the resolver's
    typed code before the ``document.get`` field access, not as the bare
    ``AttributeError`` a list would otherwise raise through the CLI."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps([{}]))
    with pytest.raises(OpsError) as excinfo:
        experiment_plan(spec_path, smoke=True)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"


def test_unused_economic_key_refuses_before_the_runner_is_invoked(tmp_path):
    spec = experiments.experiment_spec_from_document(
        _spec_document(economic_params={"fill": "mid", "slippage_bps": 5}))
    invoked = []

    def runner(*, run_dir, no_ledger):
        invoked.append(run_dir)

    with pytest.raises(OpsError) as excinfo:
        experiments.run_experiment(spec, tmp_path, tmp_path / "run", runner=runner,
                                   mode="smoke", synthetic=True)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not invoked and not (tmp_path / "run").exists()


@pytest.mark.parametrize("malformed_variant_id", (
    pytest.param("", id="empty-string"),
    pytest.param("   \t\n", id="whitespace-only"),
    pytest.param(7, id="non-string-integer"),
    pytest.param(b"sha256:deadbeef", id="non-string-bytes"),
))
def test_run_experiment_refuses_malformed_explicit_variant_ids_before_any_effect(
        tmp_path, malformed_variant_id):
    """An explicitly supplied variant ID that is not a non-empty string is the
    typed non-retryable ``INVALID_EXPERIMENT_SPEC`` refusal, fired before the
    resolved plan exists, before the destination directory is created and
    before the runner could ever be invoked -- an empty or whitespace-only ID
    is never silently defaulted to the spec hash the way a falsy ``or``
    chain would."""
    spec = experiments.experiment_spec_from_document(_spec_document(economic_params={}))
    invoked = []

    def runner(*, run_dir, no_ledger, execution_plan):
        invoked.append(run_dir)

    run_dir = tmp_path / "run"
    with pytest.raises(OpsError) as excinfo:
        experiments.run_experiment(spec, tmp_path, run_dir, runner=runner,
                                   mode="smoke", synthetic=True,
                                   variant_id=malformed_variant_id)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC", malformed_variant_id
    assert excinfo.value.problem.retryable is False
    assert not invoked
    assert not run_dir.exists()


def test_run_experiment_defaults_an_exactly_none_variant_id_to_the_spec_hash(tmp_path):
    """The default branch is the exactly-``None`` argument only: both the
    omitted ID and an explicit ``None`` resolve to the spec hash, and a valid
    supplied ID keeps its exact bytes (never normalized)."""
    spec = experiments.experiment_spec_from_document(_spec_document(economic_params={}))

    def runner(*, run_dir, no_ledger, execution_plan):
        (run_dir / "REPORT.md").write_text(
            "# variant default\n\n*Generated by engine.report v1.0.*\n")

    omitted = experiments.run_experiment(spec, tmp_path, tmp_path / "omitted",
                                         runner=runner, mode="smoke", synthetic=True)
    explicit_none = experiments.run_experiment(spec, tmp_path, tmp_path / "explicit-none",
                                               runner=runner, mode="smoke", synthetic=True,
                                               variant_id=None)
    assert omitted["status"] == "succeeded" and explicit_none["status"] == "succeeded"
    assert omitted["evidence"]["variant_id"] == spec.spec_hash
    assert explicit_none["evidence"]["variant_id"] == spec.spec_hash
    padded = experiments.run_experiment(spec, tmp_path, tmp_path / "padded",
                                        runner=runner, mode="smoke", synthetic=True,
                                        variant_id="  padded  ")
    assert padded["status"] == "succeeded"
    assert padded["evidence"]["variant_id"] == "  padded  "


@pytest.mark.parametrize("changes", (
    pytest.param({"arms": []}, id="zero-arms"),
    pytest.param({"arms": ["fixture", "control"]}, id="two-arms"),
    pytest.param({"arms": ["control"]}, id="one-arm-not-primary"),
))
def test_run_experiment_refuses_arm_shapes_that_declare_no_single_primary(tmp_path,
                                                                         changes):
    """A fixed-arm run names exactly one arm and names it as primary; zero
    arms, two, or a lone arm that is not the primary are refused as the
    resolver's ``INVALID_EXPERIMENT_SPEC`` before the run directory or any
    evidence exists, and the runner is never invoked."""
    spec = experiments.experiment_spec_from_document(_spec_document(**changes))
    invoked = []

    def runner(*, run_dir, no_ledger, execution_plan):
        invoked.append(run_dir)

    run_dir = tmp_path / "run"
    with pytest.raises(OpsError) as excinfo:
        experiments.run_experiment(spec, tmp_path, run_dir, runner=runner,
                                   mode="smoke", synthetic=True)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC", changes
    assert not invoked
    assert not run_dir.exists()


def test_resolved_plan_is_immutable_and_its_json_bytes_are_canonical():
    document = _spec_document(economic_params={"fill": "mid"})
    plan = experiments.resolve_experiment_plan(
        experiments.experiment_spec_from_document(document))
    first = plan.json_bytes()
    assert first == plan.json_bytes()
    assert b'"fill":"mid"' in first and b": " not in first
    payload = json.loads(first.decode("utf-8"))
    assert list(payload) == sorted(payload)
    with pytest.raises(TypeError):
        plan.economic_params["fill"] = "off"
    document["economic_params"]["fill"] = "off"  # no mutable mapping leaks in
    assert plan.economic_params["fill"] == "mid"
    assert plan.json_bytes() == first

    direct = experiments.ExperimentSpec(
        experiment_id="x", hypothesis="plumbing", primary_arm_id="fixture",
        arms=("fixture",), seed=7, folds=("fold-1",), economic_params={"fill": "mid"},
        price_source="synthetic", runner="synthetic")
    direct_plan = experiments.resolve_experiment_plan(direct)
    direct_bytes = direct_plan.json_bytes()
    assert direct_plan.arms == ("fixture",) and direct_plan.folds == ("fold-1",)
    assert direct_plan.json_bytes() == direct_bytes


def test_resolver_rejects_non_mapping_economic_params():
    spec = experiments.experiment_spec_from_document(_spec_document())
    for malformed in (["fill", "mid"], 7):
        with pytest.raises(OpsError) as excinfo:
            experiments.resolve_experiment_plan(replace(spec, economic_params=malformed))
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("changes", (
    pytest.param({"seed": {"n": 1}}, id="mutable-mapping-seed"),
    pytest.param({"seed": True}, id="bool-seed"),
    pytest.param({"seed": 7.5}, id="float-seed"),
    pytest.param({"arms": ["fixture", ["mutable"]]}, id="mutable-list-arms"),
    pytest.param({"arms": ["fixture"]}, id="plain-list-arms"),
    pytest.param({"arms": "fixture"}, id="string-arms"),
    pytest.param({"folds": ["fold-1", None]}, id="non-string-fold"),
    pytest.param({"folds": ["fold-1"]}, id="plain-list-folds"),
    pytest.param({"economic_params": {"fill": float("nan")}}, id="non-finite-nested"),
    pytest.param({"economic_params": {"fill": {"nested": {1: "int key"}}}},
                 id="non-string-nested-key"),
    pytest.param({"economic_params": {"fill": [{"nested": {1, 2}}]}}, id="set-nested"),
    pytest.param({"economic_params": {1: "int key", "slippage_bps": 5}},
                 id="mixed-nonstring-key-with-unknown-string-key"),
    pytest.param({"runner": ""}, id="empty-runner"),
))
def test_resolver_rejects_malformed_plan_field_types(changes):
    """Gate round-2 finding: the spec's dataclass annotations check nothing
    at runtime, so a JSON object supplied as ``seed`` was kept by reference
    in the resolved plan and mutating that source rewrote the plan's
    canonical bytes. The resolver now validates every field it copies into
    ``ResolvedExperimentPlan`` before a plan exists, and each malformed
    value is the same typed refusal -- never normalized or frozen into a
    plan. A valid integer seed still resolves and serializes as that same
    integer. Round-2 correction: a plain mutable list in ``arms``/``folds``
    contradicts the fields' declared tuple types (the parser converts valid
    document arrays; a hand-supplied list is refused), and economic keys are
    type-checked before the unused-key sort, so a mixed non-string key set
    beside an unknown string key is this typed refusal -- never the bare
    comparison ``TypeError`` sorting mixed keys would raise."""
    spec = experiments.experiment_spec_from_document(_spec_document())
    with pytest.raises(OpsError) as excinfo:
        experiments.resolve_experiment_plan(replace(spec, **changes))
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC", changes
    plan = experiments.resolve_experiment_plan(spec)
    assert plan.seed == 7 and isinstance(plan.seed, int)
    assert b'"seed":7' in plan.json_bytes()


def test_changed_fill_changes_the_plan_the_runner_receives(tmp_path):
    received = []

    def runner(*, run_dir, no_ledger, execution_plan):
        received.append(execution_plan)
        (run_dir / "REPORT.md").write_text(
            "# plan probe\n\n*Generated by engine.report v1.0.*\n")

    for fill in ("mid", "off"):
        spec = experiments.experiment_spec_from_document(
            _spec_document(economic_params={"fill": fill}))
        receipt = experiments.run_experiment(spec, tmp_path, tmp_path / fill, runner=runner,
                                             mode="smoke", synthetic=True,
                                             variant_id="registered-variant")
        assert receipt["status"] == "succeeded"
        assert receipt["evidence"]["variant_id"] == "registered-variant"
        assert receipt["evidence"]["variants_tried"] == 1
        report = (tmp_path / fill / "REPORT.md").read_text()
        assert "Variant ID: registered-variant\n" in report
        assert "Variants tried: 1\n" in report
        report_bytes = (tmp_path / fill / "REPORT.md").read_bytes()
        assert receipt["evidence"]["report_bytes"] == len(report_bytes)
        assert receipt["evidence"]["report_hash"] == "sha256:" + hashlib.sha256(report_bytes).hexdigest()
    assert [type(plan) for plan in received] == [experiments.ResolvedExperimentPlan] * 2
    assert [plan.economic_params["fill"] for plan in received] == ["mid", "off"]
    assert received[0].json_bytes() != received[1].json_bytes()

    rerun = experiments.run_experiment(spec, tmp_path, tmp_path / "off", runner=runner,
                                       mode="smoke", synthetic=True,
                                       variant_id="registered-variant")
    assert rerun["status"] == "succeeded"
    report = (tmp_path / "off" / "REPORT.md").read_text()
    assert report.count(experiments.VARIANT_MARKER_BEGIN) == 1
    assert report.count(experiments.VARIANT_MARKER_END) == 1


def test_legacy_callable_without_execution_plan_still_runs_empty_economics(tmp_path):
    spec = experiments.experiment_spec_from_document(_spec_document(economic_params={}))
    seen = []

    def runner(*, run_dir, no_ledger):
        seen.append(no_ledger)
        (run_dir / "REPORT.md").write_text(
            "# legacy\n\n*Generated by engine.report v1.0.*\n")

    receipt = experiments.run_experiment(spec, tmp_path, tmp_path / "legacy", runner=runner,
                                         mode="smoke", synthetic=True)
    assert receipt["status"] == "succeeded"
    assert seen == [True]


def test_legacy_callable_without_execution_plan_refuses_declared_economics(tmp_path):
    """A legacy callable has no channel for ``execution_plan``, so a resolved
    plan carrying economic parameters is a typed refusal before the call --
    never an invocation that silently drops the declared stance -- and the
    refusal is a preflight in ``run_experiment`` itself: no run directory,
    no ``CAPABILITIES.json``, no receipt exists when it fires."""
    spec = experiments.experiment_spec_from_document(
        _spec_document(economic_params={"fill": "mid"}))
    invoked = []

    def runner(*, run_dir, no_ledger):
        invoked.append(run_dir)

    run_dir = tmp_path / "legacy-refusal"
    with pytest.raises(OpsError) as excinfo:
        experiments.run_experiment(spec, tmp_path, run_dir, runner=runner,
                                   mode="smoke", synthetic=True)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert excinfo.value.problem.details["economic_keys"] == ["fill"]
    assert not invoked
    assert not run_dir.exists()
    assert not (run_dir / "CAPABILITIES.json").exists()
    assert not (tmp_path / "experiment_receipt.json").exists()


def test_run_experiment_hands_in_the_same_supplied_resolved_plan(tmp_path):
    """A supplied plan whose canonical bytes match the spec's resolution is
    adopted as-is -- the callable receives that exact object -- and a plan
    that resolves differently is a typed refusal before the run directory
    or any evidence can exist."""
    spec = experiments.experiment_spec_from_document(
        _spec_document(economic_params={"fill": "mid"}))
    plan = experiments.resolve_experiment_plan(spec)
    received = []

    def runner(*, run_dir, no_ledger, execution_plan):
        received.append(execution_plan)
        (run_dir / "REPORT.md").write_text(
            "# supplied plan\n\n*Generated by engine.report v1.0.*\n")

    run_dir = tmp_path / "supplied"
    receipt = experiments.run_experiment(spec, tmp_path, run_dir, runner=runner,
                                         mode="smoke", synthetic=True, resolved_plan=plan)
    assert receipt["status"] == "succeeded"
    assert len(received) == 1
    assert received[0] is plan

    mismatched = experiments.resolve_experiment_plan(
        replace(spec, economic_params={"fill": "off"}))
    refused_dir = tmp_path / "mismatched"
    invoked = []

    def spy_runner(*, run_dir, no_ledger, execution_plan):
        invoked.append(execution_plan)

    with pytest.raises(OpsError) as excinfo:
        experiments.run_experiment(spec, tmp_path, refused_dir, runner=spy_runner,
                                   mode="smoke", synthetic=True, resolved_plan=mismatched)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not invoked
    assert not refused_dir.exists()


def test_run_experiment_rejects_mutable_supplied_plan_economics(tmp_path):
    """Gate finding: canonical bytes can match while nested economics stay
    editable -- a supplied plan whose ``economic_params`` is a mutable dict
    is the same typed refusal, fired as a preflight before the run directory
    or any evidence exists, and the runner is never invoked."""
    spec = experiments.experiment_spec_from_document(
        _spec_document(economic_params={"fill": "mid"}))
    plan = replace(experiments.resolve_experiment_plan(spec), economic_params={"fill": "mid"})
    assert plan.json_bytes() == experiments.resolve_experiment_plan(spec).json_bytes()
    invoked = []

    def runner(*, run_dir, no_ledger, execution_plan):
        invoked.append(execution_plan)

    run_dir = tmp_path / "mutable-supplied"
    with pytest.raises(OpsError) as excinfo:
        experiments.run_experiment(spec, tmp_path, run_dir, runner=runner,
                                   mode="smoke", synthetic=True, resolved_plan=plan)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not invoked
    assert not run_dir.exists()
    assert not (run_dir / "CAPABILITIES.json").exists()
    assert not (tmp_path / "experiment_receipt.json").exists()


def test_run_experiment_rejects_replaced_mutable_supplied_plan(tmp_path):
    """Round-2 gate: the byte-equal impostor can hide mutability in fields
    the immutable-economics shape check alone would wave through -- a plain
    ``economic_params`` dict, a ``MappingProxyType`` over a caller-retained
    mutable backing dict, a list ``arms``. Every ``dataclasses.replace``
    rebuild loses the resolver's provenance marker, so each is the same
    typed refusal, fired as a preflight before any run directory or
    evidence exists, and the runner is never invoked. The resolver's own
    plan is the contrasting case: mutating the source economics dictionary
    after resolution -- it travels into the spec by reference -- never
    changes its content or its canonical bytes."""
    source_params = {"fill": "mid"}
    spec = experiments.ExperimentSpec(
        experiment_id="x", hypothesis="plumbing", primary_arm_id="fixture",
        arms=("fixture",), seed=7, folds=("fold-1",),
        economic_params=source_params, price_source="synthetic", runner="synthetic")
    accepted = experiments.resolve_experiment_plan(spec)
    canonical = accepted.json_bytes()
    source_params["fill"] = "off"
    assert accepted.economic_params["fill"] == "mid"
    assert accepted.arms == ("fixture",) and accepted.folds == ("fold-1",)
    assert accepted.json_bytes() == canonical

    dict_plan = replace(accepted, economic_params={"fill": "mid"})
    backing = {"fill": "mid"}
    proxy_plan = replace(accepted, economic_params=MappingProxyType(backing))
    arms_plan = replace(accepted, arms=["fixture"])
    for variant in (dict_plan, proxy_plan, arms_plan):
        assert variant.json_bytes() == canonical

    # Mutated after wrapping: the retained dict still edits the "read-only"
    # proxy, proving the variant exercises the live alias, not a snapshot.
    backing["fill"] = "off"
    assert proxy_plan.economic_params["fill"] == "off"

    invoked = []

    def runner(*, run_dir, no_ledger, execution_plan):
        invoked.append(execution_plan)

    for name, variant in (("economic-dict", dict_plan),
                          ("economic-proxy", proxy_plan),
                          ("arms-list", arms_plan)):
        run_dir = tmp_path / name
        with pytest.raises(OpsError) as excinfo:
            experiments.run_experiment(spec, tmp_path, run_dir, runner=runner,
                                       mode="smoke", synthetic=True, resolved_plan=variant)
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC", name
        assert not invoked
        assert not run_dir.exists()
        assert not (run_dir / "CAPABILITIES.json").exists()
    assert not (tmp_path / "experiment_receipt.json").exists()
