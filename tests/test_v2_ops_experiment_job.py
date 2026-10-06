"""P6 slice 10: smoke-mode experiments run as supervised v2 jobs."""
import csv
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops import cli, effects_graph, experiments, stages, supervisor, worker
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
from engine.v2.ops.submission import KindRegistry, NamespacePolicy, submit
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


def _planned_ledger(path, experiment_id="x", stage="planned"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id,spec_hash,date,stage,oos_mean_mid,sharpe_trade,promoted\n"
                    f"{experiment_id},deadbeef,2026-01-01,{stage},,,False\n")


def _ledger_rows(path):
    with open(path, newline="") as fh:
        return [{"id": row["id"], "stage": row["stage"]} for row in csv.DictReader(fh)]


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
    """A tmp checkout with a registered runner, its legacy spec.yaml, the
    inventory's declared runtime source, and a PLANNED ledger row whose
    spec_hash is the legacy ``experiments.lib`` hash."""
    from experiments import lib

    checkout = tmp_path / "checkout"
    runner = checkout / REGISTERED_RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text("if __name__ == '__main__':\n    pass\n")
    legacy_spec = runner.parent / "spec.yaml"
    legacy_spec.write_text("id: EXP-182\nprimary_spec:\n  x: 1\n")
    for source_rel in experiments.RUNNER_INVENTORY[REGISTERED_RUNNER][
            "declared_runtime_sources"]:
        declared_source = checkout / source_rel
        declared_source.parent.mkdir(parents=True, exist_ok=True)
        declared_source.write_text("if __name__ == '__main__':\n    pass\n")
    lib.ledger_append([{"id": experiment_id,
                        "spec_hash": lib.spec_hash(lib.load_spec(legacy_spec)),
                        "date": "2026-01-01", "stage": "planned",
                        "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}],
                      path=checkout / "experiments" / "LEDGER.csv")
    return checkout, legacy_spec


def _primary_cli_checkout(tmp_path, experiment_id="EXP-182"):
    """A tmp checkout carrying the REAL registered EXP-182 wrapper, its
    ``spec.yaml``, the inventory's declared runtime source(s) and the runner
    manifest's source closure -- with the copied declared source replaced by a
    tiny ``main()`` stub that asserts ``--no-ledger`` and writes an engine-marked
    ``REPORT.md`` in ``Path.cwd()`` (test-only; it never touches a ledger). A
    PLANNED row whose ``spec_hash`` is the checkout spec's legacy
    ``experiments.lib`` hash is written beside them. Every path the primary
    submit publishes -- the wrapper, the spec source, each declared source and
    each closure member -- is materialized under the checkout, so
    ``runner_manifest(checkout, runner)`` re-derives the same audited source
    set; because the copied wrapper and the stub import only stdlib, that
    closure stays tiny and self-contained. All writes stay under ``tmp_path``."""
    import shutil

    from experiments import lib

    manifest = experiments.runner_manifest(REPO, REGISTERED_RUNNER)
    declared = experiments.RUNNER_INVENTORY[REGISTERED_RUNNER]["declared_runtime_sources"]
    checkout = tmp_path / "checkout"
    for relative in dict.fromkeys(
            [manifest["runner"], manifest["spec_source"], *declared,
             *manifest["source_closure"]]):
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, destination)
    (checkout / declared[0]).write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "def main():\n"
        "    assert '--no-ledger' in sys.argv\n"
        "    Path('REPORT.md').write_text(\n"
        "        '# EXP-182 primary stub\\n\\n*Generated by engine.report v1.0.*\\n')\n")
    legacy_spec = checkout / manifest["spec_source"]
    lib.ledger_append([{"id": experiment_id,
                        "spec_hash": lib.spec_hash(lib.load_spec(legacy_spec)),
                        "date": "2026-01-01", "stage": "planned",
                        "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}],
                      path=checkout / "experiments" / "LEDGER.csv")
    return checkout


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
    runner_id = "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py"
    checkout = tmp_path
    runner_path = checkout / runner_id
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("if __name__ == '__main__':\n    pass\n")
    legacy_spec = runner_path.parent / "spec.yaml"
    legacy_spec.write_text("id: EXP-182\n")
    declared_source = checkout / experiments.RUNNER_INVENTORY[runner_id][
        "declared_runtime_sources"][0]
    declared_source.parent.mkdir(parents=True, exist_ok=True)
    declared_source.write_text("if __name__ == '__main__':\n    pass\n")
    (tmp_path / "spec.json").write_text(
        json.dumps(_spec_document(runner=runner_id, economic_params={})))

    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (tmp_path / "REPORT.md").write_text(
            "# Synthetic infrastructure report\n\n*Generated by engine.report v1.0.*\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = worker.dispatch("experiment",
                             {"expected_ids": ["experiment:x"], "runner": runner_id,
                              "no_ledger": False, "preregistration_root": str(checkout)},
                             tmp_path)
    assert result["completed_ids"] == ["experiment:x"]
    assert len(commands) == 1
    assert "--no-ledger" in commands[0], "the runner never gets ledger writes"

    from experiments import lib

    registered = lib.spec_hash(lib.load_spec(legacy_spec))
    resolved = experiments.experiment_spec_from_document(
        json.loads((tmp_path / "spec.json").read_text())).spec_hash
    report_text = (tmp_path / "REPORT.md").read_text()
    assert f"Variant ID: {registered}\nVariants tried: 1\n" in report_text, \
        "the report carries the registered identity, not the resolved hash"
    assert f"Variant ID: {resolved}" not in report_text
    report_output = next(output for output in result["outputs"]
                         if output["name"] == "experiment_variant_report")
    assert report_output["path"] == "REPORT.md"
    assert report_output["schema"] == "experiment_variant_report.v1.0"


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
    declared_source = tmp_path / experiments.RUNNER_INVENTORY[runner_id][
        "declared_runtime_sources"][0]
    declared_source.parent.mkdir(parents=True, exist_ok=True)
    declared_source.write_text("if __name__ == '__main__':\n    pass\n")
    (tmp_path / "spec.json").write_text(
        json.dumps(_spec_document(runner=runner_id, economic_params={})))

    with pytest.raises(OpsError) as excinfo:
        worker.dispatch("experiment",
                        {"expected_ids": ["experiment:x"], "runner": runner_id,
                         "no_ledger": True}, tmp_path)
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details["returncode"] == 1
    assert "runner exploded" in excinfo.value.problem.details["stderr_tail"]
    assert not (tmp_path / "experiments" / "LEDGER.csv").exists(), \
        "the worker never appends a ran row"


@pytest.mark.parametrize("link_kind", ("symlink", "hardlink"))
def test_worker_refuses_linked_report_without_mutating_target_or_committing(
        tmp_path, monkeypatch, link_kind):
    """A runner that stages ``REPORT.md`` as a LINK -- a symlink out to an
    external file, or a hardlink aliasing one -- is refused as a non-retryable
    ``VALIDATION_FAILED`` before the variant annotation can write through it:
    the external target's bytes are unchanged, the staged link itself is left
    exactly as the runner made it, the checkout ledger still holds only its
    PLANNED row, and the catalog registers neither a durable run nor a
    hypothesis. Before the ``lstat()`` guard the annotation followed the link,
    rewrote the external target's content and accepted the attempt."""
    checkout = tmp_path
    ledger = checkout / "experiments" / "LEDGER.csv"
    _planned_ledger(ledger)
    ledger_before = ledger.read_bytes()
    (checkout / "spec.json").write_text(json.dumps(_spec_document(economic_params={})))

    external = checkout / "shared" / "REPORT.md"
    external.parent.mkdir(parents=True)
    external.write_text("# Linked report\n\n*Generated by engine.report v1.0.*\n")
    target_before = external.read_bytes()

    def linked_runner(*, run_dir, no_ledger):
        staged = run_dir / "REPORT.md"
        if link_kind == "symlink":
            os.symlink(external, staged)
        else:
            os.link(external, staged)
        return {"fixture": True}

    monkeypatch.setattr(experiments, "synthetic_fixture_runner", linked_runner)
    conn, _, _ = catalog(checkout)
    try:
        with pytest.raises(OpsError) as excinfo:
            worker.dispatch("experiment",
                            {"expected_ids": ["experiment:x"], "runner": "synthetic",
                             "no_ledger": False, "preregistration_root": str(checkout)},
                            checkout)
        assert excinfo.value.code == "VALIDATION_FAILED"
        assert excinfo.value.problem.retryable is False

        staged = checkout / "REPORT.md"
        assert external.read_bytes() == target_before
        if link_kind == "symlink":
            assert staged.is_symlink()
            assert Path(os.readlink(staged)) == external
        else:
            assert not staged.is_symlink()
            assert staged.stat().st_nlink == 2
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "planned"}]
        assert ledger.read_bytes() == ledger_before
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
    finally:
        conn.close()


def test_worker_refuses_a_report_swapped_to_a_link_during_integrity_check(tmp_path, monkeypatch):
    """After the synthetic runner writes a valid report, ``_check_report_integrity``
    is wrapped to replace that report with a symlink right before the original
    check runs: the persisted receipt is a failed, non-retryable
    ``VALIDATION_FAILED`` naming the report path with its before/after hash
    fields, and the checkout ledger still holds only its PLANNED row."""
    checkout, _ = _registered_checkout(tmp_path)
    ledger = checkout / "experiments" / "LEDGER.csv"
    (checkout / "spec.json").write_text(json.dumps(_spec_document(
        experiment_id="EXP-182", economic_params={})))
    real_check = experiments._check_report_integrity

    def swap_to_link(report, before):
        external = report.parent / "external.md"
        external.write_text(report.read_text())
        report.unlink()
        os.symlink(external, report)
        return real_check(report, before)

    monkeypatch.setattr(experiments, "_check_report_integrity", swap_to_link)
    with pytest.raises(OpsError) as excinfo:
        worker.dispatch("experiment", {"expected_ids": ["experiment:EXP-182"],
                                       "runner": "synthetic", "no_ledger": False,
                                       "preregistration_root": str(checkout)}, checkout)
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.retryable is False
    receipt = json.loads((checkout / "experiment_receipt.json").read_text())
    failure = receipt["evidence"]["failure_details"]
    assert receipt["status"] == "failed"
    assert failure["path"] == str(checkout / "REPORT.md")
    assert "before_hash" in failure and "after_hash" in failure
    assert _ledger_rows(ledger) == [{"id": "EXP-182", "stage": "planned"}]


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
        # The durable run carries the SAME variant identity the ledger row
        # joins on: no legacy spec source for the synthetic runner, so the
        # registered hash is None and the resolved spec hash stands.
        expected_variant = experiments.experiment_spec_from_document(
            _spec_document(economic_params={})).spec_hash
        evidence = json.loads(conn.execute(
            "SELECT evidence_json FROM experiment_runs").fetchone()[0])
        assert evidence["variant_id"] == expected_variant
        assert evidence["variants_tried"] == 1
        assert _ledger_rows(ledger) == [{"id": "x", "stage": "planned"},
                                        {"id": "x", "stage": "ran"}]
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
        retried = json.loads(conn_.execute(
            "SELECT evidence_json FROM experiment_runs").fetchone()[0])
        assert retried["variant_id"] == expected_variant
        assert retried["variants_tried"] == 1
    finally:
        conn.close()


def _claimed_primary_effect(tmp_path, key="primary-fence", *, receipt_evidence=None):
    """A real claimed attempt + the coordinator's commit closure, plus the
    checkout ledger it appends to. Returns ``(conn, clock, claim, effect, checkout)``.
    ``receipt_evidence`` becomes the synthetic receipt's ``evidence`` object: by
    default the complete variant identity (the resolved spec hash) with exactly
    one attempt; an explicit value -- even an empty one -- is published verbatim
    so malformed-evidence refusals stay meaningful."""
    ops_root, checkout = tmp_path / "ops", tmp_path / "checkout"
    _planned_ledger(checkout / "experiments" / "LEDGER.csv")
    ops_root.mkdir()
    conn, clock, supervisor_ = catalog(ops_root)
    _submit_experiment(conn, ops_root, clock, key, no_ledger=False,
                       preregistration_root=checkout)
    claim = claim_next(conn, policy=TEST_POLICY, sample=sample(clock),
                       supervisor=supervisor_, clock=clock, registry=stages.registry())
    assert claim is not None
    store = ArtifactStore(ops_root)
    resolve_and_record(conn, store, claim)
    if receipt_evidence is None:
        receipt_evidence = {
            "variant_id": experiments.experiment_spec_from_document(
                _spec_document(economic_params={})).spec_hash,
            "variants_tried": 1}
    receipt_ref = store.publish_bytes(
        json.dumps({"input_hash": "input-" + key,
                    "evidence": receipt_evidence}).encode(),
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


def test_experiment_effect_records_synthetic_variant_in_ledger(tmp_path):
    """Review fix item 5: the ran row's ``spec_hash`` joins the durable
    identity. The default synthetic spec has no registered legacy hash, so
    the resolved ``ExperimentSpec.spec_hash`` is what both the checkout
    ledger and ``experiment_runs.evidence_json.variant_id`` carry -- the old
    implementation appended the ran row with an empty spec hash."""
    conn, clock, claim, effect, checkout = _claimed_primary_effect(tmp_path)
    ledger = checkout / "experiments" / "LEDGER.csv"
    try:
        commit_attempt(conn, claim.attempt_id, claim.fence,
                       Outcome(True, "verified_dead", 0), clock=clock,
                       effects=lambda txn: effect(txn))
        spec = experiments.experiment_spec_from_document(
            _spec_document(economic_params={}))
        with open(ledger, newline="") as fh:
            ran = [row for row in csv.DictReader(fh) if row["stage"] == "ran"]
        assert len(ran) == 1
        assert ran[0]["id"] == "x"
        assert ran[0]["spec_hash"] == spec.spec_hash
        evidence = json.loads(conn.execute(
            "SELECT evidence_json FROM experiment_runs").fetchone()[0])
        assert evidence["variant_id"] == spec.spec_hash
    finally:
        conn.close()


def test_experiment_effect_rejects_receipt_variant_mismatch_atomically(tmp_path):
    """A receipt reporting a variant identity other than the checkout's
    registered one is a typed ``INVALID_EXPERIMENT_SPEC`` refusal before the
    ledger append, and the ``commit_attempt`` transaction rolls back the
    run/hypothesis/evidence registration with it: the planned ledger bytes
    are untouched and no durable run or hypothesis row survives."""
    conn, clock, claim, effect, checkout = _claimed_primary_effect(
        tmp_path, key="primary-variant-mismatch",
        receipt_evidence={"variant_id": "not-the-registered-identity"})
    ledger = checkout / "experiments" / "LEDGER.csv"
    before = ledger.read_bytes()
    try:
        with pytest.raises(OpsError) as excinfo:
            commit_attempt(conn, claim.attempt_id, claim.fence,
                           Outcome(True, "verified_dead", 0), clock=clock,
                           effects=lambda txn: effect(txn))
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
        assert ledger.read_bytes() == before
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
    finally:
        conn.close()


def test_experiment_effect_rejects_missing_or_malformed_receipt_evidence_atomically(tmp_path):
    """CodeRabbit finding: only a *present and different* ``variant_id`` was
    refused, so a receipt with no evidence, a non-object evidence, a matching
    identity with no ``variants_tried``, or a count other than the integer 1
    was accepted and appended its ran row. Every such shape is a non-retryable
    ``INVALID_EXPERIMENT_SPEC`` refusal before the append, and the rolled-back
    transaction leaves the planned ledger bytes untouched and no run or
    hypothesis row behind."""
    variant = experiments.experiment_spec_from_document(
        _spec_document(economic_params={})).spec_hash
    cases = [("empty-evidence", {}),
             ("non-object-evidence", []),
             ("missing-variants-tried", {"variant_id": variant}),
             ("zero-variants-tried", {"variant_id": variant, "variants_tried": 0})]
    for name, receipt_evidence in cases:
        conn, clock, claim, effect, checkout = _claimed_primary_effect(
            tmp_path / name, key=f"primary-evidence-{name}",
            receipt_evidence=receipt_evidence)
        ledger = checkout / "experiments" / "LEDGER.csv"
        before = ledger.read_bytes()
        try:
            with pytest.raises(OpsError) as excinfo:
                commit_attempt(conn, claim.attempt_id, claim.fence,
                               Outcome(True, "verified_dead", 0), clock=clock,
                               effects=lambda txn: effect(txn))
            assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC", name
            assert excinfo.value.problem.retryable is False, name
            assert ledger.read_bytes() == before, name
            assert conn.execute(
                "SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0, name
            assert conn.execute(
                "SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0, name
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


def test_resolver_refuses_a_spec_that_is_not_exactly_its_primary_arm():
    """Fixed-arm guard: execution resolves one plan for one arm, so a spec
    declaring more than one arm -- or a single arm the spec does not name
    primary -- is refused by the resolver as ``INVALID_EXPERIMENT_SPEC``
    before any plan exists, leaving no artifact, receipt, report, or ledger
    row behind. The code is non-retryable, so the caller registers a
    corrected spec as a new identity rather than retrying this one. A
    well-formed one-arm spec whose primary is that arm still resolves."""
    for changes, arm_count in (
            ({"arms": []}, 0),
            ({"arms": ["fixture", "challenger"]}, 2),
            ({"primary_arm_id": "challenger"}, 1)):
        malformed = experiments.experiment_spec_from_document(_spec_document(**changes))
        with pytest.raises(OpsError) as excinfo:
            experiments.resolve_experiment_plan(malformed)
        assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC", changes
        assert excinfo.value.problem.retryable is False, changes
        assert excinfo.value.problem.details["arm_count"] == arm_count, changes
        assert excinfo.value.problem.details["arms"] == list(malformed.arms), changes

    plan = experiments.resolve_experiment_plan(
        experiments.experiment_spec_from_document(_spec_document()))
    assert plan.arms == ("fixture",)
    assert json.loads(plan.json_bytes())["arms"] == ["fixture"]


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
                                             mode="smoke", synthetic=True)
        assert receipt["status"] == "succeeded"
    assert [type(plan) for plan in received] == [experiments.ResolvedExperimentPlan] * 2
    assert [plan.economic_params["fill"] for plan in received] == ["mid", "off"]
    assert received[0].json_bytes() != received[1].json_bytes()


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


def test_primary_cli_submission_handoffs_registered_runner_sources(tmp_path, monkeypatch,
                                                                   capsys):
    """End-to-end primary CLI submission through the real registered EXP-182
    wrapper: ``ops plan --activate-ledger`` then ``ops submit`` publish the
    audited wrapper/spec/declared-source closure, the executor stages those
    verified bytes beside ``spec.json``, and the worker runs the wrapper as an
    explicit ``SourceFileLoader`` subprocess that asserts ``--no-ledger`` and
    writes an engine-marked report. This refuses the pre-fix primary path that
    staged only ``spec.json``: with the registered wrapper/source absent from
    the input stage the launch fails as a non-retryable error and commits no
    durable run or ledger row, so only the complete handoff can reach success.
    A primary run then registers exactly one durable ``experiment_runs`` row and
    appends exactly one ``ran`` row after the checkout's original ``planned``
    row."""
    checkout = _primary_cli_checkout(tmp_path)
    monkeypatch.setattr(experiments, "default_checkout_root", lambda: checkout)
    real_registry = stages.registry()
    experiment_kind = real_registry.get("experiment")
    patched_registry = KindRegistry([
        replace(experiment_kind, namespaces=experiment_kind.namespaces | {"primary"})
        if kind is experiment_kind else kind
        for kind in (real_registry.get(name) for name in real_registry.names())
    ])
    monkeypatch.setattr(
        cli,
        "NamespacePolicy",
        lambda grants: NamespacePolicy({
            "operator": frozenset(grants["operator"]) | {"primary"}
        }))
    monkeypatch.setattr(cli, "registry", lambda: patched_registry)
    ops = tmp_path / "ops"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(_spec_document(
        experiment_id="EXP-182", runner=REGISTERED_RUNNER, economic_params={})))

    assert cli.main(["--root", str(ops), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(ops), "plan", "experiment", "--spec", str(spec_path),
                     "--activate-ledger"]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(ops), "submit", "--plan", plan_ref,
                     "--idempotency-key", "cli-primary-1"]) == 0
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
        row = conn.execute("SELECT state, failure_json FROM attempts WHERE job_id=?",
                           (job_id,)).fetchone()
        assert row["state"] == "succeeded"
        assert row["failure_json"] is None
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 1
        assert _ledger_rows(checkout / "experiments" / "LEDGER.csv") == [
            {"id": "EXP-182", "stage": "planned"}, {"id": "EXP-182", "stage": "ran"}]
    finally:
        conn.close()


def test_worker_refuses_a_declared_source_tampered_during_the_run(tmp_path, monkeypatch):
    """A stub runner appends bytes to the first declared source under its cwd
    and exits 0: the worker's re-hash refuses non-retryably, retaining the
    relative path and both hashes in the details and the failed receipt, and
    committing no ledger row or durable run."""
    checkout, _ = _registered_checkout(tmp_path)
    declared = experiments.RUNNER_INVENTORY[REGISTERED_RUNNER]["declared_runtime_sources"][0]
    ledger_before = (ledger := checkout / "experiments" / "LEDGER.csv").read_bytes()
    (checkout / "spec.json").write_text(json.dumps(_spec_document(
        experiment_id="EXP-182", runner=REGISTERED_RUNNER, economic_params={})))

    def tampering_run(command, **kwargs):
        assert "--no-ledger" in command
        target = Path(kwargs["cwd"]) / declared
        target.write_bytes(target.read_bytes() + b"\n# tampered during the run\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", tampering_run)
    conn, _, _ = catalog(tmp_path)
    with pytest.raises(OpsError) as excinfo:
        worker.dispatch("experiment", {"runner": REGISTERED_RUNNER, "no_ledger": False,
                                       "preregistration_root": str(checkout)}, checkout)
    assert excinfo.value.code == "VALIDATION_FAILED" and excinfo.value.problem.retryable is False
    details = excinfo.value.problem.details
    assert details["path"] == declared and details["before_hash"] != details["after_hash"]
    receipt = json.loads((checkout / "experiment_receipt.json").read_text())
    failure = receipt["evidence"]["failure_details"]
    assert receipt["status"] == "failed" and failure.items() <= details.items()
    assert _ledger_rows(ledger) == [{"id": "EXP-182", "stage": "planned"}]
    assert ledger.read_bytes() == ledger_before
    assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
    conn.close()


def test_worker_refuses_a_declared_source_that_cannot_be_hashed(tmp_path, monkeypatch):
    """An unhashable declared source -- null before AND after -- is a typed,
    non-retryable refusal naming the path with both null hashes, never a
    silent pass on ``None == None``; the stub runner exits 0, and only the
    failed receipt and its evidence are left, with no committed run or
    ledger row."""
    from engine.v2.ops import legacy_adapter

    checkout, _ = _registered_checkout(tmp_path)
    declared = experiments.RUNNER_INVENTORY[REGISTERED_RUNNER]["declared_runtime_sources"][0]
    ledger_before = (ledger := checkout / "experiments" / "LEDGER.csv").read_bytes()
    (checkout / "spec.json").write_text(json.dumps(_spec_document(
        experiment_id="EXP-182", runner=REGISTERED_RUNNER, economic_params={})))

    original_hash_or_none = legacy_adapter._hash_or_none

    def unhashable_declared_source(path):
        return None if str(path).endswith(declared) else original_hash_or_none(path)

    def successful_run(command, **kwargs):
        assert "--no-ledger" in command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(legacy_adapter, "_hash_or_none", unhashable_declared_source)
    monkeypatch.setattr(subprocess, "run", successful_run)
    conn, _, _ = catalog(tmp_path)
    with pytest.raises(OpsError) as excinfo:
        worker.dispatch("experiment", {"runner": REGISTERED_RUNNER, "no_ledger": False,
                                       "preregistration_root": str(checkout)}, checkout)
    assert excinfo.value.code == "VALIDATION_FAILED" and excinfo.value.problem.retryable is False
    details = excinfo.value.problem.details
    assert details["path"] == declared
    assert details["before_hash"] is None and details["after_hash"] is None
    receipt = json.loads((checkout / "experiment_receipt.json").read_text())
    failure = receipt["evidence"]["failure_details"]
    assert receipt["status"] == "failed" and failure.items() <= details.items()
    assert _ledger_rows(ledger) == [{"id": "EXP-182", "stage": "planned"}]
    assert ledger.read_bytes() == ledger_before
    assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
    conn.close()
