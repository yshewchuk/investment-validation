"""Sealed sweep admission and actual independent subprocess gate evaluations."""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import DATA_FAILURE_CODES, FAILURE_CODES
from engine.v2.data.errors import DataError
from engine.v2.data.errors import make_problem as data_problem
from engine.v2.foundation import (
    ArtifactStore,
    SystemClock,
    canonical_json,
    content_hash,
    from_document,
)
from engine.v2.ops import schema, stages
from engine.v2.ops.catalog import connect
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.executor import THREAD_VARIABLES
from engine.v2.ops.experiments import register_hypothesis
from engine.v2.ops.fingerprints import snapshot_code, source_closure, worker_source_manifest
from engine.v2.ops.migrations import migrate
from engine.v2.ops.supervisor import Service
from experiments import (
    gate_report,
    native_gate_entrypoint,
    native_gate_evaluation,
    native_gate_worker,
    native_outcomes,
    native_registration,
)
from experiments import v2_gate as gate_cli
from experiments import v2_gate_sweep as sweep
from tests.data_scan_support import (
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.ops_support import TEST_POLICY, FakeClock, run_until
from tests.v2.research.scan_progress_support import assert_scan_boundaries, observe_scan_outcomes
from tests.v2.research.test_gate_report import MONTH, REPO, build_gate_source, build_gate_spec
from tests.v2.research.test_gate_report import source as source


def _request():
    return {"schema_version": "native_gate_sweep_request.v1.0", "primary_comparison": ["candidate", "control"],
            "variants": [json.loads(canonical_json(asdict(build_gate_spec(alpha, arm))))
                         for alpha, arm in ((0.0, "candidate"), (1.0, "control"))]}


def _call(source, action=sweep.register_sweep, document=None, *, code_root=REPO, **kwargs):
    return action(source.conn, source.store, source.repository, document or _request(),
                  code_root=code_root, as_of_month=MONTH, **kwargs)


def _row(source):
    return json.loads(source.conn.execute("SELECT document_json FROM native_gate_sweeps").fetchone()[0])


def _scientific(source):
    return [tuple(row) for row in source.conn.execute("SELECT * FROM experiment_runs ORDER BY run_id")]


def _forbid(*args, **kwargs):
    pytest.fail("refusal crossed the protected read/effect boundary")


def _new_pin(source):
    contract = contract_for("earnings_events")
    groups = {}
    for event in source.events + source.protected:
        groups.setdefault(str(event["event_date"].year), []).append(event)
    records = [publish_and_inspect(source.store, contract, contract_ref_for(contract),
               sorted(rows, key=lambda row: row["event_id"]), year) for year, rows in sorted(groups.items())]
    return commit_tables(source.conn, FakeClock(), {"earnings_events": records}, {"earnings_events": contract},
                         scope="alternate", receipt_id="alternate", attempt_id="alternate", store=source.store)


def _advance(source, pin):
    source.conn.execute("UPDATE data_snapshot_heads SET snapshot_id=?, generation=generation+1 WHERE scope='shadow'",
                        (pin.snapshot_id,))


def _checkout(tmp_path):
    path = tmp_path / "checkout"
    snapshot_code(REPO, path, worker_source_manifest(REPO))
    # The coordinator is an operator entrypoint, not part of the worker manifest.
    path.joinpath("experiments/v2_gate_sweep.py").write_bytes(REPO.joinpath("experiments/v2_gate_sweep.py").read_bytes())
    return path


def _replace_source(path, text):
    """Replace only a disposable test checkout's read-only snapshot copy."""
    path.unlink()
    path.write_text(text)


def _run_children(source, jobs, code_root, only=None):
    selected = jobs if only is None else only
    service = Service(source.conn, source.store.root, stages.registry(), TEST_POLICY,
                      clock=SystemClock(), code_source=code_root)
    try:
        service.start()
        for job in selected:
            run_until(service, source.conn, job.job_id, timeout=90)
    finally:
        service.close()
    results = []
    for job in selected:
        row = source.conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?", (job.job_id,)).fetchone()
        attempts = source.conn.execute("SELECT * FROM attempts WHERE job_id=?", (job.job_id,)).fetchall()
        assert len(attempts) == 1
        outputs = source.conn.execute("SELECT * FROM attempt_outputs WHERE attempt_id=?",
                                      (attempts[0]["attempt_id"],)).fetchall()
        if row[0] == "succeeded":
            assert len(outputs) == 1 and outputs[0]["name"] == "native_gate_result"
            ref = artifact(source.conn, source.store, outputs[0]["artifact_id"])
            assert ref.schema_ref == "native_gate_result.v1.0"
            results.append(json.loads(source.store.read_verified(ref)))
        else:
            assert row[0] == "failed" and outputs == []
            results.append(json.loads(row[1]))
    assert source.conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
    return results


DATA_TO_OPS = {
    "EXPERIMENT_VARIANT_FAILED": "EXPERIMENT_VARIANT_FAILED",
    "SNAPSHOT_UNRESOLVED": "SNAPSHOT_UNRESOLVED", "SNAPSHOT_NOT_FOUND": "INPUT_CHANGED",
    "SNAPSHOT_NOT_READY": "SNAPSHOT_NOT_READY", "SNAPSHOT_CONFLICT": "SNAPSHOT_NOT_READY",
    "CONTRACT_MISMATCH": "VALIDATION_FAILED", "QUERY_NOT_BOUNDED": "VALIDATION_FAILED",
    "RESULT_LIMIT_EXCEEDED": "RESOURCE_LIMIT_EXCEEDED", "CALENDAR_UNAVAILABLE": "VALIDATION_FAILED",
    "RESOURCE_UNAVAILABLE": "RESOURCE_UNAVAILABLE", "TRANSIENT_SOURCE": "TRANSIENT_SOURCE",
    "SOURCE_NOT_FINAL": "SOURCE_NOT_FINAL", "INPUT_CHANGED": "INPUT_CHANGED",
    "OBJECT_CORRUPT": "INTEGRITY_FAILED", "MANIFEST_CORRUPT": "INTEGRITY_FAILED",
    "IDENTITY_CONFLICT": "VALIDATION_FAILED", "UNSUPPORTED_CONTRACT": "VALIDATION_FAILED",
    "EVENT_NOT_FOUND": "INPUT_CHANGED", "DEADLINE_EXCEEDED": "RESOURCE_UNAVAILABLE",
    "POPULATION_COLLAPSED": "VALIDATION_FAILED", "HOLDOUT_ACCESS_DENIED": "HOLDOUT_ACCESS_DENIED",
    "DEST_ROOT_NOT_EMPTY": "VALIDATION_FAILED", "DEST_ROOT_UNSAFE": "VALIDATION_FAILED",
    "EVIDENCE_SCOPE_INCOMPLETE": "VALIDATION_FAILED", "TIER4_CACHE_STALE": "VALIDATION_FAILED",
    "STALE_EXPECTATION": "STALE_EXPECTATION",
}


@pytest.mark.parametrize("code,expected", sorted(DATA_TO_OPS.items()))
def test_data_failures_preserve_registry_policy_without_new_ops_codes(code, expected):
    assert set(DATA_TO_OPS) == set(DATA_FAILURE_CODES)
    problem = native_gate_worker.native_gate_data_failure(DataError(data_problem(code, "private input"))).problem
    assert problem.code == expected
    assert (problem.category, problem.retryable) == (FAILURE_CODES[code] if code in FAILURE_CODES
                                                    else DATA_FAILURE_CODES[code])
    if code in FAILURE_CODES:
        assert problem.code == code
    assert problem.details == {"data_failure_code": code}
    assert problem.message == "native gate data refused"


def test_evaluation_cannot_reuse_a_cross_run_cache(source):
    service = Service(source.conn, source.store.root, stages.registry(), TEST_POLICY,
                      clock=SystemClock(), code_source=REPO)
    service._store_domains = lambda claim: ()
    try:
        # An otherwise cache-eligible control proves the explicit kind guard matters.
        assert service._cache_allowed(SimpleNamespace(spec=SimpleNamespace(kind="artifact_check"))) is True
        assert service._cache_allowed(SimpleNamespace(spec=SimpleNamespace(kind="native_gate_evaluation"))) is False
    finally:
        service.close()


def test_real_corrupt_option_fragment_refuses_supervised_evaluation(source, tmp_path):
    _call(source)
    jobs = _call(source, sweep.submit_sweep)
    before = _scientific(source)
    fragment = source.repository.fragment_records(source.snapshot, "option_chains")[0]
    digest = fragment.object_ref.content_hash.removeprefix("sha256:")
    path = source.store.root / "objects" / digest[:2] / digest
    assert path.read_bytes().startswith(b"PAR1")
    path.unlink()  # Only this test's synthetic immutable fragment is replaced.
    path.write_bytes(b"corrupted synthetic parquet")
    for failed in _run_children(source, jobs, REPO):
        assert (failed["code"], failed["category"], failed["retryable"]) == ("INTEGRITY_FAILED", "integrity", False)
        assert failed["details"] == {}
        ref = artifact(source.conn, source.store, failed["diagnostic_ref"])
        assert json.loads(source.store.read_verified(ref)) == {"data_failure_code": "OBJECT_CORRUPT"}
    assert _scientific(source) == before
    assert not list(source.root.rglob("REPORT.md")) and not list(source.root.rglob("*.csv"))


def test_roster_and_submitted_input_artifacts_use_exact_schemas(source):
    _call(source)
    assert _row(source)["schema_version"] == "native_gate_sweep.v1.0"
    jobs = _call(source, sweep.submit_sweep)
    assert len(jobs) == 2
    for job in jobs:
        spec = json.loads(source.conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (job.job_id,)).fetchone()[0])
        bindings = spec["parameters"]["input_bindings"]
        assert set(bindings) == {"spec.json", "native_registration.json"}
        assert set(spec["input_refs"]) == set(bindings.values())
        for artifact_id in bindings.values():
            ref = artifact(source.conn, source.store, artifact_id)
            assert ref.schema_ref == "native_staged_input.v1.0"


def test_real_two_arm_children_consume_distinct_inputs_and_only_stage(source, tmp_path):
    checkout = _checkout(tmp_path)
    ledger = checkout / "experiments/lib.py"
    text = ledger.read_text()
    marker = "    if unique_by is not None and (not isinstance(unique_by, tuple) or not unique_by"
    assert text.count(marker) == 1
    _replace_source(ledger, text.replace(marker, "    raise AssertionError('synthetic child ledger access')\n" + marker))
    control = subprocess.run([sys.executable, "-c", "from pathlib import Path; import experiments.lib as lib; "
        "assert Path(lib.__file__).resolve() == Path('experiments/lib.py').resolve(); lib.ledger_append([])"],
        cwd=checkout, env={**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
                           "PYTHONPATH": str(checkout)}, capture_output=True, text=True, timeout=30)
    assert control.returncode == 1 and control.stdout == ""
    assert control.stderr.rstrip().endswith("AssertionError: synthetic child ledger access")
    identity = _call(source, code_root=checkout)
    seal, before = _row(source), _scientific(source)
    jobs = _call(source, sweep.submit_sweep, code_root=checkout)
    assert [job.job_id for job in _call(source, sweep.submit_sweep, code_root=checkout)] == [j.job_id for j in jobs]
    assert source.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
    _advance(source, _new_pin(source))
    assert source.repository.resolve_pinned("shadow").snapshot_id != source.snapshot.snapshot_id
    results = _run_children(source, jobs, checkout)
    assert identity == content_hash(seal)
    assert _row(source) == seal and _scientific(source) == before
    for result, alpha, cost, value, profit, saved in zip(results, (0., 1.), (3.8, 3.0), (5., 5.8), (1.2, 2.8),
                                                       seal["registrations"], strict=True):
        assert result["schema_version"] == "native_gate_result.v1.0"
        assert result["provenance"] == saved["binding"]
        assert result["variant_id"] == content_hash(saved["binding"])
        assert result["run_id"] == saved["run_id"]
        assert result["all_candidates"] == {"count": 4, "mean_mark_pnl": pytest.approx(profit)}
        assert result["availability"]["basis"] == "reconstructed_prior_completed_session"
        assert result["holdout_results"] == {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"}
        assert [row["event_id"] for row in result["predictions"]] == [e["event_id"] for e in source.events[4:]]
        for row in result["predictions"]:
            assert (row["entry_cost"], row["exit_value"], row["pnl"]) == pytest.approx((cost, value, profit))
        for fold in result["folds"]:
            for side in ("train", "test"):
                assert fold[side + "_inputs"]["fill_alpha"] == alpha
                assert fold[side + "_inputs"]["snapshot_id"] == source.snapshot.snapshot_id
                for history in fold[side + "_history"]:
                    assert history["feature_ceiling"] < history["gate_session"]
    assert results[0]["variant_id"] != results[1]["variant_id"]
    assert not list(source.root.rglob("*.csv"))
    assert not list(source.root.rglob("REPORT.md"))
    evidence = source.conn.execute("SELECT evidence_json FROM experiment_runs ORDER BY run_id").fetchall()
    assert len(evidence) == 2
    assert all(not {"native_smoke", "native_outcome"}.intersection(json.loads(row["evidence_json"])) for row in evidence)
    assert [job.job_id for job in _call(source, sweep.submit_sweep, code_root=checkout)] == [j.job_id for j in jobs]
    assert source.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


def test_earlier_nonroster_arm_stays_independent_after_sealing(source, monkeypatch, capsys):
    spec = build_gate_spec(0.5, "independent")
    binding = dict(code_root=REPO, as_of_month=MONTH, scope="shadow", event_ids=None)
    registered = native_registration.register_native(source.conn, source.store, source.repository, spec, **binding)
    assert native_registration.register_native(source.conn, source.store, source.repository, spec, **binding) == registered
    original_bytes = registered.binding_json
    identity = _call(source)
    seal = _row(source)
    roster_ids = {item["run_id"] for item in seal["registrations"]}
    assert len(roster_ids) == 2 and registered.run_id not in roster_ids
    assert len(_scientific(source)) == 3
    assert native_registration.read_native_registration(source.conn, source.store, spec).binding_json == original_bytes
    assert native_registration.require_native_registration(source.conn, source.store, source.repository,
                                                           spec, expected=registered, **binding) == registered
    # All subsequent reservation calls obey the seal, even replay of an earlier independent arm.
    later_spec = build_gate_spec(0.5, "later")
    later, later_spec = native_registration.prepare_native_registration(source.repository, later_spec, **binding)
    before = list(source.conn.iterdump())
    with monkeypatch.context() as patch:
        patch.setattr(source.store, "publish_bytes", _forbid)
        for reservation, requested in ((registered, spec), (later, later_spec)):
            with pytest.raises(OpsError) as caught:
                native_registration.reserve_native_registration(source.conn, source.store, reservation, requested)
            assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert list(source.conn.iterdump()) == before
    assert _call(source) == identity and _row(source) == seal
    jobs = _call(source, sweep.submit_sweep)
    submitted = [json.loads(row[0])["parameters"]["run_id"] for row in
                 source.conn.execute("SELECT spec_json FROM jobs")]
    assert set(submitted) == roster_ids and len(jobs) == len(submitted) == 2
    roster_rows = [row for row in _scientific(source) if row[0] in roster_ids]
    assert len(roster_rows) == 2
    # Actual single-arm CLI evaluation and outcome replay still work without adopting it into the sweep.
    spec_path = source.root / "independent-spec.json"
    spec_path.write_text(canonical_json(asdict(spec)))
    args = ["run", "--spec", str(spec_path), "--catalog", str(source.root / "catalog.sqlite"),
            "--store-root", str(source.store.root), "--as-of-month", MONTH, "--no-ledger",
            "--exp-number", "101", "--experiments-root", str(source.root / "experiments")]
    assert gate_cli.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["variant_id"] == registered.variant_id and output["recording_mode"] == "smoke"
    report = source.root.joinpath(output["report"]).read_text()
    assert report.startswith("# Exploratory reconstructed-history gate report\n")
    after = _scientific(source)
    monkeypatch.setattr(gate_report, "gate_result", _forbid)
    assert gate_cli.main(args) == 0
    replay = json.loads(capsys.readouterr().out)
    assert output["record_export"]["status"] == "written"
    assert replay["record_export"]["status"] == "unchanged"
    assert replay["recording_mode"] == "smoke"
    assert replay["variant_id"] == registered.variant_id
    assert {key: value for key, value in replay.items() if key != "record_export"} == {
        key: value for key, value in output.items() if key != "record_export"}
    assert _scientific(source) == after and _row(source) == seal
    assert [row for row in after if row[0] in roster_ids] == roster_rows
    assert source.root.joinpath(output["report"]).read_text() == report
    assert native_registration.read_native_registration(source.conn, source.store, spec).binding_json == original_bytes
    assert not list(source.root.rglob("*.csv"))


def test_pre_sweep_gate_binding_replays_without_rewriting_identity(source, tmp_path, monkeypatch, capsys):
    checkout = _checkout(tmp_path)
    spec = build_gate_spec()
    prepared, spec = native_registration.prepare_native_registration(source.repository, spec, code_root=checkout,
        as_of_month=MONTH, scope="shadow", event_ids=None)
    # Persist the pre-sweep closure rule independently of the registrar under test.
    original = prepared.document
    original["source_closure"] = source_closure(checkout, [gate_report.RUNNER])
    assert "experiments/native_gate_entrypoint.py" not in original["source_closure"]
    registered = native_registration.NativeRegistration(prepared.run_id, canonical_json(original).encode())
    native_registration.reserve_native_registration(source.conn, source.store, registered, spec)
    result = gate_report.gate_result(source.repository, registered, spec)
    report = gate_report.render_gate_report(registered, result, spec=spec, no_ledger=True)
    native_outcomes.publish_native_outcome(source.conn, source.store, registered,
                                           report=report, attempted=True, no_ledger=True)
    before = _scientific(source)
    spec_path = source.root / "spec.json"
    spec_path.write_text(canonical_json(asdict(spec)))
    monkeypatch.setattr(gate_cli, "__file__", str(checkout / gate_report.RUNNER))
    monkeypatch.setattr(gate_report, "gate_result", _forbid)
    args = ["run", "--spec", str(spec_path), "--catalog", str(source.root / "catalog.sqlite"),
            "--store-root", str(source.store.root), "--as-of-month", MONTH, "--no-ledger",
            "--exp-number", "101", "--experiments-root", str(source.root / "experiments")]
    assert gate_cli.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["variant_id"] == registered.variant_id and output["recording_mode"] == "smoke"
    assert source.root.joinpath(output["report"]).read_text() == report
    assert _scientific(source) == before
    assert native_registration.read_native_registration(source.conn, source.store, spec) == registered
    assert not list(source.root.rglob("*.csv"))
    runner = checkout / gate_report.RUNNER
    _replace_source(runner, runner.read_text() + "\n# synthetic source drift\n")
    assert gate_cli.main(args) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _scientific(source) == before
    assert native_registration.read_native_registration(source.conn, source.store, spec) == registered


@pytest.mark.parametrize("failure", ["typed", "unexpected"])
def test_real_failed_child_has_no_output_or_scientific_completion(source, tmp_path, failure):
    checkout = _checkout(tmp_path)
    path = checkout / "experiments/native_gate_evaluation.py"
    injected = ('raise fail("EXPERIMENT_VARIANT_FAILED", "synthetic evaluation failure")' if failure == "typed"
                else 'raise RuntimeError("synthetic unexpected failure")')
    _replace_source(path, path.read_text().replace("            evaluating = True", "            evaluating = True\n"
                    "            if spec.primary_arm_id == 'candidate':\n                " + injected))
    _call(source, code_root=checkout)
    before = _scientific(source)
    jobs = _call(source, sweep.submit_sweep, code_root=checkout)
    both = failure == "typed"
    results = _run_children(source, jobs, checkout, only=None if both else jobs[:1])
    failed = results[0]
    if both:
        sibling = results[1]
        assert sibling["schema_version"] == "native_gate_result.v1.0"
    assert failed["code"] == ("EXPERIMENT_VARIANT_FAILED" if failure == "typed" else "WORKER_FAILED")
    assert _scientific(source) == before
    assert set(_row(source)) == {"schema_version", "request", "snapshot", "registrations", "coordinator_source", "implementation_ref"}
    assert not list(source.root.rglob("REPORT.md")) and not list(source.root.rglob("LEDGER.csv"))


@pytest.mark.parametrize("exception,code,category,retryable,detail", [
    ('data_fail("HOLDOUT_ACCESS_DENIED", "synthetic private input")',
     "HOLDOUT_ACCESS_DENIED", "validation", False, "HOLDOUT_ACCESS_DENIED"),
    ('data_fail("SNAPSHOT_UNRESOLVED", "synthetic private input")',
     "SNAPSHOT_UNRESOLVED", "dependency", False, "SNAPSHOT_UNRESOLVED"),
    ('data_fail("RESOURCE_UNAVAILABLE", "synthetic private input")',
     "RESOURCE_UNAVAILABLE", "resource", True, "RESOURCE_UNAVAILABLE"),
    ('data_fail("INPUT_CHANGED", "synthetic private input")',
     "INPUT_CHANGED", "dependency", False, "INPUT_CHANGED"),
    ('data_fail("OBJECT_CORRUPT", "synthetic private input")',
     "INTEGRITY_FAILED", "integrity", False, "OBJECT_CORRUPT"),
    ('ValueError("synthetic private input")',
     "EXPERIMENT_VARIANT_FAILED", "internal", False, "ValueError"),
    ('OSError("synthetic private input")',
     "INPUT_CHANGED", "dependency", False, "OSError"),
    ('ArtifactError("HASH_MISMATCH", "synthetic private input")',
     "INTEGRITY_FAILED", "integrity", False, "HASH_MISMATCH"),
], ids=["holdout", "snapshot", "retryable-data", "operations-retry-policy", "data-only", "value", "io", "artifact"])
def test_supervised_evaluation_input_failures_preserve_exact_problem(
        source, tmp_path, exception, code, category, retryable, detail):
    """Inject at the real feature-read stage, inside gate_result and child admission."""
    checkout = _checkout(tmp_path)
    path = checkout / "experiments/gate_report.py"
    text = ("from engine.v2.data.errors import fail as data_fail\nfrom engine.v2.foundation import ArtifactError\n"
            + path.read_text())
    marker = "    values, inputs = [], []"
    assert text.count(marker) == 1
    _replace_source(path, text.replace(marker, "    if binding['execution_plan']['arms'] == ['candidate']:\n"
                                      "        raise " + exception + "\n" + marker))
    _call(source, code_root=checkout)
    before = _scientific(source)
    jobs = _call(source, sweep.submit_sweep, code_root=checkout)
    both = detail == "HOLDOUT_ACCESS_DENIED"
    results = _run_children(source, jobs, checkout, only=None if both else jobs[:1])
    failed = results[0]
    if both:
        sibling = results[1]
        assert sibling["schema_version"] == "native_gate_result.v1.0"
    assert (failed["code"], failed["category"], failed["retryable"]) == (code, category, retryable)
    assert failed["details"] == {}
    assert "synthetic private input" not in canonical_json(failed) and "data_failure_code" not in canonical_json(failed)
    diagnostic = artifact(source.conn, source.store, failed["diagnostic_ref"])
    assert json.loads(source.store.read_verified(diagnostic)) == {"data_failure_code": detail}
    assert _scientific(source) == before
    assert not list(source.root.rglob("REPORT.md")) and not list(source.root.rglob("LEDGER.csv"))


@pytest.mark.parametrize("phase", ["admission", "evaluation"])
@pytest.mark.parametrize("damage,code,category,detail", [
    ("corrupt", "INTEGRITY_FAILED", "integrity", "INTEGRITY_FAILED"),
    ("missing", "INTEGRITY_FAILED", "integrity", "MISSING"),
    ("file", "INPUT_CHANGED", "dependency", "FileNotFoundError"),
    ("sqlite", "INPUT_CHANGED", "dependency", "OperationalError"),
])
def test_real_child_dependency_read_preserves_exact_code(source, tmp_path, phase, damage, code, category, detail):
    """Fault injection performs an actual store/file/SQLite read, not a raised mock error."""
    payload = b"synthetic private artifact"
    ref = source.store.publish_bytes(payload, schema_ref="synthetic_probe.v1.0")
    assert source.store.read_verified(ref) == payload
    checkout = _checkout(tmp_path)
    path = checkout / ("experiments/native_gate_worker.py" if phase == "admission" else
                       "experiments/native_gate_evaluation.py")
    text = "from engine.v2.contracts import ArtifactRef\n" + path.read_text()
    marker = ("    verify_native_registration(conn, store, registration)" if phase == "admission" else
              "            evaluating = True")
    read = ("(root / 'missing-input').read_bytes()" if damage == "file" else
            "sqlite3.connect((root / 'missing-catalog').as_uri() + '?mode=ro', uri=True)" if damage == "sqlite" else
            f"store.read_verified(from_document(ArtifactRef, {asdict(ref)!r}))")
    indent = "    " if phase == "admission" else "            "
    fault = f"{indent}if spec.primary_arm_id == 'candidate':\n{indent}    {read}\n"
    assert text.count(marker) == 1
    _replace_source(path, text.replace(marker, fault + marker if phase == "admission" else marker + "\n" + fault))
    _call(source, code_root=checkout)
    jobs = _call(source, sweep.submit_sweep, code_root=checkout)
    before = _scientific(source)
    target = source.store.root / ref.storage_key
    target.unlink()  # Replace only this test's immutable synthetic object.
    if damage == "corrupt":
        target.write_bytes(b"corrupted private bytes")
    both = phase == "admission" and damage == "corrupt"
    results = _run_children(source, jobs, checkout, only=None if both else jobs[:1])
    failed = results[0]
    if both:
        sibling = results[1]
        assert sibling["schema_version"] == "native_gate_result.v1.0"
    assert (failed["code"], failed["category"], failed["retryable"]) == (code, category, False)
    assert failed["details"] == {} and ref.storage_key not in canonical_json(failed)
    diagnostic = artifact(source.conn, source.store, failed["diagnostic_ref"])
    assert json.loads(source.store.read_verified(diagnostic)) == {"data_failure_code": detail}
    assert _scientific(source) == before
    assert not list(source.root.rglob("REPORT.md")) and not list(source.root.rglob("*.csv"))


@pytest.mark.parametrize("exception", ["ValueError", "TypeError", "RecursionError"])
def test_supervised_receipt_serialization_failures_are_variant_failures(source, tmp_path, exception):
    checkout = _checkout(tmp_path)
    # Real folds/pricing are covered by the two-arm integration test above.
    # Both children keep real admission, serialization and supervisor handling;
    # repeating fitting here only delays the boundary this test exercises.
    evaluation = checkout / "experiments/native_gate_evaluation.py"
    evaluation_text = evaluation.read_text()
    calculation = "            result = gate_result(Repository(conn, store, pin_snapshots=True), registration, spec)"
    assert evaluation_text.count(calculation) == 1
    synthetic_result = ('            result = {"schema_version": "native_gate_result.v1.0", '
                        '"run_id": registration.run_id, "variant_id": registration.variant_id, '
                        '"attempted_variants": 1, "provenance": registration.document}')
    _replace_source(evaluation, evaluation_text.replace(calculation, synthetic_result))
    path = checkout / "experiments/native_gate_worker.py"
    text = path.read_text()
    marker = "    directory = Path(mkdtemp(prefix=\".native-gate-\", dir=path.parent))"
    assert text.count(marker) == 1
    fault = ("    class SyntheticOutput:\n        def __str__(self):\n"
             f"            raise {exception}('synthetic serialization failure')\n"
             "    if receipt['provenance']['execution_plan']['arms'] == ['candidate']:\n"
             "        receipt['synthetic_unserializable'] = SyntheticOutput()\n")
    _replace_source(path, text.replace(marker, fault + marker))
    _call(source, code_root=checkout)
    before = _scientific(source)
    failed, sibling = _run_children(source, _call(source, sweep.submit_sweep, code_root=checkout), checkout)
    assert (failed["code"], failed["category"], failed["retryable"]) == ("EXPERIMENT_VARIANT_FAILED", "internal", False)
    assert failed["details"] == {} and "synthetic serialization failure" not in canonical_json(failed)
    diagnostic = artifact(source.conn, source.store, failed["diagnostic_ref"])
    assert json.loads(source.store.read_verified(diagnostic)) == {"data_failure_code": exception}
    assert sibling["schema_version"] == "native_gate_result.v1.0" and _scientific(source) == before
    control = _row(source)["registrations"][1]
    assert (sibling["run_id"], sibling["variant_id"]) == (control["run_id"], content_hash(control["binding"]))
    assert sibling["provenance"] == control["binding"] and sibling["attempted_variants"] == 1
    assert not list(source.root.rglob("REPORT.md")) and not list(source.root.rglob("*.csv"))


def test_common_pin_survives_head_movement_and_partial_reservation(source, monkeypatch):
    newer = _new_pin(source)
    original, calls = native_registration.prepare_native_registration, []
    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result[0])
        if len(calls) == 1:
            _advance(source, newer)
        return result
    reserve = native_registration.reserve_native_registration
    def interrupt(conn, store, registration, spec):
        if spec.primary_arm_id == "control":
            raise OSError("synthetic reservation interruption")
        return reserve(conn, store, registration, spec)
    with monkeypatch.context() as patch:
        patch.setattr(native_registration, "prepare_native_registration", prepare)
        patch.setattr(native_registration, "reserve_native_registration", interrupt)
        with pytest.raises(OpsError) as caught:
            _call(source)
        assert caught.value.code == "INPUT_CHANGED"
        assert caught.value.problem.details == {"data_failure_code": "OSError"}
    assert len(calls) == 2 and {r.document["snapshot"]["snapshot_id"] for r in calls} == {source.snapshot.snapshot_id}
    assert source.repository.resolve_pinned("shadow") == newer
    seal, before = _row(source), _scientific(source)
    assert len(before) == 1
    with pytest.raises(OpsError) as caught:
        _call(source, sweep.submit_sweep)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    assert _call(source) == content_hash(seal)
    assert _row(source) == seal and len(_scientific(source)) == 2
    assert len(_call(source, sweep.submit_sweep)) == 2
    # The ordinary API still binds the current head, independently of the sweep seam.
    ordinary = native_registration.register_native(source.conn, source.store,
        type(source.repository)(source.conn, source.store), replace(build_gate_spec(), experiment_id="ordinary"),
        code_root=REPO, as_of_month=MONTH)
    assert ordinary.document["snapshot"]["snapshot_id"] == newer.snapshot_id


@pytest.mark.parametrize("point", ["publication", "after_seal", "after_first_arm"])
def test_registration_retry_completes_exact_interrupted_seal(source, monkeypatch, point):
    original = native_registration.reserve_native_registration
    def interrupted(conn, store, registration, spec):
        if point == "after_seal" or point == "after_first_arm" and spec.primary_arm_id == "control":
            raise OSError("synthetic interruption")
        return original(conn, store, registration, spec)
    with monkeypatch.context() as patch:
        if point == "publication":
            patch.setattr(source.store, "publish_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError("publication")))
        else:
            patch.setattr(native_registration, "reserve_native_registration", interrupted)
        with pytest.raises(OpsError) as caught:
            _call(source)
        assert caught.value.code == "INPUT_CHANGED"
        assert caught.value.problem.details == {"data_failure_code": "OSError"}
    seal = _row(source)
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert _call(source) == content_hash(seal)
    before = _scientific(source)
    assert len(before) == 2 and _call(source) == content_hash(seal) and _scientific(source) == before


@pytest.mark.parametrize("change", ["empty", "duplicate", "experiment", "hypothesis", "strategy", "folds", "seed",
                                    "runner", "unconsumed", "pair", "self_pair", "three_pair", "schema", "extra"])
def test_closed_roster_refuses_before_reads_or_reservation(source, monkeypatch, change):
    document = _request()
    arm = document["variants"][1]
    if change == "empty":
        document["variants"] = []
    elif change == "duplicate":
        arm.update(primary_arm_id="candidate", arms=["candidate"])
    elif change == "experiment":
        arm["experiment_id"] = "different"
    elif change == "hypothesis":
        arm["hypothesis"] = "different"
    elif change == "strategy":
        arm["economic_params"]["strategy"] = "STR-RUNUP"
    elif change == "folds":
        arm["folds"] = ["2024"]
    elif change == "seed":
        arm["seed"] = 7
    elif change == "runner":
        arm["runner"] = "experiments/new_experiment.py"
    elif change == "unconsumed":
        arm["economic_params"]["exit"] = "unused"
    elif change == "schema":
        document["schema_version"] = "native_gate_sweep_request.v0.0"
    elif change == "self_pair":
        document["primary_comparison"] = ["candidate", "candidate"]
    elif change == "three_pair":
        document["primary_comparison"] = ["candidate", "control", "candidate"]
    elif change == "pair":
        document["primary_comparison"] = ["candidate", "absent"]
    else:
        document["undeclared"] = True
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    with pytest.raises(OpsError) as caught:
        _call(source, document=document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert _scientific(source) == []
    assert source.conn.execute("SELECT COUNT(*) FROM native_gate_sweeps").fetchone()[0] == 0


@pytest.mark.parametrize("field", ["environment", "holdouts", "snapshot", "event_ids", "source_closure"])
def test_cross_arm_context_mismatch_refuses_before_publication(source, monkeypatch, field):
    original = native_registration.prepare_native_registration
    def prepare(repository, spec, **kwargs):
        registered, spec = original(repository, spec, **kwargs)
        if spec.primary_arm_id == "control":
            binding = registered.document
            if field == "event_ids":
                binding[field] = binding[field][:-1]
            else:
                binding[field]["synthetic_drift"] = True
            registered = native_registration.NativeRegistration(registered.run_id, canonical_json(binding).encode())
        return registered, spec
    before = list(source.conn.iterdump())
    monkeypatch.setattr(native_registration, "prepare_native_registration", prepare)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        _call(source)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert list(source.conn.iterdump()) == before


def test_canonical_order_and_conflicting_replay_preserve_all_bytes(source):
    ids = [event["event_id"] for event in source.events]
    identity = _call(source, event_ids=ids)
    seal, before = _row(source), _scientific(source)
    document = _request()
    document["variants"].reverse()
    assert _call(source, document=document, event_ids=list(reversed(ids))) == identity
    document["variants"][0]["economic_params"]["fill"] = 0.5
    with pytest.raises(OpsError) as caught:
        _call(source, document=document, event_ids=ids)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _row(source) == seal and _scientific(source) == before


def test_reversed_primary_comparison_is_a_distinct_request(source):
    document = _request()
    # Seal the nonlexical order first so sorting the pair cannot satisfy this oracle.
    document["primary_comparison"] = ["control", "candidate"]
    identity = _call(source, document=document)
    seal, before = _row(source), list(source.conn.iterdump())
    assert seal["request"]["primary_comparison"] == ["control", "candidate"]
    assert _call(source, document=document) == identity
    document["primary_comparison"].reverse()
    with pytest.raises(OpsError) as caught:
        _call(source, document=document)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _row(source) == seal and list(source.conn.iterdump()) == before


@pytest.mark.parametrize("statement", ["UPDATE native_gate_sweeps SET document_json='{}'",
    "DELETE FROM native_gate_sweeps", "INSERT OR REPLACE INTO native_gate_sweeps SELECT * FROM native_gate_sweeps"])
def test_roster_is_database_immutable(source, statement):
    _call(source)
    before = _row(source)
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        source.conn.execute(statement)
    assert _row(source) == before


@pytest.mark.parametrize("protected", [0, 1])
def test_explicit_holdout_refuses_whole_roster_without_reservation(source, protected):
    with pytest.raises(DataError) as caught:
        _call(source, event_ids=[source.events[0]["event_id"], source.protected[protected]["event_id"]])
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    assert _scientific(source) == [] and source.conn.execute("SELECT COUNT(*) FROM native_gate_sweeps").fetchone()[0] == 0


@pytest.mark.parametrize("action", [sweep.register_sweep, sweep.submit_sweep])
@pytest.mark.parametrize("membership", ["random", "rolling", "unknown", "eligible"])
def test_sealed_membership_change_admits_r5_before_final_r6(source, monkeypatch, action, membership):
    _call(source)
    before, seal = _scientific(source), _row(source)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    key = {"random": source.protected[0]["event_id"], "rolling": source.protected[1]["event_id"],
           "unknown": "unknown-event", "eligible": source.events[0]["event_id"]}[membership]
    with pytest.raises((OpsError, DataError)) as caught:
        _call(source, action, event_ids=[key])
    assert caught.value.code == ("EXPERIMENT_IDENTITY_CONFLICT" if membership == "eligible" else "HOLDOUT_ACCESS_DENIED")
    assert _scientific(source) == before and _row(source) == seal
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_atomic_graph_conflict_in_second_child_rolls_back_first(source, monkeypatch):
    _call(source)
    original = sweep.submit_graph
    def conflict(conn, registry, policy, requests, *, clock, receipt_decode_failure_code, max_failure_bytes):
        # A genuinely conflicting existing second request exercises the real SQL transaction rollback.
        assert receipt_decode_failure_code is None and max_failure_bytes is None
        other = replace(requests[1], job=replace(requests[1].job, priority=7))
        original(conn, registry, policy, [other], clock=clock, receipt_decode_failure_code=receipt_decode_failure_code,
                 max_failure_bytes=max_failure_bytes)
        return original(conn, registry, policy, requests, clock=clock,
                        receipt_decode_failure_code=receipt_decode_failure_code, max_failure_bytes=max_failure_bytes)
    monkeypatch.setattr(sweep, "submit_graph", conflict)
    with pytest.raises(OpsError) as caught:
        _call(source, sweep.submit_sweep)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.message == "sweep submission differs from its first request"
    rows = source.conn.execute("SELECT spec_json FROM jobs").fetchall()
    assert len(rows) == 1 and json.loads(rows[0][0])["priority"] == 7


def test_corrupt_arm_evidence_refuses_before_any_staging_or_job(source, monkeypatch):
    _call(source)
    run_id = _row(source)["registrations"][1]["run_id"]
    source.conn.execute("UPDATE experiment_runs SET evidence_json='{}' WHERE run_id=?", (run_id,))
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        _call(source, sweep.submit_sweep)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_preparation_artifact_read_preserves_integrity_code(source, monkeypatch, sealed, damage):
    if sealed:
        _call(source)
    payload = b"synthetic preparation dependency"
    ref = source.store.publish_bytes(payload, schema_ref="synthetic_probe.v1.0")
    assert source.store.read_verified(ref) == payload
    resolve = source.repository.resolve
    def verify_then_resolve(snapshot_id):
        # Inject a genuine foundation read at the repository dependency boundary.
        source.store.read_verified(ref)
        return resolve(snapshot_id)
    monkeypatch.setattr(source.repository, "resolve", verify_then_resolve)
    before, changes = _scientific(source), source.conn.total_changes
    target = source.store.root / ref.storage_key
    target.unlink()  # Replace only this test's immutable synthetic object.
    if damage == "corrupt":
        target.write_bytes(b"corrupted preparation dependency")
    with monkeypatch.context() as patch:
        patch.setattr(source.store, "publish_bytes", _forbid)
        with pytest.raises(OpsError) as caught:
            _call(source)
    problem = caught.value.problem
    assert (problem.code, problem.category, problem.retryable) == ("INTEGRITY_FAILED", "integrity", False)
    assert problem.details == {"data_failure_code": "INTEGRITY_FAILED" if damage == "corrupt" else "MISSING"}
    assert ref.storage_key not in problem.message and _scientific(source) == before
    assert source.conn.total_changes == changes
    assert source.conn.execute("SELECT COUNT(*) FROM native_gate_sweeps").fetchone()[0] == int(sealed)
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    target.write_bytes(payload)
    assert _call(source) == content_hash(_row(source))


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("failure", ["file", "sqlite", "typed"])
def test_preparation_io_failure_is_not_spec_or_identity_change(source, monkeypatch, sealed, failure):
    if sealed:
        _call(source)
    before = list(source.conn.iterdump())
    resolve = source.repository.resolve
    def unavailable(snapshot_id):
        if failure == "file":
            (source.root / "missing-input").read_bytes()
        elif failure == "sqlite":
            sqlite3.connect((source.root / "missing-catalog").as_uri() + "?mode=ro", uri=True)
        else:
            from engine.v2.ops.errors import fail
            raise fail("INPUT_CHANGED", "synthetic typed unavailability")
        return resolve(snapshot_id)
    with monkeypatch.context() as patch:
        patch.setattr(source.repository, "resolve", unavailable)
        patch.setattr(source.store, "publish_bytes", _forbid)
        with pytest.raises(OpsError) as caught:
            _call(source)
    problem = caught.value.problem
    assert (problem.code, problem.category, problem.retryable) == ("INPUT_CHANGED", "dependency", False)
    assert problem.details == ({} if failure == "typed" else {
        "data_failure_code": "FileNotFoundError" if failure == "file" else "OperationalError"})
    assert list(source.conn.iterdump()) == before
    assert _call(source) == content_hash(_row(source))


def test_preparation_catalog_read_failure_is_input_changed(source):
    before = list(source.conn.iterdump())
    source.conn.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ else sqlite3.SQLITE_OK)
    try:
        with pytest.raises(OpsError) as caught:
            _call(source)
    finally:
        source.conn.set_authorizer(None)
    assert caught.value.code == "INPUT_CHANGED"
    assert caught.value.problem.details == {"data_failure_code": "DatabaseError"}
    assert list(source.conn.iterdump()) == before


@pytest.mark.parametrize("phase", ["seal", "reserve_sql", "reserve_file", "stage_sql", "stage_file", "graph"])
def test_public_api_io_failures_are_typed_and_recoverable(source, monkeypatch, phase):
    submitting = phase.startswith("stage") or phase == "graph"
    if submitting:
        _call(source)
    before = _scientific(source)
    tables = {"seal": "native_gate_sweeps", "reserve_sql": "hypotheses", "stage_sql": "artifacts", "graph": "jobs"}
    with monkeypatch.context() as patch:
        if phase.endswith("file"):
            def unavailable(*args, **kwargs):
                return (source.root / "missing-private-input").read_bytes()
            patch.setattr(source.store, "publish_bytes", unavailable)
        else:
            source.conn.set_authorizer(lambda op, table, *args:
                sqlite3.SQLITE_DENY if op == sqlite3.SQLITE_INSERT and table == tables[phase] else sqlite3.SQLITE_OK)
        try:
            with pytest.raises(OpsError) as caught:
                _call(source, sweep.submit_sweep if submitting else sweep.register_sweep)
        finally:
            source.conn.set_authorizer(None)
    problem = caught.value.problem
    assert (problem.code, problem.category, problem.retryable) == ("INPUT_CHANGED", "dependency", False)
    assert problem.details == {"data_failure_code": "FileNotFoundError" if phase.endswith("file") else "DatabaseError"}
    assert "missing-private-input" not in problem.message and _scientific(source) == before
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert source.conn.execute("SELECT COUNT(*) FROM native_gate_sweeps").fetchone()[0] == (phase != "seal")
    identity = _call(source)
    assert _call(source) == identity
    jobs = _call(source, sweep.submit_sweep)
    assert len(jobs) == 2 and [job.job_id for job in _call(source, sweep.submit_sweep)] == [job.job_id for job in jobs]


@pytest.mark.parametrize("bad", [None, 5])
def test_non_path_code_root_is_a_typed_refusal(source, bad):
    with pytest.raises(OpsError) as raised:
        _call(source, code_root=bad)
    assert raised.value.code == "INVALID_EXPERIMENT_SPEC"


def test_str_code_root_is_coerced_to_the_same_sealed_identity(source, tmp_path):
    checkout = _checkout(tmp_path)
    identity = _call(source, code_root=str(checkout))
    assert identity == content_hash(_row(source))
    assert identity == _call(source, code_root=checkout)


@pytest.mark.parametrize("action", [sweep.register_sweep, sweep.submit_sweep])
def test_non_string_event_id_refuses_before_catalog_or_population(source, monkeypatch, action):
    before = list(source.conn.iterdump())
    monkeypatch.setattr(sweep, "require_current", _forbid)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    with pytest.raises(OpsError) as caught:
        _call(source, action, event_ids=[17])
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert list(source.conn.iterdump()) == before


@pytest.mark.parametrize("damage,code", [("corrupt", "INTEGRITY_FAILED"), ("symlink", "UNSAFE_PATH")])
def test_cli_corrupt_staged_object_refuses_exactly_then_recovers(source, capsys, damage, code):
    with observe_scan_outcomes({"earnings_events", "daily_market"}) as setup_scans:
        _call(source)
    setup = capsys.readouterr()
    assert setup.out == ""
    assert_scan_boundaries(setup.err, tables={"earnings_events", "daily_market"},
                           expected_scans=setup_scans, required_tables={"earnings_events"})
    payload = canonical_json(asdict(build_gate_spec())).encode()
    ref = source.store.publish_bytes(payload, schema_ref="native_staged_input.v1.0")
    target = source.store.root / ref.storage_key
    target.unlink()
    foreign = source.root / "foreign-object"
    foreign.write_bytes(payload)
    if damage == "corrupt":
        target.write_bytes(b"corrupted staged specification")
    else:
        target.symlink_to(foreign)
    before = list(source.conn.iterdump())
    roster = source.root / "roster.json"
    roster.write_text(json.dumps(_request()))
    args = ["submit", "--roster", str(roster), "--catalog", str(source.root / "catalog.sqlite"),
            "--store-root", str(source.store.root), "--as-of-month", MONTH, "--code-root", str(REPO), "--no-ledger"]
    with observe_scan_outcomes({"earnings_events", "daily_market"}) as scans:
        assert sweep.main(args) == 2
    captured = capsys.readouterr()
    assert_scan_boundaries(captured.err, tables={"earnings_events", "daily_market"},
                           expected_scans=scans, required_tables={"earnings_events"})
    assert json.loads(captured.out) == {"refused": code}
    assert list(source.conn.iterdump()) == before
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert foreign.read_bytes() == payload
    target.unlink()
    target.write_bytes(payload)
    assert sweep.main(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "state": "submitted_uncollected", "job_ids": [job.job_id for job in _call(source, sweep.submit_sweep)]}


@pytest.mark.parametrize("compatible", [True, False])
@pytest.mark.parametrize("arm,fill", [("candidate", 0.0), ("control", 1.0)])
def test_direct_arm_reservation_must_match_the_roster(source, monkeypatch, compatible, arm, fill):
    direct = native_registration.register_native(source.conn, source.store, source.repository,
        build_gate_spec(fill if compatible else 0.5, arm), code_root=REPO, as_of_month=MONTH)
    before = _scientific(source)
    if compatible:
        _call(source)
        assert next(item["binding"] for item in _row(source)["registrations"]
                    if item["run_id"] == direct.run_id) == direct.document
        assert len(_call(source, sweep.submit_sweep)) == 2
    else:
        with monkeypatch.context() as patch:
            patch.setattr(source.store, "publish_bytes", _forbid)
            with pytest.raises(OpsError) as caught:
                _call(source)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert _scientific(source) == before
        assert source.conn.execute("SELECT COUNT(*) FROM native_gate_sweeps").fetchone()[0] == 0
        with pytest.raises(OpsError) as caught:
            _call(source, sweep.submit_sweep)
        assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
        assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        # Refusal has not burned the experiment ID: a compatible roster still seals.
        corrected = _request()
        next(item for item in corrected["variants"] if item["primary_arm_id"] == arm)["economic_params"]["fill"] = 0.5
        _call(source, document=corrected)
        assert len(_scientific(source)) == 2


@pytest.mark.parametrize("action", ["register", "submit"])
def test_version_twelve_catalog_refuses_without_migration_or_population_reads(tmp_path, capsys, monkeypatch, action):
    path, store = tmp_path / "old.sqlite", ArtifactStore(tmp_path / "store")
    conn = connect(path)
    try:
        migrate(conn, schema.OWNER, schema.MIGRATIONS[:12], clock=FakeClock())
        before = list(conn.iterdump())
        monkeypatch.setattr(native_registration, "prepare_native_registration", _forbid)
        entry = sweep.register_sweep if action == "register" else sweep.submit_sweep
        with pytest.raises(OpsError) as caught:
            entry(conn, store, None, _request(), code_root=REPO, as_of_month=MONTH)
        assert caught.value.code == "INTEGRITY_FAILED"
        assert caught.value.problem.details == {"reason": "schema_older", "owner": "ops", "missing": [13]}
        roster = tmp_path / "roster.json"
        roster.write_text(json.dumps(_request()))
        assert sweep.main([action, "--roster", str(roster), "--catalog", str(path),
                           "--store-root", str(store.root), "--as-of-month", MONTH,
                           "--code-root", str(REPO), "--no-ledger"]) == 2
        assert json.loads(capsys.readouterr().out) == {"refused": "INTEGRITY_FAILED"}
        assert list(conn.iterdump()) == before
        assert not store.root.exists()
    finally:
        conn.close()


def test_non_native_primary_reservation_refuses_before_sealing(source, monkeypatch):
    register_hypothesis(source.conn, build_gate_spec(1.0, "control"), "other-input", mode="primary", run_id="non-native")
    before = _scientific(source)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        _call(source)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _scientific(source) == before
    assert source.conn.execute("SELECT COUNT(*) FROM native_gate_sweeps").fetchone()[0] == 0
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_native_reservation_requires_canonical_sealed_binding(source, tmp_path, monkeypatch):
    (tmp_path / "prepared").mkdir()
    prepared = build_gate_source(tmp_path / "prepared")
    try:
        _call(prepared)
        sealed = _row(prepared)
    finally:
        prepared.conn.close()
    sealed["registrations"][0]["binding"]["execution_plan"]["seed"] = False
    source.conn.execute("INSERT INTO native_gate_sweeps VALUES (?,?)",
                       (build_gate_spec().experiment_id, canonical_json(sealed)))
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        native_registration.register_native(source.conn, source.store, source.repository,
            build_gate_spec(), code_root=REPO, as_of_month=MONTH)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and _scientific(source) == []


@pytest.mark.parametrize("timing", ["before_publication", "during_publication"])
def test_seal_wins_direct_registration_race_without_stranding_roster(source, monkeypatch, timing):
    publish = source.store.publish_bytes
    sealed_state = []
    def seal():
        def interrupt(*args, **kwargs):
            raise OSError("synthetic stop after seal")
        with monkeypatch.context() as patch:
            patch.setattr(source.store, "publish_bytes", publish)
            patch.setattr(native_registration, "reserve_native_registration", interrupt)
            with pytest.raises(OpsError) as caught:
                _call(source)
            assert caught.value.code == "INPUT_CHANGED"
            assert caught.value.problem.details == {"data_failure_code": "OSError"}
        sealed_state.extend(_scientific(source))
    if timing == "before_publication":
        seal()
        monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    else:
        def publish_after_seal(*args, **kwargs):
            seal()
            return publish(*args, **kwargs)
        monkeypatch.setattr(source.store, "publish_bytes", publish_after_seal)
    with pytest.raises(OpsError) as caught:
        native_registration.register_native(source.conn, source.store, source.repository,
            build_gate_spec(0.5, "control"), code_root=REPO, as_of_month=MONTH)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _scientific(source) == sealed_state == []
    assert source.conn.execute("SELECT COUNT(*) FROM native_gate_sweeps").fetchone()[0] == 1
    monkeypatch.setattr(source.store, "publish_bytes", publish)
    assert _call(source) == content_hash(_row(source))
    assert len(_call(source, sweep.submit_sweep)) == 2 and len(_scientific(source)) == 2


def test_single_arm_registration_retains_version_twelve_support(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(schema, "MIGRATIONS", schema.MIGRATIONS[:12])
        old = build_gate_source(tmp_path)
    try:
        before = [tuple(row) for row in old.conn.execute("SELECT * FROM schema_versions")]
        def register():
            return native_registration.register_native(old.conn, old.store, old.repository,
                build_gate_spec(), code_root=REPO, as_of_month=MONTH)
        registered = register()
        assert register() == registered and len(_scientific(old)) == 1
        assert [tuple(row) for row in old.conn.execute("SELECT * FROM schema_versions")] == before
        assert old.conn.execute("SELECT 1 FROM sqlite_master WHERE name='native_gate_sweeps'").fetchone() is None
    finally:
        old.conn.close()


@pytest.mark.parametrize("relative", ["experiments/native_gate_evaluation.py", "experiments/native_gate_entrypoint.py"])
def test_changed_child_source_refuses_submission_before_jobs(source, tmp_path, relative):
    checkout = _checkout(tmp_path)
    _call(source, code_root=checkout)
    before = _scientific(source)
    path = checkout / relative
    _replace_source(path, path.read_text() + "\n# synthetic change\n")
    with pytest.raises(OpsError) as caught:
        _call(source, sweep.submit_sweep, code_root=checkout)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _scientific(source) == before and source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("action", [sweep.register_sweep, sweep.submit_sweep])
def test_coordinator_content_is_sealed_and_drift_refuses_without_effects(source, tmp_path, monkeypatch, action):
    checkout = _checkout(tmp_path)
    path = checkout / "experiments/v2_gate_sweep.py"
    identity = _call(source, code_root=checkout)
    seal = _row(source)
    original = path.read_bytes()
    assert seal["coordinator_source"]["experiments/v2_gate_sweep.py"] == "sha256:" + hashlib.sha256(original).hexdigest()
    assert _call(source, code_root=checkout) == identity
    before = list(source.conn.iterdump())
    _replace_source(path, original.decode() + "\n# synthetic coordinator-only drift\n")
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        _call(source, action, code_root=checkout)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert list(source.conn.iterdump()) == before and _row(source) == seal


@pytest.mark.parametrize("field,value", [("scope", "different"), ("holdout_as_of_month", "2025-05")])
@pytest.mark.parametrize("action", [sweep.register_sweep, sweep.submit_sweep])
def test_sealed_context_change_refuses_without_effects(source, monkeypatch, field, value, action):
    _call(source)
    before = list(source.conn.iterdump())
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    with pytest.raises(OpsError) as caught:
        action(source.conn, source.store, source.repository, _request(), code_root=REPO,
               as_of_month=value if field == "holdout_as_of_month" else MONTH,
               scope=value if field == "scope" else "shadow")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert list(source.conn.iterdump()) == before


@pytest.mark.parametrize("purpose", ["final", "random", "rolling"])
def test_cli_and_fixed_child_refuse_holdout_before_input_access(tmp_path, capsys, monkeypatch, purpose):
    monkeypatch.setattr(sweep, "read_native_staged_input", _forbid)
    args = ["submit", "--roster", "missing", "--catalog", "missing", "--store-root", "missing",
            "--as-of-month", MONTH, "--code-root", str(REPO), "--purpose", purpose, "--no-ledger"]
    assert sweep.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "HOLDOUT_ACCESS_DENIED"}
    with pytest.raises(OpsError) as caught:
        native_gate_entrypoint.dispatch("native_gate_evaluation", {"purpose": purpose}, tmp_path)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"


def test_cli_register_submit_and_missing_smoke_are_exact(source, capsys):
    roster = source.root / "roster.json"
    roster.write_text(json.dumps(_request()))
    args = ["--roster", str(roster), "--catalog", str(source.root / "catalog.sqlite"),
            "--store-root", str(source.store.root), "--as-of-month", MONTH, "--code-root", str(REPO)]
    assert sweep.main(["register", *args]) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_EXPERIMENT_SPEC"}
    assert _scientific(source) == []
    assert sweep.main(["register", *args, "--no-ledger"]) == 0
    assert json.loads(capsys.readouterr().out) == {"roster": content_hash(_row(source))}
    assert sweep.main(["submit", *args, "--no-ledger"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"state": "submitted_uncollected", "job_ids": [job.job_id for job in _call(source, sweep.submit_sweep)]}


@pytest.mark.parametrize("fault", ["missing", "coordinator-missing", "malformed", "unsafe-asset"])
def test_unverifiable_sealed_child_source_preserves_failure_kind(source, tmp_path, fault):
    checkout = _checkout(tmp_path)
    _call(source, code_root=checkout)
    path = checkout / ("experiments/v2_gate_sweep.py" if fault == "coordinator-missing"
                       else "experiments/native_gate_evaluation.py")
    if fault in {"missing", "coordinator-missing"}:
        path.unlink()
    elif fault == "malformed":
        _replace_source(path, "def invalid(:\n")
    else:
        (checkout / "engine/dashboard/static/unsafe\\asset.txt").write_text("synthetic")
    with pytest.raises(OpsError) as caught:
        _call(source, sweep.submit_sweep, code_root=checkout)
    assert caught.value.code == ("INPUT_CHANGED" if fault == "coordinator-missing" else "EXPERIMENT_IDENTITY_CONFLICT")
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_lost_saved_pin_is_r2_without_head_fallback(source, monkeypatch):
    _call(source)
    before = _row(source)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    def missing(_):
        from engine.v2.data.errors import fail
        raise fail("SNAPSHOT_UNRESOLVED", "synthetic missing pin")
    # Exercise the registrar's exact snapshot verification, not a replaced registrar.
    monkeypatch.setattr(source.repository, "resolve", missing)
    with pytest.raises(OpsError) as caught:
        _call(source, sweep.submit_sweep)
    assert caught.value.code == "SNAPSHOT_UNRESOLVED"
    assert _row(source) == before and source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_initial_missing_pin_is_r2(source):
    with pytest.raises(OpsError) as caught:
        _call(source, scope="nonexistent")
    assert caught.value.code == "SNAPSHOT_UNRESOLVED"
    assert source.conn.execute("SELECT COUNT(*) FROM native_gate_sweeps").fetchone()[0] == 0


def test_deep_json_roster_refuses_before_catalog_access(tmp_path, capsys, monkeypatch):
    roster = tmp_path / "roster.json"
    roster.write_text("[" * 2000 + "0" + "]" * 2000)
    monkeypatch.setattr(sweep, "connect", _forbid)
    args = ["register", "--roster", str(roster), "--catalog", "absent", "--store-root", "absent",
            "--as-of-month", MONTH, "--code-root", str(REPO), "--no-ledger"]
    assert sweep.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_EXPERIMENT_SPEC"}


@pytest.mark.parametrize("changed", ["spec", "registration", "variant", "expected", "environment", "runner"])
def test_evaluation_child_admits_exact_bound_inputs_before_fitting(source, tmp_path, monkeypatch, changed):
    _call(source)
    spec = build_gate_spec()
    registration = native_registration.read_native_registration(source.conn, source.store, spec)
    stage = tmp_path / "direct"
    stage.mkdir()
    document = asdict(spec)
    if changed == "spec":
        document["economic_params"] = {"fill": 0.5, "strategy": "STR-THRU"}
    if changed == "runner":
        document["runner"] = "experiments/native_gate_worker.py"
    (stage / "spec.json").write_text(canonical_json(document))
    (stage / "native_registration.json").write_bytes(registration.binding_json if changed != "registration" else b"{}")
    parameters = {"run_id": registration.run_id, "variant_id": registration.variant_id,
        "expected_ids": [registration.variant_id], "no_ledger": True,
        "catalog_path": str(source.root / "catalog.sqlite"), "objects_root": str(source.store.root),
        "input_bindings": {"spec.json": "spec", "native_registration.json": "binding"}}
    if changed == "variant":
        parameters["variant_id"] = "wrong"
    if changed == "expected":
        parameters["expected_ids"] = ["wrong"]
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "2" if changed == "environment" else "1")
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(native_gate_evaluation, "gate_result", _forbid)
    with pytest.raises(OpsError) as caught:
        native_gate_entrypoint.dispatch("native_gate_evaluation", parameters, stage)
    assert caught.value.code == ("INVALID_EXPERIMENT_SPEC" if changed == "runner" else "EXPERIMENT_IDENTITY_CONFLICT")
    assert not (stage / "native_gate_result.json").exists()


@pytest.mark.parametrize("parameters", [{}, {"no_ledger": False}, {"no_ledger": 1}])
def test_child_requires_smoke_before_catalog_access(tmp_path, monkeypatch, parameters):
    monkeypatch.setattr(native_gate_evaluation.sqlite3, "connect", _forbid)
    with pytest.raises(OpsError) as caught:
        native_gate_entrypoint.dispatch("native_gate_evaluation", parameters, tmp_path)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("parameters", [None, [], "sweep", 7,
    *({"purpose": purpose, "no_ledger": True} for purpose in (None, False, 0, [], {}))])
def test_child_refuses_untyped_arguments_before_catalog_access(tmp_path, monkeypatch, parameters):
    monkeypatch.setattr(native_gate_evaluation.sqlite3, "connect", _forbid)
    with pytest.raises(OpsError) as caught:
        native_gate_entrypoint.dispatch("native_gate_evaluation", parameters, tmp_path)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "native gate arguments must be typed"
    assert list(tmp_path.iterdir()) == []


def test_complete_child_parameters_require_smoke_before_catalog_access(tmp_path, monkeypatch):
    parameters = {"run_id": "registered-run", "variant_id": "registered-variant",
        "expected_ids": ["registered-variant"], "no_ledger": False, "purpose": "sweep",
        "catalog_path": str(tmp_path / "catalog.sqlite"), "objects_root": str(tmp_path / "store"),
        "input_bindings": {"spec.json": "spec", "native_registration.json": "binding"}}
    assert from_document(stages.NativeGateParameters, parameters).no_ledger is False
    monkeypatch.setattr(native_gate_evaluation.sqlite3, "connect", _forbid)
    with pytest.raises(OpsError) as caught:
        native_gate_entrypoint.dispatch("native_gate_evaluation", parameters, tmp_path)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "native gate child has no recording authority"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("unicode", [False, True])
def test_roster_decoder_exact_utf8_limit_and_predecode_refusal(monkeypatch, unicode):
    limit = 16 * 1024 * 1024
    assert native_registration.MAX_GATE_SWEEP_BYTES == limit
    payload = ('{"value":"' + "é" * ((limit - 12) // 2) + '"}' if unicode else "{}" + " " * (limit - 2))
    assert len(payload.encode("utf-8")) == limit
    result = native_registration.decode_native_gate_sweep(payload)
    assert result == ({"value": "é" * ((limit - 12) // 2)} if unicode else {})
    monkeypatch.setattr(native_registration, "json", SimpleNamespace(loads=_forbid))
    with pytest.raises(OpsError) as caught:
        native_registration.decode_native_gate_sweep(payload + " ")
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("payload", [None, b"{}", "{", "[]", "null", "[" * 2000 + "]" * 2000])
def test_malformed_stored_roster_is_typed(payload):
    with pytest.raises(OpsError) as caught:
        native_registration.decode_native_gate_sweep(payload)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("character", ["a", "é", "漢", "😀"])
def test_roster_sql_reader_bounds_encoded_payload_and_preserves_utf8_limit(monkeypatch, encoding, character):
    limit = 128
    monkeypatch.setattr(native_registration, "MAX_GATE_SWEEP_BYTES", limit)
    payload = '{"v":"' + character * ((limit - 8) // len(character.encode())) + '"}'
    payload += " " * (limit - len(payload.encode()))
    assert len(payload.encode()) == limit
    with sqlite3.connect(":memory:") as conn:
        conn.execute(f"PRAGMA encoding='{encoding}'")
        conn.execute("CREATE TABLE native_gate_sweeps (experiment_id TEXT PRIMARY KEY, document_json)")
        assert native_registration.read_native_gate_sweep(conn, "missing") is None
        conn.execute("INSERT INTO native_gate_sweeps VALUES ('exact',?)", (payload,))
        assert native_registration.read_native_gate_sweep(conn, "exact") == payload
        conn.execute("INSERT INTO native_gate_sweeps VALUES ('over',?)", (payload + " ",))
        with pytest.raises(OpsError) as caught:
            native_registration.read_native_gate_sweep(conn, "over")
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
        oversized = payload + "\0" + " " * (limit * 2)
        conn.execute("INSERT INTO native_gate_sweeps VALUES ('large',?)", (oversized,))
        fetch_limit = limit * (2 if encoding.startswith("UTF-16") else 1)
        def guarded_row(cursor, row):
            assert all(not isinstance(value, (bytes, str)) or len(value) <= fetch_limit for value in row), "unbounded SQL payload"
            return row
        conn.row_factory = guarded_row
        with pytest.raises(AssertionError, match="unbounded SQL payload"):
            conn.execute("SELECT CAST(document_json AS BLOB) FROM native_gate_sweeps WHERE experiment_id='large'").fetchone()
        statements = []
        conn.set_trace_callback(statements.append)
        with pytest.raises(OpsError) as caught:
            native_registration.read_native_gate_sweep(conn, "large")
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert len([sql for sql in statements if "FROM native_gate_sweeps" in sql]) == 1


@pytest.mark.parametrize("expression", ["NULL", "x'7b7d'", "17", "CAST(x'ff' AS TEXT)"])
def test_roster_sql_reader_rejects_wrong_type_null_or_invalid_encoding(expression):
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE native_gate_sweeps (experiment_id TEXT PRIMARY KEY, document_json)")
        conn.execute(f"INSERT INTO native_gate_sweeps VALUES ('invalid', {expression})")
        with pytest.raises(OpsError) as caught:
            native_registration.read_native_gate_sweep(conn, "invalid")
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("action", ["register", "submit", "reserve"])
def test_oversized_stored_roster_refuses_before_decode_or_effect(source, monkeypatch, action):
    spec = build_gate_spec()
    prepared, spec = native_registration.prepare_native_registration(source.repository, spec,
        code_root=REPO, as_of_month=MONTH, scope="shadow", event_ids=None)
    oversized = "{}" + " " * 1023
    source.conn.execute("INSERT INTO native_gate_sweeps VALUES (?,?)", (spec.experiment_id, oversized))
    before = list(source.conn.iterdump())
    row_factory = source.conn.row_factory
    oversized_bytes = oversized.encode()
    def bounded_row(cursor, row):
        assert oversized not in row and oversized_bytes not in row, "unbounded stored roster reached Python"
        return row_factory(cursor, row)
    source.conn.row_factory = bounded_row
    with pytest.raises(AssertionError, match="unbounded stored roster reached Python"):
        source.conn.execute("SELECT document_json FROM native_gate_sweeps").fetchone()
    monkeypatch.setattr(native_registration, "MAX_GATE_SWEEP_BYTES", 1024)
    monkeypatch.setattr(native_registration, "json", SimpleNamespace(loads=_forbid))
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    monkeypatch.setattr(source.repository, "resolve_pinned", _forbid)
    with pytest.raises(OpsError) as caught:
        if action == "reserve":
            native_registration.reserve_native_registration(source.conn, source.store, prepared, spec)
        else:
            _call(source, sweep.register_sweep if action == "register" else sweep.submit_sweep)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert list(source.conn.iterdump()) == before


def test_seal_transaction_uses_bounded_reader_after_preparation(source, monkeypatch):
    original = native_registration.native_registration_exists
    oversized = "{}" + " " * native_registration.MAX_GATE_SWEEP_BYTES
    inserted = []
    def introduce_row(conn, store, registration):
        if conn.in_transaction and not inserted:
            conn.execute("INSERT INTO native_gate_sweeps VALUES (?,?)", (build_gate_spec().experiment_id, oversized))
            inserted.append(True)
        return original(conn, store, registration)
    row_factory = source.conn.row_factory
    oversized_bytes = oversized.encode()
    def bounded_row(cursor, row):
        assert oversized not in row and oversized_bytes not in row, "unbounded stored roster reached Python"
        return row_factory(cursor, row)
    before = list(source.conn.iterdump())
    source.conn.row_factory = bounded_row
    monkeypatch.setattr(native_registration, "native_registration_exists", introduce_row)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    with pytest.raises(OpsError) as caught:
        _call(source)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and inserted == [True]
    assert list(source.conn.iterdump()) == before


def test_oversized_new_roster_cannot_create_unreplayable_seal(source, monkeypatch):
    before = list(source.conn.iterdump())
    with monkeypatch.context() as patch:
        patch.setattr(native_registration, "MAX_GATE_SWEEP_BYTES", 1)
        patch.setattr(source.store, "publish_bytes", _forbid)
        with pytest.raises(OpsError) as caught:
            _call(source)
        assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
        assert list(source.conn.iterdump()) == before
    identity = _call(source)
    assert _call(source) == identity


def test_cli_closes_catalog_when_store_initialization_fails(source, monkeypatch, capsys):
    conn = connect(source.root / "catalog.sqlite", must_exist=True)
    monkeypatch.setattr(sweep, "connect", lambda *args, **kwargs: conn)
    def unavailable_store(*args, **kwargs):
        raise OSError("synthetic store initialization failure")
    monkeypatch.setattr(sweep, "ArtifactStore", unavailable_store)
    roster = source.root / "roster.json"
    roster.write_text(json.dumps(_request()))
    args = ["register", "--roster", str(roster), "--catalog", str(source.root / "catalog.sqlite"),
            "--store-root", str(source.store.root), "--as-of-month", MONTH, "--code-root", str(REPO), "--no-ledger"]
    before = list(source.conn.iterdump())
    assert sweep.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INPUT_CHANGED"}
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        conn.execute("SELECT 1")
    assert list(source.conn.iterdump()) == before


def test_evaluation_receipt_preserves_existing_destination_and_never_records(source, tmp_path, monkeypatch):
    from engine.v2.ops import experiments
    from experiments import native_outcomes

    _call(source)
    registration = native_registration.read_native_registration(source.conn, source.store, build_gate_spec())
    stage = tmp_path / "direct"
    stage.mkdir()
    (stage / "spec.json").write_text(canonical_json(asdict(build_gate_spec())))
    (stage / "native_registration.json").write_bytes(registration.binding_json)
    target = stage / "native_gate_result.json"
    target.write_bytes(b"existing result sentinel")
    parameters = {"run_id": registration.run_id, "variant_id": registration.variant_id,
        "expected_ids": [registration.variant_id], "no_ledger": True,
        "catalog_path": str(source.root / "catalog.sqlite"), "objects_root": str(source.store.root),
        "input_bindings": {"spec.json": "spec", "native_registration.json": "binding"}}
    monkeypatch.chdir(REPO)
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(native_gate_evaluation, "gate_result", lambda *args: {"complete": "synthetic"})
    monkeypatch.setattr(experiments, "experiments_ledger_path", _forbid)
    monkeypatch.setattr(native_outcomes, "publish_native_outcome", _forbid)
    monkeypatch.setattr(native_outcomes, "replay_native_outcome", _forbid)
    before = _scientific(source)
    with pytest.raises(OpsError) as caught:
        native_gate_entrypoint.dispatch("native_gate_evaluation", parameters, stage)
    assert caught.value.code == "INPUT_CHANGED"
    assert target.read_bytes() == b"existing result sentinel" and _scientific(source) == before
    assert set(path.name for path in stage.iterdir()) == {"spec.json", "native_registration.json", "native_gate_result.json"}
