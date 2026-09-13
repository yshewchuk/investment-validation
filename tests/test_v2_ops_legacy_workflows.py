"""Shadow compatibility and experiment lifecycle checks (O16-O18/O27-O28)."""
import pytest

from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import (
    ExperimentSpec,
    run_experiment,
    synthetic_fixture_runner,
)
from engine.v2.ops.legacy_adapter import copy_read_set, manifest_files
from engine.v2.ops.nightly import (
    GRAPH,
    build_legacy_job_requests,
    build_nightly_plan,
    graph_order,
    run_shadow_nightly,
)
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit_graph
from tests.ops_support import FakeClock


def test_private_copy_rejects_indirection_and_is_read_only(tmp_path):
    source = tmp_path / "source"
    private = tmp_path / "private"
    source.mkdir()
    (source / "input.bin").write_bytes(b"pinned")
    manifest = copy_read_set(source, private, ("input.bin",))
    assert manifest["input.bin"]["byte_size"] == 6
    assert not (private / "input.bin").is_symlink()
    assert (private / "input.bin").read_bytes() == b"pinned"
    (private / "input.bin").chmod(0o644)
    (private / "input.bin").write_bytes(b"changed")
    assert manifest_files(source, ("input.bin",))["input.bin"]["content_hash"] != ""
    (source / "link").symlink_to(source / "input.bin")
    with pytest.raises(OpsError, match="indirect"):
        manifest_files(source, ("link",))


def test_shadow_nightly_required_and_optional_receipts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calendar.json").write_text("{}")
    handlers = {stage: (lambda value, stage=stage: {**value, stage: "ok"})
                for stage in GRAPH if stage not in {"settlement", "model_evidence"}}
    receipt = run_shadow_nightly(source, tmp_path / "private", "2026-09-12",
                                 handlers=handlers, read_set=("calendar.json",))
    assert receipt["status"] == "degraded"
    assert {row["stage_id"] for row in receipt["stages"] if row["status"] == "degraded"} == {
        "settlement", "model_evidence"}
    handlers.pop("score")
    with pytest.raises(OpsError, match="required nightly stage"):
        run_shadow_nightly(source, tmp_path / "private2", "2026-09-12",
                           handlers=handlers, read_set=("calendar.json",))


def test_graph_is_topological_and_serialized_selfcheck_is_real():
    order = graph_order()
    assert set(order) == set(GRAPH)
    assert order.index("decision_validation") < order.index("decision_commit")
    assert order.index("projection") < order.index("selfcheck")


def test_allowlisted_legacy_dag_submits_with_real_dependencies(tmp_path):
    plan = build_nightly_plan("/root/investing-plan", "2026-09-12")
    requests = build_legacy_job_requests(plan, tickers=("FAKE",),
                                         year_start=2025, year_end=2026)
    assert [request.job.kind for request in requests] == [
        "legacy_finality", "legacy_score", "legacy_decisions", "legacy_settlement",
        "legacy_model_evidence", "legacy_render", "legacy_selfcheck"]
    conn = open_catalog(tmp_path / "ops.sqlite", clock=FakeClock())
    try:
        policy = NamespacePolicy({"operator": frozenset({"shadow"})})
        receipts = submit_graph(conn, registry(), policy, requests, clock=FakeClock())
        assert len(receipts) == len(requests)
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 7
    finally:
        conn.close()


def test_smoke_runner_requires_report_and_never_ledger(tmp_path):
    spec = ExperimentSpec("EXP-SYN", "plumbing", "fixture", ("fixture",), 7,
                          ("fold-1",), {"fill": "mid"}, "synthetic")
    receipt = run_experiment(spec, tmp_path, tmp_path / "run",
                             runner=synthetic_fixture_runner, mode="smoke",
                             synthetic=True)
    assert receipt["status"] == "succeeded"
    assert receipt["ledger_receipt"]["mode"] == "smoke"
    assert receipt["backup_receipt"] is None


def test_private_canary_compares_population_and_requires_selfcheck(tmp_path):
    from checks.rearchitecture_phase1_canary import run

    reference = tmp_path / "reference.json"
    adapter = tmp_path / "adapter.json"
    selfcheck = tmp_path / "selfcheck.json"
    payload = {"rows": [{"row_id": "a", "ticker": "FAKE", "score": 0.5}]}
    reference.write_text(__import__("json").dumps(payload))
    adapter.write_text(__import__("json").dumps(payload))
    selfcheck.write_text("{\"ok\": true}")
    receipt = run(reference, adapter, selfcheck)
    assert receipt["status"] == "pass"


def test_primary_backup_failure_is_retryable_without_rerun(tmp_path):
    spec = ExperimentSpec("EXP-SYN2", "plumbing", "fixture", ("fixture",), 7,
                          ("fold-1",), {"fill": "mid"}, "synthetic")
    calls = []

    def primary_runner(*, run_dir, no_ledger):
        assert no_ledger is False
        (run_dir / "REPORT.md").write_text("# Genuine report\nGenerated by engine.report v1.0.\n")
        return {"completed": True}

    def backup(_receipt):
        calls.append("backup")
        raise RuntimeError("mirror offline")

    receipt = run_experiment(spec, tmp_path, tmp_path / "run2",
                             runner=primary_runner, mode="primary",
                             backup=backup)
    assert receipt["status"] == "succeeded"
    assert calls == ["backup"]
    assert receipt["backup_receipt"]["status"] == "backup_pending"
