"""CLI boundaries preserve containment, finite admission and read-only seals."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactError
from engine.v2.ops.errors import OpsError
from experiments import size_search as cli
from experiments import size_search_binding as binding
from experiments import size_search_report as report
from experiments import v2_magnitude_search_run as legacy


def args(action="report"):
    return [action, "--spec", "spec", "--request", "request", "--plan-ref", "ref",
            "--catalog", "catalog", "--store-root", "/tmp", "--no-ledger"]


def setup_cli(monkeypatch):
    monkeypatch.setattr(cli, "_contain", lambda request: None)
    monkeypatch.setattr(cli, "_read_document", lambda path: {})
    monkeypatch.setattr(cli, "experiment_spec_from_document", lambda doc: object())
    monkeypatch.setattr(cli, "connect", lambda *a, **k: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(cli, "Repository", lambda *a, **k: None)
    monkeypatch.setattr(cli, "read_native_registration", lambda *a, **k: object())
    monkeypatch.setattr(cli, "collect", lambda *a: pytest.fail("collection after invalid metadata"))


@pytest.mark.parametrize("document", [{}, {"schema_ref": "bad"}, {"digest": 1}])
def test_invalid_plan_reference_has_json_refusal(monkeypatch, capsys, document):
    setup_cli(monkeypatch)
    monkeypatch.setattr(cli, "_read_document", lambda path: document if path == "ref" else {})
    assert cli.main(args()) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_EXPERIMENT_SPEC",
        "message": "invalid SIZE metadata"}


@pytest.mark.parametrize("error", [OSError("disk"), sqlite3.OperationalError("catalog"), ArtifactError("IO_ERROR", "object")])
def test_storage_refusal_is_json(monkeypatch, capsys, error):
    setup_cli(monkeypatch)
    def broken(*a, **k):
        raise error
    monkeypatch.setattr(cli, "connect", broken)
    assert cli.main(args()) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "RESOURCE_UNAVAILABLE",
        "message": "SIZE storage unavailable"}


def test_programming_error_is_not_storage_refusal(monkeypatch):
    setup_cli(monkeypatch)
    monkeypatch.setattr(cli, "connect", lambda *a, **k: (_ for _ in ()).throw(TypeError("bug")))
    with pytest.raises(TypeError, match="bug"):
        cli.main(args())


@pytest.mark.parametrize("action", ["seal", "run", "collect"])
def test_legacy_cli_refuses_qualified_route_before_catalog_or_worker(monkeypatch, capsys, action):
    monkeypatch.setattr(legacy, "_read_document", lambda path: {"schema_version": "magnitude_search_request.v2"})
    monkeypatch.setattr(legacy, "experiment_spec_from_document", lambda doc: object())
    monkeypatch.setattr(legacy, "from_document", lambda *a: object())
    for name in ("connect", "read_native_registration", "run_search", "seal_search_run", "collect_outer_results"):
        monkeypatch.setattr(legacy, name, lambda *a, **k: pytest.fail("qualified legacy execution"))
    argv = args(action)
    if action == "run":
        argv += ["--memory-bytes", str(2 << 30), "--cpu-seconds", "30", "--wall-seconds", "60"]
    assert legacy.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["refused"]["code"] == "INVALID_EXPERIMENT_SPEC"


def test_size_plan_verification_passes_finite_native_loader_budget(monkeypatch):
    seen = []
    def verify(*a, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop before source work")
    monkeypatch.setattr(binding, "verify_native_registration", verify)
    with pytest.raises(RuntimeError, match="stop"):
        binding.prepare_plan(None, None, None, {})
    assert seen == [{"max_evidence_bytes": cli.MAX_NATIVE_EVIDENCE_BYTES}]
    assert cli.MAX_NATIVE_EVIDENCE_BYTES == 16 * 1024 * 1024


@pytest.mark.parametrize("seal", [None, {"changed": True}])
@pytest.mark.parametrize("action", ["run", "report"])
def test_report_admits_driver_before_result_reads_or_work(monkeypatch, seal, action):
    monkeypatch.setattr(legacy, "require_current", lambda *a: None)
    monkeypatch.setattr(legacy, "_binding", lambda *a: ({}, {"expected": True}))
    monkeypatch.setattr(legacy, "read_record", lambda *a: seal)
    monkeypatch.setattr(report, "collect_outer_results", lambda *a: pytest.fail("unsealed result read"))
    with pytest.raises(OpsError) as caught:
        context = (None, None, None, SimpleNamespace(run_id="run"), {}, None)
        cli._run(context) if action == "run" else report.collect(*context)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.message == "search execution requires its exact driver seal"


def test_size_register_refuses_oversized_binding_before_publication(monkeypatch, capsys):
    from tests.v2.research.test_size_review_contract import panel_request
    setup_cli(monkeypatch)
    monkeypatch.setattr(cli, "_read_document", lambda path: panel_request() if path == "panel" else {})
    registration = SimpleNamespace(binding_json=b"x" * (cli.MAX_NATIVE_EVIDENCE_BYTES + 1))
    monkeypatch.setattr(cli, "prepare", lambda *a, **k: (registration, object()))
    monkeypatch.setattr(cli, "seal_search_plan", lambda *a: pytest.fail("sealed oversized registration"))
    monkeypatch.setattr(cli, "ArtifactStore", lambda *a: SimpleNamespace(
        publish_bytes=lambda *a, **k: pytest.fail("published oversized registration")))
    argv = args("register") + ["--panel-request", "panel", "--scope", "development",
        "--training-start", "2010-01-01", "--labels-before", "2025-01-01"]
    assert cli.main(argv) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "RESOURCE_LIMIT_EXCEEDED",
        "message": "RESOURCE_LIMIT_EXCEEDED: native registration exceeds its UTF-8 byte budget"}


@pytest.mark.parametrize("bad", [None, 3])
def test_malformed_spec_decode_is_typed_but_not_general_typeerror(monkeypatch, capsys, bad):
    from engine.v2.ops.experiments import experiment_spec_from_document
    setup_cli(monkeypatch)
    monkeypatch.setattr(cli, "experiment_spec_from_document", experiment_spec_from_document)
    monkeypatch.setattr(cli, "_read_document", lambda path: {"hypothesis": "test", "primary_arm_id": "prior",
        "economic_params": {}, "price_source": "computed_moves", "input_files": bad})
    assert cli.main(args()) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_EXPERIMENT_SPEC",
        "message": "INVALID_EXPERIMENT_SPEC: invalid SIZE specification metadata"}
