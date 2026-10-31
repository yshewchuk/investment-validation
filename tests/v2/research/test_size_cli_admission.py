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
    monkeypatch.setattr(cli, "_contain", lambda *args, **kwargs: 2 ** 33)
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


@pytest.mark.parametrize("schema", ["magnitude_search_request.v2", "magnitude_search_request.v3",
                                     "magnitude_search_request.v4"])
@pytest.mark.parametrize("action", ["seal", "run", "collect"])
def test_legacy_cli_refuses_qualified_route_before_catalog_or_worker(monkeypatch, capsys, action, schema):
    monkeypatch.setattr(legacy, "_read_document", lambda path: {"schema_version": schema})
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
    from engine.v2.contracts import ArtifactRef
    from engine.v2.foundation import to_document
    from tests.v2.research.test_size_review_contract import panel_request

    RECEIPT_REF = to_document(ArtifactRef(artifact_id="a", content_hash="sha256:" + "0" * 64,
        schema_ref="size_preflight_receipt.v1", byte_size=1, storage_key="objects/00/" + "0" * 64))
    setup_cli(monkeypatch)
    monkeypatch.setattr(cli, "_read_document", lambda path: panel_request() if path == "panel" else RECEIPT_REF if path == "receipt" else {})
    registration = SimpleNamespace(binding_json=b"x" * (cli.MAX_NATIVE_EVIDENCE_BYTES + 1))
    monkeypatch.setattr(cli, "input_bindings", lambda *a, **k: {})
    monkeypatch.setattr(cli, "admit", lambda *a, **k: (registration, object()))
    monkeypatch.setattr(cli, "seal_search_plan", lambda *a: pytest.fail("sealed oversized registration"))
    monkeypatch.setattr(cli, "ArtifactStore", lambda *a: SimpleNamespace(
        publish_bytes=lambda *a, **k: pytest.fail("published oversized registration")))
    argv = args("register") + ["--panel-request", "panel", "--scope", "development",
        "--training-start", "2010-01-01", "--labels-before", "2025-01-01",
        "--receipt-ref", "receipt"]
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


def test_native_data_error_uses_supported_code_property(monkeypatch, capsys):
    from engine.v2.data.errors import DataError, fail
    setup_cli(monkeypatch)
    error = fail("INPUT_CHANGED", "native input changed")
    assert type(error) is DataError
    assert error.code == error.problem.code == "INPUT_CHANGED"
    monkeypatch.setattr(cli, "Repository", lambda *a, **k: (_ for _ in ()).throw(error))
    assert cli.main(args()) == 2
    assert json.loads(capsys.readouterr().out) == {
        "refused": "INPUT_CHANGED", "message": str(error)}


@pytest.mark.parametrize("headroom", [-1, 0, 1])
def test_prepare_and_cli_memory_admission_have_identical_refusal(monkeypatch, headroom):
    from engine.v2.research.magnitude_panel_qualified_table import recipe
    from tests.v2.research.test_magnitude_search_request import request as request_fixture
    from tests.v2.research.test_size_review_contract import panel_request

    request = request_fixture()
    request["feature_dictionary_hash"] = recipe("HMO")["dictionary_hash"]
    panel = panel_request()
    panel.update(feature_snapshot_id=request["feature_snapshot_id"], history_snapshot=request["history_snapshot_id"])
    request["limits"]["child"]["memory_bytes"] = request["limits"]["parent"]["memory_bytes"] + headroom
    monkeypatch.setattr(binding, "_sources", lambda: {})
    monkeypatch.setattr(binding, "prepare_native_registration", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("admitted limits")))
    monkeypatch.setattr(cli, "resource", SimpleNamespace(RLIM_INFINITY=-1, RLIMIT_AS=0,
        getrlimit=lambda *a: (-1, -1), setrlimit=lambda *a: None))
    monkeypatch.setattr(cli, "signal", SimpleNamespace(SIGALRM=14, signal=lambda *a: None, alarm=lambda *a: None))
    def prepare():
        binding.prepare(None, None, request, panel, scope="test",
            training_start="2020-01-01", labels_before="2025-01-01")
    if headroom <= 0:
        cli._contain(request)
        with pytest.raises(RuntimeError, match="admitted limits"):
            prepare()
    else:
        for call in (lambda: cli._contain(request), prepare):
            with pytest.raises(OpsError) as caught:
                call()
            assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
            assert caught.value.problem.message == "child memory must not exceed inherited parent limit"


def test_bounded_registry_hash_preserves_exact_valid_source_identity(monkeypatch):
    from pathlib import Path

    from engine.v2.foundation import canonical_json
    from engine.v2.ops import fingerprints

    root = Path(binding.__file__).resolve().parents[1]
    assert len(binding.read_bounded_state(root / "engine/models/registry.json", 16 * 1024 * 1024)) < 16 * 1024 * 1024
    expected = fingerprints.source_closure(root, ["experiments/size_search.py",
        "experiments/magnitude_search_worker.py", "engine/models/registry.json"])
    monkeypatch.setattr(fingerprints, "file_hash", lambda *a, **k: pytest.fail("uncapped registry fingerprint"))
    assert canonical_json(binding._source_files()) == canonical_json(expected)


def startup_probe(monkeypatch, path):
    import importlib.util

    from engine.v2.models import frozen_state_io

    native_read = frozen_state_io.read_bounded_state
    def registry_read(requested, limit):
        assert requested.name == "registry.json" and limit == 16 * 1024 * 1024
        return native_read(path, limit)
    monkeypatch.setattr(frozen_state_io, "read_bounded_state", registry_read)
    spec = importlib.util.spec_from_file_location("size_binding_startup_probe", binding.__file__)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    return probe


@pytest.mark.parametrize("kind,code", [("oversized", "RESOURCE_LIMIT_EXCEEDED"), ("missing", "RESOURCE_UNAVAILABLE"),
                                      ("fifo", "RESOURCE_UNAVAILABLE")])
def test_failed_real_module_startup_then_cli_refuses_and_never_rebases(monkeypatch, tmp_path, capsys, kind, code):
    import os
    from pathlib import Path

    from tests.v2.research.test_magnitude_search_request import request as request_fixture
    from tests.v2.research.test_size_review_contract import panel_request

    valid = (Path(binding.__file__).resolve().parents[1] / "engine/models/registry.json").read_bytes()
    path = tmp_path / "registry.json"
    if kind == "oversized":
        with path.open("wb") as stream:
            stream.truncate(16 * 1024 * 1024 + 1)
    elif kind == "fifo":
        os.mkfifo(path)
    probe = startup_probe(monkeypatch, path)
    assert probe._IMPORTED_SOURCES is None
    setup_cli(monkeypatch)
    monkeypatch.setattr(cli, "_read_document", lambda p: panel_request() if p == "panel" else request_fixture())
    monkeypatch.setattr(cli, "prepare", probe.prepare)
    monkeypatch.setattr(probe, "comparison_recipe", lambda: pytest.fail("recipe before source admission"))
    monkeypatch.setattr(probe, "prepare_native_registration", lambda *a, **k: pytest.fail("source failure admitted native data"))
    argv = args("preflight") + ["--panel-request", "panel", "--scope", "test",
        "--training-start", "2020-01-01", "--labels-before", "2025-01-01"]
    assert cli.main(argv) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["refused"] == code and str(path) not in refusal["message"]
    path.unlink(missing_ok=True)
    path.write_bytes(valid)
    assert cli.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert probe._IMPORTED_SOURCES is None


def test_registry_migration_refuses_before_recipe_and_exact_restoration_replays(monkeypatch, tmp_path):
    from pathlib import Path

    valid = (Path(binding.__file__).resolve().parents[1] / "engine/models/registry.json").read_bytes()
    path = tmp_path / "registry.json"
    path.write_bytes(valid)
    probe = startup_probe(monkeypatch, path)
    original = dict(probe._IMPORTED_SOURCES)
    document = {"schema_version": probe.SCHEMA, "environment": probe._IMPORTED_ENVIRONMENT,
        "source_closure": original, "size_comparison": {"original": True}}
    monkeypatch.setattr(probe, "comparison_recipe", lambda: pytest.fail("migrated recipe constructed before identity check"))
    assert b"20260829" in valid
    path.write_bytes(valid.replace(b"20260829", b"20260830"))
    with pytest.raises(OpsError) as caught:
        probe.verify_installation(document)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    path.write_bytes(valid)
    calls = []
    monkeypatch.setattr(probe, "comparison_recipe", lambda: calls.append("recipe") or {"original": True})
    probe.verify_installation(document)
    assert calls == ["recipe"] and probe._IMPORTED_SOURCES == original


def test_fingerprint_programming_error_remains_visible(monkeypatch):
    monkeypatch.setattr(binding, "_source_files", lambda: (_ for _ in ()).throw(TypeError("bug")))
    with pytest.raises(TypeError, match="bug"):
        binding._sources()


@pytest.mark.parametrize("stage", ["input", "catalog", "report", "serialization"])
def test_parent_memory_error_has_static_refusal_and_closes_catalog(monkeypatch, capsys, stage):
    setup_cli(monkeypatch)
    closed = []
    monkeypatch.setattr(cli, "connect", lambda *a, **k: SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(cli, "from_document", lambda *a: object())
    monkeypatch.setattr(cli, "collect", lambda *a: {"complete": True})
    def exhausted(*a, **k):
        raise MemoryError("private allocation detail")
    target = {"input": "_read_document", "catalog": "connect", "report": "collect"}.get(stage)
    if target:
        monkeypatch.setattr(cli, target, exhausted)
    else:
        monkeypatch.setattr(cli, "json", SimpleNamespace(dumps=exhausted))
    assert cli.main(args()) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {"refused": "RESOURCE_LIMIT_EXCEEDED",
        "message": "SIZE parent memory exhausted"}
    assert closed == ([True] if stage in ("report", "serialization") else [])
    assert "Traceback" not in output.err and "private allocation detail" not in output.out


def test_recoverable_real_parent_rss_failure_is_json_exit_two():
    import subprocess
    import sys
    code = '''import time
from pathlib import Path
from experiments import size_search as cli
cli._read_document = lambda path: {}
cli.experiment_spec_from_document = lambda doc: object()
def contain(request, *args, **kwargs):
    with open('/proc/self/status', encoding='ascii') as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                current = int(line.split()[1]) * 1024
                break
    return current + (32 << 20)
cli._contain = contain
def allocate(*args, **kwargs):
    block = bytearray(128 << 20)
    block[::4096] = b"\\1" * len(range(0, len(block), 4096))
    time.sleep(5)
cli.connect = allocate
raise SystemExit(cli.main(['report','--spec','spec','--request','request','--plan-ref','ref',
    '--catalog','catalog','--store-root','/tmp','--no-ledger']))
'''
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 2, completed.stderr
    assert json.loads(completed.stdout) == {"refused": "RESOURCE_LIMIT_EXCEEDED",
        "message": "SIZE parent memory exhausted"}
    assert "Traceback" not in completed.stderr
