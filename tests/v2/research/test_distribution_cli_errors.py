"""Operator input mistakes are distinct from real storage loss or CAS corruption."""
import errno
import json
import os
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactError, ArtifactStore, artifact_reference, to_document
from engine.v2.ops.errors import fail
from experiments import distribution_cli as cli
from experiments import native_gate_worker as inputs
from experiments.distribution_candidate_recipe import candidate_recipe
from tests.v2.research.magnitude_trial_support import prediction_spec
from tests.v2.research.test_distribution_run_request import request

FIELDS = ("request", "spec", "source-spec", "source-request", "source-plan-ref", "recipe", "budget-caps")


@pytest.fixture
def operator(tmp_path, monkeypatch):
    documents = {"request": request(), "spec": to_document(prediction_spec()),
        "source-spec": to_document(prediction_spec(experiment_id="synthetic-source")),
        "source-request": {}, "source-plan-ref": to_document(artifact_reference(b"source", "source.v1")),
        "recipe": candidate_recipe("H"), "budget-caps": {}}
    paths = {name: tmp_path / (name + ".json") for name in FIELDS}
    for name, value in documents.items():
        paths[name].write_text(json.dumps(value), encoding="utf-8")
    paths.update(catalog=tmp_path / "catalog.sqlite", **{"store-root": tmp_path / "store"})
    paths["store-root"].mkdir()
    with sqlite3.connect(paths["catalog"]) as conn:
        conn.execute("CREATE TABLE untouched(value TEXT)")
    conn.close()
    argv = ["run", *[arg for name, path in paths.items() for arg in ("--" + name, str(path))], "--no-ledger"]
    calls, connections = [], []
    @contextmanager
    def contained(*args):
        calls.append("contained")
        try:
            yield
        finally:
            calls.append("released")
    original_connect = sqlite3.connect
    def connect(*args, **kwargs):
        calls.append("catalog")
        conn = original_connect(*args, **kwargs)
        connections.append(conn)
        return conn
    monkeypatch.setattr(cli, "contained", contained)
    monkeypatch.setattr(cli.sqlite3, "connect", connect)
    monkeypatch.setattr(cli, "read_native_registration", lambda *args, **kwargs:
        pytest.fail("input refusal reached native registration or execution"))
    return SimpleNamespace(argv=argv, paths=paths, calls=calls, connections=connections)


def refused(operator, capsys, code, *, private=()):
    assert cli.main(operator.argv) == 2
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert set(output) == {"refused"} and output["refused"]["code"] == code
    for token in (str(operator.paths["request"].parent), "SECRET", *private):
        assert token not in captured.out + captured.err
    for conn in operator.connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    return output["refused"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("fault", ["missing", "unreadable"])
def test_real_input_reader_refuses_operator_path_errors_before_native_effects(operator, monkeypatch, capsys, field, fault):
    target, attempted = operator.paths[field], []
    if fault == "missing":
        target.unlink()  # Caller input only, never a store object.
    else:
        original = inputs.os.open
        def denied(path, *args, **kwargs):
            if os.fspath(path) == str(target):
                attempted.append(True)
                raise PermissionError(errno.EACCES, "SECRET input access", str(target))
            return original(path, *args, **kwargs)
        monkeypatch.setattr(inputs.os, "open", denied)
    refused(operator, capsys, "INVALID_EXPERIMENT_SPEC")
    assert operator.calls == ([] if field == "request" else ["contained", "catalog", "released"])
    assert len(operator.connections) == (0 if field == "request" else 1)
    if fault == "unreadable":
        assert attempted == [True]  # Privilege-independent failure of the actual bounded reader.


@pytest.mark.parametrize("fault,raw,expected", [
    ("corrupt", "INTEGRITY_FAILED", "INTEGRITY_FAILED"),
    ("missing", "MISSING", "RESOURCE_UNAVAILABLE"),
])
def test_real_cas_corruption_is_not_retryable_missing_transport(operator, monkeypatch, capsys, fault, raw, expected):
    store = ArtifactStore(operator.paths["store-root"])
    if fault == "corrupt":
        ref = store.publish_bytes(b"original", schema_ref="synthetic.v1")
        path = store.root / ref.storage_key
        path.chmod(0o644)
        path.write_bytes(b"tampered")  # Same size: hash verification, not an invented error code.
    else:
        ref = artifact_reference(b"never published", "synthetic.v1")
    with pytest.raises(ArtifactError) as native:
        store.read_verified(ref)
    assert native.value.code == raw
    def dispatch(args, conn, actual_store, *values, **kwargs):
        return actual_store.read_verified(ref)
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    problem = refused(operator, capsys, expected, private=(ref.storage_key,))
    assert problem["retryable"] is (fault == "missing")
    assert operator.calls == ["contained", "catalog", "released"]


@pytest.mark.parametrize("missing", ["catalog", "store-root"])
def test_actual_missing_runtime_storage_is_not_an_input_spec_error(operator, capsys, missing):
    index = operator.argv.index("--" + missing) + 1
    target = operator.paths[missing].with_name("missing-" + missing)
    operator.argv[index] = str(target)
    problem = refused(operator, capsys, "RESOURCE_UNAVAILABLE")
    assert problem["retryable"] is True and not target.exists()
    assert operator.calls == (["contained", "catalog", "released"] if missing == "catalog" else ["contained", "released"])


@pytest.mark.parametrize("field", ["request", "source-spec"])
@pytest.mark.parametrize("fault,message", [
    ("symlink", "native staging must not follow symlinks"),
    ("oversize", "native staging requires bounded regular files"),
    ("malformed", "search requires valid JSON object metadata"),
])
def test_native_input_symlink_byte_bound_and_json_refusals_are_preserved(operator, capsys, field, fault, message):
    path = operator.paths[field]
    if fault == "symlink":
        other = path.with_name("target.json")
        path.rename(other)
        path.symlink_to(other)
    elif fault == "oversize":
        with path.open("wb") as stream:
            stream.truncate(inputs.MAX_STAGED_BYTES + 1)  # Sparse synthetic input, no giant allocation.
    else:
        path.write_text('{"duplicate": 1, "duplicate": 2}', encoding="utf-8")
    assert refused(operator, capsys, "INVALID_EXPERIMENT_SPEC") == to_document(fail("INVALID_EXPERIMENT_SPEC", message).problem)
    assert operator.calls == ([] if field == "request" else ["contained", "catalog", "released"])
