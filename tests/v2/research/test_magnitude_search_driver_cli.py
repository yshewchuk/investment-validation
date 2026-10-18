"""Synthetic CLI admission, read-only collection and all-or-nothing JSON output."""
# ruff: noqa: F811
import errno
import io
import json
import os
import sqlite3
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.foundation import ArtifactError, DocumentError, canonical_json, to_document
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_records as records
from experiments import v2_magnitude_search_run as cli
from tests.v2.research.test_magnitude_search_driver_binding import (
    driver_case,  # noqa: F401
    snapshot,
)

LIMITS = {"memory_bytes": 1024, "cpu_seconds": 2, "wall_seconds": 3}
RESOURCE_FLAGS = tuple("--" + name.replace("_", "-") for name in LIMITS)
ACTION_NAMES = {"seal": "seal_search_run", "run": "run_search", "collect": "collect_outer_results"}
METADATA_LIMIT = 16 * 1024 * 1024


@pytest.fixture
def source(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    catalog = tmp_path / "catalog.sqlite"
    with sqlite3.connect(catalog) as connection:
        connection.execute("CREATE TABLE sentinel (value INTEGER)")
        connection.execute("INSERT INTO sentinel VALUES (7)")
    reference = ArtifactRef(artifact_id="plan", content_hash="sha256:" + "1" * 64,
                            schema_ref="search_plan.v1", byte_size=2, storage_key="objects/plan")
    documents = {
        "spec": {"hypothesis": "synthetic", "primary_arm_id": "primary", "economic_params": {},
                 "price_source": "computed_moves", "runner": "experiments/v2_prediction.py"},
        "request": {"synthetic": True},
        "plan-ref": to_document(reference),
    }
    paths = {}
    for name, document in documents.items():
        paths[name] = tmp_path / (name + ".json")
        paths[name].write_text(json.dumps(document), encoding="utf-8")
    state = SimpleNamespace(root=tmp_path, paths=paths, catalog=catalog, store=store,
                            reference=reference, documents=documents, calls=[], connections=[],
                            registration=object(), repository=object(),
                            result={"z": [2, 1], "a": "synthetic é"})

    connect = sqlite3.connect
    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        state.connections.append(connection)
        return connection
    monkeypatch.setattr(sqlite3, "connect", tracked_connect)

    def registration(connection, artifact_store, spec):
        assert artifact_store.root == store
        assert spec.hypothesis == "synthetic"
        assert connection is state.connections[-1]
        return state.registration

    def repository(connection, artifact_store):
        assert connection is state.connections[-1]
        assert artifact_store.root == store
        return state.repository

    monkeypatch.setattr(cli, "read_native_registration", registration)
    monkeypatch.setattr(cli, "Repository", repository)
    monkeypatch.setattr(cli, "_admit_driver", lambda *args, **kwargs: None)
    for action, name in ACTION_NAMES.items():
        def invoke(*args, _action=action, **kwargs):
            state.calls.append((_action, args, kwargs))
            return state.result
        monkeypatch.setattr(cli, name, invoke)
    return state


def _args(source, action="collect", *, limits=True):
    arguments = [action]
    for name, path in source.paths.items():
        arguments.extend(["--" + name, str(path)])
    arguments.extend(["--catalog", str(source.catalog), "--store-root", str(source.store)])
    if action == "run" and limits:
        for name, value in LIMITS.items():
            arguments.extend(["--" + name.replace("_", "-"), str(value)])
    return arguments


def _invoke(source, capsys, action="collect", *, status=0, extra=()):
    assert cli.main([*_args(source, action), *extra]) == status
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert captured.out == canonical_json(document) + "\n"
    return document, captured.err


def _refusal(code, message):
    return {"refused": to_document(fail(code, message).problem)}


def _closed(source):
    assert len(source.connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        source.connections[0].execute("SELECT 1")


def _parser_error(capsys, message):
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err.splitlines()[-1].endswith(": error: " + message)


def _raising(error):
    def raise_error(*args, **kwargs):
        raise error
    return raise_error


@pytest.mark.parametrize("action", ACTION_NAMES)
@pytest.mark.parametrize("no_ledger", [False, True])
def test_actions_use_public_adapters_and_emit_one_canonical_document(source, capsys, action, no_ledger):
    result, stderr = _invoke(source, capsys, action, extra=("--no-ledger",) if no_ledger else ())
    assert result == source.result
    assert not stderr
    assert len(source.calls) == 1
    name, arguments, keywords = source.calls[0]
    assert name == action
    assert arguments[0] is source.connections[0]
    assert source.registration in arguments
    assert source.documents["request"] in arguments
    assert source.reference in arguments
    if action != "seal":
        assert source.repository in arguments
    if action == "run":
        assert keywords["limits"] == LIMITS
    else:
        assert "limits" not in keywords
    _closed(source)
    assert not list(source.root.rglob("*.csv"))


@pytest.mark.parametrize("action", ACTION_NAMES)
@pytest.mark.parametrize("flag", ["--spec", "--request", "--plan-ref", "--catalog", "--store-root"])
def test_every_action_requires_explicit_metadata_and_storage(source, capsys, action, flag):
    arguments = _args(source, action)
    index = arguments.index(flag)
    del arguments[index:index + 2]
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 2
    _parser_error(capsys, "the following arguments are required: " + flag)
    assert not source.calls


@pytest.mark.parametrize("flag", RESOURCE_FLAGS)
def test_run_requires_each_resource_flag(source, capsys, flag):
    arguments = _args(source, "run")
    index = arguments.index(flag)
    del arguments[index:index + 2]
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 2
    _parser_error(capsys, "run requires all three resource limits; seal and collect accept none")
    assert not source.calls


@pytest.mark.parametrize("flag", RESOURCE_FLAGS)
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "true", str(sys.maxsize), str(sys.maxsize + 1), str(10 ** 100)])
def test_run_resource_flags_require_positive_integers(source, capsys, flag, value):
    arguments = _args(source, "run")
    arguments[arguments.index(flag) + 1] = value
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 2
    message = (f"argument {flag}: invalid int value: {value!r}" if value in ("1.5", "true")
        else "search execution requires explicit positive integer resource limits")
    _parser_error(capsys, message)
    assert not source.calls


@pytest.mark.parametrize("action", ["seal", "collect"])
@pytest.mark.parametrize("flag", RESOURCE_FLAGS)
def test_nonexecuting_actions_reject_resource_flags(source, capsys, action, flag):
    with pytest.raises(SystemExit) as caught:
        cli.main([*_args(source, action), flag, "1"])
    assert caught.value.code == 2
    _parser_error(capsys, "run requires all three resource limits; seal and collect accept none")
    assert not source.calls


def test_unknown_action_and_execution_options_are_rejected(source, capsys):
    for arguments, message in ((["prepare", *_args(source)[1:]],
            "argument action: invalid choice: 'prepare' (choose from seal, run, collect)"),
            ([*_args(source), "--jobs", "2"], "unrecognized arguments: --jobs 2")):
        with pytest.raises(SystemExit) as caught:
            cli.main(arguments)
        assert caught.value.code == 2
        _parser_error(capsys, message)
    assert not source.calls


@pytest.mark.parametrize("payload", [b'{"a":1,"a":2}', b'{"x":{"a":1,"a":2}}',
                                     b"[]", b"null", b'"text"', b"1", b"{", b"\xff",
                                     b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'])
def test_metadata_decoder_rejects_duplicates_nonobjects_and_invalid_json(tmp_path, payload):
    path = tmp_path / "metadata.json"
    path.write_bytes(payload)
    with pytest.raises(OpsError) as caught:
        cli._read_document(path)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
        "search requires valid JSON object metadata").problem
    assert str(path) not in str(caught.value)


@pytest.mark.parametrize("kind", ["oversize", "symlink", "fifo", "directory"])
def test_metadata_reader_requires_bounded_regular_nonsymlink_leaf(tmp_path, kind):
    path = tmp_path / "metadata.json"
    if kind == "oversize":
        with path.open("wb") as stream:
            stream.truncate(METADATA_LIMIT + 1)
    elif kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text("{}", encoding="utf-8")
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    with pytest.raises(OpsError) as caught:
        cli._read_document(path)
    message = "native staging must not follow symlinks" if kind == "symlink" else "native staging requires bounded regular files"
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", message).problem


def test_metadata_byte_bound_is_inclusive_and_counts_utf8(tmp_path):
    path = tmp_path / "metadata.json"
    payload = '{"value":"é"}'.encode("utf-8")
    path.write_bytes(payload + b" " * (METADATA_LIMIT - len(payload)))
    assert path.stat().st_size == METADATA_LIMIT
    assert cli._read_document(path) == {"value": "é"}
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(OpsError) as caught:
        cli._read_document(path)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "native staging requires bounded regular files").problem


@pytest.mark.parametrize("name", ["spec", "request", "plan-ref"])
def test_cli_rejects_duplicate_keys_in_every_metadata_input_before_dispatch(source, capsys, name):
    source.paths[name].write_text('{"duplicate":1,"duplicate":2}', encoding="utf-8")
    result, stderr = _invoke(source, capsys, status=2)
    assert result == _refusal("INVALID_EXPERIMENT_SPEC", "search requires valid JSON object metadata")
    assert not stderr
    assert not source.calls


@pytest.mark.parametrize("action", ACTION_NAMES)
@pytest.mark.parametrize("name", ["spec", "request", "plan-ref"])
def test_excessively_nested_metadata_refuses_before_catalog_open(source, capsys, action, name):
    # CPython's C JSON decoder has a separate stack limit from sys.getrecursionlimit().
    depth = 1 << 20
    payload = b'{"nested":' + b"[" * depth + b"0" + b"]" * depth + b"}"
    assert len(payload) < METADATA_LIMIT
    # Prove real JSON decoding reaches the recursion handler, not a shallow syntax refusal.
    with pytest.raises(RecursionError):
        json.loads(payload)
    path = source.paths[name]
    path.write_bytes(payload)
    with pytest.raises(OpsError) as caught:
        cli._read_document(path)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
        "search requires valid JSON object metadata").problem
    before = source.catalog.read_bytes()
    result, stderr = _invoke(source, capsys, action, status=2)
    assert result == _refusal("INVALID_EXPERIMENT_SPEC", "search requires valid JSON object metadata")
    assert not stderr and not source.calls and not source.connections
    assert source.catalog.read_bytes() == before


@pytest.mark.parametrize("action", ACTION_NAMES)
@pytest.mark.parametrize("damage", ["missing_reference_field", "wrong_reference_type", "null_input_files"])
def test_typed_metadata_conversion_refuses_before_catalog_open(source, capsys, action, damage):
    name = "spec" if damage == "null_input_files" else "plan-ref"
    document = dict(source.documents[name])
    if damage == "missing_reference_field":
        del document["artifact_id"]
    elif damage == "wrong_reference_type":
        document["byte_size"] = "private-invalid-size"
    else:
        document["input_files"] = None
    # Prove the real converter, rather than JSON/file admission, reaches each handler.
    with pytest.raises(TypeError if name == "spec" else DocumentError):
        if name == "spec":
            cli.experiment_spec_from_document(document)
        else:
            cli.from_document(ArtifactRef, document)
    source.paths[name].write_text(json.dumps(document), encoding="utf-8")
    before = source.catalog.read_bytes()
    result, stderr = _invoke(source, capsys, action, status=2)
    assert result == _refusal("INVALID_EXPERIMENT_SPEC", "search metadata could not be admitted")
    assert not stderr and not source.calls and not source.connections
    assert source.catalog.read_bytes() == before


@pytest.mark.parametrize("action", ACTION_NAMES)
def test_typed_spec_refusal_retains_its_details_before_catalog_open(source, capsys, action):
    document = dict(source.documents["spec"])
    del document["hypothesis"]
    source.paths["spec"].write_text(json.dumps(document), encoding="utf-8")
    result, stderr = _invoke(source, capsys, action, status=2)
    assert result == {"refused": to_document(fail("INVALID_REQUEST",
        "experiment specification is missing a required field", details={"field": "hypothesis"}).problem)}
    assert not stderr and not source.calls and not source.connections


@pytest.mark.parametrize("missing", ["catalog", "store", "metadata"])
@pytest.mark.parametrize("action", ACTION_NAMES)
def test_missing_inputs_are_not_created_and_private_paths_are_redacted(source, capsys, missing, action):
    absent = source.root / "private-operator-path" / missing
    if missing == "metadata":
        source.paths["request"] = absent
    else:
        setattr(source, missing, absent)
    result, stderr = _invoke(source, capsys, action, status=2)
    message = "search requires an existing artifact store" if missing == "store" else "search request could not be completed"
    assert result == _refusal("RESOURCE_UNAVAILABLE", message)
    assert "private-operator-path" not in canonical_json(result) + stderr
    assert not absent.exists()
    assert not absent.parent.exists()
    assert not source.calls


def test_collect_uses_actual_read_only_sqlite_connection(source, monkeypatch, capsys):
    before = source.catalog.read_bytes()

    def collect(connection, *args, **kwargs):
        assert connection.row_factory is sqlite3.Row
        assert connection.execute("SELECT value FROM sentinel").fetchone()[0] == 7
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO sentinel VALUES (8)")
        return source.result

    monkeypatch.setattr(cli, "connect", _raising(AssertionError("collect opened writable catalog")))
    monkeypatch.setattr(cli, "collect_outer_results", collect)
    assert _invoke(source, capsys)[0] == source.result
    assert source.catalog.read_bytes() == before
    assert not list(source.store.rglob("*"))


@pytest.mark.parametrize("binding", ["missing", "different"])
def test_collect_requires_its_exact_driver_seal_before_reading_results(
        driver_case, monkeypatch, capsys, binding):
    case = driver_case
    case.recorder.seal(clock=case.clock)
    if binding == "different":
        _, sealed = cli._binding(case.conn, case.store, case.registration, case.request, case.plan_ref)
        records.append_record(case.conn, case.store, "run", "driver-seal", {**sealed, "plan_id": "other"})
    documents = {"spec": {"hypothesis": "synthetic", "primary_arm_id": "primary",
                           "economic_params": {}, "price_source": "computed_moves"},
                 "request": case.request, "plan-ref": to_document(case.plan_ref)}
    paths = {}
    for name, document in documents.items():
        paths[name] = case.catalog.parent / (name + ".json")
        paths[name].write_text(json.dumps(document), encoding="utf-8")
    source = SimpleNamespace(paths=paths, catalog=case.catalog, store=case.store.root, connections=[])
    def registration(connection, *args):
        source.connections.append(connection)
        return case.registration
    monkeypatch.setattr(cli, "read_native_registration", registration)
    monkeypatch.setattr(cli, "Repository", lambda *args: None)
    monkeypatch.setattr(cli, "collect_outer_results", _raising(AssertionError("unadmitted collection")))
    before = snapshot(case)
    result, stderr = _invoke(source, capsys, status=2)
    assert result == _refusal("EXPERIMENT_IDENTITY_CONFLICT", "search execution requires its exact driver seal")
    assert not stderr
    assert snapshot(case) == before
    _closed(source)


@pytest.mark.parametrize("exception", [OpsError, DataError])
@pytest.mark.parametrize("action", ACTION_NAMES)
def test_typed_refusal_preserves_the_complete_problem(source, monkeypatch, capsys, exception, action):
    problem = replace(fail("RESOURCE_UNAVAILABLE", "safe shared refusal", stage="admission",
                           details={"field": "feature_snapshot"}, retry_after_seconds=11,
                           dependency_refs=("sha256:dependency",)).problem,
                      trace_id="trace-id", diagnostic_ref="diagnostic-id")
    monkeypatch.setattr(cli, ACTION_NAMES[action], _raising(exception(problem)))
    result, stderr = _invoke(source, capsys, action, status=2)
    assert result == {"refused": to_document(problem)}
    assert not stderr
    _closed(source)


@pytest.mark.parametrize("error", [OSError("private-disk-path"),
                                   sqlite3.OperationalError("private-table-name"),
                                   ArtifactError("UNSAFE_PATH", "private-artifact-path")])
@pytest.mark.parametrize("action", ACTION_NAMES)
def test_untyped_infrastructure_failures_are_safely_redacted(source, monkeypatch, capsys, error, action):
    monkeypatch.setattr(cli, ACTION_NAMES[action], _raising(error))
    result, stderr = _invoke(source, capsys, action, status=2)
    assert result == _refusal("RESOURCE_UNAVAILABLE", "search request could not be completed")
    assert "private" not in canonical_json(result) + stderr
    assert not stderr
    _closed(source)


@pytest.mark.parametrize("exception", [TypeError, ValueError, DocumentError, KeyboardInterrupt])
@pytest.mark.parametrize("action", ACTION_NAMES)
def test_unexpected_programming_errors_are_not_silenced(source, monkeypatch, capsys, exception, action):
    error = DocumentError("BAD_TYPE", "$", "unexpected bug") if exception is DocumentError else exception("unexpected bug")
    monkeypatch.setattr(cli, ACTION_NAMES[action], _raising(error))
    with pytest.raises(exception, match="unexpected bug") as caught:
        cli.main(_args(source, action))
    assert caught.value is error
    assert not capsys.readouterr().out
    _closed(source)


@pytest.mark.parametrize("action", ACTION_NAMES)
def test_registration_refusal_closes_catalog_before_dispatch(source, monkeypatch, capsys, action):
    error = fail("EXPERIMENT_IDENTITY_CONFLICT", "synthetic registration mismatch")
    monkeypatch.setattr(cli, "read_native_registration", _raising(error))
    result, stderr = _invoke(source, capsys, action, status=2)
    assert result == {"refused": to_document(error.problem)}
    assert not stderr and not source.calls
    _closed(source)


def test_incidental_python_stdout_moves_to_stderr(source, monkeypatch, capsys):
    def collect(*args, **kwargs):
        print("synthetic progress")
        return source.result

    monkeypatch.setattr(cli, "collect_outer_results", collect)
    result, stderr = _invoke(source, capsys)
    assert result == source.result
    assert stderr == "synthetic progress\n"


@pytest.mark.parametrize("scratch", [False, True])
def test_render_budget_counts_utf8_and_final_newline_inclusively(monkeypatch, scratch):
    assert cli.MAX_REPORT_BYTES == 64 * 1024 * 1024
    document = {"value": "é🧪"}
    expected = canonical_json(document) + "\n"
    budget = len(expected.encode("utf-8"))
    assert budget > len(expected)
    monkeypatch.setattr(cli, "MAX_REPORT_BYTES", budget)
    with cli._render(document, scratch=scratch) as stream:
        assert stream.read() == expected
    monkeypatch.setattr(cli, "MAX_REPORT_BYTES", budget - 1)
    with pytest.raises(OpsError) as caught:
        cli._render(document, scratch=scratch)
    assert caught.value.problem == fail("RESOURCE_LIMIT_EXCEEDED", "search report exceeds its stdout byte budget").problem


def test_report_overflow_emits_one_refusal_and_no_partial_success(source, monkeypatch, capsys):
    assert cli.MAX_REPORT_BYTES == 64 * 1024 * 1024
    monkeypatch.setattr(cli, "MAX_REPORT_BYTES", 1024)
    source.result = {"first_success_field": "partial must stay private", "padding": "é" * 1024}
    result, stderr = _invoke(source, capsys, status=2)
    assert result == _refusal("RESOURCE_LIMIT_EXCEEDED", "search report exceeds its stdout byte budget")
    assert "first_success_field" not in canonical_json(result) + stderr
    assert "partial must stay private" not in canonical_json(result) + stderr
    _closed(source)


def test_tempfile_allocation_failure_still_emits_redacted_refusal(source, monkeypatch, capsys):
    monkeypatch.setattr(cli.tempfile, "TemporaryFile", _raising(OSError("private-scratch-path")))
    result, stderr = _invoke(source, capsys, status=2)
    assert result == _refusal("RESOURCE_UNAVAILABLE", "search request could not be completed")
    assert "private-scratch-path" not in canonical_json(result) + stderr
    _closed(source)


def test_tempfile_write_failure_discards_partial_report_and_closes_scratch(source, monkeypatch, capsys):
    class FailingScratch(io.StringIO):
        def write(self, text):
            if self.tell():
                raise OSError("private-full-disk-path")
            return super().write(text)

    stream = FailingScratch()
    monkeypatch.setattr(cli.tempfile, "TemporaryFile", lambda *args, **kwargs: stream)
    result, stderr = _invoke(source, capsys, status=2)
    assert result == _refusal("RESOURCE_UNAVAILABLE", "search request could not be completed")
    assert "private-full-disk-path" not in canonical_json(result) + stderr
    assert stream.closed
    _closed(source)


def test_refusal_staging_failure_still_closes_catalog(source, monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_outer_results", _raising(fail("RESOURCE_UNAVAILABLE", "safe refusal")))
    monkeypatch.setattr(cli.io, "StringIO", _raising(MemoryError("no refusal buffer")))
    with pytest.raises(MemoryError, match="no refusal buffer"):
        cli.main(_args(source))
    assert not capsys.readouterr().out
    _closed(source)


def test_broken_stdout_does_not_attempt_a_second_json_document(source, monkeypatch):
    staged = io.StringIO()
    monkeypatch.setattr(cli.tempfile, "TemporaryFile", lambda **kwargs: staged)
    class BrokenOutput:
        def __init__(self):
            self.writes = []

        def write(self, text):
            self.writes.append(text)
            raise BrokenPipeError("closed consumer")

    output = BrokenOutput()
    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdout", output)
        with pytest.raises(BrokenPipeError, match="closed consumer"):
            cli.main(_args(source))
    assert output.writes == [canonical_json(source.result) + "\n"]
    assert staged.closed
    _closed(source)


@pytest.mark.parametrize("encoding", ["latin-1", "utf-16"])
def test_binary_stdout_receives_exact_utf8_without_text_encoding_or_writes(source, monkeypatch, encoding):
    class HostileTextOutput:
        def __init__(self):
            self.buffer = io.BytesIO()
            self.encoding = encoding

        def write(self, text):
            raise AssertionError("stdout text encoding must not alter canonical UTF-8")

    # Exercise several copy chunks, including multibyte characters at chunk boundaries.
    source.result = {"value": "é🧪" * 40000}
    output = HostileTextOutput()
    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdout", output)
        assert cli.main(_args(source)) == 0
    assert output.buffer.getvalue() == (canonical_json(source.result) + "\n").encode("utf-8")


def test_broken_binary_stdout_does_not_fall_back_or_emit_second_document(source, monkeypatch):
    staged = io.StringIO()
    monkeypatch.setattr(cli.tempfile, "TemporaryFile", lambda **kwargs: staged)
    class BrokenBuffer:
        def __init__(self):
            self.writes = []

        def write(self, data):
            self.writes.append(data)
            raise BrokenPipeError("closed binary consumer")

    class BinaryOutput:
        def __init__(self):
            self.buffer = BrokenBuffer()

        def write(self, text):
            raise AssertionError("broken binary stdout must not fall back to text")

    output = BinaryOutput()
    with monkeypatch.context() as patch:
        patch.setattr(cli.sys, "stdout", output)
        with pytest.raises(BrokenPipeError, match="closed binary consumer"):
            cli.main(_args(source))
    assert output.buffer.writes == [(canonical_json(source.result) + "\n").encode("utf-8")]
    assert staged.closed
    _closed(source)


@pytest.mark.parametrize("binary", [False, True], ids=["text", "binary"])
@pytest.mark.parametrize("large", [False, True], ids=["small", "buffered-tail"])
@pytest.mark.parametrize("refused", [False, True], ids=["report", "refusal"])
@pytest.mark.parametrize("failure", [None, errno.EPIPE, errno.ENOSPC], ids=["success", "broken-pipe", "full-disk"])
def test_buffered_stdout_finishes_or_fails_before_main_returns(source, monkeypatch, binary, large, refused, failure):
    source.result = {"value": "é🧪" + ("x" * 65536 if large else "small")}
    document, status = source.result, 0
    if refused:
        refusal = fail("RESOURCE_UNAVAILABLE", "safe refusal", details=source.result)
        monkeypatch.setattr(cli, "collect_outer_results", _raising(refusal))
        document, status = {"refused": to_document(refusal.problem)}, 2
    expected = (canonical_json(document) + "\n").encode("utf-8")
    staged, render = [], cli._render
    def track_render(*args, **kwargs):
        stream = render(*args, **kwargs)
        staged.append(stream)
        return stream
    monkeypatch.setattr(cli, "_render", track_render)
    error = OSError(failure, "synthetic output failure") if failure else None
    emitting = True

    class Sink(io.RawIOBase):
        def __init__(self):
            self.accepted, self.attempts = bytearray(), []
        def writable(self):
            return True
        def write(self, data):
            data = bytes(data)
            if emitting:
                assert len(staged) == 1
                _closed(source)
                self.attempts.append(data)
                # Let any complete large prefix through; only the buffered tail fails.
                if error and len(self.accepted) + len(data) == len(expected):
                    raise error
            self.accepted.extend(data)
            return len(data)

    sink = Sink()
    buffered = io.BufferedWriter(sink, buffer_size=8192)
    text = None if binary else io.TextIOWrapper(buffered, encoding="utf-8", newline="")
    output = (SimpleNamespace(buffer=buffered,
        write=_raising(AssertionError("binary output must not use text writes")),
        flush=_raising(AssertionError("flush the selected binary destination"))) if binary
        else SimpleNamespace(write=text.write, flush=text.flush))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(cli.sys, "stdout", output)
            if error:
                with pytest.raises(OSError) as caught:
                    cli.main(_args(source))
                assert caught.value is error
            else:
                assert cli.main(_args(source)) == status
        assert b"".join(sink.attempts) == expected  # no second document, retry or fallback
        assert bytes(sink.accepted) == (b"".join(sink.attempts[:-1]) if error else expected)
        assert not sink.closed and not buffered.closed  # the caller owns stdout
        assert len(staged) == 1 and staged[0].closed
        _closed(source)
    finally:
        # Drain only during test cleanup, after the assertions, without re-raising the simulated fault.
        emitting = False
        (text if text is not None else buffered).close()


@pytest.mark.parametrize("flag", RESOURCE_FLAGS)
@pytest.mark.parametrize("value", [1, sys.maxsize - 1])
def test_cli_accepts_each_exact_resource_boundary(source, capsys, flag, value):
    arguments = _args(source, "run")
    arguments[arguments.index(flag) + 1] = str(value)
    assert cli.main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == source.result
    expected = {**LIMITS, flag[2:].replace("-", "_"): value}
    assert source.calls[0][2]["limits"] == expected


def test_cli_uses_shared_resource_admission_before_any_storage(source, monkeypatch, capsys):
    calls = []
    def reject(limits):
        calls.append(limits)
        raise fail("INVALID_EXPERIMENT_SPEC", "synthetic shared resource refusal")
    monkeypatch.setattr(cli, "validate_execution_limits", reject)
    monkeypatch.setattr(cli, "_read_document", _raising(AssertionError("storage before resource admission")))
    with pytest.raises(SystemExit) as caught:
        cli.main(_args(source, "run"))
    assert caught.value.code == 2
    _parser_error(capsys, "synthetic shared resource refusal")
    assert calls == [LIMITS]
    assert not source.calls


@pytest.mark.parametrize("action", ACTION_NAMES)
@pytest.mark.parametrize("flag", ["--spec", "--request", "--plan-ref", "--catalog", "--store-root"])
def test_embedded_nul_paths_refuse_before_storage(source, monkeypatch, capsys, action, flag):
    arguments = _args(source, action)
    arguments[arguments.index(flag) + 1] = "private-path\0suffix"
    monkeypatch.setattr(cli, "_read_document", _raising(AssertionError("storage before path admission")))
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments)
    assert caught.value.code == 2
    _parser_error(capsys, "search paths must not contain NUL characters")
    assert not source.calls
