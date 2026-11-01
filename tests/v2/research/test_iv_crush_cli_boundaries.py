"""IV operator preserves input/integrity failures and probes replay read-only."""
import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactError, ArtifactStore
from experiments import iv_crush_run as cli
from experiments import size_search
from tests.v2.research.test_iv_crush_cli import arguments, request


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    for name, document in (("request", request()), ("spec", {}), ("ref", {})):
        (tmp_path / f"{name}.json").write_text(json.dumps(document))
    store = ArtifactStore(tmp_path / "store")
    store.root.mkdir()
    catalog = tmp_path / "catalog.sqlite"
    with sqlite3.connect(catalog) as conn:
        conn.execute("CREATE TABLE terminal (kind TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO terminal VALUES (?)", [("outer",), ("comparison",)])
    calls = []
    monkeypatch.setattr(cli, "_contain", lambda request: 1)
    monkeypatch.setattr(cli, "rss_watchdog", lambda memory: nullcontext())
    monkeypatch.setattr(cli.signal, "alarm", lambda seconds: 0)
    monkeypatch.setattr(cli, "experiment_spec_from_document", lambda doc: SimpleNamespace(
        runner="experiments/iv_crush_search.py", price_source="iv_crush_targets"))
    monkeypatch.setattr(cli, "from_document", lambda *a: "ref")
    monkeypatch.setattr(cli, "Repository", lambda *a, **kw: None)
    def registration(conn, *args):
        calls.append("registration")
        return SimpleNamespace(run_id="test")
    monkeypatch.setattr(cli, "read_native_registration", registration)
    monkeypatch.setattr(size_search, "_admit_driver", lambda *a: {
        "plan_id": "test", "folds": [{"outer": {"year": "2024"}}]})
    def record(conn, store, run_id, slot):
        kind = slot.split("-", 1)[0]
        return {} if conn.execute("SELECT 1 FROM terminal WHERE kind=?", (kind,)).fetchone() else None
    monkeypatch.setattr(size_search, "read_record", record)
    def collect(*context):
        calls.append("collect")
        return {"verified": True}
    monkeypatch.setattr(cli, "collect", collect)
    def forbidden(*a, **kw):
        pytest.fail("completed run opened writable catalog or execution budget")
    monkeypatch.setattr(cli, "connect", forbidden)
    monkeypatch.setattr(size_search, "remaining", forbidden)
    monkeypatch.setattr(size_search, "run_search", forbidden)
    return SimpleNamespace(root=tmp_path, catalog=catalog, store=store, calls=calls)


@pytest.mark.parametrize("name", ["request", "spec", "ref"])
@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_metadata_io_is_invalid_input_not_storage_outage(boundary, monkeypatch, capsys, name, failure):
    path = boundary.root / f"{name}.json"
    if failure == "missing":
        path.unlink()
    else:
        original = cli._read_document
        def read(candidate):
            if Path(candidate) == path:
                raise PermissionError("sensitive-private-path")
            return original(candidate)
        monkeypatch.setattr(cli, "_read_document", read)
    assert cli.main(arguments(boundary.root)) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["refused"] == "INVALID_EXPERIMENT_SPEC"
    assert "sensitive" not in output
    assert "collect" not in boundary.calls


@pytest.mark.parametrize("code, expected", [
    ("INTEGRITY_FAILED", "INTEGRITY_FAILED"), ("MISSING", "RESOURCE_UNAVAILABLE"),
    ("UNSAFE_PATH", "INTEGRITY_FAILED"), ("NOT_A_REGULAR_FILE", "INTEGRITY_FAILED"),
    ("SIZE_LIMIT", "RESOURCE_LIMIT_EXCEEDED")])
def test_artifact_failures_keep_integrity_distinct(boundary, monkeypatch, capsys, code, expected):
    def refuse(*context):
        raise ArtifactError(code, "sensitive-private-path")
    monkeypatch.setattr(cli, "collect", refuse)
    assert cli.main(arguments(boundary.root, "run")) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["refused"] == expected
    assert "sensitive" not in output
    assert boundary.calls == ["registration"]


def test_completed_delete_catalog_is_byte_stable_without_wal(boundary, capsys):
    before = boundary.catalog.read_bytes(), boundary.catalog.stat().st_mtime_ns
    assert cli.main(arguments(boundary.root, "run")) == 0
    assert json.loads(capsys.readouterr().out) == {"verified": True}
    assert (boundary.catalog.read_bytes(), boundary.catalog.stat().st_mtime_ns) == before
    assert not Path(str(boundary.catalog) + "-wal").exists()
    assert not Path(str(boundary.catalog) + "-shm").exists()
    assert boundary.calls == ["registration", "collect"]


def test_completed_run_observes_committed_live_wal(boundary, capsys):
    with sqlite3.connect(boundary.catalog, isolation_level=None) as writer:
        writer.execute("DELETE FROM terminal")
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.executemany("INSERT INTO terminal VALUES (?)", [("outer",), ("comparison",)])
        with sqlite3.connect(boundary.catalog.as_uri() + "?immutable=1", uri=True) as stale:
            assert stale.execute("SELECT COUNT(*) FROM terminal").fetchone()[0] == 0
        assert cli.main(arguments(boundary.root, "run")) == 0
        assert json.loads(capsys.readouterr().out) == {"verified": True}
    assert boundary.calls == ["registration", "collect"]


def test_incomplete_run_reopens_and_readmits_before_execution(boundary, monkeypatch, capsys):
    with sqlite3.connect(boundary.catalog) as conn:
        conn.execute("DELETE FROM terminal WHERE kind='comparison'")
    real_connect = sqlite3.connect
    def writable(path, *, must_exist):
        assert must_exist
        boundary.calls.append("writable")
        return real_connect(path, isolation_level=None)
    monkeypatch.setattr(cli, "connect", writable)
    monkeypatch.setattr(size_search, "remaining", lambda *a: 1)
    def execute(*context, **kwargs):
        boundary.calls.append("execute")
        context[0].execute("INSERT INTO terminal VALUES ('comparison')")
    monkeypatch.setattr(size_search, "run_search", execute)
    assert cli.main(arguments(boundary.root, "run")) == 0
    assert json.loads(capsys.readouterr().out) == {"verified": True}
    assert boundary.calls == ["registration", "writable", "registration", "execute", "collect"]


@pytest.mark.parametrize("action", ["run", "report"])
def test_actual_corrupt_cas_refuses_without_writable_fallback(boundary, monkeypatch, capsys, action):
    ref = boundary.store.publish_bytes(b"original", schema_ref="synthetic.v1")
    path = boundary.store.root / ref.storage_key
    path.chmod(0o600)
    path.write_bytes(b"tampered")
    monkeypatch.setattr(cli, "collect", lambda *context: boundary.store.read_verified(ref))
    before = boundary.catalog.read_bytes(), path.read_bytes()
    assert cli.main(arguments(boundary.root, action)) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["refused"] == "INTEGRITY_FAILED"
    assert ref.storage_key not in output
    assert (boundary.catalog.read_bytes(), path.read_bytes()) == before
    assert boundary.calls == ["registration"]


def test_probe_identity_refusal_never_reopens_writable(boundary, monkeypatch, capsys):
    from engine.v2.ops.errors import fail
    def refuse(*args):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "synthetic driver mismatch")
    monkeypatch.setattr(size_search, "_admit_driver", refuse)
    assert cli.main(arguments(boundary.root, "run")) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert boundary.calls == ["registration"]


@pytest.mark.parametrize("missing", ["catalog.sqlite", "store"])
def test_missing_storage_stays_resource_unavailable(boundary, capsys, missing):
    path = boundary.root / missing
    path.rmdir() if path.is_dir() else path.unlink()
    assert cli.main(arguments(boundary.root)) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == "RESOURCE_UNAVAILABLE"
