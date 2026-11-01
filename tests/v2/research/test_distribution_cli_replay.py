"""Completed CLI replay keeps SQLite read-only while honoring committed live WAL."""
import sqlite3

import pytest

from engine.models.no_fit import no_fit_guard as legacy_no_fit
from engine.v2.foundation import ArtifactStore
from engine.v2.models.no_fit import no_fit_guard
from experiments import distribution_cli as cli
from experiments import distribution_run as runner
from experiments.magnitude_search_shards import SHARD_SCHEMA
from tests.v2.research.test_distribution_cli_acceptance import operator as operator
from tests.v2.research.test_distribution_run import case as case


def object_bytes(store):
    return {str(path.relative_to(store.root)): path.read_bytes()
        for path in store.root.rglob("*") if path.is_file()}


def readonly_replay(case, operator, monkeypatch, expected):
    opened, statements, denied, verified = [], [], [], []
    original_connect, original_read = sqlite3.connect, ArtifactStore.read_verified
    def forbidden(*args, **kwargs):
        pytest.fail("completed replay attempted writable access or execution")
    def authorize(action, first, second, database, trigger):
        statements.append((action, first, second))
        if action not in {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}:
            denied.append((action, first, second))
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    def connect(target, *args, **kwargs):
        assert str(target).endswith("?mode=ro") and kwargs.get("uri") is True
        conn = original_connect(target, *args, **kwargs)
        conn.set_authorizer(authorize)
        opened.append(conn)
        return conn
    def read(store, ref):
        verified.append(ref.schema_ref)
        return original_read(store, ref)
    monkeypatch.setattr(cli, "connect", forbidden)
    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(ArtifactStore, "_commit", forbidden)
    monkeypatch.setattr(ArtifactStore, "read_verified", read)
    monkeypatch.setattr(runner.DistributionRun, "_run_contained", forbidden)
    monkeypatch.setattr(runner, "DistributionFits", forbidden)
    before, calls, instances = object_bytes(case.store), list(case.calls), len(operator.instances)
    with legacy_no_fit(), no_fit_guard():
        assert operator.invoke("run") == (0, expected)
    assert len(opened) == 1 and len(operator.instances) == instances + 1
    assert statements and not denied
    assert verified.count(SHARD_SCHEMA) >= len(expected["sections"]) > 0
    assert object_bytes(case.store) == before and case.calls == calls
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


def test_completed_run_replays_closed_delete_catalog_without_writes_or_sidecars(case, operator, monkeypatch, tmp_path):
    assert operator.invoke("seal")[0] == 0
    status, completed = operator.invoke("run")
    assert status == 0 and operator.record("completed") == completed
    assert case.conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    case.conn.close()
    path = tmp_path / "catalog.sqlite"
    before = {entry.name: entry.read_bytes() for entry in tmp_path.glob("catalog.sqlite*")}
    assert set(before) == {path.name}
    assert before[path.name][18:20] == b"\x01\x01"  # Genuine closed rollback-journal DB.
    readonly_replay(case, operator, monkeypatch, completed)
    assert {entry.name: entry.read_bytes() for entry in tmp_path.glob("catalog.sqlite*")} == before


def test_completed_run_reads_uncheckpointed_live_wal_without_writable_fallback(case, operator, monkeypatch, tmp_path):
    assert operator.invoke("seal")[0] == 0
    assert operator.record("completed") is None
    assert case.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    path = tmp_path / "catalog.sqlite"
    checkpointed = path.read_bytes()
    status, completed = operator.invoke("run")
    assert status == 0 and operator.record("completed") == completed
    # The writer remains open. Completion exists only in WAL, not this main-file image.
    assert path.read_bytes() == checkpointed
    wal, shm = path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")
    assert wal.stat().st_size > 0 and shm.is_file()
    before = {entry.name: entry.read_bytes() for entry in (path, wal)}
    names = {entry.name for entry in tmp_path.glob("catalog.sqlite*")}
    readonly_replay(case, operator, monkeypatch, completed)
    assert {entry.name: entry.read_bytes() for entry in (path, wal)} == before
    assert {entry.name for entry in tmp_path.glob("catalog.sqlite*")} == names
    # SQLite may update SHM read marks even with mode=ro; no physical SHM immutability claim.
