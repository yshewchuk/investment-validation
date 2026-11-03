"""A real reader snapshot, not an idle connection, retains live-WAL evidence."""
from contextlib import closing
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.v2.research.distribution_prediction_operator_support import catalog_bytes, pinned_wal_snapshot


@pytest.mark.parametrize("mode", ["idle", "begin_only", "snapshot"])
def test_wal_fixture_pins_snapshot_against_automatic_checkpoint(tmp_path, mode):
    catalog = tmp_path / "catalog.sqlite"
    keeper = sqlite3.connect(catalog, isolation_level=None)
    try:
        assert keeper.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        keeper.execute("CREATE TABLE completed(value TEXT)")
        assert keeper.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        checkpointed = catalog_bytes(catalog)
        uri = catalog.resolve().as_uri() + "?mode=ro"
        context = (pinned_wal_snapshot(catalog) if mode == "snapshot"
            else closing(sqlite3.connect(uri, uri=True, isolation_level=None)))
        with context as reader:
            if mode == "begin_only":
                reader.execute("BEGIN")  # No fetched read, so no pinned snapshot.
            writer = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("PRAGMA wal_autocheckpoint=1")
conn.execute("INSERT INTO completed VALUES(?)", ("committed" * 1024,))
assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
conn.close()
"""
            subprocess.run([sys.executable, "-c", writer, str(catalog)], check=True, timeout=10)
            # A real external CLI closes its last writer connection on return.
            assert (catalog_bytes(catalog) == checkpointed) is (mode == "snapshot")
            assert reader.execute("SELECT count(*) FROM completed").fetchone()[0] == (0 if mode == "snapshot" else 1)
            assert reader.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            assert catalog.with_name(catalog.name + "-wal").stat().st_size > 0
            durable = {p.name: catalog_bytes(p) for p in tmp_path.glob("catalog.sqlite*")
                if not p.name.endswith("-shm")}
            for _ in range(3):
                with closing(sqlite3.connect(uri, uri=True, isolation_level=None)) as fresh:
                    assert fresh.execute("SELECT count(*) FROM completed").fetchone()[0] == 1
                    assert fresh.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
                assert {p.name: catalog_bytes(p) for p in tmp_path.glob("catalog.sqlite*")
                    if not p.name.endswith("-shm")} == durable
        assert keeper.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        assert keeper.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    finally:
        keeper.close()
    durable = {p.name: catalog_bytes(p) for p in tmp_path.glob("catalog.sqlite*")}
    assert set(durable) == {catalog.name} and catalog_bytes(catalog)[18:20] == b"\x01\x01"
    for _ in range(3):
        with closing(sqlite3.connect(uri, uri=True, isolation_level=None)) as reader:
            assert reader.execute("SELECT count(*) FROM completed").fetchone()[0] == 1
        assert {p.name: catalog_bytes(p) for p in tmp_path.glob("catalog.sqlite*")} == durable


def test_catalog_byte_reader_never_opens_file_in_lock_owning_process(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.sqlite"
    payload = bytes(range(256))
    catalog.write_bytes(payload)

    def forbidden_read(self):
        raise AssertionError("raw catalog read in SQLite lock-owning process")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    assert catalog_bytes(catalog) == payload
