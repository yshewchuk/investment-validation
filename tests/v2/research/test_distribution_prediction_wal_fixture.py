"""A real reader snapshot, not an idle connection, retains live-WAL evidence."""
from contextlib import closing
import sqlite3

import pytest

from tests.v2.research.distribution_prediction_operator_support import pinned_wal_snapshot


@pytest.mark.parametrize("mode", ["idle", "begin_only", "snapshot"])
def test_wal_fixture_pins_snapshot_against_automatic_checkpoint(tmp_path, mode):
    catalog = tmp_path / "catalog.sqlite"
    keeper = sqlite3.connect(catalog, isolation_level=None)
    try:
        assert keeper.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        keeper.execute("CREATE TABLE completed(value TEXT)")
        assert keeper.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        checkpointed = catalog.read_bytes()
        uri = catalog.resolve().as_uri() + "?mode=ro"
        context = (pinned_wal_snapshot(catalog) if mode == "snapshot"
            else closing(sqlite3.connect(uri, uri=True, isolation_level=None)))
        with context as reader:
            if mode == "begin_only":
                reader.execute("BEGIN")  # No fetched read, so no pinned snapshot.
            with closing(sqlite3.connect(catalog, isolation_level=None)) as writer:
                assert writer.execute("PRAGMA wal_autocheckpoint=1").fetchone()[0] == 1
                writer.execute("INSERT INTO completed VALUES(?)", ("committed" * 1024,))
                # The negative controls deterministically invalidate the old native assertion.
                assert (catalog.read_bytes() == checkpointed) is (mode == "snapshot")
                assert reader.execute("SELECT count(*) FROM completed").fetchone()[0] == (0 if mode == "snapshot" else 1)
                assert catalog.with_name(catalog.name + "-wal").stat().st_size > 0
                durable = {p.name: p.read_bytes() for p in tmp_path.glob("catalog.sqlite*")
                    if not p.name.endswith("-shm")}
                for _ in range(3):
                    with closing(sqlite3.connect(uri, uri=True, isolation_level=None)) as fresh:
                        assert fresh.execute("SELECT count(*) FROM completed").fetchone()[0] == 1
                    assert {p.name: p.read_bytes() for p in tmp_path.glob("catalog.sqlite*")
                        if not p.name.endswith("-shm")} == durable
        assert keeper.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        assert keeper.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    finally:
        keeper.close()
    durable = {p.name: p.read_bytes() for p in tmp_path.glob("catalog.sqlite*")}
    assert set(durable) == {catalog.name} and catalog.read_bytes()[18:20] == b"\x01\x01"
    for _ in range(3):
        with closing(sqlite3.connect(uri, uri=True, isolation_level=None)) as reader:
            assert reader.execute("SELECT count(*) FROM completed").fetchone()[0] == 1
        assert {p.name: p.read_bytes() for p in tmp_path.glob("catalog.sqlite*")} == durable
