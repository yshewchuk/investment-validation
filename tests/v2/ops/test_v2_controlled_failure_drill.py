"""Synthetic regression for restore's existing-destination refusal."""
from __future__ import annotations

import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup
from engine.v2.ops.errors import OpsError
from tests.ops_support import catalog


def test_restore_refuses_existing_destination_without_overwriting_database(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "restore-refusal", {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="restore-refusal", owner="worker", target=backup,
               clock=clock, store=store)

    restored = restore_backup(backup, tmp_path / "restored")
    database = restored / "ops.sqlite"
    before = database.read_bytes()

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")

    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert database.read_bytes() == before