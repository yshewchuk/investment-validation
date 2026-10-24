"""Synthetic regressions for the controlled backup failure drill."""
from __future__ import annotations

import hashlib

from tools.v2_controlled_failure_drill import _manifest_is_consistent


def test_manifest_check_honors_selected_file_and_refuses_symlink(tmp_path):
    content = b"synthetic backup database"
    database = tmp_path / "selected.sqlite"
    database.write_bytes(content)
    manifest = {
        "schema_version": "catalog_backup.v1.0",
        "backup_key": "drill",
        "database_file": "selected.sqlite",
        "database_size": len(content),
        "database": "sha256:" + hashlib.sha256(content).hexdigest(),
        "artifacts": {},
    }

    assert _manifest_is_consistent(manifest, tmp_path, "drill")

    outside = tmp_path / "outside.sqlite"
    outside.write_bytes(content)
    database.unlink()
    database.symlink_to(outside)

    assert not _manifest_is_consistent(manifest, tmp_path, "drill")
