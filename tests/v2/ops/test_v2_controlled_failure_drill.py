"""Focused synthetic coverage for the controlled-failure drill's bounded
backup-manifest load (``_read_verified_backup_manifest``).

A valid JSON manifest parses and reaches ``_manifest_is_consistent``; a
malformed, oversize or symlinked manifest returns ``(None, False)`` and never
reaches the consistency check at all, so a failure is never verified. The
oversize case monkeypatches the module's ``MAX_MANIFEST_BYTES`` down to a few
bytes so the test never writes or reads a 1 MiB file.
"""
from __future__ import annotations

import json

from engine.v2.ops.fingerprints import file_hash
from tools import v2_controlled_failure_drill as drill

KEY = "focused-drill"


def _record_consistency_checks(monkeypatch):
    """Replace the consistency check with a recorder so a failure case can
    assert the check was never called, not merely that it returned False."""
    calls = []
    monkeypatch.setattr(
        drill, "_manifest_is_consistent",
        lambda manifest, root, key: calls.append((manifest, root, key)) or False)
    return calls


def test_valid_json_manifest_parses_and_verifies(tmp_path):
    database = tmp_path / f"{KEY}.sqlite"
    database.write_bytes(b"synthetic catalog bytes")
    manifest = {
        "schema_version": "catalog_backup.v1.0",
        "backup_key": KEY,
        "database": file_hash(database),
        "artifacts": {},
    }
    path = tmp_path / f"{KEY}.manifest.json"
    path.write_text(json.dumps(manifest))

    loaded, verified = drill._read_verified_backup_manifest(path, tmp_path, KEY)

    assert loaded == manifest
    assert verified is True


def test_malformed_manifest_json_is_never_verified(tmp_path, monkeypatch):
    path = tmp_path / f"{KEY}.manifest.json"
    path.write_bytes(b'{"schema_version": ')
    calls = _record_consistency_checks(monkeypatch)

    loaded, verified = drill._read_verified_backup_manifest(path, tmp_path, KEY)

    assert (loaded, verified) == (None, False)
    assert calls == []


def test_oversize_manifest_is_never_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(drill, "MAX_MANIFEST_BYTES", 8)
    path = tmp_path / f"{KEY}.manifest.json"
    path.write_bytes(b'{"schema_version": "catalog_backup.v1.0"}')
    calls = _record_consistency_checks(monkeypatch)

    loaded, verified = drill._read_verified_backup_manifest(path, tmp_path, KEY)

    assert (loaded, verified) == (None, False)
    assert calls == []


def test_symlinked_manifest_is_never_verified(tmp_path, monkeypatch):
    target = tmp_path / "real-manifest.json"
    target.write_text(json.dumps({"schema_version": "catalog_backup.v1.0"}))
    path = tmp_path / f"{KEY}.manifest.json"
    path.symlink_to(target)
    calls = _record_consistency_checks(monkeypatch)

    loaded, verified = drill._read_verified_backup_manifest(path, tmp_path, KEY)

    assert (loaded, verified) == (None, False)
    assert calls == []