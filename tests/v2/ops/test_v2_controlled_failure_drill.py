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

import pytest

from engine.v2.ops import backup as backup_module
from engine.v2.ops.fingerprints import file_hash
from tools import v2_controlled_failure_drill as drill

KEY = "focused-drill"


def _base_manifest(root, artifacts=None):
    """A minimal legacy-shaped manifest with the database file it names."""
    database = root / f"{KEY}.sqlite"
    database.write_bytes(b"synthetic catalog bytes")
    manifest = {
        "schema_version": "catalog_backup.v1.0",
        "backup_key": KEY,
        "cutoff": "2026-09-12T00:00:00.000000Z",
        "database": file_hash(database),
        "artifacts": {} if artifacts is None else artifacts,
    }
    return database, manifest


def _record_consistency_checks(monkeypatch):
    """Replace the consistency check with a recorder so a failure case can
    assert the check was never called, not merely that it returned False."""
    calls = []
    monkeypatch.setattr(
        drill, "_manifest_is_consistent",
        lambda manifest, root, key: calls.append((manifest, root, key)) or False)
    return calls


def test_valid_json_manifest_parses_and_verifies(tmp_path):
    _, manifest = _base_manifest(tmp_path)
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


def test_valid_shape_and_hashes_is_consistent(tmp_path):
    _, manifest = _base_manifest(tmp_path)
    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is True


def test_malformed_shape_is_inconsistent(tmp_path):
    _, manifest = _base_manifest(tmp_path)
    del manifest["cutoff"]

    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is False


def test_list_artifacts_is_inconsistent(tmp_path):
    _, manifest = _base_manifest(tmp_path)
    manifest["artifacts"] = ["evidence.bin"]

    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is False


@pytest.mark.parametrize("name", ["../escape.bin", "a/b.bin"])
def test_traversal_artifact_name_is_inconsistent(tmp_path, name):
    _, manifest = _base_manifest(tmp_path)
    manifest["artifacts"] = {name: {"content_hash": "sha256:" + "0" * 64}}

    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is False


def test_traversal_database_file_is_inconsistent(tmp_path):
    _, manifest = _base_manifest(tmp_path)
    manifest["database_file"] = "../escape.sqlite"

    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is False


def test_symlinked_artifact_is_inconsistent(tmp_path):
    _, manifest = _base_manifest(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"evidence")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "evidence.bin").symlink_to(outside)
    manifest["artifacts"] = {"evidence.bin": {"content_hash": file_hash(outside)}}

    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is False


def test_oversized_selected_file_is_inconsistent(tmp_path, monkeypatch):
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", 4)
    _, manifest = _base_manifest(tmp_path)

    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is False


def test_hash_mismatch_is_inconsistent(tmp_path):
    _, manifest = _base_manifest(tmp_path)
    manifest["database"] = "sha256:" + "0" * 64

    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is False


def test_artifact_byte_size_mismatch_is_inconsistent(tmp_path):
    _, manifest = _base_manifest(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "evidence.bin").write_bytes(b"evidence")
    manifest["artifacts"] = {
        "evidence.bin": {"content_hash": file_hash(artifacts / "evidence.bin"),
                         "byte_size": 1024}}

    assert drill._manifest_is_consistent(manifest, tmp_path, KEY) is False