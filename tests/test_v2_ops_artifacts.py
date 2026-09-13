"""Durable artifact publication — phase-1 guide §7.1 (slice P1-1).

Every test plants the unsafe condition the store exists to refuse, and asserts
the durable effect on disk rather than an exit status.
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.foundation import ArtifactError, ArtifactStore, safe_relative_path  # noqa: E402

SCHEMA = "synthetic_blob.v1.0"


def _stage(store: ArtifactStore, attempt: str, rel: str, data: bytes) -> Path:
    path = store.staging_dir(attempt) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _objects(root: Path) -> list[Path]:
    return sorted(p for p in (root / "objects").rglob("*") if p.is_file()) \
        if (root / "objects").exists() else []


def test_publish_copies_hashes_and_freezes_the_object(tmp_path):
    store = ArtifactStore(tmp_path)
    _stage(store, "att_1", "out/scores.json", b'{"a":1}')
    ref = store.publish_candidate("att_1", "out/scores.json", schema_ref=SCHEMA)

    digest = hashlib.sha256(b'{"a":1}').hexdigest()
    assert ref.content_hash == f"sha256:{digest}"
    assert ref.storage_key == f"objects/{digest[:2]}/{digest}"
    assert ref.byte_size == 7
    obj = tmp_path / ref.storage_key
    assert obj.read_bytes() == b'{"a":1}'
    assert stat.S_IMODE(obj.stat().st_mode) == 0o444
    assert store.verify(ref) == obj
    assert store.read_verified(ref) == b'{"a":1}'


def test_same_bytes_and_schema_are_one_artifact(tmp_path):
    store = ArtifactStore(tmp_path)
    _stage(store, "att_1", "x", b"same")
    _stage(store, "att_2", "y", b"same")
    first = store.publish_candidate("att_1", "x", schema_ref=SCHEMA)
    second = store.publish_candidate("att_2", "y", schema_ref=SCHEMA)
    other_schema = store.publish_bytes(b"same", schema_ref="other.v1.0")
    assert first == second
    assert other_schema.content_hash == first.content_hash
    assert other_schema.artifact_id != first.artifact_id
    assert len(_objects(tmp_path)) == 1


@pytest.mark.parametrize("rel", ["../escape", "/etc/passwd", "a//b", "a/./b", "", "a\\b",
                                 "a/../../b", "nul\x00"])
def test_traversal_is_refused_before_touching_the_filesystem(tmp_path, rel):
    with pytest.raises(ArtifactError) as err:
        safe_relative_path(rel)
    assert err.value.code == "UNSAFE_PATH"
    store = ArtifactStore(tmp_path)
    store.staging_dir("att_1")
    with pytest.raises(ArtifactError):
        store.publish_candidate("att_1", rel, schema_ref=SCHEMA)
    assert _objects(tmp_path) == []


@pytest.mark.parametrize("attempt", ["a/b", "..", "", "."])
def test_attempt_id_must_be_one_segment(tmp_path, attempt):
    with pytest.raises(ArtifactError):
        ArtifactStore(tmp_path).staging_dir(attempt)


def test_symlinked_directory_component_is_refused(tmp_path):
    production = tmp_path / "production"
    production.mkdir()
    (production / "ledger.jsonl").write_bytes(b"authoritative")
    store = ArtifactStore(tmp_path / "ops")
    staging = store.staging_dir("att_1")
    (staging / "sub").symlink_to(production, target_is_directory=True)
    with pytest.raises(ArtifactError) as err:
        store.publish_candidate("att_1", "sub/ledger.jsonl", schema_ref=SCHEMA)
    assert err.value.code == "UNSAFE_PATH"
    assert _objects(tmp_path / "ops") == []


def test_symlinked_file_is_refused(tmp_path):
    target = tmp_path / "outside.txt"
    target.write_bytes(b"outside")
    store = ArtifactStore(tmp_path / "ops")
    (store.staging_dir("att_1") / "link").symlink_to(target)
    with pytest.raises(ArtifactError) as err:
        store.publish_candidate("att_1", "link", schema_ref=SCHEMA)
    assert err.value.code == "UNSAFE_PATH"


def test_symlinked_staging_directory_is_refused(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "f").write_bytes(b"x")
    store = ArtifactStore(tmp_path / "ops")
    attempt_dir = store.root / "attempts" / "att_1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "staging").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ArtifactError):
        store.publish_candidate("att_1", "f", schema_ref=SCHEMA)


def test_hard_link_back_to_production_is_not_a_private_copy(tmp_path):
    production = tmp_path / "prod.jsonl"
    production.write_bytes(b"authoritative")
    store = ArtifactStore(tmp_path / "ops")
    os.link(production, store.staging_dir("att_1") / "prod.jsonl")
    with pytest.raises(ArtifactError) as err:
        store.publish_candidate("att_1", "prod.jsonl", schema_ref=SCHEMA)
    assert err.value.code == "UNSAFE_PATH"


def test_fifo_is_refused_without_blocking(tmp_path):
    store = ArtifactStore(tmp_path)
    os.mkfifo(store.staging_dir("att_1") / "pipe")
    with pytest.raises(ArtifactError) as err:
        store.publish_candidate("att_1", "pipe", schema_ref=SCHEMA)
    assert err.value.code == "NOT_A_REGULAR_FILE"


def test_missing_candidate_is_named_missing(tmp_path):
    store = ArtifactStore(tmp_path)
    store.staging_dir("att_1")
    with pytest.raises(ArtifactError) as err:
        store.publish_candidate("att_1", "absent.json", schema_ref=SCHEMA)
    assert err.value.code == "MISSING"


def test_size_limit_refuses_and_leaves_no_object(tmp_path):
    store = ArtifactStore(tmp_path)
    _stage(store, "att_1", "big", b"x" * 100)
    with pytest.raises(ArtifactError) as err:
        store.publish_candidate("att_1", "big", schema_ref=SCHEMA, max_bytes=99)
    assert err.value.code == "SIZE_LIMIT"
    assert _objects(tmp_path) == []
    assert list((tmp_path / "tmp").iterdir()) == []


def test_stale_writer_cannot_change_published_bytes(tmp_path):
    """A worker still holding its file open keeps writing after publication."""
    store = ArtifactStore(tmp_path)
    path = _stage(store, "att_1", "shard.json", b"complete")
    with open(path, "ab") as stale:
        ref = store.publish_candidate("att_1", "shard.json", schema_ref=SCHEMA)
        stale.write(b" + torn tail")
        stale.flush()
    assert store.read_verified(ref) == b"complete"
    assert path.read_bytes() == b"complete + torn tail"


def test_tampered_object_fails_verification(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.publish_bytes(b"manifest", schema_ref=SCHEMA)
    obj = tmp_path / ref.storage_key
    obj.chmod(0o644)
    obj.write_bytes(b"manifesT")
    for read in (store.verify, store.read_verified):
        with pytest.raises(ArtifactError) as err:
            read(ref)
        assert err.value.code == "INTEGRITY_FAILED"


def test_reference_whose_key_disagrees_with_its_hash_is_refused(tmp_path):
    store = ArtifactStore(tmp_path)
    good = store.publish_bytes(b"one", schema_ref=SCHEMA)
    other = store.publish_bytes(b"two", schema_ref=SCHEMA)
    forged = type(good)(**{**good.__dict__, "storage_key": other.storage_key})
    with pytest.raises(ArtifactError) as err:
        store.verify(forged)
    assert err.value.code == "INTEGRITY_FAILED"


def test_existing_corrupt_object_is_not_trusted_on_republish(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.publish_bytes(b"payload", schema_ref=SCHEMA)
    obj = tmp_path / ref.storage_key
    obj.chmod(0o644)
    obj.write_bytes(b"torn")
    with pytest.raises(ArtifactError) as err:
        store.publish_bytes(b"payload", schema_ref=SCHEMA)
    assert err.value.code == "INTEGRITY_FAILED"


class Crash(Exception):
    pass


def _crash_at(point: str):
    def fault(name: str) -> None:
        if name == point:
            raise Crash(name)
    return fault


def test_crash_before_link_leaves_no_referenceable_object(tmp_path):
    store = ArtifactStore(tmp_path, fault=_crash_at("copied"))
    _stage(store, "att_1", "shard", b"partial?")
    with pytest.raises(Crash):
        store.publish_candidate("att_1", "shard", schema_ref=SCHEMA)
    assert _objects(tmp_path) == []
    # The recovered coordinator publishes the same candidate cleanly.
    ref = ArtifactStore(tmp_path).publish_candidate("att_1", "shard", schema_ref=SCHEMA)
    assert ArtifactStore(tmp_path).read_verified(ref) == b"partial?"


def test_crash_after_link_leaves_a_complete_unreferenced_object(tmp_path):
    store = ArtifactStore(tmp_path, fault=_crash_at("linked"))
    _stage(store, "att_1", "shard", b"whole")
    with pytest.raises(Crash):
        store.publish_candidate("att_1", "shard", schema_ref=SCHEMA)
    (obj,) = _objects(tmp_path)
    assert obj.read_bytes() == b"whole"
    ref = ArtifactStore(tmp_path).publish_candidate("att_1", "shard", schema_ref=SCHEMA)
    assert tmp_path / ref.storage_key == obj
