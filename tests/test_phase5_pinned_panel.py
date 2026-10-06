"""Synthetic tests for the pinned original-panel COPY identity helper.

Catalog + ArtifactStore built with the shared ``tests.data_scan_support``
helpers (real published Parquet objects, real ``FragmentRecord``s, real
``commit_snapshot``). The helper returns a direct dict whose every key carries
the ``panel_copy_`` prefix. No private fixtures, no live provider, no fitting.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from pathlib import Path

import joblib
import pyarrow as pa
import pytest

from checks import phase5_release as layout
from engine.v2.foundation import CONTENT_HASH_PREFIX, from_document, to_document
from engine.v2.models import deployment
from engine.v2.models.serving_folds import ServingFoldDescriptor
from tests.data_scan_support import (catalog_and_store, commit_tables, contract_for,
                                     contract_ref_for, fake_hash, hand_built_record, to_bytes)
from tests.test_phase5_serving_folds import cache, _bytes, _inputs, _name, _policy  # noqa: F401
from tools import phase5_prepare_release as prep
from tools.phase5_pinned_panel import COPY_PREFIX, PANEL_TABLE, PinnedPanelContext, verify_panel_copy

IDENTITY_KEYS = {"panel_copy_mode", "panel_copy_snapshot_id", "panel_copy_dataset_version_id",
                 "panel_copy_object_id", "panel_copy_sha256"}
REFUSALS = (ValueError, getattr(prep, "PrepareRefused", ValueError))


def panel_table(rows: list[tuple[str, str, float]]) -> pa.Table:
    return pa.table({
        "ticker": pa.array([r[0] for r in rows], type=pa.string()),
        "date": pa.array([r[1] for r in rows], type=pa.string()),
        "momentum": pa.array([r[2] for r in rows], type=pa.float64()),
    })


ROWS = [("AAA", "2024-01-02", 0.5), ("MMM", "2024-06-30", 1.5)]
ALL_ONE = [(panel_table(ROWS), "all", ("AAA", "2024-01-02"), ("MMM", "2024-06-30"))]


def build_pinned(tmp_path, tables_and_keys):
    """One committed snapshot over ``feature_panel`` fragments; returns
    ``(context, store, snapshot_ref, records, source_bytes)``. The context uses
    the store's real root and the catalog filename the connection reported."""
    conn, clock, store = catalog_and_store(tmp_path)
    contract = contract_for(PANEL_TABLE)
    contract_ref = contract_ref_for(contract)
    records, source_bytes = [], None
    for table, partition_key, key_min, key_max in tables_and_keys:
        record = hand_built_record(store, contract, contract_ref, table,
                                   partition_key=partition_key, row_count=table.num_rows,
                                   primary_key_min=key_min, primary_key_max=key_max,
                                   logical_label=fake_hash(f"panel-{partition_key}"))
        records.append(record)
        if source_bytes is None:
            source_bytes = to_bytes(table)
    snapshot = commit_tables(conn, clock, {PANEL_TABLE: records}, {PANEL_TABLE: contract},
                             store=store)
    catalog_path = Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve()
    conn.close()
    context = PinnedPanelContext(catalog_path=catalog_path, snapshot_id=snapshot.snapshot_id,
                                 artifact_root=store.root)
    return context, store, snapshot, records, source_bytes


def test_real_original_object_returns_direct_prefixed_identities(tmp_path):
    context, _store, snapshot, records, source_bytes = build_pinned(tmp_path, ALL_ONE)
    identity = verify_panel_copy(context)
    assert set(identity) == IDENTITY_KEYS
    assert all(key.startswith(COPY_PREFIX) for key in identity)
    assert identity["panel_copy_mode"] == "COPY"
    assert identity["panel_copy_snapshot_id"] == snapshot.snapshot_id
    assert identity["panel_copy_dataset_version_id"] == snapshot.table_versions[PANEL_TABLE].dataset_version_id
    assert identity["panel_copy_object_id"] == records[0].object_ref.object_id
    assert identity["panel_copy_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert records[0].object_ref.content_hash == CONTENT_HASH_PREFIX + identity["panel_copy_sha256"]


def test_expected_hash_accepted_prefixed_or_bare(tmp_path):
    context, _store, _snap, _records, source_bytes = build_pinned(tmp_path, ALL_ONE)
    raw = hashlib.sha256(source_bytes).hexdigest()
    assert verify_panel_copy(context, raw) == verify_panel_copy(context, CONTENT_HASH_PREFIX + raw)


def test_expected_hash_mismatch_refused_without_path(tmp_path):
    context, _store, _snap, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    with pytest.raises(ValueError) as excinfo:
        verify_panel_copy(context, "0" * 64)
    message = str(excinfo.value)
    assert message.startswith(COPY_PREFIX)
    assert str(tmp_path) not in message


def test_missing_snapshot_refused(tmp_path):
    context, _store, _snap, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    missing = PinnedPanelContext(catalog_path=context.catalog_path,
                                 snapshot_id="snap_missing", artifact_root=context.artifact_root)
    with pytest.raises(ValueError, match=COPY_PREFIX):
        verify_panel_copy(missing)


def test_unreadable_catalog_refused(tmp_path):
    context = PinnedPanelContext(catalog_path=tmp_path / "nope.sqlite",
                                 snapshot_id="snap_any", artifact_root=tmp_path / "store")
    with pytest.raises(ValueError, match=COPY_PREFIX):
        verify_panel_copy(context)


def test_readable_but_empty_catalog_refused_without_path(tmp_path):
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).close()
    context = PinnedPanelContext(catalog_path=empty, snapshot_id="snap_any",
                                 artifact_root=tmp_path / "store")
    with pytest.raises(ValueError) as excinfo:
        verify_panel_copy(context)
    message = str(excinfo.value)
    assert message.startswith(COPY_PREFIX)
    assert str(tmp_path) not in message


def test_multi_fragment_panel_refused(tmp_path):
    context, _store, _snap, _records, _bytes = build_pinned(
        tmp_path, [
            (panel_table(ROWS[:1]), "a", ("AAA", "2024-01-02"), ("MMM", "2024-06-30")),
            (panel_table([("ZZZ", "2024-12-31", 2.0)]), "b", ("NNN", "2024-07-01"),
             ("ZZZ", "2024-12-31")),
        ])
    with pytest.raises(ValueError, match=COPY_PREFIX):
        verify_panel_copy(context)


def test_corrupt_object_bytes_refused(tmp_path):
    context, store, _snap, records, _bytes = build_pinned(tmp_path, ALL_ONE)
    digest = records[0].object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    object_path = store.root / "objects" / digest[:2] / digest
    object_path.chmod(object_path.stat().st_mode | 0o200)
    object_path.write_bytes(b"substituted, not the original panel bytes at all")
    with pytest.raises(ValueError, match=COPY_PREFIX):
        verify_panel_copy(context)


def test_global_live_panel_path_is_ignored(tmp_path, monkeypatch):
    """Verification is anchored to the pinned catalog/objects only: pointing the
    live global panel path at unrelated bytes cannot change the COPY identity."""
    context, _store, _snap, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    before = verify_panel_copy(context)
    decoy = tmp_path / "decoy-panel.parquet"
    monkeypatch.setattr("engine.paths.PANEL", decoy)
    decoy.write_bytes(b"global panel rewrite")
    (tmp_path / "panel.parquet").write_bytes(b"another global panel rewrite")
    assert verify_panel_copy(context) == before


def test_catalog_path_with_uri_ambiguous_characters_verifies(tmp_path):
    context, _store, snapshot, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    expected = verify_panel_copy(context)
    relocated = tmp_path / "snap shots" / "odd?#1.sqlite"
    relocated.parent.mkdir(parents=True)
    context.catalog_path.rename(relocated)
    catalog_bytes = relocated.read_bytes()
    moved = PinnedPanelContext(catalog_path=relocated, snapshot_id=snapshot.snapshot_id,
                               artifact_root=context.artifact_root)
    assert verify_panel_copy(moved) == expected
    assert relocated.read_bytes() == catalog_bytes  # read-only: source never rewritten


def test_verification_publishes_nothing(tmp_path):
    context, store, _snap, _records, _bytes = build_pinned(tmp_path, ALL_ONE)
    objects_dir = store.root / "objects"
    before = sorted(p.relative_to(store.root) for p in objects_dir.rglob("*") if p.is_file())
    verify_panel_copy(context)
    verify_panel_copy(context)
    after = sorted(p.relative_to(store.root) for p in objects_dir.rglob("*") if p.is_file())
    assert before == after


def test_pinned_panel_integration_writes_verified_identity_and_descriptor(tmp_path, cache, monkeypatch):
    """Real ``write_release`` over the verified original panel identity, no fabricated receipt."""
    context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
    identity = verify_panel_copy(context)
    assert set(identity) == IDENTITY_KEYS
    cache["tier3_snapshot"] = identity["panel_copy_sha256"]
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    states = prep.build_states({"tier4_folds:size": {name: raw}}, size_fold_policy=policy)
    decoy = tmp_path / "decoy-panel.parquet"
    monkeypatch.setattr("engine.paths.PANEL", decoy)
    decoy.write_bytes(b"global panel rewrite")
    loads, real_load = [], joblib.load
    monkeypatch.setattr(joblib, "load",
                        lambda *args, **kwargs: loads.append(args) or real_load(*args, **kwargs))
    out = tmp_path / "out"
    args = _inputs(cache)
    prep.write_release(out, *args, states, pinned_panel=context)
    assert len(loads) == 1
    staged_dir = out / "deployment" / "releases" / "r1"
    assert not (staged_dir / "staging-status.json").exists()
    body = layout.read_manifest(out)
    assert {key: body["sources"][key] for key in IDENTITY_KEYS} == identity
    assert deployment.current_pointer(out / "deployment") is None
    row = next(row for row in body["members"] if row["member_id"] == "tier4_folds:size")
    obj = row["objects"][0]
    descriptor = from_document(ServingFoldDescriptor, obj["serving_fold"])
    assert to_document(descriptor) == obj["serving_fold"]
    assert descriptor.policy == policy
    assert descriptor.policy.panel_sha256 == identity["panel_copy_sha256"]
    assert descriptor.estimator.content_hash == obj["content_hash"]
    assert descriptor.estimator.path == obj["path"]
    staged = deployment.stage_release(tmp_path / "parent", *args)
    assert descriptor.parent_release_id == staged.release.release_id == "r1"
    assert descriptor.parent_release_hash == staged.release_hash
    loads.clear()
    first = layout.read_manifest(out)
    prep.write_release(out, *args, states, pinned_panel=context)
    assert layout.read_manifest(out) == first
    assert deployment.current_pointer(out / "deployment") is None


def test_write_release_refuses_after_staged_manifest_publishes_no_staging_status(tmp_path, cache,
                                                                                 monkeypatch):
    """An incomplete staging workflow cannot publish success after a post-stage
    failure: ``write_manifest`` refusing once ``stage_release`` has already
    landed the model manifest leaves that manifest staged and its adjacent
    ``staging-status.json`` never written."""
    context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
    cache["tier3_snapshot"] = verify_panel_copy(context)["panel_copy_sha256"]
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    states = prep.build_states({"tier4_folds:size": {name: raw}}, size_fold_policy=policy)

    def refusing_write_manifest(*_args, **_kwargs):
        raise prep.PrepareRefused("simulated manifest write failure after staging")

    monkeypatch.setattr(prep, "write_manifest", refusing_write_manifest)
    out = tmp_path / "out"
    with pytest.raises(prep.PrepareRefused):
        prep.write_release(out, *_inputs(cache), states, pinned_panel=context)
    staged = out / "deployment" / "releases" / "r1"
    assert (staged / "manifest.json").is_file()
    assert not (staged / "staging-status.json").exists()


def test_write_release_publishes_preflighted_bytes_after_caller_mutation(tmp_path, cache, monkeypatch):
    """Swapping the caller's state payload mapping during real staging cannot change the emitted fold."""
    context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
    cache["tier3_snapshot"] = verify_panel_copy(context)["panel_copy_sha256"]
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    states = prep.build_states({"tier4_folds:size": {name: raw}}, size_fold_policy=policy)
    size_build = next(build for build in states if build.spec.member_id == "tier4_folds:size")
    actual_stage = deployment.stage_release

    def mutate_after_staging(*args, **kwargs):
        staged = actual_stage(*args, **kwargs)
        size_build.payloads.clear()
        size_build.payloads["tampered.joblib"] = b"mutated after preflight"
        return staged

    monkeypatch.setattr(deployment, "stage_release", mutate_after_staging)
    out = tmp_path / "out"
    prep.write_release(out, *_inputs(cache), states, pinned_panel=context)
    assert size_build.payloads == {"tampered.joblib": b"mutated after preflight"}
    body = layout.read_manifest(out)
    row = next(row for row in body["members"] if row["member_id"] == "tier4_folds:size")
    assert [obj["name"] for obj in row["objects"]] == [name]
    obj = row["objects"][0]
    assert obj["content_hash"] == layout.sha256_bytes(raw)
    descriptor = from_document(ServingFoldDescriptor, obj["serving_fold"])
    assert descriptor.estimator.content_hash == layout.sha256_bytes(raw)
    assert descriptor.estimator.path == obj["path"]
    assert descriptor.policy == policy
    assert deployment.current_pointer(out / "deployment") is None


@pytest.mark.parametrize("defect", ["policy", "header"])
def test_write_release_refuses_panel_policy_header_disagreement(tmp_path, cache, defect):
    """One mutated side only: the other keeps the valid digest the name/raw/policy were built with."""
    context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
    cache["tier3_snapshot"] = verify_panel_copy(context)["panel_copy_sha256"]
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    if defect == "policy":
        policy = dataclasses.replace(policy, panel_sha256="f" * 64)
    else:
        cache["tier3_snapshot"] = "f" * 64
        raw = _bytes(cache)
    states = prep.build_states({"tier4_folds:size": {name: raw}}, size_fold_policy=policy)
    out = tmp_path / "out"
    with pytest.raises(REFUSALS):
        prep.write_release(out, *_inputs(cache), states, pinned_panel=context)
    assert not out.exists()


@pytest.mark.parametrize("path_like", [False, True])
def test_write_release_refuses_non_bytes_size_fold_before_output(tmp_path, cache, path_like):
    context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
    cache["tier3_snapshot"] = verify_panel_copy(context)["panel_copy_sha256"]
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    if path_like:
        name = f"nested/{name}"
    states = prep.build_states({"tier4_folds:size": {name: bytearray(raw)}},
                               size_fold_policy=policy)
    out = tmp_path / "out"
    with pytest.raises(REFUSALS) as excinfo:
        prep.write_release(out, *_inputs(cache), states, pinned_panel=context)
    assert not out.exists()
    message = str(excinfo.value)
    assert message.startswith(COPY_PREFIX)
    assert name not in message
    if path_like:
        assert "/" not in message


def test_write_release_refuses_duplicate_size_fold_filename_before_output(tmp_path, cache):
    context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
    cache["tier3_snapshot"] = verify_panel_copy(context)["panel_copy_sha256"]
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    first = prep.build_states({"tier4_folds:size": {name: raw}}, size_fold_policy=policy)
    size = next(build for build in first if build.spec.member_id == "tier4_folds:size")
    states = first + [dataclasses.replace(size)]
    out = tmp_path / "out"
    with pytest.raises(REFUSALS):
        prep.write_release(out, *_inputs(cache), states, pinned_panel=context)
    assert not out.exists()


@pytest.mark.parametrize("mode", ["pinned", "unpinned"])
def test_caller_supplied_panel_copy_sources_refuse(tmp_path, cache, mode):
    """``panel_copy_*`` identities come only from verification, never from caller-supplied sources."""
    out = tmp_path / "out"
    kwargs = {}
    if mode == "pinned":
        context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
        cache["tier3_snapshot"] = verify_panel_copy(context)["panel_copy_sha256"]
        kwargs["pinned_panel"] = context
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    states = prep.build_states({"tier4_folds:size": {name: raw}}, size_fold_policy=policy)
    with pytest.raises(REFUSALS):
        prep.write_release(out, *_inputs(cache), states, sources={"panel_copy_mode": "COPY"}, **kwargs)
    assert not out.exists()


def test_cli_partial_pin_flags_refuse_before_inputs(tmp_path, monkeypatch, capsys):
    """Catalog pin without the snapshot is refused before any real input is loaded."""
    monkeypatch.setattr(prep, "_real_inputs",
                        lambda *a, **k: pytest.fail("inputs loaded while pin flags were incomplete"))
    target = tmp_path / "out"
    assert prep.main(["--release-id", "r1", "--out", str(target),
                      "--panel-catalog", str(tmp_path / "nonexistent")]) == 2
    err = capsys.readouterr().err
    assert COPY_PREFIX in err
    assert "all-or-none" in err
    assert not target.exists()


@pytest.mark.parametrize("mode", ["auto", "matching", "wrong"])
def test_cli_pinned_panel_verifies_catalog_and_ignores_global_panel(mode, tmp_path, cache,
                                                                     monkeypatch, capsys):
    """Real ``main()`` with all three pin flags: auto and a matching sha256-prefixed
    literal stage from the pinned catalog; a wrong literal refuses before any output.
    The global live PANEL is never hashed because verification is catalog-anchored."""
    from engine import paths
    from engine.data.features import tier4
    from engine.data import store as data_store
    from engine.v2.models import inventory

    context, _store, _snapshot, _records, source_bytes = build_pinned(tmp_path, ALL_ONE)
    identity = verify_panel_copy(context)
    digest = identity["panel_copy_sha256"]
    cache["tier3_snapshot"] = digest
    name, raw = _name(cache), _bytes(cache)

    producer = tier4.size_feature_model()
    assert producer.model_id == cache["model_id"]
    assert producer.features == tuple(cache["features"])
    release, inv, payloads = _inputs(cache)
    monkeypatch.setattr(inventory, "current_release_inventory", lambda: (inv, ()))
    monkeypatch.setattr(prep, "model_release", lambda *a, **k: (release, payloads))
    monkeypatch.setattr(prep, "modules_available", lambda *a: (False, "unused"))
    monkeypatch.setattr(tier4, "serving_model", lambda *a, **k: pytest.fail("runtime fitting path"))

    real_file_sha256 = data_store.file_sha256

    def guard_file_sha256(path, *a, **k):
        if Path(path) == Path(paths.PANEL):
            pytest.fail("pinned CLI hashed the global live PANEL")
        return real_file_sha256(path, *a, **k)

    monkeypatch.setattr(data_store, "file_sha256", guard_file_sha256)

    probe = tmp_path / "guard-forwarding-probe"
    probe.write_bytes(b"guard forwarding")
    forwarded = data_store.file_sha256(probe)
    if isinstance(forwarded, bytes):
        forwarded = forwarded.decode()
    assert forwarded.removeprefix(CONTENT_HASH_PREFIX) == hashlib.sha256(b"guard forwarding").hexdigest()

    if mode == "auto":
        expected = "auto"
    elif mode == "matching":
        expected = CONTENT_HASH_PREFIX + digest
    else:
        expected = CONTENT_HASH_PREFIX + "0" * 64

    cache_dir = tmp_path / "tier4"
    cache_dir.mkdir()
    (cache_dir / name).write_bytes(raw)
    out = tmp_path / "out"
    code = prep.main(["--release-id", "r1", "--out", str(out), "--tier4-dir", str(cache_dir),
                      "--panel-catalog", str(context.catalog_path),
                      "--panel-snapshot-id", context.snapshot_id,
                      "--panel-artifact-root", str(context.artifact_root),
                      "--tier3-snapshot", expected])

    if mode == "wrong":
        assert code == 2
        assert COPY_PREFIX in capsys.readouterr().err
        assert not out.exists()
        return

    assert code == 0
    body = layout.read_manifest(out)
    assert {key: body["sources"][key] for key in IDENTITY_KEYS} == identity
    row = next(row for row in body["members"] if row["member_id"] == "tier4_folds:size")
    obj = row["objects"][0]
    descriptor = from_document(ServingFoldDescriptor, obj["serving_fold"])
    assert descriptor.policy.panel_sha256 == digest == hashlib.sha256(source_bytes).hexdigest()
    assert descriptor.estimator.content_hash == obj["content_hash"]
    assert descriptor.estimator.path == obj["path"]


def test_rerun_over_corrupt_stored_size_fold_object_refuses_without_repair(tmp_path, cache):
    """A pinned rerun never trusts a reused destination: corrupt stored bytes
    refuse with a fixed COPY message, the published catalogs keep their exact
    bytes and the corrupted object is not silently repaired or replaced."""
    context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
    cache["tier3_snapshot"] = verify_panel_copy(context)["panel_copy_sha256"]
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    states = prep.build_states({"tier4_folds:size": {name: raw}}, size_fold_policy=policy)
    args = _inputs(cache)
    out = tmp_path / "out"
    local = prep.write_release(out, *args, states, pinned_panel=context)
    body = layout.read_manifest(out)
    row = next(row for row in body["members"] if row["member_id"] == "tier4_folds:size")
    obj = row["objects"][0]
    object_path = layout.deployment_root(out) / obj["path"]
    assert object_path.read_bytes() == raw
    object_path.chmod(object_path.stat().st_mode | 0o200)
    object_path.write_bytes(raw + b" corrupted after publication")
    corrupted = object_path.read_bytes()
    catalogs = {path: path.read_bytes() for path in (local, out / layout.MANIFEST_NAME)}
    with pytest.raises(REFUSALS) as excinfo:
        prep.write_release(out, *args, states, pinned_panel=context)
    message = str(excinfo.value)
    assert message.startswith(COPY_PREFIX)
    assert "refused" in message
    assert str(out) not in message and obj["path"] not in message
    assert {path: path.read_bytes() for path in catalogs} == catalogs
    assert object_path.read_bytes() == corrupted


def test_planted_corrupt_writer_cannot_publish_descriptor_or_catalog(tmp_path, cache, monkeypatch):
    """A write_object returning the expected digest/path while corrupting the
    destination is caught by the byte check before any descriptor is attached,
    so no phase5 catalog -- root or release-local -- is ever published."""
    context, _store, _snapshot, _records, _panel = build_pinned(tmp_path, ALL_ONE)
    cache["tier3_snapshot"] = verify_panel_copy(context)["panel_copy_sha256"]
    name, raw, policy = _name(cache), _bytes(cache), _policy(cache)
    states = prep.build_states({"tier4_folds:size": {name: raw}}, size_fold_policy=policy)
    out = tmp_path / "out"
    real_write = prep.write_object

    def corrupting_write(root, payload):
        digest, rel = real_write(root, payload)
        dest = layout.deployment_root(root) / rel
        dest.chmod(dest.stat().st_mode | 0o200)
        dest.write_bytes(payload + b" planted")
        return digest, rel

    monkeypatch.setattr(prep, "write_object", corrupting_write)
    with pytest.raises(REFUSALS) as excinfo:
        prep.write_release(out, *_inputs(cache), states, pinned_panel=context)
    message = str(excinfo.value)
    assert message.startswith(COPY_PREFIX)
    assert "refused" in message
    assert str(out) not in message
    assert not list(out.rglob(layout.MANIFEST_NAME))
    planted = layout.deployment_root(out) / layout.object_relpath(layout.sha256_bytes(raw))
    assert planted.read_bytes() == raw + b" planted"
